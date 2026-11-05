"""Queue a validated query-pair objective and idle-GPU expansion for main.

The queue uses no GPU. A controller-only handover preserves the running ranks;
all changes wait for their completed, fully validated checkpoint. The paired
experiment's weights are never used by main. Effective batch four and the
checkpoint's optimizer, LR plan and global sampler cursor are preserved.
"""
from contextlib import ExitStack
from copy import deepcopy
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time
import traceback

from multicamera4rc.config import MultiCamera4RCConfig
from multicamera4rc.datasets.mixture import read_data_config
from multicamera4rc.launch import gpu_snapshot
from multicamera4rc.losses import MultiCamera4RCLoss
from multicamera4rc.quality import compare_errors
from scripts.continue_extended_training import next_segment_end, quality_decision
from scripts.continue_quality_recovery import digest, training_argv, validate_checkpoint
from scripts.continue_syn4d_training import verify_state
from scripts.cutover_native_motion_stereo import alive, proc, write_json
from scripts.intrinsics_trial import owned_tree
from scripts.migrate_dynamic_validation import (
    SUITES, ExpandedDynamicController, build_validation_contract,
    require_fixed_training_state, require_validation_contract, suite_paths, verify_suite_report,
)
from scripts.migrate_static_validation import require_training_child, training_ranks
from scripts.query_pair_trial import QueryPairTrial, arm_losses, trial_decision
from scripts.resume_official_after_resource_failure import (
    effective_batch, execution_layout, resize_receipt,
)


def read(path):
    return json.loads(Path(path).read_text())


def verify_files(files):
    for path, expected in files.items():
        if digest(path) != expected:
            raise ValueError('frozen evidence changed: ' + path)


def clean_runtime(root, commit):
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('admission runtime differs from its tested clean commit')


def completed_gate(decision, frozen):
    """A 500-step report alone never authorizes main admission."""
    begin, end = frozen['begin'], frozen['end']
    milestones = decision.get('milestones', [])
    if (end - begin != 1000 or decision.get('begin') != begin or decision.get('end') != end
            or [(r.get('step'), r.get('updates')) for r in milestones]
            != [(begin + 500, 500), (end, 1000)]
            or decision.get('lr_changed') is not False
            or decision.get('test_used_for_selection') is not False):
        raise ValueError('both complete matched milestones and the original LR are required')
    return decision.get('selected') == 'candidate' and decision['comparison'].get('passed') is True


def trial_receipt(root):
    """Recompute admission from the five complete fixed validation reports."""
    root = Path(root)
    if not (root/'decision.json').exists():
        return None
    frozen, decision = read(root/'frozen_job.json'), read(root/'decision.json')
    if not completed_gate(decision, frozen):
        return dict(passed=False, reason='completed_pair_trial_did_not_pass')
    clean_runtime(frozen['repo'], frozen['commit'])
    # decision.json precedes the recipe by one atomic write in the trial.
    if not (root/'admission_recipe.json').exists():
        return None
    verify_files(frozen['config_hashes'])
    verify_files(frozen['protected_files'])
    for name in ('gt_audit', 'mixed_gt_audit'):
        audit = read(frozen[name])
        if audit.get('stage') != 'passed' or audit.get('failures'):
            raise ValueError('production GT admission did not pass')
    preflight = read(root/'cuda_preflight.json')
    if (not preflight.get('passed') or len(preflight.get('rows', [])) != 13
            or not all(a.get('finite_backward') for r in preflight['rows'] for a in r['arms'].values())):
        raise ValueError('complete giant-model backward checks are required')
    recipe = read(root/'admission_recipe.json')
    expected = arm_losses(frozen['source_training_state']['loss_config'], frozen.get('weight', .01))
    if (recipe['arm'] != 'query_pair' or recipe['loss'] != expected['candidate']
            or recipe['source_loss'] != frozen['source_training_state']['loss_config']
            or any(frozen['arms'][name]['loss'] != loss for name, loss in expected.items())
            or not recipe.get('experimental_checkpoint_must_not_replace_newer_main')
            or not recipe.get('restore_optimizer_rng_cursor') or recipe.get('lr_changed') is not False):
        raise ValueError('candidate recipe is not the tested additive intervention')
    trial = QueryPairTrial(root/'job.json')
    trial.job, trial.settings = frozen, frozen['arms']
    panels = {}
    evidence = [root/'decision.json', root/'admission_recipe.json', root/'frozen_job.json',
                root/'cuda_preflight.json', Path(frozen['gt_audit']), Path(frozen['mixed_gt_audit'])]
    for arm in ('control', 'candidate'):
        record = read(root/'validation'/f"{arm}_{frozen['end']}"/'quality.json')
        trial.check_trial_checkpoint(record['checkpoint'], arm, frozen['end'])
        panels[arm] = {}
        for suite in SUITES:
            path = Path(record['reports'][suite])
            plan, _ = suite_paths(frozen, suite)
            panels[arm][suite] = verify_suite_report(
                suite, read(path), record, read(plan), frozen['validation_contract'])
            evidence.append(path)
    comparison = trial_decision(panels['control'], panels['candidate'])
    if comparison != decision['comparison'] or comparison != decision['milestones'][-1]['comparison']:
        raise ValueError('recorded decision differs from complete fixed validation reports')
    return dict(passed=bool(comparison['passed']), recipe=recipe, comparison=comparison,
                frozen=frozen, evidence={str(p): digest(p) for p in evidence})


def compatible_main(payload, config, receipt):
    frozen = receipt['frozen']
    state = payload['training_state']
    if (MultiCamera4RCConfig.from_dict(payload['config'])
            != MultiCamera4RCConfig.from_dict(frozen['source_model_config'])
            or state['lr_plan'] != frozen['source_lr_plan']
            or MultiCamera4RCLoss(**state['loss_config']).to_config()
            != MultiCamera4RCLoss(**receipt['recipe']['source_loss']).to_config()
            or state.get('augmentation') != frozen['source_training_state'].get('augmentation')
            or read_data_config(config) != read_data_config(frozen['data_config'])):
        raise ValueError('current main data, model or objective differs from the tested intervention')
    if effective_batch(execution_layout(state)) != 4:
        raise ValueError('main must retain effective batch four')
    return receipt['recipe']['loss']


def free_indices(devices, processes):
    occupied = {p.gpu_uuid for p in processes}
    return {i for i, gpu in devices.items()
            if gpu.memory_used_mib < 1024 and gpu.uuid not in occupied}


def expanded_gpus(current, available):
    """Two or four ranks, never a three-rank change to the effective batch."""
    current = list(current)
    if len(current) not in (2, 4) or len(set(current)) != len(current):
        raise ValueError('expected a two/four-rank main job')
    extra = sorted(set(available) - set(current))
    return current + extra[:4-len(current)] if len(current)+len(extra) >= 4 else current


def try_lease(uuid):
    handle = (Path('/tmp')/f'multicamera4rc_{uuid}.lock').open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except BlockingIOError:
        handle.close()
        return None


def require_supervisor_command(identity, previous):
    args = shlex.split(identity['cmd'])
    pairs = dict(zip(args, args[1:]))
    ordinary = pairs.get('--controller') == str(Path(previous['control_dir'])/'job.json')
    wrapper = (pairs.get('--output-root') == str(Path(previous['control_dir']).parent)
               and any(Path(a).name in ('resume_accepted_training.py', 'assign_idle_training_gpus.py')
                       for a in args))
    if not (ordinary or wrapper):
        raise ValueError('process does not identify the recorded main controller')


def continuation_health(path, begin, end, world_size):
    """Account for every accumulated microbatch and admitted camera-only GT."""
    rows = []
    for line in Path(path).read_text().splitlines():
        start = line.find('{"step":')
        if start >= 0:
            rows.append(json.loads(line[start:]))
    if [r['step'] for r in rows] != list(range(begin+1, end+1)):
        raise ValueError('continuation did not complete every requested update')
    calibration_only = depth_without_queries = 0
    for row in rows:
        accumulation = row.get('gradient_accumulation', 1)
        if (row['world_size'] != world_size or row.get('effective_batch_size') != 4
                or world_size*accumulation != 4 or len(row['rank_batches']) != 4
                or row['optimizer_step_skipped'] or row['supervision'] != 'dense'):
            raise ValueError('continuation skipped an update or changed effective batch/supervision')
        for key in ('loss', 'grad_norm', 'lr', 'backbone_lr'):
            if not math.isfinite(row[key]):
                raise ValueError('nonfinite continuation metric: '+key)
        for batch in row['rank_batches']:
            metrics = batch['metrics']
            if not all(math.isfinite(v) for v in metrics.values() if isinstance(v, (int, float))):
                raise ValueError('nonfinite rank metric')
            if metrics.get('supervised_source_pixels', 0) <= 0:
                # Dense queries can land in filtered sky frames while other
                # observations still supervise depth. These depth metrics are
                # emitted only when depth_valid.any() is true.
                if 'depth_abs_rel' in metrics and 'depth_mae_normalized' in metrics:
                    depth_without_queries += 1
                elif 'intrinsics_log_focal_error' in metrics and 'camera_rotation_degrees' in metrics:
                    calibration_only += 1
                else:
                    raise ValueError('microbatch has neither admitted geometry nor camera supervision')
    return dict(passed=True, begin=begin, end=end, updates=len(rows), world_size=world_size,
                effective_batch_size=4, microbatches=4*len(rows), calibration_only=calibration_only,
                depth_without_queries=depth_without_queries,
                all_updates_applied=True, log=str(path))


def require_cursor_progress(before, after, updates):
    consumed = lambda state: (state['data_epoch']*state['dataset_size']
                             + state['batches_consumed']*state['world_size']*state['batch_size'])
    if (before['dataset_size'] != after['dataset_size']
            or consumed(after)-consumed(before) != updates*4):
        raise ValueError('global sampler cursor skipped or repeated training examples')


class QueryPairMainController(ExpandedDynamicController):
    def __init__(self, path):
        super().__init__(path)
        self.active = Path(self.job['main_active_path'])
        self.leases = {}

    def verify_handover(self):
        clean_runtime(self.root, self.job['commit'])
        for key in ('evidence_sha256', 'protected_files', 'official_evidence', 'admission_files'):
            verify_files(self.job.get(key, {}))
        tests = read(self.job['admission_tests'])
        if not tests.get('passed') or tests.get('commit') != self.job['commit']:
            raise ValueError('exact admission runtime must pass its regression tests')
        previous, adopted = self.job['previous'], self.job['adopted_training']
        active = read(self.active)
        keys = ('pid', 'control_dir', 'run', 'phase_target_step', 'data_config', 'physical_gpus', 'selected_checkpoint')
        if any(active[k] != previous[k] for k in keys):
            raise ValueError('main advanced while preparing the handover')
        status = read(Path(previous['control_dir'])/'progress.json')
        if (status['stage'] != f"train_{adopted['end']}_try0" or status.get('child_pid') != adopted['pid']
                or status.get('failed_quality_needs_review') or self.job.get('max_quality_retries', 0)):
            raise ValueError('handover requires an ordinary running training segment')
        require_supervisor_command(self.job['previous_identity'], previous)
        tree = owned_tree(self.job['previous_identity'])
        require_training_child(tree.get(adopted['pid']), adopted, previous['pid'])
        if len(training_ranks(tree)) != len(self.job['physical_gpus']):
            raise ValueError('expected main ranks are missing')
        if build_validation_contract(self.job) != self.job['validation_contract']:
            raise ValueError('fixed validation contract changed')
        refs = {n: read(self.job[n+'_record']) for n in ('initial', 'anchor', 'historical')}
        require_validation_contract(self.job['validation_contract'], *refs.values())
        for record in refs.values():
            self._verify_record(record)
        if (refs['initial']['checkpoint'] != active['selected_checkpoint']
                or refs['initial']['step'] != adopted['begin']
                or refs['anchor']['checkpoint'] != status['best']['checkpoint']
                or self.job['data_config'] != refs['initial']['data_config']):
            raise ValueError('accepted parent, data or best checkpoint changed')
        args = shlex.split(tree[adopted['pid']]['cmd'])
        if (set(args) & {'--override-lr', '--reset-data-cursor'}
                or '--resume' not in args or args[args.index('--resume')+1] != refs['initial']['checkpoint']):
            raise ValueError('active ranks do not resume the recorded accepted checkpoint')
        decision = quality_decision(refs['initial']['errors'], refs['anchor']['errors'],
                                    refs['historical']['errors'], refs['initial']['errors'])
        if decision['action'] != 'continue':
            raise ValueError('accepted baseline needs quality review')
        state = validate_checkpoint(refs['initial']['checkpoint'], adopted['begin'],
                                    world_size=len(self.job['physical_gpus']), batch_size=1)
        verify_state(state, self.job['target_step'])
        if effective_batch(execution_layout(state)) != 4 or state.get('supervision') != 'dense':
            raise ValueError('dense supervision and effective batch four are required')
        self._verified_adopted_state = state
        self.job['execution_resize'] = resize_receipt(
            state, refs['initial']['checkpoint'], adopted['begin'],
            self.job['physical_gpus'], state.get('gradient_accumulation', 1))
        return refs, status, tree, decision

    def _check_candidate_state(self, checkpoint, step):
        resize = self.job['execution_resize']
        source = (str(Path(checkpoint).resolve()), step) == (resize['checkpoint'], resize['step'])
        layout = resize['previous'] if source else resize['current']
        state = validate_checkpoint(checkpoint, step, world_size=layout['world_size'], batch_size=1)
        if (execution_layout(state) != layout or effective_batch(layout) != 4
                or state.get('effective_batch_size', 4) != 4):
            raise ValueError('execution layout must preserve effective batch four')
        verify_state(state, self.job['target_step'])
        expected, actual = deepcopy(self._fixed_state()), deepcopy(state)
        expected.update(world_size=layout['world_size'], batch_size=1)
        activation = self.job.get('query_pair_activation_step')
        if activation is not None and step > activation:
            expected['loss_config'] = read(self.job['query_pair_loss_config'])
        for item in (expected, actual):
            item['loss_config'] = MultiCamera4RCLoss(**item['loss_config']).to_config()
        require_fixed_training_state(expected, actual)
        if state.get('augmentation') != expected.get('augmentation'):
            raise ValueError('augmentation changed during continuation')
        return state

    def finish_adopted(self, refs):
        adopted = self.job['adopted_training']
        deadline = time.monotonic()+self.job.get('adopted_wait_hours', 48)*3600
        while alive(adopted):
            if time.monotonic() > deadline:
                raise TimeoutError('running main segment exceeded its original completion window')
            time.sleep(5)
        checkpoint = Path(adopted['run'])/'latest.pt'
        state = self._check_candidate_state(checkpoint, adopted['end'])
        require_cursor_progress(self._fixed_state(), state, adopted['end']-adopted['begin'])
        write_json(self.output/'adopted_training_proof.json', continuation_health(
            adopted['log'], adopted['begin'], adopted['end'], len(self.job['physical_gpus'])))
        self.progress.pop('child_pid', None)
        record = self.evaluate(checkpoint, f"extended_{adopted['end']}_adopted")
        record.update(step=adopted['end'], data_config=adopted['data_config'])
        return self._accept_or_review(refs['initial'], self.progress['best'], refs['historical'],
                                      record, state, adopted=True)

    def run_process(self, name, command, *, evaluation=False):
        indices = self.job['physical_gpus'][1:2] if evaluation else self.job['physical_gpus']
        while True:
            try:
                devices, processes = gpu_snapshot()
                available = free_indices(devices, processes)
                ready = all(i in available for i in indices)
                if ready:
                    for i in indices:
                        uuid = devices[i].uuid
                        if uuid not in self.leases:
                            handle = try_lease(uuid)
                            if handle is None:
                                ready = False
                                break
                            self.leases[uuid] = handle
                if ready:
                    break
                self.progress.update(pending_phase=name, busy_gpus=sorted(set(indices)-available))
                self.publish('waiting_for_main_gpus')
            except subprocess.SubprocessError as error:
                self.progress['last_probe_error'] = str(error)
                self.publish('waiting_for_gpu_driver')
            time.sleep(20)
        self.progress.pop('pending_phase', None)
        return super().run_process(name, command, evaluation=evaluation)

    def select_devices(self, reference, state):
        stable = {}
        while True:
            try:
                devices, processes = gpu_snapshot()
            except subprocess.SubprocessError as error:
                self.progress['last_probe_error'] = str(error)
                self.publish('waiting_for_gpu_driver')
                time.sleep(20)
                continue
            now = time.monotonic()
            available = free_indices(devices, processes)
            stable = {i: stable.get(i, now) for i in available}
            eligible = {i for i, since in stable.items() if now-since >= self.job.get('idle_seconds', 60)}
            for i in sorted(eligible):
                uuid = devices[i].uuid
                if uuid not in self.leases:
                    handle = try_lease(uuid)
                    if handle is None:
                        eligible.discard(i)
                    else:
                        self.leases[uuid] = handle
            current = self.job['physical_gpus']
            chosen = expanded_gpus(current, eligible)
            if set(current) <= eligible:
                break
            # Keep no reservations on optional GPUs while current ranks wait.
            for uuid in list(self.leases):
                if uuid not in {devices[i].uuid for i in current}:
                    self.leases.pop(uuid).close()
            self.publish('waiting_for_idle_main_boundary')
            time.sleep(20)
        keep = {devices[i].uuid for i in chosen}
        for uuid in list(self.leases):
            if uuid not in keep:
                self.leases.pop(uuid).close()
        resize = resize_receipt(state, reference['checkpoint'], reference['step'], chosen, 4//len(chosen))
        self.job.update(physical_gpus=chosen, gradient_accumulation=4//len(chosen),
                        global_batch_size=4, execution_resize=resize)
        self.job['environment']['CUDA_VISIBLE_DEVICES'] = ','.join(devices[i].uuid for i in chosen)
        write_json(self.output/f"execution_{reference['step']}.json", dict(
            resize, physical_gpus=chosen, stable_idle_seconds=self.job.get('idle_seconds', 60),
            lr_changed=False, reset_data_cursor=False,
            existing_rank_rng_restored=True, new_rank_rng='training_seed + rank'))

    def maybe_admit(self, reference):
        if self.job.get('query_pair_activation_step') is not None:
            verify_files(self.job['query_pair_evidence'])
            return
        try:
            receipt = trial_receipt(self.job['query_pair_trial'])
            if receipt is None or not receipt['passed']:
                self.progress['query_pair_admission'] = receipt or dict(stage='waiting_for_complete_trial')
                return
            import torch
            payload = torch.load(reference['checkpoint'], map_location='cpu', mmap=True, weights_only=True)
            loss = compatible_main(payload, self.job['data_config'], receipt)
            del payload
            path = self.output/'query_pair_loss.json'
            write_json(path, loss)
            self.job.update(query_pair_activation_step=reference['step'], query_pair_loss_config=str(path),
                            query_pair_evidence=dict(receipt['evidence'], **{str(path): digest(path)}))
            self.progress['query_pair_admission'] = dict(stage='validated_recipe_ready', step=reference['step'])
            write_json(self.output/'query_pair_activation.json', dict(
                stage='validated_recipe_ready', step=reference['step'], checkpoint=reference['checkpoint'],
                loss_config=loss, evidence=receipt['evidence'], comparison=receipt['comparison'],
                experimental_weights_used=False, lr_changed=False, reset_data_cursor=False,
                first_main_validation_after_updates=500))
        except (ValueError, KeyError, OSError) as error:
            # A changed/incompatible experiment cannot stop otherwise healthy main.
            self.progress['query_pair_admission'] = dict(stage='not_admitted', reason=str(error))
            write_json(self.output/'query_pair_not_admitted.json', self.progress['query_pair_admission'])

    def segment(self, reference, best, historical, config, end, **options):
        if options or config != reference['data_config'] or config != self.job['data_config']:
            raise ValueError('no retry, LR override or data migration is permitted')
        verify_files(self.job['admission_files'])
        verify_files(self.job['protected_files'])
        previous_state = state = self._check_candidate_state(reference['checkpoint'], reference['step'])
        self.maybe_admit(reference)
        first_pair = self.job.get('query_pair_activation_step') == reference['step']
        if first_pair:
            end = min(end, reference['step']+500)
        self.select_devices(reference, state)
        run = self.output/f"continuation_{reference['step']}_{end}_try0"
        if run.exists():
            raise ValueError('refusing to overwrite an existing segment')
        argv = training_argv(self.job, reference['checkpoint'], config, run, end,
                             loss_config=self.job.get('query_pair_loss_config'))
        argv += ['--gradient-accumulation', str(self.job['gradient_accumulation'])]
        if first_pair:
            argv[argv.index('--save-every')+1] = '500'
        if set(argv) & {'--allow-shared', '--override-lr', '--reset-data-cursor'}:
            raise ValueError('continuation must preserve optimizer, LR and sampler')
        self.job.update(run=str(run), selected_checkpoint=reference['checkpoint'], resume_step=reference['step'],
                        phase_target_step=end)
        self.run_process(f'train_{end}_try0', argv)
        checkpoint = run/'latest.pt'
        state = self._check_candidate_state(checkpoint, end)
        require_cursor_progress(previous_state, state, end-reference['step'])
        write_json(run/'health.json', continuation_health(
            self.control/f'train_{end}_try0.log', reference['step'], end, len(self.job['physical_gpus'])))
        record = self.evaluate(checkpoint, f'extended_{end}_try0')
        record.update(step=end, data_config=config)
        if first_pair:
            guard = compare_errors(reference['errors'], record['errors'])
            write_json(self.output/'query_pair_first_main_validation.json', guard)
            if guard['score_ratio'] > 1.01 or guard['worst_metric_ratio'] > 1.05:
                write_json(self.output/'latest_evaluated_checkpoint.json', record)
                self.progress['failed_quality_needs_review'] = True
                raise ValueError('first main consistency segment failed the strict regression gate')
        return self._accept_or_review(reference, best, historical, record, state)

    def run(self):
        previous_root = Path(self.job['previous']['repo'])
        with ExitStack() as stack:
            handovers = []
            for root in sorted({previous_root.resolve(), self.root.resolve()}):
                directory = root/'outputs/training_preflight'
                directory.mkdir(parents=True, exist_ok=True)
                handle = stack.enter_context((directory/'controller_handover.lock').open('a'))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                handovers.append(handle)
            old_lock = stack.enter_context((previous_root/'outputs/training_preflight/extended_training.lock').open('a'))
            if self.root.resolve() != previous_root.resolve():
                new_lock = stack.enter_context((self.root/'outputs/training_preflight/extended_training.lock').open('a'))
                fcntl.flock(new_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                refs = self.adopt(old_lock)
                for handle in handovers:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                current = self.finish_adopted(refs)
                self.remember_best(current)
                write_json(self.output/'latest_accepted_checkpoint.json', current)
                while current['step'] < self.job['target_step']:
                    end = next_segment_end(current['step'], self.job['target_step'], interval=self.job['validation_interval'])
                    current = self.segment(current, self.progress['best'], refs['historical'], self.job['data_config'], end)
                    self.remember_best(current)
                    write_json(self.output/'latest_accepted_checkpoint.json', current)
                self.publish('completed_'+str(self.job['target_step']))
            except Exception:
                self.progress['error'] = traceback.format_exc()
                if self.owns_training:
                    self.publish('failed')
                else:
                    write_json(self.control/'verification_failed.json', self.progress)
                raise
            finally:
                for handle in self.leases.values():
                    handle.close()


def prepare_controller(queue, active):
    status = read(Path(active['control_dir'])/'progress.json')
    accepted = read(Path(active['output_root'])/'latest_accepted_checkpoint.json')
    if status['stage'] != f"train_{active['phase_target_step']}_try0":
        raise ValueError('wait for the next ordinary training segment')
    root = Path(queue['output_root'])/f"main_{time.time_ns()}"
    control, output = root/'control', root/'training'
    control.mkdir(parents=True)
    output.mkdir()
    identity, child = proc(active['pid']), proc(status['child_pid'])
    if not identity or not child:
        raise ValueError('main controller or child advanced')
    job = deepcopy(active)
    for key in ('audited_data_config_replacement', 'learning_rate_trial', 'cross_camera_loss_config',
                'cross_camera_3d_weight', 'cross_camera_tests_proof'):
        job.pop(key, None)
    job.update(repo=queue['repo'], commit=queue['commit'], previous=deepcopy(active),
               environment=dict(active['environment'], PYTHONPATH=queue['repo']),
               control_dir=str(control), output_root=str(output), log=str(root/'controller.log'),
               previous_identity=identity, previous_start=identity['start'],
               adopted_training=dict(child, run=active['run'], begin=accepted['step'],
                   end=active['phase_target_step'], data_config=active['data_config'], log=status['child_log']),
               adopted_wait_hours=48, allow_shared=False, max_quality_retries=0,
               query_pair_trial=queue['query_pair_trial'], admission_tests=queue['tests_proof'],
               admission_files=queue['frozen_files'], idle_seconds=60)
    for name, record in [('initial', accepted), ('anchor', status['best'])]:
        path = control/(name+'_record.json')
        write_json(path, record)
        job[name+'_record'] = str(path)
    write_json(control/'job.json', job)
    return control/'job.json'


def run_queue(path):
    queue = read(path)
    root = Path(queue['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    def publish(stage, **values):
        write_json(root/'progress.json', dict(stage=stage, pid=os.getpid(), updated_at=time.time(), **values))
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stable = {}
        while True:
            clean_runtime(queue['repo'], queue['commit'])
            verify_files(queue['frozen_files'])
            active = read(queue['main_active_path'])
            if active.get('stage', '').startswith(('failed', 'completed')):
                publish('main_requires_review', main_stage=active['stage'])
                return
            trial = Path(queue['query_pair_trial'])
            ready = False
            if (trial/'decision.json').exists():
                ready = completed_gate(read(trial/'decision.json'), read(trial/'frozen_job.json'))
            try:
                devices, processes = gpu_snapshot()
                now = time.monotonic()
                available = free_indices(devices, processes)
                stable = {i: stable.get(i, now) for i in available}
                eligible = set()
                for i, since in stable.items():
                    if now-since >= 60:
                        handle = try_lease(devices[i].uuid)
                        if handle is not None:
                            handle.close()
                            eligible.add(i)
                expansion = expanded_gpus(active['physical_gpus'], eligible)
            except subprocess.SubprocessError as error:
                publish('waiting_for_gpu_driver', error=str(error))
                time.sleep(30)
                continue
            ordinary = active.get('stage') == f"train_{active.get('phase_target_step')}_try0"
            if ordinary and (ready or len(expansion) > len(active['physical_gpus'])):
                controller_path = prepare_controller(queue, active)
                controller = QueryPairMainController(controller_path)
                controller.verify_handover()  # read-only before dispatch/signals
                with (root/'controller_dispatch.log').open('a') as log:
                    child = subprocess.Popen([queue['python'], '-u', '-m', 'scripts.query_pair_main_admission',
                        '--controller', str(controller_path)], cwd=queue['repo'],
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=queue['repo'],
                                 OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                publish('boundary_controller_running', controller_pid=child.pid, controller=str(controller_path),
                        pair_trial_passed=ready, available_expansion=expansion)
                code = child.wait()
                publish('completed' if code == 0 else 'controller_requires_review', returncode=code)
                return
            publish('waiting_for_validation_or_idle_expansion', main_gpus=active['physical_gpus'],
                    stable_idle_gpus=sorted(eligible), pair_trial_passed=ready,
                    main_stage=active.get('stage'), pair_stage=read(trial/'progress.json').get('stage'))
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--queue')
    mode.add_argument('--controller')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.queue:
        try:
            run_queue(args.queue)
        except Exception:
            queue = read(args.queue)
            write_json(Path(queue['output_root'])/'progress.json', dict(
                stage='queue_requires_review', pid=os.getpid(), error=traceback.format_exc()))
            raise
    else:
        controller = QueryPairMainController(args.controller)
        if args.verify_only:
            controller.verify_handover()
            print(json.dumps(dict(passed=True, no_processes_signalled=True)))
        else:
            controller.run()


if __name__ == '__main__':
    main()
