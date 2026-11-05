"""Two/four-rank residual-tracker continuation with effective batch four.

The initial five updates are a production-size DDP/Adam preflight. Validation
at +500/+1000 and then every 2000 updates uses the inherited complete panels.
No automatic LR reductions or repeated training on rejected validation cases.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
from itertools import combinations
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import traceback

from multicamera4rc.quality import error_panel, compare_errors
from multicamera4rc.config import MultiCamera4RCConfig
from multicamera4rc.launch import gpu_snapshot
from scripts.continue_quality_recovery import RecoveryController, training_argv, validate_checkpoint
from scripts.cutover_native_motion_stereo import alive, write_json
from scripts.migrate_dynamic_validation import build_validation_contract, require_fixed_training_state, ExpandedDynamicController, verify_suite_report
from scripts.query_pair_main_admission import (clean_runtime, read, verify_files, try_lease,
    free_indices, continuation_health, require_cursor_progress)


def next_boundary(step, start, end):
    for boundary in (start + 5, start + 500, start + 1000):
        if step < boundary:
            return min(boundary, end)
    return min(((step // 2000) + 1) * 2000, end)


def layout_for_checkpoint(job, state):
    world=state['world_size'];accumulation=state.get('gradient_accumulation',1)
    if (world not in (2,4) or state['batch_size']!=1 or world*accumulation!=4
            or state.get('effective_batch_size')!=4):
        raise ValueError('two/four-rank execution must retain effective batch four')
    if not job.get('allow_resource_resize') and (world,accumulation)!=(4,1):
        raise ValueError('execution resize was not enabled for this job')
    expected=deepcopy(job['expected_state'])
    expected.update(world_size=world,batch_size=1)
    return expected


def expansion_candidates(current, available, allowed):
    if len(current) not in (2,4) or len(set(current))!=len(current):
        raise ValueError('expected two or four distinct GPUs')
    extra=[i for i in allowed if i in available and i not in current]
    return list(current)+extra[:2] if len(current)==2 and len(extra)>=2 else list(current)


def initial_gpu_candidates(current, available, allowed=None):
    """Prefer the previous layout, then consider only explicitly allowed GPUs."""
    current = list(current)
    if len(current) not in (2, 4) or len(set(current)) != len(current):
        raise ValueError('initial execution requires two or four distinct GPUs')
    allowed = current if allowed is None else list(allowed)
    if len(set(allowed)) != len(allowed) or not set(current) <= set(allowed):
        raise ValueError('initial GPU candidates must contain the previous layout')
    candidates = [current] if set(current) <= available else []
    for selected in combinations([i for i in allowed if i in available], len(current)):
        if set(selected) != set(current):
            candidates.append(list(selected))
    return candidates


def objective_admission_state(job, checkpoint, step, expected):
    """Accept the original native state only at the frozen migration checkpoint."""
    admission = job.get('objective_admission')
    if not admission:
        return expected
    requested = deepcopy(admission['previous_loss_config'])
    requested.setdefault('query_pair_options', {})['dynamic_temporal_fraction'] = .3
    if requested != job['expected_state']['loss_config']:
        raise ValueError('admission may change only the approved dynamic group budget')
    if read(admission['loss_config_path']) != requested:
        raise ValueError('admitted loss file changed')
    if step == admission['step']:
        if not Path(checkpoint).samefile(job['initial_checkpoint']):
            raise ValueError('original objective is allowed only at the frozen checkpoint')
        expected['loss_config'] = deepcopy(admission['previous_loss_config'])
    elif step < admission['step']:
        raise ValueError('cannot resume a checkpoint older than the admitted migration')
    return expected


class RefinementController(RecoveryController):
    def __init__(self, path):
        super().__init__(path)
        self.active = Path(self.job['main_active_path'])
        self.mutex = threading.RLock()

    def publish(self, stage):
        with self.mutex:
            self.progress['updated_utc'] = datetime.now(timezone.utc).isoformat()
            super().publish(stage)
            # The canonical record must identify a live waiter as well as a
            # live trainer. This flag is enabled only by an audited recovery.
            if not self.owns_training and self.job.get('publish_waiting_main'):
                write_json(self.active, self.job)
            now = time.monotonic()
            if stage != getattr(self, '_logged_stage', None) or now - getattr(self, '_logged_at', 0) >= 60:
                print(json.dumps(dict(utc=self.progress['updated_utc'], stage=stage,
                    step=self.progress.get('step'), physical_gpus=self.job['physical_gpus'],
                    resource_wait=self.progress.get('resource_wait'))), flush=True)
                self._logged_stage, self._logged_at = stage, now

    def device_lease(self, index, uuid):
        """Coordinate with existing local schedulers without editing their files."""
        leases = ExitStack()
        try:
            own = try_lease(uuid)
            if own is None:
                raise BlockingIOError('project GPU lease is held')
            leases.enter_context(own)
            for path in self.job.get('cooperative_gpu_locks', {}).get(str(index), []):
                # Missing/unreadable locks fail closed; never create or truncate
                # another scheduler's lock file or bypass its existing lease.
                handle = leases.enter_context(Path(path).open('rb'))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            leases.close()
            self.progress.setdefault('lease_conflicts', {})[str(index)] = str(error)
            return None
        self.progress.get('lease_conflicts', {}).pop(str(index), None)
        return leases

    def wait_for_initial_training_gpus(self, stack):
        original = list(self.job['physical_gpus'])
        allowed = self.job.get('initial_allowed_gpus', original)
        expected = self.job.get('allowed_gpu_uuids') or dict(zip(
            map(str, original), self.job['environment']['CUDA_VISIBLE_DEVICES'].split(',')))
        if set(expected) != set(map(str, allowed)):
            raise ValueError('every allowed GPU needs a frozen UUID')
        while True:
            devices, processes = gpu_snapshot()
            if any(i not in devices or devices[i].uuid != expected[str(i)] for i in allowed):
                raise ValueError('allowed GPU UUID ordering changed')
            available = free_indices(devices, processes)
            for selected in initial_gpu_candidates(original, available, allowed):
                with ExitStack() as tentative:
                    ready = True
                    for index in selected:
                        handle = self.device_lease(index, devices[index].uuid)
                        if handle is None:
                            ready = False
                            break
                        tentative.enter_context(handle)
                    if not ready:
                        continue
                    fresh, apps = gpu_snapshot()
                    if not set(selected) <= free_indices(fresh, apps):
                        continue
                    if any(fresh[i].uuid != expected[str(i)] for i in selected):
                        raise ValueError('GPU UUID changed during resource acquisition')
                    stack.enter_context(tentative.pop_all())
                self.job['physical_gpus'] = selected
                self.job['environment']['CUDA_VISIBLE_DEVICES'] = ','.join(expected[str(i)] for i in selected)
                if selected != original:
                    self.progress.setdefault('resource_changes', []).append(dict(
                        previous_gpus=original, gpus=selected, effective_batch_size=4,
                        at_step=self.progress.get('step'), reason='idle authorized GPUs before resume'))
                self.progress['lease_conflicts'] = {}
                self.progress.pop('resource_wait', None)
                return fresh
            self.progress['available_gpus'] = sorted(available)
            self.progress['resource_wait'] = dict(at_step=self.progress.get('step'),
                allowed_gpus=list(allowed), physical_gpus=original,
                occupied_pids=sorted({p.pid for p in processes if p.gpu_uuid in expected.values()}),
                available_gpus=sorted(available))
            self.publish('waiting_for_training_gpus')
            time.sleep(10)

    def check(self, checkpoint, step):
        import torch
        payload = torch.load(checkpoint, mmap=True, map_location='cpu', weights_only=True)
        expected=objective_admission_state(self.job, checkpoint, step,
            layout_for_checkpoint(self.job,payload['training_state']))
        state = validate_checkpoint(checkpoint, step, world_size=expected['world_size'], batch_size=1)
        require_fixed_training_state(expected, state)
        if (state['supervision'] != 'dense' or state['augmentation'] != self.job['expected_state']['augmentation']):
            raise ValueError('effective batch, dense supervision or augmentation changed')
        if (MultiCamera4RCConfig.from_dict(payload['config']) != MultiCamera4RCConfig.from_dict(self.job['model_config'])
                or len(payload['optimizer']['param_groups']) != 3):
            raise ValueError('model or optimizer groups changed')
        return state

    def evaluate(self, checkpoint, label):
        """Run every fixed suite, distributing work over our reserved GPUs."""
        if build_validation_contract(self.job) != self.job['validation_contract']:
            raise ValueError('fixed validation contract changed')
        self.publish('validate_' + label)
        settings = self.job['evaluation']
        suites = ('kubric', 'native_dynamic', 'static', 'variable', 'syn4d')
        uuids = self.job['environment']['CUDA_VISIBLE_DEVICES'].split(',')
        wrapper = 'import runpy,sys,torch; torch.cuda.set_per_process_memory_fraction(.85,0); runpy.run_module(sys.argv.pop(1),run_name="__main__")'

        def worker(slot):
            result = {}
            for suite in suites[slot::len(uuids)]:
                directory = self.output/'validation'/label/suite
                directory.mkdir(parents=True, exist_ok=True)
                cached = directory/'evaluation.json'
                if label in self.job.get('reuse_validation_labels', []) and cached.exists():
                    report = read(cached)
                    record = dict(checkpoint=str(checkpoint), label=label,
                                  step=self.job['objective_admission']['step'])
                    if report.get('stage') != 'passed':
                        raise ValueError('cannot reuse an incomplete validation report')
                    verify_suite_report(suite, report, record, self._plans()[suite],
                                        self.job['validation_contract'])
                    result[suite] = str(cached)
                    with self.mutex:
                        self.progress.setdefault('reused_validation_suites', []).append(suite)
                        self.publish('validate_' + label)
                    continue
                common = ['--output', str(directory), '--checkpoints', label+'='+str(checkpoint)]
                if suite in ('kubric', 'variable', 'syn4d'):
                    module = 'multicamera4rc.evaluate'
                    config = self.job['syn4d_eval_config'] if suite == 'syn4d' else settings[suite+'_config']
                    plan = self.job['syn4d_eval_plan'] if suite == 'syn4d' else settings[suite+'_plan']
                    args = ['--data-config', config, '--plan', plan, '--cpu-threads', '2']
                elif suite == 'native_dynamic':
                    module = 'scripts.evaluate_native_dynamic'
                    args = ['--data-config', settings['native_config'], '--plan', settings['native_plan'], '--memory-fraction', '.85']
                else:
                    module = 'scripts.evaluate_static_multiview'
                    plan = directory/'fixed_plan.json'
                    shutil.copyfile(settings['static_plan'], plan)
                    args = ['--train-config', settings['static_config'], '--plan', str(plan), '--memory-fraction', '.85']
                command = [self.job['python'], '-u', '-c', wrapper, module, *args, *common]
                log_path = self.control/f'validate_{label}_{suite}.log'
                with log_path.open('w') as log:
                    child = subprocess.Popen(command, cwd=self.root,
                        env=os.environ | self.job['environment'] | {'CUDA_VISIBLE_DEVICES': uuids[slot]},
                        stdout=log, stderr=subprocess.STDOUT)
                    with self.mutex:
                        self.progress.setdefault('validation_children', {})[suite] = child.pid
                        self.publish('validate_' + label)
                    code = child.wait()
                with self.mutex:
                    self.progress['validation_children'].pop(suite, None)
                    self.publish('validate_' + label)
                if code:
                    raise RuntimeError(f'{suite} validation failed ({code}); see {log_path}')
                result[suite] = str(directory/'evaluation.json')
            return result

        reports = {}
        with ThreadPoolExecutor(max_workers=len(uuids)) as pool:
            for paths in pool.map(worker, range(len(uuids))):
                reports.update(paths)
        entries = {suite: read(path)['checkpoints'][label] for suite, path in reports.items()}
        record = dict(checkpoint=str(checkpoint), label=label, errors=error_panel(entries), reports=reports,
                      validation_contract=deepcopy(self.job['validation_contract']))
        # Reuse strict case identity, frame, metric-version and plan checks.
        record['step'] = ExpandedDynamicController._verify_record(self, record)
        write_json(self.output/'validation'/label/'quality.json', record)
        return record

    def _plans(self):
        return ExpandedDynamicController._plans(self)

    def maybe_expand(self,stack):
        if not self.job.get('allow_resource_resize') or len(self.job['physical_gpus'])==4:
            return
        devices,processes=gpu_snapshot()
        old=list(self.job['physical_gpus'])
        selected=expansion_candidates(old,free_indices(devices,processes),self.job['expand_allowed_gpus'])
        if len(selected)!=4:return
        with ExitStack() as tentative:
            for index in selected[len(old):]:
                handle=self.device_lease(index, devices[index].uuid)
                if handle is None:return
                tentative.enter_context(handle)
            fresh,apps=gpu_snapshot()
            if not set(selected)<=free_indices(fresh,apps):return
            for i in selected:
                if fresh[i].uuid!=devices[i].uuid:raise ValueError('GPU UUID changed during expansion')
            stack.enter_context(tentative.pop_all())
        self.job.update(physical_gpus=selected,gradient_accumulation=1)
        self.job['environment']['CUDA_VISIBLE_DEVICES']=','.join(devices[i].uuid for i in selected)
        self.progress.setdefault('resource_changes',[]).append(dict(
            previous_gpus=old,gpus=selected,effective_batch_size=4,at_step=self.progress.get('step')))
        self.publish('expanded_to_four_at_checkpoint')

    def maybe_admit_performance(self,current,step):
        if not self.job.get('performance_admission') or self.progress.get('performance_admission_attempted'):
            return current,step
        state=self.check(current,step)
        if len(self.job['physical_gpus'])!=4 or state['world_size']!=4:
            self.progress['performance_status']='waiting_for_four_rank_checkpoint'
            return current,step
        from scripts.refinement_performance_admission import admit_at_checkpoint
        return admit_at_checkpoint(self,current,step)

    def wait_for_selected_training_gpus(self):
        """Recheck every phase: external jobs need not respect our file leases."""
        if self.job.get('allow_shared'):
            return
        selected = self.job['physical_gpus']
        expected = self.job['environment']['CUDA_VISIBLE_DEVICES'].split(',')
        waited = False
        while True:
            devices, processes = gpu_snapshot()
            if any(index not in devices for index in selected):
                raise ValueError('selected physical GPU disappeared while waiting')
            if [devices[index].uuid for index in selected] != expected:
                raise ValueError('GPU UUID ordering changed while waiting')
            available = free_indices(devices, processes)
            if set(selected) <= available:
                self.progress.pop('resource_wait', None)
                if waited:
                    self.publish('training_gpus_ready')
                return
            self.progress['resource_wait'] = dict(
                at_step=self.progress.get('step'), physical_gpus=list(selected),
                occupied_pids=sorted({p.pid for p in processes if p.gpu_uuid in expected}),
                available_gpus=sorted(available))
            self.publish('waiting_for_training_gpus')
            waited = True
            time.sleep(10)

    def prune_own_checkpoints(self, current):
        # Retain the original backup, migrated initial state, latest six full
        # training saves, best measured result, and every 10k milestone.
        paths = list((self.output/'training').glob('*/*.pt'))
        paths.sort(key=lambda p: p.stat().st_mtime)
        protected = {str(Path(current).resolve()), self.progress.get('best', {}).get('checkpoint')}
        protected.update(str(p.resolve()) for p in paths[-6:])
        for p in paths[:-6]:
            if str(p.resolve()) in protected:
                continue
            if p.name.startswith('step_') and int(p.stem.split('_')[1]) % 10000 == 0:
                continue
            p.unlink()

    def run(self):
        with ExitStack() as stack:
            lockdirs = {self.root/'outputs/training_preflight'}
            if self.job.get('main_active_path'):
                lockdirs.add(Path(self.job['main_active_path']).parent)
            for lockdir in sorted(lockdirs):
                lockdir.mkdir(parents=True, exist_ok=True)
                lock = stack.enter_context((lockdir/'extended_training.lock').open('a'))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            clean_runtime(self.root, self.job['commit'])
            verify_files(self.job['protected_files'])
            if any(alive(p) for p in self.job['stopped_identities']):
                raise ValueError('previous main process still running')
            self.progress.update(step=self.job.get('continuation_step', self.job['initial_step']),
                                 latest_checkpoint=self.job['initial_checkpoint'])
            devices = self.wait_for_initial_training_gpus(stack)
            if ','.join(devices[i].uuid for i in self.job['physical_gpus']) != self.job['environment']['CUDA_VISIBLE_DEVICES']:
                raise ValueError('GPU UUID ordering changed')
            self.owns_training = True
            start = self.job['initial_step']
            step = self.job.get('continuation_step', start)
            current = self.job['initial_checkpoint']
            if self.job.get('inherited_baseline'):
                self.progress['baseline'] = self.job['inherited_baseline']
                self.progress['best'] = self.job.get('inherited_best', self.progress['baseline'])
            self.progress.update(initial_checkpoint=current, latest_checkpoint=current,step=step)
            try:
                admission = self.job.get('objective_admission')
                if admission and step == admission['step']:
                    self.check(current, step)
                    # Finish the existing boundary using the same checkpoint,
                    # plans and scorer, reusing each strictly verified suite.
                    record = self.evaluate(admission['validation_checkpoint'],
                                           admission['validation_label'])
                    comparison = compare_errors(self.progress['baseline']['errors'], record['errors'])
                    write_json(self.output/f'comparison_{step}.json', dict(
                        baseline=self.progress['baseline'], candidate=record, comparison=comparison,
                        interpretation='completed pre-adoption validation; same fixed full panels'))
                    self.remember_best(record)
                    if comparison['score_ratio'] > 1.5:
                        raise RuntimeError('pre-adoption validation error >1.5x baseline')
                while step < self.job['target_step']:
                    verify_files(self.job['protected_files'])
                    self.wait_for_selected_training_gpus()
                    self.maybe_expand(stack)
                    current,step=self.maybe_admit_performance(current,step)
                    if step>=self.job['target_step']:break
                    before = self.check(current, step)
                    world=len(self.job['physical_gpus']);accumulation=4//world
                    if world not in (2,4):raise ValueError('unsupported training GPU count')
                    scheduled_end = next_boundary(step, start, self.job['target_step'])
                    end=scheduled_end
                    objective_preflight = bool(admission and step == admission['step'])
                    if objective_preflight:
                        end = min(end, step+5)
                    resized=before['world_size']!=world or before['gradient_accumulation']!=accumulation
                    if resized:end=min(end,step+5)
                    run = self.output/'training'/f'continuation_{step}_{end}{self.job.get("attempt_suffix", "")}'
                    if run.exists():
                        raise FileExistsError(run)
                    options = {'loss_config': admission['loss_config_path']} if admission else {}
                    command = training_argv(self.job, current, self.job['data_config'], run, end, **options)
                    command += ['--gradient-accumulation', str(accumulation)]
                    command += self.job.get('performance_args', [])
                    # Close the gap between the controller's resource check and
                    # launcher preflight without restarting or sharing a run.
                    if not self.job.get('allow_shared'):
                        command += ['--wait-until-free']
                    self.job.update(run=str(run), selected_checkpoint=str(current), resume_step=step,
                                    phase_target_step=end, lr_plan=before['lr_plan'])
                    self.run_process(f'train_{end}', command)
                    candidate = str(run/'latest.pt')
                    after = self.check(candidate, end)
                    require_cursor_progress(before, after, end-step)
                    if (after['world_size'],after['gradient_accumulation'])!=(world,accumulation):
                        raise ValueError('training did not apply the selected execution layout')
                    health = continuation_health(self.control/f'train_{end}.log', step, end, world)
                    write_json(run/'health.json', health)
                    current, step = candidate, end
                    self.progress.update(latest_checkpoint=current, step=step)
                    write_json(self.output/'latest_training_checkpoint.json', dict(checkpoint=current, step=step))
                    if step == start + 5 or ((resized or objective_preflight) and step<scheduled_end):
                        self.progress['giant_ddp_preflight'] = health
                        if objective_preflight:
                            self.progress['objective_admission_proof'] = dict(
                                health=health, native_optimizer_and_rng_resumed=True,
                                before_cursor=before['batches_consumed'],
                                after_cursor=after['batches_consumed'],
                                lr_plan_unchanged=before['lr_plan']==after['lr_plan'],
                                dense_supervision=after['supervision'],
                                next_validation_step=scheduled_end,
                                dynamic_temporal_fraction=after['loss_config']['query_pair_options']['dynamic_temporal_fraction'])
                            write_json(self.control/'objective_admission_proof.json',
                                       self.progress['objective_admission_proof'])
                        if resized:
                            self.progress['execution_resize_proof']=dict(
                                previous_world=before['world_size'],world=world,gradient_accumulation=accumulation,
                                before_cursor=before['batches_consumed'],after_cursor=after['batches_consumed'],
                                effective_batch_size=4,unchanged_lr_plan=before['lr_plan']==after['lr_plan'])
                        self.publish('giant_preflight_passed')
                        continue
                    baseline = self.progress.get('baseline')
                    if baseline is None:
                        baseline = self.evaluate(self.job['baseline_checkpoint'], 'baseline_'+str(start))
                        self.progress['baseline'] = baseline
                        self.progress['best'] = baseline
                    record = self.evaluate(current, 'refinement_'+str(step))
                    comparison = compare_errors(baseline['errors'], record['errors'])
                    write_json(self.output/f'comparison_{step}.json', dict(
                        baseline=baseline, candidate=record, comparison=comparison,
                        interpretation='same fixed full panels; joint recipe, not a single-variable trial'))
                    self.remember_best(record)
                    # Stop gross divergence for review; ordinary metric noise
                    # does not trigger a lower-LR replay or dataset exclusion.
                    if comparison['score_ratio'] > 1.5:
                        raise RuntimeError('aggregate validation error >1.5x baseline; checkpoint retained for review')
                    self.prune_own_checkpoints(current)
                self.publish('completed_200000')
            except BaseException:
                self.progress['error'] = traceback.format_exc()
                self.publish('failed')
                raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller', required=True)
    RefinementController(p.parse_args().controller).run()
