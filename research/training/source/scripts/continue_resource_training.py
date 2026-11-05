"""Explicitly authorized batch-size/resource migration of the native main run.

Model, optimizer, schedule, dense targets and validation are inherited.  Only
the execution layout changes.  The previous native checkpoint is never edited.
"""
from contextlib import ExitStack
from copy import deepcopy
import argparse
import fcntl
import math
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from multicamera4rc.config import MultiCamera4RCConfig
from multicamera4rc.launch import gpu_snapshot
from multicamera4rc.quality import compare_errors
from scripts.continue_refinement_training import RefinementController, next_boundary
from scripts.continue_quality_recovery import training_argv, validate_checkpoint
from scripts.cutover_native_motion_stereo import alive, write_json
from scripts.migrate_dynamic_validation import require_fixed_training_state, ExpandedDynamicController
from scripts.query_pair_main_admission import clean_runtime, read, verify_files, free_indices
from scripts.native_training_log import training_rows


def execution(job):
    gpus = job['physical_gpus']
    batch = job['global_batch_size']
    if (not gpus or len(set(gpus)) != len(gpus) or any(type(x) is not int or x < 0 for x in gpus)
            or type(batch) is not int or batch < len(gpus) or batch % len(gpus)):
        raise ValueError('effective batch must be a positive multiple of the selected GPU count')
    authorization = job['resource_migration']
    if (authorization.get('approved') is not True or authorization['gpus'] != gpus
            or authorization['effective_batch_size'] != batch
            or authorization.get('preserve_lr') is not True):
        raise ValueError('execution layout differs from explicit resource authorization')
    return len(gpus), batch // len(gpus), batch


def expected_state(job, state, checkpoint, step):
    world, accumulation, batch = execution(job)
    original = job['resource_migration']['source_layout']
    initial = (step == job['continuation_step']
               and Path(checkpoint).resolve() == Path(job['initial_checkpoint']).resolve())
    pending = job.get('pending_boundary_validation', {})
    activation = job.get('data_activation', {})
    historical = (step == pending.get('begin')
        and Path(checkpoint).resolve() == Path(pending['source_checkpoint']).resolve())
    reanchored = (activation.get('reset_completed_in_checkpoint') is True
        and step == activation['step']
        and Path(checkpoint).resolve() == Path(activation['balanced_reset_checkpoint']).resolve())
    initial = initial or historical or reanchored
    wanted = original if initial else dict(world_size=world, batch_size=1,
        gradient_accumulation=accumulation, effective_batch_size=batch)
    if any(state.get(k) != v for k, v in wanted.items()):
        raise ValueError('checkpoint execution layout differs from the admitted migration')
    result = deepcopy(job['expected_state'])
    result.update(wanted)
    return result


def cursor_progress(before, after, updates, batch):
    # DistributedSampler pads to a multiple of world_size (e.g. 100002 for
    # 100000 samples on three ranks). Count that real padded epoch length.
    consumed = lambda s: (s['data_epoch'] * ((s['dataset_size'] + s['world_size'] - 1)
        // s['world_size']) * s['world_size']
        + s['batches_consumed'] * s['world_size'] * s['batch_size'])
    if (before['dataset_size'] != after['dataset_size']
            or consumed(after) - consumed(before) != updates * batch):
        raise ValueError('resource migration skipped or repeated samples')


def training_health(path, begin, end, world, accumulation, batch):
    rows = training_rows(path)
    if [r['step'] for r in rows] != list(range(begin + 1, end + 1)):
        raise ValueError('missing or duplicate native optimizer updates')
    for row in rows:
        if (row['world_size'] != world or row['gradient_accumulation'] != accumulation
                or row['effective_batch_size'] != batch or len(row['rank_batches']) != batch
                or row['optimizer_step_skipped'] or row['supervision'] != 'dense'):
            raise ValueError('layout, update or dense supervision contract changed')
        for key in ('loss', 'grad_norm', 'lr', 'backbone_lr', 'refinement_lr'):
            if not math.isfinite(row[key]):
                raise ValueError('nonfinite native metric: ' + key)
        for sample in row['rank_batches']:
            metrics = sample['metrics']
            if any(not math.isfinite(v) for v in metrics.values() if isinstance(v, (int, float))):
                raise ValueError('nonfinite sample metric')
            geometry = 'depth_abs_rel' in metrics and 'depth_mae_normalized' in metrics
            camera = 'intrinsics_log_focal_error' in metrics and 'camera_rotation_degrees' in metrics
            if metrics.get('supervised_source_pixels', 0) <= 0 and not geometry and not camera:
                raise ValueError('sample has no admitted dense geometry or camera supervision')
    return dict(passed=True, begin=begin, end=end, updates=len(rows), world_size=world,
        effective_batch_size=batch, microbatches=batch * len(rows), all_updates_applied=True,
        log=str(path), last_loss=rows[-1]['loss'])


class ResourceContinuationController(RefinementController):
    def snapshot(self):
        """A failed driver query must never be mistaken for a free GPU."""
        while True:
            try:
                result = gpu_snapshot()
                self.progress.pop('gpu_query_error', None)
                return result
            except (OSError, subprocess.SubprocessError) as error:
                self.progress['gpu_query_error'] = repr(error)
                self.publish('waiting_for_gpu_query')
                time.sleep(10)

    def evaluate(self, checkpoint, label):
        from scripts.resource_validation import evaluate
        return evaluate(self, checkpoint, label)

    def run_process(self, name, command, *, evaluation=False):
        if evaluation:
            return super().run_process(name, command, evaluation=evaluation)
        from scripts.resource_resilience import run_training_process
        return run_training_process(self, name, command)

    def wait_for_selected_training_gpus(self):
        while True:
            devices, apps = self.snapshot()
            selected = self.job['physical_gpus']
            if any(i not in devices or devices[i].uuid != self.job['allowed_gpu_uuids'][str(i)] for i in selected):
                raise ValueError('authorized GPU identity changed')
            free = free_indices(devices, apps)
            if set(selected) <= free:
                self.progress.pop('resource_wait', None)
                return
            self.progress['resource_wait'] = dict(physical_gpus=selected, available_gpus=sorted(free))
            self.publish('waiting_for_training_gpus')
            time.sleep(10)

    def wait_for_checkpoint_space(self, checkpoint, begin, end):
        # No deletion of unrelated jobs or frozen evidence. Existing retention
        # runs at validated boundaries; wait if other jobs consume the reserve.
        size = Path(checkpoint).stat().st_size
        required = ((end - begin + 99) // 100 + 3) * size + 32 * 2**30
        while shutil.disk_usage(self.output).free < required:
            self.progress['disk_wait'] = dict(required_bytes=required,
                free_bytes=shutil.disk_usage(self.output).free, output=str(self.output))
            self.publish('waiting_for_checkpoint_space')
            time.sleep(30)
        self.progress.pop('disk_wait', None)

    def check(self, checkpoint, step):
        import torch
        payload = torch.load(checkpoint, mmap=True, map_location='cpu', weights_only=True)
        state = payload['training_state']
        expected = expected_state(self.job, state, checkpoint, step)
        validate_checkpoint(checkpoint, step, world_size=state['world_size'], batch_size=1)
        require_fixed_training_state(expected, state)
        if state.get('load_balancing') != expected.get('load_balancing'):
            raise ValueError('checkpoint load-balancing contract changed')
        if (state['supervision'] != 'dense' or state.get('augmentation') != expected.get('augmentation')
                or MultiCamera4RCConfig.from_dict(payload['config']) != MultiCamera4RCConfig.from_dict(self.job['model_config'])
                or len(payload['optimizer']['param_groups']) != 3):
            raise ValueError('model, optimizer or supervision contract changed')
        return state

    def wait_for_initial_training_gpus(self, stack):
        execution(self.job)
        selected = self.job['physical_gpus']
        expected = self.job['allowed_gpu_uuids']
        while True:
            devices, apps = self.snapshot()
            if any(i not in devices or devices[i].uuid != expected[str(i)] for i in selected):
                raise ValueError('authorized GPU identity changed')
            if set(selected) <= free_indices(devices, apps):
                with ExitStack() as tentative:
                    handles = []
                    for i in selected:
                        lease = self.device_lease(i, devices[i].uuid)
                        if lease is None:
                            break
                        handles.append(tentative.enter_context(lease))
                    if len(handles) == len(selected):
                        fresh, processes = self.snapshot()
                        if (set(selected) <= free_indices(fresh, processes)
                                and all(fresh[i].uuid == expected[str(i)] for i in selected)):
                            stack.enter_context(tentative.pop_all())
                            self.progress.pop('resource_wait', None)
                            return fresh
            self.progress['resource_wait'] = dict(physical_gpus=selected,
                available_gpus=sorted(free_indices(devices, apps)))
            self.publish('waiting_for_training_gpus')
            time.sleep(10)

    def activate_pending_data(self, checkpoint, step, record):
        pending = self.job.get('pending_kinoptic')
        if not pending or self.job.get('data_activation') or step != pending['step']:
            return
        from scripts.admit_kinoptic_training import verify_ready, sha, GT_SOURCE_FILES
        original = read(pending['frozen_job'])
        verify_files({pending['frozen_job']: pending['sha256']})
        verify_ready(original)
        # The original proof remains bound to its audited checkout. Every GT
        # dependency and native training file in this new checkout is identical.
        from scripts.qualify_kinoptic_dynamic import _runtime_sources
        files = set(_runtime_sources(Path(original['repo']))) | set(original['critical_runtime_sha256']) | set(GT_SOURCE_FILES)
        if any(sha(self.root / name) != sha(Path(original['repo']) / name) for name in files):
            raise ValueError('audited data or native training code changed during resource migration')
        if ExpandedDynamicController._verify_record(self, record) != step:
            raise ValueError('pending data requires the original complete boundary validation')
        before = self.check(checkpoint, step)
        activation = dict(step=step, checkpoint=str(checkpoint), validation_record=record,
            old_data_config=self.job['data_config'], new_data_config=original['candidate_data_config'],
            old_data_epoch=before['data_epoch'], old_batches_consumed=before['batches_consumed'],
            new_data_epoch=0, new_batches_consumed=0, reset_data_cursor=True,
            global_step_preserved=True, lr_plan_unchanged=True, loss_config_unchanged=True,
            data_transition=original['data_transition'])
        self.job.update(data_activation=activation, data_config=original['candidate_data_config'])
        self.job['reset_data_cursor_once'] = True
        write_json(self.control / 'data_activation.json', activation)
        self.publish('kinoptic_ready_after_original_validation')

    def finish_boundary(self, checkpoint, step):
        record = self.evaluate(checkpoint, 'refinement_' + str(step))
        comparison = compare_errors(self.progress['baseline']['errors'], record['errors'])
        write_json(self.output / f'comparison_{step}.json', dict(
            baseline=self.progress['baseline'], candidate=record, comparison=comparison))
        self.remember_best(record)
        if comparison['score_ratio'] > 1.5:
            raise RuntimeError('aggregate validation error >1.5x baseline; checkpoint retained')
        self.activate_pending_data(checkpoint, step, record)
        self.prune_own_checkpoints(checkpoint)
        return record

    def recover_pending_boundary(self, checkpoint, step):
        """Finish validation after a successful segment's controller failed.

        Never replay its optimizer updates or skip directly to the next node.
        The completed process, full checkpoints, exact update log and sample
        cursor must all agree before its boundary can be recovered.
        """
        pending = self.job.get('pending_boundary_validation')
        if not pending:
            return
        begin = pending['begin']
        if (step != pending['step'] or step != self.job['continuation_step']
                or not Path(checkpoint).samefile(pending['checkpoint'])
                or not Path(checkpoint).samefile(self.job['initial_checkpoint'])
                or begin >= step
                or next_boundary(begin, self.job['initial_step'], self.job['target_step']) != step):
            raise ValueError('recovery must match the completed original validation boundary')
        source_progress = read(pending['source_progress'])
        phases = [p for p in source_progress['phases'] if p['name'] == f'train_{step}']
        if (len(phases) != 1 or phases[0]['exit_code'] != 0
                or not Path(phases[0]['log']).samefile(pending['training_log'])):
            raise ValueError('boundary recovery requires a successful original training process')
        before = self.check(pending['source_checkpoint'], begin)
        after = self.check(checkpoint, step)
        # The completed segment used the checkpoint's original layout. Its
        # validation may now run on a different explicitly authorized GPU set.
        world, accumulation, batch = (after['world_size'], after['gradient_accumulation'],
                                     after['effective_batch_size'])
        health = training_health(pending['training_log'], begin, step, world, accumulation, batch)
        cursor_progress(before, after, step - begin, batch)
        receipt = dict(health=health, checkpoint=str(checkpoint), source_checkpoint=pending['source_checkpoint'],
            original_process_exit_code=0, sample_cursor_verified=True, optimizer_updates_replayed=False)
        write_json(self.control / 'recovered_training_health.json', receipt)
        self.progress.update(step=step, latest_checkpoint=str(checkpoint))
        write_json(self.output / 'latest_training_checkpoint.json', dict(step=step, checkpoint=str(checkpoint)))
        self.publish('recovering_boundary_validation')
        record = self.finish_boundary(checkpoint, step)
        receipt.update(validation_record=record, passed=True)
        write_json(self.control / 'recovered_boundary.json', receipt)
        self.job.pop('pending_boundary_validation')
        self.publish('boundary_recovered')

    def run(self):
        world, accumulation, batch = execution(self.job)
        with ExitStack() as stack:
            for directory in sorted({self.root / 'outputs/training_preflight', self.active.parent}):
                directory.mkdir(parents=True, exist_ok=True)
                handle = stack.enter_context((directory / 'extended_training.lock').open('a'))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            clean_runtime(self.root, self.job['commit'])
            verify_files(self.job['protected_files'])
            if any(alive(p) for p in self.job['stopped_identities']):
                raise ValueError('a previous training process is still active')
            current, step = self.job['initial_checkpoint'], self.job['continuation_step']
            self.check(current, step)
            self.progress.update(step=step, latest_checkpoint=current,
                baseline=self.job['inherited_baseline'], best=self.job['inherited_best'])
            self.wait_for_initial_training_gpus(stack)
            self.owns_training = True
            try:
                if self.job.get('performance_admission') and not self.job.get('performance_admission_attempted'):
                    from scripts.resource_performance_admission import admit
                    current, step = admit(self, current, step)
                self.recover_pending_boundary(current, step)
                while step < self.job['target_step']:
                    verify_files(self.job['protected_files'])
                    self.wait_for_selected_training_gpus()
                    before = self.check(current, step)
                    if self.job.get('reset_data_cursor_once') and before.get('load_balancing'):
                        from scripts.resource_performance_admission import reanchor_data
                        current = reanchor_data(self, current, step)
                        before = self.check(current, step)
                    boundary = next_boundary(step, self.job['initial_step'], self.job['target_step'])
                    resized = (before['world_size'], before['gradient_accumulation']) != (world, accumulation)
                    end = min(boundary, step + 5) if resized else boundary
                    retry = self.job.get('transient_attempts', {}).get(str(boundary), 0)
                    suffix = self.job.get('attempt_suffix', '_fivegpu_20261026')
                    if retry:
                        suffix += f'_retry{retry}'
                    run = self.output / 'training' / f'continuation_{step}_{end}{suffix}'
                    if run.exists():
                        raise FileExistsError(run)
                    reset = self.job.get('reset_data_cursor_once', False)
                    command = training_argv(self.job, current, self.job['data_config'], run, end,
                        loss_config=self.job['loss_config_path'], reset_cursor=reset)
                    command += ['--gradient-accumulation', str(accumulation), '--save-every', '100', '--wait-until-free']
                    self.wait_for_checkpoint_space(current, step, end)
                    phase = f'train_{end}' + (f'_retry{retry}' if retry else '')
                    self.job.update(run=str(run), selected_checkpoint=str(current), resume_step=step,
                        phase_target_step=end, gradient_accumulation=accumulation, training_phase=phase)
                    from scripts.resource_resilience import TrainingProcessFailure, recover_training_failure
                    try:
                        self.run_process(phase, command)
                    except TrainingProcessFailure as error:
                        current, step = recover_training_failure(self, error, current, step, end, boundary, before)
                        continue
                    candidate = str(run / 'latest.pt')
                    after = self.check(candidate, end)
                    origin = dict(before, data_epoch=0, batches_consumed=0) if reset else before
                    cursor_progress(origin, after, end - step, batch)
                    health = training_health(self.control / f'{phase}.log', step, end, world, accumulation, batch)
                    write_json(run / 'health.json', health)
                    if resized:
                        proof = dict(passed=True, source_checkpoint=str(current), target_checkpoint=candidate,
                            previous_world=before['world_size'], world=world,
                            previous_effective_batch=before['effective_batch_size'], effective_batch_size=batch,
                            lr_plan_unchanged=before['lr_plan'] == after['lr_plan'],
                            data_cursor_preserved=True, native_optimizer_resumed=True, health=health)
                        write_json(self.control / 'execution_resize_proof.json', proof)
                    self.job.pop('reset_data_cursor_once', None)
                    current, step = candidate, end
                    self.progress.update(step=step, latest_checkpoint=current)
                    write_json(self.output / 'latest_training_checkpoint.json', dict(step=step, checkpoint=current))
                    if step < boundary:
                        self.publish('giant_preflight_passed')
                        continue
                    self.finish_boundary(current, step)
                self.publish('completed_200000')
            except BaseException:
                self.progress['error'] = traceback.format_exc()
                self.publish('failed')
                raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', required=True)
    ResourceContinuationController(parser.parse_args().controller).run()
