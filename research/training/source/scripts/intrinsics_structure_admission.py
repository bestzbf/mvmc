"""Conditionally apply a validated recipe at a current main-training boundary.

No experimental weights overwrite newer main updates. A same-UID supervisor is
briefly suspended ONLY after its training children have finished. The complete
current checkpoint, Adam, RNG, cursor and LR are preserved. Compare the unchanged
checkpoint and config-only migration on the full panel; on failure resume the
old supervisor. Never stop any unrelated process or alter LR to force admission.
"""
from contextlib import ExitStack
from copy import deepcopy
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from multicamera4rc.config import MultiCamera4RCConfig
from multicamera4rc.datasets.mixture import read_data_config
from multicamera4rc.quality import compare_errors, error_panel
from scripts.continue_extended_training import next_segment_end
from scripts.cutover_native_motion_stereo import alive, proc, stop_tree, write_json
from scripts.intrinsics_trial import (CalibratedContinuation, calibration_decision, owned_tree)
from scripts.migrate_dynamic_validation import ExpandedDynamicController


def migrate_payload(payload, recipe, source_model_config, source_loss):
    """Metadata-only migration. All tensor dictionaries are reused untouched."""
    if MultiCamera4RCConfig.from_dict(payload['config']) != MultiCamera4RCConfig.from_dict(source_model_config):
        raise ValueError('main architecture changed since the paired experiment')
    state = payload['training_state']
    if state['loss_config'] != source_loss:
        raise ValueError('main objective changed since the paired experiment')
    result = dict(payload, config=dict(payload['config']), training_state=dict(state))
    if recipe['arm'] == 'gradient_isolation':
        expected = dict(source_loss, absolute_intrinsics_gradient='isolated')
        if recipe['settings']['loss'] != expected or recipe['settings']['intrinsics_sharing'] != 'none':
            raise ValueError('gradient admission is not a single intervention')
        result['training_state']['loss_config'] = expected
    elif recipe['arm'] == 'shared_intrinsics':
        if recipe['settings']['loss'] != source_loss or recipe['settings']['intrinsics_sharing'] != 'camera':
            raise ValueError('sharing admission is not a single intervention')
        result['config']['intrinsics_sharing'] = 'camera'
    else:
        raise ValueError('combined or unknown interventions cannot be admitted')
    return result


def boundary_decision(control, candidate):
    """Gains were established by retraining; current-weight migration must be safe."""
    camera = calibration_decision(control, candidate)
    physical = compare_errors(error_panel(control), error_panel(candidate))
    safe = (camera['macro_log_focal_ratio'] <= 1.01 and not camera['regressed_families']
            and physical['score_ratio'] <= 1.01 and physical['worst_metric_ratio'] <= 1.05
            and all(v <= 1.05 for v in camera['syn4d_metric_ratios'].values())
            and camera['mean_visibility_f1_change'] >= -.01)
    return dict(passed=bool(safe), calibration=camera, physical=physical,
                requirement='paired training gains already passed; no regression at latest main weights')


class StructureContinuation(CalibratedContinuation):
    def _check_candidate_state(self, checkpoint, step):
        import torch
        state = ExpandedDynamicController._check_candidate_state(self, checkpoint, step)
        payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
        if (payload['config'].get('intrinsics_sharing', 'none') != self.job['admitted_intrinsics_sharing']
                or state.get('augmentation') != self.job['admitted_augmentation']
                or state.get('gradient_accumulation', 1) != 1):
            raise ValueError('main continuation changed the admitted intervention')
        return state


def _training_processes(tree):
    return [p for p in tree.values() if any(x in p['cmd'] for x in
        ('-m multicamera4rc.train', '-m multicamera4rc.launch', 'torch.distributed.run'))]


def watchdog_tick(path):
    """Resume the recorded old supervisor if the admission process disappears."""
    record = json.loads(Path(path).read_text())
    if record.get('state') == 'finished':
        return False
    owner, old = record['owner'], record['old']
    if alive(owner) and time.time() < record['deadline']:
        return True
    current = proc(old['pid'])
    if (current and current['start'] == old['start'] and current['cmd'] == old['cmd']
            and current['uid'] == os.getuid() and current['state'] in ('T','t')):
        os.kill(old['pid'], signal.SIGCONT)
    write_json(Path(path), dict(record, state='finished', recovered_original_main=True))
    return False


def start_watchdog(trial, identity):
    path = trial.output/'boundary_watchdog.json'
    write_json(path, dict(state='guarding', owner=proc(os.getpid()), old=identity,
                         deadline=time.time()+4*3600))
    with (trial.output/'boundary_watchdog.log').open('a') as log:
        subprocess.Popen([trial.job['python'], '-m', 'scripts.intrinsics_structure_admission',
            '--watchdog', str(path)], cwd=trial.root,
            env=dict(trial.environment, CUDA_VISIBLE_DEVICES=''), stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    return path


def adopt_at_main_boundary(trial, recipe):
    import torch
    decision = json.loads((trial.output/'decision.json').read_text())
    if decision['selected'] != recipe['arm'] or not decision['comparisons'][recipe['arm']]['passed']:
        raise ValueError('admission requires completed independent validation gains')
    benchmark_guard = json.loads(Path(recipe['pstudio_admission_guard']).read_text())
    if not benchmark_guard['passed'] or benchmark_guard['selected'] != recipe['arm']:
        raise ValueError('PStudio regression veto did not pass for the selected arm')
    active_path = Path(trial.job['main_active_path'])
    # Do not interrupt training or choose a stale checkpoint. Wait for a phase
    # where the current segment is fully trained and main is validating it.
    while True:
        trial.check_code()
        active = json.loads(active_path.read_text())
        identity = proc(active['pid'])
        if not identity or not alive(identity):
            trial.publish('passed_main_not_running_requires_review')
            return
        progress = json.loads((Path(active['control_dir'])/'progress.json').read_text())
        if progress.get('stage', '').startswith('validate_'):
            tree = owned_tree(identity)
            if not _training_processes(tree):
                break
        trial.publish('passed_waiting_current_main_validation_boundary', main_stage=progress.get('stage'))
        time.sleep(20)

    lock_path = Path(active['repo'])/'outputs/training_preflight/controller_handover.lock'
    with lock_path.open('a') as handover:
        fcntl.flock(handover, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if json.loads(active_path.read_text())['pid'] != identity['pid']:
            raise ValueError('main ownership changed before admission')
        owned_tree(identity)  # Verify exact process start/cmd/UID before signal.
        guard_path = start_watchdog(trial, identity)
        stopped = False
        try:
            os.kill(identity['pid'], signal.SIGSTOP)
            tree = owned_tree(identity)
            if _training_processes(tree):
                trial.publish('passed_boundary_raced_training_main_unchanged')
                return
            # Existing validation child may finish; the suspended parent cannot
            # launch a new training segment or mutate latest.pt in this interval.
            deadline = time.monotonic()+1800
            while any(p['pid'] != identity['pid'] and alive(p) for p in tree.values()):
                if time.monotonic() > deadline:
                    raise TimeoutError('current validation child did not exit; resume original main')
                time.sleep(5)
            active = json.loads(active_path.read_text())
            checkpoint = Path(active['run'])/'latest.pt'
            payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
            step, state = payload['step'], payload['training_state']
            accepted = json.loads((Path(active['output_root'])/'latest_accepted_checkpoint.json').read_text())
            if (step != active['phase_target_step'] or step < accepted['step']
                    or state['lr_plan'] != trial.job['source_lr_plan']
                    or state['world_size'] != 4 or state['batch_size'] != 1
                    or read_data_config(active['data_config']) != read_data_config(trial.job['data_config'])):
                raise ValueError('current boundary no longer matches the validated data/LR contract')
            migrated = migrate_payload(payload, recipe, trial.job['source_model_config'],
                                        trial.settings['control']['loss'])
            directory = trial.output/'main_boundary'; directory.mkdir(exist_ok=True)
            candidate = directory/f'admitted_{step}.pt'
            temporary = candidate.with_suffix('.tmp')
            torch.save(migrated, temporary); temporary.replace(candidate)
            expected_state = deepcopy(migrated['training_state'])
            admitted_sharing = migrated['config'].get('intrinsics_sharing', 'none')
            del migrated, payload
            original, original_entries = trial.evaluate(checkpoint, f'boundary_control_{step}', step=step)
            record, entries = trial.evaluate(candidate, f'boundary_candidate_{step}', step=step)
            guard = boundary_decision(original_entries, entries)
            write_json(directory/'decision.json', dict(guard, source_checkpoint=str(checkpoint),
                candidate_checkpoint=str(candidate), step=step, learning_rate_changed=False,
                tensor_weights_changed=False, optimizer_and_rng_preserved=True))
            if not guard['passed']:
                trial.publish('paired_passed_boundary_failed_main_unchanged', boundary_comparison=guard)
                return
            now_active = json.loads(active_path.read_text())
            if (now_active['pid'] != identity['pid'] or now_active['run'] != active['run']
                    or now_active['phase_target_step'] != step
                    or proc(identity['pid'])['state'] not in ('T','t')
                    or _training_processes(owned_tree(identity))):
                raise ValueError('main changed during current-boundary validation')
            control = trial.output/'continuation_control'; control.mkdir(exist_ok=True)
            output = trial.output/'continued_training'; output.mkdir(exist_ok=True)
            (trial.root/'outputs/training_preflight').mkdir(parents=True, exist_ok=True)
            job = deepcopy(active); job.pop('audited_data_config_replacement', None)
            job.update(repo=str(trial.root), environment=dict(active['environment'], PYTHONPATH=str(trial.root)),
                control_dir=str(control), output_root=str(output), run=str(output), pid=os.getpid(),
                initial_checkpoint=str(candidate), initial_step=step, resume_step=step,
                selected_checkpoint=str(candidate), commit=trial.job['commit'], max_quality_retries=0,
                admitted_intrinsics_sharing=admitted_sharing, admitted_augmentation=state.get('augmentation'),
                intrinsics_structure_trial=str(trial.output/'decision.json'))
            write_json(control/'job.json', job)
            controller = StructureContinuation(control/'job.json'); controller.active = active_path
            controller.progress = deepcopy(progress)
            for key in ('child_pid','child_log','error','failed_quality_needs_review'):
                controller.progress.pop(key, None)
            record.update(data_config=active['data_config'])
            controller.progress.update(adopted_expected_state=expected_state,
                restored_training_state=expected_state, resumed_training_state=expected_state, best=original)
            controller._check_candidate_state(candidate, step)
            historical = json.loads(Path(active['historical_record']).read_text())
            tree = owned_tree(identity)
            stop_tree(tree); stopped = True
            write_json(Path(active['control_dir'])/'transition.json', dict(stage='intrinsics_structure_adopted',
                new_control_dir=str(control), checkpoint=str(candidate), stopped_pids=list(tree)))
            controller.owns_training = True
            controller.publish('intrinsics_structure_adopted')
            write_json(output/'admission.json', dict(passed=True, recipe=recipe, boundary=guard,
                source_checkpoint=str(checkpoint), checkpoint=str(candidate), step=step,
                experimental_weights_used=False, lr_changed=False))
        finally:
            if not stopped and alive(identity):
                os.kill(identity['pid'], signal.SIGCONT)
            guard = json.loads(guard_path.read_text())
            write_json(guard_path, dict(guard, state='finished'))

    if trial.lease is not None:
        trial.lease.close(); trial.lease = None
    trial.publish('adopted_main_training', continuation_control=str(control))
    with ExitStack() as stack:
        for root in sorted({Path(active['repo']),trial.root}):
            handle = stack.enter_context((root/'outputs/training_preflight/extended_training.lock').open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = record
        write_json(output/'latest_accepted_checkpoint.json', current)
        write_json(output/'latest_evaluated_checkpoint.json', current)
        while current['step'] < job['target_step']:
            end = next_segment_end(current['step'], job['target_step'], interval=job['validation_interval'])
            current = controller.segment(current, controller.progress['best'], historical, job['data_config'], end)
            controller.remember_best(current)
            write_json(output/'latest_accepted_checkpoint.json', current)
        controller.publish('completed_'+str(job['target_step']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watchdog', required=True)
    args = parser.parse_args()
    while watchdog_tick(args.watchdog):
        time.sleep(10)
