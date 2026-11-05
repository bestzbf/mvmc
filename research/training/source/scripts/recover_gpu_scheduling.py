"""Recover diagnosed GPU scheduling failures around unchanged, pinned runtimes.

Only controller-side GPU probes are adapted: an unavailable driver means no
available devices. No model, data, checkpoint, scorer or frozen job is edited.
Main training retains four ranks and resumes the last accepted optimizer/cursor.
"""
from contextlib import ExitStack
from copy import deepcopy
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unavailable_on_probe_error(probe, report, empty):
    """Fail closed on driver timeouts; never reuse an earlier idle snapshot."""
    def wrapped():
        try:
            return probe()
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
            command = error.cmd
            if not isinstance(command, (list, tuple)) or Path(command[0]).name != 'nvidia-smi':
                raise
            report(dict(at=time.time(), error=repr(error)))
            return {}, empty()
    return wrapped


def wait_idle(indices, identities, probe, publish, *, stable_seconds=60,
              interval=20, clock=time.monotonic, sleep=time.sleep):
    stable = None
    while True:
        devices, processes = probe()
        occupied = {p.gpu_uuid for p in processes}
        for index, expected in zip(indices, identities):
            if index in devices and devices[index].uuid != expected:
                raise ValueError('physical GPU identity changed; refusing to remap ranks')
        busy = [i for i in indices if i not in devices or devices[i].uuid in occupied
                or devices[i].memory_used_mib >= 1024]
        now = clock()
        if busy:
            stable = None
        elif stable is None:
            stable = now
        if stable is not None and now - stable >= stable_seconds:
            return
        publish(busy)
        sleep(interval)


def unstarted_resource_failure(run, log, code):
    """Retry only a launcher rejection that created no training outputs."""
    run = Path(run)
    if code != 2 or run.exists():
        return False
    text = Path(log).read_text()
    return ('Selected GPUs are busy; training was not started.' in text or
            ('nvidia-smi' in text and 'timed out after 20 seconds' in text))


def archive_wait_failure(root):
    root = Path(root)
    previous = read(root/'progress.json')
    if (previous.get('stage') != 'failed' or previous.get('phases')
            or previous.get('child_pid') or previous.get('completed')
            or 'nvidia-smi' not in previous.get('error', '')
            or 'TimeoutExpired' not in previous.get('error', '')):
        raise ValueError('expected a diagnosed pre-compute GPU probe timeout')
    target = root/f'progress_before_probe_recovery_{time.time_ns()}.json'
    write(target, previous)
    return str(target)


def receipt(kind, job_path, **fields):
    return dict(kind=kind, pid=os.getpid(), started_at=time.time(),
                wrapper=str(Path(__file__).resolve()), wrapper_sha256=sha(__file__),
                job_path=str(job_path), job_sha256=sha(job_path),
                model_data_and_scoring_unchanged=True, **fields)


def resume_trial(job_path):
    job = read(job_path)
    sys.path.insert(0, job['repo'])
    from scripts.intrinsics_structure_trial import StructureTrial
    from scripts import roi_ablation
    trial = StructureTrial(job_path)
    with (trial.output/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        archive = archive_wait_failure(trial.output)
        trial.check_code()
        proof = receipt('intrinsics', job_path, previous_progress=archive,
                        frozen_job_sha256=sha(trial.output/'frozen_job.json'))
        write(trial.output/'gpu_probe_recovery.json', proof)
        trial.progress.update(pid=os.getpid(), scheduling_recovery=proof)
        roi_ablation.gpu_snapshot = unavailable_on_probe_error(
            roi_ablation.gpu_snapshot,
            lambda error: trial.publish('waiting_for_gpu_driver', last_probe_error=error), list)
        try:
            trial.run()
        except Exception:
            trial.publish('failed', error=traceback.format_exc())
            raise
        finally:
            if trial.lease is not None:
                trial.lease.close()


def resume_pair(job_path):
    job = read(job_path)
    root = Path(job['output_root'])
    spec = importlib.util.spec_from_file_location('frozen_delta_queue', root/'supervisor.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # This separate lock protects the recovery check; Queue.run retains its own
    # original queue lock through inference/scoring and all idle waits.
    with (root/'probe_recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (root/'queue.lock').open('a') as original:
            fcntl.flock(original, fcntl.LOCK_EX | fcntl.LOCK_NB)
            archive = archive_wait_failure(root)
        queue = module.Queue(job_path)
        queue.verify()
        proof = receipt('benchmark', job_path, previous_progress=archive,
                        frozen_supervisor_sha256=sha(root/'supervisor.py'))
        write(root/'gpu_probe_recovery.json', proof)
        queue.progress['scheduling_recovery'] = proof
        module.snapshot = unavailable_on_probe_error(module.snapshot,
            lambda error: queue.publish('waiting_for_gpu_driver', last_probe_error=error), set)
        queue.run()


def waiting_controller_type(base, gpu_snapshot):
    class WaitingController(base):
        def run_process(self, name, command, *, evaluation=False):
            probe = unavailable_on_probe_error(gpu_snapshot, self.probe_error, list)
            indices = self.job['physical_gpus']
            identities = self.job['environment']['CUDA_VISIBLE_DEVICES'].split(',')
            if evaluation:
                indices, identities = indices[1:2], identities[1:2]
            while True:
                def publish(busy):
                    self.progress.update(pending_phase=name, busy_gpus=busy)
                    self.publish('waiting_for_main_gpus')
                wait_idle(indices, identities, probe, publish,
                          stable_seconds=0 if evaluation else 60)
                self.progress.pop('pending_phase', None)
                self.progress.pop('busy_gpus', None)
                try:
                    return super().run_process(name, command, evaluation=evaluation)
                except RuntimeError:
                    phase = self.progress['phases'][-1]
                    if (evaluation or not name.startswith('train_')
                            or phase['name'] != name
                            or not unstarted_resource_failure(self.job['run'], phase['log'], phase['exit_code'])):
                        raise
                    # Preserve the rejected attempt before the original runner
                    # opens its log again. No child rank/optimizer ever ran.
                    saved = Path(phase['log']).with_suffix(f'.resource_wait_{time.time_ns()}.log')
                    Path(phase['log']).rename(saved)
                    phase['log'] = str(saved)
                    self.progress.setdefault('resource_rejections', []).append(deepcopy(phase))

        def probe_error(self, error):
            self.progress['last_probe_error'] = error
            self.publish('waiting_for_gpu_driver')

    return WaitingController


def resume_main(job_path, output_root, verify_only=False):
    old = read(job_path)
    # Import the exact training runtime used for the accepted checkpoint.
    sys.path.insert(0, old['repo'])
    from multicamera4rc.launch import gpu_snapshot
    from scripts.resume_official_after_resource_failure import ResourceRecoveryController
    from scripts.resume_failed_dynamic_training import verify
    from scripts.continue_extended_training import next_segment_end
    from scripts.continue_official_multicam import verify_admission

    WaitingController = waiting_controller_type(ResourceRecoveryController, gpu_snapshot)
    directory = Path(old['repo'])/'outputs/training_preflight'
    with ExitStack() as stack:
        for name in ('controller_handover.lock', 'extended_training.lock'):
            handle = stack.enter_context((directory/name).open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if name == 'controller_handover.lock':
                handover = handle
        old, previous, accepted, historical, state, head = verify(job_path, ResourceRecoveryController)
        phase = previous['phases'][-1]
        if not unstarted_resource_failure(old['run'], phase['log'], phase['exit_code']):
            raise ValueError('main failure was not an unstarted GPU resource rejection')
        if len(old['physical_gpus']) != 4 or state.get('gradient_accumulation', 1) != 1:
            raise ValueError('recovery must retain the original four-rank execution')
        verify_admission(old)
        for path, expected in old['protected_files'].items():
            if sha(path) != expected:
                raise ValueError('protected experiment configuration changed')
        proof = receipt('main', job_path, checkpoint=accepted['checkpoint'], step=accepted['step'],
                        runtime=old['repo'], commit=head, physical_gpus=old['physical_gpus'],
                        learning_rate_changed=False, reset_data_cursor=False,
                        loss_config=state['loss_config'], lr_plan=state['lr_plan'])
        if verify_only:
            print(json.dumps(dict(passed=True, **proof)))
            return
        root = Path(output_root).resolve()
        control, output = root/'control', root/'training'
        if control.exists() or output.exists():
            raise ValueError('main recovery requires new output directories')
        control.mkdir(parents=True)
        output.mkdir(parents=True)
        job = deepcopy(old)
        job.update(control_dir=str(control), output_root=str(output), run=str(output),
                   initial_checkpoint=accepted['checkpoint'], initial_step=accepted['step'],
                   resume_step=accepted['step'], selected_checkpoint=accepted['checkpoint'],
                   data_config=accepted['data_config'], pid=os.getpid(),
                   log=str(root/'controller.log'), stage='resuming_after_gpu_contention')
        write(control/'job.json', job)
        write(control/'previous_job.json', old)
        write(control/'previous_progress.json', previous)
        write(output/'resume_proof.json', dict(passed=True, **proof))
        controller = WaitingController(control/'job.json')
        controller.progress = deepcopy(previous)
        for key in ('error', 'child_pid', 'child_log'):
            controller.progress.pop(key, None)
        controller.progress['scheduling_recovery'] = proof
        controller.owns_training = True
        controller.publish('resuming_after_gpu_contention')
        fcntl.flock(handover, fcntl.LOCK_UN)
        for name in ('latest_accepted_checkpoint.json', 'latest_evaluated_checkpoint.json'):
            write(output/name, accepted)
        try:
            current = accepted
            while current['step'] < job['target_step']:
                end = next_segment_end(current['step'], job['target_step'], interval=job['validation_interval'])
                current = controller.segment(current, controller.progress['best'], historical,
                                             controller.job['data_config'], end)
                controller.remember_best(current)
                write(output/'latest_accepted_checkpoint.json', current)
            controller.publish('completed_'+str(job['target_step']))
        except Exception:
            controller.progress['error'] = traceback.format_exc()
            controller.publish('failed')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('main', 'intrinsics', 'benchmark'), required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--output-root')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.kind == 'main':
        if not args.output_root and not args.verify_only:
            parser.error('main recovery requires --output-root')
        resume_main(args.job, args.output_root, args.verify_only)
    else:
        if args.verify_only or args.output_root:
            parser.error('output-root and verify-only apply to main recovery')
        (resume_trial if args.kind == 'intrinsics' else resume_pair)(args.job)
