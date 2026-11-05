"""Move a waiting controller onto idle GPUs, preserving its effective batch.

The handover only stops an identified idle supervisor with no child processes.
All model and training imports/children retain the original pinned runtime.
"""
from contextlib import contextmanager
from copy import deepcopy
import argparse
import ctypes
import fcntl
import importlib.util
import os
from pathlib import Path
import platform
import signal
import sys
import time
import traceback

# Load the scheduling helper by path, without pre-importing a different model
# or scripts package before the pinned runtime is selected below.
spec = importlib.util.spec_from_file_location('gpu_scheduling_helper',
    Path(__file__).with_name('recover_gpu_scheduling.py'))
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
read, write, sha = helper.read, helper.write, helper.sha


def pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    # Older conda Python builds lack the wrapper even on a recent kernel.
    # x86_64 Linux syscall number is checked against the server's headers.
    if sys.platform != 'linux' or platform.machine() != 'x86_64':
        raise RuntimeError('this platform needs native Python pidfd_open support')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if result < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    return int(result)


def require_waiting(job, progress, active, accepted, evaluated, children):
    if (job.get('stage') not in ('waiting_for_main_gpus', 'waiting_for_gpu_driver')
            or progress.get('stage') != job['stage'] or progress.get('child_pid') or children
            or not progress.get('pending_phase', '').startswith('train_')):
        raise ValueError('only a waiting training controller with no children can be reassigned')
    if any(active.get(k) != job.get(k) for k in ('pid', 'control_dir', 'run', 'stage')):
        raise ValueError('active training changed during reassignment')
    if Path(job['run']).exists() or progress.get('failed_quality_needs_review'):
        raise ValueError('training already started or quality review is pending')
    if any(accepted.get(k) != evaluated.get(k) for k in ('step', 'checkpoint', 'validation_contract')):
        raise ValueError('latest evaluated checkpoint is not accepted')
    decision = progress['decisions'][-1]
    if (decision['step'], decision['action']) != (accepted['step'], 'continue'):
        raise ValueError('accepted checkpoint did not pass quality review')
    if (job['resume_step'], job['selected_checkpoint']) != (accepted['step'], accepted['checkpoint']):
        raise ValueError('waiting controller is not resuming the accepted checkpoint')


@contextmanager
def stopped_waiter(job, output):
    """Use a pidfd to avoid PID reuse, and resume it on any preparation failure."""
    from scripts.cutover_native_motion_stereo import proc
    pid = job['pid']
    fd = pidfd_open(pid)
    stopped = False
    try:
        identity = proc(pid)
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if (not identity or identity['uid'] != os.getuid()
                or b'--kind' not in command or b'main' not in command
                or str(output).encode() not in command
                or not any(x.endswith(b'/recover_gpu_scheduling.py') for x in command)):
            raise ValueError('PID does not identify the expected waiting supervisor')
        signal.pidfd_send_signal(fd, signal.SIGSTOP)
        stopped = True
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            current = proc(pid)
            if current and current['state'] == 'T':
                break
            time.sleep(.05)
        else:
            raise RuntimeError('waiting supervisor did not stop for handover')
        yield fd, identity
    finally:
        if stopped:
            try:
                signal.pidfd_send_signal(fd, signal.SIGCONT)
            except ProcessLookupError:
                pass
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    old = read(args.job)
    sys.path.insert(0, old['repo'])
    from multicamera4rc.launch import gpu_snapshot
    from scripts.cutover_native_motion_stereo import proc
    from scripts.continue_extended_training import next_segment_end
    from scripts.continue_official_multicam import verify_admission
    from scripts.migrate_dynamic_validation import build_validation_contract, require_validation_contract
    from scripts.resume_official_after_resource_failure import ResourceRecoveryController, resize_receipt, effective_batch, execution_layout
    gpus = [int(v) for v in args.gpus.split(',')]
    root = Path(args.output_root).resolve()
    directory = Path(old['repo'])/'outputs/training_preflight'
    with (directory/'controller_handover.lock').open('a') as handover:
        fcntl.flock(handover, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with stopped_waiter(old, Path(old['control_dir']).parent) as (fd, identity):
            previous = read(Path(old['control_dir'])/'progress.json')
            current_job = read(args.job)
            active = read(old['main_active_path'])
            accepted = read(Path(old['output_root'])/'latest_accepted_checkpoint.json')
            evaluated = read(Path(old['output_root'])/'latest_evaluated_checkpoint.json')
            children = Path(f"/proc/{old['pid']}/task/{old['pid']}/children").read_text().split()
            require_waiting(current_job, previous, active, accepted, evaluated, children)
            old = current_job
            controller = ResourceRecoveryController(args.job)
            controller.progress = deepcopy(previous)
            historical = read(old['historical_record'])
            if build_validation_contract(old) != old['validation_contract']:
                raise ValueError('validation contract changed')
            require_validation_contract(old['validation_contract'], accepted, previous['best'], historical)
            for record in (accepted, previous['best'], historical):
                controller._verify_record(record)
            state = controller._check_candidate_state(accepted['checkpoint'], accepted['step'])
            verify_admission(old)
            for key in ('evidence_sha256', 'protected_files'):
                for path, expected in old[key].items():
                    if sha(path) != expected:
                        raise ValueError('frozen evidence changed: '+path)
            import subprocess
            head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=old['repo'], text=True).strip()
            if (head != old['commit'] or subprocess.check_output(
                    ['git','status','--porcelain'],cwd=old['repo'],text=True).strip()):
                raise ValueError('pinned runtime changed')
            batch = effective_batch(execution_layout(state))
            if not gpus or batch % len(gpus):
                raise ValueError('idle GPU count must divide the existing effective batch')
            accumulation = batch // len(gpus)
            resize = resize_receipt(state, accepted['checkpoint'], accepted['step'], gpus, accumulation)
            devices, processes = gpu_snapshot()
            occupied = {p.gpu_uuid for p in processes}
            if any(g not in devices or devices[g].uuid in occupied or devices[g].memory_used_mib >= 1024 for g in gpus):
                raise ValueError('requested GPUs are no longer completely idle')
            # These two independently pinned jobs retain their own assignments.
            protected = []
            for path in [Path('/data8/zbf/multicamera4rc_runs/intrinsics_structure_20261014/progress.json'),
                         Path('/data8/zbf/multicamera4rc_runs/cross_camera_delta_80000_82000_20261017/progress.json')]:
                assignment = read(path)
                protected.extend(assignment.get('physical_gpus', []))
            if set(gpus) & set(protected):
                raise ValueError('selected GPUs are leased to a protected experiment')
            proof = dict(passed=True, kind='idle_gpu_reassignment', wrapper=str(Path(__file__).resolve()),
                wrapper_sha256=sha(__file__), helper_sha256=sha(helper.__file__),
                previous_controller=args.job, previous_identity=identity, execution_resize=resize,
                checkpoint=accepted['checkpoint'], step=accepted['step'], runtime=old['repo'], commit=head,
                learning_rate_changed=False, reset_data_cursor=False, optimizer_restored=True,
                loss_config=state['loss_config'], lr_plan=state['lr_plan'], protected_gpus=protected)
            root.mkdir(parents=True, exist_ok=True)
            if args.verify_only:
                write(root/'verification.json', proof)
                print({'passed': True, 'step': accepted['step'], 'resize': resize})
                return
            control, output = root/'control', root/'training'
            if control.exists() or output.exists():
                raise ValueError('reassignment requires fresh output directories')
            control.mkdir(); output.mkdir()
            job = deepcopy(old)
            job.update(control_dir=str(control),output_root=str(output),run=str(output),
                physical_gpus=gpus, gradient_accumulation=accumulation, execution_resize=resize,
                initial_checkpoint=accepted['checkpoint'],initial_step=accepted['step'],
                selected_checkpoint=accepted['checkpoint'],resume_step=accepted['step'],
                environment=dict(old['environment'],CUDA_VISIBLE_DEVICES=','.join(devices[g].uuid for g in gpus)),
                log=str(root/'controller.log'),pid=os.getpid(),stage='reassigned_idle_gpus')
            write(control/'job.json',job);write(control/'previous_job.json',old)
            write(control/'previous_progress.json',previous);write(output/'resume_proof.json',proof)
            cls = helper.waiting_controller_type(ResourceRecoveryController, gpu_snapshot)
            successor = cls(control/'job.json')
            successor.progress = deepcopy(previous)
            for key in ('error','child_pid','child_log','pending_phase','busy_gpus'):
                successor.progress.pop(key,None)
            successor.progress.update(adopted_expected_state=state, idle_gpu_reassignment=proof)
            # Preparation has completed. Retire only the stopped idle parent;
            # no training, evaluation or other user's process is signalled.
            signal.pidfd_send_signal(fd, signal.SIGTERM)
            signal.pidfd_send_signal(fd, signal.SIGCONT)
            deadline = time.monotonic()+10
            while time.monotonic() < deadline:
                current = proc(old['pid'])
                if current is None or current['state'] == 'Z':
                    break
                time.sleep(.1)
            else:
                raise RuntimeError('old waiting controller did not exit')
        with (directory/'extended_training.lock').open('a') as execution:
            fcntl.flock(execution, fcntl.LOCK_EX | fcntl.LOCK_NB)
            successor.owns_training = True
            successor.publish('reassigned_idle_gpus')
            write(Path(old['control_dir'])/'superseded.json', dict(new_controller=str(control/'job.json'),pid=os.getpid(),proof=str(output/'resume_proof.json')))
            fcntl.flock(handover, fcntl.LOCK_UN)
            for name in ('latest_accepted_checkpoint.json','latest_evaluated_checkpoint.json'):
                write(output/name, accepted)
            try:
                current = accepted
                while current['step'] < job['target_step']:
                    end = next_segment_end(current['step'],job['target_step'],interval=job['validation_interval'])
                    current = successor.segment(current,successor.progress['best'],historical,successor.job['data_config'],end)
                    successor.remember_best(current)
                    write(output/'latest_accepted_checkpoint.json', current)
                successor.publish('completed_'+str(job['target_step']))
            except Exception:
                successor.progress['error'] = traceback.format_exc()
                successor.publish('failed')
                raise


if __name__ == '__main__':
    main()
