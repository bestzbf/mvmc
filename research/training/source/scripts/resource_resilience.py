"""Bounded infrastructure recovery; never retry bad data or quality failures."""
import json
import os
from pathlib import Path
import re
import subprocess
import time

from scripts.cutover_native_motion_stereo import write_json, proc
from scripts.native_training_log import training_rows


class TrainingProcessFailure(RuntimeError):
    def __init__(self, code, log):
        self.code, self.log = code, Path(log)
        super().__init__(f'training process exited {code}; see {log}')


def log_tail(path, limit=512 * 1024):
    with Path(path).open('rb') as stream:
        stream.seek(max(0, Path(path).stat().st_size - limit))
        return stream.read().decode(errors='replace')


def transient_failure(code, path):
    """Whitelist transport failures, fail closed on signals/data/numerics/OOM."""
    if code <= 0:
        return False
    text = log_tail(path)
    forbidden = ('KeyboardInterrupt', 'SignalException', 'Received 15 death signal',
        'Received 2 death signal', 'out of memory', 'OutOfMemoryError', 'device-side assert',
        'illegal memory access', 'AssertionError', 'ValueError:', 'KeyError:',
        'FileNotFoundError', 'PermissionError', 'No space left on device',
        'nonfinite', 'non-finite', '"loss": NaN', '"loss": Infinity')
    if any(word.lower() in text.lower() for word in forbidden):
        return False
    return any(word in text for word in (
        'Watchdog caught collective operation timeout',
        'NCCL watchdog thread terminated with exception',
        'DistNetworkError', 'ncclRemoteError',
        'socketStartConnect: Connect to', 'Connection reset by peer'))


def last_observed_update(path):
    """Best-effort live display only; checkpoint admission uses strict parsing."""
    text, decoder, last = log_tail(path), json.JSONDecoder(), None
    for match in re.finditer(r'\{\s*"step"\s*:', text):
        try:
            row, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue  # The writer may still be printing the last record.
        if type(row.get('step')) is int and 'loss' in row:
            last = {key: row.get(key) for key in ('step', 'loss', 'grad_norm', 'lr',
                                                'backbone_lr', 'refinement_lr')}
    return last


def run_training_process(controller, name, command):
    controller.publish(name)
    path = controller.control / (name + '.log')
    started = time.monotonic()
    with path.open('x') as log:
        child = subprocess.Popen(command, cwd=controller.root,
            env=os.environ | controller.job['environment'], stdout=log, stderr=subprocess.STDOUT)
        controller.progress.update(child_pid=child.pid, child_log=str(path))
        controller.publish(name)
        while True:
            try:
                code = child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                controller.progress['phase_seconds'] = time.monotonic() - started
                controller.progress['child_identity'] = proc(child.pid)
                # Observability must not kill a healthy optimizer.
                try:
                    row = last_observed_update(path)
                    if row:
                        controller.progress['last_observed_update'] = row
                    controller.progress['log_mtime'] = path.stat().st_mtime
                except (OSError, ValueError) as error:
                    controller.progress['observer_error'] = repr(error)
                controller.publish(name)
    controller.progress['phases'].append(dict(name=name, exit_code=code,
        seconds=time.monotonic() - started, log=str(path), argv=command))
    controller.progress.pop('child_pid', None)
    controller.progress.pop('child_identity', None)
    controller.publish(name)
    if code:
        raise TrainingProcessFailure(code, path)


def recover_training_failure(controller, error, checkpoint, begin, end, boundary, before):
    """Resume the newest complete save after at most two transport retries."""
    from scripts.continue_resource_training import execution, cursor_progress, training_health
    from scripts.query_pair_main_admission import verify_files, clean_runtime
    attempts = controller.job.setdefault('transient_attempts', {})
    count = attempts.get(str(boundary), 0)
    if count >= 2 or not transient_failure(error.code, error.log):
        raise error
    if controller.job.get('pause_requested') or (controller.control / 'PAUSE').exists():
        raise RuntimeError('explicit pause suppresses automatic recovery') from error
    clean_runtime(controller.root, controller.job['commit'])
    verify_files(controller.job['protected_files'])
    world, accumulation, batch = execution(controller.job)
    rows = training_rows(error.log)
    if rows:
        # Check even the uncommitted updates; never hide numerical failures.
        training_health(error.log, begin, rows[-1]['step'], world, accumulation, batch)
        if rows[-1]['step'] > end:
            raise ValueError('failed segment logged updates beyond its boundary')
    run = Path(controller.job['run'])
    saves = sorted((int(p.stem.split('_')[1]), p) for p in run.glob('step_*.pt'))
    if (run / 'latest.pt').exists():
        import torch
        latest = torch.load(run / 'latest.pt', mmap=True, map_location='cpu', weights_only=True)
        saves.append((int(latest['step']), run / 'latest.pt'))
        del latest
        saves.sort()
    step, candidate = (begin, Path(checkpoint))
    if saves:
        saved_step, saved_path = saves[-1]
        if not begin < saved_step <= end or saved_step > (rows[-1]['step'] if rows else begin):
            raise ValueError('checkpoint does not match completed logged updates')
        after = controller.check(saved_path, saved_step)
        cursor_progress(before, after, saved_step - begin, batch)
        step, candidate = saved_step, saved_path
    else:
        controller.check(candidate, begin)
    # Completion belongs to the boundary handler, not a no-op/replayed train.
    if step == end:
        raise RuntimeError('failed launcher saved the full boundary; requires exit diagnosis') from error
    attempts[str(boundary)] = count + 1
    event = dict(attempt=count + 1, boundary=boundary, failed_log=str(error.log),
        failed_exit_code=error.code, resume_checkpoint=str(candidate), resume_step=step,
        last_logged_step=rows[-1]['step'] if rows else begin,
        uncheckpointed_updates=(rows[-1]['step'] if rows else begin) - step,
        lr_unchanged=True, optimizer_and_rng_restored=True, data_cursor_verified=True)
    controller.progress.setdefault('transient_recoveries', []).append(event)
    controller.progress.update(step=step, latest_checkpoint=str(candidate))
    write_json(controller.control / f'transport_recovery_{boundary}_{count + 1}.json', event)
    controller.publish('recovering_transient_training')
    time.sleep(min(30 * (count + 1), 60))
    return str(candidate), step
