"""Run the unchanged complete validation panels on exclusively leased idle GPUs.

This process never publishes the canonical training job, signals training, or
changes a training checkpoint. A frozen queue entry survives either process.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import threading
import time
import traceback

from multicamera4rc.quality import compare_errors, error_panel
from scripts.continue_resource_training import ResourceContinuationController
from scripts.cutover_native_motion_stereo import write_json, proc
from scripts.migrate_dynamic_validation import build_validation_contract, ExpandedDynamicController
from scripts.query_pair_main_admission import clean_runtime, free_indices, read, verify_files
from scripts.resource_resilience import transient_failure
from scripts.resource_validation import cached_suite, validation_command


def file_identity(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.name.endswith('.tmp'):
        raise ValueError('only completed regular checkpoints may be evaluated')
    s = path.stat()
    return dict(device=s.st_dev, inode=s.st_ino, size=s.st_size, mtime_ns=s.st_mtime_ns)


def require_identity(path, expected):
    if file_identity(path) != expected:
        raise ValueError('frozen checkpoint changed during independent validation')


def validation_pool(job):
    pool = job['validation_gpus']
    if (not pool or any(type(i) is not int or i < 0 for i in pool)
            or len(set(pool)) != len(pool) or set(pool) & set(job['physical_gpus'])
            or set(job['validation_gpu_uuids']) != set(map(str, pool))):
        raise ValueError('validation must use distinct authorized GPUs outside main training')
    return pool


def acquire_idle_gpu(controller, index):
    """Free before and after both cooperative leases; never share a GPU."""
    uuid = controller.job['validation_gpu_uuids'][str(index)]
    devices, apps = controller.snapshot()
    if index not in devices or devices[index].uuid != uuid:
        raise ValueError('validation GPU identity changed')
    if index not in free_indices(devices, apps):
        return None
    lease = controller.device_lease(index, uuid)
    if lease is None:
        return None
    try:
        devices, apps = controller.snapshot()
        if index not in devices or devices[index].uuid != uuid:
            raise ValueError('validation GPU identity changed while acquiring its lease')
        if index not in free_indices(devices, apps):
            lease.close()
            return None
    except BaseException:
        lease.close()
        raise
    return lease


class IndependentValidation(ResourceContinuationController):
    def publish(self, stage):
        # Deliberately independent of RecoveryController.publish: no canonical
        # active_training.json, queued continuation or main job writes.
        with self.mutex:
            self.progress.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(self.control / 'progress.json', self.progress)

    def run(self):
        pool = validation_pool(self.job)
        self.control.mkdir(parents=True, exist_ok=True)
        with (self.control / 'validation.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write_json(self.control / 'worker_identity.json', proc(os.getpid()))
            try:
                return self._evaluate(pool)
            except BaseException:
                self.progress['error'] = traceback.format_exc()
                self.publish('failed_validation')
                raise

    def _evaluate(self, pool):
        job = self.job
        clean_runtime(self.root, job['commit'])
        verify_files(job['protected_files'])
        if build_validation_contract(job) != job['validation_contract']:
            raise ValueError('fixed validation settings changed')
        checkpoint, step, label = job['validation_checkpoint'], job['validation_step'], job['validation_label']
        require_identity(checkpoint, job['checkpoint_identity'])
        self.check(checkpoint, step)
        self.progress.update(step=step, checkpoint=checkpoint, validation_children={},
            validation_phases=[], resource_wait={}, evaluation_gpus=pool)
        self.publish('validating')
        plans, pending, reports = self._plans(), Queue(), {}
        # Cached complete suites need no GPU, even if every device is occupied.
        for suite in ('kubric', 'native_dynamic', 'static', 'variable', 'syn4d'):
            path = self.output / 'validation' / label / suite / 'evaluation.json'
            if cached_suite(path, suite, checkpoint, label, step, plans[suite], job['validation_contract']):
                reports[suite] = str(path)
            else:
                pending.put(suite)
        cancelled = threading.Event()

        def worker(index):
            while not cancelled.is_set() and not pending.empty():
                lease = acquire_idle_gpu(self, index)
                if lease is None:
                    with self.mutex:
                        self.progress['resource_wait'][str(index)] = 'occupied or leased'
                        self.publish('waiting_or_validating')
                    cancelled.wait(10)
                    continue
                with lease:
                    # Dequeue only after acquiring a free GPU, so a busy card
                    # cannot strand a suite while another worker becomes free.
                    try:
                        suite = pending.get_nowait()
                    except Empty:
                        return
                    with self.mutex:
                        self.progress['resource_wait'].pop(str(index), None)
                    directory = self.output / 'validation' / label / suite
                    directory.mkdir(parents=True, exist_ok=True)
                    command = validation_command(job, suite, directory, checkpoint, label)
                    success = False
                    for attempt in range(3):
                        if cancelled.is_set():
                            return
                        require_identity(checkpoint, job['checkpoint_identity'])
                        # Retry names remain unique after a worker restart.
                        number = attempt
                        while (self.control / f'{suite}_attempt{number}.log').exists():
                            number += 1
                        log_path = self.control / f'{suite}_attempt{number}.log'
                        began = time.monotonic()
                        environment = os.environ | job['environment'] | {
                            'CUDA_VISIBLE_DEVICES': job['validation_gpu_uuids'][str(index)]}
                        with log_path.open('x') as log:
                            child = subprocess.Popen(command, cwd=self.root, env=environment,
                                stdout=log, stderr=subprocess.STDOUT)
                            with self.mutex:
                                self.progress['validation_children'][suite] = dict(pid=child.pid, gpu=index, log=str(log_path))
                                self.publish('validating')
                            while True:
                                try:
                                    code = child.wait(timeout=30)
                                    break
                                except subprocess.TimeoutExpired:
                                    with self.mutex:
                                        self.progress.setdefault('elapsed', {})[suite] = time.monotonic() - began
                                        self.publish('validating')
                        with self.mutex:
                            self.progress['validation_children'].pop(suite, None)
                            self.progress['validation_phases'].append(dict(suite=suite, gpu=index,
                                attempt=number, exit_code=code, seconds=time.monotonic() - began, log=str(log_path)))
                            self.publish('validating')
                        path = directory / 'evaluation.json'
                        if code == 0:
                            if not cached_suite(path, suite, checkpoint, label, step, plans[suite], job['validation_contract']):
                                raise ValueError('validation exited without a complete fixed report: ' + suite)
                            require_identity(checkpoint, job['checkpoint_identity'])
                            with self.mutex:
                                reports[suite] = str(path)
                            success = True
                            break
                        if not transient_failure(code, log_path) or attempt == 2:
                            raise RuntimeError(f'{suite} exited {code}; see {log_path}')
                        cancelled.wait(30)
                    if not success:
                        raise RuntimeError('incomplete validation suite: ' + suite)

        def guarded(index):
            try:
                worker(index)
            except BaseException:
                cancelled.set()
                raise

        with ThreadPoolExecutor(max_workers=len(pool)) as executor:
            futures = [executor.submit(guarded, index) for index in pool]
            for future in futures:
                future.result()
        entries = {suite: read(path)['checkpoints'][label] for suite, path in reports.items()}
        record = dict(checkpoint=checkpoint, label=label, step=step, errors=error_panel(entries),
            reports=reports, validation_contract=deepcopy(job['validation_contract']))
        if ExpandedDynamicController._verify_record(self, record) != step:
            raise ValueError('validation suites disagree about the frozen checkpoint')
        require_identity(checkpoint, job['checkpoint_identity'])
        comparison = compare_errors(job['inherited_baseline']['errors'], record['errors'])
        result = dict(record=record, comparison=comparison,
            quality_passed=comparison['score_ratio'] <= 1.5, checkpoint_identity=job['checkpoint_identity'])
        write_json(self.output / 'validation' / label / 'quality.json', record)
        write_json(self.control / 'result.json', result)
        self.progress['result'] = str(self.control / 'result.json')
        self.publish('completed_validation' if result['quality_passed'] else 'failed_quality')
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', required=True)
    IndependentValidation(parser.parse_args().controller).run()
