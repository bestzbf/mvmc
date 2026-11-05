"""Continuous full evaluation. Only native producers write prediction receipts.

Plan writers atomically replace queue_plan.json while holding queue_plan.append.lock.
This controller never adopts prior processes or reinterprets failed cases as success.
"""
import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

GPU_UUIDS = {3: '@@GPU3_UUID@@',
             4: '@@GPU4_UUID@@'}
LOCKS = Path('@@BENCHMARK_ROOT@@/evaluation')
MIN_FREE_BYTES = 20 * 1024**3
TERMINAL = {'complete', 'failed', 'blocked', 'interrupted'}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with temp.open('w') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def process_identity(pid):
    """No environment reads; birth ticks prevent reused PID ownership."""
    try:
        text = (Path('/proc') / str(pid) / 'stat').read_text()
        fields = text[text.rfind(')') + 2:].split()
        return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]),
                'pgid': int(fields[2]), 'sid': int(fields[3]),
                'starttime_ticks': int(fields[19])}
    except (OSError, ValueError, IndexError):
        return None


def owned_group_members(item):
    """Only the new session created for this Popen; never an existing group."""
    pid, birth = item['process'].pid, item['state'].get('pid_starttime_ticks')
    if birth is None:
        return []  # Registration failed while the native-execution gate was closed.
    leader = process_identity(pid)
    if leader and leader['starttime_ticks'] != birth:
        raise RuntimeError('Refusing reused session leader PID')
    result = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        value = process_identity(int(path.name))
        if value and value['pgid'] == pid and value['sid'] == pid:
            if value['starttime_ticks'] < birth:
                raise RuntimeError('Session ownership check failed')
            if value['state'] != 'Z':
                result.append(value)
    return result


def available_gpu(index):
    def read(argv):
        return subprocess.check_output(argv, text=True, timeout=20)
    expected = GPU_UUIDS[index]
    actual = read(['nvidia-smi', '-i', str(index), '--query-gpu=uuid',
                   '--format=csv,noheader,nounits']).strip()
    if actual != expected:
        return False
    apps = read(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                 '--format=csv,noheader,nounits'])
    if any(row and row[0].strip() == expected for row in csv.reader(io.StringIO(apps))):
        return False
    used = read(['nvidia-smi', '-i', expected, '--query-gpu=memory.used',
                 '--format=csv,noheader,nounits'])
    return int(used.strip()) < 1024


def child_environment(job, index):
    env = os.environ.copy()
    env.update(job.get('env', {}))
    env.update(CUDA_VISIBLE_DEVICES=GPU_UUIDS[index] if index is not None else '',
               CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4', PYTHONUNBUFFERED='1',
               HF_HUB_OFFLINE='1', WANDB_MODE='disabled')
    return env


def target_ids(job):
    if 'case_ids' in job:
        return list(job['case_ids'])
    argv = job['argv']
    if '--manifest' in argv:
        manifest = Path(argv[argv.index('--manifest') + 1])
        if not manifest.is_absolute():
            manifest = Path(job.get('cwd', '.')) / manifest
        return [c['id'] for c in json.loads(manifest.read_text())['cases']]
    return []


def receipt_snapshot(job):
    if job['kind'] != 'gpu':
        return {'ids': [], 'count': 0, 'last_mtime_utc': None, 'invalid_receipts': 0}
    directory = Path(job.get('receipt_dir') or Path(job['expected_outputs'][0]).parent)
    selected = set(target_ids(job))
    ids, mtimes, invalid, record_hashes = set(), [], 0, {}
    files = list(directory.glob('*.receipt.json'))
    state_hash = None
    try:
        if files:
            argv = job['argv']
            manifest_path = Path(argv[argv.index('--manifest') + 1])
            if not manifest_path.is_absolute():
                manifest_path = Path(job.get('cwd', '.')) / manifest_path
            manifest_bytes = manifest_path.read_bytes()
            cases = {c['id']: c for c in json.loads(manifest_bytes)['cases']}
            state_path = directory / 'run-state.json'
            if state_path.is_symlink():
                raise ValueError('Symlink run state')
            state = json.loads(state_path.read_text())
            if state.get('schema') not in ('released-baseline-run-v1', 'released-prediction-run-v1') or state.get('manifest_sha256') != hashlib.sha256(manifest_bytes).hexdigest():
                raise ValueError('Native run state is not bound to the actual batch manifest')
            state_hash = canonical(state)
    except (OSError, ValueError, KeyError, TypeError):
        return {'ids': [], 'count': 0, 'last_mtime_utc': None,
                'invalid_receipts': len(files), 'record_digest_by_id': {}, 'run_state_sha256': None}
    for path in files:
        try:
            receipt = json.loads(path.read_text())
            record = receipt['record']
            value = record['id']
            if selected and value not in selected:
                continue
            case = cases[value]
            payload = directory / record['file']
            checksum = record['sha256']
            if (path.is_symlink() or receipt.get('schema') != 'released-prediction-case-v1' or
                receipt.get('run_state_sha256') != state_hash or
                receipt.get('case_sha256') != canonical(case) or
                record.get('input_sha256') != case['input_sha256'] or
                not isinstance(checksum,str) or len(checksum)!=64 or
                any(c not in '0123456789abcdef' for c in checksum) or
                not payload.resolve().is_relative_to(directory.resolve()) or
                payload.is_symlink() or not payload.is_file() or payload.stat().st_size == 0):
                raise ValueError('Receipt identity/payload metadata invalid')
            if state.get('method') and record.get('model') != state['method']:
                raise ValueError('Receipt model differs from actual native state')
            ids.add(value)
            record_hashes[value] = canonical(record)
            mtimes.append(path.stat().st_mtime)
        except (OSError, ValueError, KeyError, TypeError):
            invalid += 1
    latest = dt.datetime.fromtimestamp(max(mtimes), dt.timezone.utc).isoformat() if mtimes else None
    return {'ids': sorted(ids), 'count': len(ids), 'last_mtime_utc': latest,
            'invalid_receipts': invalid, 'record_digest_by_id': record_hashes,
            'run_state_sha256': state_hash}


class Runner:
    def __init__(self, plan_path):
        self.plan_path = Path(plan_path).resolve()
        self.root = self.plan_path.parent
        self.active, self.states, self.known_hashes = {}, {}, {}
        self.plan, self.last_dispatch, self.sequence = {}, {}, 0
        self.stop_requested, self.runner_error, self.disk_paused = False, None, False
        self.nonce = uuid.uuid4().hex
        self.singleton = (self.root / 'runner.lock').open('a+')
        fcntl.flock(self.singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state_dir = self.root / 'status'
        self.state_dir.mkdir(exist_ok=True)
        try:
            for path in self.state_dir.glob('*.json'):
                state = json.loads(path.read_text())
                if state['state'] not in TERMINAL | {'retry_wait'}:
                    raise RuntimeError('Explicit reconciliation required; never adopt/signal prior '
                                       'processes: ' + str(path))
                self.states[state['job_id']] = state
                self.known_hashes[state['job_id']] = state['job_hash']
                self.sequence = max(self.sequence, state.get('launch_sequence', 0))
                self.last_dispatch[state['method']] = max(
                    self.last_dispatch.get(state['method'], 0), state.get('launch_sequence', 0))
            self.reload()
            save(self.root / 'launch_receipt.json', {
                'pid': os.getpid(), 'process_identity': process_identity(os.getpid()),
                'nonce': self.nonce, 'started_at_utc': utc(), 'mode': 'continuous_full_evaluation',
                'global_deadline': None, 'plan_path': str(self.plan_path),
                'allowed_gpu_uuids': GPU_UUIDS, 'training_mutations': False,
                'foreign_process_signals': False})
        except BaseException:
            self.singleton.close()
            raise

    def reload(self):
        with (self.root / 'queue_plan.append.lock').open('a+') as handle:
            fcntl.flock(handle, fcntl.LOCK_SH)
            plan = json.loads(self.plan_path.read_text())
        if plan.get('allowed_physical_gpus') != [3, 4]:
            raise ValueError('Only physical GPU3/4 are authorized')
        if any(k in plan for k in ('duration_hours', 'deadline_utc')):
            raise ValueError('Full queue must not contain an inherited global deadline')
        jobs = plan['jobs']
        ids = [j['id'] for j in jobs]
        if len(ids) != len(set(ids)) or (plan.get('sealed') and not jobs):
            raise ValueError('Duplicate job IDs or empty sealed plan')
        if not set(self.known_hashes) <= set(ids):
            raise ValueError('Previously defined jobs cannot be removed')
        output_owners = {}
        for job in jobs:
            key = job['id']
            if not isinstance(key, str) or key in ('.', '..') or Path(key).name != key:
                raise ValueError('Unsafe job ID')
            if job['kind'] not in ('gpu', 'cpu') or not job['argv'] or not all(isinstance(a, str) for a in job['argv']):
                raise ValueError('Invalid kind/argv: ' + key)
            if Path(job['argv'][0]).name in ('sh', 'bash', 'zsh', 'env'):
                raise ValueError('Use direct native executable argv, not a shell/env wrapper')
            if 'CUDA_VISIBLE_DEVICES' in job.get('env', {}) or any('CUDA_VISIBLE_DEVICES=' in a for a in job['argv']):
                raise ValueError('Job cannot override assigned CUDA UUID')
            timeout = float(job.get('timeout_seconds', 3600))
            cap = 21600 if job['kind'] == 'cpu' or 'mose' in job['method'].lower() else 3600
            if not math.isfinite(timeout) or not 0 < timeout <= cap:
                raise ValueError('Invalid/excessive per-job timeout: ' + key)
            priority = float(job.get('priority', 100))
            if not math.isfinite(priority):
                raise ValueError('Invalid priority')
            if job['kind'] == 'gpu':
                affinity = job.get('physical_gpu_affinity', [3,4])
                affinity = [affinity] if isinstance(affinity,int) else affinity
                if not isinstance(affinity,list) or not affinity or len(affinity)!=len(set(affinity)) or any(i not in [3,4] for i in affinity):
                    raise ValueError('GPU affinity must be a nonempty subset of [3,4]')
                count = job.get('case_count', 1)
                if not isinstance(count, int) or not 1 <= count <= 5 or (timeout > 3600 and count != 1):
                    raise ValueError('GPU batches max5; long MoSE timeout requires one case')
                if not job.get('expected_outputs'):
                    raise ValueError('GPU producer requires its native prediction index')
                if not ('case_ids' in job or '--manifest' in job['argv']):
                    raise ValueError('GPU job must identify its exact native cases')
                native = str(Path(job['expected_outputs'][0]).resolve().parent)
                if native in output_owners:
                    raise ValueError('Different native jobs must not share a mutable output root')
                output_owners[native] = key
            for output in job.get('expected_outputs', []):
                if not Path(output).resolve().is_relative_to(self.root):
                    raise ValueError('New output must stay under the new full root: ' + output)
            if any(d not in ids or d == key for d in job.get('depends_on', [])):
                raise ValueError('Unknown/self dependency')
            if key in self.known_hashes and self.known_hashes[key] != digest(job):
                raise ValueError('Immutable job changed: ' + key)
        graph = {j['id']: j.get('depends_on', []) for j in jobs}
        visiting, visited = set(), set()
        def visit(key):
            if key in visiting:
                raise ValueError('Dependency cycle')
            if key in visited:
                return
            visiting.add(key)
            for dep in graph[key]:
                visit(dep)
            visiting.remove(key)
            visited.add(key)
        for key in graph:
            visit(key)
        self.known_hashes.update({j['id']: digest(j) for j in jobs})
        self.plan = plan

    def acquire(self, index):
        handle = None
        try:
            handle = (LOCKS / ('.gpu%d_single_inference.lock' % index)).open('a+')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not available_gpu(index):
                handle.close()
                return None
            return handle
        except (OSError, subprocess.SubprocessError, ValueError):
            if handle:
                handle.close()
            return None

    def write_state(self, state):
        save(self.state_dir / (state['job_id'] + '.json'), state)

    def start_job(self, job, index=None, handle=None):
        key = job['id']
        previous = self.states.get(key, {})
        attempt = previous.get('attempt_count', 0) + 1
        ids = target_ids(job)
        if job['kind'] == 'gpu' and (len(ids) != job.get('case_count', 1) or len(ids) != len(set(ids))):
            if handle:
                handle.close()
            raise ValueError('Native case IDs/count do not match the frozen batch')
        observed = receipt_snapshot(job)
        logfile = self.root / 'logs' / (key + '.attempt%d.log' % attempt)
        logfile.parent.mkdir(exist_ok=True)
        stream = logfile.open('ab', buffering=0)
        self.sequence += 1
        self.last_dispatch[job['method']] = self.sequence
        state = {'job_id': key, 'job_hash': digest(job), 'kind': job['kind'],
                 'method': job['method'], 'dataset': job['dataset'], 'state': 'preparing',
                 'attempt_count': attempt, 'launch_sequence': self.sequence,
                 'attempt_history': previous.get('attempt_history', []),
                 'receipt_origin_ids': previous.get('receipt_origin_ids', observed['ids']),
                 'receipts_before_attempt': observed['ids'], 'started_at_utc': utc(),
                 'started_epoch': time.time(), 'physical_gpu': index,
                 'gpu_uuid': GPU_UUIDS.get(index), 'argv': job['argv'], 'log': str(logfile),
                 'score_path': job.get('score_path'), 'owner_runner_nonce': self.nonce,
                 'owner_runner_identity': process_identity(os.getpid())}
        self.states[key] = state
        read_fd, write_fd = None, None
        try:
            # Durable intent precedes Popen. Child cannot execute native code until
            # active registration and its exact birth identity are also durable.
            self.write_state(state)
            read_fd, write_fd = os.pipe()
            command = [sys.executable, '-u', str(Path(__file__).resolve()),
                       '--exec-gated-child', str(read_fd), '--'] + job['argv']
            inherited = (read_fd,) + ((handle.fileno(),) if handle else ())
            env = child_environment(job, index)
            if handle:
                env['BASELINE_GPU_LOCK_FD'] = str(handle.fileno())
            process = subprocess.Popen(command, cwd=job.get('cwd', str(self.root)),
                                       env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       close_fds=True, pass_fds=inherited)
            # This assignment must be the first operation after Popen succeeds.
            self.active[key] = {'job': job, 'process': process, 'stream': stream,
                                'lock': handle, 'state': state,
                                'start_monotonic': time.monotonic(), 'term_at': None}
            identity = process_identity(process.pid)
            if not identity or identity['pgid'] != process.pid or identity['sid'] != process.pid:
                raise RuntimeError('Cannot register owned child session identity')
            state.update(state='running', pid=process.pid,
                         pid_starttime_ticks=identity['starttime_ticks'])
            self.write_state(state)
            os.write(write_fd, b'1')
        except BaseException as exc:
            if key in self.active:
                raise  # main finally owns/reaps the now-registered child.
            stream.close()
            if handle:
                handle.close()
            state.update(state='failed', error=repr(exc), finished_at_utc=utc())
            self.write_state(state)
        finally:
            if read_fd is not None:
                os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)

    def signal_owned(self, item, sig):
        process = item['process']
        identity = process_identity(process.pid)
        birth = item['state'].get('pid_starttime_ticks')
        if birth is None:
            # Only possible before registration, while the gate is still closed.
            if process.poll() is None:
                process.send_signal(sig)
            return
        if identity and (identity['starttime_ticks'] != birth or
                         identity['pgid'] != process.pid or identity['sid'] != process.pid):
            raise RuntimeError('Refusing signal: owned process identity changed')
        if identity or owned_group_members(item):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass

    def finish_attempt(self, item, code):
        job, state = item['job'], item['state']
        observed = receipt_snapshot(job)
        new_ids = sorted(set(observed['ids']) - set(state['receipts_before_attempt']))
        state.update(exit_code=code, finished_at_utc=utc(),
                     elapsed_seconds=round(time.monotonic() - item['start_monotonic'], 2),
                     observed_receipts=observed, new_receipts_this_attempt=new_ids)
        state['state'] = 'complete' if code == 0 else 'failed'
        if state.get('termination_reason'):
            state['state'] = 'interrupted'
        missing = [p for p in job.get('expected_outputs', []) if not Path(p).is_file()]
        if code == 0 and missing:
            state.update(state='failed', error='Expected native output missing', missing=missing)
        if code == 0 and not missing and job['kind'] == 'gpu':
            try:
                index = json.loads(Path(job['expected_outputs'][0]).read_text())
                actual_ids = [c['id'] for c in index['cases']]
                if actual_ids != target_ids(job) or set(observed['ids']) != set(actual_ids) or observed['invalid_receipts']:
                    raise ValueError('Native index/receipt case coverage differs from the batch')
                if index.get('schema') != 'released-tracking-predictions-v1' or index.get('run_state_sha256') != observed['run_state_sha256'] or any(canonical(c) != observed['record_digest_by_id'][c['id']] for c in index['cases']):
                    raise ValueError('Native index is not derived from the bound case receipts')
            except (OSError, ValueError, TypeError, KeyError) as exc:
                state.update(state='failed', error='Native output metadata invalid: ' + repr(exc))
        retry = (state.get('termination_reason') == 'job_timeout' and
                 state['attempt_count'] < 2 and job['kind'] == 'gpu' and
                 '--resume' in job['argv'] and bool(new_ids) and
                 job.get('allow_timeout_retry', True) and not self.stop_requested)
        archive = dict(state)
        archive.pop('attempt_history', None)
        path = self.root / 'attempts' / job['id'] / ('%04d.json' % state['attempt_count'])
        if path.exists() and digest(json.loads(path.read_text())) != digest(archive):
            raise RuntimeError('Attempt history must be immutable')
        save(path, archive)
        state['attempt_history'] = state['attempt_history'] + [str(path)]
        if retry:
            state.update(state='retry_wait', retry_reason='Signal timeout with new native receipts; one tail retry only')
        self.write_state(state)
        item['stream'].close()
        if item['lock']:
            item['lock'].close()
        del self.active[job['id']]

    def poll(self):
        now = time.monotonic()
        for item in list(self.active.values()):
            process, job, state = item['process'], item['job'], item['state']
            code = process.poll()
            timed_out = now - item['start_monotonic'] >= float(job.get('timeout_seconds', 3600))
            if code is None and (timed_out or self.stop_requested):
                if item['term_at'] is None:
                    state['termination_reason'] = 'job_timeout' if timed_out and not self.stop_requested else 'runner_stop'
                    self.signal_owned(item, signal.SIGTERM)
                    item['term_at'] = now
                elif now - item['term_at'] >= 20:
                    self.signal_owned(item, signal.SIGKILL)
                continue
            if code is None:
                continue
            # Reap all descendants of our session before releasing its GPU lock.
            if owned_group_members(item):
                if item['term_at'] is None:
                    item['term_at'] = now
                    self.signal_owned(item, signal.SIGTERM)
                elif now - item['term_at'] >= 20:
                    self.signal_owned(item, signal.SIGKILL)
                continue
            self.finish_attempt(item, code)

    def admit(self):
        if self.stop_requested:
            return
        self.disk_paused = shutil.disk_usage(self.root).free < MIN_FREE_BYTES
        ready, retries = [], []
        for position, job in enumerate(self.plan['jobs']):
            key = job['id']
            if key in self.active or self.states.get(key, {}).get('state') in TERMINAL:
                continue
            deps = job.get('depends_on', [])
            if any(self.states.get(d, {}).get('state') in TERMINAL - {'complete'} for d in deps):
                state = dict(job_id=key, job_hash=digest(job), state='blocked',
                             reason='A dependency failed; no fabricated zero', method=job['method'],
                             dataset=job['dataset'], kind=job['kind'])
                self.states[key] = state
                self.write_state(state)
                continue
            if any(self.states.get(d, {}).get('state') != 'complete' for d in deps):
                continue
            rank = (float(job.get('priority', 100)), self.last_dispatch.get(job['method'], 0), position)
            (retries if self.states.get(key, {}).get('state') == 'retry_wait' else ready).append((rank, job))
        # Recompute method order after every launch, including when both GPUs free.
        pending = [(False, rank[2], job) for rank, job in ready] + [(True, rank[2], job) for rank, job in retries]
        used = {i['state']['physical_gpu'] for i in self.active.values() if i['job']['kind']=='gpu'}
        attempted = set(used)
        cpu_busy = any(i['job']['kind']=='cpu' for i in self.active.values())
        while pending:
            chosen = min(pending, key=lambda p:(p[0], float(p[2].get('priority',100)),
                         self.last_dispatch.get(p[2]['method'],0), p[1]))
            pending.remove(chosen)
            job = chosen[2]
            if job['kind'] == 'cpu':
                if not cpu_busy:
                    self.start_job(job)
                    cpu_busy = True
                continue
            self.disk_paused = shutil.disk_usage(self.root).free < MIN_FREE_BYTES
            if self.disk_paused:
                continue
            for index in (3, 4):
                if index in attempted:
                    continue
                affinity = job.get('physical_gpu_affinity', [3,4])
                if index not in ([affinity] if isinstance(affinity,int) else affinity):
                    continue
                attempted.add(index)
                handle = self.acquire(index)
                if handle is not None:
                    self.start_job(job, index, handle)
                    used.add(index)
                    break

    def outcome(self):
        complete = all(self.states.get(j['id'], {}).get('state') == 'complete' for j in self.plan['jobs'])
        terminal = all(self.states.get(j['id'], {}).get('state') in TERMINAL for j in self.plan['jobs'])
        final = not self.active and (self.stop_requested or (self.plan.get('sealed') and terminal))
        if self.runner_error:
            status = 'needs_attention'
        elif self.stop_requested:
            status = 'stopped'
        elif complete and self.plan.get('sealed'):
            status = 'complete'
        elif final:
            status = 'needs_attention'
        elif self.disk_paused:
            status = 'paused_low_disk'
        elif not self.active:
            status = 'waiting_for_ready_jobs_or_resources'
        else:
            status = 'running'
        return final, bool(self.plan['jobs']) and complete, status

    def publish(self):
        records, counts = [], {}
        for job in self.plan['jobs']:
            record = dict(self.states.get(job['id']) or {
                'job_id':job['id'], 'state':'pending', 'kind':job['kind'],
                'method':job['method'], 'dataset':job['dataset']})
            observed = receipt_snapshot(job)
            record['receipt_progress'] = observed
            record['new_receipt_count'] = len(set(observed['ids']) - set(record.get('receipt_origin_ids', observed['ids'])))
            ids = target_ids(job)
            record['missing_case_ids'] = [v for v in ids if v not in observed['ids']] if job['kind']=='gpu' else []
            log = record.get('log')
            if log and Path(log).is_file():
                st = Path(log).stat()
                record['log_bytes'], record['log_mtime_utc'] = st.st_size, dt.datetime.fromtimestamp(st.st_mtime,dt.timezone.utc).isoformat()
            counts[record['state']] = counts.get(record['state'], 0) + 1
            records.append(record)
        final, success, status = self.outcome()
        summary = {'schema':'continuous-full-evaluation-progress-v1','updated_at_utc':utc(),
                   'runner_pid':os.getpid(),'runner_nonce':self.nonce,'mode':'continuous_full_evaluation',
                   'global_deadline':None,'overall_status':status,'final':final,
                   'all_targets_complete':success,'success':success and final and not self.runner_error,
                   'counts':counts,'plan_sealed':self.plan.get('sealed',False),
                   'allowed_physical_gpus':[3,4],'GPU4_policy':'Skip whenever foreign occupancy exists',
                   'free_bytes':shutil.disk_usage(self.root).free,'min_free_bytes':MIN_FREE_BYTES,
                   'total_new_receipts':sum(r['new_receipt_count'] for r in records),
                   'remaining_job_ids':[r['job_id'] for r in records if r['state']!='complete'],
                   'runner_error':self.runner_error,'records':records}
        save(self.root/'progress.json', summary)
        lines = ['# 全量评估实时进度','', '更新UTC：'+summary['updated_at_utc'],
                 '','状态：'+status+'；无全局截止时间；仅GPU3/4，外来占用时跳过。',
                 '', '本轮新提交receipt：'+str(summary['total_new_receipts']),
                 '', '| 方法 | 数据集 | 阶段 | 状态 | 尝试 | 新提交 | 最近receipt UTC |',
                 '| --- | --- | --- | --- | --- | --- | --- |']
        for r in records:
            lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(
                r['method'],r['dataset'],r['kind'],r['state'],r.get('attempt_count',0),
                r['new_receipt_count'],r['receipt_progress']['last_mtime_utc'] or '—'))
        if final and not summary['success']:
            lines += ['', '仍有未完成项，需检查失败/阻塞记录；没有填零或宣称全量完成。']
        atomic_text(self.root/'实时进度.md', '\n'.join(lines)+'\n')
        return summary

    def cleanup(self):
        if not self.active:
            return
        self.stop_requested = True
        items = list(self.active.values())
        cleanup_errors = []
        for item in items:
            try:
                self.signal_owned(item, signal.SIGTERM)
            except Exception as exc:
                cleanup_errors.append(repr(exc))
        limit = time.monotonic() + 20
        for item in items:
            try:
                try:
                    item['process'].wait(timeout=max(0,limit-time.monotonic()))
                except subprocess.TimeoutExpired:
                    self.signal_owned(item, signal.SIGKILL)
                    item['process'].wait(timeout=10)
                if owned_group_members(item):
                    self.signal_owned(item, signal.SIGKILL)
                item['state']['termination_reason'] = item['state'].get('termination_reason','runner_exit_cleanup')
                self.finish_attempt(item, item['process'].returncode)
            except Exception as exc:
                cleanup_errors.append(repr(exc))
                # Keep a nonterminal reconciliation record if any owned process
                # survived; a later runner must refuse to adopt it.
                state = item['state']
                try:
                    alive = item['process'].poll() is None or bool(owned_group_members(item))
                except Exception:
                    alive = True  # Identity ambiguity requires reconciliation, never a foreign signal.
                state.update(state='reconciliation_required' if alive else 'interrupted',
                             termination_reason='runner_cleanup_error',error=repr(exc),
                             finished_at_utc=utc())
                try:
                    self.write_state(state)
                except OSError:
                    pass
            finally:
                item['stream'].close()
                if item['lock']:
                    item['lock'].close()
                self.active.pop(item['job']['id'], None)
        if cleanup_errors:
            self.runner_error = self.runner_error or '; '.join(cleanup_errors)

    def main(self):
        signal.signal(signal.SIGTERM, lambda *_: setattr(self,'stop_requested',True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self,'stop_requested',True))
        published = 0
        try:
            while True:
                self.reload()
                self.poll()
                self.admit()
                if time.monotonic()-published >= 30 or self.outcome()[0]:
                    self.publish()
                    published = time.monotonic()
                if self.outcome()[0]:
                    break
                time.sleep(5)
        except BaseException as exc:
            self.runner_error = repr(exc)
            self.stop_requested = True
            try:
                save(self.root/'runner_error.json', {'at_utc':utc(),'error':repr(exc)})
            finally:
                raise
        finally:
            try:
                self.cleanup()
                self.publish()
            finally:
                self.singleton.close()


def gated_child(fd, argv):
    """A dead/unregistered parent closes the pipe: no native code is executed."""
    token = os.read(fd, 1)
    os.close(fd)
    if token != b'1':
        raise SystemExit(125)
    os.execvpe(argv[0], argv, os.environ)


if __name__ == '__main__':
    if len(sys.argv)>2 and sys.argv[1]=='--exec-gated-child':
        gated_child(int(sys.argv[2]), sys.argv[4:])
    else:
        parser=argparse.ArgumentParser()
        parser.add_argument('--plan',required=True)
        Runner(parser.parse_args().plan).main()
