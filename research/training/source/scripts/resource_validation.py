"""Fixed-plan validation with verified reuse, heartbeats and bounded retry."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from pathlib import Path
from queue import Empty, Queue
import shutil
import subprocess
import time

from multicamera4rc.quality import error_panel
from scripts.cutover_native_motion_stereo import write_json
from scripts.query_pair_main_admission import read, free_indices
from scripts.migrate_dynamic_validation import (
    ExpandedDynamicController, build_validation_contract, verify_suite_report,
)
from scripts.resource_resilience import transient_failure


def cached_suite(path, suite, checkpoint, label, step, plan, contract):
    if not path.exists():
        return None
    report = read(path)
    if report.get('stage') != 'passed':
        return None
    # A completed but incompatible report is an error, never overwritten.
    record = dict(checkpoint=str(checkpoint), label=label, step=step)
    verify_suite_report(suite, report, record, plan, contract)
    return str(path)


def validation_command(job, suite, directory, checkpoint, label):
    """One command builder for synchronous and independent full validation."""
    settings = job['evaluation']
    wrapper = ('import runpy,sys,torch; torch.cuda.set_per_process_memory_fraction(.85,0); '
               'runpy.run_module(sys.argv.pop(1),run_name="__main__")')
    common = ['--output', str(directory), '--checkpoints', label + '=' + str(checkpoint)]
    if suite in ('kubric', 'variable', 'syn4d'):
        module = 'multicamera4rc.evaluate'
        config = job['syn4d_eval_config'] if suite == 'syn4d' else settings[suite + '_config']
        plan = job['syn4d_eval_plan'] if suite == 'syn4d' else settings[suite + '_plan']
        args = ['--data-config', config, '--plan', plan, '--cpu-threads', '2']
    elif suite == 'native_dynamic':
        module = 'scripts.evaluate_native_dynamic'
        args = ['--data-config', settings['native_config'], '--plan', settings['native_plan'], '--memory-fraction', '.85']
    elif suite == 'static':
        module = 'scripts.evaluate_static_multiview'
        plan = directory / 'fixed_plan.json'
        shutil.copyfile(settings['static_plan'], plan)
        args = ['--train-config', settings['static_config'], '--plan', str(plan), '--memory-fraction', '.85']
    else:
        raise ValueError('unknown fixed validation suite: ' + suite)
    return [job['python'], '-u', '-c', wrapper, module, *args, *common]


def evaluate(controller, checkpoint, label):
    import torch
    job = controller.job
    if build_validation_contract(job) != job['validation_contract']:
        raise ValueError('fixed validation contract changed')
    payload = torch.load(checkpoint, mmap=True, map_location='cpu', weights_only=True)
    step = int(payload['step']); del payload
    controller.check(checkpoint, step)
    controller.publish('validate_' + label)
    suites = ('kubric', 'native_dynamic', 'static', 'variable', 'syn4d')
    pending = Queue()
    for suite in suites:
        pending.put(suite)
    plans = controller._plans()
    uuids = job['environment']['CUDA_VISIBLE_DEVICES'].split(',')

    def worker(slot):
        result = {}
        while True:
            try:
                suite = pending.get_nowait()
            except Empty:
                break
            directory = controller.output / 'validation' / label / suite
            directory.mkdir(parents=True, exist_ok=True)
            cached = directory / 'evaluation.json'
            reused = cached_suite(cached, suite, checkpoint, label, step, plans[suite], job['validation_contract'])
            if reused:
                result[suite] = reused
                with controller.mutex:
                    controller.progress.setdefault('reused_validation_suites', []).append(suite)
                    controller.publish('validate_' + label)
                continue
            command = validation_command(job, suite, directory, checkpoint, label)
            for attempt in range(3):
                while True:
                    devices, apps = controller.snapshot()
                    physical = job['physical_gpus'][slot]
                    if physical not in devices or devices[physical].uuid != uuids[slot]:
                        raise ValueError('validation GPU identity changed')
                    if physical in free_indices(devices, apps):
                        break
                    with controller.mutex:
                        controller.progress.setdefault('validation_resource_wait', {})[suite] = physical
                        controller.publish('validate_' + label)
                    time.sleep(10)
                log_path = controller.control / f'validate_{label}_{suite}_attempt{attempt}.log'
                began = time.monotonic()
                with log_path.open('x') as log:
                    child = subprocess.Popen(command, cwd=controller.root,
                        env=os.environ | job['environment'] | {'CUDA_VISIBLE_DEVICES': uuids[slot]},
                        stdout=log, stderr=subprocess.STDOUT)
                    with controller.mutex:
                        controller.progress.setdefault('validation_resource_wait', {}).pop(suite, None)
                        controller.progress.setdefault('validation_children', {})[suite] = child.pid
                        controller.publish('validate_' + label)
                    while True:
                        try:
                            code = child.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            with controller.mutex:
                                controller.progress.setdefault('validation_elapsed', {})[suite] = time.monotonic() - began
                                controller.publish('validate_' + label)
                with controller.mutex:
                    controller.progress['validation_children'].pop(suite, None)
                    controller.progress.setdefault('validation_phases', []).append(dict(suite=suite,
                        attempt=attempt, exit_code=code, log=str(log_path), seconds=time.monotonic() - began))
                    controller.publish('validate_' + label)
                if code == 0:
                    if not cached_suite(cached, suite, checkpoint, label, step, plans[suite], job['validation_contract']):
                        raise ValueError('validation exited without a complete fixed-plan report: ' + suite)
                    result[suite] = str(cached)
                    break
                if (attempt == 2 or not transient_failure(code, log_path)
                        or job.get('pause_requested') or (controller.control / 'PAUSE').exists()):
                    raise RuntimeError(f'{suite} validation failed ({code}); see {log_path}')
                time.sleep(30)
        return result

    reports = {}
    with ThreadPoolExecutor(max_workers=len(uuids)) as pool:
        for paths in pool.map(worker, range(len(uuids))):
            reports.update(paths)
    entries = {suite: read(path)['checkpoints'][label] for suite, path in reports.items()}
    record = dict(checkpoint=str(checkpoint), label=label, step=step, errors=error_panel(entries), reports=reports,
        validation_contract=deepcopy(job['validation_contract']))
    if ExpandedDynamicController._verify_record(controller, record) != step:
        raise ValueError('validation checkpoint steps disagree')
    write_json(controller.output / 'validation' / label / 'quality.json', record)
    return record
