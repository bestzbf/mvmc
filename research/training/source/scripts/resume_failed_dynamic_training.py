"""Explicitly resume a repaired loader failure from the last accepted checkpoint.

This does not retry quality failures, reset the LR/data cursor, or overwrite a
previous run. Invoke it once after diagnosing and testing the underlying fault.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import traceback

from scripts.continue_extended_training import next_segment_end
from scripts.continue_quality_recovery import digest
from scripts.cutover_native_motion_stereo import proc, write_json
from scripts.migrate_dynamic_validation import (
    ExpandedDynamicController, build_validation_contract,
    require_validation_contract, verify_data_config_replacement,
)


def check_failed_training(job, progress, accepted, evaluated):
    if (job.get('stage') != 'failed' or progress.get('stage') != 'failed'
            or progress.get('failed_quality_needs_review')):
        raise ValueError('only a diagnosed execution failure may be resumed')
    phases = progress.get('phases', [])
    if not phases or not phases[-1]['name'].startswith('train_') or not phases[-1]['exit_code']:
        raise ValueError('last phase must be a failed training process')
    if any(accepted.get(k) != evaluated.get(k) for k in ('step', 'checkpoint', 'validation_contract')):
        raise ValueError('a later evaluation requires review before resuming')
    decisions = progress.get('decisions', [])
    if not decisions or (decisions[-1]['step'], decisions[-1]['action']) != (accepted['step'], 'continue'):
        raise ValueError('checkpoint has not passed the latest quality decision')
    if list(Path(job['run']).glob('*.pt')):
        raise ValueError('failed segment contains a newer checkpoint; inspect it before replaying steps')


def verify(previous_path, controller_class=ExpandedDynamicController):
    previous_path = Path(previous_path).resolve()
    old = json.loads(previous_path.read_text())
    controller = controller_class(previous_path)
    controller.progress = json.loads((controller.control/'progress.json').read_text())
    accepted = json.loads((controller.output/'latest_accepted_checkpoint.json').read_text())
    evaluated = json.loads((controller.output/'latest_evaluated_checkpoint.json').read_text())
    check_failed_training(old, controller.progress, accepted, evaluated)
    active = json.loads(controller.active.read_text())
    if (active['control_dir'], active['pid'], active['stage']) != (old['control_dir'], old['pid'], 'failed'):
        raise RuntimeError('another job became active')
    if proc(old['pid']) is not None:
        raise RuntimeError('previous controller PID still exists')
    # Refuse to compete with a surviving rank, including orphaned launchers.
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        process = proc(int(path.name))
        if (process and process['state'] != 'Z'
                and str(controller.root) in process['cmd']
                and any(token in process['cmd'] for token in
                        ('-m multicamera4rc.train ', '-m multicamera4rc.launch '))):
            raise RuntimeError('training process is still alive: '+str(process['pid']))
    if build_validation_contract(old) != old['validation_contract']:
        raise ValueError('validation contract changed')
    for path, expected in old['evidence_sha256'].items():
        if digest(path) != expected:
            raise ValueError('admission evidence changed: '+path)
    verify_data_config_replacement(old)
    historical = json.loads(Path(old['historical_record']).read_text())
    require_validation_contract(old['validation_contract'], accepted, controller.progress['best'], historical)
    for record in (accepted, controller.progress['best'], historical):
        controller._verify_record(record)
    state = controller._check_candidate_state(accepted['checkpoint'], accepted['step'])
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=controller.root, text=True).strip()
    subprocess.run(['git', 'merge-base', '--is-ancestor', old['commit'], head], cwd=controller.root, check=True)
    return old, controller.progress, accepted, historical, state, head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-controller', required=True)
    parser.add_argument('--control-dir', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--supervisor-log', help='Actual stdout/stderr log of this controller')
    parser.add_argument('--loopback-bootstrap', action='store_true',
                        help='Use loopback for this single-host job after a verified NIC bootstrap failure')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    old = json.loads(Path(args.previous_controller).read_text())
    directory = Path(old['repo'])/'outputs/training_preflight'
    with (directory/'controller_handover.lock').open('a') as handover, (directory/'extended_training.lock').open('a') as lock:
        fcntl.flock(handover, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old, previous_progress, accepted, historical, state, head = verify(args.previous_controller)
        environment = deepcopy(old['environment'])
        if args.loopback_bootstrap:
            environment.update(NCCL_SOCKET_IFNAME='=lo', GLOO_SOCKET_IFNAME='lo')
        proof = dict(passed=True, previous_controller=args.previous_controller,
                     checkpoint=accepted['checkpoint'], step=accepted['step'], commit=head,
                     restored_training_state=state, learning_rate_changed=False,
                     reset_data_cursor=False, automatic_retries=False,
                     loopback_bootstrap=args.loopback_bootstrap, environment=environment)
        if args.verify_only:
            print(json.dumps(proof))
            return
        control, output = Path(args.control_dir).resolve(), Path(args.output_root).resolve()
        if control.exists() or output.exists():
            raise RuntimeError('resume requires fresh control and output directories')
        control.mkdir(parents=True)
        output.mkdir(parents=True)
        job = deepcopy(old)
        job.update(control_dir=str(control), output_root=str(output), run=str(output), commit=head,
                   initial_checkpoint=accepted['checkpoint'], initial_step=accepted['step'],
                   resume_step=accepted['step'], selected_checkpoint=accepted['checkpoint'],
                   data_config=accepted['data_config'], environment=environment,
                   pid=os.getpid(), stage='resuming_after_loader_fix')
        if args.supervisor_log:
            job['log'] = str(Path(args.supervisor_log).resolve())
        write_json(control/'job.json', job)
        write_json(control/'previous_job.json', old)
        write_json(control/'previous_progress.json', previous_progress)
        write_json(output/'resume_proof.json', proof)
        controller = ExpandedDynamicController(control/'job.json')
        controller.progress = deepcopy(previous_progress)
        for key in ('error', 'child_pid', 'child_log'):
            controller.progress.pop(key, None)
        controller.progress.update(resumed_training_state=state, execution_recovery=proof)
        controller.owns_training = True
        controller.publish('resuming_after_loader_fix')
        fcntl.flock(handover, fcntl.LOCK_UN)
        write_json(controller.output/'latest_accepted_checkpoint.json', accepted)
        write_json(controller.output/'latest_evaluated_checkpoint.json', accepted)
        try:
            current = accepted
            while current['step'] < job['target_step']:
                end = next_segment_end(current['step'], job['target_step'], interval=job['validation_interval'])
                current = controller.segment(current, controller.progress['best'], historical,
                                             controller.job['data_config'], end)
                controller.remember_best(current)
                write_json(controller.output/'latest_accepted_checkpoint.json', current)
            controller.publish('completed_'+str(job['target_step']))
        except Exception:
            controller.progress['error'] = traceback.format_exc()
            controller.publish('failed')
            raise


if __name__ == '__main__':
    main()
