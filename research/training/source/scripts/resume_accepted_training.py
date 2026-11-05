"""Resume a diagnosed admission-path failure after a fully accepted boundary.

Only the controller's mixture builder receives the versioned symlink fix.
Training children keep the original clean runtime, data, optimizer and LR plan.
"""
from contextlib import ExitStack
from copy import deepcopy
import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import traceback


def sibling(name):
    spec = importlib.util.spec_from_file_location('boundary_' + name,
        Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = sibling('recover_gpu_scheduling')
read, write, sha = helper.read, helper.write, helper.sha


def require_accepted_boundary(job, progress, active, accepted, evaluated):
    if (job.get('stage') != 'failed' or progress.get('stage') != 'failed'
            or progress.get('child_pid') or progress.get('failed_quality_needs_review')
            or job.get('max_quality_retries', 0) != 0):
        raise ValueError('expected a failed idle controller without a quality failure')
    if any(active.get(k) != job.get(k) for k in ('pid', 'control_dir', 'run', 'stage')):
        raise ValueError('another main controller became active')
    if accepted != evaluated:
        raise ValueError('latest evaluation must already be accepted')
    step = accepted['step']
    if (job['phase_target_step'] != step or job['resume_step'] >= step
            or Path(accepted['checkpoint']) != Path(job['run'])/'latest.pt'
            or accepted['data_config'] != job['data_config']
            or job['data_config'] != job['official_data_config']):
        raise ValueError('accepted checkpoint is not the completed segment boundary')
    decisions = progress.get('decisions', [])
    if not decisions or (decisions[-1]['step'], decisions[-1]['action']) != (step, 'continue'):
        raise ValueError('latest checkpoint did not pass quality review')
    phases = progress.get('phases', [])[-6:]
    names = [f'train_{step}_try0'] + [f'validate_extended_{step}_try0_{suite}'
        for suite in ('kubric', 'native_dynamic', 'static', 'variable', 'syn4d')]
    if [p['name'] for p in phases] != names or any(p['exit_code'] != 0 for p in phases):
        raise ValueError('completed training and all five validation suites are required')
    error = progress.get('error', '')
    if ('verify_mixture' not in error or not error.rstrip().endswith(
            'ValueError: new mixture changed more than the admitted source additions')):
        raise ValueError('failure is not the diagnosed admission-path mismatch')
    for path in Path(job['output_root']).glob('continuation_*'):
        fields = path.name.split('_')
        if len(fields) != 4 or int(fields[2]) > step:
            raise ValueError('a newer training segment exists and requires inspection')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--tests-proof', required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    old = read(args.job)
    # Do not import a replacement model/dataset runtime through this wrapper.
    sys.path.insert(0, old['repo'])
    from multicamera4rc.launch import gpu_snapshot
    from scripts import continue_official_multicam as admission
    from scripts.continue_extended_training import next_segment_end
    from scripts.cutover_native_motion_stereo import proc
    from scripts.migrate_dynamic_validation import build_validation_contract, require_validation_contract
    from scripts.resume_official_after_resource_failure import ResourceRecoveryController

    builder = sibling('build_official_multicam_mixture')
    # Retain the original admission hashes, full comparison, and GT checks.
    admission.build_mixture = builder.build_mixture
    wrapper_root = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=wrapper_root, text=True).strip()
    tests = read(args.tests_proof)
    if (not tests.get('passed') or tests.get('commit') != head
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=wrapper_root, text=True).strip()):
        raise ValueError('the exact clean recovery commit must have passed its tests')
    directory = Path(old['repo'])/'outputs/training_preflight'
    with ExitStack() as stack:
        for name in ('controller_handover.lock', 'extended_training.lock'):
            lock = stack.enter_context((directory/name).open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if name == 'controller_handover.lock':
                handover = lock
        previous = read(Path(old['control_dir'])/'progress.json')
        accepted = read(Path(old['output_root'])/'latest_accepted_checkpoint.json')
        evaluated = read(Path(old['output_root'])/'latest_evaluated_checkpoint.json')
        require_accepted_boundary(old, previous, read(old['main_active_path']), accepted, evaluated)
        if proc(old['pid']) is not None:
            raise ValueError('previous controller still exists')
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():
                continue
            process = proc(int(path.name))
            if process and process['state'] != 'Z' and any(token in process['cmd'] for token in
                    ('-m multicamera4rc.train ', '-m multicamera4rc.launch ')):
                try:
                    cwd = (path/'cwd').resolve()
                except OSError:
                    cwd = None
                if old['run'] in process['cmd'] or cwd == Path(old['repo']).resolve():
                    raise ValueError('main training process is still alive')
        original = ResourceRecoveryController(args.job)
        original.progress = deepcopy(previous)
        historical = read(old['historical_record'])
        if build_validation_contract(old) != old['validation_contract']:
            raise ValueError('validation contract changed')
        require_validation_contract(old['validation_contract'], accepted, previous['best'], historical)
        for record in (accepted, previous['best'], historical):
            original._verify_record(record)
        state = original._check_candidate_state(accepted['checkpoint'], accepted['step'])
        admission.verify_admission(old)
        for key in ('evidence_sha256', 'protected_files'):
            for path, expected in old[key].items():
                if sha(path) != expected:
                    raise ValueError('frozen evidence changed: ' + path)
        runtime_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=old['repo'], text=True).strip()
        if (runtime_head != old['commit'] or subprocess.check_output(
                ['git', 'status', '--porcelain'], cwd=old['repo'], text=True).strip()):
            raise ValueError('pinned model runtime changed')
        proof = dict(passed=True, kind='accepted_boundary_admission_path_recovery',
            previous_controller=args.job, checkpoint=accepted['checkpoint'], step=accepted['step'],
            wrapper=str(Path(__file__).resolve()), wrapper_commit=head, wrapper_sha256=sha(__file__),
            builder_sha256=sha(builder.__file__), helper_sha256=sha(helper.__file__),
            runtime=old['repo'], commit=runtime_head, tests_proof=args.tests_proof,
            physical_gpus=old['physical_gpus'], gradient_accumulation=old['gradient_accumulation'],
            learning_rate_changed=False, reset_data_cursor=False, optimizer_restored=True,
            loss_config=state['loss_config'], lr_plan=state['lr_plan'],
            configuration_sha256=sha(old['official_data_config']), protected_files=old['protected_files'])
        root = Path(args.output_root).resolve()
        if args.verify_only:
            write(root/'verification.json', proof)
            print({'passed': True, 'step': accepted['step'], 'runtime': old['repo']})
            return
        control, output = root/'control', root/'training'
        if control.exists() or output.exists():
            raise ValueError('recovery requires fresh output directories')
        control.mkdir(parents=True); output.mkdir()
        job = deepcopy(old)
        job.update(control_dir=str(control), output_root=str(output), run=str(output),
            initial_checkpoint=accepted['checkpoint'], initial_step=accepted['step'],
            selected_checkpoint=accepted['checkpoint'], resume_step=accepted['step'],
            pid=os.getpid(), log=str(root/'controller.log'), stage='resuming_accepted_boundary')
        write(control/'job.json', job); write(control/'previous_job.json', old)
        write(control/'previous_progress.json', previous); write(output/'resume_proof.json', proof)
        cls = helper.waiting_controller_type(ResourceRecoveryController, gpu_snapshot)
        controller = cls(control/'job.json')
        controller.progress = deepcopy(previous)
        for key in ('error', 'child_pid', 'child_log', 'pending_phase', 'busy_gpus'):
            controller.progress.pop(key, None)
        controller.progress['boundary_recovery'] = proof
        controller.owns_training = True
        controller.publish('resuming_accepted_boundary')
        write(Path(old['control_dir'])/'superseded.json', dict(new_controller=str(control/'job.json'),
            pid=os.getpid(), proof=str(output/'resume_proof.json')))
        for name in ('latest_accepted_checkpoint.json', 'latest_evaluated_checkpoint.json'):
            write(output/name, accepted)
        fcntl.flock(handover, fcntl.LOCK_UN)
        try:
            current = accepted
            while current['step'] < job['target_step']:
                end = next_segment_end(current['step'], job['target_step'], interval=job['validation_interval'])
                current = controller.segment(current, controller.progress['best'], historical,
                    controller.job['data_config'], end)
                controller.remember_best(current)
                write(output/'latest_accepted_checkpoint.json', current)
            controller.publish('completed_' + str(job['target_step']))
        except Exception:
            controller.progress['error'] = traceback.format_exc()
            controller.publish('failed')
            raise


if __name__ == '__main__':
    main()
