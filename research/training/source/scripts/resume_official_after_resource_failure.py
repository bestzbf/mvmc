"""Recover an accepted segment after GPU contention and admit checked data.

This is an explicit execution recovery, never an automatic quality retry. It
keeps Adam moments, the LR plan, validation and effective batch size; a new
mixture starts its own sampler cursor. Frozen trials retain their own inputs.
"""
from contextlib import ExitStack
from copy import deepcopy
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import traceback

from multicamera4rc.losses import MultiCamera4RCLoss
from scripts.continue_extended_training import next_segment_end
from scripts.continue_official_multicam import OfficialDataController, verify_admission
from scripts.continue_quality_recovery import digest, validate_checkpoint
from scripts.continue_syn4d_training import verify_state
from scripts.cutover_native_motion_stereo import write_json
from scripts.migrate_dynamic_validation import require_fixed_training_state
from scripts.queue_official_multicam import protected_experiments
from scripts.resume_failed_dynamic_training import verify


def execution_layout(state):
    return dict(world_size=state['world_size'], batch_size=state['batch_size'],
                gradient_accumulation=state.get('gradient_accumulation', 1))


def effective_batch(layout):
    return layout['world_size'] * layout['batch_size'] * layout['gradient_accumulation']


def resize_receipt(state, checkpoint, step, gpus, accumulation):
    old = execution_layout(state)
    new = dict(world_size=len(gpus), batch_size=1, gradient_accumulation=accumulation)
    if (not gpus or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus)
            or accumulation < 1 or effective_batch(old) != effective_batch(new)):
        raise ValueError('resource recovery must preserve the effective batch size')
    return dict(checkpoint=str(Path(checkpoint).resolve()), step=step, previous=old,
                current=new, effective_batch_size=effective_batch(new))


def verify_preflight(proof, accepted, config, state):
    rows = proof.get('rows', [])
    keys = {(r['source'], r['cameras'], r['times']) for r in rows}
    if (not proof.get('passed') or proof.get('checkpoint') != accepted['checkpoint']
            or proof.get('step') != accepted['step'] or proof.get('config_sha256') != digest(config)
            or proof.get('optimizer_update_performed') is not False or proof.get('checkpoint_modified') is not False
            or len(rows) != 12 or keys != {tuple(x) for x in proof.get('expected_cases', [])}
            or not {('syn4d_v1',8,2),('syn4d_sim',4,4)}.issubset(keys)
            or not all(r.get('finite_gradients') for r in rows)
            or MultiCamera4RCLoss(**proof['loss_config']).to_config()
               != MultiCamera4RCLoss(**state['loss_config']).to_config()):
        raise ValueError('full giant forward/backward admission is incomplete or stale')


class ResourceRecoveryController(OfficialDataController):
    def _check_candidate_state(self, checkpoint, step):
        resize = self.job['execution_resize']
        is_source = (str(Path(checkpoint).resolve()), step) == (resize['checkpoint'], resize['step'])
        layout = resize['previous'] if is_source else resize['current']
        state = validate_checkpoint(checkpoint, step, world_size=layout['world_size'], batch_size=layout['batch_size'])
        if execution_layout(state) != layout or effective_batch(layout) != resize['effective_batch_size']:
            raise ValueError('checkpoint execution layout or effective batch changed')
        if state.get('effective_batch_size', effective_batch(layout)) != effective_batch(layout):
            raise ValueError('checkpoint effective batch receipt disagrees with execution layout')
        verify_state(state, self.job['target_step'])
        expected, actual = deepcopy(self._fixed_state()), deepcopy(state)
        expected.update(world_size=layout['world_size'], batch_size=layout['batch_size'])
        for value in (expected, actual):
            value['loss_config'] = MultiCamera4RCLoss(**value['loss_config']).to_config()
        require_fixed_training_state(expected, actual)
        if state.get('augmentation') != self._fixed_state().get('augmentation'):
            raise ValueError('resource recovery changed augmentation')
        return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('previous-controller','queue','giant-preflight','execution-tests','output-root'):
        p.add_argument('--'+flag, required=True)
    p.add_argument('--gpus', required=True)
    p.add_argument('--gradient-accumulation', required=True, type=int)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    old = json.loads(Path(args.previous_controller).read_text())
    queue = json.loads(Path(args.queue).read_text())
    gpus = [int(v) for v in args.gpus.split(',')]
    output_root = Path(args.output_root).resolve()
    with ExitStack() as stack:
        handovers = []
        for repo in sorted({root.resolve(), Path(old['repo']).resolve()}):
            directory = repo/'outputs/training_preflight'; directory.mkdir(parents=True, exist_ok=True)
            for name in ('controller_handover.lock','extended_training.lock'):
                lock = stack.enter_context((directory/name).open('a'))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if name == 'controller_handover.lock':
                    handovers.append(lock)
        old, previous_progress, accepted, historical, state, _ = verify(args.previous_controller, OfficialDataController)
        head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip()
        subprocess.run(['git','merge-base','--is-ancestor',old['commit'],head], cwd=root, check=True)
        if subprocess.check_output(['git','status','--porcelain'], cwd=root, text=True).strip():
            raise ValueError('recovery checkout must be clean')
        tests = json.loads(Path(args.execution_tests).read_text())
        if not tests.get('passed') or tests.get('commit') != head:
            raise ValueError('resource recovery tests must pass at this exact commit')
        if queue['base_config'] != accepted['data_config'] or state['loss_config'].get('cross_camera_3d_weight') != .1:
            raise ValueError('expected the accepted cross-camera baseline before adding data')
        for path, sha in queue['frozen_files'].items():
            if digest(path) != sha: raise ValueError('queued evidence changed: '+path)
        preflight = json.loads(Path(args.giant_preflight).read_text())
        verify_preflight(preflight, accepted, queue['config'], state)
        protected = protected_experiments(queue['protected_jobs'], root, gpus)
        if protected is None: raise ValueError('independent trials have not frozen their inputs')
        resize = resize_receipt(state, accepted['checkpoint'], accepted['step'], gpus, args.gradient_accumulation)
        gpu_rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'], text=True)
        gpu_ids = {int(a.strip()): b.strip() for a,b in (line.split(',') for line in gpu_rows.splitlines())}
        environment = dict(old['environment'], PYTHONPATH=str(root),
                           CUDA_VISIBLE_DEVICES=','.join(gpu_ids[g] for g in gpus))
        control, output = output_root/'control', output_root/'training'
        job = deepcopy(old)
        for key in ('audited_data_config_replacement','learning_rate_trial','cross_camera_loss_config',
                    'cross_camera_3d_weight','cross_camera_tests_proof','adopted_training'):
            job.pop(key, None)
        job.update(repo=str(root), commit=head, environment=environment, physical_gpus=gpus,
                   gradient_accumulation=args.gradient_accumulation, execution_resize=resize,
                   previous=deepcopy(old), main_active_path=queue['main_active_path'],
                   control_dir=str(control), output_root=str(output), run=str(output),
                   log=str(output_root/'controller.log'), initial_checkpoint=accepted['checkpoint'],
                   initial_step=accepted['step'], resume_step=accepted['step'], selected_checkpoint=accepted['checkpoint'],
                   data_config=accepted['data_config'], official_data_config=queue['config'],
                   admission_kind='official_data', runtime_equivalence=queue['runtime_equivalence'],
                   tests_proof=queue['tests_proof'], protected_files=protected, max_quality_retries=0,
                   allow_shared=False, pid=os.getpid(), stage='prepared_resource_recovery',
                   official_admission=str(Path(queue['output_root'])/'audit/admission.json'),
                   panoptic_audit=queue['panoptic_audit'])
        admission = json.loads(Path(job['official_admission']).read_text())
        evidence = [queue['base_config'], queue['config'], queue['runtime_equivalence'], queue['tests_proof'],
                    job['official_admission'], admission['conversion_manifest'], queue['panoptic_audit'],
                    str(Path(job['official_admission']).parent/'packed_gt.jsonl'),
                    str(Path(job['official_admission']).parent/'loader_gt.jsonl'),
                    args.execution_tests, args.giant_preflight]
        job['official_evidence'] = {path: digest(path) for path in evidence}
        verify_admission(job)
        proof = dict(passed=True, previous_controller=args.previous_controller, commit=head,
                     checkpoint=accepted['checkpoint'], step=accepted['step'], execution_resize=resize,
                     learning_rate_changed=False, optimizer_restored=True, reset_data_cursor=True,
                     reset_reason='admitting audited mixture', giant_preflight=args.giant_preflight,
                     execution_tests=args.execution_tests, protected_files=protected)
        if args.verify_only:
            print(json.dumps(dict(passed=True,step=accepted['step'],execution_resize=resize)))
            return
        if control.exists() or output.exists(): raise ValueError('recovery requires fresh output directories')
        control.mkdir(parents=True); output.mkdir()
        write_json(control/'previous_job.json', old); write_json(control/'previous_progress.json', previous_progress)
        write_json(control/'job.json', job); write_json(output/'resume_proof.json', proof)
        controller = ResourceRecoveryController(control/'job.json')
        controller.progress = deepcopy(previous_progress)
        for key in ('error','child_pid','child_log','official_activation_failed'):
            controller.progress.pop(key, None)
        controller.progress.update(adopted_expected_state=state, execution_recovery=proof)
        controller.owns_training = True
        controller.publish('resuming_after_resource_failure')
        for name in ('latest_accepted_checkpoint.json','latest_evaluated_checkpoint.json'):
            write_json(output/name, accepted)
        # Future boundary handovers must be able to inspect this live job.
        # Retain the execution locks until this controller exits.
        for lock in handovers:
            fcntl.flock(lock, fcntl.LOCK_UN)
        try:
            current = accepted
            while current['step'] < job['target_step']:
                end = next_segment_end(current['step'], job['target_step'], interval=job['validation_interval'])
                current = controller.segment(current, controller.progress['best'], historical, controller.job['data_config'], end)
                controller.remember_best(current)
                write_json(output/'latest_accepted_checkpoint.json', current)
            controller.publish('completed_'+str(job['target_step']))
        except Exception:
            controller.progress['error'] = traceback.format_exc(); controller.publish('failed')
            raise


if __name__ == '__main__':
    main()
