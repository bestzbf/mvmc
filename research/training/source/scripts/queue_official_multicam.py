"""Wait for full conversion, audit data, then dispatch a boundary admission.

The queue never signals a process. The dispatched continuation verifies the
exact main process identity before replacing its controller alone.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from scripts.continue_quality_recovery import digest
from scripts.cutover_native_motion_stereo import proc, write_json
from scripts.continue_official_multicam import OfficialDataController, verify_mixture, cross_camera_loss_config
from scripts.continue_quality_recovery import validate_checkpoint


def now():return datetime.now(timezone.utc).isoformat()


def conversion_ready(root):
    path=Path(root)/'conversion_manifest.json'
    if not path.exists():return False
    manifest=json.loads(path.read_text())
    if manifest['status'] in ('failed','interrupted'):
        raise RuntimeError('MV-Kubric conversion did not finish: '+str(manifest.get('error')))
    if manifest['status']!='complete':return False
    from multicamera4rc.datasets.mv_kubric_inventory import expected_train_ids
    expected = expected_train_ids(manifest)
    if manifest.get('converted_scenes')!=len(expected) or manifest.get('official_train_scene_ids')!=expected:
        raise ValueError('completed conversion has an incomplete official inventory')
    return True


def protected_experiments(paths, integration_repo, main_gpus):
    """Preserve both paired experiments and their automatic admission contract."""
    files={}
    for path in paths:
        path=Path(path);job=json.loads(path.read_text())
        if Path(job['repo']).resolve()==Path(integration_repo).resolve():
            raise ValueError('integration must use a separate experiment checkout')
        if set(main_gpus)&(set(job['gpu_candidates'])-set(job['protected_gpus'])):
            raise ValueError('experiment and main GPU allocations overlap')
        frozen=path.parent/'frozen_job.json'
        if not frozen.exists():return None
        value=json.loads(frozen.read_text())
        if bool(value.get('auto_adopt',False)) != bool(job.get('auto_adopt',False)):
            raise ValueError('frozen and launch auto-admission policies disagree')
        files[str(path)]=digest(path)
        files[str(frozen)]=digest(frozen)
        for name,sha in value['config_hashes'].items():
            if digest(name)!=sha:raise ValueError('frozen experiment input changed: '+name)
            files[name]=sha
        if value.get('auto_adopt',False):
            # The structure trial compares its frozen data with the main recipe
            # again when admitting a successful arm. Freezing alone is not enough:
            # changing main data mid-trial would invalidate that later admission.
            progress=path.parent/'progress.json'
            stage=json.loads(progress.read_text()).get('stage') if progress.exists() else None
            settled={
                'adopted_main_training',
                'no_candidate_passed_main_unchanged',
                'validation_passed_pstudio_regressed_main_unchanged',
                'paired_passed_boundary_failed_main_unchanged',
                'passed_boundary_raced_training_main_unchanged',
                'passed_main_not_running_requires_review',
            }
            if stage not in settled:return None
    return files


def required_loss_accepted(queue, active):
    """Do not replace a loss handover while its first segment is unvalidated."""
    weight=queue.get('requires_cross_camera_3d_weight')
    if weight is None:
        return True
    accepted=json.loads((Path(active['output_root'])/'latest_accepted_checkpoint.json').read_text())
    state=validate_checkpoint(accepted['checkpoint'],accepted['step'],world_size=4,batch_size=1)
    return state['loss_config'].get('cross_camera_3d_weight',0)==weight


def build_controller(queue,active,admission,protected,root):
    root=Path(root);control=root/'control';output=root/'training'
    if control.exists() or output.exists():raise ValueError('dispatch requires fresh output directories')
    status=json.loads((Path(active['control_dir'])/'progress.json').read_text())
    accepted=json.loads((Path(active['output_root'])/'latest_accepted_checkpoint.json').read_text())
    if (status['stage']!=f"train_{active['phase_target_step']}_try0"
            or accepted['checkpoint']!=active['selected_checkpoint']
            or accepted['step']!=active['resume_step']):
        raise RuntimeError('main is not at an ordinary segment with an accepted parent')
    identity=proc(active['pid']);child=proc(status['child_pid'])
    if not identity or not child:raise RuntimeError('main controller or launcher exited')
    if active['data_config']!=queue['base_config'] or active['physical_gpus']!=queue['physical_gpus']:
        raise ValueError('main data or GPU contract changed while preparing official data')
    if active['phase_target_step']>=active['target_step']:
        raise ValueError('main has no remaining segment for official-data training')
    control.mkdir(parents=True);output.mkdir()
    job=deepcopy(active);job.pop('audited_data_config_replacement',None);job.pop('learning_rate_trial',None)
    job.update(repo=queue['repo'],main_active_path=queue['main_active_path'],
        environment=dict(active['environment'],PYTHONPATH=queue['repo']),
        control_dir=str(control),output_root=str(output),run=str(output),commit=queue['commit'],
        previous=deepcopy(active),previous_identity=identity,previous_start=identity['start'],
        adopted_training=dict(child,run=active['run'],begin=accepted['step'],end=active['phase_target_step'],
            data_config=active['data_config'],log=status['child_log']),
        adopted_wait_hours=48,max_quality_retries=0,official_data_config=queue['config'],
        admission_kind=queue.get('admission_kind','official_data'),
        runtime_equivalence=queue['runtime_equivalence'],tests_proof=queue['tests_proof'],protected_files=protected,
        log=str(root/'controller.log'),stage='prepared_official_data_handover')
    for name,record in [('initial',accepted),('anchor',status['best'])]:
        path=control/(name+'_record.json');write_json(path,record);job[name+'_record']=str(path)
    # A subsequent data-only controller must inherit the active loss from its
    # accepted checkpoint, not re-apply the previous loss-admission override.
    for key in ('cross_camera_loss_config','cross_camera_3d_weight','cross_camera_tests_proof'):
        job.pop(key,None)
    evidence=[queue['base_config'],queue['config'],queue['runtime_equivalence'],queue['tests_proof']]
    if job['admission_kind']!='cross_camera_only':
        proof=json.loads(Path(admission).read_text())
        job.update(official_admission=str(admission),panoptic_audit=queue['panoptic_audit'])
        evidence += [str(admission),proof['conversion_manifest'],queue['panoptic_audit'],
                     str(Path(admission).parent/'packed_gt.jsonl'),str(Path(admission).parent/'loader_gt.jsonl')]
    job['official_evidence']={p:digest(p) for p in evidence}
    if queue.get('cross_camera_3d_weight',0):
        # Resolve against the current accepted main recipe; independent trials
        # retain their own checkpoint/configs and never overwrite this state.
        state=validate_checkpoint(accepted['checkpoint'],accepted['step'],world_size=4,batch_size=1)
        loss_path=root/'cross_camera_loss.json'
        write_json(loss_path,cross_camera_loss_config(state['loss_config'],queue['cross_camera_3d_weight']))
        job.update(cross_camera_3d_weight=queue['cross_camera_3d_weight'],
                   cross_camera_loss_config=str(loss_path),cross_camera_tests_proof=queue['cross_camera_tests_proof'])
        for p in (str(loss_path),queue['cross_camera_tests_proof']):
            job['official_evidence'][p]=digest(p)
    write_json(control/'job.json',job)
    return control/'job.json'


def run(path):
    queue=json.loads(Path(path).read_text());root=Path(queue['output_root']);root.mkdir(parents=True,exist_ok=True)
    def publish(stage,**details):
        write_json(root/'progress.json',dict(stage=stage,pid=os.getpid(),updated_at=now(),**details))
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for p,sha in queue['frozen_files'].items():
                if digest(p)!=sha:raise ValueError('registered queue input changed: '+p)
            loss_only=queue.get('admission_kind')=='cross_camera_only'
            if loss_only:
                if queue['config']!=queue['base_config'] or queue.get('cross_camera_3d_weight',0)<=0:
                    raise ValueError('loss-only queue requires unchanged data and a positive loss weight')
            else:
                verify_mixture(queue['base_config'],queue['config'],json.loads(Path(queue['panoptic_audit']).read_text()))
            while not loss_only and not conversion_ready(queue['mv_root']):
                progress=Path(queue['mv_root'])/'full_conversion_progress.json'
                data=json.loads(progress.read_text()) if progress.exists() else {}
                publish('waiting_full_conversion',converter_status=data.get('status'),
                    published_scenes=len(list((Path(queue['mv_root'])/'train').glob('*.h5'))),expected_scenes=4970)
                time.sleep(30)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=queue['repo'])
            audit=root/'audit';admission=audit/'admission.json'
            if not loss_only and not admission.exists():
                publish('auditing_full_gt_and_loader')
                command=[queue['python'],'-u','-m','scripts.audit_official_multicam_admission',
                         '--config',queue['config'],'--output',str(audit)]
                with (root/'audit.log').open('a') as log:
                    subprocess.run(command,cwd=queue['repo'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            while True:
                protected=protected_experiments(queue['protected_jobs'],queue['repo'],queue['physical_gpus'])
                active=json.loads(Path(queue['main_active_path']).read_text())
                if active.get('stage','').startswith(('failed','completed')):
                    raise RuntimeError('main is no longer running an accepted continuation: '+active['stage'])
                loss_ready=required_loss_accepted(queue,active)
                if (protected is not None and loss_ready
                        and active.get('stage')==f"train_{active.get('phase_target_step')}_try0"):
                    break
                publish('waiting_experiment_admission_or_main_training_segment',experiments_settled=protected is not None,
                        main_stage=active.get('stage'),required_loss_accepted=loss_ready)
                time.sleep(30)
            controller_path=build_controller(queue,active,admission,protected,root)
            # Complete a non-mutating verification before dispatching any code
            # allowed to replace the main controller.
            OfficialDataController(controller_path).verify_handover()
            publish('dispatching_boundary_handover',controller=str(controller_path))
            with (root/'controller.log').open('a') as log:
                child=subprocess.Popen([queue['python'],'-u','-m','scripts.continue_official_multicam',
                    '--controller',str(controller_path)],cwd=queue['repo'],env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            publish('boundary_controller_running',controller_pid=child.pid,controller=str(controller_path))
            code=child.wait()
            publish('completed' if code==0 else 'controller_failed',returncode=code,controller=str(controller_path))
        except Exception:
            publish('failed_before_activation',error=traceback.format_exc(),main_left_running=True)
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queue',required=True)
    run(p.parse_args().queue)
