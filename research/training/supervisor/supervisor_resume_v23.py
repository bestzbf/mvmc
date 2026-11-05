#!/usr/bin/env python3
"""Resume-only recovery of the already-admitted six-card BOOT checkpoint.

Root publishes GPU7 prelaunch and CPU publisher handoffs. This owner retains
all18 GPU locks across both native launches. It never signals old/foreign/eval
processes or rewrites source, recipes, environments, or immutable checkpoints.
"""
import argparse
from contextlib import ExitStack
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
from control_common import GPUS,Leases,alive,atomic,checkpoint,descendants,gpu_snapshot,pinned,proc,read,sha,stop_owned


def owned_json(path):
    p=Path(path)
    if p.is_symlink() or p.stat().st_uid!=os.getuid():raise ValueError('Handoff must be an owned regular file')
    return read(p)
def field(tree,path):
    for key in path.split('.'):tree=tree[key]
    return tree

class Supervisor:
    def __init__(self,plan):
        self.plan=plan;self.control=Path(plan['control_dir']);self.child=None;self.child_identity=None;self.known=[]
        self.leases=None;self.promoted=False;self.eval=None
        if list(map(int,plan['physical_gpus']))!=list(GPUS):raise ValueError('Only exactsix0/1/2/5/6/7')
        self.status=dict(stage='starting',pid=os.getpid(),control_dir=str(self.control),controller=proc(os.getpid()),physical_gpus=list(GPUS),gpu_uuids=list(GPUS.values()),production_promoted=False)
    def publish(self,stage,**values):
        self.status.update(stage=stage,updated_unix=time.time(),**values);atomic(self.control/'progress.json',self.status)
        atomic(self.control/'active_training.json',self.status)
        if self.promoted and self.plan.get('canonical_active_path'):atomic(self.plan['canonical_active_path'],self.status)
    def wait_file(self,path,stage):
        while not Path(path).exists():self.publish(stage);self.idle();time.sleep(2)
        return owned_json(path)
    def idle(self):
        if (self.control/'STOP').exists():raise RuntimeError('Explicit supervisorSTOP requested')
        if self.eval is not None:self.eval.poll()
    def free_selected(self):
        inventory,apps=gpu_snapshot()
        bad=[p for p in apps if p['uuid'] in GPUS.values()]
        if bad or any(inventory[i]['memory_mib']>=1024 for i in GPUS):return False,bad
        return True,[]
    def acquire(self):
        while True:
            trial=Leases()
            try:trial.acquire()
            except BlockingIOError:
                self.publish('waiting_for_all18_leases');self.idle();time.sleep(2);continue
            free,bad=self.free_selected()
            if free:
                self.leases=trial;self.publish('all18_leases_held',leases=trial.receipt());return
            trial.close();self.publish('waiting_for_selected_gpu_occupancy',occupied=bad);self.idle();time.sleep(2)
    def run_child(self,argv,cwd,env,log_path,pass_extra=()):
        if self.child is not None:raise RuntimeError('One owned training/helperchild at a time')
        if not isinstance(argv,list) or not argv or any(not isinstance(v,str) for v in argv):raise ValueError('Require structured argv, never shell')
        handle=Path(log_path).open('x')
        child=subprocess.Popen(argv,cwd=cwd,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=self.leases.fds+tuple(pass_extra))
        identity=proc(child.pid)
        if identity is None:child.wait();handle.close();raise RuntimeError('Child exited before identity publication')
        self.child,self.child_identity,self.known=child,identity,[identity]
        self.publish(self.status['stage'],child=identity,child_log=str(log_path),child_argv=argv)
        return handle
    def clear_child(self):
        self.child=None;self.child_identity=None;self.known=[]
    def refresh_owned(self):
        if self.child is not None and alive(self.child_identity):
            records={(p['pid'],p['start']):p for p in self.known}
            records.update(((p['pid'],p['start']),p) for p in descendants(self.child_identity))
            self.known=list(records.values())
    def owned_gpu_remains(self):
        known={p['pid']:p for p in self.known}
        _,apps=gpu_snapshot()
        return [a for a in apps if a['uuid'] in GPUS.values() and a['pid'] in known and alive(known[a['pid']])]
    def wait_owned_exit(self,timeout=60):
        deadline=time.monotonic()+timeout
        while any(alive(p) for p in self.known) or self.owned_gpu_remains():
            if time.monotonic()>deadline:raise RuntimeError('Owned rank/CUDA processes remain')
            time.sleep(.5)
    def stop_child(self):
        if self.child is not None:
            stop_owned(self.child,self.child_identity,self.known)
            self.wait_owned_exit();self.clear_child()
    def cleanup_or_hold(self):
        try:self.stop_child()
        except BaseException as exc:
            while True:
                try:remaining=any(alive(p) for p in self.known) or bool(self.owned_gpu_remains())
                except BaseException:remaining=True  # unknown driver state retains18leases
                if not remaining:break
                self.publish('cleanup_blocked_retaining_leases',cleanup_error=str(exc))
                try:time.sleep(5)
                except BaseException:continue
            self.clear_child()
    def prelaunch(self,source):
        request=self.wait_file(self.control/'prelaunch_request.json','awaiting_root_gpu7_helper_request')
        if request['source_checkpoint_sha256']!=source['source']['sha256']:raise ValueError('Prelaunch bound to a stale source')
        root=Path(request['experiment_root']);template=request['guard_template'];code=Path(request['harness_dir'])
        manifest=read(root/'experiment.json')
        if manifest['checkpoint']['sha256']!=source['source']['sha256']:raise ValueError('GPU7 helper manifest uses a stale source')
        for name,digest in template['harness_sha256'].items():
            if sha(code/name)!=digest:raise ValueError('Prelaunch harness changed')
        token=secrets.token_hex(24)
        with (root/'.gpu7_experiment.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            guard=dict(template,gpu_uuid=GPUS[7],owner_pid=os.getpid(),token=token,lease_records=self.leases.receipt())
            atomic(root/'guard.json',guard)
            env=os.environ|request.get('environment',{})|{'CUDA_VISIBLE_DEVICES':GPUS[7],'TRAINING_OPT_GUARD_TOKEN':token,'PYTHONDONTWRITEBYTECODE':'1'}
            self.publish('running_root_gpu7_rng_helper')
            handle=self.run_child(request['argv'],request['cwd'],env,self.control/'gpu7_prelaunch.log',(lock.fileno(),))
            try:
                while self.child.poll() is None:self.refresh_owned();self.idle();time.sleep(.5)
                code_result=self.child.returncode
                if code_result:raise RuntimeError('GPU7 helper failed: '+str(code_result))
            finally:handle.close()
            self.wait_owned_exit();self.clear_child()
            result=owned_json(request['result'])
            if field(result,'provenance.source_checkpoint_sha256')!=source['source']['sha256']:raise ValueError('GPU7 result source mismatch')
            for name in request['required_true_fields']:
                if field(result,name) is not True:raise ValueError('GPU7 result verification failed: '+name)
            atomic(self.control/'gpu7_prelaunch_complete.json',dict(result=request['result'],result_sha256=sha(request['result']),source_checkpoint_sha256=source['source']['sha256']))
    def activation(self,source):
        a=self.wait_file(self.control/'activation_ready.json','awaiting_published_checkpoint')
        receipt=pinned(a['publisher_receipt']['path'],a['publisher_receipt']['sha256'])
        marker=pinned(receipt['marker'],receipt['marker_sha256'])
        if (not receipt.get('committed') or not receipt.get('directory_fsync_completed')
                or not receipt['independent_verification'].get('exact_all_values_verified')
                or marker['source_sha256']!=source['source']['sha256']
                or marker['source_step']!=source['source']['step']
                or marker['source']!=source['source']['path']
                or marker['mode']!='new_data_stream_reset'
                or marker['target_data_sha256']!=a['data_config']['sha256']):raise ValueError('Publisher proof/source binding failed')
        verify=pinned(a['publisher_verify']['path'],a['publisher_verify']['sha256'])
        if not all(verify.get(k) is True for k in ('committed','directory_fsync_completed','exact_all_values_verified')) or verify['marker_sha256']!=receipt['marker_sha256']:
            raise ValueError('Fresh pinned-marker bundle verification required')
        if sha(receipt['checkpoint'])!=receipt['checkpoint_sha256']:raise ValueError('Published canonical6 bytes changed')
        deployment=pinned(a['deployment']['path'],a['deployment']['sha256'])
        for path,digest in deployment['files'].items():
            if sha(path)!=digest:raise ValueError('Future deployment changed: '+path)
        for key in ('data_config','loss_config'):
            if sha(a[key]['path'])!=a[key]['sha256']:raise ValueError('Final recipe input changed')
        a['checkpoint']=dict(path=receipt['checkpoint'],sha256=receipt['checkpoint_sha256'],step=source['source']['step'])
        a['_deployment']=deployment
        adoption=dict(source_checkpoint=source['source'],published_checkpoint=a['checkpoint'],source_root=deployment['candidate_source'],base_commit=deployment['source_base_commit'],deployment=a['deployment'],data_config=a['data_config'],loss_config=a['loss_config'],global_lr_plan=source['source']['training_state']['lr_plan'],geometry_lr_options=dict(multiplier=2.0,warmup_steps=100,start_step=source['source']['step']),new_data_stream_reset_once=True,evaluation_gpus=[3,4],control_dir=str(self.control))
        atomic(self.control/'adoption_receipt.json',adoption)
        self.status.update(repo=self.plan['evaluation_repo'],source_root=deployment['candidate_source'],base_commit=deployment['source_base_commit'],deployment_sha256=a['deployment']['sha256'],data_config=a['data_config'],loss_config=a['loss_config'],source_checkpoint=a['checkpoint'],old5_source_checkpoint=dict(path=source['source']['path'],sha256=source['source']['sha256'],step=source['source']['step']),geometry_warmup_anchor=source['source']['step'],global_lr_start=95000,global_lr_end=200000,evaluation_gpus=[3,4],adoption_receipt=str(self.control/'adoption_receipt.json'),adoption_receipt_sha256=sha(self.control/'adoption_receipt.json'))
        return a
    def gate_spec(self,a,cp,phase,attempt):
        spec=dict(schema='six-rank-execution-admission-v22',phase=phase,checkpoint=cp,
            data_config=a['data_config'],loss_config=a['loss_config'],world_size=6,batch_size=1,gradient_accumulation=1,
            geometry_lr_options=dict(multiplier=2.,warmup_steps=100),expect_data_reset=phase=='boot',
            deployment_sha256=a['deployment']['sha256'],ordered_gpu_uuids=list(GPUS.values()),required_update_count=1)
        if phase=='boot':spec['expected_start_cursor']=dict(data_epoch=0,batches_consumed=0)
        path=attempt/'gate_spec.json';atomic(path,spec);return path,sha(path)
    def gate_ready(self,proof,source_step):
        return all((proof/'startup'/f'rank_{r:02d}.json').exists() and (proof/'updates'/f'step_{source_step+1:08d}'/f'rank_{r:02d}.json').exists() for r in range(6))
    def verify_gate(self,a,attempt,spec,digest):
        report=attempt/'gate_report.json'
        argv=a['gate_argv']+['--proof-root',str(attempt/'proof'),'--gate-spec',str(spec),'--gate-spec-sha256',digest,'--report',str(report)]
        done=subprocess.run(argv,cwd=a['gate_cwd'],env=os.environ|{'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
        (attempt/'gate.stdout').write_text(done.stdout);(attempt/'gate.stderr').write_text(done.stderr)
        if done.returncode:raise RuntimeError('Six-rank admission parser failed')
        result=owned_json(report)
        if not all(result.get(k) is True for k in ('passed','startup_all6_exact','routes_all6_verified','proof_scope_no_statistical_quality_claim')):raise RuntimeError('Six-rank startup/route admission failed')
        return str(report)
    def setup_evaluation(self,a):
        item=a.get('evaluation')
        if item is None:raise ValueError('Frozen evaluation queue3/4 adapter required before activation')
        if sha(item['module'])!=item['module_sha256'] or sha(item['template'])!=item['template_sha256']:raise ValueError('Evaluation adapter/template changed')
        sys.path.insert(0,str(Path(item['module']).parent))
        spec=importlib.util.spec_from_file_location('dfrt_eval_queue_v22',item['module']);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        self.eval=module.EvaluationQueue(item['template'],self.control)
    def phase(self,a,cp,phase,end):
        attempt=self.control/'attempts'/phase;attempt.mkdir(parents=True)
        spec,digest=self.gate_spec(a,cp,phase,attempt);proof=attempt/'proof';run=attempt/'training'
        wrapper=a['_deployment']['python_wrapper']
        extra=a['native_launch_args']
        denied={'--allow-shared','--resume','--pretrained','--gpus','--python','--steps','--output','--save-every','--migrate-data-config','--geometry-lr-multiplier','--geometry-lr-warmup-steps','--execution-optimizations','--override-lr','--lr-schedule','--lr-end-step','--reset-data-cursor','--reset-time-projections','--no-checkpointing','--reuse-shared-computation','--reuse-decoder-kv','--freeze-pretrained-backbone','--freeze-geometry'}
        if any(v.split('=',1)[0] in denied for v in extra):raise ValueError('Reserved/unsafe native launch argument')
        for flag,key in (('--data-config','data_config'),('--loss-config','loss_config')):
            if extra.count(flag)!=1 or str(Path(extra[extra.index(flag)+1]).resolve())!=str(Path(a[key]['path']).resolve()):
                raise ValueError('Native launch recipe file binding mismatch: '+flag)
        argv=[wrapper,'-m','multicamera4rc.launch','--gpus','0,1,2,5,6,7','--resume',cp['path'],'--output',str(run),'--steps',str(end),'--save-every','1' if phase=='boot' else '100',*extra]
        if phase=='boot':argv+=['--migrate-data-config','--geometry-lr-multiplier','2.0','--geometry-lr-warmup-steps','100']
        env=os.environ|a.get('environment',{})|{'CUDA_VISIBLE_DEVICES':','.join(GPUS.values()),'DFRT_TRAIN_EXECUTION':'fa4_metadata_prepared_kv','DFRT_FUTURE_DEPLOYMENT_SHA256':a['deployment']['sha256'],'DFRT_SIX_RANK_PROOF_ROOT':str(proof),'DFRT_SIX_RANK_GATE_SPEC':str(spec),'DFRT_SIX_RANK_GATE_SPEC_SHA256':digest,'PYTHONDONTWRITEBYTECODE':'1'}
        for _ in range(60):
            free,bad=self.free_selected()
            if free:break
            self.publish('waiting_previous_owned_cuda_contexts_exit',occupied=bad);self.idle();time.sleep(1)
        else:raise RuntimeError('Selected GPU occupied; no sharing or foreign termination')
        sys.path.insert(0,self.plan['evaluation_repo'])
        from scripts.async_training_state import UpdateAudit as NativeUpdateAudit
        from native_update_audit_v23 import top_level_audit
        UpdateAudit=top_level_audit(NativeUpdateAudit)
        log=attempt/'training.log';audit=UpdateAudit(log,cp['step'],6,1,6);checked=set();gate=None;last_publish=0
        self.publish('six_rank_'+phase+'_verifying',run=str(run),training_phase=phase,resume_checkpoint=cp['path'],target_step=end)
        handle=self.run_child(argv,self.plan['evaluation_repo'],env,log)
        try:
            while True:
                self.refresh_owned();audit.poll();self.idle()
                if audit.latest:self.status['last_observed_update']=audit.latest
                if gate is None and self.gate_ready(proof,cp['step']):
                    gate=self.verify_gate(a,attempt,spec,digest)
                    if phase=='resume':self.promoted=True;self.publish('training_200000',gate_report=gate,production_promoted=True)
                for path in sorted(run.glob('step_*.pt')):
                    step=int(path.stem.split('_')[1])
                    if step in checked or step>audit.last:continue
                    info=checkpoint(path,6);checked.add(step)
                    self.status['latest_complete_checkpoint']=dict(path=str(path),step=step)
                    if self.eval is not None and (step%5000==0 or step==200000):self.eval.enqueue(str(path),step)
                    if phase=='resume':self.prune(run,path)
                if time.monotonic()-last_publish>15:
                    self.publish(self.status['stage']);last_publish=time.monotonic()
                code=self.child.poll()
                if code is not None:break
                time.sleep(2)
            audit.poll()
            if code:raise RuntimeError('Owned native launcher failed: '+str(code))
            if audit.last!=end:raise RuntimeError('Native training exited with missing updates')
            if gate is None:gate=self.verify_gate(a,attempt,spec,digest)
            final=run/'latest.pt';info=checkpoint(final,6)
            if info['step']!=end:raise RuntimeError('Final full checkpoint step mismatch')
            self.wait_owned_exit();self.clear_child()
            return dict(path=str(final),step=end,sha256=sha(final))
        finally:handle.close()
    def prune(self,run,current):
        paths=sorted(Path(run).glob('step_*.pt'))
        protected=set(self.eval.protected_paths()) if self.eval else set()
        for p in paths[:-3]:
            if p==current or str(p) in protected or int(p.stem.split('_')[1])%10000==0:continue
            if p.is_symlink():raise RuntimeError('Unexpected symlink in owned checkpoint output')
            p.unlink()
    def verify_prior_boot(self,a,prior):
        attempt=prior/'attempts/boot';spec=attempt/'gate_spec.json'
        frozen=read(spec);digest=sha(spec)
        if (frozen['checkpoint']!=a['checkpoint'] or frozen['phase']!='boot'
                or frozen['deployment_sha256']!=a['deployment']['sha256']
                or frozen['data_config']!=a['data_config'] or frozen['loss_config']!=a['loss_config']):
            raise ValueError('Prior boot proof belongs to another admission')
        report=self.control/'prior_boot_reverified.json'
        argv=a['gate_argv']+['--proof-root',str(attempt/'proof'),'--gate-spec',str(spec),'--gate-spec-sha256',digest,'--report',str(report)]
        result=subprocess.run(argv,cwd=a['gate_cwd'],env=os.environ|{'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
        (self.control/'prior_boot_gate.stdout').write_text(result.stdout);(self.control/'prior_boot_gate.stderr').write_text(result.stderr)
        verdict=owned_json(report) if report.exists() else {}
        if result.returncode or not all(verdict.get(k) is True for k in ('passed','startup_all6_exact','routes_all6_verified')):
            raise RuntimeError('Existing successful boot proof failed independent re-verification')
        return str(report)
    def serve(self):
        self.control.mkdir(parents=True,exist_ok=True)
        with (self.control/'supervisor.lock').open('a') as own:
            fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                prior=Path(self.plan['prior_control']);failed=owned_json(prior/'progress.json')
                if alive(failed['controller']):raise RuntimeError('Previous owner remains live')
                source=pinned(self.plan['source_stopped'],self.plan['source_stopped_sha256'])
                if source.get('stage')!='complete' or not source.get('old_processes_exited'):
                    raise ValueError('Original old5 provenance missing')
                self.acquire()
                a=pinned(self.plan['activation_ready'],self.plan['activation_ready_sha256'])
                # Reuse only the already pinned publisher/source/deployment
                # admission. No RNG generation, publication, boot or reset runs.
                # activation() reads this new private handoff alias; keep old
                # failed provenance immutable and pin its exact contents above.
                atomic(self.control/'activation_ready.json',a)
                a=self.activation(source)
                boot_report=self.verify_prior_boot(a,prior)
                boot_path=Path(self.plan['boot_checkpoint'])
                if sha(boot_path)!=self.plan['boot_checkpoint_sha256']:raise ValueError('BOOT checkpoint bytes changed')
                info=checkpoint(boot_path,6)
                if (info['step']!=a['checkpoint']['step']+1
                        or info['optimizer_groups']!=['backbone','heads','refinement','geometry']):
                    raise ValueError('Recovery requires the saved4group first boot update')
                boot=dict(path=str(boot_path),step=info['step'],sha256=self.plan['boot_checkpoint_sha256'])
                atomic(self.control/'recovery_receipt.json',dict(prior_control=str(prior),prior_boot_gate=boot_report,
                    resume_input_checkpoint=boot,original_old5_source=source['source'],new_data_migration=False,
                    geometry_warmup_restarted=False,controller_fix='top-levelJSON log frames; unchanged native accept'))
                self.status.update(resume_input_checkpoint=boot,resume_only=True,prior_control=str(prior),prior_boot_gate=boot_report)
                self.setup_evaluation(a);self.eval.bind_checkpoint(boot['path'],boot['sha256'])
                self.eval.enqueue(boot['path'],boot['step']);self.eval.poll()
                final=self.phase(a,boot,'resume',200000)
                self.publish('training_complete_validation_pending',final_checkpoint=final)
                self.leases.close();self.leases=None
                while self.eval.pending():self.eval.poll();self.publish('training_complete_validation_pending');time.sleep(10)
                self.publish('completed_200000')
            except BaseException as exc:
                self.cleanup_or_hold();self.publish('failed_preserving_source_and_complete_checkpoints',error={'type':type(exc).__name__,'message':str(exc)},rollback_automatic=False)
                raise
            finally:
                if self.leases is not None:self.leases.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('check_plan','serve'));p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--lease-self',action='store_true');a=p.parse_args();plan=pinned(a.plan,a.plan_sha256)
    if a.command=='check_plan':
        if plan['physical_gpus']!=list(GPUS):raise ValueError('Unexpected GPU selection')
        print({'plan_valid':True,'source_pause_required':True,'no_GPU_or_process_action':True});return
    if not a.lease_self:p.error('Persistent owner requires explicit --lease-self')
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('root terminated supervisor')))
    Supervisor(plan).serve()
if __name__=='__main__':main()
