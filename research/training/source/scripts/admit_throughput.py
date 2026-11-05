"""Checkpoint-boundary performance admission, after complete 95,500 validation.

The CPU-only watcher leaves training and all independent trials untouched.
Once all five candidate-validation suites have been launched, only the old
supervisor is suspended to prevent its next segment racing admission. Its GPU
evaluation children finish normally. Every report/case/hash and exit status is
verified before that supervisor is replaced. Failed performance experiments
resume the unchanged recipe from the validated checkpoint, never trial weights.
"""
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
import traceback

from multicamera4rc.config import MultiCamera4RCConfig
from multicamera4rc.checkpoint import config_from_checkpoint
from multicamera4rc.launch import gpu_snapshot
from multicamera4rc.quality import error_panel, compare_errors
from scripts.continue_refinement_training import RefinementController
from scripts.continue_quality_recovery import training_argv, validate_checkpoint, digest
from scripts.cutover_native_motion_stereo import proc, write_json
from scripts.migrate_dynamic_validation import SUITES, verify_suite_report, suite_paths, build_validation_contract
from scripts.query_pair_main_admission import clean_runtime, verify_files, try_lease, free_indices, continuation_health, require_cursor_progress
from scripts.native_training_log import training_rows


def read(p):
    return json.loads(Path(p).read_text())


def speed_comparison(reference, candidate, window=8):
    if len(reference)!=24 or len(candidate)!=24:
        raise ValueError('speed experiments require all 24 updates in each arm')
    for rows in (reference,candidate):
        for r in rows:
            if (r['world_size']!=4 or r['effective_batch_size']!=4 or r['supervision']!='dense'
                    or r.get('optimizer_step_skipped') or not math.isfinite(r['loss'])):
                raise ValueError('speed arm changed training contract or skipped an update')
    for a,b in zip(reference,candidate):
        if any(a.get(k)!=b.get(k) for k in ('step','lr','backbone_lr','refinement_lr')):
            raise ValueError('matched speed arms changed the per-update learning rates')
    for first in range(0,24,window):
        counts=[]
        for rows in (reference,candidate):
            counts.append(Counter((name,index) for r in rows[first:first+window] for b in r['rank_batches']
                for name,index in zip(b['dataset'],b['sample_index'])))
        if counts[0]!=counts[1]:raise ValueError('matched speed windows consumed different sample multisets')
    # Discard the first full window, including worker/CUDA startup and warmup.
    old=[r['step_timing']['seconds'] for r in reference[window:]]
    new=[r['step_timing']['seconds'] for r in candidate[window:]]
    if not all(math.isfinite(v) and v>0 for v in old+new):raise ValueError('invalid measured timing')
    ratios=[sum(new[i:i+window])/sum(old[i:i+window]) for i in range(0,len(old),window)]
    ratio=sum(new)/sum(old)
    return dict(reference_seconds=statistics.mean(old),candidate_seconds=statistics.mean(new),
        speedup=1/ratio,ratio=ratio,window_ratios=ratios,measured_updates=16,
        identical_sample_multisets=True,passed=ratio<=.95 and max(ratios)<=1.02,
        peak_allocated_gib=max(r['step_timing'].get('peak_allocated_gib',0) for r in candidate))


class Admission:
    @property
    def boundary_step(self):
        return self.job.get('boundary_step',95500)

    def __init__(self,path):
        self.path=Path(path);self.job=read(path)
        self.root=Path(self.job['repo']);self.out=Path(self.job['output']);self.out.mkdir(parents=True,exist_ok=True)
        self.previous=read(self.job['previous_job'])
        self.pid=self.previous['pid'];self.identity=proc(self.pid)
        if not self.identity or self.identity['start']!=self.job['previous_start']:
            raise ValueError('recorded main supervisor changed')
        self.suspended=False;self.replaced=False
        self.state=dict(stage='waiting_for_95500_validation',pid=os.getpid(),previous_pid=self.pid)
        self.write()

    def write(self,stage=None,**kw):
        if stage:self.state['stage']=stage
        self.state.update(kw,updated_at=time.time());write_json(self.out/'progress.json',self.state)
        if self.replaced and self.state['stage'] not in ('main_starting','main_running'):
            active=dict(self.previous,pid=os.getpid(),stage='throughput_'+self.state['stage'],
                repo=str(self.root),commit=self.job['commit'],control_dir=str(self.out),
                child_pid=self.state.get('child_pid'),child_log=self.state.get('child_log'),
                run=self.state.get('benchmark_run',self.previous.get('run')))
            write_json(Path(self.previous['main_active_path']),active)

    def same_supervisor(self):
        p=proc(self.pid)
        return p and p['start']==self.identity['start'] and p['cmd']==self.identity['cmd']

    def record(self,label,checkpoint):
        paths={s:str(Path(self.previous['output_root'])/'validation'/label/s/'evaluation.json') for s in SUITES}
        record=dict(label=label,checkpoint=str(checkpoint),reports=paths,
            validation_contract=self.previous['validation_contract'])
        entries={}
        for suite,path in paths.items():
            if not Path(path).exists():return None
            try:report=read(path)
            except json.JSONDecodeError:return None
            plan=read(suite_paths(self.previous,suite)[0])
            entry=report.get('checkpoints',{}).get(label,{})
            if len(entry.get('records',[]))<len(plan['cases']):return None
            entries[suite]=verify_suite_report(suite,report,record,plan,self.previous['validation_contract'])
        record.update(errors=error_panel(entries))
        steps={e['step'] for e in entries.values()}
        if len(steps)!=1:raise ValueError('validation checkpoint steps disagree')
        record['step']=steps.pop()
        return record

    def validated_boundary(self):
        deadline=time.monotonic()+24*3600
        control=Path(self.previous['control_dir']);label='refinement_95500'
        while time.monotonic()<deadline:
            if not self.same_supervisor():raise RuntimeError('main supervisor changed before admission')
            active=read(self.previous['main_active_path'])
            if active['pid']!=self.pid:raise RuntimeError('another main job became active')
            progress=read(control/'progress.json')
            if progress['stage']=='failed':raise RuntimeError('main failed; performance cutover not authorized by checks')
            children=progress.get('validation_children',{})
            # Syn4D is the second task on slot 0. Other slots are either alive
            # or finished. No validation child still needs the parent to launch.
            launched=all(s in children or (Path(self.previous['output_root'])/'validation'/label/s/'evaluation.json').exists()
                         for s in ('native_dynamic','static','variable'))
            if progress['stage']=='validate_'+label and 'syn4d' in children and launched:
                if build_validation_contract(self.previous)!=self.previous['validation_contract']:
                    raise ValueError('validation inputs changed')
                os.kill(self.pid,signal.SIGSTOP);self.suspended=True
                checkpoint=progress['latest_checkpoint']
                self.write('finishing_existing_validation',checkpoint=checkpoint,children=children)
                break
            if progress.get('step',0)>95500:raise RuntimeError('missed boundary; refusing to discard newer training updates')
            time.sleep(2)
        else:raise TimeoutError('95,500 validation did not start within 24 hours')
        deadline=time.monotonic()+12*3600
        while time.monotonic()<deadline:
            if not self.same_supervisor():raise RuntimeError('suspended supervisor was replaced externally')
            alive=[]
            for pid in children.values():
                identity=proc(pid)
                if identity and identity['state']!='Z':alive.append(pid)
                elif identity:
                    fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
                    if int(fields[49]):raise RuntimeError('existing validation child exited unsuccessfully')
            if not alive:
                candidate=self.record(label,checkpoint)
                if candidate is None:raise RuntimeError('validation exited without complete reports')
                break
            time.sleep(2)
        else:raise TimeoutError('existing validation did not complete')
        baseline=self.record('baseline_95000',self.previous['baseline_checkpoint'])
        if baseline is None or candidate['step']!=95500 or baseline['step']!=95000:
            raise ValueError('complete 95k and 95,500 fixed validation required')
        comparison=compare_errors(baseline['errors'],candidate['errors'])
        if comparison['score_ratio']>1.5:raise RuntimeError('original gross-divergence gate failed; leave original controller to report it')
        validator=object.__new__(RefinementController);validator.job=self.previous
        state=validator.check(checkpoint,95500)
        proof=read(self.job['sampling_proof'])
        if (not proof['passed'] or (proof['data_epoch'],proof['cursor'])!=(state['data_epoch'],state['batches_consumed'])):
            raise ValueError('verified balancing activation cursor changed')
        clean_runtime(self.root,self.job['commit']);verify_files(self.job['proof_hashes'])
        verify_files(self.previous['protected_files'])
        root=Path(self.previous['output_root'])
        write_json(root/'validation'/label/'quality.json',candidate)
        write_json(root/'comparison_95500.json',dict(baseline=baseline,candidate=candidate,comparison=comparison,
            interpretation='same fixed full panels; verified during performance-controller handover'))
        write_json(root/'latest_evaluated_checkpoint.json',candidate)
        write_json(root/'best_checkpoint.json',candidate if comparison['accepted'] else baseline)
        self.write('validated_95500',baseline=baseline,candidate=candidate,comparison=comparison)
        return checkpoint,baseline,candidate

    def command(self,label,argv,timeout=2400):
        self.write('experiment_'+label)
        path=self.out/(label+'.log')
        with path.open('w') as log:
            child=subprocess.Popen(argv,cwd=self.root,env=os.environ|self.previous['environment']|{'PYTHONPATH':str(self.root)},
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            self.write(child_pid=child.pid,child_log=str(path))
            try:code=child.wait(timeout=timeout)
            except BaseException:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
        if code:raise RuntimeError(label+' failed; see '+str(path))
        return path

    def bench(self,name,checkpoint,flags):
        begin=self.boundary_step;end=begin+24
        run=self.out/'benchmarks'/name
        self.write(benchmark_run=str(run))
        argv=training_argv(dict(self.previous,allow_shared=False),checkpoint,self.previous['data_config'],run,end)
        argv+=['--gradient-accumulation','1','--profile-step-timing',*flags]
        path=self.command('bench_'+name,argv)
        rows=training_rows(path)
        continuation_health(path,begin,end,4)
        after=validate_checkpoint(run/'latest.pt',end,world_size=4,batch_size=1)
        before=validate_checkpoint(checkpoint,begin,world_size=4,batch_size=1)
        from scripts.migrate_dynamic_validation import require_fixed_training_state
        require_fixed_training_state(before,after);require_cursor_progress(before,after,24)
        return rows

    def experiments(self,checkpoint):
        py=self.previous['python'];equiv=self.out/'equivalence'
        self.command('equivalence',[py,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=4',
            '-m','scripts.validate_throughput_equivalence','--checkpoint',checkpoint,
            '--data-config',self.previous['data_config'],'--output',str(equiv)],timeout=1800)
        modes=set.intersection(*(set(read(equiv/f'rank_{r}.json')['passed_modes']) for r in range(4)))
        if 'shared' not in modes:raise ValueError('shared computation failed native giant-model output/gradient checks')
        # Full K/V caching is optional: its extra activation memory must pass
        # both production checks and measured full-DDP training.
        reference=self.bench('reference',checkpoint,[])
        arms={'shared':self.bench('shared',checkpoint,['--reuse-shared-computation'])}
        if 'shared_kv' in modes:
            try:arms['shared_kv']=self.bench('shared_kv',checkpoint,['--reuse-shared-computation','--reuse-decoder-kv'])
            except Exception:self.write(kv_rejection=traceback.format_exc())
        comparisons={name:speed_comparison(reference,rows) for name,rows in arms.items()}
        mode=min(comparisons,key=lambda name:comparisons[name]['candidate_seconds'])
        flags=['--reuse-shared-computation']+(['--reuse-decoder-kv'] if mode=='shared_kv' else [])
        balanced=self.bench(mode+'_balanced',checkpoint,flags+['--balance-window','8','--balance-costs',self.job['costs']])
        result=dict(cache_variants=comparisons,
                    balanced=speed_comparison(reference,balanced),selected_mode=mode)
        write_json(self.out/'speed_comparison.json',result)
        if not result['balanced']['passed']:raise ValueError('balanced/shared execution did not pass measured speed gate')
        return flags

    def migrate(self,checkpoint,flags):
        import torch
        data=torch.load(checkpoint,mmap=True,map_location='cpu',weights_only=True)
        state=deepcopy(data['training_state'])
        cfg=replace(config_from_checkpoint(data),reuse_shared_computation=True,reuse_decoder_kv='--reuse-decoder-kv' in flags)
        state['load_balancing']=dict(version=1,window_batches=8,start_epoch=state['data_epoch'],
            start_batch=state['batches_consumed'],world_size=4,dataset_size=state['dataset_size'],
            source_count=state['source_count'],max_source_time_pairs=state['max_source_time_pairs'],image_hw=read(self.job['costs']))
        # Only execution config changes; reuse the original tensor/state objects.
        data['config']=cfg.to_dict();data['training_state']=state
        dest=self.out/f'migrated_{self.boundary_step}.pt';temp=dest.with_suffix('.tmp')
        torch.save(data,temp);temp.replace(dest)
        write_json(self.out/'migration.json',dict(source=checkpoint,destination=str(dest),step=self.boundary_step,
            unchanged=['model tensors','Adam state','rank RNG','LR plan','loss','mixture','dense masks'],
            load_balancing=state['load_balancing'],model_config=cfg.to_dict()))
        return str(dest),cfg.to_dict()

    def continuation_job(self,checkpoint,step,baseline,candidate,config,accepted):
        job=deepcopy(self.previous)
        control=self.out/'main_control'
        job.update(repo=str(self.root),commit=self.job['commit'],control_dir=str(control),
            initial_checkpoint=checkpoint,continuation_step=step,initial_step=95000,
            inherited_baseline=baseline,inherited_best=(candidate if compare_errors(baseline['errors'],candidate['errors'])['accepted'] else baseline),model_config=config,
            stopped_identities=[deepcopy(self.identity)],attempt_suffix='_throughput' if accepted else '_throughput_fallback',
            allow_shared=False,performance_args=[])
        job['environment']['PYTHONPATH']=str(self.root)
        for key in ('child_pid','pid','stage','run'):job.pop(key,None)
        return job

    def launch_main(self,checkpoint,step,baseline,candidate,config,accepted):
        job=self.continuation_job(checkpoint,step,baseline,candidate,config,accepted)
        control=Path(job['control_dir']);control.mkdir(exist_ok=True)
        path=control/'job.json';write_json(path,job)
        with (control/'supervisor.log').open('a') as log:
            child=subprocess.Popen([job['python'],'-u','-m','scripts.continue_refinement_training','--controller',str(path)],
                cwd=self.root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=os.environ|job['environment'])
        self.write('main_starting',main_pid=child.pid,accepted=accepted,resume_checkpoint=checkpoint,resume_step=step)
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            if child.poll() is not None:raise RuntimeError('continuation supervisor exited; see '+str(control/'supervisor.log'))
            p=control/'progress.json'
            if p.exists() and read(p).get('child_pid'):
                self.write('main_running',main_pid=child.pid);return
            time.sleep(2)
        raise TimeoutError('continuation supervisor did not launch training')

    def run(self):
        clean_runtime(self.root,self.job['commit']);verify_files(self.job['proof_hashes'])
        if not read(self.job['cpu_proof'])['passed']:raise ValueError('CPU/DDP proof did not pass')
        try:
            checkpoint,baseline,candidate=self.validated_boundary()
            # Finished GPU children have exited. Terminate only the recorded,
            # stopped supervisor; never signal a GPU process or other trial.
            if not self.same_supervisor() or proc(self.pid)['state']!='T':
                raise RuntimeError('recorded supervisor is no longer suspended; refuse cutover')
            os.kill(self.pid,signal.SIGTERM);os.kill(self.pid,signal.SIGCONT);self.suspended=False
            for _ in range(100):
                if not self.same_supervisor() or proc(self.pid)['state']=='Z':break
                time.sleep(.1)
            else:raise RuntimeError('old supervisor did not exit')
            self.replaced=True
            write_json(Path(self.previous['control_dir'])/'transition.json',dict(
                stage='replaced_after_complete_95500_validation',checkpoint=checkpoint,
                admission_control=str(self.out),old_supervisor=self.pid,new_supervisor=os.getpid()))
            selected=checkpoint;selected_step=95500;config=self.previous['model_config'];accepted=False
            with ExitStack() as stack:
                while True:
                    devices,apps=gpu_snapshot()
                    if set(self.previous['physical_gpus'])<=free_indices(devices,apps):break
                    self.write('waiting_for_reserved_gpus',available_gpus=sorted(free_indices(devices,apps)),
                               checkpoint=checkpoint,training_paused_for_resources=True)
                    time.sleep(10)
                for i in self.previous['physical_gpus']:
                    lease=try_lease(devices[i].uuid)
                    if lease is None:raise RuntimeError('GPU lease changed')
                    stack.enter_context(lease)
                if ','.join(devices[i].uuid for i in self.previous['physical_gpus'])!=self.previous['environment']['CUDA_VISIBLE_DEVICES']:
                    raise ValueError('reserved GPU UUID ordering changed')
                try:
                    flags=self.experiments(checkpoint)
                    migrated,new_config=self.migrate(checkpoint,flags)
                    run=self.out/'production_preflight'
                    argv=training_argv(dict(self.previous,allow_shared=False),migrated,self.previous['data_config'],run,95505)
                    log=self.command('production_preflight',argv+['--gradient-accumulation','1'])
                    health=continuation_health(log,95500,95505,4)
                    before=validate_checkpoint(migrated,95500,world_size=4,batch_size=1)
                    after=validate_checkpoint(run/'latest.pt',95505,world_size=4,batch_size=1)
                    require_cursor_progress(before,after,5)
                    from scripts.migrate_dynamic_validation import require_fixed_training_state
                    require_fixed_training_state(before,after)
                    if after['load_balancing']!=before['load_balancing']:raise ValueError('balance resume state changed')
                    write_json(self.out/'production_preflight_health.json',health)
                    selected=str(run/'latest.pt');selected_step=95505;config=new_config;accepted=True
                except Exception:
                    self.write('performance_rejected',reason=traceback.format_exc(),fallback_checkpoint=checkpoint)
            self.launch_main(selected,selected_step,baseline,candidate,config,accepted)
        except BaseException:
            if self.suspended and self.same_supervisor():
                os.kill(self.pid,signal.SIGCONT);self.suspended=False
            self.write('failed',error=traceback.format_exc(),original_supervisor_resumed=not self.replaced)
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',required=True)
    args=p.parse_args()
    output=Path(read(args.job)['output']);output.mkdir(parents=True,exist_ok=True)
    def interrupted(signum,frame):
        raise InterruptedError('throughput admission interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    with (output/'admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        Admission(args.job).run()
