"""Append one allocator-only recovery per OOM-missing Ours Drive case.

This CPU metadata watcher never launches inference, changes original jobs, or
signals workers. Only the existing full runner dispatches appended GPU jobs.
"""
import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

RUNNER_SHA = 'c791bd657c7fc6db09bb0035a615bc2fedfc6c3be16eb0b75eb8e3b29f31addb'
PREP_SHA = '92e184b5dbaf1030ca1473291b6bd334dc1830e64e9820ade5def205759a4baa'
ROUTE_SHA = '679ba91943a10714b819ea72f6e6c4a7268ea802ccc4ec37e7f94d44739ccd63'
SOURCE_SHA = '345d50a1e0300f54a5e55ae7a82a572ff8d980ac1c72d718fb09456c3d5844bc'
METHOD, PANEL, ROW = 'Ours134400', 'drivetrack_minival', 'Ours134400__drivetrack_minival'
ORIGINAL_IDS = ['full_'+ROW+'_'+f'{i:03d}'+'_predict' for i in range(6)]
ALLOCATOR = {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
HOLD = 'ours_drive_allocator_oom_recovery_v1'
BENCH = '@@RELEASED_BASELINE_HARNESS_ROOT@@'
TERMINAL = {'complete', 'failed', 'blocked', 'interrupted'}


def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
def canonical(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('w') as stream:
        json.dump(obj,stream,ensure_ascii=False,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)
def save_new(path,obj):
    if Path(path).exists():
        if read(path)!=obj:raise ValueError('Preserve immutable recovery artifact: '+str(path))
    else:save(path,obj)
def load_frozen(path,expected,name):
    if sha(path)!=expected:raise ValueError('Frozen dependency changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class Watcher:
    def __init__(self,root,proof,manifest_validator=None,score_builder=None,score_builder_sha=None):
        self.root=Path(root).resolve();self.registry_path=self.root/'ours_drive_oom_recovery_registry.json'
        self.framework=load_frozen(self.root/'full_runner.py',RUNNER_SHA,'_oom_frozen_full_runner')
        self.prep=load_frozen(self.root/'scripts/prepare_missing.py',PREP_SHA,'_oom_frozen_prepare')
        self.route=self.root/'scripts/route_ours_drive_lock.py'
        if sha(self.route)!=ROUTE_SHA:raise ValueError('Routing source changed')
        self.proof=read(proof);self.proof_sha=sha(proof)
        if (self.proof.get('allocator_settings')!='expandable_segments:True' or self.proof.get('parse_supported') is not True or
            self.proof.get('cuda_initialized_before') is not False or self.proof.get('cuda_initialized_after') is not False or
            self.proof.get('CUDA_VISIBLE_DEVICES')!=''):
            raise ValueError('Need a successful CPU-only actual-runtime allocator preflight')
        if self.proof.get('source_wrapper_sha256')!=SOURCE_SHA:raise ValueError('Allocator proof did not pin the frozen native wrapper')
        self.validator=manifest_validator or self.validate_manifest
        self.score_builder_path=Path(score_builder) if score_builder else None;self.score_builder_sha=score_builder_sha
        if bool(self.score_builder_path)!=bool(score_builder_sha):raise ValueError('Score builder path and hash must be supplied together')
        self.builder=load_frozen(self.score_builder_path,score_builder_sha,'_ours_drive_recovery_score_v1') if self.score_builder_path else None
        self.inventory=read(self.root/'planning/full_inventory.json')
        self.row=next(r for r in self.inventory['rows'] if r['id']==ROW)
        self.panel=self.inventory['canonical_panels'][PANEL]
        self.canonical_path=Path(self.panel['canonical_manifest'])
        if sha(self.canonical_path)!=self.panel['canonical_manifest_sha256']:raise ValueError('Canonical inputs changed')
        self.manifest=read(self.canonical_path);self.all_ids=[c['id'] for c in self.manifest['cases']]
        self.manifest_cases={c['id']:c for c in self.manifest['cases']}
        if self.registry_path.exists():self.registry=read(self.registry_path)
        else:self.registry={'schema':'ours-drive-oom-recovery-v1','created_at_utc':utc(),'allocator':ALLOCATOR,
            'allocator_preflight_sha256':self.proof_sha,'original_jobs':{},'recoveries':{},'issues':[],'score_job':None}
        if self.registry['allocator_preflight_sha256']!=self.proof_sha:raise ValueError('Allocator proof changed across watcher restarts')
        if self.builder:
            contract={'path':str(self.score_builder_path),'sha256':score_builder_sha}
            if self.registry.get('score_builder') and self.registry['score_builder']!=contract:raise ValueError('Recovery score builder contract changed')
            self.registry['score_builder']=contract

    def validate_manifest(self,path):
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Manifest preflight is CPU only')
        if BENCH not in sys.path:sys.path.insert(0,BENCH)
        from multicamera4rc.benchmarks.released_data import load_manifest
        load_manifest(path,inference_only=True)

    def originals(self,plan):
        jobs={j['id']:j for j in plan['jobs']}
        result=[]
        for key in ORIGINAL_IDS:
            job=jobs[key]
            if (job['kind']!='gpu' or job['method']!=METHOD or job['dataset']!=PANEL or
                '--target-time-chunk' not in job['argv'] or job['argv'][job['argv'].index('--target-time-chunk')+1]!='8' or
                job['argv'][job['argv'].index('--amp')+1]!='bf16' or job.get('env')!={}):
                raise ValueError('Unexpected original Ours inference recipe')
            old=self.registry['original_jobs'].get(key)
            if old and old!=digest(job):raise ValueError('An original frozen job changed')
            self.registry['original_jobs'][key]=digest(job);result.append(job)
        return result

    def valid_snapshot(self,job):
        snap=self.framework.receipt_snapshot(job)
        if snap['invalid_receipts']:raise ValueError('Invalid original/recovery receipts: '+job['id'])
        directory=Path(job['expected_outputs'][0]).parent
        state_path=directory/'run-state.json'
        if state_path.exists():
            state=read(state_path)
            if canonical({k:v for k,v in state.items() if k!='manifest_sha256'})!=self.row['preferred_group_signature_sha256']:
                raise ValueError('Native source/runtime/model signature differs: '+job['id'])
            if state['runtime_versions']['torch']!=self.proof['torch_version']:raise ValueError('Allocator preflight used another PyTorch')
        for path in directory.glob('*.receipt.json'):
            record=read(path)['record']
            if record['id'] in snap['ids'] and sha(directory/record['file'])!=record['sha256']:
                raise ValueError('Committed native payload hash differs')
        return snap

    def failure(self,job,state):
        if state.get('state')!='failed':return None
        if state.get('job_hash')!=digest(job):raise ValueError('Failed status not tied to original job')
        log=Path(state['log'])
        if not log.resolve().is_relative_to(self.root) or log.is_symlink():raise ValueError('Untrusted failure log')
        text=log.read_text(errors='replace')
        if 'cuda out of memory' not in text.lower():return None
        snap=self.valid_snapshot(job)
        missing=[i for i in job['case_ids'] if i not in set(snap['ids'])]
        if not missing:return None
        return {'job_sha256':digest(job),'failure_status_sha256':sha(self.root/'status'/(job['id']+'.json')),
                'failure_log':str(log),'failure_log_sha256':sha(log),'failure_finished_at':state.get('finished_at_utc'),
                'valid_original_case_ids':snap['ids'],'missing_case_ids':missing}

    def make_case(self,original,case_id,info):
        case_index=original['case_ids'].index(case_id);batch=ORIGINAL_IDS.index(original['id'])
        base=self.root/'oom_recovery_v1'/f'batch_{batch:03d}'/f'case_{case_index:03d}'
        output=base/'predictions';manifest_path=base/'manifest.json';receipt=base/'routing_receipt.json'
        key=f'recover_ours_drive_oom_v1_{batch:03d}_{case_index:03d}_predict'
        manifest=self.prep.derive_manifest(self.manifest,self.canonical_path,[case_id],base/'inputs',self.root/'input_cache')
        canonical_case=self.manifest_cases[case_id];new_case=manifest['cases'][0]
        if {k:v for k,v in new_case.items() if k!='input'}!={k:v for k,v in canonical_case.items() if k!='input'}:
            raise ValueError('Recovery changed frames/queries/case metadata')
        save_new(manifest_path,manifest);self.validator(manifest_path)
        argv=list(original['argv'])
        for flag,value in [('--routing-receipt',str(receipt)),('--manifest',str(manifest_path)),('--output',str(output))]:
            argv[argv.index(flag)+1]=value
        permitted={original['argv'].index(k)+1 for k in ['--routing-receipt','--manifest','--output']}
        if any(a!=b for i,(a,b) in enumerate(zip(original['argv'],argv)) if i not in permitted):raise ValueError('Non-I/O inference argument changed')
        job={'id':key,'kind':'gpu','method':METHOD,'dataset':PANEL,'argv':argv,'env':dict(ALLOCATOR),
             'case_count':1,'case_ids':[case_id],'depends_on':[], 'expected_outputs':[str(output/'predictions.json')],
             'timeout_seconds':3600,'allow_timeout_retry':False,'manifest':str(manifest_path),'manifest_sha256':sha(manifest_path),
             'prediction_root':str(output),'priority':11,'physical_gpu_affinity':[3,4],
             'oom_recovery_of':original['id'],
             'allocator_only_recovery':{'original_job_id':original['id'],'original_job_sha256':digest(original),
                 'failure_status_sha256':info['failure_status_sha256'],'failure_log_sha256':info['failure_log_sha256'],
                 'allocator_preflight_sha256':self.proof_sha,'allocator':dict(ALLOCATOR),'once_per_case':True,
                 'model_query_frame_chunk_amp_unchanged':True,'expected_group_signature_sha256':self.row['preferred_group_signature_sha256']}}
        native=list(original['native_predict_argv'])
        for flag,value in [('--manifest',str(manifest_path)),('--output',str(output))]:native[native.index(flag)+1]=value
        job['native_predict_argv']=native
        save_new(base/'job.json',job)
        return job

    def apply(self,plan_path,jobs):
        if not jobs:return
        with (self.root/'queue_plan.append.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            plan=read(plan_path);self.originals(plan);known={j['id']:j for j in plan['jobs']}
            for job in jobs:
                original=known[job['oom_recovery_of']]
                if job.get('env')!=ALLOCATOR or len(job['argv'])!=len(original['argv']):raise ValueError('Recovery recipe/env changed before activation')
                allowed={original['argv'].index(k)+1 for k in ['--routing-receipt','--manifest','--output']}
                if any(a!=b for i,(a,b) in enumerate(zip(original['argv'],job['argv'])) if i not in allowed):raise ValueError('Recovery changed non-IO argv')
                if read(Path(job['manifest']).with_name('job.json'))!=job or sha(job['manifest'])!=job['manifest_sha256']:
                    raise ValueError('Prepared recovery job/manifest changed')
                manifest=read(job['manifest'])
                if len(manifest['cases'])!=1 or [c['id'] for c in manifest['cases']]!=job['case_ids']:raise ValueError('Recovery case scope changed')
                for case in manifest['cases']:
                    expected=self.manifest_cases[case['id']]
                    if {k:v for k,v in case.items() if k!='input'}!={k:v for k,v in expected.items() if k!='input'}:raise ValueError('Recovery modified canonical case')
                    payload=Path(job['manifest']).parent/case['input']
                    if payload.is_symlink() or not payload.resolve().is_relative_to(Path(job['manifest']).parent) or sha(payload)!=case['input_sha256']:
                        raise ValueError('Recovery input bytes/containment changed')
                if job['id'] in known:
                    if digest(known[job['id']])!=digest(job):raise ValueError('Existing recovery job changed')
                else:plan['jobs'].append(job);known[job['id']]=job
            holds=plan.setdefault('completion_holds',{})
            if HOLD not in holds:
                self.registry.setdefault('plan_was_sealed_before_recovery',plan.get('sealed',False))
            holds[HOLD]={'state':'waiting_for_recovery_sources','registered_at_utc':utc()}
            plan['sealed']=False
            save(plan_path,plan)

    def close_hold(self,reason):
        path=self.root/'queue_plan.json'
        with (self.root/'queue_plan.append.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);plan=read(path)
            holds=plan.setdefault('completion_holds',{});holds.pop(HOLD,None)
            if not holds and self.registry.get('plan_was_sealed_before_recovery') is True:plan['sealed']=True
            plan.setdefault('completed_extensions',{})[HOLD]={'state':reason,'updated_at_utc':utc()}
            save(path,plan)

    def source_pending(self,job):
        path=self.root/'status'/(job['id']+'.json')
        if not path.exists():return True
        state=read(path)
        if state.get('job_hash')!=digest(job):raise ValueError('Producer status hash changed: '+job['id'])
        if state.get('state') not in TERMINAL:return True
        pid=state.get('pid');birth=state.get('pid_starttime_ticks')
        if pid is None or birth is None:return False
        identity=self.framework.process_identity(pid)
        if identity and identity['starttime_ticks']!=birth:
            return False  # Reused PID belongs to someone else; never signal it.
        if identity and identity['state'] not in ('Z','X'):return True
        members=self.framework.owned_group_members({'process':SimpleNamespace(pid=pid),'state':state})
        return any(member['state'] not in ('Z','X') for member in members)

    def apply_score(self,intent):
        """Idempotent append + CAS target redirect; original failed jobs stay intact."""
        job=intent['job'];target=intent['target_update'];cas=intent['target_CAS']
        with (self.root/'queue_plan.append.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            plan=read(self.root/'queue_plan.json');self.originals(plan)
            producers=self.originals(plan)+[j for value in self.registry['recoveries'].values() for j in value['jobs']]
            if any(self.source_pending(j) for j in producers):raise ValueError('Source became active before score publication')
            existing={j['id']:j for j in plan['jobs']}
            for recovery in self.registry['recoveries'].values():
                for producer in recovery['jobs']:
                    if producer['id'] not in existing or digest(existing[producer['id']])!=digest(producer):
                        raise ValueError('Registered recovery changed before score publication')
            if job['id'] in existing:
                if digest(existing[job['id']])!=digest(job):raise ValueError('Recovery CPU job changed')
            else:plan['jobs'].append(job)
            targets=read(self.root/'full_targets.json');rows=targets['rows']
            index=next(i for i,row in enumerate(rows) if row['row_id']==ROW)
            if rows[index]!=target:
                if (self.builder.canonical(rows[index])!=cas['expected_old_row_canonical_sha256'] or
                    rows[index]['score_path']!=cas['expected_old_score_path'] or
                    rows[index]['score_job_id']!=cas['expected_old_score_job_id']):
                    raise ValueError('Full target CAS failed; do not overwrite another generation')
                rows[index]=target
            save(self.root/'queue_plan.json',plan)
            save(self.root/'full_targets.json',targets)
        self.registry['score_job']=job['id'];self.registry['score_phase']='registered';self.registry['score_published']=True
        save(self.registry_path,self.registry)
        self.close_hold('new_v2_recovery_score_registered')

    def score_step(self,originals,activate):
        if not activate or not self.builder or not self.registry['recoveries']:
            return {'phase':'pending_integration' if not self.builder else 'not_activated','final':False}
        if self.registry.get('score_intent'):
            producers=originals+[j for value in self.registry['recoveries'].values() for j in value['jobs']]
            if any(self.source_pending(j) for j in producers):return {'phase':'waiting_for_native_producers','final':False}
            if not self.registry.get('score_published'):self.apply_score(self.registry['score_intent'])
            elif HOLD in read(self.root/'queue_plan.json').get('completion_holds',{}):self.close_hold('new_v2_recovery_score_registered')
            key=self.registry['score_job'];path=self.root/'status'/(key+'.json')
            state=read(path) if path.exists() else {}
            final=state.get('state') in TERMINAL
            phase='complete' if state.get('state')=='complete' else ('needs_attention' if final else 'score_registered')
            self.registry['score_phase']=phase
            return {'phase':phase,'score_job_id':key,'score_state':state.get('state','pending'),'final':final}
        recoveries=[j for r in self.registry['recoveries'].values() for j in r['jobs']]
        pending=[]
        for job in [*originals,*recoveries]:
            if self.source_pending(job):pending.append(job['id'])
        if pending:return {'phase':'waiting_for_native_producers','remaining_native_jobs':pending,'final':False}
        try:bundle=self.builder.build(self.root,originals,recoveries,write=True)
        except self.builder.NotReady as error:
            # Every candidate is closed, so there is no authorized second OOM retry.
            self.registry['score_phase']='needs_attention_incomplete_native_coverage'
            self.registry['issues'].append({'at_utc':utc(),'error':str(error)})
            self.close_hold('needs_attention_incomplete_native_coverage')
            return {'phase':self.registry['score_phase'],'reason':str(error),'final':True}
        except (ValueError,OSError,KeyError) as error:
            self.registry['score_phase']='needs_attention_invalid_native_sources'
            self.registry['issues'].append({'at_utc':utc(),'error':repr(error)})
            self.close_hold('needs_attention_invalid_native_sources')
            return {'phase':self.registry['score_phase'],'reason':repr(error),'final':True}
        self.registry['score_intent']={k:bundle[k] for k in ['job','target_update','target_CAS']}
        self.registry['score_source_partition_case_count']=bundle['source_partition_case_count']
        save(self.registry_path,self.registry)
        self.apply_score(self.registry['score_intent'])
        return {'phase':'score_registered','score_job_id':self.registry['score_job'],'final':False}

    def step(self,activate=False):
        path=self.root/'queue_plan.json';plan=read(path);originals=self.originals(plan);new_jobs=[]
        existing={j['id']:j for j in plan['jobs']}
        for recovery in self.registry['recoveries'].values():
            if recovery['state']=='registered':
                for job in recovery['jobs']:
                    if job['id'] not in existing or digest(existing[job['id']])!=digest(job):
                        raise ValueError('A registered recovery disappeared or changed')
        for job in originals:
            status=self.root/'status'/(job['id']+'.json')
            state=read(status) if status.exists() else {}
            if job['id'] in self.registry['recoveries']:continue
            info=self.failure(job,state)
            if info is None:continue
            candidates=[self.make_case(job,case,info) for case in info['missing_case_ids']]
            recovery=dict(info,jobs=candidates,registered_at_utc=utc(),state='prepared')
            self.registry['recoveries'][job['id']]=recovery;new_jobs.extend(candidates)
        # Crash-safe: prepared artifacts are immutable; queue append is idempotent.
        for info in self.registry['recoveries'].values():
            if info['state']=='prepared':
                for job in info['jobs']:
                    if job not in new_jobs:new_jobs.append(job)
        if activate and new_jobs:
            self.apply(path,new_jobs)
            for info in self.registry['recoveries'].values():info['state']='registered'
        score=self.score_step(originals,activate)
        self.registry.update(updated_at_utc=utc(),mode='active' if activate else 'prepare_only',
                             recovery_gpu_jobs=sum(len(i['jobs']) for i in self.registry['recoveries'].values()))
        save(self.registry_path,self.registry)
        return {'prepared_or_registered_jobs':len(new_jobs),'recovery_gpu_jobs':self.registry['recovery_gpu_jobs'],
                'original_batches_with_confirmed_OOM':len(self.registry['recoveries']),
                'score_stage':score,'final':score['final']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--allocator-proof',type=Path,required=True)
    p.add_argument('--activate',action='store_true');p.add_argument('--once',action='store_true')
    p.add_argument('--score-builder',type=Path);p.add_argument('--score-builder-sha')
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Watcher is CPU metadata only')
    if a.activate and not a.score_builder:raise ValueError('Active recovery requires the pinned v2 score builder')
    singleton=(a.root/'ours_drive_oom_watch.lock').open('a+');fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)
    watcher=Watcher(a.root,a.allocator_proof,score_builder=a.score_builder,score_builder_sha=a.score_builder_sha)
    while True:
        try:
            summary=watcher.step(a.activate);save(a.root/'ours_drive_oom_watch_status.json',dict(summary,pid=os.getpid(),updated_at_utc=utc()))
            if a.once:print(json.dumps(summary));break
            if summary['final']:break
        except Exception as exc:
            save(a.root/'ours_drive_oom_watch_error.json',{'at_utc':utc(),'error':repr(exc)})
            if a.once:raise
        time.sleep(30)


if __name__=='__main__':main()
