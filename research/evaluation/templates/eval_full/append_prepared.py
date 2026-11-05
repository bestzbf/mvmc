"""Append only frozen CPU-preflighted missing-case jobs under the queue writer lock."""
import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path('@@EVAL_FULL_ROOT@@')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)


def priority(job):
    method,dataset=job['method'],job['dataset']
    if method=='MoSE3':return {'drivetrack_minival':40,'pstudio_minival':60,'adt_minival':80}[dataset]
    if method=='Ours134400':return 12
    if dataset=='dexycb':return 5
    if dataset=='mv-kubric':return 10
    return 20


def once():
    index=read(ROOT/'prep_index.json');appended=[]
    with (ROOT/'queue_plan.append.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        plan=read(ROOT/'queue_plan.json');existing={j['id']:j for j in plan['jobs']}
        for ident,entry in index['rows'].items():
            if entry['state']!='complete':continue
            specpath=Path(entry['spec'])
            assert sha(specpath)==entry['spec_sha256'],ident
            spec=read(specpath)
            assert spec['new_manifest_CPU_frozen_loader_preflight']=='PASS'
            assert sha(spec['canonical_manifest'])==spec['canonical_manifest_sha256']
            for candidate in spec['gpu_job_candidates']:
                if candidate['id'] in existing:continue
                job=copy.deepcopy(candidate)
                assert sha(job['manifest'])==job['manifest_sha256']
                assert len(job['case_ids'])==job['case_count'] and len(set(job['case_ids']))==job['case_count']
                job['priority']=priority(job)
                job['prediction_root']=job['argv'][job['argv'].index('--output')+1]
                job['prepared_spec']=str(specpath);job['prepared_spec_sha256']=entry['spec_sha256']
                if job['method']=='MoSE3':
                    # Profiling preserves the adapter and all native flags and is outside the producer namespace.
                    original=job['argv'];observer=ROOT/'scripts/observe_mose.py'
                    native=Path('@@BASELINE_ASSETS_ROOT@@/code/mose3/mose3/models/mose3.py')
                    logfile=ROOT/'observability'/(job['id']+'.jsonl')
                    job['argv']=[original[0],str(observer),'--adapter',original[1],'--native-model',str(native),
                                 '--baseline-root','@@RELEASED_BASELINE_HARNESS_ROOT@@',
                                 '--log',str(logfile),'--']+original[2:]
                    job['observability_only']={'wrapper':str(observer),'sha256':sha(observer),'log':str(logfile),
                                               'original_adapter_argv':original,'tensor_computation_changed':False}
                    job['allow_timeout_retry']=False
                plan['jobs'].append(job);existing[job['id']]=job;appended.append(job['id'])
        if appended:save(ROOT/'queue_plan.json',plan)
    summary={'at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'new_jobs':appended,
             'prepared_rows':index.get('complete_rows',0),'failed_rows':index.get('failed_rows',0),
             'planned_rows':index['planned_rows'],'live_jobs':len(plan['jobs']),
             'global_deadline':None,'plan_sealed':plan['sealed']}
    save(ROOT/'prepared_append_status.json',summary)
    return index,summary


if __name__=='__main__':
    while True:
        try:
            index,summary=once()
            print(json.dumps(summary,ensure_ascii=False),flush=True)
            if index.get('complete_rows',0)+index.get('failed_rows',0)>=index['planned_rows']:break
        except Exception as exc:
            save(ROOT/'prepared_append_error.json',{'at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'error':repr(exc)})
            raise
        time.sleep(15)
