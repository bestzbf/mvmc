"""CPU-only two-worker preparation coordinator; no inference/live queue writes."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

ROOT=Path('@@EVAL_FULL_ROOT@@')
PYTHON='@@BENCHMARK_ROOT@@/runtime/eval_env/bin/python'
INVENTORY=ROOT/'planning/full_inventory.json'
PREPARE=ROOT/'scripts/prepare_missing.py'
ROUTE=ROOT/'scripts/route_ours_drive_lock.py'
SHA={INVENTORY:'3b2508134d5991af293f296f4fc13002aab4433192a9e3f807eff5908ea759bd',
     PREPARE:'92e184b5dbaf1030ca1473291b6bd334dc1830e64e9820ade5def205759a4baa',
     ROUTE:'679ba91943a10714b819ea72f6e6c4a7268ea802ccc4ec37e7f94d44739ccd63'}


def read(path):return json.loads(Path(path).read_text())
def now():return dt.datetime.now(dt.timezone.utc).isoformat()


def rank(row):
    method,panel=row['method'],row['panel_id']
    if method=='VGGT-Omega-1B-512+TrackEverything' and panel=='dexycb':return (0,panel)
    if method=='ARROW' and panel=='mv-kubric':return (1,panel)
    if method=='Point4D-direct-query':return (2,panel)
    if method in ['UniQuery4R','ReLift+AllTracker+DA3']:return (3,method,panel)
    if method=='ARROW':return (4,panel)
    if method=='VGGT-Omega-1B-512+TrackEverything':return (5,panel)
    if method=='Ours134400':return (6,panel)
    return (7,panel)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU coordinator requires CUDA_VISIBLE_DEVICES empty')
    for path,expected in SHA.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise RuntimeError('Frozen preparation input differs: '+str(path))
    lock=(ROOT/'preparation_logs/prep_driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    inventory=read(INVENTORY)
    rows=[r for r in inventory['rows'] if r['missing'] and r['panel_id']!='panoptic']
    rows.sort(key=rank)
    index_path=ROOT/'prep_index.json'
    index=read(index_path) if index_path.exists() else {'schema':'full-missing-CPU-preparation-index-v1','rows':{}}
    index.update(driver_pid=os.getpid(),started_at_utc=now(),CPU_only=True,max_parallel_CPU_preparations=2,
                 live_queue_modified=False,GPU_started=False,Panoptic_excluded='Both Panoptic jobs are owned by existing bootstrap',
                 inventory_sha256=SHA[INVENTORY],planned_rows=len(rows),planned_prediction_cases=sum(r['missing'] for r in rows))
    mutex=threading.Lock()
    def publish():
        index['updated_at_utc']=now()
        index['complete_rows']=sum(r['state']=='complete' for r in index['rows'].values())
        index['failed_rows']=sum(r['state']=='failed' for r in index['rows'].values())
        temporary=index_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n');os.replace(temporary,index_path)
    with mutex:publish()
    def prepare(row):
        ident=row['id'];log=ROOT/'preparation_logs'/(ident+'.log');spec=ROOT/'specs'/(ident+'.json')
        with mutex:
            old=index['rows'].get(ident)
            if old and old['state']=='complete':
                if not spec.exists():raise RuntimeError('Completed prepared spec disappeared')
                return {'event':'already_prepared','row':ident}
            index['rows'][ident]={'state':'preparing','method':row['method'],'panel_id':row['panel_id'],
                'new_prediction_cases':row['missing'],'started_at_utc':now(),'log':str(log)};publish()
        argv=[PYTHON,str(PREPARE),'--inventory',str(INVENTORY),'--row',ident,'--output-root',str(ROOT),
              '--routing-wrapper',str(ROUTE),'--batch-size','5']
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        started=time.time()
        with log.open('a') as stream:
            stream.write('\nCPU PREPARATION '+now()+'\n');stream.flush()
            result=subprocess.run(argv,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        with mutex:
            record=index['rows'][ident];record.update(finished_at_utc=now(),elapsed_seconds=round(time.time()-started,2),returncode=result.returncode)
            if result.returncode==0 and spec.exists():
                prepared=read(spec);record.update(state='complete',spec=str(spec),spec_sha256=hashlib.sha256(spec.read_bytes()).hexdigest(),
                    candidate_GPU_batches=len(prepared['gpu_job_candidates']),candidate_case_count=sum(j['case_count'] for j in prepared['gpu_job_candidates']))
                assert record['candidate_case_count']==row['missing']
            else:record.update(state='failed',error_tail=log.read_text()[-2500:])
            publish()
            event={'event':record['state'],'row':ident,'cases':row['missing'],'elapsed_seconds':record['elapsed_seconds'],
                'spec':record.get('spec'),'log':str(log)}
        return event
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(prepare,r) for r in rows]
        for future in as_completed(futures):
            try:print(json.dumps(future.result(),ensure_ascii=False),flush=True)
            except BaseException as error:print(json.dumps({'event':'coordinator_error','error':repr(error)}),flush=True)
    with mutex:index['finished_at_utc']=now();publish()
    print(json.dumps({'event':'all_preparation_finished','complete_rows':index['complete_rows'],'failed_rows':index['failed_rows']}),flush=True)


if __name__=='__main__':main()
