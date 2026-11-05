#!/usr/bin/env python3
"""Prepare fresh missing-case manifests/specs, never run GPU or modify live queue.

Run on @@BENCHMARK_SSH_HOST@@ in eval_env, with CUDA_VISIBLE_DEVICES empty. Inputs are copied
or hardlinked inside each new manifest root. Original plans/caches remain intact.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

BENCH=Path('@@RELEASED_BASELINE_HARNESS_ROOT@@')
ROUTING_WRAPPER_SHA='679ba91943a10714b819ea72f6e6c4a7268ea802ccc4ec37e7f94d44739ccd63'


def read(path):return json.loads(Path(path).read_text())
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if path.exists():
        if path.read_text()!=encoded:raise FileExistsError('Preserve existing prepared file: '+str(path))
        return
    with path.open('x') as stream:stream.write(encoded)


def derive_manifest(original,source_path,ids,input_root,input_cache=None):
    selected=[c for c in original['cases'] if c['id'] in ids]
    if [c['id'] for c in selected]!=ids or len(ids)!=len(set(ids)):raise ValueError('IDs/order must be exact frozen missing selection')
    manifest=copy.deepcopy(original);manifest['cases']=[]
    for c in selected:
        case=copy.deepcopy(c)
        source=Path(c['input']);source=source if source.is_absolute() else source_path.parent/source
        target=input_root/(f'{len(manifest["cases"]):06d}.npz')
        target.parent.mkdir(parents=True,exist_ok=True)
        cached=None
        if input_cache is not None:
            cached=input_cache/c['input_sha256'][:2]/(c['input_sha256']+'.npz')
            cached.parent.mkdir(parents=True,exist_ok=True)
            if not cached.exists():
                temporary=cached.with_name(cached.name+f'.tmp.{os.getpid()}.{uuid.uuid4().hex}')
                try:
                    try:os.link(source,temporary)
                    except OSError:
                        with source.open('rb') as src,temporary.open('xb') as dst:shutil.copyfileobj(src,dst,8*1024*1024)
                    if sha(temporary)!=c['input_sha256']:raise ValueError('Frozen source input changed')
                    try:os.link(temporary,cached)
                    except FileExistsError:pass
                finally:
                    if temporary.exists():temporary.unlink()
            if cached.is_symlink() or sha(cached)!=c['input_sha256']:raise ValueError('Shared input cache changed')
        if not target.exists():
            try:os.link(cached or source,target)
            except OSError:
                with (cached or source).open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,8*1024*1024)
        if target.is_symlink() or sha(target)!=c['input_sha256']:raise ValueError('Copied input escaped/changed')
        case['input']=str(target.relative_to(input_root.parent));manifest['cases'].append(case)
    release=manifest.get('protocol',{}).get('release_binding')
    if release:
        release['status']='missing_case_prediction_batch_not_formal_score'
        release['coverage_verified']=False
    manifest['full_completion_batch']={'source_manifest':str(source_path),'source_manifest_sha256':sha(source_path),
        'selected_ids':ids,'selection':'Only missing canonical cases; no frame/query modification',
        'formal_score_admitted':False}
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--row',required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--batch-size',type=int,default=5)
    p.add_argument('--benchmark-code-root',type=Path,default=BENCH)
    p.add_argument('--routing-wrapper',type=Path)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU preparation requires CUDA_VISIBLE_DEVICES empty')
    if not 1<=a.batch_size<=5:raise ValueError('GPU batch size must be1..5 for fair scheduling')
    inventory=read(a.inventory);row=next(r for r in inventory['rows'] if r['id']==a.row)
    panel=inventory['canonical_panels'][row['panel_id']]
    if not row['missing_ids']:raise ValueError('Complete native cache: no GPU preparation needed')
    original_path=Path(panel['canonical_manifest'])
    if sha(original_path)!=panel['canonical_manifest_sha256']:raise ValueError('Canonical manifest changed')
    original=read(original_path);template=row['source_original'];planpath=Path(template['source_plan'])
    if sha(planpath)!=template['source_plan_sha256']:raise ValueError('Original frozen plan changed')
    sourcejob=next(j for j in read(planpath)['jobs'] if j['id']==template['id'])
    if canonical(sourcejob)!=template['job_sha256']:raise ValueError('Original frozen job changed')
    for artifact in template.get('checkpoint_artifacts') or []:
        path=Path(artifact['path'])
        if path.stat().st_size>16*1024*1024:raise ValueError('Unexpected large checkpoint artifact; no weight rehash in preparation')
        if sha(path)!=artifact['sha256']:raise ValueError('Frozen adapter/source artifact changed')
    group=next(g for g in row['groups'] if g['signature_sha256']==row['preferred_group_signature_sha256'])
    for producer in group['producers']:
        if sha(producer['run_state'])!=producer['run_state_file_sha256']:raise ValueError('Anchor producer run-state changed')
        state=read(producer['run_state'])
        if canonical({k:v for k,v in state.items() if k!='manifest_sha256'})!=group['signature_sha256']:
            raise ValueError('Anchor runtime/source/settings signature changed')
    selected=set(row['missing_ids']);expected={c['id'] for c in panel['cases']} - set(row['existing_ids'])
    if selected!=expected or len(selected)!=row['missing']:raise ValueError('Missing spec does not exactly complement native cache')
    sys.path.insert(0,str(a.benchmark_code_root))
    from multicamera4rc.benchmarks.released_data import load_manifest
    batchsize=1 if row['method']=='MoSE3' else a.batch_size
    batches=[]
    for offset in range(0,len(row['missing_ids']),batchsize):
        ids=row['missing_ids'][offset:offset+batchsize]
        base=a.output_root/'prepared'/a.row/f'{offset//batchsize:03d}'
        manifestpath=base/'manifest.json';output=base/'predictions'
        if output.exists():raise FileExistsError('Prediction root already exists; preserve real prior attempt')
        manifest=derive_manifest(original,original_path,ids,base/'inputs',a.output_root/'input_cache')
        save_new(manifestpath,manifest);load_manifest(manifestpath,inference_only=True)
        argv=list(template['predict_argv']);before=list(argv)
        for flag,value in [('--manifest',str(manifestpath)),('--output',str(output))]:
            if flag not in argv:raise ValueError('Frozen source recipe lacks replaceable '+flag)
            argv[argv.index(flag)+1]=value
        permitted={before.index('--manifest')+1,before.index('--output')+1}
        if any(x!=y for i,(x,y) in enumerate(zip(before,argv)) if i not in permitted):raise AssertionError('Non-I/O model recipe changed')
        native_argv=list(argv)
        routing=None
        if row['method']=='Ours134400' and row['panel_id']=='drivetrack_minival':
            wrapper=a.routing_wrapper or a.output_root/'scripts/route_ours_drive_lock.py'
            if sha(wrapper)!=ROUTING_WRAPPER_SHA:raise ValueError('Independent routing wrapper SHA differs')
            routing={'wrapper':str(wrapper),'wrapper_sha256':ROUTING_WRAPPER_SHA,
                'receipt':str(base/'routing_receipt.json'),'native_status':str(base/'routing_receipt.native_streamed_status.json'),
                'retry_receipts':'Preserve anchor; each further invocation uses routing_receipt.attempt_PID_ns.json and its own status',
                'changes':['parent lock routing accepts exactGPU3/4 UUID + actual inherited FD lock',
                           'only one exact old diagnostic status literal redirected to new job'],
                'native_argv_except_manifest_output_unchanged':True,'physical_gpu_affinity':[3,4]}
            argv=[argv[0],str(wrapper),'--routing-receipt',routing['receipt'],'--',*argv[2:]]
        job_id='full_'+a.row+'_'+f'{offset//batchsize:03d}'+'_predict'
        job={'id':job_id,'kind':'gpu','method':row['method'],'dataset':row['panel_id'],'argv':argv,
             'env':template.get('environment') or {},'case_count':len(ids),'case_ids':ids,'depends_on':[],
             'expected_outputs':[str(output/'predictions.json')],'timeout_seconds':21600 if row['method']=='MoSE3' else 3600,
             'manifest':str(manifestpath),'manifest_sha256':sha(manifestpath)}
        if routing:
            job['routing_provenance']=routing;job['native_predict_argv']=native_argv
        batches.append(job)
    spec={'schema':'full-task-missing-prediction-preparation-v1','row':a.row,'method':row['method'],
          'panel_id':row['panel_id'],'canonical_manifest':str(original_path),'canonical_manifest_sha256':sha(original_path),
          'canonical_full_sample_ids':[c['id'] for c in panel['cases']],
          'source_original':template,'expected_group_signature_sha256':row['preferred_group_signature_sha256'],
          'native_reuse_producers':group['producers'],'missing_ids':row['missing_ids'],'gpu_job_candidates':batches,
          'new_manifest_CPU_frozen_loader_preflight':'PASS','GPU_started':False,'live_queue_modified':False,
          'score_contract':'New full scorer must verify exact canonical full ID coverage and true native sources; no synthetic producer index/run-state.',
          'routing_note':'Only physicalGPU3/4; Ours Drive uses independently pinned lock/status metadata shim, no GPU4 affinity. Root owns live append.'}
    specpath=a.output_root/'specs'/(a.row+'.json');save_new(specpath,spec)
    print(json.dumps({'status':'READY','spec':str(specpath),'GPU_cases':len(row['missing_ids']),'GPU_batches':len(batches),
                      'CPU_loader_preflight':'PASS','GPU_started':False,'live_queue_modified':False}))


if __name__=='__main__':main()
