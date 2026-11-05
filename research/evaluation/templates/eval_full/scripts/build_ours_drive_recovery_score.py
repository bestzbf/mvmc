#!/usr/bin/env python3
"""Metadata-only preparation of a new OursDrive full50 score after OOM recovery.

Never appends a queue, edits a target, imports a metric, or opens an NPZ payload.
The sole queue writer must hold its append lock and independently prove selected
producer output roots quiescent. Original/recovery job status must be terminal.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import time

ORIGINAL_SCORE_ID = 'score_full_Ours134400__drivetrack_minival'
NEW_SCORE_ID = ORIGINAL_SCORE_ID+'_oom_recovery_v1'
ROW_ID = 'Ours134400__drivetrack_minival'
TERMINAL = {'complete','failed','interrupted','blocked'}
ALLOCATOR_KEYS = {'PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF'}
BASE_RECIPE_SHA256 = '1cb3aff36ae38f6189d0062e3ef185c749a7e50467a9c09e811ab7fde9ad50e2'
VERIFIER_SHA256 = '9e513498d6da7d9b189b59dc64189c7f1a5ed59beca1827e4bb80a926b0fef9c'
ROUTER_SHA256 = '679ba91943a10714b819ea72f6e6c4a7268ea802ccc4ec37e7f94d44739ccd63'


class NotReady(ValueError):
    """A real producer is still live or one of the canonical cases is missing."""


def require(condition,message):
    if not condition: raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+x)))


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):value.update(block)
    return value.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def job_hash(job):return hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()
def encoded(value):return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def byte_hash(value):return hashlib.sha256(encoded(value)).hexdigest()


def arg(job,flag):
    require(job['argv'].count(flag)==1,'Missing/ambiguous flag '+flag)
    return job['argv'][job['argv'].index(flag)+1]


def status(root,job):
    p=root/'status'/(job['id']+'.json')
    if not p.is_file():raise NotReady('No terminal producer status: '+job['id'])
    result=read(p)
    require(result.get('job_hash')==job_hash(job),'Immutable producer job/status hash differs')
    if result.get('state') not in TERMINAL:raise NotReady('Producer not terminal: '+job['id'])
    return result


def same_case(case,original):
    normalized=copy.deepcopy(case);normalized['input']=original['input']
    require(normalized==original,'Case frames/query/input/source identity changed')


def validate_recovery(original,recovery):
    before,after=original['argv'],recovery['argv']
    require(len(before)==len(after),'Recovery native argv length changed')
    allowed={}
    for flag in ('--manifest','--output','--routing-receipt'):
        require(before.count(flag)==after.count(flag)==1,'Missing/ambiguous recovery IO flag')
        i=before.index(flag);require(after.index(flag)==i,'Recovery argv flag order changed');allowed[i+1]=True
    require(all(a==b for i,(a,b) in enumerate(zip(before,after)) if i not in allowed),'Non-IO native model argv changed')
    require(arg(recovery,'--target-time-chunk')=='8' and arg(recovery,'--expected-step')=='134400' and '--require-refinement' in after and '--resume' in after,'Frozen Ours134400/chunk8/refinement contract changed')
    old,new=original.get('env',{}),recovery.get('env',{})
    require({k:v for k,v in old.items() if k not in ALLOCATOR_KEYS}=={k:v for k,v in new.items() if k not in ALLOCATOR_KEYS},'Recovery changed non-allocator environment')
    additions={k:v for k,v in new.items() if k in ALLOCATOR_KEYS}
    require(additions and all(v=='expandable_segments:True' for v in additions.values()),'Unapproved allocator recovery settings')
    require(Path(arg(original,'--output')).resolve()!=Path(arg(recovery,'--output')).resolve(),'Recovery must preserve original output root')
    require(Path(arg(original,'--routing-receipt')).resolve()!=Path(arg(recovery,'--routing-receipt')).resolve(),'Recovery must use a new routing receipt anchor')
    affinity=recovery.get('physical_gpu_affinity',[3,4]);affinity=[affinity] if isinstance(affinity,int) else affinity
    require(affinity and set(affinity).issubset({3,4}),'Recovery GPU affinity is outside3/4')


def committed_source(manifest_path,prediction_root,recipe,spec,canonical_cases):
    manifest_path=Path(manifest_path).resolve(strict=True);prediction_root=Path(prediction_root).resolve()
    manifest=read(manifest_path);manifest_sha=sha(manifest_path)
    original=read(recipe['source_manifest'])
    for key in ('schema','benchmark','protocol','official_sources','input_condition','coordinates'):
        require(manifest.get(key)==original.get(key),'Producer manifest convention differs: '+key)
    state_path=prediction_root/'run-state.json'
    if not state_path.is_file():return None,[]
    require(not state_path.is_symlink(),'Symlink producer state forbidden')
    state=read(state_path);state_sha=canonical(state)
    require(state.get('manifest_sha256')==manifest_sha and state.get('checkpoint')==spec['expected_checkpoint'],'Producer manifest/checkpoint differs')
    require(canonical({k:v for k,v in state.items() if k!='manifest_sha256'})==recipe['expected_group_signature_sha256'],'Producer runtime/source/numerics signature differs; never normalize it')
    require(state.get('method')==recipe['expected_producer_method'],'Producer method differs')
    require(state.get('target_time_chunk')==8,'Native producer is not the frozen chunk8 pipeline')
    ids=[];records=[]
    for i,case in enumerate(manifest['cases']):
        require(case['id'] in canonical_cases,'Producer case outside canonicalDrive50');same_case(case,canonical_cases[case['id']])
        receipt_path=prediction_root/(f'{i:06d}.receipt.json')
        if not receipt_path.is_file():continue
        require(not receipt_path.is_symlink(),'Symlink receipt forbidden')
        receipt=read(receipt_path);record=receipt['record'];payload=prediction_root/record['file']
        require(receipt.get('schema')=='released-prediction-case-v1' and receipt.get('case_sha256')==canonical(case) and receipt.get('run_state_sha256')==state_sha,'Receipt native state/case binding differs')
        require(record['id']==case['id'] and record['input_sha256']==case['input_sha256'],'Receipt record/input differs')
        checksum=record['sha256'];require(isinstance(checksum,str) and len(checksum)==64 and all(c in '0123456789abcdef' for c in checksum),'Invalid saved prediction checksum')
        require(payload.resolve().is_relative_to(prediction_root.resolve()) and not payload.is_symlink() and payload.is_file() and payload.stat().st_size>0,'Committed prediction payload missing/escaped')
        arrays=receipt.get('arrays',{});require(arrays.get('tracks_XYZ',{}).get('shape')==[case['frames'],case['queries'],3] and arrays.get('visibility',{}).get('shape')==[case['frames'],case['queries']] and arrays['tracks_XYZ'].get('dtype')=='<f4' and arrays['visibility'].get('dtype')=='|b1','Saved native output dimensions/dtypes differ')
        ids.append(case['id']);records.append(record)
    index=prediction_root/'predictions.json'
    output={'kind':'committed_case_receipts','path':str(prediction_root),'manifest':str(manifest_path),'manifest_sha256':manifest_sha,'run_state_sha256':sha(state_path),'case_ids':ids}
    if index.is_file():
        require(not index.is_symlink(),'Symlink final index forbidden');report=read(index)
        require(report.get('schema')=='released-tracking-predictions-v1' and report.get('manifest_sha256')==manifest_sha and report.get('run_state_sha256')==state_sha and report.get('checkpoint')==spec['expected_checkpoint'],'Final native index binding differs')
        require([r['id'] for r in report['cases']]==[c['id'] for c in manifest['cases']] and report['cases']==records,'Final index is not exactly complete genuine receipts')
        output.update(kind='prediction_index',path=str(index),index_sha256=sha(index))
    return output,ids


def immutable_write(path,value):
    path=Path(path);data=encoded(value);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        require(not path.is_symlink() and path.read_bytes()==data,'New immutable recovery artifact already differs: '+str(path));return
    temp=path.with_name(path.name+f'.tmp.{os.getpid()}.{time.time_ns()}')
    with temp.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    try:os.link(temp,path)
    finally:temp.unlink(missing_ok=True)
    descriptor=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def build(root,original_jobs,recovery_jobs,write=True):
    root=Path(root).resolve(strict=True);plan=read(root/'queue_plan.json')
    require(len({j['id'] for j in recovery_jobs})==len(recovery_jobs),'Duplicate recovery job in score inputs')
    old_cpu=next(j for j in plan['jobs'] if j['id']==ORIGINAL_SCORE_ID)
    require(set(j['id'] for j in original_jobs)==set(old_cpu['depends_on']) and len(original_jobs)==6,'Original six frozen batch jobs differ')
    plan_jobs={j['id']:j for j in plan['jobs']}
    for job in [*original_jobs,*recovery_jobs]:require(plan_jobs.get(job['id'])==job,'Producer is not the current immutable registered job')
    recipe_path=Path(arg(old_cpu,'--recipe'));require(sha(recipe_path)==BASE_RECIPE_SHA256,'Frozen original OursDrive recipe differs')
    recipe=read(recipe_path);spec_path=Path(recipe['consumer_config']['path']);spec=read(spec_path)
    require(sha(spec_path)==recipe['consumer_config']['sha256'],'Original score config hash differs')
    require(recipe['schema']=='frozen-full-score-recipe-v2' and recipe['consumer']['kind']=='tap' and recipe['method']=='Ours134400' and recipe['dataset']=='drivetrack_minival' and recipe['expected_cases']==50,'Unexpected original fullscore recipe')
    require(sha(recipe['source_manifest'])==recipe['source_manifest_sha256'] and sha(recipe['consumer']['path'])==recipe['consumer']['sha256'],'Pinned original source/consumer changed')
    require(sha(old_cpu['argv'][1])==VERIFIER_SHA256,'Fullscore verifier differs from the reviewed frozen v2')
    require(canonical(spec['expected_checkpoint'])==recipe['expected_checkpoint_canonical_sha256'],'Expected checkpoint canonical hash differs')
    for producer_job in [*original_jobs,*recovery_jobs]:require(sha(producer_job['argv'][1])==ROUTER_SHA256,'Native lock-routing wrapper source differs')
    original=read(recipe['source_manifest']);expected=[c['id'] for c in original['cases']];require(expected==recipe['expected_case_ids'] and len(expected)==50,'CanonicalDrive50 IDs/order differs')
    cases={c['id']:c for c in original['cases']};by_id={j['id']:j for j in original_jobs};states={j['id']:status(root,j) for j in [*original_jobs,*recovery_jobs]}
    roots={str(Path(arg(j,'--output')).resolve()):j for j in original_jobs};sources=[];claimed=set();provenance=[]
    for old in spec['sources']:
        native=Path(old['path']);native=native.parent if old['kind']=='prediction_index' else native
        require(sha(old['manifest'])==old['manifest_sha256'],'Frozen original producer manifest bytes changed')
        source,ids=committed_source(old['manifest'],native,recipe,spec,cases)
        if source is not None:
            allowed=set(old.get('case_ids',expected));ids=[ident for ident in ids if ident in allowed];source['case_ids']=ids
        if not ids:continue
        require(not claimed.intersection(ids),'Original native source overlap');claimed.update(ids);sources.append(source)
        job=roots.get(str(native.resolve()));provenance.append({'source_root':str(native),'source_kind':'original_committed_cache','producer_job_id':job['id'] if job else None,'selected_case_ids':ids})
    seen_recovery_cases=set()
    for job in recovery_jobs:
        parent=job.get('oom_recovery_of');require(parent in by_id,'Missing/unknown oom_recovery_of original job')
        require(states[parent]['state'] in ('failed','interrupted'),'OOM recovery origin is not a failed original batch')
        validate_recovery(by_id[parent],job)
        parent_cases={c['id']:c for c in read(arg(by_id[parent],'--manifest'))['cases']};manifest=read(arg(job,'--manifest'))
        require(len(manifest['cases'])==1 and job.get('case_count')==1,'Recovery must be one original complete case')
        require(job.get('case_ids',[c['id'] for c in manifest['cases']])==[c['id'] for c in manifest['cases']],'Recovery declared case IDs differ from its actual manifest')
        if job.get('manifest_sha256'):require(job['manifest_sha256']==sha(arg(job,'--manifest')),'Registered recovery manifest bytes changed')
        for case in manifest['cases']:
            require(case['id'] in parent_cases,'Recovery case outside its original batch');same_case(case,parent_cases[case['id']])
            require(case['id'] not in seen_recovery_cases,'More than one recovery job for a case');seen_recovery_cases.add(case['id'])
        source,ids=committed_source(arg(job,'--manifest'),arg(job,'--output'),recipe,spec,cases)
        require(not claimed.intersection(ids),'Recovery duplicates an already committed original case')
        if ids:
            claimed.update(ids);sources.append(source);provenance.append({'source_root':arg(job,'--output'),'source_kind':'once_OOM_allocator_recovery','producer_job_id':job['id'],'producer_job_hash':job_hash(job),'original_job_id':parent,'original_status_sha256':sha(root/'status'/(parent+'.json')),'recovery_status_sha256':sha(root/'status'/(job['id']+'.json')),'allocator_env':{k:v for k,v in job.get('env',{}).items() if k in ALLOCATOR_KEYS},'selected_case_ids':ids})
    if claimed!=set(expected):raise NotReady('CanonicalDrive50 committed coverage incomplete; missing '+str([i for i in expected if i not in claimed]))
    folder=root/'recovery_ours_drive_oom_v1'/'final_score';new_spec_path=folder/'spec.json';new_recipe_path=folder/'recipe.json';raw=root/'scores'/'raw'/(ROW_ID+'_oom_recovery_v1.json');verified=root/'scores'/('full_verified_'+ROW_ID+'_oom_recovery_v1.json')
    require(not raw.exists() and not verified.exists(),'Preserve existing recovery metric outputs; reconcile instead of rerun')
    new_spec=copy.deepcopy(spec);new_spec['sources']=sources;new_spec['recovery_provenance']={'original_recipe':str(recipe_path),'original_recipe_sha256':sha(recipe_path),'original_config_sha256':sha(spec_path),'no_native_state_synthesized':True,'payload_hash_verification_required_in_final_consumer':True,'payload_hashes_checked_by_builder':False,'sources':provenance}
    new_recipe=copy.deepcopy(recipe);new_recipe.update(consumer_config={'path':str(new_spec_path),'sha256':byte_hash(new_spec)},raw_output=str(raw),raw_output_sha256=None,score_argv=[recipe['score_argv'][0],recipe['consumer']['path'],'--spec',str(new_spec_path),'--output',str(raw)],recovery_provenance=new_spec['recovery_provenance'])
    job=copy.deepcopy(old_cpu);job.update(id=NEW_SCORE_ID,argv=[old_cpu['argv'][0],old_cpu['argv'][1],'--recipe',str(new_recipe_path),'--output',str(verified)],depends_on=[],expected_outputs=[str(verified)],score_path=str(verified),recipe_sha256=byte_hash(new_recipe),recovery_of_score_job=ORIGINAL_SCORE_ID)
    target=next(t for t in read(root/'full_targets.json')['rows'] if t['row_id']==ROW_ID)
    require(target['score_job_id']==ORIGINAL_SCORE_ID and target['recipe']==str(recipe_path),'Full target was already redirected; reconcile existing generation')
    updated=copy.deepcopy(target);updated.update(score_path=str(verified),recipe=str(new_recipe_path),score_job_id=NEW_SCORE_ID,reused_full_score=False)
    if 'recipe_sha256' in updated:updated['recipe_sha256']=byte_hash(new_recipe)
    bundle={'schema':'ours-drive-OOM-fullscore-bundle-v1','job':job,'target_update':updated,'target_CAS':{'row_id':ROW_ID,'expected_old_row_canonical_sha256':canonical(target),'expected_old_score_path':target['score_path'],'expected_old_score_job_id':ORIGINAL_SCORE_ID},'spec':new_spec,'recipe':new_recipe,'source_partition_case_count':sum(len(s['case_ids']) for s in sources),'source_count':len(sources),'source_metadata_only':True,'no_payload_or_GT_or_metric_read':True,'queue_or_target_modified_by_helper':False,'caller_must_prove_selected_roots_quiescent':True}
    if write:
        immutable_write(new_spec_path,new_spec);immutable_write(new_recipe_path,new_recipe);immutable_write(folder/'cpu_job.json',job);immutable_write(folder/'bundle.json',bundle)
    return bundle
