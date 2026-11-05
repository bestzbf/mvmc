#!/usr/bin/env python3
"""CPU-only official TAP subset scores from immutable independent cache sources.

--spec: eval-10h-tap-subset-score-spec-v1. No prediction index is edited and
record.file is always resolved relative to its own ORIGINAL prediction root.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,os,statistics,sys
from pathlib import Path


def require(ok,message):
    if not ok:raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON token '+x)))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def canonical(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def contained(root,name):
    root=Path(root).resolve(strict=True);p=(root/name).resolve(strict=True)
    require(p.is_relative_to(root),'Prediction escaped its own original root')
    return p

def pin(path,expected=None):
    actual=sha(path)
    require(not expected or actual==expected,'Pinned JSON/source bytes differ: '+str(path))
    return actual

def same_case(case,original):
    adjusted=copy.deepcopy(case);adjusted['input']=original['input']
    require(adjusted==original,'Frozen case changed beyond input path: '+str(case.get('id')))

def checkpoint_ok(cp):
    require(isinstance(cp,dict) and cp.get('sha256'),'Missing checkpoint declaration')
    if cp.get('kind') in ('author_rgb_only_native_baseline','composed_rgb_only_baseline'):
        require(cp.get('native_multicamera4rc_loading') is False and not cp.get('native_loading'),'Baseline claims native model loading')
    else:
        loading=cp.get('native_loading',{})
        require(loading.get('audit_only') is False and type(loading.get('checkpoint_step')) is int,'Native cache lacks actual inference loading identity')

def load_sources(spec,original,selected):
    expected=spec['expected_checkpoint'];checkpoint_ok(expected);original_by_id={c['id']:c for c in original['cases']};wanted=set(selected);catalog={};evidence=[];signature=None
    for number,source in enumerate(spec['sources']):
        smpath=Path(source['manifest']).resolve(strict=True);sm=read(smpath);manifest_sha=pin(smpath,source.get('manifest_sha256'))
        for key in ('benchmark','input_condition','coordinates','protocol','official_sources'):
            require(sm.get(key)==original.get(key),'Source manifest protocol differs: '+key)
        require(sm.get('schema')=='released-tracking-benchmark-v1','Wrong source manifest schema')
        source_cases=sm['cases'];require(len({c['id'] for c in source_cases})==len(source_cases),'Duplicate source manifest IDs')
        case_by_id={c['id']:c for c in source_cases}
        for case in source_cases:
            require(case['id'] in original_by_id,'Source case absent from original split')
            same_case(case,original_by_id[case['id']])
        kind=source.get('kind','prediction_index');path=Path(source['path']).resolve(strict=True)
        if kind=='prediction_index':
            index=read(path);index_sha=pin(path,source.get('index_sha256'));root=path.parent
            require(index.get('schema')=='released-tracking-predictions-v1','Wrong prediction index schema')
            require(index.get('manifest_sha256')==manifest_sha,'Index/source manifest checksum differs')
            for key in ('benchmark','input_condition','coordinates'):
                require(index.get(key)==sm.get(key),'Index source gauge/input differs: '+key)
            require(index.get('checkpoint')==expected,'Index checkpoint differs from expected real model')
            require([r['id'] for r in index.get('cases',[])]==[c['id'] for c in source_cases],'Index does not cover its own bound manifest exactly in order')
            records={r['id']:r for r in index['cases']};state_digest=index.get('run_state_sha256')
        else:
            require(kind=='committed_case_receipts','Unknown cache source kind');root=path;index_sha=None;records={};state_digest=None
            for i,case in enumerate(source_cases):
                receiptpath=root/f'{i:06d}.receipt.json'
                if receiptpath.is_file():records[case['id']]=read(receiptpath)['record']
        state_path=root/'run-state.json';state=read(state_path);state_canonical=canonical(state)
        require(not state_digest or state_digest==state_canonical,'Index actual canonical run-state differs')
        require(state.get('checkpoint')==expected and state.get('manifest_sha256')==manifest_sha,'Source actual run-state model/manifest differs')
        if source.get('run_state_sha256'):require(source['run_state_sha256']==sha(state_path),'Pinned run-state bytes differ')
        settings=copy.deepcopy(state);settings.pop('manifest_sha256',None);current=canonical(settings)
        if signature is None:signature=current
        require(current==signature,'Cross-source model/settings/runtime/source identity differs; keep prediction modes separate')
        allowed=set(source.get('case_ids',selected));added=[]
        for case_id,record in records.items():
            if case_id not in wanted or case_id not in allowed:continue
            case=case_by_id[case_id];require(record.get('id')==case_id and record.get('input_sha256')==case['input_sha256'],'Selected record case/input differs')
            payload=contained(root,record['file']);receipt_path=contained(root,str(Path(record['file']).with_suffix('.receipt.json')));receipt=read(receipt_path)
            require(receipt.get('schema')=='released-prediction-case-v1' and receipt.get('record')==record,'Original committed receipt differs')
            require(receipt.get('run_state_sha256')==state_canonical and receipt.get('case_sha256')==canonical(case),'Receipt actual run-state/case identity differs')
            require(pin(payload,record['sha256'])==record['sha256'],'Prediction payload changed')
            item={'case_id':case_id,'path':str(payload),'record':record,'receipt':receipt,'source_manifest':str(smpath),'source_manifest_sha256':manifest_sha,'source_index':str(path) if kind=='prediction_index' else None,'source_index_sha256':index_sha,'source_root':str(root),'source_kind':kind,'run_state_canonical_sha256':state_canonical,'settings_signature':current}
            require(case_id not in catalog,'Ambiguous duplicate source for case '+case_id+'; use source.case_ids to select exactly one')
            catalog[case_id]=item;added.append(case_id)
        evidence.append({'kind':kind,'path':str(path),'index_sha256':index_sha,'manifest':str(smpath),'manifest_sha256':manifest_sha,'run_state':str(state_path),'run_state_sha256':sha(state_path),'run_state_canonical_sha256':state_canonical,'settings_signature':current,'selected_case_ids':added})
    require(set(catalog)==wanted,'Selected subset coverage incomplete: '+str(sorted(wanted-set(catalog))))
    return catalog,evidence,signature

def flatten(x,prefix=''):
    out={}
    for key,value in x.items():
        name=prefix+'.'+key if prefix else key
        if isinstance(value,dict):out.update(flatten(value,name))
        else:out[name]=value
    return out

def score(spec_path,output):
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
    import numpy as np
    import torch
    torch.set_num_threads(4)
    spec=read(spec_path);require(spec.get('schema')=='eval-10h-tap-subset-score-spec-v1','Wrong subset score specification')
    output=Path(output);require(not output.exists(),'Preserve existing score output; use a fresh attempt path')
    code=Path(spec.get('benchmark_code_root','@@OURS_EVAL_HARNESS_ROOT@@')).resolve(strict=True);sys.path.insert(0,str(code))
    from multicamera4rc.benchmarks.released_official import OfficialTAPVid3D
    from multicamera4rc.benchmarks.released_runner import json_metrics
    manifestpath=Path(spec['manifest']).resolve(strict=True);sourcepath=Path(spec['source_manifest']).resolve(strict=True)
    manifest=read(manifestpath);original=read(sourcepath);require(manifest.get('schema')==original.get('schema')=='released-tracking-benchmark-v1','Wrong benchmark manifest schema');target_sha=pin(manifestpath,spec.get('manifest_sha256'));original_sha=pin(sourcepath,spec.get('source_manifest_sha256'))
    require(original.get('benchmark')=='tapvid3d' and original.get('coordinates')=='per_frame_camera' and original.get('input_condition')=='rgb_only','Only original single-camera TAP camera-coordinate contract is supported')
    for key in ('benchmark','protocol','official_sources','input_condition','coordinates'):
        require(manifest.get(key)==original.get(key),'Subset changed official protocol/input: '+key)
    require(manifest['protocol'].get('scaling')=='median','Only unchanged median protocol is supported')
    cases=manifest['cases'];ids=[c['id'] for c in cases];require(ids and len(set(ids))==len(ids),'Empty/duplicate selected IDs')
    require([c['id'] for c in original['cases'] if c['id'] in set(ids)]==ids,'Subset must keep fixed original order')
    original_by_id={c['id']:c for c in original['cases']}
    for case in cases:
        same_case(case,original_by_id[case['id']]);require(case['views']==[0],'TAP subset scorer does not support multicamera cases')
    subsets={c['subset'] for c in cases};require(len(subsets)==1,'Score each dataset separately')
    catalog,source_evidence,settings_signature=load_sources(spec,original,ids)
    official=OfficialTAPVid3D(spec.get('official_root','@@BENCHMARK_ROOT@@/code/tapnet_730cda1_20261026'));require(official.provenance==original['official_sources'],'Pinned official metric implementation differs')
    complete=[f'{subset}/{name}' for subset in original['protocol']['subsets'] for name in official.filenames(subset,original['protocol']['split'])]
    require([c['id'] for c in original['cases']]==complete,'Original source manifest must be the complete frozen official split')
    per_case=[];bindings=[]
    for case in cases:
        item=catalog[case['id']]
        with np.load(item['path'],allow_pickle=False) as arrays:
            specs={key:{'shape':list(arrays[key].shape),'dtype':arrays[key].dtype.str} for key in arrays.files};require(specs==item['receipt']['arrays'],'Actual arrays differ from original committed specifications')
            tracks=arrays['tracks_XYZ'];vis=arrays['visibility'];require(tracks.shape==(case['frames'],case['queries'],3) and vis.shape==tracks.shape[:2] and vis.dtype==np.bool_ and np.isfinite(tracks).all(),'TAP prediction shape/finiteness/native visibility differs')
            gt=Path(case['sources'][0]['path']).resolve(strict=True);pin(gt,case['sources'][0]['sha256'])
            raw=official.score(gt,tracks,vis,'median')
        metrics,unavailable=json_metrics(raw);require(not unavailable,'Official score returned nonfinite values')
        per_case.append({'id':case['id'],'metrics':metrics,'scoring_alignment':None})
        bindings.append({k:item[k] for k in ('case_id','path','source_manifest','source_manifest_sha256','source_index','source_index_sha256','source_root','source_kind','run_state_canonical_sha256','settings_signature') } | {'input_sha256':case['input_sha256'],'prediction_sha256':item['record']['sha256'],'source_case_sha256':item['receipt']['case_sha256']})
    averaged,missing=json_metrics(official.average([r['metrics'] for r in per_case]));require(not missing,'Nonfinite dataset aggregate')
    rows=[flatten(r['metrics']) for r in per_case];meanflat=flatten(averaged);robust={}
    for metric in rows[0]:
        values=[float(row[metric]) for row in rows];require(all(math.isfinite(v) for v in values),'Invalid metric cannot become zero')
        require(math.isclose(statistics.mean(values),meanflat[metric],rel_tol=1e-12,abs_tol=1e-12),'Per-case mean differs from original official aggregation')
        robust[metric]={'valid_n':len(values),'mean':meanflat[metric],'median':statistics.median(values),'population_std':statistics.pstdev(values),'sample_std':statistics.stdev(values) if len(values)>1 else None,'zero_count':sum(v==0.0 for v in values),'units':'original official raw fraction'}
    cp=spec['expected_checkpoint'];leakage=copy.deepcopy(original.get('leakage'));report={'schema':'eval-10h-tap-subset-scores-v1','result_tier':'fixed_cached_subset_not_full_benchmark','formal_score_admitted':False,'new_gpu_inference':False,'benchmark':'tapvid3d','manifest_sha256':target_sha,'source_manifest_sha256':original_sha,'protocol':original['protocol'],'official_sources':official.provenance,'input_condition':original['input_condition'],'coordinates':original['coordinates'],'sample_ids':ids,'sample_counts':{next(iter(subsets)):len(ids)},'checkpoint':cp,'leakage':leakage,'manifest_leakage_metadata':copy.deepcopy(leakage),'metrics':{next(iter(subsets)):averaged},'per_case':per_case,'metric_robustness':robust,'robustness_note':'Original official equal-video mean is primary; median/std/zero_count are separate per-video diagnostics. Raw fractions multiply100 once for percent display. Sample std uses n-1 and is null for n=1.','query_frame_policy':'Unchanged frozen official median uses evaluation_weights=ones, including query frames.','prediction_sources':source_evidence,'prediction_case_bindings':bindings,'shared_settings_signature':settings_signature,'subset_score_spec':str(Path(spec_path).resolve()),'subset_score_spec_sha256':sha(spec_path),'selection_policy':spec.get('selection_policy'),'scorer_sha256':sha(__file__),'integrity':{'prediction_payload_sha_recomputed':True,'input_video_payload_sha_recomputed':False,'gt_source_payload_sha_recomputed':True,'weight_payloads_read':False,'source_index_edited':False,'payloads_copied':False},'cpu_threads':4,'cuda_inference_initialized':torch.cuda.is_initialized()}
    if cp.get('kind') in ('author_rgb_only_native_baseline','composed_rgb_only_baseline'):
        require(not leakage.get('checkpoint_sha256'),'Evaluation manifest inventory is bound to another checkpoint')
        report['leakage']={'status':'unknown','reason':'Author baseline training scene inventory not supplied','evaluation_scene_ids':copy.deepcopy(leakage.get('evaluation_scene_ids',[]))}
    require(report['cuda_inference_initialized'] is False,'CPU scoring initialized CUDA')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();r=score(a.spec,a.output);print(json.dumps({'output':str(a.output),'sample_counts':r['sample_counts'],'metrics':r['metrics'],'formal_score_admitted':False}))

if __name__=='__main__':main()
