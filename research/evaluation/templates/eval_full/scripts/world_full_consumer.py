# Derived isolated consumer; original subset guard and official metrics unchanged.
#!/usr/bin/env python3
"""Fresh/cached WorldTrack subset CPU scoring with genuine producer receipts.

No source_score prerequisite and no synthetic run-state or prediction index.
Each selected case retains its actual producer manifest/index/state/receipt.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time


PINS={'UniQuery4R':'e6345ee1249827347638648d4a08d0fa1eab2438c827697c1277a00acbbd62ce',
      'ReLift+AllTracker+DA3':'be432e2d75708333d87e3b711d078fdcb4b7c40180674bf2c3e676d1c2f0df97'}
PRIMARY=('avg_pts_global','epe_global','avg_pts_sim3_closed','epe_sim3_closed')


def read(path):
    with Path(path).open() as f:return json.load(f,parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def clean(value):
    import numpy as np
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    a=np.asarray(value)
    if a.size!=1:raise ValueError('Non-scalar official metric')
    n=float(a.reshape(-1)[0]);return n if np.isfinite(n) else None


def metric_protocol(manifest):
    p=copy.deepcopy(manifest['protocol']);p.pop('release_binding',None);p.pop('subsets',None);return p


def case_equal(a,b):
    a=copy.deepcopy(a);a['input']=b['input'];return a==b


def expected_settings(job,task):
    method=job['method'];argv=job['predict_argv']
    def arg(flag,default):return argv[argv.index(flag)+1] if flag in argv else default
    if method=='UniQuery4R':
        return {'method':method,'prediction_kind':task,'target_time_policy':'author_joint_encoding_all_frozen_frames',
          'preprocessing':'author keep-ratio504; Pillow bicubic RGB/255; patch14','native_output_grid':True,
          'max_size':int(arg('--max-size',504)),'query_chunk_size':int(arg('--query-chunk-size',5000)),
          'reference_global_frame':0,'query_sampling':'unchanged official WorldTrack evaluator',
          'output_contract':'native dense tracks_XYZ only','adapter_sha256':PINS[method]}
    if method=='ReLift+AllTracker+DA3' and task=='trajectory':
        return {'method':method,'prediction_kind':'trajectory','target_time_policy':'all_frozen_first64_frames',
          'protocol_extension':'author pipeline on frozen64; not paper84 table',
          'geometry':'DA3NESTED-GIANT-LARGE-1.1; upper_bound_resize504; saddle_balanced',
          'tracker':'AllTracker window16 iters4; author DA3 processed frames',
          'corrector':{'group':1024,'batch':8,'seed':0},
          'query_sampling':'unchanged official WorldTrack evaluator; native DA3 grid',
          'output_contract':'native dense tracks_XYZ only','adapter_sha256':PINS[method]}
    raise ValueError('Unsupported frozen method/task')


def load_sources(job):
    if job.get('schema')!='world-fresh-subset-job-v1':raise ValueError('Unknown job schema')
    if sha(job['source_plan'])!=job['source_plan_sha256']:raise ValueError('Frozen source plan changed')
    matches=[j for j in read(job['source_plan'])['jobs'] if j['id']==job['source_job_id']]
    if len(matches)!=1 or canonical(matches[0])!=job['source_job_sha256']:raise ValueError('Frozen model job changed')
    original=matches[0]
    if original['method'] not in PINS:raise ValueError('Only frozen UniQuery/ReLift recipes supported')
    for artifact in original['checkpoint_artifacts']:
        if sha(artifact['path'])!=artifact['sha256']:raise ValueError('Frozen adapter/bridge bytes changed')
    if PINS[original['method']] not in [a['sha256'] for a in original['checkpoint_artifacts']]:raise ValueError('Unapproved adapter version')
    if sha(job['source_manifest'])!=job['source_manifest_sha256']:raise ValueError('Canonical source manifest changed')
    source=read(job['source_manifest']);target=read(job['target_manifest'])
    if sha(job['target_manifest'])!=job['target_manifest_sha256']:raise ValueError('Fixed target manifest changed')
    task=source['protocol']['task'];settings=expected_settings(original,task)
    if job['expected_inference_settings']!=settings:raise ValueError('Expected settings differ from frozen recipe')
    checkpoint=original['expected_baseline_checkpoint']
    if job['expected_checkpoint']!=checkpoint:raise ValueError('Expected model checkpoint differs')
    if checkpoint.get('native_multicamera4rc_loading') is not False:raise ValueError('Native Ours checkpoint is forbidden here')
    for key in ['benchmark','input_condition','coordinates','official_sources']:
        if source[key]!=target[key]:raise ValueError('Fixed subset protocol changed: '+key)
    if source['benchmark']!='worldtrack' or source['coordinates']!='first_camera_world' or metric_protocol(source)!=metric_protocol(target):
        raise ValueError('Only unchanged first-camera WorldTrack metric protocol is supported')
    source_cases={c['id']:c for c in source['cases']};selected=target['cases']
    if not selected or len(selected)>sum(c['subset']==job['subset'] for c in source['cases']) or [c['id'] for c in selected]!=job['sample_ids']:raise ValueError('Declared target IDs/count differ')
    if [c['id'] for c in source['cases'] if c['id'] in job['sample_ids']]!=job['sample_ids']:raise ValueError('Fixed source case order changed')
    for c in selected:
        if c['subset']!=job['subset'] or not case_equal(c,source_cases[c['id']]):raise ValueError('Selected RGB/query/frame metadata changed')
        if c['views']!=[0] or not 1<=c['frames']<=64:raise ValueError('Frozen C1/first64 contract changed')
    gathered={};evidence=[]
    for spec in job['prediction_sources']:
        mf=Path(spec['manifest']);ip=Path(spec['prediction_index'])
        if sha(mf)!=spec['manifest_sha256']:raise ValueError('Producer manifest changed')
        if spec.get('prediction_index_sha256') and sha(ip)!=spec['prediction_index_sha256']:raise ValueError('Reused producer index changed')
        producer=read(mf);index=read(ip);state_path=ip.parent/'run-state.json';state=read(state_path)
        if canonical({k:v for k,v in state.items() if k!='manifest_sha256'})!=job['expected_group_signature_sha256']:raise ValueError('Producer runtime/source/numerics identity differs')
        if index.get('schema')!='released-tracking-predictions-v1' or state.get('schema')!='released-baseline-run-v1':raise ValueError('Producer index/run-state schema differs')
        for key in ['benchmark','input_condition','coordinates','official_sources']:
            if producer[key]!=source[key]:raise ValueError('Producer data convention differs')
        if metric_protocol(producer)!=metric_protocol(source):raise ValueError('Producer metric protocol differs')
        if (job['subset'] not in producer['protocol']['subsets']
                or not set(producer['protocol']['subsets']).issubset(source['protocol']['subsets'])):
            raise ValueError('Producer dataset scope differs')
        if (index['manifest_sha256']!=sha(mf) or state['manifest_sha256']!=sha(mf)
                or canonical(state)!=index['run_state_sha256']):raise ValueError('Actual producer manifest/run-state binding differs')
        if index['checkpoint']!=checkpoint or state['checkpoint']!=checkpoint:raise ValueError('Producer model differs')
        if state['method']!=original['method'] or index['method']!=original['method'] or state['inference_settings']!=settings:
            raise ValueError('Producer method/settings differ')
        if index['target_time_policy']!=settings['target_time_policy'] or index['prediction_kind']!=task:raise ValueError('Producer temporal/task contract differs')
        pcases=producer['cases'];records=index['cases']
        if [r['id'] for r in records]!=[c['id'] for c in pcases] or len({r['id'] for r in records})!=len(records):raise ValueError('Producer is incomplete, duplicated, or reordered')
        if not index.get('metric_support',{}).get('position') or index['metric_support'].get('AJ') is not False or index['metric_support'].get('OA') is not False:raise ValueError('World position-only support declaration differs')
        allowed=set(spec.get('case_ids',job['sample_ids']))
        if not allowed.issubset(set(job['sample_ids'])):raise ValueError('Source selection exceeds scored target')
        for c,r in zip(pcases,records):
            if c['id'] not in job['sample_ids'] or c['id'] not in allowed:continue
            if c['id'] in gathered:raise ValueError('Two producers claim the same selected case')
            if c['id'] not in source_cases or not case_equal(c,source_cases[c['id']]):raise ValueError('Producer case differs from canonical source')
            path=(ip.parent/r['file']).resolve();receipt_path=path.with_suffix('.receipt.json')
            if not path.is_relative_to(ip.parent.resolve()) or not path.is_file():raise ValueError('Prediction path escaped its producer or is missing')
            receipt=read(receipt_path)
            if (receipt.get('schema')!='released-prediction-case-v1' or receipt['record']!=r
                    or receipt['run_state_sha256']!=index['run_state_sha256'] or receipt['case_sha256']!=canonical(c)
                    or r['input_sha256']!=source_cases[c['id']]['input_sha256']):raise ValueError('Genuine producer receipt/input binding differs')
            gathered[c['id']]={'case':source_cases[c['id']],'prediction':path,'record':r,'receipt':receipt,
                'producer_manifest':str(mf),'producer_manifest_sha256':sha(mf),'producer_index':str(ip),
                'producer_index_sha256':sha(ip),'producer_run_state':str(state_path),'producer_run_state_sha256':sha(state_path),
                'producer_run_state_canonical_sha256':canonical(state),'receipt_path':str(receipt_path),'receipt_sha256':sha(receipt_path)}
        evidence.append({'manifest':str(mf),'manifest_sha256':sha(mf),'prediction_index':str(ip),
          'prediction_index_sha256':sha(ip),'actual_run_state_sha256':sha(state_path),
          'source_kind':spec['source_kind'],'no_synthetic_producer_state':True})
    if set(gathered)!=set(job['sample_ids']):raise ValueError('Missing selected complete producer case')
    return source,target,checkpoint,[gathered[cid] for cid in job['sample_ids']],evidence


def score(job,job_path,metadata_only=False):
    source,target,checkpoint,rows,evidence=load_sources(job)
    expected_eligible=job.get('expected_eligible_ids',job['sample_ids'])
    if [cid for cid in job['sample_ids'] if cid in set(expected_eligible)]!=expected_eligible or len(expected_eligible)!=len(set(expected_eligible)):
        raise ValueError('Eligible IDs are not an ordered subset of canonical prediction cases')
    excluded=[cid for cid in job['sample_ids'] if cid not in set(expected_eligible)]
    exclusion_records=[]
    if excluded:
        proof_ref=job.get('eligibility_proof',{})
        if not proof_ref.get('path') or sha(proof_ref['path'])!=proof_ref.get('sha256'):
            raise ValueError('Excluded case requires pinned original official eligibility evidence')
        proof=read(proof_ref['path'])
        if proof['canonical_manifest_sha256']!=job['source_manifest_sha256'] or proof['official_sources']!=source['official_sources'] or sha(proof['source_score'])!=proof['source_score_sha256']:
            raise ValueError('Official eligibility proof source identity differs')
        entries={e['id']:e for e in proof['excluded_cases']}
        declared=[cid for cid in job['sample_ids'] if cid in entries]
        if declared!=excluded:raise ValueError('Method-specific or undeclared exclusion forbidden')
        for cid in excluded:
            entry=entries[cid]
            if entry['reason']!='official_no_first_frame_visible_points' or entry['scorer_eligible'] is not False or entry['first_frame_visible']!=0 or entry['first_frame_inbounds']!=0:
                raise ValueError('Unsupported official eligibility exclusion')
            exclusion_records.append(entry)
    if metadata_only:
        return {'status':'METADATA_BINDINGS_OK','cases':len(rows),'producer_count':len(evidence),'no_npz_read':True}
    import numpy as np
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
    if torch.cuda.is_initialized():raise ValueError('CPU scorer must not initialize CUDA')
    if sha(job['native_grid_bridge'])!=job['native_grid_bridge_sha256']:raise ValueError('Native-grid bridge changed')
    bridge=module(job['native_grid_bridge'],'world10h_native_grid')
    runner,binding=bridge.load_native_grid_wrapper(Path(job['benchmark_code_root']))
    if binding!=job['expected_wrapper_binding']:raise ValueError('Pinned wrapper binding changed')
    official=runner.OfficialWorldTrack(Path(job['official_root']))
    if official.provenance!=source['official_sources']:raise ValueError('Pinned official metric source changed')
    root=Path(job['output_dir']);root.mkdir(parents=True,exist_ok=True);result_path=root/'score.json'
    job_sha=sha(job_path)
    if result_path.exists():
        result=read(result_path)
        if result.get('job_sha256')!=job_sha:raise ValueError('Prior score belongs to a different immutable job')
        return {'status':'EXISTING_VALIDATED_RESULT','output':str(result_path)}
    work=Path(tempfile.mkdtemp(prefix='cpu_attempt_',dir=root));export=work/'native_export';export.mkdir()
    view=work/'data_view'/job['subset'];view.mkdir(parents=True);paths=[];gts=[];case_bindings=[];scored_rows=[]
    for row in rows:
        c,r,path=row['case'],row['record'],row['prediction']
        if sha(path)!=r['sha256']:raise ValueError('Selected prediction payload SHA differs')
        with np.load(path,allow_pickle=False) as archive:points=np.asarray(archive['tracks_XYZ'])
        if points.ndim!=4 or points.shape[0]!=c['frames'] or points.shape[-1]!=3 or points.dtype.kind!='f' or not np.isfinite(points).all():raise ValueError('Native dense trajectory/reconstruction contract differs')
        if list(points.shape)!=row['receipt']['arrays']['tracks_XYZ']['shape']:raise ValueError('Dense points shape differs from real producer receipt')
        dst=export/(Path(c['name']).stem+'.npy');np.save(dst,points,allow_pickle=False)
        gt_source=Path(c['sources'][0]['path']).resolve(strict=True)
        if sha(gt_source)!=c['sources'][0]['sha256']:raise ValueError('Frozen original GT payload changed')
        case_bindings.append({k:str(v) if isinstance(v,Path) else v for k,v in row.items() if k not in ['receipt','record','case']}|{
            'id':c['id'],'input_sha256':c['input_sha256'],'prediction_sha256':r['sha256'],
            'frames':c['frames'],'queries':c['queries'],'native_shape':list(points.shape)})
        if c['id'] in excluded:continue
        gt=view/c['name'];gt.symlink_to(gt_source);paths.append(dst);gts.append(gt);scored_rows.append(row)
    reconstruction=source['protocol']['task']=='reconstruction'
    metrics=clean(official.score(view.parent,job['subset'],paths,gts,work/'aggregate',reconstruction=reconstruction))
    per_case=[]
    for i,(row,path,gt) in enumerate(zip(scored_rows,paths,gts)):
        one=work/'per_case_views'/f'{i:03d}'/job['subset'];one.mkdir(parents=True);(one/gt.name).symlink_to(gt.resolve())
        values=clean(official.score(one.parent,job['subset'],[path],[one/gt.name],work/'per_case_metrics'/f'{i:03d}',reconstruction=reconstruction))
        per_case.append({'id':row['case']['id'],'metrics':values})
    robust={}
    for key in PRIMARY:
        values=[c['metrics'][key] for c in per_case]
        if any(v is None for v in values) or not np.isclose(np.mean(values),metrics[key],rtol=1e-6,atol=1e-7):raise ValueError('Official aggregate is not the same-case video mean: '+key)
        robust[key]={'mean':float(np.mean(values)),'median_diagnostic':statistics.median(values),
            'sample_std_diagnostic':statistics.stdev(values) if len(values)>1 else None,'valid_N':len(values)}
    result={'schema':'world-catalog-score-v1','result_scope':'declared_selection_requires_full_coverage_guard','derived_parent_scorer_sha256':'a2421aa88d4e3d302b2d5c3c47a4e744797ab1f3757ef9e96195acbb474e0f3a','formal_score_admitted':False,
      'job_sha256':job_sha,'job':job,'method':job['method'],'task':source['protocol']['task'],'subset':job['subset'],
      'cases':len(scored_rows),'prediction_cases':len(rows),'sample_ids':job['sample_ids'],'scored_sample_ids':expected_eligible,'excluded_cases':exclusion_records,'checkpoint':checkpoint,'inference_settings':job['expected_inference_settings'],
      'metrics':metrics,'per_case':per_case,'robust_diagnostics':robust,'official_sources':official.provenance,
      'native_grid_wrapper':binding,'actual_producer_sources':evidence,'case_bindings':case_bindings,
      'source_manifest_sha256':job['source_manifest_sha256'],'target_manifest_sha256':job['target_manifest_sha256'],
      'metric_protocol':metric_protocol(source),'source_manifest_leakage_metadata':source.get('leakage'),
      'baseline_training_inventory':{'status':'unknown','reason':'Author training inventory not supplied'},
      'prediction_payload_sha256_recomputed':True,'no_synthetic_runstate_or_prediction_index':True,
      'mean_is_official':True,'median_and_std_diagnostic_only':True,'scorer_sha256':sha(__file__),
      'comparison_references':job.get('comparison_references',[]),'cpu_only':not torch.cuda.is_initialized()}
    tmp=root/('score.'+str(time.time_ns())+'.tmp');tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,result_path)
    return {'status':'SUCCESS','output':str(result_path),'cases':len(scored_rows),'prediction_cases':len(rows),'metrics':metrics}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True);p.add_argument('--metadata-only',action='store_true');a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU scorer requires CUDA_VISIBLE_DEVICES empty')
    print(json.dumps(score(read(a.job),a.job,a.metadata_only),ensure_ascii=False))


if __name__=='__main__':main()
