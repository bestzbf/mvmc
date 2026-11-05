#!/usr/bin/env python3
"""Prepare immutable World5 GPU/CPU jobs; never modify or start a live queue."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil

ASSETS=Path('@@BASELINE_ASSETS_ROOT@@')
INTEGRATION=ASSETS/'integration_20261104'
BRIDGE=ASSETS/'adapter_cpu_tests/uniquery4r_20261104/score_worldtrack_native.py'
BENCH=Path('@@RELEASED_BASELINE_HARNESS_ROOT@@')
OFFICIAL=Path('@@BENCHMARK_ROOT@@/code/St4RTrack_0f9a3f4_20261026')
EVALPY='@@BENCHMARK_ROOT@@/runtime/eval_env/bin/python'


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if path.exists():
        if path.read_text()!=text:raise ValueError('Existing immutable preparation differs: '+str(path))
    else:path.write_text(text)


def materialize(source_path,cases,directory,helpers):
    source=helpers.read(source_path);directory.mkdir(parents=True,exist_ok=True);selected=[]
    source_positions={c['id']:i for i,c in enumerate(source['cases'])}
    for case in cases:
        src=(source_path.parent/case['input']).resolve();relative=f"inputs/{source_positions[case['id']]:06d}.npz";dst=directory/relative;dst.parent.mkdir(exist_ok=True)
        if not dst.exists():
            try:os.link(src,dst)
            except OSError:
                with src.open('rb') as reader,dst.open('xb') as writer:shutil.copyfileobj(reader,writer,8*1024*1024)
        if dst.is_symlink() or helpers.sha(dst)!=case['input_sha256']:raise ValueError('Frozen input copy differs')
        c=copy.deepcopy(case);c['input']=relative;selected.append(c)
    result=copy.deepcopy(source);result['cases']=selected
    result['source_manifest_leakage_metadata']=copy.deepcopy(source.get('leakage'))
    result['leakage']={'status':'unknown','reason':'Author baseline training inventory not supplied',
                       'evaluation_scene_ids':[c.get('scene_id',c['id']) for c in selected]}
    binding=copy.deepcopy(result['protocol'].get('release_binding',{}));binding.update(status='fixed_subset_not_formal',coverage_verified=False)
    result['protocol']['release_binding']=binding
    result['fixed_subset']={'source_manifest':str(source_path),'source_manifest_sha256':helpers.sha(source_path),
        'case_count':len(selected),'sample_ids':[c['id'] for c in selected],'formal_score_admitted':False,
        'all_original_frames_and_queries_preserved':True}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--scorer',type=Path,required=True);p.add_argument('--reference-panels',type=Path,required=True)
    a=p.parse_args();h=module(a.scorer,'fresh_world_helpers');refs=h.read(a.reference_panels)
    sysroot=str(BENCH)
    import sys
    sys.path.insert(0,sysroot)
    from multicamera4rc.benchmarks.released_data import load_manifest
    original_jobs={};plans={}
    for slug in ['uniquery4r','relift']:
        path=INTEGRATION/slug/'formal_plan.json';plan=h.read(path);plans[slug]=path
        for j in plan['jobs']:original_jobs[(slug,'reconstruction' if 'reconstruction' in j['id'] else 'trajectory')]=j
    trajectory_source=Path(original_jobs[('uniquery4r','trajectory')]['manifest']);full=h.read(trajectory_source)
    target_specs=[]
    for subset in ['adt_mini','pstudio_mini','po_mini','ds_mini']:
        panel=next(p for p in refs['panels'] if p['dataset']==subset and p['task']=='trajectory')
        candidates=[c for c in full['cases'] if c['subset']==subset]
        ids=[candidates[i]['id'] for i in [0,12,25,37,49]] if subset=='adt_mini' else panel['sample_ids']
        if len(ids)!=5 or any('decoration_skeleton_seq139_8' in cid for cid in ids):raise ValueError('World target must have5 eligible complete cases')
        target_specs.append(('trajectory',subset,trajectory_source,[c for c in full['cases'] if c['id'] in ids],panel))
    po_job=original_jobs[('uniquery4r','reconstruction')];po_source=Path(po_job['manifest']);po_full=h.read(po_source)
    panel=next(p for p in refs['panels'] if p['dataset']=='po_mini' and p['task']=='reconstruction')
    target_specs.append(('reconstruction','po_mini',po_source,[c for c in po_full['cases'] if c['id'] in panel['sample_ids']],panel))
    jobs=[];prepared=[];selection=[]
    bridge=h.module(BRIDGE,'prepare_world_native_bridge');_,wrapper_binding=bridge.load_native_grid_wrapper(BENCH)
    for task,subset,source_path,cases,reference in target_specs:
        directory=a.root/'world'/'manifests'/(task+'_'+subset)
        target=materialize(source_path,cases,directory,h);target_path=directory/'manifest_target5.json';write_new(target_path,target)
        load_manifest(target_path,inference_only=True)
        selection.append({'panel_id':task+'_'+subset,'target_manifest':str(target_path),'target_manifest_sha256':h.sha(target_path),
            'source_manifest':str(source_path),'source_manifest_sha256':h.sha(source_path),
            'sample_ids':[c['id'] for c in cases],'cases':[{'id':c['id'],'input':str((source_path.parent/c['input']).resolve()),
                'input_sha256':c['input_sha256'],'frames':c['frames'],'queries':c['queries']} for c in cases]})
        slugs=['uniquery4r','relift'] if task=='trajectory' else ['uniquery4r']
        for slug in slugs:
            original=original_jobs[(slug,task)];method=original['method'];base=a.root/'world'/slug/(task+'_'+subset)
            prediction_sources=[];reused=set()
            if slug=='relift' and subset=='adt_mini':
                oldmf=ASSETS/'quick_newmodels_20261104/relift/manifest.json';oldip=ASSETS/'quick_newmodels_20261104/relift/predictions/predictions.json'
                old=h.read(oldip);old_by_id={r['id']:r for r in old['cases']}
                if old['checkpoint']!=original['expected_baseline_checkpoint']:raise ValueError('Old ReLift checkpoint differs')
                reused={c['id'] for c in cases if c['id'] in old_by_id}
                prediction_sources.append({'source_kind':'existing_verified_cache','manifest':str(oldmf),'manifest_sha256':h.sha(oldmf),
                    'prediction_index':str(oldip),'prediction_index_sha256':h.sha(oldip)})
            missing=[c for c in cases if c['id'] not in reused]
            if len(missing)==len(cases):infer_path=target_path
            else:
                infer=copy.deepcopy(target);infer['cases']=[c for c in target['cases'] if c['id'] not in reused]
                infer['fixed_subset']['case_count']=len(infer['cases']);infer['fixed_subset']['sample_ids']=[c['id'] for c in infer['cases']]
                infer_path=directory/('manifest_'+slug+'_missing.json');write_new(infer_path,infer);load_manifest(infer_path,inference_only=True)
            argv=list(original['predict_argv']);argv[argv.index('--manifest')+1]=str(infer_path);argv[argv.index('--output')+1]=str(base/'predictions')
            gpu_id='world10h_'+slug+'_'+task+'_'+subset+'_predict'
            score_id='world10h_'+slug+'_'+task+'_'+subset+'_score'
            prediction_sources.append({'source_kind':'fresh_producer_output','producer_job_id':gpu_id,
                'manifest':str(infer_path),'manifest_sha256':h.sha(infer_path),'prediction_index':str(base/'predictions/predictions.json'),
                'prediction_index_sha256':None})
            comparison=[]
            for old in reference['methods']:
                common=[c['id'] for c in cases if c['id'] in [r['id'] for r in old['per_case']]]
                comparison.append({'method':old['method'],'existing_subset_score':old['score_path'],'existing_subset_score_sha256':old['score_sha256'],
                    'existing_per_case_reusable_ids':common,'target_sample_ids':[c['id'] for c in cases],
                    'same_target_subset_ready':len(common)==len(cases),'old_N':len(old['per_case']),
                    'action':'filter_existing_per_case' if len(common)==len(cases) else 'CPU_score_missing_target_ids_from_existing_predictions_no_GPU',
                    'source_prediction_index':old['source_prediction_index'],'source_prediction_index_sha256':old['source_prediction_index_sha256']})
            score_job={'schema':'world-fresh-subset-job-v1','method':method,'task':task,'subset':subset,
                'source_plan':str(plans[slug]),'source_plan_sha256':h.sha(plans[slug]),'source_job_id':original['id'],'source_job_sha256':h.canonical(original),
                'source_manifest':str(source_path),'source_manifest_sha256':h.sha(source_path),'target_manifest':str(target_path),'target_manifest_sha256':h.sha(target_path),
                'sample_ids':[c['id'] for c in cases],'expected_checkpoint':original['expected_baseline_checkpoint'],
                'expected_inference_settings':h.expected_settings(original,task),'prediction_sources':prediction_sources,
                'benchmark_code_root':str(BENCH),'official_root':str(OFFICIAL),'native_grid_bridge':str(BRIDGE),'native_grid_bridge_sha256':h.sha(BRIDGE),
                'expected_wrapper_binding':wrapper_binding,'output_dir':str(base/'scoring'),'comparison_references':comparison}
            score_path=base/'score_job.json';write_new(score_path,score_job)
            plan={'schema':'eval10h-world-prediction-batch-v1','gpu_job_id':gpu_id,'method':method,'task':task,'subset':subset,
                'target_count':5,'new_inference_cases':len(missing),'reused_case_ids':sorted(reused),
                'predict_argv':argv,'environment':original.get('environment',{}),'source_plan':str(plans[slug]),
                'source_plan_sha256':h.sha(plans[slug]),'source_job_id':original['id'],'source_job_sha256':h.canonical(original),
                'manifest':str(infer_path),'manifest_sha256':h.sha(infer_path),'target_manifest':str(target_path),
                'score_job':str(score_path),'score_argv':[EVALPY,str(a.scorer),'--job',str(score_path)],'formal_score_admitted':False}
            write_new(base/'prediction_plan.json',plan)
            jobs.append({'id':gpu_id,'kind':'gpu','method':method,'dataset':task+'_'+subset,'argv':argv,
                'env':original.get('environment',{}),'timeout_seconds':7200,'expected_outputs':[str(base/'predictions/predictions.json')],
                'depends_on':[]})
            jobs.append({'id':score_id,'kind':'cpu','method':method,'dataset':task+'_'+subset,'argv':plan['score_argv'],
                'env':{'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'},'timeout_seconds':7200,
                'depends_on':[gpu_id],'expected_outputs':[str(base/'scoring/score.json')],'score_path':str(base/'scoring/score.json')})
            prepared.append(plan)
    write_new(a.root/'queue_plan_world_append.json',{'duration_hours':10,'allowed_physical_gpus':[3,4],'sealed':False,'jobs':jobs})
    write_new(a.root/'world_preparation.json',{'schema':'eval10h-world-preparation-v1','status':'READY_FOR_ROOT_QUEUE_APPEND',
        'source_reader_preflight':'frozen load_manifest(inference_only=True) passed for all target/inference manifests',
        'new_gpu_work_started':False,'live_queue_modified':False,'batches':prepared,'selections':selection,
        'original_ADT3_retained_separately':True,'target5_does_not_mix_old3_aggregate':True})
    print(json.dumps({'status':'READY','gpu_jobs':len(prepared),'cpu_jobs':len(prepared),'new_prediction_cases':sum(p['new_inference_cases'] for p in prepared),
        'append':str(a.root/'queue_plan_world_append.json'),'batches':[{k:p[k] for k in ['method','task','subset','new_inference_cases','reused_case_ids','score_job']} for p in prepared]},ensure_ascii=False))


if __name__=='__main__':main()
