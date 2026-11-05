#!/usr/bin/env python3
"""Independent CPU MV subset scorer; original complete-split scorer unchanged.

Uses the same immutable-cache --spec contract as tap_subset_score.py, with schema
'eval-10h-mv-subset-score-spec-v1'. Only the declared RGB query-anchor Sim3 tier.
"""
import argparse,copy,json,math,os,statistics,sys
from pathlib import Path
import tap_subset_score as cache


def score(spec_path,output):
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
    import numpy as np
    import torch
    torch.set_num_threads(4)
    spec=cache.read(spec_path);cache.require(spec.get('schema')=='eval-10h-mv-subset-score-spec-v1','Wrong MV subset specification')
    output=Path(output);cache.require(not output.exists(),'Preserve old score output')
    code=Path(spec.get('benchmark_code_root','@@OURS_EVAL_HARNESS_ROOT@@')).resolve(strict=True);sys.path.insert(0,str(code))
    from multicamera4rc.benchmarks import released_runner as runner
    from multicamera4rc.benchmarks.official import OfficialMVTracker
    from multicamera4rc.benchmarks.released_official import OfficialWorldTrack
    manifestpath=Path(spec['manifest']).resolve(strict=True);sourcepath=Path(spec['source_manifest']).resolve(strict=True);manifest=cache.read(manifestpath);original=cache.read(sourcepath)
    manifest_sha=cache.pin(manifestpath,spec.get('manifest_sha256'));original_sha=cache.pin(sourcepath,spec.get('source_manifest_sha256'))
    cache.require(manifest.get('schema')==original.get('schema')=='released-tracking-benchmark-v1','Wrong MV manifest schema')
    cache.require(original.get('benchmark')=='mvtracker' and original.get('input_condition')=='rgb_only_3d_query_anchor_sim3' and original.get('coordinates')=='native_predicted_world','Only declared RGB query-anchor MV extension supported')
    cache.require(original['protocol'].get('mode')=='rgb_only_3d_query_anchor_sim3','Do not mix original calibrated XYZ-query reference')
    for key in ('benchmark','protocol','official_sources','input_condition','coordinates'):
        cache.require(manifest.get(key)==original.get(key),'Subset changed MV protocol: '+key)
    cases=manifest['cases'];ids=[c['id'] for c in cases];cache.require(ids and len(set(ids))==len(ids),'Empty or duplicate case IDs')
    cache.require([c['id'] for c in original['cases'] if c['id'] in set(ids)]==ids,'Selection changed original case order')
    originals={c['id']:c for c in original['cases']}
    for case in cases:cache.same_case(case,originals[case['id']])
    protocol=runner.PROTOCOLS[original['protocol']['name']]
    official=OfficialMVTracker(spec.get('official_root','@@BENCHMARK_ROOT@@/code/mvtracker'));provenance=dict(official.provenance,aggregation=official.aggregation_provenance())
    cache.require(provenance==original['official_sources'],'Official MV metric/aggregation source differs')
    # Verify the original complete split using the unchanged official reader;
    # only this new independent report deliberately selects complete cases.
    cache.require(len(original['cases'])==protocol.sequences,'Original complete split length differs')
    dataset=official.dataset(protocol,original['data_root'])
    cache.require(list(dataset.seq_names)==[c['name'] for c in original['cases']],'Original complete split reader order differs')
    for case in original['cases']:
        cache.require(case['views']==list(protocol.views) and case['queries']==protocol.queries and (not protocol.frames or case['frames']==protocol.frames),'Original MV frame/query/view contract differs')
    alignment=OfficialWorldTrack(spec.get('alignment_official_root','@@BENCHMARK_ROOT@@/code/St4RTrack_0f9a3f4_20261026'))
    cache.require(alignment.provenance==original['protocol']['alignment_sources'],'Query-anchor Sim3 source differs')
    catalog,sources,settings=cache.load_sources(spec,original,ids);per_case=[];raw_rows=[];unaligned_rows=[];bindings=[]
    for case in cases:
        item=catalog[case['id']];original_case=originals[case['id']];label_path=(sourcepath.parent/original_case['labels']).resolve(strict=True)
        cache.pin(label_path,original_case['labels_sha256'])
        with np.load(label_path,allow_pickle=False) as archive:labels={key:archive[key] for key in archive.files}
        with np.load(item['path'],allow_pickle=False) as archive:
            specs={key:{'shape':list(archive[key].shape),'dtype':archive[key].dtype.str} for key in archive.files};cache.require(specs==item['receipt']['arrays'],'Actual arrays differ from committed receipt')
            cache.require('visibility' in archive.files,'Position-only MV caches require their original position-only scorer; do not fabricate visibility')
            tracks=archive['tracks_XYZ'];visible=archive['visibility'];runner.finite_tracks(tracks,labels['tracks'].shape);runner.boolean_visibility(visible,labels['visible'].shape)
            # Same functions, order and arguments as released_runner.score_predictions.
            unaligned=official.score(labels,tracks,visible,protocol)
            aligned,alignment_record=runner.align_query_points_for_scoring(alignment,labels,tracks)
            metrics=official.score(labels,aligned,visible,protocol)
        plain,missing=runner.json_metrics(metrics);rawplain,_=runner.json_metrics(unaligned)
        per_case.append({'id':case['id'],'metrics':plain,'scoring_alignment':alignment_record,'unaligned_metrics':rawplain,'unavailable_metric_paths':missing});raw_rows.append(metrics);unaligned_rows.append(unaligned)
        bindings.append({k:item[k] for k in ('case_id','path','source_manifest','source_manifest_sha256','source_index','source_index_sha256','source_root','source_kind','run_state_canonical_sha256','settings_signature')} | {'input_sha256':case['input_sha256'],'prediction_sha256':item['record']['sha256'],'source_case_sha256':item['receipt']['case_sha256'],'official_labels':str(label_path),'official_labels_sha256':original_case['labels_sha256']})
    mean,unavailable=runner.json_metrics(official.average(raw_rows));unaligned_mean,_=runner.json_metrics(official.average(unaligned_rows));subsets={c['subset'] for c in cases};cache.require(len(subsets)==1,'Do not mix MV datasets')
    robust={}
    for metric in mean:
        values=[r['metrics'].get(metric) for r in per_case];valid=[float(v) for v in values if v is not None and math.isfinite(float(v))]
        robust[metric]={'valid_n':len(valid),'unavailable_n':len(values)-len(valid),'official_rounded_mean':mean[metric],'unrounded_per_case_mean_diagnostic':statistics.mean(valid) if valid else None,'median':statistics.median(valid) if valid else None,'population_std':statistics.pstdev(valid) if valid else None,'sample_std':statistics.stdev(valid) if len(valid)>1 else None,'zero_count':sum(v==0.0 for v in valid)}
    names={'AJ3D_percent':'average_jaccard','APD3D_percent':'average_pts_within_thresh','OA_percent':'occlusion_accuracy','ATE':'ate_visible','MTE':'mte_visible','FDE':'fde_visible','Survival_percent':'survival'}
    cp=spec['expected_checkpoint'];leakage=copy.deepcopy(original.get('leakage'));report={'schema':'eval-10h-mv-subset-scores-v1','result_tier':'rgb_only_3d_query_anchor_sim3_fixed_subset','formal_score_admitted':False,'original_protocol_comparable':False,'new_gpu_inference':False,'benchmark':'mvtracker','manifest_sha256':manifest_sha,'source_manifest_sha256':original_sha,'protocol':original['protocol'],'official_sources':provenance,'input_condition':original['input_condition'],'coordinates':original['coordinates'],'sample_ids':ids,'sample_counts':{next(iter(subsets)):len(ids)},'checkpoint':cp,'leakage':leakage,'manifest_leakage_metadata':copy.deepcopy(leakage),'metrics':{next(iter(subsets)):mean},'unaligned_metrics':{next(iter(subsets)):unaligned_mean},'per_case':per_case,'official_mean_headline_any':{key:mean.get('3dpt/model__'+part+'__any') for key,part in names.items()},'metric_robustness':robust,'unavailable_metric_paths':unavailable,'prediction_sources':sources,'prediction_case_bindings':bindings,'shared_settings_signature':settings,'subset_score_spec':str(Path(spec_path).resolve()),'subset_score_spec_sha256':cache.sha(spec_path),'selection_policy':spec.get('selection_policy'),'scorer_sha256':cache.sha(__file__),'cache_helper_sha256':cache.sha(cache.__file__),'alignment_function_source_sha256':cache.sha(runner.__file__),'claim':'Independent fixed subset of complete original cases; unchanged official MV metric/round2 nanmean and scorer-only query-anchor Sim3; not full release or original calibrated protocol','units_note':'__any headline only. Official MV percentages already percentage points; do not x100. Dex/Panoptic errors cm; MV-Kubric100xscene units. Unsupported/nonfinite values stay null.','scoring_only_GT_usage':'Prepared original normalized labels/query-time XYZ used only in CPU score; original native model predictions unchanged','integrity':{'prediction_payload_sha_recomputed':True,'input_video_payload_sha_recomputed':False,'label_payload_sha_recomputed':True,'weight_payloads_read':False,'source_index_edited':False,'payloads_copied':False},'cpu_threads':4,'cuda_inference_initialized':torch.cuda.is_initialized()}
    if cp.get('kind') in ('author_rgb_only_native_baseline','composed_rgb_only_baseline'):
        cache.require(not leakage.get('checkpoint_sha256'),'Manifest inventory bound to another checkpoint');report['leakage']={'status':'unknown','reason':'Author baseline training scene inventory not supplied','evaluation_scene_ids':copy.deepcopy(leakage.get('evaluation_scene_ids',[]))}
    cache.require(report['cuda_inference_initialized'] is False,'CPU scorer initialized CUDA');output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();r=score(a.spec,a.output);print(json.dumps({'output':str(a.output),'sample_counts':r['sample_counts'],'headline_any':r['official_mean_headline_any'],'formal_score_admitted':False}))
