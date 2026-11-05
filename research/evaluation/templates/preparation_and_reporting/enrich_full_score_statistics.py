"""Reuse already independently validated scalar caches; never rerun a scorer."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
path=ROOT/'full_score_reuse_inventory.json'
report=json.loads(path.read_text())
cache_path=Path('outputs/balanced_subset_20261104/legacy_official_stdout_per_case.json')
cache=json.loads(cache_path.read_text())

def flatten(value,prefix=''):
    out={}
    for key,v in value.items():
        name=prefix+key
        if isinstance(v,dict):out.update(flatten(v,name+'.'))
        elif isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v):out[name]=v
    return out

for row in report['rows']:
    pc=row['source_per_case']
    origin='original_full_score_per_case'
    if not pc:
        candidates=[j for j in cache['jobs'] if j['status']=='validated_exact_scalar_cache'
          and j['source_score_sha256']==row['source_score_sha256']
          and j['source_prediction_index_sha256']==row['source_prediction_index_sha256']
          and j['source_manifest_sha256']==row['source_manifest_sha256']
          and [v['id'] for v in j['per_case']]==row['panel_sample_ids']]
        if len(candidates)==1:
            old=candidates[0];pc=[{'id':v['id'],'metrics':v['metrics']} for v in old['per_case']]
            row['cached_original_stdout']={'cache_path':str(cache_path),'cache_file_sha256':hashlib.sha256(cache_path.read_bytes()).hexdigest(),
              'cache_job_id':old['job_id'],'source_original_log':old['log'],'source_original_log_sha256':old['log_sha256'],
              'full_mean_abs_deltas_recorded':old['full_mean_abs_deltas'],'previous_independent_review':'artifacts/balanced_subset_20261104/original_stdout_source_review.json'}
            row['cached_source_per_case']=pc
            origin='previously_validated_original_stdout_exact_scalar_cache'
    values=[flatten(v['metrics']) for v in pc]
    official=flatten(row['full_official_metrics'])
    stats={}
    if len(values)==row['N']:
        for name,mean in official.items():
            points=[v[name] for v in values if name in v]
            if len(points)!=row['N']:continue
            recomputed=statistics.mean(points)
            tolerance=.00500001 if name.startswith('3dpt/') else 1e-12
            if abs(recomputed-mean)>tolerance:continue
            stats[name]={'official_mean_preserved':mean,'per_case_arithmetic_mean':recomputed,
              'sample_std_ddof1':statistics.stdev(points),'N':len(points),
              'mean_abs_difference':abs(recomputed-mean),'mean_tolerance':tolerance,
              'per_case_source':origin,'units':'source metric units; MV percentage, TAP/World APD fraction, World EPE meters'}
    row['statistics_from_cached_scalars']=stats
    row['std_available']=bool(stats)
    row['std_source']=origin if stats else None
    if not stats:
        row['std_unavailable_reason']='Original score has no per-case values and no admitted exact eligible scalar cache; retain official mean without constructing std.'
    row['CPU_rescoring_needed_for_official_mean']=False
report['summary']={'rows':len(report['rows']),'original_full_means_reusable':sum(r['can_reuse_official_mean'] for r in report['rows']),
  'mean_and_std_from_existing_scalar_cache':sum(r['std_available'] for r in report['rows']),
  'mean_only_rows':[r['row_id'] for r in report['rows'] if not r['std_available']],
  'new_CPU_scoring_executed':False,'GPU_started':False,
  'eligibility_note':'WorldTrack ADT attempted50/scored49/excluded1; its official mean is reused with that denominator. No unverified50-case std is supplied.'}
path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report['summary'],ensure_ascii=False))
