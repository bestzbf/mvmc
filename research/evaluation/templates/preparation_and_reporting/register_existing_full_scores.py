"""Register verified full-score references in the new root; no scorer/model runs."""
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
report=json.loads((HERE/'full_score_reuse_inventory.json').read_text())
CODE=r'''
import datetime,hashlib,json,os
from pathlib import Path
root=Path('@@EVAL_FULL_ROOT@@')
def read(p):return json.loads(Path(p).read_text())
def digestfile(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
registered=[]
for row in ROWS:
 assert row['can_reuse_official_mean'] and all(row['checks'].values())
 score=read(row['source_score']);index=read(row['source_prediction_index']);manifest=read(row['source_manifest'])
 assert digestfile(row['source_score'])==row['source_score_sha256']
 assert digestfile(row['source_prediction_index'])==row['source_prediction_index_sha256']==score['predictions_sha256']
 assert digestfile(row['source_manifest'])==row['source_manifest_sha256']==score['manifest_sha256']
 assert canonical(score['checkpoint'])==canonical(index['checkpoint'])==row['source_checkpoint_canonical_sha256']
 subset=row['panel_subset'];ids=row['panel_sample_ids'];caseby={v['id']:v for v in manifest['cases']};source_ids=[i for i in score['sample_ids'] if i in set(ids)]
 assert source_ids==ids and len(ids)==row['N'] and all(i in caseby for i in ids)
 coverage=score.get('official_coverage',{}).get(subset)
 if coverage:
  assert coverage['attempted']==len(ids) and coverage['attempted_ids']==ids
  assert len(coverage['scored_ids'])==coverage['scored'] and len(coverage['excluded_ids'])==coverage['excluded']
  excluded=[e['id'] if isinstance(e,dict) else e for e in coverage['excluded_ids']]
  assert set(coverage['scored_ids'])|set(excluded)==set(ids)
  scored_ids=coverage['scored_ids']
 else:scored_ids=ids;excluded=[]
 wrapper={'schema':'full-frozen-official-score-reference-v1','method':row['method'],'panel_id':row['panel_id'],
  'benchmark':score['benchmark'],'input_condition':score['input_condition'],'coordinates':score['coordinates'],
  'protocol':score['protocol'],'official_sources':score['official_sources'],'checkpoint':score['checkpoint'],
  'sample_ids':ids,'sample_counts':{subset:len(ids)},'prediction_cases':len(ids),'scored_cases':len(scored_ids),
  'scored_sample_ids':scored_ids,'excluded_sample_ids':excluded,'metrics':{subset:score['metrics'][subset]},
  'full_frozen_coverage_verified':True,'formal_score_admitted':False,
  'source_formal_score_admitted':score.get('formal_score_admitted'),'source_result_tier':score.get('result_tier'),
  'source_claim':score.get('claim'),'metric_source_path':['metrics',subset],
  'official_eligibility':coverage,'official_eligibility_proof':{'source_score':row['source_score'],
    'source_score_sha256':row['source_score_sha256'],'source_field':['official_coverage',subset] if coverage else None,
    'source_audit_field':'official_eligibility_audit' if 'official_eligibility_audit' in score else None,
    'source_sample_count':score['sample_counts'][subset]},
  'source_binding':{k:row[k] for k in ['source_score','source_score_sha256','source_plan','source_job_id','source_manifest','source_manifest_sha256','source_prediction_index','source_prediction_index_sha256','source_runstate_sha256','source_checkpoint_canonical_sha256','source_checkpoint_sha256','source_checkpoint_declared_expected_step','actual_checkpoint_header_embedded_step','checkpoint_step_evidence','source_full_sample_count']},
  'statistics':row.get('statistics_from_cached_scalars',{}),
  'sample_std_ddof1':{k:v['sample_std_ddof1'] for k,v in row.get('statistics_from_cached_scalars',{}).items()} or None,
  'std_source':row.get('std_source'),'std_unavailable_reason':row.get('std_unavailable_reason'),
  'per_case':row.get('cached_source_per_case') or row.get('source_per_case',[]),
  'cached_original_stdout':row.get('cached_original_stdout'),
  'registration_only':True,'scoring_executed_here':False,'GPU_executed_here':False,
  'uses_original_official_aggregation':True,'upstream_byte_identity_claim_added':False}
 target=root/'scores'/('reused_'+row['row_id']+'.json');target.parent.mkdir(exist_ok=True)
 content=(json.dumps(wrapper,ensure_ascii=False,indent=2)+'\n').encode()
 if target.exists():assert target.read_bytes()==content,'Existing reuse reference changed; preserve it'
 else:
  tmp=target.with_name(target.name+'.registration.tmp')
  with tmp.open('wb') as f:f.write(content);f.flush();os.fsync(f.fileno())
  os.replace(tmp,target)
 registered.append({'row_id':row['row_id'],'method':row['method'],'panel_id':row['panel_id'],'path':str(target),
  'sha256':digestfile(target),'metric_path':['metrics',subset],'source_score':row['source_score'],
  'source_score_sha256':row['source_score_sha256'],'prediction_cases':len(ids),'scored_cases':len(scored_ids),
  'excluded_sample_ids':excluded,'std_available':wrapper['sample_std_ddof1'] is not None,
  'full_frozen_coverage_verified':True,'formal_score_admitted':False,'scoring_executed_here':False})
print(json.dumps({'schema':'full-score-reuse-registration-v1','registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'registered_count':len(registered),'results':registered,'liveplan_modified':False,'GPU_or_scoring_started':False},ensure_ascii=False))
'''
run=subprocess.run(['ssh','@@BENCHMARK_SSH_HOST@@','@@BENCHMARK_ROOT@@/runtime/eval_env/bin/python','-'],input='ROWS='+repr(report['rows'])+'\n'+CODE,capture_output=True,text=True)
if run.returncode:raise RuntimeError(run.stderr)
value=json.loads(run.stdout)
(HERE/'full_score_reuse_registered.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'registered':value['registered_count'],'scored_cases':[{'row':r['row_id'],'predictions':r['prediction_cases'],'scored':r['scored_cases'],'excluded':r['excluded_sample_ids']} for r in value['results'] if r['excluded_sample_ids']],'std_available':sum(r['std_available'] for r in value['results'])},ensure_ascii=False))
