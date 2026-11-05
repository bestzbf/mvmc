"""Prepare pinned full-score consumers and append CPU jobs without touching producers."""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path('@@EVAL_FULL_ROOT@@')
OLD=ROOT.parent/'eval_10h_20261104'
PY='@@BENCHMARK_ROOT@@/runtime/eval_env/bin/python'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(p,o):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
def immutable(p,o):
    text=json.dumps(o,ensure_ascii=False,indent=2)+'\n';p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_text()==text,p
    else:p.write_text(text)


def score_sources(row,prepared,expected):
    group=next(g for g in row['groups'] if g['signature_sha256']==row['preferred_group_signature_sha256'])
    checkpoint=read(group['producers'][0]['run_state'])['checkpoint']
    state=read(group['producers'][0]['run_state']);used=set();sources=[]
    for producer in group['producers']:
        allowed=[cid for cid in expected if cid in producer['eligible_case_ids'] and cid not in used]
        if not allowed:continue
        used.update(allowed)
        item=dict(kind='committed_case_receipts' if producer['receipt_only'] else 'prediction_index',
                  path=str(Path(producer['run_state']).parent) if producer['receipt_only'] else producer['prediction_index'],
                  manifest=producer['manifest'],manifest_sha256=producer['manifest_sha256'],case_ids=allowed)
        if producer['index_exists']:item['index_sha256']=producer['prediction_index_sha256']
        sources.append(item)
    for batch in prepared['gpu_job_candidates']:
        allowed=[cid for cid in batch['case_ids'] if cid not in used]
        if not allowed:continue
        used.update(allowed)
        output=batch['argv'][batch['argv'].index('--output')+1]
        sources.append(dict(kind='prediction_index',path=str(Path(output)/'predictions.json'),
                            manifest=batch['manifest'],manifest_sha256=batch['manifest_sha256'],case_ids=allowed))
    assert used==set(expected),(row['id'],set(expected)-used)
    return checkpoint,state,sources


def make_recipe(row,panel,prepared):
    source=Path(panel['canonical_manifest']);source_data=read(source)
    expected=[c['id'] for c in panel['cases']]
    checkpoint,state,sources=score_sources(row,prepared,expected)
    kind={'tapvid3d':'tap','mvtracker':'mv','worldtrack':'world'}[panel['benchmark']]
    ident=row['id'];configpath=ROOT/'full_score_configs'/(ident+'.json')
    raw=ROOT/'scores/raw'/('full_'+ident+'.json')
    eligible=expected
    if kind in ('tap','mv'):
        config=dict(schema='eval-10h-'+kind+'-subset-score-spec-v1',manifest=str(source),manifest_sha256=sha(source),
                    source_manifest=str(source),source_manifest_sha256=sha(source),expected_checkpoint=checkpoint,
                    sources=sources,selection_policy='Every original frozen case; old and missing-only producers retained individually')
        consumer=OLD/'tools'/('tap_subset_score.py' if kind=='tap' else 'mv_subset_score.py')
        immutable(configpath,config)
        argv=[PY,str(consumer),'--spec',str(configpath),'--output',str(raw)]
    else:
        key='uniquery4r' if row['method']=='UniQuery4R' else 'relift'
        template=read(OLD/'world'/key/row['panel_id']/'score_job.json')
        target=copy.deepcopy(source_data);target['cases']=[copy.deepcopy(c) for c in source_data['cases'] if c['id'] in set(expected)]
        for c in target['cases']:
            if not Path(c['input']).is_absolute():c['input']=str(source.parent/c['input'])
        target['consumer_only_selection']=True
        targetpath=ROOT/'full_score_manifests'/(row['panel_id']+'.json');immutable(targetpath,target)
        config=copy.deepcopy(template)
        config.update(target_manifest=str(targetpath),target_manifest_sha256=sha(targetpath),sample_ids=expected,
                      expected_checkpoint=checkpoint,expected_inference_settings=group_settings(row),
                      expected_group_signature_sha256=row['preferred_group_signature_sha256'],
                      output_dir=str(raw.parent/('full_'+ident)),comparison_references=[])
        subset=target['cases'][0]['subset'];config['subset']=subset
        converted=[]
        for item in sources:
            assert item['kind']=='prediction_index','World sources must be their authentic complete producer indexes'
            entry=dict(source_kind='real_independent_native_producer',manifest=item['manifest'],manifest_sha256=item['manifest_sha256'],
                       prediction_index=item['path'],case_ids=item['case_ids'])
            if item.get('index_sha256'):entry['prediction_index_sha256']=item['index_sha256']
            converted.append(entry)
        config['prediction_sources']=converted
        if row['panel_id']=='trajectory_adt_mini':
            proofpath=ROOT/'planning/world_eligibility.json';proof=read(proofpath)
            excluded={e['id'] for e in proof['excluded_cases']}
            eligible=[cid for cid in expected if cid not in excluded]
            config['eligibility_proof']=dict(path=str(proofpath),sha256=sha(proofpath))
        config['expected_eligible_ids']=eligible
        immutable(configpath,config)
        consumer=ROOT/'scripts/world_full_consumer.py';raw=Path(config['output_dir'])/'score.json'
        argv=[PY,str(consumer),'--job',str(configpath)]
    recipe=dict(schema='frozen-full-score-recipe-v2',method=row['method'],dataset=row['panel_id'],
                source_manifest=str(source),source_manifest_sha256=sha(source),
                expected_case_ids=expected,expected_cases=len(expected),
                consumer=dict(kind=kind,path=str(consumer),sha256=sha(consumer)),
                consumer_config=dict(path=str(configpath),sha256=sha(configpath)),
                expected_checkpoint_canonical_sha256=canonical(checkpoint),
                expected_group_signature_sha256=row['preferred_group_signature_sha256'],
                expected_producer_method=state.get('method'),score_argv=argv,raw_output=str(raw),raw_output_sha256=None)
    if kind=='mv':recipe['consumer_dependencies']=[dict(path=str(consumer.parent/'tap_subset_score.py'),sha256=sha(consumer.parent/'tap_subset_score.py'))]
    if kind=='world':
        recipe['source_subset']=config['subset']
        if config.get('eligibility_proof'):recipe['eligibility_proof']=config['eligibility_proof']
    return recipe,eligible


def group_settings(row):
    return next(g['inference_settings'] for g in row['groups'] if g['signature_sha256']==row['preferred_group_signature_sha256'])


def bootstrap_recipe(row,panel):
    key='arrow_mv' if row['method']=='ARROW' else 'trackeverything_mv'
    oldcpu='panoptic_'+key+'_full6_score';statuspath=ROOT/'status'/(oldcpu+'.json')
    if not statuspath.exists() or read(statuspath)['state'] not in ['complete','failed']:return None
    specpath=ROOT/'score_specs'/('panoptic_'+key+'_full6.json');spec=read(specpath)
    raw=ROOT/'scores/raw'/('panoptic_'+key+'_full6.json')
    if not raw.exists():return None
    state=read(next(g for g in row['groups'] if g['signature_sha256']==row['preferred_group_signature_sha256'])['producers'][0]['run_state'])
    consumer=OLD/'tools/mv_subset_score.py'
    recipe=dict(schema='frozen-full-score-recipe-v2',method=row['method'],dataset='panoptic',
                source_manifest=panel['canonical_manifest'],source_manifest_sha256=panel['canonical_manifest_sha256'],
                expected_case_ids=[c['id'] for c in panel['cases']],expected_cases=6,
                consumer=dict(kind='mv',path=str(consumer),sha256=sha(consumer)),
                consumer_config=dict(path=str(specpath),sha256=sha(specpath)),
                expected_checkpoint_canonical_sha256=canonical(spec['expected_checkpoint']),
                expected_group_signature_sha256=row['preferred_group_signature_sha256'],expected_producer_method=state.get('method'),
                consumer_dependencies=[dict(path=str(consumer.parent/'tap_subset_score.py'),sha256=sha(consumer.parent/'tap_subset_score.py'))],
                score_argv=[PY,str(consumer),'--spec',str(specpath),'--output',str(raw)],raw_output=str(raw),raw_output_sha256=sha(raw))
    return recipe


def main():
    inventory=read(ROOT/'planning/full_inventory.json')
    reused={r['row_id']:r for r in read(ROOT/'full_score_reuse_registered.json')['results']}
    targets={};pending=[];newjobs=[];plan=read(ROOT/'queue_plan.json');existing={j['id'] for j in plan['jobs']}
    for row in inventory['rows']:
        panel=inventory['canonical_panels'][row['panel_id']]
        entry=dict(row_id=row['id'],method=row['method'],panel_id=row['panel_id'],benchmark=panel['benchmark'],
                   total_prediction_cases=row['total'],cached_at_start=row['existing'],source_manifest=panel['canonical_manifest'],
                   source_manifest_sha256=panel['canonical_manifest_sha256'])
        if row['id'] in reused:
            entry.update(score_path=reused[row['id']]['path'],score_sha256=reused[row['id']]['sha256'],reused_full_score=True)
            targets[row['id']]=entry;continue
        if row['panel_id']=='panoptic':
            recipe=bootstrap_recipe(row,panel)
            if recipe is None:pending.append(row['id']);targets[row['id']]=entry;continue
            deps=['panoptic_'+('arrow_mv' if row['method']=='ARROW' else 'trackeverything_mv')+'_fill_predict']
        else:
            preparedpath=ROOT/'specs'/(row['id']+'.json')
            if not preparedpath.exists():pending.append(row['id']);targets[row['id']]=entry;continue
            prepared=read(preparedpath);recipe,_=make_recipe(row,panel,prepared)
            deps=[j['id'] for j in prepared['gpu_job_candidates']]
        recipepath=ROOT/'full_score_recipes'/(row['id']+'.json');immutable(recipepath,recipe)
        output=ROOT/'scores'/('full_verified_'+row['id']+'.json');ident='score_full_'+row['id']
        entry.update(score_path=str(output),recipe=str(recipepath),reused_full_score=False,score_job_id=ident)
        targets[row['id']]=entry
        if ident not in existing:
            newjobs.append(dict(id=ident,kind='cpu',method=row['method'],dataset=row['panel_id'],
                                argv=[PY,str(ROOT/'scripts/score_full_v2.py'),'--recipe',str(recipepath),'--output',str(output)],
                                env={},timeout_seconds=10800,depends_on=deps,expected_outputs=[str(output)],score_path=str(output),
                                case_count=row['total'],priority=1,recipe_sha256=sha(recipepath)))
    with (ROOT/'queue_plan.append.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);plan=read(ROOT/'queue_plan.json');known={j['id'] for j in plan['jobs']}
        plan['jobs'] += [j for j in newjobs if j['id'] not in known]
        # Seal only after every full comparison has a verified reuse result or a score job.
        if not pending:
            needed={j['id'] for row in inventory['rows'] if row['missing'] and row['panel_id']!='panoptic'
                    for j in read(ROOT/'specs'/(row['id']+'.json'))['gpu_job_candidates']}
            if needed <= {j['id'] for j in plan['jobs']}:plan['sealed']=True
        save(ROOT/'queue_plan.json',plan)
    save(ROOT/'full_targets.json',dict(rows=list(targets.values()),target_method_panels=40,pending_configuration=pending,
                                      planned_new_GPU_cases=inventory['total_missing_GPU_cases'],new_jobs_added=len(newjobs),plan_sealed=plan['sealed']))
    return pending,plan['sealed']


if __name__=='__main__':
    while True:
        waiting,sealed=main()
        if sealed:break
        time.sleep(15)
