import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

spec=importlib.util.spec_from_file_location('builder',Path(__file__).with_name('build_ours_drive_recovery_score.py'))
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)


def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b.encoded(value))


@pytest.fixture
def ready(tmp_path,monkeypatch):
    root=tmp_path;verifier=root/'scripts/score_full_v2.py';verifier.parent.mkdir();verifier.write_text('# frozen verifier fixture\n')
    router=root/'router.py';router.write_text('# frozen route fixture\n');consumer=root/'tap_consumer.py';consumer.write_text('# no score execution\n')
    monkeypatch.setattr(b,'VERIFIER_SHA256',b.sha(verifier));monkeypatch.setattr(b,'ROUTER_SHA256',b.sha(router))
    cp={'kind':'native_fixture','sha256':'c'*64,'native_loading':{'checkpoint_step':134400}}
    statebase={'schema':'released-prediction-run-v1','checkpoint':cp,'target_time_chunk':8,'runtime_versions':{'torch':'frozen'},'numerics':{'threads':8}}
    cases=[{'id':f'drivetrack/case{i:02d}.npz','input':f'inputs/{i:06d}.npz','input_sha256':f'{i:064x}','frames':2,'queries':3,'views':[0]} for i in range(50)]
    original={'schema':'released-tracking-benchmark-v1','benchmark':'tapvid3d','protocol':{'subsets':['drivetrack'],'scaling':'median'},'official_sources':{'commit':'frozen'},'input_condition':'rgb_only','coordinates':'per_frame_camera','cases':cases}
    source=root/'canonical.json';write(source,original)
    def producer(folder,selection,committed,final=False,canonical_manifest=False):
        mf=source if canonical_manifest else folder/'manifest.json'
        manifest=copy.deepcopy(original);manifest['cases']=[copy.deepcopy(cases[i]) for i in selection]
        if not canonical_manifest:write(mf,manifest)
        state=dict(statebase,manifest_sha256=b.sha(mf));pred=folder/'predictions';write(pred/'run-state.json',state);records=[]
        for i,case in enumerate(manifest['cases']):
            if selection[i] not in committed:continue
            payload=pred/f'{i:06d}.npz';payload.write_bytes(b'opaque fixture payload, never decoded')
            record={'id':case['id'],'file':payload.name,'sha256':b.sha(payload),'input_sha256':case['input_sha256']};records.append(record)
            write(payload.with_suffix('.receipt.json'),{'schema':'released-prediction-case-v1','record':record,'run_state_sha256':b.canonical(state),'case_sha256':b.canonical(case),'arrays':{'tracks_XYZ':{'shape':[2,3,3],'dtype':'<f4'},'visibility':{'shape':[2,3],'dtype':'|b1'}}})
        if final:write(pred/'predictions.json',{'schema':'released-tracking-predictions-v1','manifest_sha256':b.sha(mf),'run_state_sha256':b.canonical(state),'checkpoint':cp,'cases':records})
        return mf,pred
    cached_mf,cached_root=producer(root/'cache20',list(range(50)),list(range(20)),canonical_manifest=True)
    sources=[{'kind':'committed_case_receipts','path':str(cached_root),'manifest':str(cached_mf),'manifest_sha256':b.sha(cached_mf),'case_ids':[c['id'] for c in cases[:20]]}]
    originals=[]
    for n in range(6):
        selection=list(range(20+n*5,25+n*5));committed=selection[:2] if n==0 else selection
        mf,pred=producer(root/f'original{n}',selection,committed,final=n!=0)
        job={'id':f'original{n}','kind':'gpu','method':'Ours134400','dataset':'drivetrack_minival','argv':[sys.executable,str(router),'--routing-receipt',str(mf.parent/'route.json'),'--','predict','--manifest',str(mf),'--output',str(pred),'--resume','--target-time-chunk','8','--expected-step','134400','--require-refinement'],'env':{'OMP_NUM_THREADS':'8'},'case_count':5,'expected_outputs':[str(pred/'predictions.json')]};originals.append(job)
        write(root/'status'/(job['id']+'.json'),{'job_hash':b.job_hash(job),'state':'failed' if n==0 else 'complete'})
        sources.append({'kind':'prediction_index','path':str(pred/'predictions.json'),'manifest':str(mf),'manifest_sha256':b.sha(mf),'case_ids':[cases[i]['id'] for i in selection]})
    recoveries=[]
    for i in [22,23,24]:
        mf,pred=producer(root/f'recovery{i}',[i],[i],final=True);job=copy.deepcopy(originals[0]);job.update(id=f'recovery{i}',oom_recovery_of=originals[0]['id'],case_count=1,expected_outputs=[str(pred/'predictions.json')]);job['env']['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
        for flag,path in [('--manifest',mf),('--output',pred),('--routing-receipt',mf.parent/'route.json')]:job['argv'][job['argv'].index(flag)+1]=str(path)
        recoveries.append(job);write(root/'status'/(job['id']+'.json'),{'job_hash':b.job_hash(job),'state':'complete'})
    config=root/'old_spec.json';write(config,{'schema':'eval-10h-tap-subset-score-spec-v1','manifest':str(source),'manifest_sha256':b.sha(source),'source_manifest':str(source),'source_manifest_sha256':b.sha(source),'expected_checkpoint':cp,'sources':sources})
    recipepath=root/'old_recipe.json';recipe={'schema':'frozen-full-score-recipe-v2','method':'Ours134400','dataset':'drivetrack_minival','source_manifest':str(source),'source_manifest_sha256':b.sha(source),'expected_cases':50,'expected_case_ids':[c['id'] for c in cases],'expected_checkpoint_canonical_sha256':b.canonical(cp),'expected_group_signature_sha256':b.canonical(statebase),'expected_producer_method':None,'consumer':{'kind':'tap','path':str(consumer),'sha256':b.sha(consumer)},'consumer_config':{'path':str(config),'sha256':b.sha(config)},'score_argv':[sys.executable,str(consumer),'--spec',str(config),'--output',str(root/'old_raw.json')],'raw_output':str(root/'old_raw.json')};write(recipepath,recipe);monkeypatch.setattr(b,'BASE_RECIPE_SHA256',b.sha(recipepath))
    cpu={'id':b.ORIGINAL_SCORE_ID,'kind':'cpu','method':'Ours134400','dataset':'drivetrack_minival','argv':[sys.executable,str(verifier),'--recipe',str(recipepath),'--output',str(root/'old_verified.json')],'env':{},'depends_on':[j['id'] for j in originals],'expected_outputs':[str(root/'old_verified.json')],'score_path':str(root/'old_verified.json'),'recipe_sha256':b.sha(recipepath),'case_count':50,'timeout_seconds':21600}
    write(root/'queue_plan.json',{'jobs':[*originals,*recoveries,cpu]});write(root/'full_targets.json',{'rows':[{'row_id':b.ROW_ID,'score_job_id':cpu['id'],'score_path':cpu['score_path'],'recipe':str(recipepath),'recipe_sha256':b.sha(recipepath),'total_prediction_cases':50}]})
    return root,originals,recoveries


def test_closed_native_metadata_partitions_exact50_without_queue_or_target_mutation(ready):
    root,originals,recoveries=ready;before={p:p.read_bytes() for p in [root/'queue_plan.json',root/'full_targets.json',*list((root/'status').glob('*.json'))]}
    result=b.build(root,originals,recoveries,write=True)
    assert result['source_partition_case_count']==50
    selected=[i for s in result['spec']['sources'] for i in s['case_ids']]
    assert len(selected)==len(set(selected))==50
    assert result['job']['depends_on']==[] and result['job']['id']==b.NEW_SCORE_ID
    assert next(s for s in result['spec']['sources'] if 'original0/' in s['manifest'])['kind']=='committed_case_receipts'
    assert all(p.read_bytes()==data for p,data in before.items())
    assert b.read(result['recipe']['consumer_config']['path'])==result['spec']
    assert result['recipe']['consumer_config']['sha256']==b.sha(result['recipe']['consumer_config']['path'])
    assert result['job']['recipe_sha256']==result['target_update']['recipe_sha256']==b.byte_hash(result['recipe'])
    assert result['job']['recipe_sha256']!=b.read(root/'full_targets.json')['rows'][0]['recipe_sha256']
    assert result['target_update']['score_job_id']==b.NEW_SCORE_ID


def test_active_batch_not_ready(ready):
    root,originals,recoveries=ready;p=root/'status/original1.json';s=b.read(p);s['state']='running';write(p,s)
    with pytest.raises(b.NotReady,match='not terminal'):b.build(root,originals,recoveries,False)


def test_missing_uncommitted_case_never_becomes_full50(ready):
    root,originals,recoveries=ready;pred=Path(b.arg(recoveries[-1],'--output'));(pred/'000000.receipt.json').unlink();(pred/'predictions.json').unlink()
    with pytest.raises(b.NotReady,match='coverage incomplete'):b.build(root,originals,recoveries,False)


@pytest.mark.parametrize('change',['chunk','nonallocator_env','same_output'])
def test_recovery_only_io_and_allocator_changes(ready,change):
    root,originals,recoveries=ready;r=copy.deepcopy(recoveries[0])
    if change=='chunk':r['argv'][r['argv'].index('--target-time-chunk')+1]='1'
    elif change=='nonallocator_env':r['env']['CUDA_VISIBLE_DEVICES']='0'
    else:r['argv'][r['argv'].index('--output')+1]=b.arg(originals[0],'--output')
    with pytest.raises(ValueError):b.validate_recovery(originals[0],r)


def test_allocator_state_field_difference_is_not_normalized_away(ready):
    root,originals,recoveries=ready;pred=Path(b.arg(recoveries[0],'--output'));s=b.read(pred/'run-state.json');s['new_allocator_diagnostic']='different';write(pred/'run-state.json',s)
    with pytest.raises(ValueError,match='signature differs'):b.build(root,originals,recoveries,False)


def test_duplicate_recovery_and_wrong_receipt_shape_fail_closed(ready):
    root,originals,recoveries=ready
    with pytest.raises(ValueError,match='Duplicate recovery'):b.build(root,originals,recoveries+recoveries[:1],False)
    p=Path(b.arg(recoveries[0],'--output'))/'000000.receipt.json';r=b.read(p);r['arrays']['tracks_XYZ']['shape']=[1,3,3];write(p,r)
    with pytest.raises(ValueError,match='dimensions/dtypes'):b.build(root,originals,recoveries,False)
