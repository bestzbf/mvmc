import copy
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('world_fresh',Path(__file__).with_name('world_subset_score.py'))
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


@pytest.fixture
def sample(tmp_path,monkeypatch):
    adapter=tmp_path/'predict_relift.py';adapter.write_text('# fixture adapter; no inference\n')
    method='ReLift+AllTracker+DA3';monkeypatch.setitem(w.PINS,method,w.sha(adapter))
    checkpoint={'kind':'author_rgb_only_native_baseline','method':method,'sha256':'c'*64,'native_multicamera4rc_loading':False}
    cases=[{'id':f'adt_mini/video{i}.npz','name':f'video{i}.npz','subset':'adt_mini','input':f'inputs/{i}.npz',
        'input_sha256':str(i)*64,'frames':1,'queries':4,'views':[0],'height':2,'width':2,'sources':[{'path':f'/original/video{i}.npz','sha256':'d'*64}]} for i in [1,2]]
    manifest={'benchmark':'worldtrack','input_condition':'rgb_only_dense_first_frame','coordinates':'first_camera_world',
        'official_sources':{'commit':'pinned'},'protocol':{'task':'trajectory','subsets':['adt_mini'],'frames':'first64','release_binding':{'coverage_verified':True}},'cases':cases}
    source=tmp_path/'original.json';write(source,manifest)
    original={'id':'released_job','method':method,'manifest':str(source),'expected_baseline_checkpoint':checkpoint,
        'predict_argv':['python',str(adapter)],'checkpoint_artifacts':[{'path':str(adapter),'sha256':w.sha(adapter)}]}
    sourceplan=tmp_path/'plan.json';write(sourceplan,{'jobs':[original]})
    target=tmp_path/'target.json';t=copy.deepcopy(manifest);t['protocol']['release_binding']={'coverage_verified':False};write(target,t)
    job={'schema':'world-fresh-subset-job-v1','source_plan':str(sourceplan),'source_plan_sha256':w.sha(sourceplan),
        'source_job_id':original['id'],'source_job_sha256':w.canonical(original),'source_manifest':str(source),'source_manifest_sha256':w.sha(source),
        'target_manifest':str(target),'target_manifest_sha256':w.sha(target),'sample_ids':[c['id'] for c in cases],
        'subset':'adt_mini','expected_checkpoint':checkpoint,'expected_inference_settings':w.expected_settings(original,'trajectory'),'prediction_sources':[]}
    producers=[]
    for i,case in enumerate(cases):
        directory=tmp_path/f'producer{i}';mf=directory/'manifest.json';pm=copy.deepcopy(t);pc=copy.deepcopy(case);pc['input']=f'inputs/copied{i}.npz';pm['cases']=[pc];write(mf,pm)
        state={'schema':'released-baseline-run-v1','method':method,'manifest_sha256':w.sha(mf),'checkpoint':checkpoint,'inference_settings':job['expected_inference_settings']}
        pred=directory/'predictions';statepath=pred/'run-state.json';write(statepath,state)
        payload=pred/'000000.npz';payload.write_bytes(b'fixture-only; load_sources never opens this payload')
        record={'id':case['id'],'file':payload.name,'sha256':w.sha(payload),'input_sha256':case['input_sha256']}
        index={'schema':'released-tracking-predictions-v1','manifest_sha256':w.sha(mf),'checkpoint':checkpoint,'method':method,
            'run_state_sha256':w.canonical(state),'target_time_policy':job['expected_inference_settings']['target_time_policy'],
            'prediction_kind':'trajectory','metric_support':{'position':True,'AJ':False,'OA':False},'cases':[record]}
        ip=pred/'predictions.json';write(ip,index)
        receipt={'schema':'released-prediction-case-v1','record':record,'run_state_sha256':index['run_state_sha256'],'case_sha256':w.canonical(pc)}
        rp=payload.with_suffix('.receipt.json');write(rp,receipt)
        job['prediction_sources'].append({'source_kind':'existing_verified_cache' if i==0 else 'fresh_producer_output',
            'manifest':str(mf),'manifest_sha256':w.sha(mf),'prediction_index':str(ip),'prediction_index_sha256':None})
        producers.append({'mf':mf,'state':statepath,'index':ip,'receipt':rp,'payload':payload})
    return job,producers


def test_real_two_producer_metadata_merge_without_synthetic_state(sample):
    job,producers=sample;_,_,_,rows,evidence=w.load_sources(job)
    assert [r['case']['id'] for r in rows]==job['sample_ids']
    assert len(evidence)==2 and all(e['no_synthetic_producer_state'] for e in evidence)
    assert evidence[0]['actual_run_state_sha256']==w.sha(producers[0]['state'])


@pytest.mark.parametrize('change',['settings','checkpoint','duplicate','path_escape','case_hash','run_state_hash','input_sha','missing_case','source_plan','source_manifest','producer_scope'])
def test_identity_receipt_settings_and_merge_fail_closed(sample,change):
    job,p=sample;first=p[0]
    if change in ['settings','checkpoint','run_state_hash']:
        state=w.read(first['state'])
        if change=='settings':state['inference_settings']['corrector']['seed']=9
        elif change=='checkpoint':state['checkpoint']['sha256']='e'*64
        else:state['new_field']='changed after receipt'
        write(first['state'],state)
        if change!='run_state_hash':
            index=w.read(first['index']);index['run_state_sha256']=w.canonical(state)
            if change=='checkpoint':index['checkpoint']=state['checkpoint']
            write(first['index'],index);receipt=w.read(first['receipt']);receipt['run_state_sha256']=index['run_state_sha256'];write(first['receipt'],receipt)
    elif change=='duplicate':job['prediction_sources'].append(copy.deepcopy(job['prediction_sources'][0]))
    elif change=='path_escape':
        outside=first['payload'].parent.parent/'outside.npz';outside.write_bytes(b'outside')
        index=w.read(first['index']);index['cases'][0]['file']='../outside.npz';write(first['index'],index)
    elif change=='case_hash':
        receipt=w.read(first['receipt']);receipt['case_sha256']='f'*64;write(first['receipt'],receipt)
    elif change=='input_sha':
        index=w.read(first['index']);index['cases'][0]['input_sha256']='0'*64;write(first['index'],index)
        receipt=w.read(first['receipt']);receipt['record']=index['cases'][0];write(first['receipt'],receipt)
    elif change=='missing_case':job['prediction_sources']=job['prediction_sources'][:1]
    elif change=='source_plan':Path(job['source_plan']).write_text(Path(job['source_plan']).read_text()+' ')
    elif change=='source_manifest':Path(job['source_manifest']).write_text(Path(job['source_manifest']).read_text()+' ')
    else:
        mf=w.read(first['mf']);mf['protocol']['subsets']=['unknown'];write(first['mf'],mf);job['prediction_sources'][0]['manifest_sha256']=w.sha(first['mf'])
    with pytest.raises(ValueError):w.load_sources(job)
