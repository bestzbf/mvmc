"""CPU-only metadata/coverage guard tests; never imports a model or launches inference."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


v = load(ROOT/'score_full_v2.py', 'full_v2_tests')
w = load(ROOT/'world_full_consumer.py', 'world_full_tests')
old = load(ROOT.parents[1]/'eval_10h_20261104/scripts/test_world_subset_score.py', 'world_fixture_tests')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(old, 'w', w)
    job, producers = old.sample.__wrapped__(tmp_path, monkeypatch)
    job['method'] = 'ReLift+AllTracker+DA3'
    state = w.read(producers[0]['state'])
    job['expected_group_signature_sha256'] = w.canonical({k:x for k,x in state.items() if k != 'manifest_sha256'})
    job['output_dir'] = str(tmp_path/'raw')
    bridge = tmp_path/'bridge.py'; bridge.write_text('# fixture bridge\n')
    wrapper = tmp_path/'original_wrapper.py'; wrapper.write_text('# original wrapper\n')
    job['native_grid_bridge'] = str(bridge)
    job['native_grid_bridge_sha256'] = v.sha(bridge)
    job['expected_wrapper_binding'] = {'original_wrapper':str(wrapper), 'original_wrapper_sha256':v.sha(wrapper), 'fixture':True}
    for spec, ident in zip(job['prediction_sources'], job['sample_ids']):
        spec['case_ids'] = [ident]
    config_path = tmp_path/'job.json'
    write(config_path, job)
    original = w.read(job['source_manifest'])
    recipe = {'schema':'frozen-full-score-recipe-v2', 'method':job['method'], 'dataset':'fixture',
        'source_manifest':job['source_manifest'], 'source_manifest_sha256':job['source_manifest_sha256'],
        'source_subset':job['subset'], 'expected_case_ids':job['sample_ids'], 'expected_cases':2,
        'consumer':{'kind':'world','path':str((ROOT/'world_full_consumer.py').resolve()),'sha256':v.sha(ROOT/'world_full_consumer.py')},
        'consumer_config':{'path':str(config_path),'sha256':v.sha(config_path)},
        'expected_checkpoint_canonical_sha256':v.canonical(job['expected_checkpoint']),
        'expected_group_signature_sha256':job['expected_group_signature_sha256'],
        'expected_producer_method':job['method'],
        'score_argv':[sys.executable,str((ROOT/'world_full_consumer.py').resolve()),'--job',str(config_path)],
        'raw_output':str(tmp_path/'raw/score.json')}
    real_module = v.module
    monkeypatch.setattr(v, 'module', lambda path, name: w if Path(path).resolve() == (ROOT/'world_full_consumer.py').resolve() else real_module(path,name))
    cp, bindings, evidence = v.producer_bindings(recipe, job, original, job['sample_ids'], ROOT/'world_full_consumer.py')
    raw = {'schema':'world-catalog-score-v1','scorer_sha256':recipe['consumer']['sha256'],
        'source_manifest_sha256':recipe['source_manifest_sha256'],'sample_ids':job['sample_ids'],
        'per_case':[{'id':ident,'metrics':{'fixture':1.0}} for ident in job['sample_ids']],
        'checkpoint':cp,'official_sources':original['official_sources'],'formal_score_admitted':False,
        'job_sha256':v.sha(config_path),'job':job,'method':job['method'],
        'inference_settings':job['expected_inference_settings'], 'actual_producer_sources':evidence,
        'native_grid_wrapper':job['expected_wrapper_binding'], 'metric_protocol':w.metric_protocol(original),
        'case_bindings':bindings,'metrics':{'fixture':1.0}}
    raw_path = Path(recipe['raw_output'])
    write(raw_path, raw)
    recipe['raw_output_sha256'] = v.sha(raw_path)
    recipe_path = tmp_path/'recipe.json'
    write(recipe_path, recipe)
    return job, producers, recipe, recipe_path, raw


def test_complete_guard_preserves_producer_and_admission(fixture, tmp_path):
    job, producers, recipe, rp, raw = fixture
    before = [p['index'].read_bytes() for p in producers]
    result = v.score(rp, tmp_path/'verified.json')
    report = v.read(result['output'])
    assert report['actual_prediction_cases'] == report['actual_scored_cases'] == 2
    assert report['formal_score_admitted'] is False
    assert report['actual_producer_sources'] == raw['actual_producer_sources']
    assert before == [p['index'].read_bytes() for p in producers]
    assert report['v2_integrity']['producer_index_or_state_synthesized'] is False


@pytest.mark.parametrize('change', ['coverage','order','source_hash','config_hash','consumer_hash','raw_hash','raw_checkpoint','raw_job','raw_bindings','argv','producer_method','unproved_exclusion','output_overwrite'])
def test_wrapper_fail_closed(fixture, tmp_path, change):
    job, producers, recipe, rp, raw = fixture
    output = tmp_path/'verified.json'
    if change == 'coverage':
        recipe['expected_case_ids'] = recipe['expected_case_ids'][:-1]; recipe['expected_cases'] = 1
    elif change == 'order': recipe['expected_case_ids'] = recipe['expected_case_ids'][::-1]
    elif change == 'source_hash': Path(recipe['source_manifest']).write_text(Path(recipe['source_manifest']).read_text()+' ')
    elif change == 'config_hash': Path(recipe['consumer_config']['path']).write_text(Path(recipe['consumer_config']['path']).read_text()+' ')
    elif change == 'consumer_hash': recipe['consumer']['sha256'] = '0'*64
    elif change == 'raw_hash': recipe['raw_output_sha256'] = '0'*64
    elif change == 'raw_checkpoint': raw['checkpoint'] = dict(raw['checkpoint'], sha256='f'*64)
    elif change == 'raw_job': raw['job'] = dict(raw['job'], method='wrong-model')
    elif change == 'raw_bindings': raw['case_bindings'][0]['producer_run_state_canonical_sha256'] = '0'*64
    elif change == 'argv': recipe['score_argv'][-1] = '/another/job.json'
    elif change == 'producer_method': recipe['expected_producer_method'] = 'wrong-model'
    elif change == 'unproved_exclusion':
        job['expected_eligible_ids'] = job['sample_ids'][1:]
        write(Path(recipe['consumer_config']['path']),job)
        recipe['consumer_config']['sha256'] = v.sha(recipe['consumer_config']['path'])
    else: output.write_text('preserve')
    if change in ('raw_checkpoint','raw_job','raw_bindings'):
        write(Path(recipe['raw_output']),raw)
        recipe['raw_output_sha256'] = v.sha(recipe['raw_output'])
    write(rp,recipe)
    with pytest.raises(ValueError): v.score(rp,output)


@pytest.mark.parametrize('change', ['missing_case','overlap','different_signature','payload_hash','source_plan_hash','source_case_order'])
def test_world_sources_fail_closed(fixture, change):
    job, p, recipe, rp, raw = fixture
    if change == 'missing_case': job['prediction_sources'] = job['prediction_sources'][:1]
    elif change == 'overlap': job['prediction_sources'].append(copy.deepcopy(job['prediction_sources'][0]))
    elif change == 'different_signature':
        state = w.read(p[0]['state']); state['runtime_changed'] = True; write(p[0]['state'],state)
        index = w.read(p[0]['index']); index['run_state_sha256'] = w.canonical(state); write(p[0]['index'],index)
        receipt = w.read(p[0]['receipt']); receipt['run_state_sha256'] = index['run_state_sha256']; write(p[0]['receipt'],receipt)
    elif change == 'payload_hash': p[0]['payload'].write_bytes(b'tampered')
    elif change == 'source_plan_hash': Path(job['source_plan']).write_text(Path(job['source_plan']).read_text()+' ')
    else: job['sample_ids'] = job['sample_ids'][::-1]
    with pytest.raises(ValueError):
        v.producer_bindings(recipe,job,w.read(job['source_manifest']),job['sample_ids'],ROOT/'world_full_consumer.py')


def test_disjoint_source_case_partition_is_accepted(fixture):
    job, p, recipe, rp, raw = fixture
    _,_,_,rows,evidence = w.load_sources(job)
    assert [r['case']['id'] for r in rows] == job['sample_ids']
    assert len(evidence) == 2 and all(r['source_kind'] for r in evidence)


def test_existing_raw_requires_explicit_hash(fixture, tmp_path):
    job,p,recipe,rp,raw=fixture
    recipe.pop('raw_output_sha256')
    write(rp,recipe)
    with pytest.raises(ValueError,match='Pinned file SHA'):
        v.score(rp,tmp_path/'verified.json')


def test_commit_race_never_overwrites_another_result(fixture, tmp_path, monkeypatch):
    job,p,recipe,rp,raw = fixture
    output = tmp_path/'verified.json'
    link = v.os.link
    def race(source, destination):
        output.write_text('concurrent-writer-result')
        return link(source,destination)
    monkeypatch.setattr(v.os,'link',race)
    with pytest.raises(FileExistsError):
        v.score(rp,output)
    assert output.read_text() == 'concurrent-writer-result'
    assert not list(tmp_path.glob('verified.json.tmp.*'))


@pytest.mark.parametrize('kind', ['tap','mv'])
def test_real_tap_cache_helper_multi_index_partition_and_dependency(tmp_path, monkeypatch, kind):
    oldroot = ROOT.parents[1]/'eval_10h_20261104/tools'
    cache = load(oldroot/'tap_subset_score.py', 'cache_fixture_helper')
    monkeypatch.setitem(sys.modules, 'tap_subset_score', cache)
    taps = load(oldroot/'test_tap_subset_score.py', 'tap_fixture_helper')
    checkpoint,cases,original,make = taps.caches.__wrapped__(tmp_path)
    if kind == 'mv':
        original.update(benchmark='mvtracker',input_condition='rgb_only_3d_query_anchor_sim3',coordinates='native_predicted_world')
    a,b = make('producerA',[0,1,2]), make('producerB',[0,1,2])
    a['case_ids'] = [cases[0]['id']]
    b['case_ids'] = [c['id'] for c in cases[1:]]
    for source in (a,b):
        directory = Path(source['path']).parent
        state = cache.read(directory/'run-state.json');state['method'] = 'TestModel'
        write(directory/'run-state.json',state)
        index = cache.read(source['path']);index['run_state_sha256'] = cache.canonical(state)
        write(Path(source['path']),index);source['index_sha256'] = cache.sha(source['path'])
        for receiptpath in directory.glob('*.receipt.json'):
            receipt = cache.read(receiptpath);receipt['run_state_sha256'] = index['run_state_sha256'];write(receiptpath,receipt)
    sourcepath = tmp_path/'canonical.json';write(sourcepath,original)
    config = {'schema':'eval-10h-'+kind+'-subset-score-spec-v1','manifest':str(sourcepath),'manifest_sha256':cache.sha(sourcepath),
        'source_manifest':str(sourcepath),'source_manifest_sha256':cache.sha(sourcepath),'expected_checkpoint':checkpoint,'sources':[a,b]}
    configpath = tmp_path/'spec.json';write(configpath,config)
    consumer = (oldroot/('tap_subset_score.py' if kind=='tap' else 'mv_subset_score.py')).resolve()
    recipe = {'schema':'frozen-full-score-recipe-v2','method':'TestModel','dataset':'fixture','source_manifest':str(sourcepath),
        'source_manifest_sha256':cache.sha(sourcepath),'expected_case_ids':[c['id'] for c in cases],'expected_cases':3,
        'consumer':{'kind':kind,'path':str(consumer),'sha256':cache.sha(consumer)},
        'consumer_config':{'path':str(configpath),'sha256':cache.sha(configpath)},'expected_checkpoint_canonical_sha256':cache.canonical(checkpoint),
        'expected_producer_method':'TestModel','expected_group_signature_sha256':cache.canonical({k:x for k,x in state.items() if k!='manifest_sha256'}),
        'raw_output':str(tmp_path/'raw.json'),'score_argv':[sys.executable,str(consumer),'--spec',str(configpath),'--output',str(tmp_path/'raw.json')]}
    if kind == 'mv':recipe['consumer_dependencies'] = [{'path':str((oldroot/'tap_subset_score.py').resolve()),'sha256':cache.sha(oldroot/'tap_subset_score.py')}]
    cp,bindings,evidence = v.producer_bindings(recipe,config,original,recipe['expected_case_ids'],consumer)
    raw = {'schema':'eval-10h-'+kind+'-subset-scores-v1','scorer_sha256':cache.sha(consumer),'source_manifest_sha256':cache.sha(sourcepath),
        'sample_ids':recipe['expected_case_ids'],'per_case':[{'id':c['id'],'metrics':{}} for c in cases],'checkpoint':checkpoint,
        'official_sources':original['official_sources'],'formal_score_admitted':False,'subset_score_spec':str(configpath),'subset_score_spec_sha256':cache.sha(configpath),
        'shared_settings_signature':recipe['expected_group_signature_sha256'],'prediction_sources':evidence,'prediction_case_bindings':bindings,
        'protocol':original['protocol'],'input_condition':original['input_condition'],'coordinates':original['coordinates']}
    if kind == 'mv':raw['cache_helper_sha256'] = recipe['consumer_dependencies'][0]['sha256']
    write(Path(recipe['raw_output']),raw);recipe['raw_output_sha256'] = cache.sha(recipe['raw_output'])
    rp = tmp_path/'recipe.json';write(rp,recipe)
    report = v.read(v.score(rp,tmp_path/'verified.json')['output'])
    assert report['actual_prediction_cases'] == report['actual_scored_cases'] == 3
    assert len(report['prediction_sources']) == 2
    if kind == 'mv':
        recipe['consumer_dependencies'][0]['sha256'] = '0'*64;write(rp,recipe)
        with pytest.raises(ValueError,match='Pinned file SHA'):
            v.score(rp,tmp_path/'bad_dependency.json')
