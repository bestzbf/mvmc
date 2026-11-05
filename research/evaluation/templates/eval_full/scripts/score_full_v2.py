#!/usr/bin/env python3
"""Verify complete frozen coverage and original producer bindings around pinned CPU scorers."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ELIGIBILITY_SCORE_SHA = 'e004ca42dc918200927d0f92b9e164ecc6f9c4362981eeb464248fc05361a510'
ELIGIBILITY_MANIFEST_SHA = '388b05bd5c800ad73db099041be7351b6ef079da1afcd9419f7c802cf041f3a4'
ELIGIBILITY_GT_SHA = 'dd63be7433874d30d66ebef4ac50c98412f8e120d8c135b2a4eb5ca8480f63d6'
EXCLUDED_ID = '@@BENCHMARK_CASE_4BC7D2E490@@'
EXCLUDED_REASON = 'official_no_first_frame_visible_points'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def pin(path, checksum):
    path = Path(path).resolve(strict=True)
    require(isinstance(checksum, str) and len(checksum) == 64 and sha(path) == checksum, 'Pinned file SHA differs: '+str(path))
    return path


def same_case(case, original):
    case = copy.deepcopy(case)
    case['input'] = original['input']
    require(case == original, 'Canonical case changed beyond input path: '+str(original['id']))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def eligible_ids(recipe, config, original, expected):
    proof_spec = recipe.get('eligibility_proof')
    if not proof_spec:
        require(EXCLUDED_ID not in expected, 'ADT official ineligible case requires the original eligibility proof')
        require(config.get('expected_eligible_ids', expected) == expected, 'Unproved score exclusion is forbidden')
        return expected, [], None
    require(recipe['consumer']['kind'] == 'world', 'Eligibility exception is only for frozen WorldTrack')
    proof_path = pin(proof_spec['path'], proof_spec['sha256'])
    proof = read(proof_path)
    require(sha(recipe['source_manifest']) == ELIGIBILITY_MANIFEST_SHA, 'Eligibility belongs to another canonical source')
    require(proof['canonical_manifest_sha256'] == ELIGIBILITY_MANIFEST_SHA and proof['source_score_sha256'] == ELIGIBILITY_SCORE_SHA and proof['gt_eligibility_sha256'] == ELIGIBILITY_GT_SHA, 'Unapproved official eligibility proof')
    require(proof['official_sources'] == original['official_sources'], 'Eligibility official metric source differs')
    baseline = read(pin(proof['source_score'], ELIGIBILITY_SCORE_SHA))
    audit = baseline['official_eligibility_audit']
    all_ids = [c['id'] for c in original['cases']]
    require(audit['attempted_ids'] == all_ids and audit['gt_eligibility_sha256'] == ELIGIBILITY_GT_SHA, 'Original full official eligibility inventory differs')
    exclusion = [{'id': EXCLUDED_ID, 'reason': EXCLUDED_REASON}]
    require(audit['excluded_ids'] == exclusion and audit['scored_ids'] == [i for i in all_ids if i != EXCLUDED_ID], 'Original official exclusion differs')
    require(audit['all_predictions_validated'] is True and audit['prediction_or_input_removed'] is False and audit['official_function_bytes_changed'] is False, 'Eligibility source did not validate complete inference')
    case = original['cases'][39]
    require(case['id'] == EXCLUDED_ID and len(proof['excluded_cases']) == 1, 'Eligibility canonical index differs')
    evidence = proof['excluded_cases'][0]
    require(evidence['id'] == EXCLUDED_ID and evidence['reason'] == EXCLUDED_REASON and evidence['input_sha256'] == case['input_sha256'] and evidence['original_GT_sources'] == case['sources'] and evidence['first_frame_visible'] == 0 and evidence['first_frame_inbounds'] == 0 and evidence['scorer_eligible'] is False, 'Eligibility case/input/GT evidence differs')
    selected = [i for i in expected if i != EXCLUDED_ID]
    require(config.get('expected_eligible_ids') == selected and config.get('eligibility_proof') == proof_spec, 'Consumer eligibility is not bound to the exact proof/IDs')
    return selected, [x for x in exclusion if x['id'] in expected], {'path': str(proof_path), 'sha256': proof_spec['sha256'], 'original_full_score_sha256': ELIGIBILITY_SCORE_SHA, 'gt_eligibility_sha256': ELIGIBILITY_GT_SHA}


def validate_recipe(recipe):
    require(recipe.get('schema') == 'frozen-full-score-recipe-v2', 'Wrong full-score recipe schema')
    source = pin(recipe['source_manifest'], recipe['source_manifest_sha256'])
    original = read(source)
    cases = original['cases']
    if recipe.get('source_subset') is not None:
        cases = [c for c in cases if c['subset'] == recipe['source_subset']]
    expected = [c['id'] for c in cases]
    require(expected and len(set(expected)) == len(expected), 'Empty/duplicate canonical coverage')
    require(recipe['expected_case_ids'] == expected and recipe['expected_cases'] == len(expected), 'Complete canonical coverage/order differs')
    consumer = recipe['consumer']
    require(consumer['kind'] in ('tap', 'mv', 'world'), 'Unsupported consumer kind')
    require(original['benchmark'] == {'tap':'tapvid3d', 'mv':'mvtracker', 'world':'worldtrack'}[consumer['kind']], 'Consumer benchmark kind differs')
    consumer_path = pin(consumer['path'], consumer['sha256'])
    config_path = pin(recipe['consumer_config']['path'], recipe['consumer_config']['sha256'])
    config = read(config_path)
    require(config.get('schema') == {'tap':'eval-10h-tap-subset-score-spec-v1', 'mv':'eval-10h-mv-subset-score-spec-v1', 'world':'world-fresh-subset-job-v1'}[consumer['kind']], 'Consumer config schema differs')
    target_key = 'target_manifest' if consumer['kind'] == 'world' else 'manifest'
    target_path = pin(config[target_key], config[target_key+'_sha256'])
    target = read(target_path)
    require([c['id'] for c in target['cases']] == expected, 'Consumer target is incomplete or reordered')
    originals = {c['id']: c for c in cases}
    for case in target['cases']:
        same_case(case, originals[case['id']])
    require(Path(config['source_manifest']).resolve() == source and config['source_manifest_sha256'] == sha(source), 'Consumer source manifest binding differs')
    checkpoint = config['expected_checkpoint']
    require(canonical(checkpoint) == recipe['expected_checkpoint_canonical_sha256'], 'Recipe expected checkpoint differs from pinned config')
    require(isinstance(recipe['expected_group_signature_sha256'], str) and len(recipe['expected_group_signature_sha256']) == 64, 'Missing complete producer state signature')
    require('expected_producer_method' in recipe, 'Missing explicit producer method declaration')
    if checkpoint.get('kind') in ('author_rgb_only_native_baseline', 'composed_rgb_only_baseline'):
        require(isinstance(recipe['expected_producer_method'], str) and recipe['expected_producer_method'] and recipe['method'] == recipe['expected_producer_method'], 'Baseline method label is not the pinned actual producer method')
    rawpath = Path(recipe['raw_output']).resolve()
    argv = recipe['score_argv']
    require(isinstance(argv, list) and all(isinstance(a, str) for a in argv) and Path(argv[0]).is_absolute(), 'Use exact direct CPU interpreter argv')
    if consumer['kind'] == 'world':
        wanted = [argv[0], str(consumer_path), '--job', str(config_path)]
        require(config['sample_ids'] == expected and config['expected_group_signature_sha256'] == recipe['expected_group_signature_sha256'], 'World job coverage/group signature differs')
        require(recipe.get('source_subset') == config['subset'], 'World full subset differs')
        require(recipe['method'] == config['method'], 'World recipe relabels consumer method')
        require(rawpath == Path(config['output_dir']).resolve()/'score.json', 'World raw path differs from pinned job output')
    else:
        wanted = [argv[0], str(consumer_path), '--spec', str(config_path), '--output', str(rawpath)]
    require(argv == wanted, 'Consumer invocation/config/output argv differs')
    scored, exclusions, proof = eligible_ids(recipe, config, original, expected)
    return original, cases, expected, scored, exclusions, proof, config, consumer_path, config_path, rawpath


def producer_bindings(recipe, config, original, expected, consumer_path):
    """Revalidate real producer metadata; stream prediction SHA without decoding arrays."""
    kind = recipe['consumer']['kind']
    saved = sys.modules.get('tap_subset_score')
    dependencies = recipe.get('consumer_dependencies', [])
    try:
        if kind == 'mv':
            require(len(dependencies) == 1, 'MV requires its exact cache helper dependency pin')
            dep = dependencies[0]
            dep_path = pin(dep['path'], dep['sha256'])
            require(dep_path == consumer_path.with_name('tap_subset_score.py'), 'MV cache helper path differs')
            sys.modules['tap_subset_score'] = module(dep_path, 'tap_subset_score')
        else:
            require(not dependencies, 'Unexpected unreviewed consumer dependency declaration')
        consumer = module(consumer_path, '_full_guard_'+kind+'_'+recipe['consumer']['sha256'][:12])
        if kind == 'world':
            pin(config['native_grid_bridge'], config['native_grid_bridge_sha256'])
            binding = config['expected_wrapper_binding']
            pin(binding['original_wrapper'], binding['original_wrapper_sha256'])
            _, _, checkpoint, rows, evidence = consumer.load_sources(config)
            states = [read(row['producer_run_state']) for row in rows]
            require(all(canonical({k:v for k,v in state.items() if k != 'manifest_sha256'}) == recipe['expected_group_signature_sha256'] for state in states), 'World producer whole state signature differs')
            expected_method = recipe['expected_producer_method']
            require(all(s.get('method') == expected_method for s in states), 'World producer method differs')
            bindings = []
            for row in rows:
                require(sha(row['prediction']) == row['record']['sha256'], 'World prediction payload SHA differs')
                bindings.append({'id':row['case']['id'], 'input_sha256':row['case']['input_sha256'], 'prediction_sha256':row['record']['sha256'], **{key:row[key] for key in ('producer_manifest_sha256','producer_index_sha256','producer_run_state_canonical_sha256','receipt_sha256')}})
            return checkpoint, bindings, evidence
        cache = consumer if kind == 'tap' else consumer.cache
        catalog, evidence, signature = cache.load_sources(config, original, expected)
        require(signature == recipe['expected_group_signature_sha256'], 'Producer whole state signature differs')
        states = [read(Path(item['source_root'])/'run-state.json') for item in catalog.values()]
        require(all(state.get('method') == recipe['expected_producer_method'] for state in states), 'Producer method differs')
        if recipe['expected_producer_method'] is not None:
            require(recipe['method'] == recipe['expected_producer_method'], 'Recipe relabels actual producer method')
        bindings = []
        originals = {c['id']:c for c in original['cases']}
        for ident in expected:
            item = catalog[ident]
            bindings.append({'case_id':ident, 'path':item['path'], 'input_sha256':originals[ident]['input_sha256'], 'prediction_sha256':item['record']['sha256'], 'source_manifest_sha256':item['source_manifest_sha256'], 'source_index_sha256':item['source_index_sha256'], 'run_state_canonical_sha256':item['run_state_canonical_sha256'], 'settings_signature':item['settings_signature'], 'source_case_sha256':item['receipt']['case_sha256']})
        return config['expected_checkpoint'], bindings, evidence
    finally:
        if saved is None:
            sys.modules.pop('tap_subset_score', None)
        else:
            sys.modules['tap_subset_score'] = saved


def validate_raw(recipe, raw, original, expected, scored, config, config_path, checkpoint, bindings, evidence):
    kind = recipe['consumer']['kind']
    schemas = {'tap':'eval-10h-tap-subset-scores-v1','mv':'eval-10h-mv-subset-scores-v1','world':'world-catalog-score-v1'}
    require(raw.get('schema') == schemas[kind], 'Unexpected raw consumer schema')
    require(raw.get('scorer_sha256') == recipe['consumer']['sha256'], 'Raw belongs to another consumer source')
    require(raw.get('source_manifest_sha256') == recipe['source_manifest_sha256'], 'Raw canonical source differs')
    require(raw.get('sample_ids') == expected, 'Raw complete prediction coverage/order differs')
    require([r['id'] for r in raw['per_case']] == scored, 'Raw eligible metric coverage/order differs')
    require(raw.get('checkpoint') == checkpoint and canonical(raw['checkpoint']) == recipe['expected_checkpoint_canonical_sha256'], 'Raw model checkpoint differs')
    require(raw.get('official_sources') == original['official_sources'], 'Raw official metric source differs')
    require(raw.get('formal_score_admitted') is False, 'Coverage verification cannot add formal admission')
    if kind == 'world':
        require(raw.get('job_sha256') == recipe['consumer_config']['sha256'] and raw.get('job') == config, 'Raw World job binding differs')
        require(raw.get('method') == config['method'] and raw.get('inference_settings') == config['expected_inference_settings'], 'Raw World method/settings differs')
        require(raw.get('native_grid_wrapper') == config['expected_wrapper_binding'], 'Raw native-grid bridge binding differs')
        protocol = {k:v for k,v in original['protocol'].items() if k not in ('release_binding', 'subsets')}
        require(raw.get('metric_protocol') == protocol, 'Raw World metric protocol differs')
        require(raw.get('scored_sample_ids', expected) == scored, 'Raw World scored IDs differ')
        require(raw.get('actual_producer_sources') == evidence, 'Raw World producer evidence differs')
        actual = raw['case_bindings']
        require([b['id'] for b in actual] == expected, 'Full prediction bindings including excluded case are missing')
    else:
        require(raw.get('subset_score_spec_sha256') == recipe['consumer_config']['sha256'] and Path(raw['subset_score_spec']).resolve() == config_path, 'Raw spec binding differs')
        require(raw.get('shared_settings_signature') == recipe['expected_group_signature_sha256'], 'Raw complete state signature differs')
        require(raw.get('prediction_sources') == evidence, 'Raw producer evidence differs')
        require(raw.get('protocol') == original['protocol'] and raw.get('input_condition') == original['input_condition'] and raw.get('coordinates') == original['coordinates'], 'Raw protocol/input/coordinate convention differs')
        if kind == 'mv':
            require(raw.get('cache_helper_sha256') == recipe['consumer_dependencies'][0]['sha256'], 'Raw MV cache helper differs')
        actual = raw['prediction_case_bindings']
        require([b['case_id'] for b in actual] == expected, 'Raw prediction binding order differs')
    for wanted, actual_row in zip(bindings, actual):
        require(all(actual_row.get(k) == v for k,v in wanted.items()), 'Raw case binding differs: '+str(wanted))


def score(recipe_path, output):
    recipe_path = Path(recipe_path).resolve(strict=True)
    recipe = read(recipe_path)
    original, cases, expected, scored, exclusions, proof, config, consumer_path, config_path, rawpath = validate_recipe(recipe)
    output = Path(output).resolve()
    require(not output.exists() and not output.is_symlink() and output != rawpath, 'Preserve existing/raw score outputs')
    checkpoint, bindings, evidence = producer_bindings(recipe, config, original, expected, consumer_path)
    reused = rawpath.exists()
    if reused:
        pin(rawpath, recipe.get('raw_output_sha256'))
    else:
        require(recipe.get('raw_output_sha256') is None, 'Pinned existing raw is missing; do not silently recompute it')
        rawpath.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
        subprocess.run(recipe['score_argv'], check=True, env=env)
    pin(consumer_path, recipe['consumer']['sha256'])
    pin(config_path, recipe['consumer_config']['sha256'])
    raw = read(rawpath)
    validate_raw(recipe, raw, original, expected, scored, config, config_path, checkpoint, bindings, evidence)
    result = dict(raw)
    result.update(schema='frozen-full-tracking-scores-v2', method=recipe['method'], dataset=recipe['dataset'],
        source_result_tier=raw.get('result_tier'), source_result_scope=raw.get('result_scope'),
        result_tier='complete_frozen_source_predictions_official_eligible_scores',
        full_frozen_coverage_verified=True, expected_cases=len(expected), actual_prediction_cases=len(bindings),
        actual_scored_cases=len(scored), official_excluded_cases=exclusions, scored_sample_ids=scored,
        full_coverage_source_manifest=str(Path(recipe['source_manifest']).resolve()),
        full_coverage_source_manifest_sha256=recipe['source_manifest_sha256'], full_coverage_source_subset=recipe.get('source_subset'),
        official_eligibility_proof=proof, original_metric_consumer_output=str(rawpath), original_metric_consumer_output_sha256=sha(rawpath),
        full_score_recipe=str(recipe_path), full_score_recipe_sha256=sha(recipe_path), full_score_wrapper_sha256=sha(__file__),
        full_consumer_sha256=recipe['consumer']['sha256'], full_consumer_config_sha256=recipe['consumer_config']['sha256'],
        expected_checkpoint_canonical_sha256=recipe['expected_checkpoint_canonical_sha256'], expected_group_signature_sha256=recipe['expected_group_signature_sha256'],
        v2_integrity={'reused_pinned_raw':reused, 'original_producer_metadata_revalidated':True, 'prediction_SHA_streamed_again':True,
            'prediction_arrays_decoded_by_v2_guard':False, 'producer_index_or_state_synthesized':False, 'formal_admission_preserved':True},
        interpretation='All frozen prediction cases are covered; only the exact original method-independent official eligibility exception may reduce the scoring denominator. Original model/input/alignment qualifications and admission status are preserved.')
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name+'.tmp.'+str(os.getpid())+'.'+str(time.time_ns()))
    with temp.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    try:
        # Atomic no-overwrite publication, including a concurrent writer race.
        os.link(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    directory_fd = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {'output':str(output), 'prediction_cases':len(expected), 'scored_cases':len(scored), 'reused_pinned_raw':reused}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(score(args.recipe, args.output), ensure_ascii=False))
