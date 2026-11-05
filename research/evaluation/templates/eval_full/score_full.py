"""Run a pinned official-metric consumer and verify complete frozen coverage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def score(recipe_path, output):
    recipe = read(recipe_path)
    assert recipe['schema'] == 'frozen-full-score-recipe-v1'
    source = Path(recipe['source_manifest'])
    assert sha(source) == recipe['source_manifest_sha256']
    original = read(source)
    cases = original['cases']
    if recipe.get('source_subset') is not None:
        cases = [c for c in cases if c['subset'] == recipe['source_subset']]
    expected = [c['id'] for c in cases]
    assert expected and len(expected) == len(set(expected))
    assert expected == recipe['expected_case_ids']
    assert len(expected) == recipe['expected_cases']
    rawpath = Path(recipe['raw_output'])
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    if not rawpath.exists():
        rawpath.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(recipe['score_argv'], check=True, env=env)
    raw = read(rawpath)
    assert raw['sample_ids'] == expected, 'Full-score coverage/order differs from frozen source'
    rows = raw['per_case']
    assert [r['id'] for r in rows] == expected and len(rows) == len(expected)
    if raw.get('source_manifest_sha256'):
        assert raw['source_manifest_sha256'] == sha(source)
    result = dict(raw)
    result.update(
        schema='frozen-full-tracking-scores-v1', method=recipe['method'], dataset=recipe['dataset'],
        result_tier='complete_frozen_source_cases', full_frozen_coverage_verified=True,
        expected_cases=len(expected), actual_scored_cases=len(rows),
        coverage_scope=('complete_frozen_world_subset' if recipe.get('source_subset') else 'complete_frozen_manifest'),
        full_coverage_source_manifest=str(source), full_coverage_source_manifest_sha256=sha(source),
        full_coverage_source_subset=recipe.get('source_subset'),
        original_metric_consumer_output=str(rawpath), original_metric_consumer_output_sha256=sha(rawpath),
        full_score_recipe=str(Path(recipe_path).resolve()), full_score_recipe_sha256=sha(recipe_path),
        full_score_wrapper_sha256=sha(__file__),
        interpretation='All cases in the frozen plan are scored. Method/input/alignment extension qualifications remain those in the original metric output; no paper-protocol claim is added.',
    )
    # The consumer's method/protocol admission qualification is deliberately preserved.
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    assert not output.exists(), 'Never overwrite an existing full score'
    temp = output.with_name(output.name + '.tmp')
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, output)
    return {'output': str(output), 'method': recipe['method'], 'dataset': recipe['dataset'], 'full_cases': len(rows)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(score(args.recipe, args.output), ensure_ascii=False))
