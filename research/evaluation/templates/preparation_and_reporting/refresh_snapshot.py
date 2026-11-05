"""Fetch small full-evaluation metadata and verified scores; never copy predictions."""
import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
from publish_split_tables import publish as publish_split_tables

REMOTE = '@@EVAL_FULL_ROOT@@'
DEFAULT_DEST = Path(__file__).resolve().parents[2] / 'outputs/eval_full_20261105'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='@@BENCHMARK_SSH_HOST@@')
    parser.add_argument('--dest', type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    names = ['full_results.json', '全量评估对比.md', '实时进度.md',
             'progress.json', 'full_targets.json', 'queue_plan.json']
    subprocess.run(['scp', *[f'{args.host}:{REMOTE}/{n}' for n in names],
                    str(args.dest) + '/'], check=True)
    summary = json.loads((args.dest / 'full_results.json').read_text())
    scores = []
    for row in summary['rows']:
        if 'score_path' not in row:
            continue
        path = PurePosixPath(row['score_path'])
        if path.is_absolute() or '..' in path.parts or path.suffix != '.json':
            raise ValueError('Unexpected score path')
        scores.append(str(path))
    (args.dest / 'scores').mkdir(exist_ok=True)
    if scores:
        subprocess.run(['scp', *[f'{args.host}:{REMOTE}/{n}' for n in scores],
                        str(args.dest / 'scores') + '/'], check=True)
    publish_split_tables(args.dest.parent)
    print(json.dumps({k: summary[k] for k in [
        'updated_at_utc', 'queue_updated_at_utc', 'full_scored_method_panels',
        'target_method_panels', 'new_prediction_cases_submitted',
        'planned_new_prediction_cases', 'queue_counts', 'errors']},
        ensure_ascii=False))


if __name__ == '__main__':
    main()
