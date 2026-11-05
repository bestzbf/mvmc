"""Write a separate training-only C1/photometric generalization candidate.

No source, GT recipe, sequence whitelist, split, image size or camera layout is
invented. Source/layout budgets are normalized after gently boosting existing
reliable C1 sampling. Longer time layouts are opt-in and require both an exact
admitted-frame manifest and a real adapter-index preflight. The default changes
no time/camera layout and never assumes missing motion GT is zero motion.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

from multicamera4rc.datasets.augmentation import DEFAULT_PHOTOMETRIC, validate_photometric_augmentation
from multicamera4rc.datasets.mixture import create_dataset, read_data_config


# These are the existing dense geometry/motion readers, not dataset additions.
_RELIABLE = frozenset(('kubric_multicam', 'mv_kubric', 'syn4d', 'mugen_h5',
    'hypersim_zip', 'pointodyssey_geometry', 'pointodyssey_tracks', 'dynamic_replica',
    'vkitti2', 'tartanair_v1_zip', 'tartanair_v2_zip', 'panoptic_kinoptic'))
_TEMPORAL = _RELIABLE - {'panoptic_kinoptic'}


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def _layouts(source, config):
    return source.get('layouts') or [dict(cameras=source.get('cameras', 1),
        num_times=source.get('num_times', config.get('num_times', 3)), weight=1.)]


def _eligible(source):
    return (str(source.get('adapter', '')).lower() in _RELIABLE
            and source.get('dense_kind') != 'camera_only')


def _c1_mass(source, config):
    layouts = _layouts(source, config)
    total = sum(_positive(row.get('weight', 1.), 'layout weight') for row in layouts)
    return sum(_positive(row.get('weight', 1.), 'layout weight') for row in layouts
               if row.get('cameras', source.get('cameras', 1)) == 1)/total


def _assert_training_sources(config):
    if config.get('supervision') != 'dense' or config.get('image_mode') != 'native':
        raise ValueError('candidate requires an existing native dense training configuration')
    if config.get('sampling') != 'balanced_coverage':
        raise ValueError('C1 candidate requires balanced_coverage sampling')
    if not isinstance(config.get('sources'), list) or not config['sources']:
        raise ValueError('base configuration requires existing sources')
    for source in config['sources']:
        if source.get('split', 'train') not in ('train', 'training'):
            raise ValueError('generalization candidate only accepts training splits; validation/test are unchanged')
        labels = [source.get('adapter', ''), source.get('name', ''), source.get('options', {}).get('dataset_name', '')]
        if any(re.search(r'(^|[^a-z0-9])sintel([^a-z0-9]|$)', str(label).lower()) for label in labels):
            raise ValueError('Sintel benchmark sources may not be added to training')
        _positive(source.get('weight', 1.), 'source weight')


def _extend_times(config, requested):
    """Return index-preflight receipts; reject unproved temporal expansion."""
    if isinstance(requested, bool) or not isinstance(requested, int) or not 2 <= requested <= 48:
        raise ValueError('max_monocular_times must be an integer between 2 and 48')
    receipts = []
    for source in config['sources']:
        if not _eligible(source) or source.get('view_mode', 'temporal') != 'temporal' or not _c1_mass(source, config):
            continue
        layouts = _layouts(source, config)
        monocular = [row for row in layouts if row.get('cameras', source.get('cameras', 1)) == 1]
        if max(row['num_times'] for row in monocular) >= requested:
            continue
        adapter = str(source['adapter']).lower()
        if adapter not in _TEMPORAL or not source.get('manifest'):
            raise ValueError(f'{source.get("name", adapter)}: longer C1 layout requires a supported temporal adapter and admitted-frame manifest')
        manifest_path = Path(source['manifest']).resolve()
        manifest = json.loads(manifest_path.read_text())
        if Path(manifest.get('root', '')).resolve() != Path(source['root']).resolve():
            raise ValueError('long-layout manifest root does not match the source')
        records = manifest.get('sequences')
        selected = source.get('sequences', list(records) if isinstance(records, dict) else [])
        if not isinstance(records, dict) or not selected:
            raise ValueError('long-layout manifest must enumerate existing admitted sequences')
        strides = source.get('strides', [1])
        if not isinstance(strides, list) or not strides or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in strides):
            raise ValueError('long-layout strides must be positive integers')
        required = 1+(requested-1)*max(strides)
        sizes = []
        for sequence in selected:
            frames = records.get(sequence, {}).get('valid_frames')
            if (not isinstance(frames, list) or len(frames) < required
                    or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in frames)
                    or frames != sorted(set(frames))):
                raise ValueError(f'{sequence}: admitted frame pool cannot prove requested C1 times/strides')
            sizes.append(len(frames))
        # Keep all old time layouts and their relative probabilities; allocate
        # only 20% of the existing C1 mass to the explicitly requested length.
        expanded = deepcopy(layouts)
        mass = sum(row.get('weight', 1.) for row in monocular)
        for row in expanded:
            if row.get('cameras', source.get('cameras', 1)) == 1:
                row['weight'] = row.get('weight', 1.)*.8
        expanded.append(dict(cameras=1, num_times=requested, weight=mass*.2))
        source['layouts'] = expanded
        source['num_times'] = max(row['num_times'] for row in expanded)
        if 'max_times' in source.get('options', {}):
            source['options']['max_times'] = source['num_times']
        # Metadata counts alone cannot show that the actual adapter accepts
        # these frames. Constructor validation checks reader support, existing
        # sequence IDs and valid_frames against the actual adapter index.
        preflight = deepcopy(config)
        preflight['sources'] = [deepcopy(source)]
        preflight.pop('photometric_augmentation', None)
        preflight.pop('augmentation', None)
        indexed = create_dataset(preflight)
        actual = indexed.sources[0]
        if set(selected) != {sequence for sequence, _ in actual['pool']}:
            raise ValueError('adapter preflight changed the admitted long-layout sequence pool')
        receipts.append(dict(source=source.get('name', adapter), num_times=requested,
            sequences=len(selected), min_admitted_frames=min(sizes), required_frames=required,
            manifest=str(manifest_path), manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            adapter_index_preflight=True, rgb_or_gt_inference_executed=False))
    if not receipts:
        raise ValueError('requested long C1 times introduce no verified supported layout')
    return receipts


def build_config(base_config, *, c1_multiplier=1.25, photometric_augmentation=DEFAULT_PHOTOMETRIC,
                 max_monocular_times=None, return_receipt=False):
    """Return a new mapping, optionally with a separate preparation receipt.

    ``base_config`` may be a path (relative paths are resolved by the native
    reader) or a previously resolved mapping. This function never writes it.
    The receipt records configured clip mass before sequence eligibility
    filtering; it does not claim a measured realized data distribution.
    """
    original = read_data_config(base_config) if not isinstance(base_config, Mapping) else deepcopy(dict(base_config))
    _assert_training_sources(original)
    multiplier = _positive(c1_multiplier, 'c1_multiplier')
    if not 1 <= multiplier <= 1.5:
        raise ValueError('c1_multiplier must be between 1 and 1.5 for a mild exposure change')
    config = deepcopy(original)
    photo = validate_photometric_augmentation(photometric_augmentation)
    if photo is None:
        config.pop('photometric_augmentation', None)
    else:
        config['photometric_augmentation'] = photo
    total = sum(float(source.get('weight', 1.)) for source in original['sources'])
    c1_before = sum(float(source.get('weight', 1.))*_c1_mass(source, original)
                    for source in original['sources'] if _eligible(source))
    if c1_before <= 0:
        raise ValueError('no existing reliable C1 geometry exposure can be increased')
    for source in config['sources']:
        if not _eligible(source):
            continue
        fraction = _c1_mass(source, config)
        if fraction == 1:
            source['weight'] = float(source.get('weight', 1.))*multiplier
        elif fraction > 0 and source.get('layouts'):
            rows = source['layouts']
            before = sum(float(row.get('weight', 1.)) for row in rows)
            # Increase the source budget by its mixed C1 fraction before
            # renormalizing layouts. Consequently every existing C1 atom is
            # multiplied by the same multiplier, and every non-C1 atom by the
            # same final global normalization, including across sources.
            source['weight'] = float(source.get('weight', 1.))*(1+(multiplier-1)*fraction)
            for row in rows:
                row['weight'] = float(row.get('weight', 1.))*(multiplier if row.get('cameras', source.get('cameras', 1)) == 1 else 1)
            normalizer = before/sum(row['weight'] for row in rows)
            for row in rows:
                row['weight'] *= normalizer
    normalizer = total/sum(float(source.get('weight', 1.)) for source in config['sources'])
    for source in config['sources']:
        source['weight'] = float(source.get('weight', 1.))*normalizer
    long_receipts = [] if max_monocular_times is None else _extend_times(config, max_monocular_times)
    c1_after = sum(source['weight']*_c1_mass(source, config)
                   for source in config['sources'] if _eligible(source))
    receipt = dict(schema='monocular-generalization-config-v1', c1_multiplier=multiplier,
        total_source_weight_before=total, total_source_weight_after=sum(source['weight'] for source in config['sources']),
        configured_reliable_c1_clip_mass_before=c1_before/total,
        configured_reliable_c1_clip_mass_after=c1_after/total,
        sources_added=0, train_only=True, benchmark_sources_added=False,
        camera_time_layouts_added=len(long_receipts), time_layouts_unchanged=not long_receipts,
        long_layout_preflights=long_receipts, motion_gt_recipes_unchanged=True, training_started=False)
    return (config, receipt) if return_receipt else config


def write_config(base_config, output, **kwargs):
    config, receipt = build_config(base_config, return_receipt=True, **kwargs)
    output = Path(output).resolve()
    base = Path(base_config).resolve()
    roots = [Path(source['root']).resolve() for source in config['sources']]
    if output == base or any(output == root or root in output.parents for root in roots):
        raise ValueError('output must be outside the base config and read-only dataset roots')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(config, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    receipt.update(output=str(output), output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    return config, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--c1-multiplier', type=float, default=1.25)
    parser.add_argument('--photometric-config', type=Path, help='RGB-only options JSON; defaults to the mild preset')
    parser.add_argument('--max-monocular-times', type=int,
                        help='opt-in new time layout; requires existing admitted-frame manifests and adapter-index preflight')
    args = parser.parse_args(argv)
    photo = DEFAULT_PHOTOMETRIC if args.photometric_config is None else json.loads(args.photometric_config.read_text())
    _, receipt = write_config(args.base_config, args.output, c1_multiplier=args.c1_multiplier,
        photometric_augmentation=photo, max_monocular_times=args.max_monocular_times)
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
