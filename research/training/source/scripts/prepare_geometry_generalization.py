"""Prepare one combined geometry-generalization continuation, without launching.

This CPU-only preparation reads a complete native checkpoint with mmap, writes
separate candidate configs and an argv receipt, and never probes or claims GPUs.
The checkpoint's model/Adam/global LR schedule are resumed unchanged; only the
named geometry group receives an independent multiplier ramp at first launch.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import torch

from multicamera4rc.checkpoint import config_from_checkpoint
from multicamera4rc.datasets.mixture import read_data_config
from multicamera4rc.losses import MultiCamera4RCLoss
from multicamera4rc.schedule import LearningRatePlan
from scripts.build_monocular_generalization_config import build_config


REPO = Path(__file__).resolve().parents[1]
PRESET = REPO/'multicamera4rc/configs/geometry_generalization.json'
_OPTION_PATHS = ('cache_dir', 'validation_stamp_dir', 'precompute_root', 'fast_root',
                 'camera_metadata', 'split_metadata', 'metadata_root',
                 'depth_validity_mask_root', 'eligibility_manifest')
_WRITABLE_PATHS = {'cache_dir'}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def _positive_integer(value, label, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(f'{label} must be a {"nonnegative" if zero else "positive"} integer')
    return value


def _gpu_list(value):
    if isinstance(value, str):
        try:
            value = [int(item.strip()) for item in value.split(',')]
        except ValueError as error:
            raise ValueError('gpus must be comma-separated physical GPU integers') from error
    if (not isinstance(value, (list, tuple)) or not value
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value)
            or len(set(value)) != len(value)):
        raise ValueError('gpus must contain distinct nonnegative physical GPU integers')
    return list(value)


def _external_assets(config):
    """Check resolved paths; fingerprint metadata, not TB-sized data assets."""
    assets = []
    for index, source in enumerate(config['sources']):
        entries = [('root', source['root'], True)]
        if source.get('manifest'):
            entries.append(('manifest', source['manifest'], False))
        entries.extend((f'options.{key}', source['options'][key], False)
                       for key in _OPTION_PATHS if source.get('options', {}).get(key) is not None)
        for field, value, dataset_root in entries:
            path = Path(value)
            if not path.is_absolute() or path != path.resolve():
                raise ValueError(f'external asset did not resolve to an absolute path: {value}')
            exists = path.exists()
            optional_cache = field.removeprefix('options.') in _WRITABLE_PATHS
            if not exists and not optional_cache:
                raise FileNotFoundError(f'external training asset does not exist: {path}')
            row = dict(source_index=index, source=source.get('name', source.get('adapter')),
                       field=field, path=str(path), exists=exists)
            if exists:
                stat = path.stat()
                row.update(kind='directory' if path.is_dir() else 'file',
                           bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
                if path.is_file() and not dataset_root:
                    row['sha256'] = _sha256(path)
                else:
                    row['content_hash_omitted'] = 'dataset assets are not recursively read or rewritten'
            else:
                row['missing_optional_writable_cache'] = True
            assets.append(row)
    return assets


def _pin_checkpoint(source, output, copy_checkpoint):
    destination = output/source.name
    try:
        os.link(source, destination)
        return destination, 'hardlink'
    except OSError as error:
        if error.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP):
            raise
        if copy_checkpoint:
            shutil.copy2(source, destination)
            return destination, 'explicit_copy'
        return source, 'not_pinned_cross_device_or_link_permission'


def prepare(checkpoint, base_data_config, output, gpus, *, python=sys.executable,
            steps=200000, max_monocular_times=None, copy_checkpoint=False):
    checkpoint, base, output = (Path(value).expanduser().resolve()
                                for value in (checkpoint, base_data_config, output))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('preparation output must be absent or an empty directory')
    if not checkpoint.is_file() or not base.is_file():
        raise FileNotFoundError('checkpoint and base data config must be existing files')
    source_stat = checkpoint.stat()
    data = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if (not isinstance(data, Mapping) or data.get('architecture') != 'multicamera4rc'
            or not isinstance(data.get('model'), Mapping) or not data['model']
            or not isinstance(data.get('optimizer'), Mapping)):
        raise ValueError('preparation requires a complete native model+optimizer checkpoint')
    state = data.get('training_state')
    if not isinstance(state, Mapping) or not isinstance(state.get('loss_config'), Mapping) or not isinstance(state.get('lr_plan'), Mapping):
        raise ValueError('checkpoint must save its training loss_config and lr_plan')
    if state.get('geometry_lr_options') is not None:
        raise ValueError('checkpoint already has geometry_lr_options; use ordinary resume, not another C1/data migration')
    start = _positive_integer(data.get('step'), 'checkpoint step', zero=True)
    steps = _positive_integer(steps, 'total target steps')
    if steps <= start:
        raise ValueError('total target steps must exceed the checkpoint step')
    runtime = {name: _positive_integer(state.get(name), f'checkpoint {name}',
                    zero=name in ('source_count', 'max_source_time_pairs'))
               for name in ('world_size', 'batch_size', 'gradient_accumulation',
                            'source_count', 'max_source_time_pairs')}
    if not isinstance(state.get('prefer_motion_sources'), bool):
        raise ValueError('checkpoint must save boolean prefer_motion_sources')
    runtime['prefer_motion_sources'] = state['prefer_motion_sources']
    devices = _gpu_list(gpus)
    if len(devices) != runtime['world_size']:
        raise ValueError('GPU count must preserve the checkpoint world_size')
    config = config_from_checkpoint(data)
    lr_plan = LearningRatePlan(**dict(state['lr_plan']))
    if not isinstance(data.get('config'), Mapping) or 'embed_dim' not in data['config']:
        raise ValueError('checkpoint must contain its complete native model config')
    if (not isinstance(data['optimizer'].get('param_groups'), list)
            or not isinstance(data['optimizer'].get('state'), Mapping)
            or len(data['optimizer']['param_groups']) != len(lr_plan.base_lrs)):
        raise ValueError('checkpoint optimizer groups and saved LR plan disagree')
    amp_dtype = str(state.get('amp_dtype', 'bfloat16')).removeprefix('torch.')
    if amp_dtype not in ('bfloat16', 'float16'):
        raise ValueError('GPU continuation requires saved BF16 or FP16 precision')
    python_path = Path(python).expanduser()
    # Preserve a venv's executable path: resolving its symlink can silently
    # invoke the system interpreter without the selected environment packages.
    resolved_python = os.path.abspath(python_path) if python_path.is_file() else shutil.which(str(python))
    if not resolved_python or not os.access(resolved_python, os.X_OK):
        raise FileNotFoundError(f'training Python executable does not exist: {python}')
    resolved_base = read_data_config(base)
    saved_data_hash = state.get('data_config_sha256')
    if saved_data_hash is not None and saved_data_hash != _sha256(base):
        raise ValueError('base data config does not match the checkpoint data_config_sha256')
    dataset_roots = [Path(source['root']) for source in resolved_base['sources']]
    if output in (checkpoint, base) or any(output == root or root in output.parents for root in dataset_roots):
        raise ValueError('preparation output must be outside source configs and read-only dataset roots')
    external = _external_assets(resolved_base)
    preset = json.loads(PRESET.read_text())
    candidate, data_receipt = build_config(resolved_base,
        c1_multiplier=preset['c1_multiplier'], max_monocular_times=max_monocular_times,
        return_receipt=True)
    # Failure logging belongs to the candidate, never the old training run.
    previous_invalid_log = candidate.get('invalid_log')
    if previous_invalid_log is not None:
        candidate['invalid_log'] = str(output/'training_invalid_samples.jsonl')
    original_loss = deepcopy(dict(state['loss_config']))
    loss = deepcopy(original_loss)
    for key, value in preset['loss_overrides'].items():
        if key == 'depth_objective_options':
            loss[key] = dict(loss.get(key) or {}, **deepcopy(value))
        else:
            loss[key] = deepcopy(value)
    MultiCamera4RCLoss(**loss)  # Validate settings without constructing the model.
    output.mkdir(parents=True, exist_ok=True)
    pinned, pin_mode = _pin_checkpoint(checkpoint, output, copy_checkpoint)
    final_stat = pinned.stat()
    if ((source_stat.st_size, source_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns)
            or (pin_mode != 'explicit_copy' and source_stat.st_ino != final_stat.st_ino)):
        raise ValueError('checkpoint changed during preparation; select a complete numbered checkpoint')
    checkpoint_hash = _sha256(pinned)
    loss_path, mixture_path = output/'loss.json', output/'mixture.json'
    _write_json(loss_path, loss)
    _write_json(mixture_path, candidate)
    run = output/'run'
    common = ['--data-config', str(mixture_path), '--gpus', ','.join(map(str, devices)),
              '--python', resolved_python, '--steps', str(steps),
              '--batch-size', str(runtime['batch_size']),
              '--gradient-accumulation', str(runtime['gradient_accumulation']),
              '--source-count', str(runtime['source_count']),
              '--max-source-time-pairs', str(runtime['max_source_time_pairs']),
              '--prefer-motion-sources' if runtime['prefer_motion_sources'] else '--no-prefer-motion-sources',
              '--decoder-memory', config.decoder_memory, '--amp-dtype', amp_dtype,
              '--workers', str(state.get('workers', 2)), '--save-every', str(state.get('save_every', 100))]
    command = [resolved_python, '-m', 'multicamera4rc.launch', '--resume', str(pinned),
               '--output', str(run), '--loss-config', str(loss_path), '--migrate-data-config',
               '--geometry-lr-multiplier', str(preset['geometry_lr_multiplier']),
               '--geometry-lr-warmup-steps', str(preset['geometry_lr_warmup_steps']), *common]
    continuation = dict(schema='geometry-generalization-continuation-v1', cwd=str(REPO), argv=command,
        training_started=False, gpu_probe_executed=False, mode='all_changes_enabled_together',
        run_output=str(run), inherited_runtime=runtime,
        ordinary_resume_argv_template=[resolved_python, '-m', 'multicamera4rc.launch',
            '--resume', '{CHECKPOINT_FROM_THIS_RUN}', '--output', '{FRESH_RESUME_OUTPUT}', *common],
        ordinary_resume_note='Use a checkpoint created by this candidate; omit --migrate-data-config and geometry-LR flags so cursor migration and warmup do not repeat.')
    _write_json(output/'continuation.json', continuation)
    receipt = dict(schema='geometry-generalization-preparation-v1', checkpoint=dict(
        source=str(checkpoint), selected=str(pinned), pin_mode=pin_mode,
        pinned=pin_mode in ('hardlink', 'explicit_copy'), bytes=source_stat.st_size,
        sha256=checkpoint_hash, step=start), base_data_config=dict(path=str(base), sha256=_sha256(base)),
        preset=dict(path=str(PRESET), sha256=_sha256(PRESET)),
        output_files={name:dict(path=str(output/name), sha256=_sha256(output/name))
                      for name in ('loss.json', 'mixture.json', 'continuation.json')},
        external_assets=external, data_changes=data_receipt,
        invalid_log_redirect=dict(previous=previous_invalid_log, candidate=candidate.get('invalid_log')),
        loss_overrides=deepcopy(preset['loss_overrides']), model_config_preserved=deepcopy(data['config']),
        inherited_lr_plan=deepcopy(state['lr_plan']), inherited_loss_config=original_loss,
        base_matches_checkpoint_fingerprint=(saved_data_hash is not None),
        implementation_sources={name: _sha256(REPO/name) for name in (
            'scripts/prepare_geometry_generalization.py', 'scripts/build_monocular_generalization_config.py',
            'multicamera4rc/train.py', 'multicamera4rc/launch.py', 'multicamera4rc/optim.py',
            'multicamera4rc/schedule.py', 'multicamera4rc/losses.py', 'multicamera4rc/depth_objectives.py',
            'multicamera4rc/datasets/mixture.py', 'multicamera4rc/datasets/augmentation.py')},
        optimizer_moments_rewritten=False, rng_rewritten=False, checkpoint_rewritten=False,
        data_cursor_migration='explicit one-time reset for the new mixed stream at first launch',
        runtime_defaults_used={key: value for key, value in (('workers', 2), ('save_every', 100)) if key not in state},
        training_started=False, gpu_probe_executed=False, source_or_test_data_written=False,
        all_changes_enabled_together=True, long_context_requested=max_monocular_times,
        long_context_memory_validated=False, quality_improvement_validated=False)
    _write_json(output/'receipt.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--base-data-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--steps', type=int, default=200000, help='total target step, not additional steps')
    parser.add_argument('--max-monocular-times', type=int,
                        help='opt-in long-layout manifest/index preflight; does not prove GPU memory capacity')
    parser.add_argument('--copy-checkpoint', action='store_true',
                        help='explicitly permit a full copy if hard-link pinning is unavailable')
    args = parser.parse_args(argv)
    receipt = prepare(args.checkpoint, args.base_data_config, args.output, args.gpus,
        python=args.python, steps=args.steps, max_monocular_times=args.max_monocular_times,
        copy_checkpoint=args.copy_checkpoint)
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
