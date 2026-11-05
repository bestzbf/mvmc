"""Actual train/save/resume with every generalization feature enabled together."""
import json

import pytest
import torch

from multicamera4rc import train
from multicamera4rc.datasets.demo import make_demo_data
from scripts.build_monocular_generalization_config import build_config


def native_config(tmp_path):
    path = make_demo_data(tmp_path/'data')
    data = json.loads(path.read_text())
    data.update(image_mode='native', supervision='dense', sampling='balanced_coverage',
                num_queries=0, epoch_size=8)
    source = data['sources'][1]
    source.update(weight=1., dense_kind='geometry')
    source['options'].pop('require_precomputed_tracks', None)
    source['root'] = str((path.parent/source['root']).resolve())
    data['sources'] = [source]
    path.write_text(json.dumps(data))
    return path


def command(config, output, steps):
    return ['--data-config', str(config), '--output', str(output), '--preset', 'tiny',
            '--device', 'cpu', '--cpu-threads', '1', '--steps', str(steps),
            '--source-count', '1', '--save-every', '1', '--no-checkpointing']


def test_all_features_train_save_resume_and_preserve_migration_contract(tmp_path):
    base = native_config(tmp_path)
    first = train.main(command(base, tmp_path/'base', 2)+[
        '--lr-schedule', 'cosine', '--lr-end-step', '20'])
    payload = torch.load(first, weights_only=True)
    # Exercise the real pre-feature checkpoint format, which has no param_names.
    for group in payload['optimizer']['param_groups']:
        group.pop('param_names', None)
    torch.save(payload, first)
    data, receipt = build_config(base, return_receipt=True)
    data['photometric_augmentation']['probability'] = 1.
    candidate = tmp_path/'candidate.json'; candidate.write_text(json.dumps(data))
    loss = dict(payload['training_state']['loss_config'])
    loss.update(depth_gradient_scales=[1, 2, 4], depth_objective_options=dict(
        relative_weight=.05, log_bias_weight=.02, temporal_scale_weight=.01,
        diagnostics=True))
    loss_path = tmp_path/'loss.json'; loss_path.write_text(json.dumps(loss))
    second = train.main(command(candidate, tmp_path/'candidate_run', 4)+[
        '--resume', str(first), '--migrate-data-config', '--loss-config', str(loss_path),
        '--geometry-lr-multiplier', '2', '--geometry-lr-warmup-steps', '4'])
    saved = torch.load(second, weights_only=True)
    state = saved['training_state']
    assert state['optimizer_updates'] == 4 and saved['step'] == 4
    assert state['data_migration']['step'] == 2
    assert state['data_migration']['model_optimizer_rng_lr_preserved']
    assert state['photometric_augmentation']['probability'] == 1.
    assert state['loss_config']['depth_objective_options']['relative_weight'] == .05
    assert state['lr_plan']['start_step'] == 0 and state['lr_plan']['end_step'] == 20
    assert state['lr_plan']['group_multipliers']['geometry']['start_step'] == 2
    groups = saved['optimizer']['param_groups']
    assert [g['name'] for g in groups] == ['backbone', 'heads', 'geometry']
    assert all('param_names' in group for group in groups)
    assert all(torch.isfinite(value).all() for value in saved['model'].values())
    # Ordinary resume needs no repeat flags, data reset, loss file or LR override.
    third = train.main(command(candidate, tmp_path/'resumed', 5)+['--resume', str(second)])
    restored = torch.load(third, weights_only=True)
    assert restored['training_state']['lr_plan'] == state['lr_plan']
    assert restored['training_state']['geometry_lr_options'] == state['geometry_lr_options']
    assert restored['training_state']['data_migration'] == state['data_migration']
    assert restored['training_state']['loss_config'] == state['loss_config']
    # Updating a recipe behind the same path cannot silently change continuation.
    data['photometric_augmentation']['brightness'] = .1
    candidate.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='data configuration changed'):
        train.main(command(candidate, tmp_path/'rejected', 6)+['--resume', str(third)])


def test_fresh_training_can_enable_geometry_group(tmp_path):
    config = native_config(tmp_path)
    result = train.main(command(config, tmp_path/'fresh', 1)+[
        '--geometry-lr-multiplier', '2', '--geometry-lr-warmup-steps', '2'])
    payload = torch.load(result, weights_only=True)
    assert payload['training_state']['lr_plan']['group_names'] == ['backbone', 'heads', 'geometry']
    assert payload['training_state']['geometry_lr_options']['multiplier'] == 2
