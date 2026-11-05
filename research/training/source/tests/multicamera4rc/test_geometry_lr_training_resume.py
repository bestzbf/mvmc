"""Real tiny-model CPU training across optimizer/LR migration boundaries."""

import json
from pathlib import Path

import pytest
import torch

from multicamera4rc import launch, train
from multicamera4rc.config import MultiCamera4RCConfig
from multicamera4rc.datasets.demo import make_demo_data
from multicamera4rc.schedule import LearningRatePlan


@pytest.fixture
def recipe(tmp_path, monkeypatch):
    original = MultiCamera4RCConfig.tiny
    monkeypatch.setattr(MultiCamera4RCConfig, 'tiny', classmethod(
        lambda cls, **kwargs: original(refinement_dim=16, refinement_iterations=1,
                                      refinement_levels=1, **kwargs)))
    config = make_demo_data(tmp_path/'data')
    values = json.loads(config.read_text())
    values['epoch_size'] = 12
    config.write_text(json.dumps(values))
    base = ['--data-config', str(config), '--preset', 'tiny', '--device', 'cpu',
            '--cpu-threads', '1', '--source-count', '1', '--batch-size', '1',
            '--save-every', '1', '--seed', '71', '--no-checkpointing']

    def run(name, steps, *extra):
        path = train.main(base+['--output', str(tmp_path/name), '--steps', str(steps)]+list(extra))
        return path, torch.load(path, map_location='cpu', weights_only=True)

    return run


def group_rates(payload):
    return {group['name']: group['lr'] for group in payload['optimizer']['param_groups']}


def named_states(payload):
    result = {}
    for group in payload['optimizer']['param_groups']:
        for name, key in zip(group['param_names'], group['params']):
            result[name] = payload['optimizer']['state'].get(key, {})
    return result


def assert_states_equal(first, second):
    assert set(first) == set(second)
    for name, values in first.items():
        assert set(values) == set(second[name])
        for field, value in values.items():
            torch.testing.assert_close(value, second[name][field], rtol=0, atol=0)


def test_fresh_geometry_then_plain_resume_is_bitwise_equal_and_inherits_nondefault_warmup(recipe):
    run = recipe
    flags = ['--geometry-lr-multiplier', '2', '--geometry-lr-warmup-steps', '4',
             '--lr-schedule', 'cosine', '--lr-end-step', '100']
    checkpoint, partial = run('partial', 2, *flags)
    _, resumed = run('resumed', 3, '--resume', str(checkpoint), '--geometry-lr-multiplier', '2')
    _, direct = run('direct', 3, *flags)
    assert len(resumed['optimizer']['param_groups']) == 4
    assert resumed['training_state']['geometry_lr_options'] == dict(multiplier=2., warmup_steps=4)
    policy = resumed['training_state']['lr_plan']['group_multipliers']['geometry']
    assert policy == dict(start_step=0, multiplier=2., warmup_steps=4)
    assert resumed['training_state'] == direct['training_state']
    for name in direct['model']:
        torch.testing.assert_close(direct['model'][name], resumed['model'][name], rtol=0, atol=0)
    assert_states_equal(named_states(direct), named_states(resumed))
    rates = group_rates(resumed)
    assert rates['geometry']/rates['heads'] == pytest.approx(1.5)


def test_real_three_group_checkpoint_migrates_without_an_update_or_adam_reset(recipe):
    run = recipe
    old_path, old = run('old', 1, '--lr-schedule', 'cosine', '--lr-end-step', '100')
    new_path, migrated = run('migrated', 1, '--resume', str(old_path),
                             '--geometry-lr-multiplier', '2', '--geometry-lr-warmup-steps', '4')
    assert len(old['optimizer']['param_groups']) == 3
    assert len(migrated['optimizer']['param_groups']) == 4
    assert_states_equal(named_states(old), named_states(migrated))
    for name in old['model']:
        torch.testing.assert_close(old['model'][name], migrated['model'][name], rtol=0, atol=0)
    for key in ('data_epoch', 'batches_consumed', 'optimizer_updates'):
        assert old['training_state'][key] == migrated['training_state'][key]
    torch.testing.assert_close(old['rank_rng'][0]['torch'], migrated['rank_rng'][0]['torch'], rtol=0, atol=0)
    assert old['rank_rng'][0]['python'] == migrated['rank_rng'][0]['python']
    assert old['rank_rng'][0]['numpy'] == migrated['rank_rng'][0]['numpy']
    plan = LearningRatePlan(**migrated['training_state']['lr_plan'])
    original = LearningRatePlan(**old['training_state']['lr_plan'])
    assert plan.factor(1) == original.factor(1)
    assert plan.group_multipliers['geometry']['start_step'] == 1
    _, continued = run('continued', 2, '--resume', str(new_path))
    assert continued['training_state']['lr_plan'] == migrated['training_state']['lr_plan']


@pytest.mark.parametrize('override', [False, True])
def test_explicit_global_phase_keeps_geometry_anchor_during_warmup(recipe, override):
    run = recipe
    checkpoint, old = run('partial', 2, '--geometry-lr-multiplier', '2',
                          '--geometry-lr-warmup-steps', '4', '--lr-schedule', 'cosine',
                          '--lr-end-step', '100')
    extra = ['--resume', str(checkpoint), '--lr-schedule', 'cosine', '--lr-end-step', '100']
    if override:
        extra += ['--override-lr', '--lr', '2e-5', '--backbone-lr', '2e-6', '--refinement-lr', '2e-4']
    _, resumed = run('explicit', 3, *extra)
    before = LearningRatePlan(**old['training_state']['lr_plan'])
    after = LearningRatePlan(**resumed['training_state']['lr_plan'])
    assert after.start_step == 2  # The global phase is explicitly new.
    assert after.group_multipliers == before.group_multipliers  # Geometry is not new.
    assert after.group_factor('geometry', 2) == 1.5
    rates = group_rates(resumed)
    assert rates['geometry']/rates['heads'] == pytest.approx(1.5)
    if override:
        assert rates['heads'] == 2e-5
        assert rates['geometry'] == pytest.approx(3e-5)
    else:
        assert rates['heads'] == group_rates(old)['heads']
        assert rates['geometry'] == pytest.approx(group_rates(old)['heads']*1.5)


def test_launcher_passes_geometry_and_explicit_data_migration_options_without_launching():
    args = launch.build_parser().parse_args(['--resume', 'checkpoint.pt',
        '--geometry-lr-multiplier', '2', '--geometry-lr-warmup-steps', '50', '--migrate-data-config'])
    command = launch._command(args, 'python', Path('data.json'), Path('checkpoint.pt'), Path('out'))
    assert command[command.index('--geometry-lr-multiplier')+1] == '2.0'
    assert command[command.index('--geometry-lr-warmup-steps')+1] == '50'
    assert '--migrate-data-config' in command
