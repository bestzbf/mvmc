"""CPU evidence for named geometry-LR migration without losing Adam history."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from multicamera4rc.optim import optimizer_groups, parameter_groups, load_optimizer_state
from multicamera4rc.schedule import LearningRatePlan, migrate_geometry_plan


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.pretrained = nn.Linear(2, 2)
        self.backbone.pretrained.camera_relation_weight = nn.Parameter(torch.ones(2))
        self.head = nn.ModuleDict({'depth': nn.Linear(2, 2), 'ray': nn.Linear(2, 2)})
        self.cam_dec = nn.Linear(2, 2)
        self.motion_decoder = nn.Linear(2, 2)
        self.trajectory_refiner = nn.Linear(2, 2)


def make_optimizer(model, *, geometry=False, legacy=False):
    groups = optimizer_groups(model, 1e-5, 1e-4, 1e-3,
                              geometry_lr=1e-4 if geometry else None)
    if legacy:
        for group in groups:
            group.pop('param_names')
    return torch.optim.AdamW(groups, weight_decay=.03, betas=(.8, .95), amsgrad=True)


def update(model, optimizer):
    # Repeated equal shapes and distinct gradients make swapped state observable.
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, .01*(index+1))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def assert_moments(model, optimizer, other_model, other_optimizer):
    other = dict(other_model.named_parameters())
    for name, parameter in model.named_parameters():
        expected = other_optimizer.state[other[name]]
        actual = optimizer.state[parameter]
        assert set(actual) == set(expected)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_default_parameter_groups_and_lr_values_retain_legacy_order():
    model = SmallModel()
    expected = {name: [] for name in ('backbone', 'heads', 'refinement')}
    for name, parameter in model.named_parameters():
        group = ('refinement' if name.startswith('trajectory_refiner.') else
                 'backbone' if name.startswith('backbone.pretrained.')
                 and not name.endswith('camera_relation_weight') else 'heads')
        expected[group].append(name)
    groups = optimizer_groups(model, 1e-5, 1e-4, 1e-3)
    assert [group['name'] for group in groups] == list(expected)
    assert [group['lr'] for group in groups] == [1e-5, 1e-4, 1e-3]
    assert [group['param_names'] for group in groups] == list(expected.values())


def test_legacy_three_to_four_preserves_every_moment_and_next_update():
    torch.manual_seed(5)
    source = SmallModel()
    old = make_optimizer(source, legacy=True)
    update(source, old)
    target = deepcopy(source)
    restored = make_optimizer(target, geometry=True)
    proof = load_optimizer_state(restored, old.state_dict(), target,
                                 model_state=source.state_dict(), allow_legacy=True)
    assert proof['geometry_split'] and proof['legacy_order_reconstructed']
    assert proof['old_group_names'] == ['backbone', 'heads', 'refinement']
    assert proof['new_group_names'] == ['backbone', 'heads', 'refinement', 'geometry']
    assert_moments(target, restored, source, old)
    assert [g['lr'] for g in restored.param_groups] == [1e-5, 1e-4, 1e-3, 1e-4]
    assert all(g['betas'] == (.8, .95) and g['weight_decay'] == .03 for g in restored.param_groups)
    update(source, old)
    update(target, restored)
    for name, parameter in target.named_parameters():
        torch.testing.assert_close(parameter, dict(source.named_parameters())[name], rtol=0, atol=0)
    assert_moments(target, restored, source, old)


def test_named_checkpoint_resume_does_not_require_legacy_order_and_keeps_missing_states():
    source = SmallModel()
    old = make_optimizer(source, geometry=True)
    update(source, old)
    unused = source.cam_dec.bias
    del old.state[unused]  # Dynamic heads may never have been activated.
    target = deepcopy(source)
    restored = make_optimizer(target, geometry=True)
    proof = load_optimizer_state(restored, old.state_dict(), target)
    assert not proof['geometry_split'] and not proof['legacy_order_reconstructed']
    assert not restored.state[target.cam_dec.bias]
    assert_moments(target, restored, source, old)
    assert [g['param_names'] for g in restored.state_dict()['param_groups']] == [
        g['param_names'] for g in old.state_dict()['param_groups']]


def test_shared_parameters_have_one_owner_and_cross_group_aliases_are_rejected():
    model = SmallModel()
    model.cam_dec.weight = model.head['depth'].weight
    groups = parameter_groups(model, separate_geometry=True)
    parameters = [p for _, entries in groups for _, p in entries]
    assert len(parameters) == len({id(p) for p in parameters})
    source = make_optimizer(model, legacy=True)
    update(model, source)
    restored = make_optimizer(model, geometry=True)
    load_optimizer_state(restored, source.state_dict(), model,
                         model_state=model.state_dict(), allow_legacy=True)
    model.motion_decoder.weight = model.head['depth'].weight
    with pytest.raises(ValueError, match='shared parameter crosses'):
        parameter_groups(model, separate_geometry=True)


@pytest.mark.parametrize('fault', ['moment_shape', 'amsgrad', 'model_shape', 'model_order', 'names', 'partial_names'])
def test_unproven_or_bad_shape_mapping_fails_before_mutating_optimizer(fault):
    source = SmallModel()
    old = make_optimizer(source, legacy=True)
    update(source, old)
    state, weights = deepcopy(old.state_dict()), dict(source.state_dict())
    if fault == 'moment_shape':
        next(iter(state['state'].values()))['exp_avg'] = torch.zeros(7)
    elif fault == 'amsgrad':
        next(iter(state['state'].values())).pop('max_exp_avg_sq')
    elif fault == 'model_shape':
        first = next(iter(weights))
        weights[first] = torch.zeros(7)
    elif fault == 'model_order':
        weights = dict(reversed(list(weights.items())))
    elif fault == 'names':
        state['param_groups'][0]['name'] = 'unverified'
    else:
        state['param_groups'][0]['param_names'] = ['wrong']*len(state['param_groups'][0]['params'])
    restored = make_optimizer(source, geometry=True)
    before = deepcopy(restored.state_dict())
    with pytest.raises(ValueError):
        load_optimizer_state(restored, state, source, model_state=weights, allow_legacy=True)
    assert not restored.state
    assert restored.state_dict()['param_groups'] == before['param_groups']


def test_unnamed_legacy_requires_explicit_proof_and_does_not_guess_from_equal_shapes():
    model = SmallModel()
    old, restored = make_optimizer(model, legacy=True), make_optimizer(model, geometry=True)
    with pytest.raises(ValueError, match='explicit allow_legacy'):
        load_optimizer_state(restored, old.state_dict(), model, model_state=model.state_dict())
    with pytest.raises(ValueError, match='ordering evidence'):
        load_optimizer_state(restored, old.state_dict(), model, allow_legacy=True)


def test_geometry_multiplier_preserves_cosine_phase_and_does_not_repeat_on_resume():
    old_names = ['backbone', 'heads', 'refinement']
    new_names = old_names+['geometry']
    old = LearningRatePlan('cosine', 1000, 200000, [1e-6, 1e-5, 1e-4],
                           warmup_steps=100, min_ratio=.1, warmup_start_ratio=.1)
    new = migrate_geometry_plan(old, old_names, new_names, start_step=139000,
                                multiplier=2, warmup_steps=100)
    assert new.start_step == old.start_step and new.end_step == old.end_step
    assert new.warmup_steps == old.warmup_steps
    model = SmallModel()
    optimizer = make_optimizer(model, geometry=True)
    for step in (138999, 139000, 139025, 139100, 160000, 199999):
        new.apply(optimizer, step)
        assert new.factor(step) == old.factor(step)
        rates = [group['lr'] for group in optimizer.param_groups]
        assert rates[:3] == [lr*old.factor(step) for lr in old.base_lrs]
        expected_multiplier = (1. if step <= 139000 else
                               1.+min(1., (step-139000)/100))
        assert rates[3] == pytest.approx(old.base_lrs[1]*old.factor(step)*expected_multiplier)
    # Restoration at successive save points retains the original absolute anchor.
    restored = LearningRatePlan(**new.to_config())
    again = LearningRatePlan(**restored.to_config())
    assert again.group_factor('geometry', 139050) == 1.5
    assert again.group_factor('geometry', 139100) == 2.
    with pytest.raises(ValueError, match='already exists'):
        migrate_geometry_plan(again, new_names, new_names, start_step=139050)


def test_legacy_schedule_serialization_and_group_identity_validation():
    plan = LearningRatePlan('constant', 0, 10, [1e-5, 1e-4, 1e-3])
    assert set(plan.to_config()) == {'mode', 'start_step', 'end_step', 'base_lrs',
                                    'warmup_steps', 'min_ratio', 'warmup_start_ratio'}
    migrated = migrate_geometry_plan(plan, ['backbone', 'heads', 'refinement'],
                                    ['backbone', 'heads', 'refinement', 'geometry'],
                                    start_step=2, warmup_steps=0)
    optimizer = make_optimizer(SmallModel(), geometry=True)
    migrated.apply(optimizer, 2)
    assert optimizer.param_groups[-1]['lr'] == 2e-4
    optimizer.param_groups[-1]['name'] = 'incorrect'
    with pytest.raises(ValueError, match='identity/order'):
        migrated.apply(optimizer, 3)
