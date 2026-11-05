"""Depth objective invariants, meaningful gradients and legacy compatibility."""
from dataclasses import asdict

import pytest
import torch

from multicamera4rc.depth_objectives import DepthObjectiveOptions, dense_depth_objectives
from multicamera4rc.losses import MultiCamera4RCLoss


def case():
    # Each time sees a different surface/depth distribution; no static identity.
    gt = torch.tensor([[[[.01, .1, 1.], [2., 4., 8.]],
                        [[8., 3., .2], [1., .04, 5.]],
                        [[.3, 2., 6.], [.1, 3., 9.]]]])
    pred = dict(depth=gt.clone(), camera_id=torch.tensor([[7, 7, 7]]),
                time_idx=torch.tensor([[0, 5, 19]]),
                observation_valid=torch.ones(1, 3, dtype=torch.bool))
    target = dict(depth=gt, depth_valid=torch.ones_like(gt, dtype=torch.bool))
    return pred, target


def all_options(**overrides):
    values = dict(relative_weight=.05, log_bias_weight=.02,
                  temporal_scale_weight=.01, diagnostics=True)
    values.update(overrides)
    return DepthObjectiveOptions(**values)


def objectives(pred, target, options=None):
    return dense_depth_objectives(pred['depth'], target['depth'], target['depth_valid'],
                                  pred, options or all_options())


def test_exact_gt_zero_even_when_depth_changes_between_frames():
    pred, target = case()
    pred['depth'].requires_grad_()
    terms, loss = objectives(pred, target)
    assert loss == 0
    assert terms['depth_objective_temporal_pairs'] == 2
    assert terms['depth_objective_observations'] == 3
    loss.backward()
    assert torch.isfinite(pred['depth'].grad).all()
    assert not pred['depth'].grad.any()


def test_relative_and_anchor_detect_common_scale_temporal_detects_drift():
    pred, target = case()
    pred['depth'] *= 1.5
    terms, loss = objectives(pred, target)
    assert terms['depth_objective_relative'] > 0
    assert terms['depth_objective_log_bias'] > 0
    torch.testing.assert_close(terms['depth_objective_temporal_scale'], torch.tensor(0.), atol=1e-12, rtol=0)
    pred['depth'] = target['depth']*torch.tensor([1., 1.5, 2.])[None, :, None, None]
    drift, _ = objectives(pred, target)
    assert drift['depth_objective_temporal_scale'] > 0


def test_log_bias_anchor_detects_each_observation_not_only_mean_clip_bias():
    pred, target = case()
    pred['depth'] = target['depth']*torch.tensor([.5, 1., 2.])[None, :, None, None]
    terms, _ = objectives(pred, target)
    assert terms['depth_objective_log_bias'] > 0
    assert abs(float(terms['depth_objective_log_bias_signed'])) < 1e-6


def test_affine_shift_is_penalized_instead_of_gt_fitted_away():
    pred, target = case()
    pred['depth'] += .25
    terms, loss = objectives(pred, target)
    assert terms['depth_objective_relative'] > 0
    assert terms['depth_objective_log_bias'] > 0
    assert loss > 0


def test_unit_change_preserves_losses_and_gt_defined_bins():
    pred, target = case()
    pred['depth'] *= torch.tensor([1.1, .7, 1.5])[None, :, None, None]
    before, loss = objectives(pred, target)
    for scale in (1e-6, 1e5):
        after, scaled_loss = objectives(dict(pred, depth=pred['depth']*scale),
                                        dict(target, depth=target['depth']*scale))
        torch.testing.assert_close(loss, scaled_loss, atol=1e-6, rtol=1e-5)
        for name in ('near', 'mid', 'far'):
            assert before[f'depth_objective_{name}_pixels'] == after[f'depth_objective_{name}_pixels']


def test_invalid_nan_and_padding_have_no_loss_or_gradient():
    pred, target = case()
    target['depth_valid'][..., 0, 0] = False
    target['depth'][..., 0, 0] = float('nan')
    pred['depth'][..., 0, 0] = float('nan')
    pred['depth'][..., 1, 1] *= 1.2
    pred['depth'].requires_grad_()
    _, loss = objectives(pred, target)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred['depth'].grad).all()
    assert not pred['depth'].grad[..., 0, 0].any()
    assert pred['depth'].grad[..., 1, 1].abs().sum() > 0


def test_all_invalid_returns_differentiable_zero_without_metadata():
    pred, target = case()
    target['depth_valid'].zero_()
    target['depth'].fill_(float('nan'))
    pred = {'depth': torch.full_like(pred['depth'], float('nan'), requires_grad=True)}
    terms, loss = objectives(pred, target)
    assert terms == {} and loss == 0 and loss.requires_grad
    loss.backward()
    assert torch.isfinite(pred['depth'].grad).all() and not pred['depth'].grad.any()


def test_near_zero_gt_uses_scale_adaptive_floor_with_finite_gradients():
    pred, target = case()
    target['depth'][..., 0, 0] = 1e-20
    pred['depth'][..., 0, 0] = .1
    pred['depth'].requires_grad_()
    terms, loss = objectives(pred, target)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred['depth'].grad).all()
    assert terms['depth_objective_floor_fraction'] > 0
    # Fixed normalized scene units give a bounded derivative despite GT=1e-20.
    assert pred['depth'].grad.abs().max() < 100


def test_confidence_diagnostics_cannot_change_objective_or_eligibility():
    pred, target = case()
    pred['depth'] *= 1.1
    a, loss = objectives(dict(pred, depth_conf=torch.ones_like(pred['depth'])), target)
    b, loss2 = objectives(dict(pred, depth_conf=torch.full_like(pred['depth'], 1e5)), target)
    torch.testing.assert_close(loss, loss2, rtol=0, atol=0)
    assert a['depth_objective_confidence_saturated_fraction'] == 0
    assert b['depth_objective_confidence_saturated_fraction'] == 1


def test_multicamera_anchor_stays_active_but_c1_temporal_is_not_fabricated():
    pred, target = case()
    pred['camera_id'] = torch.tensor([[3, 3, 4]])
    pred['depth'] *= torch.tensor([1., 1.5, 2.])[None, :, None, None]
    terms, _ = objectives(pred, target)
    assert terms['depth_objective_log_bias'] > 0
    assert terms['depth_objective_temporal_pairs'] == 0
    terms2, _ = objectives(pred, target, all_options(temporal_single_camera_only=False))
    assert terms2['depth_objective_temporal_pairs'] == 1
    # Even if camera 4 has no depth GT, the input remains physically multicamera.
    target['depth_valid'][:, 2].zero_()
    terms3, _ = objectives(pred, target)
    assert terms3['depth_objective_temporal_pairs'] == 0


def test_temporal_metadata_is_required_and_duplicate_times_rejected():
    pred, target = case()
    with pytest.raises(ValueError, match='camera_id and time_idx'):
        objectives({'depth': pred['depth']}, target)
    pred['time_idx'] = torch.tensor([[5, 0, 5]])
    with pytest.raises(ValueError, match='distinct times'):
        objectives(pred, target)


def test_temporal_uses_real_sorted_times_and_permutation_invariant():
    pred, target = case()
    pred['depth'] *= torch.tensor([1., 1.5, .9])[None, :, None, None]
    _, before = objectives(pred, target)
    ids = torch.tensor([2, 0, 1])
    p2 = {k: v[:, ids] for k, v in pred.items()}
    t2 = {k: v[:, ids] for k, v in target.items()}
    _, after = objectives(p2, t2)
    torch.testing.assert_close(before, after)


def test_default_and_explicit_disabled_are_legacy_bit_exact_and_roundtrip():
    pred, target = case()
    pred['depth'] = (pred['depth']*1.1).requires_grad_()
    base = MultiCamera4RCLoss()
    disabled = MultiCamera4RCLoss(depth_objective_options={})
    assert 'depth_objective_options' not in base.to_config()
    a, b = base(pred, target), disabled(pred, target)
    assert a.keys() == b.keys()
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    grad1 = torch.autograd.grad(a['loss'], pred['depth'], retain_graph=True)[0]
    grad2 = torch.autograd.grad(b['loss'], pred['depth'])[0]
    torch.testing.assert_close(grad1, grad2, rtol=0, atol=0)
    enabled = MultiCamera4RCLoss(depth_objective_options=asdict(all_options()))
    assert MultiCamera4RCLoss(**enabled.to_config()).to_config() == enabled.to_config()


def test_new_objectives_work_when_legacy_depth_weight_zero_and_resize_confidence():
    pred, target = case()
    pred['depth'] = torch.ones(1, 3, 1, 2, requires_grad=True)
    pred['depth_conf'] = torch.ones_like(pred['depth'])*1e5
    criterion = MultiCamera4RCLoss(depth_weight=0, depth_objective_options=asdict(all_options()))
    result = criterion(pred, target)
    result['loss'].backward()
    assert result['loss'] > 0
    assert result['depth_objective_confidence_saturated_fraction'] == 1
    assert torch.isfinite(pred['depth'].grad).all() and pred['depth'].grad.abs().sum() > 0


@pytest.mark.parametrize('key,value', [('relative_weight', -1), ('log_bias_weight', float('nan')),
    ('temporal_scale_weight', True), ('relative_floor_fraction', 0), ('relative_floor_fraction', 1),
    ('robust_beta', 0), ('near_ratio', 3), ('confidence_saturation', .1),
    ('temporal_single_camera_only', 1), ('diagnostics', 1)])
def test_invalid_configuration_rejected(key, value):
    with pytest.raises(ValueError):
        DepthObjectiveOptions(**{key: value})


@pytest.mark.parametrize('bad', [0., -1., float('nan'), float('inf')])
def test_invalid_positive_prediction_fails_instead_of_masking(bad):
    pred, target = case()
    pred['depth'][0, 0, 0, 0] = bad
    with pytest.raises(FloatingPointError, match='positive predictions'):
        objectives(pred, target)
