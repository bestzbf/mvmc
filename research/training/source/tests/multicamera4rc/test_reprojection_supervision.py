from copy import deepcopy
from dataclasses import asdict

import pytest
import torch

from multicamera4rc.geometry import unproject_depth
from multicamera4rc.reprojection import ReprojectionOptions, reprojection_losses
from multicamera4rc.losses import MultiCamera4RCLoss
from multicamera4rc.schedule import LearningRatePlan, rewarm_group_plan
from multicamera4rc.distributed import _METRICS


def case():
    k = torch.tensor([[4., 0, 1.5], [0, 4., 1.], [0, 0, 1.]]).expand(1, 2, 3, 3).clone()
    e = torch.eye(4).expand(1, 2, 4, 4).clone()
    e[:, 1, 0, 3] = .2
    depth = torch.full((1, 2, 3, 4), 2.)
    base = unproject_depth(depth, k, e)
    xyz = base[:, :1, None].expand(1, 1, 2, 3, 4, 3).clone()
    pred = dict(trajectory=xyz.clone(), displacement=torch.zeros_like(xyz),
                motion_confidence=torch.ones(xyz.shape[:-1]), K=k.clone(), c2w=e.clone(),
                camera_id=torch.tensor([[7, 7]]), time_idx=torch.tensor([[0, 9]]),
                query_time_idx=torch.tensor([[0, 9]]), source_obs_idx=torch.tensor([[0]]),
                source_time_idx=torch.tensor([[0]]),
                source_valid=torch.ones(1, 1, dtype=torch.bool),
                observation_valid=torch.ones(1, 2, dtype=torch.bool), output_camera_id=torch.tensor([[7]]))
    target = dict(K=k, c2w=e, trajectory=xyz, trajectory_valid=torch.ones(xyz.shape[:-1], dtype=torch.bool),
                  visibility=torch.ones(1, 1, 2, 1, 3, 4),
                  visibility_valid=torch.ones(1, 1, 2, 1, 3, 4, dtype=torch.bool))
    return pred, target


def test_exact_gt_and_source_self_projection():
    p, t = case(); p['trajectory'].requires_grad_()
    terms, loss = reprojection_losses(p, t, ReprojectionOptions(weight=.1))
    assert terms['joint_reprojection_views'] == 1  # source identity is excluded
    assert terms['joint_reprojection_pixels'] == 9  # one column is outside target view
    assert float(loss) == pytest.approx(0, abs=1e-7)
    loss.backward(); assert torch.isfinite(p['trajectory'].grad).all()
    assert not p['trajectory'].grad[:, :, 0].any()


def test_real_direction_error_reaches_trajectory_and_camera_without_predicted_visibility():
    p, t = case(); p['trajectory'] += torch.tensor([.15, .04, 0.])
    for name in ('trajectory', 'K', 'c2w'): p[name].requires_grad_()
    p['visibility'] = torch.zeros(1, 1, 2, 1, 3, 4)
    terms, loss = reprojection_losses(p, t, ReprojectionOptions(weight=.1))
    assert terms['joint_reprojection_epe_px'] > 0
    loss.backward()
    for name in ('trajectory', 'K', 'c2w'):
        assert torch.isfinite(p[name].grad).all() and p[name].grad.abs().sum() > 0
    assert all(key in _METRICS for key in terms)


def test_unknown_gt_and_occlusion_do_not_become_zero_motion_labels():
    p, t = case(); t['visibility_valid'].zero_()
    p['trajectory'].fill_(float('nan'))
    terms, loss = reprojection_losses(p, t, ReprojectionOptions(weight=.1))
    assert not terms and loss == 0
    t['visibility_valid'].fill_(True); t['visibility'].zero_()
    terms, loss = reprojection_losses(p, t, ReprojectionOptions(weight=.1))
    assert not terms and loss == 0


def test_behind_camera_is_penalized_and_not_filtered():
    p, t = case(); p['trajectory'][..., 2] = -2; p['trajectory'].requires_grad_()
    terms, loss = reprojection_losses(p, t, ReprojectionOptions(weight=.1))
    assert terms['joint_reprojection_pixels'] == 9
    assert terms['joint_reprojection_behind_fraction'] == 1
    assert loss > 0; loss.backward()
    assert torch.isfinite(p['trajectory'].grad).all()
    assert p['trajectory'].grad[..., 2].sum() < 0  # optimizer pulls points forward


def test_unit_change_and_pixel_budget():
    p, t = case(); p['trajectory'][..., 0] += .2
    opts = ReprojectionOptions(weight=.1, max_points_per_view=3)
    before, loss = reprojection_losses(p, t, opts)
    for factor in (1e-3, 100):
        pp, tt = deepcopy(p), deepcopy(t)
        for d in (pp, tt): d['trajectory'] *= factor; d['c2w'][..., :3, 3] *= factor
        after, scaled = reprojection_losses(pp, tt, opts)
        torch.testing.assert_close(loss, scaled)
        assert before['joint_reprojection_pixels'] == after['joint_reprojection_pixels'] == 3


def test_sparse_correspondence_works_in_dense_recipe():
    p, t = case(); truth = t.pop('trajectory')[:, 0, :, 1, 2][:, None]
    del t['trajectory_valid']; del t['visibility']; del t['visibility_valid']
    t.update(sparse_queries=torch.tensor([[[0., 2., 1.]]]), sparse_query_valid=torch.ones(1, 1, dtype=torch.bool),
             sparse_trajectory=truth, sparse_trajectory_valid=torch.ones(1, 1, 2, dtype=torch.bool),
             sparse_visibility=torch.ones(1, 1, 2, 1), sparse_visibility_valid=torch.ones(1, 1, 2, 1, dtype=torch.bool))
    p['trajectory'][..., 0] += .1; p['trajectory'].requires_grad_()
    terms, loss = reprojection_losses(p, t, ReprojectionOptions(weight=.1))
    assert terms['joint_reprojection_sparse_pixels'] == 1 and terms['joint_reprojection_dense_pixels'] == 0
    loss.backward(); assert p['trajectory'].grad.abs().sum() > 0


def test_loss_config_and_continuation_lr_resume_are_explicit():
    old = MultiCamera4RCLoss().to_config(); assert 'reprojection_options' not in old
    criterion = MultiCamera4RCLoss(reprojection_options=asdict(ReprojectionOptions(weight=.05)))
    assert MultiCamera4RCLoss(**criterion.to_config()).to_config() == criterion.to_config()
    plan = LearningRatePlan('cosine', 95000, 200000, [1e-7, 1e-6, 1e-4, 1e-6],
                            group_names=['backbone', 'heads', 'refinement', 'geometry'],
                            group_multipliers={'geometry':dict(start_step=140500, multiplier=2., warmup_steps=100)})
    ramp = rewarm_group_plan(plan, 'geometry', start_step=210500, factor=1.5, warmup_steps=500)
    restored = LearningRatePlan(**ramp.to_config())
    for step, wanted in ((210500, 2e-7), (210750, 2.5e-7), (211000, 3e-7), (220000, 3e-7)):
        assert restored.base_lrs[-1]*restored.factor(step)*restored.group_factor('geometry', step) == pytest.approx(wanted)
        for i, name in enumerate(plan.group_names[:-1]):
            assert restored.base_lrs[i]*restored.factor(step)*restored.group_factor(name, step) == plan.base_lrs[i]*plan.factor(step)*plan.group_factor(name, step)


def test_reprojection_two_rank_training_and_saved_resume(tmp_path):
    import json
    from torch import distributed as dist
    if not dist.is_available() or not dist.is_gloo_available():
        import pytest
        pytest.skip('Gloo unavailable')
    from test_geometry_generalization_ddp import _torchrun, native_config
    from multicamera4rc.losses import MultiCamera4RCLoss
    config = native_config(tmp_path)
    data = json.loads(config.read_text())
    data['sources'][0]['dense_kind'] = 'geometry_tracks'
    data['sources'][0]['options'].update(prefer_precomputed_tracks=True, require_precomputed_tracks=True)
    config.write_text(json.dumps(data))
    loss = MultiCamera4RCLoss.dense().to_config()
    loss['reprojection_options'] = dict(weight=.05, max_points_per_view=64)
    loss_path = tmp_path/'reprojection-loss.json'
    loss_path.write_text(json.dumps(loss))
    first = tmp_path/'first'
    records, logs = _torchrun(tmp_path, config, first, 1, loss_config=loss_path)
    assert records[0]['digest'] == records[1]['digest']
    assert all(r['model_finite'] for r in records)
    assert any(r.get('joint_reprojection_pixels', 0) > 0 for r in logs)
    second = tmp_path/'second'
    records, logs = _torchrun(tmp_path, config, second, 2, resume=first/'latest.pt')
    assert records[0]['digest'] == records[1]['digest']
    assert all(r['model_finite'] for r in records)
    assert any(r.get('joint_reprojection_pixels', 0) > 0 for r in logs)
