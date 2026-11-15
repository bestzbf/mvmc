import numpy as np
import torch

from datasets.transforms import _resize_intrinsics, _resize_pixel_coordinates
from scripts.depth_alignment_audit import least_squares_alignment, alignment_audit, content_canvas_intrinsics


def test_partial_geometry_selects_supported_views_without_creating_labels():
    from multicamera4rc.train import _sources
    valid = torch.zeros(1, 6, 10, 10, dtype=torch.bool)
    valid[:, 1] = True
    valid[:, 4] = True
    batch = dict(observation_valid=torch.ones(1, 6, dtype=torch.bool),
                 camera_id=torch.tensor([[0, 1, 2, 0, 1, 2]]),
                 time_idx=torch.tensor([[0, 0, 0, 1, 1, 1]]),
                 images=torch.zeros(1, 6, 3, 10, 10),
                 query_time_idx=torch.tensor([[0, 1]]),
                 targets=dict(dense_partial=torch.tensor([True]), depth_valid=valid))
    for seed in range(8):
        torch.manual_seed(seed)
        chosen = _sources(batch, 2, 64, True, True, 64)
        assert set(chosen[0].tolist()) == {1, 4}
    assert int(valid.sum()) == 200
    # A real motion label still takes precedence over depth density.
    batch['targets']['dense_motion_source_valid'] = torch.tensor([[False, False, True, False, False, False]])
    assert _sources(batch, 1, 64, True, False, 64).item() == 2


def test_resize_intrinsics_and_points_preserve_the_same_rays():
    k = np.array([[500., 3., 299.3], [0, 480., 217.2], [0, 0, 1]], np.float32)
    xyz = np.array([[.3, .1, 1], [-.2, -.4, 2], [0., 0., 3.]])
    uv = xyz @ k.T; uv = uv[:, :2]/uv[:, 2:]
    changed = _resize_intrinsics(k, 439, 601, 224, 518)
    projected = xyz @ changed.T; projected = projected[:, :2]/projected[:, 2:]
    scaled = _resize_pixel_coordinates(uv, [518/601, 224/439])
    np.testing.assert_allclose(projected, scaled, atol=3e-5)
    np.testing.assert_allclose(_resize_pixel_coordinates(scaled, [601/518, 439/224]), uv, atol=6e-5)
    np.testing.assert_array_equal(_resize_intrinsics(k, 439, 601, 439, 601), k)


def test_lstsq_known_affine_and_constant_prediction():
    p = torch.tensor([[[1., 2.], [3., 4.]]]); g = 2*p+3; mask = torch.ones_like(p, dtype=torch.bool)
    aligned, _ = least_squares_alignment(p, g, mask, 100.)
    torch.testing.assert_close(aligned, g)
    constant, _ = least_squares_alignment(torch.ones_like(p), g, mask, 100.)
    torch.testing.assert_close(constant, g.mean().expand_as(g))


def test_failed_ss_pixels_remain_scored():
    p = torch.tensor([[[.01, 2., 3., 4.]]]); g = torch.tensor([[[.1, 1., 2., 3.]]])
    result = alignment_audit(p, g, torch.ones_like(p, dtype=torch.bool), 100.)
    assert not result['formal_scores_changed'] and not result['prediction_failures_excluded']
    assert {row['valid_pixels'] for row in result['variants'].values()} == {4}


def test_canvas_ablation_preserves_focal_and_learned_principal_offsets():
    k = torch.tensor([[[300., 0, 260.], [0, 300., 115.], [0, 0, 1.]]])
    # Here K is already on the content-resolution grid.
    changed = content_canvas_intrinsics(k, padded_hw=(224, 518), content_hw=(221, 518), eval_hw=(221, 518))
    torch.testing.assert_close(changed[..., 0, 0], k[..., 0, 0])
    torch.testing.assert_close(changed[..., 1, 1], k[..., 1, 1])
    torch.testing.assert_close(changed[..., 0, 2], k[..., 0, 2])
    assert changed[0, 1, 2] < k[0, 1, 2]
    torch.testing.assert_close((changed[..., 1, 2]-221/2)/221, (k[..., 1, 2]-224/2)/224)


def test_focal_ratio_uses_calibrated_ratio_and_unknown_cameras_are_masked():
    from multicamera4rc.losses import MultiCamera4RCLoss
    criterion = MultiCamera4RCLoss(intrinsics_loss_mode='log_focal', focal_loss_weight=0.,
                                  principal_loss_weight=0., focal_ratio_loss_weight=.5)
    gt = torch.tensor([[400., 0., 20.], [0., 200., 10.], [0., 0., 1.]]).expand(1, 2, 3, 3).clone()
    pred = gt.clone(); pred[:, 0, 0, 0] *= 1.2; pred[:, 0, 1, 1] *= .8
    pred[:, 1] *= 10; pred.requires_grad_()
    terms = criterion(dict(K=pred, depth=torch.ones(1, 2, 20, 40)),
                      dict(K=gt, camera_valid=torch.tensor([[True, False]])))
    torch.testing.assert_close(terms['camera_log_focal_ratio'], torch.tensor(1.5).log())
    terms['loss'].backward()
    assert pred.grad[0, 0, 0, 0] > 0 and pred.grad[0, 0, 1, 1] < 0
    assert not pred.grad[0, 1].any()
    scaled = gt.clone(); scaled[..., 0, 0] *= 3; scaled[..., 1, 1] *= 3
    exact = criterion(dict(K=scaled, depth=torch.ones(1, 2, 20, 40)), dict(K=gt))
    torch.testing.assert_close(exact['camera_log_focal_ratio'], torch.tensor(0.), atol=1e-6, rtol=0)
    assert MultiCamera4RCLoss(**criterion.to_config()).focal_ratio_loss_weight == .5
    assert 'focal_ratio_loss_weight' not in MultiCamera4RCLoss().to_config()
