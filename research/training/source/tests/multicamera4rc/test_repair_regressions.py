"""Regression tests for coordinate, crop, load-balance, and reprojection fixes."""

import torch
import pytest

from multicamera4rc.datasets.augmentation import crop_sample
from multicamera4rc.geometry import unproject_depth
from multicamera4rc.load_balance import sample_cost
from multicamera4rc.reprojection import ReprojectionOptions, reprojection_losses


def test_crop_sample_recomputes_dense_motion_source_valid():
    sample = dict(
        clip_id="crop_valid",
        images=torch.zeros(2, 3, 40, 60),
        camera_id=torch.tensor([0, 0]),
        time_idx=torch.tensor([0, 1]),
        query_time_idx=torch.tensor([0, 1]),
        observation_valid=torch.ones(2, dtype=torch.bool),
        targets=dict(
            K=torch.eye(3).repeat(2, 1, 1),
            c2w=torch.eye(4).repeat(2, 1, 1),
            joint_motion_queries=torch.tensor([[0.0, 4.0, 4.0], [1.0, 50.0, 20.0]]),
            joint_motion_query_valid=torch.tensor([True, True]),
            joint_motion_valid=torch.ones(2, 2, dtype=torch.bool),
            pseudo_motion_queries=torch.tensor([[1.0, 5.0, 5.0]]),
            pseudo_motion_query_valid=torch.tensor([False]),
            pseudo_motion_valid=torch.zeros(1, 2, dtype=torch.bool),
            dense_motion_source_valid=torch.tensor([True, True]),
        ),
    )

    left_crop = crop_sample(sample, [[0, 0, 28, 28], [0, 0, 28, 28]])
    assert left_crop["targets"]["dense_motion_source_valid"].tolist() == [True, False]

    right_crop = crop_sample(sample, [[0, 30, 28, 28], [0, 30, 28, 28]])
    assert right_crop["targets"]["dense_motion_source_valid"].tolist() == [False, True]


def test_sample_cost_recognizes_all_motion_dense_kinds():
    class _Dataset:
        def __init__(self, dense_kind):
            self.dense_kind = dense_kind

        def sample_plan(self, _index):
            return dict(
                source=dict(
                    display_name="src",
                    view_mode="temporal",
                    dense_kind=self.dense_kind,
                    camera_choices=[2],
                ),
                num_times=4,
                cameras=2,
            )

    image_hw = {"src": (224, 518)}
    geometry_cost = sample_cost(_Dataset("geometry"), 0, image_hw)
    for kind in ("camera_tracks", "geometry_tracks", "kinoptic_human", "kubric"):
        assert sample_cost(_Dataset(kind), 0, image_hw) > geometry_cost


def test_reprojection_cross_resolution_scales_pred_k_and_resizes_trajectory():
    k_lo = torch.tensor([[4.0, 0.0, 1.5], [0.0, 4.0, 1.0], [0.0, 0.0, 1.0]]).expand(1, 2, 3, 3).clone()
    k_hi = torch.tensor([[8.0, 0.0, 3.5], [0.0, 8.0, 2.5], [0.0, 0.0, 1.0]]).expand(1, 2, 3, 3).clone()
    e = torch.eye(4).expand(1, 2, 4, 4).clone()
    e[:, 1, 0, 3] = 0.2

    depth_lo = torch.full((1, 2, 3, 4), 2.0)
    xyz_lo = unproject_depth(depth_lo, k_lo, e)[:, :1, None].expand(1, 1, 2, 3, 4, 3).clone()
    depth_hi = torch.full((1, 2, 6, 8), 2.0)
    xyz_hi = unproject_depth(depth_hi, k_hi, e)[:, :1, None].expand(1, 1, 2, 6, 8, 3).clone()

    vis_valid = torch.zeros(1, 1, 2, 1, 6, 8, dtype=torch.bool)
    vis_valid[..., 1:-1, 1:-1] = True

    pred = dict(
        trajectory=xyz_lo.clone(),
        displacement=torch.zeros_like(xyz_lo),
        motion_confidence=torch.ones(xyz_lo.shape[:-1]),
        K=k_lo.clone(),
        c2w=e.clone(),
        camera_id=torch.tensor([[7, 7]]),
        time_idx=torch.tensor([[0, 9]]),
        query_time_idx=torch.tensor([[0, 9]]),
        source_obs_idx=torch.tensor([[0]]),
        source_time_idx=torch.tensor([[0]]),
        source_valid=torch.ones(1, 1, dtype=torch.bool),
        observation_valid=torch.ones(1, 2, dtype=torch.bool),
        output_camera_id=torch.tensor([[7]]),
    )
    target = dict(
        K=k_hi,
        c2w=e,
        trajectory=xyz_hi,
        trajectory_valid=torch.ones(xyz_hi.shape[:-1], dtype=torch.bool),
        visibility=torch.ones(1, 1, 2, 1, 6, 8),
        visibility_valid=vis_valid,
    )

    terms, loss = reprojection_losses(pred, target, ReprojectionOptions(weight=0.1))
    assert terms["joint_reprojection_pixels"] > 0
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    assert terms["joint_reprojection_epe_px"] == pytest.approx(0.0, abs=1e-5)

    pred_err = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in pred.items()}
    pred_err["trajectory"] = pred_err["trajectory"] + torch.tensor([0.1, 0.0, 0.0])
    for name in ("trajectory", "K", "c2w"):
        pred_err[name].requires_grad_()
    terms_err, loss_err = reprojection_losses(pred_err, target, ReprojectionOptions(weight=0.1))
    assert terms_err["joint_reprojection_epe_px"] > 0.1
    loss_err.backward()
    for name in ("trajectory", "K", "c2w"):
        assert torch.isfinite(pred_err[name].grad).all() and pred_err[name].grad.abs().sum() > 0
