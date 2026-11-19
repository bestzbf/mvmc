import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from multicamera4rc.datasets.bridge import prepare_clip
from multicamera4rc.losses import MultiCamera4RCLoss


@pytest.mark.parametrize("use_camera_encoding", [False, True])
def test_component_camera_loss_preserves_half_pixel_intrinsics(use_camera_encoding):
    k_pred = torch.tensor([[[[18., 0., 10.], [0., 15., 6.], [0., 0., 1.]]]])
    transform = torch.tensor([[4., 0., 1.5], [0., 2., .5], [0., 0., 1.]])
    k_gt = transform @ k_pred
    pred = dict(K=k_pred.clone().requires_grad_(), depth=torch.zeros(1, 1, 12, 20))
    target = dict(K=k_gt, depth=torch.zeros(1, 1, 24, 80))
    if use_camera_encoding:
        pred["camera_encoding"] = torch.zeros(1, 1, 11)
        pred["camera_encoding"][..., 7:9] = torch.tensor(
            [2 * math.atan(6 / 15), 2 * math.atan(10 / 18)]
        )
    criterion = MultiCamera4RCLoss(depth_weight=0., camera_loss_mode="components")
    exact = criterion(pred, target)
    assert exact["camera_principal"].item() == pytest.approx(0., abs=1e-7)
    assert exact["camera_fov"].item() == pytest.approx(0., abs=1e-7)

    moved = pred["K"].detach().clone()
    moved[..., 0, 2] += .25  # one GT pixel horizontally
    pred["K"] = moved.requires_grad_()
    terms = criterion(pred, target)
    assert terms["camera_principal"].item() == pytest.approx(.5 / 80)
    terms["loss"].backward()
    assert torch.isfinite(pred["K"].grad).all()
    assert pred["K"].grad[..., 0, 2].item() > 0.


def test_bridge_keeps_nonfinite_tracks_out_of_source_sampling():
    world = np.array([[[0., 0., 2.], [0., 0., 2.], [np.inf, 0., 2.], [0., 0., 2.]],
                      [[0., 0., 2.], [.25, 0., 2.], [-np.inf, 0., 2.], [0., 0., 2.]]])
    uv = np.array([[[13.5, 13.5], [13.5, 13.5], [13.5, 13.5], [np.inf, 1.]],
                   [[13.5, 13.5], [np.nan, 13.5], [13.5, 13.5], [13.5, -np.inf]]])
    source_world, source_uv = world.copy(), uv.copy()
    clip = SimpleNamespace(
        images=[np.zeros((28, 28, 3), dtype=np.uint8) for _ in range(2)],
        intrinsics=np.tile([[24., 0., 13.5], [0., 24., 13.5], [0., 0., 1.]], (2, 1, 1)),
        extrinsics=np.eye(4)[None].repeat(2, axis=0),
        trajs_3d_world=world,
        trajs_2d=uv,
        valids=np.ones((2, 4), dtype=bool),
        visibs=np.ones((2, 4), dtype=bool),
    )
    with np.errstate(invalid="raise"):
        target = prepare_clip(
            clip, image_size=(56, 42), image_mode="resize", all_track_queries=True
        )["targets"]

    assert target["sparse_queries"].tolist() == [
        [0., 20.5, 27.5], [0., 20.5, 27.5], [1., 20.5, 27.5]
    ]
    assert target["sparse_track_id"].tolist() == [0, 1, 0]
    assert target["sparse_trajectory_valid"].all()
    assert target["sparse_trajectory"][1, 1].tolist() == [.25, 0., 2.]
    np.testing.assert_equal(world, source_world)
    np.testing.assert_equal(uv, source_uv)


def test_sparse_and_auxiliary_losses_scale_queries_across_resolutions():
    from multicamera4rc.auxiliary_motion import AuxiliaryMotionOptions, auxiliary_motion_losses

    pred_h, pred_w = 12, 20
    gt_h, gt_w = 24, 80
    # Build a spatially linear field on pred canvas: x(u,v) = (u+0.5)/pred_w, y(u,v) = (v+0.5)/pred_h
    yy, xx = torch.meshgrid(
        (torch.arange(pred_h, dtype=torch.float32) + 0.5) / pred_h,
        (torch.arange(pred_w, dtype=torch.float32) + 0.5) / pred_w,
        indexing="ij",
    )
    field = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)  # [H, W, 3]
    disp_t1 = field * 0.5
    disp = torch.stack((torch.zeros_like(disp_t1), disp_t1), dim=0)[None, None]  # [1, 1, 2, H, W, 3]
    traj = field[None, None, None] + disp
    disp = disp.clone().requires_grad_()
    traj = (field[None, None, None] + disp)
    pred = dict(
        trajectory=traj,
        displacement=disp,
        motion_confidence=torch.ones(1, 1, 2, pred_h, pred_w),
        source_obs_idx=torch.tensor([[0]]),
        source_valid=torch.tensor([[True]]),
        source_time_idx=torch.tensor([[0]]),
        query_time_idx=torch.tensor([[0, 1]]),
        observation_valid=torch.tensor([[True, True]]),
        camera_id=torch.tensor([[0, 0]]),
        time_idx=torch.tensor([[0, 1]]),
    )
    # Query at GT center (u_gt=39.5, v_gt=11.5) -> normalized (0.5, 0.5); plus one out-of-GT-bounds query (-0.25, 11.5)
    u_gt, v_gt = 39.5, 11.5
    expected_base = torch.tensor([(u_gt + 0.5) / gt_w, (v_gt + 0.5) / gt_h, 1.0])
    expected_disp = expected_base * 0.5
    sparse_queries = torch.tensor([[[0.0, u_gt, v_gt], [0.0, -0.25, v_gt]]])
    orig_queries = sparse_queries.clone()
    target = dict(
        depth=torch.ones(1, 2, gt_h, gt_w),
        depth_valid=torch.zeros(1, 2, gt_h, gt_w, dtype=torch.bool),
        sparse_queries=sparse_queries,
        sparse_query_valid=torch.tensor([[True, True]]),
        sparse_trajectory=torch.stack(
            (torch.stack((expected_base, expected_base + expected_disp)),
             torch.zeros(2, 3)),
            dim=0,
        )[None],
        sparse_trajectory_valid=torch.tensor([[[True, True], [True, True]]]),
        sparse_displacement=torch.stack(
            (torch.stack((torch.zeros(3), expected_disp)),
             torch.zeros(2, 3)),
            dim=0,
        )[None],
        sparse_displacement_valid=torch.tensor([[[True, True], [True, True]]]),
        joint_motion_queries=sparse_queries.clone(),
        joint_motion_query_valid=torch.tensor([[True, True]]),
        joint_motion_displacement=torch.stack(
            (torch.stack((torch.zeros(3), expected_disp)),
             torch.zeros(2, 3)),
            dim=0,
        )[None],
        joint_motion_valid=torch.tensor([[[False, True], [False, True]]]),
        joint_motion_confidence=torch.ones(1, 2, 2),
    )
    criterion = MultiCamera4RCLoss(
        depth_weight=0.0, camera_weight=0.0, ray_weight=0.0,
        trajectory_weight=1.0, displacement_weight=1.0, visibility_weight=0.0,
        static_weight=0.0, motion_confidence_targets=(),
    )
    out = criterion(pred, target)
    torch.testing.assert_close(target["sparse_queries"], orig_queries, rtol=0, atol=0)
    assert out["native_track_queries"].item() == 1.0
    assert out["sparse_trajectory"].item() == pytest.approx(0.0, abs=1e-6)
    assert out["sparse_displacement"].item() == pytest.approx(0.0, abs=1e-6)

    aux_terms, aux_total, aux_sup = auxiliary_motion_losses(
        pred, target, AuxiliaryMotionOptions(joint_weight=0.2)
    )
    assert aux_sup is True
    assert aux_terms["kinoptic_joint_motion_points"].item() == 1.0
    assert aux_terms["kinoptic_joint_motion_loss"].item() == pytest.approx(0.0, abs=1e-6)
    assert aux_total.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_no_grad_supervised_detection_for_joint_and_reprojection():
    eye4 = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    c2w = eye4.clone()
    c2w[0, 1, 0, 3] = 0.2
    k = torch.tensor([[[20.0, 0.0, 7.5], [0.0, 20.0, 7.5], [0.0, 0.0, 1.0]]]).repeat(1, 2, 1, 1)
    pred = dict(
        depth=torch.full((1, 2, 16, 16), 2.0),
        K=k.clone(),
        c2w=c2w.clone(),
        trajectory=torch.zeros(1, 1, 1, 16, 16, 3),
        displacement=torch.zeros(1, 1, 1, 16, 16, 3),
        motion_confidence=torch.ones(1, 1, 1, 16, 16),
        camera_id=torch.tensor([[0, 1]]),
        time_idx=torch.tensor([[0, 0]]),
        query_time_idx=torch.tensor([[0]]),
        source_obs_idx=torch.tensor([[0]]),
        source_valid=torch.tensor([[True]]),
        source_time_idx=torch.tensor([[0]]),
        observation_valid=torch.tensor([[True, True]]),
        output_camera_id=torch.tensor([[0, 1]]),
    )
    target = dict(
        depth=torch.full((1, 2, 16, 16), 2.0),
        depth_valid=torch.ones(1, 2, 16, 16, dtype=torch.bool),
        K=k.clone(),
        c2w=c2w.clone(),
        camera_valid=torch.tensor([[True, True]]),
        camera_translation_valid=torch.tensor([[True, True]]),
        trajectory=torch.tensor([0.0, 0.0, 2.0]).view(1, 1, 1, 1, 1, 3).expand(1, 1, 1, 16, 16, 3).clone(),
        trajectory_valid=torch.ones(1, 1, 1, 16, 16, dtype=torch.bool),
        visibility=torch.ones(1, 1, 1, 2, 16, 16),
        visibility_valid=torch.ones(1, 1, 1, 2, 16, 16, dtype=torch.bool),
        scene_scale=torch.tensor([1.0]),
    )
    zero_weights = dict.fromkeys(MultiCamera4RCLoss().weights, 0.0)
    with torch.no_grad():
        joint_loss = MultiCamera4RCLoss(
            weights=zero_weights,
            joint_geometry_options={"relative_pose_weight": 0.1},
        )(pred, target)
        assert joint_loss["joint_rotation_pairs"].item() == 1.0
        assert joint_loss["loss"].item() == pytest.approx(0.0, abs=1e-6)

        reproj_loss = MultiCamera4RCLoss(
            weights=zero_weights,
            reprojection_options={"weight": 0.05},
        )(pred, target)
        assert reproj_loss["joint_reprojection_pixels"].item() > 0

        # Diagnostics-only joint_geometry with zero weights must still reject unsupervised batches.
        with pytest.raises(ValueError, match="no valid enabled"):
            MultiCamera4RCLoss(
                weights=zero_weights,
                joint_geometry_options={"diagnostics": True},
            )(pred, target)


def test_dense_metrics_supports_cross_resolution_predictions():
    from multicamera4rc.metrics import dense_metrics

    pred_h, pred_w = 12, 20
    gt_h, gt_w = 24, 40
    pred = dict(
        depth=torch.full((1, 2, pred_h, pred_w), 2.0),
        depth_conf=torch.full((1, 2, pred_h, pred_w), 2.0),
        base_points=torch.tensor([0.0, 0.0, 2.0]).view(1, 1, 1, 1, 3).expand(1, 2, pred_h, pred_w, 3).clone(),
        trajectory=torch.tensor([0.0, 0.0, 2.0]).view(1, 1, 1, 1, 1, 3).expand(1, 1, 2, pred_h, pred_w, 3).clone(),
        displacement=torch.zeros(1, 1, 2, pred_h, pred_w, 3),
        motion_confidence=torch.full((1, 1, 2, pred_h, pred_w), 2.0),
        visibility=torch.ones(1, 1, 2, 1, pred_h, pred_w),
        K=torch.tensor([[[10.0, 0.0, 9.5], [0.0, 10.0, 5.5], [0.0, 0.0, 1.0]]]).repeat(1, 2, 1, 1),
        c2w=torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1),
        source_time_idx=torch.tensor([[0]]),
        query_time_idx=torch.tensor([[0, 1]]),
        observation_valid=torch.tensor([[True, True]]),
    )
    ray = torch.zeros(1, 2, gt_h, gt_w, 6)
    ray[..., 2] = 1.0
    target = dict(
        depth=torch.full((1, 2, gt_h, gt_w), 2.0),
        depth_valid=torch.ones(1, 2, gt_h, gt_w, dtype=torch.bool),
        ray=ray,
        trajectory=torch.tensor([0.0, 0.0, 2.0]).view(1, 1, 1, 1, 1, 3).expand(1, 1, 2, gt_h, gt_w, 3).clone(),
        trajectory_valid=torch.ones(1, 1, 2, gt_h, gt_w, dtype=torch.bool),
        displacement=torch.zeros(1, 1, 2, gt_h, gt_w, 3),
        displacement_valid=torch.ones(1, 1, 2, gt_h, gt_w, dtype=torch.bool),
        visibility=torch.ones(1, 1, 2, 1, gt_h, gt_w),
        visibility_valid=torch.ones(1, 1, 2, 1, gt_h, gt_w, dtype=torch.bool),
        K=torch.tensor([[[20.0, 0.0, 19.5], [0.0, 20.0, 11.5], [0.0, 0.0, 1.0]]]).repeat(1, 2, 1, 1),
        c2w=torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1),
        camera_valid=torch.tensor([[True, True]]),
    )
    m = dense_metrics(pred, target)
    assert m["base_epe_normalized"].item() == pytest.approx(0.0, abs=1e-6)
    assert m["trajectory_epe_normalized"].item() == pytest.approx(0.0, abs=1e-6)
    assert m["velocity_epe_normalized"].item() == pytest.approx(0.0, abs=1e-6)
    assert m["visibility_f1"].item() == pytest.approx(1.0, abs=1e-6)
