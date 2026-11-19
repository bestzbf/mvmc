import numpy as np
import torch
from multicamera4rc.datasets.adapters.base import UnifiedClip
from multicamera4rc.datasets.bridge import prepare_clip
from multicamera4rc.datasets.camera_only import attach_camera_tracks_recipe
from multicamera4rc.datasets.mixture import collate_adapter_samples
from multicamera4rc.datasets.dense import materialize_dense_targets


def test_sparse_tracks_keep_normalized_pose_and_unknown_visibility_separate():
    e=np.repeat(np.eye(4,dtype=np.float32)[None],2,axis=0);e[1,0,3]=.1
    k=np.repeat(np.array([[20.,0,13.5],[0,20,13.5],[0,0,1]],np.float32)[None],2,axis=0)
    xyz=np.array([[[0.,0,2.],[.1,0,2.]],[[.2,0,2.],[.3,0,2.]]],np.float32)
    camera=xyz-e[:,None,:3,3];uv=camera[...,:2]/camera[...,2:]*20+13.5
    valid=np.ones((2,2),bool)
    clip=UnifiedClip(dataset_name='stereo4d',sequence_name='fixture',frame_paths=None,
        images=[np.zeros((28,28,3),np.uint8)]*2,depths=None,normals=None,
        trajs_3d_world=xyz,trajs_2d=uv,valids=valid,visibs=valid,
        intrinsics=k,extrinsics=e,visibility_valids=valid,
        metadata=dict(extrinsics_convention='c2w',frame_indices=[0,2],supervision_kind='camera_tracks'))
    sample=prepare_clip(clip,image_mode='native',all_track_queries=True,include_visibility=False)
    attach_camera_tracks_recipe(clip,sample)
    batch=collate_adapter_samples([sample]);target=materialize_dense_targets(batch,torch.tensor([[0,1]]))
    assert not target['depth_valid'].any() and not target['trajectory_valid'].any()
    assert not target['sparse_visibility_valid'].any()
    assert target['sparse_trajectory_valid'].all() and target['sparse_displacement_valid'].all()
    scale=target['scene_scale'][0]
    torch.testing.assert_close(target['c2w'][0,1,0,3]*scale,torch.tensor(.1))
    values=target['sparse_displacement'][0]*scale
    assert torch.isclose(values.norm(dim=-1).max(),torch.tensor(.2))
    assert target['camera_translation_valid'].all()
    assert target['sparse_trajectory_is_pseudo'].tolist() == [True]


def test_unknown_sparse_visibility_does_not_require_disabled_head(monkeypatch):
    import pytest
    from multicamera4rc.losses import MultiCamera4RCLoss
    from multicamera4rc import tasks
    sampled = dict(query_valid=torch.ones(1, 2, dtype=torch.bool),
                   query_time_idx=torch.tensor([[0, 1]]), source_time_idx=torch.tensor([[0, 0]]))
    monkeypatch.setattr(tasks, 'sample_tracks', lambda *args, **kwargs: sampled)
    target = dict(sparse_queries=torch.zeros(1, 2, 3),
                  sparse_visibility=torch.zeros(1, 2, 2, 1),
                  sparse_visibility_valid=torch.zeros(1, 2, 2, 1, dtype=torch.bool))
    criterion = MultiCamera4RCLoss()
    terms, total, _, supervised = criterion._sparse_losses({}, target, torch.tensor(0.))
    assert terms['sparse_visibility'] == 0 and total == 0 and not supervised
    target['sparse_visibility_valid'].fill_(True)
    with pytest.raises(KeyError, match='sparse visibility needs'):
        criterion._sparse_losses({}, target, torch.tensor(0.))


def test_absolute_pseudo_tracks_remove_only_marked_base_gradients(monkeypatch):
    from multicamera4rc import tasks
    from multicamera4rc.losses import MultiCamera4RCLoss

    base = torch.full((2, 1, 2, 3), .75, requires_grad=True)
    displacement = torch.full_like(base, .25, requires_grad=True)
    query_times = torch.tensor([[0, 1], [0, 1]])
    monkeypatch.setattr(tasks, 'sample_tracks', lambda *_args, **_kwargs: dict(
        trajectory=base + displacement, displacement=displacement,
        query_valid=torch.ones(2, 1, dtype=torch.bool),
        query_time_idx=query_times,
        source_time_idx=torch.zeros(2, 1, dtype=torch.long)))
    pred = dict(query_time_idx=query_times)
    target = dict(sparse_queries=torch.zeros(2, 1, 3), sparse_trajectory=torch.zeros_like(base),
                  sparse_displacement=torch.zeros_like(base),
                  sparse_trajectory_is_pseudo=torch.tensor([True, False]))
    config = dict(trajectory_motion_gradient_scale=.25, motion_confidence_targets=())
    control, gated = MultiCamera4RCLoss(**config), MultiCamera4RCLoss(
        **config, pseudo_trajectory_base_gradient_scale=0.)
    output = []
    for criterion in (control, gated):
        terms, loss, _, _ = criterion._sparse_losses(pred, target, torch.tensor(0.))
        output.append(torch.autograd.grad(loss, (base, displacement)))
        assert terms['sparse_trajectory'] == 1. and terms['sparse_displacement'] == .25
        assert loss == 1.25
    assert output[0][0][0].abs().sum() > 0
    assert output[1][0][0].abs().sum() == 0
    torch.testing.assert_close(output[0][0][1], output[1][0][1], rtol=0, atol=0)
    torch.testing.assert_close(output[0][1], output[1][1], rtol=0, atol=0)
    assert 'pseudo_trajectory_base_gradient_scale' not in control.to_config()
    assert MultiCamera4RCLoss(**gated.to_config()).to_config() == gated.to_config()


def test_certified_stereo_uses_measured_pixels_and_only_updates_temporal_motion():
    from multicamera4rc.datasets.augmentation import crop_sample
    from multicamera4rc.distributed import _METRICS
    from multicamera4rc.losses import MultiCamera4RCLoss
    from multicamera4rc.reprojection import ReprojectionOptions, reprojection_losses

    pose = np.repeat(np.eye(4, dtype=np.float32)[None], 4, axis=0)
    pose[2:, 0, 3] = .063
    single_K = np.array([[20., 0, 27.5], [0, 20., 27.5], [0, 0, 1.]], np.float32)
    k = np.repeat(single_K[None], 4, axis=0)
    xyz = np.array([[[0., 0, 2.], [0., 0, 2.]],
                    [[.2, 0, 2.], [.2, 0, 2.]]] * 2, np.float32)
    camera = xyz - pose[:, None, :3, 3]
    uv = camera[..., :2] / camera[..., 2:] * 20 + 27.5
    valid = np.ones((4, 2), bool)
    # The RGB-matched endpoint deliberately differs from the projected pseudo XYZ.
    right_rgb = uv[2:].copy()
    right_rgb[1, :, 0] += 0.75
    clip = UnifiedClip(
        dataset_name='stereo4d', sequence_name='stereo_fixture', frame_paths=None,
        images=[np.zeros((56, 56, 3), np.uint8)] * 4, depths=None, normals=None,
        trajs_3d_world=xyz, trajs_2d=uv, valids=valid, visibs=valid,
        intrinsics=k, extrinsics=pose, visibility_valids=valid,
        metadata=dict(extrinsics_convention='c2w', frame_indices=[0, 2, 0, 2], time_idx=[0, 2, 0, 2],
                      slot_camera=[0, 0, 1, 1], supervision_kind='camera_tracks',
                      stereo_right_lk_uv=right_rgb, stereo_right_lk_valid=valid[2:],
                      stereo_left_temporal_edges=np.array([[True, False]])),
    )
    sample = prepare_clip(clip, image_mode='native', all_track_queries=True, include_visibility=False)
    attach_camera_tracks_recipe(clip, sample)
    structural = sample["targets"]
    mask = structural["sparse_stereo_uv_valid"]
    assert mask.nonzero().tolist() == [[0, 1, 1]]
    torch.testing.assert_close(structural["sparse_stereo_uv"][mask][0], torch.from_numpy(right_rgb[1, 0]))
    dropped = crop_sample(sample, [[0, 0, 28, 28]] * 4)
    assert not dropped["targets"]["sparse_stereo_uv_valid"].any()
    sample = crop_sample(sample, [[8, 8, 28, 28]] * 2 + [[10, 11, 28, 28]] * 2)
    assert not sample["targets"]["sparse_visibility_valid"].any()
    assert sample["targets"]["sparse_stereo_uv_valid"].sum() == 1
    updated_uv = torch.from_numpy(right_rgb[1, 0]) - torch.tensor([11., 10.])
    torch.testing.assert_close(sample["targets"]["sparse_stereo_uv"][0, 1, 1], updated_uv)

    idx = torch.tensor([[0]])
    batch = collate_adapter_samples([sample])
    target = materialize_dense_targets(batch, idx)
    # Match the 3D pseudo trajectory exactly. The observed RGB error must
    # still be 0.75 px, proving this isn't a self-reprojection objective.
    delta = target['sparse_displacement'][0, 0].view(1, 1, 2, 1, 1, 3)
    delta = delta.expand(1, 1, 2, 28, 28, 3).clone().requires_grad_()
    anchor = torch.randn_like(delta).requires_grad_()
    pK = target['K'].clone().requires_grad_()
    pc2w = target['c2w'].clone().requires_grad_()
    pred = dict(trajectory=anchor + delta, displacement=delta,
                motion_confidence=torch.ones_like(delta[..., 0]),
                K=pK, c2w=pc2w, camera_id=batch['camera_id'], time_idx=batch['time_idx'],
                query_time_idx=batch['query_time_idx'], source_obs_idx=idx,
                source_time_idx=torch.tensor([[0]]), observation_valid=batch['observation_valid'],
                source_valid=torch.ones_like(idx, dtype=torch.bool), output_camera_id=torch.tensor([[0, 1]]))
    terms, disabled = reprojection_losses(pred, target, ReprojectionOptions(weight=.05))
    assert not terms and disabled == 0
    default_config = MultiCamera4RCLoss(reprojection_options={'weight': .05}).to_config()
    assert 'stereo_certified_weight' not in default_config['reprojection_options']
    configured = MultiCamera4RCLoss(reprojection_options={'stereo_certified_weight': .02})
    options = configured.to_config()['reprojection_options']
    assert options['weight'] == 0 and options['stereo_certified_weight'] == .02
    terms, loss = reprojection_losses(
        pred, target, ReprojectionOptions(**options)
    )
    assert terms['joint_reprojection_stereo_pixels'] == 1
    assert all(metric in _METRICS for metric in terms)
    torch.testing.assert_close(terms['joint_reprojection_stereo_epe_px'], torch.tensor(.75))
    assert loss > 0
    loss.backward()
    assert delta.grad is not None and torch.isfinite(delta.grad).all() and delta.grad[:, :, 1].abs().sum() > 0
    assert not delta.grad[:, :, 0].any()
    assert anchor.grad is None and pK.grad is None and pc2w.grad is None

    # Check the production loss entry point too; no visibility head is present.
    options = {'stereo_certified_weight': .02}
    motion_weights = dict.fromkeys(MultiCamera4RCLoss().weights, 0.)
    complete = MultiCamera4RCLoss(weights=motion_weights, reprojection_options=options)(pred, target)
    torch.testing.assert_close(complete['loss'], loss.detach())
    assert not complete['sparse_visibility'].item()
