import io
import json
import tarfile

import numpy as np
import pytest

from multicamera4rc.datasets.adapters.kinoptic_motion_candidates import (
    estimate_surface_candidates, official_joint_pair,
)


def plane():
    h, w = 48, 64
    images = np.full((2, 2, h, w, 3), 100, np.uint8)
    depth = np.full((2, 2, h, w), 2., np.float32)
    K = np.tile(np.array([[80., 0, 31.5], [0, 80., 23.5], [0, 0, 1.]]), (2, 2, 1, 1))
    E = np.tile(np.eye(4), (2, 2, 1, 1)); E[1, :, 0, 3] = -.1
    flow = np.zeros((2, 2, h, w, 2), np.float32)
    flow[:, 0, ..., 0] = 2.; flow[:, 1, ..., 0] = -2.
    return images, depth, K, E, flow


def run_plane(**overrides):
    image, depth, K, E, flow = plane()
    values = dict(images=image, depths=depth, intrinsics=K, extrinsics=E,
                  times_ms=[1000, 1100], flows=flow)
    values.update(overrides)
    return estimate_surface_candidates(**values)


def test_known_world_translation_and_schema():
    out = run_plane()
    assert len(out['candidate_delta_world_m']) > 2000
    np.testing.assert_allclose(out['candidate_delta_world_m'],
                               np.tile([.05, 0, 0], (len(out['candidate_delta_world_m']), 1)), atol=1e-6)
    assert out['candidate_cross_view_error_px'].max() < 1e-6
    assert out['candidate_delta_agreement_m'].max() < 1e-6
    assert not out['surface_correspondence_gt'] and not out['training_admitted']
    assert out['pseudo_label'] and not out['visibility_annotation_available']
    assert not any(k.startswith(('dense_pair_', 'sparse_', 'trajectory', 'displacement')) for k in out)


def test_cycle_failure_rejects_false_flow():
    *_, flow = plane(); flow[0, 1, ..., 0] = 2.
    assert len(run_plane(flows=flow)['candidate_delta_world_m']) == 0


def test_cross_view_motion_disagreement_rejected():
    *_, flow = plane(); flow[1, 0, ..., 0] = 7.; flow[1, 1, ..., 0] = -7.
    assert len(run_plane(flows=flow)['candidate_delta_world_m']) == 0


def test_depth_noise_cannot_create_motion_without_rgb_evidence():
    _, depth, _, _, flow = plane(); flow.fill(0)
    depth[:, 1] += .025
    # Geometric agreement alone accepts this shared depth perturbation.
    ungated = run_plane(depths=depth, flows=flow, min_independent_motion_px=0)
    assert len(ungated['candidate_delta_world_m']) > 2000
    assert np.median(np.linalg.norm(ungated['candidate_delta_world_m'], axis=-1)) > .02
    assert len(run_plane(depths=depth, flows=flow)['candidate_delta_world_m']) == 0


def test_same_camera_is_not_an_independent_witness():
    _, _, _, E, _ = plane(); E[1] = E[0]
    with pytest.raises(ValueError): run_plane(extrinsics=E)


@pytest.mark.parametrize('camera,time', [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_missing_measurement_is_not_inpainted(camera, time):
    _, depth, *_ = plane(); depth[camera, time] = 0.
    assert len(run_plane(depths=depth)['candidate_delta_world_m']) == 0


@pytest.mark.parametrize('time', [0, 1])
def test_other_view_depth_contradiction_is_rejected(time):
    _, depth, *_ = plane(); depth[1, time] += 1.
    assert len(run_plane(depths=depth)['candidate_delta_world_m']) == 0


def test_unknown_pixel_support_and_region_are_excluded():
    _, depth, *_ = plane()
    support = np.ones(depth.shape, bool); support[1, 1] = False
    assert len(run_plane(pixel_support=support)['candidate_delta_world_m']) == 0
    assert len(run_plane(region_mask=np.zeros(depth.shape[2:], bool))['candidate_delta_world_m']) == 0


@pytest.mark.parametrize('times', [[1000, 1000], [1100, 1000], [1000, 1600], [1000, float('nan')]])
def test_invalid_temporal_pair_rejected(times):
    with pytest.raises(ValueError): run_plane(times_ms=times)


def archive(tmp_path, *, duplicate_id=False, second_id=4, offset_cm=5., bad_time=False):
    path = tmp_path/'bodies.tar'
    with tarfile.open(path, 'w') as tf:
        for frame, x, identity in [(10, 100., 4), (13, 100.+offset_cm, second_id)]:
            points = np.tile([x, 20., 300., .8], (19, 1))
            points[2, 3] = .1  # Known low-confidence joint must not acquire a label.
            body = dict(id=identity, joints19=points.flatten().tolist())
            doc = dict(univTime=(1000. if frame == 10 else 1100.) + (10 if bad_time else 0),
                       bodies=[body, body] if duplicate_id else [body])
            raw = json.dumps(doc).encode(); member = tarfile.TarInfo(f'hdPose3d_stage1_coco19/body3DScene_{frame:08d}.json')
            member.size = len(raw); tf.addfile(member, io.BytesIO(raw))
    return path


def test_official_joint_units_identity_confidence_and_provenance(tmp_path):
    out = official_joint_pair(archive(tmp_path), [10, 13], [1000, 1100])
    assert len(out['keys']) == 18
    np.testing.assert_allclose(out['joint_delta_world_m'], np.tile([.05, 0, 0], (18, 1)), atol=1e-6)
    assert out['joint_start_world_m'][0, 2] == 3.
    assert all(k[0] == 4 and k[2] != 2 for k in out['keys'])
    assert all(len(h) == 64 for h in out['member_sha256'])
    assert not out['surface_correspondence_gt'] and not out['visibility_annotation_available']


def test_person_ids_never_matched_by_array_order(tmp_path):
    out = official_joint_pair(archive(tmp_path, second_id=7), [10, 13], [1000, 1100])
    assert out['joint_delta_world_m'].shape == (0, 3)


def test_implausible_identity_jump_rejected(tmp_path):
    out = official_joint_pair(archive(tmp_path, offset_cm=1000), [10, 13], [1000, 1100])
    assert len(out['keys']) == 0 and out['rejected_speed'] == 18


@pytest.mark.parametrize('option', ['bad_time', 'duplicate_id'])
def test_ambiguous_or_wrong_timestamp_annotation_rejected(tmp_path, option):
    with pytest.raises(ValueError):
        official_joint_pair(archive(tmp_path, **{option: True}), [10, 13], [1000, 1100])
