"""Diagnostic arrays and formulas remain separate from frozen benchmark scores."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from scripts.geometry_diagnostics import (archive_prediction, depth_diagnostics,
                                          prefix_clip, verify_prediction)
from scripts.monocular_protocol import BenchmarkClip, geometry_metrics


def clip():
    depth = torch.tensor([[[.2, .5, 1.], [2., 4., 8.]]]).repeat(3, 1, 1)
    K = torch.tensor([[10., 0, 1.], [0, 10., .5], [0, 0, 1.]]).repeat(3, 1, 1)
    pose = torch.eye(4).repeat(3, 1, 1)
    pose[:, 0, 3] = torch.tensor([0., .1, .3])
    return BenchmarkClip(torch.zeros(1, 3, 3, 2, 3), depth, K, pose,
                         [0, 5, 11], (2, 3), (2, 3), 'synthetic')


def test_archived_arrays_roundtrip_and_tampering_is_rejected(tmp_path):
    data = clip(); prediction = (data.depth, data.K, data.c2w)
    path = tmp_path/'predictions/example.npz'
    digest = archive_prediction(path, prediction, data.frame_indices)
    verify_prediction(path, digest, data.frame_indices)
    with np.load(path) as arrays:
        np.testing.assert_array_equal(arrays['depth'], data.depth.numpy())
    with pytest.raises(ValueError, match='frame identity'):
        verify_prediction(path, digest, [0, 5, 12])
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='changed'):
        verify_prediction(path, digest, data.frame_indices)


def test_raw_upper_clamp_can_be_diagnosed_without_changing_official_formula():
    data = clip()
    prediction = data.depth*1000
    before = geometry_metrics(prediction, data.K, data.c2w, data, dataset='sintel')
    result = depth_diagnostics(prediction, data.K, data.c2w, data, dataset='sintel')
    assert result['raw_above_limit_fraction'] == 1.
    assert result['scale_abs_rel_without_raw_upper_clamp'] < 1e-6
    assert before['depth_S_abs_rel'] > .1
    after = geometry_metrics(prediction, data.K, data.c2w, data, dataset='sintel')
    assert before == after


def test_diagnostics_detect_scale_drift_and_keep_near_far_support():
    data = clip()
    prediction = data.depth*torch.tensor([1., 2., 4.])[:, None, None]
    result = depth_diagnostics(prediction, data.K, data.c2w, data, dataset='sintel')
    assert result['raw_scale_log_std'] > .1
    assert result['raw_scale_max']/result['raw_scale_min'] == pytest.approx(4)
    assert result['depth_buckets']['near']['pixels'] > 0
    assert result['depth_buckets']['far']['pixels'] > 0
    assert all(row['frame'] in data.frame_indices for row in result['frames'])
    assert result['focal_abs_rel'] == 0 and result['principal_error_eval_pixels'] == 0


def test_prefix_preserves_identical_scoring_frames_for_context_comparisons():
    data = clip(); short = prefix_clip(data, 2)
    assert short.frame_indices == [0, 5]
    torch.testing.assert_close(short.depth, data.depth[:2], rtol=0, atol=0)
    assert short.images.shape[1] == 2 and data.images.shape[1] == 3
    with pytest.raises(ValueError):
        prefix_clip(data, 4)


def test_no_invalid_prediction_can_be_hidden_by_diagnostics():
    data = clip(); prediction = data.depth.clone(); prediction[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='invalid model depths'):
        depth_diagnostics(prediction, data.K, data.c2w, data, dataset='sintel')
