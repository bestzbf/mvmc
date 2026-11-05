"""CPU-only contract tests for the direct ARROW adapter.

Set ARROW_REPO to a clean checkout at the pinned author commit before running.
The fake model has no learned weights and implements only the public encode / 
decode_iter surface; prediction arrays and coordinate conversions are exercised
through ARROW's real Queries, image-transform, pose-query, and pose-fit code.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest


ARROW_ROOT = Path(os.environ.get("ARROW_REPO", ""))
if not (ARROW_ROOT / "src/arrow/types.py").is_file():
    pytest.skip("Set ARROW_REPO to the pinned ARROW source checkout", allow_module_level=True)
import sys
sys.path.insert(0, str(ARROW_ROOT / "src"))

import torch
from arrow.types import Predictions

SCRIPT = Path(__file__).with_name("predict_arrow.py")
SPEC = importlib.util.spec_from_file_location("arrow_adapter_under_test", SCRIPT)
adapter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(adapter)


@pytest.fixture
def queue_verifier():
    path = Path(__file__).resolve().parent.parent / "legacy_comparison_identity.py"
    assert path.is_file(), "Missing staged legacy comparison identity code"
    spec = importlib.util.spec_from_file_location("arrow_queue_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def synthetic_checkpoint(tmp_path, monkeypatch):
    """Small real files and a clean real git repo exercise identity validation."""
    source, weights = tmp_path / "source", tmp_path / "weights"
    source.mkdir()
    weights.mkdir()
    (source / "native.py").write_text("RELEASE = 'fixture'\n")
    (weights / "model.safetensors").write_bytes(b"small weight bytes for contract test")
    (weights / "config.json").write_text('{"model": "fixture"}\n')
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "native.py"], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Contract Test",
                    "-c", "user.email=contract-test@example.invalid", "commit", "-qm", "fixture"], check=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setattr(adapter, "ARROW_COMMIT", revision)
    monkeypatch.setattr(adapter, "ARROW_WEIGHT_SHA256", adapter._sha256(weights / "model.safetensors"))
    monkeypatch.setattr(adapter, "ARROW_CONFIG_SHA256", adapter._sha256(weights / "config.json"))
    record = adapter.checkpoint_record(source, weights)
    return source, weights, record


class SyntheticArrow:
    """A deterministic geometric stand-in; no CUDA or pretrained weights."""
    def __init__(self):
        self.seen = []

    def encode(self, images):
        return SimpleNamespace(images=images, num_frames=images.shape[1], device=images.device)

    def decode_iter(self, encoded, queries, chunk_size):
        q = queries.batched().cpu()
        self.seen.append(q)
        frames = int(encoded.num_frames)
        h, w = encoded.images.shape[-2:]
        for start in range(0, q.num_queries, chunk_size):
            part = q.slice(start, start + chunk_size)
            x, y = part.xy[0].unbind(-1)
            target = part.target_frame[0]
            ref = part.reference_frame[0]
            z = torch.full_like(x, 2.0)
            px = (x - (w - 1) / 2) * z / 8.0
            py = (y - (h - 1) / 2) * z / 8.0
            points = torch.stack((px, py, z), -1)
            # Distinct camera translations make the pose fit observable.
            delta = target.float() * 0.2
            points[:, 0] += torch.where(ref == 0, delta, torch.zeros_like(delta))
            uv = torch.stack((x / (w - 1), y / (h - 1)), -1)
            yield Predictions(
                points=points[None],
                point_confidence=torch.ones((1, len(x))) * 2,
                uv=uv[None],
                visibility_logits=torch.full((1, len(x)), 2.0),
                normals=torch.zeros((1, len(x), 3)),
                displacement=torch.zeros((1, len(x), 3)),
            )


def make_inputs(views, steps, height=16, width=20, queries=None):
    if queries is None:
        queries = np.array([[3.25, 4.5, 1, 0], [width - 1.25, 8, 0, views - 1]], np.float32)
    return SimpleNamespace(
        video=torch.arange(views * steps * 3 * height * width, dtype=torch.int64)
            .remainder(256).to(torch.uint8).reshape(views, steps, 3, height, width),
        queries_xytv=torch.as_tensor(queries),
    )


def test_query_layout_retains_source_time_and_target_camera_references():
    from arrow.types import Queries

    raw = np.array([[-0.49, 4, 1, 0], [17.49, 8, 0, 0]], np.float32)
    xy = torch.from_numpy(raw[:, :2])
    tap, source, count = adapter.query_layout(raw, xy, views=1, steps=3, benchmark="tapvid3d")
    assert count == 3
    assert source.tolist() == [1, 0]
    assert torch.equal(tap.source_frame, torch.tensor([1, 1, 1, 0, 0, 0]))
    assert torch.equal(tap.target_frame, torch.tensor([0, 1, 2, 0, 1, 2]))
    assert torch.equal(tap.reference_frame, tap.target_frame)
    # No clipping: both unusual-but-valid border queries survive construction.
    assert torch.equal(tap.xy[[0, 3]], xy)

    raw_mv = np.array([[1.5, 3, 0, 1], [9, 5, 1, 0]], np.float32)
    mv, source, count = adapter.query_layout(
        raw_mv, torch.from_numpy(raw_mv[:, :2]), views=2, steps=2, benchmark="mvtracker"
    )
    assert count == 4
    assert source.tolist() == [1, 2]  # time-major frame = t * V + v
    assert mv.source_frame.tolist() == [1, 1, 1, 1, 2, 2, 2, 2]
    assert mv.target_frame.tolist() == [0, 1, 2, 3] * 2
    assert mv.reference_frame.tolist() == [0] * 8


def test_tapvid_direct_queries_return_per_target_camera_tracks_and_pose_arrays():
    model = SyntheticArrow()
    inp = make_inputs(1, 3, queries=np.array([[-0.49, 4, 1, 0], [18.49, 8, 0, 0]], np.float32))
    arrays, meta = adapter.predict_case(model, inp, benchmark="tapvid3d", max_side=28,
                                        decode_chunk_size=3, pose_grid_size=4, device="cpu")
    assert arrays["tracks_XYZ"].shape == (3, 2, 3)
    assert arrays["visibility"].shape == (3, 2)
    assert arrays["predicted_c2w"].shape == (1, 3, 4, 4)
    assert arrays["predicted_K"].shape == (1, 3, 3, 3)
    assert arrays["visibility"].all()
    assert meta["coordinates"] == "per-target-camera coordinates"
    # The query at original time 1 is passed to every target without source-time
    # substitution or clipping; all target references are their own cameras.
    first_queries = model.seen[0]
    assert first_queries.source_frame[0].reshape(2, 3)[0].tolist() == [1, 1, 1]
    assert first_queries.reference_frame[0].reshape(2, 3)[0].tolist() == [0, 1, 2]
    # 20x16 is resized to 28x28 by the author's patch alignment. The negative
    # half-border seed remains negative after pixel-center transformation.
    assert first_queries.xy[0, 0, 0].item() == pytest.approx(-0.49 * 1.4 + 0.2)
    # Separate ref-0 native-gauge predictions are not mislabeled as target-camera
    # coordinates. Target 1 has a known synthetic 0.2 m gauge difference.
    assert not np.allclose(arrays["tracks_XYZ"][1], arrays["tracks_native_world"][1])
    assert np.allclose(arrays["tracks_native_world"][1, :, 0] - arrays["tracks_XYZ"][1, :, 0], 0.2)
    assert np.isfinite(arrays["predicted_K"]).all()
    expected_k = np.array([[8 / 1.4, 0, 9.5], [0, 8 / 1.75, 7.5], [0, 0, 1]])
    assert np.allclose(arrays["predicted_K"][0], expected_k, atol=1e-5)


def test_mvtracker_reduces_all_target_views_in_frame_zero_gauge():
    model = SyntheticArrow()
    inp = make_inputs(2, 2, queries=np.array([[3, 4, 0, 1], [9, 7, 1, 0]], np.float32))
    arrays, meta = adapter.predict_case(model, inp, benchmark="mvtracker", max_side=28,
                                        decode_chunk_size=4, pose_grid_size=4, device="cpu")
    assert arrays["tracks_XYZ"].shape == (2, 2, 3)
    assert arrays["visibility"].shape == (2, 2)
    assert arrays["predicted_c2w"].shape == (2, 2, 4, 4)
    assert arrays["predicted_K"].shape == (2, 2, 3, 3)
    assert arrays["visibility"].all()
    assert np.isfinite(arrays["tracks_XYZ"]).all()
    assert meta["multiview_target_reduction"] == "arithmetic mean over target cameras; no GT weighting"
    first_queries = model.seen[0]
    assert first_queries.reference_frame.unique().tolist() == [0]
    # Flattened time-major target indices advance by two across synchronized
    # timesteps in this synthetic model.
    q = arrays["tracks_XYZ"][:, 0, 0]
    assert np.isclose(q[1] - q[0], 0.4, atol=1e-5)


def test_multiview_reduction_is_unweighted_mean_and_any_positive_visibility():
    points = np.arange(2 * 3 * 2 * 3, dtype=np.float32).reshape(2, 6, 3)
    logits = np.array([
        [-1, 1, 1, -1, -1, -1],
        [1, 1, -1, -1, 0, 0],
    ], dtype=np.float32)
    tracks, visible = adapter._reduce_multiview_tracks(points, logits, n=2, steps=3, views=2)
    expected = points.reshape(2, 3, 2, 3).mean(axis=2).transpose(1, 0, 2)
    assert np.array_equal(tracks, expected)
    assert visible.tolist() == [[True, True], [True, False], [False, False]]


def test_checkpoint_record_passes_real_queue_identity_and_detects_tampering(
        synthetic_checkpoint, queue_verifier):
    source, weights, record = synthetic_checkpoint
    component = record["components"]["model"]
    assert component["path"] == str((weights / "model.safetensors").resolve())
    assert component["source_commit"] == adapter.ARROW_COMMIT
    assert {a["path"] for a in queue_verifier.baseline_artifacts(record)} == {
        str((weights / "model.safetensors").resolve()), str((weights / "config.json").resolve())
    }
    job = {"identity_kind": "baseline", "method": "ARROW",
           "expected_baseline_checkpoint": record}
    queue_verifier.verify_identity(job, record, queue_verifier.FileHashes())
    (weights / "config.json").write_text("tampered config")
    with pytest.raises(ValueError, match="checkpoint bytes differ"):
        queue_verifier.verify_identity(job, record, queue_verifier.FileHashes())
    (weights / "config.json").write_text('{"model": "fixture"}\n')
    (source / "native.py").write_text("TAMPERED = True\n")
    with pytest.raises(ValueError, match="tracked edits"):
        queue_verifier.verify_identity(job, record, queue_verifier.FileHashes())


def test_checkpoint_record_rejects_modified_weight_or_config(synthetic_checkpoint):
    source, weights, _ = synthetic_checkpoint
    (weights / "model.safetensors").write_bytes(b"wrong weights")
    with pytest.raises(ValueError, match="checkpoint does not match"):
        adapter.checkpoint_record(source, weights)
    (weights / "model.safetensors").write_bytes(b"small weight bytes for contract test")
    (weights / "config.json").write_text("wrong config")
    with pytest.raises(ValueError, match="config does not match"):
        adapter.checkpoint_record(source, weights)


def test_float_cache_rgb_is_lossless_and_fractional_rejected():
    x = torch.arange(84, dtype=torch.uint8).reshape(2, 2, 3, 1, 7)
    a, va, ta = adapter._flatten_video(x)
    b, vb, tb = adapter._flatten_video(x.float())
    np.testing.assert_array_equal(a, b)
    assert (va, ta) == (vb, tb)
    bad = x.float(); bad[0, 0, 0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="integer pixel"):
        adapter._flatten_video(bad)
