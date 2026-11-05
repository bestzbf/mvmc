import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
import pytest
import torch

spec = importlib.util.spec_from_file_location("uq_adapter", Path(__file__).with_name("predict_uniquery4r.py"))
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


@pytest.mark.skipif(not os.environ.get("UNIQUERY4R_REPO"), reason="Supply the pinned UniQuery4R source checkout")
def test_preprocessing_matches_author_pixels_and_retains_frames():
    from uniquery4r.utils.load_fn import load_and_preprocess_images
    rng = np.random.default_rng(812)
    rgb = rng.integers(0, 256, (4, 41, 73, 3), dtype=np.uint8)
    video = torch.from_numpy(rgb).permute(0, 3, 1, 2)[None]
    actual = adapter.prepare_video(video, max_size=56)
    with TemporaryDirectory() as tmp:
        paths = []
        for i, frame in enumerate(rgb):
            p = Path(tmp) / f"{i}.png"
            Image.fromarray(frame).save(p)
            paths.append(str(p))
        expected = load_and_preprocess_images(paths, max_size=56, num_workers=1)
    assert torch.equal(actual, expected)
    assert actual.shape == (4, 3, 28, 56)


def test_trajectory_uses_src0_pairs_reconstruction_uses_self_pairs():
    pairs = [(0, 1), (0, 2), (0, 0), (1, 1), (2, 2)]
    warp = torch.stack([torch.full((3, 2, 4), float(i)) for i in range(5)])[None]
    pred = {"pair_idx": pairs, "warp3d": warp}
    traj = adapter.dense_predictions(pred, frames=3, task="trajectory")
    recon = adapter.dense_predictions(pred, frames=3, task="reconstruction")
    assert traj.shape == (3, 2, 4, 3)
    assert np.array_equal(traj[:, 0, 0, 0], [2, 0, 1])
    assert np.array_equal(recon[:, 0, 0, 0], [2, 3, 4])
    with pytest.raises(ValueError, match="pair indexing"):
        adapter.dense_predictions({"pair_idx": pairs[::-1], "warp3d": warp}, frames=3, task="trajectory")


def test_nonfinite_predictions_are_fatal():
    warp = torch.zeros(1, 1, 3, 2, 2)
    warp[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        adapter.dense_predictions({"pair_idx": [(0, 0)], "warp3d": warp}, frames=1, task="trajectory")


def test_manifest_rejects_nonworldtrack_and_other_model_binding():
    good = {"benchmark": "worldtrack", "coordinates": "first_camera_world",
            "official_sources": {"commit": adapter.SCORER_COMMIT},
            "protocol": {"task": "trajectory"}, "cases": [
                {"views": [0], "frames": 64, "height": 280, "width": 504}], "leakage": {}}
    assert adapter.validate_manifest(good) == "trajectory"
    with pytest.raises(ValueError):
        adapter.validate_manifest(dict(good, benchmark="tapvid3d"))
    with pytest.raises(ValueError, match="another model"):
        adapter.validate_manifest(dict(good, leakage={"checkpoint_sha256": "ours"}))


@pytest.mark.skipif(not os.environ.get("UNIQUERY4R_REPO"), reason="Supply the pinned UniQuery4R source checkout")
def test_contract_restores_shared_runner_and_omits_fabricated_cameras():
    from types import SimpleNamespace
    baseline = SimpleNamespace(_prediction_specs=object())
    original = baseline._prediction_specs
    manifest = {"benchmark": "worldtrack", "coordinates": "first_camera_world",
                "official_sources": {"commit": adapter.SCORER_COMMIT}, "protocol": {"task": "trajectory"},
                "leakage": {}, "cases": []}
    case = {"frames": 3, "height": 41, "width": 73}
    with adapter.dense_contract(baseline, 56):
        result = baseline._prediction_specs(manifest, case, None)
        assert result == {"tracks_XYZ": (3, 28, 56, 3), "visibility": (3, 28, 56)}
    assert baseline._prediction_specs is original
    assert adapter.SUPPORT == {"position": True, "native_visibility": False, "AJ": False, "OA": False}
