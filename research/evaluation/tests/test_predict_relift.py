import importlib.util
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

spec = importlib.util.spec_from_file_location("relift_adapter", Path(__file__).with_name("predict_relift.py"))
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


@pytest.mark.skipif(not os.environ.get("RELIFT_REPO"), reason="Supply the pinned ReLift and DA3 sources")
def test_native_size_matches_real_da3_processor_including_rounding_ties():
    from relift.vendor import front_ends
    front_ends()
    from depth_anything_3.utils.io.input_processor import InputProcessor
    processor = InputProcessor()
    for h, w in [(512, 512), (1080, 1920), (41, 73), (377, 504), (49, 504), (21, 504)]:
        image = Image.new("RGB", (w, h))
        image = processor._resize_image(image, 504, "upper_bound_resize")
        image = processor._make_divisible_by_resize(image, 14)
        assert adapter.native_hw(h, w) == (image.height, image.width)


def test_prepared_window_matches_author_float16_disk_roundtrip():
    t, h, w = 3, 2, 4
    rng = np.random.default_rng(43)
    geometry = {"depth": rng.random((t, h, w)),
                "intrinsics": np.tile(np.eye(3)[None], (t, 1, 1)),
                "poses": np.tile(np.eye(4)[None], (t, 1, 1))}
    tracks = rng.random((t, h, w, 2)) * 10
    vis = rng.random((t, h, w))
    result = adapter.prepared_window(geometry, tracks, vis, "cpu")
    assert (result["T"], result["H"], result["W"]) == (t, h, w)
    assert torch.equal(result["tracks"], torch.from_numpy(tracks.astype(np.float16).astype(np.float32)))
    assert torch.equal(result["depth"], torch.from_numpy(geometry["depth"].astype(np.float16).astype(np.float32)))
    assert torch.equal(result["vis"], torch.from_numpy(vis.astype(np.float16).astype(np.float32)))
    assert all(result[k].dtype == torch.float32 for k in ["tracks", "vis", "depth", "intrinsics", "poses"])
    with pytest.raises(ValueError, match="shape mismatch"):
        adapter.prepared_window(geometry, tracks[:-1], vis, "cpu")


def test_reconstruction_and_nonworldtrack_are_explicitly_unsupported():
    manifest = {"benchmark": "worldtrack", "coordinates": "first_camera_world",
        "official_sources": {"commit": adapter.SCORER_COMMIT},
        "protocol": {"task": "trajectory"}, "leakage": {},
        "cases": [{"views": [0], "frames": 64}]}
    assert adapter.validate_manifest(manifest) == "trajectory"
    with pytest.raises(ValueError, match="trajectories only"):
        adapter.validate_manifest(dict(manifest, protocol={"task": "reconstruction"}))
    with pytest.raises(ValueError, match="another model"):
        adapter.validate_manifest(dict(manifest, leakage={"checkpoint_sha256": "ours"}))


def test_native_contract_has_only_truthful_position_support():
    from types import SimpleNamespace
    baseline = SimpleNamespace(_prediction_specs=object())
    original = baseline._prediction_specs
    manifest = {"benchmark": "worldtrack", "coordinates": "first_camera_world",
        "official_sources": {"commit": adapter.SCORER_COMMIT},
        "protocol": {"task": "trajectory"}, "leakage": {}, "cases": []}
    case = {"height": 512, "width": 512, "frames": 64}
    with adapter.dense_contract(baseline):
        assert baseline._prediction_specs(manifest, case, None) == {
            "tracks_XYZ": (64, 504, 504, 3), "visibility": (64, 504, 504)}
    assert baseline._prediction_specs is original
    assert adapter.SUPPORT["native_visibility"] is False
    assert not adapter.SUPPORT["AJ"] and not adapter.SUPPORT["OA"]


def test_strict_safetensors_loader_resolves_only_real_shared_parameters(tmp_path):
    from safetensors.torch import save_model
    model = torch.nn.Module()
    node = model
    for name in ("model", "da3", "head", "scratch"):
        child = torch.nn.Module()
        setattr(node, name, child)
        node = child
    shared = torch.nn.LayerNorm(3)
    node.output_conv2_aux = torch.nn.ModuleList([
        torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity(), shared) for _ in range(4)])
    path = tmp_path / "shared.safetensors"
    save_model(model, str(path))
    receipt = adapter.load_da3_strict(model, path)
    assert receipt["missing_keys"] == receipt["unexpected_keys"] == []
    assert len(receipt["verified_serialization_aliases"]) == 6
    node.output_conv2_aux[1][2] = torch.nn.LayerNorm(3)
    with pytest.raises(ValueError, match="same Parameter"):
        adapter.load_da3_strict(model, path)


def test_foreign_utils_cache_is_restored_and_deferred_author_imports_stay_private(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    (tmp_path / "utils").mkdir()
    (tmp_path / "utils/misc.py").write_text("def marker():\n return 'author'\n")
    (tmp_path / "utils/deferred.py").write_text("VALUE = 47\n")
    foreign = ModuleType("utils")
    foreign.__file__ = "/foreign_frontend/utils/__init__.py"
    foreign_misc = ModuleType("utils.misc")
    foreign_misc.__file__ = "/foreign_frontend/utils/misc.py"
    foreign.misc = foreign_misc
    monkeypatch.setitem(sys.modules, "utils", foreign)
    monkeypatch.setitem(sys.modules, "utils.misc", foreign_misc)
    namespace = adapter.AuthorModuleNamespace(tmp_path)
    with namespace.activate():
        import utils.misc
        assert utils.misc.marker() == "author"
    assert sys.modules["utils"] is foreign and sys.modules["utils.misc"] is foreign_misc
    with namespace.activate():
        import utils.deferred
        assert utils.deferred.VALUE == 47
        assert sys.modules["utils.misc"].marker() == "author"
    assert sys.modules["utils"] is foreign and sys.modules["utils.misc"] is foreign_misc
    assert "utils.deferred" not in sys.modules
