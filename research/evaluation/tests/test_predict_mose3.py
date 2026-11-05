"""CPU-only synthetic contract checks; these do not validate trained weights."""
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

checkout = os.environ.get("MOSE3_REPO")
if not checkout:
    pytest.skip("Supply the pinned MoSE3 source checkout", allow_module_level=True)
root = Path(checkout)
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location("mose3_adapter", Path(__file__).with_name("predict_mose3.py"))
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class FakeMoSE3:
    def __init__(self):
        self.sources = []

    def __call__(self, images, query_frame_idx):
        self.sources.append(query_frame_idx)
        _, t, _, h, w = images.shape
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        points = torch.stack((xx.float() + query_frame_idx * 100, yy.float(), torch.ones_like(xx).float()), -1)
        points = points[None, None].expand(1, t, -1, -1, -1).clone()
        poses = torch.eye(4).repeat(1, t, 1, 1)
        poses[0, :, 0, 3] = torch.arange(t).float()
        logits = torch.ones((1, t, h, w))
        logits[:, 1] = -1
        return {"tracks_local": points, "visibility_logits": logits,
                "camera_poses": poses, "track_focal": torch.tensor([[14., 14.]])}


def test_all_source_frames_coordinates_visibility_and_gauges():
    queries = np.array([[3, 2, 1, 0], [-2, 3, 0, 0], [30, 7, 1, 0]], np.float32)
    inp = SimpleNamespace(video=torch.zeros((1, 3, 3, 14, 14), dtype=torch.uint8),
                          queries_xytv=torch.from_numpy(queries))
    model = FakeMoSE3()
    arrays, meta = adapter.predict_case(model, inp, device="cpu", max_side=28)
    assert model.sources == [0, 1]
    assert arrays["tracks_XYZ"].shape == (3, 3, 3)
    np.testing.assert_allclose(arrays["tracks_XYZ"][0, :, 0], [103, 0, 113], atol=1e-5)
    np.testing.assert_allclose(arrays["tracks_native_world"][2, :, 0], [105, 2, 115], atol=1e-5)
    assert arrays["visibility"].tolist() == [[True]*3, [False]*3, [True]*3]
    assert arrays["predicted_K"].shape == (1, 3, 3, 3)
    np.testing.assert_allclose(arrays["predicted_c2w"][0, :, 0, 3], [0, 1, 2])
    assert meta["source_frames"] == [0, 1]


def test_pixel_center_resize_and_border_extension():
    grid = torch.arange(14 * 28, dtype=torch.float32).reshape(1, 14, 28, 1)
    xy = np.array([[0, 0], [13, 6], [-2, 0], [100, 0]], np.float32)
    result = adapter.sample_dense(grid, xy, 14, 7)[0, :, 0].numpy()
    # 2x input resize: original first pixel center is between four resized pixels.
    np.testing.assert_allclose(result, [14.5, 376.5, 14, 41], atol=1e-5)


def test_rejects_multicamera_and_bad_source():
    inp = SimpleNamespace(video=torch.zeros((2, 3, 3, 14, 14), dtype=torch.uint8),
                          queries_xytv=torch.tensor([[1., 2., 0., 0.]]))
    with pytest.raises(ValueError, match="single-camera"):
        adapter.predict_case(FakeMoSE3(), inp, device="cpu")
    inp.video = inp.video[:1]
    inp.queries_xytv[0, 2] = 3
    with pytest.raises(ValueError, match="Invalid source"):
        adapter.predict_case(FakeMoSE3(), inp, device="cpu")


def test_lossless_framework_float_rgb_preserves_author_uint8_pipeline():
    video = torch.arange(3 * 3 * 14 * 14).remainder(256).reshape(1, 3, 3, 14, 14).to(torch.uint8)
    queries = torch.tensor([[3., 2., 0., 0.], [10., 9., 2., 0.]])
    class Capture(FakeMoSE3):
        def __call__(self, images, query_frame_idx):
            self.images = images.clone()
            return super().__call__(images, query_frame_idx)
    byte_model, float_model = Capture(), Capture()
    byte_out, _ = adapter.predict_case(byte_model, SimpleNamespace(video=video, queries_xytv=queries), device="cpu", max_side=28)
    float_out, _ = adapter.predict_case(float_model, SimpleNamespace(video=video.float(), queries_xytv=queries), device="cpu", max_side=28)
    assert torch.equal(byte_model.images, float_model.images)
    for key in byte_out:
        np.testing.assert_array_equal(byte_out[key], float_out[key])


@pytest.mark.parametrize("bad", [float("nan"), -1., 256., .5])
def test_rejects_rgb_that_cannot_be_recovered_without_changing_pixels(bad):
    video = torch.zeros(1, 1, 3, 14, 14)
    video[0, 0, 0, 0, 0] = bad
    with pytest.raises(ValueError, match="RGB"):
        adapter.lossless_rgb_uint8(video)


def test_rejects_boolean_rgb():
    with pytest.raises(ValueError, match="RGB"):
        adapter.lossless_rgb_uint8(torch.zeros(1, 1, 3, 14, 14, dtype=torch.bool))
