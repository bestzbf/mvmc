import os
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from predict_point4d import prepare_inputs, build_queries, _source_xyz, predict_case


class ContractTests(unittest.TestCase):
    def test_source_target_reference_layout(self):
        xyz = np.array([[2, 4, 6], [8, 10, 12]], np.float32)
        uv = np.array([[1, 2], [3, 4]], np.float32)
        q, pixels = build_queries(xyz, uv, np.array([2, 1]), frames=3, scale=2)
        np.testing.assert_array_equal(q[:, 3], [2, 2, 2, 1, 1, 1])
        np.testing.assert_array_equal(q[:, 4], q[:, 3])
        np.testing.assert_array_equal(q[:, 5], [0, 1, 2, 0, 1, 2])
        np.testing.assert_array_equal(q[:, 6], q[:, 5])
        np.testing.assert_array_equal(q[:, :3], np.repeat(xyz / 2, 3, 0))
        q0, _ = build_queries(xyz, uv, np.array([2, 1]), frames=3, scale=2, native=True)
        self.assertTrue((q0[:, 6] == 0).all())
        np.testing.assert_array_equal(pixels, np.repeat(uv, 3, 0))

    def test_lossless_float_rgb_and_pixel_center(self):
        inp = SimpleNamespace(video=torch.arange(72).reshape(1, 2, 3, 3, 4).float(),
            queries_xytv=torch.tensor([[-2., 1., 1., 0.]]))
        images, raw, uv, affine = prepare_inputs(inp, (6, 8))
        self.assertEqual(images.dtype, np.uint8)
        np.testing.assert_allclose(uv, [[-3.5, 2.5]])
        np.testing.assert_array_equal(raw, inp.queries_xytv.numpy())
        inp.video[0, 0, 0, 0, 0] = .5
        with self.assertRaises(ValueError):
            prepare_inputs(inp, (6, 8))

    def test_unprojection_preserves_outside_ray_with_border_depth(self):
        cache = {"predictions": {"depth": torch.tensor([[[[2., 4.], [6., 8.]]]]),
                                 "intrinsics": torch.eye(3)[None, None]}}
        pts = _source_xyz(cache, np.array([[-1., .5], [2., .5]], np.float32),
                          np.array([0, 0]), torch.device("cpu"))
        np.testing.assert_allclose(pts, [[-4., 2., 4.], [12., 3., 6.]])

    @unittest.skipUnless(os.environ.get("POINT4D_REPO"), "Supply the pinned Point4D source checkout")
    def test_real_author_preprocessing_and_coordinate_contract(self):
        class FakeModel:
            def forward_backbone(self, images):
                _, t, _, h, w = images.shape
                ext = torch.cat([torch.eye(3), torch.zeros(3, 1)], 1).repeat(1, t, 1, 1)
                return {"images": images, "S": t, "H": h, "W": w,
                        "predictions": {"depth": torch.ones(1, t, h, w) * 2,
                            "intrinsics": torch.eye(3).repeat(1, t, 1, 1), "extrinsics": ext}}
            def compute_d4rt_kv_cache(self, cache):
                return {}
            def forward_d4rt_only(self, cache, q, **kwargs):
                # Encode p,s,t,c distinctly to detect any transposed/relabelled axis.
                return {"d4rt_pred": torch.stack([q[..., 3], q[..., 5], q[..., 6]], -1),
                        "d4rt_pred_vis_logits": q[..., 5] - q[..., 4]}
        inp = SimpleNamespace(video=torch.zeros(1, 3, 3, 14, 14, dtype=torch.uint8),
            queries_xytv=torch.tensor([[1., 2., 2., 0.], [4., 5., 1., 0.]]))
        out, meta = predict_case(FakeModel(), inp, device="cpu", image_hw=(14, 14), query_chunk_size=2)
        self.assertEqual(out["tracks_XYZ"].shape, (3, 2, 3))
        self.assertEqual(out["predicted_c2w"].shape, (1, 3, 4, 4))
        self.assertTrue((out["tracks_native_world"][..., 2] == 0).all())
        np.testing.assert_array_equal(out["visibility"], [[False, False], [False, False], [False, True]])
        self.assertTrue(meta["direct_query_extension"])


if __name__ == "__main__":
    unittest.main()
