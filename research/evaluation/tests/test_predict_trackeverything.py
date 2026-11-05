import unittest
from types import SimpleNamespace
import numpy as np
import torch
from predict_trackeverything import prepare_inputs, output_arrays, predict_case, stitch_backward, reverse_time_order


class ContractTests(unittest.TestCase):
    def test_time_major_rgb_and_source_queries(self):
        inp = SimpleNamespace(video=torch.zeros(2, 3, 3, 4, 6),
                              queries_xytv=torch.tensor([[-1., 5., 2., 1.], [2., 3., 0., 0.]]))
        for v in range(2):
            for t in range(3):
                inp.video[v, t] = 10 * t + v
        images, queries, views, frames = prepare_inputs(inp)
        np.testing.assert_array_equal(images[:, 0, 0, 0], [0, 1, 10, 11, 20, 21])
        np.testing.assert_array_equal(queries, [[5, -1, 5], [0, 2, 3]])
        self.assertEqual((views, frames), (2, 3))
        inp.video[0, 0, 0, 0, 0] = .5
        with self.assertRaises(ValueError):
            prepare_inputs(inp)

    @staticmethod
    def output(views=1):
        frames, n = 2, 2
        f = frames * views
        w2c = torch.eye(4).repeat(f, 1, 1)
        w2c[:, 0, 3] = torch.arange(f)
        coords = torch.arange(f * n * 3).reshape(1, f, n, 3).float()
        vis = torch.tensor([[[-1., 1.], [1., -1.]]]).repeat(1, views, 1)
        return SimpleNamespace(query_coords=coords, query_visibs=vis, extrinsics=w2c,
            intrinsics=torch.eye(3).repeat(f, 1, 1), rgb=torch.zeros(f, 3, 4, 6))

    def test_target_camera_and_native_world_independent(self):
        out = self.output()
        arrays = output_arrays(out, views=1, frames=2, count=2, original_hw=(7, 11), benchmark="tapvid3d")
        np.testing.assert_array_equal(arrays["tracks_native_world"], out.query_coords.numpy()[0])
        np.testing.assert_array_equal(arrays["tracks_XYZ"][1, :, 0], out.query_coords.numpy()[0, 1, :, 0] + 1)
        self.assertEqual(arrays["predicted_c2w"][0, 1, 0, 3], -1)
        self.assertEqual(arrays["predicted_K"][0, 0, 0, 0], 2)
        np.testing.assert_array_equal(arrays["visibility"], [[False, True], [True, False]])

    def test_multiview_reduction_and_union(self):
        out = self.output(2)
        arrays = output_arrays(out, views=2, frames=2, count=2, original_hw=(4, 6), benchmark="mvtracker")
        np.testing.assert_array_equal(arrays["tracks_XYZ"], out.query_coords.numpy()[0].reshape(2, 2, 2, 3).mean(1))
        self.assertTrue(arrays["visibility"].all())
        self.assertEqual(arrays["predicted_c2w"].shape, (2, 2, 4, 4))

    def test_callback_never_supplies_geometry_or_crops_frames(self):
        class FakeTracker:
            _geometry = None
            def infer(self, **kwargs):
                self.kwargs = kwargs
                return ContractTests.output(2)
        tracker = FakeTracker()
        inp = SimpleNamespace(video=torch.zeros(2, 2, 3, 4, 6, dtype=torch.uint8),
                              queries_xytv=torch.tensor([[1., 2., 0., 1.], [4., 3., 0., 0.]]))
        arrays, meta = predict_case(tracker, inp, benchmark="mvtracker", geometry_chunk_observations=7)
        self.assertEqual(tracker.kwargs["video"].shape, (4, 4, 6, 3))
        np.testing.assert_array_equal(tracker.kwargs["query_points_2d"][:, 0], [1, 0])
        self.assertIsNone(tracker.kwargs["max_frames"])
        self.assertFalse({"depths", "intrinsics", "extrinsics"} & tracker.kwargs.keys())
        self.assertEqual(meta["geometry_chunk_observations"], 6)
        self.assertEqual(meta["geometry_overlap_observations"], 2)
        self.assertEqual(arrays["tracks_XYZ"].shape, (2, 2, 3))

    def test_backward_uses_shared_geometry_and_preserves_camera_order(self):
        frames, views, count = 3, 2, 2
        f = frames * views
        rgb = torch.arange(f)[:, None, None, None].expand(f, 3, 4, 6).float()
        depth = torch.arange(f)[:, None, None].expand(f, 4, 6).float() + 10
        k = torch.eye(3).repeat(f, 1, 1)
        ext = torch.eye(4).repeat(f, 1, 1)
        ext[:, 0, 3] = torch.arange(f) + 30
        out = SimpleNamespace(rgb=rgb, depths=depth, intrinsics=k, extrinsics=ext,
            dense_source_mask=torch.ones(f, 1, 2, dtype=torch.bool),
            query_coords=torch.zeros(1, f, count, 3), query_visibs=torch.zeros(1, f, count))
        class FakeModel:
            num_views = 1
            def __call__(self, **kwargs):
                self.kwargs = kwargs
                self.views_during_call = self.num_views
                values = torch.arange(f)[None, :, None].expand(1, f, count).float() + 100
                return SimpleNamespace(query_coords=values[..., None].expand(1, f, count, 3),
                                       query_visibs=values), None
        model = FakeModel()
        tracker = SimpleNamespace(model=model)
        queries = np.array([[5, 1, 2], [1, 3, 1]], np.float32)
        def lift(q, d, kk, ee):
            # The exact existing preprocessed geometry is reused for query lift.
            self.assertIs(d, depth)
            self.assertIs(kk, k)
            self.assertIs(ee, ext)
            return torch.tensor([[[5., 20., 30., 40.], [1., 50., 60., 70.]]])
        stitched = stitch_backward(tracker, out, queries, views=views, frames=frames,
                                   original_hw=(4, 6), lift_fn=lift)
        np.testing.assert_array_equal(reverse_time_order(frames, views, device="cpu"), [4, 5, 2, 3, 0, 1])
        np.testing.assert_array_equal(model.kwargs["query_point"][0, :, 0], [1, 5])
        np.testing.assert_array_equal(model.kwargs["query_point"][0, :, 1:], [[20, 30, 40], [50, 60, 70]])
        np.testing.assert_array_equal(model.kwargs["rgb_obs"][0, :, 0, 0, 0], [4, 5, 2, 3, 0, 1])
        np.testing.assert_array_equal(model.kwargs["depth_obs"][0, :, 0, 0], [14, 15, 12, 13, 10, 11])
        np.testing.assert_array_equal(model.kwargs["extrinsics"][0, :, 0, 3], [34, 35, 32, 33, 30, 31])
        self.assertEqual(model.views_during_call, 2)
        self.assertEqual(model.num_views, 1)
        # Only logical times before source2 are replaced. Same-timestep view0
        # stays forward although its flattened index4 is less than queryflat5.
        np.testing.assert_array_equal(stitched.query_coords[0, :, 0, 0], [104, 105, 102, 103, 0, 0])
        self.assertTrue((stitched.query_coords[0, :, 1] == 0).all())
        np.testing.assert_array_equal(stitched.query_visibs[0, :, 0], [104, 105, 102, 103, 0, 0])


if __name__ == "__main__":
    unittest.main()
