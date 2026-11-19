from types import SimpleNamespace

import numpy as np
import torch

from multicamera4rc.datasets import partial_dense
from multicamera4rc.datasets.bridge import prepare_clip


def test_world_pair_delta_is_rotated_into_reference_camera_gauge(monkeypatch):
    height, width = 28, 28
    K = np.array([[50.0, 0.0, 14.0], [0.0, 50.0, 14.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    theta = np.pi / 2.0
    reference_extrinsic = np.eye(4, dtype=np.float32)
    reference_extrinsic[:3, :3] = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0],
         [np.sin(theta), np.cos(theta), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    # Point at raw world [1, 0, 5] projects to (u=14, v=24) in reference camera [0, 1, 5].
    # Moving by raw world delta [1, 0, 0] to [2, 0, 5] must land at canonical [0, 2, 5]
    # with canonical displacement [0, 1, 0].
    raw_world_delta = np.zeros((height, width, 3), dtype=np.float32)
    raw_world_delta[..., 0] = 1.0
    pair_valid = np.ones((height, width), dtype=bool)

    monkeypatch.setattr(
        partial_dense,
        "replica_pairs",
        lambda clip: [[(1, raw_world_delta, pair_valid), None], [None, None]],
    )

    image = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.full((height, width), 5.0, dtype=np.float32)
    clip = SimpleNamespace(
        images=[image, image],
        depths=[depth, depth],
        intrinsics=np.stack([K, K]),
        extrinsics=np.stack([reference_extrinsic, reference_extrinsic]),
        metadata={"extrinsics_convention": "w2c", "frame_indices": [0, 1]},
    )
    sample = prepare_clip(clip, image_size=(height, width), num_queries=0, image_mode="native")
    partial_dense.attach_partial_dense_recipe(
        clip,
        sample,
        kind="replica_flow",
        image_mode="native",
    )

    expected_delta = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).expand(height, width, 3)
    assert torch.allclose(sample["targets"]["dense_pair_delta"][0, 0], expected_delta, atol=1e-6)

    batch = {
        "images": sample["images"][None],
        "camera_id": sample["camera_id"][None],
        "time_idx": sample["time_idx"][None],
        "observation_valid": sample["observation_valid"][None],
        "query_time_idx": sample["query_time_idx"][None],
        "targets": {k: v[None] for k, v in sample["targets"].items()},
    }
    materialized = partial_dense.materialize_partial_dense_targets(
        batch, torch.tensor([[0]]), normalize_scene=False
    )
    assert torch.allclose(
        materialized["trajectory"][0, 0, 0, 24, 14],
        torch.tensor([0.0, 1.0, 5.0]),
        atol=1e-5,
    )
    assert torch.allclose(
        materialized["trajectory"][0, 0, 1, 24, 14],
        torch.tensor([0.0, 2.0, 5.0]),
        atol=1e-5,
    )

