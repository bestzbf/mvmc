from types import SimpleNamespace

import numpy as np
import torch

from multicamera4rc.datasets import partial_dense


def test_world_pair_delta_is_not_rotated_by_reference_camera(monkeypatch):
    height, width = 2, 2
    world_delta = np.zeros((height, width, 3), dtype=np.float32)
    world_delta[..., 0] = 1.0
    pair_valid = np.ones((height, width), dtype=bool)

    monkeypatch.setattr(
        partial_dense,
        "replica_pairs",
        lambda clip: [[(1, world_delta, pair_valid)], [None]],
    )

    theta = np.pi / 2.0
    reference_extrinsic = np.eye(4, dtype=np.float32)
    reference_extrinsic[:3, :3] = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0],
         [np.sin(theta), np.cos(theta), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    clip = SimpleNamespace(
        extrinsics=np.stack([reference_extrinsic, np.eye(4, dtype=np.float32)]),
        metadata={"extrinsics_convention": "w2c"},
    )
    sample = {
        "images": torch.zeros(2, 3, height, width),
        "targets": {},
        "time_idx": torch.tensor([0, 1]),
    }

    partial_dense.attach_partial_dense_recipe(
        clip,
        sample,
        kind="replica_flow",
        image_mode="native",
    )

    actual = sample["targets"]["dense_pair_delta"][0, 0]
    expected = torch.from_numpy(world_delta)
    assert torch.allclose(actual, expected)
