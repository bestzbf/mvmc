"""Check that adapter timelines survive the multi-camera supervision bridge."""

from types import SimpleNamespace

import numpy as np
import pytest

from multicamera4rc.datasets.bridge import prepare_clip


def _clip(slot_time, *, unit=None, **metadata):
    if unit is not None:
        metadata["physical_time_unit"] = unit
    return SimpleNamespace(
        dataset_name="frame_units_fixture",
        sequence_name="clip",
        images=[np.zeros((28, 28, 3), dtype=np.uint8) for _ in range(4)],
        intrinsics=np.repeat(
            np.array([[[24., 0., 13.5], [0., 24., 13.5], [0., 0., 1.]]]), 4, axis=0
        ),
        extrinsics=np.repeat(np.eye(4)[None], 4, axis=0),
        metadata={
            "slot_physical_camera_id": [0, 0, 1, 1],
            "slot_time": slot_time,
            "fps": 30.0,
            **metadata,
        },
    )


@pytest.mark.parametrize(
    ("unit", "slot_time"),
    [
        ("frames", [5, 8, 5, 8]),
        ("seconds", [5 / 30, 8 / 30, 5 / 30, 8 / 30]),
        (None, [5 / 30, 8 / 30, 5 / 30, 8 / 30]),
    ],
)
def test_shared_timeline_preserves_frame_gaps(unit, slot_time):
    sample = prepare_clip(_clip(slot_time, unit=unit), image_mode="native")
    assert sample["time_idx"].tolist() == [5, 8, 5, 8]
    assert sample["query_time_idx"].tolist() == [5, 8]


def test_explicit_frame_axis_takes_priority_over_fallback_metadata():
    clip = _clip([0, 1, 0, 1], unit="frames", time_idx=[5, 8, 5, 8])
    sample = prepare_clip(clip, image_mode="native")
    assert sample["time_idx"].tolist() == [5, 8, 5, 8]


@pytest.mark.parametrize(
    ("unit", "slot_time", "message"),
    [
        ("frames", [5.5, 8., 5.5, 8.], "nonnegative integer"),
        ("seconds", [5.5 / 30, 8 / 30, 5.5 / 30, 8 / 30], "integer synchronized"),
    ],
)
def test_does_not_reinterpret_fractional_frames(unit, slot_time, message):
    with pytest.raises(ValueError, match=message):
        prepare_clip(_clip(slot_time, unit=unit), image_mode="native")
