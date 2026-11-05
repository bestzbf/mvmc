"""Read atomically published packs of the official MV-Kubric release.

The source format is deliberately converted once, outside data-loader workers.
Its per-view instance numbering is not a shared object identity. See
``scripts.prepare_mv_kubric`` for the checked conversion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .kubric_multicam import KubricMultiCamAdapter


SCHEMA_VERSION = "mv_kubric_official_v1"


def official_scene_split(source_split, scene_id):
    """Validate the release's immutable train/test boundary."""
    scene_id = int(scene_id)
    source_split = str(source_split)
    if source_split == "train" and 31 <= scene_id <= 5000:
        return "train"
    if source_split == "test" and 1 <= scene_id <= 30:
        return "test"
    raise ValueError(f"not an official MV-Kubric scene: {source_split}/{scene_id}")


def is_validation_scene(scene_id, val_ratio=0.05, split_strategy="tail"):
    """Reserve validation independently of which packs currently exist.

    In particular, adding newly converted scenes never changes the split of
    an already published scene. Official test scenes are never used as val.
    """
    ratio = float(val_ratio)
    if not 0 <= ratio <= 0.5:
        raise ValueError("val_ratio must be between zero and .5")
    if split_strategy == "tail":
        count = round((5000 - 31 + 1) * ratio)
        return int(scene_id) > 5000 - count
    if split_strategy == "hash":
        value = int(hashlib.sha256(f"mv-kubric-v1:{int(scene_id)}".encode()).hexdigest()[:16], 16)
        return value / 2**64 < ratio
    raise ValueError("split_strategy must be 'tail' or 'hash'")


def validate_full_conversion(root, actual_train_ids):
    """Require both a completed conversion receipt and all immutable train IDs."""
    path = Path(root) / "conversion_manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"full MV-Kubric training requires a complete manifest: {path}") from error
    from multicamera4rc.datasets.mv_kubric_inventory import expected_train_ids
    expected = expected_train_ids(manifest)
    if (manifest.get("status") != "complete" or manifest.get("schema") != SCHEMA_VERSION
            or manifest.get("expected_scenes") != len(expected)
            or manifest.get("converted_scenes") != len(expected)
            or manifest.get("official_train_scene_ids") != expected):
        raise ValueError("MV-Kubric full conversion manifest is incomplete or inconsistent")
    if sorted(actual_train_ids) != expected:
        raise ValueError("MV-Kubric full conversion requires exactly the valid published train scenes from the verified inventory")


class MVKubricAdapter(KubricMultiCamAdapter):
    """Official 10-view, 24-frame scenes, with renderer-derived dense GT.

    ``split='all'`` explicitly includes both official train and test;
    ``split='train'`` excludes the deterministic held-out validation set.
    Native sparse trajectories are retained only in the pack's audit group.
    Training uses full image depth, canonical segmentation and rigid poses.
    """

    dataset_name = "mv_kubric"

    def __init__(self, root, split="train", *, verbose=True, index_workers=2,
                 val_ratio=0.05, split_strategy="tail", min_cameras=1,
                 max_cameras=10, min_times=1, max_times=24,
                 random_cameras=True, random_camera_order=False,
                 random_num_cameras=False, random_num_times=False,
                 encoder_tubelet_size=1, sampler_units="time",
                 use_direct_depth_queries=False, require_depth=True,
                 depth_is_ray_distance=True, extrinsics_are_opengl=False,
                 pixel_coordinates="array_index_v1", require_full_conversion=False,
                 strict=True):
        if split not in {"train", "val", "test", "all"}:
            raise ValueError("MV-Kubric split must be train, val, test or all")
        if sampler_units != "time" or encoder_tubelet_size != 1:
            raise ValueError("official MV-Kubric uses synchronized frame sampling, tubelet size 1")
        if use_direct_depth_queries or not require_depth:
            raise ValueError("official MV-Kubric requires renderer depth and dense supervision")
        if not depth_is_ray_distance or extrinsics_are_opengl or pixel_coordinates != "array_index_v1":
            raise ValueError("MV-Kubric packs have radial depth, OpenCV C2W and raster-center K")
        self.official_split = split
        self.official_val_ratio = float(val_ratio)
        self.official_split_strategy = split_strategy
        is_validation_scene(31, val_ratio, split_strategy)
        super().__init__(root=root, split="all", verbose=False,
            index_workers=index_workers, required_schema_version=SCHEMA_VERSION,
            min_cameras=min_cameras, max_cameras=max_cameras,
            min_times=min_times, max_times=max_times,
            random_cameras=random_cameras, random_camera_order=random_camera_order,
            random_num_cameras=random_num_cameras, random_num_times=random_num_times,
            encoder_tubelet_size=1, sampler_units="time", strict=strict,
            use_direct_depth_queries=False, require_depth=True,
            depth_is_ray_distance=True, extrinsics_are_opengl=False,
            pixel_coordinates="array_index_v1")
        if require_full_conversion:
            import h5py
            train_ids = []
            for record in self._records:
                with h5py.File(record.h5_path, "r") as handle:
                    if str(handle.attrs["source_split"]) == "train":
                        train_ids.append(int(handle.attrs["source_scene_id"]))
            validate_full_conversion(root, train_ids)
        self._records = [record for record in self._records if self._in_split(record.h5_path)]
        self.split = split
        self.sequence_names = [record.sequence_name for record in self._records]
        self._name_to_record = {record.sequence_name: record for record in self._records}
        if verbose:
            print(f"[MVKubricAdapter] split={split} scenes={len(self)} root={Path(root)}")

    def _in_split(self, path):
        import h5py
        with h5py.File(path, "r") as handle:
            source = str(handle.attrs["source_split"])
            scene_id = int(handle.attrs["source_scene_id"])
        if self.official_split == "all":
            return True
        if self.official_split == "test":
            return source == "test"
        if source != "train":
            return False
        validation = is_validation_scene(scene_id, self.official_val_ratio,
                                         self.official_split_strategy)
        return validation if self.official_split == "val" else not validation

    def _probe_h5(self, path):
        import h5py
        # .partial files never have an .h5 suffix. The completion attribute
        # additionally rejects copied/interrupted files with a final suffix.
        try:
            with h5py.File(path, "r") as handle:
                if not bool(handle.attrs.get("conversion_complete", False)):
                    return None
                official_scene_split(handle.attrs["source_split"], handle.attrs["source_scene_id"])
                required = {"depth", "segmentation", "object_trajectories", "object_is_dynamic",
                            "object_names", "object_raw_scene_ids"}
                if not required.issubset(handle):
                    return None
                objects = len(handle["object_is_dynamic"])
                frames = handle["frames"].shape
                if (len(frames) != 5 or frames[:2] != (24, 10)
                        or handle["object_trajectories"].shape != (objects, 24, 7)
                        or handle["segmentation"].shape != frames[:-1]
                        or handle["object_names"].shape != (objects,)
                        or handle["object_raw_scene_ids"].shape != (objects,)):
                    return None
        except (OSError, KeyError, ValueError, TypeError):
            return None
        return super()._probe_h5(path)
