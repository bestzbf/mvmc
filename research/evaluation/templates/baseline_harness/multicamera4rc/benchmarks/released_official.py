"""Bridges to pinned, unchanged released evaluators; never import training.

AST loading only removes unrelated imports (UI, training, inference). Function
bodies, defaults, thresholds, alignment, filtering and aggregation stay upstream.
Every executed source is hash checked before loading, including reader helpers.
"""

from __future__ import annotations

import glob
import importlib.util
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .official import compile_functions, sha256_file


def verified_sources(name, root):
    root = Path(root).resolve()
    manifest = json.loads(Path(__file__).with_name("released_sources.json").read_text())[name]
    for relative, expected in manifest["sha256"].items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} source {path}; checkout {manifest['commit']}")
        if sha256_file(path) != expected:
            raise ValueError(f"Official {name} source changed: {relative}; checkout {manifest['commit']}")
    return root, manifest


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite_tracks(tracks, shape):
    tracks = np.asarray(tracks)
    if tracks.shape != shape or not np.issubdtype(tracks.dtype, np.number) or not np.isfinite(tracks).all():
        raise ValueError(f"Expected finite tracks {shape}, got {tracks.shape}")
    return tracks


def boolean_visibility(visible, shape):
    visible = np.asarray(visible)
    if visible.shape != shape or visible.dtype != np.bool_:
        raise ValueError(f"Expected boolean visibility {shape}")
    return visible


class OfficialTAPVid3D:
    def __init__(self, root):
        self.root, self.provenance = verified_sources("tapvid3d", root)
        directory = self.root / "tapnet/tapvid3d"
        self.metrics = load_module(directory / "evaluation/metrics.py", "_dfrt_tapvid3d_metrics")
        self.splits = load_module(directory / "splits/tapvid3d_splits.py", "_dfrt_tapvid3d_splits")
        self.helpers = compile_functions(directory / "evaluation/evaluate_model.py",
            ["get_new_hw_with_given_smallest_side_length", "get_jpeg_byte_hw", "get_average_over_metrics"],
            {"np": np, "io": io, "Image": Image})

    def filenames(self, subset, split):
        if subset not in ("adt", "drivetrack", "pstudio") or split not in ("minival", "full_eval"):
            raise ValueError("TAPVid-3D requires adt/drivetrack/pstudio and minival/full_eval")
        method = self.splits.get_minival_files if split == "minival" else self.splits.get_full_eval_files
        return list(method(subset=subset))

    def score(self, path, tracks_camera, visible, scaling="median"):
        # Same call as evaluate_data_source, with strict preflight instead of its
        # broad exception -> ZERO_METRICS fallback. No metric is reimplemented.
        if scaling not in ("median", "none", "per_trajectory"):
            raise ValueError("Supported explicit scoring scales: median, none, per_trajectory")
        with np.load(path, allow_pickle=True) as gt:
            tracks = gt["tracks_XYZ"]
            vis = boolean_visibility(gt["visibility"], tracks.shape[:2])
            prediction = finite_tracks(tracks_camera, tracks.shape)
            visible = boolean_visibility(visible, vis.shape)
            finite_tracks(tracks, tracks.shape)
            h, w = self.helpers["get_jpeg_byte_hw"](gt["images_jpeg_bytes"][0])
            _, scale = self.helpers["get_new_hw_with_given_smallest_side_length"](
                orig_height=h, orig_width=w, smallest_side_length=256)
            arguments = {"gt_occluded": ~vis, "gt_tracks": tracks, "pred_occluded": ~visible,
                "pred_tracks": prediction, "intrinsics_params": gt["fx_fy_cx_cy"] * scale,
                "scaling": scaling, "query_points": gt["queries_xyt"][..., ::-1], "order": "t n"}
            if scaling == "per_trajectory":
                # Upstream's unbatched wrapper leaves query_points without the
                # batch axis its metric inserts into tracks. Use the official
                # explicit-batch API for this optional diagnostic. The default
                # median protocol/call is unchanged; no metric body is patched.
                for key in ("gt_occluded", "gt_tracks", "pred_occluded", "pred_tracks",
                            "intrinsics_params", "query_points"):
                    arguments[key] = arguments[key][None]
                arguments["order"] = "b t n"
            result = self.metrics.compute_tapvid3d_metrics(**arguments)
        if any(not np.isfinite(value).all() for value in result.values()):
            raise ValueError("Official TAPVid-3D returned nonfinite metrics (e.g. no jointly visible points)")
        return {scaling: result}

    def average(self, metrics):
        if not metrics:
            raise ValueError("Cannot aggregate an empty evaluation")
        return self.helpers["get_average_over_metrics"](metrics)


class OfficialWorldTrack:
    SUBSETS = ("adt_mini", "pstudio_mini", "po_mini", "ds_mini")
    RECON_SUBSETS = ("po_mini", "tum")
    RETURN_KEYS = ("avg_pts_global", "avg_pts_pertraj", "avg_pts_sim3",
                   "avg_pts_sim3_closed", "avg_pts_sim3_closed_dyn", "epe_global",
                   "epe_pertraj", "epe_sim3", "epe_sim3_closed")

    def __init__(self, root):
        import math
        import time
        from collections import defaultdict

        from natsort import natsorted

        self.root, self.provenance = verified_sources("worldtrack", root)
        self.util = load_module(self.root / "dust3r/track_eval_util.py", "_dfrt_worldtrack_util")
        reader = compile_functions(self.root / "dust3r/datasets/tapvid3d.py",
            ["load_npz_data", "load_npz_data_recon", "project_points_to_video_frame"],
            {"np": np, "cv2": cv2, "Image": Image, "os": os})
        self.read = reader["load_npz_data"]
        self.read_reconstruction = reader["load_npz_data_recon"]
        namespace = {"np": np, "torch": torch, "F": F, "os": os, "glob": glob,
                     "math": math, "time": time, "defaultdict": defaultdict,
                     "natsorted": natsorted, "tqdm": tqdm, "json": json,
                     "load_npz_data": self.read,
                     "load_npz_data_recon": self.read_reconstruction,
                     "get_visibility_from_depth": self.util.get_visibility_from_depth,
                     "compute_average_pts_within_thresh": self.util.compute_average_pts_within_thresh}
        self.evaluate = compile_functions(self.root / "dust3r/track_eval.py",
                                           ["eval_tapvid3d"], namespace)["eval_tapvid3d"]

    def preflight_reconstruction(self, path):
        # The original reader unconditionally inverts extrinsics before its
        # optional-pose branch. Require real poses; never substitute identity.
        with np.load(path, allow_pickle=True) as archive:
            required = {"images_jpeg_bytes", "depth_map", "fx_fy_cx_cy", "visibility", "extrinsics_w2c"}
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"WorldTrack reconstruction missing {sorted(missing)} in {path}; pinned reader requires poses")
        images, depth, intrinsics, world, _, _, _, poses = self.read_reconstruction(str(path), num_frames=64)
        shape = (len(images), images[0].height, images[0].width)
        if depth.shape != shape or world.shape != shape + (3,):
            raise ValueError(f"WorldTrack reconstruction depth/point grid shape differs from RGB: {path}")
        finite_tracks(poses, (len(images), 4, 4))
        finite_tracks(intrinsics, (4,))
        valid = self.util.get_visibility_from_depth(depth)
        if not valid.any():
            raise ValueError(f"No official valid reconstruction depth (0.1 < z < 5): {path}")
        finite_tracks(world[valid], (int(valid.sum()), 3))
        return images

    def score(self, data_root, subset, predictions, expected_files, output_dir, *, reconstruction=False):
        from natsort import natsorted
        supported = self.RECON_SUBSETS if reconstruction else self.SUBSETS
        if subset not in supported:
            raise ValueError("WorldTrack supported subsets: " + ", ".join(supported))
        actual = natsorted(Path(data_root, subset).glob("*.npz"))
        expected = [Path(p).resolve() for p in expected_files]
        if [p.resolve() for p in actual] != expected or not expected or len(predictions) != len(expected):
            raise ValueError("WorldTrack released subset differs from the frozen complete sample list")
        for gt_path, pred_path in zip(expected, predictions):
            if reconstruction:
                images = self.preflight_reconstruction(gt_path)
            else:
                images, _, _, _, gt_tracks, vis, _, _ = self.read(str(gt_path), num_frames=64)
                finite_tracks(gt_tracks, (len(images), vis.shape[1], 3))
                boolean_visibility(vis, gt_tracks.shape[:2])
                if not vis[0].any():
                    raise ValueError(f"No first-frame queries: {gt_path}; upstream would skip this video")
            pred = np.load(pred_path, allow_pickle=False)
            finite_tracks(pred, (len(images), images[0].height, images[0].width, 3))
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # Official eval.sh uses 64 frames, trajectory mode and dynamic subsets.
        # The original dense prediction lookup, OOB filtering, alignment and
        # equal-video aggregation execute here unchanged. No train.py execution.
        result = self.evaluate(output_dir=str(output_dir), num_frames=64,
            load_offline_data=[str(p) for p in predictions], is_filtered_tracks=False,
            data_type=subset, eval_recon=reconstruction, visualize=False, dyn_static=True,
            sim3_for_recon=False,
            save_predictions=False, data_root=str(data_root))
        return dict(zip(self.RETURN_KEYS, result))
