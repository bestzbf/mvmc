#!/usr/bin/env python3
"""ARROW adapter for the frozen released-benchmark prediction contract.

This adapter deliberately calls ARROW's encoder and decoder directly.  It does
not call ``tracking_queries`` (which clips coordinates and assumes one source
frame), and it does not use the sky mask.  Every frozen pixel query is decoded
against every supplied target observation while retaining the source frame.

The adapter supports the released TAPVid-3D manifests and the named RGB-only
MVTracker extension.  For MVTracker, ARROW predictions for the target cameras
at each synchronized time are reduced by an explicitly recorded arithmetic
mean; visibility is the union of ARROW's per-view visibility logits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ARROW_COMMIT = "ed126d105c4d6f5b855a5e40c67675e86495d84b"
ARROW_HF_REVISION = "fd2b2f74c49990f739264a6aee786029227a93d4"
ARROW_WEIGHT_SHA256 = "113b6f83da69fdc3f03dc623e606da0c346a1a13d796bf9ee07dc1abf07aaf25"
ARROW_CONFIG_SHA256 = "84c0599e077b5971467717c307efa0968ad088b4339249bd4f7bafa23e4f9b5a"
METHOD = "ARROW"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def checkpoint_record(author_code_root: Path, checkpoint: Path) -> dict[str, Any]:
    """Verify the author source and weight bytes before the model is loaded."""
    author_code_root = author_code_root.resolve()
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir():
        raise ValueError("Pass the local ARROW release directory containing model.safetensors and config.json")
    weight = checkpoint / "model.safetensors"
    config = checkpoint / "config.json"
    if not weight.is_file():
        raise FileNotFoundError(f"ARROW model.safetensors not found: {weight}")
    if not config.is_file():
        raise FileNotFoundError(f"ARROW config.json not found: {config}")
    if _sha256(weight) != ARROW_WEIGHT_SHA256:
        raise ValueError("ARROW checkpoint does not match the pinned author release")
    if _sha256(config) != ARROW_CONFIG_SHA256:
        raise ValueError("ARROW config does not match the pinned author release")
    revision = _git(author_code_root, "rev-parse", "HEAD")
    if revision != ARROW_COMMIT:
        raise ValueError(f"ARROW source revision differs: {revision} != {ARROW_COMMIT}")
    if _git(author_code_root, "status", "--porcelain"):
        raise ValueError("ARROW source checkout is dirty")
    sources = {
        str(p.relative_to(author_code_root)): _sha256(p)
        for p in sorted(author_code_root.rglob("*.py"))
        if ".git" not in p.parts and "__pycache__" not in p.parts
    }
    component = {
        # The durable comparison queue discovers real files recursively from
        # absolute path + sha256 records. Keep model identity hashes distinct
        # from the actual released weight/config file hashes.
        "path": str(weight),
        "resolved_path": str(weight.resolve()),
        "relative_path": str(weight.relative_to(checkpoint)),
        "sha256": ARROW_WEIGHT_SHA256,
        "bytes": weight.stat().st_size,
        "config_relative_path": str(config.relative_to(checkpoint)),
        "config_sha256": _sha256(config),
        "config": {"path": str(config), "sha256": ARROW_CONFIG_SHA256,
                   "bytes": config.stat().st_size},
        "code": "arrow",
        "revision": ARROW_COMMIT,
        "hf_repo": "ifradlin/arrow",
        "hf_revision": ARROW_HF_REVISION,
        "source_root": str(author_code_root),
        "source_commit": ARROW_COMMIT,
        "source_sha256": sources,
        "checkpoint_loading": "ARROW.from_pretrained; strict author state",
    }
    identity = {"method": METHOD, "components": {"model": component}}
    identity["sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "kind": "author_rgb_only_native_baseline",
        **identity,
        "native_multicamera4rc_loading": False,
        "refiner_loading": "not_applicable",
    }


def _flatten_video(video):
    """Return uint8 ``[F,H,W,3]``, the number of views, and frames per view."""
    import torch

    x = torch.as_tensor(video)
    if x.ndim != 5 or x.shape[2] != 3:
        raise ValueError(f"Expected video [V,T,3,H,W], got {tuple(x.shape)}")
    views, frames = int(x.shape[0]), int(x.shape[1])
    if min(views, frames, int(x.shape[3]), int(x.shape[4])) < 1:
        raise ValueError("Video views, frames, height, and width must be nonempty")
    if x.dtype != torch.uint8:
        raise ValueError(f"Frozen benchmark RGB must be uint8, got {x.dtype}")
    # Inputs are view-major; ARROW's multiview convention is time-major.
    out = x.permute(1, 0, 3, 4, 2).reshape(frames * views, int(x.shape[3]), int(x.shape[4]), 3)
    return out.cpu().numpy(), views, frames


def _prepare_observation(inputs, max_side: int):
    """Use ARROW's official resize/padding transform on frozen RGB bytes."""
    import torch
    from PIL import Image
    from arrow.data.transforms import prepare_images
    from arrow.infer.dataloader import ObservationBatch

    frames, views, steps = _flatten_video(inputs.video)
    images = [Image.fromarray(frame, mode="RGB") for frame in frames]
    info = [{"index": i, "view": i % views, "timestep": i // views} for i in range(len(images))]
    prepared = prepare_images(images, info, max_side=max_side).unsqueeze(0)
    return ObservationBatch(prepared, info, views), views, steps


def _decode(model, encoded, queries, chunk_size: int):
    from arrow.types import Predictions

    chunks = [part.cpu() for part in model.decode_iter(encoded, queries, chunk_size)]
    if not chunks:
        return Predictions.empty(queries.batched().xy.shape[0])
    return chunks[0] if len(chunks) == 1 else Predictions.concat(chunks)


def query_layout(raw, xy_in, *, views: int, steps: int, benchmark: str):
    """Construct the direct-query order used by the adapter.

    The returned rows are seed-major and target-minor, which reshapes back to
    ``[seed,target]`` without relying on ARROW's grid/query sampler.
    """
    import torch
    from arrow.types import Queries

    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 4:
        raise ValueError("raw queries must be [N,4] x,y,time,view")
    if not np.isfinite(raw).all():
        raise ValueError("raw queries must be finite")
    if not np.equal(raw[:, 2:], np.floor(raw[:, 2:])).all():
        raise ValueError("query time and view IDs must be integer-valued")
    if ((raw[:, 2] < 0) | (raw[:, 2] >= steps)).any():
        raise ValueError("query source time is outside the frozen clip")
    if ((raw[:, 3] < 0) | (raw[:, 3] >= views)).any():
        raise ValueError("query source view is outside the frozen camera set")
    n = len(raw)
    target_count = steps * views if views > 1 else steps
    source = torch.from_numpy((raw[:, 2] * views + raw[:, 3]).astype(np.int64)) if views > 1 else \
        torch.from_numpy(raw[:, 2].astype(np.int64))
    targets = torch.arange(target_count, dtype=torch.long).repeat(n)
    source_rep = source.repeat_interleave(target_count)
    xy_rep = xy_in.repeat_interleave(target_count, dim=0)
    refs = targets if benchmark == "tapvid3d" else torch.zeros_like(targets)
    return Queries(xy_rep, source_rep, targets, refs), source, target_count


def _pose_estimate(model, encoded, grid_size: int, chunk_size: int):
    from arrow.tasks.pose import estimate_poses, pose_queries

    queries = pose_queries(encoded.images, grid_size=grid_size, reference_frame=0)
    pred = _decode(model, encoded, queries, chunk_size)
    return estimate_poses(pred, queries.xy, num_frames=encoded.num_frames, reference_frame=0)


def _intrinsics_original(poses, frame_info, width: int, height: int) -> np.ndarray:
    """Map ARROW input-pixel intrinsics back through original_to_input."""
    import torch

    k = poses.intrinsics[0].detach().cpu().double()
    out = []
    for i, frame in enumerate(frame_info):
        a = torch.eye(3, dtype=torch.float64)
        a[:2] = torch.as_tensor(frame["original_to_input"], dtype=torch.float64)[:2]
        ki = k[i]
        if not torch.isfinite(ki).all() or ki[0, 0] <= 0 or ki[1, 1] <= 0:
            raise ValueError(f"ARROW pose routine could not fit finite intrinsics for observation {i}")
        ko = torch.linalg.inv(a) @ ki
        out.append(ko.numpy().astype(np.float32))
    return np.stack(out)


def _reduce_multiview_tracks(points: np.ndarray, logits: np.ndarray, *, n: int,
                             steps: int, views: int):
    """Reduce direct ref-0 predictions [N,T*V,3] to explicit [T,N,...] outputs."""
    points = np.asarray(points).reshape(n, steps, views, 3)
    logits = np.asarray(logits).reshape(n, steps, views)
    tracks = points.mean(axis=2).transpose(1, 0, 2)
    visible = (logits > 0).any(axis=2).transpose(1, 0)
    return tracks, visible


def predict_case(model, inputs, *, benchmark, max_side=504, decode_chunk_size=100_000,
                 pose_grid_size=64, device="cuda"):
    """Predict one frozen case and return the complete released array contract."""
    import torch
    obs, views, steps = _prepare_observation(inputs, max_side)
    raw = inputs.queries_xytv.detach().cpu().numpy().astype(np.float32)
    video_np = inputs.video.detach().cpu().numpy()
    height, width = int(video_np.shape[-2]), int(video_np.shape[-1])
    xy = torch.from_numpy(raw[:, :2])
    # Source views use the flattened time-major ARROW observation index.
    source_for_transform = (torch.from_numpy(raw[:, 2].astype(np.int64)) if views == 1 else
                            torch.from_numpy((raw[:, 2] * views + raw[:, 3]).astype(np.int64)))
    xy_in = obs.transform_points(xy, source_for_transform)
    n = int(len(raw))
    # TAPVid-3D reports each trajectory in each target frame's own camera.
    # The MVTracker RGB-only extension retains ARROW's frame-0 native gauge.
    if benchmark not in ("tapvid3d", "mvtracker"):
        raise ValueError("ARROW adapter supports TAPVid-3D and RGB-only MVTracker manifests")
    if benchmark == "tapvid3d" and views != 1:
        raise ValueError("TAPVid-3D is a single-view protocol")
    queries, _, target_count = query_layout(raw, xy_in, views=views, steps=steps, benchmark=benchmark)
    device = torch.device(device)
    with torch.inference_mode():
        encoded = model.encode(obs.images.to(device))
        pred = _decode(model, encoded, queries, decode_chunk_size)
        # TAPVid-3D's trajectory is intentionally in each target camera, but
        # the released array contract also carries a native reference-0 field.
        # Decode that field explicitly instead of relabelling target-camera
        # points.  This keeps both gauges independent and avoids using an
        # estimated pose as a hidden evaluation transform.
        native_pred = None
        if benchmark == "tapvid3d":
            native_queries = type(queries)(
                queries.xy, queries.source_frame, queries.target_frame,
                torch.zeros_like(queries.reference_frame),
            )
            native_pred = _decode(model, encoded, native_queries, decode_chunk_size)
        poses = _pose_estimate(model, encoded, pose_grid_size, decode_chunk_size)
    points = pred.points.reshape(n, target_count, 3).numpy().astype(np.float32)
    logits = pred.visibility_logits.reshape(n, target_count).numpy()
    if benchmark == "tapvid3d":
        tracks = points.transpose(1, 0, 2)
        native_points = native_pred.points.reshape(n, target_count, 3).numpy().astype(np.float32)
        tracks_native = native_points.transpose(1, 0, 2)
        visible = (logits > 0).transpose(1, 0)
    else:
        tracks_native, visible = _reduce_multiview_tracks(
            points, logits, n=n, steps=steps, views=views
        )
        tracks = tracks_native
    if not bool(poses.valid.all()):
        failed = (~poses.valid).nonzero().tolist()
        raise ValueError(f"ARROW pose estimation failed for observations: {failed}")
    c2w = poses.camera_to_reference[0].detach().cpu().numpy().astype(np.float32)
    k = _intrinsics_original(poses, obs.frame_info, width, height)
    if views == 1:
        c2w = c2w[None]
        k = k[None]
    else:
        c2w = c2w.reshape(steps, views, 4, 4).transpose(1, 0, 2, 3)
        k = k.reshape(steps, views, 3, 3).transpose(1, 0, 2, 3)
    metadata = {
        "adapter": "arrow-direct-query-v1",
        "input_condition": "rgb_only_unknown_calibration",
        "coordinates": ("per-target-camera coordinates" if benchmark == "tapvid3d"
                        else "ARROW reference-frame-0 native predicted gauge"),
        "source_queries": "frozen pixel x,y,time,view; exact source frame retained",
        "target_time_policy": "all encoded target observations",
        "visibility_semantics": "ARROW visibility_logit > 0; MV union over target views",
        "multiview_target_reduction": "arithmetic mean over target cameras; no GT weighting",
        "max_side": int(max_side), "decode_chunk_size": int(decode_chunk_size),
        "pose_grid_size": int(pose_grid_size), "views": int(views), "frames": int(steps),
        "input_resolution": [height, width],
    }
    arrays = {
        "tracks_XYZ": tracks,
        "visibility": visible.astype(bool),
        "tracks_native_world": tracks_native,
        "predicted_c2w": c2w,
        "predicted_K": k,
    }
    if not np.isfinite(tracks_native).all() or not np.isfinite(c2w).all() or not np.isfinite(k).all():
        raise ValueError("ARROW produced non-finite prediction arrays")
    return arrays, metadata


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--author-code-root", type=Path, required=True)
    p.add_argument("--baseline-code-root", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-side", type=int, default=504)
    p.add_argument("--decode-chunk-size", type=int, default=100_000)
    p.add_argument("--pose-grid-size", type=int, default=64)
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(argv)
    if (a.decode_chunk_size < 1 or a.pose_grid_size < 3 or a.cpu_threads < 1
            or a.max_side < 14 or a.max_side % 14):
        p.error("Chunk size/threads must be positive, pose grid size >=3, and max-side a positive multiple of 14")
    benchmark = _manifest_benchmark(a.manifest)
    if benchmark not in ("tapvid3d", "mvtracker"):
        p.error("ARROW supports only TAPVid-3D and the RGB-only MVTracker manifest")
    if benchmark == "mvtracker":
        manifest = json.loads(a.manifest.read_text())
        if manifest.get("protocol", {}).get("mode") != "rgb_only_3d_query_anchor_sim3":
            p.error("MVTracker requires the frozen RGB-only cached extension")
    sys.path.insert(0, str(a.baseline_code_root.resolve()))
    from multicamera4rc.benchmarks.released_baselines import run_adapter_predictions
    import torch
    torch.set_num_threads(a.cpu_threads)
    record = checkpoint_record(a.author_code_root, a.checkpoint)
    sys.path.insert(0, str(a.author_code_root.resolve() / "src"))
    from arrow import ARROW
    model = ARROW.from_pretrained(str(a.checkpoint), device=a.device)
    model.eval()

    def callback(inputs, case, index):
        return predict_case(model, inputs, benchmark=benchmark, max_side=a.max_side,
                            decode_chunk_size=a.decode_chunk_size,
                            pose_grid_size=a.pose_grid_size, device=a.device)

    return run_adapter_predictions(
        a.manifest, a.output, method=METHOD, checkpoint=record,
        predict_case=callback, device=a.device, resume=a.resume,
        metric_support={"position": True, "native_visibility": True, "AJ": True, "OA": True},
        inference_settings={"target_time_policy": "all_encoded_target_observations",
            "prediction_kind": "trajectory", "author_code_root": str(a.author_code_root),
            "checkpoint": str(a.checkpoint), "max_side": a.max_side,
            "decode_chunk_size": a.decode_chunk_size, "pose_grid_size": a.pose_grid_size,
            "skip_sky": False},
    )


def _manifest_benchmark(path: Path) -> str:
    return json.loads(path.read_text())["benchmark"]


if __name__ == "__main__":
    print(main())
