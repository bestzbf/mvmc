#!/usr/bin/env python3
"""Official sparse TrackEverything with its VGGT-Omega RGB geometry backend."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

COMMIT = "dfb29b505cf2ea954f8697fb612c2429668dd3a0"
WEIGHT_HASH = "4553975d372096f8c3d7c5fc35fd7025ef8a7611355e6ada8eb251ef1f5feac1"
GEOMETRY_HASH = "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934"
INITIALIZER_HASH = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
METHOD = "VGGT-Omega-1B-512+TrackEverything"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def checkpoint_record(author_code_root, checkpoint, geometry_checkpoint, initializer_checkpoint):
    code, weight, geom = Path(author_code_root).resolve(), Path(checkpoint).resolve(), Path(geometry_checkpoint).resolve()
    if weight.is_dir():
        weight = weight / "model.pth"
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=code, text=True).strip()
    if git("rev-parse", "HEAD") != COMMIT or git("status", "--porcelain"):
        raise ValueError("TrackEverything requires the pinned clean author checkout")
    sources = {str(p.relative_to(code)): sha256(p) for p in sorted(code.rglob("*"))
               if p.is_file() and p.suffix in (".py", ".yaml", ".json")
               and ".git" not in p.parts and "__pycache__" not in p.parts}
    components = {}
    for name, path, digest in (("tracker", weight, WEIGHT_HASH), ("geometry", geom, GEOMETRY_HASH)):
        if sha256(path) != digest:
            raise ValueError(f"{name} checkpoint does not match the pinned release")
        components[name] = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size,
            "source_root": str(code), "source_commit": COMMIT, "source_sha256": sources,
            "checkpoint_loading": ("author strict=True" if name == "tracker" else "author vendored VGGT-Omega strict state")}
    initializer = Path(initializer_checkpoint).resolve()
    if initializer.name != "dinov2_vits14_pretrain.pth" or initializer.parent.name != "checkpoints":
        raise ValueError("Initializer must use the official Torch Hub cache filename")
    if sha256(initializer) != INITIALIZER_HASH:
        raise ValueError("DINOv2 constructor initializer differs from its pinned official bytes")
    components["constructor_initializer"] = {"path": str(initializer), "sha256": INITIALIZER_HASH,
        "bytes": initializer.stat().st_size, "source_root": str(code), "source_commit": COMMIT,
        "source_sha256": sources, "checkpoint_loading": "official cached DINOv2 constructor initialization; strict released tracker load follows"}
    identity = {"method": METHOD, "components": components}
    identity["sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"kind": "composed_rgb_only_baseline", **identity,
            "native_multicamera4rc_loading": False, "refiner_loading": "not_applicable"}


def prepare_inputs(inputs):
    import torch
    video = torch.as_tensor(inputs.video).detach().cpu()
    if video.ndim != 5 or video.shape[2] != 3 or min(video.shape) < 1:
        raise ValueError("Expected nonempty [V,T,3,H,W] RGB")
    if video.dtype != torch.uint8:
        if (not video.is_floating_point() or not torch.isfinite(video).all() or
                (video < 0).any() or (video > 255).any() or not torch.equal(video, video.round())):
            raise ValueError("RGB must contain finite integer byte values in 0..255")
        video = video.to(torch.uint8)
    views, frames = int(video.shape[0]), int(video.shape[1])
    raw = np.asarray(inputs.queries_xytv.detach().cpu(), dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 4 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("Expected finite nonempty [x,y,time,view] queries")
    if (not np.equal(raw[:, 2:], np.floor(raw[:, 2:])).all() or
            (raw[:, 2] < 0).any() or (raw[:, 2] >= frames).any() or
            (raw[:, 3] < 0).any() or (raw[:, 3] >= views).any()):
        raise ValueError("Frozen source time/view IDs are invalid")
    images = video.permute(1, 0, 3, 4, 2).reshape(frames * views, int(video.shape[-2]), int(video.shape[-1]), 3).numpy()
    queries = np.column_stack((raw[:, 2] * views + raw[:, 3], raw[:, :2])).astype(np.float32)
    return images, queries, views, frames


def output_arrays(output, *, views, frames, count, original_hw, benchmark):
    """Convert native world trajectories, never expose reference calibration."""
    def cpu(value):
        return value.detach().float().cpu().numpy()
    world = cpu(output.query_coords)
    logits = cpu(output.query_visibs)
    if world.shape != (1, frames * views, count, 3) or logits.shape != (1, frames * views, count):
        raise ValueError("TrackEverything native sparse output shape mismatch")
    world, logits = world[0], logits[0]
    w2c, k = cpu(output.extrinsics), cpu(output.intrinsics)
    if w2c.shape != (frames * views, 4, 4) or k.shape != (frames * views, 3, 3):
        raise ValueError("TrackEverything predicted camera shape mismatch")
    height, width = original_hw
    ih, iw = output.rgb.shape[-2:]
    if min(height, width, ih, iw) < 2:
        raise ValueError("Native resize transform requires image dimensions >=2")
    # Invert precisely the author's _preprocess scaling of its camera matrix.
    k[:, 0] *= (width - 1) / (iw - 1)
    k[:, 1] *= (height - 1) / (ih - 1)
    c2w = np.linalg.inv(w2c)
    if benchmark == "tapvid3d":
        if views != 1:
            raise ValueError("TAPVid3D requires one camera")
        local = np.einsum("tij,tnj->tni", w2c[:, :3, :3], world) + w2c[:, None, :3, 3]
        native = world
        visible = logits > 0
    elif benchmark == "mvtracker":
        # Native multiview emits a position for every target view/time. This
        # arithmetic reduction is a declared adapter extension, not GT fusion.
        native = world.reshape(frames, views, count, 3).mean(1)
        local = native
        visible = (logits.reshape(frames, views, count) > 0).any(1)
    else:
        raise ValueError("Unsupported benchmark mode")
    arrays = {"tracks_XYZ": local.astype(np.float32), "tracks_native_world": native.astype(np.float32),
              "visibility": visible.astype(bool),
              "predicted_c2w": c2w.reshape(frames, views, 4, 4).transpose(1, 0, 2, 3).astype(np.float32),
              "predicted_K": k.reshape(frames, views, 3, 3).transpose(1, 0, 2, 3).astype(np.float32)}
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("TrackEverything emitted nonfinite predictions")
    return arrays


def reverse_time_order(frames, views, *, device):
    """Reverse synchronized timesteps while preserving the camera axis."""
    import torch
    return torch.arange(frames * views, device=device).reshape(frames, views).flip(0).reshape(-1)


def stitch_backward(tracker, output, queries, *, views, frames, original_hw, num_iters=3, lift_fn=None):
    """Author-style backward inference in the same forward RGB-predicted gauge.

    Reuses preprocessed tensors directly; never estimates geometry twice and
    never sends resized geometry through the wrapper's resize operation again.
    """
    import torch
    if not (queries[:, 0] // views > 0).any():
        return output
    if lift_fn is None:
        from trackeverything.demo_hf import _unproject_2d_to_3d
        lift_fn = _unproject_2d_to_3d
    height, width = original_hw
    ih, iw = output.rgb.shape[-2:]
    scaled = queries.copy()
    scaled[:, 1] *= (iw - 1) / (width - 1)
    scaled[:, 2] *= (ih - 1) / (height - 1)
    query_world = lift_fn(scaled, output.depths, output.intrinsics, output.extrinsics)
    source_flat = torch.as_tensor(queries[:, 0], device=query_world.device, dtype=torch.long)
    source_time, source_view = source_flat // views, source_flat % views
    backward_query = query_world.clone()
    backward_query[0, :, 0] = (frames - 1 - source_time) * views + source_view
    order = reverse_time_order(frames, views, device=output.rgb.device)
    model = tracker.model
    previous_views = model.num_views
    model.num_views = views
    try:
        with torch.inference_mode():
            pred, _ = model(rgb_obs=output.rgb[order][None], depth_obs=output.depths[order][None],
                intrinsics=output.intrinsics[order][None], extrinsics=output.extrinsics[order][None],
                query_point=backward_query, num_iters=num_iters,
                dense_norm_mask=output.dense_source_mask[order][None, ..., None])
    finally:
        model.num_views = previous_views
    if pred.query_coords.shape != output.query_coords.shape or pred.query_visibs.shape != output.query_visibs.shape:
        raise ValueError("Native backward sparse output shape mismatch")
    target_time = torch.arange(frames, device=output.query_coords.device).repeat_interleave(views)
    mask = target_time[None, :, None] < source_time.to(output.query_coords.device)[None, None]
    output.query_coords = torch.where(mask[..., None], pred.query_coords[:, order], output.query_coords)
    output.query_visibs = torch.where(mask, pred.query_visibs[:, order], output.query_visibs)
    return output


def predict_case(tracker, inputs, *, benchmark, cpu_threads=8, geometry_chunk_observations=60, num_iters=3):
    images, queries, views, frames = prepare_inputs(inputs)
    if benchmark == "tapvid3d" and views != 1:
        raise ValueError("TAPVid3D requires one camera")
    # Keep complete synchronized timesteps at native geometry chunk boundaries.
    chunk = max(views * 2, geometry_chunk_observations // views * views)
    overlap = max(views, (chunk // (2 * views)) * views)
    tracker._vggt_omega_chunk_size, tracker._vggt_omega_overlap = chunk, overlap
    if tracker._geometry is not None:
        tracker._geometry.chunk_size, tracker._geometry.overlap = chunk, overlap
    output = tracker.infer(video=images, num_views=views, query_points_2d=queries,
                           mode="sparse", max_frames=None, frame_stride=1, start_frame=0,
                           num_threads=cpu_threads, mem_efficient=True, num_iters=num_iters)
    output = stitch_backward(tracker, output, queries, views=views, frames=frames,
                             original_hw=images.shape[1:3], num_iters=num_iters)
    arrays = output_arrays(output, views=views, frames=frames, count=len(queries),
                           original_hw=images.shape[1:3], benchmark=benchmark)
    return arrays, {"adapter": "trackeverything-sparse-rgb-v1", "input_condition": "rgb_only_unknown_calibration",
        "geometry_backend": "official vendored VGGT-Omega-1B-512; RGB predicted",
        "source_queries": "exact frozen x,y,time,view; time-major native index t*V+v",
        "temporal_policy": "author-style forward/backward stitch; pre-source logical timesteps use backward predictions",
        "backward_geometry": "same preprocessed RGB-predicted depth/K/w2c gauge; no re-estimation or second resize",
        "query_lifting": "official nearest border predicted depth plus unmodified query ray; official invalid-depth fallback",
        "visibility_semantics": "native query_visibs logit >0; MV union over target cameras",
        "multiview_position_reduction": "arithmetic mean over native target-camera world positions" if views > 1 else "none",
        "geometry_chunk_observations": chunk, "geometry_overlap_observations": overlap,
        "views": views, "frames": frames, "num_iters": num_iters}


def main(argv=None):
    p = argparse.ArgumentParser()
    for name in ("manifest", "output", "checkpoint", "geometry-checkpoint", "initializer-checkpoint", "author-code-root", "baseline-code-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--geometry-chunk-observations", type=int, default=60)
    p.add_argument("--num-iters", type=int, default=3)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(argv)
    manifest = json.loads(a.manifest.read_text())
    benchmark = manifest["benchmark"]
    if benchmark not in ("tapvid3d", "mvtracker") or min(a.cpu_threads, a.geometry_chunk_observations, a.num_iters) < 1:
        p.error("Supports TAPVid3D and RGB-only MVTracker only; threads/chunk must be positive")
    if benchmark == "mvtracker" and manifest.get("protocol", {}).get("mode") != "rgb_only_3d_query_anchor_sim3":
        p.error("MVTracker requires RGB-only query-anchor Sim3 manifest")
    sys.path.insert(0, str(a.baseline_code_root.resolve()))
    from multicamera4rc.benchmarks.released_baselines import run_adapter_predictions
    import torch
    torch.set_num_threads(a.cpu_threads)
    record = checkpoint_record(a.author_code_root, a.checkpoint, a.geometry_checkpoint, a.initializer_checkpoint)
    os.environ["TORCH_HOME"] = str(a.initializer_checkpoint.resolve().parents[2])
    sys.path.insert(0, str(a.author_code_root.resolve().parent))
    from trackeverything.demo_hf_multiview import MultiViewTracker
    weight = a.checkpoint / "model.pth" if a.checkpoint.is_dir() else a.checkpoint
    tracker = MultiViewTracker.from_pretrained(str(weight), device=a.device,
        geometry_backend="vggt_omega", vggt_omega_checkpoint=str(a.geometry_checkpoint),
        vggt_omega_image_resolution=512)
    def callback(inputs, case, index):
        return predict_case(tracker, inputs, benchmark=benchmark, cpu_threads=a.cpu_threads,
                            geometry_chunk_observations=a.geometry_chunk_observations, num_iters=a.num_iters)
    return run_adapter_predictions(a.manifest, a.output, method=METHOD, checkpoint=record,
        predict_case=callback, device=a.device, resume=a.resume,
        metric_support={"position": True, "native_visibility": True, "AJ": True, "OA": True},
        inference_settings={"target_time_policy": "all_encoded_target_observations", "prediction_kind": "trajectory",
            "author_code_root": str(a.author_code_root), "checkpoint": str(a.checkpoint),
            "geometry_checkpoint": str(a.geometry_checkpoint), "geometry_chunk_observations": a.geometry_chunk_observations,
            "initializer_checkpoint": str(a.initializer_checkpoint), "num_iters": a.num_iters,
            "temporal_inference": "bidirectional; same forward predicted geometry; timestep-preserving reverse",
            "visibility": "native sparse logits; MV any-target-camera union", "mode": "sparse"})


if __name__ == "__main__":
    print(main())
