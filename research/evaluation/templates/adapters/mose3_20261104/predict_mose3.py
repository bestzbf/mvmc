#!/usr/bin/env python3
"""MoSE3 adapter for the frozen single-camera tracking benchmark contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

COMMIT = "bfd2ab3dbda6b2808473855567137d37cd6413d4"
WEIGHT_HASH = "fcbd51ff01c2bc2f76ca9db3037e0a4055f6bc5ea202a5e1a9f57a91107d183c"
CONFIG_HASH = "7e7c60d3ddf0781195f6e5a6f500f6321cabe031b04c7334d4bf5729f10c0d4e"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def checkpoint_record(code, checkpoint):
    code, checkpoint = Path(code).resolve(), Path(checkpoint).resolve()
    weight, config = checkpoint / "model.safetensors", checkpoint / "config.json"
    for path, expected in ((weight, WEIGHT_HASH), (config, CONFIG_HASH)):
        if sha256(path) != expected:
            raise ValueError(f"MoSE3 release hash mismatch: {path}")
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=code, text=True).strip()
    if git("rev-parse", "HEAD") != COMMIT or git("status", "--porcelain"):
        raise ValueError("MoSE3 source checkout differs from the pinned clean revision")
    sources = {str(p.relative_to(code)): sha256(p) for p in sorted(code.rglob("*.py"))
               if ".git" not in p.parts and "__pycache__" not in p.parts}
    component = {"path": str(weight), "resolved_path": str(weight), "sha256": WEIGHT_HASH,
                 "bytes": weight.stat().st_size,
                 "config": {"path": str(config), "sha256": CONFIG_HASH, "bytes": config.stat().st_size},
                 "source_root": str(code), "source_commit": COMMIT, "source_sha256": sources,
                 "checkpoint_loading": "MoSE3.from_pretrained strict=True"}
    identity = {"method": "MoSE3", "components": {"model": component}}
    identity["sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"kind": "author_rgb_only_native_baseline", **identity,
            "native_multicamera4rc_loading": False, "refiner_loading": "not_applicable"}


def sample_dense(dense, xy, width, height):
    """Sample [T,H,W,C] at original-pixel coordinates with border extension."""
    import torch
    import torch.nn.functional as F
    if dense.ndim != 4 or xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("Invalid dense predictions or query coordinates")
    # OpenCV resize maps pixel centers by (x + .5) * new/old - .5.
    grid = torch.as_tensor(xy, dtype=dense.dtype, device=dense.device).clone()
    grid[:, 0] = 2 * (grid[:, 0] + 0.5) / width - 1
    grid[:, 1] = 2 * (grid[:, 1] + 0.5) / height - 1
    grid = grid[None, None].expand(dense.shape[0], -1, -1, -1)
    return F.grid_sample(dense.permute(0, 3, 1, 2), grid, mode="bilinear",
                         padding_mode="border", align_corners=False)[:, :, 0, :].permute(0, 2, 1)


def lossless_rgb_uint8(video):
    """Recover byte RGB from a framework tensor without rounding any pixels."""
    import torch

    video = torch.as_tensor(video)
    if (video.ndim != 5 or video.shape[0] != 1 or video.shape[2] != 3
            or min(video.shape) < 1):
        raise ValueError("MoSE3 accepts nonempty single-camera [1,T,3,H,W] only")
    if (video.dtype == torch.bool or video.is_complex()
            or not torch.isfinite(video).all() or video.min() < 0 or video.max() > 255):
        raise ValueError("MoSE3 RGB must be finite byte values in 0..255")
    if video.is_floating_point() and not torch.equal(video, video.round()):
        raise ValueError("MoSE3 float RGB must be losslessly representable as uint8")
    return video.to(device="cpu", dtype=torch.uint8)


def predict_case(model, inputs, *, device="cuda", max_side=518):
    import cv2
    import torch
    from mose3.utils.video_io import _target_size

    video = lossless_rgb_uint8(inputs.video)
    raw = torch.as_tensor(inputs.queries_xytv).detach().cpu().numpy()
    _, frames, _, height, width = video.shape
    if raw.ndim != 2 or raw.shape[1] != 4 or not np.isfinite(raw).all():
        raise ValueError("Expected finite x,y,time,view queries")
    if not np.equal(raw[:, 2:], np.floor(raw[:, 2:])).all() or (raw[:, 3] != 0).any() or (raw[:, 2] < 0).any() or (raw[:, 2] >= frames).any():
        raise ValueError("Invalid source frame or camera")
    tw, th = _target_size(width, height, max_side, max_side)
    images = video[0].permute(0, 2, 3, 1).cpu().numpy()
    images = np.stack([cv2.resize(x, (tw, th), interpolation=cv2.INTER_AREA)
                       if (tw, th) != (width, height) else x for x in images])
    device = torch.device(device)
    images = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float().div_(255).unsqueeze(0)
    count = len(raw)
    local = np.empty((frames, count, 3), dtype=np.float32)
    world = np.empty_like(local)
    visibility = np.empty((frames, count), dtype=bool)
    ref_pose = ref_k = None
    with torch.inference_mode():
        for q in np.unique(raw[:, 2].astype(np.int64)):
            ids = np.flatnonzero(raw[:, 2] == q)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(images, query_frame_idx=int(q))
            points = out["tracks_local"][0].float()
            logits = out["visibility_logits"][0].float()
            poses = out["camera_poses"][0].float()
            poses = torch.linalg.inv(poses[0:1]) @ poses
            focal = out["track_focal"][0].float()
            k = torch.tensor([[1., 0., (tw - 1.) / 2], [0., 1., (th - 1.) / 2], [0., 0., 1.]], device=device)
            k[0, 0], k[1, 1] = focal[0], focal[1]
            resize = torch.tensor([[tw / width, 0., (tw / width - 1) / 2],
                                   [0., th / height, (th / height - 1) / 2], [0., 0., 1.]], device=device)
            k = torch.linalg.inv(resize) @ k
            if not torch.isfinite(poses).all() or not torch.isfinite(k).all() or (focal <= 0).any():
                raise ValueError("MoSE3 emitted invalid pose or focal length")
            if ref_pose is None:
                ref_pose, ref_k = poses.cpu(), k.cpu()
            elif not torch.allclose(poses.cpu(), ref_pose, rtol=1e-4, atol=1e-5) or not torch.allclose(k.cpu(), ref_k, rtol=1e-4, atol=1e-5):
                raise ValueError("MoSE3 camera geometry varies by query source frame")
            selected = sample_dense(points, raw[ids, :2], width, height)
            selected_vis = sample_dense(logits[..., None], raw[ids, :2], width, height)[..., 0]
            xyz_world = torch.einsum("tij,tnj->tni", poses[:, :3, :3], selected) + poses[:, None, :3, 3]
            local[:, ids] = selected.cpu().numpy()
            world[:, ids] = xyz_world.cpu().numpy()
            visibility[:, ids] = (selected_vis > 0).cpu().numpy()
    if ref_pose is None:
        raise ValueError("No queries in frozen case")
    arrays = {"tracks_XYZ": local, "tracks_native_world": world, "visibility": visibility,
              "predicted_c2w": ref_pose.numpy()[None],
              "predicted_K": np.broadcast_to(ref_k.numpy(), (1, frames, 3, 3)).copy()}
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("MoSE3 emitted non-finite output")
    return arrays, {"adapter": "mose3-dense-query-v2-lossless-rgb", "input_condition": "rgb_only_unknown_calibration",
                    "source_frames": sorted(np.unique(raw[:, 2].astype(int)).tolist()),
                    "sampling": "bilinear pixel-center, border extension", "frames": int(frames),
                    "resized_resolution": [int(th), int(tw)], "visibility_semantics": "logit > 0"}


def main(argv=None):
    p = argparse.ArgumentParser()
    for name in ("manifest", "output", "checkpoint", "author-code-root", "baseline-code-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-side", type=int, default=518)
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(argv)
    manifest = json.loads(a.manifest.read_text())
    if manifest["benchmark"] != "tapvid3d" or a.max_side < 14 or a.max_side % 14 or a.cpu_threads < 1:
        p.error("Only TAPVid-3D single-camera cases are supported; max-side must be >=14 and divisible by 14")
    sys.path.insert(0, str(a.baseline_code_root.resolve()))
    from multicamera4rc.benchmarks.released_baselines import run_adapter_predictions
    import torch
    torch.set_num_threads(a.cpu_threads)
    record = checkpoint_record(a.author_code_root, a.checkpoint)
    sys.path.insert(0, str(a.author_code_root.resolve()))
    from mose3.models.mose3 import MoSE3
    model = MoSE3.from_pretrained(str(a.checkpoint), strict=True).to(a.device).eval()
    def callback(inputs, case, index):
        return predict_case(model, inputs, device=a.device, max_side=a.max_side)
    return run_adapter_predictions(a.manifest, a.output, method="MoSE3", checkpoint=record,
        predict_case=callback, device=a.device, resume=a.resume,
        metric_support={"position": True, "native_visibility": True, "AJ": True, "OA": True},
        inference_settings={"target_time_policy": "all_encoded_target_observations",
            "prediction_kind": "trajectory", "max_side": a.max_side,
            "rgb_conversion": "lossless_uint8_from_integral_0_255_framework_values",
            "sampling": "bilinear_pixel_center_border_extension", "author_code_root": str(a.author_code_root),
            "checkpoint": str(a.checkpoint)})


if __name__ == "__main__":
    print(main())
