#!/usr/bin/env python3
"""Frozen-query Point4D single-camera adapter using the author's direct decoder.

All supplied frames are encoded together. This is an explicit direct-query
extension, not the demo's frame-zero-only long-video chaining protocol.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

COMMIT = "4da8153711a4d37afbdea9cdc789db20ac9d3f4a"
WEIGHT_HASH = "dacdabdd8678da9f247f6ac51aefcd1771a94af4b677166a5683b6971c4950cb"
METHOD = "Point4D-direct-query"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def checkpoint_record(author_code_root, checkpoint):
    code, weight = Path(author_code_root).resolve(), Path(checkpoint).resolve()
    if weight.is_dir():
        weight = weight / "point4d_final.pt"
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=code, text=True).strip()
    if git("rev-parse", "HEAD") != COMMIT or git("status", "--porcelain"):
        raise ValueError("Point4D requires the pinned clean author checkout")
    if sha256(weight) != WEIGHT_HASH:
        raise ValueError("Point4D release checkpoint SHA256 mismatch")
    sources = {str(p.relative_to(code)): sha256(p) for p in sorted(code.rglob("*"))
               if p.is_file() and p.suffix in (".py", ".yaml", ".json")
               and ".git" not in p.parts and "__pycache__" not in p.parts}
    component = {"path": str(weight), "sha256": WEIGHT_HASH, "bytes": weight.stat().st_size,
        "source_root": str(code), "source_commit": COMMIT, "source_sha256": sources,
        "checkpoint_loading": "author build_model; reject unexpected or missing learned keys"}
    identity = {"method": METHOD, "components": {"model": component}}
    identity["sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"kind": "author_rgb_only_native_baseline", **identity,
            "native_multicamera4rc_loading": False, "refiner_loading": "not_applicable"}


def prepare_inputs(inputs, image_hw):
    """Keep byte-valued RGB and unmodified time/view IDs; resize pixel centers."""
    import torch
    x = torch.as_tensor(inputs.video).detach().cpu()
    if x.ndim != 5 or x.shape[0] != 1 or x.shape[2] != 3:
        raise ValueError("Point4D-direct-query supports single-camera [1,T,3,H,W]")
    if x.dtype != torch.uint8:
        if (not x.is_floating_point() or not torch.isfinite(x).all()
                or (x < 0).any() or (x > 255).any() or not torch.equal(x, x.round())):
            raise ValueError("RGB must contain finite integer byte values in 0..255")
        x = x.to(torch.uint8)
    raw = np.asarray(inputs.queries_xytv.detach().cpu(), dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 4 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("Expected nonempty finite [x,y,time,view] queries")
    frames, height, width = int(x.shape[1]), int(x.shape[-2]), int(x.shape[-1])
    if ((raw[:, 3] != 0).any() or not np.equal(raw[:, 2], np.floor(raw[:, 2])).all()
            or (raw[:, 2] < 0).any() or (raw[:, 2] >= frames).any()):
        raise ValueError("Frozen source time/view is invalid")
    th, tw = image_hw
    affine = np.array([[tw / width, 0, (tw / width - 1) / 2],
                       [0, th / height, (th / height - 1) / 2], [0, 0, 1]], dtype=np.float32)
    uv = raw[:, :2] * np.array([tw / width, th / height]) + affine[:2, 2]
    return x[0].permute(0, 2, 3, 1).numpy(), raw, uv.astype(np.float32), affine


def build_queries(xyz, uv, source, *, frames, scale, native=False):
    """Seed-major output; [xyz,p,s,t,c] preserves source and target frame IDs."""
    n = len(xyz)
    q = np.zeros((n, frames, 7), dtype=np.float32)
    q[..., :3] = xyz[:, None] / scale
    q[..., 3] = source[:, None]
    q[..., 4] = source[:, None]
    q[..., 5] = np.arange(frames)[None]
    q[..., 6] = 0 if native else np.arange(frames)[None]
    return q.reshape(-1, 7), np.repeat(uv, frames, axis=0)


def _source_xyz(cache, uv, source, device):
    """Bilinear predicted depth with border extension; preserve the query ray."""
    import torch
    import torch.nn.functional as F
    depth = cache["predictions"]["depth"][0].float()
    ks = cache["predictions"]["intrinsics"][0].float()
    height, width = depth.shape[-2:]
    out = np.empty((len(uv), 3), np.float32)
    for frame in np.unique(source):
        ids = np.flatnonzero(source == frame)
        coords = torch.as_tensor(uv[ids], dtype=torch.float32, device=device)
        grid = coords.clone()
        grid[:, 0] = 2 * grid[:, 0] / max(width - 1, 1) - 1
        grid[:, 1] = 2 * grid[:, 1] / max(height - 1, 1) - 1
        z = F.grid_sample(depth[frame][None, None], grid[None, None],
                          mode="bilinear", padding_mode="border", align_corners=True)[0, 0, 0]
        rays = torch.linalg.solve(ks[frame], torch.cat((coords, torch.ones_like(coords[:, :1])), 1).T).T
        out[ids] = (rays * z[:, None]).cpu().numpy()
    return out


def _decode(model, cache, q, uv, *, device, query_chunk_size, native_visibility):
    import torch
    points, visibility = [], []
    for start in range(0, len(q), query_chunk_size):
        qq = torch.as_tensor(q[start:start + query_chunk_size], device=device)[None]
        uu = torch.as_tensor(uv[start:start + query_chunk_size], device=device)[None]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model.forward_d4rt_only(cache, qq, d4rt_uvs=uu,
                                         memory_kv_cache=cache["memory_kv_cache"])
        points.append(out["d4rt_pred"][0].float().cpu().numpy())
        if native_visibility:
            if "d4rt_pred_vis_logits" not in out:
                raise ValueError("Pinned Point4D aux_vis head emitted no native visibility")
            visibility.append(out["d4rt_pred_vis_logits"][0].float().cpu().numpy().reshape(-1))
    return np.concatenate(points), np.concatenate(visibility) if visibility else None


def predict_case(model, inputs, *, device="cuda", image_hw=(294, 518), query_chunk_size=40000):
    import torch
    from point4d.inference.longtrack import LongTrackConfig, _encode_chunk
    from point4d.inference.geometry import w2c_matrix
    images, raw, uv, affine = prepare_inputs(inputs, image_hw)
    device = torch.device(device)
    source = raw[:, 2].astype(np.int64)
    frames, count = len(images), len(raw)
    cfg = LongTrackConfig(image_hw=image_hw, bf16=device.type == "cuda")
    with torch.inference_mode():
        cache = _encode_chunk(model, images, list(range(frames)), cfg, device)
        scale = float(cache["scale"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Point4D predicted invalid scene scale")
        xyz = _source_xyz(cache, uv, source, device)
        q, uvs = build_queries(xyz, uv, source, frames=frames, scale=scale)
        local, logits = _decode(model, cache, q, uvs, device=device,
                                query_chunk_size=query_chunk_size, native_visibility=True)
        native_q, _ = build_queries(xyz, uv, source, frames=frames, scale=scale, native=True)
        world, _ = _decode(model, cache, native_q, uvs, device=device,
                           query_chunk_size=query_chunk_size, native_visibility=False)
        c2w = np.stack([w2c_matrix(cache, 0) @ np.linalg.inv(w2c_matrix(cache, t)) for t in range(frames)])
        k = cache["predictions"]["intrinsics"][0].float().cpu().numpy()
    arrays = {"tracks_XYZ": (local.reshape(count, frames, 3) * scale).transpose(1, 0, 2),
              "tracks_native_world": (world.reshape(count, frames, 3) * scale).transpose(1, 0, 2),
              "visibility": (logits.reshape(count, frames).T > 0),
              "predicted_c2w": c2w.astype(np.float32)[None],
              "predicted_K": (np.linalg.inv(affine)[None] @ k).astype(np.float32)[None]}
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Point4D emitted nonfinite arrays")
    return arrays, {"adapter": "point4d-direct-query-v1", "input_condition": "rgb_only_unknown_calibration",
        "direct_query_extension": True, "long_video_chaining": False,
        "frames": frames, "resized_resolution": list(image_hw),
        "source_policy": "exact frozen time; predicted source-camera XYZ; source appearance retained",
        "pixel_transform": "OpenCV pixel-center resize; predicted depth bilinear border extension",
        "visibility_semantics": "native d4rt_pred_vis_logits > 0",
        "geometry_inputs": []}


def load_model(code, checkpoint, device):
    import torch
    sys.path.insert(0, str(Path(code).resolve()))
    from point4d.loader import build_model
    model = build_model()
    path = Path(checkpoint)
    if path.is_dir():
        path = path / "point4d_final.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    learned_missing = [k for k in missing if not k.endswith(("_resnet_mean", "_resnet_std"))]
    if learned_missing or unexpected:
        raise ValueError(f"Point4D checkpoint key mismatch: {learned_missing=}, {unexpected=}")
    return model.to(device).eval()


def main(argv=None):
    p = argparse.ArgumentParser()
    for name in ("manifest", "output", "checkpoint", "author-code-root", "baseline-code-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--image-height", type=int, default=294)
    p.add_argument("--image-width", type=int, default=518)
    p.add_argument("--query-chunk-size", type=int, default=40000)
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(argv)
    if json.loads(a.manifest.read_text())["benchmark"] != "tapvid3d":
        p.error("Point4D direct-query adapter supports TAPVid-3D single-camera manifests only")
    if min(a.image_height, a.image_width) < 14 or a.image_height % 14 or a.image_width % 14 or min(a.query_chunk_size, a.cpu_threads) < 1:
        p.error("Image dimensions must be positive multiples of 14; chunks/threads must be positive")
    sys.path.insert(0, str(a.baseline_code_root.resolve()))
    from multicamera4rc.benchmarks.released_baselines import run_adapter_predictions
    import torch
    torch.set_num_threads(a.cpu_threads)
    record = checkpoint_record(a.author_code_root, a.checkpoint)
    model = load_model(a.author_code_root, a.checkpoint, a.device)
    def callback(inputs, case, index):
        return predict_case(model, inputs, device=a.device, image_hw=(a.image_height, a.image_width), query_chunk_size=a.query_chunk_size)
    return run_adapter_predictions(a.manifest, a.output, method=METHOD, checkpoint=record,
        predict_case=callback, device=a.device, resume=a.resume,
        metric_support={"position": True, "native_visibility": True, "AJ": True, "OA": True},
        inference_settings={"target_time_policy": "all_encoded_target_observations", "prediction_kind": "trajectory",
            "adapter_protocol": "direct_query_extension_no_long_video_chaining", "author_code_root": str(a.author_code_root),
            "checkpoint": str(a.checkpoint), "image_hw": [a.image_height, a.image_width], "query_chunk_size": a.query_chunk_size})


if __name__ == "__main__":
    print(main())
