#!/usr/bin/env python3
"""Pinned UniQuery4R RGB-only native-grid WorldTrack adapter.

Uses the author forward, preprocessing size rule and pair convention. The frozen
WorldTrack scorer selects queries from the exported dense grid. No visibility
head is mapped to a visibility metric, and no GT geometry enters inference.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

METHOD = "UniQuery4R"
CODE_COMMIT = "95119742e61c64c27b09349035019c8cd7b41e62"
WEIGHT_SHA = "cd71b27bbb0c7a22de36d1eacebea84d93196a4da6296c550b1ea27f3a1c5539"
SCORER_COMMIT = "0f9a3f44a7ebac76600cd31ec9eea5228ad7db91"
SUPPORT = {"position": True, "native_visibility": False, "AJ": False, "OA": False}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def checkpoint_identity(source_root, checkpoint):
    root, weight = Path(source_root).resolve(), Path(checkpoint).resolve()
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if rev != CODE_COMMIT or dirty:
        raise ValueError("UniQuery4R requires the clean pinned author checkout")
    if sha(weight) != WEIGHT_SHA:
        raise ValueError("UniQuery4R release weight hash differs")
    source_sha = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*.py"))
                  if ".git" not in p.parts and "__pycache__" not in p.parts}
    component = {"path": str(weight), "sha256": WEIGHT_SHA, "bytes": weight.stat().st_size,
                 "source_root": str(root), "source_commit": CODE_COMMIT,
                 "source_sha256": source_sha,
                 "checkpoint_loading": "author load_checkpoint(strict=True); full pinned release"}
    identity = {"method": METHOD, "components": {"model": component}}
    return {"kind": "author_rgb_only_native_baseline", **identity, "sha256": json_sha(identity),
            "native_multicamera4rc_loading": False, "refiner_loading": "not_applicable"}


def validate_manifest(manifest):
    if (manifest.get("benchmark") != "worldtrack"
            or manifest.get("coordinates") != "first_camera_world"
            or manifest.get("official_sources", {}).get("commit") != SCORER_COMMIT):
        raise ValueError("UniQuery4R requires pinned first-camera WorldTrack manifests")
    task = manifest.get("protocol", {}).get("task")
    if task not in ("trajectory", "reconstruction"):
        raise ValueError("An explicit WorldTrack task is required")
    if manifest.get("leakage", {}).get("checkpoint_sha256"):
        raise ValueError("Manifest cannot be bound to another model checkpoint")
    for case in manifest["cases"]:
        if case["views"] != [0] or not 1 <= case["frames"] <= 64:
            raise ValueError("Frozen WorldTrack requires one view and the official first64 frames")
        if min(case["height"], case["width"]) < 2:
            raise ValueError("Image dimensions must be at least2")
    return task


def prepare_video(video, max_size=504):
    """Use the author's resize rule and identical Pillow bicubic RGB transform."""
    import torch
    from PIL import Image
    from uniquery4r.utils.load_fn import _keep_ratio_target_shape
    video = torch.as_tensor(video)
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[2] != 3 or video.dtype != torch.uint8:
        raise ValueError("Expected frozen single-view uint8 [1,T,3,H,W] RGB")
    h, w = map(int, video.shape[-2:])
    ph, pw = _keep_ratio_target_shape(h, w, max_size, 14)
    images = []
    for rgb in video[0].permute(0, 2, 3, 1).cpu().numpy():
        pil = Image.fromarray(rgb).resize((pw, ph), Image.Resampling.BICUBIC)
        images.append(torch.from_numpy(np.array(pil, copy=True)).permute(2, 0, 1).float() / 255.)
    return torch.stack(images)


def dense_predictions(pred, *, frames, task):
    """Map official self/cross pairs to frame-ordered dense world-coordinate fields."""
    pairs = [tuple(map(int, p)) for p in pred["pair_idx"]]
    expected = [(0, t) for t in range(1, frames)] + [(t, t) for t in range(frames)]
    if pairs != expected:
        raise ValueError("Author pair indexing differs from the pinned forward")
    warp = pred["warp3d"]
    if warp.ndim != 5 or warp.shape[0] != 1 or warp.shape[1] != len(pairs) or warp.shape[2] != 3:
        raise ValueError("Expected official dense warp3d [1,P,3,H,W]")
    selected = [pairs.index((0, t) if task == "trajectory" and t else (t, t))
                for t in range(frames)]
    out = warp[0, selected].detach().float().permute(0, 2, 3, 1).cpu().numpy()
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite UniQuery4R points; cannot silently skip/repair a case")
    return out


def load_model(source_root, checkpoint, chunk_size, device):
    import torch
    sys.path.insert(0, str(Path(source_root).resolve()))
    from uniquery4r.models.uniquery4r import UniQuery4R
    model = UniQuery4R(chunk_size=chunk_size)
    model.load_checkpoint(checkpoint, strict=True)
    return model.eval().to(device), {"strict": True, "missing_keys": [], "unexpected_keys": [],
                                    "parameter_count": sum(p.numel() for p in model.parameters())}


class Predictor:
    def __init__(self, manifest, source_root, checkpoint, device, chunk_size, max_size):
        self.task = validate_manifest(manifest)
        self.source_root, self.checkpoint = source_root, checkpoint
        self.device, self.chunk_size, self.max_size = device, chunk_size, max_size
        self.model = self.load_receipt = None

    def load(self):
        if self.model is None:
            self.model, self.load_receipt = load_model(self.source_root, self.checkpoint,
                                                     self.chunk_size, self.device)
        return self.model

    def __call__(self, inputs, case, index):
        import torch
        started = time.monotonic()
        expected = (1, case["frames"], 3, case["height"], case["width"])
        if tuple(inputs.video.shape) != expected:
            raise ValueError("Frozen WorldTrack RGB shape differs")
        if not np.array_equal(inputs.frame_indices.cpu().numpy().reshape(-1), np.arange(case["frames"])):
            raise ValueError("Frozen WorldTrack frame order differs")
        images = prepare_video(inputs.video, self.max_size).to(self.device)
        with torch.inference_mode():
            result = self.load()(images)
            field = dense_predictions(result, frames=case["frames"], task=self.task)
        return {"tracks_XYZ": field}, {
            "prediction_seconds": time.monotonic() - started,
            "model_load_receipt": self.load_receipt,
            "native_output_hw": list(field.shape[1:3]), "task": self.task,
            "prediction_coordinate_system": "author warp3d absolute frame0-reference world",
            "dense_output_semantics": "frame0 pixel trajectories" if self.task == "trajectory"
                                      else "self-pair pixel point maps",
            "visibility_semantics": "not exported; confidence heads are not visibility probabilities",
            "input_condition": "RGB only; no GT depth/calibration/XYZ",
            "all_frozen_frames_retained": True}


@contextmanager
def dense_contract(baselines, max_size=504):
    """Use author's native grid; official scorer performs its unchanged readout."""
    from uniquery4r.utils.load_fn import _keep_ratio_target_shape
    previous = baselines._prediction_specs
    def specs(manifest, case, input_path):
        validate_manifest(manifest)
        h, w = _keep_ratio_target_shape(case["height"], case["width"], max_size, 14)
        shape = (case["frames"], h, w, 3)
        return {"tracks_XYZ": shape, "visibility": shape[:-1]}
    baselines._prediction_specs = specs
    try:
        yield
    finally:
        baselines._prediction_specs = previous


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--benchmark-code-root", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--query-chunk-size", type=int, default=5000)
    p.add_argument("--max-size", type=int, default=504)
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--memory-fraction", type=float, default=.85)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    a = p.parse_args(argv)
    if a.query_chunk_size < 1 or a.max_size < 14 or a.max_size % 14 or a.cpu_threads < 1:
        p.error("Invalid query chunk size, max-size or CPU threads")
    if not 0 < a.memory_fraction <= 1:
        p.error("Memory fraction must be in (0,1]")
    if a.preflight_only and a.device != "cpu":
        p.error("Preflight must use CPU")
    sys.path.insert(0, str(a.benchmark_code_root.resolve()))
    sys.path.insert(0, str(a.source_root.resolve()))
    import torch
    from multicamera4rc.benchmarks import released_baselines
    from multicamera4rc.benchmarks.released_data import load_manifest
    torch.set_num_threads(a.cpu_threads)
    manifest = load_manifest(a.manifest, inference_only=True)
    task = validate_manifest(manifest)
    checkpoint = checkpoint_identity(a.source_root, a.checkpoint)
    predictor = Predictor(manifest, a.source_root, a.checkpoint, a.device,
                          a.query_chunk_size, a.max_size)
    if a.preflight_only:
        predictor.load()
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU preflight unexpectedly initialized CUDA")
        print(json.dumps({"status": "UNIQUERY4R_CPU_PREFLIGHT_OK", "checkpoint": checkpoint,
            "load_receipt": predictor.load_receipt, "manifest_sha256": sha(a.manifest),
            "cases": len(manifest["cases"]), "task": task, "gpu_used": False}, indent=2))
        return
    if torch.device(a.device).type == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.memory_fraction, torch.device(a.device))
    settings = {"method": METHOD, "prediction_kind": task,
        "target_time_policy": "author_joint_encoding_all_frozen_frames",
        "preprocessing": "author keep-ratio504; Pillow bicubic RGB/255; patch14",
        "native_output_grid": True, "max_size": a.max_size,
        "query_chunk_size": a.query_chunk_size, "reference_global_frame": 0,
        "query_sampling": "unchanged official WorldTrack evaluator",
        "output_contract": "native dense tracks_XYZ only", "adapter_sha256": sha(__file__)}
    with dense_contract(released_baselines, a.max_size):
        report = released_baselines.run_adapter_predictions(a.manifest, a.output, method=METHOD,
            checkpoint=checkpoint, predict_case=predictor, metric_support=dict(SUPPORT),
            device=a.device, resume=a.resume, inference_settings=settings)
    print(report, flush=True)


if __name__ == "__main__":
    main()
