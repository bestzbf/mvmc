#!/usr/bin/env python3
"""ReLift + author AllTracker/DA3, RGB-only frozen64 WorldTrack trajectories.

This is the released pipeline on the project's frozen64 protocol, not a claim
to reproduce the paper's84-frame table. All cameras/depth come from DA3. The
corrector sees the author float16 prepared-window quantization and defaults.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np

METHOD = "ReLift+AllTracker+DA3"
CODE_COMMIT = "98b042546e13ee3abfc0ce366b0dc44cc05c0ab8"
DA3_COMMIT = "41736238f5bced4debf3f2a12375d2466874866d"
ALLTRACKER_COMMIT = "c1ed462b7cfe9b4c6d07d8880ae5ed103b0848e0"
CORRECTOR_SHA = "29b93f3241f5282a5d6dd61e8f0e2f7b218991ec7d2845a16cf30e2f075c9e26"
DA3_SHA = "8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c"
DA3_CONFIG_SHA = "09adf89474017e717bc05aa86fd3a378708ba8914b036d61874eced328069468"
ALLTRACKER_SHA = "ffd9ebcfb6d206d594b646999a150540f92c049cf9b2bf940facf7123f62aa1d"
CONVNEXT_SHA = "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"
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


def _source(root, revision):
    root = Path(root).resolve()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if actual != revision or dirty:
        raise ValueError(f"Clean pinned author source required: {root}")
    files = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*.py"))
             if ".git" not in p.parts and "__pycache__" not in p.parts}
    return {"source_root": str(root), "source_commit": revision, "source_sha256": files}


def checkpoint_identity(source_root, checkpoint, frontend_weights_root):
    root, weight, front = map(lambda p: Path(p).resolve(),
                             (source_root, checkpoint, frontend_weights_root))
    specifications = {
        "corrector": (weight, CORRECTOR_SHA, root, CODE_COMMIT),
        "geometry": (front / "da3/model.safetensors", DA3_SHA,
                     root / "third_party/Depth-Anything-3", DA3_COMMIT),
        "tracker": (front / "alltracker/alltracker.pth", ALLTRACKER_SHA,
                    root / "third_party/alltracker", ALLTRACKER_COMMIT)}
    components = {}
    for name, (path, expected, source, commit) in specifications.items():
        if sha(path) != expected:
            raise ValueError(f"Pinned author release hash differs for {name}")
        components[name] = {"path": str(path), "sha256": expected,
            "bytes": path.stat().st_size, **_source(source, commit),
            "checkpoint_loading": "full author release; strict state loading"}
    config = front / "da3/config.json"
    if sha(config) != DA3_CONFIG_SHA:
        raise ValueError("DA3 release config differs")
    components["geometry"]["config"] = {"path": str(config), "sha256": DA3_CONFIG_SHA,
                                          "bytes": config.stat().st_size}
    components["geometry"].update(hf_repo="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        hf_revision="b2359bdf726fb44ef62acca04d629dcf158053e7")
    components["tracker"].update(hf_repo="aharley/alltracker",
        hf_revision="c8ba31225828cab9ec512d0627f0f826e641a877")
    initializer = front / "torch_cache/hub/checkpoints/convnext_tiny-983f1562.pth"
    if sha(initializer) != CONVNEXT_SHA:
        raise ValueError("AllTracker's private torchvision constructor initializer differs")
    components["tracker_constructor_initializer"] = {
        "path": str(initializer), "sha256": CONVNEXT_SHA, "bytes": initializer.stat().st_size,
        "official_url": "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
        "usage": "constructor initialization only; retained model state overwritten by strict released AllTracker load"}
    identity = {"method": METHOD, "components": components}
    return {"kind": "author_rgb_only_native_baseline", **identity, "sha256": json_sha(identity),
            "native_multicamera4rc_loading": False, "refiner_loading": "not_applicable"}


def validate_manifest(manifest):
    if (manifest.get("benchmark") != "worldtrack"
            or manifest.get("coordinates") != "first_camera_world"
            or manifest.get("official_sources", {}).get("commit") != SCORER_COMMIT
            or manifest.get("protocol", {}).get("task") != "trajectory"):
        raise ValueError("ReLift supports pinned first-camera WorldTrack trajectories only")
    if manifest.get("leakage", {}).get("checkpoint_sha256"):
        raise ValueError("Manifest is bound to another model checkpoint")
    for case in manifest["cases"]:
        if case["views"] != [0] or not 1 <= case["frames"] <= 64:
            raise ValueError("Frozen WorldTrack requires one view and official first64 frames")
    return "trajectory"


def native_hw(height, width, process_res=504):
    """DA3 upper_bound_resize size followed by nearest14, with ties rounded up."""
    if min(height, width) < 1:
        raise ValueError("RGB dimensions must be positive")
    scale = process_res / max(height, width)
    h, w = max(1, int(round(height * scale))), max(1, int(round(width * scale)))
    def nearest(n):
        down = (n // 14) * 14
        up = down + 14
        return max(1, up if abs(up - n) <= abs(n - down) else down)
    return nearest(h), nearest(w)


def prepared_window(geometry, tracks, visibility, device):
    """Replay author on-disk float16 arrays without writing unbound caches."""
    import torch
    depth = np.asarray(geometry["depth"])
    if depth.ndim != 3:
        raise ValueError("DA3 depth must be [T,H,W]")
    t, h, w = depth.shape
    shapes = {"tracks": (t, h, w, 2), "vis": (t, h, w),
              "intrinsics": (t, 3, 3), "poses": (t, 4, 4)}
    arrays = {"tracks": tracks, "vis": visibility, "depth": depth,
              "intrinsics": geometry["intrinsics"], "poses": geometry["poses"]}
    for name, shape in shapes.items():
        if np.shape(arrays[name]) != shape:
            raise ValueError(f"Author provider shape mismatch for {name}: {np.shape(arrays[name])} != {shape}")
    out = {"T": t, "H": h, "W": w}
    for name, value in arrays.items():
        dtype = np.float16 if name in ("tracks", "vis", "depth") else np.float32
        value = np.asarray(value).astype(dtype).astype(np.float32)
        out[name] = torch.as_tensor(np.ascontiguousarray(value), device=device)
    return out


def load_da3_strict(model, checkpoint):
    """Strict safetensors load, validating the release's six shared LN aliases.

    The author constructs one LayerNorm object reused across four auxiliary
    heads. Safetensors stores its bytes once and records the other names as
    metadata aliases. A plain load_file/state_dict load mislabels those aliases
    as six missing tensors. load_model verifies actual shared storage instead.
    """
    from safetensors import safe_open
    from safetensors.torch import load_model
    prefix = "model.da3.head.scratch.output_conv2_aux."
    aliases = {f"{prefix}{i}.2.{kind}": f"{prefix}0.2.{kind}"
               for i in (1, 2, 3) for kind in ("weight", "bias")}
    with safe_open(str(checkpoint), framework="pt") as f:
        metadata = f.metadata() or {}
    parameters = dict(model.named_parameters(remove_duplicate=False))
    for alias, target in aliases.items():
        if metadata.get(alias) != target:
            raise ValueError("DA3 safetensors shared-alias metadata differs")
        if alias not in parameters or target not in parameters or parameters[alias] is not parameters[target]:
            raise ValueError("DA3 author LayerNorm alias is no longer the same Parameter object")
        a, b = parameters[alias], parameters[target]
        if (a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
                or a.storage_offset() != b.storage_offset() or a.shape != b.shape or a.stride() != b.stride()):
            raise ValueError("DA3 shared LayerNorm alias no longer has identical tensor storage")
    missing, unexpected = load_model(model, str(checkpoint), strict=True, device="cpu")
    return {"loading": "safetensors.load_model(strict=True); exact shared-LayerNorm aliases verified",
            "missing_keys": sorted(missing), "unexpected_keys": sorted(unexpected),
            "verified_serialization_aliases": aliases,
            "parameter_object_and_storage_identity_verified": True,
            "shared_tensor_loader_reference": "https://huggingface.co/docs/safetensors/torch_shared_tensors"}


class AuthorModuleNamespace:
    """Keep AllTracker's global utils/nets packages private across lazy imports."""
    def __init__(self, source_root):
        self.root = Path(source_root).resolve()
        self.modules = {}

    @staticmethod
    def selected(name):
        return name in ("utils", "nets") or name.startswith(("utils.", "nets."))

    @contextmanager
    def activate(self):
        previous = {k: v for k, v in sys.modules.items() if self.selected(k)}
        path = list(sys.path)
        for name in previous:
            del sys.modules[name]
        sys.modules.update(self.modules)
        sys.path.insert(0, str(self.root))
        importlib.invalidate_caches()
        # AllTracker uses implicit namespace packages. A later regular utils
        # package otherwise wins Python's namespace-package search despite the
        # author directory being first on sys.path. Bind only that directory.
        for name in ("utils", "nets"):
            directory = self.root / name
            if name not in sys.modules and directory.is_dir() and not (directory / "__init__.py").exists():
                package = types.ModuleType(name)
                package.__package__ = name
                package.__path__ = [str(directory)]
                package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
                package.__spec__.submodule_search_locations = package.__path__
                sys.modules[name] = package
        try:
            yield
            active = {k: v for k, v in sys.modules.items() if self.selected(k)}
            for name, module in active.items():
                origin = getattr(module, "__file__", None)
                package_paths = getattr(module, "__path__", [])
                owned = (origin is not None and Path(origin).resolve().is_relative_to(self.root))
                if origin is None:
                    owned = bool(package_paths) and all(Path(p).resolve().is_relative_to(self.root) for p in package_paths)
                if not owned:
                    raise ValueError(f"AllTracker imported a foreign {name} namespace")
            self.modules = active
        finally:
            for name in list(sys.modules):
                if self.selected(name):
                    del sys.modules[name]
            sys.modules.update(previous)
            sys.path[:] = path


class Predictor:
    def __init__(self, manifest, source_root, checkpoint, frontend_weights_root, device):
        validate_manifest(manifest)
        self.root, self.checkpoint, self.front = map(Path, (source_root, checkpoint, frontend_weights_root))
        self.device = device
        self.corrector = self.geometry = self.tracker = self.load_receipt = None
        self.tracker_namespace = AuthorModuleNamespace(self.root / "third_party/alltracker")

    def load(self):
        if self.corrector is None:
            initializer = self.front / "torch_cache/hub/checkpoints/convnext_tiny-983f1562.pth"
            if sha(initializer) != CONVNEXT_SHA:
                raise ValueError("Private AllTracker constructor initialization asset is missing or changed")
            os.environ["TORCH_HOME"] = str((self.front / "torch_cache").resolve())
            import torch
            torch.hub.set_dir(str((self.front / "torch_cache/hub").resolve()))
            sys.path.insert(0, str(self.root.resolve()))
            from relift.checkpoint import load
            from relift.providers.da3 import DepthAnything3Provider
            from relift.providers.alltracker import AllTrackerProvider
            self.corrector, info = load(self.checkpoint, self.device)
            self.geometry = DepthAnything3Provider(str(self.front / "da3"), self.device,
                process_res=504, process_res_method="upper_bound_resize", ref_view_strategy="saddle_balanced")
            # The generic HF mixin need not default to strict loading. Verify
            # all actual parameters using the tensor-sharing-aware strict API.
            geometry_load = load_da3_strict(self.geometry.model, self.front / "da3/model.safetensors")
            with self.tracker_namespace.activate():
                self.tracker = AllTrackerProvider(str(self.front / "alltracker/alltracker.pth"),
                                                 self.device, window=16, iters=4)
            self.load_receipt = {"strict": True, "corrector": info,
                "geometry": geometry_load,
                "tracker": "author strict=True; private utils/nets namespace during construction and every inference call"}

    def __call__(self, inputs, case, index):
        import torch
        if torch.device(self.device).type != "cuda":
            raise ValueError("Author ReLift inference sampling/group partition require CUDA; CPU preflight only")
        expected = (1, case["frames"], 3, case["height"], case["width"])
        if tuple(inputs.video.shape) != expected or inputs.video.dtype != torch.uint8:
            raise ValueError("Exact frozen single-view uint8 RGB required")
        if not np.array_equal(inputs.frame_indices.cpu().numpy().reshape(-1), np.arange(case["frames"])):
            raise ValueError("WorldTrack first64 frame order differs")
        started = time.monotonic()
        self.load()
        from relift.rays import build_rays
        from relift.infer import predict_dense
        frames = inputs.video[0].permute(0, 2, 3, 1).cpu().numpy()
        geo = self.geometry.run(frames)
        with self.tracker_namespace.activate():
            tracks, vis, _ = self.tracker.run(geo["images"])
        win = prepared_window(geo, tracks, vis, self.device)
        if (win["T"], win["H"], win["W"]) != (case["frames"], *native_hw(case["height"], case["width"])):
            raise ValueError("DA3 provider unexpectedly changed the frozen temporal/native grid")
        origin, direction, distance = build_rays(win)
        direction, distance, vis = [torch.nan_to_num(x) for x in
                                   (direction, distance, win["vis"])]
        field = predict_dense(self.corrector, origin, direction, distance, vis,
                              group=1024, batch=8, seed=0).float().cpu().numpy()
        if not np.isfinite(field).all():
            raise ValueError("Nonfinite ReLift output")
        return {"tracks_XYZ": field}, {"prediction_seconds": time.monotonic() - started,
            "model_load_receipt": self.load_receipt, "native_output_hw": list(field.shape[1:3]),
            "input_condition": "RGB only; DA3-predicted depth/intrinsics/cameras; AllTracker RGB tracks",
            "prediction_coordinate_system": "first-camera world; author build_rays relative-pose transform",
            "visibility_semantics": "not scored by WorldTrack; tracker confidence is used only by author corrector",
            "protocol_extension": "same released pipeline on frozen first64 frames; paper table uses84",
            "corrector_settings": {"group": 1024, "batch": 8, "seed": 0, "vis_gate": .85},
            "prepared_window_quantization": "author float16 tracks/visibility/depth, float32 geometry",
            "alltracker_import_isolation": "private author utils/nets cache for construction and inference; previous cache restored",
            "all_frozen_frames_retained": True}


@contextmanager
def dense_contract(baselines):
    previous = baselines._prediction_specs
    def specs(manifest, case, input_path):
        validate_manifest(manifest)
        h, w = native_hw(case["height"], case["width"])
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
    p.add_argument("--frontend-weights-root", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--memory-fraction", type=float, default=.85)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    a = p.parse_args(argv)
    if a.cpu_threads < 1 or not 0 < a.memory_fraction <= 1:
        p.error("Invalid resource limits")
    if a.preflight_only and a.device != "cpu":
        p.error("Preflight requires CPU")
    sys.path.insert(0, str(a.benchmark_code_root.resolve()))
    import torch
    from multicamera4rc.benchmarks import released_baselines
    from multicamera4rc.benchmarks.released_data import load_manifest
    torch.set_num_threads(a.cpu_threads)
    manifest = load_manifest(a.manifest, inference_only=True)
    validate_manifest(manifest)
    checkpoint = checkpoint_identity(a.source_root, a.checkpoint, a.frontend_weights_root)
    predictor = Predictor(manifest, a.source_root, a.checkpoint, a.frontend_weights_root, a.device)
    if a.preflight_only:
        predictor.load()
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU preflight unexpectedly initialized CUDA")
        print(json.dumps({"status": "RELIFT_CPU_PREFLIGHT_OK", "checkpoint": checkpoint,
            "load_receipt": predictor.load_receipt, "manifest_sha256": sha(a.manifest),
            "cases": len(manifest["cases"]), "gpu_used": False}, indent=2))
        return
    if torch.device(a.device).type == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.memory_fraction, torch.device(a.device))
    settings = {"method": METHOD, "prediction_kind": "trajectory",
        "target_time_policy": "all_frozen_first64_frames",
        "protocol_extension": "author pipeline on frozen64; not paper84 table",
        "geometry": "DA3NESTED-GIANT-LARGE-1.1; upper_bound_resize504; saddle_balanced",
        "tracker": "AllTracker window16 iters4; author DA3 processed frames",
        "corrector": {"group": 1024, "batch": 8, "seed": 0},
        "query_sampling": "unchanged official WorldTrack evaluator; native DA3 grid",
        "output_contract": "native dense tracks_XYZ only", "adapter_sha256": sha(__file__)}
    with dense_contract(released_baselines):
        report = released_baselines.run_adapter_predictions(a.manifest, a.output, method=METHOD,
            checkpoint=checkpoint, predict_case=predictor, metric_support=dict(SUPPORT),
            device=a.device, resume=a.resume, inference_settings=settings)
    print(report, flush=True)


if __name__ == "__main__":
    main()
