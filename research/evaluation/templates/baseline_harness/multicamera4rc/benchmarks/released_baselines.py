"""Composed RGB-only baselines, sharing the released offline scorer contract.

The baseline checkpoint record describes the actual geometry/tracker components;
it never claims native MultiCamera4RC or refiner checkpoint loading.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from .official import sha256_file
from .released_data import load_manifest, read_inputs
from .released_runner import (_atomic_prediction_file, _atomic_prediction_json,
    _json_hash, _prediction_numerics, _prediction_specs, _resume_prediction,
    _validate_prediction_arrays, inference_source_hashes, runtime_versions)
from .tracker_baselines import (CoTracker3GeometryBaseline, MVTrackerGeometryBaseline,
    TAPIP3DGeometryBaseline, PredictedGeometry, VGGTCommonReconstructor, rgb_identity)

METHOD = "VGGT-Omega-1B-512+MVTracker"
COMPONENTS = {
    "geometry": {"relative_path": "weights/vggt-omega/vggt_omega_1b_512.pt",
        "sha256": "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934",
        "code": "vggt-omega", "revision": "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6"},
    "tracker": {"relative_path": "weights/mvtracker/mvtracker_200000_june2025.pth",
        "sha256": "a7fa86f2a7223e3e0aa4c1d3eff0dec5fe8a9227a48572ce943b8e49d8a4f8e6",
        "code": "mvtracker", "revision": "ceea8ad2af77ed9b44148ef8e9eeba4ea3c3f072"}}
TRACKERS = {
    "mvtracker": {"name": "MVTracker", "constructor": MVTrackerGeometryBaseline,
                  "component": COMPONENTS["tracker"], "any_view_visibility": True},
    "cotracker3": {"name": "CoTracker3-Offline", "constructor": CoTracker3GeometryBaseline,
        "any_view_visibility": False,
        "component": {"relative_path": "weights/cotracker3/scaled_offline.pth",
            "sha256": "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834",
            "code": "co-tracker", "revision": "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"}},
    "tapip3d": {"name": "TAPIP3D", "constructor": TAPIP3DGeometryBaseline,
        "any_view_visibility": False,
        "component": {"relative_path": "weights/tapip3d/tapip3d_final.pth",
            "sha256": "3a9514d526559838e6158360af2b857da596d064b06a94fae7fd3b85134b2b1e",
            "code": "TAPIP3D", "revision": "4cb7e69a1687f67d56ec3e506768f51f2c581b46"}}}


def baseline_checkpoint(root, tracker="mvtracker"):
    """Verify every real component and record the exact external code bytes."""
    root = Path(root).resolve()
    if tracker not in TRACKERS:
        raise ValueError("Unknown composed tracker")
    components = {}
    specifications = {"geometry": COMPONENTS["geometry"], "tracker": TRACKERS[tracker]["component"]}
    for name, spec in specifications.items():
        weight, code = root / spec["relative_path"], root / "code" / spec["code"]
        digest = sha256_file(weight)
        if digest != spec["sha256"]:
            raise ValueError(f"{name} checkpoint differs from the pinned author release")
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=code, text=True)
        if revision != spec["revision"] or dirty:
            raise ValueError(f"{name} requires the clean pinned official source")
        sources = {str(p.relative_to(code)): sha256_file(p) for p in sorted(code.rglob("*.py"))
                   if ".git" not in p.parts and "__pycache__" not in p.parts}
        components[name] = {"path": str(weight), "sha256": digest, "bytes": weight.stat().st_size,
            "source_root": str(code), "source_commit": revision, "source_sha256": sources,
            "checkpoint_loading": "strict=True; full author checkpoint"}
    identity = {"method": "VGGT-Omega-1B-512+" + TRACKERS[tracker]["name"], "components": components}
    return {"kind": "composed_rgb_only_baseline", **identity, "sha256": _json_hash(identity),
            "native_multicamera4rc_loading": False, "refiner_loading": "not_applicable"}


def _geometry_cache(path, inputs, geometry_model, checkpoint):
    """Only checksum-bound caches from these exact RGB/code/weight bytes resume."""
    path = Path(path)
    receipt_path = path.with_suffix(".receipt.json")
    identity = {"rgb_identity_sha256": rgb_identity(inputs),
                "component": checkpoint["components"]["geometry"]}
    if path.exists() or receipt_path.exists():
        if not path.is_file() or not receipt_path.is_file() or path.is_symlink() or receipt_path.is_symlink():
            raise ValueError("Incomplete geometry commit; preserve it for inspection")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("identity") != identity or receipt.get("sha256") != sha256_file(path):
            raise ValueError("Geometry cache changed or belongs to different RGB/code/weights")
        return PredictedGeometry.load(path, inputs)
    geometry = geometry_model.predict(inputs)
    geometry.validate(inputs)
    if geometry.metadata.get("checkpoint_sha256") != checkpoint["components"]["geometry"]["sha256"]:
        raise ValueError("Geometry prediction checkpoint differs from the declared component")
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_prediction_file(path, lambda stream: np.savez_compressed(stream,
        depth=geometry.depth, depth_confidence=geometry.depth_confidence,
        predicted_K=geometry.K, predicted_c2w=geometry.c2w,
        metadata_json=np.asarray(json.dumps(geometry.metadata, sort_keys=True))))
    _atomic_prediction_json(receipt_path, {"schema": "released-baseline-geometry-cache-v1",
        "identity": identity, "sha256": sha256_file(path)})
    return geometry


def run_adapter_predictions(manifest_path, output, *, method, checkpoint, predict_case,
        metric_support, device="cpu", resume=False, inference_settings=None):
    """Shared checksummed I/O for an RGB-only baseline callback.

    ``predict_case(inputs, case, index)`` returns ``(arrays, metadata)``. Arrays
    obey the released contract; visibility is absent when native_visibility is
    false. The callback receives the four-field inference type, never labels.
    """
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    manifest = load_manifest(manifest_path, inference_only=True)
    benchmark = manifest["benchmark"]
    if benchmark == "mvtracker" and manifest["protocol"].get("mode") != "rgb_only_3d_query_anchor_sim3":
        raise ValueError("Composed baselines require the frozen RGB-only cached extension")
    if manifest["leakage"].get("checkpoint_sha256"):
        raise ValueError("Manifest training inventory is bound to another checkpoint; prepare an unbound evaluation manifest")
    if not method or not isinstance(checkpoint, dict) or checkpoint.get("kind") not in (
            "composed_rgb_only_baseline", "author_rgb_only_native_baseline"):
        raise ValueError("Record the actual baseline components, never a native/refiner loading claim")
    if "native_loading" in checkpoint or "native_multicamera4rc_loading" in checkpoint and checkpoint["native_multicamera4rc_loading"] is not False:
        raise ValueError("Baseline checkpoint must not claim native model loading")
    if not isinstance(metric_support, dict) or type(metric_support.get("native_visibility")) is not bool:
        raise ValueError("metric_support must explicitly declare boolean native_visibility")
    if not metric_support["native_visibility"] and any(metric_support.get(k) for k in ("AJ", "OA")):
        raise ValueError("A position-only baseline cannot claim AJ/OA support")
    nonempty = output.exists() and any(output.iterdir())
    if nonempty and not resume:
        raise FileExistsError("Prediction output is nonempty; preserve it or use verified --resume")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    settings = inference_settings or {}
    state = {"schema": "released-baseline-run-v1", "method": method,
        "manifest_sha256": sha256_file(manifest_path), "checkpoint": checkpoint,
        "runtime_versions": runtime_versions(), "numerics": _prediction_numerics(device),
        "device": str(device), "inference_source_sha256": inference_source_hashes(),
        "inference_settings": settings, "metric_support": metric_support,
        "query_policy": manifest.get("pixel_query_policy", "preserve_border_extension"),
        "input_boundary": "RGB, pixel queries, view IDs, frame indices only; no GT calibration/depth/XYZ"}
    state_sha256 = _json_hash(state)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "run-state.json"
    allowed = {"run-state.json", "predictions.json"}
    for index in range(len(manifest["cases"])):
        allowed.update((f"{index:06d}.npz", f"{index:06d}.receipt.json"))
    for existing in output.iterdir():
        if existing.name not in allowed:
            raise ValueError("Unexpected existing baseline artifact; preserve and inspect: " + existing.name)
    if nonempty:
        if not state_path.is_file() or state_path.is_symlink() or _json_hash(json.loads(state_path.read_text())) != state_sha256:
            raise ValueError("Resume baseline identity/settings/source differ")
    else:
        _atomic_prediction_json(state_path, state)
    records = []
    for index, case in enumerate(manifest["cases"]):
        input_path = manifest_path.parent / case["input"]
        expected = _prediction_specs(manifest, case, input_path)
        if not metric_support["native_visibility"]:
            expected.pop("visibility")
        if resume:
            previous = _resume_prediction(output, index, case, expected, state_sha256)
            if previous is not None:
                records.append(previous)
                continue
        inputs = read_inputs(input_path)
        arrays, metadata = predict_case(inputs, case, index)
        specs = _validate_prediction_arrays(arrays, expected)
        path = output / f"{index:06d}.npz"
        _atomic_prediction_file(path, lambda stream: np.savez_compressed(stream, **arrays))
        record = {"id": case["id"], "file": path.name, "sha256": sha256_file(path),
            "input_sha256": case["input_sha256"], "geometry_inputs": [],
            "prediction_kind": settings.get("prediction_kind", "trajectory"),
            "target_time_policy": settings.get("target_time_policy", "baseline_native_full_clip"),
            "pixel_query_policy": inputs.pixel_query_policy,
            "out_of_canvas_queries": case.get("out_of_canvas_queries", 0),
            "metric_support": metric_support, "model": method, **metadata}
        # Callback metadata cannot relabel frozen sample/run identity.
        if any(record[k] != v for k, v in {"id": case["id"], "file": path.name,
                "input_sha256": case["input_sha256"], "sha256": sha256_file(path),
                "metric_support": metric_support, "model": method, "geometry_inputs": []}.items()):
            raise ValueError("Adapter metadata attempted to overwrite frozen prediction identity")
        _atomic_prediction_json(output / f"{index:06d}.receipt.json", {
            "schema": "released-prediction-case-v1", "run_state_sha256": state_sha256,
            "case_sha256": _json_hash(case), "record": record, "arrays": specs})
        records.append(record)
        print(json.dumps({"method": method, "completed": len(records), "total": len(manifest["cases"]), "id": case["id"]}), flush=True)
    report = {"schema": "released-tracking-predictions-v1", "manifest_sha256": sha256_file(manifest_path),
        "benchmark": benchmark, "input_condition": manifest["input_condition"],
        "coordinates": manifest["coordinates"], "checkpoint": checkpoint, "method": method,
        "runtime_versions": state["runtime_versions"], "run_state_sha256": state_sha256,
        "target_time_policy": settings.get("target_time_policy", "baseline_native_full_clip"),
        "prediction_kind": settings.get("prediction_kind", "trajectory"),
        "model_boundary": state["input_boundary"], "metric_support": metric_support, "cases": records}
    report_path = output / "predictions.json"
    if report_path.exists() and (report_path.is_symlink() or _json_hash(json.loads(report_path.read_text())) != _json_hash(report)):
        raise ValueError("Existing complete baseline report differs from verified receipts")
    _atomic_prediction_json(report_path, report)
    return report_path


def run_baseline_predictions(manifest_path, output, geometry_cache, *, root="@@BENCHMARK_ROOT@@",
        device="cpu", resume=False, tracker="mvtracker", _checkpoint=None, _constructors=None):
    """Ω512 geometry, reused verbatim by three author-released trackers.

    Only MVTracker emits any-view visibility. CoTracker3/TAPIP3D visibility is
    query-view-only and remains metadata; their MV scores are position-only.
    """
    manifest = load_manifest(manifest_path, inference_only=True)
    if manifest["benchmark"] != "mvtracker":
        raise ValueError("Omega + tracker runner requires a multiview tracking manifest")
    if tracker not in TRACKERS:
        raise ValueError("Unknown composed tracker")
    specification = TRACKERS[tracker]
    method = "VGGT-Omega-1B-512+" + specification["name"]
    checkpoint = baseline_checkpoint(root, tracker) if _checkpoint is None else _checkpoint
    constructors = _constructors or (VGGTCommonReconstructor, specification["constructor"])
    models = {}
    class LazyGeometry:
        # Cache hits never load the large Ω network. The same cache can feed
        # successive tracker processes without recomputing a reconstruction.
        def predict(self, inputs):
            if "geometry" not in models:
                models["geometry"] = constructors[0](root, device=str(device), variant="VGGT-Omega-1B-512")
            return models["geometry"].predict(inputs)
    def predict_case(inputs, case, index):
        if "tracker" not in models:
            models["tracker"] = constructors[1](root, device=str(device))
        cache = Path(geometry_cache).resolve() / f"{index:06d}.npz"
        geometry = _geometry_cache(cache, inputs, LazyGeometry(), checkpoint)
        prediction = models["tracker"].predict(inputs, geometry)
        if prediction.metadata.get("checkpoint_sha256") != checkpoint["components"]["tracker"]["sha256"]:
            raise ValueError("Tracker prediction checkpoint differs from the declared component")
        arrays = {"tracks_XYZ": prediction.tracks_world,
            "tracks_native_world": prediction.tracks_world, "predicted_c2w": prediction.predicted_c2w,
            "predicted_K": prediction.predicted_K}
        visibility_key = "any_view_visibility" if specification["any_view_visibility"] else "query_view_visibility"
        visibility = np.asarray(prediction.metadata.get(visibility_key))
        if visibility.dtype != np.dtype(bool) or visibility.shape != (case["frames"], case["queries"]):
            raise ValueError("Native tracker " + visibility_key.replace("_visibility", " visibility").replace("_", "-") +
                " must be boolean [T,N]; no invented visibility")
        if specification["any_view_visibility"]:
            arrays["visibility"] = visibility
        metadata = {"geometry_cache": str(cache), "geometry_cache_sha256": sha256_file(cache),
            "visibility_semantics": "native_any_view_visibility" if specification["any_view_visibility"]
                                     else "query_view_only; not world-union visibility",
            "model_provenance": {k: v for k, v in prediction.metadata.items()
                                 if k not in ("any_view_visibility", "query_view_visibility")},
            "geometry_provenance": geometry.metadata}
        if not specification["any_view_visibility"]:
            metadata["native_query_view_visibility"] = visibility.tolist()
            metadata["native_query_view_visibility_usage"] = "diagnostic only; never broadcast or scored as any-view visibility"
        return arrays, metadata
    return run_adapter_predictions(manifest_path, output, method=method, checkpoint=checkpoint,
        predict_case=predict_case, device=device, resume=resume,
        metric_support={"position": True, "native_visibility": specification["any_view_visibility"],
                        "AJ": specification["any_view_visibility"], "OA": specification["any_view_visibility"]},
        inference_settings={"target_time_policy": "full_clip_joint_geometry_native_tracker",
            "geometry_preprocessing": "official Omega balanced512; outputs restored to original RGB pixels",
            "visibility_policy": "native any-view visibility" if specification["any_view_visibility"]
                                 else "position-only MV scoring; query-view visibility is diagnostic metadata"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry-cache", type=Path, required=True)
    parser.add_argument("--tracker", choices=tuple(TRACKERS), default="mvtracker")
    parser.add_argument("--root", type=Path, default=Path("@@BENCHMARK_ROOT@@"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--memory-fraction", type=float, default=.8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.cpu_threads < 1 or not 0 < args.memory_fraction <= 1:
        parser.error("Invalid CPU thread count or memory fraction")
    torch.set_num_threads(args.cpu_threads)
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    print(run_baseline_predictions(args.manifest, args.output, args.geometry_cache,
        root=args.root, device=args.device, resume=args.resume, tracker=args.tracker))


if __name__ == "__main__":
    main()
