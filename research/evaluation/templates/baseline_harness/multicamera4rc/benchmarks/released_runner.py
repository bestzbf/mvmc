"""Real checkpoint inference and strictly bound offline official scoring."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import re
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch

from .official import OfficialMVTracker, sha256_file
from .protocols import PROTOCOLS
from .released_data import load_manifest, read_inputs, write_json
from .released_geometry import ReleasedGeometryInputs, predict_released_geometry
from .released_official import (
    OfficialTAPVid3D,
    OfficialWorldTrack,
    boolean_visibility,
    finite_tracks,
)
from .tapvid_mv import predict_tapvid_mv_queries


def runtime_versions():
    versions = {}
    for package in ("torch", "numpy", "einops", "Pillow", "scipy", "pandas", "torchvision",
                    "opencv-python-headless", "kornia", "natsort"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def inference_source_hashes(model_code_root=None):
    root = Path(__file__).resolve().parents[1]
    if model_code_root is not None:
        model_root = Path(model_code_root).resolve() / "multicamera4rc"
        model_files = list(model_root.glob("*.py")) + list((model_root / "vendor").rglob("*.py"))
        hashes = {"model/" + str(p.relative_to(model_root)): sha256_file(p) for p in sorted(model_files)}
        hashes.update({"benchmark/" + p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
        return hashes
    files = list(root.glob("*.py")) + list((root / "vendor").rglob("*.py"))
    files += list((root / "benchmarks").glob("*.py"))
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(files)}


def load_prediction_model(checkpoint_path, *, model_code_root=None, expected_model_commit=None,
                          expected_step=None, require_refinement=False, model_runtime_manifest=None):
    """Reuse the existing native/refiner loader without filtering saved fields.

    External activation must happen in a fresh process before core imports;
    scripts/run_released_tracking_benchmarks.py provides that bootstrap.
    The bundled legacy path uses the same complete config/state validation.
    """
    from . import refined_current
    runtime = None
    if model_code_root is not None:
        root = Path(model_code_root).resolve()
        runtime = refined_current.build_model_runtime_manifest(root, expected_revision=expected_model_commit)
        if model_runtime_manifest is not None and model_runtime_manifest != runtime:
            raise ValueError("External model sources changed after bootstrap")
    else:
        if expected_model_commit is not None or model_runtime_manifest is not None:
            raise ValueError("External source assertions require --model-code-root")
        module = importlib.import_module("multicamera4rc.model")
        root = Path(module.__file__).resolve().parents[1]
    try:
        model, provenance = refined_current.load_native_checkpoint(
            checkpoint_path, model_code_root=root, expected_step=expected_step,
            require_refinement=require_refinement)
    except TypeError as exc:
        raise ValueError("Saved config is incompatible with the selected model architecture; use the matching "
                         "--model-code-root through scripts/run_released_tracking_benchmarks.py. "
                         "No saved fields were discarded: " + str(exc)) from exc
    except ValueError as exc:
        raise ValueError("Benchmark requires a complete native checkpoint: " + str(exc)) from exc
    if runtime is not None and refined_current.build_model_runtime_manifest(root, expected_revision=expected_model_commit) != runtime:
        raise ValueError("External model sources changed during checkpoint loading")
    provenance["model_runtime"] = runtime
    return model, provenance


def camera_tracks(tracks, c2w):
    """Predicted world -> moving camera; c2w uses column-vector rotations."""
    tracks, c2w = np.asarray(tracks), np.asarray(c2w)
    if tracks.ndim != 3 or c2w.shape != (len(tracks), 4, 4):
        raise ValueError("Expected tracks [T,N,3] and c2w [T,4,4]")
    return np.einsum("tnj,tjk->tnk", tracks - c2w[:, None, :3, 3], c2w[:, :3, :3])


def xyz_query_reference(prediction, inputs, supplied_c2w, official_queries):
    """Privileged query-conditioned baseline in the official metric world.

    Official MVTracker provides query XYZ and calibration to its trackers.
    Use only that query, its source rotation and depth to initialize each native
    predicted motion. Never fit target trajectories, query a future GT point,
    or align predicted cameras against GT. This reference is deliberately
    ineligible for the RGB-only table and is not the native MVTracker model.
    """
    tracks = np.asarray(prediction.tracks_world)
    queries = np.asarray(official_queries)
    n = tracks.shape[1]
    pixel_queries = inputs.queries_xytv.cpu().numpy()
    times, views = pixel_queries[:, 2].astype(int), pixel_queries[:, 3].astype(int)
    if queries.shape != (n, 4) or not np.array_equal(queries[:, 0], times):
        raise ValueError("Official XYZ queries differ from the frozen pixel query order")
    native = prediction.predicted_c2w[views, times]
    supplied = np.asarray(supplied_c2w)[views, times]
    anchor = tracks[times, np.arange(n)]
    native_z = np.einsum("nj,nj->n", anchor - native[:, :3, 3], native[:, :3, 2])
    supplied_z = np.einsum("nj,nj->n", queries[:, 1:] - supplied[:, :3, 3], supplied[:, :3, 2])
    if (native_z <= 1e-8).any() or (supplied_z <= 1e-8).any():
        raise ValueError("Nonpositive query depth; cannot initialize the calibrated XYZ-query reference")
    motion_camera = np.einsum("tnj,njk->tnk", tracks - anchor[None], native[:, :3, :3])
    motion_world = np.einsum("tnj,nkj->tnk", motion_camera, supplied[:, :3, :3])
    result = queries[None, :, 1:] + motion_world * (supplied_z / native_z)[None, :, None]
    return finite_tracks(result, tracks.shape)


def align_query_points_for_scoring(official_alignment, labels, native_tracks):
    """One scorer-only global Sim(3), fitted solely to matching query anchors.

    No future target point participates in the fit. The returned transformed
    copy is passed directly to official metrics, never exported as a model
    prediction or fed into model inference. Native input arrays stay unchanged.
    """
    native = finite_tracks(native_tracks, labels["tracks"].shape)
    queries = np.asarray(labels["query_points"])
    n = native.shape[1]
    if queries.shape != (n, 4) or n < 3 or not np.isfinite(queries).all():
        raise ValueError("Query-anchor Sim(3) needs at least three finite matching official queries")
    times = queries[:, 0]
    if not np.equal(times, times.round()).all() or (times < 0).any() or (times >= len(native)).any():
        raise ValueError("Query-anchor Sim(3) times must match the frozen video slots")
    times = times.astype(int)
    source = np.asarray(native[times, np.arange(n)], dtype=np.float64)
    target = np.asarray(queries[:, 1:], dtype=np.float64)
    if not np.allclose(labels["tracks"][times, np.arange(n)], target, rtol=1e-5, atol=1e-6):
        raise ValueError("Official query XYZ does not match the same track at its query time")
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    for name, centered in (("predicted", source_centered), ("GT", target_centered)):
        singular = np.linalg.svd(centered, compute_uv=False)
        if singular[0] <= 1e-12 or singular[1] <= singular[0] * 1e-10:
            raise ValueError(f"Degenerate {name} query-anchor Sim(3): require noncollinear query positions")
    # Each cloud can be noncollinear while its paired covariance has rank one:
    # then rotations about the one correlated axis are not identifiable. Test
    # rank >= 2 (not 3: ordinary planar correspondences are valid). Normalize
    # both clouds so the absolute roundoff floor is independent of their units.
    covariance = (source_centered / np.linalg.norm(source_centered)).T @ (
        target_centered / np.linalg.norm(target_centered))
    paired_singular = np.linalg.svd(covariance, compute_uv=False)
    paired_tolerance = max(paired_singular[0] * 1e-10,
                           np.finfo(np.float64).eps * max(n, 3) * 16)
    if paired_singular[1] <= paired_tolerance:
        raise ValueError("Degenerate paired query-anchor covariance: identifiable Sim(3) requires rank >= 2")
    scale, rotation, translation = official_alignment.util.estimate_sim3(source, target, ransac=False)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Official query-anchor Sim(3) returned an invalid scale")
    aligned = finite_tracks(scale * np.asarray(native, np.float64) @ rotation.T + translation, native.shape)
    residual = scale * source @ rotation.T + translation - target
    record = {"protocol": "rgb_only_3d_query_anchor_sim3_v1", "transform_count": 1,
              "fit_points": n, "fit_frames": times.tolist(), "scale": float(scale),
              "rotation": rotation.tolist(), "translation": translation.tolist(),
              "query_rmse": float(np.sqrt(np.square(residual).sum(axis=-1).mean())),
              "gt_usage": "official query-time XYZ only, scoring only; no later target frames",
              "official_alignment_sources": official_alignment.provenance,
              "original_protocol_comparable": False}
    return aligned, record


def json_metrics(value):
    """Unavailable upstream metrics are null with explicit paths, never zero."""
    missing = []
    def convert(item, path):
        if isinstance(item, dict):
            return {str(k): convert(v, path + "/" + str(k)) for k, v in item.items()}
        array = np.asarray(item)
        if array.size != 1:
            raise ValueError(f"Expected scalar official result: {path}")
        number = float(array.reshape(-1)[0])
        if not np.isfinite(number):
            missing.append(path)
            return None
        return number
    return convert(value, ""), missing


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _atomic_prediction_file(path, writer):
    """Only a complete, flushed file is installed under a public result name."""
    with tempfile.NamedTemporaryFile(prefix="." + path.name + ".", suffix=".tmp",
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _atomic_prediction_json(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    _atomic_prediction_file(path, lambda stream: stream.write(data))


def _prediction_specs(manifest, case, input_path):
    # Read only the tiny view-ID array; resuming must not decompress every RGB
    # clip again merely to verify the shape of a completed prediction.
    with np.load(input_path, allow_pickle=False) as archive:
        views = len(archive["view_ids"])
    frames, queries = case["frames"], case["queries"]
    tracks = (frames, queries, 3)
    if manifest["benchmark"] == "worldtrack":
        tracks = (frames, case["height"], case["width"], 3)
    native = ((frames, case["height"], case["width"], 3)
              if manifest["benchmark"] == "worldtrack" and manifest["protocol"].get("task") == "reconstruction"
              else (frames, queries, 3))
    return {"tracks_XYZ": tracks, "visibility": tracks[:-1],
            "tracks_native_world": native,
            "predicted_c2w": (views, frames, 4, 4), "predicted_K": (views, frames, 3, 3)}


def _validate_prediction_arrays(arrays, expected):
    if set(arrays) != set(expected):
        raise ValueError("Prediction array keys differ from the complete export contract")
    specs = {}
    for key, shape in expected.items():
        value = np.asarray(arrays[key])
        if value.shape != shape:
            raise ValueError(f"Prediction {key} shape differs: expected {shape}, got {value.shape}")
        if key == "visibility":
            if value.dtype != np.dtype(bool):
                raise ValueError("Prediction visibility must have boolean dtype")
        elif value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError(f"Prediction {key} must have finite floating-point values")
        specs[key] = {"shape": list(value.shape), "dtype": value.dtype.str}
    return specs


def _prediction_numerics(device):
    settings = {"torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
                "default_dtype": str(torch.get_default_dtype()),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cuda_runtime": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        settings["cuda_device"] = {"name": properties.name, "major": properties.major,
                                   "minor": properties.minor, "total_memory": properties.total_memory}
    return settings


def _resume_prediction(output, index, case, expected, state_sha256):
    path, receipt_path = output / f"{index:06d}.npz", output / f"{index:06d}.receipt.json"
    if not path.exists() and not receipt_path.exists():
        return None
    if not path.is_file() or not receipt_path.is_file() or path.is_symlink() or receipt_path.is_symlink():
        raise ValueError("Incomplete prediction commit: both NPZ and receipt are required; preserve and inspect it")
    try:
        receipt = json.loads(receipt_path.read_text())
        record = receipt["record"]
        if (receipt["schema"] != "released-prediction-case-v1" or receipt["run_state_sha256"] != state_sha256
                or receipt["case_sha256"] != _json_hash(case) or record["id"] != case["id"]
                or record["input_sha256"] != case["input_sha256"] or record["file"] != path.name
                or record["sha256"] != sha256_file(path)):
            raise ValueError("Prediction receipt/checksum differs from the frozen run or case")
        with np.load(path, allow_pickle=False) as archive:
            specs = _validate_prediction_arrays({key: archive[key] for key in archive.files}, expected)
        if specs != receipt["arrays"]:
            raise ValueError("Prediction array dtype/shape differs from its receipt")
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid prediction receipt or NPZ; refusing to reuse it") from exc
    return record


def run_predictions(manifest_path, checkpoint_path, output, *, device="cpu", amp="fp32",
                    target_time_chunk=8, visibility_threshold=.5, model_code_root=None,
                    expected_model_commit=None, expected_step=None, require_refinement=False,
                    model_runtime_manifest=None, resume=False):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    manifest = load_manifest(manifest_path, inference_only=True)
    nonempty = output.exists() and any(output.iterdir())
    if nonempty and not resume:
        raise FileExistsError("Prediction output must be empty; preserve existing predictions or use --resume")
    if amp not in ("fp32", "bf16"):
        raise ValueError("amp must be fp32 or bf16")
    if isinstance(target_time_chunk, bool) or not isinstance(target_time_chunk, int) or target_time_chunk <= 0:
        raise ValueError("target_time_chunk must be a positive integer")
    if not np.isfinite(visibility_threshold) or not 0 <= visibility_threshold <= 1:
        raise ValueError("visibility_threshold must be finite and in [0,1]")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu for CPU inference")
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {checkpoint_path}")
    checkpoint_hash = sha256_file(checkpoint_path)
    audited_hash = manifest["leakage"].get("checkpoint_sha256")
    if audited_hash and audited_hash != checkpoint_hash:
        raise ValueError("Training scene inventory belongs to another checkpoint")
    model, loading = load_prediction_model(checkpoint_path, model_code_root=model_code_root,
        expected_model_commit=expected_model_commit, expected_step=expected_step,
        require_refinement=require_refinement, model_runtime_manifest=model_runtime_manifest)
    model.to(device).eval()
    has_refiner = bool(getattr(model.config, "refinement_dim", 0)) or getattr(model, "trajectory_refiner", None) is not None
    reconstruction = manifest["benchmark"] == "worldtrack" and manifest["protocol"].get("task") == "reconstruction"
    time_policy = ("not_applicable_geometry" if reconstruction else
                   "all_observed_times" if has_refiner else "requested_legacy_chunk")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {"path": str(Path(checkpoint_path).resolve()), "sha256": checkpoint_hash,
                  "config": model.config.to_dict(), "device": str(device), "amp": amp,
                  "native_loading": loading,
                  "inference_source_sha256": inference_source_hashes(model_code_root)}
    versions = runtime_versions()
    state = {"schema": "released-prediction-run-v1", "manifest_sha256": sha256_file(manifest_path),
             "checkpoint": checkpoint, "runtime_versions": versions,
             "numerics": _prediction_numerics(device),
             "visibility_threshold": visibility_threshold, "target_time_chunk": target_time_chunk,
             "target_time_policy": time_policy,
             "prediction_kind": "reconstruction" if reconstruction else "trajectory",
             "loader_assertions": {"expected_model_commit": expected_model_commit,
                 "expected_step": expected_step, "require_refinement": require_refinement,
                 "model_runtime_manifest": model_runtime_manifest}}
    state_path = output / "run-state.json"
    state_sha256 = _json_hash(state)
    allowed = {"run-state.json", "predictions.json"}
    for i in range(len(manifest["cases"])):
        allowed.update((f"{i:06d}.npz", f"{i:06d}.receipt.json"))
    # Interrupted private temporary writes can be recomputed. Never remove
    # foreign files, silently adopt older outputs, or overwrite an orphan NPZ.
    temporary_name = re.compile(r"^\.(?:run-state\.json|predictions\.json|\d{6}\.(?:npz|receipt\.json))\.[A-Za-z0-9_-]+\.tmp$")
    for existing in output.iterdir():
        if existing.name not in allowed and not (temporary_name.fullmatch(existing.name) and existing.is_file()
                                                  and not existing.is_symlink()):
            raise ValueError(f"Unexpected existing prediction artifact; preserve and inspect: {existing.name}")
    if nonempty:
        if not state_path.exists() and not any(path.name in allowed for path in output.iterdir()):
            # A crash during the very first private write committed no result
            # or identity; ignoring only recognized private temporaries is safe.
            _atomic_prediction_json(state_path, state)
        else:
            if not state_path.is_file() or state_path.is_symlink():
                raise ValueError("Cannot resume existing results without a verifiable run-state.json")
            if _json_hash(json.loads(state_path.read_text())) != state_sha256:
                raise ValueError("Resume run-state differs: manifest, checkpoint, sources or inference settings changed")
    else:
        _atomic_prediction_json(state_path, state)
    cases = []
    for index, case in enumerate(manifest["cases"]):
        expected = _prediction_specs(manifest, case, manifest_path.parent / case["input"])
        if resume:
            cached = _resume_prediction(output, index, case, expected, state_sha256)
            if cached is not None:
                cases.append(cached)
                continue
        inputs = read_inputs(manifest_path.parent / case["input"])
        # A refined decode runs its complete observed timeline internally. One
        # full-time call per unique query source avoids repeating that refiner
        # for each target chunk, matching the established refined entry point.
        effective_chunk = None if reconstruction else len(inputs.frame_indices) if has_refiner else target_time_chunk
        if reconstruction:
            prediction = predict_released_geometry(model, ReleasedGeometryInputs.from_rgb(inputs), device=device,
                amp_dtype=torch.bfloat16 if amp == "bf16" else None)
        else:
            prediction = predict_tapvid_mv_queries(model, inputs, device=device,
                target_time_chunk=effective_chunk, amp_dtype=torch.bfloat16 if amp == "bf16" else None,
                visibility_threshold=visibility_threshold)
        benchmark = manifest["benchmark"]
        geometry_inputs = []
        if benchmark == "tapvid3d":
            tracks = camera_tracks(prediction.tracks_world, prediction.predicted_c2w[0])
            visible = prediction.visibility[0]
        elif benchmark == "worldtrack":
            # One predicted first-camera rigid gauge change, without GT.
            pose = np.broadcast_to(prediction.predicted_c2w[0, 0], (case["frames"], 4, 4))
            native = (prediction.base_points[0].reshape(case["frames"], -1, 3)
                      if reconstruction else prediction.tracks_world)
            tracks = camera_tracks(native, pose).reshape(
                case["frames"], case["height"], case["width"], 3)
            visible = (np.ones(tracks.shape[:-1], bool) if reconstruction
                       else prediction.visibility[0].reshape(case["frames"], case["height"], case["width"]))
        elif benchmark == "mvtracker":
            if manifest["protocol"].get("mode", "calibrated-reference") == "rgb_only_3d_query_anchor_sim3":
                # Export the untouched native world gauge. GT alignment, if
                # requested by this named extension, happens only in score.
                tracks = prediction.tracks_world
            else:
                with np.load(manifest_path.parent / case["calibration"], allow_pickle=False) as supplied:
                    tracks = xyz_query_reference(prediction, inputs, supplied["c2w"], supplied["query_points"])
                geometry_inputs = ["official_camera_calibration", "official_xyz_query",
                                   "official_query_frame_visibility_for_source_choice"]
            visible = prediction.visibility.any(axis=0)
        else:
            raise ValueError("Unknown released benchmark")
        path = output / f"{index:06d}.npz"
        arrays = {"tracks_XYZ": tracks, "visibility": visible,
                  "tracks_native_world": prediction.base_points[0] if reconstruction else prediction.tracks_world,
                  "predicted_c2w": prediction.predicted_c2w, "predicted_K": prediction.predicted_K}
        specs = _validate_prediction_arrays(arrays, expected)
        _atomic_prediction_file(path, lambda stream: np.savez_compressed(stream, **arrays))
        cases.append({"id": case["id"], "file": path.name, "sha256": sha256_file(path),
                      "input_sha256": case["input_sha256"], "geometry_inputs": geometry_inputs,
                      "effective_target_time_chunk": effective_chunk,
                      "target_time_policy": time_policy,
                      "prediction_kind": "reconstruction" if reconstruction else "trajectory"})
        if reconstruction:
            cases[-1]["geometry_inference"] = prediction.metadata
            cases[-1]["visibility_semantics"] = "all observation pixels; reconstruction scorer applies official GT validity, not visibility prediction"
        cases[-1]["pixel_query_policy"] = inputs.pixel_query_policy
        cases[-1]["out_of_canvas_queries"] = case.get("out_of_canvas_queries", 0)
        _atomic_prediction_json(output / f"{index:06d}.receipt.json",
            {"schema": "released-prediction-case-v1", "run_state_sha256": state_sha256,
             "case_sha256": _json_hash(case), "record": cases[-1], "arrays": specs})
    report = {"schema": "released-tracking-predictions-v1", "manifest_sha256": sha256_file(manifest_path),
              "checkpoint": checkpoint, "benchmark": manifest["benchmark"],
              "input_condition": manifest["input_condition"], "coordinates": manifest["coordinates"],
              "visibility_threshold": visibility_threshold, "target_time_chunk": target_time_chunk,
              "target_time_policy": time_policy,
              "prediction_kind": "reconstruction" if reconstruction else "trajectory",
              "runtime_versions": versions, "run_state_sha256": state_sha256,
              "model_boundary": ("RGB, view IDs, frame indices only; encoder geometry, no trajectory decoder or GT"
                                 if reconstruction else "RGB, pixel query, view IDs, frame indices only; no GT trajectories"),
              "cases": cases}
    report_path = output / "predictions.json"
    if report_path.exists() and (report_path.is_symlink() or _json_hash(json.loads(report_path.read_text())) != _json_hash(report)):
        raise ValueError("Existing predictions.json differs from the complete verified case receipts")
    _atomic_prediction_json(report_path, report)
    return output / "predictions.json"


def load_predictions(manifest_path, predictions_path):
    manifest_path, predictions_path = Path(manifest_path).resolve(), Path(predictions_path).resolve()
    manifest = load_manifest(manifest_path, verify_sources=True)
    prediction = json.loads(predictions_path.read_text())
    if prediction.get("schema") != "released-tracking-predictions-v1":
        raise ValueError("Invalid prediction provenance")
    for key in ("benchmark", "input_condition", "coordinates"):
        if prediction.get(key) != manifest[key]:
            raise ValueError(f"Prediction {key} differs; never mix input conditions or world gauges")
    if prediction.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Predictions belong to another frozen manifest")
    audited_hash = manifest["leakage"].get("checkpoint_sha256")
    if audited_hash and prediction.get("checkpoint", {}).get("sha256") != audited_hash:
        raise ValueError("Prediction checkpoint differs from the training scene inventory")
    if [c["id"] for c in prediction["cases"]] != [c["id"] for c in manifest["cases"]]:
        raise ValueError("Predictions must cover every frozen sample exactly once in order")
    for case, pred in zip(manifest["cases"], prediction["cases"]):
        path = (predictions_path.parent / pred["file"]).resolve()
        if not path.is_relative_to(predictions_path.parent) or sha256_file(path) != pred["sha256"]:
            raise ValueError("Prediction changed or escaped the prediction directory")
        if pred["input_sha256"] != case["input_sha256"]:
            raise ValueError("Prediction belongs to different RGB or query samples")
    return manifest, prediction


def score_predictions(manifest_path, predictions_path, official_root, output, *,
                      alignment_official_root=None, allow_unverified_release=False):
    manifest_path, predictions_path, output = map(Path, (manifest_path, predictions_path, output))
    manifest, predictions = load_predictions(manifest_path, predictions_path)
    benchmark = manifest["benchmark"]
    if benchmark != "worldtrack" and predictions.get("metric_support", {}).get("native_visibility") is False:
        raise ValueError("Position-only prediction has no native visibility; use the explicitly labeled position-only scoring bridge, never fabricate AJ/OA")
    official = {"tapvid3d": OfficialTAPVid3D, "worldtrack": OfficialWorldTrack,
                "mvtracker": OfficialMVTracker}[benchmark](official_root)
    provenance = official.provenance
    if benchmark == "mvtracker":
        provenance = dict(provenance, aggregation=official.aggregation_provenance())
    if provenance != manifest["official_sources"]:
        raise ValueError("Preparation and scoring use different official source versions")
    alignment = None
    if benchmark == "mvtracker" and manifest["protocol"].get("mode") == "rgb_only_3d_query_anchor_sim3":
        if alignment_official_root is None:
            raise ValueError("RGB-only query-Sim(3) scoring extension needs --alignment-official-root pointing to pinned St4RTrack")
        alignment = OfficialWorldTrack(alignment_official_root)
        if alignment.provenance != manifest["protocol"]["alignment_sources"]:
            raise ValueError("Scorer-only alignment implementation differs from the frozen extension")
    release_binding = manifest["protocol"].get("release_binding", {"status": "partial_unverified", "coverage_verified": False})
    if benchmark == "worldtrack" and not release_binding["coverage_verified"] and not allow_unverified_release:
        raise ValueError("WorldTrack release coverage is partial/unverified: import --release-inventory during prepare, "
                         "or explicitly use --allow-unverified-release for partial diagnostics only")
    if benchmark == "tapvid3d":
        expected = [f"{subset}/{name}" for subset in manifest["protocol"]["subsets"]
                    for name in official.filenames(subset, manifest["protocol"]["split"])]
        if [c["id"] for c in manifest["cases"]] != expected:
            raise ValueError("TAPVid-3D cases differ from the official complete split list")
    elif benchmark == "mvtracker":
        protocol = PROTOCOLS[manifest["protocol"]["name"]]
        if len(manifest["cases"]) != protocol.sequences:
            raise ValueError("MVTracker requires the complete frozen sequence list")
        dataset = official.dataset(protocol, manifest["data_root"])
        if list(dataset.seq_names) != [c["name"] for c in manifest["cases"]]:
            raise ValueError("MVTracker sequence list differs from the pinned official reader")
        for case in manifest["cases"]:
            if (case["views"] != list(protocol.views) or case["queries"] != protocol.queries
                    or (protocol.frames and case["frames"] != protocol.frames)):
                raise ValueError("MVTracker frame/query/view configuration changed")
    if output.exists():
        raise FileExistsError("Score output already exists; preserve prior evaluation")
    output.parent.mkdir(parents=True, exist_ok=True)
    grouped, per_case, diagnostics, unaligned_rows = {}, [], [], {}
    if benchmark == "worldtrack":
        # Original evaluator expects dense .npy paths containing the video stem.
        export = output.parent / (output.stem + "_worldtrack")
        export.mkdir(exist_ok=False)
        for subset in manifest["protocol"]["subsets"]:
            rows = [(case, pred) for case, pred in zip(manifest["cases"], predictions["cases"])
                    if case["subset"] == subset]
            paths = []
            subset_dir = export / subset
            subset_dir.mkdir()
            for i, (case, pred) in enumerate(rows):
                path = subset_dir / (Path(case["name"]).stem + ".npy")
                with np.load(predictions_path.parent / pred["file"], allow_pickle=False) as archive:
                    np.save(path, archive["tracks_XYZ"])
                paths.append(path)
            with (subset_dir / "official_stdout.txt").open("w") as stream, contextlib.redirect_stdout(stream):
                grouped[subset] = official.score(manifest["data_root"], subset, paths,
                    [case["sources"][0]["path"] for case, _ in rows], subset_dir,
                    reconstruction=manifest["protocol"].get("task") == "reconstruction")
    else:
        rows_by_subset = {}
        for case, pred in zip(manifest["cases"], predictions["cases"]):
            with np.load(predictions_path.parent / pred["file"], allow_pickle=False) as archive:
                tracks, visible = archive["tracks_XYZ"], archive["visibility"]
            alignment_record, unaligned_metrics = None, None
            if benchmark == "tapvid3d":
                with np.load(case["sources"][0]["path"], allow_pickle=True) as gt:
                    if manifest["protocol"]["scaling"] == "median" and not np.any(gt["visibility"] & visible):
                        diagnostics.append({"id": case["id"], "condition": "no_jointly_visible_points",
                            "behavior": "official median scale undefined; original finite metric outputs preserved"})
                metrics = official.score(case["sources"][0]["path"], tracks, visible,
                                         manifest["protocol"]["scaling"])
            else:
                with np.load(manifest_path.parent / case["labels"], allow_pickle=False) as archive:
                    labels = {key: archive[key] for key in archive.files}
                finite_tracks(tracks, labels["tracks"].shape)
                boolean_visibility(visible, labels["visible"].shape)
                if alignment is not None:
                    unaligned_metrics = official.score(labels, tracks, visible, PROTOCOLS[manifest["protocol"]["name"]])
                    unaligned_rows.setdefault(case["subset"], []).append(unaligned_metrics)
                    tracks, alignment_record = align_query_points_for_scoring(alignment, labels, tracks)
                metrics = official.score(labels, tracks, visible, PROTOCOLS[manifest["protocol"]["name"]])
            rows_by_subset.setdefault(case["subset"], []).append(metrics)
            per_case.append({"id": case["id"], "metrics": metrics, "scoring_alignment": alignment_record,
                             "unaligned_metrics": unaligned_metrics})
        for subset, metrics in rows_by_subset.items():
            if benchmark == "tapvid3d":
                grouped[subset] = official.average(metrics)
            else:
                grouped[subset] = official.average(metrics)
        if benchmark == "tapvid3d" and set(grouped) == {"adt", "drivetrack", "pstudio"}:
            grouped["Overall"] = official.average(list(grouped.values()))
    metrics, unavailable = json_metrics(grouped)
    records, per_case_unavailable = [], []
    for row in per_case:
        values, missing = json_metrics(row["metrics"])
        records.append({"id": row["id"], "metrics": values, "scoring_alignment": row["scoring_alignment"]})
        if row["unaligned_metrics"] is not None:
            records[-1]["unaligned_metrics"], missing_raw = json_metrics(row["unaligned_metrics"])
            per_case_unavailable.extend(row["id"] + "/unaligned" + p for p in missing_raw)
        per_case_unavailable.extend(row["id"] + p for p in missing)
    report = {"schema": "released-tracking-scores-v1", "benchmark": benchmark,
              "manifest_sha256": sha256_file(manifest_path), "predictions_sha256": sha256_file(predictions_path),
              "official_sources": provenance, "protocol": manifest["protocol"],
              "input_condition": manifest["input_condition"], "coordinates": manifest["coordinates"],
              "checkpoint": predictions["checkpoint"], "leakage": manifest["leakage"],
              "sample_ids": [case["id"] for case in manifest["cases"]],
              "sample_counts": {subset: sum(c["subset"] == subset for c in manifest["cases"])
                                for subset in dict.fromkeys(c["subset"] for c in manifest["cases"])},
              "inference_runtime_versions": predictions.get("runtime_versions"),
              "scoring_runtime_versions": runtime_versions(),
              "metrics": metrics, "per_case": records, "unavailable_metric_paths": unavailable,
              "official_diagnostics": diagnostics,
              "pixel_query_policy": manifest.get("pixel_query_policy", "legacy_half_pixel_inbounds"),
              "out_of_canvas_queries": {case["id"]: case.get("out_of_canvas_queries", 0) for case in manifest["cases"]},
              "unavailable_per_case_paths": per_case_unavailable,
              "unavailable_reason": "null preserves nonfinite official outputs, e.g. no dynamic tracks or disabled RANSAC Sim3",
              "claim": "official metric implementation; model/input extension as labeled; not TAPVid-MV"}
    if predictions.get("checkpoint", {}).get("kind") in ("composed_rgb_only_baseline", "author_rgb_only_native_baseline"):
        report["method"] = predictions["method"]
        report["metric_support"] = predictions.get("metric_support", {})
        report["manifest_leakage_metadata"] = report["leakage"]
        report["leakage"] = {"status": "unknown",
            "reason": "Author baseline training scene inventory was not supplied; another model's manifest audit is not baseline evidence",
            "evaluation_scene_ids": manifest["leakage"].get("evaluation_scene_ids", [])}
    if benchmark == "worldtrack":
        report["release_binding"] = release_binding
        if manifest["protocol"].get("task") == "reconstruction":
            report["not_computed_metrics"] = ["avg_pts_pertraj", "epe_pertraj", "avg_pts_sim3", "epe_sim3", "avg_pts_sim3_closed_dyn"]
            report["official_aggregation_note"] = "Disabled EPE metrics may be zero due to upstream shared denominator; these are not reconstruction results. Original raw outputs preserved."
        report["result_tier"] = "verified_inventory_protocol" if release_binding["coverage_verified"] else "partial_unverified_diagnostic"
        if not release_binding["coverage_verified"]:
            report["claim"] = "PARTIAL/UNVERIFIED WorldTrack subset diagnostic; no complete official-release comparison"
    if alignment is not None:
        report["result_tier"] = "rgb_only_3d_query_anchor_sim3"
        report["original_protocol_comparable"] = False
        report["claim"] = "RGB-only inference; GT query anchors used only by declared scorer-side global Sim(3); official MVTracker metric bodies; NOT original protocol"
        report["unaligned_metrics"], report["unaligned_unavailable_paths"] = json_metrics(
            {subset: official.average(rows) for subset, rows in unaligned_rows.items()})
        report["unaligned_metric_interpretation"] = "Raw native world vs official normalized world; arbitrary gauge diagnostic, not comparable 3D accuracy"
    write_json(output, report)
    return output
