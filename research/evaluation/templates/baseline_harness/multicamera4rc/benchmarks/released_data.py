"""Freeze released sample lists and materialize RGB-only inference boundaries."""

from __future__ import annotations

import io
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .official import OfficialMVTracker, sha256_file
from .protocols import PROTOCOLS
from .released_official import (
    OfficialTAPVid3D,
    OfficialWorldTrack,
    boolean_visibility,
    finite_tracks,
)
from .tapvid_mv import ReleasedPixelInputs

SCHEMA = "released-tracking-benchmark-v1"
INPUT_KEYS = {"video", "queries_xytv", "view_ids", "frame_indices"}
CONDITIONS = {
    "tapvid3d": ("rgb_only", "per_frame_camera"),
    "worldtrack": ("rgb_only_dense_first_frame", "first_camera_world"),
    "mvtracker": ("calibrated_xyz_query_reference", "official_normalized_world"),
}
MVTRACKER_MODES = {
    "rgb_only_3d_query_anchor_sim3": ("rgb_only_3d_query_anchor_sim3", "native_predicted_world"),
    "calibrated-reference": CONDITIONS["mvtracker"],
}
RELEASED_QUERY_POLICY = "preserve_border_extension"
WORLDTRACK_TASKS = ("trajectory", "reconstruction")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def decode_jpegs(values):
    images = []
    for value in values:
        with Image.open(io.BytesIO(bytes(value))) as image:
            images.append(np.asarray(image.convert("RGB")).copy())
    if not images or len({im.shape for im in images}) != 1:
        raise ValueError("Expected a nonempty RGB video with constant resolution")
    return np.stack(images).transpose(0, 3, 1, 2)[None]


def read_inputs(path):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != INPUT_KEYS:
            raise ValueError("Inference archive must contain only RGB, pixel queries, view IDs and frames")
        return ReleasedPixelInputs(torch.from_numpy(archive["video"]),
            torch.from_numpy(archive["queries_xytv"]), tuple(archive["view_ids"].tolist()),
            torch.from_numpy(archive["frame_indices"]))


def pixel_grid(height, width):
    y, x = np.mgrid[:height, :width]
    return np.stack((x.ravel(), y.ravel(), np.zeros(height * width), np.zeros(height * width)), -1)


def tapvid_case(path):
    with np.load(path, allow_pickle=True) as data:
        video = decode_jpegs(data["images_jpeg_bytes"])
        queries = np.asarray(data["queries_xyt"])
        if queries.ndim != 2 or queries.shape[1] != 3:
            raise ValueError("TAPVid-3D queries must be [N,3] x,y,t")
        queries = np.column_stack((queries, np.zeros(len(queries))))
    return video, queries, (0,), np.arange(video.shape[1])


def worldtrack_case(path, reconstruction=False):
    # Model receives every first-frame pixel, never the GT-selected subset.
    # Selection is left to the original evaluator after predictions are saved.
    with np.load(path, allow_pickle=True) as data:
        video = decode_jpegs(data["images_jpeg_bytes"][:64])
    h, w = video.shape[-2:]
    # Reconstruction exports predicted base_points at each observation, not
    # trajectories from frame zero. A single RGB query avoids dense track work.
    queries = np.zeros((1, 4)) if reconstruction else pixel_grid(h, w)
    return video, queries, (0,), np.arange(video.shape[1])


def mvtracker_queries(query_points, visibility, intrinsics, extrinsics, height, width):
    """Project only official query XYZ at its query frame; no future GT tracks.

    MVTracker supplies XYZ queries and calibration. Our model needs one pixel
    query, so use the first visible view in the frozen view order. This is a
    calibrated pixel-query extension, never the RGB-only or native RGBD row.
    """
    query_points = np.asarray(query_points)
    n = len(query_points)
    times = query_points[:, 0]
    if not np.isfinite(query_points).all() or not np.equal(times, times.round()).all():
        raise ValueError("Official query frames must be finite integer slots")
    times = times.astype(int)
    if (times < 0).any() or (times >= visibility.shape[1]).any():
        raise ValueError("Official query frame outside the clip")
    at_query = visibility[:, times, np.arange(n)]
    if not at_query.any(axis=0).all():
        raise ValueError("Every cached query must be visible in at least one selected camera")
    views = at_query.argmax(axis=0)
    w2c = np.asarray(extrinsics)[views, times]
    camera = np.einsum("nij,nj->ni", w2c[..., :3, :3], query_points[:, 1:]) + w2c[..., :3, 3]
    projected = np.einsum("nij,nj->ni", np.asarray(intrinsics)[views, times], camera)
    xy = projected[:, :2] / projected[:, 2:]
    if (camera[:, 2] <= 0).any() or not np.isfinite(xy).all():
        raise ValueError("Cached query projects behind the selected camera")
    queries = np.column_stack((xy, times, views))
    return queries


def mvtracker_pixel_queries(query_times, visibility, pixel_tracks, height, width):
    """Take the frozen cache's 2D observation only at its official query time.

    Annotation preparation may choose a query; the inference archive contains
    only that pixel/time/view, never XYZ, calibration, depth or future tracks.
    """
    times = np.asarray(query_times)
    if times.ndim != 1 or not np.isfinite(times).all() or not np.equal(times, times.round()).all():
        raise ValueError("Cached query times must be finite integer slots")
    times = times.astype(int)
    if (times < 0).any() or (times >= visibility.shape[1]).any():
        raise ValueError("Cached query time outside RGB clip")
    at_query = visibility[:, times, np.arange(len(times))]
    if not at_query.any(axis=0).all():
        raise ValueError("Every cached query needs a visible selected source view")
    views = at_query.argmax(axis=0)
    xy = np.asarray(pixel_tracks)[views, times, np.arange(len(times)), :2]
    if not np.isfinite(xy).all():
        raise ValueError("Cached 2D query must be finite; replacement sampling forbidden")
    return np.column_stack((xy, times, views))


def source_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def check_sources(records):
    for record in records:
        path = Path(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen benchmark source: {path}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Frozen benchmark source changed: {path}")


def leakage_audit(cases, inventory_path=None):
    """Inventory uses canonical dataset/sequence IDs, never numeric IDs alone.

    Users must supply aliases (e.g. WorldTrack PO scene -> training PO scene).
    Filename differences do not prove independence. Unknown stays unknown.
    """
    evaluation = sorted({case["scene_id"] for case in cases})
    if inventory_path is None:
        return {"status": "unknown", "reason": "No checkpoint training scene inventory supplied",
                "evaluation_scene_ids": evaluation}
    inventory = json.loads(Path(inventory_path).read_text())
    trained = inventory.get("scene_ids")
    if not isinstance(trained, list) or not all(isinstance(x, str) for x in trained):
        raise ValueError("Training inventory needs scene_ids: list[str]")
    aliases = inventory.get("evaluation_aliases", {})
    resolved = [aliases.get(scene, scene) for scene in evaluation]
    overlap = sorted(set(resolved) & set(trained))
    checkpoint_hash = inventory.get("checkpoint_sha256", "")
    hash_valid = (isinstance(checkpoint_hash, str) and len(checkpoint_hash) == 64
                  and all(c in "0123456789abcdef" for c in checkpoint_hash))
    covered = (inventory.get("complete") is True and hash_valid
               and all(scene in aliases for scene in evaluation))
    return {"status": "overlap" if overlap else "clear" if covered else "unknown",
            "overlap": overlap, "inventory": source_record(inventory_path),
            "checkpoint_sha256": checkpoint_hash if hash_valid else None,
            "reason": "clear requires a checkpoint hash, a complete inventory and explicit aliases for every evaluation scene",
            "evaluation_scene_ids": evaluation}


def worldtrack_release_binding(cases, inventory_path, provenance):
    """A local nonempty glob cannot establish official release completeness.

    The upstream git repository does not publish a complete file inventory.
    Import a separately verified inventory; never infer expected counts from
    the same directory being evaluated. Human attestation stays explicit.
    """
    selected = list(dict.fromkeys(case["subset"] for case in cases))
    observed = {s: sum(c["subset"] == s for c in cases) for s in selected}
    if inventory_path is None:
        return {"status": "partial_unverified", "coverage_verified": False,
                "observed_counts": observed, "expected_counts": None,
                "reason": "No independent verified official-release inventory supplied; local glob is not completeness evidence"}
    inventory = json.loads(Path(inventory_path).read_text())
    if (inventory.get("schema") != "worldtrack-release-inventory-v1" or inventory.get("complete") is not True
            or inventory.get("repository") != provenance["repository"]
            or inventory.get("code_commit") != provenance["commit"]):
        raise ValueError("WorldTrack inventory must independently attest the complete release and pinned source version")
    for field in ("release_id", "verified_by", "verification_evidence", "official_source_url"):
        if not isinstance(inventory.get(field), str) or not inventory[field].strip():
            raise ValueError(f"WorldTrack release inventory requires explicit {field}")
    authority = inventory.get("verification_authority", "unspecified inventory authority; verifier identity and evidence recorded")
    if not isinstance(authority, str) or not authority.strip():
        raise ValueError("WorldTrack release inventory verification_authority must be a nonempty string")
    subsets = inventory.get("subsets", {})
    allowed_scopes = (set(OfficialWorldTrack.SUBSETS), set(OfficialWorldTrack.SUBSETS) | {"tum"})
    if not isinstance(subsets, dict) or set(subsets) not in allowed_scopes or not set(selected) <= set(subsets):
        raise ValueError("Independent WorldTrack inventory must enumerate all four released trajectory subsets")
    expected = {}
    for subset, rows in subsets.items():
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"WorldTrack inventory has no independently verified files for {subset}")
        by_name = {}
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError(f"Invalid file binding in WorldTrack inventory: {subset}")
            name, digest = row.get("name"), row.get("sha256")
            if (not isinstance(name, str) or Path(name).name != name or not name.endswith(".npz")
                    or name in by_name or not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError(f"Invalid or duplicate file binding in WorldTrack inventory: {subset}")
            by_name[name] = digest
        expected[subset] = by_name
    for subset in selected:
        actual = {c["name"]: c["sources"][0]["sha256"] for c in cases if c["subset"] == subset}
        if actual != expected[subset]:
            missing = sorted(set(expected[subset]) - set(actual))
            extra = sorted(set(actual) - set(expected[subset]))
            changed = sorted(name for name in set(actual) & set(expected[subset]) if actual[name] != expected[subset][name])
            raise ValueError(f"WorldTrack release inventory mismatch for {subset}: missing={missing}, extra={extra}, changed={changed}")
    return {"status": "verified_inventory_match", "coverage_verified": True,
            "verification_authority": authority,
            "release_id": inventory["release_id"], "verified_by": inventory["verified_by"],
            "verification_evidence": inventory["verification_evidence"],
            "official_source_url": inventory["official_source_url"],
            "inventory": source_record(inventory_path), "observed_counts": observed,
            "upstream_byte_identity_verified": inventory.get("upstream_byte_identity_verified", False),
            "listing": inventory.get("listing"),
            "expected_counts": {s: len(expected[s]) for s in selected}}


def _prepare(benchmark, official_root, data_root, output, *, split="minival", subsets=None,
             protocol_name="panoptic", training_inventory=None, mvtracker_mode="calibrated-reference", release_inventory=None,
             worldtrack_task="trajectory"):
    output, data_root = Path(output).resolve(), Path(data_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preparation output must be empty; never overwrite a frozen manifest")
    output.mkdir(parents=True, exist_ok=True)
    inputs_dir = output / "inputs"
    inputs_dir.mkdir()
    cases = []

    def add(subset, name, raw, sources, *, scene_id, labels=None, calibration=None):
        index = len(cases)
        video, queries, views, frames = raw
        # Validation precedes saving. No labels/depth/calibration are accepted.
        ReleasedPixelInputs(video, queries, views, frames)
        path = inputs_dir / f"{index:06d}.npz"
        np.savez_compressed(path, video=video, queries_xytv=queries,
                            view_ids=np.asarray(views), frame_indices=frames)
        case = {"id": f"{subset}/{name}", "subset": subset, "name": name,
                "scene_id": scene_id, "input": str(path.relative_to(output)),
                "input_sha256": sha256_file(path), "sources": sources,
                "views": list(views), "frames": len(frames), "queries": len(queries),
                "height": video.shape[-2], "width": video.shape[-1],
                "pixel_query_policy": RELEASED_QUERY_POLICY,
                "out_of_canvas_queries": int(np.count_nonzero(
                    (queries[:, 0] < -.5) | (queries[:, 0] >= video.shape[-1] - .5)
                    | (queries[:, 1] < -.5) | (queries[:, 1] >= video.shape[-2] - .5)))}
        for key, arrays in (("labels", labels), ("calibration", calibration)):
            if arrays is not None:
                directory = output / key
                directory.mkdir(exist_ok=True)
                path = directory / f"{index:06d}.npz"
                np.savez_compressed(path, **arrays)
                case[key] = str(path.relative_to(output))
                case[key + "_sha256"] = sha256_file(path)
        cases.append(case)

    if benchmark == "tapvid3d":
        official = OfficialTAPVid3D(official_root)
        subsets = list(subsets or ("adt", "drivetrack", "pstudio"))
        paths = [(s, name, data_root / s / name) for s in subsets for name in official.filenames(s, split)]
        missing = [str(p) for _, _, p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} official {split} files, first: {missing[0]}")
        for subset, name, path in paths:
            add(subset, name, tapvid_case(path), [source_record(path)],
                scene_id=f"tapvid3d/{subset}/{Path(name).stem}")
        condition, coordinates = "rgb_only", "per_frame_camera"
        protocol = {"split": split, "subsets": subsets, "scaling": "median",
                    "metric_resolution_short_side": 256, "queries": "official queries_xyt, unchanged",
                    "aggregation": "official equal video within source; equal source Overall only with all 3"}
    elif benchmark == "worldtrack":
        from natsort import natsorted
        official = OfficialWorldTrack(official_root)
        if worldtrack_task not in WORLDTRACK_TASKS:
            raise ValueError("WorldTrack task must be trajectory or reconstruction")
        reconstruction = worldtrack_task == "reconstruction"
        supported = official.RECON_SUBSETS if reconstruction else official.SUBSETS
        subsets = list(subsets or supported)
        for subset in subsets:
            if subset not in supported:
                raise ValueError(f"Unsupported released WorldTrack {worldtrack_task} subset: {subset}")
            paths = natsorted((data_root / subset).glob("*.npz"))
            if not paths:
                raise FileNotFoundError(f"No released WorldTrack data in {data_root / subset}")
            for path in paths:
                if reconstruction:
                    official.preflight_reconstruction(path)
                add(subset, path.name, worldtrack_case(path, reconstruction), [source_record(path)],
                    scene_id=f"worldtrack/{subset}/{path.stem}")
        condition = "rgb_only_dense_observation_reconstruction" if reconstruction else "rgb_only_dense_first_frame"
        coordinates = "first_camera_world"
        protocol = {"task": worldtrack_task, "subsets": subsets, "frames": "first up to 64, per official eval.sh",
                    "queries": "dense first RGB frame; original scorer selects visible[0], floors pixels, removes OOB",
                    "aggregation": "official per-subset equal video; no invented combined Overall"}
        if reconstruction:
            protocol.update(queries="predicted base_points at every observation pixel; no first-frame trajectory lookup",
                depth_mask="official GT depth 0.1 < depth < 5, scoring only", sim3_for_recon=False)
        protocol["release_binding"] = worldtrack_release_binding(cases, release_inventory, official.provenance)
    elif benchmark == "mvtracker":
        if mvtracker_mode not in MVTRACKER_MODES:
            raise ValueError(f"Unknown MVTracker input mode: {mvtracker_mode}")
        official = OfficialMVTracker(official_root)
        if protocol_name not in PROTOCOLS:
            raise ValueError(f"Unknown MVTracker protocol {protocol_name}; choose from {tuple(PROTOCOLS)}")
        protocol_object = PROTOCOLS[protocol_name]
        dataset = official.dataset(protocol_object, data_root)
        sequences = list(dataset.seq_names)
        if len(sequences) != protocol_object.sequences:
            raise ValueError(f"Need all {protocol_object.sequences} official sequences; got {len(sequences)}")
        # Preflight every cache before invoking a reader: upstream otherwise
        # samples replacement queries and writes cache files silently.
        caches = [Path(dataset.data_root) / name / "cache" / protocol_object.cache_name for name in sequences]
        for path in caches:
            if not path.is_file():
                raise FileNotFoundError(f"Missing official cached queries: {path}; replacement sampling forbidden")
        for i, (name, cache) in enumerate(zip(sequences, caches)):
            loaded = dataset[i]
            if isinstance(loaded, tuple):
                point, gotit = loaded
                if not gotit:
                    raise ValueError(f"Official loader rejected {name}; fake fallback samples forbidden")
            else:
                point = loaded
            def array(value):
                return value.detach().cpu().numpy()
            video, vis = array(point.video), array(point.visibility).astype(bool)
            query = array(point.query_points_3d)
            tracks = array(point.trajectory_3d)
            if len(query) != protocol_object.queries or video.shape[0] != 4:
                raise ValueError("Official sample does not match the fixed four-view/query protocol")
            if protocol_object.frames and video.shape[1] != protocol_object.frames:
                raise ValueError("Official frame count differs from frozen protocol")
            finite_tracks(tracks, (video.shape[1], len(query), 3))
            boolean_visibility(vis, (4, video.shape[1], len(query)))
            if mvtracker_mode == "rgb_only_3d_query_anchor_sim3":
                pixel_tracks = array(point.trajectory)[..., :2]
                finite_tracks(pixel_tracks, (4, video.shape[1], len(query), 2))
                pixels = mvtracker_pixel_queries(query[:, 0], vis, pixel_tracks, *video.shape[-2:])
                labels = {"tracks": tracks, "visible": vis.any(axis=0), "query_points": query,
                          "track_upscaling_factor": np.asarray(point.track_upscaling_factor)}
                calibration = None
            else:
                K, w2c = map(array, (point.intrs, point.extrs))
                pixels = mvtracker_queries(query, vis, K, w2c, *video.shape[-2:])
                labels = {"tracks": tracks, "visible": vis.any(axis=0), "query_points": query,
                          "track_upscaling_factor": np.asarray(point.track_upscaling_factor)}
                calibration = {"query_points": query, "c2w": np.linalg.inv(w2c if w2c.shape[-2] == 4 else
                    np.concatenate((w2c, np.broadcast_to([0, 0, 0, 1], (*w2c.shape[:-2], 1, 4))), axis=-2))}
            # Hash the entire official sequence, including RGB, annotations,
            # sensor/rendered depth and cached queries used by the native reader.
            sequence_root = Path(dataset.data_root) / name
            sources = [source_record(p) for p in sorted(sequence_root.rglob("*")) if p.is_file()]
            add(protocol_name, name, (video, pixels, protocol_object.views, np.arange(video.shape[1])),
                sources, scene_id=f"mvtracker/{protocol_object.family}/{name}",
                labels=labels, calibration=calibration)
        condition, coordinates = MVTRACKER_MODES[mvtracker_mode]
        protocol = dict(protocol_object.to_dict(), mode=mvtracker_mode,
            aggregation="official equal video per dataset; no invented three-dataset Overall")
        if mvtracker_mode == "rgb_only_3d_query_anchor_sim3":
            alignment_source = json.loads(Path(__file__).with_name("released_sources.json").read_text())["worldtrack"]
            protocol.update(extension="rgb_only_3d_query_anchor_sim3_v1",
                query_conversion="cached 2D observation in first visible view at official cached query time",
                query_conditioning="none; original native predicted 3D gauge retained",
                geometry_inputs=[], coordinate_alignment="scorer-only one global query-anchor Sim(3) per video",
                scale_alignment="same global Sim(3); no per-query depth scale or anchor overwrite",
                alignment_sources=alignment_source,
                alignment_function="dust3r/track_eval_util.py:estimate_sim3(ransac=False)",
                alignment_fit="predicted query-time points -> official cached query XYZ; all queries in fixed order",
                scoring_gt_inputs=["official_cached_query_xyz"],
                scoring="unchanged official MVTracker evaluate_3dpt after declared scorer-only alignment",
                original_protocol_comparable=False)
        else:
            protocol.update(query_conversion="first visible view at official XYZ query time, supplied calibration",
                query_conditioning="source camera rotation and query depth set metric motion gauge; official XYZ anchors queries; no GT trajectory fit")
    else:
        raise ValueError("Released benchmarks: tapvid3d, mvtracker, worldtrack; TAPVid-MV is not released")
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Expected a nonempty, unique sample list")
    if len(set(subsets or ())) != len(subsets or ()):
        raise ValueError("Duplicate subsets")
    provenance = official.provenance
    if benchmark == "mvtracker":
        provenance = dict(provenance, aggregation=official.aggregation_provenance())
    manifest = {"schema": SCHEMA, "benchmark": benchmark, "data_root": str(data_root),
                "pixel_query_policy": RELEASED_QUERY_POLICY,
                "official_sources": provenance, "input_condition": condition,
                "coordinates": coordinates, "protocol": protocol, "cases": cases,
                "leakage": leakage_audit(cases, training_inventory)}
    write_json(output / "manifest.json", manifest)
    return output / "manifest.json"


def prepare(benchmark, official_root, data_root, output, **options):
    """Commit the complete prepared dataset atomically; failed runs are retryable."""
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Preparation output must be empty; never overwrite a frozen manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-preparing-", dir=output.parent))
    try:
        _prepare(benchmark, official_root, data_root, staging, **options)
        if output.exists():
            output.rmdir()  # Only the caller's verified empty directory.
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output / "manifest.json"


def load_manifest(path, *, verify_sources=False, inference_only=False):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA or not manifest.get("cases"):
        raise ValueError("Invalid or empty released benchmark manifest")
    condition = CONDITIONS.get(manifest.get("benchmark"))
    if manifest.get("benchmark") == "worldtrack" and manifest.get("protocol", {}).get("task") == "reconstruction":
        condition = ("rgb_only_dense_observation_reconstruction", "first_camera_world")
    if manifest.get("benchmark") == "mvtracker":
        condition = MVTRACKER_MODES.get(manifest.get("protocol", {}).get("mode", "calibrated-reference"))
    if condition != (manifest.get("input_condition"), manifest.get("coordinates")):
        raise ValueError("Benchmark input condition or coordinate convention changed")
    ids = [case["id"] for case in manifest["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs")
    for case in manifest["cases"]:
        keys = ("input", "labels", "calibration")
        if inference_only:
            keys = ("input", "calibration") if condition == MVTRACKER_MODES["calibrated-reference"] else ("input",)
        for key in keys:
            if key in case:
                artifact = (path.parent / case[key]).resolve()
                if not artifact.is_relative_to(path.parent) or sha256_file(artifact) != case[key + "_sha256"]:
                    raise ValueError(f"Frozen {key} changed or escaped manifest directory")
        if verify_sources:
            check_sources(case["sources"])
    if verify_sources and manifest.get("leakage", {}).get("inventory"):
        check_sources([manifest["leakage"]["inventory"]])
    release_inventory = manifest.get("protocol", {}).get("release_binding", {}).get("inventory")
    if verify_sources and release_inventory:
        check_sources([release_inventory])
        listing = manifest["protocol"]["release_binding"].get("listing")
        if listing:
            check_sources([listing])
    return manifest
