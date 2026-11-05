"""Prepare a separate Kinoptic geometry mixture without changing training.

The original Panoptic weight is split by admitted capture count. Unverified
captures retain camera-only supervision; other data sources are unchanged.
This command writes one new config and never starts a training process.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from multicamera4rc.datasets.adapters.panoptic import heldout_sequence_reason
from multicamera4rc.datasets.mixture import read_data_config


def _scene_directories(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"dataset root is not a directory: {root}")
    return {p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")}


def build_mixture(base_config, kinoptic_root, eligibility_manifest, *, cache_dir=None):
    """Return a new config; read-only input restrictions are never relaxed."""
    original = read_data_config(base_config)
    root = Path(kinoptic_root).resolve()
    available = _scene_directories(root)
    manifest_path = Path(eligibility_manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("kind") != "panoptic_hd_sync_body_anchor_eligibility"
            or manifest.get("schema_version") != 1):
        raise ValueError("unsupported Kinoptic eligibility manifest")
    records = manifest.get("scenes", [])
    if (not isinstance(records, list) or any(not isinstance(row, dict)
            or not isinstance(row.get("scene"), str) or not row["scene"]
            or Path(row["scene"]).name != row["scene"] for row in records)
            or len({row["scene"] for row in records}) != len(records)):
        raise ValueError("eligibility manifest must contain unique capture names")
    verified = {row["scene"] for row in records
                if row.get("eligible") is True and row.get("status") == "verified"}
    result = deepcopy(original)
    result["sources"] = []
    added = 0
    for source in original["sources"]:
        if source.get("adapter", "").strip().lower() != "panoptic":
            result["sources"].append(deepcopy(source))
            continue
        if source.get("dense_kind", "camera_only") != "camera_only":
            raise ValueError("the original Panoptic source must use camera_only supervision")
        if source.get("manifest"):
            raise ValueError("Panoptic frame-pool manifests require an explicit HD-to-admitted-frame conversion")
        old_root = Path(source["root"])
        options = source.get("options", {})
        candidates = {name for name in _scene_directories(old_root)
                      if (old_root/name/f"calibration_{name}.json").is_file()}
        if "sequences" in source:
            selected = source["sequences"]
            if not isinstance(selected, list) or not selected or any(not isinstance(s, str) for s in selected):
                raise ValueError("original Panoptic sequences must be a nonempty capture whitelist")
            absent = set(selected)-candidates
            if absent:
                raise ValueError(f"original Panoptic whitelist references absent captures: {sorted(absent)}")
            candidates &= set(selected)
        candidates -= set(options.get("exclude_sequences") or [])
        candidates = {name for name in candidates
                      if not heldout_sequence_reason(name, options.get("sequence_aliases"))}
        if not candidates:
            raise ValueError("original Panoptic restrictions leave no training captures")
        measured = sorted(candidates & verified)
        missing = set(measured)-available
        if missing:
            raise ValueError(f"verified captures are absent from the Kinoptic root: {sorted(missing)}")
        fallback = sorted(candidates-set(measured))
        weight = source.get("weight", 1.)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("original Panoptic weight must be finite and positive")
        if measured:
            upgraded = deepcopy(source)
            upgraded.update(adapter="panoptic_kinoptic", root=str(root), dense_kind="geometry",
                sequences=measured, weight=weight*len(measured)/len(candidates),
                name=source.get("name", "panoptic")+"_kinoptic_geometry")
            upgraded["options"] = deepcopy(options)
            # The reader indexes before the mixture applies its whitelist.
            # Exclude all other captures here as well; never index their GT.
            upgraded["options"]["exclude_sequences"] = sorted(
                set(options.get("exclude_sequences") or []) | (available-set(measured)))
            if "exclude_videos" in options:
                # An exclusion for an unindexed capture is an adapter error.
                # Preserve only the already approved exclusions applicable here.
                upgraded["options"]["exclude_videos"] = [p for p in options["exclude_videos"]
                    if Path(p).parts and Path(p).parts[0] in set(measured)]
            upgraded["options"].update(eligibility_manifest=str(manifest_path),
                max_depth_delta_ms=5., min_depth_nodes=2, min_consistent_nodes=2,
                agreement_radius_px=1, min_depth_m=.5, max_depth_m=4.5,
                occlusion_guard=True, occlusion_radius_px=2, rgb_verification="stereo",
                rgb_edge_guard=True,
                rgb_witnesses=3, rgb_min_witnesses=1, rgb_max_long_side=640, rgb_num_disparities=128)
            if cache_dir is not None:
                upgraded["options"]["cache_dir"] = str(Path(cache_dir).resolve())
            result["sources"].append(upgraded)
            added += len(measured)
        if fallback:
            retained = deepcopy(source)
            retained.update(sequences=fallback,
                weight=weight if not measured else weight-upgraded["weight"])
            result["sources"].append(retained)
    if not added:
        raise ValueError("no verified Kinoptic capture remains under the original training restrictions")
    if original.get("supervision") != "dense":
        raise ValueError("Kinoptic geometry requires a dense base mixture")
    return result


def write_mixture(base_config, kinoptic_root, eligibility_manifest, output, *, cache_dir=None):
    config = build_mixture(base_config, kinoptic_root, eligibility_manifest, cache_dir=cache_dir)
    output = Path(output).resolve()
    protected_files = {Path(base_config).resolve(), Path(eligibility_manifest).resolve()}
    protected_roots = {Path(kinoptic_root).resolve()}
    original_sources = read_data_config(base_config)["sources"]
    protected_roots.update(Path(source["root"]).resolve() for source in original_sources)
    # Unified dataset views contain per-capture links across physical disks.
    # Resolving an output through one of those links must not bypass the guard.
    capture_roots = [Path(kinoptic_root)] + [Path(source["root"]) for source in original_sources
        if source.get("adapter", "").strip().lower() == "panoptic"]
    for root in capture_roots:
        protected_roots.update(path.resolve() for path in root.iterdir() if path.is_dir())
    if output in protected_files or any(output == root or root in output.parents for root in protected_roots):
        raise ValueError("output must be a new config outside the input files and read-only dataset roots")
    payload = json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False)+"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(payload)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--kinoptic-root", required=True, type=Path)
    parser.add_argument("--eligibility-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    config = write_mixture(args.base_config, args.kinoptic_root, args.eligibility_manifest,
                           args.output, cache_dir=args.cache_dir)
    selected = [source for source in config["sources"] if source["adapter"] == "panoptic_kinoptic"]
    fallback = [source for source in config["sources"] if source["adapter"] == "panoptic"]
    print(json.dumps(dict(output=str(args.output.resolve()),
        geometry_captures=sum(len(source["sequences"]) for source in selected),
        camera_only_captures=sum(len(source["sequences"]) for source in fallback),
        geometry_weight=sum(source["weight"] for source in selected),
        camera_only_weight=sum(source["weight"] for source in fallback), training_started=False)))


if __name__ == "__main__":
    main()
