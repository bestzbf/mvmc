"""Select measured-geometry captures from frozen dynamic diagnostics.

This is a read-only evidence qualifier, not GT certification or a training
launcher. Body-core boxes supply region evidence, never segmentation, depth,
visibility or persistent material-point labels. The schema-1 result is a
strict subset of the independent HD-time manifest and remains readable by the
existing mixture builder. Manual image/GT admission is still required.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tarfile


TIME_KIND = "panoptic_hd_sync_body_anchor_eligibility"
REPORT_KIND = "kinoptic_dynamic_human_region_diagnostics_v1"
QUALIFICATION_KIND = "kinoptic_qualified_measured_geometry_subset_v1"
MINIMUM_CORE_PIXELS = 16
_SHA = re.compile(r"^[0-9a-f]{64}$")
_SCENE = re.compile(r"^[A-Za-z0-9_-]+$")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _read_frozen(path):
    before = _stat(path)
    raw = Path(path).read_bytes()
    if _stat(path) != before:
        raise ValueError("input evidence changed while being read")
    return json.loads(raw), _sha_bytes(raw)


def _inside(path, root):
    return path == root or root in path.parents


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _stat(path):
    value = Path(path).stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns, inode=value.st_ino)


def _runtime_sources(runtime):
    """Match the frozen dynamic audit's transitive source-list policy."""
    names = {p.relative_to(runtime).as_posix()
             for p in (runtime / "multicamera4rc/datasets").rglob("*.py")}
    names.update("scripts/" + name for name in (
        "audit_kinoptic_dynamic.py", "audit_kinoptic_hd_sync.py", "build_kinoptic_mixture.py"))
    required = {"multicamera4rc/datasets/" + name for name in (
        "adapters/base.py", "adapters/panoptic.py", "adapters/panoptic_kinoptic.py",
        "adapters/kinoptic_geometry.py", "adapters/kinoptic_stereo_validation.py", "mixture.py")}
    if not required <= names or any(not (runtime / name).is_file() for name in names):
        raise ValueError("runtime is missing required frozen GT/audit source files")
    return {name: _sha(runtime / name) for name in sorted(names)}


def _verify_source_map(report, current):
    recorded = report.get("source_sha256")
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("dynamic report has no frozen source map")
    for name, digest in recorded.items():
        path = PurePosixPath(name)
        if (path.is_absolute() or ".." in path.parts or str(path) != name
                or not isinstance(digest, str) or not _SHA.fullmatch(digest)):
            raise ValueError("invalid relative frozen source path or SHA-256")
    if recorded != current:
        raise ValueError("dynamic report source SHA-256 map differs from current runtime")
    if report.get("audit_script_sha256") != current["scripts/audit_kinoptic_dynamic.py"]:
        raise ValueError("dynamic audit entrypoint SHA-256 is not bound to source map")


def _read_body_members(path, members):
    """Read only bounded JSON members; never extract or alter the archive."""
    found = {}
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            if member.name not in members:
                continue
            if member.name in found or not member.isfile() or not 0 < member.size <= 4 * 1024 * 1024:
                raise ValueError("ambiguous or invalid bounded body JSON member")
            with archive.extractfile(member) as stream:
                raw = stream.read()
            found[member.name] = (_sha_bytes(raw), json.loads(raw))
    if set(found) != set(members):
        raise ValueError("body annotation member disappeared from original archive")
    return found


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _valid_hd_mapping(table):
    """Mirror the reader's running-maximum policy without changing frame IDs.

    Invalid clocks and reset/overlap rows stay unknown; sorting or demanding
    that the entire released positive clock is monotonic would respectively
    invent correspondences or incorrectly reject otherwise usable captures.
    """
    indices, times = table["index"], table["univ_time"]
    if not isinstance(indices, list) or not isinstance(times, list) or len(indices) != len(times):
        raise ValueError("malformed HD time table")
    result = {}
    previous_frame, maximum_time = -1, -math.inf
    for frame, time in zip(indices, times):
        if (isinstance(frame, bool) or not isinstance(frame, (int, float))
                or not math.isfinite(frame) or int(frame) != frame or frame <= previous_frame):
            raise ValueError("HD frame labels must be increasing nonnegative storage indices")
        previous_frame = frame
        if time is None:
            continue  # JSON null becomes NaN in the reader's float64 timeline.
        try:
            clock = float(time)
        except (TypeError, ValueError) as error:
            raise ValueError("HD clock must be numeric or null") from error
        if math.isfinite(clock) and clock > 0 and clock > maximum_time:
            result[int(frame)] = clock
            maximum_time = clock
    if not result:
        raise ValueError("HD time table has no valid increasing clock segment")
    return result


_EXPECTED_INVALID_TYPES = {
    "InsufficientKinopticGeometry": "multicamera4rc.datasets.adapters.panoptic_kinoptic.InsufficientKinopticGeometry",
    "InvalidPanopticRGBFrame": "multicamera4rc.datasets.adapters.panoptic.InvalidPanopticRGBFrame",
    "InvalidClipError": "multicamera4rc.datasets.adapters.base.InvalidClipError",
}


def _expected_unavailable_type(sample):
    """Recognize a terminal typed data error, never a keyword in its context.

    Older non-replay audits serialize a safe reader exception as ``failed``
    after C4 and C2 both lack support. They omit reader_error_type but retain
    Python's fully-qualified final exception line. An ordinary error raised
    while handling that exception must still quarantine the capture.
    """
    trace = sample.get("traceback") or ""
    if not isinstance(trace, str):
        raise ValueError("reported traceback must be text")
    terminals = re.findall(r"^((?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*):(?: |$)", trace, flags=re.MULTILINE)
    identity = terminals[-1] if terminals else sample.get("reader_error_type")
    for short, full in _EXPECTED_INVALID_TYPES.items():
        if identity in (short, full):
            return short
    return None


def _verify_scene_sources(root, row, samples):
    scene = row["scene"]
    sync_path = Path(row["sync_path"]).resolve()
    if sync_path != (root / scene / f"synctables_{scene}.json").resolve():
        raise ValueError(f"{scene}: original sync path does not match dataset root")
    before_sync = _stat(sync_path)
    if _sha(sync_path) != row.get("sync_sha256"):
        raise ValueError(f"{scene}: HD sync source changed since time admission")
    hd = _read(sync_path)["hd"]
    mapping = _valid_hd_mapping(hd)
    archive = Path(row["body_tar"]).resolve()
    if archive.parent.name != scene or not archive.is_file():
        raise ValueError(f"{scene}: invalid original body archive path")
    expected_stat = row.get("body_tar_stat")
    if expected_stat != _stat(archive):
        raise ValueError(f"{scene}: body source changed since time admission")
    anchors = row.get("anchors", [])
    if sum(a.get("verified") is True for a in anchors) < 3:
        raise ValueError(f"{scene}: original time manifest lacks three verified anchors")
    checks = [(a["member"], a["member_sha256"], a["frame_index"], a["body_univ_time_ms"])
              for a in anchors if a.get("verified") is True]
    checks.extend((s["body_member"], s["body_sha256"], s["frame"], s["time_ms"]) for s in samples)
    bodies = _read_body_members(archive, {name for name, *_ in checks})
    for member, digest, frame, timestamp in checks:
        actual_sha, body = bodies[member]
        match = re.search(r"(?:^|/)body3DScene_(\d+)\.json$", member)
        frame = _integer(frame, "body frame")
        timestamp = _finite(timestamp, "body universal time")
        body_time = _finite(body.get("univTime"), "current body universal time")
        if (not match or int(match.group(1)) != frame or digest != actual_sha
                or frame not in mapping or timestamp <= 0 or body_time <= 0
                or abs(body_time - timestamp) > 1e-6 or abs(mapping[frame] - timestamp) > 1.):
            raise ValueError(f"{scene}: body member/hash/frame/timestamp no longer matches HD evidence")
    for sample in samples:
        _, body = bodies[sample["body_member"]]
        count = _integer(sample.get("body_count"), "measured body_count", minimum=1)
        if not isinstance(body.get("bodies"), list) or len(body["bodies"]) < count:
            raise ValueError(f"{scene}: measured frame has no current body-presence evidence")
    if _stat(sync_path) != before_sync or _sha(sync_path) != row["sync_sha256"] or _stat(archive) != expected_stat:
        raise ValueError(f"{scene}: body/sync changed during read-only qualification")
    return dict(sync_path=str(sync_path), sync_sha256=row["sync_sha256"],
                body_tar=str(archive), body_tar_stat=expected_stat,
                hd_timeline_policy="original_storage_indices_above_prior_running_maximum",
                hd_timeline_total_rows=len(hd["index"]), hd_timeline_admitted_rows=len(mapping),
                checked_body_members={name: digest for name, (digest, _) in sorted(bodies.items())})


def _measured_view(view):
    name = view.get("camera")
    if not isinstance(name, str) or not re.fullmatch(r"00_\d{2}", name):
        raise ValueError("dynamic evidence requires explicit HD camera names")
    fraction = _finite(view.get("valid_depth_fraction"), "valid_depth_fraction")
    if not 0 <= fraction <= 1:
        raise ValueError("invalid measured depth fraction")
    quality = view.get("reader_quality")
    if not isinstance(quality, list) or len(quality) != 1:
        raise ValueError("a measured single-time HD view requires one reader-quality row")
    count = _integer(quality[0].get("after_rgb_pixels"), "after_rgb_pixels")
    core = _integer(view.get("core_bbox_valid_depth_pixels"), "core_bbox_valid_depth_pixels")
    core_area = _integer(view.get("core_bbox_pixels"), "core_bbox_pixels")
    if (core > min(count, core_area)
            or view.get("core_bbox_is_region_proxy_not_human_segmentation_gt") is not True):
        raise ValueError("invalid body-core region evidence")
    rgb = view.get("rgb_content", {})
    if rgb.get("valid") is not True:
        raise ValueError("a measured sample cannot contain invalid HD RGB content")
    if count == 0 or fraction == 0 or core < MINIMUM_CORE_PIXELS:
        return name, None
    return name, dict(masked_depth_pixels=count, core_region_masked_pixels=core)


def qualify(root, time_manifest, reports, runtime, output):
    root, time_manifest, runtime, output = map(lambda p: Path(p).resolve(),
                                               (root, time_manifest, runtime, output))
    report_paths = sorted({Path(p).resolve() for p in reports}, key=str)
    if not report_paths:
        raise ValueError("at least one completed frozen dynamic report is required")
    original, input_hash = _read_frozen(time_manifest)
    if original.get("schema_version") != 1 or original.get("kind") != TIME_KIND:
        raise ValueError("requires the original schema-1 HD-time eligibility manifest")
    if "geometry_qualification_kind" in original:
        raise ValueError("qualification must start from the original independent time manifest")
    rows = original.get("scenes")
    if not isinstance(rows, list) or not rows:
        raise ValueError("time manifest has no capture records")
    by_scene = {}
    for row in rows:
        name = row.get("scene")
        if not isinstance(name, str) or not _SCENE.fullmatch(name) or name in by_scene:
            raise ValueError("time manifest needs unique safe capture names")
        by_scene[name] = row
    time_verified = {name for name, row in by_scene.items()
                     if row.get("eligible") is True and row.get("status") == "verified"}
    if set(original.get("eligible_scenes", [])) != time_verified:
        raise ValueError("original time manifest eligible list disagrees with its records")
    summary_path = output.with_name(output.stem + ".summary.json")
    inputs = {time_manifest, *report_paths}
    protected_roots = {root, runtime / "multicamera4rc", runtime / "scripts"}
    protected_roots.update(path.resolve() for path in root.iterdir() if path.is_dir())
    protected_roots.update(Path(row["body_tar"]).resolve().parent.parent for row in rows if row.get("body_tar"))
    for destination in (output, summary_path):
        if (destination in inputs or destination.exists()
                or any(_inside(destination, path) for path in protected_roots)):
            raise ValueError("output must be new and outside read-only data/source directories and inputs")
    current_sources = _runtime_sources(runtime)
    gathered = {name: [] for name in by_scene}
    failures = {name: [] for name in by_scene}
    unsupported = {name: [] for name in by_scene}
    report_provenance = []
    frozen_options = None
    for path in report_paths:
        report, report_sha = _read_frozen(path)
        if (report.get("kind") != REPORT_KIND or report.get("status") != "completed"
                or report.get("source_hashes_unchanged") is not True):
            raise ValueError("requires a completed dynamic report with unchanged frozen sources")
        _verify_source_map(report, current_sources)
        options = report.get("adapter_options", {})
        if (options.get("occlusion_guard") is not True or options.get("rgb_verification") != "stereo"
                or options.get("rgb_edge_guard") is not True):
            raise ValueError("dynamic reports must explicitly enable occlusion_guard, stereo RGB validation and rgb_edge_guard")
        if frozen_options is not None and options != frozen_options:
            raise ValueError("dynamic report parts use different adapter options")
        frozen_options = deepcopy(options)
        arguments = report.get("arguments", {})
        if Path(arguments.get("root", "")).resolve() != root:
            raise ValueError("dynamic report dataset root differs from qualification root")
        prior_manifest = Path(arguments.get("manifest", "")).resolve()
        if not prior_manifest.is_file() or _sha(prior_manifest) != input_hash:
            raise ValueError("dynamic report does not bind the original time manifest")
        entries = report.get("scenes", [])
        if not isinstance(entries, list) or len({row.get("scene") for row in entries}) != len(entries):
            raise ValueError("dynamic report has duplicate or malformed scene rows")
        if set(report.get("expected_scenes", [])) != {row.get("scene") for row in entries}:
            raise ValueError("completed dynamic report is missing an expected scene")
        for entry in entries:
            name = entry.get("scene")
            if name not in by_scene:
                raise ValueError("dynamic report names a capture absent from the original manifest")
            if name not in time_verified:
                continue  # Never upgrade mismatch, unverified, heldout or prior errors.
            if entry.get("status") not in ("completed", "insufficient_body_evidence"):
                failures[name].append(dict(report=str(path), reason=entry.get("reason", "scene not completed"),
                    failure_category="scene_failure_requires_root_cause_review", traceback=entry.get("traceback")))
            body_path = entry.get("body_metadata", {}).get("body_tar")
            if body_path and Path(body_path).resolve() != Path(by_scene[name]["body_tar"]).resolve():
                raise ValueError(f"{name}: dynamic report body archive differs from time admission")
            for sample in entry.get("samples", []):
                status = sample.get("status")
                expected_type = _expected_unavailable_type(sample) if status == "failed" else None
                expected_unsupported = status == "unsupported_after_filter" or expected_type is not None
                if not expected_unsupported and status not in ("measured", "no_body_roi_depth_evidence"):
                    failures[name].append(dict(report=str(path), frame=sample.get("frame"),
                        reason=sample.get("reason", status or "missing sample status"),
                        failure_category=("invalid_rgb_content" if status == "invalid_rgb_content"
                                          else "sample_failure_requires_root_cause_review"),
                        reader_error_type=sample.get("reader_error_type"), traceback=sample.get("traceback")))
                if expected_unsupported:
                    unsupported[name].append(dict(report=str(path), frame=sample.get("frame"),
                        reason=sample.get("reason"), reader_error_type=expected_type or sample.get("reader_error_type"),
                        original_reported_status=status, traceback=sample.get("traceback"),
                        classification="expected_typed_data_unavailable",
                        rejected_cameras=sample.get("rejected_cameras", [])))
                if status == "measured":
                    gathered[name].append(dict(sample=sample, report=str(path)))
        report_provenance.append(dict(path=str(path), sha256=report_sha,
                                      scenes=[row["scene"] for row in entries]))

    result = deepcopy(original)
    scene_provenance = {}
    decisions = []
    for row in result["scenes"]:
        name = row["scene"]
        evidence = {}
        samples = [item["sample"] for item in gathered[name]]
        if name in time_verified:
            scene_provenance[name] = _verify_scene_sources(root, by_scene[name], samples)
        for item in gathered[name]:
            sample = item["sample"]
            frame = _integer(sample.get("frame"), "measured HD frame")
            time_ms = _finite(sample.get("time_ms"), "measured HD time")
            if sample.get("reader_rgb_verification") != "stereo" or sample.get("invalid_rgb_cameras") != []:
                raise ValueError("measured sample lacks clean stereo-validated RGB evidence")
            views = sample.get("views", [])
            names = sample.get("camera_names", [])
            if len(set(names)) != len(names) or set(names) != {view.get("camera") for view in views}:
                raise ValueError("measured sample has inconsistent HD view identities")
            if len({view.get("camera") for view in views}) != len(views):
                raise ValueError("a measured sample cannot double-count the same HD view")
            accepted = dict(filter(lambda item: item[1] is not None, (_measured_view(view) for view in views)))
            if len(accepted) < 2:
                continue
            previous = evidence.setdefault(frame, dict(frame=frame, time_ms=time_ms, views={}, reports=[]))
            if abs(previous["time_ms"] - time_ms) > 1e-6:
                raise ValueError("duplicate physical HD frame has conflicting timestamps")
            previous["views"].update(accepted)
            if item["report"] not in previous["reports"]:
                previous["reports"].append(item["report"])
        frames = [evidence[key] for key in sorted(evidence)]
        qualified = name in time_verified and not failures[name] and len(frames) >= 2
        if name not in time_verified:
            reason = "original_time_admission_ineligible"
        elif failures[name]:
            reason = "dynamic_report_failure_quarantine"
        elif len(frames) < 2:
            reason = "fewer_than_two_distinct_measured_multiview_body_region_frames"
        else:
            reason = "qualified_measured_geometry_subset_pending_manual_gt_admission"
        row.update(eligible=qualified, geometry_qualification_reason=reason,
            geometry_qualification=dict(distinct_measured_frames=len(frames), evidence=frames,
                                        quarantined_failures=failures[name], unsupported_samples=unsupported[name]))
        decisions.append(dict(scene=name, eligible=qualified, time_status=row["status"],
                              reason=reason, distinct_measured_frames=len(frames),
                              quarantined_failures=failures[name], unsupported_samples=unsupported[name]))
    # Recheck both input reports and all runtime bytes immediately before writing.
    if _sha(time_manifest) != input_hash or _runtime_sources(runtime) != current_sources:
        raise ValueError("frozen manifest/runtime changed during qualification")
    if any(_sha(item["path"]) != item["sha256"] for item in report_provenance):
        raise ValueError("dynamic report changed during qualification")
    for name, proof in scene_provenance.items():
        if (_stat(proof["body_tar"]) != proof["body_tar_stat"]
                or _sha(proof["sync_path"]) != proof["sync_sha256"]):
            raise ValueError(f"{name}: body/sync source changed before qualification completed")
    eligible = [row["scene"] for row in result["scenes"] if row["eligible"]]
    result.update(eligible_scenes=eligible,
        excluded_scenes=[row["scene"] for row in result["scenes"] if not row["eligible"]],
        original_time_eligibility_sha256=input_hash, geometry_qualification_kind=QUALIFICATION_KIND,
        time_verified_counts=deepcopy(original.get("counts", dict(Counter(row["status"] for row in rows)))),
        geometry_qualification_counts=dict(qualified=len(eligible), time_verified=len(time_verified),
            failure_quarantined=sum(bool(items) for items in failures.values()),
            unsupported_samples=sum(len(items) for items in unsupported.values())),
        geometry_qualification_policy=dict(minimum_distinct_measured_hd_frames=2,
            minimum_distinct_nonempty_hd_views_per_frame=2, minimum_body_core_supported_views_per_frame=2,
            minimum_body_core_masked_pixels_per_view=MINIMUM_CORE_PIXELS,
            requires_raw_occlusion_guard=True, requires_independent_rgb_stereo=True,
            requires_rgb_edge_guard=True,
            body_core_is_region_proxy_not_segmentation_gt=True, unsupported_samples_do_not_count=True,
            any_ordinary_report_failure_quarantines_scene=True, manual_gt_admission_still_required=True,
            no_motion_visibility_or_material_point_gt_created=True),
        geometry_readonly_provenance=dict(root=str(root), runtime=str(runtime),
            original_time_manifest=str(time_manifest), source_sha256=current_sources,
            qualifier_sha256=_sha(Path(__file__)), adapter_options=frozen_options,
            dynamic_reports=report_provenance, checked_scene_sources=scene_provenance))
    summary = dict(kind="kinoptic_measured_geometry_qualification_summary_v1",
        qualification_manifest=str(output), original_time_eligibility_sha256=input_hash,
        original_time_verified_count=len(time_verified), qualified_geometry_count=len(eligible),
        qualified_geometry_scenes=eligible, scenes=decisions,
        requires_failure_root_cause_review=any(failures.values()),
        ordinary_report_failure_count=sum(len(items) for items in failures.values()),
        unsupported_sample_count=sum(len(items) for items in unsupported.values()),
        manual_gt_admission_required=True, training_activated=False,
        limitations=["Qualified measured geometry subset, not perfect GT or full-image dense trajectories",
                     "Body core boxes are diagnostic regions, not human segmentation or visible depth GT",
                     "Frozen sample evidence does not certify every frame of an admitted capture"])
    output.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for destination, value in ((output, result), (summary_path, summary)):
            with destination.open("x") as stream:
                created.append(destination)
                json.dump(value, stream, indent=2, allow_nan=False)
                stream.write("\n")
    except Exception:
        for destination in created:
            destination.unlink()
        raise
    return result, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "time-manifest", "runtime", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--report", type=Path, nargs="+", required=True,
                        help="One or more completed frozen dynamic reports, deduplicated by physical HD frame")
    args = parser.parse_args(argv)
    _, summary = qualify(args.root, args.time_manifest, args.report, args.runtime, args.output)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
