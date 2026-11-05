"""Independently audit HD timestamps against body metadata, without decoding media.

The output admits geometry training scenes only. It does not certify exact
Kinect synchronization or dynamic surface correspondences. Source JSON/tars
are opened read-only; at most five body JSON members are read per scene.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tarfile
import tempfile
import time


KIND = "panoptic_hd_sync_body_anchor_eligibility"
HELD_OUT = frozenset({"basketball", "boxes", "football", "juggle", "softball", "tennis"})
MAX_ANCHOR_JSON_BYTES = 4 * 1024 * 1024
EXTRACTOR_SOURCE = (
    "https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox/"
    "blob/master/scripts/hdImgsExtractor.sh#L25"
)


def _stat(path):
    value = Path(path).stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns, inode=value.st_ino)


def _positive_time(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def _held_out(scene):
    tokens = re.split(r"[^a-z0-9]+", scene.lower())
    return any(re.fullmatch(re.escape(activity) + r"\d*", token)
               for token in tokens for activity in HELD_OUT)


def _hd_times(payload):
    hd = payload["hd"]
    indices, times = hd["index"], hd["univ_time"]
    if not isinstance(indices, list) or not isinstance(times, list) or len(indices) != len(times):
        raise ValueError("HD sync index/time lengths invalid")
    if not indices or any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("HD sync index must contain nonnegative integer frame numbers")
    if len(set(indices)) != len(indices):
        raise ValueError("HD sync indices must be unique")
    if any(after <= before for before, after in zip(indices, indices[1:])):
        raise ValueError("HD sync indices must increase")
    valid = {index: float(value) for index, value in zip(indices, times)
             if _positive_time(value)}
    if not valid:
        raise ValueError("HD sync contains no positive finite timestamps")
    if any(after <= before for before, after in zip(valid.values(), list(valid.values())[1:])):
        raise ValueError("HD sync positive timestamps must increase")
    return valid


def audit_scene(scene_dir, body_root, *, max_abs_error_ms=1.0):
    """Compare frame-labelled body times to HD ``index``, never array offsets.

    Only first three, middle and last body frames inside valid HD coverage are
    sampled. Every selected anchor must be verified; an invalid empty tail is
    explicitly unverified even when three earlier anchors pass.
    """
    scene_dir, body_root = Path(scene_dir), Path(body_root)
    scene = scene_dir.name
    row = dict(scene=scene, eligible=False, status="unverified", reasons=[], anchors=[])
    if _held_out(scene):
        row.update(status="heldout", reasons=["held-out PStudio sports capture"])
        return row
    try:
        sync_path = scene_dir / f"synctables_{scene}.json"
        sync_before = _stat(sync_path)
        raw = sync_path.read_bytes()
        hd = json.loads(raw)["hd"]
        mapping = _hd_times(dict(hd=hd))
        row.update(sync_path=str(sync_path), sync_sha256=hashlib.sha256(raw).hexdigest(),
                   hd_sync_count=len(hd["index"]),
                   valid_hd_index_min_max=[min(mapping), max(mapping)])
        if _stat(sync_path) != sync_before:
            raise ValueError("HD sync metadata changed during audit")

        archives = sorted((body_root / scene).glob("*Pose3d*.tar"),
                          key=lambda path: ("coco19" not in path.name, path.name))
        if not archives:
            row["reasons"].append("no body pose tar available")
            return row
        body_tar = archives[0]
        before = _stat(body_tar)
        row.update(body_tar=str(body_tar), body_tar_stat=before)
        with tarfile.open(body_tar, "r:") as archive:
            members = []
            for member in archive:
                if not member.isfile():
                    continue
                match = re.search(r"(?:^|/)body3DScene_(\d+)\.json$", member.name)
                if match and int(match.group(1)) in mapping:
                    members.append((int(match.group(1)), member))
            members.sort(key=lambda item: (item[0], item[1].name))
            if len({frame for frame, _ in members}) != len(members):
                raise ValueError("duplicate body frame index within HD coverage")
            row["body_members_within_hd_coverage"] = len(members)
            positions = sorted({0, 1, 2, len(members) // 2, len(members) - 1})
            for position in positions:
                if not 0 <= position < len(members):
                    continue
                frame, member = members[position]
                if not 0 < member.size <= MAX_ANCHOR_JSON_BYTES:
                    raise ValueError("body anchor JSON exceeds the metadata sampling limit")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("body anchor JSON is unreadable")
                with stream:
                    raw = stream.read(MAX_ANCHOR_JSON_BYTES + 1)
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("body anchor JSON must be an object")
                body_time = payload.get("univTime")
                recorded_time = body_time
                if isinstance(body_time, float) and not math.isfinite(body_time):
                    recorded_time = None
                anchor = dict(frame_index=frame, member=member.name,
                    member_sha256=hashlib.sha256(raw).hexdigest(),
                    body_univ_time_ms=recorded_time, hd_sync_time_ms=mapping[frame],
                    body_count=len(payload.get("bodies", [])))
                if recorded_time is not body_time:
                    anchor["body_univ_time_repr"] = repr(body_time)
                row["anchors"].append(anchor)
                if not _positive_time(body_time):
                    anchor.update(verified=False,
                                  reason="missing or nonpositive/nonfinite body univTime")
                else:
                    error = body_time - mapping[frame]
                    anchor.update(error_ms=error, verified=abs(error) <= max_abs_error_ms)
        if _stat(body_tar) != before or _stat(sync_path) != sync_before:
            raise ValueError("source metadata changed during audit")
        if not members:
            row["reasons"].append("no body frame within valid HD sync coverage")
            return row
        verified = [anchor for anchor in row["anchors"] if anchor["verified"]]
        errors = [anchor["error_ms"] for anchor in row["anchors"] if "error_ms" in anchor]
        row.update(verified_anchor_count=len(verified), selected_anchor_count=len(row["anchors"]),
                   maximum_abs_error_ms=max(map(abs, errors)) if errors else None)
        if errors:
            row.update(error_range_ms=[min(errors), max(errors)],
                       constant_offset_within_1ms=max(errors) - min(errors) <= 1.0)
        if len(row["anchors"]) < 3:
            row["reasons"].append("fewer than 3 available anchors")
        if len(verified) != len(row["anchors"]):
            row["reasons"].append("at least one body timestamp mismatches sync table or is unverified")
        if len(verified) >= 3 and len(verified) == len(row["anchors"]):
            row.update(eligible=True, status="verified")
        elif any(abs(error) > max_abs_error_ms for error in errors):
            row["status"] = "mismatch"
    except (OSError, ValueError, TypeError, KeyError, tarfile.TarError) as error:
        row.update(eligible=False, status="error")
        row["reasons"].append(f"{type(error).__name__}: {error}")
    return row


def audit(root, body_root):
    """Produce an admission manifest without reading video or depth payloads."""
    root, body_root = Path(root), Path(body_root)
    if not root.is_dir() or not body_root.is_dir():
        raise ValueError("root and body_root must be existing directories")
    started = time.perf_counter()
    scenes = [audit_scene(path, body_root) for path in sorted(root.iterdir())
              if path.is_dir() and not path.name.startswith(".")]
    return dict(schema_version=1, kind=KIND,
        scope="Downloaded training scenes; geometry only, no surface track GT",
        criterion=("At least 3 independently labelled body-frame timestamps within HD synctable "
            "coverage; all selected anchors must match sync table indexed by body frame number "
            "within 1 ms. Negative or absent timestamps are ineligible; no offset correction."),
        max_abs_error_ms=1.0, minimum_anchor_count=3,
        hd_video_index_policy=("Official hdImgsExtractor.sh uses ffmpeg -start_number 0; "
            "decoded zero-based frame f maps to synctables.hd index f."),
        hd_extractor_source=EXTRACTOR_SOURCE, scenes=scenes,
        eligible_scenes=[row["scene"] for row in scenes if row["eligible"]],
        excluded_scenes=[row["scene"] for row in scenes if not row["eligible"]],
        counts={status: sum(row["status"] == status for row in scenes)
                for status in ("verified", "mismatch", "unverified", "error", "heldout")},
        wall_seconds=time.perf_counter() - started, audited_at_unix_seconds=time.time())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Read-only Kinoptic scene view")
    parser.add_argument("--body-root", required=True, type=Path, help="Read-only raw HD/body scene root")
    parser.add_argument("--output", required=True, type=Path, help="Admission JSON outside both input roots")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if any(output == source.resolve() or source.resolve() in output.parents
           for source in (args.root, args.body_root)):
        parser.error("--output must be outside both read-only input roots")
    try:
        report = audit(args.root, args.body_root)
    except ValueError as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=output.parent,
                                         prefix=".hd-sync-audit-", suffix=".json", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps(dict(output=str(output), counts=report["counts"],
                          eligible_scenes=len(report["eligible_scenes"]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
