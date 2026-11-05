"""Hardware-synchronized HD RGB with masked, near-time Kinoptic geometry.

Kinect exposures are asynchronous. They provide measured surface geometry,
never material-point identities, temporal flow, or static-motion labels.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from .base import UnifiedClip, InvalidClipError
from .panoptic import PanopticAdapter, _positive_int, InvalidPanopticRGBFrame
from .kinoptic_geometry import (depth_rays, read_depth_frame, render_depth,
                               sensor_geometry, world_points, occluder_points,
                               render_occlusion_depth, filter_occluded_depth)
from .kinoptic_stereo_validation import RGB_EDGE_THRESHOLD, RGB_EDGE_RADIUS

FRAME_BYTES = 424 * 512 * 2


class InsufficientKinopticGeometry(InvalidClipError):
    """Requested views have no sufficiently supported measured geometry."""

    def __init__(self, sequence, cameras, times, rejected):
        self.rejected_cameras = rejected
        details = []
        for item in rejected:
            row = {k: item[k] for k in ("camera", "reason", "valid_fractions") if k in item}
            row["counts"] = [{key: value for key, value in q.items() if key.endswith("pixels")}
                             for q in item.get("quality", [])]
            details.append(row)
        super().__init__(f"{sequence}: insufficient verified geometry for C{cameras}T{times}; rejected={details}")


def _support_radius(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not 0 <= value <= 2 or not np.isfinite(value) or int(value) != value):
        raise ValueError(f"{name} must be an integer in [0,2]")
    return int(value)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_timeline(table, *, frame_count=None):
    """Preserve storage indices; reject invalid times and clock-reset overlap.

    Sorting a reset clock would silently pair two physical recording segments.
    Keep only strictly increasing times above the prior running maximum.
    """
    indices = np.asarray(table["index"])
    times = np.asarray(table["univ_time"], dtype=np.float64)
    if (indices.ndim != 1 or times.shape != indices.shape or indices.dtype.kind not in "iuf"
            or not np.isfinite(indices).all() or not np.equal(indices, np.rint(indices)).all()
            or (indices < 0).any() or (np.diff(indices) <= 0).any()):
        raise ValueError("sync table requires unique increasing zero-based storage indices")
    indices = indices.astype(np.int64)
    valid = np.isfinite(times) & (times > 0)
    positive = np.flatnonzero(valid)
    if len(positive):
        values = times[positive]
        maxima = np.maximum.accumulate(values)
        increasing = np.r_[True, values[1:] > maxima[:-1]]
        valid[positive[~increasing]] = False
    if frame_count is not None:
        # Invalid-time tail rows are released without corresponding payloads.
        # A positive-time missing depth frame is a genuine source error.
        if np.any(valid & (indices >= frame_count)):
            raise ValueError("valid sync indices exceed the available raw depth frames")
    return indices[valid], times[valid], dict(total_rows=len(times),
        admitted_rows=int(valid.sum()), invalid_or_ambiguous_rows=int((~valid).sum()))


def nearest_frames(indices, times, requested):
    """Return original raw indices and signed millisecond errors, without interpolation."""
    requested = np.asarray(requested, dtype=np.float64)
    if not len(times):
        return np.full(requested.shape, -1, np.int64), np.full(requested.shape, np.inf)
    right = np.searchsorted(times, requested).clip(0, len(times) - 1)
    left = np.maximum(0, right - 1)
    positions = np.where(np.abs(times[left] - requested) <= np.abs(times[right] - requested), left, right)
    return indices[positions], times[positions] - requested


def fuse_registered_depth(maps, *, minimum_nodes=2, absolute_tolerance_m=.02,
                          relative_tolerance=.01, support_radius_px=0):
    """Keep the frontmost supported surface; do not fill holes or reveal a rear layer."""
    values = np.stack(maps)
    measured = np.isfinite(values) & (values > 0)
    nearest = np.where(measured, values, np.inf).min(axis=0)
    tolerance = absolute_tolerance_m + relative_tolerance * np.where(np.isfinite(nearest), nearest, 0)
    agreement = measured & (np.abs(values - np.where(np.isfinite(nearest), nearest, 0)) <= tolerance)
    support_radius_px = _support_radius(support_radius_px, "cross-node support radius")
    support = agreement.sum(axis=0)
    if support_radius_px:
        # Independent depth rasters rarely land on the exact same HD pixel.
        # Neighbours certify a measured centre only; they never create a label
        # at a hole or supply an interpolated/background surface depth.
        height, width = nearest.shape
        support = np.zeros(nearest.shape, np.uint8)
        for node in range(len(values)):
            found = agreement[node].copy()
            for dy in range(-support_radius_px, support_radius_px+1):
                for dx in range(-support_radius_px, support_radius_px+1):
                    if dy == dx == 0:
                        continue
                    y = slice(max(0, -dy), min(height, height-dy))
                    x = slice(max(0, -dx), min(width, width-dx))
                    neighbour = values[node, max(0, dy):min(height, height+dy),
                                       max(0, dx):min(width, width+dx)]
                    reference = np.where(np.isfinite(nearest[y, x]), nearest[y, x], 0)
                    found[y, x] |= ((neighbour > 0) & np.isfinite(neighbour)
                        & (np.abs(neighbour-reference) <= tolerance[y, x]))
            support += found
    valid = np.isfinite(nearest) & (support >= minimum_nodes)
    depth = np.zeros(nearest.shape, dtype=np.float32)
    if valid.any():
        selected = np.where(agreement[:, valid], values[:, valid], np.nan)
        depth[valid] = np.nanmedian(selected, axis=0)
    return depth, support.astype(np.uint8)


class PanopticKinopticAdapter(PanopticAdapter):
    """HD input views retain exact shared frame labels; depth is time-bounded GT.

    An independently audited ``eligibility_manifest`` is required. All valid
    registered pixels supervise geometry, with no query subsampling. Defaults
    require two Kinect measurements within 5 ms and cross-node depth agreement.
    """
    dataset_name = "panoptic_kinoptic"

    def __init__(self, root, split="train", eligibility_manifest=None,
                 max_long_side=None, cache_dir=None, min_cameras=1, strict=True,
                 video_cache_items=4, map_cache_items=4, index_workers=4,
                 exclude_sequences=None, sequence_aliases=None, exclude_videos=None,
                 max_depth_delta_ms=5., min_depth_nodes=2, min_consistent_nodes=2,
                 min_depth_m=.5, max_depth_m=4.5, depth_edge_threshold_m=.05,
                 agreement_absolute_m=.02, agreement_relative=.01, agreement_radius_px=1,
                 max_view_angle_deg=80., minimum_valid_fraction=.001,
                 depth_ray_cache_items=10, occlusion_guard=True, occlusion_radius_px=2,
                 rgb_verification="stereo", rgb_witnesses=3, rgb_min_witnesses=1,
                 rgb_max_angle_deg=45., rgb_max_long_side=640, rgb_num_disparities=128,
                 rgb_edge_guard=True, verbose=False):
        if eligibility_manifest is None:
            raise ValueError("Kinoptic geometry requires an independently audited eligibility_manifest")
        self.eligibility_manifest = Path(eligibility_manifest)
        admission = json.loads(self.eligibility_manifest.read_text())
        if (admission.get("kind") != "panoptic_hd_sync_body_anchor_eligibility"
                or admission.get("schema_version") != 1):
            raise ValueError("unsupported Kinoptic HD synchronization admission manifest")
        self.min_depth_nodes = _positive_int(min_depth_nodes, "min_depth_nodes")
        self.min_consistent_nodes = _positive_int(min_consistent_nodes, "min_consistent_nodes")
        if self.min_consistent_nodes > self.min_depth_nodes or self.min_depth_nodes > 10:
            raise ValueError("require 1 <= min_consistent_nodes <= min_depth_nodes <= 10")
        numeric = dict(max_depth_delta_ms=max_depth_delta_ms, min_depth_m=min_depth_m,
            max_depth_m=max_depth_m, depth_edge_threshold_m=depth_edge_threshold_m,
            agreement_absolute_m=agreement_absolute_m, agreement_relative=agreement_relative,
            max_view_angle_deg=max_view_angle_deg, minimum_valid_fraction=minimum_valid_fraction)
        for key, value in numeric.items():
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or float(value) <= 0:
                raise ValueError(f"{key} must be a finite positive number")
            setattr(self, key, float(value))
        if self.max_depth_delta_ms > 17 or self.max_depth_m <= self.min_depth_m:
            raise ValueError("depth tolerance must be <=17 ms and the depth range must increase")
        if self.minimum_valid_fraction > 1 or self.max_view_angle_deg >= 90:
            raise ValueError("invalid valid fraction or front-facing normal threshold")
        self.depth_ray_cache_items = _positive_int(depth_ray_cache_items, "depth_ray_cache_items")
        self.agreement_radius_px = _support_radius(agreement_radius_px, "agreement_radius_px")
        if not isinstance(occlusion_guard, (bool, np.bool_)):
            raise ValueError("occlusion_guard must be boolean")
        self.occlusion_guard = bool(occlusion_guard)
        self.occlusion_radius_px = _support_radius(occlusion_radius_px, "occlusion_radius_px")
        if rgb_verification not in ("none", "stereo"):
            raise ValueError("rgb_verification must be none (diagnostic only) or stereo")
        self.rgb_verification = rgb_verification
        if not isinstance(rgb_edge_guard, (bool, np.bool_)):
            raise ValueError("rgb_edge_guard must be boolean")
        self.rgb_edge_guard = bool(rgb_edge_guard)
        self.rgb_witnesses = _positive_int(rgb_witnesses, "rgb_witnesses")
        self.rgb_min_witnesses = _positive_int(rgb_min_witnesses, "rgb_min_witnesses")
        if not self.rgb_min_witnesses <= self.rgb_witnesses <= 10:
            raise ValueError("require rgb_min_witnesses <= rgb_witnesses <= 10")
        if (isinstance(rgb_max_angle_deg, (bool, np.bool_)) or not np.isfinite(rgb_max_angle_deg)
                or not 0 < rgb_max_angle_deg < 90):
            raise ValueError("rgb_max_angle_deg must lie strictly between 0 and 90")
        self.rgb_max_angle_deg = float(rgb_max_angle_deg)
        self.rgb_max_long_side = _positive_int(rgb_max_long_side, "rgb_max_long_side")
        self.rgb_num_disparities = _positive_int(rgb_num_disparities, "rgb_num_disparities")
        if self.rgb_num_disparities % 16:
            raise ValueError("rgb_num_disparities must be divisible by 16")
        self._ray_cache = OrderedDict()
        entries = admission.get("scenes", [])
        if not isinstance(entries, list) or len({entry["scene"] for entry in entries}) != len(entries):
            raise ValueError("admission requires unique scene records")
        self.admission = {entry["scene"]: entry for entry in entries}
        excluded = set(exclude_sequences or [])
        for sequence in Path(root).iterdir():
            if not sequence.is_dir() or sequence.name.startswith("."):
                continue
            proof = self.admission.get(sequence.name)
            if not proof or proof.get("eligible") is not True or proof.get("status") != "verified":
                excluded.add(sequence.name)
        super().__init__(root, split=split, max_long_side=max_long_side, cache_dir=cache_dir,
            min_cameras=min_cameras, strict=strict, video_cache_items=video_cache_items,
            map_cache_items=map_cache_items, index_workers=index_workers,
            exclude_sequences=sorted(excluded), sequence_aliases=sequence_aliases,
            exclude_videos=exclude_videos, verbose=False)
        self.index_report.update(schema="panoptic_kinoptic_hd_geometry_v1",
            eligibility_manifest=str(self.eligibility_manifest.resolve()),
            eligibility_sha256=_sha(self.eligibility_manifest),
            supervision="near_time_registered_geometry_only", max_depth_delta_ms=self.max_depth_delta_ms,
            min_depth_nodes=self.min_depth_nodes, min_consistent_nodes=self.min_consistent_nodes,
            synchronization=[], geometry_exclusions=[])
        for sequence in list(self.records):
            try:
                self._index_depth(sequence)
            except (ValueError, OSError, KeyError, TypeError) as error:
                if self.strict:
                    raise ValueError(f"Kinoptic GT index error for {sequence}: {error}") from error
                self.index_report["geometry_exclusions"].append(dict(sequence=sequence, reason=str(error)))
                del self.records[sequence]
        self.sequences = sorted(self.records)
        if not self.sequences:
            raise ValueError("no independently verified, time-supported Kinoptic HD training sequence")
        self.index_report.update(sequences=len(self.sequences),
            cameras=sum(len(record["cameras"]) for record in self.records.values()),
            admitted_hd_frames=sum(record["num_frames"] for record in self.records.values()))
        if verbose:
            print(f"Kinoptic geometry: {len(self.sequences)} captures, {self.index_report['admitted_hd_frames']} admissible HD times")

    def _index_depth(self, sequence):
        directory = self.root / sequence
        sync_path = directory / f"synctables_{sequence}.json"
        ksync_path = directory / f"ksynctables_{sequence}.json"
        kcal_path = directory / f"kcalibration_{sequence}.json"
        proof = self.admission[sequence]
        if _sha(sync_path) != proof.get("sync_sha256"):
            raise ValueError("HD sync table changed after independent body-time admission")
        anchors = proof.get("anchors", [])
        if (len(anchors) < 3 or any(anchor.get("verified") is not True
                or not np.isfinite(anchor.get("error_ms", np.nan))
                or abs(anchor["error_ms"]) > 1. for anchor in anchors)):
            raise ValueError("HD synchronization lacks three verified independent body-time anchors")
        tar = Path(proof["body_tar"])
        current = tar.stat(); expected = proof["body_tar_stat"]
        if current.st_size != expected["size"] or current.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError("body timestamp source changed after admission; regenerate audit")
        psync = json.loads(sync_path.read_text())
        hd_indices, hd_times, hd_report = valid_timeline(psync["hd"])
        record = self.records[sequence]
        within = hd_indices < record["num_frames"]
        hd_indices, hd_times = hd_indices[within], hd_times[within]
        # Assert the anchors against the actual indexed HD rows as well.
        table = {int(frame): float(t) for frame, t in zip(hd_indices, hd_times)}
        for anchor in anchors:
            frame = anchor["frame_index"]
            if frame not in table or abs(table[frame] - anchor["body_univ_time_ms"]) > 1.:
                raise ValueError("independent body-time anchor does not match the indexed HD frame")
        ksync = json.loads(ksync_path.read_text())
        kcal = json.loads(kcal_path.read_text())
        calibration = json.loads(Path(record["calibration"]).read_text())
        cameras = {camera["name"]: camera for camera in calibration["cameras"]}
        sensors = []
        for node, sensor in enumerate(kcal["sensors"], 1):
            path = directory / "kinect_shared_depth" / f"KINECTNODE{node}" / "depthdata.dat"
            size = path.stat().st_size
            if size % FRAME_BYTES or not size:
                raise ValueError("truncated or empty raw depth stream")
            indices, times, report = valid_timeline(
                ksync["kinect"]["depth"][f"KINECTNODE{node}"], frame_count=size // FRAME_BYTES)
            matches, errors = nearest_frames(indices, times, hd_times)
            sensors.append(dict(node=node, path=str(path), geometry=sensor_geometry(sensor, cameras[f"50_{node:02d}"]),
                matched_frames=matches.astype(np.int32), time_errors_ms=errors.astype(np.float32),
                sync_report=report))
        if not 1 <= len(sensors) <= 10:
            raise ValueError("unsupported number of depth sensor calibrations")
        counts = sum((np.abs(sensor["time_errors_ms"]) <= self.max_depth_delta_ms).astype(np.int16)
                     for sensor in sensors)
        admitted = counts >= self.min_depth_nodes
        if not admitted.any():
            raise ValueError("no HD time has enough independently timestamped depth measurements")
        for sensor in sensors:
            sensor["matched_frames"] = sensor["matched_frames"][admitted]
            sensor["time_errors_ms"] = sensor["time_errors_ms"][admitted]
        record.update(hd_video_frames=record["num_frames"], hd_frames=hd_indices[admitted],
            hd_times_ms=hd_times[admitted], num_frames=int(admitted.sum()), depth_sensors=sensors,
            depth_metadata_sha256=dict(kcalibration=_sha(kcal_path), ksynctables=_sha(ksync_path),
                                       synctables=_sha(sync_path)))
        self.index_report["synchronization"].append(dict(sequence=sequence, hd_sync=hd_report,
            hd_video_frames=record["hd_video_frames"], admitted_hd_frames=record["num_frames"],
            filtered_hd_times=int(len(hd_indices) - admitted.sum()),
            source_world_dome_max_residual_m=max(sensor["geometry"]["world_dome_residual_m"] for sensor in sensors)))

    def _reset_runtime(self):
        super()._reset_runtime()
        self._ray_cache = OrderedDict()

    def __getstate__(self):
        state = super().__getstate__()
        state["_ray_cache"] = OrderedDict()
        return state

    def get_sequence_info(self, sequence_name):
        info = super().get_sequence_info(sequence_name)
        info.update(has_depth=True, supervision="near_time_registered_geometry_only",
            has_temporal_supervision=False, max_depth_delta_ms=self.max_depth_delta_ms,
            min_consistent_nodes=self.min_consistent_nodes,
            frame_index_domain="admitted_hd_times_with_original_frame_labels")
        return info

    def _world_clouds(self, sequence, logical_frame, *, include_occlusion=False):
        record = self.records[sequence]; clouds = []; measurements = []; occluders = []
        for sensor in record["depth_sensors"]:
            delta = float(sensor["time_errors_ms"][logical_frame])
            if abs(delta) > self.max_depth_delta_ms:
                continue
            key = (sequence, sensor["node"])
            rays = self._ray_cache.pop(key, None)
            if rays is None:
                geometry = sensor["geometry"]
                rays = depth_rays(geometry["K_depth"], geometry["dist_depth5"])
            self._ray_cache[key] = rays
            while len(self._ray_cache) > self.depth_ray_cache_items:
                self._ray_cache.popitem(last=False)
            frame = int(sensor["matched_frames"][logical_frame])
            depth = read_depth_frame(sensor["path"], frame)
            points, normals = world_points(depth, rays, sensor["geometry"]["T_world_from_depth"],
                min_depth=self.min_depth_m, max_depth=self.max_depth_m,
                edge_threshold_m=self.depth_edge_threshold_m)
            clouds.append((points, normals))
            if include_occlusion:
                occluders.append(occluder_points(depth, rays, sensor["geometry"]["T_world_from_depth"]))
            measurements.append(dict(node=sensor["node"], raw_depth_frame=frame, delta_ms=delta,
                                     surface_points=len(points)))
        return (clouds, measurements, occluders) if include_occlusion else (clouds, measurements)

    def _rgb_witness_names(self, record, name):
        source = record["cameras"][name]["E"]
        centre = -source[:3, :3].T @ source[:3, 3]
        choices = []
        for other, camera in record["cameras"].items():
            E = camera["E"]
            distance = float(np.linalg.norm((-E[:3, :3].T @ E[:3, 3])-centre))
            if (other != name and distance >= .01 and np.dot(source[2, :3], E[2, :3])
                    >= np.cos(np.deg2rad(self.rgb_max_angle_deg))):
                choices.append((distance, other))
        return [other for _, other in sorted(choices)[:self.rgb_witnesses]]

    def load_clip(self, sequence_name, frame_indices, *, layout_num_cameras=1,
                  layout_num_times=None, sampling_seed=0, canonical_anchor_required=False,
                  camera_names=None):
        if canonical_anchor_required:
            raise NotImplementedError("Kinoptic has no canonical material-point trajectory bank")
        C = _positive_int(layout_num_cameras, "layout_num_cameras")
        T = len(frame_indices) // C if layout_num_times is None else _positive_int(layout_num_times, "layout_num_times")
        indices = list(frame_indices)
        if T < 1 or len(indices) != C*T or indices != indices[:T]*C:
            raise ValueError("Kinoptic requires identical camera-major blocks of HD time indices")
        logical = indices[:T]; record = self.records[sequence_name]
        if any(isinstance(t, (bool, np.bool_)) or int(t) != t or not 0 <= t < record["num_frames"] for t in logical):
            raise ValueError("invalid admitted HD frame index")
        logical = list(map(int, logical))
        if logical != sorted(set(logical)):
            raise ValueError("clip times must be unique and increasing")
        if isinstance(sampling_seed, (bool, np.bool_)) or int(sampling_seed) != sampling_seed:
            raise ValueError("sampling_seed must be an integer")
        if self._pid != os.getpid():
            self.close(); self._reset_runtime()
        available = sorted(record["cameras"])
        if camera_names is not None:
            if (not isinstance(camera_names, (list, tuple)) or len(camera_names) != C
                    or len(set(camera_names)) != C or not set(camera_names) <= set(available)):
                raise ValueError("audit camera_names must identify exactly C distinct available HD cameras")
            available = list(camera_names)
        if C > len(available):
            raise ValueError("more HD cameras requested than available")
        raw_frames = record["hd_frames"][logical].tolist()
        images, depths, Ks, Es, supports, camera_ids, paths, affines = [], [], [], [], [], [], [], []
        selected, rejected, depths_by_camera = [], [], {}
        rgb_images, rectified, occlusion_maps, quality, invalid_rgb = {}, {}, {}, {}, {}
        with self._runtime_lock:
            cloud_data = [self._world_clouds(sequence_name, frame, include_occlusion=self.occlusion_guard)
                          for frame in logical]

            def view(name, *, images_required=False, bounds_required=False):
                if name not in rectified:
                    rectified[name] = self._rectification(sequence_name, name, record["cameras"][name])
                mx, my, support, K, _ = rectified[name]
                if images_required and name not in rgb_images:
                    if name in invalid_rgb:
                        raise invalid_rgb[name]
                    decoded = []
                    try:
                        for image in self._read_frames(record["cameras"][name], raw_frames):
                            image = cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            image[~support] = 0
                            decoded.append(image)
                    except InvalidPanopticRGBFrame as error:
                        invalid_rgb[name] = error
                        raise
                    rgb_images[name] = decoded
                if bounds_required and name not in occlusion_maps:
                    bounds = []
                    for item in cloud_data:
                        candidates = [render_occlusion_depth(points, K, record["cameras"][name]["E"],
                            support.shape, radius_px=self.occlusion_radius_px) for points in item[2]]
                        bound = np.where(np.stack(candidates) > 0, candidates, np.inf).min(axis=0)
                        bound[~np.isfinite(bound)] = 0
                        bounds.append(bound.astype(np.float32))
                    occlusion_maps[name] = bounds
                return rectified[name]

            order = (available if camera_names is not None else
                     np.random.default_rng(sampling_seed).permutation(available).tolist())
            for name in order:
                camera = record["cameras"][name]
                try:
                    mx, my, support, K, A = view(name, images_required=True,
                                                bounds_required=self.occlusion_guard)
                except InvalidPanopticRGBFrame as error:
                    rejected.append(dict(camera=name, reason="invalid_rgb_frame", detail=str(error)))
                    continue
                registered = []
                statistics = []
                for time, item in enumerate(cloud_data):
                    clouds = item[0]
                    maps = [render_depth(points, normals, K, camera["E"], support.shape,
                        max_view_angle_deg=self.max_view_angle_deg) for points, normals in clouds]
                    depth, _ = fuse_registered_depth(maps, minimum_nodes=self.min_consistent_nodes,
                        absolute_tolerance_m=self.agreement_absolute_m,
                        relative_tolerance=self.agreement_relative, support_radius_px=self.agreement_radius_px)
                    depth[~support] = 0
                    row = dict(before_occlusion_pixels=int((depth > 0).sum()))
                    if self.occlusion_guard:
                        depth = filter_occluded_depth(depth, occlusion_maps[name][time],
                            self.agreement_absolute_m, self.agreement_relative)
                    row["after_occlusion_pixels"] = int((depth > 0).sum())
                    registered.append(depth)
                    statistics.append(row)
                if self.rgb_verification == "stereo" and all(depth.any() for depth in registered):
                    from .kinoptic_stereo_validation import verify_stereo_depth
                    view(name, images_required=True)
                    witnesses = self._rgb_witness_names(record, name)
                    verified = []
                    for time, depth in enumerate(registered):
                        votes = np.zeros(depth.shape, np.uint8)
                        diagnostics = []
                        for other in witnesses:
                            try:
                                _, _, other_support, other_K, _ = view(other, images_required=True)
                            except InvalidPanopticRGBFrame as error:
                                diagnostics.append(dict(camera=other, reason="invalid_rgb_frame", detail=str(error)))
                                continue
                            _, report, mask = verify_stereo_depth(depth, rgb_images[name][time], K, camera["E"],
                                dict(image=rgb_images[other][time], K=other_K,
                                     E=record["cameras"][other]["E"], support=other_support),
                                pixel_support=support, max_long_side=self.rgb_max_long_side,
                                num_disparities=self.rgb_num_disparities, rgb_edge_guard=self.rgb_edge_guard)
                            votes += mask
                            diagnostics.append(dict(camera=other, **report))
                        depth = np.where(votes >= self.rgb_min_witnesses, depth, 0).astype(np.float32)
                        statistics[time].update(after_rgb_pixels=int((depth > 0).sum()),
                                                rgb_witnesses=diagnostics)
                        verified.append(depth)
                    registered = verified
                quality[name] = statistics
                fractions = [float((depth > 0).mean()) for depth in registered]
                if min(fractions) < self.minimum_valid_fraction:
                    rejected.append(dict(camera=name, valid_fractions=fractions, quality=statistics))
                    continue
                selected.append(name)
                depths_by_camera[name] = (registered, mx, my, support, K, A)
                if len(selected) == C:
                    break
            if len(selected) != C:
                raise InsufficientKinopticGeometry(sequence_name, C, T, rejected)
            selected.sort()
            for name in selected:
                camera = record["cameras"][name]
                registered, mx, my, support, K, A = depths_by_camera[name]
                view(name, images_required=True)
                for frame, image, depth in zip(raw_frames, rgb_images[name], registered):
                    images.append(image); depths.append(depth); Ks.append(K); Es.append(camera["E"])
                    supports.append(support); camera_ids.append(camera["physical_camera_id"]); affines.append(A)
                    paths.append(f"{camera['video']}#frame={frame}")
        metadata = dict(extrinsics_convention="w2c", pixel_coordinates="array_index_v1",
            time_idx=raw_frames*C, frame_indices=raw_frames*C, slot_time_id=list(range(T))*C,
            slot_camera=[i for i in range(C) for _ in logical], slot_physical_camera_id=camera_ids,
            camera_names=[name for name in selected for _ in logical], valid_slot=[True]*(C*T),
            camera_valid=[True]*(C*T), num_cameras=C, num_times=T, fps=30000/1001,
            supervision="geometry", supervision_kind="geometry", has_depth=True,
            has_temporal_supervision=False, has_visibility=False, has_tracks=False,
            camera_translation_supervision=True, undistorted=True, distortion_model="opencv_brown5",
            original_image_sizes=[record["cameras"][name]["resolution"][::-1] for name in selected for _ in logical],
            resize_affines=np.stack(affines), pixel_support=supports,
            calibration_sha256=record["calibration_sha256"], source_world_units="centimetres",
            world_units="metres", split_policy="capture_disjoint_from_all_six_pstudio_activities_v1",
            hd_universal_times_ms=record["hd_times_ms"][logical].tolist(),
            depth_measurements=[item[1] for item in cloud_data],
            max_depth_delta_ms=self.max_depth_delta_ms, min_consistent_nodes=self.min_consistent_nodes,
            agreement_radius_px=self.agreement_radius_px,
            occlusion_guard=self.occlusion_guard, occlusion_radius_px=self.occlusion_radius_px,
            rgb_verification=self.rgb_verification, rgb_edge_guard=self.rgb_edge_guard,
            rgb_edge_threshold=RGB_EDGE_THRESHOLD, rgb_edge_radius_px=RGB_EDGE_RADIUS,
            rgb_edge_grid="original_candidate_rgb_before_stereo_resize",
            rgb_witnesses=self.rgb_witnesses, rgb_min_witnesses=self.rgb_min_witnesses,
            rgb_max_long_side=self.rgb_max_long_side, rgb_num_disparities=self.rgb_num_disparities,
            geometry_quality={name: quality[name] for name in selected},
            geometry_is_near_time_sensor_measurement=True,
            depth_registration="inverse_Mcolor_world_HD_raw_occlusion_rgb_native_edge_v3",
            depth_metadata_sha256=record["depth_metadata_sha256"],
            eligibility_sha256=self.index_report["eligibility_sha256"],
            rejected_depth_unsupported_cameras=rejected,
            material_point_correspondence_gt=False)
        return UnifiedClip(dataset_name=self.dataset_name, sequence_name=sequence_name,
            frame_paths=paths, images=images, depths=depths, normals=None,
            trajs_2d=None, trajs_3d_world=None, valids=None, visibs=None,
            intrinsics=np.stack(Ks), extrinsics=np.stack(Es).astype(np.float32), metadata=metadata)
