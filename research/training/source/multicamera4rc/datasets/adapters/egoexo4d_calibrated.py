"""Synchronized GoPro calibration and optional manual EgoPose joint motion."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from numbers import Real

import cv2
import numpy as np

from .base import InvalidClipError, UnifiedClip
from .egoexo4d import EgoExo4DAdapter, _EGOPOSE_JOINT_NAMES


class EgoExo4DCalibratedAdapter(EgoExo4DAdapter):
    """Load only RGB, fixed GoPro calibration and (optionally) two labeled times."""

    dataset_name = "egoexo4d_calibrated"

    def __init__(self, root, split="train", *, sequence_names=None, cameras=None,
                 min_cameras=2, max_cameras=5, downscaled_size=448, fps=30.0,
                 joint_motion=False, joint_holdout_max_reprojection_px=None,
                 cache_dir=None, frame_disk_cache=False, strict=True, include_aria=False):
        self.include_aria = bool(include_aria)
        min_cameras -= int(self.include_aria)
        max_cameras -= int(self.include_aria)
        if min_cameras < 2 or max_cameras < min_cameras:
            raise ValueError("Ego-Exo4D calibration requires at least two GoPro cameras")
        self.joint_motion = bool(joint_motion)
        self.joint_holdout_max_reprojection_px = joint_holdout_max_reprojection_px
        if joint_holdout_max_reprojection_px is not None:
            if (not self.joint_motion or min_cameras < 3
                    or not isinstance(joint_holdout_max_reprojection_px, Real)
                    or isinstance(joint_holdout_max_reprojection_px, bool)
                    or not np.isfinite(joint_holdout_max_reprojection_px)
                    or joint_holdout_max_reprojection_px <= 0):
                raise ValueError("joint holdout screening requires joint motion, at least three cameras and a positive pixel tolerance")
        super().__init__(
            root, split, sequence_names=sequence_names, cameras=cameras,
            min_cameras=min_cameras, max_cameras=max_cameras, fps=fps,
            downscaled_size=downscaled_size, require_depth=False,
            require_camera_pose=self.joint_motion, undistort=True, cache_dir=cache_dir,
            frame_disk_cache=frame_disk_cache, strict=strict,
        )

    def _annotation_path(self, take_uid, split, directory):
        return self.root / "annotations" / "ego_pose" / split / directory / f"{take_uid}.json"

    def _camera_pose_path_for_uid(self, take_uid, split):
        if not self.joint_motion:
            return None
        path = self._annotation_path(take_uid, split, "camera_pose")
        body = self._annotation_path(take_uid, split, "body/annotation")
        return path if path.is_file() and body.is_file() else None

    def _calibrated_camera_ids_for_index(self, trajectory_dir, cameras, pose_path):
        path = trajectory_dir / "gopro_calibs.csv"
        if not path.is_file():
            return []
        with path.open(newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("cam_uid") in cameras]
        counts = Counter(row["cam_uid"] for row in rows)
        by_graph = defaultdict(list)
        for row in rows:
            camera = row["cam_uid"]
            graph = (row.get("graph_uid") or "").strip()
            if counts[camera] != 1 or not graph or not self._valid_gopro_calibration_row(row):
                continue
            try:
                quality, first, last = (float(row[k]) for k in ("quality", "start_frame_idx", "end_frame_idx"))
            except (KeyError, TypeError, ValueError):
                continue
            dimensions = [float(row[k]) for k in ("image_width", "image_height")]
            translation = [float(row[f"t{axis}_world_cam"]) for axis in "xyz"]
            quaternion = [float(row[f"q{axis}_world_cam"]) for axis in "xyzw"]
            intrinsics = [float(row[f"intrinsics_{i}"]) for i in range(8)]
            if (quality != 1 or (first, last) != (-1, -1)
                    or not np.isfinite(dimensions + translation + quaternion + intrinsics).all()
                    or min(dimensions + intrinsics[:2]) <= 0
                    or not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-3, rtol=0)):
                continue
            by_graph[graph].append(camera)
        # A take may have calibrations in different world frames. Use one graph.
        return sorted(max(by_graph.values(), key=lambda ids: (len(ids), sorted(ids)), default=[]))

    def _common_video_size(self, record, cameras, times):
        sizes, lengths = set(), set()
        for camera in cameras:
            path = self._video_path(record, camera)
            video = cv2.VideoCapture(str(path))
            try:
                if not video.isOpened():
                    raise InvalidClipError(f"cannot open aligned GoPro video {path}")
                fps = float(video.get(cv2.CAP_PROP_FPS))
                if not np.isfinite(fps) or abs(fps - self.fps) > 0.02:
                    raise InvalidClipError(f"{record.take_name}/{camera}: unexpected aligned video FPS {fps}")
                height, width, frames = (video.get(prop) for prop in
                    (cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_COUNT))
                if not np.isfinite([height, width, frames]).all() or min(height, width, frames) < 1:
                    raise InvalidClipError(f"{record.take_name}/{camera}: invalid video dimensions or frame count")
                sizes.add((int(height), int(width)))
                lengths.add(int(frames))
            finally:
                video.release()
        if len(sizes) != 1 or len(lengths) != 1 or times[-1] >= min(lengths):
            raise InvalidClipError(f"{record.take_name}: aligned GoPro videos disagree or lack a requested frame")
        return next(iter(sizes))

    def load_clip(self, sequence_name, frame_indices, *, layout_num_cameras=None,
                  layout_num_times=None, sampling_seed=None):
        record = self._get_record(sequence_name)
        times = list(frame_indices)
        if (not times or any(isinstance(t, bool) or not isinstance(t, (int, np.integer)) for t in times)
                or times != sorted(set(times)) or times[0] < 0
                or (layout_num_times is not None and layout_num_times != len(times))):
            raise InvalidClipError("Ego-Exo4D needs unique increasing physical frame indices")
        if self.joint_motion and (len(times) != 2 or times[1] - times[0] > 30):
            raise InvalidClipError("manual EgoPose motion needs two frames at most 30 frames apart")
        count = (min(self.max_cameras, len(record.cameras)) if layout_num_cameras is None
                 else layout_num_cameras - int(self.include_aria))
        if not self.min_cameras <= count <= min(self.max_cameras, len(record.cameras)):
            raise InvalidClipError(f"{sequence_name}: requested GoPro layout exceeds qualified cameras")
        seed = self._sample_seed(record.take_uid, times) if sampling_seed is None else sampling_seed
        cameras = sorted(random.Random(seed).sample(record.cameras, count))
        height, width = self._common_video_size(record, cameras, times)
        calibs = self._load_calibrations(record, width=width, height=height)
        intrinsics = np.stack([self._rectified_intrinsic(record, c, width, height, calibs[c]) for c in cameras])
        extrinsics = np.stack([calibs[c].extrinsic for c in cameras]).astype(np.float32)
        tracks = {}
        if self.joint_motion:
            tracks = self._manual_body_tracks(record, times, cameras, intrinsics, extrinsics, height, width)
        images, paths = [], []
        for camera in cameras:
            path = self._video_path(record, camera)
            try:
                decoded = self._read_video_frames(path, times, record, camera, width, height, calibs[camera])
            except OSError as exc:
                raise InvalidClipError(f"{sequence_name}/{camera}: {exc}") from exc
            images.extend(decoded[t] for t in times)
            paths.extend(f"{path}#{t}" for t in times)
        clip = UnifiedClip(
            dataset_name=self.dataset_name, sequence_name=sequence_name, frame_paths=paths,
            images=images, depths=None, normals=None,
            trajs_2d=tracks.get("uv"), trajs_3d_world=tracks.get("world"),
            valids=tracks.get("valid"), visibs=tracks.get("observed"),
            visibility_valids=tracks.get("observed"),
            intrinsics=np.repeat(intrinsics, len(times), axis=0),
            extrinsics=np.repeat(extrinsics, len(times), axis=0),
            metadata=dict(dataset_name=self.dataset_name, split=record.split,
                          take_uid=record.take_uid, selected_cameras=cameras,
                          slot_camera=np.repeat(np.arange(count), len(times)),
                          time_idx=times * count, extrinsics_convention="w2c",
                          supervision_kind="camera_joint_motion" if self.joint_motion else "camera_only",
                          has_temporal_supervision=self.joint_motion),
        )
        if self.include_aria:
            from .egoexo4d_aria import append_aria
            append_aria(self, record, times, clip)
        return clip

    def get_sequence_info(self, sequence_name):
        info = super().get_sequence_info(sequence_name)
        if self.include_aria:
            from .egoexo4d_aria import aria_camera_id
            info["cameras"].append(aria_camera_id(self._get_record(sequence_name)))
        return info

    def _manual_body_tracks(self, record, times, cameras, K, E, height, width):
        # The parent LRU reads automatic labels as well. Cache just the manual
        # file here and keep the two annotation sources separate.
        key = f"manual:{record.split}:{record.take_uid}"
        with self._lock:
            cached = self._egopose_cache.get(key)
            if cached is not None:
                self._egopose_cache.move_to_end(key)
        if cached is None:
            try:
                annotation = json.loads(self._annotation_path(record.take_uid, record.split, "body/annotation").read_text())
                pose = json.loads(self._annotation_path(record.take_uid, record.split, "camera_pose").read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise InvalidClipError(f"{record.take_name}: unreadable manual EgoPose labels") from exc
            cached = (annotation, pose)
            if self.egopose_cache_items:
                with self._lock:
                    self._egopose_cache[key] = cached
                    self._egopose_cache.move_to_end(key)
                    while len(self._egopose_cache) > self.egopose_cache_items:
                        self._egopose_cache.popitem(last=False)
        return _manual_joint_tracks(
            *cached, times, cameras, K, E, height, width,
            holdout_max_reprojection_px=self.joint_holdout_max_reprojection_px,
        )


def _manual_views_pass_holdout(measurements, K, E, max_reprojection_px):
    """Triangulate without each tested view, then score that view in aligned pixels."""
    if len(measurements) < 3:
        return False
    projections = np.asarray(E, dtype=np.float64)[:, :3]
    try:
        rays = []
        for c, pixel in measurements:
            ray = np.linalg.solve(K[c], np.r_[pixel, 1.0])
            if not np.isfinite(ray).all() or abs(ray[2]) <= 1e-10:
                return False
            rays.append(ray[:2] / ray[2])
        for held, (camera, pixel) in enumerate(measurements):
            rows = []
            for index, ((c, _), (u, v)) in enumerate(zip(measurements, rays)):
                if index != held:
                    ex = projections[c]
                    rows.extend((u * ex[2] - ex[0], v * ex[2] - ex[1]))
            design = np.stack(rows)
            if not np.isfinite(design).all():
                return False
            _, singular, vh = np.linalg.svd(design, full_matrices=False)
            homogeneous = vh[-1]
            tolerance = np.finfo(np.float64).eps * max(design.shape) * singular[0]
            if singular[2] <= tolerance or abs(homogeneous[3]) <= 1e-10:
                return False
            xyz = homogeneous / homogeneous[3]
            if any((projections[c] @ xyz)[2] <= 0 for c, _ in measurements):
                return False
            prediction = K[camera] @ projections[camera] @ xyz
            if not np.isfinite(prediction).all() or prediction[2] <= 0:
                return False
            error = np.linalg.norm(prediction[:2] / prediction[2] - pixel)
            if not np.isfinite(error) or error > max_reprojection_px:
                return False
    except np.linalg.LinAlgError:
        return False
    return True


def _manual_joint_tracks(annotation, pose, times, cameras, K, E, height, width, *,
                         holdout_max_reprojection_px=None):
    """Pair one wearer's joint names across both times in a common world frame."""
    if not isinstance(annotation, dict) or not isinstance(pose, dict):
        raise InvalidClipError("invalid manual EgoPose annotation or camera pose")
    people = [annotation.get(str(t)) for t in times]
    if any(not isinstance(row, list) or len(row) != 1 or not isinstance(row[0], dict) for row in people):
        raise InvalidClipError("both frames need unambiguous manual annotations for the camera wearer")
    views = []
    for i, camera in enumerate(cameras):
        item = pose.get(camera)
        try:
            annotation_K = np.asarray(item["camera_intrinsics"], dtype=np.float64)
            annotation_E = np.asarray(item["camera_extrinsics"], dtype=np.float64)
            if (annotation_K.shape != (3, 3) or annotation_E.shape != (3, 4)
                    or not np.isfinite(annotation_K).all()
                    or min(annotation_K[0, 0], annotation_K[1, 1]) <= 0
                    or not np.allclose(annotation_K[2], [0, 0, 1], atol=1e-6, rtol=0)
                    or not np.allclose(annotation_E, E[i, :3], atol=1e-3, rtol=0)):
                raise ValueError("incompatible intrinsics or extrinsics")
            views.append(K[i] @ np.linalg.inv(annotation_K))
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
            raise InvalidClipError(f"{camera}: EgoPose and GoPro calibration do not share a verified frame") from exc

    points = np.zeros((2, len(_EGOPOSE_JOINT_NAMES), 3), dtype=np.float64)
    valid_xyz = np.zeros(points.shape[:-1], dtype=bool)
    uv = np.zeros((len(cameras), 2, len(_EGOPOSE_JOINT_NAMES), 2), dtype=np.float64)
    observed = np.zeros(uv.shape[:-1], dtype=bool)
    for t, (person,) in enumerate(people):
        ann3d, ann2d = person.get("annotation3D") or {}, person.get("annotation2D") or {}
        if not isinstance(ann3d, dict) or not isinstance(ann2d, dict):
            raise InvalidClipError("malformed manual EgoPose joint dictionaries")
        for j, name in enumerate(_EGOPOSE_JOINT_NAMES):
            row = ann3d.get(name)
            try:
                point = np.asarray([row[k] for k in ("x", "y", "z")], dtype=np.float64)
                if int(row["num_views_for_3d"]) < 2 or point.shape != (3,) or not np.isfinite(point).all():
                    continue
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            points[t, j], valid_xyz[t, j] = point, True
            manual_pixels = []
            for c, camera in enumerate(cameras):
                camera_labels = ann2d.get(camera)
                label = camera_labels.get(name) if isinstance(camera_labels, dict) else None
                if not isinstance(label, dict) or label.get("placement") != "manual":
                    continue
                try:
                    ref = views[c] @ np.asarray([label["x"], label["y"], 1.], dtype=np.float64)
                except (KeyError, TypeError, ValueError):
                    continue
                if not np.isfinite(ref).all() or abs(ref[2]) <= 1e-10:
                    continue
                ref = ref[:2] / ref[2]
                if (holdout_max_reprojection_px is not None
                        and 0 <= ref[0] <= width-1 and 0 <= ref[1] <= height-1):
                    manual_pixels.append((c, ref))
                xyz = E[c, :3, :3] @ point + E[c, :3, 3]
                if xyz[2] <= 0:
                    continue
                pixel = (K[c] @ xyz)[:2] / xyz[2]
                if (0 <= pixel[0] <= width-1 and 0 <= pixel[1] <= height-1
                        and np.linalg.norm(pixel - ref) <= 4.0):
                    uv[c, t, j], observed[c, t, j] = pixel, True
            if holdout_max_reprojection_px is not None and (
                    observed[:, t, j].sum() < 3 or not _manual_views_pass_holdout(
                        manual_pixels, K, E, holdout_max_reprojection_px)):
                valid_xyz[t, j] = False
    keep = valid_xyz.all(axis=0) & ((observed.sum(axis=0) >= 2).all(axis=0))
    if int(keep.sum()) < 3:
        if holdout_max_reprojection_px is not None:
            raise InvalidClipError("EgoPose frame pair needs three common joints passing manual GoPro holdout checks at both times")
        raise InvalidClipError("EgoPose frame pair needs three common joints with two verified GoPro views per time")
    n = int(keep.sum())
    return dict(
        world=np.tile(points[:, keep], (len(cameras), 1, 1)).astype(np.float32),
        uv=uv[:, :, keep].reshape(-1, n, 2).astype(np.float32),
        valid=np.ones((len(cameras)*2, n), dtype=bool),
        observed=observed[:, :, keep].reshape(-1, n),
    )
