"""Native CMU Panoptic HD streams for camera supervision.

The released body annotations are internal human joints, not visible surface
tracks. This reader deliberately exposes no depth, trajectory or occlusion GT.
Calibration follows the official toolbox: OpenCV W2C and centimetre world units.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import threading

import cv2
import numpy as np

from .base import BaseAdapter, InvalidClipError, UnifiedClip


HELD_OUT_ACTIVITIES = frozenset({"basketball", "boxes", "football", "juggle", "softball", "tennis"})


class InvalidPanopticRGBFrame(InvalidClipError):
    """A decoded source image is an unambiguous publisher/decoder placeholder."""
    def __init__(self, *, video, frame, reason, matched_fraction=1.):
        self.video = str(video)
        self.frame = int(frame)
        self.reason = str(reason)
        self.matched_fraction = float(matched_fraction)
        super().__init__(f"invalid Panoptic RGB frame: video={self.video} frame={self.frame} "
                         f"reason={self.reason} matched_fraction={self.matched_fraction:.6f}")


def validate_panoptic_rgb_frame(rgb, *, video, frame):
    """Reject clear placeholders on raw decoded RGB, before remap or padding.

    A regular low-texture image is valid. Dark green placeholders in this
    release can be exactly RGB [0,135,0], so a 255-only check is insufficient.
    Mid-bright green must also be spatially flat; natural textured green is
    not rejected by its hue. This is a corruption check, not a scene mask.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not rgb.size or rgb.dtype != np.uint8:
        raise ValueError("Panoptic RGB validation requires a nonempty uint8 HxWx3 decoded image")
    if int(rgb.max()) <= 4:
        raise InvalidPanopticRGBFrame(video=video, frame=frame, reason="black_placeholder")
    if int(rgb.min()) >= 250:
        raise InvalidPanopticRGBFrame(video=video, frame=frame, reason="white_placeholder")
    zero_rb = (rgb[..., 0] <= 16) & (rgb[..., 2] <= 16)
    bright_green = float((zero_rb & (rgb[..., 1] >= 240)).mean())
    if bright_green > .95:
        raise InvalidPanopticRGBFrame(video=video, frame=frame,
            reason="green_placeholder", matched_fraction=bright_green)
    mean, std = cv2.meanStdDev(rgb)
    mean, std = mean.ravel(), std.ravel()
    green = float((zero_rb & (rgb[..., 1] >= 64)).mean())
    if green > .95 and float(std.max()) <= 2.:
        raise InvalidPanopticRGBFrame(video=video, frame=frame,
            reason="green_placeholder", matched_fraction=green)
    if float(std.max()) <= 1. and float(mean.min()) <= 4. and float(mean.max()) >= 64.:
        raise InvalidPanopticRGBFrame(video=video, frame=frame, reason="saturated_solid_placeholder")


def heldout_sequence_reason(sequence_name, sequence_aliases=None):
    """Block every camera/clip variant of the six TAPVid-3D sports captures.

    An alias maps a source physical sequence to its benchmark activity. Splits
    are at capture level, never by camera, clip, frame or annotation identity.
    The dated raw CMU sequences are separate captures, in the same studio rig.
    """
    aliases = sequence_aliases or {}
    values = [str(sequence_name), str(aliases.get(sequence_name, sequence_name))]
    for value in values:
        tokens = set(re.split(r"[^a-z0-9]+", value.lower()))
        overlap = sorted(activity for activity in HELD_OUT_ACTIVITIES
                         if any(re.fullmatch(re.escape(activity)+r"\d*", token) for token in tokens))
        if overlap:
            return "held_out_pstudio_capture:" + overlap[0]
    return None


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _open_video(path):
    # Thousands of indexed HD streams must not each create a decoder thread
    # pool. Newer OpenCV exposes this FFMPEG option at open time only.
    if hasattr(cv2, "CAP_PROP_N_THREADS"):
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(str(path))


class PanopticAdapter(BaseAdapter):
    """Camera-major synchronized HD RGB with authoritative camera labels only.

    ``max_long_side=None`` preserves the original 1920x1080 sampling. Set an
    explicit cap in a training recipe to lower the memory budget; aspect ratio
    and array-centre intrinsics are preserved. No original files are modified.
    """
    dataset_name = "panoptic"

    def __init__(self, root, split="train", max_long_side=None, cache_dir=None,
                 min_cameras=1, strict=True, video_cache_items=4, map_cache_items=4,
                 index_workers=4, exclude_sequences=None, sequence_aliases=None,
                 exclude_videos=None, verbose=False):
        if split != "train":
            raise ValueError("Panoptic raw adapter supports the disjoint train split only")
        self.root = Path(root)
        self.split = split
        self.max_long_side = None if max_long_side is None else _positive_int(max_long_side, "max_long_side")
        self.min_cameras = _positive_int(min_cameras, "min_cameras")
        self.video_cache_items = _positive_int(video_cache_items, "video_cache_items")
        self.map_cache_items = _positive_int(map_cache_items, "map_cache_items")
        self.strict = bool(strict)
        self.sequence_aliases = dict(sequence_aliases or {})
        self.exclude_sequences = set(exclude_sequences or [])
        self.exclude_videos = set()
        for value in exclude_videos or []:
            path = Path(value)
            if path.is_absolute() or ".." in path.parts or path.suffix != ".mp4" or len(path.parts) != 3 or path.parts[1] != "hdVideos":
                raise ValueError("exclude_videos must contain sequence/hdVideos/name.mp4 relative paths")
            self.exclude_videos.add(path.as_posix())
        matched_exclusions = set()
        self.records = {}
        self.index_report = {"root": str(self.root.resolve()), "schema": "panoptic_camera_only_v1",
                             "excluded_sequences": [], "filtered_videos": [],
                             "unavailable_cameras": [], "trimmed_timelines": []}
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        cache_path = None if cache_dir is None else Path(cache_dir)/"panoptic_video_index_v1.json"
        try:
            cache = json.loads(cache_path.read_text()) if cache_path and cache_path.exists() else {}
        except (ValueError, OSError):
            cache = {}
        probes = []
        for sequence in sorted(p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")
                               and (cache_path is None or p.resolve() != cache_path.parent.resolve())):
            reason = heldout_sequence_reason(sequence.name, self.sequence_aliases)
            if sequence.name in self.exclude_sequences:
                reason = "explicit_sequence_exclusion"
            if reason:
                self.index_report["excluded_sequences"].append(dict(sequence=sequence.name, reason=reason))
                continue
            calibration = sequence/f"calibration_{sequence.name}.json"
            if not calibration.is_file():
                self._index_failure(sequence.name, "missing_calibration", calibration)
                continue
            cameras = {}
            try:
                raw = json.loads(calibration.read_text())
                for camera in raw["cameras"]:
                    if camera["type"] != "hd":
                        continue
                    name = camera["name"]
                    if name in cameras:
                        raise ValueError("duplicate HD camera name")
                    cameras[name] = self._calibration(camera)
            except (ValueError, KeyError, TypeError) as error:
                self._index_failure(sequence.name, "invalid_calibration:"+str(error), calibration)
                continue
            record = dict(cameras={}, calibration=str(calibration),
                          calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest())
            self.records[sequence.name] = record
            folder = sequence/"hdVideos"
            for partial in sorted(folder.glob("*.part")):
                self.index_report["filtered_videos"].append(dict(sequence=sequence.name, path=str(partial), reason="partial_download"))
            for name, camera in sorted(cameras.items()):
                video = folder/f"hd_{name}.mp4"
                relative = video.relative_to(self.root).as_posix()
                if relative in self.exclude_videos:
                    matched_exclusions.add(relative)
                    self.index_report["filtered_videos"].append(dict(sequence=sequence.name,
                        path=str(video), relative_path=relative, reason="excluded_by_manifest"))
                    continue
                if not video.is_file():
                    self.index_report["unavailable_cameras"].append(dict(sequence=sequence.name, camera=name, reason="video_not_present"))
                    continue
                st = video.stat()
                key = str(video.resolve())
                stamp = [st.st_size, st.st_mtime_ns]
                probes.append((sequence.name, name, camera, video, key, stamp, cache.get(key)))
            for video in sorted(folder.glob("*.mp4")):
                if video.stem.removeprefix("hd_") not in cameras:
                    self._index_failure(sequence.name, "video_missing_HD_calibration", video)
        if self.exclude_videos-matched_exclusions:
            raise ValueError("exclude_videos did not match indexed HD cameras: "+str(sorted(self.exclude_videos-matched_exclusions)))
        def inspect_video(item):
            seq, name, camera, video, key, stamp, prior = item
            try:
                if prior and prior.get("stamp") == stamp:
                    info = prior["info"]
                else:
                    cap = _open_video(video)
                    try:
                        if not cap.isOpened():
                            raise ValueError("cannot_open_video")
                        info = dict(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                    frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                                    fps=float(cap.get(cv2.CAP_PROP_FPS)))
                    finally:
                        cap.release()
                if info["frames"] < 1 or not np.isfinite(info["fps"]) or abs(info["fps"]-30000/1001) > .01:
                    raise ValueError("invalid_HD_frame_count_or_fps")
                if [info["width"], info["height"]] != camera["resolution"]:
                    raise ValueError("video_calibration_resolution_mismatch")
                return item, info, None
            except (ValueError, OSError, KeyError, TypeError) as error:
                return item, None, str(error)
        with ThreadPoolExecutor(max_workers=_positive_int(index_workers, "index_workers")) as pool:
            for item, info, error in pool.map(inspect_video, probes):
                seq, name, camera, video, key, stamp, prior = item
                if error:
                    self._index_failure(seq, error, video)
                    continue
                cache[key] = dict(stamp=stamp, info=info)
                self.records[seq]["cameras"][name] = dict(camera, video=str(video), **info)
        for seq, record in list(self.records.items()):
            if len(record["cameras"]) < self.min_cameras:
                self.index_report["excluded_sequences"].append(dict(sequence=seq, reason="insufficient_complete_cameras"))
                del self.records[seq]
                continue
            record["num_frames"] = min(c["frames"] for c in record["cameras"].values())
            for name, camera in record["cameras"].items():
                if camera["frames"] != record["num_frames"]:
                    self.index_report["trimmed_timelines"].append(dict(sequence=seq, camera=name,
                        original_frames=camera["frames"], admitted_frames=record["num_frames"], reason="shared_HD_frame_intersection"))
        if not self.records:
            raise ValueError("no admissible disjoint Panoptic training sequences")
        self.sequences = sorted(self.records)
        self.index_report.update(sequences=len(self.sequences), cameras=sum(len(r["cameras"]) for r in self.records.values()))
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_name(cache_path.name+f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(cache, sort_keys=True))
            os.replace(temporary, cache_path)
        self._reset_runtime()
        if verbose:
            print(f"Panoptic camera-only: {len(self.sequences)} captures, {self.index_report['cameras']} HD streams")

    def _index_failure(self, sequence, reason, path):
        item = dict(sequence=sequence, reason=reason, path=str(path))
        self.index_report["filtered_videos"].append(item)
        if self.strict:
            raise ValueError(f"Panoptic index error: {item}")

    @staticmethod
    def _calibration(camera):
        K, R = np.asarray(camera["K"], float), np.asarray(camera["R"], float)
        t, distortion = np.asarray(camera["t"], float).reshape(-1), np.asarray(camera["distCoef"], float)
        resolution = list(map(int, camera["resolution"]))
        if K.shape != (3, 3) or R.shape != (3, 3) or t.shape != (3,) or distortion.shape != (5,):
            raise ValueError("invalid camera tensor shape")
        if not all(np.isfinite(a).all() for a in (K, R, t, distortion)):
            raise ValueError("nonfinite calibration")
        if min(K[0, 0], K[1, 1]) <= 0 or not np.allclose(K[2], [0, 0, 1], atol=1e-7):
            raise ValueError("invalid pinhole calibration")
        if abs(K[0, 1]) > 1e-7 or abs(K[1, 0]) > 1e-7:
            raise ValueError("nonzero skew is unsupported")
        if not np.allclose(R@R.T, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1, atol=1e-5):
            raise ValueError("nonrigid camera rotation")
        if len(resolution) != 2 or min(resolution) < 1:
            raise ValueError("invalid camera resolution")
        E = np.eye(4)
        E[:3, :3], E[:3, 3] = R, t*.01
        return dict(K=K, E=E, distortion=distortion, resolution=resolution,
                    physical_camera_id=int(camera["node"]))

    def _reset_runtime(self):
        self._pid = os.getpid()
        self._captures, self._maps = OrderedDict(), OrderedDict()
        self._runtime_lock = threading.RLock()

    def close(self):
        for cap in getattr(self, "_captures", {}).values():
            cap.release()
        if hasattr(self, "_captures"):
            self._captures.clear()

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_captures=OrderedDict(), _maps=OrderedDict(), _runtime_lock=None, _pid=None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._reset_runtime()

    def list_sequences(self): return self.sequences.copy()
    def __len__(self): return len(self.sequences)
    def get_sequence_name(self, index): return self.sequences[index]
    def get_num_frames(self, sequence_name): return self.records[sequence_name]["num_frames"]

    def get_sequence_info(self, sequence_name):
        record = self.records[sequence_name]
        first = next(iter(record["cameras"].values()))
        width, height = first["resolution"]
        scale = 1. if self.max_long_side is None else min(1., self.max_long_side/max(width, height))
        return dict(num_frames=record["num_frames"], num_cameras=len(record["cameras"]),
                    cameras=sorted(record["cameras"]), image_size=(max(1, round(height*scale)), max(1, round(width*scale))),
                    original_image_size=(height, width),
                    has_depth=False, has_tracks=False, has_visibility=False,
                    physical_scene=sequence_name, supervision="camera_only",
                    extrinsics_convention="w2c", world_units="metres", fps=30000/1001)

    def _read_frames(self, camera, times):
        key = camera["video"]
        cap = self._captures.pop(key, None)
        if cap is None:
            cap = _open_video(key)
            if not cap.isOpened():
                cap.release()
                raise ValueError(f"cannot open Panoptic video: {key}")
        self._captures[key] = cap
        while len(self._captures) > self.video_cache_items:
            self._captures.popitem(last=False)[1].release()
        output = []
        for frame in times:
            position = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            # Reuse the decoder for nearby increasing frames; random requests
            # seek through the container's keyframe index, never from frame 0.
            if frame < position or frame-position > 16:
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame):
                    raise ValueError(f"Panoptic video seek failed: {key} frame {frame}")
                position = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            while position < frame:
                if not cap.grab():
                    raise ValueError(f"truncated Panoptic video: {key} frame {frame}")
                position += 1
            if position != frame:
                raise ValueError(f"Panoptic seek returned wrong frame: {key} requested {frame}, got {position}")
            ok, bgr = cap.read()
            if not ok or bgr.shape[:2] != (camera["height"], camera["width"]):
                raise ValueError(f"Panoptic RGB decode failure: {key} frame {frame}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            validate_panoptic_rgb_frame(rgb, video=key, frame=frame)
            output.append(rgb)
        return output

    def _rectification(self, sequence, name, camera):
        key = (sequence, name)
        value = self._maps.pop(key, None)
        if value is None:
            width, height = camera["resolution"]
            scale = 1. if self.max_long_side is None else min(1., self.max_long_side/max(width, height))
            target_w, target_h = max(1, round(width*scale)), max(1, round(height*scale))
            sx, sy = target_w/width, target_h/height
            A = np.array([[sx, 0, .5*sx-.5], [0, sy, .5*sy-.5], [0, 0, 1.]])
            K = A@camera["K"]
            mx, my = cv2.initUndistortRectifyMap(camera["K"], camera["distortion"], None,
                                                K, (target_w, target_h), cv2.CV_32FC1)
            # Float32 map roundoff at an exact image boundary is harmless.
            support = (mx >= -1e-4) & (mx <= width-1+1e-4) & (my >= -1e-4) & (my <= height-1+1e-4)
            value = (mx, my, support, K.astype(np.float32), A.astype(np.float32))
        self._maps[key] = value
        while len(self._maps) > self.map_cache_items:
            self._maps.popitem(last=False)
        return value

    def load_clip(self, sequence_name, frame_indices, *, layout_num_cameras=1,
                  layout_num_times=None, sampling_seed=0, canonical_anchor_required=False):
        if canonical_anchor_required:
            raise NotImplementedError("Panoptic camera-only reader has no canonical trajectory bank")
        C = _positive_int(layout_num_cameras, "layout_num_cameras")
        T = len(frame_indices)//C if layout_num_times is None else _positive_int(layout_num_times, "layout_num_times")
        indices = list(frame_indices)
        if T < 1 or len(indices) != C*T or indices != indices[:T]*C:
            raise ValueError("Panoptic requests require C identical camera-major blocks of T times")
        times = indices[:T]
        count = self.get_num_frames(sequence_name)
        if any(isinstance(t, (bool, np.bool_)) or int(t) != t or not 0 <= t < count for t in times):
            raise ValueError("Panoptic frame indices must be valid nonnegative integers")
        times = list(map(int, times))
        if times != sorted(set(times)):
            raise ValueError("Panoptic frame times must be unique and increasing")
        if isinstance(sampling_seed, (bool, np.bool_)) or int(sampling_seed) != sampling_seed:
            raise ValueError("sampling_seed must be an integer")
        record = self.records[sequence_name]
        available = sorted(record["cameras"])
        if C > len(available):
            raise ValueError("Panoptic requested more cameras than available")
        rng = np.random.default_rng(sampling_seed)
        preferred = sorted(rng.choice(available, C, replace=False).tolist())
        if self._pid != os.getpid():
            self.close()
            self._reset_runtime()
        images, intrinsics, extrinsics, support, paths, camera_ids, original_sizes, affines = [], [], [], [], [], [], [], []
        selected, rejected_rgb_cameras = [], []
        with self._runtime_lock:
            # Preserve the original seed's chosen cameras. Only a known corrupt
            # RGB observation admits a replacement, using the same requested
            # times and a deterministic permutation of the remaining cameras.
            candidates = iter(preferred)
            replacements = None
            while len(selected) < C:
                try:
                    name = next(candidates)
                except StopIteration:
                    if replacements is None:
                        remaining = [name for name in available if name not in preferred]
                        replacements = iter(rng.permutation(remaining).tolist())
                    try:
                        name = next(replacements)
                    except StopIteration:
                        raise InvalidClipError(
                            f"Panoptic insufficient valid RGB cameras: sequence={sequence_name} "
                            f"requested={C} accepted={len(selected)} "
                            f"rejected={len(rejected_rgb_cameras)}"
                        ) from None
                camera = record["cameras"][name]
                try:
                    frames = self._read_frames(camera, times)
                except InvalidPanopticRGBFrame as error:
                    rejected_rgb_cameras.append(dict(camera=name, video=error.video,
                        frame=error.frame, reason=error.reason))
                    continue
                selected.append(name)
                mx, my, valid, K, A = self._rectification(sequence_name, name, camera)
                for frame, image in zip(times, frames):
                    output = cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    output[~valid] = 0
                    images.append(output); intrinsics.append(K); extrinsics.append(camera["E"])
                    support.append(valid); camera_ids.append(camera["physical_camera_id"])
                    paths.append(f"{camera['video']}#frame={frame}")
                    original_sizes.append(camera["resolution"][::-1]); affines.append(A)
        metadata = dict(extrinsics_convention="w2c", pixel_coordinates="array_index_v1",
            time_idx=times*C, frame_indices=times*C, slot_time_id=list(range(T))*C,
            slot_camera=[i for i in range(C) for _ in times], slot_physical_camera_id=camera_ids,
            camera_names=[name for name in selected for _ in times], valid_slot=[True]*(C*T),
            camera_valid=[True]*(C*T), num_cameras=C, num_times=T, fps=30000/1001,
            supervision="camera_only", supervision_kind="camera_only", has_depth=False, has_temporal_supervision=False,
            has_visibility=False, has_tracks=False, camera_translation_supervision=False,
            panoptic_raw_skeletons_are_surface_tracks=False, undistorted=True,
            distortion_model="opencv_brown5", original_image_sizes=original_sizes,
            resize_affines=np.stack(affines), pixel_support=support,
            calibration_sha256=record["calibration_sha256"], source_world_units="centimetres",
            rejected_rgb_cameras=rejected_rgb_cameras,
            world_units="metres", split_policy="capture_disjoint_from_all_six_pstudio_activities_v1")
        return UnifiedClip(dataset_name=self.dataset_name, sequence_name=sequence_name,
            frame_paths=paths, images=images, depths=None, normals=None, trajs_2d=None,
            trajs_3d_world=None, valids=None, visibs=None, intrinsics=np.stack(intrinsics),
            extrinsics=np.stack(extrinsics).astype(np.float32), metadata=metadata)

    def sanity_check(self, sequence_name):
        clip = self.load_clip(sequence_name, [0])
        return dict(self.get_sequence_info(sequence_name), decoded_image_size=list(clip.image_size),
                    valid_pixel_fraction=float(clip.metadata["pixel_support"][0].mean()))
