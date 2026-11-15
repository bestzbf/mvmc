"""Append synchronized Aria RGB using the official EgoPose landscape convention."""
import json

import cv2
import numpy as np

from .base import InvalidClipError


def aria_camera_id(record):
    names = [p.name.removesuffix("_214-1.mp4") for p in record.video_dir.glob("aria*_214-1.mp4")]
    if len(names) != 1:
        raise InvalidClipError(f"{record.take_name}: expected one synchronized Aria RGB stream")
    return names[0]


def _aria_metadata(adapter, record):
    key = f"aria:{record.take_uid}"
    with adapter._lock:
        cached = adapter._egopose_cache.get(key)
        if cached is not None:
            adapter._egopose_cache.move_to_end(key)
            return cached
    params = None
    try:
        with (record.trajectory_dir / "online_calibration.jsonl").open() as handle:
            for line in handle:
                cameras = json.loads(line)["CameraCalibrations"]
                rgb, = [c for c in cameras if c["Label"] == "camera-rgb"]
                current = np.asarray(rgb["Projection"]["Params"], dtype=np.float64)
                if (rgb["Projection"]["Name"] != "FisheyeRadTanThinPrism"
                        or current.shape != (15,) or not np.isfinite(current).all()
                        or not rgb["Calibrated"]):
                    raise ValueError("invalid Aria RGB factory calibration")
                if params is not None and not np.array_equal(params, current):
                    raise ValueError("time-varying RGB calibration requires timestamp matching")
                params = current
        camera = aria_camera_id(record)
        pose = json.loads(adapter._annotation_path(
            record.take_uid, record.split, "camera_pose").read_text())[camera]
        reference_K = np.asarray(pose["camera_intrinsics"], dtype=np.float64)
        if reference_K.shape != (3, 3) or not np.allclose(
                reference_K, [[150, 0, 255.5], [0, 150, 255.5], [0, 0, 1]], atol=1e-6):
            raise ValueError("unsupported EgoPose Aria rectified camera convention")
        # Official EgoPose export crops 32 factory pixels, then downsamples by 2.
        # See EGO4D/ego-exo4d-egopose handpose/data_preparation/main.py.
        if params is None or not (1200 < params[1] < 1600 and 1200 < params[2] < 1600):
            raise ValueError("expected native 2880-square Aria factory calibration")
        params = params.copy()
        params[0] /= 2
        params[1:3] = (params[1:3] - 0.5 - 32) / 2
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise InvalidClipError(f"{record.take_name}: unusable Aria calibration or pose") from exc
    with adapter._lock:
        if adapter.egopose_cache_items:
            adapter._egopose_cache[key] = (params, pose, camera)
            while len(adapter._egopose_cache) > adapter.egopose_cache_items:
                adapter._egopose_cache.popitem(last=False)
    return params, pose, camera


def append_aria(adapter, record, times, clip):
    try:
        from projectaria_tools.core import calibration
        from projectaria_tools.core.sophus import SE3
    except ImportError as exc:
        raise ImportError("include_aria requires projectaria-tools") from exc
    params, pose, camera = _aria_metadata(adapter, record)
    height, width = clip.images[0].shape[:2]
    side = min(height, width)
    left, top = (width-side)//2, (height-side)//2
    target = calibration.get_linear_camera_calibration(side, side, 150.0*side/512)
    K = np.array([[150.0*side/512, 0, (side-1)/2+left],
                  [0, 150.0*side/512, (side-1)/2+top], [0, 0, 1]], dtype=np.float32)
    path = record.video_dir / f"{camera}_214-1.mp4"
    video = cv2.VideoCapture(str(path))
    images, extrinsics = [], []
    try:
        video.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        source_width, source_height = int(video.get(3)), int(video.get(4))
        if (not video.isOpened() or source_width != source_height or source_width < 1
                or abs(video.get(cv2.CAP_PROP_FPS)-adapter.fps) > 0.02
                or int(video.get(cv2.CAP_PROP_FRAME_COUNT)) != record.num_frames):
            raise InvalidClipError(f"{record.take_name}: Aria video layout or synchronization mismatch")
        source = calibration.CameraCalibration("camera-rgb", calibration.FISHEYE624,
            params, SE3(), 1408, 1408, None, np.pi, "")
        source = source.rescale(np.array([source_width, source_height]), source_width/1408)
        for time in times:
            E = np.eye(4, dtype=np.float32)
            try:
                raw = np.asarray(pose["camera_extrinsics"][str(time)], dtype=np.float64)
                if (raw.shape != (3, 4) or not np.isfinite(raw).all()
                        or not np.allclose(raw[:, :3] @ raw[:, :3].T, np.eye(3), atol=1e-5)
                        or not np.isclose(np.linalg.det(raw[:, :3]), 1., atol=1e-5)):
                    raise ValueError("invalid rigid pose")
                E[:3] = raw
            except (KeyError, TypeError, ValueError) as exc:
                raise InvalidClipError(f"{record.take_name}: missing Aria pose at frame {time}") from exc
            if not video.set(cv2.CAP_PROP_POS_FRAMES, time):
                raise InvalidClipError(f"cannot seek Aria frame {time}")
            ok, bgr = video.read()
            if not ok or abs(video.get(cv2.CAP_PROP_POS_FRAMES)-(time+1)) > 0.1:
                raise InvalidClipError(f"cannot decode exact Aria frame {time}")
            # Frame-aligned MP4 is portrait. EgoPose W2C uses landscape axes.
            rgb = np.ascontiguousarray(np.rot90(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            rectified = calibration.distort_by_calibration(rgb, target, source)
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[top:top+side, left:left+side] = rectified
            images.append(canvas)
            extrinsics.append(E)
    finally:
        video.release()
    count = len(clip.metadata["selected_cameras"])
    clip.images.extend(images)
    clip.frame_paths.extend(f"{path}#{t}" for t in times)
    clip.intrinsics = np.concatenate([clip.intrinsics, np.repeat(K[None], len(times), axis=0)])
    clip.extrinsics = np.concatenate([clip.extrinsics, np.stack(extrinsics)])
    if clip.trajs_3d_world is not None:
        # A valid world joint can be a target without a visible Aria query.
        clip.trajs_3d_world = np.concatenate([clip.trajs_3d_world, clip.trajs_3d_world[:len(times)]])
        for name in ("trajs_2d", "valids", "visibs", "visibility_valids"):
            values = getattr(clip, name)
            setattr(clip, name, np.concatenate([values, np.zeros_like(values[:len(times)])]))
    clip.metadata["selected_cameras"].append(camera)
    clip.metadata["slot_camera"] = np.repeat(np.arange(count+1), len(times))
    clip.metadata["time_idx"] = times * (count+1)
    clip.metadata["aria_input_only_joint_mask"] = True
