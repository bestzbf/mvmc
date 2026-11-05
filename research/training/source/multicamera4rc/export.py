"""Export instantaneous predicted geometry: python -m multicamera4rc.export --help.

This exporter uses each observation's ``base_points`` in the shared predicted
world frame. It does not export trajectories, align to ground truth, or infer
unavailable metric scale. RGB comes from the exact native input images recorded
by manifest inference; padding is never exported.
"""

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch


PLY_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
])
SCALE_NOTE = (
    "Coordinates are in the model's predicted shared world frame. "
    "Uncalibrated RGB inference does not establish metric scale; "
    "do not interpret coordinate values as metres without external scale calibration."
)


def _array(value, name):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Prediction field {name!r} must be a tensor")
    return value.detach().cpu().numpy()


def _grid(height, width, maximum):
    """Sample a regular native-image grid with no more than maximum pixels."""
    if height * width <= maximum:
        y, x = np.mgrid[:height, :width]
        return y.ravel(), x.ravel()
    rows = min(height, max(1, int(math.sqrt(maximum * height / width))))
    columns = min(width, max(1, maximum // rows))
    rows = min(height, maximum // columns)
    y = np.linspace(0, height - 1, rows).round().astype(np.int64)
    x = np.linspace(0, width - 1, columns).round().astype(np.int64)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return yy.ravel(), xx.ravel()


def _write_ply(path, xyz, rgb):
    vertices = np.empty(len(xyz), dtype=PLY_DTYPE)
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = xyz[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        vertices[name] = rgb[:, index]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment instantaneous predicted geometry; scale is not necessarily metres\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    temporary = path.with_suffix(".ply.tmp")
    with temporary.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(vertices.tobytes())
    temporary.replace(path)


def _observations(payload):
    metadata = payload.get("observation_metadata")
    if not isinstance(metadata, list):
        raise ValueError(
            "Missing observation_metadata. Re-run manifest inference with the "
            "current multicamera4rc.infer before exporting; the exporter cannot "
            "guess source RGB images or native padding from an older .pt file."
        )
    for name in ("base_points", "depth", "camera_id", "time_idx", "observation_valid"):
        if name not in payload:
            raise ValueError(f"Prediction is missing required field {name!r}")
    points = _array(payload["base_points"], "base_points")
    depth = _array(payload["depth"], "depth")
    camera_ids = _array(payload["camera_id"], "camera_id")
    times = _array(payload["time_idx"], "time_idx")
    valid = _array(payload["observation_valid"], "observation_valid")
    if points.ndim != 5 or points.shape[0] != 1 or points.shape[-1] != 3:
        raise ValueError("base_points must have shape [1, S, H, W, 3]")
    slots, padded_height, padded_width = points.shape[1:4]
    if depth.shape != (1, slots, padded_height, padded_width):
        raise ValueError("depth must have shape [1, S, H, W] matching base_points")
    if any(array.shape != (1, slots) for array in (camera_ids, times, valid)):
        raise ValueError("camera_id, time_idx, observation_valid must have shape [1, S]")
    if not all(np.issubdtype(array.dtype, np.integer) for array in (camera_ids, times)):
        raise ValueError("camera_id and time_idx must contain integer labels")
    if valid.dtype != np.dtype(bool):
        raise ValueError("observation_valid must be a boolean tensor")
    if len(metadata) != slots:
        raise ValueError("observation_metadata must have one entry for every observation slot")
    observations = []
    seen = set()
    for slot, record in enumerate(metadata):
        if not isinstance(record, dict):
            raise ValueError(f"observation_metadata[{slot}] must be an object")
        camera_id, time_idx, is_valid = (
            int(camera_ids[0, slot]), int(times[0, slot]), bool(valid[0, slot])
        )
        if (record.get("camera_id"), record.get("time_idx"), record.get("valid")) != (
            camera_id, time_idx, is_valid
        ):
            raise ValueError(f"observation_metadata[{slot}] does not match tensor camera/time/valid")
        if not is_valid:
            continue
        if (camera_id, time_idx) in seen:
            raise ValueError(f"Duplicate valid camera/time observation: {(camera_id, time_idx)}")
        seen.add((camera_id, time_idx))
        native_hw = record.get("native_hw")
        if (not isinstance(native_hw, (list, tuple)) or len(native_hw) != 2
                or any(type(dimension) is not int or dimension <= 0 for dimension in native_hw)):
            raise ValueError(f"observation_metadata[{slot}].native_hw must contain positive H, W")
        height, width = native_hw
        if height > padded_height or width > padded_width:
            raise ValueError(f"observation_metadata[{slot}].native_hw exceeds predicted dimensions")
        image_value = record.get("image")
        if not isinstance(image_value, str) or not Path(image_value).is_absolute():
            raise ValueError(f"observation_metadata[{slot}].image must be an absolute input image path")
        image_path = Path(image_value)
        with Image.open(image_path) as image:
            image_width, image_height = image.size
            if not (width <= image_width <= padded_width and height <= image_height <= padded_height):
                raise ValueError(
                    f"Input image size {image.size} cannot contain native_hw {(height, width)} "
                    f"inside predicted dimensions {(padded_height, padded_width)} for observation {slot}; "
                    "use the exact transformed RGB input recorded by inference"
                )
            # Prepared manifests may store bottom/right patch padding in their
            # PNGs. The native ROI always starts at (0, 0), without resizing.
            rgb = np.asarray(image.convert("RGB"))[:height, :width].copy()
        xyz = points[0, slot, :height, :width]
        native_depth = depth[0, slot, :height, :width]
        finite = np.isfinite(xyz).all(axis=-1) & np.isfinite(native_depth) & (native_depth > 0)
        observations.append(dict(
            slot=slot, camera_id=camera_id, time_idx=time_idx, native_hw=[height, width],
            image=str(image_path), xyz=xyz, rgb=rgb, depth=native_depth, finite=finite,
        ))
    if not observations:
        raise ValueError("Prediction contains no valid observations to export")
    return observations


def export_prediction(payload, output, max_points=None, depth_percentile=None):
    """Write one shared-world PLY per observed time, returning an audit record.

    By default, export every valid native pixel without a sampling limit.
    Explicit max_points enables a bounded native-image grid for quick previews.
    Filtering applies only to display/export. With an optional percentile, each
    camera uses one predicted-depth cutoff computed from all of its native
    finite positive-depth pixels over the whole clip. Tensors remain unchanged.
    """
    if max_points is not None and (type(max_points) is not int or max_points <= 0):
        raise ValueError("max_points must be None (dense) or a positive integer")
    if depth_percentile is not None and not (0 < depth_percentile <= 100):
        raise ValueError("depth_percentile must be in (0, 100]")
    observations = _observations(payload)
    times = sorted({obs["time_idx"] for obs in observations})
    cameras = sorted({obs["camera_id"] for obs in observations})
    largest_camera_count = max(sum(obs["time_idx"] == time for obs in observations) for time in times)
    if max_points is not None and max_points < largest_camera_count:
        raise ValueError("max_points must allow at least one sample per valid camera at each time")
    cutoffs = {}
    if depth_percentile is not None:
        for camera_id in cameras:
            depths = np.concatenate([
                obs["depth"][obs["finite"]] for obs in observations if obs["camera_id"] == camera_id
            ])
            cutoffs[camera_id] = float(np.percentile(depths, depth_percentile)) if len(depths) else None
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    for time_idx in times:
        frame_observations = sorted(
            (obs for obs in observations if obs["time_idx"] == time_idx),
            key=lambda obs: obs["camera_id"],
        )
        full_resolution = max_points is None or sum(
            obs["native_hw"][0] * obs["native_hw"][1]
            for obs in frame_observations) <= max_points
        budget, remainder = divmod(max_points, len(frame_observations)) if max_points is not None else (0, 0)
        points, colors, frame_cameras = [], [], []
        for index, obs in enumerate(frame_observations):
            height, width = obs["native_hw"]
            keep = obs["finite"].copy()
            cutoff = cutoffs.get(obs["camera_id"])
            if cutoff is not None:
                keep &= obs["depth"] <= cutoff
            if full_resolution:
                # Dense native indexing avoids allocating full y/x grids.
                points.append(obs["xyz"][keep])
                colors.append(obs["rgb"][keep])
                sampled_pixels, exported_points = height * width, int(keep.sum())
            else:
                y, x = _grid(height, width, budget + (index < remainder))
                sampled_keep = keep[y, x]
                points.append(obs["xyz"][y[sampled_keep], x[sampled_keep]])
                colors.append(obs["rgb"][y[sampled_keep], x[sampled_keep]])
                sampled_pixels, exported_points = len(y), int(sampled_keep.sum())
            frame_cameras.append(dict(
                slot=obs["slot"], camera_id=obs["camera_id"], time_idx=time_idx,
                native_hw=obs["native_hw"], image=obs["image"],
                native_pixels=height * width, finite_positive_depth_pixels=int(obs["finite"].sum()),
                depth_tail_removed=int(obs["finite"].sum() - keep.sum()),
                sampled_grid_pixels=sampled_pixels, exported_points=exported_points,
            ))
        xyz, rgb = np.concatenate(points), np.concatenate(colors)
        name = f"frame_{time_idx:06d}.ply"
        _write_ply(output / name, xyz, rgb)
        frames.append(dict(
            time_idx=time_idx, file=name, point_count=len(xyz), cameras=frame_cameras,
        ))
    return dict(
        clip_id=payload.get("clip_id"), geometry="instantaneous base_points; not trajectories",
        scale_note=SCALE_NOTE, max_points_per_frame=max_points,
        sampling="all_native_pixels" if max_points is None else "bounded_native_grid",
        filtering=dict(
            ground_truth_used=False, default="finite XYZ and finite positive predicted depth",
            depth_percentile=depth_percentile,
            percentile_scope="one cutoff per camera across all native observations in this clip",
            camera_cutoffs=[dict(camera_id=camera, predicted_depth_cutoff=cutoffs.get(camera))
                            for camera in cameras],
        ), frames=frames,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Inference output directory with clip_*.pt files")
    parser.add_argument("--output", required=True, help="Output directory for per-clip PLY files and export.json")
    parser.add_argument("--max-points", type=int,
                        help="Optional sampling cap per observed time; default exports all valid native pixels")
    parser.add_argument("--depth-percentile", type=float, metavar="P", help="Optional predicted-depth tail filter, e.g. 98")
    args = parser.parse_args(argv)
    if args.max_points is not None and args.max_points <= 0:
        parser.error("--max-points must be positive")
    if args.depth_percentile is not None and not (0 < args.depth_percentile <= 100):
        parser.error("--depth-percentile must be in (0, 100]")
    predictions = Path(args.predictions).resolve()
    files = sorted(predictions.glob("clip_*.pt")) if predictions.is_dir() else []
    if not files:
        parser.error(f"No clip_*.pt predictions found in {predictions}")
    output = Path(args.output).resolve()
    clips = []
    for path in files:
        if not re.fullmatch(r"clip_\d+\.pt", path.name):
            parser.error(f"Expected a numeric clip file name, got {path.name!r}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            parser.error(f"Prediction {path} must contain a dictionary")
        try:
            record = export_prediction(payload, output / path.stem, args.max_points, args.depth_percentile)
        except (ValueError, OSError) as error:
            parser.error(f"{path.name}: {error}")
        record.update(prediction=str(path), directory=path.stem)
        clips.append(record)
    report = dict(predictions=str(predictions), scale_note=SCALE_NOTE, clips=clips)
    (output / "export.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"export": str(output / "export.json"), "clips": len(clips),
                      "frames": sum(len(clip["frames"]) for clip in clips)}))
    return output


if __name__ == "__main__":
    main()
