"""Convert official MV-Kubric scenes once, without seeking compressed tarballs.

Archives are read in streaming mode and only one scene is staged at a time.
The renderer depth is Euclidean ray distance. H5 calibration/track UV retain
the renderer's .5 raster centers; the maintained adapter shifts them once to
integer array coordinates. Object segmentation is remapped by persistent UID,
never by each camera's visibility-sorted instance position.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import time
import uuid

import h5py
import numpy as np
from PIL import Image

from multicamera4rc.datasets.adapters.mv_kubric import SCHEMA_VERSION, official_scene_split


SOURCE_CODE = {
    "mvtracker": "https://github.com/ethz-vlg/mvtracker/tree/ceea8ad2af77ed9b44148ef8e9eeba4ea3c3f072",
    "generator": "https://github.com/ethz-vlg/kubric/tree/abb16d7c387541d5c45d5efbc8aae254e33d5771",
    "dataset": "https://huggingface.co/datasets/ethz-vlg/mv3dpt-datasets",
}


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_json(path, value):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _free_space(path, minimum_gib, required_bytes=0):
    free = shutil.disk_usage(path).free
    floor = int(float(minimum_gib) * 2**30)
    if free - int(required_bytes) < floor:
        raise OSError(f"insufficient free space: {free / 2**30:.2f} GiB, "
                      f"need to preserve {minimum_gib} GiB plus {required_bytes} bytes")


def quaternion_wxyz_matrix(quaternion):
    q = np.asarray(quaternion, np.float64)
    if q.shape[-1] != 4 or not np.isfinite(q).all():
        raise ValueError("renderer quaternions must be finite WXYZ")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if (norm < 1e-8).any() or not np.allclose(norm, 1, atol=1e-4):
        raise ValueError("renderer supplied a non-unit quaternion")
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


class SceneReader:
    """Read and hash exactly the source files consumed by conversion."""

    def __init__(self, root):
        self.root = Path(root)
        self.files = {}

    def bytes(self, relative):
        relative = str(relative)
        data = (self.root / relative).read_bytes()
        self.files[relative] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return data

    def json(self, relative):
        return json.loads(self.bytes(relative))

    def npz(self, relative, key):
        with np.load(io.BytesIO(self.bytes(relative)), allow_pickle=False) as archive:
            return archive[key].copy()

    def image(self, relative):
        with Image.open(io.BytesIO(self.bytes(relative))) as image:
            return np.asarray(image).copy()


def _camera(metadata, frames, height, width):
    c = metadata["camera"]
    raw_k = np.asarray(c["K"], np.float64)
    if raw_k.shape != (3, 3):
        raise ValueError("camera.K must be a normalized Blender pinhole matrix")
    flip3 = np.diag([1., -1., -1.])
    intrinsic = np.diag([width, height, 1.]) @ raw_k @ flip3
    if (not np.isfinite(intrinsic).all() or np.any(intrinsic[[0, 1], [0, 1]] <= 0)
            or not np.allclose(intrinsic[2], [0, 0, 1])
            or not np.allclose(intrinsic[[0, 1], [1, 0]], 0)):
        raise ValueError("camera.K does not convert to OpenCV zero-skew pinhole")
    positions = np.asarray(c["positions"], np.float64)
    rotations = quaternion_wxyz_matrix(c["quaternions"])
    if positions.shape != (frames, 3) or rotations.shape != (frames, 3, 3):
        raise ValueError("camera pose timeline differs from the rendered frames")
    if not np.isfinite(positions).all():
        raise ValueError("non-finite camera positions")
    # The released fixed_random camera contract is static. Reject moving
    # variants rather than flattening a time-varying pose into [C,4,4].
    if (not np.allclose(positions, positions[:1], atol=1e-6, rtol=0)
            or not np.allclose(rotations, rotations[:1], atol=1e-6, rtol=0)):
        raise ValueError("time-varying cameras require a different packed schema")
    blender = np.eye(4)
    blender[:3, :3] = rotations[0]
    blender[:3, 3] = positions[0]
    if "R" in c and not np.allclose(blender, c["R"], atol=2e-6, rtol=0):
        raise ValueError("camera.R disagrees with authoritative quaternion/position")
    return intrinsic, blender @ np.diag([1., -1., -1., 1.])


def _pose(instance, frames):
    position = np.asarray(instance["positions"], np.float64)
    quaternion = np.asarray(instance["quaternions"], np.float64)
    if position.shape != (frames, 3) or quaternion.shape != (frames, 4):
        raise ValueError("object pose timeline differs from rendered frames")
    rotation = quaternion_wxyz_matrix(quaternion)
    if not np.isfinite(position).all() or not isinstance(instance["is_dynamic"], bool):
        raise ValueError("invalid object pose or dynamic flag")
    # WXYZ -> XYZW. Fixed asset scale cancels in the rigid relative motion.
    pose = np.concatenate((position, quaternion[:, [1, 2, 3, 0]]), -1)
    return pose, rotation, instance["is_dynamic"]


def canonical_objects(reader, camera_names, frames):
    tracked = reader.json("tracked_objects.json")
    ids, names = tracked["ids"], tracked["names"]
    if (len(ids) != len(names) or len(set(ids)) != len(ids)
            or len(set(names)) != len(names) or "dome" not in names):
        raise ValueError("invalid persistent object identity table")
    name_ids = {name: int(raw) for name, raw in zip(names, ids)}
    object_names = [name for name in names if name != "dome"]
    canonical = {name: i + 1 for i, name in enumerate(object_names)}
    poses, rotations, dynamic, view_tables, view_metadata = {}, {}, {}, {}, {}
    raw_tables = []
    for camera in camera_names:
        metadata = reader.json(f"{camera}/metadata.json")
        table = reader.json(f"{camera}/object_id_to_segmentation_id.json")
        raw_tables.append(table)
        instances = metadata["instances"]
        # A positive segmentation number addresses the current view's own
        # visibility-sorted instances, not root tracked_objects.json.
        lut = np.full(len(instances) + 1, -1, np.int64)
        lut[0] = 0
        if int(table.get("dome", -1)) != 0:
            raise ValueError("the static dome must use background segmentation 0")
        for name, local in table.items():
            if not isinstance(local, int) or isinstance(local, bool) or not 0 <= local <= len(instances):
                raise ValueError("invalid per-view segmentation mapping")
            if local == 0:
                continue
            if name not in canonical or lut[local] != -1:
                raise ValueError("untracked or duplicate positive instance identity")
            lut[local] = canonical[name]
            pose, rotation, is_dynamic = _pose(instances[local - 1], frames)
            if name in poses:
                if (not np.allclose(pose[:, :3], poses[name][:, :3], atol=2e-6, rtol=0)
                        or not np.allclose(rotation, rotations[name], atol=2e-6, rtol=0)
                        or is_dynamic != dynamic[name]):
                    raise ValueError(f"object {name} pose disagrees across physical cameras")
            else:
                poses[name], rotations[name], dynamic[name] = pose, rotation, is_dynamic
        if (lut < 0).any():
            raise ValueError(f"{camera} has an unmapped positive segmentation label")
        view_tables[camera], view_metadata[camera] = lut, metadata
    missing = [name for name in object_names if name not in poses]
    if missing:
        # Kubric can retain a never-rendered asset in the scene identity table.
        # It has no pose export when every view maps it to background. Remove
        # only identities proven absent from ALL view labels and invisible in
        # ALL native point/frame/view observations. The author samples native
        # mesh tracks before visibility filtering, so hidden tracks can exist.
        # Keep those raw tracks for auditing; never invent a static pose.
        native_ids = reader.npz("tracks_segmentation_ids.npz", "tracks_segmentation_ids")
        unsafe=[]
        for name in missing:
            selected=native_ids == name_ids[name]
            if any(table.get(name) != 0 for table in raw_tables):
                unsafe.append(name);continue
            for camera in camera_names:
                occluded=reader.npz(f"{camera}/tracks_2d.npz", "occlusion")
                if (occluded.shape != (frames,len(native_ids))
                        or not np.isin(occluded, [0,1]).all()
                        or not occluded[:,selected].astype(bool).all()):
                    unsafe.append(name);break
        if unsafe:
            raise ValueError(f"objects lack authoritative poses but may be observed: {unsafe}")
        retained = [name for name in object_names if name not in missing]
        remap = np.zeros(len(object_names) + 1, np.int64)
        for new_id, name in enumerate(retained, 1):
            remap[canonical[name]] = new_id
        view_tables = {camera: remap[lut] for camera, lut in view_tables.items()}
        object_names = retained
        if not object_names:
            raise ValueError("scene has no observed authoritative object poses")
    return dict(names=object_names, raw_ids=[name_ids[name] for name in object_names],
                poses=np.stack([poses[name] for name in object_names]),
                rotations=np.stack([rotations[name] for name in object_names]),
                dynamic=np.asarray([dynamic[name] for name in object_names]),
                view_tables=view_tables, metadata=view_metadata, raw_name_ids=name_ids,
                omitted_unobserved_objects=missing,
                unobserved_raw_ids=[name_ids[name] for name in missing])


def audit_native(reader, objects, camera_names, intrinsics, extrinsics, frames):
    """Use independently exported native tracks as a conversion-time oracle."""
    xyz = reader.npz("tracks_3d.npz", "tracks_3d")
    raw_ids = reader.npz("tracks_segmentation_ids.npz", "tracks_segmentation_ids")
    if (xyz.ndim != 3 or xyz.shape[0] != frames or xyz.shape[-1] != 3
            or raw_ids.shape != (xyz.shape[1],) or not np.isfinite(xyz).all()):
        raise ValueError("invalid official native 3D track pool")
    ids = [objects["raw_name_ids"]["dome"], *objects["raw_ids"]]
    unobserved=np.isin(raw_ids,objects['unobserved_raw_ids'])
    if not np.isin(raw_ids, ids+objects['unobserved_raw_ids']).all():
        raise ValueError("native tracks address an unknown persistent object")
    rigid_error, rigid_sum, rigid_count = 0., 0., 0
    for index, raw in enumerate(ids):
        selected = raw_ids == raw
        truth = xyz[:, selected]
        if not selected.any():
            continue
        if index == 0:
            predicted = np.broadcast_to(truth[:1], truth.shape)
        else:
            position = objects["poses"][index - 1, :, :3]
            rotation = objects["rotations"][index - 1]
            local = (truth[0] - position[0]) @ rotation[0]
            predicted = np.einsum("tij,nj->tni", rotation, local) + position[:, None]
        errors = np.linalg.norm(predicted - truth, axis=-1)
        rigid_error = max(rigid_error, float(errors.max()))
        rigid_sum += float(errors.sum()); rigid_count += errors.size
    if rigid_error > 2e-4:
        raise ValueError(f"renderer pose/native trajectory mismatch: {rigid_error:.6g} metres")
    uv_all, occlusion_all, reprojection_max = [], [], 0.
    for index, camera in enumerate(camera_names):
        uv = reader.npz(f"{camera}/tracks_2d.npz", "tracks_2d")
        occlusion = reader.npz(f"{camera}/tracks_2d.npz", "occlusion")
        if uv.shape != (*xyz.shape[:2], 2) or occlusion.shape != xyz.shape[:2]:
            raise ValueError("native 2D tracks/occlusion differ from persistent track pool")
        if not occlusion[:,unobserved].astype(bool).all():
            raise ValueError('pose-less native object has a visible track')
        inverse = np.linalg.inv(extrinsics[index])
        camera_xyz = xyz @ inverse[:3, :3].T + inverse[:3, 3]
        projected = camera_xyz @ intrinsics[index].T
        valid = (camera_xyz[..., 2] > 1e-4) & np.isfinite(uv).all(-1) & ~occlusion.astype(bool)
        if not valid.any():
            raise ValueError(f"{camera} has no visible positive-depth native track")
        projected = projected[..., :2] / np.where(valid, camera_xyz[..., 2], 1)[..., None]
        error = np.linalg.norm(projected[valid] - uv[valid], axis=-1)
        reprojection_max = max(reprojection_max, float(error.max()))
        uv_all.append(uv); occlusion_all.append(occlusion.astype(bool))
    if reprojection_max > .05:
        raise ValueError(f"OpenCV calibration/native UV mismatch: {reprojection_max:.6g} pixels")
    return dict(tracks_3d=xyz, tracks_raw_scene_ids=raw_ids,
                tracks_2d=np.stack(uv_all, 1), occlusion=np.stack(occlusion_all, 1)), {
                    "native_points": xyz.shape[1], "rigid_max_error_m": rigid_error,
                    "unobserved_native_points": int(unobserved.sum()),
                    "rigid_audited_native_points": int((~unobserved).sum()),
                    "rigid_mean_error_m": rigid_sum / max(1, rigid_count),
                    "reprojection_max_error_pixels": reprojection_max}


def published_pack(path):
    """A final suffix alone does not make an interrupted conversion usable."""
    try:
        with h5py.File(path, "r") as handle:
            official_scene_split(handle.attrs["source_split"], handle.attrs["source_scene_id"])
            return (handle.attrs.get("schema_version") == SCHEMA_VERSION
                    and bool(handle.attrs.get("conversion_complete", False))
                    and {"frames", "depth", "segmentation", "intrinsics", "extrinsics",
                         "object_trajectories", "object_is_dynamic"}.issubset(handle)
                    and handle["frames"].shape[:2] == (24, 10))
    except (OSError, KeyError, ValueError, TypeError):
        return False


def convert_scene(scene, output_root, *, source_split=None, scene_id=None,
                  compression="lzf", keep_native_tracks=True, min_free_gib=20,
                  source_info=None):
    scene, output_root = Path(scene), Path(output_root)
    source_split = scene.parent.name if source_split is None else source_split
    scene_id = int(scene.name) if scene_id is None else int(scene_id)
    official_scene_split(source_split, scene_id)
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / source_split / f"{source_split}_{scene_id:06d}.h5"
    target.parent.mkdir(parents=True, exist_ok=True)
    if published_pack(target):
        return {"status": "already_complete", "scene_id": scene_id, "source_split": source_split,
                "path": str(target), "packed_bytes": target.stat().st_size}
    if target.exists():
        raise FileExistsError(f"refusing to replace an incomplete final pack: {target}")
    _free_space(output_root, min_free_gib)
    started = time.monotonic()
    reader = SceneReader(scene)
    camera_names = sorted([p.name for p in scene.glob("view_*") if p.is_dir()], key=lambda n: int(n.split("_")[-1]))
    if camera_names != [f"view_{i}" for i in range(10)]:
        raise ValueError("official scene must contain exactly view_0..view_9")
    # Validate all identities, poses and native oracle before allocating the
    # final-size arrays; a failed scene never receives a public .h5 filename.
    metadata = reader.json("view_0/metadata.json")
    info = metadata["metadata"]
    frames = int(info["num_frames"])
    height, width = map(int, info["resolution"])
    if frames != 24 or min(height, width) < 1:
        raise ValueError("official scene must contain 24 frames at its native resolution")
    fps = float(info["frame_rate"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("invalid render frame rate")
    objects = canonical_objects(reader, camera_names, frames)
    intrinsics, extrinsics = [], []
    for camera in camera_names:
        meta = objects["metadata"][camera]
        other = meta["metadata"]
        if (int(other["num_frames"]) != frames or list(other["resolution"]) != [height, width]
                or float(other["frame_rate"]) != fps):
            raise ValueError("camera image/time metadata disagree")
        K, E = _camera(meta, frames, height, width)
        intrinsics.append(K); extrinsics.append(E)
    intrinsics, extrinsics = np.stack(intrinsics), np.stack(extrinsics)
    native, audit = audit_native(reader, objects, camera_names, intrinsics, extrinsics, frames)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    filters = {} if compression == "none" else {"compression": compression}
    invalid_depth_pixels = 0
    try:
        with h5py.File(temporary, "w") as handle:
            shape = (frames, len(camera_names), height, width)
            image_ds = handle.create_dataset("frames", shape=(*shape, 3), dtype="u1",
                chunks=(1, 1, min(height, 128), width, 3), **filters)
            depth_ds = handle.create_dataset("depth", shape=shape, dtype="f4",
                chunks=(1, 1, min(height, 128), width), **filters)
            segmentation_ds = handle.create_dataset("segmentation", shape=shape, dtype="u2",
                chunks=(1, 1, min(height, 128), width), **filters)
            for camera_index, camera in enumerate(camera_names):
                lut = objects["view_tables"][camera]
                for frame in range(frames):
                    rgba = reader.image(f"{camera}/rgba_{frame:05d}.png")
                    depth = reader.image(f"{camera}/depth_{frame:05d}.tiff")
                    segmentation = reader.image(f"{camera}/segmentation_{frame:05d}.png")
                    if (rgba.shape not in ((height, width, 3), (height, width, 4))
                            or rgba.dtype != np.uint8 or depth.shape != (height, width)
                            or depth.dtype.kind != "f" or segmentation.shape != (height, width)
                            or segmentation.dtype.kind not in "ui"
                            or segmentation.min() < 0 or segmentation.max() >= len(lut)):
                        raise ValueError(f"invalid renderer RGB/depth/instance array: {camera}/{frame}")
                    good = np.isfinite(depth) & (depth > 0)
                    if not good.any():
                        raise ValueError(f"render has no valid depth: {camera}/{frame}")
                    invalid_depth_pixels += int((~good).sum())
                    _free_space(output_root, min_free_gib, rgba.nbytes + depth.nbytes + segmentation.nbytes)
                    image_ds[frame, camera_index] = rgba[..., :3]
                    depth_ds[frame, camera_index] = np.where(good, depth, 0)
                    segmentation_ds[frame, camera_index] = lut[segmentation]
            handle.create_dataset("intrinsics", data=intrinsics.astype(np.float32))
            handle.create_dataset("extrinsics", data=extrinsics.astype(np.float32))
            handle.create_dataset("camera_ids", data=np.asarray(camera_names, dtype="S16"))
            handle.create_dataset("object_trajectories", data=objects["poses"].astype(np.float32))
            handle.create_dataset("object_is_dynamic", data=objects["dynamic"].astype(bool))
            handle.create_dataset("object_names", data=np.asarray(objects["names"], dtype="S128"))
            handle.create_dataset("object_raw_scene_ids", data=np.asarray(objects["raw_ids"], np.int64))
            handle.create_dataset("unobserved_object_raw_scene_ids", data=np.asarray(objects['unobserved_raw_ids'],np.int64))
            if keep_native_tracks:
                group = handle.create_group("audit_native")
                for key, value in native.items():
                    group.create_dataset(key, data=value, **filters)
                group.attrs["purpose"] = "independent official native-track oracle; not densified for training"
            handle.attrs.update(schema_version=SCHEMA_VERSION, source_split=source_split,
                source_scene_id=scene_id, coordinate_system="opencv_camera__blender_world",
                depth_representation="ray_distance", pixel_coordinate_convention="raster_center_0.5",
                sampling_pixel_center_offset=.5, fps=fps, ground_plane_world_z=0.,
                depth_absolute_tolerance_m=.03, depth_relative_tolerance=.01,
                visibility_search_radius_px=1, object_quaternion_order="xyzw",
                background_policy="objects_ground", source_code=json.dumps(SOURCE_CODE, sort_keys=True),
                native_track_audit=json.dumps(audit, sort_keys=True),
                source_info=json.dumps(source_info or {}, sort_keys=True))
            handle.attrs["omitted_unobserved_objects"] = json.dumps(objects["omitted_unobserved_objects"])
            handle.attrs['background_raw_scene_id'] = objects['raw_name_ids']['dome']
            source_digest = hashlib.sha256(json.dumps(reader.files, sort_keys=True).encode()).hexdigest()
            handle.attrs["source_scene_consumed_files_sha256"] = source_digest
            handle.attrs["conversion_complete"] = True
            handle.flush()
        with temporary.open("rb") as file:
            os.fsync(file.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    result = {"status": "complete", "path": str(target), "source_split": source_split,
        "scene_id": scene_id, "schema_version": SCHEMA_VERSION, "cameras": len(camera_names),
        "frames": frames, "image_hw": [height, width], "objects": len(objects["names"]),
        "dynamic_objects": int(objects["dynamic"].sum()), "packed_bytes": target.stat().st_size,
        "consumed_source_bytes": sum(v["bytes"] for v in reader.files.values()),
        "consumed_source_files": len(reader.files), "invalid_depth_pixels": invalid_depth_pixels,
        "source_scene_consumed_files_sha256": source_digest, "native_track_audit": audit,
        "source_info": source_info or {}, "elapsed_seconds": time.monotonic() - started,
        "omitted_unobserved_objects": objects['omitted_unobserved_objects']}
    _atomic_json(target.with_suffix(".json"), result)
    return result


def archive_scene_member(member_name):
    path = PurePosixPath(member_name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive path: {member_name}")
    parts = path.parts
    if "kubric-multiview" not in parts:
        return None
    start = parts.index("kubric-multiview")
    if len(parts) < start + 4:
        return None
    source_split, scene = parts[start + 1:start + 3]
    if not scene.isdecimal():
        return None
    scene_id = int(scene)
    official_scene_split(source_split, scene_id)
    return (source_split, scene_id), Path(*parts[start + 3:])


@contextmanager
def archive_lease(output_root, archive):
    """Do not allow two jobs to replay the same archive simultaneously."""
    import fcntl
    key = hashlib.sha256(str(Path(archive).resolve()).encode()).hexdigest()[:16]
    path = Path(output_root) / f"archive_{key}.lock"
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"archive already has a running converter: {archive}") from error
        yield key


def convert_archive(archive, output_root, *, max_scenes=1, staging_root=None,
                    compression="lzf", keep_native_tracks=True, min_free_gib=20):
    """One gzip pass, one raw scene on disk. Completed packs are skipped.

    Replaying a partially converted gzip must decompress its earlier bytes;
    this cost is explicitly recorded. Parallelism is across distinct archives,
    never repeated workers seeking the same gzip. Scene interleaving is an
    error because it would violate bounded staging and atomic completeness.
    """
    archive, output_root = Path(archive), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stage_parent = Path(staging_root) if staging_root else output_root / ".staging"
    stage_parent.mkdir(parents=True, exist_ok=True)
    _free_space(stage_parent, min_free_gib)
    results, closed, current, stage, skipping = [], set(), None, None, False
    preserve_failed_scene = False
    stat = archive.stat()
    source_info = {"archive": str(archive.resolve()), "archive_bytes": stat.st_size,
        "archive_mtime_ns": stat.st_mtime_ns, "archive_sha256_verified_in_this_run": False,
        "archive_checksum_note": "No full-archive checksum is claimed by conversion; source download verification is separate.",
        "read_mode": "single sequential gzip replay; completed scene H5 files skipped"}
    with archive_lease(output_root, archive) as key:
        progress = output_root / f"archive_{key}.progress.json"
        log = output_root / f"archive_{key}.scenes.jsonl"
        begun = time.monotonic()
        def finish():
            if current is None:
                return
            split, scene_id = current
            if skipping:
                result = {"status": "already_complete", "source_split": split, "scene_id": scene_id}
            else:
                result = convert_scene(stage, output_root, source_split=split, scene_id=scene_id,
                    compression=compression, keep_native_tracks=keep_native_tracks,
                    min_free_gib=min_free_gib, source_info=source_info)
            results.append(result)
            _append_json(log, result)
            print(json.dumps(result, sort_keys=True), flush=True)
            if stage is not None:
                shutil.rmtree(stage)
            closed.add(current)
        try:
            with tarfile.open(archive, mode="r|gz") as tar:
                for member in tar:
                    # Even r|gz retains TarInfo objects by default. Millions
                    # of frames must not accumulate an archive-sized RAM index.
                    tar.members.clear()
                    parsed = archive_scene_member(member.name)
                    if parsed is None or member.isdir():
                        continue
                    if not member.isfile():
                        raise ValueError(f"archive contains a link or special entry: {member.name}")
                    identity, relative = parsed
                    if identity != current:
                        finish()
                        if max_scenes and sum(r["status"] == "complete" for r in results) >= max_scenes:
                            break
                        if identity in closed:
                            raise ValueError(f"archive scenes are interleaved: {identity}")
                        current = identity
                        split, scene_id = identity
                        final = output_root / split / f"{split}_{scene_id:06d}.h5"
                        skipping = published_pack(final)
                        stage = None if skipping else Path(tempfile.mkdtemp(prefix=f"{split}_{scene_id}_", dir=stage_parent))
                        _atomic_json(progress, dict(status="reading_scene", current_scene=list(identity),
                            scenes_finished=len(results), source_info=source_info,
                            elapsed_seconds=time.monotonic()-begun,
                            resume_note="gzip replays earlier bytes; completed H5 scenes are not reconverted"))
                    if skipping:
                        continue
                    # Diagnostic videos and redundant camera summaries are not
                    # needed to establish authoritative training geometry.
                    if relative.suffix in {".mp4", ".blend"}:
                        continue
                    _free_space(stage_parent, min_free_gib, member.size)
                    destination = stage / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        raise ValueError(f"duplicate archive member: {member.name}")
                    with tar.extractfile(member) as source, destination.open("wb") as sink:
                        shutil.copyfileobj(source, sink, length=2**20)
                else:
                    finish()
            _atomic_json(progress, dict(status="bounded_run_complete" if max_scenes else "archive_complete",
                scenes_finished=len(results), source_info=source_info,
                elapsed_seconds=time.monotonic()-begun, packed_bytes=sum(x.get("packed_bytes", 0) for x in results)))
        except Exception as error:
            preserve_failed_scene = stage is not None and stage.exists()
            _atomic_json(progress, dict(status="failed", current_scene=current, error=repr(error),
                scenes_finished=len(results), source_info=source_info,
                failed_scene_path=str(stage) if preserve_failed_scene else None,
                elapsed_seconds=time.monotonic()-begun))
            raise
        finally:
            if stage is not None and stage.exists() and not preserve_failed_scene:
                shutil.rmtree(stage)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scene", help="one extracted official train/ID or test/ID directory")
    source.add_argument("--archive", help="one official scene-contiguous .tar.gz; run one job per archive")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--staging-root")
    parser.add_argument("--max-scenes", type=int, default=1,
                        help="new scenes to convert, default 1 for audit; 0 for full archive; completed packs do not count")
    parser.add_argument("--min-free-gib", type=float, default=20.)
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--omit-native-tracks", action="store_true",
                        help="audit native tracks before conversion but omit their duplicate audit copy")
    args = parser.parse_args(argv)
    if args.max_scenes < 0 or args.min_free_gib < 0:
        parser.error("max-scenes and min-free-gib must be nonnegative")
    options = dict(compression=args.compression, keep_native_tracks=not args.omit_native_tracks,
                   min_free_gib=args.min_free_gib)
    if args.scene:
        result = convert_scene(args.scene, args.output_root, **options)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    else:
        convert_archive(args.archive, args.output_root, max_scenes=args.max_scenes,
                        staging_root=args.staging_root, **options)


if __name__ == "__main__":
    main()
