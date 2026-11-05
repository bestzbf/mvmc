"""Prepare all five official MV-Kubric training archives on at most two CPU jobs.

The complete-data manifest is a publication gate, not a scene-count guess.
It is published only after every archive child succeeds and all 4,970 H5
scenes pass independent metadata/shape validation. An interrupted run can be
restarted: compressed archives replay sequentially, while valid atomic H5
scenes are skipped by the scene converter.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import h5py
import numpy as np

from multicamera4rc.datasets.adapters.mv_kubric import SCHEMA_VERSION, official_scene_split
from scripts.prepare_mv_kubric import _atomic_json, _free_space


ARCHIVES = (
    "kubric-multiview--train.full.0031-1000.tar.gz",
    "kubric-multiview--train.full.1001-2000.tar.gz",
    "kubric-multiview--train.full.2001-3000.tar.gz",
    "kubric-multiview--train.full.3001-4000.tar.gz",
    "kubric-multiview--train.full.4001-5000.tar.gz",
)
EXPECTED_IDS = list(range(31, 5001))


def _now():
    return datetime.now(timezone.utc).isoformat()


def archive_inventory(root):
    """Use only the immutable full-train whitelist, never test or preview data."""
    root = Path(root).resolve()
    result = []
    for name in ARCHIVES:
        path = root / name
        if not path.is_file() or path.stat().st_size < 10:
            raise FileNotFoundError(f"missing complete official training archive: {path}")
        with path.open("rb") as handle:
            if handle.read(2) != b"\x1f\x8b":
                raise ValueError(f"not a gzip archive: {path}")
        stat = path.stat()
        result.append(dict(name=name, path=str(path), bytes=stat.st_size,
                           mtime_ns=stat.st_mtime_ns,
                           checksum_verified_in_this_run=False))
    return result


def inspect_pack(path):
    """Validate each published H5, without decompressing every image twice."""
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if (handle.attrs.get("schema_version") != SCHEMA_VERSION
                or not bool(handle.attrs.get("conversion_complete", False))):
            raise ValueError(f"incomplete or foreign-schema final H5: {path}")
        source_split = str(handle.attrs["source_split"])
        scene_id = int(handle.attrs["source_scene_id"])
        official_scene_split(source_split, scene_id)
        if source_split != "train":
            raise ValueError(f"official test scene must not enter full training: {path}")
        image_shape = (24, 10, 512, 512)
        expected = {"frames": (*image_shape, 3), "depth": image_shape,
                    "segmentation": image_shape, "intrinsics": (10, 3, 3),
                    "extrinsics": (10, 4, 4), "camera_ids": (10,)}
        for key, shape in expected.items():
            if key not in handle or handle[key].shape != shape:
                raise ValueError(f"malformed official H5 field {key}: {path}")
        if (handle["frames"].dtype != np.uint8 or handle["depth"].dtype != np.float32
                or handle["segmentation"].dtype.kind not in "ui"):
            raise ValueError(f"invalid official RGB/depth dtype: {path}")
        objects = len(handle["object_is_dynamic"])
        if (objects < 1 or handle["object_trajectories"].shape != (objects, 24, 7)
                or handle["object_names"].shape != (objects,)
                or handle["object_raw_scene_ids"].shape != (objects,)):
            raise ValueError(f"invalid canonical object tables: {path}")
        if (handle.attrs.get("coordinate_system") != "opencv_camera__blender_world"
                or handle.attrs.get("depth_representation") != "ray_distance"
                or float(handle.attrs.get("sampling_pixel_center_offset", -1)) != .5):
            raise ValueError(f"invalid official coordinate/depth convention: {path}")
        geometry = {key: handle[key][()] for key in ("intrinsics", "extrinsics", "object_trajectories")}
        for key, value in geometry.items():
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite small geometry table {key}: {path}")
        K, E, poses = (geometry[key] for key in ("intrinsics", "extrinsics", "object_trajectories"))
        rotations = E[:, :3, :3]
        if (np.any(K[:, [0, 1], [0, 1]] <= 0) or not np.allclose(K[:, 2], [0, 0, 1])
                or not np.allclose(K[:, [0, 1], [1, 0]], 0)
                or not np.allclose(E[:, 3], [0, 0, 0, 1])
                or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-5)
                or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5)
                or not np.allclose(np.linalg.norm(poses[..., 3:], axis=-1), 1, atol=1e-4)):
            raise ValueError(f"non-rigid or invalid pinhole/object geometry: {path}")
    return dict(path=str(path), source_split=source_split, scene_id=scene_id,
                packed_bytes=path.stat().st_size, schema=SCHEMA_VERSION)


def validate_pack_inventory(records, expected_ids=None):
    """Require an exact unique ID set; a release gap needs separate raw proof."""
    expected_ids = EXPECTED_IDS if expected_ids is None else expected_ids
    ids = []
    for record in records:
        if record.get("schema") != SCHEMA_VERSION or record.get("source_split") != "train":
            raise ValueError("foreign-schema or official test data in full training inventory")
        official_scene_split("train", record["scene_id"])
        ids.append(int(record["scene_id"]))
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate official train scene IDs in packed inventory")
    if sorted(ids) != expected_ids:
        missing = sorted(set(expected_ids) - set(ids))
        extra = sorted(set(ids) - set(expected_ids))
        raise ValueError(f"full training inventory incomplete: valid={len(ids)}, "
                         f"missing={len(missing)} examples={missing[:10]}, extra={extra[:10]}")
    return records


def verify_converted_packs(root):
    paths = sorted(Path(root).rglob("*.h5"))
    return validate_pack_inventory([inspect_pack(path) for path in paths])


@contextmanager
def supervisor_lease(root):
    import fcntl
    with (Path(root) / "full_conversion.lock").open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another full MV-Kubric conversion supervisor is already running") from error
        yield handle.fileno()


def _stop_children(children):
    """Signal only this supervisor's unreaped child process groups."""
    for child in children:
        process = child["process"]
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    for child in children:
        process = child["process"]
        if process.poll() is None:
            try:
                process.wait(timeout=max(.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


def _archive_progress(root, archive):
    key = hashlib.sha256(str(Path(archive).resolve()).encode()).hexdigest()[:16]
    path = Path(root) / f"archive_{key}.progress.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def run_supervisor(archive_root, output_root, *, workers=2, min_free_gib=300,
                   poll_seconds=5):
    if isinstance(workers, bool) or workers not in (1, 2):
        raise ValueError("workers must be 1 or 2")
    if min_free_gib < 0:
        raise ValueError("min_free_gib must be nonnegative")
    archives = archive_inventory(archive_root)
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with supervisor_lease(output_root) as lease_fd:
        manifest_path = output_root / "conversion_manifest.json"
        progress_path = output_root / "full_conversion_progress.json"
        log_root = output_root / "child_logs"
        log_root.mkdir(exist_ok=True)
        started = _now()
        common = dict(schema=SCHEMA_VERSION, expected_scenes=len(EXPECTED_IDS),
            started_at=started, supervisor_pid=os.getpid(), workers=workers,
            cuda_visible_devices="", cpu_threads_per_child=2,
            min_free_gib=min_free_gib, archives=archives,
            archive_checksums_verified_in_this_run=False,
            resume_policy="Sequential gzip replay; atomically completed H5 scenes are skipped. "
                          "Earlier compressed bytes must be decompressed again after interruption.",
            source_download_checksum_note="Source download verification is separate; this supervisor does not claim a new full-archive SHA256 pass.")
        _atomic_json(manifest_path, dict(common, status="preparing", converted_scenes=0,
                                        official_train_scene_ids=[]))
        children, launched = [], []
        jobs = [dict(archive=a["path"], name=a["name"], state="pending") for a in archives]
        interrupted = []
        old_handlers = {}
        def signal_handler(number, _frame):
            interrupted.append(number)
        for number in (signal.SIGTERM, signal.SIGINT):
            old_handlers[number] = signal.signal(number, signal_handler)
        def publish(status, **extra):
            snapshot = dict(common, status=status, updated_at=_now(), jobs=jobs,
                            child_pids=[c["process"].pid for c in children if c["process"].poll() is None],
                            archives_succeeded=sum(j["state"] == "succeeded" for j in jobs), **extra)
            _atomic_json(progress_path, snapshot)
        repo = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
        env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        try:
            _free_space(output_root, min_free_gib)
            while any(job["state"] != "succeeded" for job in jobs):
                if interrupted:
                    raise InterruptedError(f"supervisor received signal {interrupted[-1]}")
                running = []
                for child in children:
                    process, job = child["process"], child["job"]
                    code = process.poll()
                    job["converter_progress"] = _archive_progress(output_root, job["archive"])
                    if code is None:
                        running.append(child)
                        continue
                    job.update(returncode=code, finished_at=_now(),
                               state="succeeded" if code == 0 else "failed")
                    child["log_handle"].close()
                    if code:
                        raise RuntimeError(f"archive conversion failed ({code}): {job['archive']}; log={job['log']}")
                children = running
                for job in jobs:
                    if len(children) >= workers:
                        break
                    if job["state"] != "pending":
                        continue
                    _free_space(output_root, min_free_gib)
                    log_path = log_root / (Path(job["name"]).name + ".log")
                    log_handle = log_path.open("a", buffering=1)
                    command = [sys.executable, "-u", "-m", "scripts.prepare_mv_kubric",
                        "--archive", job["archive"], "--output-root", str(output_root),
                        "--max-scenes", "0", "--min-free-gib", str(min_free_gib)]
                    try:
                        process = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                            stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True,
                            # If the supervisor is killed without cleanup,
                            # its live children keep the lock until they exit.
                            # A restart cannot duplicate their archive jobs.
                            pass_fds=(lease_fd,))
                    except BaseException:
                        log_handle.close()
                        raise
                    job.update(state="running", pid=process.pid, started_at=_now(),
                               command=command, log=str(log_path))
                    child = dict(process=process, job=job, log_handle=log_handle)
                    children.append(child); launched.append(child)
                publish("converting")
                if children:
                    time.sleep(poll_seconds)
            publish("validating_all_published_packs")
            records = verify_converted_packs(output_root)
            if interrupted:
                raise InterruptedError(f"supervisor received signal {interrupted[-1]} during final validation")
            complete = dict(common, status="complete", completed_at=_now(),
                converted_scenes=len(records), official_train_scene_ids=EXPECTED_IDS,
                packed_bytes=sum(record["packed_bytes"] for record in records),
                archive_results=jobs,
                validation="Every final H5 checked for schema/completion, renderer field shapes and finite small geometry tables; exact unique train IDs 31..5000; no official test data.")
            _atomic_json(manifest_path, complete)
            publish("complete", converted_scenes=len(records),
                    packed_bytes=complete["packed_bytes"], completed_at=complete["completed_at"])
            print(json.dumps({"status": "complete", "converted_scenes": len(records),
                              "manifest": str(manifest_path)}, sort_keys=True), flush=True)
            return complete
        except BaseException as error:
            _stop_children(children)
            for child in children:
                if child["job"]["state"] == "running":
                    child["job"].update(state="interrupted" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "cancelled",
                                        returncode=child["process"].poll(), finished_at=_now())
            status = "interrupted" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed"
            _atomic_json(manifest_path, dict(common, status=status, failed_at=_now(),
                converted_scenes=None, official_train_scene_ids=[], error=repr(error),
                archive_results=jobs))
            publish(status, error=repr(error))
            raise
        finally:
            for child in launched:
                if not child["log_handle"].closed:
                    child["log_handle"].close()
            for number, previous in old_handlers.items():
                signal.signal(number, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--min-free-gib", type=float, default=300.)
    args = parser.parse_args(argv)
    if args.min_free_gib < 0:
        parser.error("min-free-gib must be nonnegative")
    run_supervisor(args.archive_root, args.output_root, workers=args.workers,
                   min_free_gib=args.min_free_gib)


if __name__ == "__main__":
    main()
