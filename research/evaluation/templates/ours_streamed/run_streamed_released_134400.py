"""Run the released benchmark runner with the validated low-memory decoder."""
from __future__ import annotations

import importlib
from contextlib import contextmanager
import importlib.util
import fcntl
import json
import os
import time
import sys
from pathlib import Path


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def inference_lock(lock_path):
    """Reuse an inherited physical GPU lock without a second flock."""
    inherited = os.environ.get("SERIAL_EVAL_GPU_LOCK_FD")
    if inherited is not None:
        try:
            descriptor = int(inherited)
            if descriptor < 0:
                raise ValueError("negative lock descriptor")
            inherited_stat = os.fstat(descriptor)
            lock_stat = lock_path.stat()
        except (ValueError, OSError) as error:
            raise RuntimeError("Invalid inherited SERIAL_EVAL_GPU_LOCK_FD") from error
        if (inherited_stat.st_dev, inherited_stat.st_ino) != (lock_stat.st_dev, lock_stat.st_ino):
            raise RuntimeError("Inherited GPU lock descriptor points to a different lock")
        yield descriptor
        return
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield lock.fileno()


def main():
    args = sys.argv[1:]
    if not args or args[0] != "predict":
        raise SystemExit("This wrapper only supports the released predict command")
    root = Path("@@OURS_EVAL_HARNESS_ROOT@@")
    benchmarks = root / "multicamera4rc/benchmarks"
    bootstrap = load_file("_streamed_released_bootstrap", benchmarks / "refined_current.py")
    early = importlib.import_module("argparse").ArgumentParser(add_help=False)
    early.add_argument("--model-code-root", type=Path)
    early.add_argument("--expected-model-commit")
    options, _ = early.parse_known_args(args[1:])
    runtime = bootstrap.activate_external_model(
        options.model_code_root,
        benchmarks,
        expected_revision=options.expected_model_commit,
    )

    import torch
    torch.set_num_threads(8)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.cuda.set_per_process_memory_fraction(1.0, 0)
    import multicamera4rc.motion as motion
    import multicamera4rc.vendor.heads.dpt_head as dpt
    from multicamera4rc.benchmarks import released_runner

    cache_root = Path("@@OURS_STREAM_COMPONENT_ROOT@@/eval_streaming_candidate_20261102")
    memory = load_file(
        "_streamed_memory_exact",
        Path("@@OURS_STREAM_COMPONENT_ROOT@@/eval_exact_v9_fixed20_20261102/memory_exact.py"),
    )
    stream_model = load_file("multicamera4rc._stream_model", cache_root / "model.py")
    visibility = load_file(
        "multicamera4rc._stream_visibility",
        Path("@@OURS_STREAM_COMPONENT_ROOT@@/eval_exact_v2_20261102/visibility.py"),
    )
    stream = load_file("multicamera4rc._stream_helper", cache_root / "streaming.py")
    motion.custom_interpolate = memory.custom_interpolate
    dpt.custom_interpolate = memory.custom_interpolate

    original_load = released_runner.load_prediction_model

    def load_prediction_model(*load_args, **load_kwargs):
        model, loading = original_load(*load_args, **load_kwargs)
        model.__class__ = stream_model.MultiCamera4RC
        model.visibility_head.__class__ = visibility.VisibilityHead
        model.evaluation_cache_enabled = True
        memory.install(model)
        return model, loading

    original_predict = released_runner.predict_tapvid_mv_queries

    def predict_streamed(model, inputs, **predict_kwargs):
        with stream.streamed(model, inputs, time_chunk=8):
            return original_predict(model, inputs, **predict_kwargs)

    released_runner.load_prediction_model = load_prediction_model
    released_runner.predict_tapvid_mv_queries = predict_streamed
    released = importlib.import_module("multicamera4rc.benchmarks.released")
    lock_path = Path("@@BENCHMARK_ROOT@@/evaluation/.gpu4_single_inference.lock")
    with inference_lock(lock_path):
        report = Path("@@OURS_STREAMED_ROOT@@/drive_minival_134400_streamed_status.json")
        status = {"stage": "running", "started_at": time.time(), "pid": os.getpid(),
                  "device": torch.cuda.get_device_name(0), "per_process_memory_fraction": 1.0,
                  "manifest": args[args.index("--manifest") + 1],
                  "output": args[args.index("--output") + 1], "stream_time_chunk": 8}
        report.write_text(json.dumps(status, indent=2))
        try:
            released.main(args, model_runtime_manifest=runtime)
            status.update(stage="predictions_complete", finished_at=time.time())
            report.write_text(json.dumps(status, indent=2))
        except BaseException as error:
            status.update(stage="failed", error=type(error).__name__ + ": " + str(error), updated_at=time.time())
            report.write_text(json.dumps(status, indent=2))
            raise


if __name__ == "__main__":
    main()
