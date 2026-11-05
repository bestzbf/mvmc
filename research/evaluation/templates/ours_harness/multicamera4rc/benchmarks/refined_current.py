"""Strict bridge from the current benchmark to an external native model tree.

The bootstrap functions in this file intentionally use only the standard
library. The entry point loads them by filename before importing the external
``multicamera4rc`` package, then binds only its benchmark subpackage to this
checkout. Model/config/refinement modules always stay in the external tree.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
import hashlib
import importlib
from importlib.machinery import ModuleSpec
import inspect
import json
import numbers
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType


_SOURCE_SUFFIXES = frozenset((".py", ".pyi", ".yaml", ".yml", ".json", ".toml", ".cu", ".cuh", ".cpp", ".h"))
_EXCLUDED_DIRECTORIES = frozenset((".git", "__pycache__", "build", "dist", ".venv", "venv"))


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _git(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments])


def build_model_runtime_manifest(model_code_root, *, expected_revision=None):
    """Require a clean external source tree and bind its actual source bytes."""
    root = Path(model_code_root).resolve(strict=True)
    package = root / "multicamera4rc"
    for required in ("__init__.py", "model.py", "config.py", "checkpoint.py"):
        if not (package / required).is_file():
            raise ValueError(f"External model tree is missing multicamera4rc/{required}")
    revision = _git(root, "rev-parse", "HEAD").decode().strip()
    if expected_revision is not None:
        expected = _git(root, "rev-parse", "--verify", str(expected_revision) + "^{commit}").decode().strip()
        if revision != expected:
            raise ValueError(f"External model revision differs: expected {expected}, got {revision}")
    if _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=no"):
        raise ValueError("External model has modified tracked files; use a clean frozen checkout")
    files = set()
    for raw in _git(root, "ls-files", "-z").split(b"\0"):
        if raw:
            relative = os.fsdecode(raw)
            if Path(relative).suffix in _SOURCE_SUFFIXES:
                files.add(relative)
    # Untracked modules inside the model package can be imported too. Bind
    # them, while avoiding unrelated outputs/downloaded data outside it.
    for directory, children, names in os.walk(package, followlinks=False):
        children[:] = sorted(c for c in children if c not in _EXCLUDED_DIRECTORIES
                             and not c.endswith(".egg-info"))
        for name in names:
            path = Path(directory) / name
            if path.suffix in _SOURCE_SUFFIXES:
                files.add(path.relative_to(root).as_posix())
    hashes = {}
    for relative in sorted(files):
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError(f"External model source resolves outside its checkout: {relative}")
        hashes[relative] = _file_sha256(path)
    return {"manifest_version": 1, "model_code_root": str(root), "model_package": str(package),
            "revision": revision, "tracked_worktree": "clean", "files_sha256": hashes}


def _module_origin(module):
    path = getattr(module, "__file__", None)
    return None if path is None else Path(path).resolve()


def _verify_loaded_sources(package, benchmarks):
    for name, module in list(sys.modules.items()):
        if name != "multicamera4rc" and not name.startswith("multicamera4rc."):
            continue
        allowed = benchmarks if name == "multicamera4rc.benchmarks" or name.startswith("multicamera4rc.benchmarks.") else package
        origin = _module_origin(module)
        if origin is None or not origin.is_relative_to(allowed):
            raise RuntimeError(f"Another {name} source tree is already loaded; start a fresh process")


def activate_external_model(model_code_root, benchmark_root, *, expected_revision=None):
    """Activate one model tree before model imports; expose only latest benches.

    Existing classes from a different tree cannot be made safe by deleting
    sys.modules entries. Fail closed and require a fresh evaluation process.
    ``benchmark_root`` is the directory containing the benchmark modules.
    """
    manifest = build_model_runtime_manifest(model_code_root, expected_revision=expected_revision)
    package = Path(manifest["model_package"])
    benchmarks = Path(benchmark_root).resolve(strict=True)
    if not (benchmarks / "tapvid_mv.py").is_file():
        raise ValueError("benchmark_root must contain the current benchmark adapters")
    _verify_loaded_sources(package, benchmarks)
    sys.path.insert(0, manifest["model_code_root"])
    model_package = importlib.import_module("multicamera4rc")
    if _module_origin(model_package) != package / "__init__.py":
        raise RuntimeError("Evaluation imported a different model package")
    if any(not Path(path).resolve().is_relative_to(package) for path in model_package.__path__):
        raise RuntimeError("External model package search path contains another source tree")
    # __init__ can itself import submodules. Validate them before accepting any
    # preexisting benchmark namespace or constructing the model.
    _verify_loaded_sources(package, benchmarks)
    # Explicitly select this benchmark package even if a future external model
    # checkout happens to bundle an older benchmarks directory of its own.
    name = "multicamera4rc.benchmarks"
    if name not in sys.modules:
        module = ModuleType(name)
        module.__path__ = [str(benchmarks)]
        module.__file__ = str(benchmarks / "__init__.py")
        module.__package__ = name
        module.__spec__ = ModuleSpec(name, loader=None, is_package=True)
        module.__spec__.submodule_search_locations = module.__path__
        sys.modules[name] = module
        model_package.benchmarks = module
    return manifest


def _modern_components(model_code_root):
    root = Path(model_code_root).resolve(strict=True)
    model_module = importlib.import_module("multicamera4rc.model")
    checkpoint_module = importlib.import_module("multicamera4rc.checkpoint")
    config_module = importlib.import_module("multicamera4rc.config")
    for module in (model_module, checkpoint_module, config_module):
        origin = _module_origin(module)
        if origin is None or not origin.is_relative_to(root / "multicamera4rc"):
            raise RuntimeError("Native model/config/checkpoint must come from the selected external model tree")
    return model_module.MultiCamera4RC, checkpoint_module.config_from_checkpoint


def validate_native_payload(payload, *, expected_step=None):
    """Reject foreign weights or config fallbacks before constructing a model."""
    if not isinstance(payload, Mapping) or payload.get("architecture") != "multicamera4rc":
        raise ValueError("Expected a native multicamera4rc checkpoint")
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, numbers.Integral) or step < 0:
        raise ValueError("Native checkpoint must contain a nonnegative integer step")
    if expected_step is not None:
        if isinstance(expected_step, bool) or not isinstance(expected_step, numbers.Integral) or expected_step < 0:
            raise ValueError("expected_step must be a nonnegative integer")
        if step != expected_step:
            raise ValueError(f"Checkpoint step mismatch: expected {expected_step}, got {step}")
    config = payload.get("config")
    if not isinstance(config, Mapping) or "embed_dim" not in config:
        raise ValueError("Native checkpoint must contain its complete saved model config")
    if not isinstance(payload.get("model"), Mapping) or not payload["model"]:
        raise ValueError("Native checkpoint must contain its full model state")
    return int(step)


def validate_native_config(saved, restored):
    """Forbid missing semantic settings even when tensor shapes still match.

    Historical config parsers may fill missing time/refinement fields with
    defaults. This refined entry requires the native config in full; checkpoint
    metadata outside that config remains unrestricted.
    """
    if not is_dataclass(restored) or isinstance(restored, type):
        raise ValueError("Native model config must be a restored dataclass instance")
    names = [field.name for field in fields(restored) if field.init]
    missing = sorted(set(names) - set(saved))
    if missing:
        raise ValueError(f"Native checkpoint has incomplete saved model config; missing={missing}")
    unexpected = sorted(set(saved) - set(names))
    if unexpected:
        raise ValueError(f"Native checkpoint has unsupported saved model config fields: {unexpected}")
    # JSON normalization permits the saved list -> native tuple conversion for
    # out_layers/dpt_channels, while preserving boolean/numeric value types.
    changed = [name for name in names
               if json.dumps(saved[name], sort_keys=True, allow_nan=False)
               != json.dumps(getattr(restored, name), sort_keys=True, allow_nan=False)]
    if changed:
        raise ValueError(f"Native config restoration changed saved settings: {changed}")
    return {"saved_init_fields": len(names), "all_init_fields_saved": True,
            "no_unknown_config_fields": True,
            "saved_restored_values_equal": True,
            "inference_overrides": {"gradient_checkpointing": False}}


def validate_native_state(model, state, *, require_refinement=False):
    """Verify all parameter/buffer keys and shapes, including trained refiner."""
    import torch
    expected = model.state_dict()
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("Native model state must contain named tensors only")
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise ValueError(f"Native checkpoint must cover the complete model; missing={missing}, unexpected={unexpected}")
    mismatches = {key: [list(state[key].shape), list(expected[key].shape)]
                  for key in expected if state[key].shape != expected[key].shape}
    if mismatches:
        raise ValueError(f"Native checkpoint architecture/shape mismatch: {mismatches}")
    refinement_dim = getattr(model.config, "refinement_dim", 0)
    refiner_keys = sorted(key for key in expected if key.startswith("trajectory_refiner."))
    refiner = getattr(model, "trajectory_refiner", None)
    if (require_refinement or refinement_dim > 0) and (refiner is None or not refiner_keys):
        raise ValueError("The saved refinement configuration must instantiate and load its full trajectory refiner")
    nonzero = None
    delta_key = "trajectory_refiner.delta_head.weight"
    if delta_key in state:
        nonzero = bool(state[delta_key].detach().count_nonzero().item())
    return {"parameter_tensors": len(expected), "checkpoint_tensors": len(state),
            "keys_exact": True, "shapes_exact": True, "refiner_tensors": len(refiner_keys),
            "refiner_configured": bool(refinement_dim), "refiner_delta_nonzero": nonzero}


def _validate_refiner_origin(model, model_code_root):
    refiner = getattr(model, "trajectory_refiner", None)
    if refiner is not None:
        origin = Path(inspect.getfile(type(refiner))).resolve()
        if not origin.is_relative_to(Path(model_code_root).resolve() / "multicamera4rc"):
            raise RuntimeError("Trajectory refiner was imported from a different model tree")


def load_native_checkpoint(checkpoint, *, model_code_root, expected_step=None,
                           require_refinement=False, audit_only=False):
    """Restore native config and every trained tensor without step hardcoding.

    Audit mode builds the model on meta and assigns the CPU-mapped checkpoint
    strictly, avoiding random giant CPU allocations or any GPU work. Inference
    retains the model's own strict native load_4rc_weights implementation.
    """
    import torch
    model_cls, config_from_checkpoint = _modern_components(model_code_root)
    payload = torch.load(checkpoint, mmap=True, map_location="cpu", weights_only=True)
    step = validate_native_payload(payload, expected_step=expected_step)
    restored_config = config_from_checkpoint(payload)
    config_coverage = validate_native_config(payload["config"], restored_config)
    config = replace(restored_config, gradient_checkpointing=False)
    if audit_only:
        with torch.device("meta"):
            model = model_cls(config).eval()
    else:
        model = model_cls(config).eval()
    _validate_refiner_origin(model, model_code_root)
    state = payload["model"]
    coverage = validate_native_state(model, state, require_refinement=require_refinement)
    if audit_only:
        result = model.load_state_dict(state, strict=True, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError("Strict CPU audit assignment did not cover the native model")
        report = {"matched": len(state), "missing": [], "missing_base": [], "unexpected": []}
    else:
        report = model.load_4rc_weights(payload, strict_base=True)
    if (report.get("matched") != coverage["parameter_tensors"] or report.get("missing")
            or report.get("missing_base") or report.get("unexpected")):
        raise ValueError("Native load must include every trained model/refinement tensor")
    provenance = {"checkpoint_step": step, "config": config.to_dict(), "load_report": report,
                  "config_coverage": config_coverage,
                  "state_coverage": coverage, "model_code_root": str(Path(model_code_root).resolve()),
                  "checkpoint_loading": "meta construction; strict CPU mmap assignment" if audit_only else "native strict_base=True",
                  "audit_only": bool(audit_only)}
    del payload
    return model, provenance


def load_refined_predictor(args):
    """Shared runner callback; full observed-time decoding keeps the refiner."""
    import torch
    from .tapvid_mv import predict_tapvid_mv_queries
    runtime = build_model_runtime_manifest(
        args.model_code_root, expected_revision=getattr(args, "expected_model_commit", None))
    if runtime != args.model_runtime_manifest:
        raise ValueError("External model sources changed between signature and loading")
    model, provenance = load_native_checkpoint(
        args.checkpoint, model_code_root=args.model_code_root,
        expected_step=getattr(args, "expected_step", None),
        require_refinement=getattr(args, "require_refinement", False))
    model.to(args.device)
    if build_model_runtime_manifest(args.model_code_root) != runtime:
        raise ValueError("External model sources changed during checkpoint loading")
    provenance["model_runtime_sha256"] = _digest_json(runtime)
    def predict(inputs):
        return predict_tapvid_mv_queries(model, inputs, args.device,
                                        target_time_chunk=len(inputs.frame_indices), amp_dtype=torch.bfloat16)
    return predict, provenance
