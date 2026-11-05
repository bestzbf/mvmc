#!/usr/bin/env python3
"""CPU-only5->6 layout planning and in-memory transformation; never writes weights.

A supplied new-rank CUDA RNG must be generated separately under the GPU guard.
This module does not initialize CUDA or synthesize Philox state bytes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

OFFSET = "historical_padding_sample_offset"
LINEAGE = "execution_layout_migration"
SCHEMA = "zero-skip-canonical-expansion-v14"


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid {name}")
    return value


def padded_epoch(size, world):
    return ((size + world - 1) // world) * world


def _layout(state):
    world = _integer(state.get("world_size"), "world_size", 1)
    size = _integer(state.get("dataset_size"), "dataset_size", 1)
    epoch = _integer(state.get("data_epoch"), "data_epoch")
    cursor = _integer(state.get("batches_consumed"), "batches_consumed")
    if (world != 5 or state.get("batch_size") != 1
            or state.get("gradient_accumulation") != 1
            or state.get("effective_batch_size") != world
            or state.get("sampler_kind") != "distributed_v1"):
        raise ValueError("requires saved5-rank distributed batch1/accumulation1")
    if cursor > math.ceil(size / world):
        raise ValueError("cursor exceeds padded epoch")
    balance = state.get("load_balancing")
    if (not isinstance(balance, dict) or balance.get("version") != 1
            or balance.get("world_size") != world or balance.get("dataset_size") != size):
        raise ValueError("requires the saved deterministic balancing layout")
    _integer(balance.get("window_batches"), "window_batches", 2)
    _integer(balance.get("start_epoch"), "start_epoch")
    _integer(balance.get("start_batch"), "start_batch")
    if type(balance.get(OFFSET, 0)) is not int:
        raise ValueError("invalid historical padding offset")
    if balance["start_epoch"] > epoch:
        raise ValueError("balancing activation is in the future")
    return world, size, epoch, cursor, balance


def consumed_samples(state):
    world = state["world_size"]
    return (state["load_balancing"].get(OFFSET, 0)
            + state["data_epoch"] * padded_epoch(state["dataset_size"], world)
            + state["batches_consumed"] * world)


def plan_expansion(state, world_size=6):
    old, size, epoch, cursor, balance = _layout(state)
    if type(world_size) is not int or world_size != 6:
        raise ValueError("only independently scoped5->6 expansion is supported")
    anchor = balance["start_batch"] if epoch == balance["start_epoch"] else 0
    final_batch = math.ceil(size / old)
    if cursor < anchor or ((cursor - anchor) % balance["window_batches"] and cursor != final_batch):
        raise ValueError("wait for a fully consumed old balancing window")
    if cursor == final_batch:
        new_epoch, new_cursor = epoch + 1, 0
    else:
        if cursor * old % world_size:
            raise ValueError("consumed sample prefix is not aligned to target batch")
        new_epoch, new_cursor = epoch, cursor * old // world_size
    offset = balance.get(OFFSET, 0) + new_epoch * (
        padded_epoch(size, old) - padded_epoch(size, world_size))
    new_balance = deepcopy(balance)
    new_balance.update(world_size=world_size, start_epoch=new_epoch,
                       start_batch=new_cursor, **{OFFSET: offset})
    result = dict(schema=SCHEMA, old_world_size=old, world_size=world_size,
        data_epoch=new_epoch, batches_consumed=new_cursor, load_balancing=new_balance,
        retained_rank_rng_indices=list(range(old)), added_rank_rng_indices=[5],
        reset_data_cursor=False, skipped_real_samples=0,
        future_padding_note="new DistributedSampler padding may repeat epoch-prefix samples",
        source_global_samples_consumed=consumed_samples(state))
    after = migrated_training_state(state, result)
    result["global_samples_consumed"] = consumed_samples(after)
    if result["global_samples_consumed"] != result["source_global_samples_consumed"]:
        raise ValueError("historical consumed sample accounting changed")
    return result


def migrated_training_state(state, plan):
    result = deepcopy(state)
    result.update(world_size=plan["world_size"], effective_batch_size=plan["world_size"],
        data_epoch=plan["data_epoch"], batches_consumed=plan["batches_consumed"],
        load_balancing=deepcopy(plan["load_balancing"]))
    return result


def next_aligned_step(state):
    _, size, epoch, cursor, _ = _layout(state)
    base = _integer(state.get("optimizer_updates"), "optimizer_updates")
    length = math.ceil(size / 5)
    for delta in range(length - cursor + 1):
        projected = deepcopy(state)
        projected.update(batches_consumed=cursor + delta, optimizer_updates=base + delta)
        try:
            return base + delta, plan_expansion(projected)
        except ValueError:
            pass
    raise ValueError("no aligned boundary within the saved epoch")


def _equal(a, b, path="payload"):
    import torch
    if isinstance(a, torch.Tensor):
        if (not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape
                or a.layout != b.layout or not torch.equal(a, b)):
            raise ValueError(f"tensor changed: {path}")
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise ValueError(f"mapping keys changed: {path}")
        for key in a:
            _equal(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError(f"sequence changed: {path}")
        for index, (x, y) in enumerate(zip(a, b)):
            _equal(x, y, f"{path}[{index}]")
    elif type(a) is not type(b) or a != b:
        raise ValueError(f"metadata changed: {path}")


def validate_rank_rng(rng, *, cuda_bytes):
    import numpy as np
    import torch
    if not isinstance(rng, dict) or set(rng) != {"torch", "python", "numpy", "cuda"}:
        raise ValueError("requires native torch/python/numpy/cuda RNG fields")
    for key in ("torch", "cuda"):
        tensor = rng[key]
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                or tensor.dtype != torch.uint8 or tensor.ndim != 1
                or not tensor.is_contiguous() or tensor.numel() == 0):
            raise ValueError(f"invalid native {key} RNG tensor")
    if rng["cuda"].numel() != cuda_bytes:
        raise ValueError("CUDA RNG size differs from saved native rank states")
    # Validate CPU/Python/NumPy structure with private generators, never globals.
    torch.Generator(device="cpu").set_state(rng["torch"])
    random.Random().setstate(rng["python"])
    n = rng["numpy"]
    if (not isinstance(n, (list, tuple)) or len(n) != 5 or n[0] != "MT19937"
            or not isinstance(n[1], list) or len(n[1]) != 624
            or any(type(x) is not int or not 0 <= x < 2**32 for x in n[1])
            or type(n[2]) is not int or not 0 <= n[2] <= 624
            or type(n[3]) is not int or n[3] not in (0, 1)
            or not isinstance(n[4], (int, float)) or not math.isfinite(n[4])):
        raise ValueError("invalid native NumPy RNG state")
    np.random.RandomState().set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))


def validate_source_payload(payload):
    import torch
    if not isinstance(payload, dict) or LINEAGE in payload:
        raise ValueError("source must be a native payload without ambiguous expansion lineage")
    for name in ("model", "optimizer", "config", "step", "rank_rng", "training_state"):
        if name not in payload:
            raise ValueError(f"native checkpoint missing {name}")
    if payload.get("architecture") != "multicamera4rc":
        raise ValueError("source is not a native multicamera4rc checkpoint")
    state = payload["training_state"]
    _layout(state)
    if payload["step"] != state.get("optimizer_updates"):
        raise ValueError("requires all-updates-applied optimizer boundary")
    if (not isinstance(payload["model"], dict) or not payload["model"]
            or not isinstance(payload["config"], dict)
            or not isinstance(state.get("loss_config"), dict)
            or not isinstance(state.get("lr_plan"), dict)):
        raise ValueError("source lacks complete model/config/loss/LR state")
    optimizer = payload["optimizer"]
    if not isinstance(optimizer, dict) or set(optimizer) != {"state", "param_groups"}:
        raise ValueError("source lacks standard Adam state")
    groups = optimizer["param_groups"]
    if (not isinstance(groups, list) or not isinstance(optimizer["state"], dict)
            or [g.get("name") for g in groups] not in (
                ["backbone", "heads", "refinement"],
                ["backbone", "heads", "refinement", "geometry"])
            or len(groups) != len(state["lr_plan"].get("base_lrs", []))):
        raise ValueError("source Adam/LR group identities disagree")
    rngs = payload["rank_rng"]
    if not isinstance(rngs, list) or len(rngs) != 5:
        raise ValueError("source must contain exactly5rank RNG states")
    reference = rngs[0].get("cuda") if isinstance(rngs[0], dict) else None
    if not isinstance(reference, torch.Tensor):
        raise ValueError("source must contain native CUDA RNG states")
    for rng in rngs:
        validate_rank_rng(rng, cuda_bytes=reference.numel())
    return state


def validate_new_rank_rng(envelope, payload, source_sha256):
    import torch
    old_rng = payload.get("rank_rng")
    if not isinstance(old_rng, list) or len(old_rng) != 5:
        raise ValueError("source must contain exactly5rank RNG states")
    reference = old_rng[0].get("cuda") if isinstance(old_rng[0], dict) else None
    if not isinstance(reference, torch.Tensor):
        raise ValueError("source must contain native CUDA RNG states")
    for rng in old_rng:
        validate_rank_rng(rng, cuda_bytes=reference.numel())
    if not isinstance(envelope, dict) or set(envelope) != {"state", "provenance"}:
        raise ValueError("new-rank RNG requires an explicit state/provenance envelope")
    provenance = envelope["provenance"]
    seed = _integer(payload["training_state"].get("seed"), "saved training seed") + 5
    if (not isinstance(provenance, dict)
            or provenance.get("schema") != "guarded-cuda-generator-rank-rng-v1"
            or provenance.get("rank") != 5 or provenance.get("seed") != seed
            or provenance.get("source_checkpoint_sha256") != source_sha256
            or provenance.get("method") != "torch.Generator(device=cuda).manual_seed(seed).get_state()"
            or provenance.get("physical_gpu_uuid") != "GPU-00000000-0000-0000-0000-000000000007"
            or not isinstance(provenance.get("torch_version"), str)
            or not provenance.get("torch_version")
            or provenance.get("guard_verified") is not True):
        raise ValueError("new-rank CUDA RNG provenance does not match guarded rank5initialization")
    validate_rank_rng(envelope["state"], cuda_bytes=reference.numel())
    return provenance


def transform_payload(payload, new_rank_rng, *, source_sha256):
    if (not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in source_sha256)):
        raise ValueError("source SHA256 is required")
    state = validate_source_payload(payload)
    plan = plan_expansion(state)
    provenance = validate_new_rank_rng(new_rank_rng, payload, source_sha256)
    transformed = dict(payload)
    transformed["training_state"] = migrated_training_state(state, plan)
    transformed["rank_rng"] = list(payload["rank_rng"]) + [deepcopy(new_rank_rng["state"])]
    transformed[LINEAGE] = dict(schema=SCHEMA, source_checkpoint_sha256=source_sha256,
        source_step=payload["step"], physical_gpus=[0, 1, 2, 5, 6, 7], plan=deepcopy(plan),
        new_rank_rng_provenance=deepcopy(provenance),
        recipe_migration=False, checkpoint_written=False)
    verify_transformed_payload(payload, transformed, new_rank_rng, source_sha256=source_sha256)
    return transformed


def verify_transformed_payload(source, result, new_rank_rng, *, source_sha256):
    plan = plan_expansion(source["training_state"])
    validate_new_rank_rng(new_rank_rng, source, source_sha256)
    if result.keys() != source.keys() | {LINEAGE}:
        raise ValueError("unexpected top-level mutation")
    _equal(migrated_training_state(source["training_state"], plan), result["training_state"], "training_state")
    _equal(source["rank_rng"], result["rank_rng"][:5], "retained_rank_rng")
    if len(result["rank_rng"]) != 6:
        raise ValueError("expanded rank RNG count must be6")
    _equal(new_rank_rng["state"], result["rank_rng"][5], "new_rank_rng")
    for name in source:
        if name not in ("training_state", "rank_rng"):
            _equal(source[name], result[name], name)
    expected_lineage = dict(schema=SCHEMA, source_checkpoint_sha256=source_sha256,
        source_step=source["step"], physical_gpus=[0, 1, 2, 5, 6, 7], plan=deepcopy(plan),
        new_rank_rng_provenance=deepcopy(new_rank_rng["provenance"]),
        recipe_migration=False, checkpoint_written=False)
    _equal(expected_lineage, result[LINEAGE], LINEAGE)
    return plan


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "dry-run"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--new-rank-rng", type=Path)
    parser.add_argument("--new-rank-rng-sha256")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.source.is_symlink() or not args.source.is_file():
        raise ValueError("source must be a pinned regular file")
    source_stat = args.source.stat()
    source_identity = (source_stat.st_dev, source_stat.st_ino, source_stat.st_size, source_stat.st_mtime_ns)
    if sha256_file(args.source) != args.source_sha256:
        raise ValueError("source SHA256 mismatch")
    payload = torch.load(args.source, map_location="cpu", mmap=True, weights_only=True)
    validate_source_payload(payload)
    step, plan = next_aligned_step(payload["training_state"])
    receipt = dict(schema=SCHEMA, command=args.command, checkpoint_written=False,
        source=str(args.source.resolve()), source_sha256=args.source_sha256,
        next_aligned_step=step, plan=plan)
    if args.command == "dry-run":
        if args.new_rank_rng is None or args.new_rank_rng_sha256 is None:
            raise ValueError("dry-run requires explicit pinned new-rank RNG state")
        if args.new_rank_rng.is_symlink() or not args.new_rank_rng.is_file():
            raise ValueError("RNG envelope must be a pinned regular file")
        if sha256_file(args.new_rank_rng) != args.new_rank_rng_sha256:
            raise ValueError("new-rank RNG SHA256 mismatch")
        envelope = torch.load(args.new_rank_rng, map_location="cpu", weights_only=True)
        transformed = transform_payload(payload, envelope, source_sha256=args.source_sha256)
        receipt.update(dry_run_verified=True, rank_rng_count=len(transformed["rank_rng"]))
    after = args.source.stat()
    if source_identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("source checkpoint changed during CPU dry-run")
    text = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    if args.report:
        with args.report.open("x") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
