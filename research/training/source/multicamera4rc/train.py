"""Supervised training entry point: python -m multicamera4rc.train --help."""

import argparse
from dataclasses import replace
import inspect
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn, distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from . import MultiCamera4RC, MultiCamera4RCConfig
from .checkpoint import config_from_checkpoint, read_checkpoint, save_checkpoint
from .data import MultiCameraManifestDataset, collate_multicamera, select_source_targets
from .losses import MultiCamera4RCLoss
from .metrics import METRICS_VERSION, dense_metrics
from .schedule import LearningRatePlan
from .training_execution_future import TrainingExecution, execution_scope, validate_resume_mode, validate_resume_identity
from .six_rank_gate_v22 import SixRankProof
from .metadata import build_metadata, gather_observations
from .distributed import (DistributedContext, aggregate_metrics, resolve_amp_dtype,
                          restore_rng)


def _sparse_supervised_queries(targets, index):
    """Prefer sources that carry finite, valid labels rather than padded queries."""
    queries = targets["sparse_queries"][index]
    supervised = torch.zeros(len(queries), dtype=torch.bool, device=queries.device)
    if not len(queries):
        return supervised
    for name in ("trajectory", "displacement", "visibility"):
        key = f"sparse_{name}"
        if key not in targets:
            continue
        truth = targets[key][index]
        vector = name != "visibility"
        finite = torch.isfinite(truth).all(-1) if vector else torch.isfinite(truth)
        mask_key = f"{key}_valid"
        supplied = targets.get(mask_key)
        if supplied is None and name == "displacement":
            supplied = targets.get("sparse_trajectory_valid")
        if supplied is not None:
            mask = supplied[index].bool()
            if vector and mask.shape == truth.shape:
                mask = mask.all(-1)
            finite = finite & torch.broadcast_to(mask, finite.shape)
        supervised |= finite.reshape(len(queries), -1).any(-1)
    if "sparse_query_valid" in targets:
        supervised &= targets["sparse_query_valid"][index].bool()
    return supervised


def _sources(batch, count, max_source_time_pairs=0, prefer_motion=False, prefer_paired=False):
    valid = batch["observation_valid"]
    b, s = valid.shape
    if max_source_time_pairs:
        times = batch["query_time_idx"].shape[1]
        budget_count = max(1, max_source_time_pairs // times)
        count = min(count or s, budget_count)
    if count == 0:
        return torch.arange(s, device=valid.device)[None].expand(b, -1)
    count = min(count, int(valid.sum(1).min()))
    rows = []
    for index in range(b):
        candidates = valid[index].nonzero(as_tuple=False).flatten()
        labels = candidates[:0]
        prioritized = "sparse_queries" in batch["targets"]
        if prioritized:
            queries = batch["targets"]["sparse_queries"][index]
            active = _sparse_supervised_queries(batch["targets"], index)
            height, width = batch["images"].shape[-2:]
            active &= (torch.isfinite(queries).all(-1)
                       & (queries[:, 0] == queries[:, 0].round())
                       & (queries[:, 1] >= 0) & (queries[:, 1] <= width-1)
                       & (queries[:, 2] >= 0) & (queries[:, 2] <= height-1))
            labels = queries[active, 0].long().unique()
            labels = labels[(labels >= 0) & (labels < s)]
            labels = labels[valid[index, labels]]
        if prefer_motion and "dense_pair_valid" in batch["targets"]:
            known = batch["targets"]["dense_pair_valid"][index].bool().flatten(1).any(1)
            labels = torch.cat((labels, (known & valid[index]).nonzero().flatten())).unique()
            prioritized = True
        if prefer_motion and "dense_motion_source_valid" in batch["targets"]:
            known = batch["targets"]["dense_motion_source_valid"][index].bool()
            labels = torch.cat((labels, (known & valid[index]).nonzero().flatten())).unique()
            prioritized = True
        if prioritized:
            labels = labels[torch.randperm(len(labels), device=labels.device)]
            rest = candidates[~torch.isin(candidates, labels)]
            rest = rest[torch.randperm(len(rest), device=rest.device)]
            candidates = torch.cat((labels, rest))
        else:
            candidates = candidates[torch.randperm(len(candidates), device=candidates.device)]
        if prefer_paired:
            from .query_pair import paired_source_order
            candidates = paired_source_order(candidates, batch['camera_id'][index], batch['time_idx'][index], count)
        rows.append(candidates[:count])
    return torch.stack(rows)


class _CursorBatchSampler(BatchSampler):
    """Skip sampler indices before dataset reads, preserving the epoch order."""
    start_batch = 0

    def __iter__(self):
        for index, batch in enumerate(super().__iter__()):
            if index >= self.start_batch:
                yield batch

    def __len__(self):
        return max(0, super().__len__()-self.start_batch)


def _resume_data_cursor(state, step, dataset_size, world_size, batch_size, *, reset=False, sampler_kind=None):
    """Preserve the consumed permutation prefix when the number of ranks changes.

    DistributedSampler partitions one shared permutation by rank. A saved
    batch cursor therefore represents old_world * old_batch samples, not the
    optimizer step count (which may span earlier data configurations).
    """
    if reset:
        return 0, 0
    old_world = int(state.get("world_size", world_size))
    old_batch = int(state.get("batch_size", batch_size))
    if min(dataset_size, old_world, old_batch, world_size, batch_size) <= 0:
        raise ValueError("dataset and training layout sizes must be positive")
    if state.get("dataset_size", dataset_size) != dataset_size:
        raise ValueError("dataset size changed; use --reset-data-cursor for a new data configuration")
    old_samples = (dataset_size + old_world - 1) // old_world
    old_batches = (old_samples + old_batch - 1) // old_batch
    epoch = int(state.get("data_epoch", step // old_batches))
    cursor = int(state.get("batches_consumed", step % old_batches))
    if epoch < 0 or not 0 <= cursor <= old_batches:
        raise ValueError("checkpoint has an invalid training data cursor")
    if cursor == old_batches:
        return epoch + 1, 0
    if (old_world, old_batch) == (world_size, batch_size):
        return epoch, cursor
    if old_world == 1 or world_size == 1:
        # Accumulation uses the exact DistributedSampler permutation with
        # num_replicas=1. Legacy RandomSampler checkpoints still need a reset.
        before=state.get('sampler_kind','distributed_v1' if old_world>1 else 'random_v1')
        after=sampler_kind or ('distributed_v1' if world_size>1 else 'random_v1')
        if before!='distributed_v1' or after!='distributed_v1':
            raise ValueError("single/distributed sampler migration requires --reset-data-cursor")
    consumed = cursor * old_world * old_batch
    new_global_batch = world_size * batch_size
    if consumed % new_global_batch:
        raise ValueError("saved sample cursor is not aligned with the new global batch; "
                         "resume at an aligned checkpoint or use --reset-data-cursor")
    return epoch, consumed // new_global_batch


def _criterion(args, dense_supervision, training_state):
    profile = args.loss_profile
    if args.loss_config:
        path = Path(args.loss_config)
        if path.suffix.lower() == ".json":
            config = json.loads(path.read_text())
        else:
            import yaml
            config = yaml.safe_load(path.read_text())
        return MultiCamera4RCLoss(**config), "custom"
    if profile == "auto" and "loss_config" in training_state:
        return MultiCamera4RCLoss(**training_state["loss_config"]), training_state.get("loss_profile", "custom")
    if profile in ("dense", "dense_balanced") and not dense_supervision:
        raise ValueError("dense loss profiles require renderer-derived full-pixel supervision")
    if profile == "dense_balanced":
        return MultiCamera4RCLoss.balanced_dense(), profile
    return (MultiCamera4RCLoss.dense(), "dense") if dense_supervision else (MultiCamera4RCLoss(), "default")


class _TrainingStep(nn.Module):
    """Return the actual loss graph so DDP can identify unused task heads.

    Wrapping the prediction-only model would expose ray/visibility outputs to
    DDP even when no corresponding GT participates in backward. Restricting
    this wrapper's differentiable outputs to total loss avoids that mismatch.
    """

    def __init__(self, model, criterion):
        super().__init__()
        self.model, self.criterion = model, criterion

    def forward(self, images, camera_id, time_idx, query_time_idx, source,
                observation_valid, targets, need_visibility, principal_canvas=None):
        if self.model.config.intrinsics_sharing == 'camera' and 'K' in targets:
            from .geometry import check_constant_intrinsics
            # Known native preprocessing and per-camera ROI are constant across
            # time. This asserts the contract; GT never decides model grouping.
            check_constant_intrinsics(targets['K'], camera_id,
                observation_valid & targets.get('camera_valid', observation_valid).bool())
        geometry_only = ("displacement_valid" in targets and not targets["displacement_valid"].any()
                         and not targets.get("sparse_displacement_valid", torch.zeros((), device=images.device)).any()
                         and not targets.get("sparse_trajectory_valid", torch.zeros((), device=images.device)).any()
                         and not need_visibility and not targets.get("static_mask", torch.zeros((), device=images.device)).any())
        if geometry_only:
            # A geometry-only sample has no temporal labels. Train geometry
            # on every observation and source base point; avoid an expensive
            # motion forward whose outputs would receive no gradient.
            scene = self.model.encode(images, camera_id, time_idx, observation_valid,
                                      principal_canvas=principal_canvas)
            meta = build_metadata(scene.camera_id, scene.time_idx, query_time_idx, source, scene.observation_valid)
            base = gather_observations(scene.geometry["base_points"], meta.source_index)
            identity = (meta.source_times[..., None] == meta.query_times[:, None]) & meta.source_valid[..., None]
            xyz = torch.where(identity[..., None, None, None], base[:, :, None], 0)
            prediction = dict(scene.geometry)
            prediction.update(trajectory=xyz, displacement=torch.zeros_like(xyz),
                source_obs_idx=meta.source_index, source_valid=meta.source_valid,
                source_time_idx=meta.source_times, query_time_idx=meta.query_times,
                camera_id=scene.camera_id, time_idx=scene.time_idx,
                observation_valid=scene.observation_valid)
        else:
            prediction = self.model(images, camera_id, time_idx, query_time_idx, source,
                                    observation_valid=observation_valid,
                                    return_visibility=need_visibility, principal_canvas=principal_canvas)
        terms = self.criterion(prediction, targets)
        for name in ('refinement_update', 'refinement_match_fraction'):
            if name in prediction:
                terms[name] = prediction[name][prediction['source_valid']].mean()
        if "scene_scale" in targets and "trajectory_valid" in targets and "trajectory" in targets:
            terms.update(dense_metrics(prediction, targets))
        return terms["loss"], {name: value.detach() for name, value in terms.items()}


def _optimizer_step(loss, model, optimizer, scaler, clip_grad):
    if loss is not None:
        scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    # FP16 overflow is handled by GradScaler's standard skipped step. A hard
    # clipping error would abort before it can lower the scale. Unscaled/BF16
    # gradients still fail promptly instead of silently updating with NaNs.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad,
                                         error_if_nonfinite=not scaler.is_enabled())
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    skipped = scaler.is_enabled() and scaler.get_scale() < previous_scale
    return norm, skipped


def _reset_absolute_time_projections(model, optimizer):
    """Explicit migration only: retain relative-time weights and all other Adam states."""
    if model.config.time_origin != 'clip_start':
        raise ValueError('time projection reset requires time_origin=clip_start')
    names = ('backbone.frame_projection.weight', 'target_projection.weight')
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name in names:
            parameter = parameters[name]
            parameter.zero_()
            optimizer.state.pop(parameter, None)
    return names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    data_input = parser.add_mutually_exclusive_group(required=True)
    data_input.add_argument("--manifest", help="Prepared JSON/JSONL observations and tensor targets")
    data_input.add_argument("--data-config", help="Native adapter mixture JSON/YAML configuration")
    parser.add_argument("--output", required=True)
    parser.add_argument("--execution-optimizations", choices=("native", "fa4_metadata_prepared_kv"), default="native",
                        help="Optional isolated execution scope; native/default leaves kernels unchanged")
    parser.add_argument("--preset", choices=("giant", "tiny"), default="giant")
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--pretrained", help="Original 4RC .pth/.safetensors; requires matching giant preset")
    weights.add_argument("--resume", help="Native model+optimizer training checkpoint")
    parser.add_argument("--reset-data-cursor", action="store_true",
                        help="With --resume, restart sampling at epoch/batch zero for a new data configuration")
    parser.add_argument("--migrate-data-config", action="store_true",
                        help="Explicitly restart a changed native data stream, including balancing, while preserving model/Adam/RNG/LR")
    parser.add_argument("--partial-pretrained", action="store_true", help="Report, rather than reject, missing original modules")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Local batch size per process; torchrun multiplies it by WORLD_SIZE")
    parser.add_argument("--gradient-accumulation", type=int, default=1,
                        help="Microbatches per rank per Adam update; preserves optimizer-step LR scheduling")
    parser.add_argument("--source-count", type=int, default=2, help="0 decodes all sources")
    parser.add_argument("--max-source-time-pairs", type=int, default=0,
                        help="Limit decoded reference count by Q*T; 0 disables. All pixels/particles of each selected reference remain supervised")
    parser.add_argument("--prefer-motion-sources", action=argparse.BooleanOptionalAction, default=None,
                        help="Prefer references with valid native pair labels; resume restores its saved policy")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument('--balance-window', type=int, default=None)
    parser.add_argument('--balance-costs', help='Frozen JSON map from source name to observed image [H,W]')
    parser.add_argument('--reuse-shared-computation', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--reuse-decoder-kv', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--profile-step-timing', action='store_true')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--refinement-lr", type=float, default=1e-4,
                        help="Peak LR for newly initialized residual tracking modules")
    parser.add_argument("--geometry-lr-multiplier", type=float, default=None,
                        help="Split depth/ray/camera parameters into a named group and multiply their inherited LR; preserves the original cosine phase")
    parser.add_argument("--geometry-lr-warmup-steps", type=int, default=None,
                        help="One-time warmup for a newly enabled geometry LR multiplier (default 100)")
    parser.add_argument("--override-lr", action="store_true", help="With resume, apply CLI learning rates while preserving optimizer moments")
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default=None,
                        help="Start an explicit LR phase; otherwise restore the checkpoint schedule")
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--warmup-start-ratio", type=float, default=None,
                        help="Resume warmup from this fraction of the new peak LR")
    parser.add_argument("--lr-end-step", type=int, default=None,
                        help="End of the LR phase, independent of this curriculum stage's --steps")
    parser.add_argument("--min-lr-ratio", type=float, default=None)
    loss_settings = parser.add_mutually_exclusive_group()
    loss_settings.add_argument("--loss-profile", choices=("auto", "dense", "dense_balanced"), default="auto")
    loss_settings.add_argument("--loss-config", help="JSON/YAML loss constructor settings; checkpointed in full")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("auto", "bfloat16", "float16"), default="auto",
                        help="Dtype with --amp: auto prefers CUDA BF16 and uses CPU BF16")
    parser.add_argument("--freeze-pretrained-backbone", action="store_true")
    parser.add_argument("--freeze-geometry", action="store_true")
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--decoder-memory", choices=("all_observations", "target_time"),
                        help="Explicit memory scope; resume otherwise preserves checkpoint behavior")
    parser.add_argument("--time-origin", choices=("clip_start", "absolute"),
                        help="Explicit time encoding migration; resume otherwise preserves checkpoint behavior")
    parser.add_argument("--principal-point-mode", choices=("image", "crop_original"),
                        help="Explicit principal crop parameterization; defaults to checkpoint semantics")
    parser.add_argument("--intrinsics-sharing", choices=("none", "camera"),
                        help="Explicit constant-intrinsics camera grouping; restores checkpoint by default")
    parser.add_argument("--reset-time-projections", action="store_true",
                        help="On resume with clip_start, reset only the two absolute-time projections and their Adam states")
    parser.add_argument("--cpu-threads", type=int, default=4)
    attention = parser.add_mutually_exclusive_group()
    attention.add_argument("--attention-chunk-size", type=int,
                        help="Override attention query chunk size, including native checkpoint resume")
    attention.add_argument("--no-query-chunking", action="store_true",
                           help="Compute whole attention queries and visibility pixels without splitting")
    args = parser.parse_args(argv)
    if min(args.steps, args.batch_size, args.save_every, args.cpu_threads, args.gradient_accumulation) <= 0 or args.source_count < 0 or args.workers < 0 or args.max_source_time_pairs < 0:
        parser.error("steps, batch size, save interval and threads must be positive; source count/workers nonnegative")
    if min(args.lr, args.backbone_lr, args.clip_grad) <= 0 or args.weight_decay < 0:
        parser.error("learning rates and clip-grad must be positive; weight-decay nonnegative")
    if args.attention_chunk_size is not None and args.attention_chunk_size <= 0:
        parser.error("attention-chunk-size must be positive")
    if args.reset_data_cursor and not args.resume:
        parser.error("reset-data-cursor requires --resume")
    if args.migrate_data_config and (not args.resume or not args.data_config):
        parser.error("migrate-data-config requires --resume and --data-config")
    if args.geometry_lr_multiplier is not None and (not math.isfinite(args.geometry_lr_multiplier) or args.geometry_lr_multiplier <= 0):
        parser.error("geometry-lr-multiplier must be finite and positive")
    if args.geometry_lr_warmup_steps is not None and args.geometry_lr_warmup_steps < 0:
        parser.error("geometry-lr-warmup-steps must be nonnegative")
    if args.override_lr and not args.resume:
        parser.error("override-lr requires --resume")
    if args.reset_time_projections and not args.resume:
        parser.error("reset-time-projections requires --resume")
    context = DistributedContext.initialize(args.device)
    try:
        return _train(args, context)
    finally:
        context.close()


@execution_scope
def _train(args, context):
    device = context.device
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.data_config:
        from .datasets.mixture import create_dataset, collate_adapter_samples
        dataset = create_dataset(args.data_config)
        collate_fn = collate_adapter_samples
    else:
        dataset = MultiCameraManifestDataset(args.manifest)
        collate_fn = collate_multicamera
    resume = read_checkpoint(args.resume) if args.resume else None
    training_state = resume.get("training_state", {}) if resume is not None else {}
    data_fingerprint = (hashlib.sha256(Path(args.data_config).read_bytes()).hexdigest()
                        if args.data_config else None)
    saved_fingerprint = training_state.get('data_config_sha256')
    data_migration = training_state.get('data_migration')
    if args.migrate_data_config:
        if saved_fingerprint is not None and saved_fingerprint == data_fingerprint:
            raise ValueError('data migration requested for an unchanged data configuration')
        args.reset_data_cursor = True
        data_migration = dict(step=int(resume.get('step', 0)),
                              previous_sha256=saved_fingerprint,
                              new_sha256=data_fingerprint,
                              cursor_policy='explicit_reset_once',
                              model_optimizer_rng_lr_preserved=True)
    elif saved_fingerprint is not None and saved_fingerprint != data_fingerprint:
        raise ValueError('data configuration changed; use --migrate-data-config for an explicit new stream')
    if resume is not None and not args.migrate_data_config:
        for name in ('augmentation', 'photometric_augmentation'):
            saved = training_state.get(name)
            if saved != getattr(dataset, name, None):
                raise ValueError(f'{name} changed; use --migrate-data-config for an explicit new stream')
    prefer_motion_sources = (args.prefer_motion_sources if args.prefer_motion_sources is not None
                             else training_state.get("prefer_motion_sources", False))
    training_seed = int(training_state.get("seed", args.seed))
    use_distributed_sampler = (context.distributed or args.gradient_accumulation > 1
                               or training_state.get('sampler_kind') == 'distributed_v1')
    sampler_kind = 'distributed_v1' if use_distributed_sampler else 'random_v1'
    from .load_balance import BalancedDistributedSampler, validate_balance
    balance = training_state.get('load_balancing')
    if balance and args.migrate_data_config:
        balance = dict(balance, start_epoch=0, start_batch=0,
                       world_size=context.world_size, dataset_size=len(dataset))
    if args.balance_window is not None:
        if balance is not None:
            if args.balance_window != balance['window_batches']:
                raise ValueError('cannot change a resumed balancing window')
        else:
            if args.balance_window < 2 or not args.balance_costs or not use_distributed_sampler:
                raise ValueError('balancing requires window >=2, a cost catalogue and distributed sampling')
            epoch, cursor = _resume_data_cursor(training_state, int(resume.get('step', 0)) if resume else 0,
                len(dataset), context.world_size, args.batch_size, reset=args.reset_data_cursor, sampler_kind=sampler_kind)
            balance = dict(version=1, window_batches=args.balance_window, start_epoch=epoch, start_batch=cursor,
                world_size=context.world_size, dataset_size=len(dataset), source_count=args.source_count,
                max_source_time_pairs=args.max_source_time_pairs,
                image_hw=json.loads(Path(args.balance_costs).read_text()))
    if balance:
        validate_balance(balance, context.world_size, args.batch_size, len(dataset))
        if args.reset_data_cursor and not args.migrate_data_config:
            raise ValueError('cannot reset the cursor of a balanced stream')
        if args.balance_costs and json.loads(Path(args.balance_costs).read_text()) != balance['image_hw']:
            raise ValueError('saved balancing cost catalogue changed')
        if (balance['source_count'], balance['max_source_time_pairs']) != (args.source_count, args.max_source_time_pairs):
            raise ValueError('balanced source/time budget changed')
    sampler = (DistributedSampler(dataset, num_replicas=context.world_size,
                                  rank=context.rank, shuffle=True, seed=training_seed)
               if use_distributed_sampler else None)
    if balance:
        sampler = BalancedDistributedSampler(dataset, balance=balance, num_replicas=context.world_size,
            rank=context.rank, shuffle=True, seed=training_seed)
    loader_generator = torch.Generator()
    if sampler is None:
        sampler = RandomSampler(dataset, generator=loader_generator)
    batch_sampler = _CursorBatchSampler(sampler, args.batch_size, drop_last=False)
    loader_options = {}
    if args.workers and device.type == "cuda":
        # CUDA/NCCL has already initialized in the parent; fork would inherit
        # unsafe CUDA state. Native adapters support spawn-safe H5 reopening.
        loader_options["multiprocessing_context"] = "spawn"
    loader = DataLoader(dataset, batch_sampler=batch_sampler, generator=loader_generator,
                        num_workers=args.workers, collate_fn=collate_fn,
                        pin_memory=device.type == "cuda", **loader_options)
    if not len(loader):
        raise ValueError("the training dataset supplies no batches")
    batches_per_epoch = len(loader)
    cfg = (MultiCamera4RCConfig.tiny(gradient_checkpointing=not args.no_checkpointing)
           if args.preset == "tiny" else MultiCamera4RCConfig(gradient_checkpointing=not args.no_checkpointing))
    cfg = config_from_checkpoint(resume, fallback=cfg)
    for name in ('reuse_shared_computation', 'reuse_decoder_kv'):
        if getattr(args, name) is not None:
            cfg = replace(cfg, **{name: getattr(args, name)})
    if args.no_checkpointing:
        cfg = replace(cfg, gradient_checkpointing=False)
    if args.decoder_memory is not None:
        cfg = replace(cfg, decoder_memory=args.decoder_memory)
    if args.time_origin is not None:
        cfg = replace(cfg, time_origin=args.time_origin)
    if args.principal_point_mode is not None:
        cfg = replace(cfg, principal_point_mode=args.principal_point_mode)
    if getattr(args, 'intrinsics_sharing', None) is not None:
        cfg = replace(cfg, intrinsics_sharing=args.intrinsics_sharing)
    if args.attention_chunk_size is not None:
        cfg = replace(cfg, attention_chunk_size=args.attention_chunk_size)
    if args.no_query_chunking:
        cfg = replace(cfg, attention_chunk_size=None, visibility_chunk_size=None)
    if context.primary:
        print(json.dumps({"model_config": cfg.to_dict()}), flush=True)
    if resume is not None:
        model = MultiCamera4RC.from_checkpoint(resume, config=cfg)
    else:
        model = MultiCamera4RC(cfg)
        if args.pretrained:
            report = model.load_4rc_weights(args.pretrained, strict_base=not args.partial_pretrained)
            if context.primary:
                print(json.dumps({"pretrained": report}), flush=True)
    model = model.to(device).train()
    if args.freeze_pretrained_backbone:
        model.backbone.pretrained.requires_grad_(False)
        for name, parameter in model.backbone.pretrained.named_parameters():
            if name.endswith("camera_relation_weight"):
                parameter.requires_grad_(True)
    if args.freeze_geometry:
        model.head.requires_grad_(False)
        model.cam_dec.requires_grad_(False)
    from .optim import optimizer_groups, load_optimizer_state
    geometry_options = training_state.get('geometry_lr_options')
    if args.geometry_lr_multiplier is not None:
        requested_geometry = dict(multiplier=args.geometry_lr_multiplier,
                                  warmup_steps=(geometry_options.get('warmup_steps', 100) if geometry_options else 100)
                                  if args.geometry_lr_warmup_steps is None else args.geometry_lr_warmup_steps)
        if geometry_options is not None and requested_geometry != geometry_options:
            raise ValueError('a resumed geometry LR recipe must remain unchanged; start an explicit audited phase to change it')
        geometry_options = requested_geometry
    elif args.geometry_lr_warmup_steps is not None:
        raise ValueError('geometry-lr-warmup-steps requires geometry-lr-multiplier')
    optimizer = torch.optim.AdamW(optimizer_groups(model, args.backbone_lr, args.lr,
                                                   args.refinement_lr,
                                                   geometry_lr=args.lr if geometry_options is not None else None),
                                  weight_decay=args.weight_decay)
    step = 0
    optimizer_migration = None
    old_group_names = [g.get('name') for g in optimizer.param_groups]
    if resume is not None:
        if "optimizer" not in resume:
            raise ValueError("resume checkpoint has no optimizer; use --pretrained for weights-only initialization")
        old_group_names = [g.get('name') for g in resume['optimizer']['param_groups']]
        optimizer_migration = load_optimizer_state(optimizer, resume['optimizer'], model,
                                                   model_state=resume.get('model'), allow_legacy=True)
        step = int(resume.get("step", 0))
        if args.override_lr:
            rates = dict(backbone=args.backbone_lr, heads=args.lr,
                         refinement=args.refinement_lr, geometry=args.lr)
            for group in optimizer.param_groups:
                group['lr'] = rates[group['name']]
    time_projection_reset_step = training_state.get('time_projection_reset_step')
    if args.reset_time_projections:
        reset_names = _reset_absolute_time_projections(model, optimizer)
        time_projection_reset_step = step
        if context.primary:
            print(json.dumps(dict(reset_time_projections=reset_names, reset_step=step)), flush=True)
    dense_supervision = getattr(dataset, "supervision", None) == "dense"
    if context.primary and hasattr(dataset, "sources"):
        print(json.dumps({"dataset_plan": {"sampling": dataset.sampling, "epoch_size": len(dataset),
            "coverage_windows": dataset.coverage_size,
            "sources": [{"name": s["display_name"], "sequences": len(s["pool"]),
                "frames": sum(n for _, n in s["pool"]), "cameras": s["cameras"],
                "camera_choices": list(s["camera_choices"]), "view_mode": s["view_mode"],
                "max_times": s["num_times"], "dense_kind": s["dense_kind"],
                "layouts": s.get("layouts", []), "weight": dataset.weights[i],
                "sampling_fraction": dataset.weights[i]/sum(dataset.weights)} for i, s in enumerate(dataset.sources)]}}), flush=True)
    criterion, loss_profile = _criterion(args, dense_supervision, training_state)
    explicit_lr_phase = args.override_lr or any(v is not None for v in
        (args.lr_schedule, args.warmup_steps, args.min_lr_ratio, args.warmup_start_ratio, args.lr_end_step))
    if not explicit_lr_phase and "lr_plan" in training_state:
        lr_plan = LearningRatePlan(**training_state["lr_plan"])
    else:
        lr_plan = LearningRatePlan(mode=args.lr_schedule or "constant", start_step=step,
                    end_step=args.lr_end_step if args.lr_end_step is not None else max(step+1, args.steps),
                    base_lrs=[g["lr"] for g in optimizer.param_groups],
                    warmup_steps=args.warmup_steps or 0,
                    warmup_start_ratio=args.warmup_start_ratio or 0.,
                    min_ratio=.1 if args.min_lr_ratio is None else args.min_lr_ratio)
        if geometry_options is not None and 'geometry' in old_group_names and 'lr_plan' in training_state:
            saved_plan = LearningRatePlan(**training_state['lr_plan'])
            names = [group['name'] for group in optimizer.param_groups]
            if saved_plan.group_names != names or 'geometry' not in (saved_plan.group_multipliers or {}):
                raise ValueError('resumed geometry optimizer has no matching saved LR policy')
            rates = list(lr_plan.base_lrs)
            if not args.override_lr:
                position = names.index('geometry')
                # Checkpoints are saved after an update whose LR was applied
                # at step-1. Preserve the geometry/head ratio while the saved
                # absolute warmup policy advances to the next update.
                rates[position] /= saved_plan.group_factor('geometry', max(0, step-1))
            lr_plan = replace(lr_plan, base_lrs=rates, group_names=names,
                              group_multipliers=saved_plan.group_multipliers)
    if args.lr_end_step is not None and lr_plan.end_step < args.steps:
        raise ValueError("LR phase ends before the requested training target; start an explicit continuation phase")
    if geometry_options is not None:
        from .schedule import migrate_geometry_plan
        new_group_names = [g['name'] for g in optimizer.param_groups]
        if len(lr_plan.base_lrs) != len(new_group_names) or not getattr(lr_plan, 'group_multipliers', None):
            source_groups = (old_group_names if len(old_group_names) == len(lr_plan.base_lrs)
                             else new_group_names)
            if 'geometry' in source_groups:
                # A fresh/explicit phase was built from the already-split
                # optimizer. Recover its unsplit base plan before attaching
                # the one-time group multiplier.
                source_rates = dict(zip(source_groups, lr_plan.base_lrs))
                source_groups = [name for name in source_groups if name != 'geometry']
                lr_plan = replace(lr_plan, base_lrs=[source_rates[name] for name in source_groups],
                                  group_names=source_groups, group_multipliers=None)
            lr_plan = migrate_geometry_plan(lr_plan, source_groups, new_group_names,
                        start_step=step, multiplier=geometry_options['multiplier'],
                        warmup_steps=geometry_options['warmup_steps'])
    if context.primary:
        print(json.dumps({"loss_profile": loss_profile, "loss_config": criterion.to_config(),
                          "lr_plan": lr_plan.to_config(), "resume_step": step,
                          "optimizer_moments_restored": resume is not None,
                          "optimizer_migration": optimizer_migration,
                          "geometry_lr_options": geometry_options,
                          "data_migration": data_migration}), flush=True)
    amp = args.amp and device.type in ("cuda", "cpu")
    amp_dtype = resolve_amp_dtype(device, args.amp_dtype) if amp else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda" and amp_dtype == torch.float16)
    if resume is not None and scaler.is_enabled() and resume.get("grad_scaler"):
        scaler.load_state_dict(resume["grad_scaler"])
    training_step = _TrainingStep(model, criterion)
    if context.distributed:
        # Newer PyTorch renamed this option. Retain 2.5 compatibility without
        # triggering its replacement's deprecation warning on newer releases.
        ddp_options = {"forward_sync_buffers": False} if "forward_sync_buffers" in inspect.signature(DistributedDataParallel).parameters else {"broadcast_buffers": False}
        training_step = DistributedDataParallel(
            training_step, device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True, gradient_as_bucket_view=True, **ddp_options)
    # Parameters have already been broadcast by DDP. Vary per-rank source
    # sampling while keeping model initialization and dataset planning shared.
    torch.manual_seed(training_seed + context.rank)
    random.seed(training_seed + context.rank)
    np.random.seed((training_seed + context.rank) % (2**32))
    if device.type == "cuda":
        torch.cuda.manual_seed(training_seed + context.rank)
    context.initialize_data_sync()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    same_layout = (training_state.get("world_size", context.world_size) == context.world_size
                   and training_state.get("batch_size", args.batch_size) == args.batch_size)
    data_epoch, batches_consumed = _resume_data_cursor(
        training_state, step, len(dataset), context.world_size, args.batch_size,
        reset=args.reset_data_cursor, sampler_kind=sampler_kind)
    optimizer_updates = int(training_state.get("optimizer_updates", step))
    rank_rng = resume.get("rank_rng", []) if resume is not None else []
    if context.rank < len(rank_rng):
        restore_rng(rank_rng[context.rank], device)
    if resume is not None and not same_layout and context.primary:
        print(json.dumps({"resume_layout": {
            "previous_world_size": training_state.get("world_size"),
            "world_size": context.world_size, "batch_size": args.batch_size,
            "data_epoch": data_epoch, "batches_consumed": batches_consumed,
            "global_samples_consumed_in_epoch": batches_consumed * context.world_size * args.batch_size,
            "preserved_rank_rng": min(len(rank_rng), context.world_size),
            "new_rank_rng": "training_seed + rank", "reset_data_cursor": args.reset_data_cursor,
        }}), flush=True)

    execution = TrainingExecution(model, args.execution_optimizations)
    execution.identity["resume_policy"] = validate_resume_mode(
        training_state.get("execution_optimizations"), args.execution_optimizations)
    validate_resume_identity(training_state.get("execution_optimizations"), execution.identity)
    six_rank_proof = SixRankProof(args=args, context=context, resume=resume, model=model,
        optimizer=optimizer, lr_plan=lr_plan, criterion=criterion, scaler=scaler, execution=execution,
        data_epoch=data_epoch, batches_consumed=batches_consumed, balance=balance, dataset_size=len(dataset))

    def checkpoint(path):
        execution.assert_checkpoint_safe()
        from .datasets.dense import DENSE_TARGET_VERSION
        # All ranks contribute RNG metadata; only rank zero writes weights,
        # optimizer/scaler state and the common data cursor. Save the bare model
        # so existing native/4RC checkpoint key names remain unchanged.
        rng = context.gather_rng()
        if context.primary:
            save_checkpoint(path, model, optimizer, step=step,
                            grad_scaler=scaler.state_dict(), rank_rng=rng,
                            training_state={"execution_optimizations": execution.checkpoint_metadata(),
                                            "data_epoch": data_epoch,
                                            "batches_consumed": batches_consumed,
                                            "world_size": context.world_size,
                                            "batch_size": args.batch_size,
                                            "gradient_accumulation": args.gradient_accumulation,
                                            "effective_batch_size": context.world_size*args.batch_size*args.gradient_accumulation,
                                            "sampler_kind": sampler_kind,
                                            "load_balancing": balance,
                                            "dataset_size": len(dataset),
                                            "source_count": args.source_count,
                                            "max_source_time_pairs": args.max_source_time_pairs,
                                            "prefer_motion_sources": prefer_motion_sources,
                                            "time_projection_reset_step": time_projection_reset_step,
                                            "seed": training_seed,
                                            "optimizer_updates": optimizer_updates,
                                            "supervision": "dense" if dense_supervision else "sparse_or_manifest",
                                            "scene_normalized": dense_supervision,
                                            "dense_target_version": DENSE_TARGET_VERSION,
                                            "kubric_background_policy": getattr(dataset, "kubric_background_policy", None),
                                            "augmentation": getattr(dataset, "augmentation", None),
                                            "photometric_augmentation": getattr(dataset, "photometric_augmentation", None),
                                            "data_config_sha256": data_fingerprint,
                                            "data_migration": data_migration,
                                            "geometry_lr_options": geometry_options,
                                             "loss_weights": criterion.weights,
                                             "loss_profile": loss_profile,
                                             "loss_config": criterion.to_config(),
                                             "lr_plan": lr_plan.to_config(),
                                            "amp_dtype": str(amp_dtype).removeprefix("torch.")})

    micro_metrics, micro_records = [], []
    while step < args.steps:
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(data_epoch)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(data_epoch)
        loader_generator.manual_seed(training_seed + data_epoch)
        batch_sampler.start_batch = batches_consumed
        if balance:
            sampler.start_batch = batches_consumed
        from .performance import StepTimer
        timer = StepTimer(device, args.profile_step_timing)
        for batch_index, batch in enumerate(loader, start=batches_consumed):
            if step >= args.steps:
                break
            context.synchronize_data_ready()
            timer.mark('data_wait')
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device, non_blocking=True)
            batch["targets"] = {key: value.to(device, non_blocking=True)
                                for key, value in batch["targets"].items()}
            source = _sources(batch, args.source_count, args.max_source_time_pairs, prefer_motion_sources,
                              criterion.query_pair_source_sampling)
            if dense_supervision:
                from .datasets.dense import materialize_dense_targets
                targets = materialize_dense_targets(batch, source)
            else:
                targets = select_source_targets(batch["targets"], source, batch["images"].shape[1])
            # Native adapters publish fixed fields even when labels are
            # unavailable. Avoid a visibility forward/loss without supervision.
            targets = dict(targets)
            if criterion.query_pair_weight > 0 or criterion.query_pair_source_sampling:
                # Retain native persistent surface IDs for strict GT-only
                # pairing; these labels never enter the encoder/decoder.
                from .query_pair import attach_pair_surface_ids
                targets = attach_pair_surface_ids(targets, batch['targets'])
            for name in ("visibility", "sparse_visibility"):
                mask = targets.get(f"{name}_valid")
                if name in targets and mask is not None and not mask.any():
                    targets.pop(name)
                    targets.pop(f"{name}_valid")
            need_visibility = any(key in targets for key in ("visibility", "sparse_visibility"))
            if not micro_metrics:
                optimizer.zero_grad(set_to_none=True)
                lr_plan.apply(optimizer, step)
            execution.before_microbatch(step)
            forward_error = None
            timer.mark('prepare_targets')
            try:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                    loss, metrics = training_step(
                        batch["images"], batch["camera_id"], batch["time_idx"],
                        batch["query_time_idx"], source, batch["observation_valid"],
                        targets, need_visibility, batch.get("principal_canvas"))
            except Exception as error:
                forward_error = error
            locally_finite = forward_error is None and torch.isfinite(loss).item()
            timer.mark('forward_loss')
            if not context.all_true(locally_finite):
                if forward_error is not None:
                    raise forward_error
                if locally_finite:
                    raise RuntimeError("another distributed rank failed during forward or loss evaluation")
                raise FloatingPointError("training loss is not finite")
            timer.mark('forward_sync')
            if args.gradient_accumulation == 1:
                norm, skipped = _optimizer_step(loss, model, optimizer, scaler, args.clip_grad)
            else:
                # Reduce every microbatch, including its unused-parameter mask.
                # Different datasets can exercise different heads on each rank;
                # retaining DDP's ordinary reductions handles these transitions.
                # Gradients already accumulated are identical across ranks, so
                # averaging them again leaves their contribution unchanged.
                scaler.scale(loss/args.gradient_accumulation).backward()
            batches_consumed = batch_index + 1
            timer.mark('backward_optimizer')
            averaged = aggregate_metrics(context, metrics)
            rank_batch = dict(dataset=batch.get("dataset_name", []), clip_id=batch.get("clip_id", []),
                input_hw=list(batch["images"].shape[-2:]), observations=int(batch["observation_valid"].sum()),
                times=batch["query_time_idx"].shape[1], decoded_sources=source.shape[1])
            if 'sample_index' in batch:
                rank_batch['sample_index'] = batch['sample_index']
            if args.profile_step_timing:
                rank_batch['timing'] = dict(timer.phases)
            if criterion.joint_geometry_options is not None:
                rank_batch.update(source_indices=source.detach().cpu().tolist(),
                    camera_ids=batch['camera_id'].detach().cpu().tolist(),
                    time_indices=batch['time_idx'].detach().cpu().tolist())
            # Averaged metrics can mix unrelated datasets. Retain local error
            # measurements to attribute outliers without another forward.
            local_keys = ("loss", "depth_abs_rel", "depth_mae_normalized", "trajectory_epe_normalized",
                          "displacement_epe_dynamic_normalized", "depth_confidence_saturated_fraction",
                          "trajectory_epe_temporal_normalized", "displacement_epe_temporal_normalized",
                          "supervised_source_pixels", "supervised_trajectory_pixels",
                          "sparse_trajectory", "sparse_displacement", "sparse_velocity", "sparse_visibility",
                          "native_track_queries", "native_track_point_times", "native_temporal_point_times",
                          "native_trajectory_epe_normalized", "native_displacement_epe_normalized",
                          "native_trajectory_epe_temporal_normalized", "native_displacement_epe_temporal_normalized",
                          "native_velocity_epe_normalized", "camera_rotation_degrees",
                          "excluded_background_pixels", "excluded_background_fraction")
            local_metrics = {k: v.detach() for k, v in metrics.items()
                             if k in local_keys or k.startswith(("intrinsics_", "cross_camera_3d", "query_pair_", "joint_"))
                             or k in ("camera_log_focal", "camera_principal_normalized")}
            # One device-to-host copy, rather than a blocking scalar copy for
            # each diagnostic. No metric is removed or sampled less often.
            rank_batch['metrics'] = dict(zip(local_metrics,
                torch.stack(list(local_metrics.values())).cpu().tolist())) if local_metrics else {}
            if "augmentation" in batch:
                rank_batch["augmentation"] = batch["augmentation"]
            if "scene_scale" in targets:
                rank_batch["scene_scale"] = targets["scene_scale"].detach().cpu().tolist()
            if "depth_filter" in batch:
                rank_batch["depth_filter"] = batch["depth_filter"]
            if "static_multiview" in batch:
                rank_batch["static_multiview"] = batch["static_multiview"]
            if "native_tracks" in batch:
                rank_batch["native_tracks"] = batch["native_tracks"]
            if "syn4d" in batch:
                rank_batch["syn4d"] = batch["syn4d"]
            rank_batches = [rank_batch]
            if context.distributed:
                rank_batches = [None]*context.world_size
                dist.all_gather_object(rank_batches, rank_batch)
            micro_metrics.append(averaged)
            micro_records.extend(rank_batches)
            if len(micro_metrics) < args.gradient_accumulation:
                del loss, metrics, batch, targets
                continue
            if args.gradient_accumulation > 1:
                norm, skipped = _optimizer_step(None, model, optimizer, scaler, args.clip_grad)
                from .distributed import average_microbatch_metrics
                averaged = average_microbatch_metrics(micro_metrics)
            execution.finish_update()
            six_rank_proof.after_update(step_before=step, skipped=skipped, norm=norm, execution=execution,
                targets=targets, need_visibility=need_visibility, source_count=source.shape[1])
            averaged['grad_norm'] = float(norm.detach())
            optimizer_updates += int(not skipped)
            step += 1
            step_timing = timer.finish()
            if args.profile_step_timing and device.type == 'cuda':
                step_timing['peak_allocated_gib'] = torch.cuda.max_memory_allocated(device) / 2**30
                step_timing['peak_reserved_gib'] = torch.cuda.max_memory_reserved(device) / 2**30
            if context.primary:
                print(json.dumps({"step": step, "world_size": context.world_size,
                                  "execution_optimizations": execution.last_receipt,
                                  "step_timing": step_timing,
                                  "gradient_accumulation": args.gradient_accumulation,
                                  "effective_batch_size": context.world_size*args.batch_size*args.gradient_accumulation,
                                  "metrics_version": METRICS_VERSION,
                                  "supervision": "dense" if dense_supervision else "sparse_or_manifest",
                                  "input_hw": list(batch["images"].shape[-2:]),
                                  "rank_batches": micro_records,
                                  "data_epoch": data_epoch,
                                  "optimizer_step_skipped": bool(skipped),
                                   "amp_dtype": str(amp_dtype).removeprefix("torch."),
                                   "loss_profile": loss_profile,
                                   "backbone_lr": optimizer.param_groups[0]["lr"],
                                   "lr": optimizer.param_groups[1]["lr"],
                                  **{group['name']+'_lr': group['lr'] for group in optimizer.param_groups
                                     if group['name'] in ('refinement', 'geometry')},
                                  **averaged}, allow_nan=False), flush=True)
            micro_metrics, micro_records = [], []
            if step % args.save_every == 0 and step < args.steps:
                checkpoint(output/f"step_{step:08d}.pt")
        if batches_consumed >= batches_per_epoch:
            data_epoch += 1
            batches_consumed = 0
    checkpoint(output/"latest.pt")
    if context.primary and step % args.save_every == 0:
        from .checkpoint import publish_checkpoint_alias
        publish_checkpoint_alias(output/"latest.pt", output/f"step_{step:08d}.pt")
    return output/"latest.pt"


if __name__ == "__main__":
    main()
