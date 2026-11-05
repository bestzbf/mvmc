"""Evaluate native dense checkpoints on an immutable, disjoint validation plan."""

import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import random
import re

import torch
from torch import distributed as dist

from . import MultiCamera4RC, MultiCamera4RCConfig
from .checkpoint import config_from_checkpoint, read_checkpoint
from .datasets.bridge import prepare_clip
from .datasets.dense import DENSE_TARGET_VERSION, attach_kubric_dense_recipe, materialize_dense_targets
from .datasets.mixture import create_dataset, collate_adapter_samples, read_data_config
from .distributed import DistributedContext, resolve_amp_dtype
from .metrics import METRICS_VERSION, dense_metrics
from .losses import MultiCamera4RCLoss


def make_validation_plan(dataset, *, seed=9157, scenes_per_variant=2):
    cases = []
    for layout, source in enumerate(dataset.sources):
        if source.get('layouts'):
            raise ValueError('fixed validation needs explicit C/T sources without layouts; use a separate evaluation YAML')
        groups = {}
        for sequence, length in sorted(source["pool"]):
            match = re.search(r"(?:^|_)([CDEF])_", sequence)
            groups.setdefault(match.group(1) if match else "other", []).append((sequence, length))
        for variant, group in sorted(groups.items()):
            rng = random.Random(f"{seed}:{variant}")
            for scene, (sequence, length) in enumerate(rng.sample(group, min(scenes_per_variant, len(group)))):
                rng = random.Random(f"{seed}:{sequence}:{layout}")
                stride = rng.choice([v for v in source["strides"] if 1+(source["num_times"]-1)*v <= length])
                start = rng.randrange(length-(source["num_times"]-1)*stride)
                cases.append(dict(layout=layout, variant=variant, sequence=sequence,
                                  times=[start+t*stride for t in range(source["num_times"])],
                                  sampling_seed=seed+layout*100003+scene))
    syn4d = any(source.get('dense_kind') == 'syn4d' for source in dataset.sources)
    if syn4d:
        for case in cases:
            if dataset.sources[case['layout']].get('dense_kind') == 'syn4d':
                case.update(renderer_expected_observations(dataset.sources[case['layout']], case))
    return {"format_version": 3 if syn4d else 1, "seed": seed, "scenes_per_variant": scenes_per_variant,
            "layouts": [[s["cameras"], s["num_times"]] for s in dataset.sources], "cases": cases}


def fixed_sources(batch, count):
    valid = batch["observation_valid"][0]
    cameras, times = batch["camera_id"][0], batch["time_idx"][0]
    choices = []
    endpoints = (times[valid].min(), times[valid].max())
    camera_values = cameras[valid].unique().sort().values
    # Two simultaneous views at each endpoint allow cross-camera evaluation.
    for time in endpoints:
        for camera in camera_values[:2]:
            slots = (valid & (times == time) & (cameras == camera)).nonzero().flatten()
            if len(slots):
                choices.append(int(slots[0]))
    choices = list(dict.fromkeys(choices))
    choices += [int(i) for i in valid.nonzero().flatten() if int(i) not in choices]
    return torch.tensor([choices[:min(count, int(valid.sum()))]], device=valid.device)


def summarize_records(records):
    if not records:
        raise ValueError("evaluation needs at least one record")
    def average(rows):
        keys = sorted(set().union(*(r["metrics"] for r in rows)))
        return {key: sum(r["metrics"][key] for r in rows if key in r["metrics"])/sum(key in r["metrics"] for r in rows) for key in keys}
    groups, scenes = {}, {}
    for record in records:
        groups.setdefault(str(record["layout"]), []).append(record)
        scenes.setdefault(record["sequence"], []).append(record)
    return {"clips": len(records), "scenes": len(scenes), "mean": average(records),
            "by_layout": {key: average(rows) for key, rows in groups.items()},
            "by_sequence": {key: average(rows) for key, rows in scenes.items()}}


def prepare_renderer_case(dataset, case):
    """Materialize an immutable Kubric or Syn4D evaluation observation layout."""
    source = dataset.sources[case["layout"]]
    if source["dense_kind"] not in ("kubric", "syn4d"):
        raise ValueError("renderer evaluation supports native Kubric/Syn4D sources")
    clip = source["adapter"].load_clip(case["sequence"], case["times"]*source["cameras"],
         layout_num_cameras=source["cameras"], layout_num_times=source["num_times"], sampling_seed=case["sampling_seed"])
    sample = prepare_clip(clip, image_mode=dataset.image_mode, image_size=dataset.image_size, num_queries=0)
    if source["dense_kind"] == "syn4d":
        from .datasets.syn4d_dense import attach_syn4d_dense_recipe
        attach_syn4d_dense_recipe(clip, sample, image_mode=dataset.image_mode, include_visibility=True)
    else:
        attach_kubric_dense_recipe(clip, sample, image_mode=dataset.image_mode, include_visibility=True,
                                  background_policy=dataset.kubric_background_policy)
    return sample


def renderer_expected_observations(source, case):
    """Resolve sampling positions to physical frames before checking the model input.

    Syn4D's admitted/reference frames can contain gaps; its adapter correctly
    returns original frame numbers. A position in that list is not a timestamp.
    """
    C, T = source['cameras'], source['num_times']
    times = case['times']
    result = {}
    if source.get('dense_kind') == 'syn4d':
        import numpy as np
        info = source['adapter'].get_sequence_info(case['sequence'])
        frames = info['frame_indices']
        if any(isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < len(frames) for t in times):
            raise ValueError('Syn4D plan positions are outside the admitted frame list')
        times = [frames[t] for t in times]
        cameras = sorted(np.random.default_rng(case['sampling_seed']).choice(sorted(info['cameras']), C, replace=False).tolist())
        result['expected_camera_ids'] = [camera for camera in cameras for _ in range(T)]
    result['expected_time_indices'] = times*C
    return result


def verify_renderer_observations(sample, case, source):
    """Check the exact loaded observations, not just a held-out sequence name."""
    cameras, length = source['cameras'], source['num_times']
    if len(case['times']) != length or len(set(case['times'])) != length:
        raise ValueError('renderer plan has a different number of distinct times')
    resolved = renderer_expected_observations(source, case)
    for key, values in resolved.items():
        if key in case and case[key] != values:
            raise ValueError('renderer metadata differs from the frozen plan: '+key)
    if sample['time_idx'].tolist() != resolved['expected_time_indices']:
        raise ValueError('loaded renderer times differ from the fixed plan')
    ids = sample['camera_id'].tolist()
    if 'expected_camera_ids' in resolved and ids != resolved['expected_camera_ids']:
        raise ValueError('loaded renderer cameras differ from the fixed plan')
    if (len(ids) != cameras*length or len(set(ids)) != cameras
            or any(len(set(ids[i*length:(i+1)*length])) != 1 for i in range(cameras))):
        raise ValueError('loaded renderer camera layout differs from the fixed plan')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="LABEL=PATH entries")
    parser.add_argument("--plan", required=True, help="Create once, then reuse exact cameras, frames and sources")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=9157)
    parser.add_argument("--scenes-per-variant", type=int, default=2)
    parser.add_argument("--source-count", type=int, default=4)
    parser.add_argument("--decoder-memory", choices=("all_observations", "target_time"),
                        help="Explicit ablation override; otherwise evaluate each checkpoint's saved scope")
    parser.add_argument("--time-origin", choices=("clip_start", "absolute"),
                        help="Explicit ablation override; otherwise preserve checkpoint time encoding")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    if min(args.scenes_per_variant, args.source_count, args.cpu_threads) <= 0:
        parser.error("scene/source counts and CPU threads must be positive")
    context = DistributedContext.initialize(args.device)
    try:
        torch.set_num_threads(args.cpu_threads)
        data = read_data_config(args.data_config)
        if data.get("supervision") != "dense":
            raise ValueError("renderer-derived dense evaluation requires supervision: dense")
        training = create_dataset(data)
        train_sequences = set().union(*(set(s for s, _ in source["pool"]) for source in training.sources))
        for source in data["sources"]:
            source["split"] = "val"
        dataset = create_dataset(data)
        val_sequences = set().union(*(set(s for s, _ in source["pool"]) for source in dataset.sources))
        if train_sequences & val_sequences:
            raise ValueError("validation sequences overlap training; configure a disjoint split")
        del training
        plan_path = Path(args.plan)
        plan = json.loads(plan_path.read_text()) if plan_path.exists() else make_validation_plan(dataset, seed=args.seed, scenes_per_variant=args.scenes_per_variant)
        layouts = [[s["cameras"], s["num_times"]] for s in dataset.sources]
        if plan.get("layouts") != layouts or any(c["sequence"] not in val_sequences for c in plan["cases"]):
            raise ValueError("validation plan does not match the held-out dataset/layouts")
        if any(c['sequence'] not in dict(dataset.sources[c['layout']]['pool']) for c in plan['cases']):
            raise ValueError('validation sequence does not belong to its requested layout pool')
        if context.primary:
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            if not plan_path.exists():
                plan_path.write_text(json.dumps(plan, indent=2)+"\n")
        plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        result = {"stage": "running", "metrics_version": METRICS_VERSION,
                  "plan_sha256": plan_hash, "source_count": args.source_count,
                  "data_config_sha256": hashlib.sha256(Path(args.data_config).read_bytes()).hexdigest(),
                  "train_sequences": len(train_sequences), "val_sequences": len(val_sequences),
                  "sequence_overlap": 0, "checkpoints": {}}
        dtype = resolve_amp_dtype(context.device, args.amp_dtype)
        for entry in args.checkpoints:
            label, path = entry.split("=", 1)
            torch.manual_seed(args.seed)
            payload = (torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                       if Path(path).suffix != ".safetensors" else read_checkpoint(path))
            cfg = config_from_checkpoint(payload)
            if args.decoder_memory is not None:
                cfg = replace(cfg, decoder_memory=args.decoder_memory)
            if args.time_origin is not None:
                cfg = replace(cfg, time_origin=args.time_origin)
            cfg = replace(cfg, gradient_checkpointing=False, attention_chunk_size=None, visibility_chunk_size=None)
            model = MultiCamera4RC(cfg)
            loading = model.load_4rc_weights(payload)
            loss_config = payload.get("training_state", {}).get("loss_config")
            criterion = MultiCamera4RCLoss(**loss_config) if loss_config else MultiCamera4RCLoss.dense()
            step = int(payload.get("step", 0))
            del payload
            model = model.to(context.device).eval()
            records = []
            for index in range(context.rank, len(plan["cases"]), context.world_size):
                case = plan["cases"][index]
                source = dataset.sources[case["layout"]]
                sample = prepare_renderer_case(dataset, case)
                verify_renderer_observations(sample, case, source)
                batch = collate_adapter_samples([sample])
                for key, value in list(batch.items()):
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(context.device)
                batch["targets"] = {key: value.to(context.device) for key, value in batch["targets"].items()}
                sources = fixed_sources(batch, args.source_count)
                with torch.inference_mode():
                    target = materialize_dense_targets(batch, sources)
                    if criterion.query_pair_weight > 0 or criterion.query_pair_source_sampling:
                        from .query_pair import attach_pair_surface_ids
                        target = attach_pair_surface_ids(target, batch['targets'])
                    with torch.autocast(device_type=context.device.type, dtype=dtype):
                        pred = model(batch["images"], batch["camera_id"], batch["time_idx"], batch["query_time_idx"],
                                     sources, observation_valid=batch["observation_valid"])
                    metrics = {key: float(value) for key, value in criterion(pred, target).items()}
                    metrics.update({key: float(value) for key, value in dense_metrics(pred, target).items()})
                record = dict(case=index, sequence=case["sequence"], variant=case["variant"],
                    planned_times=case['times'], sampling_seed=case['sampling_seed'],
                    layout=[source["cameras"], source["num_times"]], source_indices=sources.tolist()[0],
                    camera_ids=batch["camera_id"].tolist()[0], time_indices=batch["time_idx"].tolist()[0],
                    background_filter_enabled=bool(batch["targets"]["dense_filter_background"].all()) if "dense_filter_background" in batch["targets"] else False,
                    scene_scale=target["scene_scale"].tolist(), metrics=metrics)
                if "syn4d" in sample:
                    record["syn4d"] = sample["syn4d"]
                records.append(record)
                print(json.dumps({"evaluation": label, "rank": context.rank, **record}), flush=True)
                del pred, target, batch, sample
            if context.distributed:
                all_records = [None]*context.world_size
                dist.all_gather_object(all_records, records)
                records = [record for rank_records in all_records for record in rank_records]
            records.sort(key=lambda r: r["case"])
            if context.primary:
                result["checkpoints"][label] = {"path": path, "step": step, "loading": loading,
                                               "model_config": cfg.to_dict(),
                                               "loss_config": criterion.to_config(),
                                               "dense_target_version": DENSE_TARGET_VERSION,
                                               "kubric_background_policy": dataset.kubric_background_policy,
                                               "amp_dtype": str(dtype).removeprefix("torch."),
                                               **summarize_records(records), "records": records}
                (output/"evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
                print(json.dumps({"evaluation_summary": label, **result["checkpoints"][label]["mean"]}), flush=True)
            del model
            gc.collect()
            if context.device.type == "cuda":
                torch.cuda.empty_cache()
        if context.primary:
            result["stage"] = "passed"
            (output/"evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        return output/"evaluation.json"
    finally:
        context.close()


if __name__ == "__main__":
    main()
