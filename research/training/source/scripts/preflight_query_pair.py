"""One-GPU native-resolution giant-model forward/backward; no Adam updates.

The parent owns the GPU UUID lease. Optimizer moments are restored for a real
memory budget, but no source checkpoint or training process is modified.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import time

import torch

from multicamera4rc import MultiCamera4RC
from multicamera4rc.checkpoint import config_from_checkpoint
from multicamera4rc.datasets.dense import materialize_dense_targets
from multicamera4rc.datasets.mixture import create_dataset, read_data_config, collate_adapter_samples
from multicamera4rc.losses import MultiCamera4RCLoss
from multicamera4rc.query_pair import attach_pair_surface_ids
from multicamera4rc.train import _TrainingStep, _sources
from scripts.cutover_native_motion_stereo import write_json


def stress_specs(config):
    names = ('kubric_variable', 'mv_kubric_official', 'syn4d_v1', 'syn4d_sim',
             'vkitti2_stereo', 'pointodyssey', 'panoptic_calibration', 'blended')
    specs = []
    for name in names:
        spec = next((s for s in config['sources'] if s['name'] == name), None)
        if spec is None: raise ValueError('missing stress source '+name)
        layouts = spec.get('layouts') or [spec]
        if not spec.get('layouts') and isinstance(spec['cameras'], list):
            layouts = [dict(cameras=C, num_times=spec['num_times']) for C in spec['cameras']]
        largest = max(layouts, key=lambda x: (x['cameras'], x['num_times']))
        multi = [v for v in layouts if v['cameras'] > 1] or layouts
        longest = max(multi, key=lambda x: (x['num_times'], x['cameras']))
        for C, T in sorted({(v['cameras'], v['num_times']) for v in (largest, longest)}):
            value = deepcopy(spec)
            value.update(cameras=C, num_times=T)
            if value.get('view_mode') == 'static_multiview':
                value.pop('layouts', None)
            else:
                value['layouts'] = [dict(cameras=C, num_times=T)]
            if 'index_workers' in value.setdefault('options', {}): value['options']['index_workers'] = 2
            specs.append(value)
    return specs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('config', 'checkpoint', 'loss-config', 'output', 'gpu-uuid'):
        parser.add_argument('--'+flag, required=True)
    args = parser.parse_args(argv)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid:
        raise ValueError('preflight requires exactly the parent-leased GPU UUID')
    torch.set_num_threads(4); torch.cuda.set_device(0)
    config = read_data_config(args.config)
    payload = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)
    cfg = replace(config_from_checkpoint(payload), attention_chunk_size=None, visibility_chunk_size=None,
                  gradient_checkpointing=True)
    model = MultiCamera4RC.from_checkpoint(payload, config=cfg).cuda().train()
    groups = [[], []]
    for name, parameter in model.named_parameters():
        groups[0 if name.startswith('backbone.pretrained.') and not name.endswith('camera_relation_weight') else 1].append(parameter)
    optimizer = torch.optim.AdamW([dict(params=values) for values in groups])
    optimizer.load_state_dict(payload['optimizer'])
    baseline = MultiCamera4RCLoss(**payload['training_state']['loss_config'])
    candidate = MultiCamera4RCLoss(**json.loads(Path(args.loss_config).read_text()))
    report = dict(passed=False, optimizer_update_performed=False, optimizer_restored=True,
        checkpoint=args.checkpoint, step=payload['step'], gpu_uuid=args.gpu_uuid, rows=[])
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report); del payload
    for spec in stress_specs(config):
        data = create_dataset(dict(config, sources=[spec], epoch_size=1, skip_invalid=False, invalid_log=None))
        sample = data[0]; batch = collate_adapter_samples([sample])
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch['targets'] = {k: v.cuda() for k, v in batch['targets'].items()}
        torch.manual_seed(49)
        source = _sources(batch, 4, 64, True, True)
        target = attach_pair_surface_ids(materialize_dense_targets(batch, source), batch['targets'])
        for key in ('visibility', 'sparse_visibility'):
            if key in target and not target[key+'_valid'].any():
                target.pop(key); target.pop(key+'_valid')
        need_visibility = any(k in target for k in ('visibility', 'sparse_visibility'))
        row = dict(source=spec['name'], cameras=spec['cameras'], times=spec['num_times'],
            clip_id=sample['clip_id'], input_shape=list(batch['images'].shape), source_indices=source.tolist(), arms={})
        for label, criterion in (('baseline', baseline), ('candidate', candidate)):
            model.zero_grad(set_to_none=True); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(49); torch.cuda.manual_seed(49)
            start = time.monotonic(); step = _TrainingStep(model, criterion)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, metrics = step(batch['images'], batch['camera_id'], batch['time_idx'], batch['query_time_idx'],
                    source, batch['observation_valid'], target, need_visibility, batch.get('principal_canvas'))
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            torch.cuda.synchronize()
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite preflight loss')
            row['arms'][label] = dict(loss=float(loss.detach()), grad_norm=float(norm), finite_backward=True,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30, seconds=time.monotonic()-start,
                metrics={k: float(v) for k, v in metrics.items() if k.startswith(('query_pair_', 'cross_camera_3d'))})
            del loss, metrics, step
        report['rows'].append(row); write_json(output, report); print(json.dumps(row), flush=True)
        model.zero_grad(set_to_none=True)
        del sample, batch, target, source, data
        gc.collect(); torch.cuda.empty_cache()
    report['passed'] = True; write_json(output, report)


if __name__ == '__main__': main()
