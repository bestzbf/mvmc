"""Evaluate MultiCamera4RC on legacy D4RT Sintel / ScanNet geometry benchmarks."""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from multicamera4rc import MultiCamera4RC
from multicamera4rc.checkpoint import config_from_checkpoint
from scripts.monocular_protocol import (SINTEL_14_SEQUENCES, frame_indices, load_clip,
                                        geometry_metrics, geometry_on_eval_grid)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def build_plan(args):
    import h5py
    roots = [Path(p).resolve() for p in args.roots]
    cases = []
    if args.dataset == 'sintel':
        names = args.scene_names or list(SINTEL_14_SEQUENCES)
        for name in names:
            matches = [r for r in roots if (r/'training/final'/name).is_dir()]
            if len(matches) != 1:
                raise ValueError('Sintel scene must resolve to exactly one root: '+name)
            n = len(list((matches[0]/'training/final'/name).glob('frame_*.png')))
            cases.append(dict(dataset='sintel', root=str(matches[0]), scene=name,
                frame_indices=frame_indices(n, args.num_frames, args.frame_stride, args.start_frame, dataset='sintel')))
    else:
        found = {}
        for r in roots:
            for p in sorted(r.glob('scene*')):
                if not p.is_dir(): continue
                if p.name in found: raise ValueError('duplicate ScanNet scene: '+p.name)
                found[p.name] = r
        names = args.scene_names or sorted(found)
        for name in names:
            r = found[name]
            with h5py.File(r/name/'processed/precomputed.h5', 'r') as f:
                n = int(f['num_frames'][()])
            cases.append(dict(dataset='scannet', root=str(r), scene=name,
                frame_indices=frame_indices(n, args.num_frames, args.frame_stride, args.start_frame, dataset='scannet')))
    if args.max_scenes:
        cases = cases[:args.max_scenes]
    if not cases: raise ValueError('empty benchmark')
    return dict(format_version=1, dataset=args.dataset, eval_size=args.eval_size,
                long_side=args.long_side, cases=cases)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=('sintel', 'scannet'))
    p.add_argument('--roots', nargs='+', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--scene-names', nargs='+')
    p.add_argument('--num-frames', type=int, default=48)
    p.add_argument('--frame-stride', type=int, default=1)
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--eval-size', type=int, default=256)
    p.add_argument('--long-side', type=int, default=518, help='Aspect-preserving input maximum; 0 keeps native RGB')
    p.add_argument('--max-scenes', type=int)
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--memory-fraction', type=float, default=.8)
    p.add_argument('--resume-results', action='store_true')
    p.add_argument('--save-predictions', action=argparse.BooleanOptionalAction, default=True,
                   help='Archive exact depth/K/c2w arrays for reproducible diagnostics')
    p.add_argument('--diagnostics', action=argparse.BooleanOptionalAction, default=True,
                   help='Record near/far errors, raw-depth clamp sensitivity and scale drift without changing official scores')
    p.add_argument('--context-lengths', type=int, nargs='+', default=None,
                   help='Optional additional forwards; compare the same prefix frames under different input contexts')
    args = p.parse_args(argv)
    if args.cpu_threads <= 0 or args.long_side < 0 or args.eval_size < 2 or not 0 < args.memory_fraction <= 1:
        p.error('invalid resource / resolution settings')
    if not args.audit_only and not args.checkpoint:
        p.error('--checkpoint is required for inference')
    if args.context_lengths and any(length < 2 for length in args.context_lengths):
        p.error('context lengths must be at least two frames')
    torch.set_num_threads(args.cpu_threads)
    if args.plan.exists():
        plan = json.loads(args.plan.read_text())
        if any(plan[k] != getattr(args, k) for k in ('dataset', 'eval_size', 'long_side')):
            raise ValueError('requested settings differ from frozen plan')
    else:
        plan = build_plan(args)
        write_json(args.plan, plan)
    plan_sha = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        audit = dict(passed=False, plan_sha256=plan_sha, requested_cases=len(plan['cases']), cases=[], failed_cases=[])
        for case in plan['cases']:
            try:
                clip = load_clip(case, eval_size=args.eval_size, long_side=args.long_side)
                limit = 70 if args.dataset == 'sintel' else 10
                mask = torch.isfinite(clip.depth) & (clip.depth > 0) & (clip.depth < limit)
                row = dict(scene=case['scene'], frame_indices=case['frame_indices'], raw_hw=clip.raw_hw,
                    content_hw=clip.content_hw, input_shape=list(clip.images.shape), calibration_sha256=clip.calibration_sha256,
                    valid_depth_fraction=float(mask.float().mean()), valid_depth_pixels=int(mask.sum()))
                audit['cases'].append(row)
                del clip
                print(json.dumps(dict(audited=row['scene'], progress=len(audit['cases']))), flush=True)
            except Exception as e:
                audit['failed_cases'].append(dict(scene=case['scene'], error=repr(e)))
            write_json(args.output/'gt_audit.json', audit)
        audit['passed'] = not audit['failed_cases'] and len(audit['cases']) == len(plan['cases'])
        write_json(args.output/'gt_audit.json', audit)
        if not audit['passed']: raise RuntimeError('benchmark GT audit failed; see gt_audit.json')
        return audit
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device.index if device.index is not None else 0)
    payload = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)
    cfg = replace(config_from_checkpoint(payload), gradient_checkpointing=False,
                  attention_chunk_size=None, visibility_chunk_size=None)
    model = MultiCamera4RC(cfg); model.load_4rc_weights(payload)
    step = int(payload['step']); del payload
    model.to(device).eval()
    from scripts.geometry_diagnostics import file_sha256, archive_prediction, verify_prediction, depth_diagnostics, prefix_clip
    checkpoint_digest = file_sha256(args.checkpoint)
    diagnostic_settings = dict(save_predictions=args.save_predictions, diagnostics=args.diagnostics,
                               context_lengths=sorted(set(args.context_lengths or [])))
    report = dict(stage='running', checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_step=step,
        checkpoint_sha256=checkpoint_digest, diagnostic_settings=diagnostic_settings,
        plan_sha256=plan_sha, model_config=json.loads(json.dumps(cfg.to_dict())), amp_dtype='bfloat16' if device.type == 'cuda' else 'float32',
        protocol=dict(dataset=args.dataset, single_camera=True, long_side=args.long_side,
            eval_grid=[args.eval_size]*2, input_resize='aspect_preserving_then_bottom_right_patch_padding',
            depth='D4RT per-frame median scale / median-centred scale+shift; all valid pixels',
            gt_depth_max=70 if args.dataset == 'sintel' else 10, pose='camera_head; legacy D4RT Sim3 ATE/RPE and adjacent-pair AUC@30',
            pointcloud='reference camera; one clip depth scale + mean shift; separate Sim3 result',
            confidence_filter=False, intrinsics_input=False, sparse_queries=False,
            legacy_input_resolution_differs=True, legacy_pose_estimator_differs=True,
            legacy_mean_shift_only_uses_a_different_scale_gauge=True), records=[], failed_cases=[])
    source_root = Path(__file__).resolve().parents[1]
    report['evaluation_source_sha256'] = {
        relative: file_sha256(source_root/relative) for relative in (
            'scripts/evaluate_monocular.py', 'scripts/monocular_protocol.py',
            'scripts/geometry_diagnostics.py', 'utils/camera.py', 'utils/metrics.py')}
    audit_path = args.output/'gt_audit.json'
    audit = json.loads(audit_path.read_text())
    if not audit['passed'] or audit['plan_sha256'] != plan_sha:
        raise ValueError('requires passed GT audit for the exact evaluation plan')
    audit_rows = {r['scene']: r for r in audit['cases']}
    path = args.output/'evaluation.json'
    if args.resume_results and path.exists():
        old = json.loads(path.read_text())
        if any(old[k] != report[k] for k in ('checkpoint', 'plan_sha256', 'model_config', 'protocol')):
            raise ValueError('cannot resume a different checkpoint or benchmark protocol')
        if old.get('checkpoint_sha256') != checkpoint_digest or old.get('diagnostic_settings') != diagnostic_settings:
            raise ValueError('checkpoint bytes or diagnostic settings changed; use a fresh output directory')
        if old.get('evaluation_source_sha256') != report['evaluation_source_sha256']:
            raise ValueError('evaluation source changed; use a fresh output directory')
        cases = {case['scene']: case for case in plan['cases']}
        if len({row['scene'] for row in old['records']}) != len(old['records']):
            raise ValueError('duplicate scenes in resumed results')
        for row in old['records']:
            case = cases.get(row['scene'])
            if case is None or row['frame_indices'] != case['frame_indices']:
                raise ValueError('resumed scene/frame identity differs from frozen plan')
            if args.diagnostics and not isinstance(row.get('diagnostics'), dict):
                raise ValueError('resumed result is missing requested diagnostics')
            if args.context_lengths:
                lengths = sorted(set(min(length, len(case['frame_indices'])) for length in args.context_lengths))
                contexts = row.get('context_diagnostics', {}).get('records', [])
                if ([record.get('input_frames') for record in contexts] != lengths
                        or any(record.get('frame_indices') != case['frame_indices'][:min(lengths)] for record in contexts)):
                    raise ValueError('resumed result is missing matched-frame context diagnostics')
            if args.save_predictions:
                if not row.get('prediction_file'):
                    raise ValueError('old results have no archived predictions; use a fresh output directory')
                expected_path = 'predictions/'+row['scene']+'.npz'
                if row['prediction_file'] != expected_path:
                    raise ValueError('unexpected archived prediction path')
                verify_prediction(args.output/expected_path, row['prediction_sha256'], case['frame_indices'])
                # Validate recorded numbers from immutable arrays, without a
                # second model forward or trusting edited metric JSON.
                clip = load_clip(case, eval_size=args.eval_size, long_side=args.long_side, patch_size=cfg.patch_size)
                if clip.calibration_sha256 != audit_rows[case['scene']]['calibration_sha256']:
                    raise ValueError('GT changed after the recorded audit')
                with np.load(args.output/expected_path, allow_pickle=False) as arrays:
                    archived = tuple(torch.from_numpy(arrays[key].copy()) for key in ('depth', 'K', 'c2w'))
                checked = geometry_metrics(*archived, clip, dataset=args.dataset)
                if set(checked) != set(row['metrics']) or any(
                        not math.isclose(float(value), float(row['metrics'][name]), rel_tol=1e-6, abs_tol=1e-8)
                        for name, value in checked.items()):
                    raise ValueError('resumed metrics disagree with archived predictions')
                del clip, archived
        report['records'] = old['records']
    done = {r['scene'] for r in report['records']}
    write_json(path, report)
    for case in plan['cases']:
        if case['scene'] in done: continue
        began = time.monotonic()
        try:
            clip = load_clip(case, eval_size=args.eval_size, long_side=args.long_side, patch_size=cfg.patch_size)
            if clip.calibration_sha256 != audit_rows[case['scene']]['calibration_sha256']:
                raise ValueError('GT calibration/depth changed after audit')
            images = clip.images.to(device)
            cameras = torch.zeros(images.shape[:2], device=device, dtype=torch.long)
            times = torch.tensor(clip.frame_indices, device=device, dtype=torch.long)[None]
            if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                encoded = model.encode(images, cameras, times)
            prediction = geometry_on_eval_grid(encoded.geometry, clip)
            peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
            del encoded, images, cameras, times
            if device.type == 'cuda': torch.cuda.empty_cache()
            metrics = geometry_metrics(*prediction, clip, dataset=args.dataset)
            row = dict(scene=case['scene'], root=case['root'], frame_indices=clip.frame_indices,
                actual_times=len(clip.frame_indices), input_shape=list(clip.images.shape), raw_hw=list(clip.raw_hw),
                content_hw=list(clip.content_hw), metrics=metrics, seconds=time.monotonic()-began, peak_cuda_bytes=peak)
            if args.save_predictions:
                relative_path = Path('predictions')/(case['scene']+'.npz')
                row['prediction_sha256'] = archive_prediction(args.output/relative_path, prediction, clip.frame_indices)
                row['prediction_file'] = relative_path.as_posix()
            if args.diagnostics:
                row['diagnostics'] = depth_diagnostics(*prediction, clip, dataset=args.dataset)
            if args.context_lengths:
                lengths = sorted(set(min(length, len(clip.frame_indices)) for length in args.context_lengths))
                score_length = min(lengths)
                score_clip = prefix_clip(clip, score_length)
                context_rows = []
                for length in lengths:
                    if length == len(clip.frame_indices):
                        context_prediction = tuple(value[:score_length] for value in prediction)
                    else:
                        context_clip = prefix_clip(clip, length)
                        context_images = context_clip.images.to(device)
                        context_cameras = torch.zeros(context_images.shape[:2], device=device, dtype=torch.long)
                        context_times = torch.tensor(context_clip.frame_indices, device=device, dtype=torch.long)[None]
                        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                            context_scene = model.encode(context_images, context_cameras, context_times)
                        arrays = geometry_on_eval_grid(context_scene.geometry, context_clip)
                        context_prediction = tuple(value[:score_length] for value in arrays)
                        del context_scene, context_images, context_cameras, context_times, arrays
                    context_rows.append(dict(input_frames=length, scored_frames=score_length,
                        frame_indices=score_clip.frame_indices,
                        metrics=geometry_metrics(*context_prediction, score_clip, dataset=args.dataset)))
                row['context_diagnostics'] = dict(kind='matched_scoring_frames_not_official_score', records=context_rows)
            report['records'].append(row)
            print(json.dumps(dict(completed=len(report['records']), **row)), flush=True)
            del clip, prediction
        except Exception:
            report['failed_cases'].append(dict(scene=case['scene'], error=traceback.format_exc()))
            print(json.dumps(report['failed_cases'][-1]), flush=True)
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
        write_json(path, report)
    order = {case['scene']: index for index, case in enumerate(plan['cases'])}
    report['records'].sort(key=lambda row: order[row['scene']])
    rows = report['records']
    report['mean_metrics'] = {k:sum(r['metrics'][k] for r in rows)/len(rows) for k in rows[0]['metrics']} if rows else {}
    report['num_requested_scenes'], report['num_scenes'] = len(plan['cases']), len(rows)
    report['stage'] = 'passed' if not report['failed_cases'] and len(rows) == len(plan['cases']) else 'failed'
    write_json(path, report)
    if report['stage'] != 'passed': raise RuntimeError('evaluation has failed/missing scenes; see evaluation.json')
    return report


if __name__ == '__main__': main()
