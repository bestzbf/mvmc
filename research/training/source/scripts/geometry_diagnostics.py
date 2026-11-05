"""Additional GT-assisted diagnostics; never alter the frozen benchmark scores."""
from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import torch

from scripts.monocular_protocol import MAX_DEPTH, align_depth_per_frame


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def archive_prediction(path, prediction, frame_indices):
    """Atomically retain the exact evaluation-grid arrays for later diagnosis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    depth, intrinsics, c2w = prediction
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, depth=depth.detach().float().cpu().numpy(),
                            K=intrinsics.detach().float().cpu().numpy(),
                            c2w=c2w.detach().float().cpu().numpy(),
                            frame_indices=np.asarray(frame_indices, dtype=np.int64))
    temporary.replace(path)
    return file_sha256(path)


def verify_prediction(path, expected_sha256, frame_indices):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or file_sha256(path) != expected_sha256:
        raise ValueError('archived prediction is missing or changed: '+str(path))
    with np.load(path, allow_pickle=False) as arrays:
        if not np.array_equal(arrays['frame_indices'], np.asarray(frame_indices)):
            raise ValueError('archived prediction frame identity changed')
        count = len(frame_indices)
        if (arrays['depth'].ndim != 3 or arrays['depth'].shape[0] != count
                or arrays['K'].shape != (count, 3, 3)
                or arrays['c2w'].shape != (count, 4, 4)
                or not all(np.isfinite(arrays[key]).all() for key in ('depth', 'K', 'c2w'))
                or (arrays['depth'] <= 0).any()):
            raise ValueError('archived prediction has invalid geometry')


def prefix_clip(clip, length):
    if not 1 <= length <= len(clip.frame_indices):
        raise ValueError('invalid clip prefix length')
    return replace(clip, images=clip.images[:, :length], depth=clip.depth[:length],
                   K=clip.K[:length], c2w=clip.c2w[:length],
                   frame_indices=clip.frame_indices[:length])


def depth_diagnostics(pred_depth, pred_K, pred_c2w, clip, *, dataset):
    """Separate clamp sensitivity, near-depth errors and inter-frame scale drift.

    All extra measurements use evaluation GT and are explicitly diagnostic.
    The S/SS results from geometry_metrics remain untouched.
    """
    limit = MAX_DEPTH[dataset]
    truth = clip.depth.float()
    prediction = pred_depth.detach().float().cpu()
    mask = torch.isfinite(truth) & (truth > 0) & (truth < limit)
    if prediction.shape != truth.shape or not mask.flatten(1).any(1).all():
        raise ValueError('diagnostics require matching depth and nonempty GT in every frame')
    if not torch.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError('diagnostics refuse invalid model depths')
    aligned_s = align_depth_per_frame(prediction, truth, mask, 'scale', limit)
    aligned_ss = align_depth_per_frame(prediction, truth, mask, 'scale_shift', limit)
    unbounded = prediction.clone()
    scales, biases, records = [], [], []
    for index in range(len(prediction)):
        valid = mask[index]
        p, g = prediction[index][valid], truth[index][valid]
        scale = (g/p).median()
        scales.append(float(scale))
        unbounded[index] *= scale
        clipped = p.clamp(1e-6, limit)
        pc, gc = clipped-clipped.median()+1e-6, g-g.median()+1e-6
        eligible = torch.isfinite(pc) & torch.isfinite(gc) & (pc.abs() > 1e-6)
        ss_scale = (gc[eligible]/pc[eligible]).median() if eligible.any() else (g/clipped).median()
        ss_shift = float((g-ss_scale*clipped).median()) if torch.isfinite(ss_scale) and ss_scale > 0 else 0.
        biases.append(ss_shift)
        records.append(dict(frame=int(clip.frame_indices[index]), raw_scale=float(scale),
                            ss_shift=ss_shift,
                            raw_above_limit_fraction=float((p > limit).float().mean()),
                            ss_nonpositive_fraction=float((aligned_ss[index][valid] <= 0).float().mean())))
    values = prediction[mask]
    quantiles = torch.quantile(values, torch.tensor([0., .01, .5, .99, 1.]))
    log_scales = torch.tensor(scales).log()
    errors = (aligned_s-truth).abs()/truth.clamp_min(1e-9)
    ss_errors = (aligned_ss.clamp_min(1e-6)-truth).abs()/truth.clamp_min(1e-9)
    unbounded_errors = (unbounded-truth).abs()/truth.clamp_min(1e-9)
    bucket_values = {key: [] for key in ('near', 'middle', 'far')}
    bucket_ss = {key: [] for key in bucket_values}
    for index in range(len(prediction)):
        valid = mask[index]
        median = truth[index][valid].median()
        regions = dict(near=valid & (truth[index] < .5*median),
                       middle=valid & (truth[index] >= .5*median) & (truth[index] <= 2*median),
                       far=valid & (truth[index] > 2*median))
        for name, region in regions.items():
            if region.any():
                bucket_values[name].append(errors[index][region])
                bucket_ss[name].append(ss_errors[index][region])
    buckets = {}
    for name, tensors in bucket_values.items():
        count = sum(t.numel() for t in tensors)
        buckets[name] = dict(pixels=count,
                             scale_abs_rel=float(torch.cat(tensors).mean()) if count else None,
                             scale_shift_abs_rel=float(torch.cat(bucket_ss[name]).mean()) if count else None)
    focal = torch.stack((pred_K[:, 0, 0], pred_K[:, 1, 1]), -1)
    gt_focal = torch.stack((clip.K[:, 0, 0], clip.K[:, 1, 1]), -1)
    return dict(kind='gt_assisted_diagnostic_not_official_score',
                raw_depth_quantiles=dict(zip(('min','p01','median','p99','max'), map(float, quantiles))),
                raw_above_limit_fraction=float((prediction[mask] > limit).float().mean()),
                ss_nonpositive_fraction=float((aligned_ss[mask] <= 0).float().mean()),
                scale_abs_rel_without_raw_upper_clamp=float(unbounded_errors[mask].mean()),
                raw_scale_log_std=float(log_scales.std(unbiased=False)),
                raw_scale_min=min(scales), raw_scale_max=max(scales),
                focal_abs_rel=float(((focal-gt_focal).abs()/gt_focal).mean()),
                principal_error_eval_pixels=float((pred_K[:, :2, 2]-clip.K[:, :2, 2]).norm(dim=-1).mean()),
                depth_buckets=buckets, frames=records)
