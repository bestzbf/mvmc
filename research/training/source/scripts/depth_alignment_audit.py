"""Separate alignment diagnostics; never replace frozen benchmark scores.

The valid population is defined exclusively by GT. A failed/nonpositive
aligned prediction remains in the denominator, using the existing fixed log
floor. No GT-dependent epsilon or prediction-dependent pixel removal is used.
"""
import torch

from scripts.monocular_protocol import align_depth_per_frame
from utils.metrics import compute_depth_metrics


def least_squares_alignment(pred, gt, mask, max_depth):
    result = pred.clamp(1e-6, max_depth).clone()
    parameters = []
    for i in range(len(pred)):
        p, g = result[i][mask[i]].double(), gt[i][mask[i]].double()
        if not len(p) or not torch.isfinite(p).all() or not torch.isfinite(g).all():
            raise ValueError('least-squares alignment requires finite nonempty GT support')
        centered = p-p.mean()
        variance = centered.square().sum()
        scale = ((centered*(g-g.mean())).sum()/variance).clamp_min(0) if variance > 1e-20 else p.new_zeros(())
        shift = g.mean()-scale*p.mean()
        result[i] = result[i]*scale.to(pred.dtype)+shift.to(pred.dtype)
        parameters.append(dict(scale=float(scale), shift=float(shift)))
    return result, parameters


def alignment_audit(pred, gt, mask, max_depth):
    rows = {}
    for name, aligned in (
            ('legacy_scale', align_depth_per_frame(pred, gt, mask, 'scale', max_depth)),
            ('legacy_scale_shift', align_depth_per_frame(pred, gt, mask, 'scale_shift', max_depth)),
            ('diagnostic_least_squares_scale_shift', least_squares_alignment(pred, gt, mask, max_depth)[0])):
        row = {key:float(value) for key,value in compute_depth_metrics(aligned, gt, mask=mask, scale_invariant=False).items()}
        row.update(valid_pixels=int(mask.sum()), nonpositive_pixels=int(((aligned <= 0) & mask).sum()),
                   nonpositive_fraction=float(((aligned <= 0) & mask).sum()/mask.sum()))
        rows[name] = row
    return dict(formal_scores_changed=False, prediction_failures_excluded=False, fixed_log_floor=1e-6, variants=rows)


def content_canvas_intrinsics(pred_k, *, padded_hw, content_hw, eval_hw):
    """CPU reproduction of an explicit content-canvas inference ablation.

    Reconstruct the saved dimensionless learned principal offset, then change
    its canvas. Focal length and learned offsets are preserved. This ablation
    can double-correct a head trained to compensate padding; it is not enabled
    in default inference or used to overwrite an archived score.
    """
    out = pred_k.clone()
    for axis, old, new, evaluation in ((0, padded_hw[1], content_hw[1], eval_hw[1]),
                                       (1, padded_hw[0], content_hw[0], eval_hw[0])):
        native = (out[..., axis, 2]+.5)*new/evaluation-.5
        delta = (native-old/2)/old
        changed = new/2+delta*new
        out[..., axis, 2] = (changed+.5)*evaluation/new-.5
    return out
