"""Masked supervision for synchronized, uncalibrated multi-camera 4RC.

Ground-truth poses and trajectories must already use the same world gauge as
the predictions. Visibility scores never mask geometry or motion supervision.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _resize_scalar(value: Tensor, size: tuple[int, int], *, mask: bool = False) -> Tensor:
    if value.shape[-2:] == size:
        return value.bool() if mask else value
    leading = value.shape[:-2]
    image = value.reshape(-1, 1, *value.shape[-2:]).float()
    resized = (F.interpolate(image, size=size, mode="nearest") if mask
               else F.interpolate(image, size=size, mode="bilinear", align_corners=False))
    result = resized.reshape(*leading, *size)
    return result.bool() if mask else result.to(value.dtype)


def _resize_vector(value: Tensor, size: tuple[int, int]) -> Tensor:
    if value.shape[-3:-1] == size:
        return value
    leading, channels = value.shape[:-3], value.shape[-1]
    image = value.reshape(-1, *value.shape[-3:]).permute(0, 3, 1, 2)
    resized = F.interpolate(image.float(), size=size, mode="bilinear", align_corners=False)
    return resized.permute(0, 2, 3, 1).reshape(*leading, *size, channels).to(value.dtype)


def _valid_mask(target: Tensor, supplied: Tensor | None, *, vector: bool = False) -> Tensor:
    finite = torch.isfinite(target).all(-1) if vector else torch.isfinite(target)
    if supplied is None:
        return finite
    mask = supplied.to(device=target.device, dtype=torch.bool)
    if vector and mask.shape == target.shape:
        mask = mask.all(-1)
    try:
        return finite & torch.broadcast_to(mask, finite.shape)
    except RuntimeError as exc:
        raise ValueError(f"valid mask {tuple(mask.shape)} cannot cover target {tuple(finite.shape)}") from exc


def isolate_absolute_intrinsics_gradient(pred: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Identical XYZ values, but no direct K gradient from absolute XYZ loss.

    Recompute only the reference base, not the motion decoder. Subtracting the
    original base path and adding the K-detached base retains depth, extrinsic
    and motion derivatives. Visibility logits and direct K/ray losses retain
    their original graph. Shared feature gradients are intentionally not frozen.
    """
    from .geometry import unproject_depth
    from .metadata import gather_observations
    required = {'depth', 'K', 'c2w', 'base_points', 'source_obs_idx', 'source_valid', 'trajectory'}
    if required-set(pred):
        raise ValueError('intrinsics isolation needs model geometry and source metadata')
    with torch.autocast(device_type=pred['K'].device.type, enabled=False):
        isolated = unproject_depth(pred['depth'], pred['K'].detach(), pred['c2w'])
        if 'observation_valid' in pred:
            isolated = torch.where(pred['observation_valid'][..., None, None, None], isolated, 0)
        correction = gather_observations(isolated-pred['base_points'], pred['source_obs_idx'])
        correction = torch.where(pred['source_valid'][..., None, None, None], correction, 0)
        return dict(pred, trajectory=pred['trajectory']+correction[:, :, None])


class MultiCamera4RCLoss(nn.Module):
    """Combine available labels; raise when a batch supplies no valid supervision.

    Dense target fields may use their native source-image size: predictions are
    resized to that size, and K is scaled from prediction pixels to target pixels.
    Confidence inputs are the heads' positive ``1 + exp(raw)`` outputs, rather
    than raw logits. Source-time identity slots train geometry without applying
    motion-confidence weighting or its negative log regularizer.
    """

    def __init__(self, depth_weight: float = 1.0, camera_weight: float = 1.0,
                 ray_weight: float = 1.0, trajectory_weight: float = 1.0,
                 displacement_weight: float = 1.0, visibility_weight: float = 0.1,
                 static_weight: float = 0.1, confidence_alpha: float = 0.2,
                 depth_gradient_weight: float = 0.0, velocity_weight: float = 0.0,
                 regression_norm: str = "l1", foreground_weight: float = 1.0,
                 visibility_projection_mask: bool = True,
                 regression_norms: Mapping[str, str] | None = None,
                 unweighted_regression_weight: float = 0.0,
                 foreground_fraction: float | None = None,
                 motion_confidence_targets=("trajectory", "displacement", "velocity"),
                 depth_gradient_scales=(1,), depth_gradient_confidence: bool = True,
                 camera_loss_mode: str = "matrix",
                 intrinsics_loss_mode: str = "legacy",
                 focal_loss_weight: float = .5, principal_loss_weight: float = .5,
                 focal_ratio_loss_weight: float = 0.,
                 absolute_intrinsics_gradient: str = "full",
                 trajectory_motion_gradient_scale: float = 1.0,
                 cross_camera_3d_weight: float = 0.0,
                 query_pair_weight: float = 0.0,
                 query_pair_source_sampling: bool | None = None,
                 geometry_source_min_pixels: int = 0,
                 query_pair_options: Mapping | None = None,
                 joint_geometry_options: Mapping | None = None,
                 reprojection_options: Mapping | None = None,
                 depth_objective_options: Mapping | None = None,
                 auxiliary_motion_options: Mapping | None = None,
                 weights: Mapping[str, float] | None = None, *,
                 pseudo_trajectory_base_gradient_scale: float = 1.0):
        super().__init__()
        self.weights = {"depth": depth_weight, "camera": camera_weight, "ray": ray_weight,
                        "trajectory": trajectory_weight, "displacement": displacement_weight,
                        "visibility": visibility_weight, "static": static_weight,
                        "depth_gradient": depth_gradient_weight, "velocity": velocity_weight}
        if weights is not None:
            unknown = set(weights) - set(self.weights)
            if unknown:
                raise ValueError(f"unknown loss weights: {sorted(unknown)}")
            self.weights.update(weights)
        if any(not math.isfinite(weight) or weight < 0 for weight in self.weights.values()) or not math.isfinite(confidence_alpha) or confidence_alpha < 0:
            raise ValueError("loss weights and confidence_alpha must be nonnegative")
        self.confidence_alpha = confidence_alpha
        norms = {"l1", "l2", "l1_sum"}
        if regression_norm not in norms or not math.isfinite(foreground_weight) or foreground_weight < 1:
            raise ValueError("regression_norm must be l1/l2 and foreground_weight at least 1")
        self.regression_norm, self.foreground_weight = regression_norm, foreground_weight
        self.visibility_projection_mask = visibility_projection_mask
        self.regression_norms = dict(regression_norms or {})
        if set(self.regression_norms)-set(self.weights) or any(n not in norms for n in self.regression_norms.values()):
            raise ValueError("regression_norms must name known tasks and l1/l2/l1_sum norms")
        if not math.isfinite(unweighted_regression_weight) or unweighted_regression_weight < 0:
            raise ValueError("unweighted_regression_weight must be finite and nonnegative")
        if foreground_fraction is not None and (not math.isfinite(foreground_fraction) or not 0 < foreground_fraction < 1):
            raise ValueError("foreground_fraction must be strictly between zero and one")
        self.unweighted_regression_weight = unweighted_regression_weight
        self.foreground_fraction = foreground_fraction
        self.motion_confidence_targets = tuple(motion_confidence_targets)
        if set(self.motion_confidence_targets)-{"trajectory", "displacement", "velocity"}:
            raise ValueError("motion confidence targets must be trajectory/displacement/velocity")
        self.depth_gradient_scales = tuple(depth_gradient_scales)
        if not self.depth_gradient_scales or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in self.depth_gradient_scales):
            raise ValueError("depth_gradient_scales must contain positive integer strides")
        self.depth_gradient_confidence = depth_gradient_confidence
        if camera_loss_mode not in ("matrix", "components"):
            raise ValueError("camera_loss_mode must be matrix/components")
        self.camera_loss_mode = camera_loss_mode
        if intrinsics_loss_mode not in ("legacy", "log_focal"):
            raise ValueError("intrinsics_loss_mode must be legacy/log_focal")
        if intrinsics_loss_mode != "legacy" and camera_loss_mode != "matrix":
            raise ValueError("independent intrinsics retain matrix extrinsics supervision")
        if any(not math.isfinite(v) or v < 0 for v in (focal_loss_weight, principal_loss_weight)):
            raise ValueError("intrinsics weights must be finite and nonnegative")
        self.intrinsics_loss_mode = intrinsics_loss_mode
        self.focal_loss_weight, self.principal_loss_weight = focal_loss_weight, principal_loss_weight
        if (isinstance(focal_ratio_loss_weight, bool) or not math.isfinite(focal_ratio_loss_weight)
                or focal_ratio_loss_weight < 0):
            raise ValueError('focal ratio weight must be finite and nonnegative')
        if focal_ratio_loss_weight and intrinsics_loss_mode != 'log_focal':
            raise ValueError('focal ratio supervision requires log_focal intrinsics')
        self.focal_ratio_loss_weight = float(focal_ratio_loss_weight)
        if absolute_intrinsics_gradient not in ("full", "isolated"):
            raise ValueError("absolute_intrinsics_gradient must be full or isolated")
        self.absolute_intrinsics_gradient = absolute_intrinsics_gradient
        if (not math.isfinite(trajectory_motion_gradient_scale)
                or not 0 <= trajectory_motion_gradient_scale <= 1):
            raise ValueError('trajectory_motion_gradient_scale must be in [0,1]')
        self.trajectory_motion_gradient_scale = float(trajectory_motion_gradient_scale)
        if (isinstance(pseudo_trajectory_base_gradient_scale, bool)
                or not math.isfinite(pseudo_trajectory_base_gradient_scale)
                or not 0 <= pseudo_trajectory_base_gradient_scale <= 1):
            raise ValueError('pseudo_trajectory_base_gradient_scale must be in [0,1]')
        self.pseudo_trajectory_base_gradient_scale = float(pseudo_trajectory_base_gradient_scale)
        if not math.isfinite(cross_camera_3d_weight) or cross_camera_3d_weight < 0:
            raise ValueError('cross_camera_3d_weight must be finite and nonnegative')
        self.cross_camera_3d_weight = float(cross_camera_3d_weight)
        from .query_pair import QueryPairOptions
        if not math.isfinite(query_pair_weight) or query_pair_weight < 0:
            raise ValueError('query_pair_weight must be finite and nonnegative')
        if query_pair_source_sampling is not None and not isinstance(query_pair_source_sampling, bool):
            raise ValueError('query_pair_source_sampling must be boolean or None')
        self.query_pair_weight = float(query_pair_weight)
        if (isinstance(geometry_source_min_pixels, bool) or not isinstance(geometry_source_min_pixels, int)
                or geometry_source_min_pixels < 0):
            raise ValueError('geometry source pixel threshold must be a nonnegative integer')
        self.geometry_source_min_pixels = geometry_source_min_pixels
        self.query_pair_source_sampling = (query_pair_weight > 0 if query_pair_source_sampling is None
                                           else query_pair_source_sampling)
        self.query_pair_options = QueryPairOptions(**dict(query_pair_options or {}))
        from .joint_geometry import JointGeometryOptions
        from .reprojection import ReprojectionOptions
        self.reprojection_options = (None if reprojection_options is None else
                                     ReprojectionOptions(**dict(reprojection_options)))
        self.joint_geometry_options = (None if joint_geometry_options is None else
                                       JointGeometryOptions(**dict(joint_geometry_options)))
        from .depth_objectives import DepthObjectiveOptions
        self.depth_objective_options = (None if depth_objective_options is None else
                                        DepthObjectiveOptions(**dict(depth_objective_options)))
        from .auxiliary_motion import AuxiliaryMotionOptions
        self.auxiliary_motion_options = (None if auxiliary_motion_options is None else
                                         AuxiliaryMotionOptions(**dict(auxiliary_motion_options)))

    def to_config(self):
        reprojection = asdict(self.reprojection_options) if self.reprojection_options is not None else None
        if reprojection is not None and not reprojection["stereo_certified_weight"]:
            del reprojection["stereo_certified_weight"]
        return dict(weights=dict(self.weights), confidence_alpha=self.confidence_alpha,
                    regression_norm=self.regression_norm, foreground_weight=self.foreground_weight,
                    visibility_projection_mask=self.visibility_projection_mask,
                    regression_norms=dict(self.regression_norms),
                    unweighted_regression_weight=self.unweighted_regression_weight,
                    foreground_fraction=self.foreground_fraction,
                    motion_confidence_targets=list(self.motion_confidence_targets),
                    depth_gradient_scales=list(self.depth_gradient_scales),
                    depth_gradient_confidence=self.depth_gradient_confidence,
                    camera_loss_mode=self.camera_loss_mode,
                    **(dict(focal_ratio_loss_weight=self.focal_ratio_loss_weight)
                       if self.focal_ratio_loss_weight else {}),
                    **(dict(geometry_source_min_pixels=self.geometry_source_min_pixels)
                       if self.geometry_source_min_pixels else {}),
                    **(dict(reprojection_options=reprojection) if reprojection is not None else {}),
                    **(dict(joint_geometry_options=asdict(self.joint_geometry_options))
                       if self.joint_geometry_options is not None else {}),
                    **(dict(depth_objective_options=asdict(self.depth_objective_options))
                       if self.depth_objective_options is not None else {}),
                    **(dict(auxiliary_motion_options=asdict(self.auxiliary_motion_options))
                       if self.auxiliary_motion_options is not None else {}),
                    **(dict(cross_camera_3d_weight=self.cross_camera_3d_weight)
                       if self.cross_camera_3d_weight else {}),
                    **(dict(query_pair_weight=self.query_pair_weight,
                            query_pair_source_sampling=self.query_pair_source_sampling,
                            query_pair_options=asdict(self.query_pair_options))
                       if self.query_pair_weight or self.query_pair_source_sampling else {}),
                    **(dict(absolute_intrinsics_gradient=self.absolute_intrinsics_gradient)
                       if self.absolute_intrinsics_gradient != "full" else {}),
                    **(dict(trajectory_motion_gradient_scale=self.trajectory_motion_gradient_scale)
                       if self.trajectory_motion_gradient_scale != 1 else {}),
                    **(dict(pseudo_trajectory_base_gradient_scale=self.pseudo_trajectory_base_gradient_scale)
                       if self.pseudo_trajectory_base_gradient_scale != 1 else {}),
                    **(dict(intrinsics_loss_mode=self.intrinsics_loss_mode,
                            focal_loss_weight=self.focal_loss_weight,
                            principal_loss_weight=self.principal_loss_weight)
                       if self.intrinsics_loss_mode != "legacy" else {}))

    @classmethod
    def dense(cls):
        """4RC spatial/temporal constraints with OmniX L2 and foreground weights.

        Unlike a point-query objective, all valid source pixels participate.
        Foreground weights are normalized by their sum to preserve loss scale.
        """
        return cls(depth_gradient_weight=1.0, velocity_weight=1.0,
                   regression_norm="l2", foreground_weight=10.0,
                    visibility_projection_mask=False)

    @classmethod
    def balanced_dense(cls):
        """Full-pixel foreground/background balance and factorized motion supervision.

        These are experiment defaults, not weights claimed by either paper.
        The ordinary residual plus the confidence residual follows OmniX code.
        """
        return cls(trajectory_weight=.25, static_weight=0,
                   depth_gradient_weight=1., velocity_weight=1.,
                   regression_norm="l2", regression_norms={"ray": "l1_sum"},
                   foreground_fraction=.5, unweighted_regression_weight=1.,
                   motion_confidence_targets=("displacement",),
                   depth_gradient_scales=(1, 2, 4, 8), depth_gradient_confidence=False,
                   camera_loss_mode="components", visibility_projection_mask=False)

    def _foreground_weights(self, target, valid):
        if "trajectory_foreground" not in target or (self.foreground_weight == 1 and self.foreground_fraction is None):
            return None
        foreground = torch.broadcast_to(target["trajectory_foreground"].to(valid.device).bool(), valid.shape)
        if self.foreground_fraction is not None:
            # Give each source/time map the same class budget regardless of how
            # much of the image the moving object occupies. No pixel is dropped.
            fg = (foreground & valid).sum((-2, -1), keepdim=True).float()
            bg = (~foreground & valid).sum((-2, -1), keepdim=True).float()
            both = (fg > 0) & (bg > 0)
            w_fg = torch.where(both, self.foreground_fraction*(fg+bg)/fg.clamp_min(1), 1.)
            w_bg = torch.where(both, (1-self.foreground_fraction)*(fg+bg)/bg.clamp_min(1), 1.)
            return torch.where(foreground, w_fg, w_bg)
        return torch.where(foreground, float(self.foreground_weight), 1.0)

    def _absolute_trajectory(self, trajectory, displacement):
        """Preserve residual values and base gradients; attenuate only motion.

        This applies to absolute-world and other-camera XYZ objectives. Direct
        displacement, velocity, and query-pair supervision retain full gradients.
        The shared image encoder still couples geometry and motion features.
        """
        if self.trajectory_motion_gradient_scale == 1 or not torch.is_grad_enabled():
            return trajectory
        if displacement is None or displacement.shape != trajectory.shape:
            raise ValueError('absolute trajectory routing requires matching displacement')
        return trajectory + (self.trajectory_motion_gradient_scale - 1) * (displacement - displacement.detach())

    def _regression(self, prediction: Tensor, target: Tensor, valid: Tensor,
                    confidence: Tensor | None = None,
                    confidence_eligible: Tensor | None = None,
                     pixel_weights: Tensor | None = None,
                     task: str | None = None) -> tuple[Tensor, Tensor]:
        # Select before arithmetic: NaN padding in invalid targets cannot leak.
        p, t = prediction[valid].float(), target[valid].float()
        if not torch.isfinite(p).all():
            raise FloatingPointError("non-finite prediction at a supervised location")
        error = (p - t).abs()
        norm = self.regression_norms.get(task, self.regression_norm)
        if prediction.ndim == valid.ndim + 1:
            error = (torch.linalg.vector_norm(p-t, dim=-1) if norm == "l2"
                     else error.sum(-1) if norm == "l1_sum" else error.mean(-1))
        confidence_penalty = error.new_zeros(())
        penalty = torch.zeros_like(error)
        if confidence is not None:
            q = confidence[valid].float()
            if not torch.isfinite(q).all():
                raise FloatingPointError("non-finite confidence at a supervised location")
            # Bound the confidence weight when an exactly fitted regression
            # has zero residual; -log(q) otherwise has no finite optimum.
            q = q.clamp(min=1.0, max=1e4)
            eligible = (torch.ones_like(q, dtype=torch.bool) if confidence_eligible is None
                        else torch.broadcast_to(confidence_eligible, valid.shape)[valid])
            effective = torch.where(eligible, q, torch.ones_like(q))
            error = error * (effective+self.unweighted_regression_weight)
            penalty = -self.confidence_alpha * effective.log()
        if pixel_weights is None:
            return error.mean(), penalty.mean()
        weight = torch.broadcast_to(pixel_weights, valid.shape)[valid].float()
        denominator = weight.sum().clamp_min(1e-8)
        return (error*weight).sum()/denominator, (penalty*weight).sum()/denominator

    @staticmethod
    def _observation_mask(valid: Tensor, pred: Mapping[str, Tensor]) -> Tensor:
        if "observation_valid" in pred:
            observation = pred["observation_valid"].bool().to(valid.device)
            if valid.shape[:2] != observation.shape:
                raise ValueError("observation_valid must match the supervised observation axes")
            valid = valid & observation.reshape(*observation.shape, *((1,) * (valid.ndim - 2)))
        return valid

    @staticmethod
    def _source_mask(valid: Tensor, pred: Mapping[str, Tensor]) -> Tensor:
        if "source_valid" in pred:
            source = pred["source_valid"].bool().to(valid.device)
            if valid.shape[:2] != source.shape:
                raise ValueError("source_valid must match the supervised source axes")
            valid = valid & source.reshape(*source.shape, *((1,) * (valid.ndim - 2)))
        return valid

    @staticmethod
    def _confidence_eligible(pred: Mapping[str, Tensor], target_shape: tuple[int, ...]) -> Tensor | None:
        if "source_time_idx" not in pred or "query_time_idx" not in pred:
            return None
        source, query = pred["source_time_idx"], pred["query_time_idx"]
        if source.ndim != 2 or query.ndim != 2 or source.shape[0] != query.shape[0]:
            raise ValueError("source_time_idx/query_time_idx must be [B,Q] and [B,T]")
        eligible = source[:, :, None] != query[:, None, :]
        if eligible.shape != target_shape[:3]:
            raise ValueError("time indices do not match motion query dimensions")
        return eligible.reshape(*eligible.shape, *((1,) * (len(target_shape) - 3)))

    def _camera_components(self, pred, target):
        from .vendor.utils.transform import mat_to_quat

        terms = {}
        for name in ("c2w", "K"):
            if name not in target:
                continue
            estimate = pred[name]
            truth = target[name].to(estimate.device)
            shape = (4, 4) if name == "c2w" else (3, 3)
            if truth.ndim != 4 or truth.shape[-2:] != shape or estimate.shape != truth.shape:
                raise ValueError(f"component camera {name} needs matching [B,S,{shape[0]},{shape[1]}]")
            valid = torch.isfinite(truth).all(dim=(-2, -1))
            if "camera_valid" in target:
                valid &= target["camera_valid"].to(valid.device).bool()
            valid = self._observation_mask(valid, pred)
            if not valid.any():
                continue
            p, t = estimate[valid].float(), truth[valid].float()
            if not torch.isfinite(p).all():
                raise FloatingPointError("non-finite component camera prediction")
            if name == "c2w":
                translation_valid = target.get("camera_translation_valid")
                translation_valid = (torch.ones_like(valid) if translation_valid is None else
                    torch.broadcast_to(translation_valid.to(valid.device).bool(), valid.shape))
                translated = translation_valid[valid]
                if translated.any():
                    terms["camera_translation"] = (p[translated, :3, 3]-t[translated, :3, 3]).abs().mean()
                if "camera_encoding" in pred:
                    q = pred["camera_encoding"][valid][..., 3:7].float()
                    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
                    identity = torch.zeros_like(q)
                    identity[..., 3] = 1
                    q = torch.where(norm > 1e-8, q/norm.clamp_min(1e-8), identity)
                else:
                    q = mat_to_quat(p[..., :3, :3])
                gt_q = mat_to_quat(t[..., :3, :3])
                # Quaternion antipodes represent the same physical rotation.
                terms["camera_rotation"] = torch.minimum((q-gt_q).abs().mean(-1), (q+gt_q).abs().mean(-1)).mean()
            else:
                height, width = target["depth"].shape[-2:] if "depth" in target else pred["depth"].shape[-2:]
                source_h, source_w = pred["depth"].shape[-2:]
                ratio = p.new_tensor((width/source_w, height/source_h, 1))
                principal = p[..., :2, 2]*ratio[:2] + (ratio[:2]-1)/2
                gt_focal = torch.stack((t[..., 1, 1], t[..., 0, 0]), -1)
                if (gt_focal <= 0).any():
                    raise ValueError("supervised camera focal lengths must be positive")
                half_size = p.new_tensor((height/2, width/2))
                gt_fov = 2*torch.atan(half_size/gt_focal)
                if "camera_encoding" in pred:
                    fov = pred["camera_encoding"][valid][..., 7:9].float().clamp(0, math.pi-1e-4)
                else:
                    focal = torch.stack((p[..., 1, 1]*ratio[1], p[..., 0, 0]*ratio[0]), -1)
                    fov = 2*torch.atan(half_size/focal.clamp_min(1e-8))
                terms["camera_fov"] = (fov-gt_fov).abs().mean()
                normalizer = p.new_tensor((width, height))
                terms["camera_principal"] = ((principal-t[..., :2, 2])/normalizer).abs().mean()
        return terms

    def _sparse_losses(self, pred: Mapping[str, Tensor], target: Mapping[str, Tensor],
                       zero: Tensor) -> tuple[dict[str, Tensor], Tensor, Tensor, bool]:
        """Sample explicitly matched source pixels; never pair pixels across views."""
        terms = {f"sparse_{name}": zero for name in ("trajectory", "displacement", "visibility", "velocity")}
        total, regularizers, supervised = zero, zero, False
        enabled = [name for name in ("trajectory", "displacement", "visibility")
                   if f"sparse_{name}" in target and self.weights[name] > 0]
        if not enabled and not ("sparse_displacement" in target and self.weights["velocity"] > 0):
            return terms, total, regularizers, supervised
        if "sparse_queries" not in target:
            raise KeyError("sparse supervision needs sparse_queries [B,N,3]=[source_obs,u,v]")
        # Deferred import keeps the dense geometry/loss API usable independently.
        from .tasks import rescale_queries_to_prediction, sample_tracks
        queries = target["sparse_queries"].to(zero.device)
        gt_query_valid = target.get("sparse_query_valid")
        if gt_query_valid is not None:
            gt_query_valid = gt_query_valid.to(zero.device).bool()
        if "trajectory" in pred:
            queries, gt_query_valid = rescale_queries_to_prediction(
                queries, pred["trajectory"].shape[-3:-1], target, gt_query_valid
            )
        sampled = sample_tracks(pred, queries, strict=False)
        if gt_query_valid is not None:
            sampled = dict(sampled, query_valid=sampled["query_valid"].bool() & gt_query_valid)
        metadata = dict(sampled)
        if "query_time_idx" in pred:
            metadata["query_time_idx"] = pred["query_time_idx"]
        query_valid = sampled["query_valid"].bool()
        for name in enabled:
            key = f"sparse_{name}"
            truth = target[key].to(zero.device)
            if name == "visibility":
                valid = _valid_mask(truth, target.get("sparse_visibility_valid"))
                valid = valid & query_valid[..., None, None]
                # Sources without visibility annotations intentionally skip the
                # visibility head. Unknown labels must not require its outputs.
                if not valid.any():
                    continue
                if "visibility_logits" not in sampled or "projection_valid" not in sampled:
                    raise KeyError("sparse visibility needs sampled logits and projection_valid")
                logits = sampled["visibility_logits"]
                if truth.ndim != 4 or logits.shape != truth.shape:
                    raise ValueError("sparse_visibility must match [B,N,T,C] sampled logits")
                if self.visibility_projection_mask:
                    valid &= sampled["projection_valid"].bool()
                if not valid.any():
                    continue
                labels = truth[valid].float()
                if ((labels < 0) | (labels > 1)).any():
                    raise ValueError("sparse visibility labels must lie in [0,1]")
                if not torch.isfinite(logits[valid]).all():
                    raise FloatingPointError("non-finite supervised sparse visibility logits")
                terms[key] = F.binary_cross_entropy_with_logits(logits[valid].float(), labels)
                total = total + self.weights[name] * terms[key]
            else:
                if name not in sampled:
                    raise KeyError(f"sparse {name} supervision needs sampled {name}")
                estimate = sampled[name]
                if name == 'trajectory':
                    displacement = sampled.get('displacement')
                    estimate = self._absolute_trajectory(estimate, displacement)
                    is_pseudo = target.get('sparse_trajectory_is_pseudo')
                    if (is_pseudo is not None and self.pseudo_trajectory_base_gradient_scale != 1
                            and torch.is_grad_enabled()):
                        if is_pseudo.dtype != torch.bool or is_pseudo.shape != estimate.shape[:1]:
                            raise ValueError('sparse_trajectory_is_pseudo must be boolean [B]')
                        if displacement is None or displacement.shape != estimate.shape:
                            raise ValueError('pseudo trajectory gradient routing requires matching displacement')
                        base = sampled[name] - displacement
                        # An exact zero in the forward pass. Direct calibration,
                        # dense geometry, and relative motion remain supervised.
                        base_gradient = torch.where(is_pseudo.to(zero.device)[:, None, None, None],
                                                    base - base.detach(), 0.)
                        estimate = estimate + (self.pseudo_trajectory_base_gradient_scale - 1) * base_gradient
                if truth.ndim != 4 or truth.shape[-1] != 3 or estimate.shape != truth.shape:
                    raise ValueError(f"{key} must match [B,N,T,3] sampled {name}")
                supplied = target.get(f"{key}_valid", target.get("sparse_trajectory_valid"))
                valid = _valid_mask(truth, supplied, vector=True) & query_valid[..., None]
                if not valid.any():
                    continue
                confidence = sampled.get("motion_confidence") if name in self.motion_confidence_targets else None
                eligible = self._confidence_eligible(metadata, tuple(valid.shape))
                if confidence is not None and eligible is None:
                    raise ValueError("sparse motion-confidence supervision needs source/query time indices")
                fit, regularizer = self._regression(estimate, truth, valid, confidence, eligible, task=name)
                terms[key] = fit
                total = total + self.weights[name] * (fit + regularizer)
                regularizers = regularizers + self.weights[name] * regularizer
            supervised = True
        if "sparse_displacement" in target and self.weights["velocity"] > 0 and pred["query_time_idx"].shape[1] > 1:
            truth = target["sparse_displacement"].to(zero.device).float()
            estimate = sampled["displacement"].float()
            valid = _valid_mask(truth, target.get("sparse_displacement_valid"), vector=True) & query_valid[..., None]
            order = pred["query_time_idx"].argsort(1)
            interval = pred["query_time_idx"].gather(1, order).diff(dim=1).float()
            if (interval <= 0).any():
                raise ValueError("native track velocity requires unique frame labels")
            ids = order[:, None, :, None].expand_as(truth)
            truth, estimate = truth.gather(2, ids), estimate.gather(2, ids)
            valid = valid.gather(2, ids[..., 0])
            pairs = valid[..., 1:] & valid[..., :-1]
            if pairs.any():
                confidence = sampled.get("motion_confidence") if "velocity" in self.motion_confidence_targets else None
                if confidence is not None:
                    eligible = self._confidence_eligible(metadata, tuple(query_valid.shape)+(order.shape[1],))
                    confidence = torch.where(eligible, confidence, 1.).gather(2, ids[..., 0])
                    confidence = (confidence[..., 1:]+confidence[..., :-1])/2
                dt = interval[:, None, :, None]
                fit, regularizer = self._regression(estimate.diff(dim=2)/dt, truth.diff(dim=2)/dt,
                    pairs, confidence, task="velocity")
                terms["sparse_velocity"] = fit
                total += self.weights["velocity"]*(fit+regularizer)
                regularizers += self.weights["velocity"]*regularizer
                supervised = True
        from .metrics import native_track_metrics
        terms.update(native_track_metrics(sampled, target))
        return terms, total, regularizers, supervised

    def forward(self, pred: Mapping[str, Tensor], target: Mapping[str, Tensor]) -> dict[str, Tensor]:
        if not isinstance(pred, Mapping) or not isinstance(target, Mapping):
            raise TypeError("pred and target must be mappings")
        if (self.absolute_intrinsics_gradient == "isolated" and torch.is_grad_enabled()
                and (self.weights['trajectory'] > 0 or self.cross_camera_3d_weight > 0
                     or self.query_pair_weight > 0) and 'trajectory' in pred
                and ('trajectory' in target or 'sparse_trajectory' in target)):
            pred = isolate_absolute_intrinsics_gradient(pred)
        tensor = next((value for value in pred.values() if isinstance(value, Tensor) and value.is_floating_point()), None)
        if tensor is None:
            raise ValueError("pred contains no floating-point tensors")
        zero = tensor.new_zeros((), dtype=torch.float32)
        terms = {name: zero for name in self.weights}
        confidence_term = zero
        total = zero
        supervised = False

        for name, vector, confidence_key in (("depth", False, "depth_conf"),
                                              ("ray", True, "ray_conf"),
                                              ("trajectory", True, "motion_confidence"),
                                              ("displacement", True, "motion_confidence")):
            if name not in target or self.weights[name] == 0:
                continue
            if name not in pred:
                raise KeyError(f"target supplies {name} but pred does not")
            truth = target[name].to(device=tensor.device)
            expected_ndim = {"depth": 4, "ray": 5, "trajectory": 6, "displacement": 6}[name]
            allowed_channels = (3, 6) if name == "ray" else (3,)
            if truth.ndim != expected_ndim or (vector and truth.shape[-1] not in allowed_channels):
                raise ValueError(f"{name} target has an incompatible shape")
            size = tuple(truth.shape[-3:-1] if vector else truth.shape[-2:])
            estimate = (_resize_vector(pred[name], size) if vector
                        else _resize_scalar(pred[name], size))
            if name == 'trajectory' and self.trajectory_motion_gradient_scale != 1:
                estimate = self._absolute_trajectory(estimate, _resize_vector(pred['displacement'], size))
            if estimate.shape != truth.shape:
                raise ValueError(f"{name} prediction and target shapes disagree")
            supplied = target.get(f"{name}_valid")
            if supplied is None and name == "displacement":
                supplied = target.get("trajectory_valid")
            valid = _valid_mask(truth, supplied, vector=vector)
            if name == "depth":
                valid = valid & (truth > 0)
            motion = name in ("trajectory", "displacement")
            valid = self._source_mask(valid, pred) if motion else self._observation_mask(valid, pred)
            if not valid.any():
                continue
            confidence = pred.get(confidence_key)
            if motion and name not in self.motion_confidence_targets:
                confidence = None
            if confidence is not None:
                confidence = _resize_scalar(confidence, size)
                if confidence.shape != valid.shape:
                    raise ValueError(f"{confidence_key} shape does not match {name}")
            eligible = self._confidence_eligible(pred, tuple(valid.shape)) if motion else None
            if motion and confidence is not None and eligible is None:
                raise ValueError("motion-confidence supervision needs source_time_idx and query_time_idx to exclude identity slots")
            pixel_weights = self._foreground_weights(target, valid) if motion else None
            fit, regularizer = self._regression(estimate, truth, valid, confidence, eligible, pixel_weights, task=name)
            terms[name] = fit
            confidence_term = confidence_term + self.weights[name] * regularizer
            total = total + self.weights[name] * (fit + regularizer)
            supervised = True

        if self.cross_camera_3d_weight > 0:
            from .cross_camera import cross_camera_3d
            if 'trajectory' in target:
                truth = target['trajectory'].to(tensor.device)
                estimate = _resize_vector(pred['trajectory'], tuple(truth.shape[-3:-1]))
                if self.trajectory_motion_gradient_scale != 1:
                    estimate = self._absolute_trajectory(estimate, _resize_vector(
                        pred['displacement'], tuple(truth.shape[-3:-1])))
                valid = self._source_mask(_valid_mask(truth, target.get('trajectory_valid'), vector=True), pred)
                cross = cross_camera_3d(pred, target, estimate, truth, valid,
                                        self._foreground_weights(target, valid),
                                        report_delta=(self.joint_geometry_options is not None
                                                      and self.joint_geometry_options.diagnostics))
                terms.update(cross)
                if cross['cross_camera_3d_points'] > 0:
                    total = total + self.cross_camera_3d_weight * cross['cross_camera_3d']
                    supervised = True

        if (self.query_pair_weight > 0 or self.query_pair_source_sampling) and 'trajectory' in target:
            from .query_pair import query_pair_consistency
            truth = target['trajectory'].to(tensor.device)
            estimate = _resize_vector(pred['trajectory'], tuple(truth.shape[-3:-1]))
            valid = self._source_mask(_valid_mask(truth, target.get('trajectory_valid'), vector=True), pred)
            pair = query_pair_consistency(pred, target, estimate, truth, valid,
                                         self._foreground_weights(target, valid), options=self.query_pair_options)
            terms.update(pair)
            if self.query_pair_weight > 0 and pair['query_pair_points'] > 0:
                total = total+self.query_pair_weight*pair['query_pair_consistency']
                supervised = True

        if "depth" in target and self.weights["depth_gradient"] > 0:
            truth = target["depth"].to(tensor.device).float()
            estimate = _resize_scalar(pred["depth"], tuple(truth.shape[-2:])).float()
            valid = self._observation_mask(_valid_mask(truth, target.get("depth_valid")) & (truth > 0), pred)
            confidence = pred.get("depth_conf") if self.depth_gradient_confidence else None
            if confidence is not None:
                confidence = _resize_scalar(confidence, tuple(truth.shape[-2:]))
            fits, penalties = [], []
            for stride in self.depth_gradient_scales:
                gt, p, mask = truth[..., ::stride, ::stride], estimate[..., ::stride, ::stride], valid[..., ::stride, ::stride]
                conf = None if confidence is None else confidence[..., ::stride, ::stride]
                scale_fits, scale_penalties = [], []
                for axis in (-1, -2):
                    left, right = [slice(None)]*gt.ndim, [slice(None)]*gt.ndim
                    left[axis], right[axis] = slice(None, -1), slice(1, None)
                    left, right = tuple(left), tuple(right)
                    pair_valid = mask[left] & mask[right]
                    if pair_valid.any():
                        pair_conf = None if conf is None else (conf[left]+conf[right])/2
                        fit, penalty = self._regression(p[right]-p[left], gt[right]-gt[left], pair_valid, pair_conf, task="depth_gradient")
                        scale_fits.append(fit)
                        scale_penalties.append(penalty)
                if scale_fits:
                    fits.append(torch.stack(scale_fits).mean())
                    penalties.append(torch.stack(scale_penalties).mean())
            if fits:
                supervised = True
                terms["depth_gradient"] = torch.stack(fits).mean()
                penalty = torch.stack(penalties).mean()
                total = total + self.weights["depth_gradient"] * (terms["depth_gradient"]+penalty)
                confidence_term = confidence_term + self.weights["depth_gradient"] * penalty

        if (self.depth_objective_options is not None and self.depth_objective_options.enabled
                and "depth" in target):
            from .depth_objectives import dense_depth_objectives
            truth = target['depth'].to(tensor.device).float()
            estimate = _resize_scalar(pred['depth'], tuple(truth.shape[-2:])).float()
            valid = self._observation_mask(_valid_mask(truth, target.get('depth_valid')) & (truth > 0), pred)
            metadata = dict(pred)
            if 'depth_conf' in metadata:
                metadata['depth_conf'] = _resize_scalar(metadata['depth_conf'], tuple(truth.shape[-2:]))
            depth_terms, depth_total = dense_depth_objectives(
                estimate, truth, valid, metadata, self.depth_objective_options)
            terms.update(depth_terms)
            total = total+depth_total
            depth_enabled = (self.depth_objective_options.relative_weight
                             or self.depth_objective_options.log_bias_weight
                             or self.depth_objective_options.temporal_scale_weight)
            supervised = supervised or (bool(depth_enabled) and bool(valid.any()))

        if "displacement" in target and self.weights["velocity"] > 0:
            truth = target["displacement"].to(tensor.device).float()
            estimate = _resize_vector(pred["displacement"], tuple(truth.shape[-3:-1])).float()
            times = pred["query_time_idx"]
            if times.shape[1] > 1:
                order = times.argsort(dim=1)
                sorted_times = times.gather(1, order)
                interval = (sorted_times[:, 1:]-sorted_times[:, :-1]).float()
                if (interval <= 0).any():
                    raise ValueError("velocity needs unique physical frame indices")
                indices = order[:, None, :, None, None, None].expand_as(truth)
                truth, estimate = truth.gather(2, indices), estimate.gather(2, indices)
                valid = self._source_mask(_valid_mask(target["displacement"].to(tensor.device), target.get("displacement_valid", target.get("trajectory_valid")), vector=True), pred)
                weights = self._foreground_weights(target, valid)
                valid = valid.gather(2, indices[..., 0])
                pair_valid = valid[:, :, 1:] & valid[:, :, :-1]
                if pair_valid.any():
                    supervised = True
                    dt = interval[:, None, :, None, None, None]
                    confidence = pred.get("motion_confidence") if "velocity" in self.motion_confidence_targets else None
                    if confidence is not None:
                        confidence = _resize_scalar(confidence, tuple(truth.shape[-3:-1]))
                        eligible = self._confidence_eligible(pred, tuple(confidence.shape))
                        if eligible is None:
                            raise ValueError("velocity confidence needs source_time_idx and query_time_idx")
                        confidence = torch.where(eligible, confidence, 1.0).gather(2, indices[..., 0])
                        confidence = (confidence[:, :, 1:]+confidence[:, :, :-1])/2
                    if weights is not None:
                        weights = weights.gather(2, indices[..., 0])
                        weights = (weights[:, :, 1:]+weights[:, :, :-1])/2
                    fit, penalty = self._regression((estimate[:, :, 1:]-estimate[:, :, :-1])/dt,
                                                    (truth[:, :, 1:]-truth[:, :, :-1])/dt,
                                                    pair_valid, confidence, pixel_weights=weights, task="velocity")
                    terms["velocity"] = fit
                    total = total + self.weights["velocity"] * (fit+penalty)
                    confidence_term = confidence_term + self.weights["velocity"] * penalty

        camera_terms = []
        intrinsic_extra = zero
        if self.camera_loss_mode == "components" and self.weights["camera"] > 0:
            camera_parts = self._camera_components(pred, target)
            if camera_parts:
                terms.update(camera_parts)
                factors = {"camera_translation": 1., "camera_rotation": 1., "camera_fov": .5, "camera_principal": .5}
                camera_terms.append(sum(factors[name]*value for name, value in camera_parts.items()))
                supervised = True
        for name in ("K", "c2w"):
            if self.camera_loss_mode == "components":
                continue
            if name not in target or self.weights["camera"] == 0:
                continue
            if name not in pred:
                raise KeyError(f"target supplies {name} but pred does not")
            truth, estimate = target[name].to(tensor.device), pred[name]
            matrix_shape = (3, 3) if name == "K" else (4, 4)
            if truth.ndim != 4 or truth.shape[-2:] != matrix_shape or estimate.shape != truth.shape:
                raise ValueError(f"{name} must be matching [B,S,{matrix_shape[0]},{matrix_shape[1]}] tensors")
            if name == "K":
                source_size = pred["depth"].shape[-2:] if "depth" in pred else None
                target_size = target["depth"].shape[-2:] if "depth" in target else source_size
                if source_size is not None and target_size is not None:
                    ratio = estimate.new_tensor((target_size[1] / source_size[1],
                                                 target_size[0] / source_size[0], 1))
                    estimate = estimate * ratio[None, None, :, None]
                    if self.intrinsics_loss_mode != "legacy":
                        estimate = estimate.clone()
                        estimate[..., :2, 2] += (ratio[:2]-1)/2
            valid = torch.isfinite(truth).all(dim=(-2, -1))
            if "camera_valid" in target:
                valid = valid & torch.broadcast_to(target["camera_valid"].to(valid.device).bool(), valid.shape)
            valid = self._observation_mask(valid, pred)
            if not valid.any():
                continue
            p, t = estimate[valid].float(), truth[valid].float()
            if not torch.isfinite(p).all():
                raise FloatingPointError("non-finite camera prediction at a supervised location")
            if name == "K":
                # Normalize pixels so focal/principal losses do not grow with resolution.
                scale = t[..., :2, :2].diagonal(dim1=-2, dim2=-1).abs().clamp_min(1.0)
                if self.intrinsics_loss_mode == "legacy":
                    camera_terms.append(((p[..., :2, :] - t[..., :2, :]) / scale[..., :, None]).abs().mean())
                else:
                    focal = p[..., :2, :2].diagonal(dim1=-2, dim2=-1)
                    gt_focal = t[..., :2, :2].diagonal(dim1=-2, dim2=-1)
                    if (focal <= 0).any() or (gt_focal <= 0).any():
                        raise ValueError("log focal supervision requires positive intrinsics")
                    terms["camera_log_focal"] = (focal.log()-gt_focal.log()).abs().mean()
                    if self.focal_ratio_loss_weight:
                        log_error = focal.log()-gt_focal.log()
                        terms['camera_log_focal_ratio'] = (log_error[..., 0]-log_error[..., 1]).abs().mean()
                    h, w = target_size if target_size is not None else pred["depth"].shape[-2:]
                    terms["camera_principal_normalized"] = ((p[..., :2, 2]-t[..., :2, 2])/p.new_tensor([w,h])).abs().mean()
                    intrinsic_extra = (self.focal_loss_weight*terms["camera_log_focal"]
                                       +self.principal_loss_weight*terms["camera_principal_normalized"])
                    if self.focal_ratio_loss_weight:
                        intrinsic_extra = intrinsic_extra + self.focal_ratio_loss_weight*terms['camera_log_focal_ratio']
                    # Preserve the original extrinsics coefficient exactly:
                    # with K+c2w present the old mean contributes 0.5 L_c2w.
                    camera_terms.append(intrinsic_extra*0)
            else:
                residual = (p[..., :3, :] - t[..., :3, :]).abs()
                if "camera_translation_valid" in target:
                    translated = torch.broadcast_to(target["camera_translation_valid"].to(valid.device).bool(), valid.shape)[valid]
                    component_mask = torch.ones_like(residual)
                    component_mask[..., 3] = translated[:, None]
                    residual = residual * component_mask
                # Keep the established 12-component denominator and camera
                # coefficient. Unknown translation contributes no gradient.
                camera_terms.append(residual.mean())
            supervised = True
        if camera_terms:
            terms["camera"] = torch.stack(camera_terms).mean()+intrinsic_extra
            total = total + self.weights["camera"] * terms["camera"]

        if "visibility" in target and self.weights["visibility"] > 0:
            if "visibility_logits" not in pred or "projection_valid" not in pred:
                raise KeyError("visibility supervision needs logits and projection_valid")
            truth = target["visibility"].to(tensor.device)
            if truth.ndim != 6:
                raise ValueError("visibility target must be [B,Q,T,C,H,W]")
            size = tuple(truth.shape[-2:])
            logits = _resize_scalar(pred["visibility_logits"], size)
            projection = _resize_scalar(pred["projection_valid"], size, mask=True)
            if logits.shape != truth.shape or projection.shape != truth.shape:
                raise ValueError("visibility prediction and target shapes disagree")
            valid = _valid_mask(truth, target.get("visibility_valid"))
            if self.visibility_projection_mask:
                valid = valid & projection
            valid = self._source_mask(valid, pred)
            if valid.any():
                labels = truth[valid].float()
                if ((labels < 0) | (labels > 1)).any():
                    raise ValueError("visibility labels must lie in [0,1]")
                if not torch.isfinite(logits[valid]).all():
                    raise FloatingPointError("non-finite supervised visibility logits")
                terms["visibility"] = F.binary_cross_entropy_with_logits(logits[valid].float(), labels)
                total = total + self.weights["visibility"] * terms["visibility"]
                supervised = True

        if "static_mask" in target and self.weights["static"] > 0:
            if "displacement" not in pred:
                raise KeyError("static supervision needs predicted displacement")
            displacement = pred["displacement"]
            mask = target["static_mask"].to(tensor.device).bool()
            if mask.ndim == 4:
                mask = mask[:, :, None]
            if mask.ndim != 5:
                raise ValueError("static_mask must be [B,Q,H,W] or [B,Q,T,H,W]")
            size = tuple(mask.shape[-2:])
            displacement = _resize_vector(displacement, size)
            valid = torch.broadcast_to(mask, displacement.shape[:-1])
            valid = self._source_mask(valid, pred)
            if "trajectory_valid" in target:
                valid = valid & _resize_scalar(target["trajectory_valid"].to(valid.device), size, mask=True)
            eligible = self._confidence_eligible(pred, tuple(valid.shape))
            if eligible is not None:
                valid = valid & eligible
            if valid.any():
                if not torch.isfinite(displacement[valid]).all():
                    raise FloatingPointError("non-finite static displacement prediction")
                terms["static"] = displacement[valid].float().abs().mean()
                total = total + self.weights["static"] * terms["static"]
                supervised = True

        if self.joint_geometry_options is not None:
            from .joint_geometry import joint_geometry_losses
            joint_terms, joint_total = joint_geometry_losses(pred, target, self.joint_geometry_options)
            terms.update(joint_terms)
            total = total + joint_total
            opts = self.joint_geometry_options
            joint_supervised = (
                (opts.source_anchor_weight > 0 and "joint_source_pixels" in joint_terms)
                or (opts.relative_pose_weight > 0 and ("joint_rotation_pairs" in joint_terms
                                                      or "joint_translation_pairs" in joint_terms))
                or (opts.dynamic_displacement_weight > 0 and "joint_dynamic_point_times" in joint_terms)
                or (opts.static_displacement_weight > 0 and "joint_static_point_times" in joint_terms)
                or (opts.baseline_relative_weight > 0 and "joint_baseline_pairs" in joint_terms)
                or (opts.depth_baseline_scale_weight > 0 and "joint_scale_pairs" in joint_terms)
            )
            supervised = supervised or joint_supervised

        if self.auxiliary_motion_options is not None:
            from .auxiliary_motion import auxiliary_motion_losses
            aux_terms, aux_total, aux_supervised = auxiliary_motion_losses(pred, target, self.auxiliary_motion_options)
            terms.update(aux_terms)
            total = total + aux_total
            supervised = supervised or aux_supervised

        sparse_terms = {}
        if self.reprojection_options is not None:
            from .reprojection import reprojection_losses
            reprojection_terms, reprojection_total = reprojection_losses(pred, target, self.reprojection_options)
            terms.update(reprojection_terms)
            total = total + reprojection_total
            supervised = supervised or bool(reprojection_terms)
        if "sparse_queries" in target:
            sparse_terms, sparse_total, sparse_confidence, sparse_supervised = self._sparse_losses(pred, target, zero)
            total = total + sparse_total
            confidence_term = confidence_term + sparse_confidence
            supervised = supervised or sparse_supervised

        if not supervised:
            raise ValueError("batch has no valid enabled depth, camera, ray, trajectory, visibility, or static supervision")
        return {"loss": total, **terms, **sparse_terms, "confidence": confidence_term}
