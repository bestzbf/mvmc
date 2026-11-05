"""Dense residual tracking from predicted geometry and local visual matches.

No ground-truth geometry enters this module. Correspondence coordinates and
depth cues are detached for the first-stage motion refinement, while image
features and the update network remain trainable. Shared updates operate on a
source-pixel lattice across a fixed set of observed times. Learned convex
upsampling restores every input pixel; the original dense prediction is never
replaced by an upsampled coarse initialization.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .metadata import gather_observations
from .embeddings import SinusoidalFrameEmbedding


class TemporalUpdate(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim),
                                 nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, hidden, time_valid):
        # [B,U,N,D] -> [B*N,heads,U,head_dim]. Each row is one source point.
        b, u, n, d = hidden.shape
        x = hidden.permute(0, 2, 1, 3).reshape(b * n, u, d)
        q, k, v = self.qkv(self.norm(x)).reshape(
            b * n, u, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        valid = time_valid[:, None].expand(b, n, u).reshape(b * n, u)
        # Metadata guarantees a valid observed time in every batch element.
        x = x + self.proj(F.scaled_dot_product_attention(
            q, k, v, attn_mask=valid[:, None, None, :]).transpose(1, 2).reshape(b * n, u, d))
        x = (x + self.ffn(x)) * valid[..., None]
        return x.reshape(b, n, u, d).permute(0, 2, 1, 3)


def convex_upsample(residual, mask, factor):
    """Upsample world-XYZ residuals; unlike pixel flow, do not multiply by factor."""
    b, u, h, w, _ = residual.shape
    values = residual.permute(0, 1, 4, 2, 3).reshape(b * u, 3, h, w)
    values = F.unfold(F.pad(values, (1, 1, 1, 1), mode="replicate"), 3)
    values = values.reshape(b, u, 3, 9, 1, 1, h, w)
    weights = mask.reshape(b, 1, 1, 9, factor, factor, h, w).softmax(3)
    out = (values * weights).sum(3)
    return out.permute(0, 1, 5, 3, 6, 4, 2).reshape(b, u, h * factor, w * factor, 3)


class TrajectoryRefiner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d, p = config.refinement_dim, config.patch_size
        self.feature_projection = nn.Sequential(nn.LayerNorm(2 * config.embed_dim),
                                                 nn.Linear(2 * config.embed_dim, d))
        self.motion_projection = nn.Conv2d(config.dpt_features // 2, d, 1)
        self.time_embedding = SinusoidalFrameEmbedding(d)
        self.time_projection = nn.Linear(d, d)
        correlations = config.refinement_levels * (2 * config.refinement_radius + 1) ** 2
        # Camera XYZ, image XY, signed depth residual, confidence, centre
        # support, search support, same-camera relation, and world-coordinate
        # Jacobians for one image patch in u/v and a relative depth update.
        self.view_encoder = nn.Sequential(nn.Linear(correlations + 19, d), nn.GELU(), nn.Linear(d, d))
        self.view_gate = nn.Linear(d, 1)
        self.state_encoder = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.temporal = TemporalUpdate(d, config.refinement_heads)
        self.update = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.delta_head = nn.Linear(d, 3)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        self.upsample_mask = nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.GELU(),
                                           nn.Conv2d(d, 9 * p * p, 1))
        r = config.refinement_radius
        y, x = torch.meshgrid(torch.arange(-r, r + 1), torch.arange(-r, r + 1), indexing="ij")
        self.register_buffer("offsets", torch.stack((x, y), -1).reshape(-1, 2).float(), persistent=False)

    def prepare(self, patches, patch_grid):
        b, s, _, _ = patches.shape
        h, w = patch_grid
        features = self.feature_projection(patches.float())
        return features.transpose(-1, -2).reshape(b, s, self.config.refinement_dim, h, w)

    @staticmethod
    def scene_scale(depth, observation_valid):
        # One predicted scale for the entire clip, shared by every source/view.
        values = []
        with torch.no_grad():
            for b in range(len(depth)):
                z = depth[b, observation_valid[b]].float()
                z = z[torch.isfinite(z) & (z > 0)]
                values.append(z.median().clamp_min(1e-4) if z.numel() else depth.new_tensor(1.))
        return torch.stack(values)

    def prepare_pyramid(self, target):
        d, h, w = target.shape[-3:]
        target = target.reshape(-1, d, h, w)
        return tuple(F.normalize(target if level == 0 else F.interpolate(
            target, (max(1, h // 2 ** level), max(1, w // 2 ** level)),
            mode='bilinear', align_corners=False), dim=1)
            for level in range(self.config.refinement_levels))

    def matching_evidence(self, points, source, target, depth, confidence, K, c2w,
                          observation_valid, same_camera, scale, image_hw, target_pyramid=None):
        """Compress local windows immediately into scalar correlations.

        Each scale uses one batched grid_sample. No Python loop over pixels,
        queries or time, and no persistent [N,U,C,window,D] cost volume.
        """
        b, u, n, _ = points.shape
        c, d, h, w = target.shape[2:]
        H, W = image_hw
        with torch.no_grad():
            relative = points.detach()[:, :, None] - c2w.detach()[..., None, :3, 3]
            pc = torch.einsum("bucni,bucij->bucnj", relative, c2w.detach()[..., :3, :3])
            finite = torch.isfinite(pc).all(-1)
            pc = torch.nan_to_num(pc)
            z = pc[..., 2]
            uvw = torch.einsum("bucni,bucji->bucnj", pc, K.detach())
            uv = uvw[..., :2] / z[..., None].clamp_min(1e-6)
            grid = torch.stack((2 * (uv[..., 0] + .5) / W - 1,
                                2 * (uv[..., 1] + .5) / H - 1), -1)
            grid = torch.nan_to_num(grid).clamp(-1e4, 1e4)
            eligible = finite & (z > 1e-6) & observation_valid[..., None]
            centre = eligible & (uv[..., 0] >= 0) & (uv[..., 0] <= W - 1)
            centre &= (uv[..., 1] >= 0) & (uv[..., 1] <= H - 1)
            g = grid.reshape(b * u * c, n, 1, 2)
            z_image = F.grid_sample(depth.detach().reshape(b * u * c, 1, H, W).float(),
                                    g, align_corners=False).reshape(b, u, c, n)
            conf = F.grid_sample(confidence.detach().reshape(b * u * c, 1, H, W).float(),
                                 g, align_corners=False).reshape(b, u, c, n)
        descriptor = F.normalize(source.flatten(-2).transpose(-1, -2), dim=-1)
        descriptor = descriptor[:, None, None, :, None, :]
        target_pyramid = self.prepare_pyramid(target) if target_pyramid is None else target_pyramid
        correlation, supports = [], []
        for feature in target_pyramid:
            lh, lw = feature.shape[-2:]
            offsets = self.offsets.to(grid) * grid.new_tensor((2 / lw, 2 / lh))
            window = grid[..., None, :] + offsets
            support = eligible[..., None] & (window.abs() <= 1).all(-1)
            samples = F.grid_sample(feature, window.reshape(b * u * c, n, -1, 2), align_corners=False)
            samples = samples.reshape(b, u, c, d, n, -1).permute(0, 1, 2, 4, 5, 3)
            score = (samples * descriptor).sum(-1).masked_fill(~support, -1.)
            correlation.append(score)
            supports.append(support.any(-1))
        supported = torch.stack(supports).any(0)
        with torch.no_grad():
            divisor = scale[:, None, None, None]
            xyz = pc / divisor[..., None]
            xyz = xyz.sign() * torch.log1p(xyz.abs())
            dz = (z_image - z) / divisor
            dz = dz.sign() * torch.log1p(dz.abs())
            rotation = c2w.detach()[..., :3, :3]
            fx = K.detach()[..., 0, 0].clamp_min(1e-6)
            fy = K.detach()[..., 1, 1].clamp_min(1e-6)
            ju = rotation[..., None, :, 0] * (z / fx[..., None] / divisor)[..., None] * self.config.patch_size
            jv = rotation[..., None, :, 1] * (z / fy[..., None] / divisor)[..., None] * self.config.patch_size
            jz = relative / divisor[..., None]
            jacobian = torch.cat((ju, jv, jz), -1)
            jacobian = (jacobian.sign() * torch.log1p(jacobian.abs())).clamp(-10, 10)
            geometry = torch.cat((xyz.clamp(-10, 10), grid.clamp(-2, 2),
                dz[..., None].clamp(-10, 10), conf.clamp(1, 1e4).log()[..., None],
                centre[..., None].float(), supported[..., None].float(),
                same_camera[:, None, :, None, None].expand(b, u, c, n, 1).float(), jacobian), -1)
        evidence = self.view_encoder(torch.cat((*correlation, geometry), -1))
        weights = self.view_gate(evidence).sigmoid() * supported[..., None]
        pooled = (evidence * weights).sum(2) / weights.sum(2).clamp_min(1e-6)
        return pooled, supported.float().mean((1, 2, 3))

    def forward(self, trajectory, motion_features, source_features, target_features,
                target_depth, target_conf, target_K, target_c2w, target_valid,
                same_camera, times, time_valid, anchor_slot, scale, target_pyramid=None):
        b, u, H, W, _ = trajectory.shape
        h, w = source_features.shape[-2:]
        n, d = h * w, self.config.refinement_dim
        coarse = F.interpolate(trajectory.permute(0, 1, 4, 2, 3).reshape(b * u, 3, H, W),
                               (h, w), mode="bilinear", align_corners=False)
        initial = coarse.reshape(b, u, 3, n).permute(0, 1, 3, 2)
        anchor = gather_observations(initial, anchor_slot)
        feature = F.interpolate(motion_features.reshape(b * u, -1, H, W),
                                (h, w), mode="bilinear", align_corners=False)
        feature = self.motion_projection(feature).reshape(b, u, d, n).permute(0, 1, 3, 2)
        source = source_features.flatten(-2).transpose(-1, -2)
        source_time = gather_observations(times, anchor_slot)
        time = self.time_projection(self.time_embedding(times - source_time[:, None]).float())
        hidden = feature + source[:, None] + time[:, :, None]
        correction = torch.zeros_like(initial)
        support = trajectory.new_zeros(b)
        for _ in range(self.config.refinement_iterations):
            current = initial + correction
            xyz = torch.cat((current.detach() - anchor.detach()[:, None],
                             anchor.detach()[:, None].expand_as(current)), -1) / scale[:, None, None, None]
            state = xyz.sign() * torch.log1p(xyz.abs())
            evidence = torch.zeros_like(hidden)
            if self.config.refinement_matching:
                evidence, support = self.matching_evidence(current, source_features, target_features,
                    target_depth, target_conf, target_K, target_c2w, target_valid,
                    same_camera, scale, (H, W), target_pyramid)
            hidden = hidden + evidence + self.state_encoder(state.clamp(-10, 10))
            if self.config.refinement_temporal:
                hidden = self.temporal(hidden, time_valid)
            hidden = (hidden + self.update(hidden)) * time_valid[:, :, None, None]
            step = self.delta_head(hidden).tanh() * scale[:, None, None, None] * self.config.refinement_max_update
            step = step - gather_observations(step, anchor_slot)[:, None]
            correction = correction + step * time_valid[:, :, None, None]
        mask = self.upsample_mask(source_features)
        dense = convex_upsample(correction.reshape(b, u, h, w, 3), mask, self.config.patch_size)
        return dict(correction=dense, match_fraction=support.detach(),
                    correction_relative_norm=(correction.detach().norm(dim=-1) / scale[:, None, None]).mean((1, 2)))
