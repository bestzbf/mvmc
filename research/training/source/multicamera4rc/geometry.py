"""Differentiable pinhole geometry using the original 4RC camera convention."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .vendor.heads.cam_dec import CameraDec


class ExtendedCameraDec(CameraDec):
    """Append principal-point offsets without renaming pretrained pose9 weights.

    The encoding is ``[c2w_translation, quaternion_xyzw, fov_h, fov_w,
    principal_dx, principal_dy]``. Offsets are fractions of image width/height.
    Zero offsets preserve the original camera head's centered principal point.
    """

    def __init__(self, dim_in: int = 1536):
        super().__init__(dim_in=dim_in)
        # Scratch initialization needs positive FOV logits: a dead ReLU gives
        # huge clamped focal lengths with no direct focal gradient. Loading
        # a 4RC checkpoint subsequently replaces this original-head bias.
        nn.init.constant_(self.fc_fov[0].bias, math.pi/3)
        self.fc_principal = nn.Linear(dim_in, 2)
        nn.init.zeros_(self.fc_principal.weight)
        nn.init.zeros_(self.fc_principal.bias)

    def forward(self, feat: Tensor, camera_encoding: Tensor | None = None,
                *args, **kwargs) -> Tensor:
        if feat.ndim != 3:
            raise ValueError("camera features must have shape [B, S, dim_in]")
        batch, observations = feat.shape[:2]
        hidden = self.backbone(feat.reshape(batch * observations, -1)).float()
        translation = self.fc_t(hidden).reshape(batch, observations, 3)
        if camera_encoding is None:
            quaternion = self.fc_qvec(hidden).reshape(batch, observations, 4)
            fov = self.fc_fov(hidden).reshape(batch, observations, 2)
            principal = self.fc_principal(hidden).reshape(batch, observations, 2)
        else:
            if camera_encoding.shape[:-1] != feat.shape[:-1] or camera_encoding.shape[-1] not in (9, 11):
                raise ValueError("camera_encoding must be [B, S, 9] or [B, S, 11]")
            quaternion = camera_encoding[..., 3:7]
            fov = camera_encoding[..., 7:9]
            principal = (camera_encoding[..., 9:11] if camera_encoding.shape[-1] == 11
                         else self.fc_principal(hidden).reshape(batch, observations, 2))
        return torch.cat((translation, quaternion, fov, principal), dim=-1)


def quaternion_xyzw_to_matrix(quaternion: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize XYZW quaternions, mapping an all-zero quaternion to identity."""
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    identity = torch.zeros_like(quaternion)
    identity[..., 3] = 1
    q = torch.where(norm > eps, quaternion / norm.clamp_min(eps), identity)
    x, y, z, w = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ), dim=-1).reshape(quaternion.shape[:-1] + (3, 3))


def decode_camera(pose11: Tensor, height: int, width: int, *,
                  principal_canvas: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Decode pose11 to zero-skew intrinsics and camera-to-world transforms.

    Pixel coordinates are integer pixel centers, matching 4RC's unprojection;
    the pretrained centered principal point is ``(width / 2, height / 2)``.
    The first nine entries retain Arc's camera-to-world semantics (Arc inverts
    the decoded transform only when it exposes world-to-camera extrinsics).

    Optional ``principal_canvas[B,S,4] = [H0,W0,top,left]`` describes a known
    integer crop of the original padded canvas. Only principal offsets use
    that canvas; FOV/focal decoding still uses the current image dimensions.
    It contains preprocessing metadata, never ground-truth calibration.
    """
    if pose11.ndim != 3 or pose11.shape[-1] != 11:
        raise ValueError("pose11 must have shape [B, S, 11]")
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    # Geometry in float32 remains stable under mixed-precision model inference.
    pose = pose11.float() if pose11.dtype in (torch.float16, torch.bfloat16) else pose11
    rotation_c2w = quaternion_xyzw_to_matrix(pose[..., 3:7])
    fov = pose[..., 7:9].clamp(min=0.0, max=math.pi - 1e-4)
    fy = (height / 2.0) / torch.tan(fov[..., 0] / 2.0).clamp_min(1e-6)
    fx = (width / 2.0) / torch.tan(fov[..., 1] / 2.0).clamp_min(1e-6)
    if principal_canvas is None:
        cx = width / 2.0 + pose[..., 9] * width
        cy = height / 2.0 + pose[..., 10] * height
    else:
        canvas = torch.as_tensor(principal_canvas, device=pose.device, dtype=pose.dtype)
        if canvas.shape != pose.shape[:-1] + (4,):
            raise ValueError("principal_canvas must have shape [B,S,4]")
        if (not torch.isfinite(canvas).all() or (canvas[..., :2] <= 0).any()
                or (canvas[..., 2:] < 0).any() or (canvas[..., 2:] >= canvas[..., :2]).any()):
            raise ValueError("invalid principal crop canvas or origin")
        h0, w0, top, left = canvas.unbind(-1)
        cx = w0 / 2.0 + pose[..., 9] * w0 - left
        cy = h0 / 2.0 + pose[..., 10] * h0 - top
    zero, one = torch.zeros_like(fx), torch.ones_like(fx)
    intrinsics = torch.stack((fx, zero, cx, zero, fy, cy, zero, zero, one), -1)
    intrinsics = intrinsics.reshape(pose.shape[:-1] + (3, 3))
    center = pose[..., :3]
    top = torch.cat((rotation_c2w, center[..., None]), dim=-1)
    bottom = torch.stack((zero, zero, zero, one), -1)[..., None, :]
    return intrinsics, torch.cat((top, bottom), dim=-2)


def unproject_depth(depth: Tensor, K: Tensor, c2w: Tensor) -> Tensor:
    """Lift camera-z depth [B,S,H,W] into world points [B,S,H,W,3]."""
    if depth.ndim != 4:
        raise ValueError("depth must have shape [B, S, H, W]")
    if K.shape != depth.shape[:2] + (3, 3) or c2w.shape != depth.shape[:2] + (4, 4):
        raise ValueError("K/c2w observation dimensions must match depth")
    dtype = torch.promote_types(torch.promote_types(depth.dtype, K.dtype), c2w.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    z, intrinsics, camera = depth.to(dtype), K.to(dtype), c2w.to(dtype)
    height, width = depth.shape[-2:]
    v, u = torch.meshgrid(torch.arange(height, device=z.device, dtype=dtype),
                          torch.arange(width, device=z.device, dtype=dtype), indexing="ij")
    fx, fy = intrinsics[..., 0, 0, None, None], intrinsics[..., 1, 1, None, None]
    cx, cy = intrinsics[..., 0, 2, None, None], intrinsics[..., 1, 2, None, None]
    points = torch.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), dim=-1)
    return torch.einsum("bsij,bshwj->bshwi", camera[..., :3, :3], points) + camera[..., :3, 3][:, :, None, None, :]


def share_camera_intrinsics(K: Tensor, camera_id: Tensor, valid: Tensor,
                            pixel_affine: Tensor | None = None) -> Tensor:
    """Average only fx/fy/cx/cy over the valid times of each physical camera.

    The caller asserts constant physical intrinsics (no optical zoom). Known
    preprocessing can differ by frame: K_input = pixel_affine @ K_original.
    Neither calibration GT nor depth/pose/confidence enters this operation.
    Singleton groups and invalid observations are returned bitwise unchanged.
    """
    if K.shape != camera_id.shape + (3, 3) or valid.shape != camera_id.shape:
        raise ValueError('K/camera_id/valid observation shapes disagree')
    canonical = K
    if pixel_affine is not None:
        if pixel_affine.shape != K.shape or not torch.isfinite(pixel_affine).all():
            raise ValueError('pixel_affine must be finite [B,S,3,3]')
        canonical = torch.linalg.solve(pixel_affine.to(K), K)
    same = ((camera_id[:, :, None] == camera_id[:, None, :])
            & valid[:, :, None] & valid[:, None, :])
    counts = same.sum(-1, keepdim=True)
    weights = same.to(K.dtype)/counts.clamp_min(1)
    values = torch.stack((canonical[..., 0, 0], canonical[..., 1, 1],
                          canonical[..., 0, 2], canonical[..., 1, 2]), -1)
    means = weights @ torch.where(valid[..., None], values, 0)
    shared = canonical.clone()
    shared[..., 0, 0], shared[..., 1, 1] = means[..., 0], means[..., 1]
    shared[..., 0, 2], shared[..., 1, 2] = means[..., 2], means[..., 3]
    if pixel_affine is not None:
        shared = pixel_affine.to(K) @ shared
    return torch.where((counts > 1)[..., None], shared, K)


def replace_pose_intrinsics(pose: Tensor, K: Tensor, height: int, width: int,
                            principal_canvas: Tensor | None = None) -> Tensor:
    """Keep pose11 and decoded K consistent after camera-wise pooling."""
    fov_h = 2*torch.atan((height/2)/K[..., 1, 1])
    fov_w = 2*torch.atan((width/2)/K[..., 0, 0])
    if principal_canvas is None:
        dx, dy = K[..., 0, 2]/width-.5, K[..., 1, 2]/height-.5
    else:
        h0, w0, top, left = principal_canvas.to(K).unbind(-1)
        dx = (K[..., 0, 2]+left)/w0-.5
        dy = (K[..., 1, 2]+top)/h0-.5
    return torch.cat((pose[..., :7], torch.stack((fov_h, fov_w, dx, dy), -1)), -1)


def check_constant_intrinsics(K: Tensor, camera_id: Tensor, valid: Tensor) -> None:
    """Training GT assertion only: never use labels to select sharing groups."""
    for batch in range(len(K)):
        for camera in camera_id[batch, valid[batch]].unique():
            values = K[batch, valid[batch] & (camera_id[batch] == camera)]
            if len(values) < 2:
                continue
            focal = values[:, [0, 1], [0, 1]]
            principal = values[:, :2, 2]
            if (not torch.isfinite(values).all() or (focal <= 0).any()
                    or ((focal.amax(0)-focal.amin(0))/focal.mean(0)).max() > 1e-4
                    or (principal.amax(0)-principal.amin(0)).max() > 1e-3):
                raise ValueError('camera-wise sharing encountered changing GT intrinsics; audit this source')
