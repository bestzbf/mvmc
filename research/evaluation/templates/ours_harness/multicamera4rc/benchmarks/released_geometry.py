"""RGB-only observation geometry, without trajectory decoding or GT access.

WorldTrack reconstruction consumes each observation's predicted base points,
not a point-query trajectory. Encoding already predicts these points and the
cameras, so executing the motion decoder/refiner cannot change this output.
Preprocessing and the float32 CPU export match ``tapvid_mv`` exactly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .tapvid_mv import _integer_indices


@dataclass(frozen=True, slots=True)
class ReleasedGeometryInputs:
    """Only RGB and observation identifiers; no queries or annotation fields."""

    video: torch.Tensor                         # [V,T,3,H,W], RGB in 0..255
    view_ids: tuple[int, ...]
    frame_indices: torch.Tensor | None = None    # [T], increasing original labels

    def __post_init__(self):
        video = torch.as_tensor(self.video)
        if video.ndim != 5 or video.shape[2] != 3 or min(video.shape) == 0:
            raise ValueError("video must be nonempty [V,T,3,H,W] RGB")
        if video.dtype == torch.bool or video.is_complex() or not torch.isfinite(video).all():
            raise ValueError("video must contain finite RGB values in 0..255")
        if video.min() < 0 or video.max() > 255:
            raise ValueError("video must contain RGB values in 0..255")
        views, frames = video.shape[:2]
        ids = _integer_indices(self.view_ids, "view_ids")
        if ids.shape != (views,) or (ids < 0).any() or len(ids.unique()) != views:
            raise ValueError("view_ids must be distinct nonnegative IDs matching the video view axis")
        labels = (torch.arange(frames) if self.frame_indices is None else
                  _integer_indices(self.frame_indices, "frame_indices"))
        if labels.shape != (frames,) or (labels < 0).any() or (labels[1:] <= labels[:-1]).any():
            raise ValueError("frame_indices must be nonnegative, strictly increasing [T] labels")
        object.__setattr__(self, "video", video)
        object.__setattr__(self, "view_ids", tuple(ids.tolist()))
        object.__setattr__(self, "frame_indices", labels)

    @classmethod
    def from_rgb(cls, source):
        """Read precisely these RGB fields, never queries or ground truth."""
        return cls(source.video, source.view_ids, source.frame_indices)


@dataclass(frozen=True, slots=True)
class ReleasedGeometryPrediction:
    """CPU observation geometry in input order and native predicted world gauge."""

    depth: np.ndarray                    # [V,T,H,W]
    depth_confidence: np.ndarray         # [V,T,H,W]
    base_points: np.ndarray              # [V,T,H,W,3]
    predicted_K: np.ndarray              # [V,T,3,3], original pixel coordinates
    predicted_c2w: np.ndarray            # [V,T,4,4]
    metadata: dict


def _validate_model_device(model, device):
    """Fail before encoding when a module was not moved to the requested device."""
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use device='cpu' for CPU inference")
    if device.type == "meta":
        raise ValueError("Geometry inference requires a real device, not meta")
    expected = device
    if device.type == "cuda" and device.index is None:
        expected = torch.device("cuda", torch.cuda.current_device())
    for kind in ("parameters", "buffers"):
        tensors = getattr(model, kind, None)
        if tensors is None:
            continue
        for tensor in tensors():
            actual = tensor.device
            if actual.type != expected.type or (expected.type != "cpu" and actual.index != expected.index):
                raise ValueError(f"Model {kind} are on {actual}, but geometry input device is {expected}; "
                                 "move the model before inference")


@torch.inference_mode()
def predict_released_geometry(model, inputs: ReleasedGeometryInputs, device="cpu", amp_dtype=None):
    """Encode all RGB once and export true geometry, without executing decode.

    The native checkpoint/refiner must still be fully and strictly loaded by
    the caller. This path skips motion computation, not checkpoint validation.
    It never resizes RGB, invents trajectories/visibility, or accesses GT.
    """
    if not isinstance(inputs, ReleasedGeometryInputs):
        raise TypeError("inputs must be ReleasedGeometryInputs, not a benchmark label object")
    # Owners can mutate tensors after construction; validate again at inference.
    inputs = ReleasedGeometryInputs(inputs.video, inputs.view_ids, inputs.frame_indices)
    if model.training:
        raise ValueError("Set the model to eval() before benchmark inference")
    device = torch.device(device)
    _validate_model_device(model, device)
    if amp_dtype is not None and amp_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("amp_dtype must be None, torch.float16 or torch.bfloat16")
    views, frames, _, height, width = inputs.video.shape
    patch = model.config.patch_size
    if isinstance(patch, bool) or not isinstance(patch, int) or patch <= 0:
        raise ValueError("Model patch_size must be a positive integer")
    padded_h = max(2 * patch, math.ceil(height / patch) * patch)
    padded_w = max(2 * patch, math.ceil(width / patch) * patch)
    images = inputs.video.to(device=device, dtype=torch.float32).reshape(views * frames, 3, height, width) / 255
    images = F.pad(images, (0, padded_w - width, 0, padded_h - height))[None]
    camera_ids = torch.tensor(inputs.view_ids, device=device).repeat_interleave(frames)[None]
    labels = inputs.frame_indices.to(device=device, dtype=torch.long)
    times = labels.repeat(views)[None]
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        scene = model.encode(images, camera_ids, times)
    geometry = scene.geometry
    if not isinstance(geometry, Mapping):
        raise ValueError("Encoder geometry must be a mapping of prediction tensors")
    shapes = {"depth": (1, views * frames, padded_h, padded_w),
              "depth_conf": (1, views * frames, padded_h, padded_w),
              "base_points": (1, views * frames, padded_h, padded_w, 3),
              "K": (1, views * frames, 3, 3), "c2w": (1, views * frames, 4, 4)}
    for key, shape in shapes.items():
        value = geometry.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"Encoder geometry {key} must be a tensor with shape {shape}")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"Encoder geometry {key} must contain finite floating-point predictions")
        if value.device != images.device:
            raise ValueError(f"Encoder geometry {key} device differs from the encoded RGB device")
    depth = geometry["depth"][0].reshape(views, frames, padded_h, padded_w)[..., :height, :width]
    confidence = geometry["depth_conf"][0].reshape(views, frames, padded_h, padded_w)[..., :height, :width]
    base = geometry["base_points"][0].reshape(views, frames, padded_h, padded_w, 3)[..., :height, :width, :]
    K = geometry["K"][0].reshape(views, frames, 3, 3)
    c2w = geometry["c2w"][0].reshape(views, frames, 4, 4)

    def array(tensor):
        return tensor.detach().float().cpu().numpy().copy()

    arrays = tuple(array(value) for value in (depth, confidence, base, K, c2w))
    if any(not np.isfinite(value).all() for value in arrays):
        raise ValueError("Model produced non-finite exported geometry predictions")
    metadata = {"adapter": "multicamera4rc-released-rgb-geometry-v1", "prediction_kind": "geometry",
                "input_condition": "rgb_only_unknown_calibration", "coordinates": "native predicted world gauge",
                "alignment": "none", "view_ids": list(inputs.view_ids), "frame_indices": labels.cpu().tolist(),
                "image_hw": [height, width], "padded_hw": [padded_h, padded_w], "resize": "none",
                "padding": "black, right/bottom only; encoder padding is unmasked", "patch_size": patch,
                "amp_dtype": str(amp_dtype) if amp_dtype is not None else None,
                "input_observations": views * frames, "encoder_passes": 1, "decoder_passes": 0,
                "target_time_policy": "not_applicable_geometry", "long_video_window_chaining": False,
                "official_scoring_implemented": False}
    return ReleasedGeometryPrediction(*arrays, metadata)
