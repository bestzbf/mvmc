"""RGB-only pixel-query inference for the TAPVid-MV task.

This is a model adapter, not a reproduction of the unpublished official scorer.
It preserves the model's predicted geometry and world gauge. No calibration,
depth, query XYZ, ground-truth visibility, or tracking labels are model inputs.
The complete clip is encoded once; decoder chunks do not implement long-video
window chaining or remove any encoder observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
from torch.nn import functional as F


def _integer_indices(values, name):
    tensor = torch.as_tensor(values)
    if tensor.dtype == torch.bool or tensor.is_complex():
        raise ValueError(f"{name} must contain integer indices")
    if not torch.isfinite(tensor).all() or not torch.equal(tensor, tensor.round()):
        raise ValueError(f"{name} must contain finite integer indices")
    return tensor.long()


@dataclass(frozen=True, slots=True)
class TAPVidMVInputs:
    """Only RGB and single-view pixel queries cross the inference boundary.

    ``queries_xytv[:, 2:4]`` contains indices into the supplied time/view axes,
    not camera IDs or physical frame labels. ``frame_indices`` supplies those
    physical labels to the model without changing the query/returned time order.
    Pixel centers follow ``[-.5, W-.5) x [-.5, H-.5)``.
    """

    video: torch.Tensor                    # [V,T,3,H,W], RGB in 0..255
    queries_xytv: torch.Tensor             # [N,4], (x,y,time slot,view slot)
    view_ids: tuple[int, ...]
    frame_indices: torch.Tensor | None = None  # [T], original increasing labels
    pixel_query_policy: ClassVar[str] = "half_pixel_inbounds"

    def __post_init__(self):
        video = torch.as_tensor(self.video)
        if video.ndim != 5 or video.shape[2] != 3 or min(video.shape) == 0:
            raise ValueError("video must be nonempty [V,T,3,H,W] RGB")
        if video.dtype == torch.bool or video.is_complex() or not torch.isfinite(video).all():
            raise ValueError("video must contain finite RGB values in 0..255")
        if video.min() < 0 or video.max() > 255:
            raise ValueError("video must contain RGB values in 0..255")
        views, frames, _, height, width = video.shape
        ids = _integer_indices(self.view_ids, "view_ids")
        if ids.shape != (views,) or (ids < 0).any() or len(ids.unique()) != views:
            raise ValueError("view_ids must be distinct nonnegative IDs matching the video view axis")
        queries = torch.as_tensor(self.queries_xytv)
        if queries.ndim != 2 or queries.shape[1] != 4 or len(queries) == 0:
            raise ValueError("queries_xytv must be nonempty [N,4]")
        if queries.dtype == torch.bool or queries.is_complex() or not torch.isfinite(queries).all():
            raise ValueError("queries_xytv must contain finite numeric coordinates")
        times = _integer_indices(queries[:, 2], "query frames")
        cameras = _integer_indices(queries[:, 3], "query views")
        if ((times < 0) | (times >= frames)).any():
            raise ValueError("query frame is outside the supplied video time axis")
        if ((cameras < 0) | (cameras >= views)).any():
            raise ValueError("query view is outside the supplied video view axis")
        if self.pixel_query_policy not in ("half_pixel_inbounds", "preserve_border_extension"):
            raise ValueError("Unknown pixel query policy")
        if self.pixel_query_policy == "half_pixel_inbounds" and ((queries[:, 0] < -.5) | (queries[:, 0] >= width - .5)
                | (queries[:, 1] < -.5) | (queries[:, 1] >= height - .5)).any():
            raise ValueError("query pixels must lie in [-.5,W-.5) x [-.5,H-.5)")
        labels = (torch.arange(frames) if self.frame_indices is None else
                  _integer_indices(self.frame_indices, "frame_indices"))
        if labels.shape != (frames,) or (labels < 0).any() or (labels[1:] <= labels[:-1]).any():
            raise ValueError("frame_indices must be nonnegative, strictly increasing [T] labels")
        object.__setattr__(self, "video", video)
        object.__setattr__(self, "queries_xytv", queries)
        object.__setattr__(self, "view_ids", tuple(ids.tolist()))
        object.__setattr__(self, "frame_indices", labels)

    @classmethod
    def from_rgb(cls, source):
        """Read only these four RGB/query attributes, never an annotation object."""
        return cls(source.video, source.queries_xytv, source.view_ids, source.frame_indices)


@dataclass(frozen=True, slots=True)
class ReleasedPixelInputs(TAPVidMVInputs):
    """Released official queries can extend outside the recorded RGB canvas.

    Coordinates remain unchanged. The dense model's defined border extension is
    an adapter behavior, never an alteration of the official query inventory.
    The original four-field input type and its strict default remain unchanged.
    """

    pixel_query_policy: ClassVar[str] = "preserve_border_extension"


@dataclass(frozen=True, slots=True)
class TAPVidMVPrediction:
    """CPU arrays in the supplied time/view/query order and native world gauge."""

    tracks_world: np.ndarray              # [T,N,3]
    visibility: np.ndarray                # [V,T,N], bool
    visibility_probability: np.ndarray    # [V,T,N]
    depth: np.ndarray                     # [V,T,H,W], cropped prediction
    depth_confidence: np.ndarray          # [V,T,H,W], cropped prediction
    base_points: np.ndarray               # [V,T,H,W,3], cropped prediction
    predicted_K: np.ndarray               # [V,T,3,3], original pixel coordinates
    predicted_c2w: np.ndarray             # [V,T,4,4]
    source_obs_idx: np.ndarray            # [N], fixed view-major source indices
    metadata: dict


def sample_pixel_queries(field, xy):
    """Sample [U,D,H,W] at [N,2] integer-center pixels, including half borders.

    Cropped model fields use border replication for legal subpixel locations
    outside their outermost pixel centers; padding pixels never enter sampling.
    """
    height, width = field.shape[-2:]
    xy = xy.to(device=field.device, dtype=torch.float32)
    grid = torch.stack((2 * (xy[:, 0] + .5) / width - 1,
                        2 * (xy[:, 1] + .5) / height - 1), -1)
    grid = grid[None, :, None].expand(field.shape[0], -1, -1, -1)
    sampled = F.grid_sample(field.float(), grid, mode="bilinear",
                            padding_mode="border", align_corners=False)
    return sampled[..., 0].transpose(1, 2)  # [U,N,D]


def _original_image_fov(tracks, K, c2w, height, width):
    """Exclude padded target pixels using predicted geometry alone: [U,N,V]."""
    delta = tracks[:, :, None] - c2w[:, None, :, :3, 3]
    camera = torch.einsum("uncj,ucjk->unck", delta, c2w[..., :3, :3])
    pixels = torch.einsum("ucij,uncj->unci", K, camera)
    xy = pixels[..., :2] / camera[..., 2:].clamp_min(1e-8)
    return (torch.isfinite(xy).all(-1) & (camera[..., 2] > 0)
            & (xy[..., 0] >= -.5) & (xy[..., 0] < width - .5)
            & (xy[..., 1] >= -.5) & (xy[..., 1] < height - .5))


@torch.inference_mode()
def predict_tapvid_mv_queries(model, inputs: TAPVidMVInputs, device="cpu",
                             target_time_chunk=8, amp_dtype=None,
                             visibility_threshold=.5):
    """Encode all RGB observations and read pixel-query tracks without GT fixes.

    The model must already be in evaluation mode. Input images are never resized;
    black padding is added at the right/bottom to satisfy the patch size and the
    minimum two-patch canvas. No pixel mask is added to encoder attention.
    """
    if not isinstance(inputs, TAPVidMVInputs):
        raise TypeError("inputs must be TAPVidMVInputs, not a benchmark label object")
    # Revalidate tensor values in case their owner modified them after creation.
    input_type = ReleasedPixelInputs if isinstance(inputs, ReleasedPixelInputs) else TAPVidMVInputs
    inputs = input_type(inputs.video, inputs.queries_xytv, inputs.view_ids, inputs.frame_indices)
    if model.training:
        raise ValueError("Set the model to eval() before benchmark inference")
    if isinstance(target_time_chunk, bool) or not isinstance(target_time_chunk, int) or target_time_chunk <= 0:
        raise ValueError("target_time_chunk must be a positive integer")
    if not math.isfinite(visibility_threshold) or not 0 <= visibility_threshold <= 1:
        raise ValueError("visibility_threshold must be finite and in [0,1]")
    device = torch.device(device)
    views, frames, _, height, width = inputs.video.shape
    patch = model.config.patch_size
    padded_h = max(2 * patch, math.ceil(height / patch) * patch)
    padded_w = max(2 * patch, math.ceil(width / patch) * patch)
    images = inputs.video.to(device=device, dtype=torch.float32).reshape(views * frames, 3, height, width) / 255
    images = F.pad(images, (0, padded_w - width, 0, padded_h - height))[None]
    camera_ids = torch.tensor(inputs.view_ids, device=device).repeat_interleave(frames)[None]
    labels = inputs.frame_indices.to(device=device, dtype=torch.long)
    times = labels.repeat(views)[None]
    queries = inputs.queries_xytv.to(device=device, dtype=torch.float32)
    source_indices = queries[:, 3].long() * frames + queries[:, 2].long()
    n = len(queries)
    tracks = np.empty((frames, n, 3), dtype=np.float32)
    probabilities = np.empty((views, frames, n), dtype=np.float32)
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        scene = model.encode(images, camera_ids, times)
        K = scene.geometry["K"][0].reshape(views, frames, 3, 3)
        c2w = scene.geometry["c2w"][0].reshape(views, frames, 4, 4)
        for source in source_indices.unique(sorted=True).tolist():
            mask = source_indices == source
            point_ids = mask.nonzero().flatten().cpu().numpy()
            xy = queries[mask, :2]
            for start in range(0, frames, target_time_chunk):
                stop = min(frames, start + target_time_chunk)
                result = model.decode(scene, query_time_idx=labels[None, start:stop],
                                      source_obs_idx=torch.tensor([[source]], device=device),
                                      return_visibility=True)
                output_ids = result["output_camera_id"][0].tolist()
                if set(output_ids) != set(inputs.view_ids) or not result["output_camera_valid"].all():
                    raise ValueError("Decoder cameras do not match the complete input view set")
                order = [output_ids.index(camera) for camera in inputs.view_ids]
                field = result["trajectory"][0, 0, :, :height, :width].permute(0, 3, 1, 2)
                vis_field = result["visibility"][0, 0, :, order, :height, :width]
                with torch.autocast(device.type, enabled=False):
                    predicted = sample_pixel_queries(field, xy)
                    probability = sample_pixel_queries(vis_field, xy)
                    fov = _original_image_fov(predicted, K[:, start:stop].transpose(0, 1),
                                              c2w[:, start:stop].transpose(0, 1), height, width)
                    probability = probability * fov
                tracks[start:stop, point_ids] = predicted.cpu().numpy()
                probabilities[:, start:stop, point_ids] = probability.permute(2, 0, 1).cpu().numpy()
                del result, field, vis_field, predicted, probability
    def array(tensor):
        return tensor.detach().float().cpu().numpy().copy()
    depth = scene.geometry["depth"][0].reshape(views, frames, padded_h, padded_w)[..., :height, :width]
    depth_conf = scene.geometry["depth_conf"][0].reshape(views, frames, padded_h, padded_w)[..., :height, :width]
    base = scene.geometry["base_points"][0].reshape(views, frames, padded_h, padded_w, 3)[..., :height, :width, :]
    arrays = (tracks, probabilities, array(depth), array(depth_conf), array(base), array(K), array(c2w))
    if any(not np.isfinite(value).all() for value in arrays):
        raise ValueError("Model produced non-finite tracking, visibility, or geometry predictions")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Model visibility probabilities must be in [0,1]")
    metadata = {"adapter": "multicamera4rc-tapvid-mv-rgb-v1", "input_condition": "rgb_only_unknown_calibration",
                "coordinates": "native predicted world gauge", "alignment": "none",
                "query_anchor": "predicted source base point; no GT XYZ correction",
                "view_ids": list(inputs.view_ids), "frame_indices": labels.cpu().tolist(),
                "image_hw": [height, width], "padded_hw": [padded_h, padded_w],
                "resize": "none", "padding": "black, right/bottom only; encoder padding is unmasked",
                "visibility_threshold": float(visibility_threshold), "visibility_comparison": ">",
                "visibility_fov": "predicted geometry inside original image half-pixel bounds",
                "target_time_chunk": target_time_chunk, "input_observations": views * frames,
                "patch_size": patch, "amp_dtype": str(amp_dtype) if amp_dtype is not None else None,
                "decoder_memory": model.config.decoder_memory, "time_origin": model.config.time_origin,
                "encoder_passes": 1, "long_video_window_chaining": False,
                "official_scoring_implemented": False}
    if isinstance(inputs, ReleasedPixelInputs):
        metadata.update(pixel_query_policy=inputs.pixel_query_policy,
                        query_sampling="bilinear integer-center grid; constant border extension; coordinates unchanged")
    return TAPVidMVPrediction(tracks, probabilities > visibility_threshold, probabilities,
                             arrays[2], arrays[3], arrays[4], arrays[5], arrays[6],
                             source_indices.cpu().numpy().copy(), metadata)
