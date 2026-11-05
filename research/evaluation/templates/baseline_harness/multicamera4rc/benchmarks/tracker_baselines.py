"""Tracking baselines driven exclusively by RGB-predicted common geometry.

The public VGGT-1B variant is not the TAPVid-MV paper's VGGT-Omega variant.
Missing native per-camera visibility remains missing; an any-camera score or
query-camera score is never broadcast into the other cameras.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .tapvid_mv import TAPVidMVInputs, TAPVidMVPrediction


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rgb_identity(inputs):
    if not isinstance(inputs, TAPVidMVInputs):
        raise TypeError("Only TAPVidMVInputs may cross the inference boundary")
    digest = hashlib.sha256()
    video = inputs.video.detach().cpu().contiguous().numpy()
    digest.update(json.dumps({"shape": list(video.shape), "dtype": str(video.dtype),
                              "view_ids": inputs.view_ids,
                              "frame_indices": inputs.frame_indices.tolist()}, sort_keys=True).encode())
    digest.update(memoryview(video).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True)
class PredictedGeometry:
    depth: np.ndarray                       # [V,T,H,W], camera z
    depth_confidence: np.ndarray            # [V,T,H,W]
    K: np.ndarray                           # [V,T,3,3], original pixels
    c2w: np.ndarray                         # [V,T,4,4]
    metadata: dict

    def validate(self, inputs):
        views, frames, _, height, width = inputs.video.shape
        expected = ((views, frames, height, width), (views, frames, height, width),
                    (views, frames, 3, 3), (views, frames, 4, 4))
        for value, shape in zip((self.depth, self.depth_confidence, self.K, self.c2w), expected):
            if np.shape(value) != shape or not np.isfinite(value).all():
                raise ValueError("Predicted geometry has a wrong shape or nonfinite values")
        if (self.depth <= 0).any():
            raise ValueError("Common predicted depth must be positive; no GT filling is permitted")
        if not np.allclose(self.c2w[..., 3, :], [0, 0, 0, 1], atol=1e-5):
            raise ValueError("Predicted c2w must be homogeneous matrices")
        if self.metadata.get("rgb_identity_sha256") != rgb_identity(inputs):
            raise ValueError("Geometry cache belongs to different RGB, view order or frame order")
        if (self.metadata.get("input_condition") != "rgb_only_unknown_calibration"
                or self.metadata.get("reconstruction_source") not in ("VGGT-1B", "VGGT-Omega-1B-512", "VGGT-Omega-1B-416-Reproduction")
                or self.metadata.get("ground_truth_inputs") != []):
            raise ValueError("Geometry cache is not admitted as RGB-predicted common geometry")
        return self

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, depth=self.depth, depth_confidence=self.depth_confidence,
                            predicted_K=self.K, predicted_c2w=self.c2w,
                            metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)))

    @classmethod
    def load(cls, path, inputs):
        with np.load(path, allow_pickle=False) as archive:
            result = cls(archive["depth"], archive["depth_confidence"], archive["predicted_K"],
                         archive["predicted_c2w"], json.loads(str(archive["metadata_json"])))
        return result.validate(inputs)


def preprocess_vggt_rgb(video, side=518, mode="pad"):
    """Tensor equivalent of the official PIL 'pad' mode, retaining all pixels.

    A pixel-center affine records the rounded resize and symmetric white pad,
    so both the dense depths and intrinsics can be restored to native pixels.
    """
    views, frames, _, height, width = video.shape
    if side <= 0 or side % 14:
        raise ValueError("VGGT canvas side must be a positive multiple of 14")
    if mode not in ("pad", "landscape_crop"):
        raise ValueError("Unsupported VGGT preprocessing mode")
    if width >= height:
        resized_w, resized_h = side, max(14, round(height * side / width / 14) * 14)
    else:
        resized_h, resized_w = side, max(14, round(width * side / height / 14) * 14)
    canvas_h, canvas_w = ((resized_h, resized_w) if mode == "landscape_crop" and width >= height else (side, side))
    top, left = (canvas_h - resized_h) // 2, (canvas_w - resized_w) // 2
    sx, sy = resized_w / width, resized_h / height
    transform = torch.tensor([[sx, 0, left + (sx - 1) / 2],
                              [0, sy, top + (sy - 1) / 2], [0, 0, 1]], dtype=torch.float32)
    source = video.detach().cpu().permute(0, 1, 3, 4, 2).numpy()
    processed = []
    for image in source.reshape(views * frames, height, width, 3):
        resized = Image.fromarray(image.round().astype(np.uint8), mode="RGB").resize(
            (resized_w, resized_h), Image.Resampling.BICUBIC)
        tensor = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float() / 255
        processed.append(F.pad(tensor, (left, canvas_w - resized_w - left, top, canvas_h - resized_h - top), value=1))
    return torch.stack(processed), transform, (top, left, resized_h, resized_w)


def preprocess_omega_rgb(video, resolution=512, mode="balanced"):
    """Official Omega bicubic sizing on complete supported-aspect RGB images.

    The published datasets fall inside [0.5,2.0]. Reject extreme shapes instead
    of silently following the official loader's content crop and losing queries.
    """
    views, frames, _, height, width = video.shape
    if mode not in ("balanced", "max_size") or resolution <= 0 or resolution % 16:
        raise ValueError("Omega requires balanced/max_size preprocessing and a multiple-of-16 resolution")
    aspect = height / width
    if not .5 <= aspect <= 2.:
        raise ValueError("Extreme aspect ratio would require content cropping; full-field evaluation rejects this input")
    if mode == "balanced":
        tokens = (resolution // 16) ** 2
        w_patches = np.sqrt(tokens / aspect)
        h_patches = tokens / w_patches
        target_h, target_w = max(1, int(np.round(h_patches))) * 16, max(1, int(np.round(w_patches))) * 16
    elif aspect >= 1:
        target_h, target_w = resolution, max(16, int(np.round(resolution / aspect / 16)) * 16)
    else:
        target_h, target_w = max(16, int(np.round(resolution * aspect / 16)) * 16), resolution
    sx, sy = target_w / width, target_h / height
    transform = torch.tensor([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]], dtype=torch.float32)
    source = video.detach().cpu().permute(0, 1, 3, 4, 2).numpy()
    processed = []
    for image in source.reshape(views * frames, height, width, 3):
        resized = Image.fromarray(image.round().astype(np.uint8), mode="RGB").resize((target_w, target_h), Image.Resampling.BICUBIC)
        processed.append(torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float() / 255)
    return torch.stack(processed), transform, (0, 0, target_h, target_w)


def _load_official_code(root, name):
    code = Path(root) / "code" / name
    if not code.exists():
        raise FileNotFoundError(code)
    sys.path.insert(0, str(code))
    return code


def _official_revision(code):
    return subprocess.check_output(["git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip()


class VGGTCommonReconstructor:
    """One shared predicted world gauge for all cameras and timesteps in a clip."""

    def __init__(self, root="@@BENCHMARK_ROOT@@", device="cuda", variant="VGGT-1B", preprocess_mode=None):
        self.root, self.device, self.variant = Path(root), torch.device(device), variant
        preprocess_mode = preprocess_mode or ("landscape_crop" if variant == "VGGT-1B" else "balanced")
        if preprocess_mode not in (("pad", "landscape_crop") if variant == "VGGT-1B" else ("balanced", "max_size")):
            raise ValueError("Unsupported VGGT preprocessing mode")
        self.preprocess_mode = preprocess_mode
        if variant == "VGGT-1B":
            self.code = _load_official_code(root, "vggt")
            from vggt.models.vggt import VGGT
            self.model = VGGT()
            self.checkpoint = self.root / "weights/vggt/model.pt"
            self.side = 518
        elif variant in ("VGGT-Omega-1B-512", "VGGT-Omega-1B-416-Reproduction"):
            self.code = _load_official_code(root, "vggt-omega")
            from vggt_omega.models import VGGTOmega
            self.model = VGGTOmega()
            filename = "vggt_omega_1b_512.pt" if variant == "VGGT-Omega-1B-512" else "vggt_omega_1b_416_reproduce.pt"
            self.checkpoint = self.root / "weights/vggt-omega" / filename
            self.side = 512 if variant == "VGGT-Omega-1B-512" else 416
        else:
            raise ValueError("Unsupported reconstruction variant")
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.checkpoint_sha256 = file_sha256(self.checkpoint)

    @torch.inference_mode()
    def predict(self, inputs):
        inputs = TAPVidMVInputs.from_rgb(inputs)
        preprocess = preprocess_vggt_rgb if self.variant == "VGGT-1B" else preprocess_omega_rgb
        images, transform, crop = preprocess(inputs.video, self.side, self.preprocess_mode)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]), torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            result = self.model(images.to(self.device))
        if self.variant == "VGGT-1B":
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            extrinsic, intrinsics = pose_encoding_to_extri_intri(result["pose_enc"], images.shape[-2:])
        else:
            from vggt_omega.utils.pose_enc import encoding_to_camera
            extrinsic, intrinsics = encoding_to_camera(result["pose_enc"], images.shape[-2:])
        extrinsic, intrinsics = extrinsic[0].float(), intrinsics[0].float()
        views, frames, _, height, width = inputs.video.shape
        top, left, resized_h, resized_w = crop
        depth = result["depth"][0, ..., 0].float()[:, None, top:top + resized_h, left:left + resized_w]
        confidence = result["depth_conf"][0].float()[:, None, top:top + resized_h, left:left + resized_w]
        depth = F.interpolate(depth, (height, width), mode="bilinear", align_corners=False)[:, 0]
        confidence = F.interpolate(confidence, (height, width), mode="bilinear", align_corners=False)[:, 0]
        native_K = torch.linalg.inv(transform.to(self.device)) @ intrinsics
        w2c = torch.eye(4, device=self.device)[None].repeat(views * frames, 1, 1)
        w2c[:, :3, :] = extrinsic
        c2w = torch.linalg.inv(w2c)
        def array(value, shape):
            return value.detach().float().cpu().numpy().reshape(shape).copy()
        metadata = {"schema": "rgb-predicted-common-geometry-v1", "input_condition": "rgb_only_unknown_calibration",
                    "ground_truth_inputs": [], "reconstruction_source": self.variant,
                    "paper_exact_reconstruction": False, "rgb_identity_sha256": rgb_identity(inputs),
                    "code_commit": _official_revision(self.code), "checkpoint_sha256": self.checkpoint_sha256,
                    "image_hw": [height, width], "model_hw": list(images.shape[-2:]), "preprocess_mode": self.preprocess_mode,
                    "rgb_preprocess": ("official Omega PIL bicubic " + self.preprocess_mode + "; patch16; supported full aspect, no content cropping" if self.variant != "VGGT-1B" else
                                       "official PIL bicubic pad mode; max side 518; shorter side rounded to 14; white center pad" if self.preprocess_mode == "pad" else
                                       "official PIL bicubic crop mode for landscape/square without spatial cropping; portrait uses pad to preserve full image"),
                    "native_to_model_pixel_affine": transform.tolist(), "input_order": "view-major, then time",
                    "joint_rgb_observations": views * frames, "geometry_alignment": "none",
                    "amp_dtype": "bfloat16" if self.device.type == "cuda" else None,
                    "sdpa_backends": ["flash", "efficient", "math"],
                    "sdpa_cudnn_disabled": "cuDNN Frontend plan failure observed on B300"}
        geometry = PredictedGeometry(array(depth, (views, frames, height, width)),
                                     array(confidence, (views, frames, height, width)),
                                     array(native_K, (views, frames, 3, 3)),
                                     array(c2w, (views, frames, 4, 4)), metadata)
        return geometry.validate(inputs)


def lift_pixel_tracks(xy, depth, K, c2w):
    """Lift [T,N,2] tracks with predicted z-depth and geometry of one view."""
    xy, depth, K, c2w = (torch.as_tensor(value, dtype=torch.float32) for value in (xy, depth, K, c2w))
    frames, points, _ = xy.shape
    height, width = depth.shape[-2:]
    grid = torch.stack((2 * (xy[..., 0] + .5) / width - 1,
                        2 * (xy[..., 1] + .5) / height - 1), -1)[:, :, None]
    z = F.grid_sample(depth[:, None], grid, mode="bilinear", padding_mode="border", align_corners=False)[:, 0, :, 0]
    rays = torch.einsum("tij,tnj->tni", torch.linalg.inv(K), torch.cat([xy, torch.ones(frames, points, 1)], -1))
    camera_xyz = rays * z[..., None]
    world = torch.einsum("tij,tnj->tni", c2w[:, :3, :3], camera_xyz) + c2w[:, None, :3, 3]
    return world


def predicted_queries_world(inputs, geometry):
    geometry.validate(inputs)
    queries = inputs.queries_xytv.detach().cpu().float()
    result = torch.empty(len(queries), 4)
    result[:, 0] = queries[:, 2]
    for view in queries[:, 3].long().unique().tolist():
        ids = (queries[:, 3].long() == view).nonzero().flatten()
        t = queries[ids, 2].long()
        tracks = queries[ids, :2][None].expand(len(inputs.frame_indices), -1, -1)
        world = lift_pixel_tracks(tracks, geometry.depth[view], geometry.K[view], geometry.c2w[view])
        result[ids, 1:] = world[t, torch.arange(len(ids))]
    return result


def _prediction(inputs, geometry, tracks, metadata):
    geometry.validate(inputs)
    tracks = np.asarray(tracks, dtype=np.float32)
    if tracks.shape != (inputs.video.shape[1], len(inputs.queries_xytv), 3) or not np.isfinite(tracks).all():
        raise ValueError("Baseline produced invalid world tracks")
    # The scorer derives the same-pixel world grid from predicted depth/K/c2w.
    # No separate world point head is used by this composed baseline. Avoid
    # materializing hundreds of millions of identical dense ray computations.
    views, frames, _, height, width = inputs.video.shape
    queries = inputs.queries_xytv.cpu().numpy()
    metadata = {"input_condition": "rgb_only_unknown_calibration", "ground_truth_inputs": [],
                "reconstruction_source": geometry.metadata["reconstruction_source"],
                "paper_exact_reconstruction": geometry.metadata["paper_exact_reconstruction"],
                "geometry_rgb_identity_sha256": geometry.metadata["rgb_identity_sha256"],
                "geometry_checkpoint_sha256": geometry.metadata["checkpoint_sha256"],
                "geometry_code_commit": geometry.metadata["code_commit"],
                "per_view_visibility_supported": False, "visibility_supported": False,
                "visibility_policy": "missing per-view visibility; never broadcast native query/any-view visibility",
                "query_anchor": "single-view query pixel lifted using common predicted depth/K/c2w",
                "base_points_source": "scorer unprojection of common predicted depth/K/c2w; no separate point head",
                "coordinates": "common reconstruction's native predicted world gauge", "alignment": "none",
                "view_ids": list(inputs.view_ids), "frame_indices": inputs.frame_indices.tolist(), **metadata}
    return TAPVidMVPrediction(tracks, None, None, geometry.depth, geometry.depth_confidence, None,
                             geometry.K, geometry.c2w, (queries[:, 3] * frames + queries[:, 2]).astype(np.int64), metadata)


class CoTracker3GeometryBaseline:
    """Author's offline CoTracker3, query-view 2D tracks, common depth lifting."""

    def __init__(self, root="@@BENCHMARK_ROOT@@", device="cuda"):
        self.root, self.device = Path(root), torch.device(device)
        code = _load_official_code(root, "co-tracker")
        from cotracker.predictor import CoTrackerPredictor
        checkpoint = self.root / "weights/cotracker3/scaled_offline.pth"
        self.model = CoTrackerPredictor(checkpoint=str(checkpoint), offline=True, window_len=60, v2=False).to(self.device).eval()
        self.provenance = {"model": "CoTracker3-Offline + common predicted geometry", "code_commit": _official_revision(code),
                           "checkpoint_sha256": file_sha256(checkpoint), "native_visibility": "query-view-only"}

    @torch.inference_mode()
    def predict(self, inputs, geometry):
        inputs = TAPVidMVInputs.from_rgb(inputs)
        geometry.validate(inputs)
        frames, points = inputs.video.shape[1], len(inputs.queries_xytv)
        world = np.empty((frames, points, 3), np.float32)
        query_vis = np.empty((frames, points), bool)
        queries = inputs.queries_xytv.detach().cpu().float()
        for view in queries[:, 3].long().unique().tolist():
            ids = (queries[:, 3].long() == view).nonzero().flatten()
            native_queries = queries[ids][:, [2, 0, 1]][None].to(self.device)
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                xy, visibility = self.model(inputs.video[view][None].float().to(self.device), queries=native_queries,
                                            backward_tracking=True, grid_size=0)
            world[:, ids.numpy()] = lift_pixel_tracks(xy[0].cpu(), geometry.depth[view], geometry.K[view], geometry.c2w[view]).numpy()
            query_vis[:, ids.numpy()] = visibility[0].cpu().numpy()
        return _prediction(inputs, geometry, world, self.provenance | {"adapter": "cotracker3-common-geometry-v1",
                           "query_view_visibility": query_vis.tolist(), "bidirectional": True,
                           "depth_outside_image": "border sampling of predicted depth; no GT replacement"})


class MVTrackerGeometryBaseline:
    """Official MVTracker predictor with predicted 3D query anchors."""

    def __init__(self, root="@@BENCHMARK_ROOT@@", device="cuda", n_iters=6):
        self.root, self.device = Path(root), torch.device(device)
        code = _load_official_code(root, "mvtracker")
        spec = importlib.util.spec_from_file_location("pinned_mvtracker_hubconf", code / "hubconf.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module._build_model()
        checkpoint = self.root / "weights/mvtracker/mvtracker_200000_june2025.pth"
        state = module._extract_model_state(torch.load(checkpoint, map_location="cpu", weights_only=False))
        model.load_state_dict(state, strict=True)
        from mvtracker.models.evaluation_predictor_3dpt import EvaluationPredictor
        self.model = EvaluationPredictor(model, interp_shape=(384, 512), visibility_threshold=.5,
                                          grid_size=4, n_grids_per_view=1, local_grid_size=18, local_extent=50,
                                          sift_size=0, num_uniformly_sampled_pts=0, n_iters=n_iters).to(self.device).eval()
        self.provenance = {"model": "MVTracker + common predicted geometry", "code_commit": _official_revision(code),
                           "checkpoint_sha256": file_sha256(checkpoint), "native_visibility": "any-view-only",
                           "n_iters": n_iters, "interp_shape": [384, 512]}

    @torch.inference_mode()
    def predict(self, inputs, geometry):
        inputs = TAPVidMVInputs.from_rgb(inputs)
        geometry.validate(inputs)
        queries = predicted_queries_world(inputs, geometry)[None].to(self.device)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            # The pinned official core normalizes 0..255 with rgbs / 255.
            # Its README quickstart divides beforehand, but its evaluation
            # loader/core contract uses raw RGB. Never normalize twice.
            result = self.model(rgbs=inputs.video[None].float().to(self.device),
                                depths=torch.as_tensor(geometry.depth[:, :, None], device=self.device)[None],
                                intrs=torch.as_tensor(geometry.K, device=self.device)[None],
                                extrs=torch.linalg.inv(torch.as_tensor(geometry.c2w, device=self.device))[None, :, :, :3],
                                query_points_3d=queries,
                                query_points_view=inputs.queries_xytv[:, 3].long()[None].to(self.device))
        return _prediction(inputs, geometry, result["traj_e"][0].cpu().numpy(), self.provenance | {
                           "adapter": "mvtracker-common-geometry-v1", "any_view_visibility": result["vis_e"][0].cpu().numpy().tolist()})


class TAPIP3DGeometryBaseline:
    """Official TAPIP3D, run in each query camera using the shared reconstruction."""

    def __init__(self, root="@@BENCHMARK_ROOT@@", device="cuda", n_iters=6):
        self.root, self.device, self.n_iters = Path(root), torch.device(device), n_iters
        code = _load_official_code(root, "TAPIP3D")
        # TAPIP3D's upstream package uses top-level 'models', 'utils',
        # 'datasets' and 'training'. Use a dedicated process/environment.
        import models as tapip_models
        from utils.inference_utils import inference
        self.inference = inference
        checkpoint = self.root / "weights/tapip3d/tapip3d_final.pth"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = deepcopy(state["cfg"])
        # The released checkpoint includes every encoder weight. Upstream also
        # downloads CoTracker solely for initialization, immediately overwriting
        # it with these same full-state weights. Disable that redundant network
        # initialization; strict=True proves no random parameter remains.
        config["model"]["encoder"]["pretrained"] = False
        self.model = tapip_models.from_config(config["model"], image_size=config["train_dataset"]["resolution"])
        self.model.load_state_dict(state["weight"], strict=True)
        self.model.set_eval_mode("raw")
        self.model.to(self.device).eval()
        self.provenance = {"model": "TAPIP3D + common predicted geometry", "code_commit": _official_revision(code),
                           "checkpoint_sha256": file_sha256(checkpoint), "native_visibility": "query-view-only",
                           "n_iters": n_iters, "grid_size": 8, "inference_hw": [384, 512],
                           "checkpoint_loading": "strict full state; redundant CoTracker initialization download disabled"}

    @torch.inference_mode()
    def predict(self, inputs, geometry):
        inputs = TAPVidMVInputs.from_rgb(inputs)
        geometry.validate(inputs)
        frames, points = inputs.video.shape[1], len(inputs.queries_xytv)
        height, width = inputs.video.shape[-2:]
        world = np.empty((frames, points, 3), np.float32)
        query_vis = np.empty((frames, points), bool)
        queries = predicted_queries_world(inputs, geometry)
        for view in inputs.queries_xytv[:, 3].long().unique().tolist():
            ids = (inputs.queries_xytv[:, 3].long() == view).nonzero().flatten()
            rgb = F.interpolate(inputs.video[view].float().to(self.device) / 255, (384, 512), mode="bilinear", align_corners=False)
            depth = F.interpolate(torch.as_tensor(geometry.depth[view], device=self.device)[:, None], (384, 512), mode="bilinear", align_corners=False)[:, 0]
            affine = torch.tensor([[512 / width, 0, (512 / width - 1) / 2],
                                    [0, 384 / height, (384 / height - 1) / 2], [0, 0, 1]], device=self.device)
            K = affine @ torch.as_tensor(geometry.K[view], device=self.device)
            w2c = torch.linalg.inv(torch.as_tensor(geometry.c2w[view], device=self.device))
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]), torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                xyz, visibility = self.inference(model=self.model, video=rgb, depths=depth, intrinsics=K,
                                                  extrinsics=w2c, query_point=queries[ids].to(self.device),
                                                  num_iters=self.n_iters, grid_size=8, bidrectional=True, vis_threshold=.9)
            world[:, ids.numpy()] = xyz.detach().float().cpu().numpy().reshape(frames, len(ids), 3)
            query_vis[:, ids.numpy()] = visibility.cpu().numpy().reshape(frames, len(ids))
        return _prediction(inputs, geometry, world, self.provenance | {
                           "adapter": "tapip3d-common-geometry-v1", "query_view_visibility": query_vis.tolist(), "bidirectional": True})
