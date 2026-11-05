"""Multi-camera 4RC: one geometry backbone, per-source world trajectories."""

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .config import MultiCamera4RCConfig
from .encoder import MultiCameraEncoder
from .embeddings import SinusoidalFrameEmbedding
from .geometry import (ExtendedCameraDec, decode_camera, unproject_depth,
                       share_camera_intrinsics, replace_pose_intrinsics)
from .metadata import build_metadata, gather_observations, validate_observations
from .motion import MultiViewMotionDecoder, MotionDPT
from .vendor.heads.dualdpt import DualDPT
from .visibility import VisibilityHead


@dataclass
class EncodedScene:
    """Reusable differentiable features; do not reuse a freed graph across training steps."""
    features: list
    geometry: dict
    camera_id: torch.Tensor
    time_idx: torch.Tensor
    observation_valid: torch.Tensor
    image_size: tuple[int, int]
    execution_cache: dict = field(default_factory=dict, repr=False)


class MultiCamera4RC(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or MultiCamera4RCConfig()
        if isinstance(self.config, dict):
            self.config = MultiCamera4RCConfig.from_dict(self.config)
        cfg = self.config
        d = cfg.embed_dim
        # Names intentionally match the original Arc state_dict prefixes.
        self.backbone = MultiCameraEncoder(
            embed_dim=d, depth=cfg.encoder_depth, num_heads=cfg.encoder_heads,
            out_layers=cfg.out_layers, alt_start=cfg.alt_start,
            img_size=cfg.positional_image_size, patch_size=cfg.patch_size,
            ffn_layer=cfg.ffn_layer, attention_chunk_size=cfg.attention_chunk_size,
            checkpoint=cfg.gradient_checkpointing)
        self.head = DualDPT(dim_in=2*d, output_dim=2, features=cfg.dpt_features,
                            out_channels=cfg.dpt_channels, patch_size=cfg.patch_size)
        self.cam_dec = ExtendedCameraDec(dim_in=2*d)
        self.motion_decoder = MultiViewMotionDecoder(
            embed_dim=d, patch_size=cfg.patch_size, depth=cfg.motion_depth,
            num_heads=cfg.motion_heads, attention_chunk_size=cfg.attention_chunk_size,
            checkpoint=cfg.gradient_checkpointing)
        self.track_head = MotionDPT(embed_dim=d, features=cfg.dpt_features,
                                   out_channels=cfg.dpt_channels, patch_size=cfg.patch_size)
        self.frame_embedding = SinusoidalFrameEmbedding(d)
        self.target_projection = nn.Linear(d, d, bias=False)
        self.delta_projection = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.target_projection.weight)
        nn.init.zeros_(self.delta_projection.weight)
        self.visibility_head = VisibilityHead(
            motion_feature_dim=cfg.dpt_features//2, patch_feature_dim=d,
            query_chunk_size=cfg.visibility_chunk_size)
        if cfg.refinement_dim:
            from .refinement import TrajectoryRefiner
            self.trajectory_refiner = TrajectoryRefiner(cfg)
        else:
            self.trajectory_refiner = None

    def _time_origin(self, times, valid):
        if self.config.time_origin == "absolute":
            return torch.zeros_like(times[:, :1])
        # Use every valid observation, never just the current target subset.
        # Keep original frame labels in metadata and preserve all frame gaps.
        return times.masked_fill(~valid, torch.iinfo(times.dtype).max).amin(1, keepdim=True)

    def _cache_enabled(self):
        return (not self.training and not torch.is_grad_enabled()
                and self.trajectory_refiner is not None
                and getattr(self, "evaluation_cache_enabled", False))

    def _shared_inference_value(self, scene, key, factory):
        """Reuse exact, scene-constant tensors across source decodes in inference."""
        if not self._cache_enabled():
            return factory()
        if key in scene.execution_cache:
            return scene.execution_cache[key]
        value = factory()

        def tensor_bytes(item):
            if isinstance(item, torch.Tensor):
                return item.numel() * item.element_size()
            if isinstance(item, (tuple, list)):
                return sum(tensor_bytes(child) for child in item)
            return 0

        size = tensor_bytes(value)
        used = sum(tensor_bytes(item) for item in scene.execution_cache.values())
        if scene.camera_id.is_cuda:
            free, _ = torch.cuda.mem_get_info(scene.camera_id.device)
            budget = min(8 * 1024**3, max(0, free + size) // 3)
        else:
            budget = 8 * 1024**3
        if used + size <= budget:
            scene.execution_cache[key] = value
        return value

    def encode(self, images, camera_id, time_idx, observation_valid=None, *, principal_canvas=None):
        """Encode all observed cameras/times once; images are RGB in [0,1]."""
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must have shape [B,S,3,H,W]")
        h, w = images.shape[-2:]
        p = self.config.patch_size
        if min(h, w) < 2*p or h % p or w % p:
            raise ValueError(f"H and W must be multiples of {p} and at least {2*p}")
        if images.dtype == torch.uint8:
            images = images.float()/255
        elif not images.is_floating_point():
            raise ValueError("images must be uint8 or floating RGB in [0,1]")
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("images must be finite RGB in [0,1]; do not ImageNet-normalize twice")
        camera_id = torch.as_tensor(camera_id, device=images.device)
        time_idx = torch.as_tensor(time_idx, device=images.device)
        camera_id, time_idx, valid = validate_observations(camera_id, time_idx, observation_valid)
        if camera_id.shape != images.shape[:2]:
            raise ValueError("camera/time metadata must match images' [B,S] axes")
        features = self.backbone(images, camera_id, time_idx-self._time_origin(time_idx, valid), valid)
        # Regression and camera math remain float32 when the encoder uses AMP.
        with torch.autocast(device_type=images.device.type, enabled=False):
            def geometry_head(*flat):
                taps = [tuple(flat[i:i+3]) for i in range(0, len(flat), 3)]
                return dict(self.head(taps, h, w, patch_start_idx=0,
                                      chunk_size=self.config.geometry_chunk_size))
            flat = tuple(value.float() for tap in features for value in tap)
            geometry = (checkpoint(geometry_head, *flat, use_reentrant=False)
                        if self.config.gradient_checkpointing and torch.is_grad_enabled()
                        else geometry_head(*flat))
            pose = self.cam_dec(features[-1][1].float())
            K, c2w = decode_camera(pose, h, w, principal_canvas=(principal_canvas
                if self.config.principal_point_mode == "crop_original" else None))
            if self.config.intrinsics_sharing == "camera":
                # Native clips have identical preprocessing over time per
                # camera. Undo the optional known integer crop before pooling.
                affine = None
                if principal_canvas is not None:
                    affine = torch.eye(3, device=K.device, dtype=K.dtype).expand_as(K).clone()
                    affine[..., 0, 2] = -principal_canvas[..., 3]
                    affine[..., 1, 2] = -principal_canvas[..., 2]
                K = share_camera_intrinsics(K, camera_id, valid, affine)
                pose = replace_pose_intrinsics(pose, K, h, w, principal_canvas=(principal_canvas
                    if self.config.principal_point_mode == "crop_original" else None))
            base = unproject_depth(geometry["depth"], K, c2w)
            base = torch.where(valid[..., None, None, None], base, torch.zeros_like(base))
        geometry.update(K=K, c2w=c2w, camera_encoding=pose, base_points=base)
        return EncodedScene(features, geometry, camera_id, time_idx, valid, (h, w))

    def decode(self, scene, query_time_idx=None, source_obs_idx=None, return_visibility=True):
        """Decode dense source trajectories with the configured memory scope.

        All-observation memory is independent of the requested target subset.
        Target-time grouping is still used for time conditions and visibility.
        """
        meta = build_metadata(scene.camera_id, scene.time_idx, query_time_idx,
                              source_obs_idx, scene.observation_valid,
                              all_observed_times=self.trajectory_refiner is not None)
        if self._cache_enabled():
            signature = (meta.memory_index, meta.memory_valid, meta.internal_times)
            previous = scene.execution_cache.get("signature")
            if previous is None or not all(torch.equal(a, b) for a, b in zip(signature, previous)):
                scene.execution_cache.clear()
                scene.execution_cache["signature"] = tuple(x.clone() for x in signature)
        d = self.config.embed_dim
        h, w = scene.image_size
        b, sources = meta.source_index.shape
        internal_count = meta.internal_times.shape[1]
        camera_count = meta.camera_values.shape[1]
        l = (h//self.config.patch_size)*(w//self.config.patch_size)
        patch_grid = (h//self.config.patch_size, w//self.config.patch_size)
        memories, grouped_features, pooled_times = [], [], []
        all_observations = self.config.decoder_memory == "all_observations"
        for tap_index, (patch, _, time) in enumerate(scene.features):
            global_patch = patch[..., d:]
            needs_visibility_features = return_visibility and tap_index == len(scene.features)-1
            cached_visibility = (self._cache_enabled()
                                 and "visibility_prepared" in scene.execution_cache)
            if not all_observations or (needs_visibility_features and not cached_visibility):
                identity_memory = (getattr(self, '_stream_identity_memory', False)
                    and b == 1 and camera_count == 1 and internal_count == global_patch.shape[1]
                    and bool(meta.memory_valid.all())
                    and torch.equal(meta.memory_index.reshape(-1), torch.arange(internal_count, device=global_patch.device)))
                if identity_memory:
                    grouped = global_patch[:, :, None]
                else:
                    grouped = gather_observations(global_patch, meta.memory_index)
                    grouped = torch.where(meta.memory_valid[..., None, None], grouped, 0)
            if needs_visibility_features:
                grouped_features.append(None if cached_visibility else grouped)
            memories.append(global_patch.reshape(b, -1, d) if all_observations
                            else grouped.reshape(b, internal_count, camera_count*l, d))
            tokens = gather_observations(time[..., d:], meta.memory_index)
            weights = meta.memory_valid[..., None].to(tokens.dtype)
            pooled_times.append((tokens*weights).sum(2)/weights.sum(2).clamp_min(1))
        if all_observations:
            memory_camera = scene.camera_id[..., None].expand(-1, -1, l).reshape(b, -1)
            memory_valid = scene.observation_valid[..., None].expand(-1, -1, l).reshape(b, -1)
        else:
            memory_camera = meta.camera_values[:, None, :, None].expand(
                b, internal_count, camera_count, l).reshape(b, internal_count, camera_count*l)
            memory_valid = meta.memory_valid[..., None].expand(
                b, internal_count, camera_count, l).reshape(b, internal_count, camera_count*l)
        origin = self._time_origin(scene.time_idx, scene.observation_valid)
        target_pe = self.target_projection(self.frame_embedding(meta.internal_times-origin))
        if return_visibility or self.trajectory_refiner is not None:
            target_depth, target_conf, target_K, target_c2w = self._shared_inference_value(
                scene, "target_geometry", lambda: (
                    gather_observations(scene.geometry["depth"], meta.memory_index),
                    gather_observations(scene.geometry["depth_conf"], meta.memory_index),
                    gather_observations(scene.geometry["K"], meta.memory_index),
                    gather_observations(scene.geometry["c2w"], meta.memory_index)))
        prepared, target_prepared, refinement_scale = None, None, None
        target_pyramid, visibility_prepared = None, None
        # Bound additional live activations on long/native-resolution clips.
        # Estimate both K and V as FP32 even under AMP. Falling back changes
        # execution only, with the same weights, full queries and supervision.
        kv_bytes = sum(memory.numel() * 8 * self.config.motion_depth for memory in memories)
        use_kv_cache = (self.config.reuse_decoder_kv and sources > 1
                        and kv_bytes <= self.config.decoder_kv_cache_limit_mb * 2**20)
        prepared_memories = ([self.motion_decoder.prepare_memory(memory, patch_grid) for memory in memories]
                             if use_kv_cache else [None] * len(memories))
        if self.trajectory_refiner is not None:
            with torch.autocast(device_type=scene.camera_id.device.type, enabled=False):
                prepared = self._shared_inference_value(scene, "refinement_features",
                    lambda: self.trajectory_refiner.prepare(scene.features[-1][0], patch_grid))
                target_prepared = self._shared_inference_value(scene, "refinement_targets",
                    lambda: gather_observations(prepared, meta.memory_index))
                refinement_scale = self._shared_inference_value(scene, "refinement_scale",
                    lambda: self.trajectory_refiner.scene_scale(
                        scene.geometry["depth"], scene.observation_valid))
                if (self._cache_enabled() or self.config.reuse_shared_computation) and self.config.refinement_matching:
                    target_pyramid = self._shared_inference_value(scene, "refinement_pyramid",
                        lambda: self.trajectory_refiner.prepare_pyramid(target_prepared))
        if return_visibility and (self._cache_enabled() or self.config.reuse_shared_computation):
            with torch.autocast(device_type=scene.camera_id.device.type, enabled=False):
                cached = scene.execution_cache.get("visibility_prepared") if self._cache_enabled() else None
                if cached is not None:
                    visibility_prepared = cached
                else:
                    visibility_prepared = self._shared_inference_value(scene, "visibility_prepared",
                        lambda: self.visibility_head.prepare_targets(grouped_features[-1].float(), patch_grid))
        trajectories, displacements, confidences = [], [], []
        visibilities, visibility_logits, projection_valids = [], [], []
        refinement_updates, refinement_matches = [], []

        def render_source(taps, base, anchor_slot, source_valid,
                          source_feature, source_camera):
            # Checkpoint the complete dense head, including visibility. Its
            # full-resolution feature tensor is an intermediate, not a saved
            # output. Whole images/times are still computed together.
            with torch.autocast(device_type=scene.camera_id.device.type, enabled=False):
                motion = self.track_head(taps, output_hw=(h, w))
                raw = motion["displacement"]
                anchor = gather_observations(raw, anchor_slot)
                delta = raw-anchor[:, None]
                xyz = base[:, None]+delta
                refinement = None
                if self.trajectory_refiner is not None:
                    refinement = self.trajectory_refiner(
                        xyz, motion["features"], source_feature, target_prepared,
                        target_depth, target_conf, target_K, target_c2w, meta.memory_valid,
                        source_camera[:, None] == meta.camera_values,
                        meta.internal_times, meta.internal_valid, anchor_slot, refinement_scale, target_pyramid)
                    delta = delta + refinement["correction"]
                    xyz = base[:, None] + delta
                delta = torch.where(source_valid[:, None, None, None, None], delta, 0)
                xyz = torch.where(source_valid[:, None, None, None, None], xyz, 0)
                confidence = torch.where(source_valid[:, None, None, None], motion["confidence"], 0)
                result = dict(trajectory=xyz, displacement=delta, confidence=confidence)
                if refinement is not None:
                    result.update(refinement_update=refinement["correction_relative_norm"],
                                  refinement_match_fraction=refinement["match_fraction"])
                if return_visibility:
                    vis = self.visibility_head(
                        trajectory=xyz, motion_features=motion["features"],
                        target_features=(None if visibility_prepared is not None else grouped_features[-1].float()),
                        target_depth=target_depth, target_depth_conf=target_conf,
                        target_K=target_K, target_c2w=target_c2w,
                        target_valid=meta.memory_valid, patch_grid=patch_grid, prepared_targets=visibility_prepared)
                    active = source_valid[:, None, None, None, None]
                    result.update(visibility=vis["visibility"]*active,
                                  visibility_logits=vis["visibility_logits"],
                                  projection_valid=vis["projection_valid"] & active)
                return result

        for source_slot in range(sources):
            source_index = meta.source_index[:, source_slot]
            source_time = meta.source_times[:, source_slot]
            source_camera = gather_observations(scene.camera_id, source_index)
            relative = meta.internal_times-source_time[:, None]
            delta_pe = self.delta_projection(self.frame_embedding(relative))
            decoded = []
            for m, (patch, _, _) in enumerate(scene.features):
                src = gather_observations(patch[..., d:], source_index)
                condition = pooled_times[m]+target_pe+delta_pe
                decoded.append(self.motion_decoder(src, memories[m], condition,
                                source_camera, memory_camera, memory_valid, patch_grid, prepared_memories[m]))
            base = gather_observations(scene.geometry["base_points"], source_index)
            source_feature = (gather_observations(prepared, source_index) if prepared is not None
                              else base.new_empty(0))
            args = ([x.float() for x in decoded], base, meta.source_slot[:, source_slot],
                    meta.source_valid[:, source_slot], source_feature, source_camera)
            del decoded
            if not torch.is_grad_enabled() and source_slot+1 == sources:
                memories.clear()
                prepared_memories.clear()
                del patch, global_patch, src, condition, time, _
                if visibility_prepared is not None:
                    grouped_features.clear()
                    if 'grouped' in locals():
                        del grouped
            rendered = (checkpoint(render_source, *args, use_reentrant=False)
                        if self.config.gradient_checkpointing and torch.is_grad_enabled()
                        else render_source(*args))
            if self.trajectory_refiner is not None:
                refinement_updates.append(rendered["refinement_update"])
                refinement_matches.append(rendered["refinement_match_fraction"])
            if return_visibility:
                visibilities.append(gather_observations(rendered["visibility"], meta.target_slot))
                visibility_logits.append(gather_observations(rendered["visibility_logits"], meta.target_slot))
                projection_valids.append(gather_observations(rendered["projection_valid"], meta.target_slot))
            trajectories.append(gather_observations(rendered["trajectory"], meta.target_slot))
            displacements.append(gather_observations(rendered["displacement"], meta.target_slot))
            confidences.append(gather_observations(rendered["confidence"], meta.target_slot))
        output = dict(scene.geometry)
        output.update(trajectory=torch.stack(trajectories, 1),
                      displacement=torch.stack(displacements, 1),
                      motion_confidence=torch.stack(confidences, 1),
                      source_obs_idx=meta.source_index, source_valid=meta.source_valid,
                      source_time_idx=meta.source_times, query_time_idx=meta.query_times,
                      camera_id=scene.camera_id, time_idx=scene.time_idx,
                      observation_valid=scene.observation_valid,
                      output_camera_id=meta.camera_values, output_camera_valid=meta.camera_valid,
                      target_observation_valid=gather_observations(meta.memory_valid, meta.target_slot))
        if return_visibility:
            output.update(visibility=torch.stack(visibilities, 1),
                          visibility_logits=torch.stack(visibility_logits, 1),
                          projection_valid=torch.stack(projection_valids, 1))
        if self.trajectory_refiner is not None:
            output.update(refinement_update=torch.stack(refinement_updates, 1),
                          refinement_match_fraction=torch.stack(refinement_matches, 1))
        return output

    def forward(self, images, camera_id, time_idx, query_time_idx=None,
                source_obs_idx=None, observation_valid=None, return_visibility=True, *, principal_canvas=None):
        return self.decode(self.encode(images, camera_id, time_idx, observation_valid,
                                       principal_canvas=principal_canvas),
                           query_time_idx, source_obs_idx, return_visibility)

    def load_4rc_weights(self, checkpoint, strict_base=True):
        from .checkpoint import load_4rc_weights
        return load_4rc_weights(self, checkpoint, strict_base=strict_base)

    @classmethod
    def from_checkpoint(cls, checkpoint, config=None, strict_base=True):
        from .checkpoint import config_from_checkpoint, read_checkpoint
        data = read_checkpoint(checkpoint)
        if config is None:
            config = config_from_checkpoint(data)
        model = cls(config)
        model.load_4rc_weights(data, strict_base=strict_base)
        return model
