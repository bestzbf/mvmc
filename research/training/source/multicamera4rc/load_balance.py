"""Deterministic window permutations of the existing distributed sample stream.

No rejection/resampling, resizing, weight changes or extra examples. The exact
global sample multiset is retained within each window, including DDP padding.
Save the activation anchor and cost catalogue to resume inside a window.
"""
from __future__ import annotations

import math
import random

import torch
from torch.utils.data.distributed import DistributedSampler


def validate_balance(config, world_size, batch_size, dataset_size):
    if not config:
        return
    if (config.get('version') != 1 or config['window_batches'] < 2
            or config['world_size'] != world_size or batch_size != 1
            or config['dataset_size'] != dataset_size
            or config['start_epoch'] < 0 or config['start_batch'] < 0):
        raise ValueError('load balancing requires its saved dataset, four/unchanged ranks and batch 1')
    for shape in config['image_hw'].values():
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError('invalid cost-catalogue image dimensions')


def sample_cost(dataset, index, image_hw, source_count=4, max_source_time_pairs=64):
    """Metadata-only estimate; imperfect costs affect speed, never sample GT."""
    plan = dataset.sample_plan(index)
    src, t, c = plan['source'], plan['num_times'], plan['cameras']
    name = src['display_name']
    if name not in image_hw:
        raise ValueError('missing fixed cost estimate for source ' + name)
    h, w = image_hw[name]
    n = math.ceil(h / 14) * math.ceil(w / 14)
    if src['view_mode'] == 'static_multiview':
        c, t = max(src['camera_choices']), 1
    observations = c * t
    q = min(source_count or observations, observations,
            max(1, max_source_time_pairs // t) if max_source_time_pairs else observations)
    # Linear projection/MLP and quadratic global attention, plus dense heads.
    # Geometry/camera-only adapters skip motion in _TrainingStep.
    tokens = observations * n
    motion = src['dense_kind'] in (
        'kubric', 'replica_flow', 'vkitti_scene_flow', 'pointodyssey_tracks', 'syn4d',
        'camera_joint_motion', 'camera_tracks', 'geometry_tracks', 'kinoptic_human',
    )
    return tokens + tokens * tokens / 32768 + (q * t * n * (6 + c) if motion else 0)


class BalancedDistributedSampler(DistributedSampler):
    def __init__(self, dataset, *, balance, **kwargs):
        super().__init__(dataset, **kwargs)
        validate_balance(balance, self.num_replicas, 1, len(dataset))
        self.balance = balance
        self.start_batch = 0

    def __iter__(self):
        # Reproduce DistributedSampler exactly before rearranging full windows.
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist() if self.shuffle else list(range(len(self.dataset)))
        if not self.drop_last:
            missing = self.total_size - len(indices)
            indices += (indices * math.ceil(missing / len(indices)))[:missing] if missing else []
        else:
            indices = indices[:self.total_size]
        cfg, world = self.balance, self.num_replicas
        if self.epoch < cfg['start_epoch']:
            yield from indices[self.rank::world]
            return
        anchor = cfg['start_batch'] if self.epoch == cfg['start_epoch'] else 0
        if anchor > self.num_samples:
            raise ValueError('balance activation cursor exceeds epoch')
        window = cfg['window_batches']
        # Completed windows will be skipped by _CursorBatchSampler; avoid
        # metadata IO for their indices. Rebuild the partially consumed window.
        first = anchor + max(0, (self.start_batch - anchor) // window) * window
        yield from indices[self.rank:first * world:world]
        for begin in range(first, self.num_samples, window):
            stop = min(begin + window, self.num_samples)
            block = indices[begin * world:stop * world]
            block.sort(key=lambda i: sample_cost(self.dataset, i, cfg['image_hw'],
                       cfg['source_count'], cfg['max_source_time_pairs']))
            groups = [block[i:i + world] for i in range(0, len(block), world)]
            rng = random.Random(self.seed + self.epoch * 1000003 + begin * 97)
            rng.shuffle(groups)
            for group in groups:
                # Avoid always assigning the greatest local cost to one rank.
                rng.shuffle(group)
                yield group[self.rank]

