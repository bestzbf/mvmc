"""Explicit, checksummed native-pixel GT exclusions from an offline audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


class DepthValidityMasks:
    def __init__(self, root, manifest_sha256=None):
        self.root = Path(root).resolve()
        raw = (self.root/'manifest.json').read_bytes()
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        if manifest_sha256 is not None and self.manifest_sha256 != manifest_sha256:
            raise ValueError('depth validity manifest checksum mismatch')
        self.manifest = json.loads(raw)
        if self.manifest.get('format_version') != 1 or not isinstance(self.manifest.get('frames'), dict):
            raise ValueError('unsupported depth validity manifest')
        threshold = self.manifest.get('minimum_trusted_clip_fraction')
        if threshold is not None and (isinstance(threshold, bool) or not 0 < float(threshold) <= 1):
            raise ValueError('minimum trusted clip fraction must be in (0,1]')
        self.severe_frames = {(r['sequence'], int(r['frame']))
                              for r in self.manifest.get('statistics', []) if r.get('severely_corrupted')}

    def clip_admission(self, sequence, frames, original_depths, filtered_depths):
        before = sum(int((d > 0).sum()) for d in original_depths) if original_depths is not None else 0
        after = sum(int((d > 0).sum()) for d in filtered_depths) if filtered_depths is not None else 0
        fraction = after/before if before else 0.
        severe = any((sequence, int(frame)) in self.severe_frames for frame in frames)
        threshold = self.manifest.get('minimum_trusted_clip_fraction')
        quarantine = threshold is not None and severe and original_depths is not None and fraction < threshold
        return dict(quarantined=quarantine, valid_depth_before=before, valid_depth_after=after,
                    retained_depth_fraction=fraction, severe_frame_present=severe,
                    minimum_trusted_clip_fraction=threshold,
                    reason='insufficient trustworthy scene-normalization support' if quarantine else None)

    def load(self, sequence, frame, shape):
        entry = self.manifest['frames'].get(sequence, {}).get(str(int(frame)))
        if entry is None:
            return None
        path = (self.root/entry['path']).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('depth validity mask path escapes its root')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError(f'depth validity mask checksum mismatch: {sequence}/{frame}')
        with Image.open(path) as image:
            value = np.asarray(image)
        if tuple(entry['shape']) != tuple(shape) or value.shape != tuple(shape):
            raise ValueError('depth validity mask and native image shapes disagree')
        if value.dtype != np.uint8 or not np.isin(value, [0, 255]).all():
            raise ValueError('depth validity masks must be binary uint8 PNGs')
        return value > 0


def apply_depth_validity(depths, masks, uv=None, valids=None, visibs=None):
    """Apply one exclusion to geometry and native track observations together."""
    if depths is not None:
        if len(depths) != len(masks):
            raise ValueError('depth validity mask count disagrees with frames')
        depths = [d if m is None else np.where(m, d, 0).astype(np.float32)
                  for d, m in zip(depths, masks)]
    if uv is not None:
        if valids is None or visibs is None or len(uv) != len(masks):
            raise ValueError('depth validity needs matching native track masks')
        valids, visibs = valids.copy(), visibs.copy()
        for i, mask in enumerate(masks):
            if mask is None:
                continue
            h, w = mask.shape
            points = np.asarray(uv[i])
            finite = np.isfinite(points).all(-1)
            xy = np.rint(np.where(finite[:, None], points, -1)).astype(np.int64)
            inside = finite & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
            keep = np.zeros(len(points), bool)
            keep[inside] = mask[xy[inside, 1], xy[inside, 0]]
            valids[i] &= keep
            visibs[i] &= keep
    return depths, valids, visibs
