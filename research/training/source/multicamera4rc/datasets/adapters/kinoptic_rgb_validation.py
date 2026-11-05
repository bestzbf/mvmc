"""Independent synchronized-HD appearance checks for measured geometry.

RGB supplies a rejection mask, never a new depth value or a motion label.
Flat or non-discriminative checked patches remain unknown. Finite appearance
checks cannot prove visibility or resolve every possible periodic texture.
The local plane approximation may reject a correct curved surface.
"""
from __future__ import annotations

import cv2
import numpy as np

from .kinoptic_geometry import _intrinsic, _rigid


def _gray(image):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RGB verification requires uint8 HxWx3 RGB")
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _patch_score(source_patch, xy, depths, witness, matrix, translation,
                 offsets, min_std, absolute_tolerance_m, relative_tolerance,
                 check_occlusion=True):
    """ZNCC at the measured depth under a local source-frontoparallel plane."""
    pixels = xy[:, None, :] + offsets[None]
    homogeneous = np.concatenate((pixels, np.ones((*pixels.shape[:-1], 1))), -1)
    projected = homogeneous @ matrix.T + translation[None, None] / depths[:, None, None]
    positive = projected[..., 2] > 1e-8
    uv = projected[..., :2] / np.where(positive[..., None], projected[..., 2:], 1.)
    mx, my = uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32)
    gray, support, bound = witness
    inside = (positive & (mx >= 0) & (my >= 0)
              & (mx <= gray.shape[1]-1) & (my <= gray.shape[0]-1))
    observed = cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # Linear support requires the entire bilinear interpolation footprint.
    valid_support = cv2.remap(support.astype(np.float32), mx, my, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT) > .999
    valid = (inside & valid_support).all(axis=1)
    if check_occlusion and bound is not None:
        nearest = cv2.remap(bound, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        z = projected[..., 2] * depths[:, None]
        valid &= ((nearest <= 0) | (z <= nearest + absolute_tolerance_m
                                  + relative_tolerance*nearest)).all(axis=1)
    a = source_patch-source_patch.mean(axis=1, keepdims=True)
    b = observed-observed.mean(axis=1, keepdims=True)
    std_a, std_b = np.sqrt((a*a).mean(1)), np.sqrt((b*b).mean(1))
    valid &= (std_a >= min_std) & (std_b >= min_std)
    score = np.full(len(xy), -1., np.float32)
    score[valid] = ((a[valid]*b[valid]).mean(1)/(std_a[valid]*std_b[valid])).clip(-1, 1)
    return score, valid


def verify_rgb_depth(depth, image, K, E_w2c, witnesses, *, pixel_support=None,
                     patch_radius=2, min_std=4., min_ncc=.8, min_margin=.03,
                     depth_factors=(.5, .8, .9, 1.1, 1.2, 2.), min_witnesses=1,
                     max_angle_deg=45., chunk_size=8192,
                     absolute_tolerance_m=.02, relative_tolerance=.01):
    """Retain only original measurements with discriminative cross-HD support.

    A witness is a dict with ``image``, ``K``, ``E`` and optional rectification
    ``support`` and raw ``occlusion_depth``. RGBs must share the exact HD frame.
    Tests at alternative depths reject appearance ambiguity; they do not search
    for a replacement GT depth. Every positive candidate is checked.
    """
    for value, name, lower, upper in ((patch_radius, "patch_radius", 1, 4),
            (chunk_size, "chunk_size", 1, 8192), (min_witnesses, "min_witnesses", 1, 32)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lower <= value <= upper):
            raise ValueError(f"{name} must be an integer in [{lower},{upper}]")
    options = np.array([min_std, min_ncc, min_margin, max_angle_deg,
                        absolute_tolerance_m, relative_tolerance], np.float64)
    if (not np.isfinite(options).all() or not 0 < min_std <= 128
            or not 0 < min_ncc < 1 or not 0 <= min_margin < 1
            or not 0 < max_angle_deg < 90 or min(absolute_tolerance_m, relative_tolerance) < 0):
        raise ValueError("invalid RGB verification thresholds")
    factors = np.asarray(depth_factors, np.float64)
    if (factors.ndim != 1 or len(factors) < 2 or not np.isfinite(factors).all()
            or (factors <= 0).any() or np.any(np.abs(factors-1) < .01)
            or not (np.any(factors < 1) and np.any(factors > 1))):
        raise ValueError("alternative depth factors must bracket 1 without including it")
    gray = _gray(image)
    depth = np.asarray(depth, np.float32)
    if depth.shape != gray.shape or not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError("candidate depth must be finite nonnegative HxW")
    K, E = _intrinsic(K), _rigid(E_w2c, "source E")
    support = np.ones(depth.shape, bool) if pixel_support is None else np.asarray(pixel_support, bool)
    if support.shape != depth.shape:
        raise ValueError("source RGB support must match candidate depth")
    r = patch_radius
    yoff, xoff = np.mgrid[-r:r+1, -r:r+1]
    offsets = np.stack((xoff.ravel(), yoff.ravel()), -1)
    whole_patch = cv2.erode(support.astype(np.uint8), np.ones((2*r+1, 2*r+1), np.uint8),
                            borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    ys, xs = np.nonzero((depth > 0) & whole_patch)
    counts = np.zeros(depth.shape, np.uint8)
    checked_witnesses = 0
    camera_poses = set()
    for item in witnesses:
        other_gray = _gray(item["image"])
        other_K, other_E = _intrinsic(item["K"]), _rigid(item["E"], "witness E")
        identity = np.round(other_E, 10).tobytes()
        if identity in camera_poses:
            continue
        camera_poses.add(identity)
        relative = other_E @ np.linalg.inv(E)
        # Separate cameras are required. A duplicated source image/camera does
        # not constitute independent evidence of depth.
        if np.linalg.norm(relative[:3, 3]) < .01:
            continue
        if np.dot(E[2, :3], other_E[2, :3]) < np.cos(np.deg2rad(max_angle_deg)):
            continue
        other_support = np.asarray(item.get("support", np.ones_like(other_gray, bool)), bool)
        if other_support.shape != other_gray.shape:
            raise ValueError("witness RGB support shape mismatch")
        bound = item.get("occlusion_depth")
        if bound is not None:
            bound = np.asarray(bound, np.float32)
            if bound.shape != other_gray.shape or not np.isfinite(bound).all() or (bound < 0).any():
                raise ValueError("witness occlusion depth must be finite nonnegative HxW")
        witness = (other_gray, other_support, bound)
        matrix = other_K @ relative[:3, :3] @ np.linalg.inv(K)
        translation = other_K @ relative[:3, 3]
        checked_witnesses += 1
        if checked_witnesses > 32:
            raise ValueError("at most 32 independent witness cameras are supported")
        for start in range(0, len(xs), chunk_size):
            x, y = xs[start:start+chunk_size], ys[start:start+chunk_size]
            xy = np.stack((x, y), -1)
            patches = gray[y[:, None]+offsets[:, 1], x[:, None]+offsets[:, 0]]
            z = depth[y, x]
            score, valid = _patch_score(patches, xy, z, witness, matrix, translation,
                                 offsets, min_std, absolute_tolerance_m, relative_tolerance)
            alternate = np.full(len(x), -1., np.float32)
            for factor in factors:
                # Alternative hypotheses test appearance at a different image
                # position; their geometric visibility is not an admission
                # decision. All their RGB patches must be observable/textured.
                alternative, observed = _patch_score(patches, xy, z*factor, witness, matrix, translation,
                    offsets, min_std, absolute_tolerance_m, relative_tolerance, check_occlusion=False)
                valid &= observed
                alternate = np.maximum(alternate, alternative)
            passed = valid & (score >= min_ncc) & (score >= alternate+min_margin)
            counts[y, x] += passed
    filtered = np.where(counts >= min_witnesses, depth, 0).astype(np.float32)
    diagnostics = dict(candidate_pixels=int((depth > 0).sum()),
        source_patch_supported_pixels=len(xs), witnesses_checked=checked_witnesses,
        verified_pixels=int((filtered > 0).sum()), patch_radius=patch_radius,
        min_std=min_std, min_ncc=min_ncc, min_margin=min_margin,
        alternative_depth_factors=factors.tolist(), min_witnesses=min_witnesses,
        verification="synchronized_hd_zncc_measured_depth_vs_alternatives_v1")
    return filtered, diagnostics, counts
