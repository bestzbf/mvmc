"""Independent RGB-only stereo evidence used only to reject measured GT.

The input RGBs must be undistorted pinhole views of the exact same HD frame.
StereoSGBM never sees candidate depth. Its disparities are evidence masks,
never replacement depth labels or a claim of complete visibility ground truth.
"""
from __future__ import annotations

import cv2
import numpy as np

from .kinoptic_geometry import _intrinsic, _rigid


RGB_EDGE_THRESHOLD = 48
RGB_EDGE_RADIUS = 1


def _rgb_gray(image):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("stereo validation requires uint8 HxWx3 RGB")
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _integer(value, name, minimum, maximum):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not minimum <= value <= maximum):
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
    return int(value)


def _support(value, shape):
    if value is None:
        return np.ones(shape, bool)
    result = np.asarray(value, bool)
    if result.shape != shape:
        raise ValueError("RGB support shape mismatch")
    return result


def _texture(gray, minimum_std, size):
    image = gray.astype(np.float32)
    mean = cv2.boxFilter(image, -1, (size, size), normalize=True)
    square = cv2.boxFilter(image * image, -1, (size, size), normalize=True)
    return np.maximum(square - mean * mean, 0) >= minimum_std * minimum_std


def _native_rgb_edges(image):
    """Conservatively flag strong edges on the original candidate RGB grid.

    Both pixels beside an adjacent RGB change >=48 are marked, then expanded
    by one pixel. This is an uncertainty mask, not foreground segmentation.
    It is never applied to the independent SGBM matcher or used to add depth.
    """
    value = np.asarray(image, dtype=np.int16)
    edges = np.zeros(value.shape[:2], bool)
    dx = np.max(np.abs(value[:, 1:] - value[:, :-1]), axis=-1) >= RGB_EDGE_THRESHOLD
    dy = np.max(np.abs(value[1:] - value[:-1]), axis=-1) >= RGB_EDGE_THRESHOLD
    edges[:, 1:] |= dx
    edges[:, :-1] |= dx
    edges[1:] |= dy
    edges[:-1] |= dy
    kernel = np.ones((2 * RGB_EDGE_RADIUS + 1, 2 * RGB_EDGE_RADIUS + 1), np.uint8)
    return cv2.dilate(edges.astype(np.uint8), kernel,
        borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def _full_bilinear_support(mask, xy, positive):
    """Require every nonzero interpolation neighbour, without remap rounding."""
    height, width = mask.shape
    inside = (positive & np.isfinite(xy).all(axis=-1)
        & (xy[:, 0] >= 0) & (xy[:, 0] <= width - 1)
        & (xy[:, 1] >= 0) & (xy[:, 1] <= height - 1))
    safe = np.where(inside[:, None], xy, 0.)
    low = np.floor(safe).astype(np.int64)
    high = np.ceil(safe).astype(np.int64)
    return (inside & mask[low[:, 1], low[:, 0]] & mask[low[:, 1], high[:, 0]]
            & mask[high[:, 1], low[:, 0]] & mask[high[:, 1], high[:, 0]])


def _rectify_gray(gray, support, K, R, P, image_size, roi):
    xmap, ymap = cv2.initUndistortRectifyMap(K, np.zeros(5), R, P[:3, :3],
                                          image_size, cv2.CV_32FC1)
    output = cv2.remap(gray, xmap, ymap, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid = cv2.remap(support.astype(np.float32), xmap, ymap,
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) > .999
    # stereoRectify's ROI is an inscribed all-valid rectangle, not the entire
    # valid domain. Rotated cameras may even get an empty rectangle while
    # remap has many valid pixels. Check the actual interpolation footprint
    # rather than declaring everything outside that rectangle invalid.
    return output, valid


def _rgb_evidence(image, K, E, witness, pixel_support, *, max_long_side,
                  num_disparities, block_size, uniqueness_ratio,
                  lr_tolerance_px, min_texture_std, max_angle_deg):
    """Build independent evidence without receiving any candidate depth."""
    gray, other = _rgb_gray(image), _rgb_gray(witness["image"])
    if gray.shape != other.shape:
        raise ValueError("stereo HD views must have matching image sizes")
    other_K, other_E = _intrinsic(witness["K"]), _rigid(witness["E"], "witness E")
    support = _support(pixel_support, gray.shape)
    other_support = _support(witness.get("support"), other.shape)
    relative = other_E @ np.linalg.inv(E)
    diagnostics = dict(baseline_m=float(np.linalg.norm(relative[:3, 3])),
        matching_uses_candidate_depth=False, rgb_only_matched_pixels=0,
        vertical_baseline=False, disparity_sign_flipped=False)
    if diagnostics["baseline_m"] < .01:
        return None, dict(diagnostics, reason="same_or_negligible_baseline_camera")
    if np.dot(E[2, :3], other_E[2, :3]) < np.cos(np.deg2rad(max_angle_deg)):
        return None, dict(diagnostics, reason="camera_view_angle_exceeds_gate")
    original_h, original_w = gray.shape
    scale = min(1., max_long_side / max(gray.shape))
    width, height = max(1, round(original_w * scale)), max(1, round(original_h * scale))
    sx, sy = width / original_w, height / original_h
    affine = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1.]])
    if (height, width) != gray.shape:
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
        other = cv2.resize(other, (width, height), interpolation=cv2.INTER_AREA)
        support = cv2.resize(support.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA) > .999
        other_support = cv2.resize(other_support.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA) > .999
    scaled_K, scaled_other_K = affine @ K, affine @ other_K
    R1, R2, P1, P2, _, roi1, roi2 = cv2.stereoRectify(
        scaled_K, np.zeros(5), scaled_other_K, np.zeros(5), (width, height),
        np.ascontiguousarray(relative[:3, :3]),
        np.ascontiguousarray(relative[:3, 3]).reshape(3, 1), flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=1, newImageSize=(width, height))
    if not all(np.isfinite(value).all() for value in (R1, R2, P1, P2)):
        return None, dict(diagnostics, reason="invalid_rectification")
    left, left_support = _rectify_gray(gray, support, scaled_K, R1, P1, (width, height), roi1)
    right, right_support = _rectify_gray(other, other_support, scaled_other_K, R2, P2, (width, height), roi2)
    axis = int(abs(P2[1, 3]) > abs(P2[0, 3]))
    vertical = axis == 1
    flipped = bool(P2[axis, 3] > 0)
    if abs(P2[axis, 3]) < 1e-8:
        return None, dict(diagnostics, reason="degenerate_rectified_baseline")
    if vertical:
        left, right = left.T, right.T
        left_support, right_support = left_support.T, right_support.T
    if flipped:
        left, right = left[:, ::-1], right[:, ::-1]
        left_support, right_support = left_support[:, ::-1], right_support[:, ::-1]
    left, right = np.ascontiguousarray(left), np.ascontiguousarray(right)
    match_h, match_w = left.shape
    diagnostics.update(validation_image_hw=[height, width], matching_image_hw=[match_h, match_w],
                       vertical_baseline=vertical, disparity_sign_flipped=flipped,
                       num_disparities=num_disparities, block_size=block_size,
                       uniqueness_ratio=uniqueness_ratio, lr_tolerance_px=float(lr_tolerance_px),
                       rectification_alpha=1., zero_disparity_principal_points=True,
                       rectified_focal_px=float(P1[axis, axis]),
                       principal_point_difference_px=float(P1[axis, 2] - P2[axis, 2]),
                       opencv_inscribed_roi_left=list(map(int, roi1)),
                       opencv_inscribed_roi_right=list(map(int, roi2)),
                       rectified_support_policy="actual_bilinear_source_support")
    if match_w <= num_disparities + 2 * block_size:
        return None, dict(diagnostics, reason="matching_axis_too_short_for_disparity_range")
    kernel = np.ones((block_size, block_size), np.uint8)
    left_support = cv2.erode(left_support.astype(np.uint8), kernel,
        borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    right_support = cv2.erode(right_support.astype(np.uint8), kernel,
        borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    left_support &= _texture(left, min_texture_std, max(5, block_size))
    right_support &= _texture(right, min_texture_std, max(5, block_size))
    diagnostics.update(left_support_textured_pixels=int(left_support.sum()),
                       right_support_textured_pixels=int(right_support.sum()))
    settings = dict(minDisparity=0, numDisparities=num_disparities,
        blockSize=block_size, P1=8 * block_size ** 2, P2=32 * block_size ** 2,
        disp12MaxDiff=1, preFilterCap=31, uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=50, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM)
    # Right-to-left is another independent positive-disparity computation in
    # flipped coordinates; flip its result back before checking consistency.
    disparity = cv2.StereoSGBM_create(**settings).compute(left, right).astype(np.float32) / 16.
    reverse = cv2.StereoSGBM_create(**settings).compute(
        np.ascontiguousarray(right[:, ::-1]), np.ascontiguousarray(left[:, ::-1]))
    reverse = np.ascontiguousarray(reverse[:, ::-1], dtype=np.float32) / 16.
    y, x = np.indices(left.shape, dtype=np.float32)
    right_x = x - disparity
    reverse_valid = right_support & (reverse > .5) & (reverse < num_disparities)
    sampled_reverse = cv2.remap(reverse, right_x, y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    sampled_support = cv2.remap(reverse_valid.astype(np.float32), right_x, y,
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) > .999
    valid = (left_support & (disparity > .5) & (disparity < num_disparities)
             & sampled_support & (np.abs(disparity - sampled_reverse) <= lr_tolerance_px))
    diagnostics.update(left_sgbm_in_range_pixels=int(((disparity > .5) & (disparity < num_disparities)).sum()),
                       left_right_consistent_pixels=int(valid.sum()))
    # Bilinear lookup below must not manufacture intermediate disparities at
    # a foreground boundary, even if both sides separately pass stereo LR.
    local_min = cv2.erode(disparity, np.ones((3, 3), np.uint8))
    local_max = cv2.dilate(disparity, np.ones((3, 3), np.uint8))
    valid &= local_max - local_min <= 1.5 + .01 * np.maximum(disparity, 0)
    diagnostics["rgb_only_matched_pixels"] = int(valid.sum())
    return dict(disparity=disparity, valid=valid, R1=R1, P1=P1, P2=P2,
                vertical=vertical, flipped=flipped), diagnostics


def verify_stereo_depth(depth, image, K, E_w2c, witness, *, pixel_support=None,
                        max_long_side=640, num_disparities=128, block_size=5,
                        uniqueness_ratio=15, lr_tolerance_px=1.,
                        absolute_disparity_tolerance_px=1.5,
                        relative_disparity_tolerance=.01,
                        min_texture_std=4., max_angle_deg=45., rgb_edge_guard=True):
    """Keep original-depth pixels consistent with RGB-only bidirectional SGBM.

    Returns ``(filtered_depth, diagnostics, accepted_mask)`` at the original
    candidate resolution. Rectification supports, texture, uniqueness, LR and
    depth-disparity agreement are mandatory. By default strong source/witness
    RGB edges on the original candidate grid also veto final labels, since
    block stereo can overlook a 1–2 pixel foreground. SGBM itself is unchanged.
    Unknown regions stay invalid; same-color occlusion is not certified.
    A witness contains ``image``, ``K``, ``E`` and optional ``support``.
    """
    max_long_side = _integer(max_long_side, "max_long_side", 32, 4096)
    num_disparities = _integer(num_disparities, "num_disparities", 16, 512)
    block_size = _integer(block_size, "block_size", 3, 11)
    uniqueness_ratio = _integer(uniqueness_ratio, "uniqueness_ratio", 1, 100)
    if not isinstance(rgb_edge_guard, (bool, np.bool_)):
        raise ValueError("rgb_edge_guard must be boolean")
    rgb_edge_guard = bool(rgb_edge_guard)
    if num_disparities % 16 or block_size % 2 == 0:
        raise ValueError("num_disparities must be a multiple of 16 and block_size must be odd")
    parameters = (lr_tolerance_px, absolute_disparity_tolerance_px,
                  relative_disparity_tolerance, min_texture_std, max_angle_deg)
    if (any(isinstance(value, (bool, np.bool_)) for value in parameters)
            or not np.isfinite(parameters).all()
            or not 0 < lr_tolerance_px <= 1 or not 0 < min_texture_std <= 128
            or absolute_disparity_tolerance_px < 0 or relative_disparity_tolerance < 0
            or not 0 < max_angle_deg < 90):
        raise ValueError("invalid stereo LR, texture, angle or disparity tolerances")
    gray = _rgb_gray(image)
    depth = np.asarray(depth, np.float32)
    if depth.shape != gray.shape or not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError("candidate depth must be finite nonnegative HxW matching RGB")
    K, E = _intrinsic(K), _rigid(E_w2c, "source E")
    original_support = _support(pixel_support, depth.shape)
    evidence, diagnostics = _rgb_evidence(image, K, E, witness, pixel_support,
        max_long_side=max_long_side, num_disparities=num_disparities, block_size=block_size,
        uniqueness_ratio=uniqueness_ratio, lr_tolerance_px=lr_tolerance_px,
        min_texture_std=min_texture_std, max_angle_deg=max_angle_deg)
    accepted = np.zeros(depth.shape, bool)
    diagnostics.update(candidate_pixels=int((depth > 0).sum()), verified_pixels=0,
        candidate_inside_pair_pixels=0, candidate_outside_disparity_search_pixels=0,
        candidate_with_rgb_stereo_support_pixels=0,
        verification="independent_rgb_bidirectional_sgbm_rejection_v1",
        rgb_edge_guard=rgb_edge_guard, rgb_edge_threshold=RGB_EDGE_THRESHOLD,
        rgb_edge_radius_px=RGB_EDGE_RADIUS,
        rgb_edge_grid="original_candidate_rgb_before_stereo_resize",
        rgb_edge_masks_evaluated=False, source_rgb_edge_pixels=0, witness_rgb_edge_pixels=0,
        candidate_stereo_consistent_pixels=0, candidate_source_edge_rejected_pixels=0,
        candidate_witness_support_rejected_pixels=0, candidate_witness_edge_rejected_pixels=0,
        matching_uses_rgb_edge_guard=False,
        absolute_disparity_tolerance_px=float(absolute_disparity_tolerance_px),
        relative_disparity_tolerance=float(relative_disparity_tolerance))
    if evidence is not None:
        ys, xs = np.nonzero((depth > 0) & original_support)
        rays = np.stack((xs, ys, np.ones(len(xs))), axis=-1) @ np.linalg.inv(K).T
        camera = rays * depth[ys, xs, None]
        if rgb_edge_guard:
            source_edges = _native_rgb_edges(image)
            witness_edges = _native_rgb_edges(witness["image"])
            witness_support = _support(witness.get("support"), witness_edges.shape)
            relative = _rigid(witness["E"], "witness E") @ np.linalg.inv(E)
            other_camera = camera @ relative[:3, :3].T + relative[:3, 3]
            positive_other = np.isfinite(other_camera).all(axis=-1) & (other_camera[:, 2] > 1e-8)
            other_homogeneous = other_camera @ _intrinsic(witness["K"]).T
            other_xy = other_homogeneous[:, :2] / np.where(positive_other[:, None], other_camera[:, 2:], 1.)
            source_edge_reject = source_edges[ys, xs]
            target_support = _full_bilinear_support(witness_support, other_xy, positive_other)
            target_edge_clear = _full_bilinear_support(~witness_edges, other_xy, positive_other)
            diagnostics.update(rgb_edge_masks_evaluated=True,
                source_rgb_edge_pixels=int(source_edges.sum()), witness_rgb_edge_pixels=int(witness_edges.sum()),
                verification="independent_rgb_bidirectional_sgbm_native_edge_rejection_v2")
        rectified = camera @ evidence["R1"].T
        homogeneous = np.concatenate((rectified, np.ones((len(xs), 1))), axis=-1)
        left = homogeneous @ evidence["P1"].T
        right = homogeneous @ evidence["P2"].T
        positive = (left[:, 2] > 1e-8) & (right[:, 2] > 1e-8)
        left = left[:, :2] / np.where(positive[:, None], left[:, 2:], 1.)
        right = right[:, :2] / np.where(positive[:, None], right[:, 2:], 1.)
        if evidence["vertical"]:
            left, right = left[:, ::-1], right[:, ::-1]
        match_h, match_w = evidence["disparity"].shape
        if evidence["flipped"]:
            left[:, 0] = match_w - 1 - left[:, 0]
            right[:, 0] = match_w - 1 - right[:, 0]
        expected = left[:, 0] - right[:, 0]
        finite = positive & np.isfinite(left).all(axis=-1) & np.isfinite(right).all(axis=-1)
        inside_pair = (finite
                  & (left[:, 0] >= 0) & (left[:, 0] <= match_w - 1)
                  & (left[:, 1] >= 0) & (left[:, 1] <= match_h - 1)
                  & (right[:, 0] >= 0) & (right[:, 0] <= match_w - 1)
                  & (right[:, 1] >= 0) & (right[:, 1] <= match_h - 1)
                  & (np.abs(left[:, 1] - right[:, 1]) < .01))
        in_range = (expected > .5) & (expected < num_disparities)
        inside = inside_pair & in_range
        diagnostics.update(candidate_inside_pair_pixels=int(inside_pair.sum()),
            candidate_outside_disparity_search_pixels=int((finite & ~in_range).sum()),
            candidate_expected_disparity_quantiles_px=np.quantile(expected[finite], [0, .05, .5, .95, 1]).tolist()
                if finite.any() else [])
        # OpenCV remap's map dimensions must stay below SHRT_MAX. Original
        # full-HD candidate sets therefore use bounded one-row lookup chunks.
        for start in range(0, len(xs), 16384):
            stop = min(len(xs), start + 16384)
            mx = left[start:stop, 0].astype(np.float32)[None]
            my = left[start:stop, 1].astype(np.float32)[None]
            observed = cv2.remap(evidence["disparity"], mx, my, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=-1)[0]
            supported = cv2.remap(evidence["valid"].astype(np.float32), mx, my,
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) > .999
            tolerance = absolute_disparity_tolerance_px + relative_disparity_tolerance * expected[start:stop]
            keep = (inside[start:stop] & supported[0]
                    & (np.abs(observed - expected[start:stop]) <= tolerance))
            diagnostics["candidate_with_rgb_stereo_support_pixels"] += int((inside[start:stop] & supported[0]).sum())
            diagnostics["candidate_stereo_consistent_pixels"] += int(keep.sum())
            if rgb_edge_guard:
                source_bad = source_edge_reject[start:stop]
                target_valid = target_support[start:stop]
                edge_clear = target_edge_clear[start:stop]
                diagnostics["candidate_source_edge_rejected_pixels"] += int((keep & source_bad).sum())
                diagnostics["candidate_witness_support_rejected_pixels"] += int((keep & ~source_bad & ~target_valid).sum())
                diagnostics["candidate_witness_edge_rejected_pixels"] += int((keep & ~source_bad & target_valid & ~edge_clear).sum())
                keep &= ~source_bad & target_valid & edge_clear
            accepted[ys[start:stop], xs[start:stop]] = keep
    filtered = np.where(accepted, depth, 0).astype(np.float32)
    diagnostics["verified_pixels"] = int(accepted.sum())
    return filtered, diagnostics, accepted
