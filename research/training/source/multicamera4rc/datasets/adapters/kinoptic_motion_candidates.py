"""Isolated Kinoptic motion evidence; none of these outputs is surface GT.

Official skeleton displacement and estimated surface correspondence deliberately
have different schemas. No function writes ``dense_pair_delta``, trajectory,
visibility, or displacement training targets. Admission needs independent motion
accuracy evidence in addition to the consistency checks implemented here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile

import cv2
import numpy as np

from .kinoptic_geometry import _intrinsic, _rigid
from .kinoptic_stereo_validation import _native_rgb_edges


def official_joint_pair(path, frames, times_ms, *, confidence=.3, max_speed_mps=8.):
    """Read two timestamp-checked official frames, pairing person ID and joint.

    Panoptic's centimetres become metres. These are joint-centre labels, never
    skin depth, material points, or per-camera visibility annotations.
    """
    if len(frames) != 2 or len(times_ms) != 2:
        raise ValueError('joint pair requires exactly two frames and timestamps')
    if any(isinstance(f, (bool, np.bool_)) or int(f) != f or f < 0 for f in frames):
        raise ValueError('frame labels must be nonnegative integers')
    times = np.asarray(times_ms, float)
    dt = float(np.diff(times)[0]/1000)
    if not np.isfinite(times).all() or times.min() <= 0 or not 0 < dt <= .5 or frames[0] >= frames[1]:
        raise ValueError('joint pair needs increasing frames within 500 ms')
    if not np.isfinite([confidence, max_speed_mps]).all() or not 0 < confidence <= 1 or max_speed_mps <= 0:
        raise ValueError('invalid confidence or speed gate')
    path = Path(path); before = path.stat(); documents = []; hashes = []
    with tarfile.open(path, 'r:') as archive:
        members = {m.name: m for m in archive if m.isfile()}
        for frame, time_ms in zip(frames, times):
            suffix = f'/body3DScene_{int(frame):08d}.json'
            matches = [m for name, m in members.items() if name.endswith(suffix)]
            if len(matches) != 1 or not 0 < matches[0].size <= 4*1024*1024:
                raise ValueError('missing, ambiguous, or oversized body annotation')
            raw = archive.extractfile(matches[0]).read()
            doc = json.loads(raw)
            t = doc.get('univTime')
            if isinstance(t, bool) or not isinstance(t, (int, float)) or not np.isfinite(t) or abs(t-time_ms) > 1.:
                raise ValueError('body annotation does not match the HD synchronization table')
            bodies = {}
            for body in doc.get('bodies', []):
                identity = body.get('id')
                if isinstance(identity, bool) or not isinstance(identity, int) or identity < 0:
                    continue
                if identity in bodies:
                    raise ValueError('ambiguous repeated person ID in one frame')
                key = 'joints19' if 'joints19' in body else 'joints15'
                n = 19 if key == 'joints19' else 15
                points = np.asarray(body.get(key, []), float)
                if points.size != 4*n:
                    continue
                points = points.reshape(n, 4).copy(); points[:, :3] *= .01
                bodies[identity] = (key, points)
            documents.append(bodies); hashes.append(hashlib.sha256(raw).hexdigest())
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('body annotation archive changed during reading')
    keys, start, end, conf = [], [], [], []
    rejected_speed = 0
    for identity in sorted(set(documents[0]) & set(documents[1])):
        kind0, a = documents[0][identity]; kind1, b = documents[1][identity]
        if kind0 != kind1:
            continue
        valid = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
        valid &= (a[:, 3] >= confidence) & (b[:, 3] >= confidence)
        speed = np.linalg.norm(b[:, :3]-a[:, :3], axis=-1)/dt
        rejected_speed += int((valid & (speed > max_speed_mps)).sum())
        for j in np.flatnonzero(valid & (speed <= max_speed_mps)):
            keys.append((identity, kind0, int(j))); start.append(a[j, :3]); end.append(b[j, :3])
            conf.append(min(a[j, 3], b[j, 3]))
    start = np.asarray(start, np.float32).reshape(-1, 3)
    end = np.asarray(end, np.float32).reshape(-1, 3)
    return dict(kind='official_sparse_joint_displacement', keys=keys,
                joint_start_world_m=start, joint_end_world_m=end,
                joint_delta_world_m=end-start, confidence=np.asarray(conf, np.float32),
                frames=list(map(int, frames)), times_ms=times.tolist(), member_sha256=hashes,
                rejected_speed=rejected_speed, surface_correspondence_gt=False,
                visibility_annotation_available=False, training_admitted=False)


def _sample(field, uv):
    """Bilinear flow sampling with explicit bounds and finite-value checks."""
    h, w = field.shape[:2]
    good = np.isfinite(uv).all(-1) & (uv[:, 0] >= 0) & (uv[:, 0] < w-1)
    good &= (uv[:, 1] >= 0) & (uv[:, 1] < h-1)
    safe = np.where(good[:, None], uv, 0.)
    x, y = np.floor(safe).astype(np.int64).T
    dx, dy = (safe-np.floor(safe)).T
    v = ((1-dx)[:, None]*(1-dy)[:, None]*field[y, x]
         + dx[:, None]*(1-dy)[:, None]*field[y, x+1]
         + (1-dx)[:, None]*dy[:, None]*field[y+1, x]
         + dx[:, None]*dy[:, None]*field[y+1, x+1])
    good &= np.isfinite(v).all(-1)
    return np.where(good[:, None], v, 0.), good


def _at_depth(depth, uv, pixel_support, edge):
    h, w = depth.shape
    valid = np.isfinite(uv).all(-1) & (uv[:, 0] >= 0) & (uv[:, 0] <= w-1)
    valid &= (uv[:, 1] >= 0) & (uv[:, 1] <= h-1)
    ij = np.floor(np.where(valid[:, None], uv, 0.)+.5).astype(np.int64)
    z = depth[ij[:, 1], ij[:, 0]]
    valid &= np.isfinite(z) & (z > 0) & pixel_support[ij[:, 1], ij[:, 0]] & ~edge[ij[:, 1], ij[:, 0]]
    return ij.astype(float), np.where(valid, z, 0.), valid


def _world(uv, z, K, E):
    ray = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T
    camera = ray*z[:, None]
    return (camera-E[:3, 3]) @ E[:3, :3]


def _project(world, K, E):
    cam = world @ E[:3, :3].T + E[:3, 3]
    p = cam @ K.T
    positive = np.isfinite(p).all(-1) & (cam[:, 2] > 0)
    uv = np.divide(p[:, :2], p[:, 2, None], out=np.zeros((len(p), 2)), where=positive[:, None])
    return uv, cam[:, 2], positive


def estimate_surface_candidates(images, depths, intrinsics, extrinsics, times_ms, *,
                                pixel_support=None, region_mask=None, flows=None,
                                cycle_tolerance_px=1., cross_view_tolerance_px=1.5,
                                depth_absolute_m=.02, depth_relative=.01,
                                max_speed_mps=8., max_candidates=100000,
                                min_independent_motion_px=1.):
    """Two views x two HD times, using measured depth and RGB-only flow.

    Every accepted start/end has measured depth. Both cameras must support
    forward/backward flow, source depth agreement, endpoint reprojection, and
    displacement agreement. Both RGB views must also show motion beyond the
    camera-induced flow; depth noise alone cannot create a moving-point label.
    A passed check establishes consistency, not accuracy.
    ``region_mask`` may be a body bounding-box proxy, never segmentation GT.
    """
    images = np.asarray(images); depths = np.asarray(depths, np.float32)
    if images.dtype != np.uint8 or images.ndim != 5 or images.shape[:2] != (2, 2) or images.shape[-1] != 3:
        raise ValueError('images must be uint8 [2 cameras,2 times,H,W,3]')
    if depths.shape != images.shape[:-1] or min(images.shape[2:4]) < 8:
        raise ValueError('depths must match RGB grids')
    if not np.isfinite(depths).all() or (depths < 0).any():
        raise ValueError('depth holes must be zero, not NaN or negative')
    if np.shape(intrinsics) != (2, 2, 3, 3) or np.shape(extrinsics) != (2, 2, 4, 4):
        raise ValueError('provide calibration separately for every camera and time')
    Ks = np.array([[_intrinsic(k) for k in row] for row in intrinsics])
    Es = np.array([[_rigid(e, 'E_w2c') for e in row] for row in extrinsics])
    centers = np.array([[-e[:3, :3].T @ e[:3, 3] for e in row] for row in Es])
    if (np.linalg.norm(centers[0]-centers[1], axis=-1) < .01).any():
        raise ValueError('independent views must have at least 1 cm camera baseline')
    times = np.asarray(times_ms, float)
    if times.shape != (2,) or not np.isfinite(times).all() or times[0] <= 0 or not 0 < times[1]-times[0] <= 500:
        raise ValueError('require two shared HD timestamps within 500 ms')
    params = [cycle_tolerance_px, cross_view_tolerance_px, depth_absolute_m, depth_relative, max_speed_mps]
    if not np.isfinite(params).all() or min(params) <= 0:
        raise ValueError('thresholds must be finite and positive')
    if (isinstance(min_independent_motion_px, (bool, np.bool_))
            or not np.isfinite(min_independent_motion_px) or min_independent_motion_px < 0):
        raise ValueError('motion evidence threshold must be finite and nonnegative')
    if isinstance(max_candidates, bool) or int(max_candidates) != max_candidates or max_candidates <= 0:
        raise ValueError('max_candidates must be a positive integer')
    supports = np.ones(depths.shape, bool) if pixel_support is None else np.asarray(pixel_support, bool)
    if supports.shape != depths.shape:
        raise ValueError('pixel support must match depth')
    region = np.ones(depths.shape[2:], bool) if region_mask is None else np.asarray(region_mask, bool)
    if region.shape != depths.shape[2:]:
        raise ValueError('region mask must match source grid')
    edges = np.array([[_native_rgb_edges(image) for image in row] for row in images])
    if flows is None:
        flow_rows = []
        for camera in range(2):
            gray = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in images[camera]]
            matcher = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            flow_rows.append([matcher.calc(gray[0], gray[1], None), matcher.calc(gray[1], gray[0], None)])
        flows = np.asarray(flow_rows)
    else:
        flows = np.asarray(flows)
    if flows.shape != (*depths.shape, 2) or not np.isfinite(flows).all():
        raise ValueError('flows must be finite [2 cameras, forward/backward,H,W,2]')
    y, x = np.where((depths[0, 0] > 0) & supports[0, 0] & region & ~edges[0, 0])
    if len(x) > max_candidates:
        select = np.linspace(0, len(x)-1, int(max_candidates), dtype=int)
        y, x = y[select], x[select]
    uv0 = np.c_[x, y].astype(float)
    z0 = depths[0, 0, y, x]
    p0 = _world(uv0, z0, Ks[0, 0], Es[0, 0])
    uv1 = uv0 + flows[0, 0, y, x]
    back, good_back = _sample(flows[0, 1], uv1)
    cycle_a = np.linalg.norm(uv1+back-uv0, axis=-1)
    uv1d, z1, depth1_ok = _at_depth(depths[0, 1], uv1, supports[0, 1], edges[0, 1])
    p1 = _world(uv1d, z1, Ks[0, 1], Es[0, 1])
    good = good_back & depth1_ok & (cycle_a <= cycle_tolerance_px)
    stages = dict(source_measured_pixels=len(uv0), temporal_depth_cycle_pixels=int(good.sum()))
    # Independently track the same starting surface in camera B.
    b0, expected_z0, positive0 = _project(p0, Ks[1, 0], Es[1, 0])
    b0d, bz0, b0_ok = _at_depth(depths[1, 0], b0, supports[1, 0], edges[1, 0])
    f_b, f_ok = _sample(flows[1, 0], b0d)
    b1 = b0d + f_b
    rev_b, rev_ok = _sample(flows[1, 1], b1)
    cycle_b = np.linalg.norm(b1+rev_b-b0d, axis=-1)
    b1d, bz1, b1_ok = _at_depth(depths[1, 1], b1, supports[1, 1], edges[1, 1])
    q0 = _world(b0d, bz0, Ks[1, 0], Es[1, 0])
    q1 = _world(b1d, bz1, Ks[1, 1], Es[1, 1])
    good &= positive0 & b0_ok & b1_ok & f_ok & rev_ok & (cycle_b <= cycle_tolerance_px)
    good &= np.abs(bz0-expected_z0) <= depth_absolute_m + depth_relative*np.maximum(expected_z0, 0)
    stages['both_view_depth_cycle_pixels'] = int(good.sum())
    expected_b1, expected_z1, positive1 = _project(p1, Ks[1, 1], Es[1, 1])
    reprojection = np.linalg.norm(expected_b1-b1, axis=-1)
    displacement_agreement = np.linalg.norm((p1-p0)-(q1-q0), axis=-1)
    tolerance = depth_absolute_m + depth_relative*np.minimum(z0, z1)
    speed = np.linalg.norm(p1-p0, axis=-1)/((times[1]-times[0])/1000)
    good &= positive1 & (reprojection <= cross_view_tolerance_px)
    good &= np.abs(bz1-expected_z1) <= depth_absolute_m + depth_relative*np.maximum(expected_z1, 0)
    good &= displacement_agreement <= tolerance
    good &= np.isfinite(speed) & (speed <= max_speed_mps)
    stages['cross_view_agreed_pixels'] = int(good.sum())
    static_a, _, static_a_ok = _project(p0, Ks[0, 1], Es[0, 1])
    static_b, _, static_b_ok = _project(q0, Ks[1, 1], Es[1, 1])
    motion_evidence = np.minimum(np.linalg.norm(uv1-static_a, axis=-1),
                                 np.linalg.norm(b1-static_b, axis=-1))
    if min_independent_motion_px > 0:
        good &= static_a_ok & static_b_ok & (motion_evidence >= min_independent_motion_px)
    stages['independent_rgb_motion_pixels'] = int(good.sum())
    return dict(kind='multiview_rgbd_surface_motion_candidate',
        candidate_source_uv=uv0[good].astype(np.float32),
        candidate_target_uv=uv1d[good].astype(np.float32),
        candidate_start_world_m=p0[good].astype(np.float32),
        candidate_delta_world_m=(p1-p0)[good].astype(np.float32),
        candidate_cycle_error_px=np.maximum(cycle_a, cycle_b)[good].astype(np.float32),
        candidate_cross_view_error_px=reprojection[good].astype(np.float32),
        candidate_delta_agreement_m=displacement_agreement[good].astype(np.float32),
        candidate_motion_evidence_px=motion_evidence[good].astype(np.float32),
        times_ms=times.tolist(), stages=stages, surface_correspondence_gt=False,
        visibility_annotation_available=False, training_admitted=False,
        pseudo_label=True, region_is_segmentation_gt=False,
        min_independent_motion_px=float(min_independent_motion_px))
