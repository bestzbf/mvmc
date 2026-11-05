"""CMU Kinoptic measurements registered into calibrated pinhole cameras.

The point-cloud demo's world chain is authoritative here: ``M_color`` maps
color coordinates into the Kinect local frame. The older color-projection
helper uses the opposite convention. ``domeCenter`` provides an independent
check of the selected direction and of the centimetre/metre conversion.
No function interpolates depth holes or constructs temporal correspondences.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


DEPTH_HW = (424, 512)
DEPTH_FRAME_BYTES = 2 * DEPTH_HW[0] * DEPTH_HW[1]


def _intrinsic(value, name="K"):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    if (matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0., 0., 1.], atol=1e-8, rtol=0)
            or abs(matrix[0, 1]) > 1e-8 or abs(matrix[1, 0]) > 1e-8):
        raise ValueError(f"{name} must have positive focal lengths and zero skew")
    return matrix.copy()


def _rigid(value, name):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite homogeneous 4x4 matrix")
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0., 0., 0., 1.], atol=1e-8, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5, rtol=0)
            or abs(np.linalg.det(rotation) - 1.) > 2e-5):
        raise ValueError(f"{name} must be a rigid transform without scale/reflection")
    return matrix.copy()


def _brown5(value):
    distortion = np.asarray(value, dtype=np.float64).reshape(-1)
    # The final two entries of CMU's seven-entry depth calibration include
    # 1000 and 0. They are NOT OpenCV rational distortion coefficients.
    if distortion.size < 5 or not np.isfinite(distortion[:5]).all():
        raise ValueError("depth distortion must contain five finite Brown coefficients")
    return distortion[:5].copy()


def read_depth_frame(path, index):
    """Read zero-based Kinect raw frame as horizontally flipped metres.

    Official MATLAB ``idx`` is one-based; Python callers pass the original
    zero-based sync-table index. A truncated stream is never silently padded.
    """
    if isinstance(index, (bool, np.bool_)) or int(index) != index or int(index) < 0:
        raise ValueError("depth index must be a non-negative integer")
    index = int(index)
    path = Path(path)
    size = path.stat().st_size
    if size == 0 or size % DEPTH_FRAME_BYTES:
        raise ValueError(f"raw depth file has an incomplete frame: {path}")
    if index >= size // DEPTH_FRAME_BYTES:
        raise IndexError(f"depth frame {index} outside {size // DEPTH_FRAME_BYTES} frames")
    with path.open("rb") as stream:
        stream.seek(index * DEPTH_FRAME_BYTES)
        data = np.fromfile(stream, dtype="<u2", count=DEPTH_HW[0] * DEPTH_HW[1])
    if data.size != DEPTH_HW[0] * DEPTH_HW[1]:
        raise ValueError(f"raw depth frame changed or truncated while reading: {path}")
    return np.ascontiguousarray(data.reshape(DEPTH_HW)[:, ::-1], dtype=np.float32) * np.float32(.001)


def depth_rays(K, D):
    """Undistorted normalized rays for the flipped 512x424 depth raster."""
    K, distortion = _intrinsic(K, "K_depth"), _brown5(D)
    y, x = np.indices(DEPTH_HW, dtype=np.float64)
    pixels = np.stack((x, y), axis=-1).reshape(-1, 1, 2)
    xy = cv2.undistortPoints(pixels, K, distortion).reshape(*DEPTH_HW, 2)
    rays = np.concatenate((xy, np.ones((*DEPTH_HW, 1))), axis=-1)
    if not np.isfinite(rays).all():
        raise ValueError("depth undistortion produced non-finite rays")
    return rays


def sensor_geometry(sensor, calib50):
    """Validate a Kinect sensor and return its raw-depth-to-world transform.

    Returned transforms act on column vectors. Kinect local coordinates use
    metres, while ``calib50['t']`` uses centimetres. The independent supplied
    dome centre must map to the Panoptic world origin within 1 mm.
    """
    K_depth = _intrinsic(sensor["K_depth"], "K_depth")
    distortion = _brown5(sensor["distCoeffs_depth"])
    M_depth = _rigid(sensor["M_depth"], "M_depth")
    M_color = _rigid(sensor["M_color"], "M_color")
    rotation = np.asarray(calib50["R"], dtype=np.float64)
    translation = np.asarray(calib50["t"], dtype=np.float64).reshape(-1)
    if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("calibration50 must contain 3x3 R and finite three-entry t")
    color_from_world = np.eye(4)
    color_from_world[:3, :3] = rotation
    color_from_world[:3, 3] = translation * .01
    color_from_world = _rigid(color_from_world, "calibration50 world-to-color")
    world_from_local = np.linalg.inv(color_from_world) @ np.linalg.inv(M_color)
    world_from_depth = world_from_local @ np.linalg.inv(M_depth)
    dome = np.asarray(sensor["domeCenter"], dtype=np.float64).reshape(-1)
    if dome.shape != (3,) or not np.isfinite(dome).all():
        raise ValueError("domeCenter must contain three finite metre coordinates")
    residual = float(np.linalg.norm((world_from_local @ np.r_[dome, 1.])[:3]))
    if residual > .001:
        raise ValueError(f"domeCenter/world origin mismatch {residual:.6g} m; check transform direction and units")
    return dict(K_depth=K_depth, dist_depth5=distortion,
                T_world_from_depth=world_from_depth,
                T_world_from_local=world_from_local,
                T_color_from_depth=np.linalg.inv(M_color) @ np.linalg.inv(M_depth),
                depth_camera_center_world=world_from_depth[:3, 3].copy(),
                world_dome_residual_m=residual)


def occluder_points(depth, rays, T, min_depth=.1, max_depth=8.):
    """Keep raw measured surfaces as occlusion evidence, never as GT labels.

    Depth edges, isolated measurements and points without a trustworthy normal
    still block a farther supervision candidate. They therefore must survive
    the stricter normal/neighbourhood filtering in :func:`world_points`.
    The broader depth interval is only for this conservative veto stream.
    """
    depth = np.asarray(depth, dtype=np.float64)
    rays = np.asarray(rays, dtype=np.float64)
    T = _rigid(T, "T_world_from_depth")
    if depth.ndim != 2 or min(depth.shape, default=0) < 1 or rays.shape != (*depth.shape, 3):
        raise ValueError("depth/rays must have matching nonempty HxW and HxWx3 shapes")
    if (isinstance(min_depth, (bool, np.bool_)) or isinstance(max_depth, (bool, np.bool_))
            or not np.isfinite([min_depth, max_depth]).all() or not 0 < min_depth < max_depth):
        raise ValueError("invalid occluder depth interval")
    valid = (np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
             & np.isfinite(rays).all(axis=-1)
             & np.isclose(rays[..., 2], 1., atol=1e-6, rtol=0))
    points = rays[valid] * depth[valid, None]
    points = points @ T[:3, :3].T + T[:3, 3]
    return points.astype(np.float32)


def world_points(depth, rays, T, min_depth=.3, max_depth=8., edge_threshold_m=.05):
    """Return every supported measurement with a source-facing unit normal.

    A complete 3x3 valid neighbourhood is required. A neighbourhood spanning
    more than ``edge_threshold_m`` in optical depth is discarded, preventing
    foreground/background edges and holes from acquiring fabricated normals.
    """
    depth = np.asarray(depth, dtype=np.float64)
    rays = np.asarray(rays, dtype=np.float64)
    T = _rigid(T, "T_world_from_depth")
    if depth.ndim != 2 or min(depth.shape) < 3 or rays.shape != (*depth.shape, 3):
        raise ValueError("depth/rays must have matching HxW and HxWx3 shapes, with H,W >= 3")
    if (not np.isfinite([min_depth, max_depth, edge_threshold_m]).all()
            or not 0 < min_depth < max_depth or edge_threshold_m <= 0):
        raise ValueError("invalid depth interval or edge threshold")
    ray_valid = np.isfinite(rays).all(axis=-1) & np.isclose(rays[..., 2], 1., atol=1e-6, rtol=0)
    valid = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth) & ray_valid
    kernel = np.ones((3, 3), np.uint8)
    supported = cv2.erode(valid.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    safe_depth = np.where(valid, depth, 0.)
    minimum = cv2.erode(safe_depth, kernel)
    maximum = cv2.dilate(safe_depth, kernel)
    supported &= maximum - minimum <= edge_threshold_m
    local = np.where(valid[..., None], rays * safe_depth[..., None], 0.)
    dx = local[1:-1, 2:] - local[1:-1, :-2]
    dy = local[2:, 1:-1] - local[:-2, 1:-1]
    normal = np.zeros_like(local)
    normal[1:-1, 1:-1] = np.cross(dx, dy)
    magnitude = np.linalg.norm(normal, axis=-1)
    supported &= np.isfinite(normal).all(axis=-1) & (magnitude > 1e-12)
    points, normal = local[supported], normal[supported]
    if not len(points):
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    normal[np.einsum("ij,ij->i", normal, points) > 0] *= -1
    points = points @ T[:3, :3].T + T[:3, 3]
    normal = normal @ T[:3, :3].T
    return points.astype(np.float32), normal.astype(np.float32)


def render_depth(points_world, normals_world, K, E_w2c, image_hw, max_view_angle_deg=80):
    """Point z-buffer into a rectified pinhole image, without filling holes.

    Pixel centres are integer array indices. Their nearest-neighbour support
    is [-0.5,W-0.5) x [-0.5,H-0.5), with ties rounded using floor(u+0.5).
    Normals face the source depth camera; only surfaces facing the target are
    admitted. The result is target-camera optical Z, never world Z/range.
    """
    points = np.asarray(points_world, dtype=np.float64)
    normals = np.asarray(normals_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or normals.shape != points.shape:
        raise ValueError("points/normals must be matching Nx3 arrays")
    K, E = _intrinsic(K), _rigid(E_w2c, "E_w2c")
    if len(image_hw) != 2 or any(isinstance(v, (bool, np.bool_)) or int(v) != v or v <= 0 for v in image_hw):
        raise ValueError("image_hw must contain positive integer height and width")
    height, width = map(int, image_hw)
    if not np.isfinite(max_view_angle_deg) or not 0 < max_view_angle_deg <= 90:
        raise ValueError("max_view_angle_deg must be in (0,90]")
    output = np.full(height * width, np.inf, dtype=np.float64)
    camera = points @ E[:3, :3].T + E[:3, 3]
    camera_normals = normals @ E[:3, :3].T
    distance = np.linalg.norm(camera, axis=-1)
    normal_size = np.linalg.norm(camera_normals, axis=-1)
    finite = np.isfinite(camera).all(axis=-1) & np.isfinite(camera_normals).all(axis=-1)
    valid = finite & (camera[:, 2] > 0) & (normal_size > 1e-12)
    cosine = np.zeros(len(points), dtype=np.float64)
    np.divide(-np.einsum("ij,ij->i", camera_normals, camera), normal_size * distance,
              out=cosine, where=valid)
    valid &= cosine >= math.cos(math.radians(max_view_angle_deg))
    camera = camera[valid]
    if len(camera):
        pixels = camera @ K.T
        uv = pixels[:, :2] / pixels[:, 2:]
        inside = (np.isfinite(uv).all(axis=-1) & (uv[:, 0] >= -.5) & (uv[:, 0] < width - .5)
                  & (uv[:, 1] >= -.5) & (uv[:, 1] < height - .5))
        ij = np.floor(uv[inside] + .5).astype(np.int64)
        np.minimum.at(output, ij[:, 1] * width + ij[:, 0], camera[inside, 2])
    output[~np.isfinite(output)] = 0.
    return output.reshape(height, width).astype(np.float32)


def render_occlusion_depth(points, K, E, image_hw, radius_px=2):
    """Render a conservative nearest-Z bound without any normal filtering.

    Each measurement blocks a square of radius ``radius_px`` in the target
    image. This is an occlusion-only splat: its extra pixels never acquire GT
    labels. An expanded buffer handles near-border measurements without wrap
    around or clipping their centres onto the image. Zero denotes no bound.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("occluder points must be an Nx3 array")
    K, E = _intrinsic(K), _rigid(E, "E_w2c")
    if len(image_hw) != 2 or any(isinstance(v, (bool, np.bool_)) or int(v) != v or v <= 0 for v in image_hw):
        raise ValueError("image_hw must contain positive integer height and width")
    if (isinstance(radius_px, (bool, np.bool_)) or not np.isfinite(radius_px)
            or int(radius_px) != radius_px or not 0 <= radius_px <= 32):
        raise ValueError("occlusion radius_px must be an integer in [0,32]")
    height, width = map(int, image_hw)
    radius = int(radius_px)
    canvas_h, canvas_w = height + 2 * radius, width + 2 * radius
    buffer = np.full((canvas_h, canvas_w), np.inf, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=-1)]
    camera = points @ E[:3, :3].T + E[:3, 3]
    camera = camera[np.isfinite(camera).all(axis=-1) & (camera[:, 2] > 0)]
    if len(camera):
        homogeneous = camera @ K.T
        uv = homogeneous[:, :2] / homogeneous[:, 2:]
        # The border exists only to let an outside splat reach the image; no
        # unseen geometry is synthesized and points outside this halo vanish.
        inside = (np.isfinite(uv).all(axis=-1)
                  & (uv[:, 0] >= -.5 - radius) & (uv[:, 0] < width - .5 + radius)
                  & (uv[:, 1] >= -.5 - radius) & (uv[:, 1] < height - .5 + radius))
        ij = np.floor(uv[inside] + .5).astype(np.int64) + radius
        np.minimum.at(buffer.reshape(-1), ij[:, 1] * canvas_w + ij[:, 0], camera[inside, 2])
    if radius:
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        buffer = cv2.erode(buffer, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=float("inf"))
        buffer = buffer[radius:radius + height, radius:radius + width].copy()
    buffer[~np.isfinite(buffer)] = 0.
    return buffer.astype(np.float32)


def filter_occluded_depth(candidate, occlusion, absolute_tolerance_m=.02, relative_tolerance=.01):
    """Veto a candidate behind a measured occluder; never create a depth label.

    The tolerance is ``absolute_tolerance_m + relative_tolerance * bound_Z``.
    Unknown/non-finite bounds cannot validate a candidate. Invalid candidates
    stay zero, and both inputs are left unchanged.
    """
    candidate = np.asarray(candidate, dtype=np.float64)
    bound = np.asarray(occlusion, dtype=np.float64)
    if candidate.ndim != 2 or candidate.shape != bound.shape:
        raise ValueError("candidate and occlusion must be matching HxW arrays")
    parameters = (absolute_tolerance_m, relative_tolerance)
    if (any(isinstance(v, (bool, np.bool_)) for v in parameters)
            or not np.isfinite(parameters).all() or any(v < 0 for v in parameters)):
        raise ValueError("occlusion tolerances must be finite and non-negative")
    candidate_valid = np.isfinite(candidate) & (candidate > 0)
    bound_valid = np.isfinite(bound) & (bound > 0)
    tolerance = absolute_tolerance_m + relative_tolerance * np.where(bound_valid, bound, 0.)
    rejected = candidate_valid & bound_valid & (candidate > bound + tolerance)
    return np.where(candidate_valid & ~rejected, candidate, 0.).astype(np.float32)
