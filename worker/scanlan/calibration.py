from __future__ import annotations

import numpy as np

from .io import PhaseData, RgbCameraModel


def distort_normalized(
    x: np.ndarray,
    y: np.ndarray,
    camera: RgbCameraModel,
) -> tuple[np.ndarray, np.ndarray]:
    if camera.model in {"pinhole", "none"} or not camera.distortion:
        return x, y
    coefficients = (*camera.distortion, 0.0, 0.0, 0.0, 0.0, 0.0)
    k1, k2, p1, p2, k3 = coefficients[:5]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    return (
        x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
        y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
    )


def project_rgb(
    points: np.ndarray,
    camera: RgbCameraModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    z = points[:, 2]
    safe_z = np.where(np.abs(z) > 1e-12, z, 1.0)
    x, y = distort_normalized(points[:, 0] / safe_z, points[:, 1] / safe_z, camera)
    return camera.fx * x + camera.cx, camera.fy * y + camera.cy, z


def depth_camera_points(
    depth: np.ndarray,
    phase: PhaseData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = phase.camera
    yy, xx = np.indices(depth.shape, dtype=np.float64)
    z = depth.astype(np.float64) / camera.depth_scale
    valid = (z > 0.0) & (z <= camera.max_depth_m)
    points = np.column_stack(
        (
            (xx[valid] - camera.cx) * z[valid] / camera.fx,
            (yy[valid] - camera.cy) * z[valid] / camera.fy,
            z[valid],
            np.ones(int(valid.sum()), dtype=np.float64),
        )
    )
    return points, valid, np.column_stack((xx[valid], yy[valid]))


def rgb_depth_zbuffer(
    depth: np.ndarray,
    phase: PhaseData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproject a depth image into the native RGB camera with nearest-depth collision handling."""
    from .io import effective_rgb_camera

    rgb_camera = effective_rgb_camera(phase)
    depth_points, depth_valid, _ = depth_camera_points(depth, phase)
    rgb_points = (phase.rgb_from_depth @ depth_points.T).T[:, :3]
    u, v, z = project_rgb(rgb_points, rgb_camera)
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    projected = (
        (z > 0.0)
        & np.isfinite(u)
        & np.isfinite(v)
        & (ui >= 0)
        & (ui < rgb_camera.width)
        & (vi >= 0)
        & (vi < rgb_camera.height)
    )
    zbuffer = np.full(rgb_camera.width * rgb_camera.height, np.inf, dtype=np.float64)
    flat = vi[projected] * rgb_camera.width + ui[projected]
    np.minimum.at(zbuffer, flat, z[projected])
    zbuffer = zbuffer.reshape(rgb_camera.height, rgb_camera.width)
    zbuffer[~np.isfinite(zbuffer)] = 0.0

    uv_map = np.full((*depth.shape, 2), np.nan, dtype=np.float32)
    visibility = np.zeros(depth.shape, dtype=bool)
    valid_indices = np.flatnonzero(depth_valid)
    uv_map.reshape(-1, 2)[valid_indices, 0] = u.astype(np.float32)
    uv_map.reshape(-1, 2)[valid_indices, 1] = v.astype(np.float32)
    accepted = np.zeros(len(z), dtype=bool)
    accepted[projected] = (
        np.abs(z[projected] - zbuffer[vi[projected], ui[projected]])
        <= np.maximum(0.015, z[projected] * 0.006)
    )
    visibility.reshape(-1)[valid_indices] = accepted
    return zbuffer.astype(np.float32), uv_map, visibility


def robust_depth_mask(depth_m: np.ndarray) -> np.ndarray:
    valid = depth_m > 0.0
    horizontal = np.zeros_like(valid)
    vertical = np.zeros_like(valid)
    horizontal[:, 1:] = np.abs(depth_m[:, 1:] - depth_m[:, :-1]) > np.maximum(
        0.04, np.minimum(depth_m[:, 1:], depth_m[:, :-1]) * 0.025
    )
    vertical[1:, :] = np.abs(depth_m[1:, :] - depth_m[:-1, :]) > np.maximum(
        0.04, np.minimum(depth_m[1:, :], depth_m[:-1, :]) * 0.025
    )
    discontinuity = horizontal | vertical
    discontinuity[:, :-1] |= horizontal[:, 1:]
    discontinuity[:-1, :] |= vertical[1:, :]
    return valid & ~discontinuity


def world_from_depth_opencv(camera_to_global: np.ndarray, image_y_up: bool) -> np.ndarray:
    matrix = np.asarray(camera_to_global, dtype=np.float64)
    if not image_y_up:
        return matrix
    # Known-pose fixtures historically used +Y up camera coordinates. Canonical
    # datasets always expose OpenCV camera axes (+X right, +Y down, +Z forward).
    return matrix @ np.diag([1.0, -1.0, 1.0, 1.0])


def world_from_rgb_camera(
    camera_to_global: np.ndarray,
    image_y_up: bool,
    rgb_from_depth: np.ndarray,
) -> np.ndarray:
    return world_from_depth_opencv(camera_to_global, image_y_up) @ np.linalg.inv(rgb_from_depth)
