from __future__ import annotations

import numpy as np

from .io import (
    FrameRecord,
    PhaseData,
    RgbCameraModel,
    frame_rgb_camera,
    frame_rgb_from_depth,
)


def distort_normalized(
    x: np.ndarray,
    y: np.ndarray,
    camera: RgbCameraModel,
) -> tuple[np.ndarray, np.ndarray]:
    if camera.model in {"pinhole", "none"}:
        return x, y
    if camera.model == "brown_conrady":
        if not camera.distortion:
            return x, y
        coefficients = (*camera.distortion, 0.0, 0.0, 0.0, 0.0, 0.0)
        k1, k2, p1, p2, k3 = coefficients[:5]
        denominator = None
    elif camera.model == "opencv_rational":
        if len(camera.distortion) != 8:
            raise ValueError("OpenCV rational calibration requires eight coefficients")
        k1, k2, p1, p2, k3, k4, k5, k6 = camera.distortion
        r2 = x * x + y * y
        denominator = 1.0 + k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2
    else:
        raise ValueError(f"Unsupported RGB lens model: {camera.model}")
    r2 = x * x + y * y
    numerator = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    radial = numerator if denominator is None else np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=np.abs(denominator) > 1e-12,
    )
    return (
        x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
        y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
    )


def scaled_pinhole_camera(
    camera: RgbCameraModel,
    max_dimension: int,
) -> RgbCameraModel:
    if max_dimension <= 0:
        raise ValueError("Pinhole output dimension must be positive")
    scale = min(1.0, max_dimension / max(camera.width, camera.height))
    width = max(1, round(camera.width * scale))
    height = max(1, round(camera.height * scale))
    scale_x = width / camera.width
    scale_y = height / camera.height
    return RgbCameraModel(
        width,
        height,
        camera.fx * scale_x,
        camera.fy * scale_y,
        (camera.cx + 0.5) * scale_x - 0.5,
        (camera.cy + 0.5) * scale_y - 0.5,
        "pinhole",
        (),
    )


def undistort_rgb_to_pinhole(
    image: np.ndarray,
    source_camera: RgbCameraModel,
    target_camera: RgbCameraModel,
    *,
    tile_rows: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample calibrated RGB directly onto a target pinhole ray grid."""
    image = np.asarray(image, dtype=np.uint8)
    if image.shape != (source_camera.height, source_camera.width, 3):
        raise ValueError("RGB dimensions must match their source calibration")
    if target_camera.model != "pinhole" or target_camera.distortion:
        raise ValueError("Undistortion output must use a pinhole camera")
    if tile_rows <= 0:
        raise ValueError("Undistortion tile size must be positive")
    if source_camera == target_camera:
        return image.copy(), np.ones(image.shape[:2], dtype=bool)

    output_image = np.zeros(
        (target_camera.height, target_camera.width, 3), dtype=np.uint8
    )
    output_valid = np.zeros((target_camera.height, target_camera.width), dtype=bool)
    target_x = np.arange(target_camera.width, dtype=np.float32)[None, :]
    normalized_x = (target_x - target_camera.cx) / target_camera.fx
    for top in range(0, target_camera.height, tile_rows):
        bottom = min(target_camera.height, top + tile_rows)
        target_y = np.arange(top, bottom, dtype=np.float32)[:, None]
        normalized_y = (target_y - target_camera.cy) / target_camera.fy
        source_x, source_y = distort_normalized(
            normalized_x, normalized_y, source_camera
        )
        source_u = source_camera.fx * source_x + source_camera.cx
        source_v = source_camera.fy * source_y + source_camera.cy
        valid = (
            np.isfinite(source_u)
            & np.isfinite(source_v)
            & (source_u >= 0.0)
            & (source_u <= source_camera.width - 1)
            & (source_v >= 0.0)
            & (source_v <= source_camera.height - 1)
        )
        safe_u = np.where(valid, source_u, 0.0)
        safe_v = np.where(valid, source_v, 0.0)
        x0 = np.floor(safe_u).astype(np.int32)
        y0 = np.floor(safe_v).astype(np.int32)
        x1 = np.minimum(x0 + 1, source_camera.width - 1)
        y1 = np.minimum(y0 + 1, source_camera.height - 1)
        wx = (safe_u - x0).astype(np.float32)
        wy = (safe_v - y0).astype(np.float32)
        sampled = (
            image[y0, x0].astype(np.float32) * ((1.0 - wx) * (1.0 - wy))[..., None]
            + image[y0, x1].astype(np.float32) * (wx * (1.0 - wy))[..., None]
            + image[y1, x0].astype(np.float32) * ((1.0 - wx) * wy)[..., None]
            + image[y1, x1].astype(np.float32) * (wx * wy)[..., None]
        )
        output_image[top:bottom][valid] = np.clip(
            np.rint(sampled[valid]), 0, 255
        ).astype(np.uint8)
        output_valid[top:bottom] = valid
    return output_image, output_valid


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
    frame: FrameRecord | None = None,
    output_camera: RgbCameraModel | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproject depth to RGB rays with nearest-depth collision handling."""
    rgb_camera = frame_rgb_camera(frame, phase) if frame is not None else phase.rgb_camera
    projection_camera = output_camera or rgb_camera
    rgb_from_depth = frame_rgb_from_depth(frame, phase) if frame is not None else phase.rgb_from_depth
    depth_points, depth_valid, _ = depth_camera_points(depth, phase)
    rgb_points = (rgb_from_depth @ depth_points.T).T[:, :3]
    source_u, source_v, z = project_rgb(rgb_points, rgb_camera)
    u, v, _ = project_rgb(rgb_points, projection_camera)
    finite_projection = np.isfinite(u) & np.isfinite(v)
    ui = np.rint(np.where(finite_projection, u, 0.0)).astype(np.int64)
    vi = np.rint(np.where(finite_projection, v, 0.0)).astype(np.int64)
    projected = (
        (z > 0.0)
        & finite_projection
        & (ui >= 0)
        & (ui < projection_camera.width)
        & (vi >= 0)
        & (vi < projection_camera.height)
    )
    zbuffer = np.full(
        projection_camera.width * projection_camera.height,
        np.inf,
        dtype=np.float64,
    )
    flat = vi[projected] * projection_camera.width + ui[projected]
    np.minimum.at(zbuffer, flat, z[projected])
    zbuffer = zbuffer.reshape(projection_camera.height, projection_camera.width)
    zbuffer[~np.isfinite(zbuffer)] = 0.0

    uv_map = np.full((*depth.shape, 2), np.nan, dtype=np.float32)
    visibility = np.zeros(depth.shape, dtype=bool)
    valid_indices = np.flatnonzero(depth_valid)
    uv_map.reshape(-1, 2)[valid_indices, 0] = source_u.astype(np.float32)
    uv_map.reshape(-1, 2)[valid_indices, 1] = source_v.astype(np.float32)
    accepted = np.zeros(len(z), dtype=bool)
    accepted[projected] = (
        np.abs(z[projected] - zbuffer[vi[projected], ui[projected]])
        <= np.maximum(0.015, z[projected] * 0.006)
    )
    accepted &= (
        np.isfinite(source_u)
        & np.isfinite(source_v)
        & (source_u >= 0.0)
        & (source_u <= rgb_camera.width - 1)
        & (source_v >= 0.0)
        & (source_v <= rgb_camera.height - 1)
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
