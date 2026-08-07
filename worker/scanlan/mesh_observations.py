from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .calibration import world_from_depth_opencv
from .io import load_depth


OUTSIDE_VIEW = "outside_view"
MISSING_DEPTH = "missing_depth"
SUPPORTED = "supported"
FREE_SPACE_VIOLATION = "free_space_violation"
OCCLUDED = "occluded"


@dataclass(frozen=True)
class DepthProjection:
    camera_points: np.ndarray
    image_u: np.ndarray
    image_v: np.ndarray
    pixel_x: np.ndarray
    pixel_y: np.ndarray
    in_view: np.ndarray
    observed_depth_m: np.ndarray
    tolerance_m: np.ndarray


def project_world_points_to_depth(
    points: np.ndarray,
    frame: Any,
    voxel_size_m: float,
    *,
    depth_image: np.ndarray | None = None,
) -> DepthProjection:
    """Project metric world points into one native RGB-D depth image."""

    world_points = np.asarray(points, dtype=np.float64)
    if world_points.ndim != 2 or world_points.shape[1] != 3:
        raise ValueError("World points must have shape (N, 3)")
    camera = frame.source.camera
    world_from_camera = world_from_depth_opencv(frame.camera_to_global, frame.image_y_up)
    camera_from_world = np.linalg.inv(world_from_camera)
    camera_points = world_points @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    z = camera_points[:, 2]
    safe_z = np.where(z > 1e-8, z, 1.0)
    image_u = camera.fx * camera_points[:, 0] / safe_z + camera.cx
    image_v = camera.fy * camera_points[:, 1] / safe_z + camera.cy
    pixel_x = np.rint(image_u).astype(np.int64)
    pixel_y = np.rint(image_v).astype(np.int64)
    in_view = (
        np.isfinite(camera_points).all(axis=1)
        & (z > 0.25)
        & (z <= camera.max_depth_m)
        & (pixel_x >= 0)
        & (pixel_x < camera.width)
        & (pixel_y >= 0)
        & (pixel_y < camera.height)
    )
    observed = np.zeros(len(world_points), dtype=np.float32)
    if not frame.depthless:
        if depth_image is None:
            source_frame = frame.source.frames[frame.frame_index]
            depth_image = load_depth(source_frame, camera)
        depth_m = np.asarray(depth_image, dtype=np.float32) / float(camera.depth_scale)
        if depth_m.shape != (camera.height, camera.width):
            raise ValueError("Depth image dimensions do not match its camera calibration")
        observed[in_view] = depth_m[pixel_y[in_view], pixel_x[in_view]]
    tolerance = np.maximum(
        max(float(voxel_size_m) * 2.0, 0.022),
        np.maximum(z, 0.0) * 0.009,
    )
    return DepthProjection(
        camera_points,
        image_u,
        image_v,
        pixel_x,
        pixel_y,
        in_view,
        observed,
        tolerance,
    )


def classify_world_points(
    points: np.ndarray,
    frame: Any,
    voxel_size_m: float,
    *,
    depth_image: np.ndarray | None = None,
) -> np.ndarray:
    """Classify proposed surface samples against one metric depth frame."""

    projection = project_world_points_to_depth(
        points,
        frame,
        voxel_size_m,
        depth_image=depth_image,
    )
    classifications = np.full(len(projection.camera_points), OUTSIDE_VIEW, dtype="<U20")
    classifications[projection.in_view] = MISSING_DEPTH
    if frame.depthless:
        return classifications
    observed = projection.observed_depth_m
    patch_depth = projection.camera_points[:, 2]
    valid = projection.in_view & (observed > 0.0) & np.isfinite(observed)
    delta = patch_depth - observed
    classifications[valid & (np.abs(delta) <= projection.tolerance_m)] = SUPPORTED
    classifications[valid & (delta < -projection.tolerance_m)] = FREE_SPACE_VIOLATION
    classifications[valid & (delta > projection.tolerance_m)] = OCCLUDED
    return classifications
