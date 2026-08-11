from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .io import PhaseData, load_color, load_depth


def depth_to_world_points(
    phase: PhaseData,
    frame_index: int,
    pixel_stride: int = 2,
    depth_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    frame = phase.frames[frame_index]
    if frame.pose is None:
        raise ValueError(
            "The NumPy engine requires known poses. Install Open3D to estimate poses from real captures."
        )
    camera = phase.camera
    if depth_path is None:
        depth_values = load_depth(frame, camera)
    else:
        depth_values = np.fromfile(depth_path, dtype="<u2")
        expected = camera.width * camera.height
        if depth_values.size != expected:
            raise ValueError(f"Depth override {depth_path} has an unexpected size")
        depth_values = depth_values.reshape(camera.height, camera.width)
    depth = depth_values[::pixel_stride, ::pixel_stride].astype(np.float64)
    color = load_color(frame, camera)[::pixel_stride, ::pixel_stride]
    y_pixels, x_pixels = np.mgrid[0 : camera.height : pixel_stride, 0 : camera.width : pixel_stride]
    z = depth / camera.depth_scale
    valid = (z > 0.25) & (z <= camera.max_depth_m)
    z = z[valid]
    x = (x_pixels[valid] - camera.cx) * z / camera.fx
    y = -(y_pixels[valid] - camera.cy) * z / camera.fy
    camera_points = np.column_stack((x, y, z, np.ones_like(z)))
    world_points = (frame.pose @ camera_points.T).T[:, :3]
    return world_points.astype(np.float32), color[valid].astype(np.uint8)


def voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return points.reshape(0, 3), colors.reshape(0, 3)
    if voxel_size_m <= 0:
        raise ValueError("Voxel size must be positive")

    voxel_keys = np.floor(points / voxel_size_m).astype(np.int64)
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(map(tuple, voxel_keys)):
        groups[key].append(index)

    output_points = np.empty((len(groups), 3), dtype=np.float32)
    output_colors = np.empty((len(groups), 3), dtype=np.uint8)
    for output_index, indices in enumerate(groups.values()):
        output_points[output_index] = points[indices].mean(axis=0)
        output_colors[output_index] = np.rint(colors[indices].mean(axis=0)).astype(np.uint8)
    return output_points, output_colors


def reconstruct_known_poses(
    phases: list[PhaseData],
    voxel_size_m: float,
    progress: Callable[..., None] | None = None,
    depth_overrides: dict[tuple[str, int], Any] | None = None,
    accepted_frame_keys: set[tuple[str, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not phases:
        raise ValueError("At least one capture phase is required")
    point_batches: list[np.ndarray] = []
    color_batches: list[np.ndarray] = []
    total_frames = sum(
        1
        for phase in phases
        for frame_index in range(len(phase.frames))
        if accepted_frame_keys is None
        or (str(phase.root), frame_index) in accepted_frame_keys
    )
    if total_frames == 0:
        raise RuntimeError("Camera validation left no frames for NumPy fusion")
    completed_frames = 0
    for phase in phases:
        for frame_index in range(len(phase.frames)):
            if (
                accepted_frame_keys is not None
                and (str(phase.root), frame_index) not in accepted_frame_keys
            ):
                continue
            override = (depth_overrides or {}).get((str(phase.root), frame_index))
            depth_paths = (
                [None]
                if override is None
                else [override.measured_depth_path]
                + ([override.refined_depth_path] if override.generated_pixels > 0 else [])
            )
            for depth_path in depth_paths:
                points, colors = depth_to_world_points(
                    phase, frame_index, depth_path=depth_path
                )
                point_batches.append(points)
                color_batches.append(colors)
            completed_frames += 1
            if progress:
                progress(
                    "Placing frames",
                    f"Placed validated frame {completed_frames} of {total_frames}",
                    1,
                    sum(batch.shape[0] for batch in point_batches),
                    completed_frames / total_frames,
                )
    if progress:
        progress("Fusing points", "Voxel-downsampling placed frames", 4, None, 1.0)
    return voxel_downsample(
        np.concatenate(point_batches, axis=0),
        np.concatenate(color_batches, axis=0),
        voxel_size_m,
    )
