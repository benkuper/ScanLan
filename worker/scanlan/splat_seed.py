from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .calibration import robust_depth_mask


SEED_VERSION = "rgbd-quadtree-discs-v4-confidence-500k"
DEFAULT_SEED_VOXEL_M = 0.01
DEFAULT_CONTRAST_THRESHOLD = 2.0e-4
DEFAULT_MAX_CELL_SIZE = 24
DEFAULT_MIN_CELL_SIZE = 2
MAX_INITIAL_GAUSSIANS = 500_000


@dataclass(frozen=True)
class GaussianSeeds:
    points: np.ndarray
    colors: np.ndarray
    scales: np.ndarray
    quaternions: np.ndarray
    confidence: np.ndarray | None = None
    # 0 = calibrated sensor measurement, 1 = validated generated depth.
    # Learned RGB-only geometry uses code 2 in the shared dense-fusion
    # contract written by the media worker.
    provenance: np.ndarray | None = None
    source_frame_indices: np.ndarray | None = None


def _integral_image(values: np.ndarray) -> np.ndarray:
    return np.pad(values.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))


def _region_sum(integral: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> Any:
    return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]


def adaptive_quadtree_cells(
    image: np.ndarray,
    valid: np.ndarray,
    contrast_threshold: float = DEFAULT_CONTRAST_THRESHOLD,
    max_cell_size: int = DEFAULT_MAX_CELL_SIZE,
    min_cell_size: int = DEFAULT_MIN_CELL_SIZE,
) -> list[tuple[int, int, int, int]]:
    """Divide an RGB-D view into compact, contrast-aware seed regions.

    Color variance preserves thin visual detail. A maximum leaf size also
    preserves geometry on low-texture walls, while mixed valid/invalid cells
    are subdivided so depth boundaries do not receive oversized splats.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3")
    if valid.shape != image.shape[:2]:
        raise ValueError("valid mask must match image dimensions")
    height, width = valid.shape
    normalized = np.asarray(image, dtype=np.float64) / 255.0
    valid_float = valid.astype(np.float64)
    count_integral = _integral_image(valid_float)
    sums = [_integral_image(normalized[..., channel] * valid_float) for channel in range(3)]
    square_sums = [
        _integral_image(np.square(normalized[..., channel]) * valid_float)
        for channel in range(3)
    ]

    leaves: list[tuple[int, int, int, int]] = []
    pending = [(0, 0, width, height)]
    while pending:
        x0, y0, x1, y1 = pending.pop()
        cell_width = x1 - x0
        cell_height = y1 - y0
        area = cell_width * cell_height
        count = float(_region_sum(count_integral, x0, y0, x1, y1))
        if count <= 0:
            continue

        variance = 0.0
        for channel_sum, channel_square_sum in zip(sums, square_sums, strict=True):
            mean = float(_region_sum(channel_sum, x0, y0, x1, y1)) / count
            second_moment = float(
                _region_sum(channel_square_sum, x0, y0, x1, y1)
            ) / count
            variance += max(0.0, second_moment - mean * mean)
        variance /= 3.0
        coverage = count / max(area, 1)
        can_split = cell_width > min_cell_size * 2 and cell_height > min_cell_size * 2
        should_split = can_split and (
            cell_width > max_cell_size
            or cell_height > max_cell_size
            or variance > contrast_threshold
            or (0.05 < coverage < 0.95)
        )
        if not should_split:
            leaves.append((x0, y0, x1, y1))
            continue

        xm = x0 + cell_width // 2
        ym = y0 + cell_height // 2
        if xm in (x0, x1) or ym in (y0, y1):
            leaves.append((x0, y0, x1, y1))
            continue
        pending.extend(
            [
                (x0, y0, xm, ym),
                (xm, y0, x1, ym),
                (x0, ym, xm, y1),
                (xm, ym, x1, y1),
            ]
        )
    return leaves


def _cell_centers(
    cells: list[tuple[int, int, int, int]],
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centers: list[tuple[int, int]] = []
    sizes: list[tuple[int, int]] = []
    for x0, y0, x1, y1 in cells:
        local = np.argwhere(valid[y0:y1, x0:x1])
        if not len(local):
            continue
        target_x = (x0 + x1 - 1) * 0.5
        target_y = (y0 + y1 - 1) * 0.5
        distances = (local[:, 1] + x0 - target_x) ** 2 + (local[:, 0] + y0 - target_y) ** 2
        row, column = local[int(np.argmin(distances))]
        centers.append((int(column + x0), int(row + y0)))
        sizes.append((x1 - x0, y1 - y0))
    return np.asarray(centers, dtype=np.int64), np.asarray(sizes, dtype=np.float32)


def _rotation_quaternions(rotation: np.ndarray) -> np.ndarray:
    """Convert right-handed rotation matrices to normalized wxyz quaternions."""
    m = np.asarray(rotation, dtype=np.float64)
    w = np.sqrt(np.maximum(0.0, 1.0 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2])) * 0.5
    x = np.copysign(
        np.sqrt(np.maximum(0.0, 1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2])) * 0.5,
        m[:, 2, 1] - m[:, 1, 2],
    )
    y = np.copysign(
        np.sqrt(np.maximum(0.0, 1.0 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2])) * 0.5,
        m[:, 0, 2] - m[:, 2, 0],
    )
    z = np.copysign(
        np.sqrt(np.maximum(0.0, 1.0 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2])) * 0.5,
        m[:, 1, 0] - m[:, 0, 1],
    )
    quaternions = np.column_stack((w, x, y, z))
    quaternions /= np.maximum(np.linalg.norm(quaternions, axis=1, keepdims=True), 1e-8)
    return quaternions.astype(np.float32)


def _surface_frames(
    depth_m: np.ndarray,
    valid: np.ndarray,
    centers: np.ndarray,
    camera: Any,
    world_from_depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth_m.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    points = np.stack(
        (
            (xx - float(camera.cx)) * depth_m / float(camera.fx),
            (yy - float(camera.cy)) * depth_m / float(camera.fy),
            depth_m,
        ),
        axis=-1,
    )
    tangent_x = np.zeros_like(points)
    tangent_y = np.zeros_like(points)
    tangent_x[:, 1:-1] = points[:, 2:] - points[:, :-2]
    tangent_y[1:-1, :] = points[2:, :] - points[:-2, :]
    neighbor_valid = np.zeros_like(valid)
    neighbor_valid[1:-1, 1:-1] = (
        valid[1:-1, :-2]
        & valid[1:-1, 2:]
        & valid[:-2, 1:-1]
        & valid[2:, 1:-1]
    )

    u = centers[:, 0]
    v = centers[:, 1]
    camera_points = points[v, u].astype(np.float64)
    tx = tangent_x[v, u].astype(np.float64)
    ty = tangent_y[v, u].astype(np.float64)
    fallback = ~neighbor_valid[v, u]
    tx[fallback] = np.column_stack(
        (
            depth_m[v[fallback], u[fallback]] / float(camera.fx),
            np.zeros(int(fallback.sum())),
            np.zeros(int(fallback.sum())),
        )
    )
    ty[fallback] = np.column_stack(
        (
            np.zeros(int(fallback.sum())),
            depth_m[v[fallback], u[fallback]] / float(camera.fy),
            np.zeros(int(fallback.sum())),
        )
    )

    normal = np.cross(tx, ty)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
    away_from_camera = np.sum(normal * camera_points, axis=1) > 0.0
    normal[away_from_camera] *= -1.0
    tx -= np.sum(tx * normal, axis=1, keepdims=True) * normal
    tx /= np.maximum(np.linalg.norm(tx, axis=1, keepdims=True), 1e-8)
    ty = np.cross(normal, tx)
    ty /= np.maximum(np.linalg.norm(ty, axis=1, keepdims=True), 1e-8)

    linear = np.asarray(world_from_depth, dtype=np.float64)[:3, :3]
    world_tx = (linear @ tx.T).T
    world_normal = (linear @ normal.T).T
    world_normal /= np.maximum(np.linalg.norm(world_normal, axis=1, keepdims=True), 1e-8)
    world_tx -= np.sum(world_tx * world_normal, axis=1, keepdims=True) * world_normal
    world_tx /= np.maximum(np.linalg.norm(world_tx, axis=1, keepdims=True), 1e-8)
    world_ty = np.cross(world_normal, world_tx)
    world_ty /= np.maximum(np.linalg.norm(world_ty, axis=1, keepdims=True), 1e-8)
    rotations = np.stack((world_tx, world_ty, world_normal), axis=2)

    homogeneous = np.column_stack((camera_points, np.ones(len(camera_points))))
    world_points = (np.asarray(world_from_depth, dtype=np.float64) @ homogeneous.T).T[:, :3]
    return world_points.astype(np.float32), _rotation_quaternions(rotations)


def seed_rgbd_gaussians(
    depth: np.ndarray,
    image: np.ndarray,
    uv_map: np.ndarray,
    visibility: np.ndarray,
    camera: Any,
    world_from_depth: np.ndarray,
    confidence: np.ndarray | None = None,
) -> GaussianSeeds:
    """Create surface-aligned 2D Gaussian seeds from one posed RGB-D frame."""
    depth_m = np.asarray(depth, dtype=np.float32) / float(camera.depth_scale)
    valid = visibility & robust_depth_mask(depth_m)
    projected = valid & np.isfinite(uv_map[..., 0]) & np.isfinite(uv_map[..., 1])
    aligned = np.zeros((*depth.shape, 3), dtype=np.uint8)
    flat = np.flatnonzero(projected)
    if len(flat):
        uv = uv_map.reshape(-1, 2)[flat]
        u = np.rint(uv[:, 0]).astype(np.int64).clip(0, image.shape[1] - 1)
        v = np.rint(uv[:, 1]).astype(np.int64).clip(0, image.shape[0] - 1)
        aligned.reshape(-1, 3)[flat] = image[v, u]
    cells = adaptive_quadtree_cells(aligned, projected)
    centers, sizes = _cell_centers(cells, projected)
    if not len(centers):
        empty3 = np.empty((0, 3), dtype=np.float32)
        return GaussianSeeds(
            empty3,
            np.empty((0, 3), dtype=np.uint8),
            empty3,
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    points, quaternions = _surface_frames(
        depth_m,
        projected,
        centers,
        camera,
        world_from_depth,
    )
    u = centers[:, 0]
    v = centers[:, 1]
    z = depth_m[v, u]
    scale_x = np.maximum(z * sizes[:, 0] * 0.5 / float(camera.fx), DEFAULT_SEED_VOXEL_M * 0.35)
    scale_y = np.maximum(z * sizes[:, 1] * 0.5 / float(camera.fy), DEFAULT_SEED_VOXEL_M * 0.35)
    scale_z = np.maximum(np.minimum(scale_x, scale_y) * 0.08, 5.0e-4)
    scales = np.column_stack((scale_x, scale_y, scale_z)).astype(np.float32)
    if confidence is None:
        seed_confidence = np.ones(len(points), dtype=np.float32)
    else:
        confidence_array = np.asarray(confidence)
        if confidence_array.shape != depth.shape:
            raise ValueError("confidence must match the depth raster")
        seed_confidence = np.clip(
            confidence_array[v, u].astype(np.float32) / 255.0,
            0.0,
            1.0,
        )
    return GaussianSeeds(points, aligned[v, u], scales, quaternions, seed_confidence)


def compact_seed_batches(
    batches: list[GaussianSeeds],
    voxel_size_m: float = DEFAULT_SEED_VOXEL_M,
    limit: int = MAX_INITIAL_GAUSSIANS,
) -> GaussianSeeds:
    if not batches:
        empty3 = np.empty((0, 3), dtype=np.float32)
        return GaussianSeeds(
            empty3,
            np.empty((0, 3), dtype=np.uint8),
            empty3,
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.uint8),
            np.empty((0,), dtype=np.int32),
        )
    points = np.concatenate([batch.points for batch in batches])
    colors = np.concatenate([batch.colors for batch in batches])
    scales = np.concatenate([batch.scales for batch in batches])
    quaternions = np.concatenate([batch.quaternions for batch in batches])
    confidence = np.concatenate(
        [
            np.ones(len(batch.points), dtype=np.float32)
            if batch.confidence is None
            else np.asarray(batch.confidence, dtype=np.float32)
            for batch in batches
        ]
    )
    provenance = np.concatenate(
        [
            np.zeros(len(batch.points), dtype=np.uint8)
            if batch.provenance is None
            else np.asarray(batch.provenance, dtype=np.uint8)
            for batch in batches
        ]
    )
    source_frame_indices = np.concatenate(
        [
            np.full(len(batch.points), -1, dtype=np.int32)
            if batch.source_frame_indices is None
            else np.asarray(batch.source_frame_indices, dtype=np.int32)
            for batch in batches
        ]
    )
    voxel_keys = np.floor(points / voxel_size_m).astype(np.int64)
    _, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)
    quality = np.maximum(scales[:, 0], scales[:, 1])
    # Prefer measured/high-confidence geometry within a voxel; footprint is the
    # deterministic tiebreaker when provenance confidence matches.
    order = np.lexsort((quality, -confidence, inverse))
    first = np.r_[True, inverse[order][1:] != inverse[order][:-1]]
    selected = order[first]
    # The voxel sort makes deterministic uniform subsampling spatially balanced.
    selected = selected[np.lexsort((voxel_keys[selected, 2], voxel_keys[selected, 1], voxel_keys[selected, 0]))]
    if len(selected) > limit:
        selected = selected[np.linspace(0, len(selected) - 1, limit, dtype=np.int64)]
    return GaussianSeeds(
        points[selected],
        colors[selected],
        scales[selected],
        quaternions[selected],
        confidence[selected],
        provenance[selected],
        source_frame_indices[selected],
    )
