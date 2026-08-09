from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .live_contract import CoverageSummary
from .stream import RgbdFrame


VOXEL_BLOCK_RESOLUTION = 16
VOXEL_BYTES = 4 + 2 + 3 * 2  # Float32 TSDF, UInt16 weight, UInt16 RGB.
VOXEL_BLOCK_BYTES = VOXEL_BLOCK_RESOLUTION**3 * VOXEL_BYTES


@dataclass(frozen=True)
class SubmapLimits:
    gpu_budget_bytes: int
    block_capacity: int
    rollover_block_count: int
    maximum_distance_m: float = 2.5
    maximum_rotation_degrees: float = 100.0
    maximum_integrated_frames: int = 450
    maximum_submaps: int = 64
    maximum_host_points: int = 750_000
    maximum_host_triangles: int = 600_000

    @classmethod
    def from_mebibytes(cls, budget_mib: int) -> "SubmapLimits":
        if not 256 <= budget_mib <= 4096:
            raise ValueError("Live-map memory budget must be between 256 and 4096 MiB")
        budget_bytes = budget_mib * 1024 * 1024
        # Keep 12.5% for hash keys and allocator overhead. The active grid is
        # replaced before the reserved value pool becomes saturated.
        capacity = max(2_048, int(budget_bytes * 0.875) // VOXEL_BLOCK_BYTES)
        return cls(
            gpu_budget_bytes=budget_bytes,
            block_capacity=capacity,
            rollover_block_count=max(1, int(capacity * 0.82)),
        )


class AdaptiveBudgetController:
    """Hysteretic degradation controller for lower-priority live work."""

    def __init__(self) -> None:
        self.level = 0
        self._healthy_samples = 0

    def observe(
        self,
        *,
        map_latency_ms: float,
        mapping_queue_ratio: float,
        memory_ratio: float,
    ) -> int:
        pressure = 0
        if map_latency_ms > 90.0 or mapping_queue_ratio >= 0.50:
            pressure = 1
        if map_latency_ms > 180.0 or mapping_queue_ratio >= 0.75:
            pressure = 3
        if map_latency_ms > 400.0 or mapping_queue_ratio >= 0.90:
            pressure = 5
        if memory_ratio >= 0.80:
            pressure = max(pressure, 4)
        if pressure:
            self.level = min(6, max(self.level + 1, pressure))
            self._healthy_samples = 0
        else:
            self._healthy_samples += 1
            if self._healthy_samples >= 20 and self.level:
                self.level -= 1
                self._healthy_samples = 0
        return self.level

    @property
    def point_interval_seconds(self) -> float:
        return (0.10, 0.13, 0.18, 0.25, 0.35, 0.50, 0.75)[self.level]

    @property
    def coverage_interval_seconds(self) -> float:
        return (0.25, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0)[self.level]

    @property
    def mesh_enabled(self) -> bool:
        return self.level == 0

    @property
    def integration_stride(self) -> int:
        return 2 if self.level >= 6 else 1


@dataclass
class _CoverageCell:
    observations: int
    best_pixel_density: float
    pose_confidence: float
    last_sequence: int


def frame_world_samples(
    frame: RgbdFrame,
    world_to_camera: np.ndarray,
    *,
    stride: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(frame.depth[::stride, ::stride], dtype=np.float32)
    rows, columns = np.mgrid[0 : frame.camera.height : stride, 0 : frame.camera.width : stride]
    z = depth / frame.camera.depth_scale
    valid = (
        np.isfinite(z)
        & (z >= frame.camera.min_depth_m)
        & (z <= frame.camera.max_depth_m)
    )
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.float32)
    z_valid = z[valid]
    camera_points = np.column_stack(
        (
            (columns[valid] - frame.camera.cx) * z_valid / frame.camera.fx,
            (rows[valid] - frame.camera.cy) * z_valid / frame.camera.fy,
            z_valid,
            np.ones_like(z_valid),
        )
    )
    camera_to_world = np.linalg.inv(world_to_camera)
    world = (camera_to_world @ camera_points.T).T[:, :3]
    pixel_density = frame.camera.fx / np.maximum(z_valid, 1e-3)
    return world.astype(np.float32), pixel_density.astype(np.float32)


class CoverageField:
    def __init__(self, voxel_size_m: float, maximum_cells: int = 200_000) -> None:
        self.voxel_size_m = max(0.04, voxel_size_m)
        self.maximum_cells = maximum_cells
        self.cells: dict[tuple[int, int, int], _CoverageCell] = {}
        self.dropped_cells = 0

    def observe(
        self,
        frame: RgbdFrame,
        world_to_camera: np.ndarray,
        pose_confidence: float,
    ) -> None:
        points, pixel_density = frame_world_samples(frame, world_to_camera)
        if not len(points):
            return
        keys = np.floor(points / self.voxel_size_m).astype(np.int32)
        unique_keys, indices = np.unique(keys, axis=0, return_index=True)
        for key_values, index in zip(unique_keys, indices, strict=True):
            key = tuple(int(value) for value in key_values)
            cell = self.cells.get(key)
            if cell is None:
                if len(self.cells) >= self.maximum_cells:
                    self.dropped_cells += 1
                    continue
                self.cells[key] = _CoverageCell(
                    observations=1,
                    best_pixel_density=float(pixel_density[index]),
                    pose_confidence=pose_confidence,
                    last_sequence=frame.sequence,
                )
            elif cell.last_sequence != frame.sequence:
                cell.observations = min(65_535, cell.observations + 1)
                cell.best_pixel_density = max(
                    cell.best_pixel_density, float(pixel_density[index])
                )
                cell.pose_confidence = max(cell.pose_confidence, pose_confidence)
                cell.last_sequence = frame.sequence

    def summary(self, tracking_confidence: float) -> CoverageSummary:
        total = len(self.cells)
        if total == 0:
            return CoverageSummary(guidance=("Aim at a surface with valid depth",))
        counts = np.fromiter(
            (cell.observations for cell in self.cells.values()),
            dtype=np.int32,
            count=total,
        )
        observed = float(np.count_nonzero(counts >= 3) / total)
        weak = float(np.count_nonzero(counts == 2) / total)
        single = float(np.count_nonzero(counts == 1) / total)
        guidance: list[str] = []
        if tracking_confidence < 0.35:
            guidance.append("Return to the last trusted region")
        if single > 0.35:
            guidance.append("Increase parallax and revisit orange surfaces")
        if observed < 0.45:
            guidance.append("Revisit weakly observed surfaces")
        if not guidance:
            guidance.append("Coverage is stable; close the loop before stopping")
        return CoverageSummary(
            observed_ratio=observed,
            weak_ratio=weak,
            single_view_ratio=single,
            hole_boundary_ratio=0.0,
            guidance=tuple(guidance),
        )

    def colors(self, world_points: np.ndarray) -> np.ndarray:
        colors = np.full((len(world_points), 3), [86, 68, 104], dtype=np.uint8)
        keys = np.floor(world_points / self.voxel_size_m).astype(np.int32)
        for index, key_values in enumerate(keys):
            cell = self.cells.get(tuple(int(value) for value in key_values))
            if cell is None:
                continue
            if cell.observations >= 3:
                colors[index] = [50, 214, 145]
            elif cell.observations == 2:
                colors[index] = [242, 196, 74]
            else:
                colors[index] = [242, 124, 58]
        return colors


def tracking_colors(point_count: int, state: str, confidence: float) -> np.ndarray:
    if state in {"searching", "failed"}:
        color = [230, 68, 84]
    elif state == "relocalized":
        color = [86, 180, 255]
    elif state == "frozen" or confidence < 0.45:
        color = [244, 180, 58]
    else:
        green = int(round(130 + 100 * min(1.0, confidence)))
        color = [48, green, 142]
    return np.full((point_count, 3), color, dtype=np.uint8)


def rotation_degrees(matrix: np.ndarray) -> float:
    trace = min(3.0, max(-1.0, float(np.trace(matrix[:3, :3]))))
    return math.degrees(math.acos(min(1.0, max(-1.0, (trace - 1.0) * 0.5))))

