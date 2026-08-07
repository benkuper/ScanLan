from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from .mesh_observations import (
    FREE_SPACE_VIOLATION,
    MISSING_DEPTH,
    OCCLUDED,
    OUTSIDE_VIEW,
    SUPPORTED,
    classify_world_points,
)


MESH_REPAIR_REPORT_SCHEMA_VERSION = 1
MESH_REPAIR_ALGORITHM_VERSION = "1.0.0"
MeshRepairProfile = Literal["faithful", "architectural", "natural", "watertight"]


@dataclass(frozen=True)
class MeshRepairSettings:
    enabled: bool = True
    profile: MeshRepairProfile = "faithful"
    max_hole_diameter_m: float | None = None
    min_support_ratio: float = 0.60
    max_free_space_ratio: float = 0.01
    min_supporting_views: int = 2
    fill_inferred_holes: bool = False
    repair_non_manifold: bool = True
    repair_self_intersections: bool = False
    produce_watertight_copy: bool = False
    allow_unrepaired_fallback: bool = True

    def resolved_max_hole_diameter_m(self, mesh_voxel_size_m: float) -> float:
        if self.max_hole_diameter_m is not None:
            return max(0.0, float(self.max_hole_diameter_m))
        return float(np.clip(12.0 * mesh_voxel_size_m, 0.04, 0.15))

    def validate(self) -> None:
        if self.profile not in {"faithful", "architectural", "natural", "watertight"}:
            raise ValueError(f"Unknown mesh repair profile: {self.profile}")
        if not 0.0 <= self.min_support_ratio <= 1.0:
            raise ValueError("min_support_ratio must be between 0 and 1")
        if not 0.0 <= self.max_free_space_ratio <= 1.0:
            raise ValueError("max_free_space_ratio must be between 0 and 1")
        if self.min_supporting_views < 1:
            raise ValueError("min_supporting_views must be at least 1")

    def report_payload(self, mesh_voxel_size_m: float) -> dict[str, Any]:
        payload = asdict(self)
        payload["max_hole_diameter_m"] = self.resolved_max_hole_diameter_m(
            mesh_voxel_size_m
        )
        return payload


def _plane_axes(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(normal, dtype=np.float64)
    length = float(np.linalg.norm(normal))
    if length <= 1e-12:
        normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        normal = normal / length
    reference = np.zeros(3, dtype=np.float64)
    reference[int(np.argmin(np.abs(normal)))] = 1.0
    axis_u = np.cross(normal, reference)
    axis_u /= max(float(np.linalg.norm(axis_u)), 1e-12)
    axis_v = np.cross(normal, axis_u)
    return axis_u, axis_v


def _inside_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        if (y1 > y) != (y2 > y):
            crossing = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing:
                inside = not inside
        previous = current
    return inside


def _boundary_distance(point: np.ndarray, polygon: np.ndarray) -> float:
    minimum = math.inf
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        edge = end - start
        length_squared = float(np.dot(edge, edge))
        if length_squared <= 1e-20:
            distance = float(np.linalg.norm(point - start))
        else:
            position = float(np.clip(np.dot(point - start, edge) / length_squared, 0.0, 1.0))
            distance = float(np.linalg.norm(point - (start + position * edge)))
        minimum = min(minimum, distance)
    return minimum


def _halton(index: int, base: int) -> float:
    result = 0.0
    fraction = 1.0 / base
    while index:
        result += fraction * (index % base)
        index //= base
        fraction /= base
    return result


def sample_boundary_loop(
    loop: dict[str, Any], mesh_voxel_size_m: float
) -> np.ndarray:
    """Generate deterministic interior samples, buffered from depth discontinuities."""

    positions = np.asarray(loop["orderedBoundaryPositions"], dtype=np.float64)
    plane = loop["bestFitPlane"]
    origin = np.asarray(plane["origin"], dtype=np.float64)
    normal = np.asarray(plane["normal"], dtype=np.float64)
    axis_u, axis_v = _plane_axes(normal)
    relative = positions - origin
    polygon = np.column_stack((relative @ axis_u, relative @ axis_v))
    area = max(float(loop.get("approximateEnclosedAreaM2", 0.0)), 1e-8)
    target = int(np.clip(round(area / max(mesh_voxel_size_m**2, 1e-8) * 2.0), 32, 256))
    minimum = polygon.min(axis=0)
    maximum = polygon.max(axis=0)
    diameter = max(float(loop.get("diameterM", 0.0)), mesh_voxel_size_m)
    margin = min(mesh_voxel_size_m * 0.75, diameter * 0.08)

    selected: list[np.ndarray] = []
    for pass_index in range(2):
        selected.clear()
        pass_margin = margin if pass_index == 0 else margin * 0.35
        for sequence in range(1, 20_001):
            point = minimum + np.asarray(
                [_halton(sequence, 2), _halton(sequence, 3)], dtype=np.float64
            ) * (maximum - minimum)
            if _inside_polygon(point, polygon) and _boundary_distance(point, polygon) >= pass_margin:
                selected.append(point)
                if len(selected) >= target:
                    break
        if len(selected) >= min(32, target):
            break
    if not selected:
        selected.append(np.mean(polygon, axis=0))
    samples_2d = np.asarray(selected, dtype=np.float64)
    return (
        origin
        + samples_2d[:, :1] * axis_u
        + samples_2d[:, 1:] * axis_v
    ).astype(np.float64)


def _geometric_classification(loop: dict[str, Any], mesh_voxel_size_m: float) -> str:
    diameter = max(float(loop.get("diameterM", 0.0)), mesh_voxel_size_m)
    residual = float(loop.get("planeRmsResidualM", math.inf))
    coherence = float(loop.get("boundaryNormalCoherence", 0.0))
    planar = residual <= max(mesh_voxel_size_m, diameter * 0.03) and coherence >= 0.70
    return "planar" if planar else "freeform"


def classify_boundary_loop(
    loop: dict[str, Any],
    frames: list[Any],
    mesh_voxel_size_m: float,
    settings: MeshRepairSettings,
) -> dict[str, Any]:
    settings.validate()
    samples = sample_boundary_loop(loop, mesh_voxel_size_m)
    counts = {
        OUTSIDE_VIEW: 0,
        MISSING_DEPTH: 0,
        SUPPORTED: 0,
        FREE_SPACE_VIOLATION: 0,
        OCCLUDED: 0,
    }
    supporting_views = 0
    useful_view_count = 0
    for frame in frames:
        if frame.depthless:
            continue
        evidence = classify_world_points(samples, frame, mesh_voxel_size_m)
        frame_counts = {name: int(np.count_nonzero(evidence == name)) for name in counts}
        for name, count in frame_counts.items():
            counts[name] += count
        metric_count = frame_counts[SUPPORTED] + frame_counts[FREE_SPACE_VIOLATION] + frame_counts[OCCLUDED]
        if metric_count:
            useful_view_count += 1
        if frame_counts[SUPPORTED] >= max(3, math.ceil(len(samples) * 0.15)):
            supporting_views += 1

    metric_evidence = counts[SUPPORTED] + counts[FREE_SPACE_VIOLATION] + counts[OCCLUDED]
    denominator = max(metric_evidence, 1)
    support_ratio = counts[SUPPORTED] / denominator
    free_space_ratio = counts[FREE_SPACE_VIOLATION] / denominator
    occluded_ratio = counts[OCCLUDED] / denominator
    max_diameter = settings.resolved_max_hole_diameter_m(mesh_voxel_size_m)
    diameter = float(loop.get("diameterM", math.inf))
    coherence = float(loop.get("boundaryNormalCoherence", 0.0))
    residual = float(loop.get("planeRmsResidualM", math.inf))
    geometric = _geometric_classification(loop, mesh_voxel_size_m)

    if diameter > max_diameter:
        classification = "preserve_too_large"
    elif free_space_ratio > settings.max_free_space_ratio:
        classification = "preserve_opening"
    elif (
        support_ratio >= settings.min_support_ratio
        and supporting_views >= settings.min_supporting_views
    ):
        classification = "fill_measured"
    elif occluded_ratio >= 0.25 and counts[SUPPORTED] == 0:
        classification = "preserve_occluded"
    elif (
        settings.fill_inferred_holes
        and coherence >= 0.85
        and residual <= max(2.0 * mesh_voxel_size_m, diameter * 0.04)
        and counts[FREE_SPACE_VIOLATION] == 0
    ):
        classification = "fill_inferred"
    else:
        classification = "preserve_unknown"

    return {
        "loopId": str(loop["loopId"]),
        "classification": classification,
        "geometricClassification": geometric,
        "sampleCount": len(samples),
        "supportRatio": round(support_ratio, 6),
        "freeSpaceViolationRatio": round(free_space_ratio, 6),
        "occludedRatio": round(occluded_ratio, 6),
        "supportingViewCount": supporting_views,
        "usefulMetricViewCount": useful_view_count,
        "evidenceCounts": counts,
        "diameterM": diameter,
        "areaM2": float(loop.get("approximateEnclosedAreaM2", 0.0)),
        "bestFitPlane": loop["bestFitPlane"],
        "planeRmsResidualM": residual,
        "boundaryNormalCoherence": coherence,
    }


def classify_topology_report(
    topology_report: dict[str, Any],
    frames: list[Any],
    mesh_voxel_size_m: float,
    settings: MeshRepairSettings,
) -> dict[str, Any]:
    decisions = [
        classify_boundary_loop(loop, frames, mesh_voxel_size_m, settings)
        for loop in topology_report.get("boundaryLoops", [])
    ]
    selected = [
        decision["loopId"]
        for decision in decisions
        if decision["classification"] in {"fill_measured", "fill_inferred"}
    ]
    summary: dict[str, int] = {}
    for decision in decisions:
        classification = str(decision["classification"])
        summary[classification] = summary.get(classification, 0) + 1
    return {
        "schemaVersion": MESH_REPAIR_REPORT_SCHEMA_VERSION,
        "algorithmVersion": MESH_REPAIR_ALGORITHM_VERSION,
        "status": "classified",
        "settings": settings.report_payload(mesh_voxel_size_m),
        "topology": topology_report.get("topology", {}),
        "inputMeshFingerprint": topology_report.get("inputMeshFingerprint"),
        "holes": decisions,
        "selectedLoopIds": selected,
        "summary": summary,
    }
