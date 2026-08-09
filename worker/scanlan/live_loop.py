from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np


AnchorT = TypeVar("AnchorT")


def rotation_degrees(transform: np.ndarray) -> float:
    trace = float(np.trace(np.asarray(transform, dtype=np.float64)[:3, :3]))
    cosine = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return math.degrees(math.acos(cosine))


def transform_delta(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
    delta = np.linalg.inv(np.asarray(before, dtype=np.float64)) @ np.asarray(
        after, dtype=np.float64
    )
    return float(np.linalg.norm(delta[:3, 3])), rotation_degrees(delta)


def _axis_angle(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    angle = math.radians(rotation_degrees(rotation))
    if angle < 1e-9:
        return np.asarray((1.0, 0.0, 0.0), dtype=np.float64), 0.0
    sine = math.sin(angle)
    if abs(sine) < 1e-6:
        values, vectors = np.linalg.eigh(np.asarray(rotation, dtype=np.float64))
        axis = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
    else:
        axis = np.asarray(
            (
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ),
            dtype=np.float64,
        ) / (2.0 * sine)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    return axis, angle


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = np.asarray(axis, dtype=np.float64)
    cross = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def interpolate_transform(before: np.ndarray, after: np.ndarray, fraction: float) -> np.ndarray:
    fraction = max(0.0, min(1.0, float(fraction)))
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    if fraction <= 0.0:
        return before.copy()
    if fraction >= 1.0:
        return after.copy()
    relative = np.linalg.inv(before) @ after
    axis, angle = _axis_angle(relative[:3, :3])
    step = np.eye(4, dtype=np.float64)
    step[:3, :3] = _rodrigues(axis, angle * fraction)
    step[:3, 3] = relative[:3, 3] * fraction
    return before @ step


class LocalAnchorDatabase(Generic[AnchorT]):
    """Bounded, capture-wide anchor bank with deterministic rotating queries."""

    def __init__(self, maximum_entries: int = 48, recent_entries: int = 8) -> None:
        if maximum_entries < 2 or not 0 < recent_entries < maximum_entries:
            raise ValueError("Anchor database limits are invalid")
        self.maximum_entries = maximum_entries
        self.recent_entries = recent_entries
        self.entries: list[AnchorT] = []
        self.features: dict[int, tuple[Any, Any]] = {}
        self.cursor = 0

    @staticmethod
    def _sequence(anchor: AnchorT) -> int:
        return int(getattr(getattr(anchor, "frame"), "sequence"))

    def add(self, anchor: AnchorT) -> None:
        self.entries.append(anchor)
        if len(self.entries) <= self.maximum_entries:
            return
        history_end = max(1, len(self.entries) - self.recent_entries)
        self.entries = self.entries[:history_end:2] + self.entries[history_end:]
        retained = {self._sequence(entry) for entry in self.entries}
        self.features = {
            sequence: value
            for sequence, value in self.features.items()
            if sequence in retained
        }
        self.cursor %= max(len(self.entries), 1)

    def candidates(
        self,
        *,
        previous_sequence: int | None,
        pending_sequence: int | None,
        limit: int,
    ) -> list[AnchorT]:
        available = [
            entry
            for entry in self.entries
            if self._sequence(entry) != previous_sequence
        ]
        if not available or limit <= 0:
            return []
        selected: list[AnchorT] = []
        if pending_sequence is not None:
            pending = next(
                (
                    entry
                    for entry in available
                    if self._sequence(entry) == pending_sequence
                ),
                None,
            )
            if pending is not None:
                selected.append(pending)
        if len(selected) < limit and all(
            self._sequence(entry) != self._sequence(available[0])
            for entry in selected
        ):
            selected.append(available[0])
        selected_sequences = {self._sequence(entry) for entry in selected}
        rotating = [
            entry
            for entry in available[1:]
            if self._sequence(entry) not in selected_sequences
        ]
        budget = limit - len(selected)
        if rotating and budget > 0:
            start = self.cursor % len(rotating)
            count = min(budget, len(rotating))
            selected.extend(rotating[(start + offset) % len(rotating)] for offset in range(count))
            self.cursor = (start + count) % len(rotating)
        return selected


@dataclass(frozen=True)
class LoopVerification:
    accepted: bool
    target_from_source: np.ndarray
    information: np.ndarray
    fitness: float
    rmse_m: float
    correspondence_count: int
    reason: str


@dataclass(frozen=True)
class PoseGraphLoop:
    source_id: str
    target_id: str
    target_from_source: np.ndarray
    information: np.ndarray
    fitness: float
    rmse_m: float
    sequence: int


@dataclass(frozen=True)
class PoseGraphSolution:
    accepted: bool
    transforms: dict[str, np.ndarray]
    maximum_translation_m: float
    maximum_rotation_degrees: float
    reason: str


class SubmapPoseGraph:
    """Small Open3D pose graph with rigid submap nodes and fail-closed loops."""

    def __init__(self, o3d: Any, voxel_size_m: float) -> None:
        self.o3d = o3d
        self.voxel_size_m = voxel_size_m
        self.ids: list[str] = []
        self.transforms: list[np.ndarray] = []
        self.odometry_information: list[np.ndarray] = []
        self.loops: list[PoseGraphLoop] = []

    def add_submap(
        self,
        submap_id: str,
        global_from_local: np.ndarray,
        odometry_information: np.ndarray | None = None,
    ) -> None:
        if submap_id in self.ids:
            raise ValueError(f"Duplicate pose-graph node {submap_id}")
        self.ids.append(submap_id)
        self.transforms.append(np.asarray(global_from_local, dtype=np.float64).copy())
        if len(self.transforms) > 1:
            self.odometry_information.append(
                np.eye(6, dtype=np.float64) * 100.0
                if odometry_information is None
                else np.asarray(odometry_information, dtype=np.float64).copy()
            )

    def add_loop(self, loop: PoseGraphLoop) -> None:
        if loop.source_id not in self.ids or loop.target_id not in self.ids:
            raise ValueError("Loop endpoints must already exist in the pose graph")
        self.loops.append(loop)

    def _build(self) -> Any:
        registration = self.o3d.pipelines.registration
        graph = registration.PoseGraph()
        for transform in self.transforms:
            graph.nodes.append(registration.PoseGraphNode(transform))
        for source_index in range(len(self.transforms) - 1):
            target_index = source_index + 1
            target_from_source = (
                np.linalg.inv(self.transforms[target_index]) @ self.transforms[source_index]
            )
            graph.edges.append(
                registration.PoseGraphEdge(
                    source_index,
                    target_index,
                    target_from_source,
                    self.odometry_information[source_index],
                    uncertain=False,
                )
            )
        for loop in self.loops:
            graph.edges.append(
                registration.PoseGraphEdge(
                    self.ids.index(loop.source_id),
                    self.ids.index(loop.target_id),
                    loop.target_from_source,
                    loop.information,
                    uncertain=True,
                )
            )
        return graph

    def optimize(self) -> PoseGraphSolution:
        if not self.loops:
            return PoseGraphSolution(False, {}, 0.0, 0.0, "no verified loop constraint")
        registration = self.o3d.pipelines.registration
        graph = self._build()
        registration.global_optimization(
            graph,
            registration.GlobalOptimizationLevenbergMarquardt(),
            registration.GlobalOptimizationConvergenceCriteria(),
            registration.GlobalOptimizationOption(
                max_correspondence_distance=max(0.03, self.voxel_size_m * 3.0),
                edge_prune_threshold=0.25,
                preference_loop_closure=5.0,
                reference_node=0,
            ),
        )
        solved = [np.asarray(node.pose, dtype=np.float64) for node in graph.nodes]
        corrections = [
            transform_delta(before, after)
            for before, after in zip(self.transforms, solved, strict=True)
        ]
        maximum_translation = max((value[0] for value in corrections), default=0.0)
        maximum_rotation = max((value[1] for value in corrections), default=0.0)
        if maximum_translation > 0.75 or maximum_rotation > 25.0:
            return PoseGraphSolution(
                False,
                {},
                maximum_translation,
                maximum_rotation,
                "pose-graph correction exceeded the live safety gate",
            )
        for loop in self.loops:
            source = solved[self.ids.index(loop.source_id)]
            target = solved[self.ids.index(loop.target_id)]
            predicted = np.linalg.inv(target) @ source
            residual = np.linalg.inv(loop.target_from_source) @ predicted
            translation_error = float(np.linalg.norm(residual[:3, 3]))
            rotation_error = rotation_degrees(residual)
            if translation_error > 0.10 or rotation_error > 6.0:
                return PoseGraphSolution(
                    False,
                    {},
                    maximum_translation,
                    maximum_rotation,
                    "optimized loop residual exceeded the live safety gate",
                )
        self.transforms = [value.copy() for value in solved]
        return PoseGraphSolution(
            True,
            dict(zip(self.ids, self.transforms, strict=True)),
            maximum_translation,
            maximum_rotation,
            "verified loop optimized",
        )


def verify_loop_candidate(
    o3d: Any,
    *,
    source_points: np.ndarray,
    target_points: np.ndarray,
    initial_target_from_source: np.ndarray,
    voxel_size_m: float,
) -> LoopVerification:
    identity = np.eye(4, dtype=np.float64)
    empty_information = np.eye(6, dtype=np.float64)
    if len(source_points) < 500 or len(target_points) < 500:
        return LoopVerification(False, identity, empty_information, 0.0, math.inf, 0, "too few points")
    downsample = max(0.025, voxel_size_m * 3.0)
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_points))
    source = source.voxel_down_sample(downsample)
    target = target.voxel_down_sample(downsample)
    if len(source.points) < 200 or len(target.points) < 200:
        return LoopVerification(False, identity, empty_information, 0.0, math.inf, 0, "too few downsampled points")
    normal_radius = downsample * 2.5
    search = o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    source.estimate_normals(search)
    target.estimate_normals(search)
    maximum_distance = max(0.04, voxel_size_m * 4.0)
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        maximum_distance,
        np.asarray(initial_target_from_source, dtype=np.float64),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.HuberLoss(maximum_distance * 0.5)
        ),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
    )
    correspondences = len(result.correspondence_set)
    delta = np.asarray(result.transformation) @ np.linalg.inv(initial_target_from_source)
    correction_translation = float(np.linalg.norm(delta[:3, 3]))
    correction_rotation = rotation_degrees(delta)
    accepted = (
        float(result.fitness) >= 0.35
        and float(result.inlier_rmse) <= max(0.025, voxel_size_m * 2.5)
        and correspondences >= 200
        and correction_translation <= 0.50
        and correction_rotation <= 15.0
    )
    if float(result.fitness) < 0.35:
        reason = "insufficient geometric overlap"
    elif float(result.inlier_rmse) > max(0.025, voxel_size_m * 2.5):
        reason = "loop alignment residual too high"
    elif correspondences < 200:
        reason = "too few verified correspondences"
    elif correction_translation > 0.50 or correction_rotation > 15.0:
        reason = "loop correction outside the live motion gate"
    else:
        reason = "strict geometric verification passed"
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source,
        target,
        maximum_distance,
        result.transformation,
    )
    return LoopVerification(
        accepted,
        np.asarray(result.transformation, dtype=np.float64),
        np.asarray(information, dtype=np.float64),
        float(result.fitness),
        float(result.inlier_rmse),
        correspondences,
        reason,
    )


def submap_odometry_information(
    o3d: Any,
    *,
    source_points: np.ndarray,
    target_points: np.ndarray,
    target_from_source: np.ndarray,
    voxel_size_m: float,
) -> np.ndarray:
    """Measure the adjacent-edge information matrix, with a safe finite fallback."""
    fallback = np.eye(6, dtype=np.float64) * 100.0
    if len(source_points) < 200 or len(target_points) < 200:
        return fallback
    downsample = max(0.025, voxel_size_m * 3.0)
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_points))
    source = source.voxel_down_sample(downsample)
    target = target.voxel_down_sample(downsample)
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source,
        target,
        max(0.04, voxel_size_m * 4.0),
        np.asarray(target_from_source, dtype=np.float64),
    )
    values = np.asarray(information, dtype=np.float64)
    return values if values.shape == (6, 6) and np.isfinite(values).all() else fallback


def loop_event(
    *,
    sequence: int,
    source_id: str,
    target_id: str,
    verification: LoopVerification,
    solution: PoseGraphSolution | None,
) -> dict[str, Any]:
    accepted = bool(verification.accepted and solution is not None and solution.accepted)
    return {
        "schemaVersion": 1,
        "recordedAtUnixNs": time.time_ns(),
        "sequence": int(sequence),
        "sourceSubmapId": source_id,
        "targetSubmapId": target_id,
        "accepted": accepted,
        "verification": {
            "fitness": verification.fitness,
            "rmseM": verification.rmse_m if math.isfinite(verification.rmse_m) else None,
            "correspondenceCount": verification.correspondence_count,
            "reason": verification.reason,
            "targetFromSource": verification.target_from_source.reshape(-1).tolist(),
            "information": verification.information.reshape(-1).tolist(),
        },
        "optimization": None
        if solution is None
        else {
            "accepted": solution.accepted,
            "maximumTranslationM": solution.maximum_translation_m,
            "maximumRotationDegrees": solution.maximum_rotation_degrees,
            "reason": solution.reason,
        },
        "requiresProductionRevalidation": True,
    }
