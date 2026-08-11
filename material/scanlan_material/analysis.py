from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import MATERIAL_CLASSES, OPTICAL_RISKS, MaterialPrediction


ANALYSIS_VERSION = "scanlan-two-pass-material-v1"
SURFACE_CONTRACT_VERSION = "scanlan-material-surface-v1"


@dataclass(frozen=True)
class MaterialCamera:
    """One calibrated, undistorted source view used by material analysis."""

    identifier: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_from_camera: np.ndarray
    pose_confidence: float = 1.0
    depth_m: np.ndarray | None = None
    metadata: Mapping[str, Any] | None = None

    def validated(self) -> "MaterialCamera":
        if not self.identifier:
            raise ValueError("material camera identifier cannot be empty")
        if self.width <= 0 or self.height <= 0 or self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError(f"material camera {self.identifier} has invalid calibration")
        calibration = np.asarray([self.fx, self.fy, self.cx, self.cy], dtype=np.float64)
        if not np.isfinite(calibration).all():
            raise ValueError(f"material camera {self.identifier} has non-finite calibration")
        pose = np.asarray(self.world_from_camera, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f"material camera {self.identifier} must have a finite 4x4 pose")
        if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
            raise ValueError(f"material camera {self.identifier} has an invalid homogeneous row")
        rotation = pose[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3) or not math.isclose(
            float(np.linalg.det(rotation)), 1.0, abs_tol=2e-3
        ):
            raise ValueError(f"material camera {self.identifier} pose is not rigid")
        confidence = float(self.pose_confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"material camera {self.identifier} has invalid pose confidence")
        depth = None
        if self.depth_m is not None:
            depth = np.asarray(self.depth_m, dtype=np.float32)
            if depth.shape != (self.height, self.width):
                raise ValueError(
                    f"material camera {self.identifier} depth must match its source grid"
                )
            if not np.isfinite(depth).all() or np.any(depth < 0.0):
                raise ValueError(f"material camera {self.identifier} has invalid metric depth")
        return MaterialCamera(
            self.identifier,
            int(self.width),
            int(self.height),
            float(self.fx),
            float(self.fy),
            float(self.cx),
            float(self.cy),
            pose,
            confidence,
            depth,
            dict(self.metadata or {}),
        )


@dataclass(frozen=True)
class MaterialView:
    camera: MaterialCamera
    prediction: MaterialPrediction

    def validated(self) -> "MaterialView":
        camera = self.camera.validated()
        prediction = self.prediction.validated()
        if prediction.valid_mask.shape != (camera.height, camera.width):
            raise ValueError(
                f"material prediction for {camera.identifier} is not source-aligned"
            )
        return MaterialView(camera, prediction)


@dataclass(frozen=True)
class FusedMaterialSurface:
    class_probabilities: np.ndarray
    optical_risk_probabilities: np.ndarray
    valid_mask: np.ndarray
    confidence: np.ndarray
    support_count: np.ndarray
    effective_view_count: np.ndarray
    region_ids: np.ndarray
    albedo_linear: np.ndarray | None = None
    roughness: np.ndarray | None = None
    metallic: np.ndarray | None = None
    transmission: np.ndarray | None = None
    normal_world: np.ndarray | None = None
    emission_linear: np.ndarray | None = None
    metadata: Mapping[str, Any] | None = None

    def validated(self) -> "FusedMaterialSurface":
        classes = np.asarray(self.class_probabilities, dtype=np.float32)
        if classes.ndim != 2 or classes.shape[1] != len(MATERIAL_CLASSES):
            raise ValueError("surface class probabilities must be VxC")
        vertex_count = len(classes)
        risks = np.asarray(self.optical_risk_probabilities, dtype=np.float32)
        if risks.shape != (vertex_count, len(OPTICAL_RISKS)):
            raise ValueError("surface optical-risk probabilities must be VxR")
        valid = np.asarray(self.valid_mask, dtype=bool)
        confidence = np.asarray(self.confidence, dtype=np.float32)
        support = np.asarray(self.support_count, dtype=np.uint16)
        effective = np.asarray(self.effective_view_count, dtype=np.float32)
        regions = np.asarray(self.region_ids, dtype=np.int32)
        for name, value in {
            "valid mask": valid,
            "confidence": confidence,
            "support count": support,
            "effective view count": effective,
            "region identifiers": regions,
        }.items():
            if value.shape != (vertex_count,):
                raise ValueError(f"surface {name} must have one value per vertex")
        if not np.isfinite(classes).all() or not np.isfinite(risks).all():
            raise ValueError("surface probabilities must be finite")
        if (
            np.any(classes < 0.0)
            or np.any(classes > 1.0)
            or np.any(risks < 0.0)
            or np.any(risks > 1.0)
        ):
            raise ValueError("surface probabilities must remain in [0, 1]")
        if np.any(valid & (np.abs(np.sum(classes, axis=1) - 1.0) > 2e-3)):
            raise ValueError("valid surface class probabilities must sum to one")
        if np.any(~valid & ((confidence > 0.0) | (np.sum(risks, axis=1) > 0.0))):
            raise ValueError("unsupported surface vertices cannot retain confidence or risk")
        if (
            not np.isfinite(confidence).all()
            or np.any(confidence < 0.0)
            or np.any(confidence > 1.0)
        ):
            raise ValueError("surface confidence must remain in [0, 1]")
        if not np.isfinite(effective).all() or np.any(effective < 0.0):
            raise ValueError("effective view count must be finite and non-negative")

        scalar_fields: dict[str, np.ndarray | None] = {}
        for name in ("roughness", "metallic", "transmission"):
            value = getattr(self, name)
            if value is None:
                scalar_fields[name] = None
                continue
            array = np.asarray(value, dtype=np.float32)
            if array.shape != (vertex_count,) or not np.isfinite(array).all():
                raise ValueError(f"surface {name} must be finite V")
            if np.any(array < 0.0) or np.any(array > 1.0):
                raise ValueError(f"surface {name} must remain in [0, 1]")
            scalar_fields[name] = array

        vector_fields: dict[str, np.ndarray | None] = {}
        for name in ("albedo_linear", "normal_world", "emission_linear"):
            value = getattr(self, name)
            if value is None:
                vector_fields[name] = None
                continue
            array = np.asarray(value, dtype=np.float32)
            if array.shape != (vertex_count, 3) or not np.isfinite(array).all():
                raise ValueError(f"surface {name} must be finite Vx3")
            if name != "normal_world" and np.any(array < 0.0):
                raise ValueError(f"surface {name} cannot be negative")
            if name == "albedo_linear" and np.any(array > 1.0):
                raise ValueError("surface linear albedo must remain in [0, 1]")
            if name == "normal_world":
                lengths = np.linalg.norm(array, axis=1)
                if np.any(valid & (np.abs(lengths - 1.0) > 2e-2)):
                    raise ValueError("valid world-space material normals must be unit length")
            vector_fields[name] = array

        return FusedMaterialSurface(
            classes,
            risks,
            valid,
            confidence,
            support,
            effective,
            regions,
            vector_fields["albedo_linear"],
            scalar_fields["roughness"],
            scalar_fields["metallic"],
            scalar_fields["transmission"],
            vector_fields["normal_world"],
            vector_fields["emission_linear"],
            dict(self.metadata or {}),
        )


def vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    points, faces = _validated_mesh(vertices, triangles)
    normals = np.zeros_like(points, dtype=np.float64)
    face_normals = np.cross(
        points[faces[:, 1]] - points[faces[:, 0]],
        points[faces[:, 2]] - points[faces[:, 0]],
    )
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    usable = lengths > 1e-12
    normals[usable] /= lengths[usable, None]
    return normals.astype(np.float32)


def _validated_mesh(
    vertices: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("material surface vertices must be non-empty Vx3")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("material surface triangles must be non-empty Fx3")
    if not np.isfinite(points).all():
        raise ValueError("material surface vertices must be finite")
    if np.any(faces < 0) or np.any(faces >= len(points)):
        raise ValueError("material surface triangles reference invalid vertices")
    return points, faces


def _rotation_distance(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(left.T @ right) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cosine)


def select_coarse_cameras(
    cameras: Sequence[MaterialCamera], maximum_views: int = 48
) -> tuple[int, ...]:
    """K-center pose sampling with a scale derived from the actual camera path."""

    checked = tuple(camera.validated() for camera in cameras)
    if not checked:
        raise ValueError("two-pass material analysis requires calibrated cameras")
    limit = max(1, min(int(maximum_views), len(checked)))
    if limit == len(checked):
        return tuple(range(len(checked)))
    centers = np.stack([camera.world_from_camera[:3, 3] for camera in checked])
    rotations = np.stack([camera.world_from_camera[:3, :3] for camera in checked])
    adjacent = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    moving = adjacent[adjacent > 1e-6]
    translation_scale = (
        float(np.median(moving)) * 4.0
        if len(moving)
        else max(float(np.linalg.norm(np.ptp(centers, axis=0))) / 8.0, 0.1)
    )
    translation_scale = max(translation_scale, 1e-3)
    rotation_scale = math.radians(20.0)

    chosen = [0]
    if limit > 1 and len(checked) > 1:
        chosen.append(len(checked) - 1)
    while len(chosen) < limit:
        best_index = -1
        best_score = -1.0
        for index, camera in enumerate(checked):
            if index in chosen:
                continue
            nearest = min(
                math.hypot(
                    float(np.linalg.norm(centers[index] - centers[other])) / translation_scale,
                    _rotation_distance(rotations[index], rotations[other]) / rotation_scale,
                )
                for other in chosen
            )
            score = nearest * (0.25 + 0.75 * camera.pose_confidence)
            if score > best_score + 1e-12:
                best_index, best_score = index, score
        chosen.append(best_index)
    return tuple(sorted(chosen))


def _project_surface(
    vertices: np.ndarray,
    normals: np.ndarray,
    camera: MaterialCamera,
    *,
    absolute_depth_tolerance_m: float = 0.025,
    relative_depth_tolerance: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera_from_world = np.linalg.inv(camera.world_from_camera)
    camera_points = vertices @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    z = camera_points[:, 2]
    positive = z > 1e-6
    u = np.full(len(vertices), -1.0, dtype=np.float64)
    v = np.full(len(vertices), -1.0, dtype=np.float64)
    u[positive] = camera.fx * camera_points[positive, 0] / z[positive] + camera.cx
    v[positive] = camera.fy * camera_points[positive, 1] / z[positive] + camera.cy
    visible = positive & (u >= 0.0) & (u < camera.width) & (v >= 0.0) & (v < camera.height)
    if not np.any(visible):
        empty = np.empty(0, dtype=np.int64)
        return empty, u[empty], v[empty], np.empty(0, dtype=np.float32)

    indices = np.flatnonzero(visible)
    rounded_u = np.clip(np.rint(u[indices]).astype(np.int64), 0, camera.width - 1)
    rounded_v = np.clip(np.rint(v[indices]).astype(np.int64), 0, camera.height - 1)
    tolerance = np.maximum(
        absolute_depth_tolerance_m, relative_depth_tolerance * z[indices]
    )
    if camera.depth_m is not None:
        reference_depth = camera.depth_m[rounded_v, rounded_u]
        depth_visible = (reference_depth > 0.0) & (
            np.abs(reference_depth - z[indices]) <= tolerance
        )
    else:
        pixel = rounded_v * camera.width + rounded_u
        zbuffer = np.full(camera.width * camera.height, np.inf, dtype=np.float32)
        np.minimum.at(zbuffer, pixel, z[indices].astype(np.float32))
        depth_visible = z[indices] <= zbuffer[pixel] + tolerance
    indices = indices[depth_visible]
    if not len(indices):
        return indices, u[indices], v[indices], np.empty(0, dtype=np.float32)

    to_camera = camera.world_from_camera[:3, 3] - vertices[indices]
    distance = np.linalg.norm(to_camera, axis=1)
    to_camera /= np.maximum(distance[:, None], 1e-9)
    signed_facing = np.sum(normals[indices] * to_camera, axis=1)
    # Reconstruction backends can disagree about winding. Pick one global sign
    # per view, never a per-vertex absolute value that would expose back faces.
    sign = -1.0 if np.mean(signed_facing > 0.0) < 0.35 else 1.0
    facing = np.clip(signed_facing * sign, 0.0, 1.0)
    border = np.minimum.reduce(
        (u[indices], v[indices], camera.width - 1.0 - u[indices], camera.height - 1.0 - v[indices])
    )
    border_scale = max(1.0, min(16.0, 0.05 * min(camera.width, camera.height)))
    weights = (
        camera.pose_confidence
        * facing**2
        * np.clip(border / border_scale, 0.0, 1.0)
        / np.sqrt(np.maximum(distance, 0.25))
    )
    useful = weights > 1e-5
    return (
        indices[useful],
        u[indices][useful],
        v[indices][useful],
        np.clip(weights[useful], 0.0, 1.0).astype(np.float32),
    )


def _sample_prediction(
    prediction: MaterialPrediction, u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.clip(np.rint(u).astype(np.int64), 0, prediction.valid_mask.shape[1] - 1)
    y = np.clip(np.rint(v).astype(np.int64), 0, prediction.valid_mask.shape[0] - 1)
    return (
        prediction.class_probabilities[y, x],
        prediction.optical_risk_probabilities[y, x],
        prediction.valid_mask[y, x],
        prediction.confidence[y, x],
    )


def select_final_views(
    vertices: np.ndarray,
    normals: np.ndarray,
    coarse_views: Sequence[MaterialView],
    maximum_views: int = 24,
    planning_vertex_limit: int = 50_000,
) -> tuple[int, ...]:
    """Greedily cover geometry while giving coarse optical risk extra authority."""

    points = np.asarray(vertices, dtype=np.float64)
    surface_normals = np.asarray(normals, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or surface_normals.shape != points.shape:
        raise ValueError("final-view planning requires aligned Vx3 vertices and normals")
    checked = tuple(view.validated() for view in coarse_views)
    if not checked:
        raise ValueError("final-view planning requires coarse optical-risk predictions")
    limit = max(1, min(int(maximum_views), len(checked)))
    sample_count = min(len(points), max(1, int(planning_vertex_limit)))
    sample_indices = np.unique(np.linspace(0, len(points) - 1, sample_count, dtype=np.int64))
    sampled_points = points[sample_indices]
    sampled_normals = surface_normals[sample_indices]
    scores = np.zeros((len(checked), len(sample_indices)), dtype=np.float16)
    for view_index, view in enumerate(checked):
        indices, u, v, geometry_weight = _project_surface(
            sampled_points, sampled_normals, view.camera
        )
        if not len(indices):
            continue
        _, risks, valid, confidence = _sample_prediction(view.prediction, u, v)
        risk = np.max(risks, axis=1)
        score = geometry_weight * confidence * valid * (1.0 + 2.0 * risk)
        scores[view_index, indices] = score.astype(np.float16)

    chosen: list[int] = []
    covered = np.zeros(len(sample_indices), dtype=np.float32)
    while len(chosen) < limit:
        best_index = -1
        best_gain = -1.0
        for index in range(len(checked)):
            if index in chosen:
                continue
            candidate = scores[index].astype(np.float32)
            gain = float(np.sum(np.maximum(covered, candidate) - covered))
            if gain > best_gain + 1e-8:
                best_index, best_gain = index, gain
        minimum_redundant_views = min(2, limit)
        if best_index < 0 or (
            len(chosen) >= minimum_redundant_views and best_gain <= 1e-8
        ):
            break
        chosen.append(best_index)
        covered = np.maximum(covered, scores[best_index].astype(np.float32))
    return tuple(chosen)


def _connected_regions(
    triangles: np.ndarray,
    classes: np.ndarray,
    risks: np.ndarray,
    valid: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    vertex_count = len(classes)
    labels = np.argmax(classes, axis=1).astype(np.int16)
    usable = valid & (confidence >= 0.10) & (labels != MATERIAL_CLASSES.index("unknown"))
    risk_signature = np.zeros(vertex_count, dtype=np.uint16)
    for risk_index in range(len(OPTICAL_RISKS)):
        risk_signature |= ((risks[:, risk_index] >= 0.5).astype(np.uint16) << risk_index)
    edges = np.concatenate((triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]))
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    accepted = (
        usable[edges[:, 0]]
        & usable[edges[:, 1]]
        & (labels[edges[:, 0]] == labels[edges[:, 1]])
        & (risk_signature[edges[:, 0]] == risk_signature[edges[:, 1]])
        & (np.sum(np.abs(classes[edges[:, 0]] - classes[edges[:, 1]]), axis=1) <= 0.6)
    )
    parent = np.arange(vertex_count, dtype=np.int32)
    rank = np.zeros(vertex_count, dtype=np.uint8)

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = int(parent[root])
        while parent[value] != value:
            next_value = int(parent[value])
            parent[value] = root
            value = next_value
        return root

    for left, right in edges[accepted]:
        left_root, right_root = find(int(left)), find(int(right))
        if left_root == right_root:
            continue
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    regions = np.full(vertex_count, -1, dtype=np.int32)
    usable_indices = np.flatnonzero(usable)
    roots = np.fromiter((find(int(index)) for index in usable_indices), dtype=np.int32)
    _, compact = np.unique(roots, return_inverse=True)
    regions[usable_indices] = compact.astype(np.int32)
    return regions


def fuse_material_surface(
    vertices: np.ndarray,
    normals: np.ndarray,
    triangles: np.ndarray,
    views: Sequence[MaterialView],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> FusedMaterialSurface:
    """Fuse calibrated 2D observations into conservative 3D material regions."""

    points, faces = _validated_mesh(vertices, triangles)
    surface_normals = np.asarray(normals, dtype=np.float64)
    if surface_normals.shape != points.shape or not np.isfinite(surface_normals).all():
        raise ValueError("material surface normals must be finite and vertex-aligned")
    normal_lengths = np.linalg.norm(surface_normals, axis=1)
    usable_normals = normal_lengths >= 1e-6
    if not np.any(usable_normals):
        raise ValueError("material surface has no usable vertex normals")
    # A repaired/indexed mesh can retain isolated vertices. They remain
    # unsupported; one unused vertex must not invalidate the observed surface.
    surface_normals[usable_normals] /= normal_lengths[usable_normals, None]
    surface_normals[~usable_normals] = 0.0
    checked = tuple(view.validated() for view in views)
    if not checked:
        raise ValueError("material fusion requires final-pass predictions")
    for field_name in (
        "albedo_linear",
        "roughness",
        "metallic",
        "transmission",
        "normal_camera",
        "emission_linear",
    ):
        presence = [getattr(view.prediction, field_name) is not None for view in checked]
        if any(presence) and not all(presence):
            raise ValueError(
                f"final material views must agree on optional {field_name} availability"
            )

    vertex_count = len(points)
    class_evidence = np.zeros((vertex_count, len(MATERIAL_CLASSES)), dtype=np.float64)
    risk_evidence = np.zeros((vertex_count, len(OPTICAL_RISKS)), dtype=np.float64)
    risk_peak = np.zeros_like(risk_evidence)
    weight_sum = np.zeros(vertex_count, dtype=np.float64)
    weight_square_sum = np.zeros(vertex_count, dtype=np.float64)
    support_count = np.zeros(vertex_count, dtype=np.uint16)
    field_sums: dict[str, np.ndarray] = {}
    field_weights: dict[str, np.ndarray] = {}

    for view in checked:
        indices, u, v, geometry_weight = _project_surface(points, surface_normals, view.camera)
        if not len(indices):
            continue
        classes, risks, prediction_valid, prediction_confidence = _sample_prediction(
            view.prediction, u, v
        )
        useful = prediction_valid & (prediction_confidence > 0.0)
        indices = indices[useful]
        if not len(indices):
            continue
        weights = geometry_weight[useful] * prediction_confidence[useful]
        classes = classes[useful]
        risks = risks[useful]
        class_evidence[indices] += weights[:, None] * classes
        risk_evidence[indices] += weights[:, None] * risks
        # A high-confidence optical warning from one geometrically sound view
        # must survive averaging with many easy opaque views.
        risk_peak[indices] = np.maximum(
            risk_peak[indices],
            risks
            * prediction_confidence[useful, None]
            * np.sqrt(
                np.clip(geometry_weight[useful, None] / 0.25, 0.0, 1.0)
            ),
        )
        weight_sum[indices] += weights
        weight_square_sum[indices] += weights**2
        support_count[indices] = np.minimum(
            np.iinfo(np.uint16).max, support_count[indices].astype(np.uint32) + 1
        ).astype(np.uint16)

        for name in (
            "albedo_linear",
            "roughness",
            "metallic",
            "transmission",
            "normal_camera",
            "emission_linear",
        ):
            value = getattr(view.prediction, name)
            if value is None:
                continue
            sampled = value[
                np.clip(np.rint(v[useful]).astype(np.int64), 0, view.camera.height - 1),
                np.clip(np.rint(u[useful]).astype(np.int64), 0, view.camera.width - 1),
            ].astype(np.float64)
            output_name = "normal_world" if name == "normal_camera" else name
            if name == "normal_camera":
                sampled = sampled @ view.camera.world_from_camera[:3, :3].T
            if output_name not in field_sums:
                shape = (vertex_count, 3) if sampled.ndim == 2 else (vertex_count,)
                field_sums[output_name] = np.zeros(shape, dtype=np.float64)
                field_weights[output_name] = np.zeros(vertex_count, dtype=np.float64)
            contribution = (
                weights[:, None] * sampled if sampled.ndim == 2 else weights * sampled
            )
            field_sums[output_name][indices] += contribution
            field_weights[output_name][indices] += weights

    valid = weight_sum > 1e-6
    classes = np.zeros_like(class_evidence, dtype=np.float32)
    classes[:, MATERIAL_CLASSES.index("unknown")] = 1.0
    classes[valid] = (class_evidence[valid] / weight_sum[valid, None]).astype(np.float32)
    classes[valid] /= np.maximum(np.sum(classes[valid], axis=1, keepdims=True), 1e-9)
    risk_mean = np.zeros_like(risk_evidence, dtype=np.float32)
    risk_mean[valid] = (risk_evidence[valid] / weight_sum[valid, None]).astype(np.float32)
    risks = np.maximum(risk_mean, risk_peak.astype(np.float32))
    risks[~valid] = 0.0
    effective = np.zeros(vertex_count, dtype=np.float32)
    effective[valid] = (
        weight_sum[valid] ** 2 / np.maximum(weight_square_sum[valid], 1e-12)
    ).astype(np.float32)
    entropy = np.zeros(vertex_count, dtype=np.float64)
    entropy[valid] = -np.sum(
        classes[valid] * np.log(np.maximum(classes[valid], 1e-9)), axis=1
    ) / math.log(len(MATERIAL_CLASSES))
    confidence = np.zeros(vertex_count, dtype=np.float32)
    confidence[valid] = np.clip(
        (1.0 - np.exp(-weight_sum[valid]))
        * (1.0 - entropy[valid])
        * np.clip(effective[valid] / 2.0, 0.25, 1.0),
        0.0,
        1.0,
    ).astype(np.float32)

    fused_fields: dict[str, np.ndarray | None] = {}
    for name in (
        "albedo_linear",
        "roughness",
        "metallic",
        "transmission",
        "normal_world",
        "emission_linear",
    ):
        if name not in field_sums:
            fused_fields[name] = None
            continue
        output = np.zeros_like(field_sums[name], dtype=np.float32)
        field_valid = field_weights[name] > 1e-6
        if output.ndim == 2:
            output[field_valid] = (
                field_sums[name][field_valid] / field_weights[name][field_valid, None]
            ).astype(np.float32)
        else:
            output[field_valid] = (
                field_sums[name][field_valid] / field_weights[name][field_valid]
            ).astype(np.float32)
        if name == "normal_world":
            lengths = np.linalg.norm(output, axis=1)
            usable = field_valid & (lengths > 1e-6)
            output[usable] /= lengths[usable, None]
            output[field_valid & ~usable] = surface_normals[field_valid & ~usable]
            output[valid & ~field_valid] = surface_normals[valid & ~field_valid]
        fused_fields[name] = output

    regions = _connected_regions(faces, classes, risks, valid, confidence)
    region_count = int(np.max(regions) + 1) if np.any(regions >= 0) else 0
    result_metadata = {
        "analysisVersion": ANALYSIS_VERSION,
        "sourceViews": [view.camera.identifier for view in checked],
        "sourceViewCount": len(checked),
        "supportedVertexCount": int(np.count_nonzero(valid)),
        "multiViewVertexCount": int(np.count_nonzero(effective >= 1.5)),
        "regionCount": region_count,
        **dict(metadata or {}),
    }
    return FusedMaterialSurface(
        classes,
        risks,
        valid,
        confidence,
        support_count,
        effective,
        regions,
        fused_fields["albedo_linear"],
        fused_fields["roughness"],
        fused_fields["metallic"],
        fused_fields["transmission"],
        fused_fields["normal_world"],
        fused_fields["emission_linear"],
        result_metadata,
    ).validated()


def run_two_pass_analysis(
    vertices: np.ndarray,
    normals: np.ndarray,
    triangles: np.ndarray,
    cameras: Sequence[MaterialCamera],
    coarse_inference: Callable[[MaterialCamera], MaterialPrediction],
    final_inference: Callable[[MaterialCamera], MaterialPrediction],
    *,
    maximum_coarse_views: int = 48,
    maximum_final_views: int = 24,
) -> FusedMaterialSurface:
    coarse_indices = select_coarse_cameras(cameras, maximum_coarse_views)
    coarse_views = tuple(
        MaterialView(cameras[index], coarse_inference(cameras[index])).validated()
        for index in coarse_indices
    )
    selected_positions = select_final_views(
        vertices, normals, coarse_views, maximum_final_views
    )
    final_views = tuple(
        MaterialView(
            coarse_views[position].camera,
            final_inference(coarse_views[position].camera),
        ).validated()
        for position in selected_positions
    )
    return fuse_material_surface(
        vertices,
        normals,
        triangles,
        final_views,
        metadata={
            "coarseViews": [view.camera.identifier for view in coarse_views],
            "finalViews": [view.camera.identifier for view in final_views],
        },
    )


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_surface_material(path: Path, surface: FusedMaterialSurface) -> None:
    checked = surface.validated()
    arrays: dict[str, np.ndarray] = {
        "contract": np.asarray(SURFACE_CONTRACT_VERSION),
        "class_probabilities": checked.class_probabilities.astype(np.float16),
        "optical_risk_probabilities": checked.optical_risk_probabilities.astype(np.float16),
        "valid_mask": checked.valid_mask.astype(np.uint8),
        "confidence": checked.confidence.astype(np.float16),
        "support_count": checked.support_count.astype(np.uint16),
        "effective_view_count": checked.effective_view_count.astype(np.float16),
        "region_ids": checked.region_ids.astype(np.int32),
        "metadata_json": np.asarray(
            json.dumps(dict(checked.metadata or {}), sort_keys=True, separators=(",", ":"))
        ),
    }
    for name in (
        "albedo_linear",
        "roughness",
        "metallic",
        "transmission",
        "normal_world",
        "emission_linear",
    ):
        value = getattr(checked, name)
        if value is not None:
            arrays[name] = value.astype(np.float16)
    _atomic_npz(path, arrays)


def read_surface_material(path: Path) -> FusedMaterialSurface:
    with np.load(path, allow_pickle=False) as archive:
        contract = str(archive["contract"].item())
        if contract != SURFACE_CONTRACT_VERSION:
            raise ValueError(f"unsupported surface material contract: {contract}")
        optional = {
            name: np.asarray(archive[name], dtype=np.float32) if name in archive else None
            for name in (
                "albedo_linear",
                "roughness",
                "metallic",
                "transmission",
                "normal_world",
                "emission_linear",
            )
        }
        result = FusedMaterialSurface(
            np.asarray(archive["class_probabilities"], dtype=np.float32),
            np.asarray(archive["optical_risk_probabilities"], dtype=np.float32),
            np.asarray(archive["valid_mask"], dtype=bool),
            np.asarray(archive["confidence"], dtype=np.float32),
            np.asarray(archive["support_count"], dtype=np.uint16),
            np.asarray(archive["effective_view_count"], dtype=np.float32),
            np.asarray(archive["region_ids"], dtype=np.int32),
            optional["albedo_linear"],
            optional["roughness"],
            optional["metallic"],
            optional["transmission"],
            optional["normal_world"],
            optional["emission_linear"],
            json.loads(str(archive["metadata_json"].item())),
        )
    return result.validated()
