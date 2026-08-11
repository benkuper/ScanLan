from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


VALIDATION_CONTRACT_VERSION = 1


def _mad(values: np.ndarray) -> float:
    if not len(values):
        return 0.0
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _finite_max(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite, initial=0.0))


def _report(kind: str, accepted: bool, metrics: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    return {
        "contractVersion": VALIDATION_CONTRACT_VERSION,
        "kind": kind,
        "accepted": bool(accepted),
        "metrics": metrics,
        "reasons": reasons,
    }


@dataclass(frozen=True)
class CameraValidationConfig:
    minimum_confidence: float = 0.5
    maximum_rotation_step_degrees: float = 75.0
    maximum_translation_step: float | None = None
    adaptive_translation_limit: bool = True
    translation_median_multiplier: float = 4.0
    robust_sigma_multiplier: float = 6.0
    orthogonality_tolerance: float = 2e-2
    determinant_tolerance: float = 2e-2


@dataclass(frozen=True)
class CameraValidationResult:
    accepted: bool
    frame_mask: np.ndarray
    drift_risk: float
    translation_limit: float | None
    metrics: dict[str, Any]
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _report("camera", self.accepted, self.metrics, list(self.reasons))


def _rotation_step_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def validate_camera_trajectory(
    world_from_cameras: np.ndarray,
    confidence: np.ndarray | None = None,
    config: CameraValidationConfig | None = None,
    sample_positions: np.ndarray | None = None,
) -> CameraValidationResult:
    config = config or CameraValidationConfig()
    poses = np.asarray(world_from_cameras, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
        mask = np.zeros(len(poses) if poses.ndim else 0, dtype=bool)
        return CameraValidationResult(
            False, mask, 1.0, None, {"frameCount": int(len(mask))}, ("camera poses must be a non-empty Nx4x4 array",)
        )
    scores = (
        np.ones(len(poses), dtype=np.float64)
        if confidence is None
        else np.asarray(confidence, dtype=np.float64)
    )
    if scores.shape != (len(poses),):
        raise ValueError("camera confidence must contain one value per pose")
    positions = (
        np.arange(len(poses), dtype=np.float64)
        if sample_positions is None
        else np.asarray(sample_positions, dtype=np.float64)
    )
    if positions.shape != (len(poses),):
        raise ValueError("camera sample positions must contain one value per pose")
    position_steps = np.diff(positions)
    if np.any(~np.isfinite(positions)) or np.any(position_steps <= 0.0):
        raise ValueError("camera sample positions must be finite and strictly increasing")

    finite = np.isfinite(poses).all(axis=(1, 2)) & np.isfinite(scores)
    homogeneous = np.max(np.abs(poses[:, 3] - np.asarray([0.0, 0.0, 0.0, 1.0])), axis=1) <= 1e-5
    rotations = poses[:, :3, :3]
    identity = np.eye(3)
    orthogonality_error = np.linalg.norm(
        np.transpose(rotations, (0, 2, 1)) @ rotations - identity,
        axis=(1, 2),
    )
    determinants = np.linalg.det(rotations)
    rigid = (
        finite
        & homogeneous
        & (orthogonality_error <= config.orthogonality_tolerance)
        & (np.abs(determinants - 1.0) <= config.determinant_tolerance)
    )
    candidate_mask = rigid & (scores >= config.minimum_confidence)

    translations = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    translation_rates = translations / position_steps
    rotations_deg = np.asarray(
        [_rotation_step_degrees(rotations[index - 1], rotations[index]) for index in range(1, len(poses))],
        dtype=np.float64,
    )
    rotation_rates = rotations_deg / position_steps
    finite_steps = translation_rates[
        np.isfinite(translation_rates) & (translation_rates > 1e-9)
    ]
    translation_limit: float | None = config.maximum_translation_step
    if config.adaptive_translation_limit and len(finite_steps) >= 2:
        median = float(np.median(finite_steps))
        robust_limit = median + config.robust_sigma_multiplier * 1.4826 * _mad(finite_steps)
        adaptive_limit = max(median * config.translation_median_multiplier, robust_limit, 1e-6)
        translation_limit = adaptive_limit if translation_limit is None else min(translation_limit, adaptive_limit)

    frame_mask = np.zeros(len(poses), dtype=bool)
    rejected_translation = 0
    rejected_rotation = 0
    trusted_index: int | None = None
    for index in range(len(poses)):
        if not candidate_mask[index]:
            continue
        if trusted_index is None:
            frame_mask[index] = True
            trusted_index = index
            continue
        distance = float(
            np.linalg.norm(poses[index, :3, 3] - poses[trusted_index, :3, 3])
        )
        angle = _rotation_step_degrees(rotations[trusted_index], rotations[index])
        sample_delta = float(positions[index] - positions[trusted_index])
        distance_rate = distance / sample_delta
        angle_rate = angle / sample_delta
        translation_ok = translation_limit is None or (
            np.isfinite(distance_rate) and distance_rate <= translation_limit
        )
        rotation_ok = (
            np.isfinite(angle_rate)
            and angle_rate <= config.maximum_rotation_step_degrees
        )
        if translation_ok and rotation_ok:
            frame_mask[index] = True
            trusted_index = index
        else:
            rejected_translation += int(not translation_ok)
            rejected_rotation += int(not rotation_ok)

    finite_scores = scores[np.isfinite(scores)]
    mean_confidence = (
        float(np.mean(np.clip(finite_scores, 0.0, 1.0)))
        if len(finite_scores)
        else 0.0
    )
    confidence_risk = 1.0 - mean_confidence
    rotation_risk = _finite_max(rotation_rates) / max(
        config.maximum_rotation_step_degrees, 1e-9
    )
    translation_risk = (
        _finite_max(translation_rates) / translation_limit
        if translation_limit is not None and translation_limit > 0.0
        else 0.0
    )
    invalid_ratio = 1.0 - float(np.mean(frame_mask))
    drift_risk = float(np.clip(max(confidence_risk, invalid_ratio, rotation_risk - 1.0, translation_risk - 1.0), 0.0, 1.0))
    reasons: list[str] = []
    if not np.all(rigid):
        reasons.append("one or more camera transforms are not finite rigid SE(3) poses")
    if np.any(scores < config.minimum_confidence):
        reasons.append("one or more camera proposals are below the confidence gate")
    if rejected_translation:
        reasons.append("one or more camera translations break robust trajectory continuity")
    if rejected_rotation:
        reasons.append("one or more camera rotations break trajectory continuity")
    metrics = {
        "frameCount": int(len(poses)),
        "acceptedFrameCount": int(np.count_nonzero(frame_mask)),
        "rejectedFrameCount": int(len(poses) - np.count_nonzero(frame_mask)),
        "maximumTranslationStep": _finite_max(translations),
        "maximumTranslationPerSample": _finite_max(translation_rates),
        "translationLimit": translation_limit,
        "maximumRotationStepDegrees": _finite_max(rotations_deg),
        "maximumRotationPerSampleDegrees": _finite_max(rotation_rates),
        "meanConfidence": mean_confidence,
        "driftRisk": drift_risk,
    }
    return CameraValidationResult(bool(np.all(frame_mask)), frame_mask, drift_risk, translation_limit, metrics, tuple(reasons))


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        return self.scale * (values @ self.rotation.T) + self.translation


@dataclass(frozen=True)
class ScaleValidationConfig:
    minimum_correspondences: int = 6
    minimum_inlier_ratio: float = 0.65
    maximum_relative_rmse: float = 0.05
    robust_sigma_multiplier: float = 3.5


@dataclass(frozen=True)
class ScaleValidationResult:
    accepted: bool
    status: str
    transform: SimilarityTransform | None
    inlier_mask: np.ndarray
    metrics: dict[str, Any]
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        report = _report("scale", self.accepted, self.metrics, list(self.reasons))
        report["status"] = self.status
        return report


def _fit_similarity(source: np.ndarray, target: np.ndarray) -> SimilarityTransform | None:
    if len(source) < 3:
        return None
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if variance <= 1e-12:
        return None
    covariance = (target_centered.T @ source_centered) / len(source)
    left, singular, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_t) < 0.0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_t
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    translation = target_mean - scale * (rotation @ source_mean)
    return SimilarityTransform(scale, rotation, translation)


def validate_scale(
    source_points: np.ndarray,
    metric_points: np.ndarray,
    config: ScaleValidationConfig | None = None,
    *,
    accepted_status: str = "MODEL_METRIC_VALIDATED",
    rejected_status: str = "MODEL_METRIC_UNVERIFIED",
) -> ScaleValidationResult:
    config = config or ScaleValidationConfig()
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(metric_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1:] != (3,):
        raise ValueError("scale correspondences must be matching Nx3 arrays")
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[finite]
    target = target[finite]
    original_indices = np.flatnonzero(finite)
    final_mask = np.zeros(len(finite), dtype=bool)
    if len(source) < config.minimum_correspondences:
        return ScaleValidationResult(
            False,
            rejected_status,
            None,
            final_mask,
            {"correspondenceCount": int(len(source))},
            ("too few finite metric correspondences to validate scale",),
        )
    inliers = np.ones(len(source), dtype=bool)
    transform: SimilarityTransform | None = None
    residual = np.full(len(source), np.inf)
    for _ in range(4):
        transform = _fit_similarity(source[inliers], target[inliers])
        if transform is None:
            break
        residual = np.linalg.norm(transform.apply(source) - target, axis=1)
        median = float(np.median(residual))
        sigma = 1.4826 * _mad(residual)
        scene_scale = max(float(np.median(np.linalg.norm(target - np.median(target, axis=0), axis=1))), 1e-6)
        threshold = max(median + config.robust_sigma_multiplier * sigma, scene_scale * 0.005, 1e-5)
        updated = residual <= threshold
        if np.array_equal(updated, inliers) or np.count_nonzero(updated) < 3:
            break
        inliers = updated
    if transform is None:
        return ScaleValidationResult(False, rejected_status, None, final_mask, {"correspondenceCount": int(len(source))}, ("metric similarity fit is degenerate",))
    final_mask[original_indices[inliers]] = True
    scene_scale = max(float(np.median(np.linalg.norm(target - np.median(target, axis=0), axis=1))), 1e-6)
    rmse = float(np.sqrt(np.mean(np.square(residual[inliers])))) if np.any(inliers) else float("inf")
    relative_rmse = rmse / scene_scale
    inlier_ratio = float(np.mean(inliers))
    accepted = bool(inlier_ratio >= config.minimum_inlier_ratio and relative_rmse <= config.maximum_relative_rmse)
    reasons = () if accepted else ("metric anchors do not support a stable similarity scale",)
    metrics = {
        "correspondenceCount": int(len(source)),
        "inlierCount": int(np.count_nonzero(inliers)),
        "inlierRatio": inlier_ratio,
        "scale": transform.scale,
        "rmse": rmse,
        "relativeRmse": relative_rmse,
    }
    return ScaleValidationResult(accepted, accepted_status if accepted else rejected_status, transform, final_mask, metrics, reasons)


@dataclass(frozen=True)
class DepthValidationConfig:
    minimum_samples: int = 256
    maximum_scale_bias: float = 0.04
    minimum_inlier_ratio: float = 0.65
    absolute_inlier_tolerance: float = 0.04
    relative_inlier_tolerance: float = 0.025
    maximum_median_relative_residual: float = 0.015
    maximum_p90_relative_residual: float = 0.05


@dataclass(frozen=True)
class DepthValidationResult:
    accepted: bool
    comparison_mask: np.ndarray
    metrics: dict[str, Any]
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _report("depth", self.accepted, self.metrics, list(self.reasons))


def validate_depth(
    measured: np.ndarray,
    proposed: np.ndarray,
    measured_mask: np.ndarray | None = None,
    proposed_mask: np.ndarray | None = None,
    config: DepthValidationConfig | None = None,
) -> DepthValidationResult:
    config = config or DepthValidationConfig()
    reference = np.asarray(measured, dtype=np.float64)
    candidate = np.asarray(proposed, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError("measured and proposed depth must have the same shape")
    comparison = np.isfinite(reference) & np.isfinite(candidate) & (reference > 0.0) & (candidate > 0.0)
    if measured_mask is not None:
        comparison &= np.asarray(measured_mask, dtype=bool)
    if proposed_mask is not None:
        comparison &= np.asarray(proposed_mask, dtype=bool)
    sample_count = int(np.count_nonzero(comparison))
    if sample_count < config.minimum_samples:
        return DepthValidationResult(False, comparison, {"sampleCount": sample_count}, ("too few overlapping depth samples",))
    reference_values = reference[comparison]
    candidate_values = candidate[comparison]
    if sample_count > 100_000:
        indices = np.linspace(0, sample_count - 1, 100_000, dtype=np.int64)
        reference_values = reference_values[indices]
        candidate_values = candidate_values[indices]
    residual = np.abs(candidate_values - reference_values)
    scene_depth = float(np.median(reference_values))
    median_residual = float(np.median(residual))
    p90_residual = float(np.percentile(residual, 90))
    scale_bias = float(abs(np.median(candidate_values / np.maximum(reference_values, 1e-9)) - 1.0))
    tolerance = np.maximum(config.absolute_inlier_tolerance, reference_values * config.relative_inlier_tolerance)
    inlier_ratio = float(np.mean(residual <= tolerance))
    accepted = bool(
        median_residual <= max(0.03, scene_depth * config.maximum_median_relative_residual)
        and p90_residual <= max(0.10, scene_depth * config.maximum_p90_relative_residual)
        and scale_bias <= config.maximum_scale_bias
        and inlier_ratio >= config.minimum_inlier_ratio
    )
    reasons = () if accepted else ("proposed depth changes measured scale or geometry beyond the gate",)
    metrics = {
        "sampleCount": sample_count,
        "sceneDepth": scene_depth,
        "medianResidual": median_residual,
        "p90Residual": p90_residual,
        "scaleBias": scale_bias,
        "inlierRatio": inlier_ratio,
    }
    return DepthValidationResult(accepted, comparison, metrics, reasons)


@dataclass(frozen=True)
class RayConsistencyResult:
    support_mask: np.ndarray
    free_space_violation_mask: np.ndarray
    occluded_mask: np.ndarray
    unknown_mask: np.ndarray

    @property
    def accepted_mask(self) -> np.ndarray:
        return self.support_mask | self.occluded_mask | self.unknown_mask

    def to_dict(self) -> dict[str, Any]:
        return _report(
            "free-space",
            not bool(np.any(self.free_space_violation_mask)),
            {
                "sampleCount": int(self.support_mask.size),
                "supportCount": int(np.count_nonzero(self.support_mask)),
                "freeSpaceViolationCount": int(np.count_nonzero(self.free_space_violation_mask)),
                "occludedCount": int(np.count_nonzero(self.occluded_mask)),
                "unknownCount": int(np.count_nonzero(self.unknown_mask)),
            },
            [] if not np.any(self.free_space_violation_mask) else ["proposed geometry occupies observed free space"],
        )


def validate_ray_depths(
    proposed_depth: np.ndarray,
    observed_depth: np.ndarray,
    observed_mask: np.ndarray | None = None,
    *,
    absolute_tolerance: float = 0.035,
    relative_tolerance: float = 0.02,
) -> RayConsistencyResult:
    proposed = np.asarray(proposed_depth, dtype=np.float64)
    observed = np.asarray(observed_depth, dtype=np.float64)
    if proposed.shape != observed.shape:
        raise ValueError("proposed and observed ray depths must have the same shape")
    known = np.isfinite(observed) & (observed > 0.0) & np.isfinite(proposed) & (proposed > 0.0)
    if observed_mask is not None:
        known &= np.asarray(observed_mask, dtype=bool)
    tolerance = np.maximum(absolute_tolerance, proposed * relative_tolerance)
    delta = proposed - observed
    support = known & (np.abs(delta) <= tolerance)
    free_space = known & (delta < -tolerance)
    occluded = known & (delta > tolerance)
    unknown = ~known
    return RayConsistencyResult(support, free_space, occluded, unknown)


@dataclass(frozen=True)
class GeometryValidationConfig:
    minimum_confidence: float = 0.0
    maximum_absolute_coordinate: float = 1e6
    minimum_points: int = 1


@dataclass(frozen=True)
class GeometryValidationResult:
    accepted: bool
    point_mask: np.ndarray
    metrics: dict[str, Any]
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _report("geometry", self.accepted, self.metrics, list(self.reasons))


def validate_geometry(
    points: np.ndarray,
    confidence: np.ndarray | None = None,
    *,
    free_space_violation_mask: np.ndarray | None = None,
    observation_count: np.ndarray | None = None,
    config: GeometryValidationConfig | None = None,
) -> GeometryValidationResult:
    config = config or GeometryValidationConfig()
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (3,):
        raise ValueError("geometry points must be an Nx3 array")
    mask = np.isfinite(values).all(axis=1) & (np.max(np.abs(values), axis=1, initial=0.0) <= config.maximum_absolute_coordinate)
    if confidence is not None:
        scores = np.asarray(confidence, dtype=np.float64)
        if scores.shape != (len(values),):
            raise ValueError("geometry confidence must contain one value per point")
        mask &= np.isfinite(scores) & (scores >= config.minimum_confidence)
    if free_space_violation_mask is not None:
        violations = np.asarray(free_space_violation_mask, dtype=bool)
        if violations.shape != (len(values),):
            raise ValueError("free-space mask must contain one value per point")
        mask &= ~violations
    if observation_count is not None:
        observations = np.asarray(observation_count)
        if observations.shape != (len(values),):
            raise ValueError("observation count must contain one value per point")
        mask &= observations >= 1
    accepted_count = int(np.count_nonzero(mask))
    accepted = accepted_count >= config.minimum_points
    reasons = () if accepted else ("no admissible geometry remains after validation",)
    metrics = {
        "pointCount": int(len(values)),
        "acceptedPointCount": accepted_count,
        "rejectedPointCount": int(len(values) - accepted_count),
        "acceptedRatio": float(accepted_count / max(1, len(values))),
    }
    return GeometryValidationResult(accepted, mask, metrics, reasons)
