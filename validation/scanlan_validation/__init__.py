"""Fail-closed validation shared by ScanLan reconstruction backends."""

from .engine import (
    VALIDATION_CONTRACT_VERSION,
    CameraValidationConfig,
    CameraValidationResult,
    DepthValidationConfig,
    DepthValidationResult,
    GeometryValidationConfig,
    GeometryValidationResult,
    RayConsistencyResult,
    ScaleValidationConfig,
    ScaleValidationResult,
    SimilarityTransform,
    validate_camera_trajectory,
    validate_depth,
    validate_geometry,
    validate_ray_depths,
    validate_scale,
)

__all__ = [
    "VALIDATION_CONTRACT_VERSION",
    "CameraValidationConfig",
    "CameraValidationResult",
    "DepthValidationConfig",
    "DepthValidationResult",
    "GeometryValidationConfig",
    "GeometryValidationResult",
    "RayConsistencyResult",
    "ScaleValidationConfig",
    "ScaleValidationResult",
    "SimilarityTransform",
    "validate_camera_trajectory",
    "validate_depth",
    "validate_geometry",
    "validate_ray_depths",
    "validate_scale",
]
