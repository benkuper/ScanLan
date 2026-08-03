from __future__ import annotations

from typing import Any


POSE_TRANSLATION_LIMIT_M = 0.05
POSE_ROTATION_PARAMETER_LIMIT = 0.12


def rotation_6d_to_matrix(values: Any) -> Any:
    """Convert the continuous 6D rotation representation to rotation matrices."""
    import torch
    import torch.nn.functional as functional

    first = values[..., :3]
    second = values[..., 3:]
    basis_x = functional.normalize(first, dim=-1, eps=1e-8)
    basis_y = functional.normalize(
        second - torch.sum(basis_x * second, dim=-1, keepdim=True) * basis_x,
        dim=-1,
        eps=1e-8,
    )
    basis_z = torch.cross(basis_x, basis_y, dim=-1)
    return torch.stack((basis_x, basis_y, basis_z), dim=-2)


def pose_delta_matrix(offsets: Any) -> Any:
    """Build local camera-to-world corrections from translation + 6D rotation."""
    import torch

    identity_rotation = offsets.new_tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    )
    rotation = rotation_6d_to_matrix(offsets[..., 3:] + identity_rotation)
    transform = torch.eye(4, dtype=offsets.dtype, device=offsets.device).expand(
        offsets.shape[:-1] + (4, 4)
    ).clone()
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = offsets[..., :3]
    return transform


def pose_regularization(offsets: Any, neighbor_pairs: Any) -> Any:
    """Keep RGB-D pose corrections small and temporally smooth within phases."""
    translation = offsets[..., :3] / POSE_TRANSLATION_LIMIT_M
    rotation = offsets[..., 3:] / POSE_ROTATION_PARAMETER_LIMIT
    prior = translation.square().mean() + rotation.square().mean()
    if neighbor_pairs.numel() == 0:
        return prior
    differences = offsets[neighbor_pairs[:, 1]] - offsets[neighbor_pairs[:, 0]]
    smooth_translation = differences[..., :3] / POSE_TRANSLATION_LIMIT_M
    smooth_rotation = differences[..., 3:] / POSE_ROTATION_PARAMETER_LIMIT
    smoothness = smooth_translation.square().mean() + smooth_rotation.square().mean()
    return prior + 2.0 * smoothness


def constrain_pose_offsets_(offsets: Any) -> None:
    """Project corrections into conservative metric bounds and fix the gauge."""
    import torch

    with torch.no_grad():
        translations = offsets[:, :3]
        norms = torch.linalg.vector_norm(translations, dim=-1, keepdim=True)
        translations.mul_(
            torch.clamp(POSE_TRANSLATION_LIMIT_M / norms.clamp_min(1e-12), max=1.0)
        )
        offsets[:, 3:].clamp_(
            -POSE_ROTATION_PARAMETER_LIMIT,
            POSE_ROTATION_PARAMETER_LIMIT,
        )
        offsets[0].zero_()


def pose_correction_statistics(offsets: Any) -> dict[str, float]:
    """Summarize the actual rigid corrections represented by pose offsets."""
    import torch

    with torch.no_grad():
        transforms = pose_delta_matrix(offsets)
        translations = torch.linalg.vector_norm(transforms[..., :3, 3], dim=-1)
        traces = torch.diagonal(transforms[..., :3, :3], dim1=-2, dim2=-1).sum(-1)
        angles = torch.acos(torch.clamp((traces - 1.0) * 0.5, -1.0, 1.0))
        return {
            "maximumTranslationM": float(translations.max().cpu()),
            "meanTranslationM": float(translations.mean().cpu()),
            "maximumRotationDegrees": float(torch.rad2deg(angles).max().cpu()),
            "meanRotationDegrees": float(torch.rad2deg(angles).mean().cpu()),
            "translationLimitM": POSE_TRANSLATION_LIMIT_M,
            "rotationParameterLimit": POSE_ROTATION_PARAMETER_LIMIT,
        }
