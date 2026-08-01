from __future__ import annotations

import math

import numpy as np

from .io import ImuSample


def _rotation_vector_matrix(rotation_vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotation_vector / angle
    x, y, z = axis
    cross = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def odometry_rotation_prior(
    samples: list[ImuSample],
    start_us: int,
    end_us: int,
) -> np.ndarray | None:
    """Return a source-camera to target-camera rotation seeded by gyroscope data."""
    if end_us <= start_us:
        return None
    gyro = [
        sample
        for sample in samples
        if sample.kind == "gyro" and start_us <= sample.timestamp_us <= end_us
    ]
    duration = (end_us - start_us) / 1_000_000.0
    if len(gyro) < 2 or duration > 0.5:
        return None
    if gyro[0].timestamp_us - start_us > 80_000 or end_us - gyro[-1].timestamp_us > 80_000:
        return None

    orientation_change = np.eye(3, dtype=np.float64)
    previous_time = start_us
    previous_rate = gyro[0].value
    for sample in gyro:
        step = (sample.timestamp_us - previous_time) / 1_000_000.0
        if step < 0.0 or step > 0.08:
            return None
        mean_rate = 0.5 * (previous_rate + sample.value)
        orientation_change = orientation_change @ _rotation_vector_matrix(mean_rate * step)
        previous_time = sample.timestamp_us
        previous_rate = sample.value
    final_step = (end_us - previous_time) / 1_000_000.0
    if final_step < 0.0 or final_step > 0.08:
        return None
    orientation_change = orientation_change @ _rotation_vector_matrix(previous_rate * final_step)

    angle = math.acos(np.clip((np.trace(orientation_change) - 1.0) * 0.5, -1.0, 1.0))
    if not np.isfinite(orientation_change).all() or angle > math.radians(65.0):
        return None

    prior = np.eye(4, dtype=np.float64)
    # Integrated body rotation is previous-camera to current-camera orientation.
    # Open3D expects the point transform from the previous frame into the current frame.
    prior[:3, :3] = orientation_change.T
    return prior
