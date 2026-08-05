from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .imu import odometry_rotation_prior
from .io import load_color, load_depth, read_phase
from .stream import RgbdFrame, StreamCamera, encode_rgbd_frame


def _rotation_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.asarray(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        diagonal = int(np.argmax(np.diag(matrix)))
        if diagonal == 0:
            scale = 2.0 * np.sqrt(max(1e-12, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]))
            quaternion = np.asarray(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif diagonal == 1:
            scale = 2.0 * np.sqrt(max(1e-12, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]))
            quaternion = np.asarray(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(max(1e-12, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]))
            quaternion = np.asarray(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    return quaternion / max(float(np.linalg.norm(quaternion)), 1e-12)


def archive_frames(root: Path) -> Iterator[RgbdFrame]:
    phase = read_phase(root, include_tracking_rejected=True)
    sensor_kind = str((phase.manifest.get("sensor") or {}).get("kind", ""))
    camera = StreamCamera(
        phase.camera.width,
        phase.camera.height,
        phase.camera.fx,
        phase.camera.fy,
        phase.camera.cx,
        phase.camera.cy,
        phase.camera.depth_scale,
        0.5 if sensor_kind == "kinect_v2" else 0.25,
        phase.camera.max_depth_m,
    )
    for position, record in enumerate(phase.frames):
        gyro = None
        if position:
            prior = odometry_rotation_prior(
                phase.imu_samples,
                phase.frames[position - 1].timestamp_us,
                record.timestamp_us,
            )
            if prior is not None:
                gyro = _rotation_quaternion_xyzw(prior[:3, :3])
        yield RgbdFrame(
            sequence=record.source_sequence,
            depth_timestamp_us=record.timestamp_us,
            color_timestamp_us=record.rgb_timestamp_us or record.timestamp_us,
            camera=camera,
            depth=load_depth(record, phase.camera),
            color=load_color(record, phase.camera),
            gyro_delta_xyzw=gyro,
            camera_to_world=record.pose,
            mirror_x=sensor_kind == "kinect_v2",
        )


def replay_archive(root: Path, output: BinaryIO) -> dict[str, int]:
    frame_count = 0
    first_sequence: int | None = None
    last_sequence: int | None = None
    for frame in archive_frames(root):
        output.write(encode_rgbd_frame(frame))
        if first_sequence is None:
            first_sequence = frame.sequence
        last_sequence = frame.sequence
        frame_count += 1
    output.flush()
    return {
        "frameCount": frame_count,
        "firstSequence": first_sequence or 0,
        "lastSequence": last_sequence or 0,
    }
