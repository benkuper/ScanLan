from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
from scanlan_validation import (
    VALIDATION_CONTRACT_VERSION,
    CameraValidationConfig,
    validate_camera_trajectory,
)

if TYPE_CHECKING:
    from .mesh import PosedFrame


def validate_posed_frames(frames: list[PosedFrame]) -> tuple[list[PosedFrame], dict[str, Any]]:
    """Reject unsafe production camera poses without crossing phase boundaries."""
    grouped: dict[str, list[tuple[int, PosedFrame]]] = defaultdict(list)
    for index, frame in enumerate(frames):
        grouped[frame.phase_id].append((index, frame))

    accepted_indices: set[int] = set()
    phase_reports: list[dict[str, Any]] = []
    for phase_id, entries in grouped.items():
        poses = np.asarray([frame.camera_to_global for _, frame in entries], dtype=np.float64)
        sample_positions = np.asarray(
            [getattr(frame, "frame_index", order) for order, (_, frame) in enumerate(entries)],
            dtype=np.float64,
        )
        result = validate_camera_trajectory(
            poses,
            config=CameraValidationConfig(
                maximum_translation_step=0.5,
                adaptive_translation_limit=False,
            ),
            sample_positions=sample_positions,
        )
        accepted_indices.update(
            original_index
            for (original_index, _), accepted in zip(entries, result.frame_mask, strict=True)
            if accepted
        )
        phase_reports.append(
            {
                "phaseId": phase_id,
                "phaseName": entries[0][1].phase_name,
                **result.to_dict(),
            }
        )

    accepted = [frame for index, frame in enumerate(frames) if index in accepted_indices]
    if frames and not accepted:
        raise RuntimeError("Camera validation rejected every production frame")
    report = {
        "contractVersion": VALIDATION_CONTRACT_VERSION,
        "accepted": not frames or bool(accepted),
        "allInputAccepted": len(accepted) == len(frames),
        "scaleStatus": "SENSOR_METRIC",
        "inputFrameCount": len(frames),
        "acceptedFrameCount": len(accepted),
        "rejectedFrameCount": len(frames) - len(accepted),
        "phases": phase_reports,
    }
    return accepted, report
