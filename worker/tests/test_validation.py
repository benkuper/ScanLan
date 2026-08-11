from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scanlan.validation import validate_posed_frames
from scanlan.numpy_engine import reconstruct_known_poses


class ProductionValidationTests(unittest.TestCase):
    @staticmethod
    def _frame(phase: str, x: float, frame_index: int):
        pose = np.eye(4)
        pose[0, 3] = x
        return SimpleNamespace(
            phase_id=phase,
            phase_name=f"Phase {phase}",
            camera_to_global=pose,
            frame_index=frame_index,
        )

    def test_camera_continuity_is_validated_within_each_phase(self) -> None:
        frames = [
            self._frame("a", 0.0, 0),
            self._frame("a", 0.1, 1),
            self._frame("a", 0.2, 2),
            self._frame("a", 9.0, 3),
            self._frame("a", 0.4, 4),
            self._frame("b", 100.0, 0),
            self._frame("b", 100.1, 1),
        ]
        accepted, report = validate_posed_frames(frames)
        self.assertEqual(len(accepted), 6)
        self.assertTrue(report["accepted"])
        self.assertFalse(report["allInputAccepted"])
        self.assertEqual(report["rejectedFrameCount"], 1)
        self.assertEqual(report["scaleStatus"], "SENSOR_METRIC")

    def test_numpy_fusion_never_consumes_a_rejected_camera(self) -> None:
        phase = SimpleNamespace(
            root=Path("phase-a"),
            frames=[object(), object(), object()],
            manifest={"name": "Phase A"},
        )
        accepted = {(str(phase.root), 0), (str(phase.root), 2)}
        visited: list[int] = []

        def fake_depth_to_world_points(_phase, frame_index, **_kwargs):
            visited.append(frame_index)
            return (
                np.asarray([[float(frame_index), 0.0, 0.0]], dtype=np.float32),
                np.asarray([[frame_index, 0, 0]], dtype=np.uint8),
            )

        with patch(
            "scanlan.numpy_engine.depth_to_world_points",
            side_effect=fake_depth_to_world_points,
        ):
            reconstruct_known_poses(
                [phase],
                0.001,
                accepted_frame_keys=accepted,
            )

        self.assertEqual(visited, [0, 2])


if __name__ == "__main__":
    unittest.main()
