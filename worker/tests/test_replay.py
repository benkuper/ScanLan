from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scanlan.io import read_project
from scanlan.mock_data import create_mock_project
from scanlan.replay import _rotation_quaternion_xyzw, replay_archive
from scanlan.stream import read_rgbd_frame


class ReplayTests(unittest.TestCase):
    def test_archive_replay_preserves_source_sequence_and_pose(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = create_mock_project(Path(temporary) / "scan", phase_count=1, frame_count=3)
            project = read_project(root)
            capture = root / "phases" / project["phases"][0]["id"]
            output = io.BytesIO()

            result = replay_archive(capture, output)
            output.seek(0)
            frames = [read_rgbd_frame(output) for _ in range(3)]

        self.assertEqual(result["frameCount"], 3)
        self.assertEqual([frame.sequence for frame in frames], [0, 1, 2])
        self.assertTrue(all(frame.camera_to_world is not None for frame in frames))
        self.assertEqual(output.read(), b"")

    def test_rotation_quaternion_round_trip_for_quarter_turn(self) -> None:
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        quaternion = _rotation_quaternion_xyzw(rotation)

        np.testing.assert_allclose(
            np.abs(quaternion),
            [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
