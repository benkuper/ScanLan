from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from scanlan.dataset import MAX_CANONICAL_FRAMES, _select_training_frames


class DatasetSelectionTests(unittest.TestCase):
    def test_view_selection_is_bounded_deterministic_and_keeps_take_boundaries(self) -> None:
        frames = []
        for index in range(650):
            pose = np.eye(4)
            pose[0, 3] = index * 0.01
            frames.append(
                SimpleNamespace(
                    phase_id="first" if index < 325 else "second",
                    camera_to_global=pose,
                )
            )

        selected = _select_training_frames(frames)
        repeated = _select_training_frames(frames)

        self.assertEqual(len(selected), MAX_CANONICAL_FRAMES)
        self.assertIs(selected[0], frames[0])
        self.assertIs(selected[-1], frames[-1])
        self.assertTrue(any(frame is frames[324] for frame in selected))
        self.assertTrue(any(frame is frames[325] for frame in selected))
        self.assertEqual(
            [id(frame) for frame in selected],
            [id(frame) for frame in repeated],
        )


if __name__ == "__main__":
    unittest.main()
