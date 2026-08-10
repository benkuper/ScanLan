from __future__ import annotations

import unittest

import numpy as np

from scanlan_splat.neural_sdf import validate_candidate


class NeuralSdfValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.triangles = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)

    def test_small_topology_preserving_refinement_is_accepted(self) -> None:
        candidate = self.vertices.copy()
        candidate[:, 2] = np.asarray([0.001, -0.001, 0.001, -0.001])
        report = validate_candidate(self.vertices, self.triangles, candidate, 0.01)
        self.assertTrue(report["accepted"])
        self.assertLess(report["p95DisplacementM"], 0.01)

    def test_excessive_displacement_is_rejected(self) -> None:
        candidate = self.vertices.copy()
        candidate[0, 2] = 0.05
        report = validate_candidate(self.vertices, self.triangles, candidate, 0.01)
        self.assertFalse(report["accepted"])
        self.assertGreater(report["maximumDisplacementM"], 0.02)

    def test_flipped_surface_is_rejected(self) -> None:
        candidate = self.vertices.copy()
        candidate[[0, 1]] = candidate[[1, 0]]
        report = validate_candidate(self.vertices, self.triangles, candidate, 1.0)
        self.assertFalse(report["accepted"])
        self.assertGreater(report["flippedTriangleFraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
