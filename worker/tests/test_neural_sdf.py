from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scanlan.neural_sdf import refine_surface_with_worker


class NeuralSdfBridgeTests(unittest.TestCase):
    def test_camera_validation_gate_skips_worker_and_preserves_surface(self) -> None:
        vertices = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        triangles = np.asarray([[0, 1, 2]], dtype=np.int64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            refined, refined_triangles, report = refine_surface_with_worker(
                vertices,
                triangles,
                project_root=root,
                worker=root / "worker-that-must-not-run.exe",
                voxel_size_m=0.01,
                validation_report={"accepted": False, "reason": "unsafe camera path"},
            )
            np.testing.assert_array_equal(refined, vertices)
            np.testing.assert_array_equal(refined_triangles, triangles)
            self.assertEqual(report["status"], "skipped")
            persisted = json.loads(
                (root / "outputs" / "neural-sdf-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
