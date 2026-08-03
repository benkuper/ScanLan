from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scanlan_splat.export import export_3dgs_ply


class TwoDimensionalGaussianExportTests(unittest.TestCase):
    def test_export_preserves_degree_three_spherical_harmonics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "room-splat.ply"
            sh = np.arange(48, dtype=np.float32).reshape(1, 16, 3) / 100.0
            export_3dgs_ply(
                path,
                np.zeros((1, 3), dtype=np.float32),
                np.zeros((1, 3), dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                np.zeros((1, 3), dtype=np.float32),
                np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                sh_coefficients=sh,
            )

            with path.open("rb") as handle:
                properties: list[str] = []
                while True:
                    line = handle.readline().decode("ascii").strip()
                    if line.startswith("property float "):
                        properties.append(line.split()[-1])
                    if line == "end_header":
                        break
                vertex = np.fromfile(
                    handle,
                    dtype=np.dtype([(name, "<f4") for name in properties]),
                    count=1,
                )[0]

            self.assertAlmostEqual(float(vertex["f_dc_0"]), float(sh[0, 0, 0]))
            self.assertAlmostEqual(float(vertex["f_dc_2"]), float(sh[0, 0, 2]))
            self.assertAlmostEqual(float(vertex["f_rest_0"]), float(sh[0, 1, 0]))
            self.assertAlmostEqual(float(vertex["f_rest_14"]), float(sh[0, 15, 0]))
            self.assertAlmostEqual(float(vertex["f_rest_15"]), float(sh[0, 1, 1]))
            self.assertAlmostEqual(float(vertex["f_rest_44"]), float(sh[0, 15, 2]))


if __name__ == "__main__":
    unittest.main()
