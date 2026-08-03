from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scanlan_splat.train import _read_seed_parameters


def _write_initialization(path: Path) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "element vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    vertex = np.asarray(
        [(1.0, 2.0, 3.0, 255, 128, 0)],
        dtype=np.dtype(
            [
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ]
        ),
    )
    with path.open("wb") as handle:
        handle.write(header)
        vertex.tofile(handle)


class SeedParameterTests(unittest.TestCase):
    def test_rgbd_sidecar_supplies_surface_scales_and_rotations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_initialization(root / "initialization.ply")
            np.savez(
                root / "initialization-2dgs.npz",
                points=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
                colors=np.asarray([[255, 128, 0]], dtype=np.uint8),
                scales=np.asarray([[0.02, 0.03, 0.001]], dtype=np.float32),
                quaternions=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            )

            points, colors, scales, quaternions = _read_seed_parameters(
                root,
                {
                    "initialization": "initialization.ply",
                    "initializationParameters": "initialization-2dgs.npz",
                },
            )

            self.assertTrue(np.allclose(points, [[1.0, 2.0, 3.0]]))
            self.assertTrue(np.allclose(colors, [[1.0, 128.0 / 255.0, 0.0]]))
            self.assertTrue(np.allclose(scales, [[0.02, 0.03, 0.001]]))
            self.assertTrue(np.allclose(quaternions, [[1.0, 0.0, 0.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
