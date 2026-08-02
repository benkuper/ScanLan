from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scanlan_splat.export import export_3dgs_ply, write_splat_sidecars


class ExportTests(unittest.TestCase):
    def test_canonical_ply_contains_required_3dgs_properties(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "room-splat.ply"
            export_3dgs_ply(
                path,
                np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
                np.asarray([[0.25, 0.5, 0.75]], dtype=np.float32),
                np.asarray([0.0], dtype=np.float32),
                np.asarray([[-4.0, -4.0, -4.0]], dtype=np.float32),
                np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            )
            header = path.read_bytes().split(b"end_header\n", 1)[0].decode("ascii")
            for name in ("f_dc_0", "f_rest_44", "opacity", "scale_2", "rot_3"):
                self.assertIn(f"property float {name}", header)
            write_splat_sidecars(
                root,
                "fixture",
                True,
                {"gsplat": "test"},
                {"usesDepth": True, "iterations": 1},
            )
            self.assertTrue((root / "room-splat.transform.json").is_file())
            self.assertTrue((root / "splat-manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
