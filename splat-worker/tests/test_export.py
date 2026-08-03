from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scanlan_splat.export import export_3dgs_ply, export_splat_preview, write_splat_sidecars


class ExportTests(unittest.TestCase):
    def test_compact_preview_encodes_one_32_byte_splat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "room-splat.preview.splat"
            export_splat_preview(
                path,
                np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
                np.asarray([[0.25, 0.5, 0.75]], dtype=np.float32),
                np.asarray([0.0], dtype=np.float32),
                np.log(np.asarray([[0.1, 0.2, 0.4]], dtype=np.float32)),
                np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            )
            payload = path.read_bytes()
            self.assertEqual(len(payload), 32)
            self.assertTrue(np.allclose(np.frombuffer(payload[:24], dtype="<f4"), [1, 2, 3, 0.1, 0.2, 0.4]))
            self.assertEqual(list(payload[24:28]), [64, 128, 191, 128])
            self.assertEqual(list(payload[28:32]), [255, 128, 128, 128])

    def test_live_preview_can_bound_the_published_splat_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "room-splat.preview.splat"
            means = np.asarray([[float(index), 0.0, 0.0] for index in range(5)], dtype=np.float32)
            export_splat_preview(
                path,
                means,
                np.ones((5, 3), dtype=np.float32),
                np.zeros(5, dtype=np.float32),
                np.zeros((5, 3), dtype=np.float32),
                np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (5, 1)),
                limit=2,
            )
            payload = path.read_bytes()
            self.assertEqual(len(payload), 64)
            self.assertEqual(np.frombuffer(payload[:12], dtype="<f4")[0], 0.0)
            self.assertEqual(np.frombuffer(payload[32:44], dtype="<f4")[0], 4.0)
            self.assertEqual(list(Path(temporary).glob("*.tmp")), [])

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
