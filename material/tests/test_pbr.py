from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scanlan_material.analysis import FusedMaterialSurface
from scanlan_material.contracts import MATERIAL_CLASSES, OPTICAL_RISKS
from scanlan_material.pbr import PBR_CONTRACT_VERSION, build_pbr_artifacts


def _surface(confidence: float = 1.0) -> FusedMaterialSurface:
    count = 4
    valid = np.full(count, confidence > 0.0, dtype=bool)
    classes = np.zeros((count, len(MATERIAL_CLASSES)), dtype=np.float32)
    classes[:, MATERIAL_CLASSES.index("opaque_dielectric")] = 1.0
    return FusedMaterialSurface(
        class_probabilities=classes,
        optical_risk_probabilities=np.zeros((count, len(OPTICAL_RISKS)), dtype=np.float32),
        valid_mask=valid,
        confidence=np.full(count, confidence, dtype=np.float32),
        support_count=np.full(count, 2 if confidence else 0, dtype=np.uint16),
        effective_view_count=np.full(count, 2.0 if confidence else 0.0, dtype=np.float32),
        region_ids=np.zeros(count, dtype=np.int32),
        albedo_linear=np.full((count, 3), [0.25, 0.5, 0.75], dtype=np.float32),
        roughness=np.full(count, 0.25, dtype=np.float32),
        metallic=np.full(count, 0.75, dtype=np.float32),
        transmission=np.full(count, 0.5, dtype=np.float32),
        normal_world=np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (count, 1)),
        emission_linear=np.tile(np.asarray([[2.0, 0.0, 0.0]], dtype=np.float32), (count, 1)),
    )


def _geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    normals = np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (4, 1))
    triangles = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    uvs = np.asarray(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
        dtype=np.float32,
    )
    return vertices, normals, triangles, uvs


def _glb_json(path: Path) -> tuple[dict[str, object], int]:
    payload = path.read_bytes()
    magic, version, total = struct.unpack_from("<4sII", payload)
    if magic != b"glTF" or version != 2 or total != len(payload):
        raise AssertionError("invalid GLB header")
    json_length, json_kind = struct.unpack_from("<I4s", payload, 12)
    if json_kind != b"JSON":
        raise AssertionError("missing GLB JSON chunk")
    document = json.loads(payload[20 : 20 + json_length].decode("utf-8"))
    binary_length, binary_kind = struct.unpack_from("<I4s", payload, 20 + json_length)
    if binary_kind != b"BIN\0":
        raise AssertionError("missing GLB binary chunk")
    return document, binary_length


class PbrExportTests(unittest.TestCase):
    def test_bakes_all_pbr_channels_and_self_contained_glb(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observed = np.full((16, 16, 3), 128, dtype=np.uint8)
            artifacts = build_pbr_artifacts(root, *_geometry(), observed, _surface())

            self.assertEqual(artifacts.emissive_strength, 2.0)
            self.assertEqual(artifacts.material_coverage, 1.0)
            for name in (
                "room-observed.png", "room-base-color.png", "room-metallic-roughness.png",
                "room-transmission.png", "room-normal.png", "room-emission.png",
                "room-pbr.glb", "pbr-report.json",
            ):
                self.assertTrue((root / name).is_file(), name)
            mr = np.asarray(Image.open(root / "room-metallic-roughness.png"))
            self.assertAlmostEqual(float(np.median(mr[..., 1])) / 255.0, 0.25, delta=0.01)
            self.assertAlmostEqual(float(np.median(mr[..., 2])) / 255.0, 0.75, delta=0.01)
            transmission = np.asarray(Image.open(root / "room-transmission.png"))
            self.assertAlmostEqual(float(np.median(transmission[..., 0])) / 255.0, 0.5, delta=0.01)
            normal = np.asarray(Image.open(root / "room-normal.png"))
            np.testing.assert_allclose(np.median(normal, axis=(0, 1)), [128, 128, 255], atol=1)

            document, binary_length = _glb_json(root / "room-pbr.glb")
            self.assertEqual(document["asset"]["version"], "2.0")
            self.assertEqual(len(document["images"]), 6)
            self.assertIn("KHR_materials_transmission", document["extensionsUsed"])
            self.assertIn("KHR_materials_emissive_strength", document["extensionsUsed"])
            declared_binary_length = document["buffers"][0]["byteLength"]
            self.assertGreaterEqual(binary_length, declared_binary_length)
            self.assertLess(binary_length - declared_binary_length, 4)
            material = document["materials"][0]
            self.assertEqual(material["extras"]["contract"], PBR_CONTRACT_VERSION)
            self.assertEqual(material["pbrMetallicRoughness"]["baseColorTexture"]["index"], 1)

    def test_unsupported_material_falls_back_to_observed_rough_opaque_dielectric(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observed = np.full((8, 8, 3), [64, 128, 192], dtype=np.uint8)
            artifacts = build_pbr_artifacts(root, *_geometry(), observed, _surface(0.0))

            self.assertEqual(artifacts.material_coverage, 0.0)
            base = np.asarray(Image.open(root / "room-base-color.png"))
            np.testing.assert_allclose(np.median(base, axis=(0, 1)), [64, 128, 192], atol=1)
            mr = np.asarray(Image.open(root / "room-metallic-roughness.png"))
            self.assertTrue(np.all(mr[..., 1] == 255))
            self.assertTrue(np.all(mr[..., 2] == 0))
            document, _ = _glb_json(root / "room-pbr.glb")
            self.assertNotIn("extensionsUsed", document)

    def test_rejects_surface_with_different_vertex_layout(self) -> None:
        vertices, normals, triangles, uvs = _geometry()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "match the PBR mesh vertices"):
                build_pbr_artifacts(
                    Path(temporary), vertices[:3], normals[:3], triangles[:1], uvs[:1],
                    np.zeros((4, 4, 3), dtype=np.uint8), _surface(),
                )


if __name__ == "__main__":
    unittest.main()
