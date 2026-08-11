from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scanlan_material.contracts import (
    MATERIAL_CLASSES,
    OPTICAL_RISKS,
    MaterialPrediction,
    read_prediction,
    write_prediction,
)


class MaterialContractTests(unittest.TestCase):
    def prediction(self) -> MaterialPrediction:
        classes = np.zeros((3, 4, len(MATERIAL_CLASSES)), dtype=np.float32)
        classes[..., MATERIAL_CLASSES.index("metal")] = 1.0
        risks = np.zeros((3, 4, len(OPTICAL_RISKS)), dtype=np.float32)
        risks[..., OPTICAL_RISKS.index("high_specular")] = 0.9
        risks[..., OPTICAL_RISKS.index("mirror")] = 0.35
        valid = np.ones((3, 4), dtype=bool)
        return MaterialPrediction(
            classes,
            risks,
            valid,
            np.full((3, 4), 0.8, dtype=np.float32),
            albedo_linear=np.full((3, 4, 3), 0.25, dtype=np.float32),
            roughness=np.full((3, 4), 0.15, dtype=np.float32),
            metallic=np.ones((3, 4), dtype=np.float32),
            transmission=np.zeros((3, 4), dtype=np.float32),
            normal_camera=np.broadcast_to(
                np.asarray([0.0, 0.0, 1.0], dtype=np.float32), (3, 4, 3)
            ).copy(),
            metadata={"backend": "fixture", "frameIndex": 7},
        )

    def test_overlapping_optical_risks_round_trip_independently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prediction.npz"
            write_prediction(path, self.prediction())
            restored = read_prediction(path)
        self.assertEqual(restored.metadata["frameIndex"], 7)
        self.assertAlmostEqual(
            float(restored.optical_risk_probabilities[0, 0, OPTICAL_RISKS.index("mirror")]),
            0.35,
            places=3,
        )
        self.assertEqual(
            int(np.argmax(restored.class_probabilities[0, 0])),
            MATERIAL_CLASSES.index("metal"),
        )

    def test_invalid_pixels_cannot_retain_risk_or_confidence(self) -> None:
        value = self.prediction()
        valid = value.valid_mask.copy()
        valid[0, 0] = False
        with self.assertRaisesRegex(ValueError, "invalid pixels"):
            MaterialPrediction(
                value.class_probabilities,
                value.optical_risk_probabilities,
                valid,
                value.confidence,
            ).validated()

    def test_pbr_fields_are_linear_and_bounded(self) -> None:
        value = self.prediction()
        albedo = value.albedo_linear.copy()
        albedo[1, 1, 0] = 1.1
        with self.assertRaisesRegex(ValueError, "linear albedo"):
            MaterialPrediction(
                value.class_probabilities,
                value.optical_risk_probabilities,
                value.valid_mask,
                value.confidence,
                albedo_linear=albedo,
            ).validated()


if __name__ == "__main__":
    unittest.main()
