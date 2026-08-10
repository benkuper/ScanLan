from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from scanlan_material.radiometry import (
    RADIOMETRY_VERSION,
    linear_to_srgb,
    load_linear_rgb,
    prepare_dataset_radiometry,
    srgb_to_linear,
)


class RadiometryTests(unittest.TestCase):
    def test_srgb_reference_transfer_and_round_trip(self) -> None:
        encoded = np.asarray([0.0, 0.04045, 0.5, 1.0], dtype=np.float32)
        linear = srgb_to_linear(encoded)
        np.testing.assert_allclose(
            linear,
            np.asarray([0.0, 0.0031308, 0.21404114, 1.0], dtype=np.float32),
            atol=2e-7,
        )
        np.testing.assert_allclose(linear_to_srgb(linear), encoded, atol=2e-7)

    def test_embedded_profile_is_normalized_before_linearization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profiled.png"
            profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
            Image.new("RGB", (2, 1), (128, 64, 32)).save(path, icc_profile=profile)
            linear, metadata = load_linear_rgb(path)
        self.assertTrue(metadata["embeddedIcc"])
        self.assertEqual(metadata["conversion"], "embedded-icc-to-srgb-relative-colorimetric")
        np.testing.assert_allclose(
            linear[0, 0],
            srgb_to_linear(np.asarray([128, 64, 32], dtype=np.float32) / 255.0),
            atol=1e-6,
        )

    def test_unprofiled_non_srgb_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unprofiled-cmyk.jpg"
            Image.new("CMYK", (1, 1), (0, 0, 0, 0)).save(path)
            with self.assertRaisesRegex(ValueError, "unsupported unprofiled image mode"):
                load_linear_rgb(path)

    def test_dataset_preparation_is_content_addressed_and_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            (dataset / "images").mkdir(parents=True)
            Image.new("RGB", (4, 3), (128, 64, 32)).save(dataset / "images" / "000.jpg")
            (dataset / "dataset.json").write_text(
                json.dumps(
                    {
                        "schemaVersion": 3,
                        "fingerprint": "fixture",
                        "frames": [{"image": "images/000.jpg"}],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "radiometry"
            first = prepare_dataset_radiometry(dataset, output)
            second = prepare_dataset_radiometry(dataset, output)
            linear_path = output / first["fingerprint"] / "linear" / "000000.npy"
            array = np.load(linear_path)
            linear_path.unlink()
            rebuilt = prepare_dataset_radiometry(dataset, output)
        self.assertEqual(first, second)
        self.assertEqual(first, rebuilt)
        self.assertEqual(first["radiometryVersion"], RADIOMETRY_VERSION)
        self.assertEqual(array.dtype, np.float16)
        self.assertEqual(array.shape, (3, 4, 3))

    def test_dataset_paths_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "dataset.json").write_text(
                json.dumps({"frames": [{"image": "../outside.jpg"}]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsafe image path"):
                prepare_dataset_radiometry(dataset, root / "output")


if __name__ == "__main__":
    unittest.main()
