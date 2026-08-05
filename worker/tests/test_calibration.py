from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scanlan.calibration import (
    distort_normalized,
    rgb_depth_zbuffer,
    scaled_pinhole_camera,
    undistort_rgb_to_pinhole,
)
from scanlan.io import RgbCameraModel, load_depth, read_phase
from scanlan.mock_data import create_mock_project


class CalibrationTests(unittest.TestCase):
    def test_opencv_rational_projection_uses_denominator_coefficients(self) -> None:
        camera = RgbCameraModel(
            64,
            48,
            50.0,
            51.0,
            31.5,
            23.5,
            "opencv_rational",
            (0.12, -0.03, 0.004, -0.002, 0.01, 0.08, -0.02, 0.005),
        )
        x = np.asarray([0.4], dtype=np.float64)
        y = np.asarray([-0.2], dtype=np.float64)

        actual_x, actual_y = distort_normalized(x, y, camera)

        r2 = 0.4**2 + (-0.2) ** 2
        radial = (1 + 0.12 * r2 - 0.03 * r2**2 + 0.01 * r2**3) / (
            1 + 0.08 * r2 - 0.02 * r2**2 + 0.005 * r2**3
        )
        expected_x = 0.4 * radial + 2 * 0.004 * 0.4 * -0.2 - 0.002 * (r2 + 2 * 0.4**2)
        expected_y = -0.2 * radial + 0.004 * (r2 + 2 * (-0.2) ** 2) + 2 * -0.002 * 0.4 * -0.2
        np.testing.assert_allclose(actual_x, [expected_x], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual_y, [expected_y], rtol=1e-12, atol=1e-12)

    def test_zero_rational_distortion_remaps_rgb_without_changing_pixels(self) -> None:
        width, height = 9, 7
        yy, xx = np.indices((height, width))
        image = np.stack((xx * 20, yy * 25, (xx + yy) * 10), axis=-1).astype(np.uint8)
        camera = RgbCameraModel(
            width,
            height,
            6.0,
            6.0,
            4.0,
            3.0,
            "opencv_rational",
            (0.0,) * 8,
        )

        target = scaled_pinhole_camera(camera, width)
        remapped_image, valid = undistort_rgb_to_pinhole(
            image, camera, target, tile_rows=3
        )

        np.testing.assert_array_equal(remapped_image, image)
        self.assertTrue(valid.all())

    def test_scaled_pinhole_camera_preserves_pixel_center_rays(self) -> None:
        source = RgbCameraModel(9, 7, 6.0, 6.5, 4.0, 3.0, "pinhole", ())

        target = scaled_pinhole_camera(source, 5)

        self.assertEqual((target.width, target.height), (5, 4))
        scale_x = 5 / 9
        scale_y = 4 / 7
        self.assertAlmostEqual(target.fx, source.fx * scale_x)
        self.assertAlmostEqual(target.fy, source.fy * scale_y)
        self.assertAlmostEqual(target.cx, (source.cx + 0.5) * scale_x - 0.5)
        self.assertAlmostEqual(target.cy, (source.cy + 0.5) * scale_y - 0.5)

    def test_unknown_lens_model_is_rejected(self) -> None:
        camera = RgbCameraModel(8, 8, 4.0, 4.0, 3.5, 3.5, "fisheye", ())
        with self.assertRaisesRegex(ValueError, "Unsupported RGB lens model"):
            distort_normalized(np.asarray([0.0]), np.asarray([0.0]), camera)

    def test_phase_reader_rejects_incomplete_rational_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = create_mock_project(Path(temporary) / "scan", phase_count=1, frame_count=1)
            phase_root = next((root / "phases").iterdir())
            manifest_path = phase_root / "phase.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["rgbCamera"]["model"] = "opencv_rational"
            manifest["rgbCamera"]["distortion"] = [0.0] * 5
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unsupported or incomplete"):
                read_phase(phase_root)

    def test_depth_projects_directly_to_a_scaled_pinhole_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = create_mock_project(Path(temporary) / "scan", phase_count=1, frame_count=1)
            phase = read_phase(next((root / "phases").iterdir()))
            target = scaled_pinhole_camera(phase.rgb_camera, 24)

            zbuffer, uv_map, visibility = rgb_depth_zbuffer(
                load_depth(phase.frames[0], phase.camera),
                phase,
                phase.frames[0],
                output_camera=target,
            )

            self.assertEqual(zbuffer.shape, (18, 24))
            self.assertEqual(uv_map.shape, (36, 48, 2))
            self.assertGreater(int(visibility.sum()), 100)


if __name__ == "__main__":
    unittest.main()
