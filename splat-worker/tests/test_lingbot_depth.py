from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scanlan_splat.lingbot_depth import refine_depth_request


class _Predictor:
    backend = "test predictor"

    def __init__(self) -> None:
        self.intrinsics: np.ndarray | None = None

    def infer(
        self,
        color: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.intrinsics = intrinsics.copy()
        self.color_shape = color.shape
        return depth_m + 0.125, np.ones(depth_m.shape, dtype=bool)


class LingbotDepthRequestTests(unittest.TestCase):
    def test_request_keeps_the_aligned_grid_and_normalizes_intrinsics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            width, height = 6, 4
            color_path = root / "aligned.rgb"
            depth_path = root / "depth.u16"
            prediction_path = root / "prediction.npy"
            mask_path = root / "mask.npy"
            result_path = root / "result.json"
            progress_path = root / "progress.json"
            np.arange(width * height * 3, dtype=np.uint8).tofile(color_path)
            np.full((height, width), 2000, dtype="<u2").tofile(depth_path)
            request = {
                "schemaVersion": 1,
                "resultPath": str(result_path),
                "frames": [
                    {
                        "key": "phase:0",
                        "colorPath": str(color_path),
                        "depthPath": str(depth_path),
                        "predictionPath": str(prediction_path),
                        "modelMaskPath": str(mask_path),
                        "width": width,
                        "height": height,
                        "fx": 5.0,
                        "fy": 4.0,
                        "cx": 2.5,
                        "cy": 1.5,
                        "depthScale": 1000.0,
                    }
                ],
            }
            request_path = root / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            predictor = _Predictor()

            result = refine_depth_request(
                request_path, progress_path, predictor=predictor
            )

            self.assertEqual(result["status"], "complete")
            self.assertEqual(predictor.color_shape, (height, width, 3))
            np.testing.assert_allclose(
                predictor.intrinsics,
                [[5.0 / width, 0.0, 2.5 / width], [0.0, 1.0, 1.5 / height], [0.0, 0.0, 1.0]],
            )
            np.testing.assert_allclose(np.load(prediction_path), 2.125)
            np.testing.assert_array_equal(np.load(mask_path), 1)
            self.assertEqual(json.loads(progress_path.read_text())["progress"], 1.0)


if __name__ == "__main__":
    unittest.main()
