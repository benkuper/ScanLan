from __future__ import annotations

import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scanlan_splat.mapanything import (
    MAPANYTHING_MODEL_FILENAME,
    _bundled_dinov2_loader,
    _confidence_probability,
    resolve_mapanything_model,
    refine_mapanything_depth_request,
    restore_processed_raster,
)


class _Hub:
    def __init__(self) -> None:
        self.load = lambda *_args, **_kwargs: "network"


class _Torch:
    def __init__(self) -> None:
        self.hub = _Hub()


class MapAnythingAdapterTests(unittest.TestCase):
    def test_rgbd_request_batches_frames_and_publishes_aligned_arrays(self) -> None:
        class Predictor:
            backend = "fixture MapAnything"

            def infer_rgbd(self, colors, depths, intrinsics, camera_poses=None):
                self.camera_poses = camera_poses
                return [
                    (
                        depth + 0.1,
                        np.ones(depth.shape, dtype=bool),
                        np.full(depth.shape, 0.8, dtype=np.float32),
                    )
                    for depth in depths
                ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = []
            width, height = 8, 6
            for index in range(3):
                color_path = root / f"{index}.rgb"
                depth_path = root / f"{index}.u16"
                np.full((height, width, 3), index, dtype=np.uint8).tofile(color_path)
                np.full((height, width), 2000, dtype="<u2").tofile(depth_path)
                frames.append(
                    {
                        "key": str(index),
                        "colorPath": str(color_path),
                        "depthPath": str(depth_path),
                        "predictionPath": str(root / f"{index}.npy"),
                        "modelMaskPath": str(root / f"{index}-mask.npy"),
                        "confidencePath": str(root / f"{index}-confidence.npy"),
                        "width": width,
                        "height": height,
                        "fx": 10.0,
                        "fy": 11.0,
                        "cx": 4.0,
                        "cy": 3.0,
                        "depthScale": 1000.0,
                        "cameraPose": np.eye(4).reshape(-1).tolist(),
                    }
                )
            request_path = root / "request.json"
            request_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "cancelPath": "",
                        "resultPath": str(root / "result.json"),
                        "frames": frames,
                    }
                ),
                encoding="utf-8",
            )
            predictor = Predictor()
            result = refine_mapanything_depth_request(
                request_path, root / "progress.json", predictor=predictor
            )
            self.assertEqual(result["modelSha256"], "fa06c0fdccefc5048e072c85935d5789b1e36b307f3859033c17f9dcb9fd5201")
            self.assertIsNotNone(predictor.camera_poses)
            for frame in frames:
                prediction = np.load(frame["predictionPath"], allow_pickle=False)
                mask = np.load(frame["modelMaskPath"], allow_pickle=False)
                confidence = np.load(frame["confidencePath"], allow_pickle=False)
                np.testing.assert_allclose(prediction, 2.1)
                self.assertTrue(mask.all())
                np.testing.assert_allclose(confidence, 0.8)

    def test_confidence_head_is_mapped_to_bounded_probability(self) -> None:
        values = _confidence_probability(
            np.asarray([1.0, 2.0, 10.0, np.inf, np.nan], dtype=np.float32)
        )
        np.testing.assert_allclose(values[:3], [0.0, 0.5, 0.9], atol=1e-6)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(np.all((values >= 0.0) & (values <= 1.0)))

    def test_restore_processed_raster_rejects_cropped_source_borders(self) -> None:
        processed = np.full((6, 6), 2.0, dtype=np.float32)
        valid = np.ones((6, 6), dtype=bool)
        restored, restored_valid = restore_processed_raster(
            processed,
            valid,
            (12, 6),
        )
        self.assertEqual(restored.shape, (6, 12))
        self.assertFalse(restored_valid[:, 0].any())
        self.assertFalse(restored_valid[:, -1].any())
        self.assertTrue(restored_valid[:, 3:9].all())
        np.testing.assert_allclose(restored[restored_valid], 2.0, atol=1e-5)

    def test_bundled_loader_restores_torch_hub_function(self) -> None:
        torch = _Torch()
        original = torch.hub.load
        factory_result = object()
        with patch(
            "mapanything.models.external.dinov2.hub.backbones.dinov2_vitg14",
            return_value=factory_result,
        ) as factory:
            with _bundled_dinov2_loader(torch):
                result = torch.hub.load(
                    "facebookresearch/dinov2",
                    "dinov2_vitg14",
                    force_reload=True,
                    pretrained=True,
                )
                self.assertIs(result, factory_result)
                factory.assert_called_once_with(pretrained=False)
        self.assertIs(torch.hub.load, original)

    def test_model_resolution_requires_config_and_pinned_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / MAPANYTHING_MODEL_FILENAME).write_bytes(b"checkpoint")
            (root / "config.json").write_text("{}", encoding="utf-8")
            with patch.dict("os.environ", {"SCANLAN_MAPANYTHING_MODEL": str(root)}):
                self.assertEqual(resolve_mapanything_model(verify=False), root.resolve())
                with self.assertRaisesRegex(RuntimeError, "pinned digest"):
                    resolve_mapanything_model(verify=True)


if __name__ == "__main__":
    unittest.main()
