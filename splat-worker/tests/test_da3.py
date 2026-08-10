from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scanlan_splat.da3 import (
    DA3_MODEL_FILENAME,
    DA3_MODEL_SHA256,
    _confidence_probability,
    _direct_gaussian_mask,
    _metric_scaled_gaussians,
    infer_da3_geometry_streaming,
    refine_da3_depth_request,
    resolve_da3_model,
)
from scanlan_splat.lingbot import LingbotGeometry


class Da3AdapterTests(unittest.TestCase):
    def test_direct_gaussian_mask_preserves_learned_opacity_and_crops_border(self) -> None:
        confidence = np.full((2, 256, 256), 0.8, dtype=np.float32)
        opacity = np.linspace(0.0, 1.0, confidence.size, dtype=np.float32)
        keep = _direct_gaussian_mask(confidence, opacity).reshape(confidence.shape)

        self.assertFalse(keep[:, :8, :].any())
        self.assertFalse(keep[:, -8:, :].any())
        self.assertFalse(keep[:, :, :8].any())
        self.assertFalse(keep[:, :, -8:].any())
        self.assertTrue(keep[:, 8:-8, 8:-8].all())
        opacity[256 * 256 + 128 * 256 + 128] = np.nan
        self.assertFalse(
            _direct_gaussian_mask(confidence, opacity).reshape(confidence.shape)[1, 128, 128]
        )

    def test_confidence_head_is_mapped_to_probability(self) -> None:
        values = _confidence_probability(
            np.asarray([1.0, 2.0, 10.0, np.inf, np.nan], dtype=np.float32)
        )
        np.testing.assert_allclose(values[:3], [0.0, 0.5, 0.9], atol=1e-6)
        self.assertTrue(np.isfinite(values).all())

    def test_model_resolution_requires_pinned_nested_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / DA3_MODEL_FILENAME).write_bytes(b"checkpoint")
            (root / "config.json").write_text("{}", encoding="utf-8")
            with patch.dict("os.environ", {"SCANLAN_DA3_MODEL": str(root)}):
                self.assertEqual(resolve_da3_model(verify=False), root.resolve())
                with self.assertRaisesRegex(RuntimeError, "pinned digest"):
                    resolve_da3_model(verify=True)

    def test_nested_metric_scale_is_applied_to_direct_gaussians(self) -> None:
        means, scales = _metric_scaled_gaussians(
            np.asarray([[1.0, -2.0, 3.0]], dtype=np.float32),
            np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32),
            2.5,
        )
        np.testing.assert_allclose(means, [[2.5, -5.0, 7.5]])
        np.testing.assert_allclose(scales, [[0.25, 0.5, 0.75]])
        with self.assertRaisesRegex(RuntimeError, "metric scale"):
            _metric_scaled_gaussians(means, scales, float("nan"))

    def test_pose_conditioned_request_is_bounded_and_publishes_aligned_arrays(self) -> None:
        class Predictor:
            backend = "fixture DA3 Nested"

            def infer_pose_conditioned_depth(self, colors, intrinsics, poses):
                self.maximum_batch = max(getattr(self, "maximum_batch", 0), len(colors))
                return [
                    (
                        np.full(color.shape[:2], 2.25, dtype=np.float32),
                        np.ones(color.shape[:2], dtype=bool),
                        np.full(color.shape[:2], 0.85, dtype=np.float32),
                    )
                    for color in colors
                ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = []
            width, height = 8, 6
            for index in range(19):
                color_path = root / f"{index}.rgb"
                np.full((height, width, 3), index, dtype=np.uint8).tofile(color_path)
                frames.append(
                    {
                        "key": str(index),
                        "colorPath": str(color_path),
                        "predictionPath": str(root / f"{index}.npy"),
                        "modelMaskPath": str(root / f"{index}-mask.npy"),
                        "width": width,
                        "height": height,
                        "fx": 10.0,
                        "fy": 11.0,
                        "cx": 4.0,
                        "cy": 3.0,
                        "cameraPose": np.eye(4).reshape(-1).tolist(),
                    }
                )
            request_path = root / "request.json"
            request_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "chunkSize": 8,
                        "cancelPath": "",
                        "resultPath": str(root / "result.json"),
                        "frames": frames,
                    }
                ),
                encoding="utf-8",
            )
            predictor = Predictor()
            result = refine_da3_depth_request(
                request_path, root / "progress.json", predictor=predictor
            )
            self.assertEqual(result["modelSha256"], DA3_MODEL_SHA256)
            self.assertLessEqual(predictor.maximum_batch, 8)
            for frame in frames:
                np.testing.assert_allclose(
                    np.load(frame["predictionPath"], allow_pickle=False), 2.25
                )
                self.assertTrue(
                    np.load(frame["modelMaskPath"], allow_pickle=False).all()
                )

    def test_streaming_windows_align_overlap_without_duplicate_geometry(self) -> None:
        class Predictor:
            backend = "fixture DA3 Nested"
            model_root = Path("fixture")

            def __init__(self) -> None:
                self.calls = 0

            def infer_geometry_window(
                self, paths, *, maximum_seeds, infer_gaussians=False
            ):
                indices = np.asarray([int(path.stem) for path in paths], dtype=np.int32)
                global_centers = np.column_stack(
                    (
                        indices * 0.08,
                        np.sin(indices * 0.31) * 0.3,
                        np.cos(indices * 0.19) * 0.2,
                    )
                )
                if self.calls == 0:
                    scale = 1.0
                    rotation = np.eye(3)
                    translation = np.zeros(3)
                else:
                    angle = 0.17 * self.calls
                    rotation = np.asarray(
                        [
                            [np.cos(angle), -np.sin(angle), 0.0],
                            [np.sin(angle), np.cos(angle), 0.0],
                            [0.0, 0.0, 1.0],
                        ]
                    )
                    scale = 1.0 + 0.2 * self.calls
                    translation = np.asarray([0.4, -0.2, 0.1]) * self.calls
                local_centers = (global_centers - translation) @ rotation / scale
                poses = np.repeat(np.eye(4)[None], len(paths), axis=0)
                poses[:, :3, :3] = rotation.T
                poses[:, :3, 3] = local_centers
                points = local_centers + np.asarray([0.0, 0.0, 1.0])
                self.calls += 1
                return LingbotGeometry(
                    world_from_cameras=poses,
                    intrinsics=np.repeat(np.eye(3)[None], len(paths), axis=0),
                    points=points.astype(np.float32),
                    colors=np.full((len(paths), 3), 127, dtype=np.uint8),
                    scales=np.full((len(paths), 3), 0.01, dtype=np.float32),
                    quaternions=np.repeat(
                        np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                        len(paths),
                        axis=0,
                    ),
                    source_frame_indices=np.arange(len(paths), dtype=np.int32),
                    frame_confidence=np.full(len(paths), 0.9, dtype=np.float32),
                    backend=self.backend,
                    model_path="fixture",
                    processed_size=(70, 56),
                    opacities=np.linspace(0.1, 0.9, len(paths), dtype=np.float32),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index in range(55):
                path = root / f"{index}.png"
                path.write_bytes(b"fixture")
                paths.append(path)
            geometry, telemetry = infer_da3_geometry_streaming(
                Predictor(), paths, window_size=24, overlap=6
            )
            indices = np.arange(55)
            expected = np.column_stack(
                (
                    indices * 0.08,
                    np.sin(indices * 0.31) * 0.3,
                    np.cos(indices * 0.19) * 0.2,
                )
            )
            np.testing.assert_allclose(
                geometry.world_from_cameras[:, :3, 3], expected, atol=1e-5
            )
            self.assertEqual(len(geometry.points), 55)
            self.assertIsNotNone(geometry.opacities)
            self.assertEqual(len(geometry.opacities), 55)
            self.assertEqual(telemetry["mode"], "bounded_overlap_streaming")
            self.assertEqual(telemetry["windowCount"], 3)
            self.assertLess(telemetry["maximumOverlapResidual"], 1e-5)


if __name__ == "__main__":
    unittest.main()
