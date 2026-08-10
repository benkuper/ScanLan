from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scanlan_splat.geometry_ipc import (
    ProgressivePreviewAccumulator,
    _load_geometry,
    run_da3_request,
    run_lingbot_map_request,
)
from scanlan_splat.lingbot import LingbotGeometry


class GeometryIpcTests(unittest.TestCase):
    @staticmethod
    def _preview_chunk(first_frame: int, count: int = 4) -> LingbotGeometry:
        poses = np.repeat(np.eye(4)[None], count, axis=0)
        poses[:, 0, 3] = np.arange(count) * 0.1 + first_frame * 0.1
        intrinsics = np.repeat(np.eye(3)[None], count, axis=0)
        intrinsics[:, 0, 0] = intrinsics[:, 1, 1] = 400.0
        point_count = 12_000
        return LingbotGeometry(
            world_from_cameras=poses,
            intrinsics=intrinsics,
            points=np.column_stack(
                (
                    np.linspace(0, 1, point_count),
                    np.zeros(point_count),
                    np.ones(point_count),
                )
            ).astype(np.float32),
            colors=np.full((point_count, 3), 128, dtype=np.uint8),
            scales=np.full((point_count, 3), 0.01, dtype=np.float32),
            quaternions=np.tile(np.asarray([1, 0, 0, 0], dtype=np.float32), (point_count, 1)),
            source_frame_indices=(
                np.arange(point_count, dtype=np.int32) % count + first_frame
            ),
            frame_confidence=np.resize(
                np.asarray([0.9, 0.8, 0.4, 0.9], dtype=np.float32), count
            ),
            backend="fixture",
            model_path="fixture.pt",
            processed_size=(4, 3),
        )

    def test_progressive_preview_is_bounded_and_labels_unverified_scale(self) -> None:
        accumulator = ProgressivePreviewAccumulator(maximum_points=10_000, resident_submaps=2)
        status = None
        for first_frame in (0, 4, 8):
            status = accumulator.update(
                self._preview_chunk(first_frame),
                first_frame,
                first_frame + 4,
            )
        assert status is not None
        self.assertLessEqual(len(status["points"]), 10_000)
        self.assertEqual(status["status"]["scaleStatus"], "MODEL_METRIC_UNVERIFIED")
        self.assertTrue(status["status"]["learnedOnly"])
        self.assertEqual(status["status"]["residentSubmapCount"], 2)
        self.assertEqual(status["status"]["archivedSubmapCount"], 1)
        self.assertGreater(status["status"]["rejectedFrameCount"], 0)

    def test_isolated_lingbot_contract_preserves_every_output_array_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = [root / f"{index}.png" for index in range(3)]
            for path in images:
                path.write_bytes(b"fixture")
            rays = np.arange(24, dtype=np.float64).reshape(3, 4, 2)
            np.save(root / "rays.npy", rays, allow_pickle=False)
            world_from_cameras = np.repeat(np.eye(4)[None], 3, axis=0)
            world_from_cameras[:, 0, 3] = (0.0, 0.1, 0.2)
            intrinsics = np.repeat(np.eye(3)[None], 3, axis=0)
            intrinsics[:, 0, 0] = 400.0
            intrinsics[:, 1, 1] = 405.0
            quaternions = np.zeros((5, 4), dtype=np.float32)
            quaternions[:, 0] = 1.0
            expected = LingbotGeometry(
                world_from_cameras=world_from_cameras,
                intrinsics=intrinsics,
                points=np.arange(15, dtype=np.float32).reshape(5, 3),
                colors=np.arange(15, dtype=np.uint8).reshape(5, 3),
                scales=np.arange(15, dtype=np.float32).reshape(5, 3) + 0.1,
                quaternions=quaternions,
                source_frame_indices=np.asarray([0, 0, 1, 2, 2], dtype=np.int32),
                frame_confidence=np.asarray([0.7, 0.8, 0.9], dtype=np.float32),
                backend="fixture backend",
                model_path="fixture.pt",
                processed_size=(4, 3),
                opacities=np.linspace(0.1, 0.9, 5, dtype=np.float32),
            )
            observed: dict[str, object] = {}

            def infer(paths, **kwargs):
                observed["paths"] = paths
                observed.update(kwargs)
                return expected

            request = {
                "schemaVersion": 1,
                "imagePaths": [str(path) for path in images],
                "maximumSeeds": 123,
                "normalizedRaysPath": str(root / "rays.npy"),
                "outputIndices": [0, 2],
                "cancelPath": "",
                "arraysPath": str(root / "geometry.npz"),
                "resultPath": str(root / "result.json"),
            }
            request_path = root / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")

            result = run_lingbot_map_request(
                request_path,
                root / "progress.json",
                inference=infer,
            )
            restored = _load_geometry(root / "geometry.npz", result)

            self.assertEqual(observed["maximum_seeds"], 123)
            self.assertEqual(observed["output_indices"], [0, 2])
            np.testing.assert_array_equal(observed["normalized_rays"], rays)
            for field in (
                "world_from_cameras",
                "intrinsics",
                "points",
                "colors",
                "scales",
                "quaternions",
                "source_frame_indices",
                "frame_confidence",
                "opacities",
            ):
                np.testing.assert_array_equal(getattr(restored, field), getattr(expected, field))
            self.assertEqual(restored.backend, expected.backend)
            self.assertEqual(restored.model_path, expected.model_path)
            self.assertEqual(restored.processed_size, expected.processed_size)

            incompatible = dict(result, modelRevision="different")
            with self.assertRaisesRegex(RuntimeError, "incompatible LingBot-Map revision"):
                _load_geometry(root / "geometry.npz", incompatible)

    def test_da3_direct_gaussian_cuda_oom_falls_back_with_explicit_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = [root / f"{index}.png" for index in range(3)]
            for path in images:
                path.write_bytes(b"fixture")
            request = {
                "schemaVersion": 1,
                "imagePaths": [str(path) for path in images],
                "maximumSeeds": 123,
                "directGaussians": True,
                "cancelPath": "",
                "arraysPath": str(root / "geometry.npz"),
                "resultPath": str(root / "result.json"),
            }
            request_path = root / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            geometry = self._preview_chunk(0, 3)
            calls: list[bool] = []

            def infer(_predictor, _paths, **kwargs):
                direct = bool(kwargs["infer_gaussians"])
                calls.append(direct)
                if direct:
                    raise RuntimeError("CUDA out of memory while allocating direct GS head")
                return geometry, {"mode": "single_window"}

            class Predictor:
                backend = "fixture DA3"

            with patch(
                "scanlan_splat.geometry_ipc.infer_da3_geometry_streaming", infer
            ):
                result = run_da3_request(
                    request_path,
                    root / "progress.json",
                    predictor=Predictor(),
                )

            self.assertEqual(calls, [True, False])
            self.assertEqual(result["proposalType"], "camera-depth")
            self.assertTrue(result["streaming"]["directGaussiansRequested"])
            self.assertFalse(result["streaming"]["directGaussiansUsed"])
            self.assertIn("out of memory", result["streaming"]["memoryFallback"].lower())


if __name__ == "__main__":
    unittest.main()
