from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scanlan_splat.geometry_ipc import _load_geometry, run_lingbot_map_request
from scanlan_splat.lingbot import LingbotGeometry


class GeometryIpcTests(unittest.TestCase):
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
            ):
                np.testing.assert_array_equal(getattr(restored, field), getattr(expected, field))
            self.assertEqual(restored.backend, expected.backend)
            self.assertEqual(restored.model_path, expected.model_path)
            self.assertEqual(restored.processed_size, expected.processed_size)

            incompatible = dict(result, modelRevision="different")
            with self.assertRaisesRegex(RuntimeError, "incompatible LingBot-Map revision"):
                _load_geometry(root / "geometry.npz", incompatible)


if __name__ == "__main__":
    unittest.main()
