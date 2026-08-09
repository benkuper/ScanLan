from __future__ import annotations

import unittest
import zipfile
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scanlan_splat.lingbot import (
    LingbotGeometry,
    _restore_bundled_flashinfer_cache,
    _rotation_matrices_to_quaternions,
    _inference_streaming_progressive,
    _streaming_configuration,
    _surface_seeds,
)
from scanlan_splat.media import (
    _align_lingbot_geometry,
    _anchor_lingbot_geometry,
    _restrict_lingbot_seeds_to_frames,
)


class _Transform:
    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = matrix

    def matrix(self) -> np.ndarray:
        return self._matrix


class _Image:
    def __init__(self, world_from_camera: np.ndarray) -> None:
        self.has_pose = True
        self._camera_from_world = np.linalg.inv(world_from_camera)[:3]

    def cam_from_world(self) -> _Transform:
        return _Transform(self._camera_from_world)


class _Reconstruction:
    def __init__(self, poses: list[np.ndarray]) -> None:
        self._images = {
            f"frame-{index:03}.jpg": _Image(pose)
            for index, pose in enumerate(poses)
        }

    def find_image_with_name(self, name: str) -> _Image | None:
        return self._images.get(name)


class LingbotGeometryTests(unittest.TestCase):
    def test_progressive_streaming_preserves_all_outputs_and_chunk_order(self) -> None:
        import torch

        class _Model:
            pred_normalization = False

            def __init__(self) -> None:
                self.parameter = torch.zeros(1)
                self.skip_events: list[bool] = []

            def parameters(self):
                yield self.parameter

            def clean_kv_cache(self) -> None:
                pass

            def _set_skip_append(self, value: bool) -> None:
                self.skip_events.append(value)

            def forward(self, images, **_kwargs):
                ids = images[:, :, 0, 0, 0]
                batch, frames = ids.shape
                return {
                    "pose_enc": ids[:, :, None].repeat(1, 1, 9),
                    "depth": ids[:, :, None, None, None].repeat(1, 1, 2, 2, 1),
                    "depth_conf": (ids + 10)[:, :, None, None].repeat(1, 1, 2, 2),
                }

        images = torch.stack(
            [torch.full((3, 2, 2), float(index)) for index in range(6)]
        )
        chunks: list[tuple[int, list[float]]] = []
        model = _Model()
        predictions = _inference_streaming_progressive(
            model,
            images,
            torch=torch,
            num_scale_frames=2,
            keyframe_interval=2,
            output_device=torch.device("cpu"),
            chunk_frames=2,
            on_chunk=lambda start, values: chunks.append(
                (start, values["pose_enc"][0, :, 0].tolist())
            ),
        )

        self.assertEqual(chunks, [(0, [0.0, 1.0]), (2, [2.0, 3.0]), (4, [4.0, 5.0])])
        self.assertEqual(predictions["pose_enc"][0, :, 0].tolist(), list(map(float, range(6))))
        self.assertEqual(predictions["depth_conf"][0, :, 0, 0].tolist(), [10, 11, 12, 13, 14, 15])
        self.assertEqual(model.skip_events, [True, False, True, False])

    def test_streaming_cache_is_bounded_for_16_gib_sdpa(self) -> None:
        class _Properties:
            total_memory = 16 * 1024**3

        class _Cuda:
            @staticmethod
            def get_device_properties(_device: int) -> _Properties:
                return _Properties()

        class _Torch:
            cuda = _Cuda()

        self.assertEqual(
            _streaming_configuration(_Torch(), 181, use_sdpa=False),
            (64, 1, 320),
        )
        self.assertEqual(
            _streaming_configuration(_Torch(), 181, use_sdpa=True),
            (64, 2, 320),
        )

    def test_bundled_flashinfer_cache_restores_only_safe_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "cache.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("0.6.11/120f/cached_ops/kernel/kernel.dll", b"dll")
            drive = root / "drive"
            with patch.dict(
                "os.environ",
                {
                    "SCANLAN_FLASHINFER_CACHE_ARCHIVE": str(archive),
                    "SystemDrive": str(drive),
                },
                clear=False,
            ):
                _restore_bundled_flashinfer_cache()

            self.assertEqual(
                (drive / "_fij/0.6.11/120f/cached_ops/kernel/kernel.dll").read_bytes(),
                b"dll",
            )

    def test_similarity_alignment_transforms_cameras_points_scales_and_orientations(self) -> None:
        source_centers = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.2],
                [1.7, 0.4, 0.3],
                [2.1, 1.1, 0.1],
                [2.4, 1.8, -0.2],
            ],
            dtype=np.float64,
        )
        angle = np.deg2rad(37.0)
        alignment_rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        alignment_scale = 2.75
        alignment_translation = np.asarray([4.0, -3.0, 1.5])
        source_poses = np.repeat(np.eye(4)[None], len(source_centers), axis=0)
        source_poses[:, :3, 3] = source_centers
        target_poses: list[np.ndarray] = []
        for source_pose in source_poses:
            target = np.eye(4)
            target[:3, :3] = alignment_rotation @ source_pose[:3, :3]
            target[:3, 3] = (
                alignment_scale * (alignment_rotation @ source_pose[:3, 3])
                + alignment_translation
            )
            target_poses.append(target)
        points = np.asarray([[0.2, 0.3, 1.0], [1.0, -0.4, 2.0]], dtype=np.float32)
        geometry = LingbotGeometry(
            world_from_cameras=source_poses,
            intrinsics=np.repeat(np.eye(3)[None], len(source_centers), axis=0),
            points=points,
            colors=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
            scales=np.full((2, 3), 0.1, dtype=np.float32),
            quaternions=np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float32),
            source_frame_indices=np.asarray([0, 1], dtype=np.int32),
            frame_confidence=np.ones(len(source_centers), dtype=np.float32),
            backend="test",
            model_path="test.pt",
            processed_size=(8, 8),
        )

        aligned, quality = _align_lingbot_geometry(
            geometry,
            _Reconstruction(target_poses),
            list(_Reconstruction(target_poses)._images),
        )

        np.testing.assert_allclose(aligned.world_from_cameras, target_poses, atol=1e-6)
        np.testing.assert_allclose(
            aligned.points,
            alignment_scale * (points @ alignment_rotation.T) + alignment_translation,
            atol=1e-6,
        )
        np.testing.assert_allclose(aligned.scales, geometry.scales * alignment_scale, atol=1e-6)
        np.testing.assert_array_equal(
            aligned.source_frame_indices,
            geometry.source_frame_indices,
        )
        expected_quaternion = _rotation_matrices_to_quaternions(alignment_rotation[None])[0]
        np.testing.assert_allclose(aligned.quaternions[0], expected_quaternion, atol=1e-6)
        self.assertEqual(quality["alignmentInlierCount"], len(source_centers))
        self.assertLess(quality["maximumCameraCenterResidual"], 1e-6)

    def test_local_anchoring_preserves_colmap_cameras_and_warps_owned_seeds(self) -> None:
        source_poses = np.repeat(np.eye(4)[None], 5, axis=0)
        source_poses[:, 0, 3] = np.arange(5, dtype=np.float64)
        target_poses = source_poses.copy()
        target_poses[:, :3, 3] *= 2.0
        angle = np.deg2rad(20.0)
        rotation = np.asarray(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        target_poses[:, :3, :3] = rotation
        target_poses[:, :3, 3] = 2.0 * (source_poses[:, :3, 3] @ rotation.T)
        reconstruction = _Reconstruction(list(target_poses))
        reconstruction._images.pop("frame-001.jpg")
        reconstruction._images.pop("frame-003.jpg")
        geometry = LingbotGeometry(
            world_from_cameras=source_poses,
            intrinsics=np.repeat(np.eye(3)[None], 5, axis=0),
            points=source_poses[:, :3, 3].astype(np.float32),
            colors=np.full((5, 3), 128, dtype=np.uint8),
            scales=np.full((5, 3), 0.1, dtype=np.float32),
            quaternions=np.asarray([[1.0, 0.0, 0.0, 0.0]] * 5, dtype=np.float32),
            source_frame_indices=np.arange(5, dtype=np.int32),
            frame_confidence=np.ones(5, dtype=np.float32),
            backend="test",
            model_path="test.pt",
            processed_size=(8, 8),
        )

        anchored, quality = _anchor_lingbot_geometry(
            geometry,
            reconstruction,
            [f"frame-{index:03}.jpg" for index in range(5)],
        )

        np.testing.assert_allclose(anchored.world_from_cameras, target_poses, atol=1e-6)
        np.testing.assert_allclose(anchored.points, target_poses[:, :3, 3], atol=1e-6)
        np.testing.assert_allclose(anchored.scales, 0.2, atol=1e-6)
        self.assertEqual(quality["anchorCameraCount"], 3)
        self.assertLess(quality["maximumAnchorCenterResidual"], 1e-8)

        restricted = _restrict_lingbot_seeds_to_frames(anchored, [0, 2, 4])
        np.testing.assert_array_equal(restricted.source_frame_indices, [0, 2, 4])
        self.assertEqual(len(restricted.points), 3)

    def test_surface_seeding_rejects_depth_discontinuity_shards(self) -> None:
        height = width = 12
        depths = np.full((1, height, width), 2.0, dtype=np.float32)
        depths[0, 6, 6] = 20.0
        confidence = np.full_like(depths, 2.0)
        images = np.full((1, height, width, 3), 128, dtype=np.uint8)
        intrinsics = np.asarray(
            [[[10.0, 0.0, 5.5], [0.0, 10.0, 5.5], [0.0, 0.0, 1.0]]],
            dtype=np.float64,
        )
        poses = np.eye(4, dtype=np.float64)[None]

        with patch("scanlan_splat.lingbot._sky_confidence", return_value=None):
            points, _colors, scales, quaternions, source_indices, scores = _surface_seeds(
                depths,
                confidence,
                images,
                intrinsics,
                poses,
                maximum_seeds=1_000,
            )

        self.assertGreater(len(points), 20)
        self.assertLess(float(np.max(points[:, 2])), 3.0)
        self.assertTrue(np.isfinite(scales).all())
        self.assertTrue(np.isfinite(quaternions).all())
        np.testing.assert_allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-5)
        np.testing.assert_array_equal(source_indices, np.zeros(len(points), dtype=np.int32))
        self.assertGreater(float(scores[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
