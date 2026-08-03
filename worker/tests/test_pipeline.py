from __future__ import annotations

import json
import math
import csv
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scanlan.imu import odometry_rotation_prior
from scanlan.calibration import rgb_depth_zbuffer
from scanlan.io import (
    CameraModel,
    FrameRecord,
    ImuSample,
    PhaseData,
    load_depth,
    load_source_rgb,
    read_phase,
    read_project,
)
from scanlan.mock_data import create_mock_project
from scanlan.compute import (
    ComputeBackend,
    _tensor_surfel_merge,
    _tensor_tsdf,
    tensor_odometry,
    tensor_refine_registration,
    tensor_rgbd,
)
from scanlan.open3d_engine import (
    PhaseAlignment,
    _apply_fragment_corrections,
    _display_points,
    _interpolate_rigid_transform,
    _prefer_alignment,
    _tracking_fragment_ranges,
    _trajectory_alignment_acceptable,
    reconstruct_open3d,
)
from scanlan.mesh import PosedFrame, _bake_triangle_atlas, _weld_depth_meshes
from scanlan.reconstruct import reconstruct_project


class PipelineTests(unittest.TestCase):
    def test_x_mirror_is_only_applied_for_kinect_v2(self) -> None:
        cloud = type("Cloud", (), {"points": np.asarray([[1.0, 2.0, 3.0]])})()
        np.testing.assert_array_equal(_display_points(cloud, False), [[1.0, -2.0, -3.0]])
        np.testing.assert_array_equal(_display_points(cloud, True), [[-1.0, -2.0, -3.0]])

    def test_long_tracking_fragments_do_not_leave_a_tiny_tail(self) -> None:
        self.assertEqual(_tracking_fragment_ranges(5), [(0, 5)])
        self.assertEqual(_tracking_fragment_ranges(6), [(0, 4), (4, 6)])
        self.assertEqual(
            _tracking_fragment_ranges(10),
            [(0, 4), (4, 8), (8, 10)],
        )

    def test_fragment_corrections_are_smoothed_across_the_trajectory(self) -> None:
        poses = []
        for frame_index in range(11):
            pose = np.eye(4)
            pose[0, 3] = frame_index * 0.1
            poses.append(pose)
        identity = np.eye(4)
        corrected_end = np.eye(4)
        corrected_end[:3, :3] = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        corrected_end[1, 3] = -1.0

        corrected = _apply_fragment_corrections(
            poses,
            [0, 10],
            [identity, corrected_end],
        )

        np.testing.assert_allclose(corrected[0], poses[0], atol=1e-8)
        np.testing.assert_allclose(corrected[-1], corrected_end @ poses[-1], atol=1e-8)
        self.assertAlmostEqual(corrected[5][1, 3], -0.5 + math.sqrt(0.5) * 0.5)
        np.testing.assert_allclose(
            corrected[5][:3, :3].T @ corrected[5][:3, :3],
            np.eye(3),
            atol=1e-8,
        )

    def test_trajectory_relocalization_keeps_strict_quality_and_motion_gates(self) -> None:
        accepted = PhaseAlignment(2, "test", 0.4, 0.02, 0.3, 0.3, 0.8, 2.0, 72)
        better = PhaseAlignment(2, "test", 0.5, 0.015, 0.4, 0.4, 0.85, 1.0, 80)
        poor_overlap = PhaseAlignment(2, "test", 0.4, 0.02, 0.05, 0.3, 0.8, 2.0, 72)
        correction = np.eye(4)
        correction[0, 3] = 0.2

        self.assertTrue(_trajectory_alignment_acceptable(accepted, correction))
        self.assertTrue(_prefer_alignment(better, accepted))
        self.assertFalse(_trajectory_alignment_acceptable(poor_overlap, correction))
        correction[0, 3] = 1.3
        self.assertFalse(_trajectory_alignment_acceptable(accepted, correction))

    def test_rigid_interpolation_preserves_endpoints(self) -> None:
        left = np.eye(4)
        right = np.eye(4)
        right[:3, :3] = np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )
        right[:3, 3] = [1.0, 2.0, 3.0]
        np.testing.assert_allclose(_interpolate_rigid_transform(left, right, 0.0), left)
        np.testing.assert_allclose(_interpolate_rigid_transform(left, right, 1.0), right)

    def test_mock_project_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = create_mock_project(Path(temporary) / "scan", phase_count=2, frame_count=4)
            project = read_project(root)
            self.assertEqual(len(project["phases"]), 2)
            phase = read_phase(root / "phases" / project["phases"][0]["id"])
            self.assertEqual(len(phase.frames), 4)
            self.assertIsNotNone(phase.frames[0].pose)

    def test_numpy_reconstruction_writes_ply_and_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = create_mock_project(Path(temporary) / "scan", phase_count=2, frame_count=4)
            result = reconstruct_project(root, engine="numpy")
            self.assertGreater(result["pointCount"], 100)
            self.assertTrue((root / "outputs" / "room-cloud.ply").exists())
            self.assertTrue((root / "outputs" / "room-mesh.obj").exists())
            self.assertTrue((root / "outputs" / "room-mesh.mtl").exists())
            self.assertTrue((root / "outputs" / "room-texture.png").exists())
            mesh_preview = (root / "outputs" / "room-mesh.preview.bin").read_bytes()
            current_dataset = root / "outputs" / "cache" / "datasets" / "current.json"
            self.assertFalse(current_dataset.exists())
            self.assertEqual(len(result["datasetFingerprint"]), 24)
            self.assertGreater(result["meshTriangleCount"], 100)
            self.assertFalse(result["meshCacheHit"])
            obj_lines = (root / "outputs" / "room-mesh.obj").read_text(encoding="utf-8").splitlines()
            self.assertEqual(sum(line.startswith("v ") for line in obj_lines), result["meshVertexCount"])
            self.assertEqual(sum(line.startswith("vn ") for line in obj_lines), result["meshVertexCount"])
            self.assertEqual(
                sum(line.startswith("vt ") for line in obj_lines),
                result["meshTextureVertexCount"],
            )
            self.assertEqual(sum(line.startswith("f ") for line in obj_lines), result["meshTriangleCount"])
            self.assertEqual(mesh_preview[:4], b"K2M1")
            preview_vertex_count, preview_index_count = struct.unpack("<II", mesh_preview[4:12])
            self.assertEqual(preview_vertex_count, result["meshRenderVertexCount"])
            self.assertEqual(preview_index_count, result["meshTriangleCount"] * 3)
            self.assertEqual(len(mesh_preview), 12 + preview_vertex_count * 20 + preview_index_count * 4)
            self.assertIn(result["meshFusionMethod"], {"tsdf", "welded_depth"})
            self.assertEqual(result["textureSource"], "best_view_native_rgb_texel_projection")
            self.assertGreaterEqual(result["textureCoveragePercent"], 99.0)
            self.assertEqual((root / "outputs" / "room-texture.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            with (root / "outputs" / "camera-poses.json").open("r", encoding="utf-8") as handle:
                camera_frames = json.load(handle)
            self.assertEqual(len(camera_frames), 8)
            self.assertTrue(any(frame["textureFrame"] for frame in camera_frames))
            with (root / "outputs" / "preview.json").open("r", encoding="utf-8") as handle:
                preview = json.load(handle)
            self.assertGreater(len(preview), 100)
            project = read_project(root)
            self.assertEqual(project["processingStatus"], "complete")
            self.assertEqual(project["meshTriangleCount"], result["meshTriangleCount"])
            with (root / "outputs" / "progress.json").open("r", encoding="utf-8") as handle:
                progress = json.load(handle)
            self.assertEqual(progress["stage"], "Complete")
            self.assertEqual(progress["progress"], 1.0)
            self.assertEqual(progress["pointCount"], result["pointCount"])
            self.assertEqual(result["computeBackend"], "NumPy CPU")
            self.assertGreaterEqual(result["processingSeconds"], 0)
            self.assertIn("Exporting", result["stageTimingsSeconds"])
            self.assertEqual(project["processingBackend"], "NumPy CPU")

            cached_result = reconstruct_project(root, engine="numpy")
            self.assertTrue(cached_result["meshCacheHit"])
            self.assertEqual(cached_result["meshTriangleCount"], result["meshTriangleCount"])

    def test_gaussian_target_builds_the_canonical_posed_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = create_mock_project(Path(temporary) / "scan", phase_count=1, frame_count=2)

            result = reconstruct_project(root, engine="numpy", targets=("gaussian_splat",))

            current = root / "outputs" / "cache" / "datasets" / "current.json"
            pointer = json.loads(current.read_text(encoding="utf-8"))
            dataset_root = current.parent / pointer["path"]
            dataset = json.loads((dataset_root / "dataset.json").read_text(encoding="utf-8"))
            self.assertEqual(dataset["fingerprint"], result["datasetFingerprint"])
            self.assertEqual(len(dataset["frames"]), 2)
            self.assertTrue((dataset_root / "initialization.ply").exists())

    def test_depth_mesh_welding_merges_positions_and_duplicate_faces(self) -> None:
        first_vertices = np.asarray(
            [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.float32
        )
        second_vertices = first_vertices + np.asarray([0.004, 0, 0], dtype=np.float32)
        triangles = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)

        vertices, fused = _weld_depth_meshes(
            [(first_vertices, triangles), (second_vertices, triangles)],
            voxel_size_m=0.01,
        )

        self.assertEqual(vertices.shape, (4, 3))
        self.assertEqual(fused.shape, (2, 3))
        np.testing.assert_allclose(vertices[:, 0].min(), 0.002, atol=1e-6)

    def test_triangle_charts_share_blended_edge_colors(self) -> None:
        colors = np.asarray(
            [[240, 20, 20], [20, 240, 20], [20, 20, 240], [220, 220, 220]],
            dtype=np.uint8,
        )
        triangles = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)

        atlas, uvs, _ = _bake_triangle_atlas(colors, triangles)

        def atlas_color(uv: np.ndarray) -> np.ndarray:
            x = min(atlas.shape[1] - 1, int(uv[0] * atlas.shape[1]))
            y = min(atlas.shape[0] - 1, int((1.0 - uv[1]) * atlas.shape[0]))
            return atlas[y, x]

        np.testing.assert_allclose(atlas_color(uvs[0, 1]), atlas_color(uvs[1, 0]), atol=1)
        np.testing.assert_allclose(atlas_color(uvs[0, 2]), atlas_color(uvs[1, 2]), atol=1)

    def test_triangle_charts_retain_native_image_detail_between_vertices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            depth_path = root / "depth.u16"
            color_path = root / "color.rgb"
            np.full((8, 8), 2000, dtype="<u2").tofile(depth_path)
            yy, xx = np.indices((8, 8))
            checker = (((xx + yy) % 2) * 255).astype(np.uint8)
            np.repeat(checker[..., None], 3, axis=2).tofile(color_path)
            camera = CameraModel(8, 8, 4.0, 4.0, 3.5, 3.5, 1000.0, 4.0)
            record = FrameRecord(0, 0, depth_path, color_path, None, None, np.eye(4))
            phase = PhaseData(root, {}, camera, None, np.eye(4), [record], [])
            frame = PosedFrame("test", "test", phase, 0, np.eye(4), (1.0, 1.0, 1.0), False)
            image_pixels = np.asarray([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]])
            z = np.full(3, 2.0)
            vertices = np.column_stack(
                (
                    (image_pixels[:, 0] - camera.cx) * z / camera.fx,
                    (image_pixels[:, 1] - camera.cy) * z / camera.fy,
                    z,
                )
            ).astype(np.float32)
            triangles = np.asarray([[0, 1, 2]], dtype=np.int64)

            atlas, _, _ = _bake_triangle_atlas(
                np.full((3, 3), 128, dtype=np.uint8),
                triangles,
                vertices=vertices,
                frames=[frame],
                triangle_frames=np.asarray([0], dtype=np.int16),
                exposure_gains=np.asarray([1.0], dtype=np.float32),
            )

            self.assertGreater(int(np.ptp(atlas[..., 0])), 150)

    def test_numpy_reconstruction_honors_one_mm_point_spacing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = create_mock_project(Path(temporary) / "scan", phase_count=1, frame_count=2)
            project_path = root / "project.json"
            with project_path.open("r", encoding="utf-8") as handle:
                project = json.load(handle)
            project["settings"]["voxelSizeMm"] = 1
            with project_path.open("w", encoding="utf-8") as handle:
                json.dump(project, handle)

            result = reconstruct_project(root, engine="numpy")

            self.assertEqual(result["voxelSizeM"], 0.001)

    def test_open3d_local_phase_cache_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = create_mock_project(Path(temporary) / "scan", phase_count=1, frame_count=4)
            project = read_project(root)
            phase_root = root / "phases" / project["phases"][0]["id"]
            manifest_path = phase_root / "phase.json"
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["poseSource"] = "kinect_fusion"
            with manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            phase = read_phase(phase_root)
            cache_root = root / "outputs" / "cache"
            preview_path = root / "outputs" / "build-preview.json"
            first_messages: list[str] = []
            first_points, _, _ = reconstruct_open3d(
                [phase],
                0.03,
                lambda stage, detail, *args, **kwargs: first_messages.append(detail),
                preview_path,
                requested_device="cpu",
                cache_root=cache_root,
            )
            self.assertEqual(len(list((cache_root / "local-phases").glob("*.npz"))), 1)
            second_messages: list[str] = []
            second_points, _, _ = reconstruct_open3d(
                [phase],
                0.03,
                lambda stage, detail, *args, **kwargs: second_messages.append(detail),
                preview_path,
                requested_device="cpu",
                cache_root=cache_root,
            )
            self.assertEqual(first_points.shape, second_points.shape)
            self.assertTrue(any("Reused tracking" in value for value in second_messages))

    def test_tensor_tsdf_and_icp_run_on_tensor_api(self) -> None:
        import open3d as o3d

        with tempfile.TemporaryDirectory() as temporary:
            root = create_mock_project(Path(temporary) / "scan", phase_count=1, frame_count=3)
            project = read_project(root)
            phase = read_phase(root / "phases" / project["phases"][0]["id"])
            entries = [
                (phase, index, np.linalg.inv(phase.frames[index].pose))
                for index in range(3)
            ]
            backend = ComputeBackend(
                "test-tensor-cpu",
                "Tensor CPU test",
                o3d.core.Device("CPU:0"),
                True,
            )
            odometry = tensor_odometry(
                o3d,
                tensor_rgbd(o3d, phase, 0, backend.device),
                tensor_rgbd(o3d, phase, 1, backend.device),
                phase,
                np.eye(4),
                0.09,
                backend,
            )
            self.assertTrue(np.isfinite(odometry).all())
            cloud = _tensor_tsdf(o3d, entries, 0.03, 0.12, backend, None)
            self.assertGreater(len(cloud.points), 100)
            source = cloud.voxel_down_sample(0.05)
            source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.15, 30))
            target = o3d.geometry.PointCloud(source)
            target.translate((0.02, 0.0, 0.0))
            target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.15, 30))
            refined, _, _ = tensor_refine_registration(
                o3d, source, target, np.eye(4), backend
            )
            self.assertGreater(refined.fitness, 0.9)
            self.assertAlmostEqual(refined.transformation[0, 3], 0.02, places=3)

    def test_tensor_surfel_merge_streams_into_unique_voxels(self) -> None:
        import open3d as o3d

        with tempfile.TemporaryDirectory() as temporary:
            root = create_mock_project(Path(temporary) / "scan", phase_count=1, frame_count=3)
            project = read_project(root)
            phase = read_phase(root / "phases" / project["phases"][0]["id"])
            entries = [
                (phase, index, np.linalg.inv(phase.frames[index].pose))
                for index in range(3)
            ]
            backend = ComputeBackend(
                "test-tensor-cpu",
                "Tensor CPU test",
                o3d.core.Device("CPU:0"),
                True,
            )
            merged: list[tuple[int, bool]] = []

            cloud = _tensor_surfel_merge(
                o3d,
                entries,
                0.01,
                backend,
                lambda index, repeated: merged.append((index, repeated)),
            )

            points = np.asarray(cloud.points)
            voxel_keys = np.floor(points / 0.01).astype(np.int32)
            self.assertGreater(len(points), 100)
            self.assertEqual(len(np.unique(voxel_keys, axis=0)), len(points))
            self.assertEqual(merged, [(0, False), (1, False), (2, False)])

    def test_empty_project_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = create_mock_project(Path(temporary) / "scan", phase_count=0)
            with self.assertRaisesRegex(ValueError, "Capture at least one phase"):
                reconstruct_project(root, engine="numpy")

    def test_failed_empty_phase_does_not_block_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = create_mock_project(Path(temporary) / "scan", phase_count=1, frame_count=4)
            project_path = root / "project.json"
            with project_path.open("r", encoding="utf-8") as handle:
                project = json.load(handle)
            project["phases"].insert(
                0,
                {
                    "id": "failed-empty-phase",
                    "name": "Failed phase",
                    "status": "failed",
                    "frameCount": 0,
                },
            )
            with project_path.open("w", encoding="utf-8") as handle:
                json.dump(project, handle)
            result = reconstruct_project(root, engine="numpy")
            self.assertGreater(result["pointCount"], 100)

    def test_phase_loads_calibrated_imu_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = create_mock_project(Path(temporary) / "scan", phase_count=1, frame_count=2)
            project = read_project(root)
            phase_root = root / "phases" / project["phases"][0]["id"]
            manifest_path = phase_root / "phase.json"
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["imu"] = {
                "path": "imu.csv",
                "coordinateFrame": "depth_camera",
                "accelerationUnit": "m/s^2",
                "angularVelocityUnit": "rad/s",
            }
            with manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            (phase_root / "imu.csv").write_text(
                "timestamp_us,type,x,y,z,temperature_c\n"
                "0,gyro,0,0,0.25,31.0\n"
                "0,accel,0,-9.81,0,31.0\n",
                encoding="utf-8",
            )
            phase = read_phase(phase_root)
            self.assertEqual(len(phase.imu_samples), 2)
            self.assertEqual(phase.imu_samples[0].kind, "gyro")

    def test_legacy_aligned_rgb_uses_identity_native_rgb_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = create_mock_project(Path(temporary) / "scan", phase_count=1, frame_count=1)
            project = read_project(root)
            phase = read_phase(root / "phases" / project["phases"][0]["id"])
            self.assertIsNone(phase.rgb_camera)
            np.testing.assert_array_equal(phase.rgb_from_depth, np.eye(4))
            image = load_source_rgb(phase.frames[0], phase)
            self.assertEqual(image.shape, (phase.camera.height, phase.camera.width, 3))
            zbuffer, uv_map, visibility = rgb_depth_zbuffer(
                load_depth(phase.frames[0], phase.camera), phase
            )
            self.assertEqual(zbuffer.shape, (phase.camera.height, phase.camera.width))
            self.assertEqual(uv_map.shape, (phase.camera.height, phase.camera.width, 2))
            self.assertGreater(int(visibility.sum()), 100)

    def test_phase_v3_loads_native_rgb_calibration_and_timestamp(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = create_mock_project(Path(temporary) / "scan", phase_count=1, frame_count=1)
            project = read_project(root)
            phase_root = root / "phases" / project["phases"][0]["id"]
            manifest_path = phase_root / "phase.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update(
                schemaVersion=3,
                rgbCamera={
                    "width": 96,
                    "height": 72,
                    "fx": 88.0,
                    "fy": 88.0,
                    "cx": 47.5,
                    "cy": 35.5,
                    "model": "brown_conrady",
                    "distortion": [0, 0, 0, 0, 0],
                },
                rgbFromDepth=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                sourceRgb={"format": "jpeg", "quality": 92, "nativeResolution": True},
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            rgb_root = phase_root / "rgb"
            rgb_root.mkdir()
            Image.fromarray(np.full((72, 96, 3), [20, 80, 160], dtype=np.uint8)).save(
                rgb_root / "000000.jpg", quality=95
            )
            frames_path = phase_root / "frames.csv"
            with frames_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
                original_fields = list(rows[0])
            fields = original_fields[:4] + ["rgb_path", "rgb_timestamp_us"] + original_fields[4:]
            rows[0]["rgb_path"] = "rgb/000000.jpg"
            rows[0]["rgb_timestamp_us"] = "123456"
            with frames_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            phase = read_phase(phase_root)
            self.assertEqual(phase.rgb_camera.width, 96)
            self.assertEqual(phase.frames[0].rgb_timestamp_us, 123456)
            self.assertEqual(load_source_rgb(phase.frames[0], phase).shape, (72, 96, 3))

    def test_gyro_prior_maps_previous_camera_into_current_camera(self) -> None:
        samples = [
            ImuSample(timestamp, "gyro", np.asarray([0.0, 0.0, 1.0]), 30.0)
            for timestamp in (0, 50_000, 100_000)
        ]
        prior = odometry_rotation_prior(samples, 0, 100_000)
        self.assertIsNotNone(prior)
        assert prior is not None
        angle = math.atan2(prior[1, 0], prior[0, 0])
        self.assertAlmostEqual(angle, -0.1, places=3)


if __name__ == "__main__":
    unittest.main()
