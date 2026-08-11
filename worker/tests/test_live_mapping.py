from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from scanlan.live_mapping import (
    AdaptiveBudgetController,
    CoverageField,
    SubmapLimits,
    tracking_colors,
)
from scanlan.compute import ComputeBackend
from scanlan.realtime import AlignmentQuality, LiveSubmapManager, TrackedFrame
from scanlan.stream import RgbdFrame, StreamCamera


class LiveMappingTests(unittest.TestCase):
    def test_memory_budget_derives_a_bounded_sparse_block_pool(self) -> None:
        limits = SubmapLimits.from_mebibytes(1024)

        self.assertLess(limits.block_capacity, 24_000)
        self.assertLess(limits.rollover_block_count, limits.block_capacity)
        self.assertEqual(limits.gpu_budget_bytes, 1024**3)

    def test_adaptive_controller_degrades_fast_and_recovers_slowly(self) -> None:
        controller = AdaptiveBudgetController()

        controller.observe(map_latency_ms=500, mapping_queue_ratio=1.0, memory_ratio=0.2)
        self.assertGreaterEqual(controller.level, 5)
        self.assertFalse(controller.mesh_enabled)
        for _ in range(19):
            controller.observe(map_latency_ms=10, mapping_queue_ratio=0, memory_ratio=0.2)
        self.assertGreaterEqual(controller.level, 5)
        controller.observe(map_latency_ms=10, mapping_queue_ratio=0, memory_ratio=0.2)
        self.assertEqual(controller.level, 4)

    def test_coverage_distinguishes_single_and_repeated_observations(self) -> None:
        camera = StreamCamera(8, 8, 8.0, 8.0, 3.5, 3.5, 1000.0, 0.25, 5.0)
        depth = np.full((8, 8), 1000, dtype=np.uint16)
        field = CoverageField(0.25)
        for sequence in range(3):
            frame = RgbdFrame(sequence, sequence, sequence, camera, depth, None, None, None)
            field.observe(frame, np.eye(4), 0.9)

        summary = field.summary(0.9)

        self.assertGreater(summary.observed_ratio, 0.9)
        self.assertLess(summary.single_view_ratio, 0.1)

    def test_tracking_overlay_makes_searching_geometry_red(self) -> None:
        colors = tracking_colors(2, "searching", 0.0)

        self.assertEqual(colors.tolist(), [[230, 68, 84], [230, 68, 84]])

    def test_coverage_field_moves_with_a_loop_correction(self) -> None:
        camera = StreamCamera(8, 8, 8.0, 8.0, 3.5, 3.5, 1000.0, 0.25, 5.0)
        frame = RgbdFrame(
            0,
            0,
            0,
            camera,
            np.full((8, 8), 1000, dtype=np.uint16),
            None,
            None,
            None,
        )
        field = CoverageField(0.25)
        field.observe(frame, np.eye(4), 0.9)
        before = set(field.cells)
        correction = np.eye(4)
        correction[0, 3] = 0.5

        field.transform(correction)

        self.assertEqual(
            {key[0] for key in field.cells},
            {key[0] + 2 for key in before},
        )

    def test_travel_rollover_keeps_completed_submap_on_host(self) -> None:
        import open3d as o3d

        camera = StreamCamera(32, 24, 30.0, 30.0, 15.5, 11.5, 1000.0, 0.25, 5.0)
        depth = np.full((24, 32), 1000, dtype=np.uint16)
        color = np.full((24, 32, 3), 128, dtype=np.uint8)
        quality = AlignmentQuality(True, 1.0, 1.0, 0.0, depth.size, "captured")
        limits = replace(
            SubmapLimits.from_mebibytes(256),
            maximum_distance_m=0.10,
        )
        manager = LiveSubmapManager(
            o3d,
            0.02,
            ComputeBackend("test", "CPU", o3d.core.Device("CPU:0"), False),
            "points",
            limits,
        )

        first = RgbdFrame(0, 0, 0, camera, depth, color, None, None)
        second = RgbdFrame(1, 100_000, 100_000, camera, depth, color, None, None)
        first_pose = np.eye(4)
        second_pose = np.eye(4)
        second_pose[0, 3] = -0.20
        manager.integrate(
            TrackedFrame(first, first_pose, quality, True, "tracking", "first")
        )
        manager.integrate(
            TrackedFrame(second, second_pose, quality, True, "tracking", "second")
        )

        self.assertEqual(len(manager.completed), 1)
        self.assertIsNotNone(manager.active)
        self.assertEqual(manager.completed[0].descriptor.resident, "host")
        self.assertEqual(manager.rollover_count, 1)

    def test_verified_loop_moves_submaps_without_duplicating_geometry(self) -> None:
        import open3d as o3d

        camera = StreamCamera(64, 48, 60.0, 60.0, 31.5, 23.5, 1000.0, 0.25, 5.0)
        depth = np.full((48, 64), 1000, dtype=np.uint16)
        color = np.full((48, 64, 3), 128, dtype=np.uint8)
        quality = AlignmentQuality(True, 1.0, 1.0, 0.0, depth.size, "captured")
        limits = replace(
            SubmapLimits.from_mebibytes(256),
            maximum_distance_m=0.10,
        )
        manager = LiveSubmapManager(
            o3d,
            0.02,
            ComputeBackend("test", "CPU", o3d.core.Device("CPU:0"), False),
            "points",
            limits,
        )
        for sequence, camera_x in enumerate((0.0, 0.20, 0.02)):
            world_to_camera = np.eye(4)
            world_to_camera[0, 3] = -camera_x
            frame = RgbdFrame(
                sequence,
                sequence * 100_000,
                sequence * 100_000,
                camera,
                depth,
                color,
                None,
                None,
            )
            manager.integrate(
                TrackedFrame(
                    frame, world_to_camera, quality, True, "tracking", "closed loop"
                )
            )
        manager.complete_active("capture stop")

        expected_points = sum(len(submap.points) for submap in manager.completed)
        combined_points, _ = manager.world_points()
        self.assertEqual(len(combined_points), expected_points)
        self.assertEqual(manager.correction_count, 1)
        self.assertEqual(len(manager.pose_graph.loops), 1)
        self.assertTrue(manager.loop_events[-1]["accepted"])


if __name__ == "__main__":
    unittest.main()
