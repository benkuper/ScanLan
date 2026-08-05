from __future__ import annotations

import io
import json
import struct
import tempfile
from pathlib import Path
import unittest

import numpy as np

from scanlan.compute import ComputeBackend
from scanlan.realtime import (
    ENGINE_STATUS,
    AlignmentQuality,
    EngineMessageWriter,
    TrackedFrame,
    TrackingJournal,
    RealtimeTracker,
    evaluate_depth_alignment,
    mesh_packet,
    point_packet,
    read_engine_message,
)
from scanlan.stream import RgbdFrame, StreamCamera


class RealtimeQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = StreamCamera(160, 120, 145.0, 145.0, 79.5, 59.5, 1000.0, 0.25, 5.0)
        self.depth = np.full((120, 160), 2000, dtype=np.uint16)

    def test_identical_depth_accepts_identity_alignment(self) -> None:
        quality = evaluate_depth_alignment(
            self.depth,
            self.depth,
            self.camera,
            np.eye(4),
            stride=4,
        )

        self.assertTrue(quality.accepted)
        self.assertGreater(quality.overlap, 0.95)
        self.assertGreater(quality.inlier_ratio, 0.95)
        self.assertLess(quality.rmse_m, 1e-6)

    def test_geometrically_wrong_finite_transform_is_rejected(self) -> None:
        wrong = np.eye(4)
        wrong[2, 3] = 0.25
        quality = evaluate_depth_alignment(
            self.depth,
            self.depth,
            self.camera,
            wrong,
            stride=4,
        )

        self.assertFalse(quality.accepted)
        self.assertLess(quality.inlier_ratio, 0.1)

    def test_engine_messages_are_framed_for_mixed_status_and_geometry(self) -> None:
        stream = io.BytesIO()
        writer = EngineMessageWriter(stream)
        writer.status(11, {"active": True, "state": "tracking"})
        stream.seek(0)

        kind, sequence, payload = read_engine_message(stream)

        self.assertEqual(kind, ENGINE_STATUS)
        self.assertEqual(sequence, 11)
        self.assertEqual(json.loads(payload)["state"], "tracking")

    def test_point_packet_matches_the_desktop_binary_contract(self) -> None:
        points = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]], dtype=np.float32)
        colors = np.asarray([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)

        packet = point_packet(17, 123_456, 4.5, points, colors)

        magic, frame_count, timestamp_us, update_fps, point_count = struct.unpack(
            "<4sIQfI", packet[:24]
        )
        self.assertEqual((magic, frame_count, timestamp_us), (b"K2P1", 17, 123_456))
        self.assertAlmostEqual(update_fps, 4.5)
        self.assertEqual(len(packet), 24 + point_count * 15)

    def test_mesh_packet_reverses_winding_for_mirrored_display_axes(self) -> None:
        packet = mesh_packet(
            9,
            np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            np.full((3, 3), 180, dtype=np.uint8),
            np.asarray([[0, 1, 2]], dtype=np.uint32),
            flip_winding=True,
        )
        magic, frame_count, vertex_count, index_count = struct.unpack("<4sIII", packet[:16])
        self.assertEqual((magic, frame_count, vertex_count, index_count), (b"K2M2", 9, 3, 3))
        index_start = 16 + vertex_count * 15
        self.assertEqual(
            np.frombuffer(packet, dtype="<u4", offset=index_start).tolist(),
            [0, 2, 1],
        )

    def test_tracking_journal_persists_pose_and_quality_without_frame_pixels(self) -> None:
        frame = RgbdFrame(
            7,
            123_000,
            123_500,
            self.camera,
            self.depth,
            None,
            None,
            None,
        )
        tracked = TrackedFrame(
            frame,
            np.eye(4),
            AlignmentQuality(True, 0.8, 0.9, 0.004, 500, "accepted"),
            True,
            "tracking",
            "accepted",
        )
        with tempfile.TemporaryDirectory() as temporary:
            journal = TrackingJournal(Path(temporary))
            journal.append(tracked)
            journal.close()

            entry = json.loads((Path(temporary) / "tracking.jsonl").read_text())

        self.assertEqual(entry["sequence"], 7)
        self.assertEqual(len(entry["worldToCamera"]), 16)
        self.assertNotIn("depth", entry)

    def test_tracker_relocalizes_to_a_recent_quality_gated_anchor(self) -> None:
        class ScriptedTracker(RealtimeTracker):
            def _representation(self, frame: RgbdFrame) -> int:
                return frame.sequence

            def _odometry(
                self,
                source: int,
                current: int,
                frame: RgbdFrame,
                initial: np.ndarray,
            ) -> tuple[bool, np.ndarray]:
                del frame, initial
                transform = np.eye(4)
                if (source, current) == (0, 1):
                    transform[0, 3] = 0.05
                    return True, transform
                if (source, current) == (1, 2):
                    return False, transform
                return (source, current) == (0, 2), transform

        def frame(sequence: int) -> RgbdFrame:
            return RgbdFrame(
                sequence,
                sequence * 100_000,
                sequence * 100_000,
                self.camera,
                self.depth,
                None,
                None,
                None,
            )

        tracker = ScriptedTracker(None, ComputeBackend("test", "test", None, False), 0.01)
        self.assertTrue(tracker.track(frame(0)).integrate)
        self.assertTrue(tracker.track(frame(1)).integrate)

        recovered = tracker.track(frame(2))

        self.assertIsNotNone(recovered.world_to_camera)
        self.assertIn("relocalized", recovered.detail)

    def test_kinect_pose_must_agree_with_metric_depth(self) -> None:
        class CapturedPoseTracker(RealtimeTracker):
            def _representation(self, frame: RgbdFrame) -> int:
                return frame.sequence

        def frame(sequence: int, camera_to_world: np.ndarray) -> RgbdFrame:
            return RgbdFrame(
                sequence,
                sequence * 100_000,
                sequence * 100_000,
                self.camera,
                self.depth,
                None,
                None,
                camera_to_world,
            )

        tracker = CapturedPoseTracker(
            None,
            ComputeBackend("test", "test", None, False),
            0.01,
        )
        self.assertIsNotNone(tracker.track(frame(0, np.eye(4))).world_to_camera)
        wrong_pose = np.eye(4)
        wrong_pose[2, 3] = 0.10

        rejected = tracker.track(frame(1, wrong_pose))

        self.assertIsNone(rejected.world_to_camera)
        self.assertIn("pose rejected", rejected.detail)


if __name__ == "__main__":
    unittest.main()
