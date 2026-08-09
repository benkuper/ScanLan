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
    ENGINE_CAMERA_POINTS,
    ENGINE_STATUS,
    AlignmentQuality,
    EngineMessageWriter,
    TrackedFrame,
    TrackingJournal,
    RealtimeTracker,
    _recovery_pose_is_credible,
    evaluate_depth_alignment,
    frame_point_cloud,
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

    def test_marginal_tracking_pose_is_never_fused_into_the_map(self) -> None:
        tracker = RealtimeTracker(
            None,
            ComputeBackend("test", "test", None, False),
            0.01,
        )

        self.assertFalse(
            tracker._quality_is_safe_to_integrate(
                AlignmentQuality(True, 0.80, 0.70, 0.012, 500, "tracking only")
            )
        )
        self.assertTrue(
            tracker._quality_is_safe_to_integrate(
                AlignmentQuality(True, 0.80, 0.90, 0.012, 500, "fusion safe")
            )
        )

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

    def test_camera_preview_has_a_distinct_engine_stream(self) -> None:
        stream = io.BytesIO()
        packet = point_packet(
            23,
            456_000,
            12.0,
            np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
            np.asarray([[10, 20, 30]], dtype=np.uint8),
        )
        EngineMessageWriter(stream).write(ENGINE_CAMERA_POINTS, 23, packet)
        stream.seek(0)

        kind, sequence, payload = read_engine_message(stream)

        self.assertEqual((kind, sequence), (ENGINE_CAMERA_POINTS, 23))
        self.assertEqual(payload, packet)

    def test_fresh_capture_tracker_uses_first_usable_frame_as_identity(self) -> None:
        class LightweightTracker(RealtimeTracker):
            def _representation(self, frame: RgbdFrame) -> int:
                return frame.sequence

        frame = RgbdFrame(23, 123_000, 123_500, self.camera, self.depth, None, None, None)
        tracker = LightweightTracker(
            None, ComputeBackend("test", "test", None, False), 0.01
        )

        tracked = tracker.track(frame)

        self.assertIsNotNone(tracked.world_to_camera)
        self.assertTrue(np.allclose(tracked.world_to_camera, np.eye(4)))
        self.assertTrue(tracked.integrate)
        self.assertEqual(tracked.detail, "RGB-D odometry initialized")

    def test_single_rgbd_frame_provides_an_immediate_preview_fallback(self) -> None:
        color = np.zeros((120, 160, 3), dtype=np.uint8)
        color[..., 0] = 25
        color[..., 1] = 50
        color[..., 2] = 75
        frame = RgbdFrame(0, 0, 0, self.camera, self.depth, color, None, None)

        points, colors = frame_point_cloud(frame)

        self.assertEqual(points.shape, (120 * 160, 3))
        self.assertTrue(np.allclose(points[:, 2], -2.0))
        self.assertEqual(colors[0].tolist(), [25, 50, 75])

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

    def test_tracker_relocalizes_to_a_saved_quality_gated_anchor(self) -> None:
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
                    transform[0, 3] = 0.10
                    return True, transform
                if source == 1 and current >= 2:
                    return False, transform
                return source == 0 and current >= 2, transform

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

        tentative_one = tracker.track(frame(2))
        tentative_two = tracker.track(frame(3))
        recovered = tracker.track(frame(4))

        self.assertIsNone(tentative_one.world_to_camera)
        self.assertIsNone(tentative_two.world_to_camera)
        self.assertIsNotNone(recovered.world_to_camera)
        self.assertIn("relocalized", recovered.detail)

    def test_tracker_can_recover_to_a_distant_saved_anchor(self) -> None:
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
                    transform[0, 3] = 0.10
                    return True, transform
                if source == 0 and 2 <= current <= 4:
                    # A valid saved view that is intentionally far beyond the
                    # local 15 cm continuity gate around the lost pose.
                    transform[0, 3] = 0.60
                    return True, transform
                if (source, current) == (1, 5):
                    transform[0, 3] = 0.50
                    return True, transform
                return False, transform

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

        tracker = ScriptedTracker(
            None,
            ComputeBackend("test", "test", None, False),
            0.01,
        )
        tracker.track(frame(0))
        tracker.track(frame(1))

        tentative_one = tracker.track(frame(2))
        tentative_two = tracker.track(frame(3))
        recovered = tracker.track(frame(4))
        resumed = tracker.track(frame(5))

        self.assertIsNone(tentative_one.world_to_camera)
        self.assertIsNone(tentative_two.world_to_camera)
        self.assertIsNotNone(recovered.world_to_camera)
        self.assertFalse(recovered.integrate)
        assert recovered.world_to_camera is not None
        self.assertAlmostEqual(float(recovered.world_to_camera[0, 3]), 0.60)
        self.assertIn("relocalization locked", recovered.detail)
        self.assertTrue(resumed.integrate)

    def test_saved_anchor_recovery_overrides_a_misleading_local_solve(self) -> None:
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
                    transform[0, 3] = 0.10
                    return True, transform
                if source == 1 and current >= 2:
                    # This superficially valid local solve lands far from the
                    # last accepted pose and used to suppress relocalization.
                    transform[0, 3] = 0.50
                    return True, transform
                if source == 0 and current >= 2:
                    transform[0, 3] = 0.40
                    return True, transform
                return False, transform

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

        tracker = ScriptedTracker(
            None,
            ComputeBackend("test", "test", None, False),
            0.01,
        )
        tracker.track(frame(0))
        tracker.track(frame(1))
        # Enter recovery before presenting the ambiguous frame pair.
        tracker.rejected_since_accept = 1

        tentative_one = tracker.track(frame(2))
        tentative_two = tracker.track(frame(3))
        recovered = tracker.track(frame(4))

        self.assertIsNone(tentative_one.world_to_camera)
        self.assertIsNone(tentative_two.world_to_camera)
        self.assertIsNotNone(recovered.world_to_camera)
        assert recovered.world_to_camera is not None
        self.assertAlmostEqual(float(recovered.world_to_camera[0, 3]), 0.40)
        self.assertIn("relocalization locked", recovered.detail)
        self.assertFalse(recovered.integrate)

    def test_duplicate_wall_recovery_jump_is_rejected(self) -> None:
        previous = np.eye(4)
        duplicate = np.eye(4)
        angle = np.deg2rad(43.0)
        duplicate[:3, :3] = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        duplicate[0, 3] = 0.463

        self.assertFalse(
            _recovery_pose_is_credible(previous, duplicate, previous)
        )

    def test_relocalization_bank_preserves_and_searches_the_whole_capture(self) -> None:
        tracker = RealtimeTracker(
            None,
            ComputeBackend("test", "test", None, False),
            0.01,
        )

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

        for sequence in range(80):
            pose = np.eye(4)
            pose[0, 3] = sequence * 0.10
            tracker._remember_anchor(frame(sequence), sequence, pose)

        tracker.previous_frame = frame(79)
        saved_sequences = {
            anchor.frame.sequence
            for anchor in tracker.anchors
            if anchor.frame.sequence != 79
        }
        searched_sequences: set[int] = set()
        for _ in range(len(tracker.anchors)):
            searched_sequences.update(
                anchor.frame.sequence for anchor in tracker._relocalization_anchors()
            )

        self.assertIn(0, saved_sequences)
        self.assertIn(0, searched_sequences)
        self.assertEqual(searched_sequences, saved_sequences)

    def test_tracker_uses_global_features_after_local_overlap_is_lost(self) -> None:
        class ScriptedTracker(RealtimeTracker):
            def _representation(self, frame: RgbdFrame) -> int:
                return frame.sequence

            def _feature_geometry(self, frame: RgbdFrame) -> tuple[None, None]:
                del frame
                return None, None

            def _feature_relocalization_transform(
                self,
                anchor: object,
                current_geometry: tuple[None, None],
            ) -> np.ndarray | None:
                del current_geometry
                if anchor.frame.sequence != 0:
                    return None
                transform = np.eye(4)
                transform[0, 3] = 0.20
                return transform

            def _odometry(
                self,
                source: int,
                current: int,
                frame: RgbdFrame,
                initial: np.ndarray,
            ) -> tuple[bool, np.ndarray]:
                del frame
                transform = np.eye(4)
                if (source, current) == (0, 1):
                    transform[0, 3] = 0.10
                    return True, transform
                if source == 0 and current >= 2 and initial[0, 3] > 0.15:
                    return True, initial
                return False, transform

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

        tracker = ScriptedTracker(
            None,
            ComputeBackend("test", "test", None, False),
            0.01,
        )
        tracker.track(frame(0))
        tracker.track(frame(1))

        tentative = [tracker.track(frame(sequence)) for sequence in range(2, 9)]
        recovered = tracker.track(frame(9))

        self.assertTrue(all(candidate.world_to_camera is None for candidate in tentative))
        self.assertIsNotNone(recovered.world_to_camera)
        self.assertIn("globally relocalized", recovered.detail)

    def test_tracker_uses_latest_keyframe_instead_of_accumulating_frame_drift(self) -> None:
        class ScriptedTracker(RealtimeTracker):
            calls: list[tuple[int, int]]

            def __init__(self, *args: object) -> None:
                super().__init__(*args)
                self.calls = []

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
                self.calls.append((source, current))
                transform = np.eye(4)
                if (source, current) == (0, 1):
                    transform[0, 3] = 0.010
                elif (source, current) == (0, 2):
                    transform[0, 3] = 0.020
                elif (source, current) == (1, 2):
                    transform[0, 3] = 0.015
                else:
                    return False, transform
                return True, transform

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
        tracker.track(frame(0))
        tracker.track(frame(1))
        tracked = tracker.track(frame(2))

        self.assertIn((0, 2), tracker.calls)
        self.assertNotIn((1, 2), tracker.calls)
        self.assertIsNotNone(tracked.world_to_camera)
        assert tracked.world_to_camera is not None
        self.assertAlmostEqual(float(tracked.world_to_camera[0, 3]), 0.020)
        self.assertIn("Keyframe", tracked.detail)
        self.assertEqual(tracked.state, "tracking")

    def test_tracker_waits_for_usable_depth_before_initializing(self) -> None:
        class CapturedRepresentationTracker(RealtimeTracker):
            def _representation(self, frame: RgbdFrame) -> int:
                return frame.sequence

        empty = RgbdFrame(
            0,
            0,
            0,
            self.camera,
            np.zeros_like(self.depth),
            None,
            None,
            None,
        )
        usable = RgbdFrame(
            1,
            100_000,
            100_000,
            self.camera,
            self.depth,
            None,
            None,
            None,
        )
        tracker = CapturedRepresentationTracker(
            None,
            ComputeBackend("test", "test", None, False),
            0.01,
        )

        waiting = tracker.track(empty)
        initialized = tracker.track(usable)

        self.assertIsNone(waiting.world_to_camera)
        self.assertIn("Waiting for usable depth", waiting.detail)
        self.assertIsNotNone(initialized.world_to_camera)
        self.assertIs(tracker.previous_frame, usable)

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
