from __future__ import annotations

import io
import unittest
from dataclasses import replace

import numpy as np

from scanlan.stream import (
    LatestFrameQueue,
    RgbdFrame,
    RgbdStreamError,
    StreamCamera,
    decode_rgbd_frame,
    encode_rgbd_frame,
    read_rgbd_frame,
    reject_depth_speckles,
)


def frame(sequence: int = 7) -> RgbdFrame:
    camera = StreamCamera(4, 3, 220.0, 221.0, 1.5, 1.0, 1000.0, 0.25, 5.0)
    return RgbdFrame(
        sequence=sequence,
        depth_timestamp_us=123_000 + sequence,
        color_timestamp_us=123_100 + sequence,
        camera=camera,
        depth=np.arange(12, dtype=np.uint16).reshape(3, 4) * 100,
        color=np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
        gyro_delta_xyzw=np.asarray([0.0, 0.01, 0.0, 0.99995]),
        camera_to_world=np.asarray(
            [[1, 0, 0, 0.1], [0, 1, 0, 0.2], [0, 0, 1, 0.3], [0, 0, 0, 1]],
            dtype=np.float64,
        ),
        mirror_x=True,
    )


class RgbdStreamTests(unittest.TestCase):
    def test_round_trip_preserves_calibration_pose_and_pixels(self) -> None:
        original = frame()
        decoded = decode_rgbd_frame(encode_rgbd_frame(original))

        self.assertEqual(decoded.sequence, original.sequence)
        self.assertEqual(decoded.camera, original.camera)
        np.testing.assert_array_equal(decoded.depth, original.depth)
        np.testing.assert_array_equal(decoded.color, original.color)
        np.testing.assert_allclose(decoded.gyro_delta_xyzw, original.gyro_delta_xyzw, atol=1e-6)
        np.testing.assert_allclose(decoded.camera_to_world, original.camera_to_world, atol=1e-6)
        self.assertTrue(decoded.mirror_x)

    def test_truncated_payload_is_rejected(self) -> None:
        payload = encode_rgbd_frame(frame())
        with self.assertRaisesRegex(RgbdStreamError, "bytes missing"):
            read_rgbd_frame(io.BytesIO(payload[:-17]))

    def test_latest_queue_discards_stale_frames_instead_of_blocking(self) -> None:
        frames = LatestFrameQueue(capacity=2)
        frames.put(frame(1))
        frames.put(frame(2))
        frames.put(frame(3))

        self.assertEqual(frames.dropped, 1)
        self.assertEqual(frames.get().sequence, 2)
        self.assertEqual(frames.get().sequence, 3)

    def test_latest_queue_preserves_rotation_across_dropped_frames(self) -> None:
        frames = LatestFrameQueue(capacity=2)
        first = frame(1)
        second = frame(2)
        third = frame(3)
        first = replace(
            first,
            gyro_delta_xyzw=np.asarray(
                [0.0, 0.0, np.sin(0.05), np.cos(0.05)]
            ),
        )
        second = replace(
            second,
            gyro_delta_xyzw=np.asarray(
                [0.0, 0.0, np.sin(0.10), np.cos(0.10)]
            ),
        )
        frames.put(first)
        frames.put(second)
        frames.put(third)

        retained = frames.get()

        self.assertEqual(retained.sequence, 2)
        np.testing.assert_allclose(
            retained.gyro_delta_xyzw,
            [0.0, 0.0, np.sin(0.15), np.cos(0.15)],
            atol=1e-7,
        )

    def test_depth_filter_removes_speckles_but_keeps_supported_edges(self) -> None:
        camera = StreamCamera(5, 5, 4.0, 4.0, 2.0, 2.0, 1000.0, 0.2, 5.0)
        depth = np.full((5, 5), 2000, dtype=np.uint16)
        depth[2, 2] = 3500
        depth[1:4, 4] = 3000

        cleaned = reject_depth_speckles(depth, camera)

        self.assertEqual(cleaned[2, 2], 0)
        self.assertTrue(np.all(cleaned[1:4, 4] == 3000))


if __name__ == "__main__":
    unittest.main()
