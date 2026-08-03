from __future__ import annotations

import csv
import json
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from scanlan.io import read_phase, read_project
from scanlan.live import MESH_MAGIC, POINT_MAGIC, live_reconstruct, mesh_packet, point_packet
from scanlan.mock_data import create_mock_project


class LiveReconstructionTests(unittest.TestCase):
    def test_worker_warms_before_capture_and_keeps_a_late_stop_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            phase_root = Path(temporary) / "phase"
            phase_root.mkdir()
            outcome: dict[str, object] = {}

            def run() -> None:
                outcome["result"] = live_reconstruct(
                    phase_root, 0.03, "points", "cpu", poll_seconds=0.01
                )

            thread = threading.Thread(target=run)
            thread.start()
            status_path = phase_root / "live-reconstruction.json"
            deadline = time.monotonic() + 10
            while not status_path.is_file() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(status_path.is_file())
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertTrue(status["active"])
            self.assertIn("ready", status["trackingStatus"].lower())

            stop_path = phase_root / "live-reconstruction.stop"
            stop_path.touch()
            thread.join(10)
            self.assertFalse(thread.is_alive())
            self.assertTrue(stop_path.exists())
            self.assertEqual(outcome["result"]["processedFrames"], 0)  # type: ignore[index]

    def test_point_packet_matches_the_desktop_binary_contract(self) -> None:
        points = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]], dtype=np.float32)
        colors = np.asarray([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
        packet = point_packet(17, 123_456, 4.5, points, colors)

        magic, frame_count, timestamp_us, update_fps, point_count = struct.unpack(
            "<4sIQfI", packet[:24]
        )
        self.assertEqual(magic, POINT_MAGIC)
        self.assertEqual(frame_count, 17)
        self.assertEqual(timestamp_us, 123_456)
        self.assertAlmostEqual(update_fps, 4.5)
        self.assertEqual(point_count, 2)
        self.assertEqual(len(packet), 24 + point_count * 15)

    def test_mesh_packet_reverses_winding_for_kinect_display_axes(self) -> None:
        packet = mesh_packet(
            9,
            np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            np.full((3, 3), 180, dtype=np.uint8),
            np.asarray([[0, 1, 2]], dtype=np.uint32),
            flip_winding=True,
        )
        magic, frame_count, vertex_count, index_count = struct.unpack("<4sIII", packet[:16])
        self.assertEqual((magic, frame_count, vertex_count, index_count), (MESH_MAGIC, 9, 3, 3))
        index_start = 16 + vertex_count * 12 + vertex_count * 3
        self.assertEqual(
            np.frombuffer(packet, dtype="<u4", offset=index_start).tolist(),
            [0, 2, 1],
        )

    def test_tracking_gap_frames_are_excluded_from_offline_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = create_mock_project(Path(temporary) / "scan", phase_count=1, frame_count=5)
            project = read_project(root)
            phase_root = root / "phases" / project["phases"][0]["id"]
            with (phase_root / "live-frame-selection.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(("index", "accepted", "reason"))
                writer.writerow((0, "true", "tracking initialized"))
                writer.writerow((1, "true", "tracking locked"))
                writer.writerow((2, "false", "tracking lost"))
                writer.writerow((3, "false", "return to known area"))
                writer.writerow((4, "true", "tracking recovered"))

            clean = read_phase(phase_root)
            raw = read_phase(phase_root, respect_live_selection=False)
            self.assertEqual([frame.index for frame in clean.frames], [0, 1, 4])
            self.assertEqual([frame.index for frame in raw.frames], [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
