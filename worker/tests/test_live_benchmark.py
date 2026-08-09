from __future__ import annotations

import unittest
from pathlib import Path

from scanlan.live_benchmark import _latency_summary, summarize_live_benchmark


class LiveBenchmarkTests(unittest.TestCase):
    def test_latency_summary_interpolates_p95(self) -> None:
        summary = _latency_summary([10.0, 20.0, 30.0, 40.0])

        self.assertEqual(summary["samples"], 4)
        self.assertEqual(summary["medianMs"], 25.0)
        self.assertEqual(summary["p95Ms"], 38.5)
        self.assertEqual(summary["maximumMs"], 40.0)

    def test_summary_protects_rejected_frames_and_reports_provisional_output(self) -> None:
        report = summarize_live_benchmark(
            capture=Path("capture"),
            mode="mesh",
            voxel_size_m=0.01,
            live_map_mib=768,
            device="cuda",
            paced=True,
            frame_count=3,
            source_duration_seconds=0.2,
            wall_seconds=0.3,
            statuses=[
                {"state": "tracking"},
                {
                    "state": "complete",
                    "backend": "CUDA",
                    "processedFrames": 3,
                    "acceptedFrames": 2,
                    "rejectedFrames": 1,
                    "integratedFrames": 2,
                    "trackingQueueDrops": 0,
                    "mappingDrops": 0,
                },
            ],
            pose_latencies_ms=[4.0, 8.0],
            point_latencies_ms=[20.0],
            mesh_latencies_ms=[30.0],
            point_snapshots=1,
            mesh_snapshots=1,
            final_point_count=40,
            final_triangle_count=12,
            coverage_snapshots=1,
            tracking_snapshots=1,
            coverage_message={"observedRatio": 0.5},
            submap_message={"submaps": [{"id": "submap-0"}]},
            working_set_samples=[10 * 1024 * 1024],
            gpu_samples=[20 * 1024 * 1024],
            journal_entries=[
                {"accepted": True, "integrated": True, "state": "tracking"},
                {"accepted": False, "integrated": False, "state": "searching"},
            ],
            exit_code=0,
        )

        self.assertTrue(report["tracking"]["integrationFrozenForEveryRejectedFrame"])
        self.assertTrue(report["preview"]["provisionalAvailableAfterStop"])
        self.assertEqual(report["runtime"]["peakGpuMemoryMiB"], 20.0)
        self.assertEqual(report["liveMap"]["finalSubmapCount"], 1)
        self.assertEqual(report["configuration"]["liveMapMemoryMiB"], 768)


if __name__ == "__main__":
    unittest.main()
