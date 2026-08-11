from __future__ import annotations

import unittest

from scanlan_validation import BackendPolicyError, select_backend_policy


def benchmark(
    benchmark_id: str,
    backend: str,
    *,
    registration: float,
    residual: float,
    commercial: bool = True,
) -> dict:
    return {
        "benchmarkId": benchmark_id,
        "lane": "productionCamera",
        "backend": backend,
        "accepted": True,
        "commercialUseAllowed": commercial,
        "requiresRuntimes": [backend],
        "runtimeRevisions": {backend: "pinned"},
        "source": {
            "kinds": ["photos"],
            "minimumFrames": 8,
            "maximumFrames": 32,
            "maximumImageDimension": 2048,
            "characteristics": ["overlap"],
        },
        "hardware": {
            "requiresCuda": True,
            "minimumCudaCapability": "8.0",
            "minimumVramMiB": 8192,
            "reserveVramMiB": 1024,
        },
        "gates": {"cameraAgreement": True, "bundleAdjustment": True},
        "metrics": {
            "registrationRatio": registration,
            "medianCameraResidual": residual,
            "medianReprojectionErrorPx": 0.7,
            "wallSeconds": 20.0,
            "peakVramMiB": 6000,
        },
    }


class BackendPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = {
            "kind": "photos",
            "frameCount": 16,
            "maximumImageDimension": 1600,
            "characteristics": ["overlap"],
        }
        self.hardware = {
            "cudaValidated": True,
            "cudaCapability": "12.0",
            "gpuName": "test gpu",
            "vramTotalMiB": 12288,
            "vramFreeMiB": 11000,
            "cpuThreads": 16,
        }
        self.runtimes = {
            "da3-guided-colmap": {
                "available": True,
                "validated": True,
                "revision": "pinned",
            },
            "mapanything-guided-colmap": {
                "available": True,
                "validated": True,
                "revision": "pinned",
            },
            "da3": True,
            "mapanything": True,
            "colmap-learned": True,
            "gsplat": True,
        }

    def test_quality_first_ranking_prefers_camera_coverage(self) -> None:
        result = select_backend_policy(
            source=self.source,
            hardware=self.hardware,
            runtimes=self.runtimes,
            records=[
                benchmark("fast", "da3-guided-colmap", registration=0.82, residual=0.001),
                benchmark(
                    "complete",
                    "mapanything-guided-colmap",
                    registration=0.95,
                    residual=0.010,
                ),
            ],
        )

        decision = result["decisions"]["productionCamera"]
        self.assertEqual(decision["selected"], "mapanything-guided-colmap")
        self.assertTrue(decision["benchmarked"])
        self.assertEqual(decision["evidence"], ["complete"])
        self.assertIn("da3-guided-colmap", decision["fallbackChain"])

    def test_incompatible_source_and_memory_retain_protected_baseline(self) -> None:
        constrained_hardware = {**self.hardware, "vramTotalMiB": 7000}
        result = select_backend_policy(
            source={**self.source, "maximumImageDimension": 4096},
            hardware=constrained_hardware,
            runtimes=self.runtimes,
            records=[
                benchmark(
                    "candidate", "da3-guided-colmap", registration=0.95, residual=0.01
                )
            ],
        )

        decision = result["decisions"]["productionCamera"]
        self.assertEqual(decision["selected"], "validated-learned-challenger-bakeoff")
        self.assertFalse(decision["benchmarked"])
        assessment = result["candidateAssessments"][0]
        self.assertFalse(assessment["eligible"])
        self.assertTrue(any("resolution" in reason for reason in assessment["reasons"]))
        self.assertTrue(any("VRAM" in reason for reason in assessment["reasons"]))

    def test_runtime_revision_and_commercial_license_are_hard_gates(self) -> None:
        result = select_backend_policy(
            source=self.source,
            hardware=self.hardware,
            runtimes={
                **self.runtimes,
                "da3-guided-colmap": {
                    "available": True,
                    "validated": True,
                    "revision": "different",
                },
            },
            quality={"commercialUse": True},
            records=[
                benchmark(
                    "research-only",
                    "da3-guided-colmap",
                    registration=0.99,
                    residual=0.001,
                    commercial=False,
                )
            ],
        )

        reasons = result["candidateAssessments"][0]["reasons"]
        self.assertTrue(any("revision" in reason for reason in reasons))
        self.assertTrue(any("commercial" in reason for reason in reasons))

    def test_missing_task_quality_cannot_win_by_being_the_only_record(self) -> None:
        incomplete = benchmark(
            "incomplete", "da3-guided-colmap", registration=0.95, residual=0.01
        )
        del incomplete["metrics"]["medianReprojectionErrorPx"]
        result = select_backend_policy(
            source=self.source,
            hardware=self.hardware,
            runtimes=self.runtimes,
            records=[incomplete],
        )
        self.assertFalse(result["decisions"]["productionCamera"]["benchmarked"])
        self.assertTrue(
            any(
                "medianReprojectionErrorPx" in reason
                for reason in result["candidateAssessments"][0]["reasons"]
            )
        )

    def test_explicit_override_is_audited_and_missing_runtime_fails(self) -> None:
        result = select_backend_policy(
            source=self.source,
            hardware=self.hardware,
            runtimes=self.runtimes,
            overrides={"productionCamera": "da3-guided-colmap"},
            records=[
                benchmark(
                    "first-benchmark",
                    "da3-guided-colmap",
                    registration=0.9,
                    residual=0.01,
                )
            ],
        )
        decision = result["decisions"]["productionCamera"]
        self.assertEqual(decision["selectionMode"], "explicit-override")
        self.assertFalse(decision["benchmarked"])

        with self.assertRaisesRegex(BackendPolicyError, "unavailable runtime"):
            select_backend_policy(
                source=self.source,
                hardware=self.hardware,
                runtimes={**self.runtimes, "da3-guided-colmap": False},
                overrides={"productionCamera": "da3-guided-colmap"},
                records=[
                    benchmark(
                        "first-benchmark",
                        "da3-guided-colmap",
                        registration=0.9,
                        residual=0.01,
                    )
                ],
            )

    def test_free_vram_and_cuda_smoke_are_not_inferred_from_installation(self) -> None:
        result = select_backend_policy(
            source=self.source,
            hardware={**self.hardware, "cudaValidated": False, "vramFreeMiB": 6200},
            runtimes=self.runtimes,
            records=[
                benchmark(
                    "candidate", "da3-guided-colmap", registration=0.95, residual=0.01
                )
            ],
        )

        reasons = result["candidateAssessments"][0]["reasons"]
        self.assertTrue(any("smoke test" in reason for reason in reasons))
        self.assertTrue(any("free VRAM" in reason for reason in reasons))

    def test_packaged_femto_evidence_selects_lower_residual_depth_backend(self) -> None:
        result = select_backend_policy(
            source={
                "kind": "rgbd",
                "sensorKind": "femto_mega",
                "frameCount": 81,
                "maximumImageDimension": 1920,
                "characteristics": ["physical-capture"],
            },
            hardware={
                "cudaValidated": True,
                "cudaCapability": "12.0",
                "gpuName": "NVIDIA GeForce RTX 5080 Laptop GPU",
                "vramTotalMiB": 16302,
                "vramFreeMiB": 15000,
                "cpuThreads": 24,
            },
            runtimes={
                "open3d-cuda": {
                    "available": True,
                    "validated": True,
                    "revision": "0.19.0",
                },
                "open3d-cpu": True,
                "mapanything": {
                    "available": True,
                    "validated": True,
                    "revision": "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9",
                },
                "lingbot-depth": {"available": True, "validated": True},
                "gsplat": True,
            },
        )

        depth = result["decisions"]["depthCompletion"]
        self.assertEqual(depth["selected"], "mapanything")
        self.assertEqual(depth["evidence"], ["p7-femto-mapanything-depth"])


if __name__ == "__main__":
    unittest.main()
