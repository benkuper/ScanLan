from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scanlan_validation import (
    ReleaseMatrixError,
    evaluate_release_matrix,
    load_release_evidence,
    load_release_requirements,
    write_default_promotion,
)


def requirements() -> dict:
    return {
        "schemaVersion": 1,
        "revision": "test-matrix",
        "candidateDefaults": {"depthRefinement": "adaptive"},
        "scenarios": [
            {
                "scenarioId": "rgbd",
                "category": "rgbd",
                "matches": {"source.kind": "rgbd", "source.sensorKind": "camera"},
                "requiredGates": ["safe"],
                "requiredMetrics": {
                    "quality.coverage": {"minimum": 0.9},
                    "latency.p95Ms": {"maximum": 100.0},
                },
                "requiredArtifacts": ["preview"],
                "visualInspection": True,
            },
            {
                "scenarioId": "cancel-resume",
                "category": "cancellation-resume",
                "matches": {"source.kind": ["rgbd", "photos"]},
                "requiredGates": ["atomic", "equivalent"],
                "requiredMetrics": {},
                "requiredArtifacts": [],
                "visualInspection": False,
            },
        ],
    }


def evidence(artifact_digest: str) -> list[dict]:
    return [
        {
            "schemaVersion": 1,
            "scenarioId": "rgbd",
            "status": "passed",
            "realInput": True,
            "source": {"kind": "rgbd", "sensorKind": "camera"},
            "gates": {"safe": True},
            "metrics": {"quality": {"coverage": 0.95}, "latency": {"p95Ms": 80}},
            "artifacts": {"preview": {"path": "preview.bin", "sha256": artifact_digest}},
            "visualInspection": {
                "passed": True,
                "reviewer": "release reviewer",
                "inspectedAt": "2026-08-11T00:00:00Z",
                "artifactSha256": artifact_digest,
            },
        },
        {
            "schemaVersion": 1,
            "scenarioId": "cancel-resume",
            "status": "passed",
            "realInput": True,
            "source": {"kind": "photos"},
            "gates": {"atomic": True, "equivalent": True},
            "metrics": {},
            "artifacts": {},
        },
    ]


class ReleaseMatrixTests(unittest.TestCase):
    def test_complete_matrix_is_the_only_default_promotion_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "preview.bin"
            artifact.write_bytes(b"inspected final preview")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            records = evidence(digest)
            for record in records:
                record["_evidencePath"] = str(root / "evidence.json")
            report = evaluate_release_matrix(
                requirements=requirements(), evidence_records=records
            )
            self.assertTrue(report["complete"])
            self.assertTrue(report["defaultPromotion"]["eligible"])
            promotion = root / "promotion.json"
            write_default_promotion(promotion, report)
            self.assertEqual(
                json.loads(promotion.read_text(encoding="utf-8"))["defaults"],
                {"depthRefinement": "adaptive"},
            )

    def test_missing_scenario_withholds_defaults(self) -> None:
        report = evaluate_release_matrix(
            requirements=requirements(), evidence_records=[], verify_artifacts=False
        )
        self.assertFalse(report["complete"])
        self.assertEqual(
            report["summary"]["blockedScenarios"], ["rgbd", "cancel-resume"]
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ReleaseMatrixError, "forbidden"):
                write_default_promotion(Path(directory) / "defaults.json", report)

    def test_tampered_artifact_fails_even_when_declared_gates_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "preview.bin").write_bytes(b"tampered")
            records = evidence(hashlib.sha256(b"original").hexdigest())
            records[0]["_evidencePath"] = str(root / "evidence.json")
            report = evaluate_release_matrix(
                requirements=requirements(), evidence_records=records
            )
            reasons = report["scenarios"][0]["reasons"]
            self.assertTrue(any("digest does not match" in reason for reason in reasons))

    def test_successful_process_cannot_replace_real_input_and_visual_review(self) -> None:
        records = evidence(hashlib.sha256(b"unused").hexdigest())
        records[0]["realInput"] = False
        records[0].pop("visualInspection")
        report = evaluate_release_matrix(
            requirements=requirements(),
            evidence_records=records,
            verify_artifacts=False,
        )
        reasons = report["scenarios"][0]["reasons"]
        self.assertTrue(any("representative real input" in reason for reason in reasons))
        self.assertTrue(any("visual inspection" in reason for reason in reasons))

    def test_diagnostic_artifact_bypass_can_never_complete_the_matrix(self) -> None:
        records = evidence(hashlib.sha256(b"unavailable artifact").hexdigest())
        report = evaluate_release_matrix(
            requirements=requirements(),
            evidence_records=records,
            verify_artifacts=False,
        )
        self.assertFalse(report["complete"])
        self.assertTrue(
            any(
                "verification was disabled" in reason
                for reason in report["scenarios"][0]["reasons"]
            )
        )

    def test_duplicate_evidence_is_rejected_as_ambiguous(self) -> None:
        record = evidence(hashlib.sha256(b"unused").hexdigest())[1]
        with self.assertRaisesRegex(ReleaseMatrixError, "Duplicate"):
            evaluate_release_matrix(
                requirements=requirements(),
                evidence_records=[record, record],
                verify_artifacts=False,
            )

    def test_packaged_requirements_and_evidence_set_load(self) -> None:
        packaged = load_release_requirements()
        self.assertEqual(packaged["revision"], "scanlan-v2-p19-2026-08-11")
        path = (
            Path(__file__).parents[1]
            / "release-evidence"
            / "v2-p19-audit-2026-08-11.json"
        )
        records = load_release_evidence([path])
        self.assertEqual(len(records), len(packaged["scenarios"]))


if __name__ == "__main__":
    unittest.main()
