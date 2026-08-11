from __future__ import annotations

import unittest

import numpy as np

from scanlan_material.bakeoff import (
    CandidateEvidence,
    evaluate_candidate,
    measure_candidate_evidence,
    rank_candidates,
)


def evidence(identifier: str, **changes: float | int) -> CandidateEvidence:
    values: dict[str, float | int | str] = {
        "identifier": identifier,
        "material_mean_iou": 0.7,
        "optical_risk_recall": 0.94,
        "optical_risk_precision": 0.78,
        "multiview_consistency": 0.86,
        "expected_calibration_error": 0.06,
        "peak_vram_gb": 8.5,
        "median_seconds_per_megapixel": 1.2,
        "evaluated_frames": 60,
        "representative_real_frames": 32,
    }
    values.update(changes)
    return CandidateEvidence(**values)  # type: ignore[arg-type]


class BakeoffTests(unittest.TestCase):
    def test_quality_evidence_is_measured_from_source_aligned_arrays(self) -> None:
        probabilities = np.zeros((20, 1, 2, 7), dtype=np.float32)
        probabilities[..., 1] = 1.0
        labels = np.ones((20, 1, 2), dtype=np.int64)
        risks = np.zeros((20, 1, 2, 7), dtype=np.float32)
        risks[..., 0] = 0.95
        risk_labels = np.zeros_like(risks, dtype=bool)
        risk_labels[..., 0] = True
        measured = measure_candidate_evidence(
            "measured",
            probabilities,
            labels,
            risks,
            risk_labels,
            np.ones_like(labels, dtype=bool),
            np.asarray([0.82, 0.9]),
            peak_vram_gb=8.0,
            median_seconds_per_megapixel=1.0,
            representative_real_frames=20,
        )
        self.assertEqual(measured.material_mean_iou, 1.0)
        self.assertEqual(measured.optical_risk_recall, 1.0)
        self.assertEqual(measured.optical_risk_precision, 1.0)
        self.assertAlmostEqual(measured.multiview_consistency, 0.86)
        self.assertEqual(measured.expected_calibration_error, 0.0)

    def test_optical_risk_recall_is_a_hard_gate(self) -> None:
        result = evaluate_candidate(evidence("unsafe", optical_risk_recall=0.89))
        self.assertFalse(result.accepted)
        self.assertIn("optical-risk recall", " ".join(result.reasons))

    def test_real_scanlan_evidence_is_mandatory(self) -> None:
        result = evaluate_candidate(evidence("synthetic-only", representative_real_frames=0))
        self.assertFalse(result.accepted)
        self.assertIn("representative real", " ".join(result.reasons))

    def test_safe_quality_frontier_precedes_rejected_candidate(self) -> None:
        ranked = rank_candidates(
            [
                evidence("balanced"),
                evidence("safer", optical_risk_recall=0.98, peak_vram_gb=9.0),
                evidence("small", peak_vram_gb=5.0, material_mean_iou=0.62),
                evidence("rejected", peak_vram_gb=11.5),
            ]
        )
        self.assertEqual(ranked[-1].identifier, "rejected")
        self.assertEqual(ranked[0].identifier, "safer")


if __name__ == "__main__":
    unittest.main()
