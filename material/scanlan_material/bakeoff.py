from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .contracts import MATERIAL_CLASSES, OPTICAL_RISKS


@dataclass(frozen=True)
class BakeoffGates:
    minimum_material_mean_iou: float = 0.55
    minimum_optical_risk_recall: float = 0.90
    minimum_optical_risk_precision: float = 0.65
    minimum_multiview_consistency: float = 0.80
    maximum_expected_calibration_error: float = 0.10
    maximum_peak_vram_gb: float = 10.5


@dataclass(frozen=True)
class CandidateEvidence:
    identifier: str
    material_mean_iou: float
    optical_risk_recall: float
    optical_risk_precision: float
    multiview_consistency: float
    expected_calibration_error: float
    peak_vram_gb: float
    median_seconds_per_megapixel: float
    evaluated_frames: int
    representative_real_frames: int


@dataclass(frozen=True)
class BakeoffResult:
    identifier: str
    accepted: bool
    reasons: tuple[str, ...]
    evidence: CandidateEvidence

    def to_dict(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "evidence": asdict(self.evidence),
        }


def measure_candidate_evidence(
    identifier: str,
    class_probabilities: np.ndarray,
    class_labels: np.ndarray,
    optical_risk_probabilities: np.ndarray,
    optical_risk_labels: np.ndarray,
    valid_mask: np.ndarray,
    multiview_agreement: np.ndarray,
    *,
    peak_vram_gb: float,
    median_seconds_per_megapixel: float,
    representative_real_frames: int,
    calibration_bins: int = 10,
) -> CandidateEvidence:
    """Measure bake-off quality from source-aligned annotated predictions.

    Resource values come from the supervised worker because CUDA allocation and
    wall time cannot be reconstructed from saved arrays. All quality values are
    recomputed here, preventing adapters from supplying self-reported scores.
    """

    classes = np.asarray(class_probabilities, dtype=np.float64)
    labels = np.asarray(class_labels, dtype=np.int64)
    risks = np.asarray(optical_risk_probabilities, dtype=np.float64)
    risk_labels = np.asarray(optical_risk_labels, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    if classes.ndim != 4 or classes.shape[-1] != len(MATERIAL_CLASSES):
        raise ValueError("bake-off class probabilities must be NxHxWxC")
    if labels.shape != classes.shape[:3] or valid.shape != labels.shape:
        raise ValueError("bake-off labels and validity must be source-aligned")
    expected_risk_shape = (*labels.shape, len(OPTICAL_RISKS))
    if risks.shape != expected_risk_shape or risk_labels.shape != expected_risk_shape:
        raise ValueError("bake-off optical-risk arrays must be source-aligned NxHxWxR")
    if not np.isfinite(classes).all() or not np.isfinite(risks).all():
        raise ValueError("bake-off probabilities must be finite")
    if (
        np.any(classes < 0.0)
        or np.any(classes > 1.0)
        or np.any(risks < 0.0)
        or np.any(risks > 1.0)
    ):
        raise ValueError("bake-off probabilities must remain in [0, 1]")
    if not np.any(valid):
        raise ValueError("bake-off contains no valid annotated pixels")
    if np.any(valid & ((labels < 0) | (labels >= len(MATERIAL_CLASSES)))):
        raise ValueError("bake-off material labels are outside the contract")
    sums = np.sum(classes, axis=-1)
    if np.any(valid & (np.abs(sums - 1.0) > 2e-3)):
        raise ValueError("bake-off class probabilities must sum to one")

    predicted = np.argmax(classes, axis=-1)
    ious: list[float] = []
    # Unknown is excluded: rewarding a model for abstention would hide poor
    # identity predictions on annotated material.
    for index in range(1, len(MATERIAL_CLASSES)):
        target = valid & (labels == index)
        proposed = valid & (predicted == index)
        union = np.count_nonzero(target | proposed)
        if union:
            ious.append(float(np.count_nonzero(target & proposed) / union))
    mean_iou = float(np.mean(ious)) if ious else 0.0

    risk_valid = np.broadcast_to(valid[..., None], risks.shape)
    predicted_risk = risks >= 0.5
    true_positive = int(np.count_nonzero(risk_valid & predicted_risk & risk_labels))
    false_negative = int(np.count_nonzero(risk_valid & ~predicted_risk & risk_labels))
    false_positive = int(np.count_nonzero(risk_valid & predicted_risk & ~risk_labels))
    recall = true_positive / max(1, true_positive + false_negative)
    precision = true_positive / max(1, true_positive + false_positive)

    confidence = np.max(classes, axis=-1)[valid]
    correctness = (predicted == labels)[valid].astype(np.float64)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, max(2, calibration_bins) + 1)
    for bin_index in range(len(edges) - 1):
        selected = (confidence >= edges[bin_index]) & (
            confidence <= edges[bin_index + 1]
            if bin_index == len(edges) - 2
            else confidence < edges[bin_index + 1]
        )
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(
                float(np.mean(confidence[selected])) - float(np.mean(correctness[selected]))
            )

    agreement = np.asarray(multiview_agreement, dtype=np.float64)
    if agreement.size == 0 or not np.isfinite(agreement).all():
        raise ValueError("bake-off requires finite warped multiview agreement")
    if np.any(agreement < 0.0) or np.any(agreement > 1.0):
        raise ValueError("multiview agreement must remain in [0, 1]")
    return CandidateEvidence(
        identifier=identifier,
        material_mean_iou=mean_iou,
        optical_risk_recall=float(recall),
        optical_risk_precision=float(precision),
        multiview_consistency=float(np.mean(agreement)),
        expected_calibration_error=float(ece),
        peak_vram_gb=peak_vram_gb,
        median_seconds_per_megapixel=median_seconds_per_megapixel,
        evaluated_frames=int(classes.shape[0]),
        representative_real_frames=int(representative_real_frames),
    )


def evaluate_candidate(
    evidence: CandidateEvidence,
    gates: BakeoffGates | None = None,
) -> BakeoffResult:
    gates = gates or BakeoffGates()
    bounded = {
        "material mean IoU": evidence.material_mean_iou,
        "optical-risk recall": evidence.optical_risk_recall,
        "optical-risk precision": evidence.optical_risk_precision,
        "multiview consistency": evidence.multiview_consistency,
        "expected calibration error": evidence.expected_calibration_error,
    }
    if any(value < 0.0 or value > 1.0 for value in bounded.values()):
        raise ValueError("bake-off quality metrics must remain in [0, 1]")
    if evidence.peak_vram_gb <= 0.0 or evidence.median_seconds_per_megapixel <= 0.0:
        raise ValueError("bake-off resource measurements must be positive")
    reasons: list[str] = []
    if evidence.representative_real_frames <= 0:
        reasons.append("no representative real ScanLan frame was evaluated")
    if evidence.evaluated_frames < 20:
        reasons.append("fewer than 20 annotated frames were evaluated")
    if evidence.material_mean_iou < gates.minimum_material_mean_iou:
        reasons.append("material mean IoU misses the quality gate")
    if evidence.optical_risk_recall < gates.minimum_optical_risk_recall:
        reasons.append("optical-risk recall misses the fail-safe geometry gate")
    if evidence.optical_risk_precision < gates.minimum_optical_risk_precision:
        reasons.append("optical-risk precision would suppress too much valid geometry")
    if evidence.multiview_consistency < gates.minimum_multiview_consistency:
        reasons.append("source-aligned predictions are not multiview consistent")
    if evidence.expected_calibration_error > gates.maximum_expected_calibration_error:
        reasons.append("prediction confidence is not calibrated")
    if evidence.peak_vram_gb > gates.maximum_peak_vram_gb:
        reasons.append("peak VRAM leaves insufficient headroom on the 12 GB target")
    return BakeoffResult(evidence.identifier, not reasons, tuple(reasons), evidence)


def _dominates(left: CandidateEvidence, right: CandidateEvidence) -> bool:
    higher = (
        "material_mean_iou",
        "optical_risk_recall",
        "optical_risk_precision",
        "multiview_consistency",
    )
    lower = (
        "expected_calibration_error",
        "peak_vram_gb",
        "median_seconds_per_megapixel",
    )
    no_worse = all(getattr(left, name) >= getattr(right, name) for name in higher)
    no_worse &= all(getattr(left, name) <= getattr(right, name) for name in lower)
    strictly_better = any(getattr(left, name) > getattr(right, name) for name in higher)
    strictly_better |= any(getattr(left, name) < getattr(right, name) for name in lower)
    return no_worse and strictly_better


def rank_candidates(
    evidence: Iterable[CandidateEvidence],
    gates: BakeoffGates | None = None,
) -> tuple[BakeoffResult, ...]:
    results = [evaluate_candidate(value, gates) for value in evidence]
    accepted = [result for result in results if result.accepted]
    rejected = [result for result in results if not result.accepted]
    # Preserve the Pareto frontier. When several safe candidates trade quality
    # against resources, prioritize the failure mode that can corrupt geometry,
    # then identity quality, consistency, calibration, memory and time.
    accepted.sort(
        key=lambda result: (
            any(
                _dominates(other.evidence, result.evidence)
                for other in accepted
                if other.identifier != result.identifier
            ),
            -result.evidence.optical_risk_recall,
            -result.evidence.material_mean_iou,
            -result.evidence.multiview_consistency,
            result.evidence.expected_calibration_error,
            result.evidence.peak_vram_gb,
            result.evidence.median_seconds_per_megapixel,
            result.identifier,
        )
    )
    rejected.sort(key=lambda result: (len(result.reasons), result.identifier))
    return tuple(accepted + rejected)
