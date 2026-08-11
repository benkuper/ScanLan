from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .analysis import FusedMaterialSurface
from .contracts import MATERIAL_CLASSES, OPTICAL_RISKS, MaterialPrediction


GEOMETRY_POLICY_VERSION = "scanlan-material-geometry-v1"
GEOMETRY_RESULT_VERSION = "scanlan-material-refinement-v1"

PROVENANCE_MEASURED = 0
PROVENANCE_GENERATED = 1
PROVENANCE_LEARNED = 2
_PROVENANCE_VALUES = (PROVENANCE_MEASURED, PROVENANCE_GENERATED, PROVENANCE_LEARNED)


def _unit_interval(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must remain in [0, 1]")
    return array


@dataclass(frozen=True)
class MaterialGeometryPolicy:
    """Conservative multipliers applied to existing geometry confidence.

    The policy never creates geometric evidence. A value of one leaves the
    upstream confidence unchanged, while zero removes that source's authority.
    """

    sensor_depth_multiplier: np.ndarray
    generated_depth_multiplier: np.ndarray
    learned_depth_multiplier: np.ndarray
    repair_authority: np.ndarray
    refinement_authority: np.ndarray
    protected_mask: np.ndarray
    discard_mask: np.ndarray
    metadata: Mapping[str, Any] | None = None

    def validated(self) -> "MaterialGeometryPolicy":
        sensor = np.asarray(self.sensor_depth_multiplier, dtype=np.float32)
        if sensor.ndim < 1:
            raise ValueError("material geometry policy must have a non-scalar grid")
        shape = sensor.shape
        generated = _unit_interval(
            "generated-depth multiplier", self.generated_depth_multiplier, shape
        )
        learned = _unit_interval("learned-depth multiplier", self.learned_depth_multiplier, shape)
        repair = _unit_interval("repair authority", self.repair_authority, shape)
        refinement = _unit_interval("refinement authority", self.refinement_authority, shape)
        sensor = _unit_interval("sensor-depth multiplier", sensor, shape)
        protected = np.asarray(self.protected_mask, dtype=bool)
        discard = np.asarray(self.discard_mask, dtype=bool)
        if protected.shape != shape or discard.shape != shape:
            raise ValueError("geometry policy masks must match the policy grid")
        if np.any(protected & (repair > 0.0)):
            raise ValueError("protected material regions cannot authorize blind repair")
        if np.any(
            discard
            & (
                (sensor > 0.0)
                | (generated > 0.0)
                | (learned > 0.0)
                | (repair > 0.0)
                | (refinement > 0.0)
            )
        ):
            raise ValueError("discarded material regions cannot retain geometry authority")
        return MaterialGeometryPolicy(
            sensor,
            generated,
            learned,
            repair,
            refinement,
            protected,
            discard,
            dict(self.metadata or {}),
        )

    def multiplier_for(self, provenance: np.ndarray) -> np.ndarray:
        checked = self.validated()
        values = np.asarray(provenance, dtype=np.uint8)
        if values.shape != checked.sensor_depth_multiplier.shape:
            raise ValueError("geometry provenance must match the material policy grid")
        if not np.isin(values, _PROVENANCE_VALUES).all():
            raise ValueError("geometry provenance contains an unsupported value")
        return np.choose(
            values,
            (
                checked.sensor_depth_multiplier,
                checked.generated_depth_multiplier,
                checked.learned_depth_multiplier,
            ),
        ).astype(np.float32)


def neutral_geometry_policy(
    shape: Sequence[int], *, metadata: Mapping[str, Any] | None = None
) -> MaterialGeometryPolicy:
    resolved = tuple(int(value) for value in shape)
    if not resolved or any(value <= 0 for value in resolved):
        raise ValueError("neutral material geometry policy requires a positive grid")
    ones = np.ones(resolved, dtype=np.float32)
    zeros = np.zeros(resolved, dtype=bool)
    return MaterialGeometryPolicy(
        ones,
        ones.copy(),
        ones.copy(),
        ones.copy(),
        ones.copy(),
        zeros,
        zeros.copy(),
        {
            "policyVersion": GEOMETRY_POLICY_VERSION,
            "materialEvidence": "missing",
            **dict(metadata or {}),
        },
    ).validated()


def _derive_geometry_policy(
    class_probabilities: np.ndarray,
    optical_risk_probabilities: np.ndarray,
    valid_mask: np.ndarray,
    confidence: np.ndarray,
    *,
    risk_already_calibrated: bool,
    metadata: Mapping[str, Any] | None = None,
) -> MaterialGeometryPolicy:
    classes = np.asarray(class_probabilities, dtype=np.float32)
    risks = np.asarray(optical_risk_probabilities, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    confidence_values = np.asarray(confidence, dtype=np.float32)
    shape = valid.shape
    if classes.shape != (*shape, len(MATERIAL_CLASSES)):
        raise ValueError("material classes do not match the requested geometry grid")
    if risks.shape != (*shape, len(OPTICAL_RISKS)):
        raise ValueError("optical risks do not match the requested geometry grid")
    _unit_interval("material confidence", confidence_values, shape)
    if not np.isfinite(classes).all() or not np.isfinite(risks).all():
        raise ValueError("material geometry evidence must be finite")
    if (
        np.any(classes < 0.0)
        or np.any(classes > 1.0)
        or np.any(risks < 0.0)
        or np.any(risks > 1.0)
    ):
        raise ValueError("material geometry evidence must remain in [0, 1]")

    # Per-view probabilities still need their calibrated prediction
    # confidence. P14's fused surface risks already retain confidence through
    # both the consensus and conservative-peak paths and must not be attenuated
    # a second time by identity entropy.
    calibrated_risks = risks if risk_already_calibrated else risks * confidence_values[..., None]
    calibrated_risks = np.where(valid[..., None], calibrated_risks, 0.0)
    identity = np.where(valid[..., None], classes * confidence_values[..., None], 0.0)

    def risk(name: str) -> np.ndarray:
        return calibrated_risks[..., OPTICAL_RISKS.index(name)]

    glass = risk("glass_or_transmissive")
    mirror = risk("mirror")
    specular = risk("high_specular")
    emissive = risk("emissive")
    thin = risk("thin_geometry")
    dynamic = np.maximum(
        risk("dynamic"), identity[..., MATERIAL_CLASSES.index("dynamic")]
    )
    sky = np.maximum(risk("sky"), identity[..., MATERIAL_CLASSES.index("sky")])

    sensor_penalty = np.maximum.reduce(
        (0.95 * glass, mirror, 0.65 * specular, 0.45 * thin, dynamic, sky)
    )
    generated_penalty = np.maximum.reduce(
        (glass, mirror, 0.80 * specular, 0.40 * emissive, 0.75 * thin, dynamic, sky)
    )
    learned_penalty = np.maximum.reduce(
        (0.80 * glass, 0.90 * mirror, 0.55 * specular, 0.30 * emissive, 0.55 * thin, dynamic, sky)
    )
    repair_penalty = np.maximum.reduce((glass, mirror, thin, dynamic, sky))
    refinement_penalty = np.maximum.reduce(
        (glass, mirror, 0.50 * specular, 0.60 * thin, dynamic, sky)
    )

    sensor = np.where(valid, 1.0 - sensor_penalty, 1.0).astype(np.float32)
    generated = np.where(valid, 1.0 - generated_penalty, 1.0).astype(np.float32)
    learned = np.where(valid, 1.0 - learned_penalty, 1.0).astype(np.float32)
    repair = np.where(valid, 1.0 - repair_penalty, 1.0).astype(np.float32)
    refinement = np.where(valid, 1.0 - refinement_penalty, 1.0).astype(np.float32)
    protected = valid & (
        (glass >= 0.55)
        | (mirror >= 0.55)
        | (thin >= 0.65)
        | (dynamic >= 0.50)
        | (sky >= 0.50)
    )
    discard = valid & ((dynamic >= 0.80) | (sky >= 0.80))
    repair[protected] = 0.0
    for value in (sensor, generated, learned, repair, refinement):
        value[discard] = 0.0

    return MaterialGeometryPolicy(
        np.clip(sensor, 0.0, 1.0),
        np.clip(generated, 0.0, 1.0),
        np.clip(learned, 0.0, 1.0),
        np.clip(repair, 0.0, 1.0),
        np.clip(refinement, 0.0, 1.0),
        protected,
        discard,
        {
            "policyVersion": GEOMETRY_POLICY_VERSION,
            "materialEvidence": "available",
            "protectedCount": int(np.count_nonzero(protected)),
            "discardCount": int(np.count_nonzero(discard)),
            **dict(metadata or {}),
        },
    ).validated()


def prediction_geometry_policy(prediction: MaterialPrediction) -> MaterialGeometryPolicy:
    checked = prediction.validated()
    return _derive_geometry_policy(
        checked.class_probabilities,
        checked.optical_risk_probabilities,
        checked.valid_mask,
        checked.confidence,
        risk_already_calibrated=False,
        metadata={"sourceContract": "scanlan-material-v1", **dict(checked.metadata or {})},
    )


def surface_geometry_policy(surface: FusedMaterialSurface) -> MaterialGeometryPolicy:
    checked = surface.validated()
    return _derive_geometry_policy(
        checked.class_probabilities,
        checked.optical_risk_probabilities,
        checked.valid_mask,
        checked.confidence,
        risk_already_calibrated=True,
        metadata={
            "sourceContract": "scanlan-material-surface-v1",
            "sourceViewCount": int((checked.metadata or {}).get("sourceViewCount", 0)),
        },
    )


def apply_depth_confidence_policy(
    confidence: np.ndarray,
    policy: MaterialGeometryPolicy,
    provenance: int | np.ndarray,
) -> np.ndarray:
    checked = policy.validated()
    values = _unit_interval(
        "upstream depth confidence", confidence, checked.sensor_depth_multiplier.shape
    )
    provenance_values = np.asarray(provenance, dtype=np.uint8)
    if provenance_values.ndim == 0:
        provenance_values = np.full(values.shape, int(provenance_values), dtype=np.uint8)
    return (values * checked.multiplier_for(provenance_values)).astype(np.float32)


@dataclass(frozen=True)
class RepairBoundaryDecision:
    allowed: bool
    authority: float
    matched_sample_count: int
    protected_fraction: float
    discard_fraction: float
    reason: str


def evaluate_repair_boundary(
    vertices: np.ndarray,
    boundary_positions: np.ndarray,
    policy: MaterialGeometryPolicy,
    maximum_distance_m: float,
) -> RepairBoundaryDecision:
    points = np.asarray(vertices, dtype=np.float64)
    boundary = np.asarray(boundary_positions, dtype=np.float64)
    checked = policy.validated()
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("repair policy requires finite Vx3 surface vertices")
    if (
        boundary.ndim != 2
        or boundary.shape[1] != 3
        or not len(boundary)
        or not np.isfinite(boundary).all()
    ):
        raise ValueError("repair policy requires finite non-empty Bx3 boundary positions")
    if checked.sensor_depth_multiplier.shape != (len(points),):
        raise ValueError("repair policy must contain one value per surface vertex")
    radius = float(maximum_distance_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("repair material lookup radius must be positive")

    nearest: list[int] = []
    maximum_distance_squared = radius * radius
    # Boundary loops are deliberately bounded by the native repair stage. The
    # chunked search avoids adding SciPy to the shared offline package and
    # keeps memory independent of production mesh size.
    for start in range(0, len(boundary), 32):
        chunk = boundary[start : start + 32]
        best_distance = np.full(len(chunk), np.inf)
        best_index = np.full(len(chunk), -1, dtype=np.int64)
        for vertex_start in range(0, len(points), 32_768):
            vertex_chunk = points[vertex_start : vertex_start + 32_768]
            distance = np.sum((chunk[:, None, :] - vertex_chunk[None, :, :]) ** 2, axis=2)
            local_index = np.argmin(distance, axis=1)
            local_distance = distance[np.arange(len(chunk)), local_index]
            improved = local_distance < best_distance
            best_distance[improved] = local_distance[improved]
            best_index[improved] = vertex_start + local_index[improved]
        nearest.extend(best_index[best_distance <= maximum_distance_squared].tolist())
    if not nearest:
        return RepairBoundaryDecision(
            True,
            1.0,
            0,
            0.0,
            0.0,
            "no nearby material evidence; preserve the existing depth-based repair decision",
        )
    indices = np.unique(np.asarray(nearest, dtype=np.int64))
    protected_fraction = float(np.mean(checked.protected_mask[indices]))
    discard_fraction = float(np.mean(checked.discard_mask[indices]))
    authority = float(np.quantile(checked.repair_authority[indices], 0.20))
    if discard_fraction >= 0.10:
        reason = "dynamic or sky evidence vetoes static surface repair"
        allowed = False
    elif protected_fraction > 0.0:
        reason = "reflective, transmissive, or thin material evidence preserves the opening"
        allowed = False
    elif authority < 0.50:
        reason = "material evidence provides insufficient repair authority"
        allowed = False
    else:
        reason = "material evidence does not veto the depth-supported repair"
        allowed = True
    return RepairBoundaryDecision(
        allowed,
        authority,
        len(indices),
        protected_fraction,
        discard_fraction,
        reason,
    )


@dataclass(frozen=True)
class GeometryProposal:
    vertices: np.ndarray
    confidence: np.ndarray
    effective_view_count: np.ndarray
    heldout_residual_m: np.ndarray
    provenance: np.ndarray
    metadata: Mapping[str, Any] | None = None

    def validated(self, vertex_count: int) -> "GeometryProposal":
        vertices = np.asarray(self.vertices, dtype=np.float32)
        if vertices.shape != (vertex_count, 3) or not np.isfinite(vertices).all():
            raise ValueError("geometry proposal vertices must be finite Vx3")
        confidence = _unit_interval(
            "geometry proposal confidence", self.confidence, (vertex_count,)
        )
        effective = np.asarray(self.effective_view_count, dtype=np.float32)
        residual = np.asarray(self.heldout_residual_m, dtype=np.float32)
        if (
            effective.shape != (vertex_count,)
            or not np.isfinite(effective).all()
            or np.any(effective < 0.0)
        ):
            raise ValueError("geometry proposal effective view count must be finite non-negative V")
        if (
            residual.shape != (vertex_count,)
            or not np.isfinite(residual).all()
            or np.any(residual < 0.0)
        ):
            raise ValueError("geometry proposal held-out residual must be finite non-negative V")
        provenance = np.asarray(self.provenance, dtype=np.uint8)
        if provenance.shape != (vertex_count,) or not np.isin(provenance, _PROVENANCE_VALUES).all():
            raise ValueError(
                "geometry proposal provenance must contain one supported value per vertex"
            )
        return GeometryProposal(
            vertices,
            confidence,
            effective,
            residual,
            provenance,
            dict(self.metadata or {}),
        )


@dataclass(frozen=True)
class MaterialGeometryResult:
    vertices: np.ndarray
    accepted_mask: np.ndarray
    authority: np.ndarray
    displacement_m: np.ndarray
    metadata: Mapping[str, Any] | None = None

    def validated(self) -> "MaterialGeometryResult":
        vertices = np.asarray(self.vertices, dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError("material geometry result vertices must be finite Vx3")
        count = len(vertices)
        accepted = np.asarray(self.accepted_mask, dtype=bool)
        authority = _unit_interval("material geometry authority", self.authority, (count,))
        displacement = np.asarray(self.displacement_m, dtype=np.float32)
        if (
            accepted.shape != (count,)
            or displacement.shape != (count,)
            or not np.isfinite(displacement).all()
            or np.any(displacement < 0.0)
        ):
            raise ValueError("material geometry result masks and displacement must be finite V")
        if np.any(~accepted & ((authority > 0.0) | (displacement > 1e-8))):
            raise ValueError("rejected geometry vertices cannot retain authority or displacement")
        return MaterialGeometryResult(
            vertices,
            accepted,
            authority,
            displacement,
            dict(self.metadata or {}),
        )


def _triangle_quality(
    original: np.ndarray, candidate: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    original_edges_a = original[triangles[:, 1]] - original[triangles[:, 0]]
    original_edges_b = original[triangles[:, 2]] - original[triangles[:, 0]]
    candidate_edges_a = candidate[triangles[:, 1]] - candidate[triangles[:, 0]]
    candidate_edges_b = candidate[triangles[:, 2]] - candidate[triangles[:, 0]]
    original_cross = np.cross(original_edges_a, original_edges_b)
    candidate_cross = np.cross(candidate_edges_a, candidate_edges_b)
    original_area = np.linalg.norm(original_cross, axis=1)
    candidate_area = np.linalg.norm(candidate_cross, axis=1)
    if np.any(original_area <= 1e-12):
        raise ValueError("material refinement requires a non-degenerate input mesh")
    cosine = np.sum(original_cross * candidate_cross, axis=1) / np.maximum(
        original_area * candidate_area, 1e-18
    )
    ratio = candidate_area / original_area
    return (
        (candidate_area > 1e-12)
        & (cosine >= 0.50)
        & (ratio >= 0.20)
        & (ratio <= 5.0)
    )


def refine_material_geometry(
    vertices: np.ndarray,
    triangles: np.ndarray,
    material_surface: FusedMaterialSurface,
    proposal: GeometryProposal,
    voxel_size_m: float,
) -> MaterialGeometryResult:
    original = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(triangles, dtype=np.int64)
    if original.ndim != 2 or original.shape[1] != 3 or not np.isfinite(original).all():
        raise ValueError("material refinement requires finite Vx3 vertices")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or int(faces.min()) < 0
        or int(faces.max()) >= len(original)
    ):
        raise ValueError("material refinement requires indexed triangle geometry")
    voxel = float(voxel_size_m)
    if not math.isfinite(voxel) or voxel <= 0.0:
        raise ValueError("material refinement voxel size must be positive")
    surface = material_surface.validated()
    if len(surface.valid_mask) != len(original):
        raise ValueError("material surface must correspond one-to-one with refinement vertices")
    candidate = proposal.validated(len(original))
    policy = surface_geometry_policy(surface)
    source_multiplier = policy.multiplier_for(candidate.provenance)
    displacement = np.linalg.norm(candidate.vertices - original, axis=1)

    minimum_confidence = np.choose(candidate.provenance, (0.35, 0.70, 0.75))
    minimum_views = np.choose(candidate.provenance, (0.75, 1.75, 2.00))
    maximum_residual = voxel * np.choose(candidate.provenance, (1.50, 1.00, 0.75))
    maximum_displacement = voxel * np.choose(candidate.provenance, (2.50, 2.00, 1.50))
    confidence_ok = candidate.confidence >= minimum_confidence
    support_ok = candidate.effective_view_count >= minimum_views
    residual_ok = candidate.heldout_residual_m <= maximum_residual
    displacement_ok = displacement <= maximum_displacement

    protected_recovery = (
        policy.protected_mask
        & (candidate.provenance != PROVENANCE_MEASURED)
        & (candidate.confidence >= 0.90)
        & (candidate.effective_view_count >= 2.50)
        & (candidate.heldout_residual_m <= voxel * 0.50)
        & (displacement <= voxel)
    )
    material_ok = (~policy.protected_mask & (source_multiplier > 0.0)) | protected_recovery
    evidence_strength = candidate.confidence * np.minimum(
        1.0,
        candidate.effective_view_count
        / np.choose(candidate.provenance, (1.0, 2.0, 2.5)),
    )
    authority = evidence_strength * source_multiplier * policy.refinement_authority
    # A protected surface may only move through the explicit, strict recovery
    # gate above. Give that independently verified proposal bounded authority
    # even though the ordinary material multiplier intentionally approaches zero.
    authority[protected_recovery] = np.maximum(
        authority[protected_recovery],
        0.35 * evidence_strength[protected_recovery],
    )
    accepted = (
        confidence_ok
        & support_ok
        & residual_ok
        & displacement_ok
        & material_ok
        & ~policy.discard_mask
        & (authority >= 0.15)
    )
    authority = np.where(accepted, np.clip(authority, 0.0, 1.0), 0.0).astype(np.float32)
    refined = original + authority[:, None] * (candidate.vertices - original)

    rejected_topology = np.zeros(len(original), dtype=bool)
    # Reject all moved vertices participating in a newly folded, collapsed, or
    # explosively stretched face. Repeating is necessary because resetting one
    # face can expose a neighboring face whose other vertices remain moved.
    for _ in range(8):
        bad_faces = ~_triangle_quality(original, refined, faces)
        if not np.any(bad_faces):
            break
        bad_vertices = np.unique(faces[bad_faces].reshape(-1))
        moved_bad = bad_vertices[accepted[bad_vertices]]
        if not len(moved_bad):
            raise RuntimeError("material refinement cannot preserve input mesh topology")
        accepted[moved_bad] = False
        rejected_topology[moved_bad] = True
        authority[moved_bad] = 0.0
        refined[moved_bad] = original[moved_bad]
    if not np.all(_triangle_quality(original, refined, faces)):
        raise RuntimeError("material refinement failed its final topology gate")

    final_displacement = np.linalg.norm(refined - original, axis=1).astype(np.float32)
    final_displacement[~accepted] = 0.0
    metadata = {
        "resultVersion": GEOMETRY_RESULT_VERSION,
        "policyVersion": GEOMETRY_POLICY_VERSION,
        "candidateVertexCount": len(original),
        "acceptedVertexCount": int(np.count_nonzero(accepted)),
        "protectedVertexCount": int(np.count_nonzero(policy.protected_mask)),
        "protectedRecoveredVertexCount": int(np.count_nonzero(accepted & protected_recovery)),
        "discardedVertexCount": int(np.count_nonzero(policy.discard_mask)),
        "rejectedConfidenceCount": int(np.count_nonzero(~confidence_ok)),
        "rejectedSupportCount": int(np.count_nonzero(~support_ok)),
        "rejectedResidualCount": int(np.count_nonzero(~residual_ok)),
        "rejectedDisplacementCount": int(np.count_nonzero(~displacement_ok)),
        "rejectedTopologyVertexCount": int(np.count_nonzero(rejected_topology)),
        "rmsAcceptedDisplacementM": (
            float(np.sqrt(np.mean(final_displacement[accepted] ** 2))) if np.any(accepted) else 0.0
        ),
        "maximumAcceptedDisplacementM": (
            float(np.max(final_displacement[accepted])) if np.any(accepted) else 0.0
        ),
        "voxelSizeM": voxel,
        **dict(candidate.metadata or {}),
    }
    return MaterialGeometryResult(
        refined.astype(np.float32),
        accepted,
        authority,
        final_displacement,
        metadata,
    ).validated()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_geometry_result(path: Path, result: MaterialGeometryResult) -> None:
    checked = result.validated()
    _atomic_npz(
        path,
        {
            "contract": np.asarray(GEOMETRY_RESULT_VERSION),
            "vertices": checked.vertices.astype(np.float32),
            "accepted_mask": checked.accepted_mask.astype(np.uint8),
            "authority": checked.authority.astype(np.float16),
            "displacement_m": checked.displacement_m.astype(np.float32),
            "metadata_json": np.asarray(
                json.dumps(dict(checked.metadata or {}), sort_keys=True, separators=(",", ":"))
            ),
        },
    )


def read_geometry_result(path: Path) -> MaterialGeometryResult:
    with np.load(path, allow_pickle=False) as archive:
        contract = str(archive["contract"].item())
        if contract != GEOMETRY_RESULT_VERSION:
            raise ValueError(f"unsupported material geometry result contract: {contract}")
        result = MaterialGeometryResult(
            np.asarray(archive["vertices"], dtype=np.float32),
            np.asarray(archive["accepted_mask"], dtype=bool),
            np.asarray(archive["authority"], dtype=np.float32),
            np.asarray(archive["displacement_m"], dtype=np.float32),
            json.loads(str(archive["metadata_json"].item())),
        )
    return result.validated()
