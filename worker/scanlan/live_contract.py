from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Protocol


LIVE_CONTRACT_VERSION = 2


class TrackingState(str, Enum):
    READY = "ready"
    PREVIEW = "preview"
    TRACKING = "tracking"
    SEARCHING = "searching"
    RELOCALIZED = "relocalized"
    FROZEN = "frozen"
    FAILED = "failed"
    COMPLETE = "complete"


class ScaleStatus(str, Enum):
    SENSOR_METRIC = "SENSOR_METRIC"
    MODEL_METRIC_UNVERIFIED = "MODEL_METRIC_UNVERIFIED"
    MODEL_METRIC_VALIDATED = "MODEL_METRIC_VALIDATED"
    USER_CALIBRATED = "USER_CALIBRATED"
    RELATIVE_SCALE = "RELATIVE_SCALE"


class PreviewRepresentation(str, Enum):
    FUSED_POINTS = "fused_points"
    TSDF_RAYCAST = "tsdf_raycast"
    MESH = "mesh"
    GAUSSIANIZED_SURFELS = "gaussianized_surfels"


class _AlignmentQuality(Protocol):
    accepted: bool
    overlap: float
    inlier_ratio: float
    rmse_m: float


@dataclass(frozen=True)
class LiveFailurePolicy:
    archive_continues_on_preview_failure: bool = True
    rejected_pose_freezes_integration: bool = True
    raw_observations_are_authoritative: bool = True
    production_may_replace_live_trajectory: bool = True
    preview_failure_is_fatal_to_capture: bool = False


@dataclass(frozen=True)
class LiveSubmapDescriptor:
    id: str
    local_origin: tuple[float, ...]
    global_from_local: tuple[float, ...]
    state: str
    first_sequence: int
    last_sequence: int
    voxel_size_m: float
    voxel_count: int
    point_count: int
    observation_count: int
    confidence: float
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    resident: str

    def __post_init__(self) -> None:
        if len(self.local_origin) != 16 or len(self.global_from_local) != 16:
            raise ValueError("Submap transforms must be row-major 4x4 matrices")
        if self.first_sequence < 0 or self.last_sequence < self.first_sequence:
            raise ValueError("Submap sequence range is invalid")
        if self.voxel_size_m <= 0 or self.voxel_count < 0 or self.point_count < 0:
            raise ValueError("Submap geometry counts must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Submap confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "localOrigin": list(self.local_origin),
            "globalFromLocal": list(self.global_from_local),
            "state": self.state,
            "firstSequence": self.first_sequence,
            "lastSequence": self.last_sequence,
            "voxelSizeM": self.voxel_size_m,
            "voxelCount": self.voxel_count,
            "pointCount": self.point_count,
            "observationCount": self.observation_count,
            "confidence": self.confidence,
            "boundsMin": list(self.bounds_min),
            "boundsMax": list(self.bounds_max),
            "resident": self.resident,
        }


@dataclass(frozen=True)
class CoverageSummary:
    observed_ratio: float = 0.0
    weak_ratio: float = 0.0
    single_view_ratio: float = 0.0
    hole_boundary_ratio: float = 0.0
    guidance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value in (
            self.observed_ratio,
            self.weak_ratio,
            self.single_view_ratio,
            self.hole_boundary_ratio,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Coverage ratios must be in [0, 1]")

    def to_message(self, frame_sequence: int) -> dict[str, Any]:
        return {
            "contractVersion": LIVE_CONTRACT_VERSION,
            "frameSequence": frame_sequence,
            "observedRatio": self.observed_ratio,
            "weakRatio": self.weak_ratio,
            "singleViewRatio": self.single_view_ratio,
            "holeBoundaryRatio": self.hole_boundary_ratio,
            "guidance": list(self.guidance),
        }


@dataclass(frozen=True)
class LiveTelemetry:
    pose_latency_ms: float | None = None
    map_update_latency_ms: float | None = None
    map_update_hz: float = 0.0
    allocated_live_map_bytes: int = 0
    active_voxel_count: int = 0
    active_surfel_count: int = 0
    resident_submap_count: int = 0
    host_cached_submap_count: int = 0
    dropped_preview_jobs: int = 0
    tracking_queue_depth: int = 0
    mapping_queue_depth: int = 0
    degradation_level: int = 0

    def __post_init__(self) -> None:
        numeric = (
            self.map_update_hz,
            self.allocated_live_map_bytes,
            self.active_voxel_count,
            self.active_surfel_count,
            self.resident_submap_count,
            self.host_cached_submap_count,
            self.dropped_preview_jobs,
            self.tracking_queue_depth,
            self.mapping_queue_depth,
            self.degradation_level,
        )
        if any(value < 0 for value in numeric):
            raise ValueError("Live telemetry cannot contain negative counters")


def tracking_confidence(quality: _AlignmentQuality) -> float:
    if not quality.accepted:
        return 0.0
    residual_score = (
        max(0.0, 1.0 - quality.rmse_m / 0.03)
        if math.isfinite(quality.rmse_m)
        else 0.0
    )
    return max(
        0.0,
        min(
            1.0,
            0.45 * float(quality.overlap)
            + 0.45 * float(quality.inlier_ratio)
            + 0.10 * residual_score,
        ),
    )


def pose_uncertainty(quality: _AlignmentQuality) -> tuple[float | None, float | None]:
    if not quality.accepted or not math.isfinite(quality.rmse_m):
        return None, None
    overlap = max(float(quality.overlap), 0.05)
    position_mm = max(1.0, quality.rmse_m * 1000.0 / math.sqrt(overlap))
    angle_degrees = max(0.1, (1.0 - float(quality.inlier_ratio)) * 8.0)
    return position_mm, angle_degrees


def contract_status(
    value: dict[str, Any],
    *,
    telemetry: LiveTelemetry | None = None,
    failure_policy: LiveFailurePolicy | None = None,
) -> dict[str, Any]:
    state = TrackingState(str(value.get("state", TrackingState.FAILED.value)))
    telemetry = telemetry or LiveTelemetry(
        pose_latency_ms=value.get("poseLatencyMs"),
        map_update_latency_ms=value.get("mapUpdateLatencyMs"),
        map_update_hz=float(value.get("mapUpdateHz", 0.0)),
        allocated_live_map_bytes=int(value.get("allocatedLiveMapBytes", 0)),
        active_voxel_count=int(value.get("activeVoxelCount", 0)),
        active_surfel_count=int(value.get("activeSurfelCount", 0)),
        resident_submap_count=int(value.get("residentSubmapCount", 0)),
        host_cached_submap_count=int(value.get("hostCachedSubmapCount", 0)),
        dropped_preview_jobs=int(value.get("droppedPreviewJobs", 0)),
        tracking_queue_depth=int(value.get("trackingQueueDepth", 0)),
        mapping_queue_depth=int(value.get("mappingQueueDepth", 0)),
        degradation_level=int(value.get("degradationLevel", 0)),
    )
    result = dict(value)
    result.update(
        contractVersion=LIVE_CONTRACT_VERSION,
        trackingState=state.value,
        scaleStatus=str(value.get("scaleStatus", ScaleStatus.SENSOR_METRIC.value)),
        integrationFrozen=bool(
            value.get(
                "integrationFrozen",
                state
                in {
                    TrackingState.SEARCHING,
                    TrackingState.FROZEN,
                    TrackingState.FAILED,
                },
            )
        ),
        **{
            "poseLatencyMs": telemetry.pose_latency_ms,
            "mapUpdateLatencyMs": telemetry.map_update_latency_ms,
            "mapUpdateHz": telemetry.map_update_hz,
            "allocatedLiveMapBytes": telemetry.allocated_live_map_bytes,
            "activeVoxelCount": telemetry.active_voxel_count,
            "activeSurfelCount": telemetry.active_surfel_count,
            "residentSubmapCount": telemetry.resident_submap_count,
            "hostCachedSubmapCount": telemetry.host_cached_submap_count,
            "droppedPreviewJobs": telemetry.dropped_preview_jobs,
            "trackingQueueDepth": telemetry.tracking_queue_depth,
            "mappingQueueDepth": telemetry.mapping_queue_depth,
            "degradationLevel": telemetry.degradation_level,
        },
    )
    if failure_policy is not None:
        result["failurePolicy"] = asdict(failure_policy)
    return result


def submap_message(
    frame_sequence: int, submaps: list[LiveSubmapDescriptor]
) -> dict[str, Any]:
    return {
        "contractVersion": LIVE_CONTRACT_VERSION,
        "frameSequence": frame_sequence,
        "submaps": [submap.to_dict() for submap in submaps],
    }
