from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


BACKEND_POLICY_VERSION = 1
BENCHMARK_MANIFEST_VERSION = 1


class BackendPolicyError(ValueError):
    """Raised when a policy input or an explicit override is unsafe."""


@dataclass(frozen=True)
class CandidateAssessment:
    benchmark_id: str
    lane: str
    backend: str
    eligible: bool
    reasons: tuple[str, ...]
    metrics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmarkId": self.benchmark_id,
            "lane": self.lane,
            "backend": self.backend,
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
        }


_APPLICABLE_LANES = {
    "rgbd": ("liveRgbd", "productionCamera", "depthCompletion", "surface", "gaussian"),
    "photos": ("productionCamera", "surface", "gaussian"),
    "video": ("liveVideo", "productionCamera", "surface", "gaussian"),
    "hybrid": (
        "liveRgbd",
        "productionCamera",
        "depthCompletion",
        "surface",
        "gaussian",
    ),
}

_KNOWN_BACKENDS = {
    "liveRgbd": {
        "open3d-cuda-tsdf": ("open3d-cuda",),
        "open3d-cpu-tsdf": ("open3d-cpu",),
        "numpy-cpu-surfel": (),
    },
    "liveVideo": {"off": (), "lingbot-map": ("lingbot-map",)},
    "productionCamera": {
        "validated-rgbd-trajectory": (),
        "validated-learned-challenger-bakeoff": (),
        "colmap": (),
        "da3-guided-colmap": ("da3", "colmap-learned"),
        "mapanything-guided-colmap": ("mapanything", "colmap-learned"),
    },
    "depthCompletion": {
        "off": (),
        "lingbot": ("lingbot-depth",),
        "mapanything": ("mapanything",),
        "da3": ("da3",),
    },
    "surface": {"validated-dense-fusion": (), "neural-sdf": ("neural-sdf",)},
    "gaussian": {
        "off": (),
        "gsplat-2dgs": ("gsplat",),
        "gsplat-3dgs": ("gsplat",),
    },
}

_REQUIRED_METRICS = {
    "liveRgbd": ("acceptedFrameRatio", "poseP95Ms", "pointMapP95Ms"),
    "liveVideo": ("driftRisk", "acceptedFrameRatio", "firstGeometrySeconds"),
    "productionCamera": ("registrationRatio", "medianReprojectionErrorPx"),
    "depthCompletion": ("medianResidualMm", "inlierRatio"),
    "surface": ("heldOutError", "p95DisplacementVoxels"),
    "gaussian": ("medianPsnr", "medianSsim", "medianL1"),
}

_UNIT_INTERVAL_METRICS = {
    "acceptedFrameRatio",
    "driftRisk",
    "registrationRatio",
    "inlierRatio",
    "acceptedHoleRatio",
    "medianSsim",
    "medianL1",
}

_NONCOMMERCIAL_BACKENDS = {"da3", "da3-guided-colmap"}


def packaged_benchmark_manifest_path() -> Path:
    return Path(__file__).with_name("backend-benchmarks.json")


def load_benchmark_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = (path or packaged_benchmark_manifest_path()).resolve(strict=True)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(value.get("schemaVersion", 0)) != BENCHMARK_MANIFEST_VERSION:
        raise BackendPolicyError(
            f"Backend benchmark manifest must use schema {BENCHMARK_MANIFEST_VERSION}"
        )
    records = value.get("records")
    if not isinstance(records, list):
        raise BackendPolicyError("Backend benchmark manifest records must be a list")
    return value


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _normal_name(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _version_tuple(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        parts = tuple(int(part) for part in str(value).split("."))
    except ValueError:
        return None
    return parts if parts else None


def _runtime_available(runtimes: Mapping[str, Any], name: str) -> bool:
    value = runtimes.get(name)
    if isinstance(value, Mapping):
        return bool(value.get("available", False)) and bool(
            value.get("validated", value.get("available", False))
        )
    return bool(value)


def _runtime_revision(runtimes: Mapping[str, Any], name: str) -> str | None:
    value = runtimes.get(name)
    if not isinstance(value, Mapping):
        return None
    revision = value.get("revision")
    return str(revision) if revision is not None else None


def _source_compatibility(
    envelope: Mapping[str, Any], source: Mapping[str, Any]
) -> list[str]:
    reasons: list[str] = []
    source_kind = str(source.get("kind", ""))
    kinds = tuple(str(value) for value in envelope.get("kinds", ()))
    if kinds and source_kind not in kinds:
        reasons.append(f"source kind {source_kind or 'unknown'} is outside {list(kinds)}")

    sensor_kinds = tuple(str(value) for value in envelope.get("sensorKinds", ()))
    sensor_kind = str(source.get("sensorKind", ""))
    if sensor_kinds and sensor_kind not in sensor_kinds:
        reasons.append(
            f"sensor {sensor_kind or 'unknown'} is outside the benchmark sensor envelope"
        )

    frame_count = _finite_number(source.get("frameCount"))
    minimum_frames = _finite_number(envelope.get("minimumFrames"))
    maximum_frames = _finite_number(envelope.get("maximumFrames"))
    if minimum_frames is not None or maximum_frames is not None:
        if frame_count is None:
            reasons.append("source frame count is unknown")
        elif minimum_frames is not None and frame_count < minimum_frames:
            reasons.append("source has fewer frames than the benchmark envelope")
        elif maximum_frames is not None and frame_count > maximum_frames:
            reasons.append("source has more frames than the benchmark envelope")

    dimension = _finite_number(source.get("maximumImageDimension"))
    benchmark_dimension = _finite_number(envelope.get("maximumImageDimension"))
    if benchmark_dimension is not None:
        if dimension is None:
            reasons.append("source image dimension is unknown")
        elif dimension > benchmark_dimension:
            reasons.append("source resolution exceeds the measured benchmark envelope")

    required = set(str(value) for value in envelope.get("characteristics", ()))
    actual = set(str(value) for value in source.get("characteristics", ()))
    missing = sorted(required - actual)
    if missing:
        reasons.append(f"source lacks measured characteristics: {', '.join(missing)}")
    return reasons


def _hardware_compatibility(
    envelope: Mapping[str, Any],
    metrics: Mapping[str, Any],
    hardware: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    requires_cuda = bool(envelope.get("requiresCuda", False))
    if requires_cuda and not bool(hardware.get("cudaValidated", False)):
        reasons.append("CUDA has not passed a runtime kernel smoke test")

    gpu_names = {_normal_name(value) for value in envelope.get("gpuNames", ())}
    gpu_name = _normal_name(hardware.get("gpuName"))
    if gpu_names and gpu_name not in gpu_names:
        reasons.append("GPU model differs from the measured hardware envelope")

    actual_capability = _version_tuple(hardware.get("cudaCapability"))
    minimum_capability = _version_tuple(envelope.get("minimumCudaCapability"))
    if minimum_capability is not None and (
        actual_capability is None or actual_capability < minimum_capability
    ):
        reasons.append("CUDA compute capability is below the benchmark requirement")

    total_vram = _finite_number(hardware.get("vramTotalMiB"))
    free_vram = _finite_number(hardware.get("vramFreeMiB"))
    minimum_vram = _finite_number(envelope.get("minimumVramMiB"))
    peak_vram = _finite_number(metrics.get("peakVramMiB"))
    reserve_vram = _finite_number(envelope.get("reserveVramMiB")) or 1024.0
    required_vram = max(minimum_vram or 0.0, (peak_vram or 0.0) + reserve_vram)
    if required_vram > 0.0 and (total_vram is None or total_vram < required_vram):
        reasons.append(
            f"total VRAM does not provide the measured {required_vram:.0f} MiB safety envelope"
        )
    if peak_vram is not None and free_vram is not None and free_vram < peak_vram + 256.0:
        reasons.append("currently free VRAM is below the measured working set plus launch reserve")

    minimum_cpu_threads = _finite_number(envelope.get("minimumCpuThreads"))
    cpu_threads = _finite_number(hardware.get("cpuThreads"))
    if minimum_cpu_threads is not None and (
        cpu_threads is None or cpu_threads < minimum_cpu_threads
    ):
        reasons.append("CPU thread count is below the measured hardware envelope")
    return reasons


def _assessment(
    record: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    hardware: Mapping[str, Any],
    runtimes: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> CandidateAssessment:
    benchmark_id = str(record.get("benchmarkId", ""))
    lane = str(record.get("lane", ""))
    backend = str(record.get("backend", ""))
    if not benchmark_id or not lane or not backend:
        raise BackendPolicyError("Each benchmark requires benchmarkId, lane, and backend")
    reasons: list[str] = []
    if backend not in _KNOWN_BACKENDS.get(lane, {}):
        reasons.append("backend is not implemented for this policy lane")
    if not bool(record.get("accepted", False)):
        reasons.append("benchmark outcome was not accepted")
    gates = record.get("gates", {})
    if not isinstance(gates, Mapping) or not gates:
        reasons.append("benchmark has no release-gate evidence")
    elif any(value is not True for value in gates.values()):
        reasons.append("one or more benchmark release gates did not pass")

    source_envelope = record.get("source", {})
    hardware_envelope = record.get("hardware", {})
    metrics = record.get("metrics", {})
    if not isinstance(source_envelope, Mapping) or not isinstance(hardware_envelope, Mapping):
        raise BackendPolicyError(f"Benchmark {benchmark_id} has an invalid envelope")
    if not isinstance(metrics, Mapping):
        raise BackendPolicyError(f"Benchmark {benchmark_id} metrics must be an object")
    for metric in _REQUIRED_METRICS.get(lane, ()):
        value = _finite_number(metrics.get(metric))
        if value is None:
            reasons.append(f"required quality metric {metric} is missing or non-finite")
        elif value < 0.0:
            reasons.append(f"required quality metric {metric} cannot be negative")
    for metric in _UNIT_INTERVAL_METRICS:
        value = _finite_number(metrics.get(metric))
        if value is not None and not 0.0 <= value <= 1.0:
            reasons.append(f"quality metric {metric} must be between zero and one")
    reasons.extend(_source_compatibility(source_envelope, source))
    reasons.extend(_hardware_compatibility(hardware_envelope, metrics, hardware))

    requirements = tuple(str(value) for value in record.get("requiresRuntimes", ()))
    for runtime in requirements:
        if not _runtime_available(runtimes, runtime):
            reasons.append(f"runtime {runtime} is unavailable or has not been validated")
    revisions = record.get("runtimeRevisions", {})
    if not isinstance(revisions, Mapping):
        raise BackendPolicyError(f"Benchmark {benchmark_id} runtimeRevisions must be an object")
    for runtime, expected in revisions.items():
        actual = _runtime_revision(runtimes, str(runtime))
        if actual != str(expected):
            reasons.append(
                f"runtime {runtime} revision is {actual or 'unknown'}, expected {expected}"
            )

    if bool(quality.get("commercialUse", False)) and (
        backend in _NONCOMMERCIAL_BACKENDS
        or not bool(record.get("commercialUseAllowed", True))
    ):
        reasons.append("backend license does not allow the requested commercial use")
    return CandidateAssessment(
        benchmark_id,
        lane,
        backend,
        not reasons,
        tuple(reasons),
        metrics,
    )


def _metric(metrics: Mapping[str, Any], name: str, default: float) -> float:
    value = _finite_number(metrics.get(name))
    return default if value is None else value


def _rank_key(candidate: CandidateAssessment) -> tuple[float, ...]:
    metrics = candidate.metrics
    if candidate.lane == "liveRgbd":
        return (
            -_metric(metrics, "acceptedFrameRatio", -1.0),
            _metric(metrics, "poseP95Ms", math.inf),
            _metric(metrics, "pointMapP95Ms", math.inf),
            _metric(metrics, "peakVramMiB", math.inf),
        )
    if candidate.lane == "liveVideo":
        return (
            _metric(metrics, "driftRisk", math.inf),
            -_metric(metrics, "acceptedFrameRatio", -1.0),
            _metric(metrics, "firstGeometrySeconds", math.inf),
            _metric(metrics, "peakVramMiB", math.inf),
        )
    if candidate.lane == "productionCamera":
        return (
            -_metric(metrics, "registrationRatio", -1.0),
            _metric(metrics, "medianCameraResidual", math.inf),
            _metric(metrics, "medianReprojectionErrorPx", math.inf),
            _metric(metrics, "wallSeconds", math.inf),
        )
    if candidate.lane == "depthCompletion":
        return (
            _metric(metrics, "medianResidualMm", math.inf),
            -_metric(metrics, "inlierRatio", -1.0),
            -_metric(metrics, "acceptedHoleRatio", -1.0),
            _metric(metrics, "peakVramMiB", math.inf),
        )
    if candidate.lane == "surface":
        return (
            _metric(metrics, "heldOutError", math.inf),
            _metric(metrics, "p95DisplacementVoxels", math.inf),
            _metric(metrics, "wallSeconds", math.inf),
        )
    if candidate.lane == "gaussian":
        return (
            -_metric(metrics, "medianPsnr", -1.0),
            -_metric(metrics, "medianSsim", -1.0),
            _metric(metrics, "medianL1", math.inf),
            _metric(metrics, "peakVramMiB", math.inf),
        )
    return (candidate.backend,)


def _baseline(lane: str, source_kind: str, runtimes: Mapping[str, Any]) -> str:
    if lane == "liveRgbd":
        if _runtime_available(runtimes, "open3d-cuda"):
            return "open3d-cuda-tsdf"
        if _runtime_available(runtimes, "open3d-cpu"):
            return "open3d-cpu-tsdf"
        return "numpy-cpu-surfel"
    if lane == "liveVideo":
        return "off"
    if lane == "productionCamera":
        return (
            "validated-learned-challenger-bakeoff"
            if source_kind in {"photos", "video", "hybrid"}
            else "validated-rgbd-trajectory"
        )
    if lane == "depthCompletion":
        return "off"
    if lane == "surface":
        return "validated-dense-fusion"
    if lane == "gaussian":
        if not _runtime_available(runtimes, "gsplat"):
            return "off"
        return "gsplat-2dgs" if source_kind in {"rgbd", "hybrid"} else "gsplat-3dgs"
    raise BackendPolicyError(f"Unknown backend lane: {lane}")


def _manual_override(
    lane: str,
    override: str,
    runtimes: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    known = _KNOWN_BACKENDS.get(lane, {})
    matching = [
        record
        for record in records
        if str(record.get("lane")) == lane and str(record.get("backend")) == override
    ]
    if override not in known:
        raise BackendPolicyError(f"Unknown explicit {lane} backend: {override}")
    required = set(known.get(override, ()))
    if matching:
        required.update(
            str(runtime)
            for record in matching
            for runtime in record.get("requiresRuntimes", ())
        )
    unavailable = sorted(runtime for runtime in required if not _runtime_available(runtimes, runtime))
    if unavailable:
        raise BackendPolicyError(
            f"Explicit {lane} backend {override} requires unavailable runtime(s): "
            + ", ".join(unavailable)
        )
    if bool(quality.get("commercialUse", False)) and override in _NONCOMMERCIAL_BACKENDS:
        raise BackendPolicyError(
            f"Explicit {lane} backend {override} is not licensed for commercial use"
        )
    if matching:
        if bool(quality.get("commercialUse", False)) and all(
            not bool(record.get("commercialUseAllowed", True)) for record in matching
        ):
            raise BackendPolicyError(
                f"Explicit {lane} backend {override} is not licensed for commercial use"
            )
    return {
        "selected": override,
        "selectionMode": "explicit-override",
        "benchmarked": False,
        "evidence": [],
        "fallbackChain": [],
        "reason": "The explicit user/backend override takes precedence over automatic ranking.",
    }


def select_backend_policy(
    *,
    source: Mapping[str, Any],
    hardware: Mapping[str, Any],
    runtimes: Mapping[str, Any],
    quality: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    records: Iterable[Mapping[str, Any]] | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Select a reproducible backend plan from compatible accepted measurements.

    Automatic selection is deliberately fail-closed: an inference runtime being installed is
    not benchmark evidence. A record must match the source envelope, runtime revisions, license,
    kernel validation, compute capability, and memory headroom. If none does, the current
    validation-gated baseline is retained and identified as unbenchmarked for this input.
    """

    source_kind = str(source.get("kind", ""))
    if source_kind not in _APPLICABLE_LANES:
        raise BackendPolicyError(
            "Backend policy source kind must be rgbd, photos, video, or hybrid"
        )
    quality = dict(quality or {})
    overrides = dict(overrides or {})
    if records is None:
        manifest = load_benchmark_manifest(manifest_path)
        record_list = list(manifest["records"])
        manifest_revision = str(manifest.get("revision", "unknown"))
    else:
        record_list = list(records)
        manifest_revision = "caller-supplied"

    lanes = _APPLICABLE_LANES[source_kind]
    assessments = [
        _assessment(
            record,
            source=source,
            hardware=hardware,
            runtimes=runtimes,
            quality=quality,
        )
        for record in record_list
        if str(record.get("lane", "")) in lanes
    ]
    decisions: dict[str, Any] = {}
    for lane in lanes:
        override = str(overrides.get(lane, "auto") or "auto")
        if override != "auto":
            decisions[lane] = _manual_override(
                lane, override, runtimes, record_list, quality
            )
            continue
        eligible = sorted(
            (assessment for assessment in assessments if assessment.lane == lane and assessment.eligible),
            key=_rank_key,
        )
        baseline = _baseline(lane, source_kind, runtimes)
        if eligible:
            selected = eligible[0]
            fallback = []
            for backend in [*(candidate.backend for candidate in eligible[1:]), baseline]:
                if backend != selected.backend and backend not in fallback:
                    fallback.append(backend)
            decisions[lane] = {
                "selected": selected.backend,
                "selectionMode": "compatible-benchmark",
                "benchmarked": True,
                "evidence": [selected.benchmark_id],
                "fallbackChain": fallback,
                "reason": (
                    "Selected by the lane's quality-first ordering from compatible, "
                    "release-gated benchmark evidence."
                ),
            }
        else:
            decisions[lane] = {
                "selected": baseline,
                "selectionMode": "protected-baseline",
                "benchmarked": False,
                "evidence": [],
                "fallbackChain": [],
                "reason": (
                    "No accepted benchmark matched this source, hardware, runtime revision, "
                    "license, and memory envelope; retained the established guarded path."
                ),
            }

    return {
        "schemaVersion": BACKEND_POLICY_VERSION,
        "kind": "scanlan-adaptive-backend-policy",
        "selectedAt": datetime.now(timezone.utc).isoformat(),
        "manifestRevision": manifest_revision,
        "source": dict(source),
        "hardware": dict(hardware),
        "quality": quality,
        "runtimeEvidence": dict(runtimes),
        "decisions": decisions,
        "candidateAssessments": [assessment.to_dict() for assessment in assessments],
    }
