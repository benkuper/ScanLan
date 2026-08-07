from __future__ import annotations

import math
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .mesh_observations import (
    FREE_SPACE_VIOLATION,
    MISSING_DEPTH,
    OCCLUDED,
    OUTSIDE_VIEW,
    SUPPORTED,
    classify_world_points,
)
from .io import write_json


MESH_REPAIR_REPORT_SCHEMA_VERSION = 1
MESH_REPAIR_ALGORITHM_VERSION = "1.0.0"
MeshRepairProfile = Literal["faithful", "architectural", "natural", "watertight"]


@dataclass(frozen=True)
class MeshRepairSettings:
    enabled: bool = True
    profile: MeshRepairProfile = "faithful"
    max_hole_diameter_m: float | None = None
    min_support_ratio: float = 0.60
    max_free_space_ratio: float = 0.01
    min_supporting_views: int = 2
    fill_inferred_holes: bool = False
    repair_non_manifold: bool = True
    repair_self_intersections: bool = False
    produce_watertight_copy: bool = False
    allow_unrepaired_fallback: bool = True

    def resolved_max_hole_diameter_m(self, mesh_voxel_size_m: float) -> float:
        if self.max_hole_diameter_m is not None:
            return max(0.0, float(self.max_hole_diameter_m))
        return float(np.clip(12.0 * mesh_voxel_size_m, 0.04, 0.15))

    def validate(self) -> None:
        if self.profile not in {"faithful", "architectural", "natural", "watertight"}:
            raise ValueError(f"Unknown mesh repair profile: {self.profile}")
        if not 0.0 <= self.min_support_ratio <= 1.0:
            raise ValueError("min_support_ratio must be between 0 and 1")
        if not 0.0 <= self.max_free_space_ratio <= 1.0:
            raise ValueError("max_free_space_ratio must be between 0 and 1")
        if self.min_supporting_views < 1:
            raise ValueError("min_supporting_views must be at least 1")

    def report_payload(self, mesh_voxel_size_m: float) -> dict[str, Any]:
        payload = asdict(self)
        payload["max_hole_diameter_m"] = self.resolved_max_hole_diameter_m(
            mesh_voxel_size_m
        )
        return payload


def settings_from_project(project: dict[str, Any]) -> MeshRepairSettings:
    values = project.get("settings", {})
    return MeshRepairSettings(
        enabled=bool(values.get("repairMesh", True)),
        profile=str(values.get("meshRepairProfile", "faithful")),  # type: ignore[arg-type]
        fill_inferred_holes=bool(values.get("fillInferredMeshHoles", False)),
        produce_watertight_copy=bool(values.get("produceWatertightMesh", False)),
    )


def _plane_axes(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(normal, dtype=np.float64)
    length = float(np.linalg.norm(normal))
    if length <= 1e-12:
        normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        normal = normal / length
    reference = np.zeros(3, dtype=np.float64)
    reference[int(np.argmin(np.abs(normal)))] = 1.0
    axis_u = np.cross(normal, reference)
    axis_u /= max(float(np.linalg.norm(axis_u)), 1e-12)
    axis_v = np.cross(normal, axis_u)
    return axis_u, axis_v


def _inside_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        if (y1 > y) != (y2 > y):
            crossing = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing:
                inside = not inside
        previous = current
    return inside


def _boundary_distance(point: np.ndarray, polygon: np.ndarray) -> float:
    minimum = math.inf
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        edge = end - start
        length_squared = float(np.dot(edge, edge))
        if length_squared <= 1e-20:
            distance = float(np.linalg.norm(point - start))
        else:
            position = float(np.clip(np.dot(point - start, edge) / length_squared, 0.0, 1.0))
            distance = float(np.linalg.norm(point - (start + position * edge)))
        minimum = min(minimum, distance)
    return minimum


def _halton(index: int, base: int) -> float:
    result = 0.0
    fraction = 1.0 / base
    while index:
        result += fraction * (index % base)
        index //= base
        fraction /= base
    return result


def sample_boundary_loop(
    loop: dict[str, Any], mesh_voxel_size_m: float
) -> np.ndarray:
    """Generate deterministic interior samples, buffered from depth discontinuities."""

    positions = np.asarray(loop["orderedBoundaryPositions"], dtype=np.float64)
    plane = loop["bestFitPlane"]
    origin = np.asarray(plane["origin"], dtype=np.float64)
    normal = np.asarray(plane["normal"], dtype=np.float64)
    axis_u, axis_v = _plane_axes(normal)
    relative = positions - origin
    polygon = np.column_stack((relative @ axis_u, relative @ axis_v))
    area = max(float(loop.get("approximateEnclosedAreaM2", 0.0)), 1e-8)
    target = int(np.clip(round(area / max(mesh_voxel_size_m**2, 1e-8) * 2.0), 32, 256))
    minimum = polygon.min(axis=0)
    maximum = polygon.max(axis=0)
    diameter = max(float(loop.get("diameterM", 0.0)), mesh_voxel_size_m)
    margin = min(mesh_voxel_size_m * 0.75, diameter * 0.08)

    selected: list[np.ndarray] = []
    for pass_index in range(2):
        selected.clear()
        pass_margin = margin if pass_index == 0 else margin * 0.35
        for sequence in range(1, 20_001):
            point = minimum + np.asarray(
                [_halton(sequence, 2), _halton(sequence, 3)], dtype=np.float64
            ) * (maximum - minimum)
            if _inside_polygon(point, polygon) and _boundary_distance(point, polygon) >= pass_margin:
                selected.append(point)
                if len(selected) >= target:
                    break
        if len(selected) >= min(32, target):
            break
    if not selected:
        selected.append(np.mean(polygon, axis=0))
    samples_2d = np.asarray(selected, dtype=np.float64)
    return (
        origin
        + samples_2d[:, :1] * axis_u
        + samples_2d[:, 1:] * axis_v
    ).astype(np.float64)


def _geometric_classification(loop: dict[str, Any], mesh_voxel_size_m: float) -> str:
    diameter = max(float(loop.get("diameterM", 0.0)), mesh_voxel_size_m)
    residual = float(loop.get("planeRmsResidualM", math.inf))
    coherence = float(loop.get("boundaryNormalCoherence", 0.0))
    planar = residual <= max(mesh_voxel_size_m, diameter * 0.03) and coherence >= 0.70
    return "planar" if planar else "freeform"


def classify_boundary_loop(
    loop: dict[str, Any],
    frames: list[Any],
    mesh_voxel_size_m: float,
    settings: MeshRepairSettings,
) -> dict[str, Any]:
    settings.validate()
    samples = sample_boundary_loop(loop, mesh_voxel_size_m)
    counts = {
        OUTSIDE_VIEW: 0,
        MISSING_DEPTH: 0,
        SUPPORTED: 0,
        FREE_SPACE_VIOLATION: 0,
        OCCLUDED: 0,
    }
    supporting_views = 0
    useful_view_count = 0
    for frame in frames:
        if frame.depthless:
            continue
        evidence = classify_world_points(samples, frame, mesh_voxel_size_m)
        frame_counts = {name: int(np.count_nonzero(evidence == name)) for name in counts}
        for name, count in frame_counts.items():
            counts[name] += count
        metric_count = frame_counts[SUPPORTED] + frame_counts[FREE_SPACE_VIOLATION] + frame_counts[OCCLUDED]
        if metric_count:
            useful_view_count += 1
        if frame_counts[SUPPORTED] >= max(3, math.ceil(len(samples) * 0.15)):
            supporting_views += 1

    metric_evidence = counts[SUPPORTED] + counts[FREE_SPACE_VIOLATION] + counts[OCCLUDED]
    denominator = max(metric_evidence, 1)
    support_ratio = counts[SUPPORTED] / denominator
    free_space_ratio = counts[FREE_SPACE_VIOLATION] / denominator
    occluded_ratio = counts[OCCLUDED] / denominator
    max_diameter = settings.resolved_max_hole_diameter_m(mesh_voxel_size_m)
    diameter = float(loop.get("diameterM", math.inf))
    coherence = float(loop.get("boundaryNormalCoherence", 0.0))
    residual = float(loop.get("planeRmsResidualM", math.inf))
    geometric = _geometric_classification(loop, mesh_voxel_size_m)

    if free_space_ratio > settings.max_free_space_ratio:
        classification = "preserve_opening"
    elif diameter > max_diameter:
        classification = "preserve_too_large"
    elif (
        support_ratio >= settings.min_support_ratio
        and supporting_views >= settings.min_supporting_views
    ):
        classification = "fill_measured"
    elif occluded_ratio >= 0.25 and counts[SUPPORTED] == 0:
        classification = "preserve_occluded"
    elif (
        settings.fill_inferred_holes
        and coherence >= 0.85
        and residual <= max(2.0 * mesh_voxel_size_m, diameter * 0.04)
        and counts[FREE_SPACE_VIOLATION] == 0
    ):
        classification = "fill_inferred"
    else:
        classification = "preserve_unknown"

    return {
        "loopId": str(loop["loopId"]),
        "classification": classification,
        "geometricClassification": geometric,
        "sampleCount": len(samples),
        "supportRatio": round(support_ratio, 6),
        "freeSpaceViolationRatio": round(free_space_ratio, 6),
        "occludedRatio": round(occluded_ratio, 6),
        "supportingViewCount": supporting_views,
        "usefulMetricViewCount": useful_view_count,
        "evidenceCounts": counts,
        "diameterM": diameter,
        "areaM2": float(loop.get("approximateEnclosedAreaM2", 0.0)),
        "bestFitPlane": loop["bestFitPlane"],
        "planeRmsResidualM": residual,
        "boundaryNormalCoherence": coherence,
    }


def classify_topology_report(
    topology_report: dict[str, Any],
    frames: list[Any],
    mesh_voxel_size_m: float,
    settings: MeshRepairSettings,
) -> dict[str, Any]:
    decisions = [
        classify_boundary_loop(loop, frames, mesh_voxel_size_m, settings)
        for loop in topology_report.get("boundaryLoops", [])
    ]
    selected = [
        decision["loopId"]
        for decision in decisions
        if decision["classification"] in {"fill_measured", "fill_inferred"}
    ]
    summary: dict[str, int] = {}
    for decision in decisions:
        classification = str(decision["classification"])
        summary[classification] = summary.get(classification, 0) + 1
    return {
        "schemaVersion": MESH_REPAIR_REPORT_SCHEMA_VERSION,
        "algorithmVersion": MESH_REPAIR_ALGORITHM_VERSION,
        "status": "classified",
        "settings": settings.report_payload(mesh_voxel_size_m),
        "topology": topology_report.get("topology", {}),
        "inputMeshFingerprint": topology_report.get("inputMeshFingerprint"),
        "holes": decisions,
        "selectedLoopIds": selected,
        "summary": summary,
    }


def _mesh_fingerprint(vertices: np.ndarray, triangles: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(vertices, dtype="<f4").tobytes())
    digest.update(np.asarray(triangles, dtype="<i8").tobytes())
    return digest.hexdigest()


def _depth_dataset_fingerprint(frames: list[Any]) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        if frame.depthless:
            continue
        source = frame.source
        camera = source.camera
        record = source.frames[frame.frame_index]
        stat = record.depth_path.stat()
        digest.update(str(record.depth_path.resolve()).encode("utf-8"))
        digest.update(np.asarray([stat.st_size, stat.st_mtime_ns], dtype="<i8").tobytes())
        digest.update(
            np.asarray(
                [
                    camera.width,
                    camera.height,
                    camera.fx,
                    camera.fy,
                    camera.cx,
                    camera.cy,
                    camera.depth_scale,
                    camera.max_depth_m,
                ],
                dtype="<f8",
            ).tobytes()
        )
        digest.update(np.asarray(frame.camera_to_global, dtype="<f8").tobytes())
        digest.update(bytes((int(frame.image_y_up),)))
    return digest.hexdigest()


def _write_triangle_ply(path: Path, vertices: np.ndarray, triangles: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    with temporary.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(
            "ply\n"
            "format ascii 1.0\n"
            "comment ScanLan metric triangle mesh\n"
            f"element vertex {len(vertices)}\n"
            "property double x\n"
            "property double y\n"
            "property double z\n"
            f"element face {len(triangles)}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        )
        for x, y, z in vertices:
            handle.write(f"{x:.17g} {y:.17g} {z:.17g}\n")
        for a, b, c in triangles:
            handle.write(f"3 {a} {b} {c}\n")
    os.replace(temporary, path)


def _read_triangle_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="ascii") as handle:
        header: list[str] = []
        for line in handle:
            line = line.strip()
            header.append(line)
            if line == "end_header":
                break
        if not header or header[0] != "ply" or "format ascii 1.0" not in header:
            raise RuntimeError("The repaired mesh is not an ASCII PLY file")
        vertex_count = 0
        face_count = 0
        for line in header:
            fields = line.split()
            if fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
            elif fields[:2] == ["element", "face"]:
                face_count = int(fields[2])
        if vertex_count <= 0 or face_count <= 0:
            raise RuntimeError("The repaired PLY contains no triangle mesh")
        vertices = np.asarray(
            [[float(value) for value in handle.readline().split()[:3]] for _ in range(vertex_count)],
            dtype=np.float32,
        )
        faces: list[list[int]] = []
        for _ in range(face_count):
            values = [int(value) for value in handle.readline().split()]
            if not values or values[0] != 3 or len(values) < 4:
                raise RuntimeError("The repaired PLY contains a non-triangle face")
            faces.append(values[1:4])
    triangles = np.asarray(faces, dtype=np.int64)
    if int(triangles.max(initial=-1)) >= len(vertices):
        raise RuntimeError("The repaired PLY references an invalid vertex")
    return vertices, triangles


def find_mesh_repair_backend() -> Path | None:
    executable_name = "scanlan-mesh-repair.exe" if os.name == "nt" else "scanlan-mesh-repair"
    candidates: list[Path] = []
    configured = os.environ.get("SCANLAN_MESH_REPAIR")
    if configured:
        candidates.append(Path(configured))
    runtime_root = Path(sys.executable).resolve().parent
    candidates.extend(
        [
            runtime_root / executable_name,
            runtime_root / "mesh-repair" / executable_name,
            runtime_root.parent / "mesh-repair" / executable_name,
        ]
    )
    repository_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            repository_root / "build" / "mesh-repair-cgal" / "Release" / executable_name,
            repository_root / "build" / "mesh-repair" / "Release" / executable_name,
            repository_root / "build" / "mesh-repair" / executable_name,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _run_backend(arguments: list[str], report_path: Path, timeout_seconds: int = 600) -> dict[str, Any]:
    completed = subprocess.run(
        arguments,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    report: dict[str, Any] = {}
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = {}
    if completed.returncode != 0:
        message = str(report.get("error", {}).get("message", "")).strip()
        if not message:
            message = completed.stderr.strip() or "Native mesh repair failed"
        raise RuntimeError(message)
    if report.get("status") not in {"ok", "classified"}:
        raise RuntimeError("Native mesh repair returned an invalid report")
    return report


def _backend_version(backend: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(backend), "version", "--json"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "Mesh-repair backend did not start")
    try:
        version = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Mesh-repair backend returned an invalid version response") from error
    if (
        version.get("schemaVersion") != MESH_REPAIR_REPORT_SCHEMA_VERSION
        or version.get("algorithmVersion") != MESH_REPAIR_ALGORITHM_VERSION
        or version.get("backend", {}).get("name") != "CGAL"
    ):
        raise RuntimeError("The bundled mesh-repair backend is incompatible with this worker")
    return version


def _repair_summary(
    classification: dict[str, Any], native: dict[str, Any] | None, *, fallback: bool
) -> dict[str, Any]:
    decisions = classification.get("summary", {})
    operations = native.get("operations", {}) if native else {}
    defects_fixed = sum(
        int(operations.get(name, 0))
        for name in (
            "duplicateVerticesRemoved",
            "duplicateTrianglesRemoved",
            "degenerateTrianglesRemoved",
            "topologicallyIncompatibleTrianglesRemoved",
            "coincidentBorderPairsStitched",
            "nonManifoldVerticesDuplicated",
        )
    )
    return {
        "topologyDefectsFixed": defects_fixed,
        "holesFilled": len(native.get("filledLoops", [])) if native else 0,
        "openingsPreserved": int(decisions.get("preserve_opening", 0)),
        "occludedBoundariesPreserved": int(decisions.get("preserve_occluded", 0)),
        "unknownBoundariesPreserved": int(decisions.get("preserve_unknown", 0)),
        "largeBoundariesPreserved": int(decisions.get("preserve_too_large", 0)),
        "fallbackOccurred": fallback,
    }


def repair_mesh_geometry(
    output_dir: Path,
    vertices: np.ndarray,
    triangles: np.ndarray,
    frames: list[Any],
    mesh_voxel_size_m: float,
    settings: MeshRepairSettings,
    progress: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Analyze, classify, and repair geometry before any texture projection."""

    settings.validate()
    final_report_path = output_dir / "mesh-repair-report.json"
    raw_fingerprint = _mesh_fingerprint(vertices, triangles)
    settings_fingerprint = hashlib.sha256(
        (
            raw_fingerprint
            + MESH_REPAIR_ALGORITHM_VERSION
            + json.dumps(
                settings.report_payload(mesh_voxel_size_m),
                sort_keys=True,
                separators=(",", ":"),
            )
            + f"{mesh_voxel_size_m:.17g}"
        ).encode("utf-8")
    ).hexdigest()[:24]
    disabled_classification = {
        "schemaVersion": MESH_REPAIR_REPORT_SCHEMA_VERSION,
        "algorithmVersion": MESH_REPAIR_ALGORITHM_VERSION,
        "status": "disabled",
        "settings": settings.report_payload(mesh_voxel_size_m),
        "summary": {},
        "holes": [],
        "selectedLoopIds": [],
        "rawMeshFingerprint": raw_fingerprint,
        "repairCacheFingerprint": settings_fingerprint,
    }
    if not settings.enabled:
        disabled_classification["repairSummary"] = _repair_summary(
            disabled_classification, None, fallback=False
        )
        write_json(final_report_path, disabled_classification)
        return vertices, triangles, disabled_classification

    backend = find_mesh_repair_backend()
    if backend is None:
        report = {
            **disabled_classification,
            "status": "fallback",
            "fallbackReason": "The bundled CGAL mesh-repair backend was not found",
        }
        report["repairSummary"] = _repair_summary(report, None, fallback=True)
        write_json(final_report_path, report)
        if settings.allow_unrepaired_fallback:
            return vertices, triangles, report
        raise RuntimeError(report["fallbackReason"])
    try:
        backend_version = _backend_version(backend)
    except Exception as error:
        report = {
            **disabled_classification,
            "status": "fallback",
            "fallbackReason": str(error),
        }
        report["repairSummary"] = _repair_summary(report, None, fallback=True)
        write_json(final_report_path, report)
        if settings.allow_unrepaired_fallback:
            return vertices, triangles, report
        raise

    cache_root = output_dir / "cache" / "mesh-repair"
    cache_root.mkdir(parents=True, exist_ok=True)
    raw_path = cache_root / f"raw-{raw_fingerprint[:24]}.ply"
    if not raw_path.is_file():
        _write_triangle_ply(raw_path, vertices, triangles)
    dataset_fingerprint = _depth_dataset_fingerprint(frames)
    cache_digest = hashlib.sha256()
    cache_digest.update(raw_fingerprint.encode("ascii"))
    cache_digest.update(MESH_REPAIR_ALGORITHM_VERSION.encode("ascii"))
    cache_digest.update(
        json.dumps(
            settings.report_payload(mesh_voxel_size_m), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    cache_digest.update(np.asarray([mesh_voxel_size_m], dtype="<f8").tobytes())
    cache_digest.update(dataset_fingerprint.encode("ascii"))
    cache_key = cache_digest.hexdigest()[:24]
    repaired_path = cache_root / f"repaired-{cache_key}.ply"
    cached_report_path = cache_root / f"report-{cache_key}.json"
    if repaired_path.is_file() and cached_report_path.is_file():
        cached_report = json.loads(cached_report_path.read_text(encoding="utf-8"))
        if cached_report.get("status") == "ok":
            cached_report["repairedCacheHit"] = True
            cached_report["repairCacheFingerprint"] = cache_key
            write_json(final_report_path, cached_report)
            repaired_vertices, repaired_triangles = _read_triangle_ply(repaired_path)
            return repaired_vertices, repaired_triangles, cached_report

    topology_path = cache_root / f"topology-{raw_fingerprint[:24]}.json"
    if progress:
        progress("Analyzing topology", "Finding mesh defects and boundary loops", 0, None, 0.60)
    topology = _run_backend(
        [
            str(backend),
            "analyze",
            "--input",
            str(raw_path),
            "--report",
            str(topology_path),
            "--voxel-size-m",
            str(mesh_voxel_size_m),
        ],
        topology_path,
    )
    if progress:
        progress(
            "Checking openings against depth",
            f"Projecting {len(topology.get('boundaryLoops', []))} boundaries into original depth frames",
            0,
            None,
            0.66,
        )
    classification = classify_topology_report(topology, frames, mesh_voxel_size_m, settings)
    selected_by_id = {
        decision["loopId"]: decision
        for decision in classification["holes"]
        if decision["classification"] in {"fill_measured", "fill_inferred"}
    }
    policy = {
        "schemaVersion": MESH_REPAIR_REPORT_SCHEMA_VERSION,
        "algorithmVersion": MESH_REPAIR_ALGORITHM_VERSION,
        "inputMeshFingerprint": topology["inputMeshFingerprint"],
        "profile": settings.profile if settings.profile != "watertight" else "faithful",
        "repairNonManifold": settings.repair_non_manifold,
        "repairSelfIntersections": settings.repair_self_intersections,
        "selectedLoops": [selected_by_id[key] for key in sorted(selected_by_id)],
    }
    policy_path = cache_root / f"policy-{cache_key}.json"
    native_report_path = cache_root / f"native-{cache_key}.json"
    write_json(policy_path, policy)
    if progress:
        progress(
            "Repairing supported holes",
            f"Repairing {len(selected_by_id)} depth-supported boundaries with the {settings.profile} profile",
            0,
            None,
            0.72,
        )
    try:
        native = _run_backend(
            [
                str(backend),
                "repair",
                "--input",
                str(raw_path),
                "--policy",
                str(policy_path),
                "--output",
                str(repaired_path),
                "--report",
                str(native_report_path),
            ],
            native_report_path,
        )
    except Exception as error:
        report = {
            **classification,
            "status": "fallback",
            "fallbackReason": str(error),
            "rawMeshFingerprint": raw_fingerprint,
            "depthDatasetFingerprint": dataset_fingerprint,
            "repairCacheFingerprint": cache_key,
        }
        report["repairSummary"] = _repair_summary(report, None, fallback=True)
        write_json(final_report_path, report)
        if settings.allow_unrepaired_fallback:
            return vertices, triangles, report
        raise

    if progress:
        progress("Validating repaired mesh", "Rechecking topology before texturing", 0, None, 0.78)
    validation_path = cache_root / f"validation-{cache_key}.json"
    try:
        validation = _run_backend(
            [
                str(backend),
                "analyze",
                "--input",
                str(repaired_path),
                "--report",
                str(validation_path),
                "--voxel-size-m",
                str(mesh_voxel_size_m),
            ],
            validation_path,
        )
        if int(validation["topology"]["nonManifoldVertexCount"]) > int(
            topology["topology"]["nonManifoldVertexCount"]
        ):
            raise RuntimeError("Mesh repair increased non-manifold topology")
    except Exception as error:
        repaired_path.unlink(missing_ok=True)
        report = {
            **classification,
            "status": "fallback",
            "fallbackReason": f"Repaired mesh validation failed: {error}",
            "rawMeshFingerprint": raw_fingerprint,
            "depthDatasetFingerprint": dataset_fingerprint,
            "nativeRepair": native,
            "backendVersion": backend_version,
            "repairCacheFingerprint": cache_key,
        }
        report["repairSummary"] = _repair_summary(report, None, fallback=True)
        write_json(final_report_path, report)
        if settings.allow_unrepaired_fallback:
            return vertices, triangles, report
        raise

    report = {
        **classification,
        "status": "ok",
        "rawMeshFingerprint": raw_fingerprint,
        "depthDatasetFingerprint": dataset_fingerprint,
        "nativeRepair": native,
        "validationTopology": validation["topology"],
        "repairedCacheHit": False,
        "repairCacheFingerprint": cache_key,
        "reportPath": "outputs/mesh-repair-report.json",
        "backendVersion": backend_version,
    }
    report["repairSummary"] = _repair_summary(report, native, fallback=False)

    if settings.produce_watertight_copy and topology.get("boundaryLoops"):
        watertight_path = output_dir / "room-mesh-watertight.ply"
        watertight_policy_path = cache_root / f"watertight-policy-{cache_key}.json"
        watertight_report_path = cache_root / f"watertight-native-{cache_key}.json"
        watertight_policy = {
            **policy,
            "profile": "faithful",
            "repairSelfIntersections": True,
            "selectedLoops": [
                {
                    "loopId": loop["loopId"],
                    "classification": "fill_inferred",
                    "bestFitPlane": loop["bestFitPlane"],
                }
                for loop in topology["boundaryLoops"]
            ],
        }
        write_json(watertight_policy_path, watertight_policy)
        try:
            watertight_native = _run_backend(
                [
                    str(backend),
                    "repair",
                    "--input",
                    str(raw_path),
                    "--policy",
                    str(watertight_policy_path),
                    "--output",
                    str(watertight_path),
                    "--report",
                    str(watertight_report_path),
                ],
                watertight_report_path,
            )
            watertight_validation_path = (
                cache_root / f"watertight-validation-{cache_key}.json"
            )
            watertight_validation = _run_backend(
                [
                    str(backend),
                    "analyze",
                    "--input",
                    str(watertight_path),
                    "--report",
                    str(watertight_validation_path),
                    "--voxel-size-m",
                    str(mesh_voxel_size_m),
                ],
                watertight_validation_path,
            )
            watertight_topology = watertight_validation["topology"]
            if (
                int(watertight_topology["boundaryLoopCount"]) != 0
                or int(watertight_topology["nonManifoldVertexCount"]) != 0
                or int(watertight_topology["selfIntersectionCount"]) != 0
            ):
                raise RuntimeError(
                    "CGAL derivative did not pass closed-manifold validation"
                )
            report["watertightCopy"] = {
                "status": "ok",
                "path": "outputs/room-mesh-watertight.ply",
                "intentionalOpeningsMayBeSealed": True,
                "nativeRepair": watertight_native,
                "validationTopology": watertight_topology,
            }
        except Exception as error:
            watertight_path.unlink(missing_ok=True)
            report["watertightCopy"] = {"status": "failed", "error": str(error)}

    write_json(cached_report_path, report)
    write_json(final_report_path, report)
    repaired_vertices, repaired_triangles = _read_triangle_ply(repaired_path)
    return repaired_vertices, repaired_triangles, report
