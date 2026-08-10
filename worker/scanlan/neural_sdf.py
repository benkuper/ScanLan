from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .io import write_json


NEURAL_SDF_CONTRACT_VERSION = "scanlan-neural-sdf-v1"


def _worker_command(worker: Path) -> list[str]:
    return [sys.executable, str(worker)] if worker.suffix.lower() == ".py" else [str(worker)]


def _surface_fingerprint(
    vertices: np.ndarray,
    triangles: np.ndarray,
    voxel_size_m: float,
) -> str:
    digest = hashlib.sha256(NEURAL_SDF_CONTRACT_VERSION.encode())
    digest.update(np.asarray(vertices, dtype=np.float32).tobytes())
    digest.update(np.asarray(triangles, dtype=np.int64).tobytes())
    digest.update(np.asarray(voxel_size_m, dtype=np.float64).tobytes())
    return digest.hexdigest()[:24]


def _independent_candidate_validation(
    source_vertices: np.ndarray,
    triangles: np.ndarray,
    candidate_vertices: np.ndarray,
    candidate_triangles: np.ndarray,
    voxel_size_m: float,
) -> dict[str, Any]:
    if candidate_vertices.shape != source_vertices.shape:
        raise RuntimeError("Neural SDF worker changed the surface vertex contract")
    if not np.array_equal(candidate_triangles, triangles):
        raise RuntimeError("Neural SDF worker changed topology outside the repair stage")
    if not np.isfinite(candidate_vertices).all():
        raise RuntimeError("Neural SDF worker returned non-finite geometry")
    displacement = np.linalg.norm(candidate_vertices - source_vertices, axis=1)
    median = float(np.median(displacement))
    p95 = float(np.percentile(displacement, 95.0))
    maximum = float(np.max(displacement, initial=0.0))
    accepted = (
        median <= voxel_size_m * 0.55
        and p95 <= voxel_size_m * 1.25
        and maximum <= voxel_size_m * 2.05
    )
    return {
        "accepted": accepted,
        "medianDisplacementM": median,
        "p95DisplacementM": p95,
        "maximumDisplacementM": maximum,
    }


def refine_surface_with_worker(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    project_root: Path,
    worker: Path,
    voxel_size_m: float,
    validation_report: dict[str, Any] | None,
    progress: Callable[..., None] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source_vertices = np.asarray(vertices, dtype=np.float32)
    source_triangles = np.asarray(triangles, dtype=np.int64)
    public_report_path = project_root / "outputs" / "neural-sdf-report.json"
    gate = validation_report or {}
    if not bool(gate.get("accepted", False)):
        report = {
            "schemaVersion": 1,
            "method": NEURAL_SDF_CONTRACT_VERSION,
            "status": "skipped",
            "reason": "Camera/depth validation did not authorize neural surface refinement",
            "inputValidation": gate,
        }
        write_json(public_report_path, report)
        return source_vertices, source_triangles, report

    fingerprint = _surface_fingerprint(source_vertices, source_triangles, voxel_size_m)
    cache_root = project_root / "outputs" / "cache" / "neural-sdf" / fingerprint
    input_path = cache_root / "input.npz"
    candidate_path = cache_root / "candidate.npz"
    report_path = cache_root / "report.json"
    progress_path = project_root / "outputs" / "progress.json"
    cache_root.mkdir(parents=True, exist_ok=True)
    if report_path.is_file() and candidate_path.is_file():
        cached_report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            cached_report.get("status") == "accepted"
            and cached_report.get("sourceFingerprint") == fingerprint
        ):
            with np.load(candidate_path, allow_pickle=False) as values:
                candidate_vertices = np.asarray(values["vertices"], dtype=np.float32)
                candidate_triangles = np.asarray(values["triangles"], dtype=np.int64)
            independent = _independent_candidate_validation(
                source_vertices,
                source_triangles,
                candidate_vertices,
                candidate_triangles,
                voxel_size_m,
            )
            if independent["accepted"]:
                cached_report["cacheHit"] = True
                cached_report["independentValidation"] = independent
                cached_report["inputValidation"] = gate
                write_json(public_report_path, cached_report)
                if progress:
                    progress(
                        "Neural SDF",
                        "Reused the accepted Max Quality neural surface",
                        0,
                        None,
                        0.68,
                    )
                return candidate_vertices, candidate_triangles, cached_report

    temporary_input = input_path.with_suffix(".npz.tmp")
    with temporary_input.open("wb") as handle:
        np.savez(
            handle,
            vertices=source_vertices,
            triangles=source_triangles,
            voxel_size_m=np.asarray(voxel_size_m, dtype=np.float64),
            fingerprint=np.asarray(fingerprint),
        )
    temporary_input.replace(input_path)
    if progress:
        progress(
            "Neural SDF",
            "Fitting an optional Max Quality surface from validated geometry",
            0,
            None,
            0.60,
        )
    command = [
        *_worker_command(worker),
        "refine-sdf",
        "--input",
        str(input_path),
        "--output",
        str(candidate_path),
        "--report",
        str(report_path),
        "--progress",
        str(progress_path),
        "--device",
        "cuda",
    ]
    completed = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    report: dict[str, Any] = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.is_file()
        else {
            "schemaVersion": 1,
            "method": NEURAL_SDF_CONTRACT_VERSION,
            "status": "rejected",
        }
    )
    report["cacheHit"] = False
    report["inputValidation"] = gate
    if completed.returncode != 0 or report.get("status") != "accepted" or not candidate_path.is_file():
        detail = completed.stderr.strip().splitlines()
        report["status"] = "rejected"
        report["reason"] = report.get("reason") or (
            detail[-1] if detail else "Neural SDF candidate did not pass its quality gates"
        )
        write_json(public_report_path, report)
        if progress:
            progress(
                "Neural SDF",
                f"Kept the validated baseline surface - {report['reason']}",
                0,
                None,
                0.68,
            )
        return source_vertices, source_triangles, report

    with np.load(candidate_path, allow_pickle=False) as values:
        candidate_vertices = np.asarray(values["vertices"], dtype=np.float32)
        candidate_triangles = np.asarray(values["triangles"], dtype=np.int64)
    independent = _independent_candidate_validation(
        source_vertices,
        source_triangles,
        candidate_vertices,
        candidate_triangles,
        voxel_size_m,
    )
    report["independentValidation"] = independent
    if not independent["accepted"]:
        report["status"] = "rejected"
        report["reason"] = "The reconstruction worker rejected excessive neural surface displacement"
        write_json(public_report_path, report)
        return source_vertices, source_triangles, report
    write_json(public_report_path, report)
    return candidate_vertices, candidate_triangles, report
