from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import PhaseData, write_json


def _nvidia_hardware(cuda_validated: bool) -> dict[str, Any]:
    hardware: dict[str, Any] = {
        "cudaValidated": bool(cuda_validated),
        "gpuName": None,
        "cudaCapability": None,
        "vramTotalMiB": None,
        "vramFreeMiB": None,
        "cpuThreads": os.cpu_count() or 1,
    }
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,compute_cap",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=3.0,
            creationflags=flags,
        )
        values = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
        if result.returncode == 0 and len(values) == 4:
            hardware.update(
                gpuName=values[0],
                vramTotalMiB=float(values[1]),
                vramFreeMiB=float(values[2]),
                cudaCapability=values[3],
            )
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        pass
    return hardware


def select_live_backend(
    session_root: Path | None,
    *,
    sensor_kind: str,
    requested_device: str,
    cuda_active: bool,
    open3d_revision: str,
    expected_frame_count: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Audit/select the live implementation after its real Open3D kernel probe."""

    from scanlan_validation import select_backend_policy

    runtimes = {
        "open3d-cuda": {
            "available": cuda_active,
            "validated": cuda_active,
            "revision": open3d_revision,
        },
        "open3d-cpu": {
            "available": True,
            "validated": True,
            "revision": open3d_revision,
        },
    }
    overrides: dict[str, str] = {}
    if requested_device == "cuda":
        overrides["liveRgbd"] = "open3d-cuda-tsdf"
    elif requested_device == "cpu":
        overrides["liveRgbd"] = "open3d-cpu-tsdf"
    manifest = os.environ.get("SCANLAN_BACKEND_BENCHMARKS", "").strip()
    source = {
        "kind": "rgbd",
        "sensorKind": sensor_kind or "unknown",
        "maximumImageDimension": 4096,
        "characteristics": ["physical-capture"],
    }
    if expected_frame_count is not None and expected_frame_count > 0:
        source["frameCount"] = int(expected_frame_count)
    try:
        report = select_backend_policy(
            source=source,
            hardware=_nvidia_hardware(cuda_active),
            runtimes=runtimes,
            quality={"preference": "interactive-quality", "commercialUse": False},
            overrides=overrides,
            manifest_path=Path(manifest) if manifest else None,
        )
    except Exception as error:
        current = "open3d-cuda-tsdf" if cuda_active else "open3d-cpu-tsdf"
        report = {
            "schemaVersion": 1,
            "kind": "scanlan-adaptive-backend-policy",
            "source": source,
            "decisions": {
                "liveRgbd": {
                    "selected": current,
                    "selectionMode": "protected-baseline",
                    "benchmarked": False,
                    "reason": f"Live policy evaluation failed safely: {error}",
                }
            },
        }
    selected = str(report["decisions"]["liveRgbd"]["selected"])
    selected_device = (
        "cuda"
        if selected == "open3d-cuda-tsdf"
        else "cpu"
        if selected in {"open3d-cpu-tsdf", "numpy-cpu-surfel"}
        else requested_device
    )
    if session_root is not None:
        write_json(session_root / "backend-policy.json", report)
    return selected_device, report


def _source_profile(project: Mapping[str, Any], phases: Sequence[PhaseData]) -> dict[str, Any]:
    settings = project.get("settings", {})
    return {
        "kind": "hybrid" if project.get("mediaSources") else "rgbd",
        "sensorKind": str(settings.get("sensorKind", "unknown")),
        "frameCount": sum(len(phase.frames) for phase in phases),
        "maximumImageDimension": max(
            (max(phase.camera.width, phase.camera.height) for phase in phases),
            default=0,
        ),
        "characteristics": ["physical-capture"],
    }


def select_depth_backend(
    project_root: Path,
    project: Mapping[str, Any],
    phases: Sequence[PhaseData],
    geometry_worker: Path,
) -> tuple[str, dict[str, Any]]:
    """Resolve opt-in automatic completion through the isolated policy probe."""

    geometry_worker = geometry_worker.resolve(strict=True)
    report_path = project_root / "outputs" / "backend-policy.json"
    request: dict[str, Any] = {
        "source": _source_profile(project, phases),
        "quality": {"preference": "max-quality", "commercialUse": False},
        "overrides": {"depthCompletion": "auto"},
        "lanes": ["depthCompletion"],
        "validateRuntimes": True,
    }
    manifest = os.environ.get("SCANLAN_BACKEND_BENCHMARKS", "").strip()
    if manifest:
        request["manifestPath"] = str(Path(manifest).resolve(strict=True))
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        with tempfile.TemporaryDirectory(prefix="scanlan-backend-policy-") as directory:
            request_path = Path(directory) / "request.json"
            request_path.write_text(
                json.dumps(request, separators=(",", ":"), allow_nan=False),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    str(geometry_worker),
                    "backend-policy",
                    "--request",
                    str(request_path),
                    "--report",
                    str(report_path),
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
                creationflags=flags,
            )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip().splitlines()
            raise RuntimeError(detail[-1] if detail else "policy worker returned no diagnostic")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        decision = report.get("decisions", {}).get("depthCompletion", {})
        backend = str(decision.get("selected", "off"))
        if backend not in {"off", "lingbot", "mapanything", "da3"}:
            raise RuntimeError(f"policy selected unsupported depth backend {backend}")
        return backend, report
    except Exception as error:
        report = {
            "schemaVersion": 1,
            "kind": "scanlan-adaptive-backend-policy",
            "source": request["source"],
            "decisions": {
                "depthCompletion": {
                    "selected": "off",
                    "selectionMode": "protected-baseline",
                    "benchmarked": False,
                    "reason": f"Automatic policy probe failed safely: {error}",
                }
            },
        }
        write_json(report_path, report)
        return "off", report
