from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from scanlan_validation import load_benchmark_manifest, select_backend_policy
from scanlan_splat.da3 import DA3_CODE_REVISION, da3_runtime_status
from scanlan_splat.lingbot import LINGBOT_CODE_REVISION, lingbot_runtime_status
from scanlan_splat.lingbot_depth import (
    LINGBOT_DEPTH_CODE_REVISION,
    lingbot_depth_runtime_status,
)
from scanlan_splat.mapanything import (
    MAPANYTHING_CODE_REVISION,
    mapanything_runtime_status,
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _runtime_entry(status: Mapping[str, Any], revision: str) -> dict[str, Any]:
    return {
        "available": bool(status.get("available", False)),
        "validated": bool(
            status.get("runtimeValidated", status.get("flashinferValidated", False))
        ),
        "revision": revision,
        "backend": status.get("backend", status.get("attentionBackend")),
        "error": status.get("error", status.get("modelError")),
    }


def _hardware(torch: Any, *, cuda_validated: bool) -> dict[str, Any]:
    cpu_threads = os.cpu_count() or 1
    if not torch.cuda.is_available():
        return {
            "cudaValidated": False,
            "gpuName": None,
            "cudaCapability": None,
            "vramFreeMiB": None,
            "vramTotalMiB": None,
            "cpuThreads": cpu_threads,
        }
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    return {
        "cudaValidated": bool(cuda_validated),
        "gpuName": torch.cuda.get_device_name(0),
        "cudaCapability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "vramFreeMiB": round(free_bytes / (1024 * 1024), 2),
        "vramTotalMiB": round(total_bytes / (1024 * 1024), 2),
        "cpuThreads": cpu_threads,
    }


def _cuda_smoke(torch: Any) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        probe = torch.tensor([1.0], device="cuda")
        if float((probe + 1.0).cpu()[0]) != 2.0:
            return False
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def evaluate_backend_policy(request: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    validate = bool(request.get("validateRuntimes", False))
    manifest_path = (
        Path(str(request["manifestPath"]))
        if str(request.get("manifestPath", "")).strip()
        else None
    )
    manifest = load_benchmark_manifest(manifest_path)

    # First inspect packages/assets without loading a multi-gigabyte model. A
    # cheap real CUDA launch establishes hardware capability. We then smoke only
    # runtimes belonging to a source/hardware-compatible record (or an explicit
    # override), avoiding a redundant three-model bake-off before ordinary media
    # inference when no benchmark could apply to the input anyway.
    mapanything = mapanything_runtime_status()
    da3 = da3_runtime_status()
    lingbot_depth = lingbot_depth_runtime_status()
    lingbot_map = lingbot_runtime_status(allow_download=False, validate_flashinfer=False)
    runtimes: dict[str, Any] = {
        "mapanything": _runtime_entry(mapanything, MAPANYTHING_CODE_REVISION),
        "da3": _runtime_entry(da3, DA3_CODE_REVISION),
        "lingbot-depth": _runtime_entry(lingbot_depth, LINGBOT_DEPTH_CODE_REVISION),
        "lingbot-map": _runtime_entry(lingbot_map, LINGBOT_CODE_REVISION),
    }
    supplied = request.get("runtimes", {})
    if isinstance(supplied, Mapping):
        runtimes.update({str(key): value for key, value in supplied.items()})
    hardware = _hardware(torch, cuda_validated=_cuda_smoke(torch))
    supplied_hardware = request.get("hardware", {})
    if isinstance(supplied_hardware, Mapping):
        hardware.update(supplied_hardware)

    preflight = select_backend_policy(
        source=request.get("source", {}),
        hardware=hardware,
        runtimes=runtimes,
        quality=request.get("quality", {}),
        overrides={},
        records=manifest["records"],
    )
    potential_ids = {
        str(assessment["benchmarkId"])
        for assessment in preflight.get("candidateAssessments", ())
        if assessment.get("reasons")
        and all("runtime " in str(reason) for reason in assessment["reasons"])
    }
    runtime_names = {
        str(runtime)
        for record in manifest["records"]
        if str(record.get("benchmarkId")) in potential_ids
        for runtime in record.get("requiresRuntimes", ())
    }
    overrides = request.get("overrides", {})
    if isinstance(overrides, Mapping):
        explicit_requirements = {
            "lingbot-map": ("lingbot-map",),
            "lingbot": ("lingbot-depth",),
            "mapanything": ("mapanything",),
            "mapanything-guided-colmap": ("mapanything",),
            "da3": ("da3",),
            "da3-guided-colmap": ("da3",),
        }
        for lane, override in overrides.items():
            runtime_names.update(explicit_requirements.get(str(override), ()))
            runtime_names.update(
                str(runtime)
                for record in manifest["records"]
                if str(record.get("lane")) == str(lane)
                and str(record.get("backend")) == str(override)
                for runtime in record.get("requiresRuntimes", ())
            )

    if validate and "mapanything" in runtime_names:
        mapanything = mapanything_runtime_status(verify_model=True, smoke_test=True)
        runtimes["mapanything"] = _runtime_entry(mapanything, MAPANYTHING_CODE_REVISION)
    if validate and "da3" in runtime_names:
        da3 = da3_runtime_status(verify_model=True, smoke_test=True)
        runtimes["da3"] = _runtime_entry(da3, DA3_CODE_REVISION)
    if validate and "lingbot-depth" in runtime_names:
        lingbot_depth = lingbot_depth_runtime_status(verify_model=True, smoke_test=True)
        runtimes["lingbot-depth"] = _runtime_entry(
            lingbot_depth, LINGBOT_DEPTH_CODE_REVISION
        )
    if validate and "lingbot-map" in runtime_names:
        lingbot_map = lingbot_runtime_status(
            allow_download=False, validate_flashinfer=True
        )
        runtimes["lingbot-map"] = _runtime_entry(lingbot_map, LINGBOT_CODE_REVISION)
    hardware = _hardware(torch, cuda_validated=bool(hardware["cudaValidated"]))
    if isinstance(supplied_hardware, Mapping):
        hardware.update(supplied_hardware)

    return select_backend_policy(
        source=request.get("source", {}),
        hardware=hardware,
        runtimes=runtimes,
        quality=request.get("quality", {}),
        overrides=overrides if isinstance(overrides, Mapping) else {},
        manifest_path=manifest_path,
    )


def evaluate_backend_policy_file(request_path: Path, report_path: Path | None) -> dict[str, Any]:
    request = json.loads(request_path.resolve(strict=True).read_text(encoding="utf-8"))
    result = evaluate_backend_policy(request)
    if report_path is not None:
        _atomic_json(report_path.resolve(), result)
    return result
