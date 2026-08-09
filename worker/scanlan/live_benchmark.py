from __future__ import annotations

import ctypes
import json
import math
import os
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .realtime import ENGINE_MESH, ENGINE_POINTS, ENGINE_STATUS, read_engine_message
from .replay import archive_frames
from .stream import encode_rgbd_frame


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "samples": len(values),
        "medianMs": _rounded(_percentile(values, 0.50)),
        "p95Ms": _rounded(_percentile(values, 0.95)),
        "maximumMs": _rounded(max(values) if values else None),
    }


def _rounded(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(float(value), digits)


def _worker_command() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable]
    # Windows virtual environments use a tiny redirector executable that
    # launches the base interpreter. Starting the base interpreter directly
    # keeps benchmark RSS/GPU sampling attached to the process doing the work.
    interpreter = str(getattr(sys, "_base_executable", sys.executable))
    return [interpreter, "-m", "scanlan.cli"]


def _process_working_set_bytes(process_id: int) -> int | None:
    if sys.platform == "win32":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        query_information = 0x0400
        handle = kernel32.OpenProcess(query_information, False, process_id)
        if not handle:
            return None
        try:
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            ):
                return None
            return int(counters.WorkingSetSize)
        finally:
            kernel32.CloseHandle(handle)

    status = Path(f"/proc/{process_id}/status")
    try:
        for line in status.read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _process_gpu_memory_bytes(process_id: int) -> int | None:
    try:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
            creationflags=flags,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    total_mib = 0
    found = False
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 2:
            continue
        try:
            if int(values[0]) == process_id:
                total_mib += int(values[1])
                found = True
        except ValueError:
            continue
    return total_mib * 1024 * 1024 if found else None


def _monitor_memory(
    process: subprocess.Popen[bytes],
    stop: threading.Event,
    samples: dict[str, list[int]],
) -> None:
    next_gpu_sample = 0.0
    while not stop.wait(0.05):
        if process.poll() is not None:
            break
        working_set = _process_working_set_bytes(process.pid)
        if working_set is not None:
            samples["workingSet"].append(working_set)
        now = time.monotonic()
        if now >= next_gpu_sample:
            gpu = _process_gpu_memory_bytes(process.pid)
            if gpu is not None:
                samples["gpu"].append(gpu)
            next_gpu_sample = now + 0.25


def summarize_live_benchmark(
    *,
    capture: Path,
    mode: str,
    voxel_size_m: float,
    device: str,
    paced: bool,
    frame_count: int,
    source_duration_seconds: float,
    wall_seconds: float,
    statuses: list[dict[str, Any]],
    pose_latencies_ms: list[float],
    point_latencies_ms: list[float],
    mesh_latencies_ms: list[float],
    point_snapshots: int,
    mesh_snapshots: int,
    final_point_count: int,
    final_triangle_count: int,
    working_set_samples: list[int],
    gpu_samples: list[int],
    journal_entries: list[dict[str, Any]],
    exit_code: int,
) -> dict[str, Any]:
    final_status = statuses[-1] if statuses else {}
    state_counts = Counter(str(status.get("state", "unknown")) for status in statuses)
    rejected_entries = [entry for entry in journal_entries if not entry.get("accepted", False)]
    invalid_integration = [entry for entry in rejected_entries if entry.get("integrated", False)]
    relocalized = sum(
        "relocal" in str(entry.get("state", "")).lower()
        or "relocal" in str(entry.get("reason", "")).lower()
        for entry in journal_entries
    )
    return {
        "schemaVersion": 1,
        "kind": "scanlan-live-baseline",
        "capturedAt": datetime.now(timezone.utc).isoformat(),
        "capture": str(capture.resolve()),
        "configuration": {
            "mode": mode,
            "voxelSizeM": voxel_size_m,
            "device": device,
            "realtimePacing": paced,
        },
        "source": {
            "frameCount": frame_count,
            "durationSeconds": _rounded(source_duration_seconds, 3),
        },
        "runtime": {
            "exitCode": exit_code,
            "wallSeconds": _rounded(wall_seconds, 3),
            "effectiveFps": _rounded(frame_count / max(wall_seconds, 1e-9)),
            "backend": final_status.get("backend"),
            "peakWorkingSetMiB": _rounded(
                max(working_set_samples) / (1024 * 1024)
                if working_set_samples
                else None
            ),
            "peakGpuMemoryMiB": _rounded(
                max(gpu_samples) / (1024 * 1024) if gpu_samples else None
            ),
        },
        "latency": {
            "pose": _latency_summary(pose_latencies_ms),
            "pointMap": _latency_summary(point_latencies_ms),
            "mesh": _latency_summary(mesh_latencies_ms),
        },
        "tracking": {
            "processedFrames": int(final_status.get("processedFrames", 0)),
            "acceptedFrames": int(final_status.get("acceptedFrames", 0)),
            "rejectedFrames": int(final_status.get("rejectedFrames", 0)),
            "integratedFrames": int(final_status.get("integratedFrames", 0)),
            "states": dict(sorted(state_counts.items())),
            "relocalizationEvents": relocalized,
            "integrationFrozenForEveryRejectedFrame": not invalid_integration,
        },
        "queues": {
            "sourceDrops": int(final_status.get("sourceDrops", 0)),
            "trackingDrops": int(final_status.get("trackingQueueDrops", 0)),
            "mappingDrops": int(final_status.get("mappingDrops", 0)),
            "journalDrops": int(final_status.get("journalDrops", 0)),
        },
        "preview": {
            "pointSnapshots": point_snapshots,
            "meshSnapshots": mesh_snapshots,
            "finalPointCount": final_point_count,
            "finalTriangleCount": final_triangle_count,
            "provisionalAvailableAfterStop": point_snapshots > 0 and final_point_count > 0,
            "trackingJournalAvailableAfterStop": bool(journal_entries),
        },
    }


def benchmark_live_capture(
    capture: Path,
    *,
    mode: str = "mesh",
    voxel_size_m: float = 0.01,
    device: str = "auto",
    paced: bool = True,
    session_root: Path | None = None,
) -> dict[str, Any]:
    capture = capture.resolve(strict=True)
    frames = archive_frames(capture)
    owned_session: tempfile.TemporaryDirectory[str] | None = None
    if session_root is None:
        owned_session = tempfile.TemporaryDirectory(prefix="scanlan-live-benchmark-")
        session_root = Path(owned_session.name)
    else:
        session_root.mkdir(parents=True, exist_ok=True)
    session_root = session_root.resolve()

    command = _worker_command() + [
        "realtime",
        "--mode",
        mode,
        "--voxel-size",
        str(voxel_size_m),
        "--device",
        device,
        "--session",
        str(session_root),
    ]
    environment = os.environ.copy()
    package_root = str(Path(__file__).resolve().parents[1])
    import_paths = [package_root]
    import_paths.extend(
        value
        for value in sys.path
        if value and "site-packages" in value.lower() and Path(value).is_dir()
    )
    if environment.get("PYTHONPATH"):
        import_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(import_paths)
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=package_root,
        env=environment,
        creationflags=flags,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    send_times: dict[int, float] = {}
    sent_lock = threading.Lock()
    statuses: list[dict[str, Any]] = []
    pose_latencies_ms: list[float] = []
    point_latencies_ms: list[float] = []
    mesh_latencies_ms: list[float] = []
    point_snapshots = 0
    mesh_snapshots = 0
    final_point_count = 0
    final_triangle_count = 0
    read_error: list[BaseException] = []

    def latency_for(sequence: int, arrived: float) -> float | None:
        with sent_lock:
            sent = send_times.get(sequence)
        return None if sent is None else max(0.0, (arrived - sent) * 1000.0)

    def read_messages() -> None:
        nonlocal point_snapshots, mesh_snapshots, final_point_count, final_triangle_count
        try:
            while True:
                try:
                    kind, sequence, payload = read_engine_message(process.stdout)
                except EOFError:
                    return
                arrived = time.monotonic()
                latency = latency_for(sequence, arrived)
                if kind == ENGINE_STATUS:
                    status = json.loads(payload)
                    statuses.append(status)
                    if latency is not None and status.get("state") not in {"ready", "complete"}:
                        pose_latencies_ms.append(latency)
                elif kind == ENGINE_POINTS:
                    point_snapshots += 1
                    if len(payload) >= 24:
                        _, _, _, _, final_point_count = struct.unpack("<4sIQfI", payload[:24])
                    if latency is not None:
                        point_latencies_ms.append(latency)
                elif kind == ENGINE_MESH:
                    mesh_snapshots += 1
                    if len(payload) >= 16:
                        _, _, _, index_count = struct.unpack("<4sIII", payload[:16])
                        final_triangle_count = index_count // 3
                    if latency is not None:
                        mesh_latencies_ms.append(latency)
        except BaseException as error:
            read_error.append(error)

    reader = threading.Thread(target=read_messages, name="benchmark-engine-reader", daemon=True)
    reader.start()
    memory_samples: dict[str, list[int]] = {"workingSet": [], "gpu": []}
    monitor_stop = threading.Event()
    monitor = threading.Thread(
        target=_monitor_memory,
        args=(process, monitor_stop, memory_samples),
        name="benchmark-memory",
        daemon=True,
    )
    monitor.start()

    started = time.monotonic()
    first_timestamp_us: int | None = None
    last_timestamp_us: int | None = None
    frame_count = 0
    write_error: BaseException | None = None
    try:
        for frame in frames:
            if first_timestamp_us is None:
                first_timestamp_us = frame.depth_timestamp_us
            if paced:
                target = started + max(
                    0.0, (frame.depth_timestamp_us - first_timestamp_us) / 1_000_000.0
                )
                delay = target - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            with sent_lock:
                send_times[frame.sequence] = time.monotonic()
            process.stdin.write(encode_rgbd_frame(frame))
            process.stdin.flush()
            last_timestamp_us = frame.depth_timestamp_us
            frame_count += 1
    except BaseException as error:
        write_error = error
    finally:
        process.stdin.close()

    try:
        exit_code = process.wait(timeout=max(60.0, frame_count * 2.0))
    except subprocess.TimeoutExpired:
        process.kill()
        exit_code = process.wait()
        write_error = write_error or RuntimeError("Realtime benchmark timed out")
    reader.join(timeout=5.0)
    monitor_stop.set()
    monitor.join(timeout=3.0)
    wall_seconds = time.monotonic() - started
    stderr = process.stderr.read().decode("utf-8", errors="replace").strip()

    journal_path = session_root / "tracking.jsonl"
    journal_entries = []
    if journal_path.is_file():
        journal_entries = [
            json.loads(line)
            for line in journal_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    source_duration_seconds = (
        max(0.0, (last_timestamp_us - first_timestamp_us) / 1_000_000.0)
        if first_timestamp_us is not None and last_timestamp_us is not None
        else 0.0
    )
    report = summarize_live_benchmark(
        capture=capture,
        mode=mode,
        voxel_size_m=voxel_size_m,
        device=device,
        paced=paced,
        frame_count=frame_count,
        source_duration_seconds=source_duration_seconds,
        wall_seconds=wall_seconds,
        statuses=statuses,
        pose_latencies_ms=pose_latencies_ms,
        point_latencies_ms=point_latencies_ms,
        mesh_latencies_ms=mesh_latencies_ms,
        point_snapshots=point_snapshots,
        mesh_snapshots=mesh_snapshots,
        final_point_count=final_point_count,
        final_triangle_count=final_triangle_count,
        working_set_samples=memory_samples["workingSet"],
        gpu_samples=memory_samples["gpu"],
        journal_entries=journal_entries,
        exit_code=exit_code,
    )
    if owned_session is not None:
        owned_session.cleanup()
    if read_error:
        raise RuntimeError(f"Could not read realtime benchmark output: {read_error[0]}")
    if write_error is not None:
        raise RuntimeError(f"Could not replay benchmark capture: {write_error}")
    if exit_code != 0:
        raise RuntimeError(stderr or f"Realtime benchmark exited with code {exit_code}")
    return report
