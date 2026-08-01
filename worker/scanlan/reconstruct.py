from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np

from .io import phase_roots, read_phase, read_project, save_binary_ply, save_preview, write_json
from .mesh import PosedFrame, build_mesh_artifacts
from .numpy_engine import reconstruct_known_poses

Engine = Literal["auto", "numpy", "open3d"]
Device = Literal["auto", "cpu", "cuda"]


class ProgressReporter:
    def __init__(self, project_root: Path, total_units: int) -> None:
        self.path = project_root / "outputs" / "progress.json"
        self.total_units = max(total_units, 1)
        self.processed_units = 0
        self.point_count: int | None = None
        self.started = perf_counter()
        self.compute_backend: str | None = None
        self.current_stage: str | None = None
        self.stage_started = self.started
        self.completed_stage_seconds: dict[str, float] = {}
        self.update("Preparing", "Loading captured RGB-D frames", 0)

    def stage_timings(self, include_current: bool = True) -> dict[str, float]:
        timings = dict(self.completed_stage_seconds)
        if include_current and self.current_stage is not None:
            timings[self.current_stage] = timings.get(self.current_stage, 0.0) + (
                perf_counter() - self.stage_started
            )
        return {key: round(value, 2) for key, value in timings.items()}

    def update(
        self,
        stage: str,
        detail: str,
        advance: int = 1,
        point_count: int | None = None,
        stage_progress: float | None = None,
        stage_eta_seconds: int | None = None,
        elapsed_seconds: int | None = None,
        compute_backend: str | None = None,
    ) -> None:
        now = perf_counter()
        if stage != self.current_stage:
            if self.current_stage is not None:
                self.completed_stage_seconds[self.current_stage] = (
                    self.completed_stage_seconds.get(self.current_stage, 0.0)
                    + now
                    - self.stage_started
                )
            self.current_stage = stage
            self.stage_started = now
        if compute_backend is not None:
            self.compute_backend = compute_backend
        self.processed_units = min(self.total_units, self.processed_units + advance)
        if point_count is not None:
            self.point_count = point_count
        progress = self.processed_units / self.total_units
        elapsed = perf_counter() - self.started
        eta = None
        if self.processed_units > 1 and progress < 1:
            eta = max(0, round(elapsed * (1.0 - progress) / progress))
        payload = {
            "stage": stage,
            "detail": detail,
            "progress": round(progress, 4),
            "processedUnits": self.processed_units,
            "totalUnits": self.total_units,
            "etaSeconds": eta,
            "pointCount": self.point_count,
            "stageProgress": None if stage_progress is None else round(stage_progress, 4),
            "stageEtaSeconds": stage_eta_seconds,
            "elapsedSeconds": elapsed_seconds,
            "computeBackend": self.compute_backend,
            "stageTimingsSeconds": self.stage_timings(),
        }
        write_json(self.path, {key: value for key, value in payload.items() if value is not None})

    def fail(self, detail: str) -> None:
        self.update("Failed", detail, 0)


def reconstruct_project(
    project_root: Path,
    engine: Engine = "auto",
    device: Device = "auto",
) -> dict:
    project_root = project_root.resolve()
    project = read_project(project_root)
    phases = [read_phase(path) for path in phase_roots(project_root, project)]
    if not phases:
        raise ValueError("Capture at least one phase before building a point cloud")

    voxel_size_m = max(float(project["settings"].get("voxelSizeMm", 15)) / 1000.0, 0.001)
    total_frames = sum(len(phase.frames) for phase in phases)
    known_global_poses = all(
        phase.manifest.get("poseSource") == "known_global"
        and all(frame.pose is not None for frame in phase.frames)
        for phase in phases
    )
    selected_engine = engine
    if selected_engine == "auto":
        selected_engine = "numpy" if known_global_poses else "open3d"
    total_units = (
        total_frames + 12
        if selected_engine == "numpy"
        else total_frames * 3 + len(phases) * 2 + max(len(phases) - 1, 0) * 30 + 12
    )
    reporter = ProgressReporter(project_root, total_units)
    build_preview_path = project_root / "outputs" / "build-preview.json"
    build_preview_path.unlink(missing_ok=True)

    try:
        artifact_context: dict = {}
        if selected_engine == "numpy":
            reporter.update(
                "Preparing",
                "Compute backend: NumPy CPU",
                0,
                compute_backend="NumPy CPU",
            )
            points, colors = reconstruct_known_poses(phases, voxel_size_m, reporter.update)
            flip_x = all(
                phase.manifest.get("sensor", {}).get("kind", "kinect_v2") == "kinect_v2"
                for phase in phases
            )
            points = points * ([-1.0, 1.0, -1.0] if flip_x else [1.0, 1.0, -1.0])
            display_axes = (-1.0, 1.0, -1.0) if flip_x else (1.0, 1.0, -1.0)
            artifact_context["posed_frames"] = [
                PosedFrame(
                    phase_name=str(phase.manifest.get("name", f"Phase {phase_index + 1}")),
                    phase_id=str(phase.manifest.get("id", phase.root.name)),
                    source=phase,
                    frame_index=frame_index,
                    camera_to_global=np.asarray(frame.pose, dtype=np.float64),
                    display_axes=display_axes,
                    image_y_up=True,
                )
                for phase_index, phase in enumerate(phases)
                for frame_index, frame in enumerate(phase.frames)
                if frame.pose is not None
            ]
            quality = {
                "score": 96,
                "label": "High",
                "detail": f"High confidence from known global poses; used all {total_frames} frames.",
                "framesUsed": total_frames,
                "framesCaptured": total_frames,
                "tracking": [],
                "phaseMatches": [],
            }
        elif selected_engine == "open3d":
            from .open3d_engine import reconstruct_open3d

            points, colors, quality = reconstruct_open3d(
                phases,
                voxel_size_m,
                reporter.update,
                build_preview_path,
                requested_device=device,
                cache_root=project_root / "outputs" / "cache",
                artifact_context=artifact_context,
            )
        else:
            raise ValueError(f"Unknown reconstruction engine: {selected_engine}")

        if points.shape[0] == 0:
            raise RuntimeError("Reconstruction produced no valid points")

        output_dir = project_root / "outputs"
        mesh = build_mesh_artifacts(
            output_dir,
            artifact_context.get("posed_frames", []),
            reporter.update,
        )
        reporter.update("Exporting", "Writing point-cloud and textured-mesh artifacts", 4, len(points))
        output_path = output_dir / "room-cloud.ply"
        save_binary_ply(output_path, points, colors)
        save_preview(output_dir / "preview.json", points, colors)
        result = {
            "engine": selected_engine,
            "pointCount": int(points.shape[0]),
            "phaseCount": len(phases),
            "voxelSizeM": voxel_size_m,
            "outputPath": str(output_path),
            "confidenceScore": quality["score"],
            "confidenceLabel": quality["label"],
            "confidenceDetail": quality["detail"],
            "framesUsed": quality["framesUsed"],
            "framesCaptured": quality["framesCaptured"],
            "quality": quality,
            "computeBackend": quality.get("computeBackend", reporter.compute_backend or "NumPy CPU"),
            **mesh,
        }

        project["processingStatus"] = "complete"
        project.pop("processingError", None)
        project["pointCount"] = result["pointCount"]
        project["outputPath"] = "outputs/room-cloud.ply"
        project["meshTriangleCount"] = result["meshTriangleCount"]
        project["meshOutputPath"] = result.get("meshOutputPath")
        project["cameraFrameCount"] = result["cameraFrameCount"]
        project["confidenceScore"] = result["confidenceScore"]
        project["confidenceLabel"] = result["confidenceLabel"]
        project["confidenceDetail"] = result["confidenceDetail"]
        project["framesUsed"] = result["framesUsed"]
        reporter.update(
            "Complete",
            f"3D model ready · {result['meshTriangleCount']:,} textured triangles · "
            f"{quality['score']}/100 {quality['label'].lower()} confidence",
            max(0, reporter.total_units - reporter.processed_units),
            len(points),
            1.0,
            0,
            round(perf_counter() - reporter.started),
        )
        result["processingSeconds"] = round(perf_counter() - reporter.started, 2)
        result["stageTimingsSeconds"] = reporter.stage_timings(include_current=False)
        project["processingBackend"] = result["computeBackend"]
        project["processingDurationSeconds"] = result["processingSeconds"]
        write_json(output_dir / "result.json", result)
        write_json(project_root / "project.json", project)
        return result
    except Exception as error:
        reporter.fail(str(error))
        raise
