from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np

from .dataset import build_posed_dataset, dataset_fingerprint
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
        stage_elapsed = max(0, round(now - self.stage_started))
        if elapsed_seconds is None:
            elapsed_seconds = stage_elapsed
        if (
            stage_eta_seconds is None
            and stage_progress is not None
            and 0.01 < stage_progress < 1.0
            and stage_elapsed >= 1
        ):
            stage_eta_seconds = max(
                1,
                round(stage_elapsed * (1.0 - stage_progress) / stage_progress),
            )
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
    targets: tuple[str, ...] = ("point_cloud", "textured_mesh"),
) -> dict:
    project_root = project_root.resolve()
    project = read_project(project_root)
    # Live tracking rejection marks a pose estimate as unsafe; it does not make
    # the archived RGB-D pixels unusable.  Keep those consecutive frames for
    # production odometry so a lost live tracker cannot turn a smooth recording
    # into large temporal gaps that offline tracking cannot bridge.
    phases = [
        read_phase(path, include_tracking_rejected=True)
        for path in phase_roots(project_root, project)
    ]
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
        artifact_context: dict = {
            "needs_mesh": "textured_mesh" in targets,
            "mesh_voxel_size_m": max(voxel_size_m, 0.008),
        }
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
        # Keep the last usable geometry visible while later mesh/dataset/splat
        # stages continue, including the fast known-pose (NumPy) pipeline.
        save_preview(build_preview_path, points, colors, limit=30_000)

        output_dir = project_root / "outputs"
        posed_frames = artifact_context.get("posed_frames", [])
        # The canonical RGB/depth dataset and Gaussian initialization are only
        # consumed by the splat trainer.  Building them for a mesh-only job used
        # to reproject and compress every captured frame before meshing could
        # even begin, despite the mesh path reading its selected source frames
        # directly.
        needs_dataset = "gaussian_splat" in targets
        dataset = (
            build_posed_dataset(output_dir / "cache", posed_frames, reporter.update)
            if needs_dataset
            else None
        )
        source_fingerprint = (
            str(dataset["fingerprint"])
            if dataset is not None
            else dataset_fingerprint(posed_frames)
        )
        mesh = (
            build_mesh_artifacts(
                output_dir,
                posed_frames,
                reporter.update,
                voxel_size_m=voxel_size_m,
                prebuilt_mesh=artifact_context.get("fused_mesh"),
                prebuilt_mesh_method=artifact_context.get("fused_mesh_method"),
            )
            if "textured_mesh" in targets
            else {
                "cameraFrameCount": len(posed_frames),
                "meshVertexCount": 0,
                "meshTriangleCount": 0,
            }
        )
        reporter.update(
            "Exporting",
            "Writing selected reconstruction artifacts",
            4,
            len(points),
            0.0,
        )
        output_path = output_dir / "room-cloud.ply"
        if "point_cloud" in targets:
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
            "targets": list(targets),
            "datasetFingerprint": source_fingerprint,
        }

        # Gaussian training is a separate worker launched by the desktop job
        # orchestrator. Keep the overall project active while that selected
        # target is still pending so live splat previews remain available.
        project["processingStatus"] = (
            "processing" if "gaussian_splat" in targets else "complete"
        )
        project.pop("processingError", None)
        project["schemaVersion"] = 3
        artifacts = project.setdefault("artifacts", {})
        updated_at = datetime.now(timezone.utc).isoformat()
        fingerprint = source_fingerprint
        if "point_cloud" in targets:
            project["pointCount"] = result["pointCount"]
            project["outputPath"] = "outputs/room-cloud.ply"
            artifacts["pointCloud"] = {
                "path": "outputs/room-cloud.ply",
                "status": "ready",
                "sourceFingerprint": fingerprint,
                "updatedAt": updated_at,
                "metric": True,
                "stale": False,
            }
        if "textured_mesh" in targets:
            mesh_fingerprint = hashlib.sha256(
                (
                    fingerprint
                    + ":"
                    + str(result.get("supplementalTextureFingerprint", "none"))
                ).encode("utf-8")
            ).hexdigest()[:24]
            project["meshTriangleCount"] = result["meshTriangleCount"]
            project["meshOutputPath"] = result.get("meshOutputPath")
            project["cameraFrameCount"] = result["cameraFrameCount"]
            artifacts["texturedMesh"] = {
                "path": "outputs/room-mesh.obj",
                "status": "ready",
                "sourceFingerprint": mesh_fingerprint,
                "updatedAt": updated_at,
                "metric": True,
                "stale": False,
            }
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
