from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .io import read_project, save_binary_ply, save_preview, write_json


PROVENANCE_MEASURED = 0
PROVENANCE_GENERATED = 1
PROVENANCE_LEARNED = 2
FUSION_CONTRACT_VERSION = "dense-surface-samples-v1"


@dataclass(frozen=True)
class DenseSamples:
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray
    scales: np.ndarray
    confidence: np.ndarray
    provenance: np.ndarray
    source_frame_indices: np.ndarray


def _empty_samples() -> DenseSamples:
    return DenseSamples(
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 3), dtype=np.uint8),
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 3), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=np.uint8),
        np.empty((0,), dtype=np.int32),
    )


def _quaternion_normals(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    w, x, y, z = values.T
    normals = np.column_stack(
        (
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        )
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return normals.astype(np.float32)


def samples_from_arrays(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
    provenance: int = PROVENANCE_MEASURED,
    voxel_size_m: float = 0.01,
) -> DenseSamples:
    point_values = np.asarray(points, dtype=np.float32)
    color_values = np.asarray(colors, dtype=np.uint8)
    if point_values.ndim != 2 or point_values.shape[1] != 3 or color_values.shape != point_values.shape:
        raise ValueError("Dense samples require matching N x 3 points and colors")
    count = len(point_values)
    confidence_values = (
        np.ones(count, dtype=np.float32)
        if confidence is None
        else np.asarray(confidence, dtype=np.float32)
    )
    if confidence_values.shape != (count,):
        raise ValueError("Dense sample confidence must contain one value per point")
    scale = max(float(voxel_size_m), 1e-5)
    return DenseSamples(
        point_values,
        color_values,
        np.zeros((count, 3), dtype=np.float32),
        np.full((count, 3), scale, dtype=np.float32),
        np.clip(confidence_values, 0.0, 1.0),
        np.full(count, provenance, dtype=np.uint8),
        np.full(count, -1, dtype=np.int32),
    )


def _resolve_dataset(pointer: Path) -> tuple[Path, dict[str, Any]]:
    pointer = pointer.resolve(strict=True)
    if pointer.is_file():
        link = json.loads(pointer.read_text(encoding="utf-8"))
        relative = Path(str(link.get("path", "")))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Dense dataset pointer contains an unsafe path")
        root = (pointer.parent / relative).resolve(strict=True)
        if not root.is_relative_to(pointer.parent):
            raise ValueError("Dense dataset pointer escapes its cache root")
    else:
        root = pointer
    manifest = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    return root, manifest


def load_dense_samples(pointer: Path) -> tuple[DenseSamples, dict[str, Any], Path]:
    root, dataset = _resolve_dataset(pointer)
    relative = Path(str(dataset.get("initializationParameters", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Dataset has no safe dense initialization parameter path")
    parameters = (root / relative).resolve(strict=True)
    if not parameters.is_relative_to(root):
        raise ValueError("Dense initialization parameters escape the dataset")
    with np.load(parameters, allow_pickle=False) as values:
        points = np.asarray(values["points"], dtype=np.float32)
        colors = np.asarray(values["colors"], dtype=np.uint8)
        scales = np.asarray(values["scales"], dtype=np.float32)
        quaternions = np.asarray(values["quaternions"], dtype=np.float32)
        count = len(points)
        if "fusion_confidence" in values:
            confidence_source = values["fusion_confidence"]
        elif dataset.get("directGaussianPrior"):
            # Legacy v18 sidecars stored direct-head opacity in `confidence`.
            # Opacity is a rendering parameter and must not be reinterpreted
            # as geometric confidence.
            confidence_source = np.ones(count)
        elif "confidence" in values:
            confidence_source = values["confidence"]
        else:
            confidence_source = np.ones(count)
        confidence = np.asarray(confidence_source, dtype=np.float32)
        provenance = np.asarray(
            values["provenance"]
            if "provenance" in values
            else np.full(count, PROVENANCE_LEARNED if not dataset.get("metric") else PROVENANCE_MEASURED),
            dtype=np.uint8,
        )
        owners = np.asarray(
            values["source_frame_indices"]
            if "source_frame_indices" in values
            else np.full(count, -1),
            dtype=np.int32,
        )
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or colors.shape != points.shape
        or scales.shape != points.shape
        or quaternions.shape != (len(points), 4)
        or confidence.shape != (len(points),)
        or provenance.shape != (len(points),)
        or owners.shape != (len(points),)
        or not np.isfinite(points).all()
        or not np.isfinite(scales).all()
        or not np.isfinite(quaternions).all()
        or not np.isfinite(confidence).all()
    ):
        raise ValueError("Dense initialization parameters violate the shared fusion contract")
    valid = (
        np.all(scales > 0.0, axis=1)
        & (confidence > 0.0)
        & np.isin(provenance, [PROVENANCE_MEASURED, PROVENANCE_GENERATED, PROVENANCE_LEARNED])
    )
    if not np.any(valid):
        raise RuntimeError("Dense initialization contains no confidence-bearing surface samples")
    return (
        DenseSamples(
            points[valid],
            colors[valid],
            _quaternion_normals(quaternions[valid]),
            scales[valid],
            np.clip(confidence[valid], 0.0, 1.0),
            provenance[valid],
            owners[valid],
        ),
        dataset,
        root,
    )


def _similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    covariance = centered_target.T @ centered_source / len(source)
    left, singular, right = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(left @ right) < 0.0:
        sign[-1] = -1.0
    rotation = left @ np.diag(sign) @ right
    variance = float(np.mean(np.sum(centered_source * centered_source, axis=1)))
    if variance <= 1e-12:
        raise ValueError("Shared media cameras do not span a solvable baseline")
    scale = float(np.sum(singular * sign) / variance)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Dense media alignment produced a non-positive scale")
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _rotation_error_degrees(left: np.ndarray, right: np.ndarray) -> float:
    delta = left.T @ right
    cosine = np.clip((float(np.trace(delta)) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def _frame_key(source_path: str | None, timestamp: float | None) -> tuple[str, int | None]:
    source = str(Path(source_path).resolve()).casefold() if source_path else ""
    stamp = None if timestamp is None else round(float(timestamp) * 1_000_000)
    return source, stamp


def align_media_samples(
    samples: DenseSamples,
    dataset: dict[str, Any],
    target_frames: Sequence[Any],
) -> tuple[DenseSamples, dict[str, Any]]:
    targets = {
        _frame_key(frame.media_source_path, frame.media_timestamp_seconds): frame
        for frame in target_frames
        if frame.media_source_path
    }
    source_poses: list[np.ndarray] = []
    target_poses: list[np.ndarray] = []
    source_indices: list[int] = []
    target_confidence: dict[int, float] = {}
    for frame in dataset.get("frames", []):
        timestamp = (
            float(frame["timestampUs"]) / 1_000_000.0
            if frame.get("timestampUs") is not None
            else None
        )
        target = targets.get(_frame_key(frame.get("sourcePath"), timestamp))
        if target is None:
            continue
        source_index = int(frame.get("sourceFrameIndex", frame.get("frameIndex", -1)))
        source_pose = np.asarray(frame["worldFromRgbCamera"], dtype=np.float64).reshape(4, 4)
        display = np.diag([*target.display_axes, 1.0])
        target_pose = display @ np.asarray(target.camera_to_global, dtype=np.float64)
        source_poses.append(source_pose)
        target_poses.append(target_pose)
        source_indices.append(source_index)
        inlier_score = min(1.0, max(0.0, target.localization_inliers / 120.0))
        rmse_score = min(1.0, max(0.0, (4.0 - target.localization_rmse_px) / 3.5))
        target_confidence[source_index] = max(0.2, 0.55 * inlier_score + 0.45 * rmse_score)
    if len(source_poses) < 3:
        raise RuntimeError("Dense media fusion needs at least three independently localized shared cameras")
    source_array = np.asarray(source_poses)
    target_array = np.asarray(target_poses)
    inliers = np.ones(len(source_array), dtype=bool)
    for _ in range(5):
        scale, rotation, translation = _similarity(
            source_array[inliers, :3, 3], target_array[inliers, :3, 3]
        )
        predicted = scale * (source_array[:, :3, 3] @ rotation.T) + translation
        residuals = np.linalg.norm(predicted - target_array[:, :3, 3], axis=1)
        median = float(np.median(residuals[inliers]))
        mad = float(np.median(np.abs(residuals[inliers] - median)))
        next_inliers = residuals <= max(median + 3.5 * max(mad, 1e-6), median * 2.5, 0.015)
        if np.count_nonzero(next_inliers) < 3 or np.array_equal(next_inliers, inliers):
            break
        inliers = next_inliers
    scale, rotation, translation = _similarity(
        source_array[inliers, :3, 3], target_array[inliers, :3, 3]
    )
    predicted = scale * (source_array[:, :3, 3] @ rotation.T) + translation
    residuals = np.linalg.norm(predicted - target_array[:, :3, 3], axis=1)
    baseline = max(
        float(np.linalg.norm(np.ptp(target_array[inliers, :3, 3], axis=0))), 0.05
    )
    rotation_errors = np.asarray(
        [
            _rotation_error_degrees(rotation @ source[:3, :3], target[:3, :3])
            for source, target in zip(source_array, target_array, strict=True)
        ]
    )
    normalized_residual = float(np.median(residuals[inliers]) / baseline)
    median_rotation = float(np.median(rotation_errors[inliers]))
    if normalized_residual > 0.10 or median_rotation > 8.0:
        raise RuntimeError(
            "Learned media geometry disagrees with metric localized cameras "
            f"({normalized_residual:.3f} normalized center residual, {median_rotation:.2f} deg rotation)"
        )
    points = scale * (samples.points @ rotation.T) + translation
    normals = samples.normals @ rotation.T
    confidence = samples.confidence.copy()
    for owner, value in target_confidence.items():
        confidence[samples.source_frame_indices == owner] *= value
    accepted_owners = np.asarray(source_indices, dtype=np.int32)[inliers]
    confidence[~np.isin(samples.source_frame_indices, accepted_owners)] *= 0.35
    aligned = DenseSamples(
        points.astype(np.float32),
        samples.colors,
        normals.astype(np.float32),
        (samples.scales * scale).astype(np.float32),
        np.clip(confidence, 0.0, 1.0),
        samples.provenance,
        samples.source_frame_indices,
    )
    return aligned, {
        "sharedCameraCount": len(source_array),
        "inlierCameraCount": int(np.count_nonzero(inliers)),
        "scale": scale,
        "medianCameraResidualM": float(np.median(residuals[inliers])),
        "normalizedMedianCameraResidual": normalized_residual,
        "medianRotationErrorDegrees": median_rotation,
    }


def fuse_dense_samples(batches: Sequence[DenseSamples], voxel_size: float) -> DenseSamples:
    usable = [batch for batch in batches if len(batch.points)]
    if not usable:
        return _empty_samples()
    points = np.concatenate([batch.points for batch in usable])
    colors = np.concatenate([batch.colors for batch in usable])
    normals = np.concatenate([batch.normals for batch in usable])
    scales = np.concatenate([batch.scales for batch in usable])
    confidence = np.concatenate([batch.confidence for batch in usable])
    provenance = np.concatenate([batch.provenance for batch in usable])
    owners = np.concatenate([batch.source_frame_indices for batch in usable])
    voxel = max(float(voxel_size), 1e-6)
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    provenance_weight = np.choose(provenance, [1.0, 0.5, 0.65]).astype(np.float32)
    weights = np.maximum(confidence * provenance_weight, 1e-5)
    count = int(inverse.max()) + 1
    measured_voxel = np.zeros(count, dtype=bool)
    np.logical_or.at(measured_voxel, inverse, provenance == PROVENANCE_MEASURED)
    # Calibrated sensor evidence is authoritative at conflicts. Generated or
    # learned observations still fill genuinely unobserved voxels, but cannot
    # pull an already measured surface away from its metric position.
    weights[(provenance != PROVENANCE_MEASURED) & measured_voxel[inverse]] = 0.0
    totals = np.bincount(inverse, weights=weights, minlength=count)
    fused_points = np.column_stack(
        [np.bincount(inverse, weights=points[:, axis] * weights, minlength=count) / totals for axis in range(3)]
    )
    fused_colors = np.column_stack(
        [np.bincount(inverse, weights=colors[:, axis] * weights, minlength=count) / totals for axis in range(3)]
    )
    order = np.lexsort((np.arange(len(points)), -weights, inverse))
    first = np.r_[True, inverse[order][1:] != inverse[order][:-1]]
    representative = order[first]
    return DenseSamples(
        fused_points.astype(np.float32),
        np.rint(fused_colors).clip(0, 255).astype(np.uint8),
        normals[representative],
        scales[representative],
        np.minimum(1.0, np.bincount(inverse, weights=confidence * weights, minlength=count) / totals).astype(np.float32),
        provenance[representative],
        owners[representative],
    )


def adaptive_voxel_size(samples: DenseSamples) -> float:
    lower, upper = np.percentile(samples.points, [1.0, 99.0], axis=0)
    extent = upper - lower
    diagonal = max(float(np.linalg.norm(extent)), 1e-3)
    footprint = float(np.median(np.max(samples.scales, axis=1)))
    return float(np.clip(footprint * 0.75, diagonal / 2500.0, diagonal / 180.0))


def bounded_mesh_samples(
    samples: DenseSamples,
    voxel_size: float,
    maximum_samples: int = 250_000,
) -> tuple[DenseSamples, float]:
    mesh_voxel = float(voxel_size)
    fused = fuse_dense_samples([samples], mesh_voxel)
    for _ in range(4):
        if len(fused.points) <= maximum_samples:
            break
        ratio = len(fused.points) / maximum_samples
        mesh_voxel *= max(1.08, ratio ** (1.0 / 3.0) * 1.04)
        fused = fuse_dense_samples([samples], mesh_voxel)
    if len(fused.points) > maximum_samples:
        indices = np.linspace(0, len(fused.points) - 1, maximum_samples, dtype=np.int64)
        fused = DenseSamples(
            fused.points[indices],
            fused.colors[indices],
            fused.normals[indices],
            fused.scales[indices],
            fused.confidence[indices],
            fused.provenance[indices],
            fused.source_frame_indices[indices],
        )
    return fused, mesh_voxel


def dense_surface_mesh(samples: DenseSamples, voxel_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import open3d as o3d

    accepted = samples.confidence >= 0.25
    if np.count_nonzero(accepted) < 100:
        raise RuntimeError("Dense fusion retained too few confident samples for meshing")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(samples.points[accepted].astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(samples.colors[accepted].astype(np.float64) / 255.0)
    radius = max(float(voxel_size), 1e-5)
    # Some learned backends publish true tangent discs; bounded depth-seed
    # fallbacks publish camera-facing or isotropic Gaussian axes. Re-estimate
    # mesh normals from the fused neighbourhood so BPA never mistakes camera
    # orientation for surface orientation.
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(max(radius * 5.0, 0.02), 32)
    )
    try:
        cloud.orient_normals_consistent_tangent_plane(24)
    except RuntimeError:
        # Disconnected fragments can defeat global orientation. BPA can still
        # use locally estimated normals and the guarded Poisson fallback below.
        pass
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        cloud,
        o3d.utility.DoubleVector([radius * 1.5, radius * 2.5, radius * 4.0]),
    )
    if not len(mesh.triangles):
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud, depth=9, width=0, scale=1.05, linear_fit=False
        )
        density = np.asarray(densities)
        if len(density):
            mesh.remove_vertices_by_mask(density < np.percentile(density, 8.0))
        mesh = mesh.crop(cloud.get_axis_aligned_bounding_box())
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > 600_000:
        mesh = mesh.simplify_quadric_decimation(600_000)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if not len(triangles):
        raise RuntimeError("Confidence-gated dense fusion produced no mesh triangles")
    if mesh.has_vertex_colors():
        vertex_colors = np.rint(np.asarray(mesh.vertex_colors) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        tree = o3d.geometry.KDTreeFlann(cloud)
        source_colors = samples.colors[accepted]
        vertex_colors = np.asarray(
            [source_colors[tree.search_knn_vector_3d(vertex.astype(float), 1)[1][0]] for vertex in vertices],
            dtype=np.uint8,
        )
    return vertices, triangles, vertex_colors


def publish_media_dense_artifacts(
    project_root: Path,
    dataset_pointer: Path,
    targets: Sequence[str],
    neural_sdf_worker: Path | None = None,
) -> dict[str, Any]:
    samples, dataset, _root = load_dense_samples(dataset_pointer)
    voxel_size = adaptive_voxel_size(samples)
    fused = fuse_dense_samples([samples], voxel_size)
    output = project_root / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    if "point_cloud" in targets:
        save_binary_ply(output / "room-cloud.ply", fused.points, fused.colors)
        save_preview(output / "preview.json", fused.points, fused.colors)
    mesh_metrics: dict[str, Any] = {"meshVertexCount": 0, "meshTriangleCount": 0}
    if "textured_mesh" in targets:
        from .mesh import _bake_triangle_atlas, _vertex_normals, _write_mesh, _write_png

        mesh_samples, mesh_voxel = bounded_mesh_samples(fused, voxel_size)
        vertices, triangles, vertex_colors = dense_surface_mesh(mesh_samples, mesh_voxel)
        neural_sdf_report: dict[str, Any] = {
            "status": "disabled",
            "method": "scanlan-neural-sdf-v1",
        }
        if neural_sdf_worker is not None:
            from .neural_sdf import refine_surface_with_worker

            vertices, triangles, neural_sdf_report = refine_surface_with_worker(
                vertices,
                triangles,
                project_root=project_root,
                worker=neural_sdf_worker,
                voxel_size_m=mesh_voxel,
                validation_report={
                    "accepted": True,
                    "contractVersion": "media-camera-depth-validation-v1",
                    "scaleStatus": "LEARNED_VALIDATED",
                    "quality": dataset.get("quality", {}),
                },
            )
        atlas, uvs, _resolution = _bake_triangle_atlas(vertex_colors, triangles)
        _write_png(output / "room-texture.png", atlas)
        (output / "room-mesh.mtl").write_text(
            "# ScanLan confidence-fused media material\n"
            "newmtl room_rgb\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\n"
            "Ks 0.0 0.0 0.0\nd 1.0\nillum 1\nmap_Kd room-texture.png\n",
            encoding="utf-8",
            newline="\n",
        )
        _write_mesh(output, vertices, _vertex_normals(vertices, triangles), triangles, uvs)
        mesh_metrics = {
            "meshVertexCount": len(vertices),
            "meshTriangleCount": len(triangles),
            "meshOutputPath": "outputs/room-mesh.obj",
            "meshMaterialPath": "outputs/room-mesh.mtl",
            "meshTexturePath": "outputs/room-texture.png",
            "meshFusionMethod": (
                "confidence_weighted_learned_surface_bpa+validated_neural_sdf"
                if neural_sdf_report.get("status") == "accepted"
                else "confidence_weighted_learned_surface_bpa"
            ),
            "neuralSdf": neural_sdf_report,
            "meshVoxelSize": mesh_voxel,
            "textureSource": "learned_surface_vertex_color_atlas",
        }
    fingerprint = hashlib.sha256(
        f"{FUSION_CONTRACT_VERSION}:{dataset.get('fingerprint')}:{voxel_size:.9g}".encode()
    ).hexdigest()[:24]
    result = {
        "fusionContract": FUSION_CONTRACT_VERSION,
        "sourceMode": dataset.get("sourceType", "media"),
        "metric": bool(dataset.get("metric", False)),
        "pointCount": len(fused.points),
        "voxelSize": voxel_size,
        "provenance": {"learned": int(np.count_nonzero(fused.provenance == PROVENANCE_LEARNED))},
        "sourceFingerprint": fingerprint,
        **mesh_metrics,
    }
    write_json(output / "dense-fusion-report.json", result)
    project = read_project(project_root)
    artifacts = project.setdefault("artifacts", {})
    updated_at = datetime.now(timezone.utc).isoformat()
    if "point_cloud" in targets:
        project["pointCount"] = len(fused.points)
        project["outputPath"] = "outputs/room-cloud.ply"
        artifacts["pointCloud"] = {
            "path": "outputs/room-cloud.ply", "status": "ready",
            "sourceFingerprint": fingerprint, "updatedAt": updated_at,
            "metric": bool(dataset.get("metric", False)), "stale": False,
        }
    if "textured_mesh" in targets:
        project["meshTriangleCount"] = result["meshTriangleCount"]
        project["meshOutputPath"] = "outputs/room-mesh.obj"
        project["neuralSdf"] = result.get(
            "neuralSdf",
            {"status": "disabled", "method": "scanlan-neural-sdf-v1"},
        )
        artifacts["texturedMesh"] = {
            "path": "outputs/room-mesh.obj", "status": "ready",
            "sourceFingerprint": fingerprint, "updatedAt": updated_at,
            "metric": bool(dataset.get("metric", False)), "stale": False,
        }
    project["processingStatus"] = "processing" if "gaussian_splat" in targets else "complete"
    project.pop("processingError", None)
    write_json(project_root / "project.json", project)
    return result
