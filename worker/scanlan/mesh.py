from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .calibration import project_rgb, world_from_depth_opencv
from .io import (
    PhaseData,
    RgbCameraModel,
    frame_rgb_camera,
    frame_rgb_from_depth,
    load_color,
    load_depth,
    load_source_rgb,
    write_json,
)
from .mesh_observations import project_world_points_to_depth
from .mesh_repair import MeshRepairSettings, repair_mesh_geometry


MAX_TEXTURE_FRAMES = 24
MAX_ATLAS_SIZE = int(os.environ.get("SCANLAN_ATLAS_SIZE", "8192"))
MAX_ATLAS_SIZE = min(16384, max(4096, MAX_ATLAS_SIZE))
TARGET_MESH_SAMPLES = 280
MIN_MESH_VOXEL_SIZE = float(os.environ.get("SCANLAN_MIN_MESH_VOXEL_M", "0.008"))
MIN_MESH_VOXEL_SIZE = min(0.025, max(0.004, MIN_MESH_VOXEL_SIZE))
MAX_MESH_TRIANGLES = int(os.environ.get("SCANLAN_MAX_MESH_TRIANGLES", "600000"))
MAX_MESH_TRIANGLES = min(2_000_000, max(50_000, MAX_MESH_TRIANGLES))
MAX_CHART_SIZE = 12
CHART_PADDING = 2
ATLAS_PAGE_PADDING = 8
CALIBRATION_SAMPLE_LIMIT = 40_000
CALIBRATION_GRID_SIZE = 16
LABEL_SMOOTHNESS = 0.42
LABEL_OPTIMIZATION_PASSES = 5
POSE_REFINEMENT_ITERATIONS = 3
MESH_CACHE_VERSION = "all-keyframe-shared-tsdf-v2"


@dataclass(frozen=True)
class PosedFrame:
    phase_name: str
    phase_id: str
    source: PhaseData
    frame_index: int
    camera_to_global: np.ndarray
    display_axes: tuple[float, float, float]
    image_y_up: bool
    image_path: Path | None = None
    rgb_camera_override: RgbCameraModel | None = None
    depthless: bool = False
    localization_inliers: int = 0
    localization_rmse_px: float = 0.0


@dataclass(frozen=True)
class TextureCalibration:
    gains: np.ndarray
    biases: np.ndarray
    spatial_biases: np.ndarray
    reference_frame: int
    overlap_edge_count: int
    sample_count: int


@dataclass(frozen=True)
class CalibrationSamples:
    vertex_indices: np.ndarray
    colors: np.ndarray
    weights: np.ndarray
    uvs: np.ndarray
    image_sizes: np.ndarray


def _select_texture_frames(frames: list[PosedFrame]) -> list[PosedFrame]:
    if len(frames) <= MAX_TEXTURE_FRAMES:
        return frames
    phase_indices: dict[str, list[int]] = {}
    for index, frame in enumerate(frames):
        phase_indices.setdefault(frame.phase_id, []).append(index)
    mandatory = [indices[len(indices) // 2] for indices in phase_indices.values()]
    if len(mandatory) > MAX_TEXTURE_FRAMES:
        sampled = np.linspace(0, len(mandatory) - 1, MAX_TEXTURE_FRAMES, dtype=np.int64)
        mandatory = [mandatory[int(index)] for index in np.unique(sampled)]
    chosen = set(mandatory)
    remaining_slots = MAX_TEXTURE_FRAMES - len(chosen)
    candidates = [index for index in range(len(frames)) if index not in chosen]
    if remaining_slots > 0 and candidates:
        sampled = np.linspace(0, len(candidates) - 1, remaining_slots, dtype=np.int64)
        chosen.update(candidates[int(index)] for index in np.unique(sampled))
    return [frames[index] for index in sorted(chosen)]


def _select_texture_frames_for_mesh(
    frames: list[PosedFrame],
    vertices: np.ndarray,
    normals: np.ndarray,
) -> list[PosedFrame]:
    """Greedily retain views that add surface coverage and source-pixel density."""
    if len(frames) <= MAX_TEXTURE_FRAMES:
        return frames
    sample_indices = _calibration_vertex_indices(len(vertices))
    points = vertices[sample_indices].astype(np.float64, copy=False)
    sampled_normals = normals[sample_indices]
    scores = np.zeros((len(frames), len(sample_indices)), dtype=np.float32)
    for frame_index, frame in enumerate(frames):
        camera = _texture_camera(frame)
        world_from_camera = world_from_depth_opencv(frame.camera_to_global, frame.image_y_up)
        camera_from_world = np.linalg.inv(world_from_camera)
        camera_points = points @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
        rgb_from_depth = _texture_rgb_from_depth(frame)
        rgb_points = camera_points @ rgb_from_depth[:3, :3].T + rgb_from_depth[:3, 3]
        u, v, z = project_rgb(rgb_points, camera)
        valid = (
            (z > 0.0)
            & (u >= 0.0)
            & (u <= camera.width - 1.0)
            & (v >= 0.0)
            & (v <= camera.height - 1.0)
        )
        camera_center = world_from_camera[:3, 3]
        to_camera = camera_center - points
        distance = np.linalg.norm(to_camera, axis=1)
        to_camera /= np.maximum(distance[:, None], 1e-8)
        signed = np.sum(sampled_normals * to_camera, axis=1)
        sign = -1.0 if np.count_nonzero(valid) and np.mean(signed[valid] > 0.0) < 0.35 else 1.0
        facing = np.clip(signed * sign, 0.0, 1.0)
        border = np.minimum.reduce((u, v, camera.width - 1.0 - u, camera.height - 1.0 - v))
        score = (
            facing**2
            * np.clip(border / 32.0, 0.0, 1.0)
            * (math.sqrt(max(camera.fx * camera.fy, 1.0)) / 1000.0)
            / np.maximum(distance, 0.5)
        )
        if frame.image_path is not None:
            score *= 1.15
        scores[frame_index, valid] = score[valid].astype(np.float32)

    chosen: list[int] = []
    current = np.zeros(len(sample_indices), dtype=np.float32)
    phase_groups: dict[str, list[int]] = {}
    for frame_index, frame in enumerate(frames):
        phase_groups.setdefault(frame.phase_id, []).append(frame_index)
    for candidates in phase_groups.values():
        if len(chosen) >= MAX_TEXTURE_FRAMES:
            break
        best = max(
            candidates,
            key=lambda index: float(np.sum(np.maximum(current, scores[index]) - current)),
        )
        if best not in chosen:
            chosen.append(best)
            current = np.maximum(current, scores[best])
    while len(chosen) < MAX_TEXTURE_FRAMES:
        remaining = [index for index in range(len(frames)) if index not in chosen]
        if not remaining:
            break
        gains = [float(np.sum(np.maximum(current, scores[index]) - current)) for index in remaining]
        best_position = int(np.argmax(gains))
        if gains[best_position] <= 1e-8:
            break
        best = remaining[best_position]
        chosen.append(best)
        current = np.maximum(current, scores[best])
    return [frames[index] for index in chosen]


def _display_matrix(frame: PosedFrame) -> np.ndarray:
    axes = np.diag([*frame.display_axes, 1.0])
    return axes @ np.asarray(frame.camera_to_global, dtype=np.float64)


def _texture_camera(frame: PosedFrame) -> RgbCameraModel:
    if frame.rgb_camera_override is not None:
        return frame.rgb_camera_override
    return frame_rgb_camera(frame.source.frames[frame.frame_index], frame.source)


def _texture_rgb_from_depth(frame: PosedFrame) -> np.ndarray:
    if frame.depthless:
        return np.eye(4, dtype=np.float64)
    return frame_rgb_from_depth(frame.source.frames[frame.frame_index], frame.source)


def _texture_timestamp_us(frame: PosedFrame) -> int:
    if frame.image_path is not None:
        return 0
    return frame.source.frames[frame.frame_index].timestamp_us


def _camera_payload(frame: PosedFrame, textured: bool) -> dict[str, Any]:
    camera = _texture_camera(frame)
    world_from_depth_camera = world_from_depth_opencv(frame.camera_to_global, frame.image_y_up)
    world_from_rgb_camera = world_from_depth_camera @ np.linalg.inv(_texture_rgb_from_depth(frame))
    return {
        "phaseName": frame.phase_name,
        "phaseId": frame.phase_id,
        "frameIndex": frame.frame_index,
        "timestampUs": _texture_timestamp_us(frame),
        "matrix": [round(float(value), 8) for value in _display_matrix(frame).reshape(-1)],
        "worldFromDepthCameraOpenCv": [
            round(float(value), 10) for value in world_from_depth_camera.reshape(-1)
        ],
        "worldFromRgbCameraOpenCv": [
            round(float(value), 10) for value in world_from_rgb_camera.reshape(-1)
        ],
        "camera": {
            "width": camera.width,
            "height": camera.height,
            "fx": round(camera.fx, 8),
            "fy": round(camera.fy, 8),
            "cx": round(camera.cx, 8),
            "cy": round(camera.cy, 8),
            "model": camera.model,
            "distortion": [round(float(value), 10) for value in camera.distortion],
        },
        "aspect": round(camera.width / max(camera.height, 1), 6),
        "fovYDegrees": round(math.degrees(2.0 * math.atan(camera.height / (2.0 * camera.fy))), 5),
        "imageYUp": frame.image_y_up,
        "textureFrame": textured,
        "supplementalPhoto": frame.image_path is not None,
        "localizationInliers": frame.localization_inliers,
        "localizationRmsePixels": round(frame.localization_rmse_px, 4),
    }


def _load_supplemental_texture_frames(
    project_root: Path,
    reference_frame: PosedFrame,
) -> list[PosedFrame]:
    manifest_path = project_root / "supplemental-photos.json"
    if not manifest_path.is_file():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(payload.get("schemaVersion", 0)) != 1:
        raise ValueError("Supplemental photo manifest must use schema 1")
    result: list[PosedFrame] = []
    for index, photo in enumerate(payload.get("photos", [])):
        relative_path = Path(str(photo["path"]))
        image_path = (project_root / relative_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Supplemental texture photo is missing: {relative_path}")
        camera_payload = photo["camera"]
        camera = RgbCameraModel(
            int(camera_payload["width"]),
            int(camera_payload["height"]),
            float(camera_payload["fx"]),
            float(camera_payload["fy"]),
            float(camera_payload["cx"]),
            float(camera_payload["cy"]),
            str(camera_payload.get("model", "pinhole")),
            tuple(float(value) for value in camera_payload.get("distortion", [])),
        )
        pose = np.asarray(photo["worldFromCamera"], dtype=np.float64).reshape(4, 4)
        if not np.all(np.isfinite(pose)):
            raise ValueError(f"Supplemental photo {relative_path} has an invalid pose")
        result.append(
            PosedFrame(
                phase_name=str(photo.get("name", relative_path.stem)),
                phase_id=f"supplemental:{photo.get('id', index)}",
                source=reference_frame.source,
                frame_index=reference_frame.frame_index,
                camera_to_global=pose,
                display_axes=reference_frame.display_axes,
                image_y_up=False,
                image_path=image_path,
                rgb_camera_override=camera,
                depthless=True,
                localization_inliers=int(photo.get("inlierCount", 0)),
                localization_rmse_px=float(photo.get("reprojectionRmsePixels", 0.0)),
            )
        )
    return result


def _write_png(path: Path, image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("PNG texture must be an H x W x 3 uint8 image")
    height, width, _ = image.shape

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    scanlines = b"".join(b"\x00" + row.tobytes() for row in image)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        # PNG is lossless at every level.  Level 2 cuts the large atlas export
        # pause substantially while retaining identical texture pixels.
        + chunk(b"IDAT", zlib.compress(scanlines, level=2))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def _append_lines(handle: Any, lines: list[str]) -> None:
    if lines:
        handle.write("".join(lines))
        lines.clear()


def _vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    normals = np.zeros(vertices.shape, dtype=np.float64)
    if not len(triangles):
        return normals.astype(np.float32)
    face_normals = np.cross(
        vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
        vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
    )
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    return normals.astype(np.float32)


def _frame_depth_mesh(frame: PosedFrame) -> tuple[np.ndarray, np.ndarray]:
    camera = frame.source.camera
    depth = load_depth(frame.source.frames[frame.frame_index], camera).astype(np.float64)
    stride = max(1, math.ceil(max(camera.width, camera.height) / TARGET_MESH_SAMPLES))
    y_pixels = np.arange(0, camera.height, stride, dtype=np.int64)
    x_pixels = np.arange(0, camera.width, stride, dtype=np.int64)
    sampled = depth[np.ix_(y_pixels, x_pixels)] / camera.depth_scale
    valid = (sampled > 0.25) & (sampled <= camera.max_depth_m)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int64)

    yy, xx = np.meshgrid(y_pixels, x_pixels, indexing="ij")
    z = sampled[valid]
    x = (xx[valid] - camera.cx) * z / camera.fx
    y = (yy[valid] - camera.cy) * z / camera.fy
    if frame.image_y_up:
        y *= -1.0
    camera_points = np.column_stack((x, y, z, np.ones_like(z)))
    world = (np.asarray(frame.camera_to_global) @ camera_points.T).T[:, :3]

    vertex_map = np.full(sampled.shape, -1, dtype=np.int64)
    local_ids = np.arange(int(valid.sum()), dtype=np.int64)
    vertex_map[valid] = local_ids

    if sampled.shape[0] < 2 or sampled.shape[1] < 2:
        return world.astype(np.float32), np.empty((0, 3), dtype=np.int64)
    a, b = vertex_map[:-1, :-1], vertex_map[1:, :-1]
    c, d = vertex_map[:-1, 1:], vertex_map[1:, 1:]
    za, zb = sampled[:-1, :-1], sampled[1:, :-1]
    zc, zd = sampled[:-1, 1:], sampled[1:, 1:]
    threshold_abc = np.maximum(0.045, np.minimum(np.minimum(za, zb), zc) * 0.025)
    threshold_bdc = np.maximum(0.045, np.minimum(np.minimum(zb, zd), zc) * 0.025)
    mask_abc = (
        (a >= 0)
        & (b >= 0)
        & (c >= 0)
        & (np.maximum(np.maximum(za, zb), zc) - np.minimum(np.minimum(za, zb), zc) <= threshold_abc)
    )
    mask_bdc = (
        (b >= 0)
        & (d >= 0)
        & (c >= 0)
        & (np.maximum(np.maximum(zb, zd), zc) - np.minimum(np.minimum(zb, zd), zc) <= threshold_bdc)
    )
    triangles_abc = np.column_stack((a[mask_abc], b[mask_abc], c[mask_abc]))
    triangles_bdc = np.column_stack((b[mask_bdc], d[mask_bdc], c[mask_bdc]))
    reverses_winding = frame.image_y_up != (np.prod(frame.display_axes) < 0)
    if reverses_winding:
        triangles_abc = triangles_abc[:, [0, 2, 1]]
        triangles_bdc = triangles_bdc[:, [0, 2, 1]]
    return world.astype(np.float32), np.concatenate((triangles_abc, triangles_bdc), axis=0)


def _remove_unreferenced_vertices(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    referenced = np.unique(triangles.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[referenced] = np.arange(len(referenced), dtype=np.int64)
    return vertices[referenced], remap[triangles]


def _weld_depth_meshes(
    meshes: list[tuple[np.ndarray, np.ndarray]],
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Spatially weld depth sheets and discard coincident faces.

    This is the dependency-free fallback. Production builds use TSDF extraction,
    but known-pose projects still get one indexed surface instead of one OBJ
    object laid directly on top of another.
    """
    vertex_batches: list[np.ndarray] = []
    triangle_batches: list[np.ndarray] = []
    vertex_offset = 0
    for vertices, triangles in meshes:
        if not len(triangles):
            continue
        vertex_batches.append(vertices)
        triangle_batches.append(triangles + vertex_offset)
        vertex_offset += len(vertices)
    if not triangle_batches:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int64)

    source_vertices = np.concatenate(vertex_batches, axis=0).astype(np.float64, copy=False)
    source_triangles = np.concatenate(triangle_batches, axis=0)
    weld_size = max(float(voxel_size_m), MIN_MESH_VOXEL_SIZE)
    voxel_keys = np.floor(source_vertices / weld_size + 0.5).astype(np.int64)
    _, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)
    vertices = np.zeros((int(inverse.max()) + 1, 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=len(vertices)).astype(np.float64)
    for axis in range(3):
        vertices[:, axis] = np.bincount(
            inverse,
            weights=source_vertices[:, axis],
            minlength=len(vertices),
        ) / counts
    triangles = inverse[source_triangles]
    valid = (
        (triangles[:, 0] != triangles[:, 1])
        & (triangles[:, 1] != triangles[:, 2])
        & (triangles[:, 2] != triangles[:, 0])
    )
    triangles = triangles[valid]
    canonical = np.sort(triangles, axis=1)
    _, first = np.unique(canonical, axis=0, return_index=True)
    triangles = triangles[np.sort(first)]
    areas = np.linalg.norm(
        np.cross(
            vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
            vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
        ),
        axis=1,
    )
    triangles = triangles[areas > weld_size * weld_size * 1e-4]
    return _remove_unreferenced_vertices(vertices.astype(np.float32), triangles)


def _prepare_fused_mesh(
    o3d: Any,
    mesh: Any,
    mesh_voxel: float,
    progress: Callable[..., None] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean and bound a legacy Open3D mesh extracted from any TSDF backend."""

    raw_triangle_count = len(mesh.triangles)
    if len(mesh.triangles) > MAX_MESH_TRIANGLES:
        # QEM directly on a multi-million-triangle marching-cubes mesh is the
        # dominant CPU cost (about 60 seconds for a 4.25M-face room). A small
        # spatial pre-pass removes redundant sub-centimetre vertices in a few
        # seconds, after which QEM still supplies the high-quality final shape.
        # On the same room this preserves the 600k target while reducing the
        # combined simplification time to about eight seconds.
        precluster_voxel = mesh_voxel * 1.75
        if progress:
            progress(
                "Meshing",
                f"Condensing {raw_triangle_count:,} raw triangles at {precluster_voxel * 1000.0:.0f} mm",
                0,
                None,
                0.34,
            )
        mesh = mesh.simplify_vertex_clustering(
            precluster_voxel,
            contraction=o3d.geometry.SimplificationContraction.Average,
        )
    if len(mesh.triangles) > MAX_MESH_TRIANGLES:
        if progress:
            progress(
                "Meshing",
                f"Refining {len(mesh.triangles):,} triangles to the {MAX_MESH_TRIANGLES:,} quality budget",
                0,
                None,
                0.44,
            )
        mesh = mesh.simplify_quadric_decimation(MAX_MESH_TRIANGLES)
    if progress:
        progress(
            "Meshing",
            f"Cleaning {len(mesh.triangles):,} simplified triangles",
            0,
            None,
            0.53,
        )
    # TSDF extraction is already effectively manifold. The previous full raw
    # non-manifold scan took ten seconds and removed only 14 of 4.25M faces.
    # Cleanup after simplification is both sufficient and dramatically cheaper.
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    if len(mesh.triangles):
        clusters, counts, _ = mesh.cluster_connected_triangles()
        clusters = np.asarray(clusters)
        counts = np.asarray(counts)
        if len(counts):
            mesh.remove_triangles_by_mask(counts[clusters] < 12)
        mesh.remove_unreferenced_vertices()
        mesh = mesh.filter_smooth_taubin(number_of_iterations=1)
    if progress:
        progress(
            "Meshing",
            f"Prepared {len(mesh.triangles):,} fused triangles for native RGB texturing",
            0,
            None,
            0.58,
        )
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if not len(triangles):
        raise RuntimeError("TSDF fusion produced no mesh triangles")
    return vertices, triangles


def _open3d_fused_mesh(
    frames: list[PosedFrame],
    voxel_size_m: float,
    progress: Callable[..., None] | None,
) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    mesh_voxel = max(float(voxel_size_m), MIN_MESH_VOXEL_SIZE)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=mesh_voxel,
        sdf_trunc=max(mesh_voxel * 4.0, 0.03),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )
    intrinsic_cache: dict[str, Any] = {}
    for index, frame in enumerate(frames, start=1):
        phase = frame.source
        phase_key = str(phase.root)
        intrinsic = intrinsic_cache.get(phase_key)
        if intrinsic is None:
            camera = phase.camera
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                camera.width,
                camera.height,
                camera.fx,
                camera.fy,
                camera.cx,
                camera.cy,
            )
            intrinsic_cache[phase_key] = intrinsic
        camera = phase.camera
        depth = np.ascontiguousarray(load_depth(phase.frames[frame.frame_index], camera))
        # Open3D requires an RGBD image even for a geometry-only TSDF.
        color = np.ascontiguousarray(load_color(phase.frames[frame.frame_index], camera))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth),
            depth_scale=camera.depth_scale,
            depth_trunc=camera.max_depth_m,
            convert_rgb_to_intensity=False,
        )
        world_from_camera = world_from_depth_opencv(frame.camera_to_global, frame.image_y_up)
        volume.integrate(rgbd, intrinsic, np.linalg.inv(world_from_camera))
        if progress:
            progress(
                "Meshing",
                f"Fused depth keyframe {index} of {len(frames)}",
                0,
                None,
                0.25 * index / len(frames),
            )

    if progress:
        progress(
            "Meshing",
            "Extracting the fused TSDF surface",
            0,
            None,
            0.28,
        )
    return _prepare_fused_mesh(o3d, volume.extract_triangle_mesh(), mesh_voxel, progress)


def _fused_mesh(
    frames: list[PosedFrame],
    voxel_size_m: float,
    progress: Callable[..., None] | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        vertices, triangles = _open3d_fused_mesh(frames, voxel_size_m, progress)
        return vertices, triangles, "tsdf"
    except Exception as error:
        if progress:
            progress(
                "Meshing",
                f"TSDF mesh unavailable; welding depth surfaces instead - {str(error).splitlines()[0]}",
                0,
                None,
                0.0,
            )
        meshes: list[tuple[np.ndarray, np.ndarray]] = []
        for index, frame in enumerate(frames, start=1):
            meshes.append(_frame_depth_mesh(frame))
            if progress:
                progress(
                    "Meshing",
                    f"Welded depth keyframe {index} of {len(frames)}",
                    0,
                    None,
                    0.25 * index / len(frames),
                )
        vertices, triangles = _weld_depth_meshes(meshes, voxel_size_m)
        if not len(triangles):
            raise RuntimeError("Depth fusion produced no mesh triangles")
        return vertices, triangles, "welded_depth"


def _bilinear_rgb(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    left = np.floor(u).astype(np.int64)
    top = np.floor(v).astype(np.int64)
    right = np.minimum(left + 1, image.shape[1] - 1)
    bottom = np.minimum(top + 1, image.shape[0] - 1)
    fx = (u - left).astype(np.float32)[:, None]
    fy = (v - top).astype(np.float32)[:, None]
    upper = image[top, left].astype(np.float32) * (1.0 - fx) + image[top, right].astype(np.float32) * fx
    lower = image[bottom, left].astype(np.float32) * (1.0 - fx) + image[bottom, right].astype(np.float32) * fx
    return upper * (1.0 - fy) + lower * fy


def _load_texture_image(frame: PosedFrame) -> np.ndarray:
    if frame.image_path is not None:
        from PIL import Image

        with Image.open(frame.image_path) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.uint8)
    else:
        source_frame = frame.source.frames[frame.frame_index]
        image = load_source_rgb(source_frame, frame.source)
    camera = _texture_camera(frame)
    if image.shape[:2] == (camera.height, camera.width):
        return image
    from PIL import Image

    return np.asarray(
        Image.fromarray(image).resize(
            (camera.width, camera.height),
            Image.Resampling.LANCZOS,
        ),
        dtype=np.uint8,
    )


_SRGB_LINEAR_LUT = np.asarray(
    [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in np.linspace(0.0, 1.0, 256)
    ],
    dtype=np.float32,
)


def _linear_to_srgb(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    encoded = np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    )
    return np.rint(encoded * 255.0).clip(0, 255).astype(np.uint8)


def _bilinear_linear_rgb(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    left = np.floor(u).astype(np.int64)
    top = np.floor(v).astype(np.int64)
    right = np.minimum(left + 1, image.shape[1] - 1)
    bottom = np.minimum(top + 1, image.shape[0] - 1)
    fx = (u - left).astype(np.float32)[:, None]
    fy = (v - top).astype(np.float32)[:, None]
    upper = _SRGB_LINEAR_LUT[image[top, left]] * (1.0 - fx) + _SRGB_LINEAR_LUT[
        image[top, right]
    ] * fx
    lower = _SRGB_LINEAR_LUT[image[bottom, left]] * (1.0 - fx) + _SRGB_LINEAR_LUT[
        image[bottom, right]
    ] * fx
    return upper * (1.0 - fy) + lower * fy


def _depthless_zbuffer(
    vertices: np.ndarray,
    frame: PosedFrame,
    camera_from_world: np.ndarray,
    max_dimension: int = 1024,
) -> tuple[np.ndarray, float, float]:
    camera = _texture_camera(frame)
    points = vertices.astype(np.float64, copy=False)
    camera_points = points @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    u, v, z = project_rgb(camera_points, camera)
    scale = min(1.0, max_dimension / max(camera.width, camera.height))
    width = max(1, round(camera.width * scale))
    height = max(1, round(camera.height * scale))
    x = np.rint((u + 0.5) * width / camera.width - 0.5).astype(np.int64)
    y = np.rint((v + 0.5) * height / camera.height - 0.5).astype(np.int64)
    valid = (
        (z > 0.0)
        & np.isfinite(u)
        & np.isfinite(v)
        & (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
    )
    zbuffer = np.full(width * height, np.inf, dtype=np.float32)
    np.minimum.at(zbuffer, y[valid] * width + x[valid], z[valid].astype(np.float32))
    return zbuffer.reshape(height, width), width / camera.width, height / camera.height


def _frame_observations(
    vertices: np.ndarray,
    normals: np.ndarray,
    frame: PosedFrame,
    voxel_size_m: float,
    vertex_indices: np.ndarray,
    *,
    image: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return linear RGB, geometric confidence, and native-image UV for vertices."""
    if image is None:
        image = _load_texture_image(frame)
    rgb_camera = _texture_camera(frame)
    rgb_from_depth = _texture_rgb_from_depth(frame)
    world_from_camera = world_from_depth_opencv(frame.camera_to_global, frame.image_y_up)
    camera_from_world = np.linalg.inv(world_from_camera)
    camera_center = world_from_camera[:3, 3]
    points = vertices[vertex_indices].astype(np.float64, copy=False)
    camera_points = points @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    rgb_points = camera_points @ rgb_from_depth[:3, :3].T + rgb_from_depth[:3, 3]
    rgb_u, rgb_v, rgb_z = project_rgb(rgb_points, rgb_camera)
    valid = (
        (rgb_z > 0.0)
        & np.isfinite(rgb_u)
        & np.isfinite(rgb_v)
        & (rgb_u >= 0.0)
        & (rgb_u <= rgb_camera.width - 1.001)
        & (rgb_v >= 0.0)
        & (rgb_v <= rgb_camera.height - 1.001)
    )
    residual = np.full(len(points), np.inf, dtype=np.float64)
    tolerance = np.maximum(
        max(voxel_size_m * 2.0, 0.022),
        np.maximum(camera_points[:, 2], 0.0) * 0.009,
    )

    if frame.depthless:
        zbuffer, scale_x, scale_y = _depthless_zbuffer(vertices, frame, camera_from_world)
        x = np.rint((rgb_u + 0.5) * scale_x - 0.5).astype(np.int64)
        y = np.rint((rgb_v + 0.5) * scale_y - 0.5).astype(np.int64)
        in_buffer = valid & (x >= 0) & (x < zbuffer.shape[1]) & (y >= 0) & (y < zbuffer.shape[0])
        observed = np.full(len(points), np.inf, dtype=np.float32)
        observed[in_buffer] = zbuffer[y[in_buffer], x[in_buffer]]
        residual[in_buffer] = np.abs(observed[in_buffer] - rgb_z[in_buffer])
        tolerance = np.maximum(tolerance, np.maximum(rgb_z, 0.0) * 0.004 + 0.008)
        valid &= in_buffer & np.isfinite(observed)
    else:
        projection = project_world_points_to_depth(points, frame, voxel_size_m)
        observed = projection.observed_depth_m
        residual[projection.in_view] = np.abs(
            observed[projection.in_view] - projection.camera_points[projection.in_view, 2]
        )
        valid &= projection.in_view & (observed > 0.0)

    to_camera = camera_center - points
    distances = np.linalg.norm(to_camera, axis=1)
    to_camera /= np.maximum(distances[:, None], 1e-8)
    signed_facing = np.sum(normals[vertex_indices] * to_camera, axis=1)
    visible_facing = signed_facing[valid & (residual <= tolerance)]
    normal_sign = -1.0 if len(visible_facing) and np.mean(visible_facing > 0.0) < 0.35 else 1.0
    facing = np.clip(signed_facing * normal_sign, 0.0, 1.0)
    border = np.minimum.reduce(
        (
            rgb_u,
            rgb_v,
            rgb_camera.width - 1.0 - rgb_u,
            rgb_camera.height - 1.0 - rgb_v,
        )
    )
    border_weight = np.clip(border / 24.0, 0.0, 1.0)
    base_weights = (
        np.maximum(facing, 0.02) ** 2
        * border_weight
        * (math.sqrt(max(rgb_camera.fx * rgb_camera.fy, 1.0)) / 1000.0)
        / np.maximum(distances, 0.5)
    ).astype(np.float32)
    weights = base_weights * np.exp(-np.square(residual / np.maximum(tolerance, 1e-6)))
    weights[~valid | (residual > tolerance)] = 0.0
    loose_weights = base_weights * np.exp(
        -np.square(residual / np.maximum(tolerance * 4.0, 1e-6))
    )
    loose_weights[~valid] = 0.0
    colors = np.zeros((len(points), 3), dtype=np.float32)
    accepted = loose_weights > 0.0
    if np.any(accepted):
        colors[accepted] = _bilinear_linear_rgb(image, rgb_u[accepted], rgb_v[accepted])
    uvs = np.column_stack((rgb_u, rgb_v))
    uvs[~valid] = np.nan
    return colors, weights, loose_weights.astype(np.float32), uvs.astype(np.float32)


def _calibration_vertex_indices(vertex_count: int) -> np.ndarray:
    if vertex_count <= CALIBRATION_SAMPLE_LIMIT:
        return np.arange(vertex_count, dtype=np.int64)
    return np.linspace(0, vertex_count - 1, CALIBRATION_SAMPLE_LIMIT, dtype=np.int64)


def _collect_calibration_samples(
    vertices: np.ndarray,
    normals: np.ndarray,
    frames: list[PosedFrame],
    voxel_size_m: float,
) -> CalibrationSamples:
    indices = _calibration_vertex_indices(len(vertices))
    colors = np.zeros((len(frames), len(indices), 3), dtype=np.float32)
    weights = np.zeros((len(frames), len(indices)), dtype=np.float32)
    uvs = np.full((len(frames), len(indices), 2), np.nan, dtype=np.float32)
    image_sizes = np.zeros((len(frames), 2), dtype=np.int32)
    for frame_index, frame in enumerate(frames):
        image = _load_texture_image(frame)
        image_sizes[frame_index] = (image.shape[1], image.shape[0])
        colors[frame_index], weights[frame_index], _, uvs[frame_index] = _frame_observations(
            vertices,
            normals,
            frame,
            voxel_size_m,
            indices,
            image=image,
        )
    return CalibrationSamples(indices, colors, weights, uvs, image_sizes)


def _estimate_texture_calibration(samples: CalibrationSamples) -> TextureCalibration:
    frame_count, sample_count = samples.weights.shape
    gains = np.ones((frame_count, 3), dtype=np.float32)
    biases = np.zeros((frame_count, 3), dtype=np.float32)
    spatial = np.zeros((frame_count, CALIBRATION_GRID_SIZE, CALIBRATION_GRID_SIZE, 3), dtype=np.float32)
    if frame_count < 2 or sample_count == 0:
        return TextureCalibration(gains, biases, spatial, 0, 0, sample_count)

    overlap = np.zeros((frame_count, frame_count), dtype=np.int32)
    equations: list[tuple[int, int, np.ndarray, float]] = []
    for first in range(frame_count):
        for second in range(first + 1, frame_count):
            common = (samples.weights[first] > 0.0) & (samples.weights[second] > 0.0)
            count = int(np.count_nonzero(common))
            overlap[first, second] = overlap[second, first] = count
            if count < 96:
                continue
            first_colors = samples.colors[first, common]
            second_colors = samples.colors[second, common]
            usable = (
                (first_colors > 0.015)
                & (first_colors < 0.97)
                & (second_colors > 0.015)
                & (second_colors < 0.97)
            )
            offsets = np.zeros(3, dtype=np.float64)
            channel_valid = True
            for channel in range(3):
                mask = usable[:, channel]
                if np.count_nonzero(mask) < 64:
                    channel_valid = False
                    break
                delta = np.log(second_colors[mask, channel]) - np.log(first_colors[mask, channel])
                median = np.median(delta)
                deviation = np.median(np.abs(delta - median))
                robust = np.abs(delta - median) <= max(0.04, deviation * 3.5)
                offsets[channel] = np.median(delta[robust]) if np.any(robust) else median
            if channel_valid:
                equations.append((first, second, offsets, math.sqrt(count)))

    reference = int(np.argmax(overlap.sum(axis=1))) if frame_count else 0
    if equations:
        rows = len(equations) + 1
        matrix = np.zeros((rows, frame_count), dtype=np.float64)
        targets = np.zeros((rows, 3), dtype=np.float64)
        for row, (first, second, offset, confidence) in enumerate(equations):
            # corrected_first == corrected_second, so log(g_first)-log(g_second)
            # equals log(color_second)-log(color_first).
            matrix[row, first] = confidence
            matrix[row, second] = -confidence
            targets[row] = offset * confidence
        matrix[-1, reference] = max(10.0, math.sqrt(sample_count))
        log_gains = np.linalg.lstsq(matrix, targets, rcond=None)[0]
        gains = np.exp(log_gains).clip(0.55, 1.8).astype(np.float32)

    corrected = samples.colors * gains[:, None, :]
    visible = samples.weights > 0.0
    for _ in range(3):
        numerator = np.sum(
            (corrected + biases[:, None, :]) * samples.weights[..., None], axis=0
        )
        denominator = samples.weights.sum(axis=0)[:, None]
        consensus = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0.0,
        )
        for frame_index in range(frame_count):
            if frame_index == reference:
                continue
            mask = visible[frame_index] & (denominator[:, 0] > samples.weights[frame_index])
            if np.count_nonzero(mask) < 96:
                continue
            residual = consensus[mask] - corrected[frame_index, mask]
            biases[frame_index] = np.median(residual, axis=0).clip(-0.08, 0.08)

    corrected += biases[:, None, :]
    numerator = np.sum(corrected * samples.weights[..., None], axis=0)
    denominator = samples.weights.sum(axis=0)[:, None]
    consensus = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    for frame_index in range(frame_count):
        mask = visible[frame_index] & (denominator[:, 0] > samples.weights[frame_index])
        if np.count_nonzero(mask) < 96:
            continue
        camera_uv = samples.uvs[frame_index, mask]
        finite = np.all(np.isfinite(camera_uv), axis=1)
        if not np.any(finite):
            continue
        camera_uv = camera_uv[finite]
        residual = (consensus[mask][finite] - corrected[frame_index, mask][finite]).clip(-0.12, 0.12)
        maximum = samples.image_sizes[frame_index].astype(np.float64) - 1.0
        if np.any(maximum <= 0.0):
            continue
        gx = np.minimum(
            CALIBRATION_GRID_SIZE - 1,
            np.floor(camera_uv[:, 0] / maximum[0] * CALIBRATION_GRID_SIZE).astype(np.int64),
        )
        gy = np.minimum(
            CALIBRATION_GRID_SIZE - 1,
            np.floor(camera_uv[:, 1] / maximum[1] * CALIBRATION_GRID_SIZE).astype(np.int64),
        )
        cell_sum = np.zeros((CALIBRATION_GRID_SIZE, CALIBRATION_GRID_SIZE, 3), dtype=np.float64)
        cell_count = np.zeros((CALIBRATION_GRID_SIZE, CALIBRATION_GRID_SIZE), dtype=np.float64)
        np.add.at(cell_sum, (gy, gx), residual)
        np.add.at(cell_count, (gy, gx), 1.0)
        data = np.divide(
            cell_sum,
            cell_count[..., None],
            out=np.zeros_like(cell_sum),
            where=cell_count[..., None] > 0.0,
        )
        field = data.copy()
        for _ in range(24):
            neighbors = (
                np.roll(field, 1, axis=0)
                + np.roll(field, -1, axis=0)
                + np.roll(field, 1, axis=1)
                + np.roll(field, -1, axis=1)
            ) * 0.25
            data_weight = np.minimum(cell_count / 8.0, 4.0)[..., None]
            field = (data * data_weight + neighbors * 1.5) / (data_weight + 1.5)
        spatial[frame_index] = field.clip(-0.08, 0.08).astype(np.float32)

    return TextureCalibration(
        gains,
        biases,
        spatial,
        reference,
        len(equations),
        sample_count,
    )


def _rotation_angle_degrees(matrix: np.ndarray) -> float:
    cosine = np.clip((float(np.trace(matrix[:3, :3])) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def _refine_texture_poses(
    vertices: np.ndarray,
    normals: np.ndarray,
    frames: list[PosedFrame],
    voxel_size_m: float,
    progress: Callable[..., None] | None,
) -> tuple[list[PosedFrame], int, float, float]:
    """Conservatively refine RGB-D poses against the final fused surface."""
    try:
        import open3d as o3d
    except Exception:
        return frames, 0, 0.0, 0.0

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(vertices.astype(np.float64, copy=False))
    target.normals = o3d.utility.Vector3dVector(normals.astype(np.float64, copy=False))
    target = target.voxel_down_sample(max(voxel_size_m, 0.008))
    if not target.has_normals():
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(max(voxel_size_m * 5.0, 0.04), 30)
        )
    threshold = max(voxel_size_m * 2.5, 0.025)
    refined_frames: list[PosedFrame] = []
    translation_updates: list[float] = []
    rotation_updates: list[float] = []

    for frame_index, frame in enumerate(frames):
        if frame.depthless:
            refined_frames.append(frame)
            continue
        camera = frame.source.camera
        depth = load_depth(frame.source.frames[frame.frame_index], camera).astype(np.float64)
        depth /= camera.depth_scale
        stride = max(1, int(round(max(camera.width, camera.height) / 320.0)))
        y, x = np.indices(depth.shape)
        valid = (depth > 0.25) & (depth <= camera.max_depth_m)
        valid &= (x % stride == 0) & (y % stride == 0)
        z = depth[valid]
        if len(z) < 400:
            refined_frames.append(frame)
            continue
        points = np.column_stack(
            (
                (x[valid] - camera.cx) * z / camera.fx,
                (y[valid] - camera.cy) * z / camera.fy,
                z,
            )
        )
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(points)
        initial = world_from_depth_opencv(frame.camera_to_global, frame.image_y_up)
        try:
            result = o3d.pipelines.registration.registration_icp(
                source,
                target,
                threshold,
                initial,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20),
            )
        except Exception:
            refined_frames.append(frame)
            continue
        candidate = np.asarray(result.transformation, dtype=np.float64)
        correction = np.linalg.inv(initial) @ candidate
        translation = float(np.linalg.norm(correction[:3, 3]))
        rotation = _rotation_angle_degrees(correction)
        if (
            not np.all(np.isfinite(candidate))
            or float(result.fitness) < 0.35
            or translation > 0.025
            or rotation > 2.0
        ):
            refined_frames.append(frame)
            continue
        camera_to_global = (
            candidate @ np.diag([1.0, -1.0, 1.0, 1.0])
            if frame.image_y_up
            else candidate
        )
        refined_frames.append(replace(frame, camera_to_global=camera_to_global))
        translation_updates.append(translation)
        rotation_updates.append(rotation)
        if progress:
            progress(
                "Texturing",
                f"Refined texture camera {frame_index + 1} of {len(frames)} against the fused surface",
                0,
                None,
                0.50 + 0.04 * (frame_index + 1) / len(frames),
            )

    return (
        refined_frames,
        len(translation_updates),
        float(np.mean(translation_updates)) if translation_updates else 0.0,
        float(np.mean(rotation_updates)) if rotation_updates else 0.0,
    )


def _photometric_refine_texture_poses(
    vertices: np.ndarray,
    triangles: np.ndarray,
    frames: list[PosedFrame],
    voxel_size_m: float,
    progress: Callable[..., None] | None,
) -> tuple[list[PosedFrame], int]:
    """Run bounded rigid color-map optimization and retain only small corrections."""
    captured_indices = [index for index, frame in enumerate(frames) if not frame.depthless]
    if len(captured_indices) < 3:
        return frames, 0
    try:
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64, copy=False))
        mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32, copy=False))
        mesh.compute_vertex_normals()
        rgbd_images = []
        parameters = []
        maximum_depth = 0.0
        for frame_index in captured_indices:
            frame = frames[frame_index]
            camera = frame.source.camera
            source_frame = frame.source.frames[frame.frame_index]
            color = load_color(source_frame, camera)
            depth = load_depth(source_frame, camera)
            rgbd_images.append(
                o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.ascontiguousarray(color)),
                    o3d.geometry.Image(np.ascontiguousarray(depth)),
                    depth_scale=camera.depth_scale,
                    depth_trunc=camera.max_depth_m,
                    convert_rgb_to_intensity=False,
                )
            )
            parameter = o3d.camera.PinholeCameraParameters()
            parameter.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                camera.width,
                camera.height,
                camera.fx,
                camera.fy,
                camera.cx,
                camera.cy,
            )
            world_from_camera = world_from_depth_opencv(
                frame.camera_to_global, frame.image_y_up
            )
            parameter.extrinsic = np.linalg.inv(world_from_camera)
            parameters.append(parameter)
            maximum_depth = max(maximum_depth, camera.max_depth_m)
        trajectory = o3d.camera.PinholeCameraTrajectory()
        trajectory.parameters = parameters
        options = o3d.pipelines.color_map.RigidOptimizerOption(
            maximum_iteration=18,
            maximum_allowable_depth=maximum_depth,
            depth_threshold_for_visibility_check=max(voxel_size_m * 2.0, 0.02),
            depth_threshold_for_discontinuity_check=0.08,
            half_dilation_kernel_size_for_discontinuity_map=2,
            image_boundary_margin=12,
            invisible_vertex_color_knn=0,
        )
        _, optimized = o3d.pipelines.color_map.run_rigid_optimizer(
            mesh, rgbd_images, trajectory, options
        )
    except Exception:
        return frames, 0

    refined = list(frames)
    accepted = 0
    for parameter_index, frame_index in enumerate(captured_indices):
        frame = frames[frame_index]
        initial = world_from_depth_opencv(frame.camera_to_global, frame.image_y_up)
        candidate = np.linalg.inv(
            np.asarray(optimized.parameters[parameter_index].extrinsic, dtype=np.float64)
        )
        correction = np.linalg.inv(initial) @ candidate
        translation = float(np.linalg.norm(correction[:3, 3]))
        rotation = _rotation_angle_degrees(correction)
        if (
            not np.all(np.isfinite(candidate))
            or translation > 0.012
            or rotation > 0.8
        ):
            continue
        if translation < 1e-6 and rotation < 1e-4:
            continue
        camera_to_global = (
            candidate @ np.diag([1.0, -1.0, 1.0, 1.0])
            if frame.image_y_up
            else candidate
        )
        refined[frame_index] = replace(frame, camera_to_global=camera_to_global)
        accepted += 1
    if progress and accepted:
        progress(
            "Texturing",
            f"Photometrically refined {accepted} calibrated texture cameras",
            0,
            None,
            0.545,
        )
    return refined, accepted


def _sample_spatial_bias(
    field: np.ndarray,
    uvs: np.ndarray,
    image_size: np.ndarray,
) -> np.ndarray:
    grid_height, grid_width = field.shape[:2]
    gx = np.clip(uvs[:, 0] / max(float(image_size[0] - 1), 1.0) * (grid_width - 1), 0.0, grid_width - 1)
    gy = np.clip(uvs[:, 1] / max(float(image_size[1] - 1), 1.0) * (grid_height - 1), 0.0, grid_height - 1)
    left = np.floor(gx).astype(np.int64)
    top = np.floor(gy).astype(np.int64)
    right = np.minimum(left + 1, grid_width - 1)
    bottom = np.minimum(top + 1, grid_height - 1)
    fx = (gx - left)[:, None]
    fy = (gy - top)[:, None]
    upper = field[top, left] * (1.0 - fx) + field[top, right] * fx
    lower = field[bottom, left] * (1.0 - fx) + field[bottom, right] * fx
    return upper * (1.0 - fy) + lower * fy


def _triangle_neighbors(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangle_ids = np.repeat(np.arange(len(triangles), dtype=np.int64), 3)
    edges = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]), axis=0
    )
    # Concatenation groups by edge kind, whereas triangle ids above group by
    # triangle. Tile to preserve that edge-to-face correspondence.
    triangle_ids = np.tile(np.arange(len(triangles), dtype=np.int64), 3)
    edges = np.sort(edges, axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    sorted_edges = edges[order]
    sorted_triangles = triangle_ids[order]
    shared = np.all(sorted_edges[:-1] == sorted_edges[1:], axis=1)
    first = sorted_triangles[:-1][shared]
    second = sorted_triangles[1:][shared]
    distinct = first != second
    first, second = first[distinct], second[distinct]
    shared_edges = sorted_edges[:-1][shared][distinct]
    if not len(first):
        return first, second, np.empty(0, dtype=np.float32)
    face_normals = np.cross(
        vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
        vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
    )
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-12)
    coplanar = np.clip(np.sum(face_normals[first] * face_normals[second], axis=1), 0.0, 1.0) ** 4
    edge_lengths = np.linalg.norm(
        vertices[shared_edges[:, 1]] - vertices[shared_edges[:, 0]], axis=1
    )
    median_length = max(float(np.median(edge_lengths)), 1e-8)
    weights = (coplanar * np.clip(edge_lengths / median_length, 0.25, 4.0)).astype(np.float32)
    return first, second, weights


def _label_switch_percent(labels: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    valid = (labels[first] >= 0) & (labels[second] >= 0)
    if not np.any(valid):
        return 0.0
    return 100.0 * float(np.mean(labels[first[valid]] != labels[second[valid]]))


def _coherent_triangle_labels(
    scores: np.ndarray,
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    best_scores = np.max(scores, axis=0)
    labels = np.argmax(scores, axis=0).astype(np.int16)
    labels[best_scores <= 0.0] = -1
    first, second, edge_weights = _triangle_neighbors(vertices, triangles)
    before = _label_switch_percent(labels, first, second)
    if not len(first) or scores.shape[0] < 2:
        return labels, before, before
    normalized = np.divide(
        scores,
        best_scores[None, :],
        out=np.zeros_like(scores),
        where=best_scores[None, :] > 0.0,
    )
    degree = np.zeros(len(triangles), dtype=np.float32)
    np.add.at(degree, first, edge_weights)
    np.add.at(degree, second, edge_weights)
    degree = np.maximum(degree, 1e-6)

    triangle_parity = np.arange(len(triangles), dtype=np.int64) & 1
    for _ in range(LABEL_OPTIMIZATION_PASSES):
        changed = 0
        # Red/black ICM avoids the whole checkerboard swapping labels in lockstep.
        for parity in (0, 1):
            active = triangle_parity == parity
            best_objective = np.full(len(triangles), -np.inf, dtype=np.float32)
            next_labels = labels.copy()
            for frame_index in range(scores.shape[0]):
                support = np.zeros(len(triangles), dtype=np.float32)
                np.add.at(support, first, edge_weights * (labels[second] == frame_index))
                np.add.at(support, second, edge_weights * (labels[first] == frame_index))
                objective = normalized[frame_index] + LABEL_SMOOTHNESS * support / degree
                objective[scores[frame_index] <= 0.0] = -np.inf
                better = active & (objective > best_objective)
                best_objective[better] = objective[better]
                next_labels[better] = frame_index
            changed += int(np.count_nonzero(next_labels[active] != labels[active]))
            labels[active] = next_labels[active]
        if changed == 0:
            break
    return labels, before, _label_switch_percent(labels, first, second)


def _sample_surface_colors(
    vertices: np.ndarray,
    normals: np.ndarray,
    triangles: np.ndarray,
    frames: list[PosedFrame],
    voxel_size_m: float,
    calibration: TextureCalibration,
    progress: Callable[..., None] | None,
) -> tuple[np.ndarray, float, float, np.ndarray, float, float, float]:
    # Vertex colors remain a seamless fallback for boundary geometry. The atlas
    # itself keeps one sharp source image per coherent patch.
    top_weights = np.zeros((len(vertices), 4), dtype=np.float32)
    top_colors = np.zeros((len(vertices), 4, 3), dtype=np.float32)
    fallback_weights = np.zeros(len(vertices), dtype=np.float32)
    fallback_colors = np.zeros((len(vertices), 3), dtype=np.float32)
    triangle_scores = np.zeros((len(frames), len(triangles)), dtype=np.float32)
    loose_triangle_scores = np.zeros_like(triangle_scores)
    all_indices = np.arange(len(vertices), dtype=np.int64)

    for frame_index, frame in enumerate(frames):
        image = _load_texture_image(frame)
        sampled, view_weights, loose_view_weights, uvs = _frame_observations(
            vertices,
            normals,
            frame,
            voxel_size_m,
            all_indices,
            image=image,
        )
        sampled_valid = loose_view_weights > 0.0
        if np.any(sampled_valid):
            sampled[sampled_valid] = (
                sampled[sampled_valid] * calibration.gains[frame_index]
                + calibration.biases[frame_index]
                + _sample_spatial_bias(
                    calibration.spatial_biases[frame_index],
                    uvs[sampled_valid],
                    np.asarray([image.shape[1], image.shape[0]]),
                )
            ).clip(0.0, 1.0)
            fallback_globals = np.flatnonzero(sampled_valid)
            better_fallback = loose_view_weights[fallback_globals] > fallback_weights[fallback_globals]
            replacement = fallback_globals[better_fallback]
            fallback_weights[replacement] = loose_view_weights[replacement]
            fallback_colors[replacement] = sampled[replacement]
        accepted = view_weights > 0.0
        if np.any(accepted):
            global_indices = np.flatnonzero(accepted)
            weakest_slots = np.argmin(top_weights[global_indices], axis=1)
            weakest_weights = top_weights[global_indices, weakest_slots]
            stronger = view_weights[global_indices] > weakest_weights
            replacement_indices = global_indices[stronger]
            replacement_slots = weakest_slots[stronger]
            top_weights[replacement_indices, replacement_slots] = view_weights[replacement_indices]
            top_colors[replacement_indices, replacement_slots] = sampled[replacement_indices]
        triangle_scores[frame_index] = np.min(view_weights[triangles], axis=1)
        loose_triangle_scores[frame_index] = np.min(loose_view_weights[triangles], axis=1)
        if progress:
            progress(
                "Texturing",
                f"Scored calibrated RGB view {frame_index + 1} of {len(frames)}",
                0,
                None,
                0.58 + 0.12 * (frame_index + 1) / len(frames),
            )

    weight_sum = top_weights.sum(axis=1, dtype=np.float64)
    color_sum = np.sum(top_colors * top_weights[..., None], axis=1, dtype=np.float64)
    colors = np.full((len(vertices), 3), 0.21586, dtype=np.float64)
    textured = weight_sum > 1e-10
    colors[textured] = color_sum[textured] / weight_sum[textured, None]
    direct_coverage = 100.0 * float(np.mean(textured)) if len(textured) else 0.0
    fallback = (~textured) & (fallback_weights > 1e-5)
    colors[fallback] = fallback_colors[fallback]
    textured[fallback] = True

    edges = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]), axis=0
    )
    for _ in range(16):
        missing = ~textured
        if not np.any(missing):
            break
        neighbor_sum = np.zeros_like(colors)
        neighbor_count = np.zeros(len(vertices), dtype=np.int32)
        first, second = edges[:, 0], edges[:, 1]
        second_textured = textured[second]
        first_textured = textured[first]
        np.add.at(neighbor_sum, first[second_textured], colors[second[second_textured]])
        np.add.at(neighbor_count, first[second_textured], 1)
        np.add.at(neighbor_sum, second[first_textured], colors[first[first_textured]])
        np.add.at(neighbor_count, second[first_textured], 1)
        fill = missing & (neighbor_count > 0)
        if not np.any(fill):
            break
        colors[fill] = neighbor_sum[fill] / neighbor_count[fill, None]
        textured[fill] = True
    final_coverage = 100.0 * float(np.mean(textured)) if len(textured) else 0.0
    labels, switch_before, switch_after = _coherent_triangle_labels(
        triangle_scores, vertices, triangles
    )
    direct_triangle_coverage = 100.0 * float(np.mean(labels >= 0)) if len(labels) else 0.0
    missing_labels = labels < 0
    if np.any(missing_labels):
        loose_best = np.max(loose_triangle_scores[:, missing_labels], axis=0)
        loose_labels = np.argmax(loose_triangle_scores[:, missing_labels], axis=0).astype(np.int16)
        loose_labels[loose_best <= 1e-8] = -1
        labels[missing_labels] = loose_labels
    return (
        _linear_to_srgb(colors),
        direct_coverage,
        final_coverage,
        labels,
        direct_triangle_coverage,
        switch_before,
        switch_after,
    )


def _bake_triangle_atlas(
    vertex_colors: np.ndarray,
    triangles: np.ndarray,
    *,
    vertices: np.ndarray | None = None,
    frames: list[PosedFrame] | None = None,
    triangle_frames: np.ndarray | None = None,
    exposure_gains: np.ndarray | None = None,
    progress: Callable[..., None] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    triangle_count = len(triangles)
    columns = max(1, math.ceil(math.sqrt(triangle_count)))
    rows = math.ceil(triangle_count / columns)
    cell_size = min(
        MAX_CHART_SIZE,
        MAX_ATLAS_SIZE // columns,
        MAX_ATLAS_SIZE // rows,
    )
    if cell_size < CHART_PADDING * 2 + 2:
        raise RuntimeError("The fused mesh has too many triangles for the texture atlas")
    width = columns * cell_size
    height = rows * cell_size
    atlas = np.full((height, width, 3), 24, dtype=np.uint8)
    triangle_colors = vertex_colors[triangles].astype(np.float32)
    triangle_ids = np.arange(triangle_count, dtype=np.int64)
    cell_x = (triangle_ids % columns) * cell_size
    cell_y = (triangle_ids // columns) * cell_size
    denominator = max(cell_size - CHART_PADDING * 2 - 1, 1)

    # Start with a seamless vertex-color fallback for boundary triangles that
    # no camera sees completely.  Direct native-RGB samples overwrite it below.
    for pixel_y in range(cell_size):
        for pixel_x in range(cell_size):
            bary_x = float(np.clip((pixel_x - CHART_PADDING) / denominator, 0.0, 1.0))
            bary_y = float(np.clip((pixel_y - CHART_PADDING) / denominator, 0.0, 1.0))
            if bary_x + bary_y > 1.0:
                projected_x = np.clip((bary_x - bary_y + 1.0) * 0.5, 0.0, 1.0)
                projected_y = np.clip((bary_y - bary_x + 1.0) * 0.5, 0.0, 1.0)
                bary_x, bary_y = float(projected_x), float(projected_y)
            bary_origin = 1.0 - bary_x - bary_y
            colors = (
                triangle_colors[:, 0] * bary_origin
                + triangle_colors[:, 1] * bary_x
                + triangle_colors[:, 2] * bary_y
            )
            atlas[cell_y + pixel_y, cell_x + pixel_x] = np.rint(colors).clip(0, 255).astype(np.uint8)

    direct_inputs = (vertices, frames, triangle_frames, exposure_gains)
    if any(value is not None for value in direct_inputs):
        if any(value is None for value in direct_inputs):
            raise ValueError("Direct atlas baking requires vertices, frames, triangle_frames, and exposure_gains")
        assert vertices is not None
        assert frames is not None
        assert triangle_frames is not None
        assert exposure_gains is not None
        if len(triangle_frames) != triangle_count:
            raise ValueError("triangle_frames must contain one source view per triangle")

        # A fused mesh must not turn native imagery into three interpolated
        # colors per face.  Each face instead uses its strongest common,
        # depth-validated view, and every atlas texel is projected back into
        # that source image.  This retains fine detail without multi-view blur.
        for frame_index, frame in enumerate(frames):
            frame_triangles = np.flatnonzero(triangle_frames == frame_index)
            if len(frame_triangles):
                image = _load_texture_image(frame)
                phase = frame.source
                source_frame = phase.frames[frame.frame_index]
                rgb_camera = frame_rgb_camera(source_frame, phase)
                rgb_from_depth = frame_rgb_from_depth(source_frame, phase)
                world_from_camera = world_from_depth_opencv(
                    frame.camera_to_global,
                    frame.image_y_up,
                )
                camera_from_world = np.linalg.inv(world_from_camera)
                source_triangles = vertices[triangles[frame_triangles]].astype(
                    np.float64,
                    copy=False,
                )
                frame_cell_x = cell_x[frame_triangles]
                frame_cell_y = cell_y[frame_triangles]
                gain = float(exposure_gains[frame_index])
                for pixel_y in range(cell_size):
                    for pixel_x in range(cell_size):
                        bary_x = float(
                            np.clip((pixel_x - CHART_PADDING) / denominator, 0.0, 1.0)
                        )
                        bary_y = float(
                            np.clip((pixel_y - CHART_PADDING) / denominator, 0.0, 1.0)
                        )
                        if bary_x + bary_y > 1.0:
                            projected_x = np.clip((bary_x - bary_y + 1.0) * 0.5, 0.0, 1.0)
                            projected_y = np.clip((bary_y - bary_x + 1.0) * 0.5, 0.0, 1.0)
                            bary_x, bary_y = float(projected_x), float(projected_y)
                        points = (
                            source_triangles[:, 0] * (1.0 - bary_x - bary_y)
                            + source_triangles[:, 1] * bary_x
                            + source_triangles[:, 2] * bary_y
                        )
                        camera_points = (
                            points @ camera_from_world[:3, :3].T
                            + camera_from_world[:3, 3]
                        )
                        rgb_points = (
                            camera_points @ rgb_from_depth[:3, :3].T
                            + rgb_from_depth[:3, 3]
                        )
                        rgb_u, rgb_v, rgb_z = project_rgb(rgb_points, rgb_camera)
                        valid = (
                            (rgb_z > 0.0)
                            & (rgb_u >= 0.0)
                            & (rgb_u <= rgb_camera.width - 1.001)
                            & (rgb_v >= 0.0)
                            & (rgb_v <= rgb_camera.height - 1.001)
                        )
                        if np.any(valid):
                            sampled = np.clip(
                                _bilinear_rgb(image, rgb_u[valid], rgb_v[valid]) * gain,
                                0.0,
                                255.0,
                            )
                            atlas[
                                frame_cell_y[valid] + pixel_y,
                                frame_cell_x[valid] + pixel_x,
                            ] = np.rint(sampled).astype(np.uint8)
            if progress:
                progress(
                    "Texturing",
                    f"Baked calibrated RGB detail from keyframe {frame_index + 1} of {len(frames)}",
                    0,
                    None,
                    0.75 + 0.20 * (frame_index + 1) / len(frames),
                )

    left = cell_x + CHART_PADDING + 0.5
    top = cell_y + CHART_PADDING + 0.5
    right = cell_x + cell_size - CHART_PADDING - 0.5
    bottom = cell_y + cell_size - CHART_PADDING - 0.5
    uvs = np.empty((triangle_count, 3, 2), dtype=np.float32)
    uvs[:, 0, 0] = left / width
    uvs[:, 0, 1] = 1.0 - top / height
    uvs[:, 1, 0] = right / width
    uvs[:, 1, 1] = 1.0 - top / height
    uvs[:, 2, 0] = left / width
    uvs[:, 2, 1] = 1.0 - bottom / height
    return atlas, uvs, cell_size


def _apply_frame_calibration(
    image: np.ndarray,
    calibration: TextureCalibration,
    frame_index: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    output = np.empty_like(image)
    field = calibration.spatial_biases[frame_index]
    grid_height, grid_width = field.shape[:2]
    gx = np.linspace(0.0, grid_width - 1, width, dtype=np.float32)
    left = np.floor(gx).astype(np.int64)
    right = np.minimum(left + 1, grid_width - 1)
    fx = (gx - left)[None, :, None]
    for start in range(0, height, 128):
        stop = min(start + 128, height)
        gy = np.linspace(
            start / max(height - 1, 1) * (grid_height - 1),
            (stop - 1) / max(height - 1, 1) * (grid_height - 1),
            stop - start,
            dtype=np.float32,
        )
        top = np.floor(gy).astype(np.int64)
        bottom = np.minimum(top + 1, grid_height - 1)
        fy = (gy - top)[:, None, None]
        upper = field[top[:, None], left[None, :]] * (1.0 - fx) + field[
            top[:, None], right[None, :]
        ] * fx
        lower = field[bottom[:, None], left[None, :]] * (1.0 - fx) + field[
            bottom[:, None], right[None, :]
        ] * fx
        spatial = upper * (1.0 - fy) + lower * fy
        linear = _SRGB_LINEAR_LUT[image[start:stop]]
        linear = (
            linear * calibration.gains[frame_index]
            + calibration.biases[frame_index]
            + spatial
        )
        output[start:stop] = _linear_to_srgb(linear)
    return output


def _best_page_grid(images: list[np.ndarray], page_count: int) -> tuple[int, int, int, int]:
    best: tuple[float, int, int, int, int] | None = None
    for columns in range(1, page_count + 1):
        rows = math.ceil(page_count / columns)
        cell_width = MAX_ATLAS_SIZE // columns
        cell_height = MAX_ATLAS_SIZE // rows
        if cell_width <= ATLAS_PAGE_PADDING * 2 or cell_height <= ATLAS_PAGE_PADDING * 2:
            continue
        scales = [
            min(
                (cell_width - ATLAS_PAGE_PADDING * 2) / image.shape[1],
                (cell_height - ATLAS_PAGE_PADDING * 2) / image.shape[0],
                1.0,
            )
            for image in images
        ]
        score = min(scales) * 0.7 + float(np.mean(scales)) * 0.3
        candidate = (score, columns, rows, cell_width, cell_height)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("Texture views cannot fit into the configured atlas")
    return best[1], best[2], best[3], best[4]


def _bake_shared_view_atlas(
    vertex_colors: np.ndarray,
    vertices: np.ndarray,
    triangles: np.ndarray,
    frames: list[PosedFrame],
    triangle_frames: np.ndarray,
    calibration: TextureCalibration,
    progress: Callable[..., None] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pack corrected source images once and share their pixels across face patches."""
    images = [_load_texture_image(frame) for frame in frames]
    missing_triangles = np.flatnonzero(triangle_frames < 0)
    if len(missing_triangles) == len(triangles):
        return _bake_triangle_atlas(vertex_colors, triangles)
    page_count = len(images) + (1 if len(missing_triangles) else 0)
    layout_images = images + (
        [np.empty((1024, 1024, 3), dtype=np.uint8)] if len(missing_triangles) else []
    )
    columns, rows, cell_width, cell_height = _best_page_grid(layout_images, page_count)
    width = columns * cell_width
    height = rows * cell_height
    atlas = np.full((height, width, 3), 24, dtype=np.uint8)
    uvs = np.zeros((len(triangles), 3, 2), dtype=np.float32)
    effective_page_resolutions: list[int] = []

    for frame_index, (frame, image) in enumerate(zip(frames, images, strict=True)):
        column = frame_index % columns
        row = frame_index // columns
        scale = min(
            (cell_width - ATLAS_PAGE_PADDING * 2) / image.shape[1],
            (cell_height - ATLAS_PAGE_PADDING * 2) / image.shape[0],
            1.0,
        )
        target_width = max(2, round(image.shape[1] * scale))
        target_height = max(2, round(image.shape[0] * scale))
        effective_page_resolutions.append(max(target_width, target_height))
        from PIL import Image

        if (target_width, target_height) != (image.shape[1], image.shape[0]):
            image = np.asarray(
                Image.fromarray(image).resize(
                    (target_width, target_height), Image.Resampling.LANCZOS
                ),
                dtype=np.uint8,
            )
        corrected = _apply_frame_calibration(image, calibration, frame_index)
        origin_x = column * cell_width + (cell_width - target_width) // 2
        origin_y = row * cell_height + (cell_height - target_height) // 2
        padding = min(
            ATLAS_PAGE_PADDING,
            origin_x - column * cell_width,
            origin_y - row * cell_height,
            (column + 1) * cell_width - origin_x - target_width,
            (row + 1) * cell_height - origin_y - target_height,
        )
        padding = max(0, int(padding))
        if padding:
            padded = np.pad(corrected, ((padding, padding), (padding, padding), (0, 0)), mode="edge")
            atlas[
                origin_y - padding : origin_y + target_height + padding,
                origin_x - padding : origin_x + target_width + padding,
            ] = padded
        else:
            atlas[origin_y : origin_y + target_height, origin_x : origin_x + target_width] = corrected

        frame_triangles = np.flatnonzero(triangle_frames == frame_index)
        if len(frame_triangles):
            world_from_camera = world_from_depth_opencv(frame.camera_to_global, frame.image_y_up)
            camera_from_world = np.linalg.inv(world_from_camera)
            source_points = vertices[triangles[frame_triangles]].reshape(-1, 3).astype(
                np.float64, copy=False
            )
            camera_points = (
                source_points @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
            )
            rgb_from_depth = _texture_rgb_from_depth(frame)
            rgb_points = camera_points @ rgb_from_depth[:3, :3].T + rgb_from_depth[:3, 3]
            projected_u, projected_v, _ = project_rgb(rgb_points, _texture_camera(frame))
            projected_u = np.clip(projected_u, 0.0, images[frame_index].shape[1] - 1.0)
            projected_v = np.clip(projected_v, 0.0, images[frame_index].shape[0] - 1.0)
            atlas_x = origin_x + projected_u * (target_width - 1) / max(images[frame_index].shape[1] - 1, 1)
            atlas_y = origin_y + projected_v * (target_height - 1) / max(images[frame_index].shape[0] - 1, 1)
            frame_uvs = np.column_stack(
                ((atlas_x + 0.5) / width, 1.0 - (atlas_y + 0.5) / height)
            ).reshape(-1, 3, 2)
            uvs[frame_triangles] = frame_uvs.astype(np.float32)
        if progress:
            progress(
                "Texturing",
                f"Packed sharp corrected view {frame_index + 1} of {len(frames)}",
                0,
                None,
                0.72 + 0.25 * (frame_index + 1) / len(frames),
            )

    if len(missing_triangles):
        fallback_page = len(frames)
        column = fallback_page % columns
        row = fallback_page // columns
        page_x = column * cell_width + ATLAS_PAGE_PADDING
        page_y = row * cell_height + ATLAS_PAGE_PADDING
        available_width = cell_width - ATLAS_PAGE_PADDING * 2
        available_height = cell_height - ATLAS_PAGE_PADDING * 2
        chart_columns = max(1, math.ceil(math.sqrt(len(missing_triangles))))
        chart_rows = math.ceil(len(missing_triangles) / chart_columns)
        chart_size = min(
            MAX_CHART_SIZE,
            available_width // chart_columns,
            available_height // chart_rows,
        )
        fallback_colors = vertex_colors[triangles[missing_triangles]].astype(np.float32)
        if chart_size < CHART_PADDING * 2 + 2:
            # A large open-boundary mesh may have hundreds of thousands of
            # unobserved faces. They carry no source-image detail to preserve,
            # so use one average-color texel per face instead of failing the
            # entire sharp-view atlas for lack of padded fallback charts.
            chart_columns = min(
                available_width,
                max(1, math.ceil(math.sqrt(len(missing_triangles)))),
            )
            chart_rows = math.ceil(len(missing_triangles) / chart_columns)
            if chart_rows > available_height:
                raise RuntimeError("Unobserved mesh faces do not fit into the texture fallback page")
            local_ids = np.arange(len(missing_triangles), dtype=np.int64)
            texel_x = page_x + local_ids % chart_columns
            texel_y = page_y + local_ids // chart_columns
            atlas[texel_y, texel_x] = np.rint(
                np.mean(fallback_colors, axis=1)
            ).clip(0, 255).astype(np.uint8)
            centers = np.column_stack(
                (
                    (texel_x + 0.5) / width,
                    1.0 - (texel_y + 0.5) / height,
                )
            ).astype(np.float32)
            uvs[missing_triangles] = np.repeat(centers[:, None, :], 3, axis=1)
            return (
                atlas,
                uvs,
                min(effective_page_resolutions) if effective_page_resolutions else 0,
            )
        local_ids = np.arange(len(missing_triangles), dtype=np.int64)
        cell_x = page_x + (local_ids % chart_columns) * chart_size
        cell_y = page_y + (local_ids // chart_columns) * chart_size
        denominator = max(chart_size - CHART_PADDING * 2 - 1, 1)
        for pixel_y in range(chart_size):
            for pixel_x in range(chart_size):
                bary_x = float(np.clip((pixel_x - CHART_PADDING) / denominator, 0.0, 1.0))
                bary_y = float(np.clip((pixel_y - CHART_PADDING) / denominator, 0.0, 1.0))
                if bary_x + bary_y > 1.0:
                    bary_x, bary_y = (
                        float(np.clip((bary_x - bary_y + 1.0) * 0.5, 0.0, 1.0)),
                        float(np.clip((bary_y - bary_x + 1.0) * 0.5, 0.0, 1.0)),
                    )
                colors = (
                    fallback_colors[:, 0] * (1.0 - bary_x - bary_y)
                    + fallback_colors[:, 1] * bary_x
                    + fallback_colors[:, 2] * bary_y
                )
                atlas[cell_y + pixel_y, cell_x + pixel_x] = np.rint(colors).clip(0, 255).astype(np.uint8)
        left = cell_x + CHART_PADDING + 0.5
        top = cell_y + CHART_PADDING + 0.5
        right = cell_x + chart_size - CHART_PADDING - 0.5
        bottom = cell_y + chart_size - CHART_PADDING - 0.5
        fallback_uvs = np.empty((len(missing_triangles), 3, 2), dtype=np.float32)
        fallback_uvs[:, 0] = np.column_stack((left / width, 1.0 - top / height))
        fallback_uvs[:, 1] = np.column_stack((right / width, 1.0 - top / height))
        fallback_uvs[:, 2] = np.column_stack((left / width, 1.0 - bottom / height))
        uvs[missing_triangles] = fallback_uvs

    return atlas, uvs, min(effective_page_resolutions) if effective_page_resolutions else 0


def _write_mesh(
    output_dir: Path,
    vertices: np.ndarray,
    normals: np.ndarray,
    triangles: np.ndarray,
    uvs: np.ndarray,
) -> int:
    with (output_dir / "room-mesh.obj").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "# ScanLan fused RGB-D mesh\n"
            "mtllib room-mesh.mtl\n"
            "o room_mesh\n"
            "usemtl room_rgb\n"
        )
        lines: list[str] = []
        for point in vertices:
            lines.append(f"v {point[0]:.7g} {point[1]:.7g} {point[2]:.7g}\n")
            if len(lines) >= 8192:
                _append_lines(handle, lines)
        _append_lines(handle, lines)
        for normal in normals:
            lines.append(f"vn {normal[0]:.7g} {normal[1]:.7g} {normal[2]:.7g}\n")
            if len(lines) >= 8192:
                _append_lines(handle, lines)
        _append_lines(handle, lines)
        for texture_u, texture_v in uvs.reshape(-1, 2):
            lines.append(f"vt {texture_u:.7g} {texture_v:.7g}\n")
            if len(lines) >= 8192:
                _append_lines(handle, lines)
        _append_lines(handle, lines)
        for triangle_index, (first, second, third) in enumerate(triangles):
            uv_offset = triangle_index * 3 + 1
            first += 1
            second += 1
            third += 1
            lines.append(
                f"f {first}/{uv_offset}/{first} "
                f"{second}/{uv_offset + 1}/{second} "
                f"{third}/{uv_offset + 2}/{third}\n"
            )
            if len(lines) >= 8192:
                _append_lines(handle, lines)
        _append_lines(handle, lines)

    preview_path = output_dir / "room-mesh.preview.bin"
    preview_temporary = output_dir / "room-mesh.preview.bin.tmp"
    render_positions = np.asarray(vertices[triangles].reshape(-1, 3), dtype="<f4")
    render_uvs = np.asarray(uvs.reshape(-1, 2), dtype="<f4")
    render_indices = np.arange(len(render_positions), dtype="<u4")
    with preview_temporary.open("wb") as preview:
        preview.write(b"K2M1")
        preview.write(struct.pack("<II", len(render_positions), len(render_indices)))
        preview.write(render_positions.tobytes())
        preview.write(render_uvs.tobytes())
        preview.write(render_indices.tobytes())
    preview_temporary.replace(preview_path)
    return len(render_positions)


def _mesh_cache_path(
    output_dir: Path,
    frames: list[PosedFrame],
    voxel_size_m: float,
) -> Path:
    digest = hashlib.sha256()
    digest.update(MESH_CACHE_VERSION.encode("ascii"))
    digest.update(np.asarray([voxel_size_m], dtype="<f8").tobytes())
    seen_phases: set[Path] = set()
    for frame in frames:
        phase = frame.source
        if phase.root not in seen_phases:
            for name in ("phase.json", "frames.csv"):
                path = phase.root / name
                stat = path.stat()
                digest.update(str(path).encode("utf-8"))
                digest.update(np.asarray([stat.st_size, stat.st_mtime_ns], dtype="<i8").tobytes())
            seen_phases.add(phase.root)
        record = phase.frames[frame.frame_index]
        stat = record.depth_path.stat()
        digest.update(frame.phase_id.encode("utf-8"))
        digest.update(np.asarray([frame.frame_index, stat.st_size, stat.st_mtime_ns], dtype="<i8").tobytes())
        digest.update(np.asarray(frame.camera_to_global, dtype="<f8").tobytes())
        digest.update(bytes((int(frame.image_y_up),)))
    key = digest.hexdigest()[:24]
    return output_dir / "cache" / "meshes" / f"{key}.npz"


def _read_mesh_cache(path: Path) -> tuple[np.ndarray, np.ndarray, str] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            vertices = np.asarray(payload["vertices"], dtype=np.float32)
            triangles = np.asarray(payload["triangles"], dtype=np.int64)
            method = str(payload["method"].item())
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or triangles.ndim != 2
            or triangles.shape[1] != 3
            or not len(triangles)
            or int(triangles.max()) >= len(vertices)
        ):
            return None
        return vertices, triangles, method
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _write_mesh_cache(
    path: Path,
    vertices: np.ndarray,
    triangles: np.ndarray,
    method: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            vertices=np.asarray(vertices, dtype=np.float32),
            triangles=np.asarray(triangles, dtype=np.int64),
            method=np.asarray(method),
        )
    temporary.replace(path)


def build_mesh_artifacts(
    output_dir: Path,
    frames: list[PosedFrame],
    progress: Callable[..., None] | None = None,
    voxel_size_m: float = 0.015,
    prebuilt_mesh: Any | None = None,
    prebuilt_mesh_method: str | None = None,
    repair_settings: MeshRepairSettings | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not frames:
        write_json(output_dir / "camera-poses.json", [])
        return {"cameraFrameCount": 0, "meshVertexCount": 0, "meshTriangleCount": 0}

    if progress:
        progress(
            "Meshing",
            f"Preparing one continuous surface from all {len(frames)} accepted depth keyframes",
            0,
            None,
            0.0,
        )
    mesh_voxel_size = max(float(voxel_size_m), MIN_MESH_VOXEL_SIZE)
    cache_path = _mesh_cache_path(output_dir, frames, mesh_voxel_size)
    cached_mesh = _read_mesh_cache(cache_path)
    mesh_cache_hit = cached_mesh is not None
    if cached_mesh is None:
        if prebuilt_mesh is not None:
            try:
                import open3d as o3d

                if progress:
                    progress(
                        "Meshing",
                        "Reusing the final reconstruction TSDF instead of fusing depth twice",
                        0,
                        None,
                        0.28,
                    )
                vertices, triangles = _prepare_fused_mesh(
                    o3d,
                    prebuilt_mesh,
                    mesh_voxel_size,
                    progress,
                )
                fusion_method = prebuilt_mesh_method or "shared_tsdf"
            except Exception as error:
                if progress:
                    progress(
                        "Meshing",
                        f"Shared TSDF surface unavailable; rebuilding geometry · {str(error).splitlines()[0]}",
                        0,
                        None,
                        0.0,
                    )
                vertices, triangles, fusion_method = _fused_mesh(
                    frames,
                    voxel_size_m,
                    progress,
                )
        else:
            vertices, triangles, fusion_method = _fused_mesh(
                frames,
                voxel_size_m,
                progress,
            )
        _write_mesh_cache(cache_path, vertices, triangles, fusion_method)
    else:
        vertices, triangles, fusion_method = cached_mesh
        if progress:
            progress(
                "Meshing",
                f"Reused cached fused surface with {len(triangles):,} triangles",
                0,
                None,
                0.58,
            )
    repair_settings = repair_settings or MeshRepairSettings()
    vertices, triangles, repair_report = repair_mesh_geometry(
        output_dir,
        vertices,
        triangles,
        frames,
        mesh_voxel_size,
        repair_settings,
        progress,
    )
    repair_summary = repair_report.get("repairSummary", {})
    texture_progress = progress
    if progress:
        def texture_progress(
            stage: str,
            detail: str,
            advance: int = 0,
            point_count: int | None = None,
            stage_progress: float | None = None,
            *extra: Any,
        ) -> None:
            mapped_progress = stage_progress
            if stage_progress is not None:
                mapped_progress = 0.72 + 0.27 * min(
                    1.0, max(0.0, (stage_progress - 0.50) / 0.47)
                )
            progress(
                stage,
                detail,
                advance,
                point_count,
                mapped_progress,
                *extra,
            )
    normals = _vertex_normals(vertices, triangles)
    supplemental_frames = _load_supplemental_texture_frames(output_dir.parent, frames[0])
    texture_candidates = [*frames, *supplemental_frames]
    texture_frames = _select_texture_frames_for_mesh(texture_candidates, vertices, normals)
    texture_frames, refined_pose_count, mean_pose_translation, mean_pose_rotation = (
        _refine_texture_poses(
            vertices,
            normals,
            texture_frames,
            mesh_voxel_size,
            texture_progress,
        )
    )
    texture_frames, photometric_pose_count = _photometric_refine_texture_poses(
        vertices,
        triangles,
        texture_frames,
        mesh_voxel_size,
        texture_progress,
    )
    if texture_progress:
        texture_progress(
            "Texturing",
            f"Calibrating color across {len(texture_frames)} overlapping texture views",
            0,
            None,
            0.55,
        )
    calibration_samples = _collect_calibration_samples(
        vertices,
        normals,
        texture_frames,
        mesh_voxel_size,
    )
    calibration = _estimate_texture_calibration(calibration_samples)
    selected_keys = {(frame.phase_id, frame.frame_index) for frame in texture_frames}
    all_camera_frames = [*frames, *supplemental_frames]
    refined_by_key = {
        (frame.phase_id, frame.frame_index): frame for frame in texture_frames
    }
    write_json(
        output_dir / "camera-poses.json",
        [
            _camera_payload(
                refined_by_key.get((frame.phase_id, frame.frame_index), frame),
                (frame.phase_id, frame.frame_index) in selected_keys,
            )
            for frame in all_camera_frames
        ],
    )
    (
        vertex_colors,
        direct_texture_coverage,
        texture_coverage,
        triangle_frames,
        direct_triangle_coverage,
        switch_before,
        switch_after,
    ) = _sample_surface_colors(
        vertices,
        normals,
        triangles,
        texture_frames,
        mesh_voxel_size,
        calibration,
        texture_progress,
    )
    atlas, uvs, page_resolution = _bake_shared_view_atlas(
        vertex_colors,
        vertices,
        triangles,
        texture_frames,
        triangle_frames,
        calibration,
        texture_progress,
    )
    display_axes = np.asarray(texture_frames[0].display_axes, dtype=np.float64)
    display_vertices = (vertices * display_axes).astype(np.float32)
    display_triangles = triangles.copy()
    display_uvs = uvs
    if np.prod(display_axes) < 0.0:
        display_triangles = display_triangles[:, [0, 2, 1]]
        display_uvs = uvs[:, [0, 2, 1]]
    display_normals = _vertex_normals(display_vertices, display_triangles)
    texture_path = output_dir / "room-texture.png"
    _write_png(texture_path, atlas)
    (output_dir / "room-mesh.mtl").write_text(
        "# ScanLan fused multi-view RGB material\n"
        "newmtl room_rgb\n"
        "Ka 1.0 1.0 1.0\n"
        "Kd 1.0 1.0 1.0\n"
        "Ks 0.0 0.0 0.0\n"
        "d 1.0\n"
        "illum 1\n"
        "map_Kd room-texture.png\n",
        encoding="utf-8",
        newline="\n",
    )
    render_vertex_count = _write_mesh(
        output_dir,
        display_vertices,
        display_normals,
        display_triangles,
        display_uvs,
    )
    vertex_count = len(display_vertices)
    triangle_count = len(display_triangles)
    supplemental_manifest_path = output_dir.parent / "supplemental-photos.json"
    supplemental_fingerprint = (
        hashlib.sha256(supplemental_manifest_path.read_bytes()).hexdigest()[:24]
        if supplemental_manifest_path.is_file()
        else "none"
    )
    if progress:
        progress(
            "Meshing",
            f"Fused textured mesh contains {triangle_count:,} triangles",
            4,
            None,
            1.0,
        )
    return {
        "cameraFrameCount": len(frames),
        "textureFrameCount": len(texture_frames),
        "meshVertexCount": vertex_count,
        "meshRenderVertexCount": render_vertex_count,
        "meshTextureVertexCount": triangle_count * 3,
        "meshTriangleCount": triangle_count,
        "meshOutputPath": "outputs/room-mesh.obj",
        "meshMaterialPath": "outputs/room-mesh.mtl",
        "meshTexturePath": "outputs/room-texture.png",
        "meshFusionMethod": fusion_method,
        "meshCacheHit": mesh_cache_hit,
        "meshRepairEnabled": repair_settings.enabled,
        "meshRepairProfile": repair_settings.profile,
        "meshRepairStatus": repair_report.get("status", "unknown"),
        "meshRepairReportPath": "outputs/mesh-repair-report.json",
        "meshRepairFallback": bool(repair_summary.get("fallbackOccurred", False)),
        "meshRepairDefectsFixed": int(
            repair_summary.get("topologyDefectsFixed", 0)
        ),
        "meshRepairHolesFilled": int(repair_summary.get("holesFilled", 0)),
        "meshRepairOpeningsPreserved": int(
            repair_summary.get("openingsPreserved", 0)
        ),
        "meshRepairUnknownPreserved": int(
            repair_summary.get("unknownBoundariesPreserved", 0)
        ),
        "meshRepairCacheHit": bool(repair_report.get("repairedCacheHit", False)),
        "meshRepairFingerprint": str(
            repair_report.get("repairCacheFingerprint", "")
        ),
        "watertightMeshOutputPath": (
            repair_report.get("watertightCopy", {}).get("path")
            if repair_report.get("watertightCopy", {}).get("status") == "ok"
            else None
        ),
        "textureCandidateFrameCount": len(texture_candidates),
        "supplementalTextureFrameCount": len(supplemental_frames),
        "supplementalTextureFingerprint": supplemental_fingerprint,
        "textureSource": "coherent_best_view_shared_image_atlas",
        "textureBlend": "single_source_patches_linear_rgb_overlap_and_spatial_seam_corrected",
        "textureDirectCoveragePercent": round(direct_texture_coverage, 2),
        "textureCoveragePercent": round(texture_coverage, 2),
        "textureDirectTriangleCoveragePercent": round(direct_triangle_coverage, 2),
        "textureLabelSwitchPercentBefore": round(switch_before, 2),
        "textureLabelSwitchPercentAfter": round(switch_after, 2),
        "textureCalibrationReferenceFrame": calibration.reference_frame,
        "textureCalibrationOverlapEdges": calibration.overlap_edge_count,
        "textureCalibrationSampleCount": calibration.sample_count,
        "texturePoseRefinedFrameCount": refined_pose_count,
        "texturePhotometricPoseRefinedFrameCount": photometric_pose_count,
        "texturePoseMeanTranslationMm": round(mean_pose_translation * 1000.0, 3),
        "texturePoseMeanRotationDegrees": round(mean_pose_rotation, 4),
        "texturePageResolution": page_resolution,
        "textureAtlasSize": int(max(atlas.shape[:2])),
    }
