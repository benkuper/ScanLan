from __future__ import annotations

import hashlib
import math
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .calibration import project_rgb, world_from_depth_opencv
from .io import (
    PhaseData,
    frame_rgb_camera,
    frame_rgb_from_depth,
    load_color,
    load_depth,
    load_source_rgb,
    write_json,
)


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


def _display_matrix(frame: PosedFrame) -> np.ndarray:
    axes = np.diag([*frame.display_axes, 1.0])
    return axes @ np.asarray(frame.camera_to_global, dtype=np.float64)


def _camera_payload(frame: PosedFrame, textured: bool) -> dict[str, Any]:
    camera = frame_rgb_camera(frame.source.frames[frame.frame_index], frame.source)
    return {
        "phaseName": frame.phase_name,
        "phaseId": frame.phase_id,
        "frameIndex": frame.frame_index,
        "timestampUs": frame.source.frames[frame.frame_index].timestamp_us,
        "matrix": [round(float(value), 8) for value in _display_matrix(frame).reshape(-1)],
        "aspect": round(camera.width / max(camera.height, 1), 6),
        "fovYDegrees": round(math.degrees(2.0 * math.atan(camera.height / (2.0 * camera.fy))), 5),
        "imageYUp": frame.image_y_up,
        "textureFrame": textured,
    }


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
    source_frame = frame.source.frames[frame.frame_index]
    image = load_source_rgb(source_frame, frame.source)
    camera = frame_rgb_camera(source_frame, frame.source)
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


def _sample_surface_colors(
    vertices: np.ndarray,
    normals: np.ndarray,
    triangles: np.ndarray,
    frames: list[PosedFrame],
    voxel_size_m: float,
    progress: Callable[..., None] | None,
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray, float]:
    luminances: list[float] = []
    for frame in frames:
        image = _load_texture_image(frame)
        sample = image[::8, ::8].reshape(-1, 3).astype(np.float32)
        luminances.append(float(np.median(sample.mean(axis=1))) if len(sample) else 128.0)
    target_luminance = float(np.median(luminances))
    # Blending only the strongest geometrically valid views avoids ghosting
    # from distant oblique frames while retaining smooth transitions where the
    # best camera changes across the surface.
    top_weights = np.zeros((len(vertices), 4), dtype=np.float32)
    top_colors = np.zeros((len(vertices), 4, 3), dtype=np.float32)
    fallback_weights = np.zeros(len(vertices), dtype=np.float32)
    fallback_colors = np.zeros((len(vertices), 3), dtype=np.float32)
    best_triangle_scores = np.zeros(len(triangles), dtype=np.float32)
    best_triangle_frames = np.full(len(triangles), -1, dtype=np.int16)
    loose_triangle_scores = np.zeros(len(triangles), dtype=np.float32)
    loose_triangle_frames = np.full(len(triangles), -1, dtype=np.int16)
    exposure_gains = np.asarray(
        [np.clip(target_luminance / max(value, 1.0), 0.72, 1.38) for value in luminances],
        dtype=np.float32,
    )
    chunk_size = 200_000

    for frame_number, (frame, luminance) in enumerate(
        zip(frames, luminances, strict=True), start=1
    ):
        phase = frame.source
        camera = phase.camera
        source_frame = phase.frames[frame.frame_index]
        image = _load_texture_image(frame)
        depth_m = load_depth(source_frame, camera).astype(np.float32)
        depth_m /= camera.depth_scale
        rgb_camera = frame_rgb_camera(source_frame, phase)
        rgb_from_depth = frame_rgb_from_depth(source_frame, phase)
        world_from_camera = world_from_depth_opencv(frame.camera_to_global, frame.image_y_up)
        camera_from_world = np.linalg.inv(world_from_camera)
        camera_center = world_from_camera[:3, 3]
        gain = float(exposure_gains[frame_number - 1])
        view_weights = np.zeros(len(vertices), dtype=np.float32)
        loose_view_weights = np.zeros(len(vertices), dtype=np.float32)

        for start in range(0, len(vertices), chunk_size):
            stop = min(start + chunk_size, len(vertices))
            points = vertices[start:stop].astype(np.float64, copy=False)
            camera_points = points @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
            z = camera_points[:, 2]
            safe_z = np.where(z > 1e-8, z, 1.0)
            depth_u = camera.fx * camera_points[:, 0] / safe_z + camera.cx
            depth_v = camera.fy * camera_points[:, 1] / safe_z + camera.cy
            depth_x = np.rint(depth_u).astype(np.int64)
            depth_y = np.rint(depth_v).astype(np.int64)
            in_depth = (
                (z > 0.25)
                & (z <= camera.max_depth_m)
                & (depth_x >= 0)
                & (depth_x < camera.width)
                & (depth_y >= 0)
                & (depth_y < camera.height)
            )
            observed = np.zeros(len(points), dtype=np.float32)
            observed[in_depth] = depth_m[depth_y[in_depth], depth_x[in_depth]]
            tolerance = np.maximum.reduce(
                (
                    np.full(len(points), max(voxel_size_m * 2.0, 0.022), dtype=np.float64),
                    np.maximum(z, 0.0) * 0.009,
                )
            )
            residual = np.abs(observed - z)

            rgb_points = camera_points @ rgb_from_depth[:3, :3].T + rgb_from_depth[:3, 3]
            rgb_u, rgb_v, rgb_z = project_rgb(rgb_points, rgb_camera)
            in_rgb = (
                in_depth
                & (observed > 0.0)
                & (rgb_z > 0.0)
                & (rgb_u >= 0.0)
                & (rgb_u <= rgb_camera.width - 1.001)
                & (rgb_v >= 0.0)
                & (rgb_v <= rgb_camera.height - 1.001)
            )
            candidates = np.flatnonzero(in_rgb)
            if not len(candidates):
                continue

            to_camera = camera_center - points[candidates]
            distances = np.linalg.norm(to_camera, axis=1)
            to_camera /= np.maximum(distances[:, None], 1e-8)
            facing = np.abs(np.sum(normals[start:stop][candidates] * to_camera, axis=1))
            border = np.minimum.reduce(
                (
                    rgb_u[candidates],
                    rgb_v[candidates],
                    rgb_camera.width - 1.0 - rgb_u[candidates],
                    rgb_camera.height - 1.0 - rgb_v[candidates],
                )
            )
            border_weight = np.clip(border / 24.0, 0.08, 1.0)
            base_weights = (
                np.maximum(facing, 0.08) ** 2
                * border_weight
                / np.maximum(distances, 0.5)
            )
            sampled = _bilinear_rgb(image, rgb_u[candidates], rgb_v[candidates])
            sampled = np.clip(sampled * gain, 0.0, 255.0)
            candidate_globals = start + candidates
            fallback_scores = base_weights * np.exp(
                -np.square(residual[candidates] / (tolerance[candidates] * 4.0))
            )
            loose_view_weights[candidate_globals] = fallback_scores
            better_fallback = fallback_scores > fallback_weights[candidate_globals]
            fallback_globals = candidate_globals[better_fallback]
            fallback_weights[fallback_globals] = fallback_scores[better_fallback]
            fallback_colors[fallback_globals] = sampled[better_fallback]

            directly_visible = residual[candidates] <= tolerance[candidates]
            if not np.any(directly_visible):
                continue
            accepted = candidates[directly_visible]
            global_indices = candidate_globals[directly_visible]
            weights = base_weights[directly_visible] * np.exp(
                -np.square(residual[accepted] / tolerance[accepted])
            )
            view_weights[global_indices] = weights
            sampled = sampled[directly_visible]
            weakest_slots = np.argmin(top_weights[global_indices], axis=1)
            weakest_weights = top_weights[global_indices, weakest_slots]
            stronger = weights > weakest_weights
            replacement_indices = global_indices[stronger]
            replacement_slots = weakest_slots[stronger]
            top_weights[replacement_indices, replacement_slots] = weights[stronger]
            top_colors[replacement_indices, replacement_slots] = sampled[stronger]

        if len(triangles):
            triangle_scores = np.min(view_weights[triangles], axis=1)
            better = triangle_scores > best_triangle_scores
            best_triangle_scores[better] = triangle_scores[better]
            best_triangle_frames[better] = frame_number - 1
            triangle_scores = np.min(loose_view_weights[triangles], axis=1)
            better = triangle_scores > loose_triangle_scores
            loose_triangle_scores[better] = triangle_scores[better]
            loose_triangle_frames[better] = frame_number - 1

        if progress:
            progress(
                "Texturing",
                f"Blended RGB keyframe {frame_number} of {len(frames)}",
                0,
                None,
                0.58 + 0.17 * frame_number / len(frames),
            )

    weight_sum = top_weights.sum(axis=1, dtype=np.float64)
    color_sum = np.sum(top_colors * top_weights[..., None], axis=1, dtype=np.float64)
    colors = np.full((len(vertices), 3), 128.0, dtype=np.float64)
    textured = weight_sum > 1e-10
    colors[textured] = color_sum[textured] / weight_sum[textured, None]
    direct_coverage = 100.0 * float(np.mean(textured)) if len(textured) else 0.0
    fallback = (~textured) & (fallback_weights > 1e-4)
    colors[fallback] = fallback_colors[fallback]
    textured[fallback] = True

    # Marching cubes can create a narrow boundary row just outside the valid
    # depth projection. Diffuse only from directly textured mesh neighbors so
    # those boundary vertices inherit the same local appearance instead of a
    # conspicuous neutral-gray fringe.
    edges = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]),
        axis=0,
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
    triangle_frames = best_triangle_frames.copy()
    missing_triangles = triangle_frames < 0
    triangle_frames[missing_triangles] = loose_triangle_frames[missing_triangles]
    direct_triangle_coverage = (
        100.0 * float(np.mean(best_triangle_frames >= 0)) if len(triangles) else 0.0
    )
    return (
        np.rint(colors).clip(0, 255).astype(np.uint8),
        direct_coverage,
        final_coverage,
        triangle_frames,
        exposure_gains,
        direct_triangle_coverage,
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
) -> dict[str, bool | int | float | str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    texture_frames = _select_texture_frames(frames)
    selected_keys = {(frame.phase_id, frame.frame_index) for frame in texture_frames}
    write_json(
        output_dir / "camera-poses.json",
        [_camera_payload(frame, (frame.phase_id, frame.frame_index) in selected_keys) for frame in frames],
    )
    if not frames or not texture_frames:
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
    normals = _vertex_normals(vertices, triangles)
    (
        vertex_colors,
        direct_texture_coverage,
        texture_coverage,
        triangle_frames,
        exposure_gains,
        direct_triangle_coverage,
    ) = _sample_surface_colors(
        vertices,
        normals,
        triangles,
        texture_frames,
        mesh_voxel_size,
        progress,
    )
    atlas, uvs, chart_size = _bake_triangle_atlas(
        vertex_colors,
        triangles,
        vertices=vertices,
        frames=texture_frames,
        triangle_frames=triangle_frames,
        exposure_gains=exposure_gains,
        progress=progress,
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
        "textureSource": "best_view_native_rgb_texel_projection",
        "textureBlend": "single_best_depth_visibility_angle_exposure_corrected",
        "textureDirectCoveragePercent": round(direct_texture_coverage, 2),
        "textureCoveragePercent": round(texture_coverage, 2),
        "textureDirectTriangleCoveragePercent": round(direct_triangle_coverage, 2),
        "textureChartSize": chart_size,
        "textureAtlasSize": int(max(atlas.shape[:2])),
    }
