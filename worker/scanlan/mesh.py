from __future__ import annotations

import math
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .calibration import rgb_depth_zbuffer
from .io import PhaseData, effective_rgb_camera, load_depth, load_source_rgb, write_json


MAX_TEXTURE_FRAMES = 16
MAX_ATLAS_SIZE = int(os.environ.get("SCANLAN_ATLAS_SIZE", "8192"))
MAX_ATLAS_SIZE = min(16384, max(4096, MAX_ATLAS_SIZE))
ATLAS_PADDING = 6
TARGET_MESH_SAMPLES = 280


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
    camera = effective_rgb_camera(frame.source)
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
        + chunk(b"IDAT", zlib.compress(scanlines, level=6))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def _texture_atlas(
    frames: list[PosedFrame],
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, int, int, int, int, int, int]]]:
    tiles: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    luminances: list[float] = []
    for index, frame in enumerate(frames, start=1):
        image = load_source_rgb(frame.source.frames[frame.frame_index], frame.source)
        depth = load_depth(frame.source.frames[frame.frame_index], frame.source.camera)
        _, uv_map, visibility = rgb_depth_zbuffer(depth, frame.source)
        projected = uv_map[visibility]
        if len(projected):
            left = max(0, int(np.floor(projected[:, 0].min())) - ATLAS_PADDING)
            top = max(0, int(np.floor(projected[:, 1].min())) - ATLAS_PADDING)
            right = min(image.shape[1], int(np.ceil(projected[:, 0].max())) + ATLAS_PADDING + 1)
            bottom = min(image.shape[0], int(np.ceil(projected[:, 1].max())) + ATLAS_PADDING + 1)
        else:
            left, top, right, bottom = 0, 0, image.shape[1], image.shape[0]
        crop = image[top:bottom, left:right]
        tiles.append((crop, (left, top, right - left, bottom - top)))
        valid_pixels = crop.reshape(-1, 3)
        luminances.append(float(np.median(valid_pixels.mean(axis=1))) if len(valid_pixels) else 128.0)
        if progress:
            progress(index, len(frames))

    columns = max(1, math.ceil(math.sqrt(len(frames))))
    rows = math.ceil(len(frames) / columns)
    max_width = max(tile.shape[1] for tile, _ in tiles) + ATLAS_PADDING * 2
    max_height = max(tile.shape[0] for tile, _ in tiles) + ATLAS_PADDING * 2
    scale = min(
        1.0,
        MAX_ATLAS_SIZE / max(columns * max_width, 1),
        MAX_ATLAS_SIZE / max(rows * max_height, 1),
    )
    cell_width = max(1, int(math.floor(max_width * scale)))
    cell_height = max(1, int(math.floor(max_height * scale)))
    atlas = np.full((rows * cell_height, columns * cell_width, 3), 24, dtype=np.uint8)
    placements: list[tuple[int, int, int, int, int, int, int, int]] = []
    target_luminance = float(np.median(luminances))
    for index, ((color, crop), luminance) in enumerate(zip(tiles, luminances, strict=True)):
        target_width = max(1, min(cell_width - ATLAS_PADDING * 2, int(round(color.shape[1] * scale))))
        target_height = max(1, min(cell_height - ATLAS_PADDING * 2, int(round(color.shape[0] * scale))))
        x_samples = np.linspace(0, color.shape[1] - 1, target_width, dtype=np.int64)
        y_samples = np.linspace(0, color.shape[0] - 1, target_height, dtype=np.int64)
        gain = np.clip(target_luminance / max(luminance, 1.0), 0.72, 1.38)
        resized = np.clip(color[np.ix_(y_samples, x_samples)].astype(np.float32) * gain, 0, 255).astype(np.uint8)
        cell_x = (index % columns) * cell_width
        cell_y = (index // columns) * cell_height
        offset_x = cell_x + (cell_width - target_width) // 2
        offset_y = cell_y + (cell_height - target_height) // 2
        atlas[offset_y : offset_y + target_height, offset_x : offset_x + target_width] = resized
        # Extend edge texels into the padding to prevent bilinear filtering from
        # sampling unrelated neighboring atlas islands.
        atlas[offset_y : offset_y + target_height, max(cell_x, offset_x - ATLAS_PADDING) : offset_x] = resized[:, :1]
        atlas[offset_y : offset_y + target_height, offset_x + target_width : min(cell_x + cell_width, offset_x + target_width + ATLAS_PADDING)] = resized[:, -1:]
        atlas[max(cell_y, offset_y - ATLAS_PADDING) : offset_y, max(cell_x, offset_x - ATLAS_PADDING) : min(cell_x + cell_width, offset_x + target_width + ATLAS_PADDING)] = atlas[offset_y : offset_y + 1, max(cell_x, offset_x - ATLAS_PADDING) : min(cell_x + cell_width, offset_x + target_width + ATLAS_PADDING)]
        atlas[offset_y + target_height : min(cell_y + cell_height, offset_y + target_height + ATLAS_PADDING), max(cell_x, offset_x - ATLAS_PADDING) : min(cell_x + cell_width, offset_x + target_width + ATLAS_PADDING)] = atlas[offset_y + target_height - 1 : offset_y + target_height, max(cell_x, offset_x - ATLAS_PADDING) : min(cell_x + cell_width, offset_x + target_width + ATLAS_PADDING)]
        placements.append((offset_x, offset_y, target_width, target_height, *crop))
    return atlas, placements


def _append_lines(handle: Any, lines: list[str]) -> None:
    if lines:
        handle.write("".join(lines))
        lines.clear()


def _append_frame_geometry(
    handle: Any,
    frame: PosedFrame,
    placement: tuple[int, int, int, int, int, int, int, int],
    atlas_width: int,
    atlas_height: int,
    vertex_offset: int,
) -> tuple[int, int]:
    camera = frame.source.camera
    depth = load_depth(frame.source.frames[frame.frame_index], camera).astype(np.float64)
    stride = max(1, math.ceil(max(camera.width, camera.height) / TARGET_MESH_SAMPLES))
    y_pixels = np.arange(0, camera.height, stride, dtype=np.int64)
    x_pixels = np.arange(0, camera.width, stride, dtype=np.int64)
    sampled = depth[np.ix_(y_pixels, x_pixels)] / camera.depth_scale
    valid = (sampled > 0.25) & (sampled <= camera.max_depth_m)
    _, uv_map, visibility = rgb_depth_zbuffer(depth, frame.source)
    sampled_uv = uv_map[np.ix_(y_pixels, x_pixels)]
    valid &= visibility[np.ix_(y_pixels, x_pixels)]
    if not np.any(valid):
        return 0, 0

    yy, xx = np.meshgrid(y_pixels, x_pixels, indexing="ij")
    z = sampled[valid]
    x = (xx[valid] - camera.cx) * z / camera.fx
    y = (yy[valid] - camera.cy) * z / camera.fy
    if frame.image_y_up:
        y *= -1.0
    camera_points = np.column_stack((x, y, z, np.ones_like(z)))
    world = (np.asarray(frame.camera_to_global) @ camera_points.T).T[:, :3]
    world *= np.asarray(frame.display_axes, dtype=np.float64)

    vertex_map = np.full(sampled.shape, -1, dtype=np.int64)
    local_ids = np.arange(int(valid.sum()), dtype=np.int64) + vertex_offset + 1
    vertex_map[valid] = local_ids
    offset_x, offset_y, tile_width, tile_height, crop_x, crop_y, crop_width, crop_height = placement
    source_x = (sampled_uv[..., 0][valid] - crop_x) / max(crop_width - 1, 1)
    source_y = (sampled_uv[..., 1][valid] - crop_y) / max(crop_height - 1, 1)
    source_x = np.clip(source_x, 0.0, 1.0)
    source_y = np.clip(source_y, 0.0, 1.0)
    u = (offset_x + source_x * max(tile_width - 1, 0) + 0.5) / atlas_width
    v = 1.0 - (offset_y + source_y * max(tile_height - 1, 0) + 0.5) / atlas_height

    lines: list[str] = []
    for point in world:
        lines.append(f"v {point[0]:.7g} {point[1]:.7g} {point[2]:.7g}\n")
        if len(lines) >= 8192:
            _append_lines(handle, lines)
    _append_lines(handle, lines)
    for texture_u, texture_v in zip(u, v, strict=True):
        lines.append(f"vt {texture_u:.7g} {texture_v:.7g}\n")
        if len(lines) >= 8192:
            _append_lines(handle, lines)
    _append_lines(handle, lines)

    if sampled.shape[0] < 2 or sampled.shape[1] < 2:
        return int(valid.sum()), 0
    a, b = vertex_map[:-1, :-1], vertex_map[1:, :-1]
    c, d = vertex_map[:-1, 1:], vertex_map[1:, 1:]
    za, zb = sampled[:-1, :-1], sampled[1:, :-1]
    zc, zd = sampled[:-1, 1:], sampled[1:, 1:]
    threshold_abc = np.maximum(0.045, np.minimum(np.minimum(za, zb), zc) * 0.025)
    threshold_bdc = np.maximum(0.045, np.minimum(np.minimum(zb, zd), zc) * 0.025)
    mask_abc = (
        (a > 0)
        & (b > 0)
        & (c > 0)
        & (np.maximum(np.maximum(za, zb), zc) - np.minimum(np.minimum(za, zb), zc) <= threshold_abc)
    )
    mask_bdc = (
        (b > 0)
        & (d > 0)
        & (c > 0)
        & (np.maximum(np.maximum(zb, zd), zc) - np.minimum(np.minimum(zb, zd), zc) <= threshold_bdc)
    )
    triangles_abc = np.column_stack((a[mask_abc], b[mask_abc], c[mask_abc]))
    triangles_bdc = np.column_stack((b[mask_bdc], d[mask_bdc], c[mask_bdc]))
    reverses_winding = frame.image_y_up != (np.prod(frame.display_axes) < 0)
    if reverses_winding:
        triangles_abc = triangles_abc[:, [0, 2, 1]]
        triangles_bdc = triangles_bdc[:, [0, 2, 1]]
    triangle_count = int(triangles_abc.shape[0] + triangles_bdc.shape[0])
    for triangle in (triangles_abc, triangles_bdc):
        for first, second, third in triangle:
            lines.append(f"f {first}/{first} {second}/{second} {third}/{third}\n")
            if len(lines) >= 8192:
                _append_lines(handle, lines)
    _append_lines(handle, lines)
    return int(valid.sum()), triangle_count


def build_mesh_artifacts(
    output_dir: Path,
    frames: list[PosedFrame],
    progress: Callable[..., None] | None = None,
) -> dict[str, int | str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = _select_texture_frames(frames)
    selected_keys = {(frame.phase_id, frame.frame_index) for frame in selected}
    write_json(
        output_dir / "camera-poses.json",
        [_camera_payload(frame, (frame.phase_id, frame.frame_index) in selected_keys) for frame in frames],
    )
    if not selected:
        return {"cameraFrameCount": 0, "meshVertexCount": 0, "meshTriangleCount": 0}

    if progress:
        progress("Meshing", f"Packing RGB texture atlas from {len(selected)} keyframes", 0, None, 0.0)
    atlas, placements = _texture_atlas(
        selected,
        lambda done, total: progress(
            "Meshing",
            f"Prepared RGB atlas keyframe {done} of {total}",
            0,
            None,
            0.45 * done / total,
        ) if progress else None,
    )
    texture_path = output_dir / "room-texture.png"
    _write_png(texture_path, atlas)
    (output_dir / "room-mesh.mtl").write_text(
        "# ScanLan reprojected RGB material\n"
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
    vertex_count = 0
    triangle_count = 0
    with (output_dir / "room-mesh.obj").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "# ScanLan RGB-reprojected mesh\n"
            "mtllib room-mesh.mtl\n"
            "o room_mesh\n"
            "usemtl room_rgb\n"
        )
        for index, (frame, placement) in enumerate(zip(selected, placements, strict=True), start=1):
            added_vertices, added_triangles = _append_frame_geometry(
                handle,
                frame,
                placement,
                atlas.shape[1],
                atlas.shape[0],
                vertex_count,
            )
            vertex_count += added_vertices
            triangle_count += added_triangles
            if progress:
                progress(
                    "Meshing",
                    f"Reprojected RGB keyframe {index} of {len(selected)}",
                    0,
                    None,
                    0.45 + 0.55 * index / len(selected),
                )
    if progress:
        progress(
            "Meshing",
            f"Textured mesh contains {triangle_count:,} triangles",
            4,
            None,
            1.0,
        )
    return {
        "cameraFrameCount": len(frames),
        "textureFrameCount": len(selected),
        "meshVertexCount": vertex_count,
        "meshTriangleCount": triangle_count,
        "meshOutputPath": "outputs/room-mesh.obj",
        "meshMaterialPath": "outputs/room-mesh.mtl",
        "meshTexturePath": "outputs/room-texture.png",
        "textureSource": "native_rgb_with_aligned_fallback",
        "textureAtlasSize": int(max(atlas.shape[:2])),
    }
