from __future__ import annotations

import json
import os
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .analysis import FusedMaterialSurface
from .radiometry import linear_to_srgb, srgb_to_linear


PBR_CONTRACT_VERSION = "scanlan-pbr-v1"


@dataclass(frozen=True)
class PbrArtifacts:
    glb_path: Path
    observed_atlas_path: Path
    intrinsic_atlas_path: Path
    report_path: Path
    material_coverage: float
    emissive_strength: float


def _validated_geometry(
    vertices: np.ndarray,
    normals: np.ndarray,
    triangles: np.ndarray,
    uvs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(vertices, dtype=np.float32)
    vertex_normals = np.asarray(normals, dtype=np.float32)
    faces = np.asarray(triangles, dtype=np.int64)
    texture_coordinates = np.asarray(uvs, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("PBR vertices must be non-empty Vx3")
    if vertex_normals.shape != points.shape:
        raise ValueError("PBR normals must be Vx3")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("PBR triangles must be non-empty Fx3")
    if texture_coordinates.shape != (len(faces), 3, 2):
        raise ValueError("PBR UVs must be source triangle-corner coordinates Fx3x2")
    if not all(np.isfinite(value).all() for value in (points, vertex_normals, texture_coordinates)):
        raise ValueError("PBR geometry contains non-finite values")
    if np.any(faces < 0) or np.any(faces >= len(points)):
        raise ValueError("PBR triangles reference invalid vertices")
    if np.any(texture_coordinates < 0.0) or np.any(texture_coordinates > 1.0):
        raise ValueError("PBR UV coordinates must remain in [0, 1]")
    lengths = np.linalg.norm(vertex_normals, axis=1)
    if np.any(np.abs(lengths - 1.0) > 2e-2):
        raise ValueError("PBR geometry normals must be unit length")
    return points, vertex_normals, faces, texture_coordinates


def _validated_observed_atlas(value: np.ndarray | Image.Image) -> np.ndarray:
    array = np.asarray(value.convert("RGB") if isinstance(value, Image.Image) else value)
    if array.ndim != 3 or array.shape[2] != 3 or not len(array) or not array.shape[1]:
        raise ValueError("observed atlas must be non-empty HxWx3")
    if array.dtype != np.uint8:
        if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
            raise ValueError("floating observed atlas must remain in [0, 1]")
        array = np.rint(array * 255.0).astype(np.uint8)
    return np.ascontiguousarray(array)


def _sample_atlas_vertices(atlas: np.ndarray, faces: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    height, width = atlas.shape[:2]
    accumulated = np.zeros((int(faces.max()) + 1, 3), dtype=np.float64)
    counts = np.zeros(len(accumulated), dtype=np.float64)
    x = np.clip(np.rint(uvs[..., 0] * (width - 1)).astype(np.int64), 0, width - 1)
    y = np.clip(np.rint((1.0 - uvs[..., 1]) * (height - 1)).astype(np.int64), 0, height - 1)
    samples = atlas[y, x].astype(np.float64) / 255.0
    np.add.at(accumulated, faces.reshape(-1), samples.reshape(-1, 3))
    np.add.at(counts, faces.reshape(-1), 1.0)
    return srgb_to_linear((accumulated / np.maximum(counts[:, None], 1.0)).astype(np.float32))


def _complete_vertex_material(
    surface: FusedMaterialSurface,
    observed_linear: np.ndarray,
    geometry_normals: np.ndarray,
) -> dict[str, np.ndarray]:
    checked = surface.validated()
    count = len(geometry_normals)
    if len(checked.valid_mask) != count:
        raise ValueError("fused material surface must match the PBR mesh vertices")
    confidence = np.where(checked.valid_mask, checked.confidence, 0.0).astype(np.float32)
    confidence = np.clip(confidence, 0.0, 1.0)

    def scalar(name: str, fallback: float) -> np.ndarray:
        value = getattr(checked, name)
        predicted = np.full(count, fallback, dtype=np.float32) if value is None else value
        return confidence * predicted + (1.0 - confidence) * fallback

    albedo = observed_linear.copy()
    if checked.albedo_linear is not None:
        albedo = confidence[:, None] * checked.albedo_linear + (1.0 - confidence[:, None]) * albedo
    emission = np.zeros((count, 3), dtype=np.float32)
    if checked.emission_linear is not None:
        emission = confidence[:, None] * checked.emission_linear
    material_normals = geometry_normals.copy()
    if checked.normal_world is not None:
        material_normals = (
            confidence[:, None] * checked.normal_world
            + (1.0 - confidence[:, None]) * geometry_normals
        )
        material_normals /= np.maximum(
            np.linalg.norm(material_normals, axis=1, keepdims=True), 1e-8
        )
    return {
        "albedo": np.clip(albedo, 0.0, 1.0),
        "roughness": np.clip(scalar("roughness", 1.0), 0.04, 1.0),
        "metallic": np.clip(scalar("metallic", 0.0), 0.0, 1.0),
        "transmission": np.clip(scalar("transmission", 0.0), 0.0, 1.0),
        "normal": material_normals,
        "emission": np.maximum(emission, 0.0),
        "confidence": confidence,
    }


def _triangle_tangent_frame(
    positions: np.ndarray, normals: np.ndarray, uv: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    edge1 = positions[1] - positions[0]
    edge2 = positions[2] - positions[0]
    delta1 = uv[1] - uv[0]
    delta2 = uv[2] - uv[0]
    determinant = float(delta1[0] * delta2[1] - delta1[1] * delta2[0])
    averaged_normal = np.sum(normals, axis=0)
    averaged_normal /= max(float(np.linalg.norm(averaged_normal)), 1e-8)
    if abs(determinant) > 1e-10:
        tangent = (edge1 * delta2[1] - edge2 * delta1[1]) / determinant
        bitangent_reference = (edge2 * delta1[0] - edge1 * delta2[0]) / determinant
    else:
        tangent = edge1
        bitangent_reference = edge2
    tangent -= averaged_normal * float(np.dot(tangent, averaged_normal))
    if float(np.linalg.norm(tangent)) <= 1e-8:
        axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(axis, averaged_normal))) > 0.9:
            axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        tangent = np.cross(axis, averaged_normal)
    tangent /= max(float(np.linalg.norm(tangent)), 1e-8)
    sign = -1.0 if float(np.dot(np.cross(averaged_normal, tangent), bitangent_reference)) < 0.0 else 1.0
    bitangent = np.cross(averaged_normal, tangent) * sign
    return tangent.astype(np.float32), bitangent.astype(np.float32), sign


def _bake_material_atlases(
    vertices: np.ndarray,
    normals: np.ndarray,
    triangles: np.ndarray,
    uvs: np.ndarray,
    observed: np.ndarray,
    values: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    height, width = observed.shape[:2]
    albedo = np.zeros((height, width, 3), dtype=np.float32)
    roughness = np.ones((height, width), dtype=np.float32)
    metallic = np.zeros((height, width), dtype=np.float32)
    transmission = np.zeros((height, width), dtype=np.float32)
    normal_map = np.zeros((height, width, 3), dtype=np.float32)
    normal_map[..., 2] = 1.0
    emission = np.zeros((height, width, 3), dtype=np.float32)
    covered = np.zeros((height, width), dtype=bool)
    corner_tangents = np.empty((len(triangles), 3, 4), dtype=np.float32)

    pixel_uvs = uvs.copy()
    pixel_uvs[..., 0] *= width - 1
    pixel_uvs[..., 1] = (1.0 - pixel_uvs[..., 1]) * (height - 1)
    for triangle_index, face in enumerate(triangles):
        xy = pixel_uvs[triangle_index]
        tangent, _bitangent, sign = _triangle_tangent_frame(
            vertices[face], normals[face], uvs[triangle_index]
        )
        for corner, corner_normal in enumerate(normals[face]):
            corner_tangent = tangent - corner_normal * float(np.dot(tangent, corner_normal))
            corner_tangent /= max(float(np.linalg.norm(corner_tangent)), 1e-8)
            corner_tangents[triangle_index, corner, :3] = corner_tangent
            corner_tangents[triangle_index, corner, 3] = sign
        minimum = np.maximum(np.floor(np.min(xy, axis=0)).astype(np.int64), 0)
        maximum = np.minimum(
            np.ceil(np.max(xy, axis=0)).astype(np.int64),
            np.asarray([width - 1, height - 1]),
        )
        if np.any(maximum < minimum):
            continue
        x, y = np.meshgrid(
            np.arange(minimum[0], maximum[0] + 1, dtype=np.float32),
            np.arange(minimum[1], maximum[1] + 1, dtype=np.float32),
        )
        denominator = (
            (xy[1, 1] - xy[2, 1]) * (xy[0, 0] - xy[2, 0])
            + (xy[2, 0] - xy[1, 0]) * (xy[0, 1] - xy[2, 1])
        )
        if abs(float(denominator)) <= 1e-8:
            continue
        w0 = ((xy[1, 1] - xy[2, 1]) * (x - xy[2, 0]) + (xy[2, 0] - xy[1, 0]) * (y - xy[2, 1])) / denominator
        w1 = ((xy[2, 1] - xy[0, 1]) * (x - xy[2, 0]) + (xy[0, 0] - xy[2, 0]) * (y - xy[2, 1])) / denominator
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not np.any(inside):
            continue
        rows = y.astype(np.int64)[inside]
        columns = x.astype(np.int64)[inside]
        weights = np.stack((w0[inside], w1[inside], w2[inside]), axis=1)
        for name, destination in (
            ("albedo", albedo),
            ("roughness", roughness),
            ("metallic", metallic),
            ("transmission", transmission),
            ("emission", emission),
        ):
            source = values[name][face]
            destination[rows, columns] = weights @ source
        interpolated_geometry_normal = weights @ normals[face]
        interpolated_geometry_normal /= np.maximum(
            np.linalg.norm(interpolated_geometry_normal, axis=1, keepdims=True), 1e-8
        )
        interpolated_material_normal = weights @ values["normal"][face]
        interpolated_material_normal /= np.maximum(
            np.linalg.norm(interpolated_material_normal, axis=1, keepdims=True), 1e-8
        )
        local_tangent = tangent - interpolated_geometry_normal * np.sum(
            interpolated_geometry_normal * tangent[None, :], axis=1, keepdims=True
        )
        local_tangent /= np.maximum(np.linalg.norm(local_tangent, axis=1, keepdims=True), 1e-8)
        local_bitangent = np.cross(interpolated_geometry_normal, local_tangent) * sign
        tangent_normal = np.column_stack(
            (
                np.sum(interpolated_material_normal * local_tangent, axis=1),
                np.sum(interpolated_material_normal * local_bitangent, axis=1),
                np.sum(interpolated_material_normal * interpolated_geometry_normal, axis=1),
            )
        )
        tangent_normal /= np.maximum(np.linalg.norm(tangent_normal, axis=1, keepdims=True), 1e-8)
        normal_map[rows, columns] = tangent_normal
        covered[rows, columns] = True

    observed_linear_pixels = srgb_to_linear(observed.astype(np.float32) / 255.0)
    albedo[~covered] = observed_linear_pixels[~covered]
    return {
        "base_color": np.rint(linear_to_srgb(np.clip(albedo, 0.0, 1.0)) * 255.0).astype(np.uint8),
        "metallic_roughness": np.stack(
            (
                np.full_like(roughness, 1.0),
                np.clip(roughness, 0.0, 1.0),
                np.clip(metallic, 0.0, 1.0),
            ),
            axis=2,
        ),
        "transmission": np.clip(transmission, 0.0, 1.0),
        "normal": np.clip(normal_map * 0.5 + 0.5, 0.0, 1.0),
        "emission_linear": np.maximum(emission, 0.0),
        "covered": covered,
    }, corner_tangents


def _atomic_png(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        Image.fromarray(np.asarray(value, dtype=np.uint8)).save(temporary, format="PNG", compress_level=6)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_aligned(payload: bytearray, value: bytes) -> tuple[int, int]:
    while len(payload) % 4:
        payload.append(0)
    offset = len(payload)
    payload.extend(value)
    return offset, len(value)


def _png_bytes(value: np.ndarray) -> bytes:
    import io

    stream = io.BytesIO()
    Image.fromarray(np.asarray(value, dtype=np.uint8)).save(stream, format="PNG", compress_level=6)
    return stream.getvalue()


def _write_glb(
    path: Path,
    vertices: np.ndarray,
    normals: np.ndarray,
    triangles: np.ndarray,
    uvs: np.ndarray,
    tangents: np.ndarray,
    images: list[tuple[str, np.ndarray]],
    emissive_strength: float,
    has_transmission: bool,
) -> None:
    positions = np.ascontiguousarray(vertices[triangles].reshape(-1, 3), dtype="<f4")
    render_normals = np.ascontiguousarray(normals[triangles].reshape(-1, 3), dtype="<f4")
    texture_coordinates = np.ascontiguousarray(uvs.reshape(-1, 2), dtype="<f4")
    render_tangents = np.ascontiguousarray(tangents.reshape(-1, 4), dtype="<f4")
    indices = np.arange(len(positions), dtype="<u4")
    binary = bytearray()
    buffer_views: list[dict[str, Any]] = []

    def add_view(value: bytes, target: int | None = None) -> int:
        offset, length = _append_aligned(binary, value)
        view: dict[str, Any] = {"buffer": 0, "byteOffset": offset, "byteLength": length}
        if target is not None:
            view["target"] = target
        buffer_views.append(view)
        return len(buffer_views) - 1

    position_view = add_view(positions.tobytes(), 34962)
    normal_view = add_view(render_normals.tobytes(), 34962)
    uv_view = add_view(texture_coordinates.tobytes(), 34962)
    tangent_view = add_view(render_tangents.tobytes(), 34962)
    index_view = add_view(indices.tobytes(), 34963)
    gltf_images: list[dict[str, Any]] = []
    for name, image in images:
        gltf_images.append(
            {"name": name, "bufferView": add_view(_png_bytes(image)), "mimeType": "image/png"}
        )

    count = len(positions)
    accessors = [
        {
            "bufferView": position_view,
            "componentType": 5126,
            "count": count,
            "type": "VEC3",
            "min": np.min(positions, axis=0).astype(float).tolist(),
            "max": np.max(positions, axis=0).astype(float).tolist(),
        },
        {"bufferView": normal_view, "componentType": 5126, "count": count, "type": "VEC3"},
        {"bufferView": uv_view, "componentType": 5126, "count": count, "type": "VEC2"},
        {"bufferView": tangent_view, "componentType": 5126, "count": count, "type": "VEC4"},
        {"bufferView": index_view, "componentType": 5125, "count": count, "type": "SCALAR"},
    ]
    material: dict[str, Any] = {
        "name": "ScanLan intrinsic PBR",
        "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 1},
            "metallicRoughnessTexture": {"index": 2},
            "metallicFactor": 1.0,
            "roughnessFactor": 1.0,
        },
        "normalTexture": {"index": 4},
        "emissiveTexture": {"index": 5},
        "emissiveFactor": [1.0, 1.0, 1.0],
        "extras": {"scanlanObservedTexture": 0, "contract": PBR_CONTRACT_VERSION},
    }
    extensions_used: list[str] = []
    material_extensions: dict[str, Any] = {}
    if has_transmission:
        extensions_used.append("KHR_materials_transmission")
        material_extensions["KHR_materials_transmission"] = {
            "transmissionFactor": 1.0,
            "transmissionTexture": {"index": 3},
        }
    if emissive_strength > 1.0 + 1e-6:
        extensions_used.append("KHR_materials_emissive_strength")
        material_extensions["KHR_materials_emissive_strength"] = {
            "emissiveStrength": emissive_strength
        }
    if material_extensions:
        material["extensions"] = material_extensions
    document: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": f"ScanLan {PBR_CONTRACT_VERSION}"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "room", "mesh": 0}],
        "meshes": [
            {
                "name": "room-mesh",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2, "TANGENT": 3},
                        "indices": 4,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "materials": [material],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "textures": [{"sampler": 0, "source": index} for index in range(len(images))],
        "images": gltf_images,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(binary)}],
    }
    if extensions_used:
        document["extensionsUsed"] = extensions_used
    json_payload = json.dumps(document, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    json_payload += b" " * ((-len(json_payload)) % 4)
    binary.extend(b"\0" * ((-len(binary)) % 4))
    total_length = 12 + 8 + len(json_payload) + 8 + len(binary)
    payload = (
        struct.pack("<4sII", b"glTF", 2, total_length)
        + struct.pack("<I4s", len(json_payload), b"JSON")
        + json_payload
        + struct.pack("<I4s", len(binary), b"BIN\0")
        + bytes(binary)
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_pbr_artifacts(
    output_dir: Path,
    vertices: np.ndarray,
    normals: np.ndarray,
    triangles: np.ndarray,
    uvs: np.ndarray,
    observed_atlas: np.ndarray | Image.Image,
    surface: FusedMaterialSurface,
) -> PbrArtifacts:
    """Bake intrinsic PBR atlases and a self-contained glTF 2.0 GLB.

    Material confidence blends predictions toward safe measured/neutral values.
    This prevents unsupported inverse-rendering pixels from becoming glossy,
    metallic, transmissive, or emissive merely because an atlas was exported.
    """

    points, vertex_normals, faces, texture_coordinates = _validated_geometry(
        vertices, normals, triangles, uvs
    )
    observed = _validated_observed_atlas(observed_atlas)
    observed_linear = _sample_atlas_vertices(observed, faces, texture_coordinates)
    if len(observed_linear) != len(points):
        padded = np.zeros((len(points), 3), dtype=np.float32)
        padded[: len(observed_linear)] = observed_linear
        observed_linear = padded
    material = _complete_vertex_material(surface, observed_linear, vertex_normals)
    baked, tangents = _bake_material_atlases(
        points, vertex_normals, faces, texture_coordinates, observed, material
    )
    emission_max = float(np.max(baked["emission_linear"]))
    emissive_strength = max(1.0, emission_max)
    emission_texture = np.rint(
        linear_to_srgb(np.clip(baked["emission_linear"] / emissive_strength, 0.0, 1.0))
        * 255.0
    ).astype(np.uint8)
    metallic_roughness = np.rint(baked["metallic_roughness"] * 255.0).astype(np.uint8)
    transmission = np.zeros((*baked["transmission"].shape, 3), dtype=np.uint8)
    transmission[..., 0] = np.rint(baked["transmission"] * 255.0).astype(np.uint8)
    normal = np.rint(baked["normal"] * 255.0).astype(np.uint8)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "observed": output_dir / "room-observed.png",
        "base": output_dir / "room-base-color.png",
        "metallic_roughness": output_dir / "room-metallic-roughness.png",
        "transmission": output_dir / "room-transmission.png",
        "normal": output_dir / "room-normal.png",
        "emission": output_dir / "room-emission.png",
        "glb": output_dir / "room-pbr.glb",
        "report": output_dir / "pbr-report.json",
    }
    for name, value in (
        ("observed", observed),
        ("base", baked["base_color"]),
        ("metallic_roughness", metallic_roughness),
        ("transmission", transmission),
        ("normal", normal),
        ("emission", emission_texture),
    ):
        _atomic_png(paths[name], value)
    _write_glb(
        paths["glb"],
        points,
        vertex_normals,
        faces,
        texture_coordinates,
        tangents,
        [
            ("observed", observed),
            ("baseColor", baked["base_color"]),
            ("metallicRoughness", metallic_roughness),
            ("transmission", transmission),
            ("normal", normal),
            ("emission", emission_texture),
        ],
        emissive_strength,
        bool(np.any(baked["transmission"] > 1.0 / 255.0)),
    )
    coverage = float(np.mean(material["confidence"] > 0.0))
    report = {
        "contract": PBR_CONTRACT_VERSION,
        "status": "ready",
        "vertexCount": len(points),
        "triangleCount": len(faces),
        "atlasWidth": int(observed.shape[1]),
        "atlasHeight": int(observed.shape[0]),
        "rasterCoverage": float(np.mean(baked["covered"])),
        "materialVertexCoverage": coverage,
        "emissiveStrength": emissive_strength,
        "colorSpace": {
            "observed": "sRGB",
            "baseColor": "sRGB encoding of linear intrinsic albedo",
            "emission": "sRGB encoding of linear emission divided by emissiveStrength",
            "metallicRoughness": "linear; G=roughness, B=metallic",
            "normal": "linear OpenGL tangent space",
            "transmission": "linear R channel",
        },
        "fallback": "confidence blend to observed base color, rough dielectric, opaque, non-emissive",
        "artifacts": {key: path.name for key, path in paths.items() if key != "report"},
    }
    temporary_report = paths["report"].with_name(f".{paths['report'].name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_report, paths["report"])
    finally:
        temporary_report.unlink(missing_ok=True)
    return PbrArtifacts(
        paths["glb"],
        paths["observed"],
        paths["base"],
        paths["report"],
        coverage,
        emissive_strength,
    )
