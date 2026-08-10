from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


NEURAL_SDF_VERSION = "scanlan-neural-sdf-v1"


@dataclass(frozen=True)
class SurfaceNormalization:
    center: np.ndarray
    radius: float


def _normalization(vertices: np.ndarray) -> SurfaceNormalization:
    lower, upper = np.percentile(vertices, [1.0, 99.0], axis=0)
    center = (lower + upper) * 0.5
    radius = max(float(np.linalg.norm(upper - lower) * 0.5), 1e-4)
    return SurfaceNormalization(center.astype(np.float32), radius)


def _face_geometry(
    vertices: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    edges_a = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    edges_b = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    crosses = np.cross(edges_a, edges_b)
    doubled_area = np.linalg.norm(crosses, axis=1)
    normals = crosses / np.maximum(doubled_area[:, None], 1e-12)
    return normals.astype(np.float32), (doubled_area * 0.5).astype(np.float32)


def _vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    face_normals, areas = _face_geometry(vertices, triangles)
    normals = np.zeros_like(vertices, dtype=np.float32)
    weighted = face_normals * areas[:, None]
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], weighted)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-12)


def validate_candidate(
    source_vertices: np.ndarray,
    triangles: np.ndarray,
    candidate_vertices: np.ndarray,
    voxel_size_m: float,
) -> dict[str, Any]:
    source = np.asarray(source_vertices, dtype=np.float32)
    candidate = np.asarray(candidate_vertices, dtype=np.float32)
    faces = np.asarray(triangles, dtype=np.int64)
    if candidate.shape != source.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Neural SDF candidate changed the vertex contract")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("Neural SDF input has no triangle surface")
    if not np.isfinite(candidate).all():
        raise ValueError("Neural SDF candidate contains non-finite vertices")

    displacement = np.linalg.norm(candidate - source, axis=1)
    source_normals, source_areas = _face_geometry(source, faces)
    candidate_normals, candidate_areas = _face_geometry(candidate, faces)
    source_valid = source_areas > max(float(voxel_size_m) ** 2 * 1e-6, 1e-12)
    candidate_valid = candidate_areas > max(float(voxel_size_m) ** 2 * 1e-6, 1e-12)
    comparable = source_valid & candidate_valid
    flipped_fraction = (
        float(np.mean(np.sum(source_normals[comparable] * candidate_normals[comparable], axis=1) <= 0.0))
        if np.any(comparable)
        else 1.0
    )
    degenerate_fraction = float(np.mean(~candidate_valid))
    source_degenerate_fraction = float(np.mean(~source_valid))
    median = float(np.median(displacement))
    p95 = float(np.percentile(displacement, 95.0))
    maximum = float(np.max(displacement, initial=0.0))
    accepted = (
        median <= voxel_size_m * 0.55
        and p95 <= voxel_size_m * 1.25
        and maximum <= voxel_size_m * 2.05
        and flipped_fraction <= 0.005
        and degenerate_fraction <= source_degenerate_fraction + 0.001
    )
    return {
        "accepted": accepted,
        "medianDisplacementM": median,
        "p95DisplacementM": p95,
        "maximumDisplacementM": maximum,
        "flippedTriangleFraction": flipped_fraction,
        "degenerateTriangleFraction": degenerate_fraction,
        "sourceDegenerateTriangleFraction": source_degenerate_fraction,
    }


def _load_input(path: Path) -> tuple[np.ndarray, np.ndarray, float, str]:
    with np.load(path, allow_pickle=False) as values:
        vertices = np.asarray(values["vertices"], dtype=np.float32)
        triangles = np.asarray(values["triangles"], dtype=np.int64)
        voxel_size_m = float(np.asarray(values["voxel_size_m"]).reshape(()))
        fingerprint = str(np.asarray(values["fingerprint"]).reshape(()))
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 32:
        raise ValueError("Neural SDF input needs at least 32 finite 3D vertices")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
        raise ValueError("Neural SDF input needs an indexed triangle mesh")
    if not np.isfinite(vertices).all() or np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        raise ValueError("Neural SDF input mesh is invalid")
    if not math.isfinite(voxel_size_m) or voxel_size_m <= 0.0:
        raise ValueError("Neural SDF voxel size must be positive")
    return vertices, triangles, voxel_size_m, fingerprint


def _sample_training_surface(
    vertices: np.ndarray,
    triangles: np.ndarray,
    maximum_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    normals = _vertex_normals(vertices, triangles)
    count = min(len(vertices), maximum_samples)
    if count == len(vertices):
        indices = np.arange(len(vertices), dtype=np.int64)
    else:
        # A stable random sample avoids the spatial bias of selecting a prefix
        # from TSDF and Poisson mesh vertex buffers.
        indices = np.random.default_rng(seed).choice(len(vertices), count, replace=False)
        indices.sort()
    usable = np.linalg.norm(normals[indices], axis=1) > 0.5
    return vertices[indices][usable], normals[indices][usable]


def refine_surface(
    vertices: np.ndarray,
    triangles: np.ndarray,
    voxel_size_m: float,
    *,
    iterations: int = 1_600,
    device_name: str = "cuda",
    seed: int = 20260810,
    progress_path: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from torch import nn

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Max Quality neural SDF refinement requires CUDA")
    device = torch.device(device_name)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    normalization = _normalization(vertices)
    normalized = (vertices - normalization.center) / normalization.radius
    surface, surface_normals = _sample_training_surface(
        normalized, triangles, 140_000, seed
    )
    if len(surface) < 32:
        raise RuntimeError("Neural SDF input has too few oriented surface samples")
    voxel_normalized = voxel_size_m / normalization.radius
    offsets = np.asarray([0.0, 0.45, -0.45, 1.1, -1.1], dtype=np.float32) * voxel_normalized
    sample_points = np.concatenate(
        [surface + surface_normals * offset for offset in offsets], axis=0
    ).astype(np.float32)
    targets = np.concatenate(
        [np.full(len(surface), offset, dtype=np.float32) for offset in offsets]
    )
    held_out = (np.arange(len(sample_points), dtype=np.int64) * 2654435761 + seed) % 17 == 0
    train_points = torch.from_numpy(sample_points[~held_out]).to(device)
    train_targets = torch.from_numpy(targets[~held_out]).to(device)
    held_points = torch.from_numpy(sample_points[held_out]).to(device)
    held_targets = torch.from_numpy(targets[held_out]).to(device)

    class ProgressiveSdf(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("frequencies", 2.0 ** torch.arange(7, dtype=torch.float32))
            width = 128
            self.network = nn.Sequential(
                nn.Linear(3 + 3 * 2 * 7, width),
                nn.Softplus(beta=40.0),
                nn.Linear(width, width),
                nn.Softplus(beta=40.0),
                nn.Linear(width, width),
                nn.Softplus(beta=40.0),
                nn.Linear(width, 1),
            )

        def forward(self, points: torch.Tensor, level: float = 7.0) -> torch.Tensor:
            phase = points[..., None] * self.frequencies * math.pi
            weights = torch.clamp(
                torch.as_tensor(level, device=points.device) - torch.arange(7, device=points.device),
                0.0,
                1.0,
            )
            encoded = torch.cat(
                (points, (torch.sin(phase) * weights).flatten(-2), (torch.cos(phase) * weights).flatten(-2)),
                dim=-1,
            )
            return self.network(encoded).squeeze(-1)

    model = ProgressiveSdf().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-6)
    batch_size = min(16_384, len(train_points))
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    smoothed_loss = None
    iterations = max(200, min(int(iterations), 5_000))
    for iteration in range(iterations):
        indices = torch.randint(
            len(train_points), (batch_size,), generator=generator, device=device
        )
        points = train_points[indices]
        target = train_targets[indices]
        level = min(7.0, 1.0 + 7.0 * iteration / max(iterations * 0.72, 1.0))
        prediction = model(points, level)
        data_loss = torch.nn.functional.smooth_l1_loss(
            prediction, target, beta=max(voxel_normalized * 0.2, 1e-5)
        )
        loss = data_loss
        if iteration % 4 == 0:
            eikonal_points = points[: min(1_024, len(points))].detach().requires_grad_(True)
            eikonal_sdf = model(eikonal_points, level)
            gradient = torch.autograd.grad(
                eikonal_sdf.sum(), eikonal_points, create_graph=True
            )[0]
            eikonal_loss = (gradient.norm(dim=-1) - 1.0).abs().mean()
            loss = loss + 0.04 * eikonal_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        value = float(loss.detach())
        smoothed_loss = value if smoothed_loss is None else smoothed_loss * 0.96 + value * 0.04
        if progress_path is not None and (iteration % 20 == 0 or iteration + 1 == iterations):
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_temporary = progress_path.with_suffix(progress_path.suffix + ".tmp")
            progress_temporary.write_text(
                json.dumps(
                    {
                        "stage": "neural_sdf",
                        "detail": "Fitting a validation-gated metric signed-distance surface",
                        "progress": (iteration + 1) / iterations,
                        "stageProgress": (iteration + 1) / iterations,
                        "iteration": iteration + 1,
                        "totalIterations": iterations,
                        "loss": value,
                        "smoothedLoss": smoothed_loss,
                        "computeBackend": (
                            torch.cuda.get_device_name(device) if device.type == "cuda" else "PyTorch CPU"
                        ),
                    }
                ),
                encoding="utf-8",
            )
            progress_temporary.replace(progress_path)

    model.eval()
    with torch.no_grad():
        held_error = torch.abs(model(held_points, 7.0) - held_targets)
        held_mae = float(held_error.mean())
        held_p95 = float(torch.quantile(held_error, 0.95))

    all_vertices = torch.from_numpy(normalized.astype(np.float32)).to(device)
    refined_chunks: list[np.ndarray] = []
    for start in range(0, len(all_vertices), 32_768):
        points = all_vertices[start : start + 32_768].detach().requires_grad_(True)
        sdf = model(points, 7.0)
        gradient = torch.autograd.grad(sdf.sum(), points)[0]
        step = sdf[:, None] * gradient / gradient.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)
        limit = voxel_normalized * 2.0
        scale = torch.clamp(limit / step.norm(dim=-1, keepdim=True).clamp_min(1e-8), max=1.0)
        refined_chunks.append((points - step * scale).detach().cpu().numpy())
    refined_normalized = np.concatenate(refined_chunks)
    refined = refined_normalized * normalization.radius + normalization.center
    validation = validate_candidate(vertices, triangles, refined, voxel_size_m)
    fit_accepted = held_mae <= voxel_normalized * 0.45 and held_p95 <= voxel_normalized * 1.05
    validation["accepted"] = bool(validation["accepted"] and fit_accepted)
    validation.update(
        {
            "heldOutSdfMaeM": held_mae * normalization.radius,
            "heldOutSdfP95M": held_p95 * normalization.radius,
            "trainingIterations": iterations,
            "trainingSampleCount": len(train_points),
            "heldOutSampleCount": len(held_points),
            "normalizationRadiusM": normalization.radius,
        }
    )
    return refined.astype(np.float32), validation


def run_refinement(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    progress_path: Path | None,
    iterations: int,
    device: str,
) -> dict[str, Any]:
    vertices, triangles, voxel_size_m, fingerprint = _load_input(input_path)
    refined, validation = refine_surface(
        vertices,
        triangles,
        voxel_size_m,
        iterations=iterations,
        device_name=device,
        progress_path=progress_path,
    )
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "method": NEURAL_SDF_VERSION,
        "sourceFingerprint": fingerprint,
        "device": device,
        "vertexCount": len(vertices),
        "triangleCount": len(triangles),
        "voxelSizeM": voxel_size_m,
        "validation": validation,
        "status": "accepted" if validation["accepted"] else "rejected",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary_report.replace(report_path)
    if validation["accepted"]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary_output.open("wb") as handle:
            np.savez(handle, vertices=refined, triangles=triangles)
        temporary_output.replace(output_path)
    return report
