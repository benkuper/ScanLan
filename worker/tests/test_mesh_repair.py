from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from scanlan.io import CameraModel, FrameRecord, PhaseData, RgbCameraModel
from scanlan.mesh_observations import (
    FREE_SPACE_VIOLATION,
    MISSING_DEPTH,
    OCCLUDED,
    OUTSIDE_VIEW,
    SUPPORTED,
    classify_world_points,
)
from scanlan.mesh_repair import (
    MeshRepairSettings,
    classify_boundary_loop,
    find_mesh_repair_backend,
    repair_mesh_geometry,
)


def _frame(root: Path, depth_m: float | None, *, depthless: bool = False) -> SimpleNamespace:
    width = height = 100
    camera = CameraModel(width, height, 100.0, 100.0, 49.5, 49.5, 1000.0, 8.0)
    rgb_camera = RgbCameraModel(width, height, 100.0, 100.0, 49.5, 49.5, "pinhole", ())
    depth_path = root / f"depth-{len(list(root.glob('depth-*')))}.u16"
    depth = np.zeros((height, width), dtype="<u2")
    if depth_m is not None:
        depth.fill(round(depth_m * camera.depth_scale))
    depth.tofile(depth_path)
    color_path = root / f"color-{len(list(root.glob('color-*')))}.rgb"
    np.zeros((height, width, 3), dtype=np.uint8).tofile(color_path)
    record = FrameRecord(0, 0, 0, depth_path, color_path, None, None, np.eye(4))
    phase = PhaseData(root, {}, camera, rgb_camera, np.eye(4), [record], [])
    return SimpleNamespace(
        source=phase,
        frame_index=0,
        camera_to_global=np.eye(4),
        image_y_up=False,
        depthless=depthless,
    )


def _loop(size: float = 0.03, z: float = 2.0) -> dict:
    positions = [
        [-size, -size, z],
        [size, -size, z],
        [size, size, z],
        [-size, size, z],
    ]
    return {
        "loopId": "loop-test",
        "orderedBoundaryPositions": positions,
        "perimeterM": size * 8.0,
        "approximateEnclosedAreaM2": (size * 2.0) ** 2,
        "diameterM": size * 2.0**0.5 * 2.0,
        "bestFitPlane": {"origin": [0.0, 0.0, z], "normal": [0.0, 0.0, 1.0]},
        "planeRmsResidualM": 0.0,
        "boundaryNormalCoherence": 1.0,
        "distanceFromMeshBoundingBoxBoundaryM": 0.2,
        "vertexCount": 4,
    }


def _square_ring(z: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [-0.5, -0.5, z],
            [0.5, -0.5, z],
            [0.5, 0.5, z],
            [-0.5, 0.5, z],
            [-0.03, -0.03, z],
            [0.03, -0.03, z],
            [0.03, 0.03, z],
            [-0.03, 0.03, z],
        ],
        dtype=np.float32,
    )
    triangles: list[list[int]] = []
    for index in range(4):
        following = (index + 1) % 4
        triangles.extend(
            [
                [index, following, 4 + following],
                [index, 4 + following, 4 + index],
            ]
        )
    return vertices, np.asarray(triangles, dtype=np.int64)


def test_arbitrary_point_depth_classifications(tmp_path: Path) -> None:
    frame = _frame(tmp_path, 2.0)
    points = np.asarray(
        [
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 3.0],
            [20.0, 0.0, 2.0],
        ]
    )
    evidence = classify_world_points(points, frame, 0.008)
    assert evidence.tolist() == [SUPPORTED, FREE_SPACE_VIOLATION, OCCLUDED, OUTSIDE_VIEW]


def test_planar_wall_dropout_with_matching_depth_is_filled(tmp_path: Path) -> None:
    frames = [_frame(tmp_path, 2.0), _frame(tmp_path, 2.0)]
    decision = classify_boundary_loop(_loop(), frames, 0.008, MeshRepairSettings())
    assert decision["classification"] == "fill_measured"
    assert decision["supportRatio"] == 1.0
    assert decision["supportingViewCount"] == 2


def test_doorway_with_back_wall_is_preserved_as_opening(tmp_path: Path) -> None:
    frames = [_frame(tmp_path, 3.0), _frame(tmp_path, 3.0)]
    decision = classify_boundary_loop(_loop(), frames, 0.008, MeshRepairSettings())
    assert decision["classification"] == "preserve_opening"
    assert decision["freeSpaceViolationRatio"] == 1.0


def test_patch_behind_foreground_geometry_is_preserved_occluded(tmp_path: Path) -> None:
    frames = [_frame(tmp_path, 1.0), _frame(tmp_path, 1.0)]
    decision = classify_boundary_loop(_loop(), frames, 0.008, MeshRepairSettings())
    assert decision["classification"] == "preserve_occluded"
    assert decision["occludedRatio"] == 1.0


def test_missing_depth_everywhere_is_unknown(tmp_path: Path) -> None:
    frames = [_frame(tmp_path, None), _frame(tmp_path, None)]
    decision = classify_boundary_loop(_loop(), frames, 0.008, MeshRepairSettings())
    assert decision["classification"] == "preserve_unknown"
    assert decision["evidenceCounts"][MISSING_DEPTH] > 0


def test_loop_exceeding_threshold_is_too_large(tmp_path: Path) -> None:
    frames = [_frame(tmp_path, 2.0), _frame(tmp_path, 2.0)]
    decision = classify_boundary_loop(_loop(size=0.2), frames, 0.008, MeshRepairSettings())
    assert decision["classification"] == "preserve_too_large"


def test_candidate_outside_every_camera_is_unknown(tmp_path: Path) -> None:
    loop = _loop()
    for position in loop["orderedBoundaryPositions"]:
        position[0] += 20.0
    loop["bestFitPlane"]["origin"][0] += 20.0
    frames = [_frame(tmp_path, 2.0), _frame(tmp_path, 2.0)]
    decision = classify_boundary_loop(loop, frames, 0.008, MeshRepairSettings())
    assert decision["classification"] == "preserve_unknown"
    assert decision["evidenceCounts"][OUTSIDE_VIEW] > 0


def test_depthless_photographs_never_provide_metric_support(tmp_path: Path) -> None:
    frames = [_frame(tmp_path, 2.0, depthless=True), _frame(tmp_path, 2.0, depthless=True)]
    decision = classify_boundary_loop(_loop(), frames, 0.008, MeshRepairSettings())
    assert decision["classification"] == "preserve_unknown"
    assert decision["supportingViewCount"] == 0


def test_inferred_fill_requires_explicit_opt_in(tmp_path: Path) -> None:
    frames = [_frame(tmp_path, None), _frame(tmp_path, None)]
    decision = classify_boundary_loop(
        _loop(),
        frames,
        0.008,
        MeshRepairSettings(fill_inferred_holes=True),
    )
    assert decision["classification"] == "fill_inferred"


def test_disabled_repair_preserves_geometry_and_writes_report(tmp_path: Path) -> None:
    vertices, triangles = _square_ring()
    repaired_vertices, repaired_triangles, report = repair_mesh_geometry(
        tmp_path,
        vertices,
        triangles,
        [],
        0.008,
        MeshRepairSettings(enabled=False),
    )
    assert np.array_equal(repaired_vertices, vertices)
    assert np.array_equal(repaired_triangles, triangles)
    assert report["status"] == "disabled"
    assert (tmp_path / "mesh-repair-report.json").is_file()


@pytest.mark.skipif(find_mesh_repair_backend() is None, reason="CGAL test backend not built")
def test_supported_wall_hole_is_filled_by_native_backend(tmp_path: Path) -> None:
    vertices, triangles = _square_ring()
    frames = [_frame(tmp_path, 2.0), _frame(tmp_path, 2.0)]
    repaired_vertices, repaired_triangles, report = repair_mesh_geometry(
        tmp_path,
        vertices,
        triangles,
        frames,
        0.008,
        MeshRepairSettings(),
    )
    assert report["status"] == "ok"
    assert report["repairSummary"]["holesFilled"] == 1
    assert len(repaired_vertices) >= len(vertices)
    assert len(repaired_triangles) > len(triangles)


@pytest.mark.skipif(find_mesh_repair_backend() is None, reason="CGAL test backend not built")
def test_native_repair_preserves_doorway_free_space(tmp_path: Path) -> None:
    vertices, triangles = _square_ring()
    frames = [_frame(tmp_path, 3.0), _frame(tmp_path, 3.0)]
    _, repaired_triangles, report = repair_mesh_geometry(
        tmp_path,
        vertices,
        triangles,
        frames,
        0.008,
        MeshRepairSettings(),
    )
    assert report["status"] == "ok"
    assert report["repairSummary"]["holesFilled"] == 0
    assert report["repairSummary"]["openingsPreserved"] == 1
    assert len(repaired_triangles) == len(triangles)


@pytest.mark.skipif(find_mesh_repair_backend() is None, reason="CGAL test backend not built")
def test_profile_change_reuses_raw_mesh_but_invalidates_repair_cache(tmp_path: Path) -> None:
    vertices, triangles = _square_ring()
    frames = [_frame(tmp_path, 2.0), _frame(tmp_path, 2.0)]
    repair_mesh_geometry(
        tmp_path,
        vertices,
        triangles,
        frames,
        0.008,
        MeshRepairSettings(profile="faithful"),
    )
    repair_mesh_geometry(
        tmp_path,
        vertices,
        triangles,
        frames,
        0.008,
        MeshRepairSettings(profile="architectural"),
    )
    cache = tmp_path / "cache" / "mesh-repair"
    assert len(list(cache.glob("raw-*.ply"))) == 1
    assert len(list(cache.glob("repaired-*.ply"))) == 2


def test_missing_backend_fallback_is_explicit_valid_json(tmp_path: Path) -> None:
    vertices, triangles = _square_ring()
    with patch("scanlan.mesh_repair.find_mesh_repair_backend", return_value=None):
        _, _, report = repair_mesh_geometry(
            tmp_path,
            vertices,
            triangles,
            [],
            0.008,
            MeshRepairSettings(),
        )
    saved = json.loads((tmp_path / "mesh-repair-report.json").read_text(encoding="utf-8"))
    assert report["status"] == "fallback"
    assert saved["repairSummary"]["fallbackOccurred"] is True
