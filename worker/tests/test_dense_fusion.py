from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scanlan.dense_fusion import (
    PROVENANCE_LEARNED,
    PROVENANCE_MEASURED,
    DenseSamples,
    align_media_samples,
    fuse_dense_samples,
    load_dense_samples,
    samples_from_arrays,
    publish_media_dense_artifacts,
)


def _learned_samples(points: np.ndarray, owners: np.ndarray) -> DenseSamples:
    count = len(points)
    return DenseSamples(
        points.astype(np.float32),
        np.full((count, 3), 128, dtype=np.uint8),
        np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (count, 1)),
        np.full((count, 3), 0.02, dtype=np.float32),
        np.full(count, 0.8, dtype=np.float32),
        np.full(count, PROVENANCE_LEARNED, dtype=np.uint8),
        owners.astype(np.int32),
    )


def test_measured_surface_wins_a_voxel_conflict() -> None:
    measured = samples_from_arrays(
        np.asarray([[0.001, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[255, 0, 0]], dtype=np.uint8),
        provenance=PROVENANCE_MEASURED,
    )
    learned = _learned_samples(
        np.asarray([[0.009, 0.0, 0.0]], dtype=np.float32),
        np.asarray([0]),
    )

    fused = fuse_dense_samples([learned, measured], 0.01)

    np.testing.assert_allclose(fused.points, measured.points)
    np.testing.assert_array_equal(fused.colors, measured.colors)
    np.testing.assert_array_equal(fused.provenance, [PROVENANCE_MEASURED])


def test_media_alignment_uses_independently_localized_camera_similarity() -> None:
    source_centers = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.2]]
    )
    angle = np.deg2rad(18.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    scale = 2.4
    translation = np.asarray([3.0, -1.0, 0.7])
    frames = []
    targets = []
    for index, center in enumerate(source_centers):
        source_pose = np.eye(4)
        source_pose[:3, 3] = center
        source_pose[:3, :3] = np.eye(3)
        target_pose = np.eye(4)
        target_pose[:3, 3] = scale * (rotation @ center) + translation
        target_pose[:3, :3] = rotation
        path = str(Path(f"view-{index}.jpg").resolve())
        frames.append(
            {
                "sourcePath": path,
                "timestampUs": None,
                "sourceFrameIndex": index,
                "worldFromRgbCamera": source_pose.reshape(-1).tolist(),
            }
        )
        targets.append(
            SimpleNamespace(
                media_source_path=path,
                media_timestamp_seconds=None,
                display_axes=(1.0, 1.0, 1.0),
                camera_to_global=target_pose,
                localization_inliers=140,
                localization_rmse_px=0.5,
            )
        )
    samples = _learned_samples(source_centers, np.arange(len(source_centers)))

    aligned, report = align_media_samples(samples, {"frames": frames}, targets)

    np.testing.assert_allclose(
        aligned.points,
        scale * (source_centers @ rotation.T) + translation,
        atol=1e-5,
    )
    assert report["inlierCameraCount"] == 4
    assert report["normalizedMedianCameraResidual"] < 1e-6
    assert report["medianRotationErrorDegrees"] < 1e-5


def test_dense_sidecar_preserves_confidence_ownership_and_provenance(tmp_path: Path) -> None:
    dataset_root = tmp_path / "media-test"
    dataset_root.mkdir()
    (dataset_root / "dataset.json").write_text(
        json.dumps(
            {
                "metric": False,
                "initializationParameters": "initialization-parameters.npz",
                "frames": [],
            }
        ),
        encoding="utf-8",
    )
    np.savez(
        dataset_root / "initialization-parameters.npz",
        points=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        colors=np.asarray([[10, 20, 30]], dtype=np.uint8),
        scales=np.asarray([[0.01, 0.02, 0.001]], dtype=np.float32),
        quaternions=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        fusion_confidence=np.asarray([0.72], dtype=np.float32),
        source_frame_indices=np.asarray([7], dtype=np.int32),
        provenance=np.asarray([PROVENANCE_LEARNED], dtype=np.uint8),
    )
    pointer = tmp_path / "media-current.json"
    pointer.write_text(json.dumps({"path": "media-test"}), encoding="utf-8")

    samples, _dataset, resolved = load_dense_samples(pointer)

    assert resolved == dataset_root
    np.testing.assert_allclose(samples.confidence, [0.72])
    np.testing.assert_array_equal(samples.source_frame_indices, [7])
    np.testing.assert_array_equal(samples.provenance, [PROVENANCE_LEARNED])
    np.testing.assert_allclose(samples.normals, [[0.0, 0.0, 1.0]])


def test_media_fusion_publishes_loadable_point_and_mesh_bundle(tmp_path: Path) -> None:
    project = tmp_path / "project"
    dataset_root = project / "outputs" / "cache" / "datasets" / "media-sphere"
    dataset_root.mkdir(parents=True)
    (project / "project.json").write_text(
        json.dumps({"schemaVersion": 3, "settings": {}, "phases": [], "artifacts": {}}),
        encoding="utf-8",
    )
    latitude = np.linspace(0.12, np.pi - 0.12, 32)
    longitude = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    phi, theta = np.meshgrid(latitude, longitude, indexing="ij")
    points = np.column_stack(
        (
            np.sin(phi).reshape(-1) * np.cos(theta).reshape(-1),
            np.cos(phi).reshape(-1),
            np.sin(phi).reshape(-1) * np.sin(theta).reshape(-1),
        )
    ).astype(np.float32)
    colors = np.rint((points + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    quaternion_w = np.sqrt(np.maximum(1e-8, (1.0 + points[:, 2]) * 0.5))
    quaternions = np.column_stack(
        (
            quaternion_w,
            -points[:, 1] / (2.0 * quaternion_w),
            points[:, 0] / (2.0 * quaternion_w),
            np.zeros(len(points)),
        )
    ).astype(np.float32)
    np.savez(
        dataset_root / "initialization-parameters.npz",
        points=points,
        colors=colors,
        scales=np.full((len(points), 3), 0.055, dtype=np.float32),
        quaternions=quaternions,
        fusion_confidence=np.full(len(points), 0.9, dtype=np.float32),
        source_frame_indices=np.zeros(len(points), dtype=np.int32),
        provenance=np.full(len(points), PROVENANCE_LEARNED, dtype=np.uint8),
    )
    (dataset_root / "dataset.json").write_text(
        json.dumps(
            {
                "schemaVersion": 3,
                "fingerprint": "sphere",
                "metric": False,
                "sourceType": "photos",
                "initializationParameters": "initialization-parameters.npz",
                "frames": [],
            }
        ),
        encoding="utf-8",
    )
    pointer = dataset_root.parent / "media-current.json"
    pointer.write_text(json.dumps({"path": "media-sphere"}), encoding="utf-8")

    result = publish_media_dense_artifacts(
        project, pointer, ("point_cloud", "textured_mesh")
    )

    assert result["pointCount"] > 1_000
    assert result["meshTriangleCount"] > 100
    assert (project / "outputs" / "room-cloud.ply").read_bytes().startswith(b"ply\n")
    obj = (project / "outputs" / "room-mesh.obj").read_text(encoding="utf-8")
    assert "mtllib room-mesh.mtl" in obj
    assert "\nf " in obj
    assert (project / "outputs" / "room-texture.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    published = json.loads((project / "project.json").read_text(encoding="utf-8"))
    assert published["artifacts"]["pointCloud"]["metric"] is False
    assert published["artifacts"]["texturedMesh"]["metric"] is False
