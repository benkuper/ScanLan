from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .io import PhaseData, load_color, load_depth


@dataclass(frozen=True)
class ComputeBackend:
    key: str
    label: str
    device: Any | None
    uses_cuda: bool


def select_compute_backend(o3d: Any, requested: str = "auto") -> ComputeBackend:
    requested = os.environ.get("SCANLAN_DEVICE", requested).strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("SCANLAN_DEVICE must be auto, cpu, or cuda")

    build_has_cuda = bool(getattr(o3d, "_build_config", {}).get("BUILD_CUDA_MODULE", False))
    cuda_available = build_has_cuda and o3d.core.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA reconstruction was requested, but this Open3D build has no usable CUDA device. "
            "Install the CUDA-enabled Open3D worker or use --device cpu."
        )
    if requested != "cpu" and cuda_available:
        device = o3d.core.Device("CUDA:0")
        # Force a tiny allocation and kernel launch now so an incompatible wheel
        # fails before a long reconstruction has started.
        probe = o3d.core.Tensor([1.0], dtype=o3d.core.Dtype.Float32, device=device)
        float((probe + 1.0).cpu().numpy()[0])
        return ComputeBackend(
            key=f"open3d-{o3d.__version__}-cuda0",
            label="Open3D CUDA:0 + CPU hybrid",
            device=device,
            uses_cuda=True,
        )
    return ComputeBackend(
        key=f"open3d-{o3d.__version__}-cpu",
        label="Open3D CPU (OpenMP)",
        device=None,
        uses_cuda=False,
    )


def tensor_intrinsic(o3d: Any, phase: PhaseData, device: Any) -> Any:
    camera = phase.camera
    return o3d.core.Tensor(
        [
            [camera.fx, 0.0, camera.cx],
            [0.0, camera.fy, camera.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=o3d.core.Dtype.Float64,
        device=device,
    )


def tensor_rgbd(o3d: Any, phase: PhaseData, frame_index: int, device: Any) -> Any:
    frame = phase.frames[frame_index]
    color = np.ascontiguousarray(load_color(frame, phase.camera))
    depth = np.ascontiguousarray(load_depth(frame, phase.camera))
    color_image = o3d.t.geometry.Image(o3d.core.Tensor(color, device=device))
    depth_image = o3d.t.geometry.Image(o3d.core.Tensor(depth, device=device))
    return o3d.t.geometry.RGBDImage(color_image, depth_image, True)


def tensor_odometry(
    o3d: Any,
    source: Any,
    target: Any,
    phase: PhaseData,
    initial: np.ndarray,
    depth_difference_max: float,
    backend: ComputeBackend,
) -> np.ndarray:
    assert backend.device is not None
    result = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
        source,
        target,
        tensor_intrinsic(o3d, phase, backend.device),
        o3d.core.Tensor(
            np.ascontiguousarray(initial),
            dtype=o3d.core.Dtype.Float64,
            device=backend.device,
        ),
        depth_scale=phase.camera.depth_scale,
        depth_max=phase.camera.max_depth_m,
        method=o3d.t.pipelines.odometry.Method.Hybrid,
        params=o3d.t.pipelines.odometry.OdometryLossParams(
            depth_outlier_trunc=depth_difference_max,
            depth_huber_delta=min(depth_difference_max, 0.05),
            intensity_huber_delta=0.10,
        ),
    )
    transformation = result.transformation.cpu().numpy()
    if not np.isfinite(transformation).all():
        raise RuntimeError("CUDA RGB-D odometry produced a non-finite transform")
    return transformation


TsdfEntry = tuple[PhaseData, int, np.ndarray]
IntegrationCallback = Callable[[int, bool], None]
FallbackCallback = Callable[[str], None]


def _legacy_tsdf(
    o3d: Any,
    entries: Sequence[TsdfEntry],
    voxel_size_m: float,
    sdf_trunc_m: float,
    on_integrated: IntegrationCallback | None,
    already_reported: int = 0,
) -> Any:
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size_m,
        sdf_trunc=sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    intrinsic_cache: dict[str, Any] = {}
    for entry_index, (phase, frame_index, extrinsic) in enumerate(entries):
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
        frame = phase.frames[frame_index]
        color = np.ascontiguousarray(load_color(frame, phase.camera))
        depth = np.ascontiguousarray(load_depth(frame, phase.camera))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth),
            depth_scale=phase.camera.depth_scale,
            depth_trunc=phase.camera.max_depth_m,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, extrinsic)
        if on_integrated:
            on_integrated(entry_index, entry_index < already_reported)
    return volume.extract_point_cloud()


def _tensor_tsdf(
    o3d: Any,
    entries: Sequence[TsdfEntry],
    voxel_size_m: float,
    sdf_trunc_m: float,
    backend: ComputeBackend,
    on_integrated: IntegrationCallback | None,
) -> Any:
    assert backend.device is not None
    host = o3d.core.Device("CPU:0")
    # UInt16 weights and colors use the optimized Open3D kernels while keeping
    # a 50k-block room volume comfortably inside a 16 GB GPU.
    volume = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(
            o3d.core.Dtype.Float32,
            o3d.core.Dtype.UInt16,
            o3d.core.Dtype.UInt16,
        ),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel_size_m,
        block_resolution=16,
        block_count=50_000,
        device=backend.device,
    )
    intrinsic_cache: dict[str, Any] = {}
    truncation_multiplier = sdf_trunc_m / voxel_size_m
    for entry_index, (phase, frame_index, extrinsic_matrix) in enumerate(entries):
        phase_key = str(phase.root)
        intrinsic = intrinsic_cache.get(phase_key)
        if intrinsic is None:
            # Open3D keeps camera matrices on the host even when images, voxel
            # blocks, hashing, and integration kernels live on CUDA.
            intrinsic = tensor_intrinsic(o3d, phase, host)
            intrinsic_cache[phase_key] = intrinsic
        rgbd = tensor_rgbd(o3d, phase, frame_index, backend.device)
        extrinsic = o3d.core.Tensor(
            np.ascontiguousarray(extrinsic_matrix),
            dtype=o3d.core.Dtype.Float64,
            device=host,
        )
        block_coordinates = volume.compute_unique_block_coordinates(
            rgbd.depth,
            intrinsic,
            extrinsic,
            phase.camera.depth_scale,
            phase.camera.max_depth_m,
            truncation_multiplier,
        )
        volume.integrate(
            block_coordinates,
            rgbd.depth,
            rgbd.color,
            intrinsic,
            extrinsic,
            phase.camera.depth_scale,
            phase.camera.max_depth_m,
            truncation_multiplier,
        )
        if on_integrated:
            on_integrated(entry_index, False)
    o3d.core.cuda.synchronize(backend.device)
    return volume.extract_point_cloud(weight_threshold=1.0).cpu().to_legacy()


def integrate_tsdf(
    o3d: Any,
    entries: Sequence[TsdfEntry],
    voxel_size_m: float,
    sdf_trunc_m: float,
    backend: ComputeBackend,
    on_integrated: IntegrationCallback | None = None,
    on_fallback: FallbackCallback | None = None,
) -> Any:
    if not backend.uses_cuda:
        return _legacy_tsdf(
            o3d,
            entries,
            voxel_size_m,
            sdf_trunc_m,
            on_integrated,
        )
    reported = 0

    def track_reported(index: int, repeated: bool) -> None:
        nonlocal reported
        reported = max(reported, index + 1)
        if on_integrated:
            on_integrated(index, repeated)

    try:
        return _tensor_tsdf(
            o3d,
            entries,
            voxel_size_m,
            sdf_trunc_m,
            backend,
            track_reported,
        )
    except RuntimeError as error:
        if on_fallback:
            on_fallback(str(error).splitlines()[0])
        return _legacy_tsdf(
            o3d,
            entries,
            voxel_size_m,
            sdf_trunc_m,
            on_integrated,
            already_reported=reported,
        )


def _legacy_surfel_merge(
    o3d: Any,
    entries: Sequence[TsdfEntry],
    voxel_size_m: float,
    on_merged: IntegrationCallback | None,
    already_reported: int = 0,
) -> Any:
    """CPU fallback for fine-spacing point-cloud fusion."""
    cloud = o3d.geometry.PointCloud()
    intrinsic_cache: dict[str, Any] = {}
    for entry_index, (phase, frame_index, extrinsic) in enumerate(entries):
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
        frame = phase.frames[frame_index]
        color = np.ascontiguousarray(load_color(frame, phase.camera))
        depth = np.ascontiguousarray(load_depth(frame, phase.camera))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth),
            depth_scale=phase.camera.depth_scale,
            depth_trunc=phase.camera.max_depth_m,
            convert_rgb_to_intensity=False,
        )
        frame_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        frame_cloud.transform(np.linalg.inv(extrinsic))
        cloud += frame_cloud
        if (entry_index + 1) % 6 == 0:
            cloud = cloud.voxel_down_sample(voxel_size_m)
        if on_merged:
            on_merged(entry_index, entry_index < already_reported)
    return cloud.voxel_down_sample(voxel_size_m)


def _tensor_surfel_merge(
    o3d: Any,
    entries: Sequence[TsdfEntry],
    voxel_size_m: float,
    backend: ComputeBackend,
    on_merged: IntegrationCallback | None,
) -> Any:
    """Fuse fine RGB-D surfels into a streaming voxel hash on one device.

    Each frame is reduced to one sample per output voxel before it reaches the
    persistent map. Consequently, integration work is proportional to the new
    frame instead of repeatedly downsampling the entire accumulated room.
    """
    assert backend.device is not None
    device = backend.device
    host = o3d.core.Device("CPU:0")
    largest_frame = max(
        (phase.camera.width * phase.camera.height for phase, _, _ in entries),
        default=1,
    )
    initial_capacity = min(max(1_000_000, largest_frame * 4), 8_000_000)
    voxel_map = o3d.core.HashMap(
        initial_capacity,
        o3d.core.Dtype.Int32,
        (3,),
        o3d.core.Dtype.Float32,
        (7,),
        device,
    )
    intrinsic_cache: dict[str, Any] = {}

    for entry_index, (phase, frame_index, extrinsic_matrix) in enumerate(entries):
        phase_key = str(phase.root)
        intrinsic = intrinsic_cache.get(phase_key)
        if intrinsic is None:
            intrinsic = tensor_intrinsic(o3d, phase, host)
            intrinsic_cache[phase_key] = intrinsic
        extrinsic = o3d.core.Tensor(
            np.ascontiguousarray(extrinsic_matrix),
            dtype=o3d.core.Dtype.Float64,
            device=host,
        )
        frame_cloud = o3d.t.geometry.PointCloud.create_from_rgbd_image(
            tensor_rgbd(o3d, phase, frame_index, device),
            intrinsic,
            extrinsic,
            phase.camera.depth_scale,
            phase.camera.max_depth_m,
        )
        # The second pass makes the output idempotent at floating-point voxel
        # boundaries, guaranteeing unique keys for the indexed accumulation.
        frame_cloud = frame_cloud.voxel_down_sample(voxel_size_m)
        frame_cloud = frame_cloud.voxel_down_sample(voxel_size_m)
        point_count = int(frame_cloud.point.positions.shape[0])
        if point_count:
            keys = (
                (frame_cloud.point.positions / voxel_size_m)
                .floor()
                .to(o3d.core.Dtype.Int32)
            )
            maximum_size = voxel_map.size() + point_count
            if maximum_size > voxel_map.capacity():
                voxel_map.reserve(
                    max(maximum_size, int(voxel_map.capacity() * 1.5))
                )
            zeros = o3d.core.Tensor.zeros(
                (point_count, 7), o3d.core.Dtype.Float32, device
            )
            voxel_map.insert(keys, zeros)
            buffer_indices, found = voxel_map.find(keys)
            if not bool(found.all().item()):
                raise RuntimeError("Fine-cloud voxel insertion was incomplete")
            ones = o3d.core.Tensor.ones(
                (point_count, 1), o3d.core.Dtype.Float32, device
            )
            updates = o3d.core.concatenate(
                [frame_cloud.point.positions, frame_cloud.point.colors, ones],
                axis=1,
            )
            values = voxel_map.value_tensor()
            values[buffer_indices] = values[buffer_indices] + updates
        if str(device).startswith("CUDA"):
            o3d.core.cuda.synchronize(device)
        if on_merged:
            on_merged(entry_index, False)

    if voxel_map.size() == 0:
        return o3d.geometry.PointCloud()
    active = voxel_map.active_buf_indices()
    accumulated = voxel_map.value_tensor()[active]
    weights = accumulated[:, 6:7]
    result = o3d.t.geometry.PointCloud(accumulated[:, 0:3] / weights)
    result.point.colors = accumulated[:, 3:6] / weights
    if str(device).startswith("CUDA"):
        o3d.core.cuda.synchronize(device)
    return result.cpu().to_legacy()


def merge_surfel_cloud(
    o3d: Any,
    entries: Sequence[TsdfEntry],
    voxel_size_m: float,
    backend: ComputeBackend,
    on_merged: IntegrationCallback | None = None,
    on_fallback: FallbackCallback | None = None,
) -> Any:
    """Build a fine-spaced cloud on CUDA, with the original CPU path as fallback."""
    if not backend.uses_cuda:
        return _legacy_surfel_merge(o3d, entries, voxel_size_m, on_merged)
    reported = 0

    def track_reported(index: int, repeated: bool) -> None:
        nonlocal reported
        reported = max(reported, index + 1)
        if on_merged:
            on_merged(index, repeated)

    try:
        return _tensor_surfel_merge(
            o3d, entries, voxel_size_m, backend, track_reported
        )
    except RuntimeError as error:
        if on_fallback:
            on_fallback(str(error).splitlines()[0])
        return _legacy_surfel_merge(
            o3d,
            entries,
            voxel_size_m,
            on_merged,
            already_reported=reported,
        )


def tensor_refine_registration(
    o3d: Any,
    source: Any,
    target: Any,
    initial: np.ndarray,
    backend: ComputeBackend,
) -> tuple[Any, Any, Any]:
    assert backend.device is not None
    source_tensor = o3d.t.geometry.PointCloud.from_legacy(
        source, o3d.core.Dtype.Float32, backend.device
    )
    target_tensor = o3d.t.geometry.PointCloud.from_legacy(
        target, o3d.core.Dtype.Float32, backend.device
    )
    source_tensor.estimate_normals(max_nn=50, radius=0.18)
    target_tensor.estimate_normals(max_nn=50, radius=0.18)
    initial_tensor = o3d.core.Tensor(
        np.ascontiguousarray(initial),
        dtype=o3d.core.Dtype.Float64,
        # Open3D registration results and transforms are always host tensors;
        # point clouds and nearest-neighbor kernels remain on CUDA.
        device=o3d.core.Device("CPU:0"),
    )
    geometric = o3d.t.pipelines.registration.multi_scale_icp(
        source_tensor,
        target_tensor,
        o3d.utility.DoubleVector([0.06, 0.035, 0.025]),
        [
            o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=35),
            o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=45),
            o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=35),
        ],
        o3d.utility.DoubleVector([0.16, 0.085, 0.05]),
        initial_tensor,
        o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    transformation = geometric.transformation
    try:
        colored = o3d.t.pipelines.registration.icp(
            source_tensor,
            target_tensor,
            0.05,
            transformation,
            o3d.t.pipelines.registration.TransformationEstimationForColoredICP(0.94),
            o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=20),
            voxel_size=0.025,
        )
        if colored.fitness >= geometric.fitness * 0.98 and colored.inlier_rmse <= geometric.inlier_rmse:
            transformation = colored.transformation
    except RuntimeError:
        pass
    o3d.core.cuda.synchronize(backend.device)

    source_fine = source.voxel_down_sample(0.025)
    target_fine = target.voxel_down_sample(0.025)
    if len(source_fine.points) > 120_000:
        source_fine = source_fine.uniform_down_sample(
            int(np.ceil(len(source_fine.points) / 120_000))
        )
    if len(target_fine.points) > 120_000:
        target_fine = target_fine.uniform_down_sample(
            int(np.ceil(len(target_fine.points) / 120_000))
        )
    evaluated = o3d.pipelines.registration.evaluate_registration(
        source_fine,
        target_fine,
        0.05,
        transformation.cpu().numpy(),
    )
    return evaluated, source_fine, target_fine
