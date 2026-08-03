from __future__ import annotations

import importlib.util
import re
import shutil
from pathlib import Path


FEATURE_STAMP = "2dgs-rgbd-v3"
COMPATIBLE_EXTENSION_STAMPS = {"2dgs-rgbd-v2", FEATURE_STAMP}


def _replace_between(text: str, start: str, end: str, replacement: str) -> str:
    start_index = text.find(start)
    if start_index < 0:
        return text
    end_index = text.find(end, start_index)
    if end_index < 0:
        raise RuntimeError(f"gsplat source marker is missing: {end}")
    return text[:start_index] + replacement + text[end_index:]


def _limit_2dgs_channels(text: str, channels: tuple[int, ...]) -> str:
    calls = "".join(f"__INS__({channel})\n" for channel in channels)
    return re.sub(
        r"(?m)(?:^__INS__\(\d+\)\r?\n)+",
        calls,
        text,
    )


def _limit_2dgs_switches(text: str, channels: tuple[int, ...]) -> str:
    start = text.find("////////////////////////////////////////////////////\n// 2DGS")
    end = text.find("////////////////////////////////////////////////////\n// 3DGS (from world)", start)
    if end < 0:
        end = text.find("// ScanLan omits the unused 3DGUT from-world wrappers.", start)
    if start < 0 or end < 0:
        raise RuntimeError("gsplat 2DGS rasterization markers are missing")
    section = text[start:end]
    calls = "".join(f"        __LAUNCH_KERNEL__({channel})\n" for channel in channels)
    section = re.sub(
        r"(?m)(?:^\s*__LAUNCH_KERNEL__\(\d+\)\r?\n)+",
        calls,
        section,
    )
    return text[:start] + section + text[end:]


def _fix_2dgs_bwd_instantiation(text: str) -> str:
    """Match gsplat's 2DGS backward instantiation to its public declaration.

    gsplat 1.5.3 marks the six output tensors const only in its explicit
    instantiation. MSVC includes that top-level qualifier in the CUDA object's
    symbol name, so Rasterization.cpp cannot link against it on Windows.
    """
    start = text.find("#define __INS__(CDIM)")
    end = text.find("#undef __INS__", start)
    if start < 0 or end < 0:
        raise RuntimeError("gsplat 2DGS backward instantiation marker is missing")
    section = text[start:end]
    for parameter in (
        "v_means2d",
        "v_ray_transforms",
        "v_colors",
        "v_opacities",
        "v_normals",
        "v_densify",
    ):
        section = section.replace(
            f"const at::Tensor {parameter}",
            f"at::Tensor {parameter}",
        )
    return text[:start] + section + text[end:]


def _fix_2dgs_packed_spherical_harmonics(text: str) -> str:
    """Gather SH coefficients for packed 2DGS projections in gsplat 1.5.3."""
    old = """        if packed:
            dirs = means[..., gaussian_ids, :] - camtoworlds[..., camera_ids, :3, 3]
        else:
            dirs = means[..., None, :, :] - camtoworlds[..., None, :3, 3]

        if colors.dim() == num_batch_dims + 3:
            # Turn [..., N, K, 3] into [..., C, N, K, 3]
            shs = torch.broadcast_to(
                colors[..., None, :, :, :], batch_dims + (C, N, -1, 3)
            )  # [..., C, N, K, 3]
        else:
            # colors is already [..., C, N, K, 3]
            shs = colors
"""
    new = """        if packed:
            means_flat = means.reshape(B, N, 3)
            cameras_flat = camtoworlds.reshape(B, C, 4, 4)
            dirs = (
                means_flat[batch_ids, gaussian_ids]
                - cameras_flat[batch_ids, camera_ids, :3, 3]
            )
        else:
            dirs = means[..., None, :, :] - camtoworlds[..., None, :3, 3]

        if colors.dim() == num_batch_dims + 3:
            if packed:
                shs = colors.reshape(B, N, -1, 3)[batch_ids, gaussian_ids]
            else:
                # Turn [..., N, K, 3] into [..., C, N, K, 3]
                shs = torch.broadcast_to(
                    colors[..., None, :, :, :], batch_dims + (C, N, -1, 3)
                )  # [..., C, N, K, 3]
        else:
            if packed:
                shs = colors.reshape(B, C, N, -1, 3)[
                    batch_ids, camera_ids, gaussian_ids
                ]
            else:
                # colors is already [..., C, N, K, 3]
                shs = colors
"""
    if old in text:
        return text.replace(old, new, 1)
    if "means_flat = means.reshape(B, N, 3)" not in text:
        raise RuntimeError("gsplat packed 2DGS spherical-harmonics marker is missing")
    return text


def patch_scanlan_sources(package_root: Path) -> None:
    # ScanLan renders RGB plus expected depth, so the 2DGS pixel kernels only
    # need the three-channel diagnostic path and four-channel training path.
    # Avoiding hundreds of unused template instantiations keeps the CUDA 13
    # Windows link reliable and materially shortens the extension build.
    two_dgs_channels = (3, 4)
    cuda_root = package_root / "cuda"
    extension = cuda_root / "ext.cpp"
    extension_text = extension.read_text(encoding="utf-8")
    extension_text = _replace_between(
        extension_text,
        '    m.def("rasterize_to_pixels_from_world_3dgs_fwd"',
        "    // Cameras from 3DGUT",
        "    // ScanLan uses the standard 3DGS rasterizer; the optional from-world\n"
        "    // entry points are omitted because their channel templates do not link\n"
        "    // with CUDA 13 on Windows.\n\n",
    )
    extension.write_text(extension_text, encoding="utf-8", newline="\n")

    rasterization = cuda_root / "csrc" / "Rasterization.cpp"
    rasterization_text = rasterization.read_text(encoding="utf-8")
    rasterization_text = _limit_2dgs_switches(rasterization_text, two_dgs_channels)
    rasterization_text = _replace_between(
        rasterization_text,
        "////////////////////////////////////////////////////\n// 3DGS (from world)",
        "} // namespace gsplat",
        "// ScanLan omits the unused 3DGUT from-world wrappers.\n\n",
    )
    rasterization.write_text(rasterization_text, encoding="utf-8", newline="\n")

    for source_name in ("RasterizeToPixels2DGSFwd.cu", "RasterizeToPixels2DGSBwd.cu"):
        source = cuda_root / "csrc" / source_name
        source_text = _limit_2dgs_channels(
            source.read_text(encoding="utf-8"), two_dgs_channels
        )
        if source_name == "RasterizeToPixels2DGSBwd.cu":
            source_text = _fix_2dgs_bwd_instantiation(source_text)
        source.write_text(source_text, encoding="utf-8", newline="\n")

    backend = cuda_root / "_backend.py"
    backend_text = backend.read_text(encoding="utf-8")
    backend_text = backend_text.replace('name = "gsplat_cuda"', 'name = "csrc"')
    backend_text = backend_text.replace(
        'extra_cflags = [opt_level, "-Wno-attributes"]',
        'extra_cflags = ["/O2"] if os.name == "nt" else [opt_level, "-Wno-attributes"]',
    )
    backend_text = backend_text.replace(
        "            shutil.rmtree(build_dir)",
        "            os.makedirs(build_dir, exist_ok=True)",
    )
    source_marker = '            + [os.path.join(PATH, "ext.cpp")]\n        )'
    source_replacement = (
        '            + [os.path.join(PATH, "ext.cpp")]\n'
        "        )\n"
        "        sources = [\n"
        "            source for source in sources\n"
        "            if 'RasterizeToPixelsFromWorld3DGS' not in os.path.basename(source)\n"
        "        ]"
    )
    if source_marker in backend_text:
        backend_text = backend_text.replace(source_marker, source_replacement)
    elif "if 'RasterizeToPixelsFromWorld3DGS' not in" not in backend_text:
        raise RuntimeError("gsplat backend source-list marker is missing")
    backend.write_text(backend_text, encoding="utf-8", newline="\n")

    rendering = package_root / "rendering.py"
    rendering.write_text(
        _fix_2dgs_packed_spherical_harmonics(
            rendering.read_text(encoding="utf-8")
        ),
        encoding="utf-8",
        newline="\n",
    )


def _write_feature_stamp(package_root: Path) -> None:
    (package_root / "scanlan-build.txt").write_text(
        f"{FEATURE_STAMP}\n", encoding="ascii", newline="\n"
    )


def main() -> None:
    specification = importlib.util.find_spec("gsplat")
    if specification is None or specification.submodule_search_locations is None:
        raise RuntimeError("gsplat is not installed")
    package_root = Path(next(iter(specification.submodule_search_locations))).resolve()
    compiled = package_root / "csrc.pyd"
    stamp = package_root / "scanlan-build.txt"
    if compiled.is_file():
        installed_features = stamp.read_text(encoding="ascii").strip() if stamp.is_file() else ""
        if installed_features in COMPATIBLE_EXTENSION_STAMPS:
            patch_scanlan_sources(package_root)
            _write_feature_stamp(package_root)
            print(compiled)
            return
        compiled.unlink()
    patch_scanlan_sources(package_root)

    from gsplat.cuda._backend import _C

    extension_path = Path(_C.__file__).resolve()
    if extension_path.suffix.lower() != ".pyd":
        raise RuntimeError(f"Unexpected gsplat extension output: {extension_path}")
    shutil.copy2(extension_path, compiled)
    _write_feature_stamp(package_root)
    print(compiled)


if __name__ == "__main__":
    main()
