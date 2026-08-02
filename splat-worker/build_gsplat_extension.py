from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path


def _replace_between(text: str, start: str, end: str, replacement: str) -> str:
    start_index = text.find(start)
    if start_index < 0:
        return text
    end_index = text.find(end, start_index)
    if end_index < 0:
        raise RuntimeError(f"gsplat source marker is missing: {end}")
    return text[:start_index] + replacement + text[end_index:]


def patch_3d_only_sources(package_root: Path) -> None:
    cuda_root = package_root / "cuda"
    extension = cuda_root / "ext.cpp"
    extension_text = extension.read_text(encoding="utf-8")
    extension_text = _replace_between(
        extension_text,
        '    m.def("projection_2dgs_fused_fwd"',
        '    m.def("projection_ut_3dgs_fused"',
        "    // ScanLan packages only the 3DGS API. The CUDA 13 Windows linker\n"
        "    // cannot resolve gsplat 1.5.3's unused 2DGS channel templates.\n\n",
    )
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
    rasterization_text = _replace_between(
        rasterization_text,
        "////////////////////////////////////////////////////\n// 2DGS",
        "////////////////////////////////////////////////////\n// 3DGS (from world)",
        "// ScanLan 3D-only build: 2DGS wrappers intentionally omitted.\n\n",
    )
    rasterization_text = _replace_between(
        rasterization_text,
        "////////////////////////////////////////////////////\n// 3DGS (from world)",
        "} // namespace gsplat",
        "// ScanLan standard-3DGS build: from-world wrappers intentionally omitted.\n\n",
    )
    rasterization.write_text(rasterization_text, encoding="utf-8", newline="\n")

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
        "            if not any(name in os.path.basename(source) for name in (\n"
        "                'RasterizeToIndices2DGS',\n"
        "                'RasterizeToPixels2DGS',\n"
        "                'RasterizeToPixelsFromWorld3DGS',\n"
        "            ))\n"
        "        ]"
    )
    if source_marker in backend_text:
        backend_text = backend_text.replace(source_marker, source_replacement)
    elif "RasterizeToPixels2DGS" not in backend_text:
        raise RuntimeError("gsplat backend source-list marker is missing")
    if "RasterizeToPixelsFromWorld3DGS" not in backend_text:
        backend_text = backend_text.replace(
            "                'RasterizeToPixels2DGS',\n",
            "                'RasterizeToPixels2DGS',\n"
            "                'RasterizeToPixelsFromWorld3DGS',\n",
        )
    backend.write_text(backend_text, encoding="utf-8", newline="\n")


def main() -> None:
    specification = importlib.util.find_spec("gsplat")
    if specification is None or specification.submodule_search_locations is None:
        raise RuntimeError("gsplat is not installed")
    package_root = Path(next(iter(specification.submodule_search_locations))).resolve()
    compiled = package_root / "csrc.pyd"
    if compiled.is_file():
        print(compiled)
        return
    patch_3d_only_sources(package_root)

    from gsplat.cuda._backend import _C

    extension_path = Path(_C.__file__).resolve()
    if extension_path.suffix.lower() != ".pyd":
        raise RuntimeError(f"Unexpected gsplat extension output: {extension_path}")
    shutil.copy2(extension_path, compiled)
    print(compiled)


if __name__ == "__main__":
    main()
