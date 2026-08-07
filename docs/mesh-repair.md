# Mesh repair

ScanLan repairs the fused triangle mesh with a separate CGAL executable before normals, UVs, atlas generation, and texture projection. The Python observation layer checks every candidate patch against the original metric depth frames, so a missing wall sample can be filled while a doorway, window, scan boundary, occluded cavity, or unsupported boundary stays open.

Repair is enabled by default with the **Faithful** profile. It can be disabled to retain the unrepaired geometry. A failed or missing native backend is reported as an explicit fallback; it is never presented as a successful repair.

## Profiles and controls

- **Faithful** triangulates only authorized holes and never smooths or repositions existing valid vertices.
- **Architectural** refines authorized patches and projects only their new interior vertices to the boundary's fitted plane.
- **Natural** refines and fairs only new patch vertices to better meet freeform surroundings.
- **Fill inferred holes** additionally permits small, geometrically coherent holes without direct measured support. It is off by default.
- **Produce watertight copy** writes a separate `outputs/room-mesh-watertight.ply`. It never replaces the textured master and may seal intentional openings.

The primary mesh always uses the conservative depth-aware policy. The default maximum hole diameter is `clamp(12 * mesh voxel size, 0.04, 0.15)` metres. A measured fill requires at least 60% support from at least two RGB-D views and no more than 1% free-space evidence. Depthless supplemental photographs never count as metric support.

## Pipeline and outputs

The fused mesh cache is independent from repair settings. Repair uses a second cache fingerprint containing the raw mesh fingerprint, settings, algorithm version, metric voxel size, and relevant camera/depth dataset fingerprint. Changing a repair profile therefore reruns repair without recomputing TSDF fusion.

The build writes:

- `outputs/mesh-repair-report.json` — classification evidence, native operations, validation topology, summary, backend version, and any fallback reason;
- `outputs/cache/mesh-repair/raw-*.ply` — serialized fused geometry;
- `outputs/cache/mesh-repair/repaired-*.ply` — repaired geometry keyed by repair inputs;
- `outputs/room-mesh-watertight.ply` — optional separate derivative.

Every accepted patch records support ratio, free-space violation ratio, supporting-view count, geometric classification, area added, triangles added, and maximum seam-normal discontinuity. The reconstruction UI summarizes defects fixed, holes filled, openings and unknown boundaries preserved, and fallback state.

## Build and packaging

The manifest pins the vcpkg registry baseline and resolves CGAL 6.2, Eigen, Boost dependencies, GMP/MPFR, and nlohmann-json. From the repository root:

```powershell
git clone https://github.com/microsoft/vcpkg.git build/vcpkg
build/vcpkg/bootstrap-vcpkg.bat -disableMetrics
cmake -S native/mesh-repair -B build/mesh-repair-cgal -A x64 `
  "-DCMAKE_TOOLCHAIN_FILE=$PWD/build/vcpkg/scripts/buildsystems/vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build/mesh-repair-cgal --config Release
ctest --test-dir build/mesh-repair-cgal -C Release --output-on-failure
```

`scripts/build-workers.ps1` performs the same configuration and copies `scanlan-mesh-repair.exe` plus its runtime DLLs beside `scanlan-worker.exe`. The existing Tauri `scanlan-worker` resource directory therefore bundles both executables. Development discovery also checks `build/mesh-repair-cgal/Release`, and runtime startup validates the schema, algorithm version, and CGAL backend name.

## Commands

```text
scanlan-mesh-repair version --json

scanlan-mesh-repair analyze \
  --input raw-mesh.ply \
  --report topology.json \
  --voxel-size-m 0.008

scanlan-mesh-repair repair \
  --input raw-mesh.ply \
  --policy repair-policy.json \
  --output repaired-mesh.ply \
  --report native-repair-report.json
```

The repair policy is versioned and contains the source fingerprint plus only classifier-authorized loop IDs. Stale fingerprints, incompatible versions, duplicate IDs, or non-fill classifications are rejected. Reports and geometry use atomic replacement so readers do not observe partial output.

The reconstruction CLI exposes the same product controls:

```text
scanlan-worker reconstruct PROJECT \
  --mesh-repair on \
  --mesh-repair-profile faithful \
  --no-fill-inferred-holes \
  --no-produce-watertight-copy \
  --mesh-repair-fallback
```

## Report schema

Schema version 1 uses algorithm version `1.0.0`. Topology reports include the input SHA-256, counts for duplicate/degenerate/non-manifold/self-intersecting topology, stable connected components, and deterministic boundary loops. Each loop includes ordered positions, perimeter, approximate area, diameter, best-fit plane, plane RMS residual, boundary-normal coherence, bounding-box distance, and vertex count.

Repair reports add exact cleanup and stitching counts, non-manifold duplication, optional self-intersection repair, filled-loop patch metrics, topology before/after, unauthorized-fill count, and original-vertex maximum displacement. The Python report combines those results with depth evidence and final validation.

## Licensing

ScanLan remains GPL-3.0-only. The native executable uses CGAL `Surface_mesh`, Polygon Mesh Processing repair, manifoldness, stitching, self-intersection/autorefinement, hole triangulation/refinement/fairing, least-squares fitting, and PLY I/O. CGAL packages and their Eigen, Boost, GMP/MPFR, and nlohmann-json dependencies retain their upstream GPL-3.0-or-later, LGPL-3.0-or-later, BSL-1.0, MPL-2.0, or other compatible notices supplied through vcpkg. MeshLib is not included.
