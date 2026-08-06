# Mesh repair

ScanLan's mesh-repair backend is a separate native executable built with CGAL. The first implementation milestone is deliberately report-only: it can inspect a triangular PLY mesh, but it has no command that writes or mutates mesh geometry.

## Build

The manifest pins the vcpkg registry baseline and currently resolves CGAL 6.2, Boost, GMP/MPFR, and nlohmann-json. From the repository root:

```powershell
git clone https://github.com/microsoft/vcpkg.git build/vcpkg
build/vcpkg/bootstrap-vcpkg.bat -disableMetrics
cmake -S native/mesh-repair -B build/mesh-repair -A x64 `
  "-DCMAKE_TOOLCHAIN_FILE=$PWD/build/vcpkg/scripts/buildsystems/vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build/mesh-repair --config Release
ctest --test-dir build/mesh-repair -C Release --output-on-failure
```

The same manifest works with another vcpkg installation by changing only `CMAKE_TOOLCHAIN_FILE`.

## Commands

Runtime/backend diagnostics are available without loading a mesh:

```text
scanlan-mesh-repair version --json
```

Analyze a triangular ASCII or binary PLY mesh:

```text
scanlan-mesh-repair analyze \
  --input raw-mesh.ply \
  --report topology.json \
  --voxel-size-m 0.008
```

Analysis returns zero only after the report has been written successfully. Invalid input returns a non-zero status and, when `--report` was parsed successfully, writes a JSON error document. Reports are written through a temporary file so readers never observe a partial document.

## Topology report schema, version 1

Successful reports contain these top-level fields:

| Field | Meaning |
| --- | --- |
| `schemaVersion` | Integer report contract version; currently `1`. |
| `algorithmVersion` | Analyzer behavior version; currently `1.0.0`. |
| `command`, `status` | `analyze` and `ok`. |
| `inputMeshFingerprint` | SHA-256 algorithm name and lowercase digest of the original PLY bytes. |
| `voxelSizeM` | Metric voxel size supplied by the caller. |
| `mesh` | Original vertex/triangle counts and the canonical-position bounding box. |
| `topology` | Duplicate, degenerate, non-manifold, self-intersection, component, and boundary counts. |
| `connectedComponents` | Stable component IDs plus vertex/triangle counts, area, boundary-loop count, and bounds. |
| `boundaryLoops` | Stable loop IDs and the geometric measurements below. |
| `warnings` | Conditions that limited validation without making the report unusable. |

Each `boundaryLoops` entry contains:

- `loopId` and `componentId`
- `orderedBoundaryPositions`
- `perimeterM`, `approximateEnclosedAreaM2`, and `diameterM`
- `bestFitPlane.origin` and `bestFitPlane.normal`
- `planeRmsResidualM`
- `boundaryNormalCoherence`, in the range 0–1
- `distanceFromMeshBoundingBoxBoundaryM`
- `vertexCount`

Loop ordering is canonicalized across rotations and winding directions before its ID is calculated. Component IDs are based on sorted geometric triangle signatures. Reports intentionally contain no timestamp or machine-specific input path, so analyzing identical bytes with the same arguments is deterministic and produces byte-for-byte identical JSON.

Defect counts describe the original polygon soup after exact-coordinate vertex canonicalization. CGAL self-intersection checks run on a non-mutating validation mesh with duplicate and degenerate triangles excluded. If CGAL cannot insert a topologically incompatible triangle into that validation mesh, the report retains the raw topology counts and records the limitation in `warnings`.

An error report has the same version and algorithm fields, `status: "error"`, and an `error` object containing stable `code` and human-readable `message` fields.

## Licensing

ScanLan remains licensed under GPL-3.0-only. The bundled backend uses CGAL packages distributed under GPL-3.0-or-later/LGPL-3.0-or-later/BSL-1.0 terms, together with their Boost and GMP/MPFR dependencies. The report-only analyzer uses CGAL stream support, `Surface_mesh`, least-squares plane fitting, and Polygon Mesh Processing self-intersection detection. Release packaging must retain the dependency notices produced by vcpkg.
