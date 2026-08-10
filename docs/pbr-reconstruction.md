# PBR reconstruction and GLB export

P16 turns P14's fused, confidence-bearing material surface into a portable intrinsic material.
The exporter preserves the original observed atlas, bakes a distinct intrinsic atlas, and writes a
self-contained glTF 2.0 binary asset. It follows the Khronos metallic-roughness layout and does not
reinterpret display RGB as linear reflectance.

## Output contract

`build_pbr_artifacts` atomically publishes `scanlan-pbr-v1` artifacts:

- `room-observed.png`: the illumination-bearing sRGB atlas used by the conventional mesh;
- `room-base-color.png`: sRGB encoding of fused linear intrinsic albedo;
- `room-metallic-roughness.png`: linear channels with roughness in G and metallic in B;
- `room-transmission.png`: linear transmission in R;
- `room-normal.png`: OpenGL tangent-space normal vectors;
- `room-emission.png`: sRGB encoding of normalized linear emission;
- `room-pbr.glb`: geometry, tangents, every atlas, and the PBR material in one binary asset;
- `pbr-report.json`: dimensions, coverage, color semantics, fallbacks, and artifact names.

The GLB uses core glTF base color, metallic-roughness, normal, and emissive properties. Transmission
is declared through `KHR_materials_transmission`. Emission above the core texture range is normalized
and restored with `KHR_materials_emissive_strength`. The observed atlas is embedded and referenced
from material extras as ScanLan provenance; the intrinsic base-color texture drives standard PBR
rendering.

## Confidence and fallback policy

Every predicted field is blended by P14's fused confidence. Unsupported vertices retain observed
base color and become rough, nonmetallic, opaque, non-emissive dielectrics with the measured mesh
normal. Thus missing inverse-rendering evidence cannot manufacture highlights, glass, metal, light,
or normal detail. Partial evidence degrades continuously to those safe values instead of creating a
hard material seam.

The mesh is exported with triangle-corner vertices so UV seams remain exact. Tangents are derived
from each UV chart and the fused world-space material normals are transformed into glTF tangent
space during atlas rasterization.

## Verification

Deterministic tests parse the GLB container and its JSON/binary chunks, verify all texture bindings
and Khronos extensions, check metallic-roughness/transmission/normal channel encodings, exercise HDR
emission, and prove that unsupported predictions use the neutral observed-color fallback.

P16 supplies the reconstruction and export path; automatic use still depends on a P13/P14 model
pack passing the real-capture, licensing, memory, and visual gates.
