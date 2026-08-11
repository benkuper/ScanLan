# Material and radiometric foundation

P13 freezes the data and licensing boundary used by the later material-aware geometry and PBR
stages. It does not claim that a material model is production-ready. A model can enter P14 only
after it passes the representative bake-off and its complete code, checkpoint, dependency, and
training-asset terms fit the requested model pack.

## Shared contract

`scanlan-material` is a small NumPy/Pillow package installed into both frozen reconstruction
runtimes. Its `scanlan-material-v1` observation contract separates:

- mutually exclusive material identity (`unknown`, `opaque_dielectric`, `metal`, `emissive`,
  `thin_or_fibrous`, `dynamic`, and `sky`);
- overlapping optical risks (`glass_or_transmissive`, `mirror`, `high_specular`, `emissive`,
  `thin_geometry`, `dynamic`, and `sky`);
- calibrated prediction confidence and a source-aligned validity mask;
- optional PBR fields whose semantics are already fixed for P16: linear-sRGB albedo and emission,
  linear roughness/metallic/transmission, and unit camera-space normals.

Identity and optical risk cannot be inferred from each other. For example, polished metal remains
the `metal` class while independently carrying `high_specular` and possibly `mirror` risk. Invalid
pixels must have zero confidence and zero risk; valid class probabilities must sum to one. Unknown
versions, malformed shapes, non-finite arrays, out-of-range probabilities, non-unit normals, and
nonlinear or over-range albedo fail closed. Publication uses an atomic compressed NumPy sidecar.

## Linear-light preparation

Every material model receives `scanlan-linear-srgb-v1` input:

1. EXIF orientation is applied once.
2. An embedded ICC profile is converted to sRGB with LittleCMS and relative-colorimetric intent.
3. An unprofiled ScanLan JPEG/PNG is accepted only under the capture/dataset declaration that it is
   IEC sRGB; unknown HDR and wide-gamut modes are rejected.
4. The exact IEC 61966-2-1 sRGB transfer is decoded before statistics, fusion, or PBR math.
5. Source-aligned linear RGB is stored as content-addressed float16 `.npy` data. The manifest keeps
   the source digest, conversion decision, dimensions, dataset fingerprint, and working space.

Canonical photo import now converts embedded profiles before writing its sRGB JPEG and embeds the
resulting sRGB profile. The on-demand command avoids adding tens of gigabytes to projects that do
not request material output:

```powershell
scanlan-splat prepare-material --dataset <dataset-root> --output <project>\outputs\material-input
```

The base-color and emissive rules intentionally match the
[Khronos glTF 2.0 material specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#materials):
color textures are sRGB-encoded for interchange but decoded before computation, while roughness and
metallic values remain linear. ICC conversion follows the
[International Color Consortium interpretation of IEC sRGB](https://registry.color.org/rgb-registry/files/sRGB.pdf).

## Candidate bake-off

The initial frozen shortlist compares complementary approaches rather than choosing a winner from
paper figures:

| Candidate | Frozen source/model revision | Intended evidence | Pack status |
| --- | --- | --- | --- |
| [Material Anything](https://github.com/3DTopia/MaterialAnything) estimator | `be3d6b3` / `dcd4e4c` | Mesh-conditioned albedo, roughness, metallic, and normal maps | Commercial candidate: MIT code, Apache-2.0 checkpoint; transitive audit still required |
| [RGB↔X](https://github.com/zheng95z/rgbx) RGB-to-X | `977e0df` / `b38b3fd` | Indoor intrinsic decomposition and lighting | Research only: Adobe Research source and unverified checkpoint terms |
| [DiffusionRenderer](https://github.com/nv-tlabs/diffusion-renderer) inverse model | `8fcf005` | Video material/lighting decomposition and consistency | Research only: NVIDIA noncommercial license |

Material Anything is designed to estimate PBR maps from existing or generated textured meshes.
RGB↔X provides an interior-scene intrinsic challenger, and DiffusionRenderer provides a strong
video inverse-rendering challenger. These roles are not treated as interchangeable, and none is a
substitute for explicit glass/mirror risk validation.

The bake-off harness rejects a candidate unless it evaluates at least 20 annotated frames including
real ScanLan captures and passes all of these gates on the 12 GB reference machine:

```text
material mean IoU:             >= 0.55
optical-risk recall:           >= 0.90
optical-risk precision:        >= 0.65
warped multiview consistency:  >= 0.80
confidence calibration ECE:    <= 0.10
peak VRAM:                     <= 10.5 GB
```

The accepted set is Pareto-ranked across quality, calibration, memory, and measured time. The
deterministic tie policy puts optical-risk recall first because a false negative can contaminate
geometry, then material IoU and multiview consistency. No single weighted score can hide a failed
safety or memory gate.

## Commercial and research packs

The pack resolver validates immutable 40-character source and model revisions and refuses to place
unverified, noncommercial, or output-restricted assets in a commercial manifest. The commercial
manifest currently contains only the Material Anything candidate. The research manifest may list
all candidates and explicitly records its output restrictions. This is a candidate manifest, not a
download list: P14 packaging may include a model only after the bake-off and full transitive license
audit.

Generate the manifests with:

```powershell
scanlan-splat material-pack --pack commercial --output model-pack.json
scanlan-splat material-pack --pack research --output model-pack.json
```

This separation is fail-closed. A missing or unverified model license can never be interpreted as
permission for commercial use.

## Validation

The material package tests cover exact sRGB reference values, ICC conversion, content-addressed
reuse, path confinement, invalid-pixel behavior, overlapping optical risks, PBR bounds, atomic
round trips, hard bake-off gates, Pareto ordering, immutable revisions, and commercial-pack
exclusion. P13 also runs radiometric preparation on the repository's representative phone capture;
P14 must add human-reviewed optical-risk and material labels before any model can pass.
