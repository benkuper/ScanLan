# Material-aware Gaussian reconstruction

P17 replaces one entangled Gaussian appearance tensor with four explicit behaviors in the
production gsplat trainer: diffuse linear reflectance, view-dependent spherical harmonics,
non-negative emission, and transmission. It retains the P12 geometry, source-tile coverage,
quality, memory, cancellation, and resume gates.

## Why the decomposition is prior-gated

Appearance decomposition is underconstrained from ordinary RGB supervision.
[Spec-Gloss Surfels](https://openaccess.thecvf.com/content/WACV2026/html/Kouros_Spec-Gloss_Surfels_and_Normal-Diffuse_Priors_for_Relightable_Glossy_Objects_WACV_2026_paper.html)
uses intrinsic diffuse/normal priors to reduce this ambiguity, while
[RT-Splatting](https://openaccess.thecvf.com/content/CVPR2026/html/Shi_RT-Splatting_Joint_Reflection-Transmission_Modeling_with_Gaussian_Splatting_CVPR_2026_paper.html)
separates geometric occupancy from optical opacity and gates misleading specular gradients.
ScanLan applies those principles conservatively: the material path
activates only when a versioned P14/P16 linear-sRGB prior is declared. Existing datasets keep their
previous display-space radiance path exactly and do not acquire guessed glass or emission.

`scanlan-gaussian-material-v1` requires every initialization Gaussian to carry:

- linear-sRGB albedo and emission;
- roughness, metallic, and transmission in `[0, 1]`;
- fused material confidence in `[0, 1]`.

Partial arrays, undeclared arrays, unsafe paths, the wrong vertex count, non-finite values, and the
wrong color space fail closed. Confidence blends albedo toward the observed initialization color
and blends every risky component toward rough, nonmetallic, opaque, non-emissive behavior.

## Trainer behavior

The parameter dictionary and each fused Adam optimizer have matching component tensors, so gsplat
1.5.3's bounded duplicate/split/prune operations resize material state together with geometry.
Checkpoints therefore preserve the exact decomposition and remain protected by the bumped trainer
version.

With a material prior, targets and diffuse/emissive radiance use linear sRGB. The rasterizer receives
camera-invariant diffuse plus emission in the degree-zero coefficient and a separate degree-1-to-3
view-dependent lobe. Geometric opacity continues to represent occupancy. Optical opacity is
`geometric opacity * (1 - transmission)`, allowing content behind a transmissive surface to remain
visible without erasing its geometric primitive. This is a bounded raster approximation, not a
claim of ray-traced refraction.

Transmission and emission stay anchored to the fused prior. Unsupported components are penalized
toward zero, and rough regions penalize excess high-order view dependence. High-confidence smooth
or metallic evidence stops photometric gradients from leaking from the reflection lobe into the
transmission branch; the material prior still supplies its valid gradient.

## Publication

`room-splat.ply` remains an interoperable display-space 3DGS artifact: its degree-zero color uses
the exact transfer and higher-order linear radiance uses a first-order sRGB projection. The lossless
linear coefficients remain in the material sidecar. The
`room-splat.preview.splat` uses optical opacity. A material-aware job additionally publishes the
atomic `room-splat-material.npz`, aligned one-to-one with PLY vertex order, containing:

- `diffuse_linear` and `emission_linear`;
- `view_sh_linear`;
- `transmission`, `roughness`, `metallic`, and `confidence`;
- both `geometric_opacity` and `optical_opacity`;
- contract, color-space, and alignment declarations.

The splat manifest describes these semantics and the project artifact retains the material path.
An ordinary transformed PLY export copies the material sidecar and updates its exported name.
Bounding-box clipping deliberately omits it because filtering the PLY would otherwise break row
alignment; that omission is explicit in the exported manifest.

## Verification

Tests cover fail-closed contracts, neutral legacy initialization, component composition, optical
opacity, specular gradient gating, prior regularization, and lossless sidecar round trips. The CUDA
smoke runs the real 2DGS rasterizer and fused optimizers on a declared material dataset through final
PLY, preview, manifest, camera, quality, and material publication.
