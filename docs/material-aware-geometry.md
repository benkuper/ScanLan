# Material-aware geometry

P15 turns P14's optical-risk evidence into conservative geometry decisions. It does not make a
material model mandatory and it does not treat a material label as geometry. Instead, the shared
package publishes bounded confidence multipliers, repair vetoes, and a validation-gated second-pass
refinement contract. If no material sidecar exists, the policy is an explicit neutral no-op and the
current production result is unchanged.

## Why the policy is conservative

Commodity RGB-D sensors can return missing or incorrect depth on transparent and reflective
surfaces. [TransFusion](https://openaccess.thecvf.com/content/ICCV2021/html/Zhu_Transfusion_A_Novel_SLAM_Method_Focused_on_Transparent_Objects_ICCV_2021_paper.html)
showed why transparent observations should be excluded from ordinary RGB-D tracking/fusion and
reconstructed separately. Single-view completion alone is not enough for production reintegration:
[Consistent Depth Prediction for Transparent Object Reconstruction](https://openaccess.thecvf.com/content/ICCV2023/html/Cai_Consistent_Depth_Prediction_for_Transparent_Object_Reconstruction_from_RGB-D_Camera_ICCV_2023_paper.html)
demonstrated that cross-view consistency is necessary to prevent distorted reconstruction. The P15
policy therefore distinguishes protection from recovery:

- calibrated material evidence can only preserve or reduce upstream depth confidence;
- dynamic and sky evidence discards static-scene geometry;
- glass, mirror, and sufficiently strong thin-geometry evidence protects a region from blind hole
  filling and ordinary refinement;
- a protected region can move only through an explicit generated/learned proposal with at least 2.5
  effective views, calibrated confidence of at least 0.90, held-out residual no greater than half a
  voxel, and displacement no greater than one voxel;
- measured sensor depth cannot use the protected-recovery exception.

This leaves room for a later validated transparent-depth backend. ScanLan does not package one in
P15 because no candidate has yet passed P13's real-capture optical-risk, calibration, memory, and
license gates. Methods such as
[RGB-D Local Implicit Function](https://openaccess.thecvf.com/content/CVPR2021/html/Zhu_RGB-D_Local_Implicit_Function_for_Depth_Completion_of_Transparent_Objects_CVPR_2021_paper.html)
remain candidates for measured comparison rather than silently becoming a runtime dependency.

## Depth authority

`scanlan-material-geometry-v1` supplies five per-pixel or per-vertex values in `[0, 1]`:

- measured sensor-depth multiplier;
- validated generated-depth multiplier;
- learned RGB-only geometry multiplier;
- repair authority;
- refinement authority.

The multipliers are applied to existing confidence and can never increase it. Glass and mirror have
the strongest sensor/generated penalties; high specularity and thin geometry receive graded
penalties; emissive appearance primarily reduces generated/learned authority. Unsupported material
pixels are neutral instead of being interpreted as safe. Strong dynamic or sky identity/risk sets
every authority to zero.

Per-view risk is multiplied by its calibrated prediction confidence. P14's fused surface risk is
not multiplied again: its consensus and conservative-peak paths already retain calibrated evidence,
and class entropy must not erase a sound single-view glass warning.

## Material-aware repair

Boundary positions are matched to the preliminary surface with a bounded-memory nearest-vertex
search. Missing nearby material evidence preserves the existing depth/free-space repair decision.
Nearby protected evidence vetoes the fill, as does a material-authority lower-tail below 0.50 or a
dynamic/sky fraction of at least 10 percent. This policy is deliberately asymmetric: preserving a
possible glass opening is reversible in a later specialized pass, while filling it contaminates the
surface and every downstream texture/PBR result.

## Second-pass refinement

`GeometryProposal` is a one-to-one candidate for the preliminary indexed surface. Each candidate
vertex carries provenance, calibrated confidence, effective independent-view count, and held-out
metric residual. `refine_material_geometry` applies provenance-specific gates:

| Evidence | Minimum confidence | Minimum effective views | Maximum held-out residual | Maximum displacement |
| --- | ---: | ---: | ---: | ---: |
| measured | 0.35 | 0.75 | 1.50 voxels | 2.50 voxels |
| generated | 0.70 | 1.75 | 1.00 voxel | 2.00 voxels |
| learned | 0.75 | 2.00 | 0.75 voxel | 1.50 voxels |

Accepted proposals are blended by their remaining material/evidence authority. The final indexed
mesh must retain triangle orientation within 60 degrees and triangle area within 0.2x-5x of the
input. Vertices participating in a new fold, collapse, or explosive stretch are reset; failure to
restore a valid mesh rejects the refinement. The atomically written
`scanlan-material-refinement-v1` result records accepted vertices, authority, displacement, every
gate count, protected recovery, and topology rejection.

## Verification

Deterministic tests cover neutral missing-material behavior, confidence monotonicity, glass/mirror
protection, dynamic rejection, repair vetoes, ordinary opaque refinement, strict protected recovery,
topology rollback, and atomic contract round trips.

The production-scale smoke used the retained real 315,543-vertex / 578,097-triangle reconstructed
mesh. Ten percent of vertices were marked as high-confidence glass with insufficient independent
support; the remaining vertices received a 1 mm generated proposal. P15 accepted 283,956 vertices,
accepted none of the 31,555 protected vertices, reset 32 vertices at topology gates, and bounded the
final displacement to 0.951 mm. Refinement took 0.585 seconds; loading plus refinement took 1.538
seconds with a 173.9 MiB peak process working set on the reference Windows machine.

This smoke validates policy scale, protection, and topology behavior. It does not validate material
model accuracy. Automatic activation still depends on a P13/P14 candidate passing the annotated
real-capture bake-off and visual inspection.
