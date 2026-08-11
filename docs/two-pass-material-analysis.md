# Two-pass material analysis

P14 turns the P13 observation contract into a bounded multiview pipeline without pretending an
unvalidated material checkpoint is production-ready. The shared `scanlan-material` package owns
planning, geometric acceptance, fusion, and the 3D sidecar. A separately packaged backend owns
inference and can enter either the commercial or research runtime only after the existing measured
bake-off and transitive-license gates pass.

## Why the model remains replaceable

The current candidates solve different problems. [Material Anything](https://github.com/3DTopia/MaterialAnything)
is a mesh-conditioned PBR estimator/refiner. [RGB↔X](https://github.com/zheng95z/rgbx) targets
interior intrinsic decomposition but its released terms do not qualify for the commercial pack.
[DiffusionRenderer](https://github.com/nv-tlabs/diffusion-renderer) is a strong temporally coherent
video inverse renderer, but its upstream instructions report more than 22 GB of VRAM after the
provided memory reductions, which is incompatible with ScanLan's 12 GB target. None replaces an
explicit, measured glass/mirror safety detector.

This is why the runner accepts coarse and final inference callbacks that must return the frozen
`scanlan-material-v1` contract. Unknown shapes, grids, poses, probability ranges, invalid-pixel
semantics, and camera transforms fail before any evidence reaches the surface.

## Pass 1: coarse optical risk

`select_coarse_cameras` measures adjacent camera translation to derive the scene's path scale and
combines it with rotation distance and pose confidence in deterministic k-center selection. The
first and last calibrated views remain represented. This avoids both fixed frame strides and an
unbounded full-video pass.

The coarse backend may run at a lower calibrated resolution, but its prediction and intrinsics must
describe that same grid. `select_final_views` projects a bounded deterministic surface sample into
every coarse view, rejects occluded/back-facing/border-unsafe evidence, and greedily maximizes new
surface coverage. Optical-risk probability increases the gain rather than merely adding a fixed
number of frames. At least two useful final views are retained where available so apparent
single-view certainty is not mislabeled as multiview support.

## Pass 2: final material evidence

The final backend runs only on the selected views. For each production-surface vertex, ScanLan
requires:

- a rigid calibrated world-from-camera pose;
- positive camera-space depth and valid source coordinates;
- agreement with measured depth, or the nearest projected surface when depth is absent;
- one consistent front-face winding per view;
- sufficient border margin, pose confidence, and prediction confidence.

Accepted material-class distributions use normalized confidence-weighted evidence, following the
probabilistic multiview principle demonstrated by
[SemanticFusion](https://arxiv.org/abs/1609.05130). Independent optical risks retain the greater of
the multiview consensus and the strongest calibrated warning from a geometrically sound view. This
asymmetry is intentional: a false glass or mirror negative can corrupt P15 geometry, whereas a
single class disagreement should reduce identity confidence.

Confidence combines accumulated authority, class-distribution agreement, and effective view count
`(Σw)² / Σw²`. A repeated near-duplicate view therefore cannot masquerade as independent support.
Optional camera-space PBR normals are rotated into world space before averaging and normalization;
linear albedo, emission, roughness, metallic, and transmission retain the P13/P16 semantics.

## 3D surface contract

`scanlan-material-surface-v1` is an atomically published compressed NumPy sidecar containing:

- `class_probabilities` (`float16`, V×C);
- `optical_risk_probabilities` (`float16`, V×R);
- `valid_mask`, `confidence`, `support_count`, and `effective_view_count`;
- connected `region_ids`, with `-1` for unsupported/unknown vertices;
- optional linear PBR fields and unit world-space normals;
- analysis revision, coarse/final source views, supported/multiview vertex counts, and region count.

Regions are connected components on the actual triangle graph. An edge crosses a boundary when
identity differs, the thresholded overlapping-risk signature differs, or the full class
distribution changes sharply. P15 can therefore apply conservative geometry policy to coherent 3D
regions instead of noisy independent pixels.

## Quality and activation boundary

The deterministic tests cover adaptive path sampling, risk-prioritized final selection, metric
occlusion rejection, conservative single-view risk retention, multiview support, connected regions,
atomic round trips, callback isolation, and source-grid failure. The production-scale integration
smoke used the retained nine-view phone solve and its 315,618-vertex / 594,669-triangle reconstructed
mesh. Eight coarse cameras selected four final cameras; geometric fusion supported 145,208 vertices,
34,923 from at least 1.5 effective views, and built 1,446 connected regions in 1.281 seconds. Neutral
fixture predictions were deliberately used so this run measures camera projection, bounded planning,
fusion, and region construction—not material accuracy.

This validates P14's integration and geometry semantics, not model quality. Default model
activation still requires at least 20 human-reviewed frames including the retained real capture,
all P13 bake-off gates, visual inspection of the fused region artifact, 12 GB peak-memory evidence,
and a complete transitive asset-license audit. Until then, the shipped pack contains no automatic
material inference and downstream P15 must treat missing material output as unknown.
