# ScanLan Reconstruction 2.0 — Revised Master Plan

## 1. Core product definition

ScanLan must produce three distinct reconstruction products from every capture:

### A. Live reconstruction preview

Available during recording or progressive video processing:

* continuously estimated camera pose;
* accumulated scene geometry;
* camera trajectory and frustum;
* tracking confidence;
* coverage and missing-area guidance;
* visible relocalization and loop-correction behavior;
* bounded latency and memory.

This is a real reconstruction, not merely a current-frame point cloud.

### B. Immediate provisional reconstruction

Available as soon as recording stops:

* preserves the final live map;
* remains inspectable while production processing starts;
* can be exported as a clearly identified provisional point cloud or mesh;
* never blocks access to the raw capture.

### C. Production reconstruction

Built asynchronously after capture:

* reruns or refines the trajectory;
* performs global optimization;
* reintegrates original observations;
* runs learned geometry, material, PBR and Gaussian stages;
* replaces the provisional result only after validation.

The current ScanLan live path already performs quality-gated RGB-D tracking, weighted TSDF integration, point snapshots and an optional low-rate mesh. That becomes a protected first-class subsystem rather than an implementation detail.

---

# 2. Revised architectural principle

The application must have two deliberately separate compute lanes:

```text
                    CAPTURE-TIME LANE
                    hard latency budget
                           │
RGB-D / video input ───────┼──► tracking
                           │
                           ├──► local map integration
                           │
                           ├──► coverage analysis
                           │
                           └──► reconstruction preview
                                      │
                                      ▼
                            provisional live map
                                      │
                                      │ raw observations remain authoritative
                                      ▼
                   PRODUCTION-TIME LANE
                    quality-first processing
                                      │
                           global camera refinement
                                      │
                           learned geometry backends
                                      │
                           final dense fusion / SDF
                                      │
                           materials and PBR
                                      │
                           optimized Gaussian splat
                                      │
                                      ▼
                           final validated outputs
```

The production lane may use LingBot, MapAnything, DA3, neural SDF, material segmentation and inverse rendering.

The RGB-D capture-time lane must not depend on any of them.

For RGB-only video, a learned streaming backend is necessary to obtain depth and poses, but it still operates under a separate bounded preview profile rather than the full production profile.

---

# 3. Updated non-negotiable requirements

1. **Realtime reconstruction preview is a release-blocking feature.**

2. **Capture never waits for reconstruction preview.** Sensor acquisition, archive writing, preview integration and viewport rendering remain independent bounded queues.

3. **Realtime preview never modifies raw observations.**

4. **Production work never runs concurrently with an active RGB-D capture on the same GPU unless an explicit resource test proves sufficient headroom.**

5. **The viewport render rate is independent from map-update rate.** A 60 Hz viewport may render geometry that is updated at 10 Hz.

6. **Tracking failure freezes integration rather than corrupting the live map.**

7. **Loop corrections transform live submaps instead of rebuilding the entire scene during capture.**

8. **The production pass always has the right to reject and rebuild the live trajectory.**

9. **Live RGB-only video geometry is explicitly marked provisional and scale-qualified.**

10. **Material segmentation and PBR reconstruction are not allowed to jeopardize capture latency.**

11. **A low-confidence live preview must communicate uncertainty visibly rather than hide it behind smooth rendering.**

12. **The final live map survives application workspace transitions and remains visible immediately after capture.**

---

# 4. Realtime RGB-D reconstruction architecture

## 4.1 Capture stages

```text
sensor acquisition
      │
      ├──► bounded archive writer
      │
      └──► bounded realtime queue
                    │
                    ▼
             depth preparation
                    │
                    ▼
             camera tracking
                    │
             accepted pose?
               │         │
              yes        no
               │         └──► freeze integration + relocalize
               ▼
          keyframe decision
               │
               ▼
        active submap fusion
               │
        ┌──────┴─────────┐
        ▼                ▼
 preview geometry   coverage map
        │                │
        └──────┬─────────┘
               ▼
             viewport
```

## 4.2 Tracking

Keep the current deterministic tracking foundation:

* calibrated RGB-D odometry;
* gyro initialization where available;
* overlap and depth-residual gates;
* physical-motion gates;
* recent-anchor relocalization;
* fail-closed integration.

Upgrade it with:

* explicit tracking-state confidence;
* pose covariance or a practical uncertainty approximation;
* tracking-quality history;
* separate `TRACKING`, `SEARCHING`, `RELOCALIZED`, `FROZEN` and `FAILED` states;
* keyframe descriptors retained for local relocalization and loop detection;
* deterministic replay tests for every tracker change.

Learned models may assist production relocalization later, but do not become mandatory for live RGB-D tracking.

---

# 5. Live map representation

## 5.1 Submap-based fusion

Replace the conceptual single ever-growing live volume with bounded local submaps.

Each submap contains:

* local origin;
* local trajectory segment;
* voxel-hashed TSDF or surfels;
* color;
* normal;
* observation count;
* confidence;
* approximate coverage;
* bounding volume.

A new submap begins when one or more conditions are reached:

* distance from current submap origin;
* rotation accumulation;
* voxel or memory budget;
* tracking discontinuity;
* take boundary;
* substantial loop correction.

Completed submaps become immutable during normal capture except for their global transform.

## 5.2 Why submaps are required

They allow ScanLan to:

* bound active GPU memory;
* avoid rebuilding an entire room after loop closure;
* correct drift by moving submaps;
* stream old submaps to host memory;
* preserve responsive rendering in large scenes;
* isolate tracking failures;
* reuse the same representation for RGB-only video preview.

## 5.3 Live representation hierarchy

The live engine should support three progressively heavier display forms.

### Default: surfel or fused-point preview

* fastest update;
* suitable for continuous capture;
* preserves color and normals;
* straightforward confidence visualization;
* preferred when tracking headroom is limited.

### Enhanced: TSDF raycast preview

* visually denser and easier to interpret;
* updated from the active sparse volume;
* avoids constant triangle extraction;
* recommended balanced default where the CUDA implementation supports it efficiently.

### Optional: asynchronous mesh preview

* extracted from completed or active submaps;
* lower update rate;
* never blocks tracking or fusion;
* useful for topology inspection rather than primary responsiveness.

A lightweight “Gaussianized surfel” renderer may later draw each fused surfel as an oriented disc. This can improve visual density without running Gaussian optimization during capture.

Full 2DGS/3DGS training does not belong in the RGB-D capture loop.

---

# 6. Live submap corrections and loop closure

## 6.1 During capture

The live system performs bounded loop detection:

* recent keyframe relocalization continuously;
* nonlocal loop queries at a controlled rate;
* strict geometric verification;
* local pose-graph correction;
* submap-transform update.

It does not globally reintegrate all archived depth during recording.

## 6.2 Visual correction behavior

When a loop is accepted:

1. calculate corrected submap transforms;
2. interpolate viewport transforms over a short interval;
3. update camera trajectory and coverage;
4. preserve active tracking continuity;
5. mark the live map as corrected;
6. record the loop for production validation.

This prevents a distracting single-frame map jump while avoiding false geometry blending.

## 6.3 After capture

Production reconstruction:

* verifies all loops again;
* performs the full pose-graph solve;
* rejects unsafe corrections;
* reintegrates original depth using final poses;
* replaces the provisional submaps with final geometry.

The live map is not treated as the final fusion source.

---

# 7. Realtime coverage guidance

The preview must help the user improve the scan, not merely show geometry.

Maintain a low-resolution coverage field containing:

* observation count;
* best viewing angle;
* estimated pixel density;
* depth confidence;
* pose confidence;
* recent visibility;
* material-risk status where available.

Viewport modes:

### Normal

Colored accumulated reconstruction.

### Coverage

* well observed;
* weakly observed;
* single-view only;
* unseen or hole boundary.

### Tracking

* trusted map;
* recently integrated geometry;
* frozen geometry;
* uncertain trajectory;
* relocalization anchors.

### Geometry confidence

* measured and repeatedly observed;
* single observation;
* learned or repaired;
* rejected/unknown.

### Material-risk preview

Later optional overlay:

* likely glass;
* mirror;
* polished metal;
* emissive display;
* thin/fibrous material;
* dynamic object.

User guidance should be concrete:

* move slower;
* return to last trusted region;
* increase parallax;
* revisit this surface;
* close the loop;
* surface may be glass or reflective;
* coverage is one-sided;
* depth is unavailable here.

---

# 8. Compute and memory scheduling

## 8.1 Capture has absolute priority

Resource priority:

```text
1. camera acquisition
2. archive persistence
3. pose tracking
4. live fusion
5. reconstruction rendering
6. coverage analysis
7. optional loop queries
8. optional material-risk inference
```

Lower-priority work is skipped or delayed before higher-priority work is allowed to miss its budget.

## 8.2 Bounded queues

Use latest-wins queues for:

* viewport geometry snapshots;
* coverage overlays;
* optional mesh extraction;
* optional material analysis.

Use ordered queues for:

* archived raw observations;
* tracking frames;
* pose journal entries.

Queue pressure is visible in diagnostics.

## 8.3 Adaptive degradation ladder

When the live system exceeds its latency or memory budget:

1. reduce viewport geometry publication frequency;
2. pause live mesh extraction;
3. reduce raycast resolution;
4. reduce coverage-analysis frequency;
5. reduce nonlocal loop-query frequency;
6. integrate fewer keyframes while continuing to track every frame;
7. temporarily switch from TSDF raycast to surfel rendering;
8. freeze integration if pose quality becomes unsafe.

It must not:

* reduce archive fidelity silently;
* skip tracking solely to preserve UI smoothness;
* continue integrating stale poses;
* allow memory to grow without a hard ceiling.

## 8.4 GPU isolation

On the 12 GB target GPU:

* the realtime engine receives a fixed configurable VRAM budget;
* the active submap remains on GPU;
* completed submaps may be compacted or moved to host memory;
* preview rendering uses bounded buffers;
* production learned models are unloaded during capture;
* post-capture workers release one large model before loading another.

The system should explicitly report:

* allocated live-map memory;
* active voxel/surfel count;
* resident submap count;
* host-cached submap count;
* dropped preview jobs;
* pose and map-update latency.

---

# 9. Realtime material handling

Material-aware reconstruction remains in the roadmap, but its capture-time role is deliberately limited.

## Stage 1: initial release

During RGB-D capture:

* no full material segmentation model;
* no inverse rendering;
* no PBR estimation;
* no material-aware remeshing.

Use fast geometric signals only:

* missing depth;
* unstable depth;
* multiview disagreement;
* high RGB variation;
* likely dynamic regions.

Material analysis begins immediately after capture.

## Stage 2: optional asynchronous optical-risk model

Run a compact model on selected keyframes at a low rate to identify:

* glass/transmission;
* mirror;
* polished metal;
* emissive screens;
* dynamic people or objects;
* sky for outdoor captures.

Rules:

* runs only when tracking and fusion have headroom;
* uses a latest-wins queue;
* never blocks capture;
* initially affects the overlay only;
* does not alter live fusion until separately benchmarked.

## Stage 3: conservative live policy

After validation, optical risk may influence live fusion:

* reflective/transmissive regions receive lower generated-depth authority;
* unstable regions are displayed but not repaired live;
* screen content is excluded from appearance normalization;
* suspected glass holes remain unknown.

The final material segmentation, PBR estimation and material-aware second fusion remain production stages.

---

# 10. RGB-only video reconstruction preview

This becomes an explicit secondary workstream rather than an undefined future possibility.

## 10.1 Supported scenarios

### Live RGB camera

Use a webcam, phone stream or supported video camera as a live reconstruction source.

### Progressive imported video

Play through an existing video file while progressively constructing the scene.

Both use the same ordered-frame reconstruction API.

## 10.2 Pipeline

```text
video decode or live frames
        │
        ▼
bounded quality/motion sampling
        │
        ▼
streaming learned pose + depth
 LingBot-Map initially
        │
        ▼
confidence and continuity gates
        │
        ▼
local RGB-derived submap
        │
        ▼
provisional reconstruction preview
```

LingBot-Map is the initial backend because ScanLan already integrates its ordered streaming model and bounded context logic.

DA3-Streaming and MapAnything become alternative backends after the generic geometry-worker contract exists.

## 10.3 Video preview behavior

The preview must show:

* reconstructed local geometry;
* provisional camera trajectory;
* scale status;
* pose confidence;
* submap boundaries;
* drift estimate;
* accepted and rejected frames;
* areas supported by multiple views;
* learned-only geometry.

Scale labels:

```text
MODEL_METRIC_UNVERIFIED
MODEL_METRIC_VALIDATED
USER_CALIBRATED
RELATIVE_SCALE
```

The UI must not imply calibrated metric accuracy when it has not been established.

## 10.4 Drift control

Use:

* streaming model context;
* local feature tracking;
* optical-flow continuity;
* keyframe overlap;
* local submaps;
* periodic loop candidates;
* optional sparse high-resolution anchors;
* user-visible drift state.

During live video capture, do not run full COLMAP mapping on every frame.

After stop:

1. run learned-first camera proposal;
2. select high-value anchors;
3. perform high-resolution feature verification;
4. run bundle adjustment;
5. correct submap transforms;
6. reintegrate dense depth;
7. produce final point cloud, mesh and GS.

## 10.5 Failure behavior

When live video tracking becomes unsafe:

* stop integrating geometry;
* keep recording video;
* continue searching for relocalization;
* display the last valid map;
* allow a new provisional submap if relocalization fails;
* attempt multi-submap registration after capture.

RGB-only live preview never contaminates the final result merely because it rendered successfully.

---

# 11. Live reconstruction artifacts

Add a dedicated live-session artifact:

```text
outputs/live/
    session.json
    poses.jsonl
    submaps/
    coverage/
    latest-preview.ply
    latest-preview.glb
    tracking-summary.json
```

`session.json` records:

* source type;
* live-engine revision;
* calibration;
* submap definitions;
* provisional scale status;
* tracking statistics;
* accepted loops;
* rejected loops;
* queue drops;
* peak memory;
* final live-map fingerprint.

At capture stop:

* publish the live artifact atomically;
* make it immediately visible in Reconstruct;
* begin production reconstruction from raw observations;
* retain the live artifact for A/B diagnosis;
* never overwrite it with the production result.

---

# 12. Revised implementation order

The previous roadmap should be reordered so realtime reconstruction is secured before expanding the offline model stack.

## P0 — Freeze current baseline

Benchmark:

* current RGB-D tracking;
* accumulated point preview;
* live mesh;
* latency;
* queue behavior;
* memory;
* relocalization;
* current post-stop artifact availability.

Include live-preview metrics in the canonical benchmark report.

## P1 — Realtime reconstruction contract

Define:

* tracking states;
* live submap format;
* preview geometry messages;
* coverage messages;
* latency telemetry;
* memory telemetry;
* capture/preview failure policy;
* immediate provisional artifact.

This phase changes interfaces, not algorithms.

## P2 — Harden the RGB-D live engine

Implement:

* submap management;
* sparse bounded fusion;
* independent render/update rates;
* adaptive degradation ladder;
* persistent post-stop live map;
* coverage visualization;
* tracking-confidence visualization.

Maintain the existing path as fallback during development.

## P3 — Live relocalization and bounded loop correction

Implement:

* local anchor database;
* verified loop candidates;
* submap pose graph;
* smooth viewport correction;
* deterministic replay tests;
* production revalidation.

## P4 — Geometry-worker scaffold

Create the isolated learned-model worker and migrate:

* LingBot-Map;
* LingBot-Depth.

Preserve output equivalence.

## P5 — RGB-only progressive preview

Implement experimental:

* LingBot streaming inference;
* local learned-depth submaps;
* drift/confidence visualization;
* progressive imported-video processing;
* optional live RGB camera input.

This remains feature-flagged until acceptance gates pass.

## P6 — ScanLan validation engine

Implement generic camera, scale, depth, free-space and geometry validation shared by live-video and production backends.

## P7 — MapAnything integration

Implemented:

* RGB-D completion with source-grid restoration, held-out sensor anchoring, and shared metric/multi-view/free-space gates;
* photo camera/depth proposals aligned and verified against COLMAP before dense seeding;
* a bounded short-video challenger that replaces LingBot only when both pass and MapAnything has lower normalized camera residual;
* pinned Apache-2.0 source/model assets and empty-cache offline frozen-runtime diagnostics.

## P8 — DA3 integration

Implemented with the strongest refreshed noncommercial checkpoint:

* photo and video camera/depth challenger selected only after COLMAP agreement;
* pose-conditioned metric RGB-D depth behind held-out sensor-anchor and shared validation gates;
* 24-frame/six-overlap bounded streaming with Sim(3), center-residual, and rotation continuity gates;
* direct-Gaussian initialization in bounded windows, with an explicit measured-memory depth-seed
  fallback when the 12 GB quality-preserving memory gate requires it;
* end-to-end preservation of learned opacity and anisotropic scale orientation, with upstream-style
  border filtering and no confidence heuristic that discards valid rendered surfaces;
* pinned offline source/model assets, digest verification, frozen-runtime inference diagnostics,
  and explicit CC BY-NC 4.0 provenance.

## P9 — Learned-first production camera solving

Implemented:

* DA3 proposes the ordered photo/video trajectory before feature matching, with MapAnything as
  the bounded Apache-2.0 proposal fallback and conventional SfM retained when neither proposal is
  valid;
* learned centers, view directions, confidence, and video time build a bounded connected pair
  graph instead of making the learned trajectory a camera observation;
* every proposed edge still passes source-detail ALIKED/LightGlue or SIFT matching and COLMAP
  two-view geometric verification;
* missing views receive a targeted learned-neighbour recovery graph, while an under-gate model
  automatically expands to the established conventional graph and keeps the stronger result;
* the winning reconstruction receives an explicit robust 100-iteration global bundle adjustment
  before learned/COLMAP camera-agreement validation and source-resolution undistortion;
* dataset telemetry records proposal backend, guided/recovery/fallback pair counts, verified
  pairs, inlier matches, recovered cameras, and final-BA status.

## P10 — Unified dense fusion

Implemented with one versioned dense-surface sample contract:

* calibrated RGB-D, validated generated depth, photo geometry, and video geometry retain explicit
  confidence, provenance, surface orientation, footprint, and source-frame ownership;
* confidence/provenance-weighted voxel fusion lets generated and learned surfaces fill unseen
  space while calibrated sensor samples win every occupied metric voxel;
* photo/video projects publish learned-scale colored PLY and OBJ/MTL/PNG artifacts directly from
  the same camera-validated dense prior used for Gaussian initialization;
* hybrid projects align learned media geometry to independently localized RGB-D cameras with a
  robust Sim(3) solve and strict center/rotation gates, then add only non-overlapping learned
  surface triangles to the calibrated TSDF mesh;
* an incompatible learned alignment is recorded and rejected atomically rather than contaminating
  the metric reconstruction;
* the desktop exposes point clouds and meshes for every source mode and labels media-only scale as
  learned rather than metric.

## P11 — Optional neural-SDF production surface

Implemented as an explicit Max Quality option after camera/depth validation:

* the isolated CUDA worker fits a metric neural signed-distance field to the accepted dense
  surface with progressive Fourier levels, near-surface signed supervision, and an
  automatic-differentiation Eikonal constraint;
* RGB-D, hybrid, photo, and video meshes enter the stage only after their existing trajectory,
  scale/depth, and dense-geometry gates have passed;
* a deterministic held-out SDF set measures generalization instead of accepting the training
  loss as proof of surface quality;
* the candidate preserves indexed topology and must pass bounded median/p95/maximum displacement,
  triangle-orientation, and degeneracy gates in the CUDA worker plus an independent displacement
  check in the reconstruction worker;
* accepted geometry reaches depth-aware repair and multiview texturing; rejected, failed, or
  unavailable CUDA refinement leaves the validated TSDF/learned dense surface unchanged;
* content-addressed candidate/report caches make the optional stage reproducible and expose its
  exact outcome in `outputs/neural-sdf-report.json`.

## P12 — Gaussian initialization and production optimization

Implemented with one explicit initialization contract and a bounded source-detail finish:

* sparse SfM, confidence-bearing dense surfaces, and direct learned Gaussians declare distinct
  initialization kinds and representations instead of relying on trainer-side boolean inference;
* direct-head opacity remains renderer state while geometric confidence remains independent fusion
  and loss evidence; contradictory manifests and missing direct opacity fail closed;
* metric 2D surface discs remain fixed, while sparse and learned 3D priors retain bounded adaptive
  densification under the existing hardware Gaussian ceilings;
* global bounded-resolution passes establish appearance and geometry efficiently, then an adaptive
  final schedule covers every calibrated source-resolution tile with exact focal length and shifted
  principal point, without allocating an unsafe full 2.5K/4K raster;
* requested iterations are extended only when required to complete one source-tile pass, and the
  published training sidecar records effective iterations, initialization policy, tile coverage,
  and source-resolution start iteration;
* checkpoint resume preserves per-camera tile position, while incomplete source-pixel coverage
  rejects publication rather than mislabeling a downsample-only result as production quality.
* a deterministic five-view raw-Ply-equivalent photometric gate rejects undertrained or divergent
  media splats, preserves the final checkpoint for more optimization, and records PSNR, SSIM, and
  L1 evidence in `outputs/splat-quality-report.json`.

## P13 — Material and radiometric foundation

Implemented as the fail-closed boundary for every later material stage:

* one source-aligned observation contract separates material identity, overlapping optical risk,
  confidence/validity, and optional linear PBR fields;
* EXIF and embedded ICC color are normalized to sRGB before the exact IEC transfer is decoded into
  content-addressed linear-light inputs, while undeclared HDR/wide-gamut input is rejected;
* a representative-data bake-off uses hard material, optical-risk, multiview, calibration, 12 GB
  memory, and real-capture gates followed by Pareto ranking rather than a paper-only or weighted
  leaderboard;
* frozen candidate revisions and legal terms drive distinct commercial and research manifests;
  unverified, noncommercial, and output-restricted assets cannot enter the commercial pack;
* the initial shortlist covers Material Anything, RGB-to-X, and DiffusionRenderer, but P14 cannot
  package a winner until measured ScanLan evidence and a complete transitive license audit pass.

## P14 — Two-pass material analysis

Implemented without weakening P13's model admission gate:

* calibrated cameras are sampled with path-scale-adaptive translation/rotation k-center selection
  for a bounded coarse optical-risk pass;
* final views are selected by incremental visible-surface coverage, with coarse glass, mirror,
  specular, emissive, thin, dynamic, and sky evidence receiving extra priority;
* final source-aligned predictions are filtered by pose, visibility/depth agreement, viewing angle,
  image border, validity, and calibrated confidence before multiview fusion;
* mutually exclusive identity uses normalized evidence fusion, while independent optical risks use
  both consensus and a conservative high-confidence peak so one sound warning cannot be averaged
  away by many opaque views;
* the versioned surface contract publishes per-vertex probability, confidence, support count,
  effective view count, optional world-space PBR fields, and connected 3D material/risk regions;
* inference remains a callback boundary: no candidate checkpoint becomes a packaged default until
  its real ScanLan annotations, bake-off gates, 12 GB measurement, and transitive license audit pass.

## P15 — Material-aware geometry

Implemented as a conservative post-capture policy rather than an unconditional completion pass:

* source-aligned and fused material evidence produces monotonic confidence multipliers for measured,
  generated, and learned depth, so a material prediction can never manufacture authority;
* strong glass, mirror, thin, dynamic, and sky evidence protects regions from blind filling, while
  missing material output is an explicit neutral no-op;
* boundary repair uses bounded nearest-surface material evidence to veto reflective/transmissive,
  low-authority, and non-static regions without weakening the existing depth/free-space checks;
* second-pass proposals carry provenance, calibrated confidence, effective independent-view count,
  and held-out metric residual, with stricter recovery gates for protected regions;
* accepted displacement is bounded by voxel scale and every changed triangle must pass orientation,
  collapse, and area-stretch gates before an atomic versioned result can replace the baseline;
* no transparent-depth checkpoint is packaged until it passes the P13/P14 real-capture, optical-risk,
  12 GB, visual, and transitive-license gates.

## P16 — PBR reconstruction and export

Implemented as a confidence-preserving intrinsic-material export boundary:

* fused linear albedo, roughness, metallic, transmission, world normal, and emission are baked into
  standards-compliant glTF texture encodings rather than reinterpreting observed RGB as reflectance;
* unsupported material evidence falls back to observed color and a rough, opaque, nonmetallic,
  non-emissive dielectric, so inference cannot manufacture high-risk appearance;
* UV-chart tangents transform fused world normals into OpenGL tangent-space normal maps;
* observed and intrinsic atlases remain distinct and the observed image is embedded as provenance;
* one self-contained GLB uses core metallic-roughness PBR plus `KHR_materials_transmission` and
  `KHR_materials_emissive_strength` when their data requires them;
* atomic atlas, GLB, and `outputs/pbr-report.json` publication makes channel/color semantics and
  material coverage explicit.

## P17 — Material-aware Gaussian reconstruction

Implemented as a prior-gated decomposition in the production trainer:

* a versioned, fail-closed linear-sRGB seed contract aligns P14/P16 albedo, roughness, metallic,
  transmission, emission, and confidence with every initialization Gaussian;
* diffuse degree-zero radiance, higher-order view dependence, non-negative emission, and
  transmission are distinct optimizer/checkpoint tensors that follow gsplat topology changes;
* geometric occupancy remains separate from optical opacity, which is reduced by transmission
  without deleting the surface primitive;
* prior anchoring, unsupported-region penalties, roughness regularization, and specular-aware
  transmission-gradient gating limit decomposition ambiguity;
* datasets without a declared material prior retain the established display-space trainer as an
  exact neutral compatibility path;
* an atomic `outputs/room-splat-material.npz` preserves the lossless linear decomposition beside
  the interoperable display-space PLY, and manifests expose its component and opacity semantics.

## P18 — Adaptive backend policy

Select live and production backends from benchmarked source, hardware and quality characteristics.

## P19 — Full validation and default selection

Run the complete:

* RGB-D;
* video;
* photos;
* hybrid;
* material;
* reflective/transmissive;
* 12 GB memory;
* cancellation/resume;

release matrix before changing defaults.

---

# 13. Realtime acceptance gates

## RGB-D live preview

Target acceptance criteria:

* capture and archive fidelity are unchanged by preview load;
* tracking processes the full intended sensor stream or explicitly reports degraded operation;
* p95 visible preview latency remains within the defined interactive budget;
* point/surfel or raycast geometry updates are frequent enough for active scan guidance;
* viewport rendering remains responsive independently of geometry-update frequency;
* optional mesh extraction cannot stall tracking;
* memory remains bounded during long room-scale captures;
* integration stops within one frame of an unsafe tracking decision;
* relocalization does not integrate intermediate invalid poses;
* loop correction moves submaps without generating duplicate geometry;
* stopping capture immediately exposes the final provisional map;
* raw capture remains usable if the live engine crashes.

Recommended performance targets for the 12 GB reference machine:

```text
viewport rendering:             30 FPS minimum sustained, 60 FPS target
pose age visible to viewport:   below 100 ms p95
point/surfel map update:        10 Hz or better target
TSDF raycast update:            10 Hz or better target where enabled
mesh update:                    0.5–1 Hz target
tracking-loss integration stop: immediate
capture queue growth:           bounded
live-map memory:                hard capped
```

These are engineering targets and must be adjusted only from measured reference-scene data.

## RGB-only video preview

Initial acceptance criteria:

* first provisional geometry appears quickly after startup;
* the preview progresses without storing an unbounded model context;
* frame processing never blocks recording;
* stale inference work is dropped safely;
* local geometry remains stable over short windows;
* drift is measured and displayed;
* integration freezes on unsafe pose/depth;
* post-stop BA can correct the provisional trajectory;
* final production output is not degraded by provisional errors;
* scale status is always explicit.

RGB-only live preview may remain experimental longer than RGB-D preview, but its architecture must be established before the production learned-backend integrations are finalized.

---

# 14. Updated definition of done

The complete program now requires all of the following:

1. RGB-D recording immediately builds an accumulated reconstruction preview.

2. The live preview contains fused scene geometry rather than only the current camera frame.

3. Tracking, integration and rendering have independent bounded pipelines.

4. The user can identify unscanned, weakly scanned and uncertain areas during capture.

5. Tracking loss visibly freezes integration and supports relocalization.

6. Verified loop closures can correct live submaps without full reintegration.

7. The provisional live reconstruction remains available immediately after capture.

8. Production reconstruction always starts from raw observations, not the provisional mesh.

9. Production work never compromises an active RGB-D capture.

10. Long captures remain bounded through active and completed submaps.

11. Photos and video use LingBot, MapAnything and DA3 through the shared geometry contracts.

12. RGB-only video supports progressive reconstruction preview with explicit drift and scale status.

13. RGB-only video recording continues safely even when preview tracking fails.

14. Full material segmentation and PBR reconstruction remain off the capture-critical path.

15. Optional optical-risk detection can eventually assist live guidance without blocking tracking.

16. Final material-aware geometry can reinterpret uncertain live regions after capture.

17. Every live and final surface retains confidence and provenance.

18. Point cloud, mesh, PBR mesh and Gaussian outputs remain independent validated products.

19. The 12 GB target retains enough GPU headroom for stable realtime RGB-D reconstruction.

20. Realtime reconstruction quality, latency and reliability become protected benchmark gates for every subsequent architecture change.
