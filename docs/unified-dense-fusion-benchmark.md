# Unified dense fusion validation

P10 was validated on 10 August 2026 with the retained P9 real-phone fixture: 16 indoor photos at
up to 1600 px, 15 bundle-adjusted production cameras, and a camera-agreement-gated DA3 direct
prior containing 750,000 learned samples.

## Media-only artifact result

| Measure | Result |
| --- | ---: |
| Input learned samples | 750,000 |
| Confidence/provenance-fused PLY points | 589,679 |
| Adaptive point voxel | 0.04484 learned units |
| Bounded mesh voxel | 0.11074 learned units |
| OBJ geometry vertices | 103,714 |
| OBJ triangles | 153,302 |
| Point + mesh publication time | 26.3 s |
| Mesh edge-manifold check (boundary allowed) | Pass |

The published bundle contained a binary colored `room-cloud.ply`, indexed
`room-mesh.obj`, `room-mesh.mtl`, `room-texture.png`, and preview buffer. Orthographic inspection
of both the cloud and uniformly sampled mesh showed the same coherent indoor layout, furniture,
and dominant room surfaces. The learned reconstruction still contains the expected thin and
weakly observed fragments; P10 does not relabel those arbitrary-scale predictions as metric.

The first artifact run exposed a contract error: legacy direct-Gaussian `confidence` stores
renderer opacity, not geometry confidence, and reduced the surface to 203 triangles when treated
as a fusion gate. The final contract keeps direct opacity unchanged for Gaussian training and
publishes independent `fusion_confidence`; legacy direct-prior sidecars are interpreted by their
declared `directGaussianPrior` flag. Revalidation produced the 153,302-triangle result above.

## Automated gates

- exact Sim(3) camera alignment and measured-voxel precedence fixtures;
- sidecar shape, finiteness, confidence, provenance, and ownership validation;
- synthetic end-to-end PLY plus OBJ/MTL/PNG publication and reload;
- 121 reconstruction-worker tests;
- 73 media/splat tests;
- 35 Rust desktop tests;
- zero Svelte diagnostics and a successful production web build.
