# Experimental RGB video preview

ScanLan can progressively publish provisional geometry while LingBot-Map processes an imported
RGB video. The feature is disabled by default and is enabled per project with **Progressive
learned-depth preview**. Its output is guidance only: production COLMAP verification, bundle
adjustment, alignment gates, and Gaussian optimization retain full authority over the result.

## Streaming contract

The pinned LingBot model is driven frame by frame with its upstream causal KV-cache behavior.
The first eight scale frames establish the model gauge; subsequent frames are processed in order.
ScanLan publishes completed eight-frame chunks without rerunning the prefix. The final pose,
depth, and confidence tensors are the same ordered outputs consumed by production reconstruction.

Each chunk becomes one local learned-depth submap. Publication is bounded to:

- 16 resident submaps;
- a compacted archive for older submaps;
- 120,000 preview points total;
- the upstream VRAM-dependent sliding KV window and bounded keyframe interval.

The geometry worker writes an atomic NumPy snapshot. The splat worker validates its finite point
contract and publishes `outputs/build-preview.json` for the desktop. Latest-wins publication means
slow rendering never queues model snapshots or blocks inference.

## Safety and visualization

Every preview reports `MODEL_METRIC_UNVERIFIED`; ScanLan does not imply calibrated metric scale.
Point colors blend source RGB with red-to-green confidence. The reconstruction overlay reports:

- accepted and rejected frame counts;
- confidence;
- drift risk derived from confidence and local translation/rotation continuity;
- resident local-submap count;
- explicit learned-only and integration-frozen state.

A low-confidence frame or implausible local pose step freezes its geometry. Video processing and
production camera solving continue. A FlashInfer-to-SDPA retry replaces the provisional stream
from frame zero so output from two inference executions cannot mix.

The desktop flag currently targets progressive imported video. Live webcam/phone ingest is not
enabled in the UI: recording must first gain a nonblocking frame relay and stale-work policy. The
worker boundary already consumes the required ordered-frame API, so a future live source can feed
the same bounded submap and validation path without changing model ownership.

## Upstream basis

The implementation follows the pinned [LingBot-Map streaming code](https://github.com/Robbyant/lingbot-map/blob/1f480aeb8a47a24656090d46d053115b7fe60435/lingbot_map/models/gct_stream.py)
and its [Geometric Context Transformer paper](https://arxiv.org/abs/2604.14141). ScanLan adds the
publication, uncertainty, cancellation, and production-isolation policy around that upstream
causal inference path.
