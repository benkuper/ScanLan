# ScanLan Engineering Expectations

These instructions apply to the entire repository.

## Own the technical solution

When the user asks to improve quality, add a feature, or make something the
"best possible" / "AAA":

- Treat the requested outcome as the goal, not the current implementation as
  the design. Reconsider architecture, algorithms, data flow, and dependencies
  when the existing approach limits the result.
- Do not wait for the user to prescribe implementation steps. Proactively
  identify what determines quality, choose the approach, implement it, and
  validate it.
- Do not merely tune an existing constant when the real problem calls for a
  data-driven or structurally different solution.
- Make decisions from the input and task semantics. Prefer adaptive behavior
  based on measured signal (for example motion, overlap, confidence, or error)
  over arbitrary fixed sampling or limits.

## Find and use the best fitting tools

- Research the current state of the art before making a significant quality or
  architecture decision. Use primary sources, official documentation, papers,
  and upstream implementations.
- Compare mature external tools and models with an in-house implementation.
  Prefer the strongest proven solution that is compatible with ScanLan's
  platform, hardware, licensing, packaging, and maintenance constraints.
- Integrate useful acceleration paths and high-quality algorithms when they
  materially improve the result. Verify that optional acceleration is actually
  active; do not claim it from installation alone.
- Pin external code and model revisions required for reproducibility. Package
  all runtime assets needed for an offline desktop build.
- If the objectively strongest tool cannot be used, document the concrete
  incompatibility and implement the next-best option. Do not silently settle
  for an easier approach.

## Quality first, then measured speed

- Establish a measurable quality bar before optimizing. Never trade away the
  required result for speed without surfacing the tradeoff.
- Optimize measured bottlenecks with appropriate GPU kernels, mixed precision,
  batching, streaming, caching, and bounded-memory algorithms.
- A fast unusable result is a failure. A high-quality path that crashes or has
  unbounded memory is also a failure.
- Use quality gates and safe fallbacks. Reject incompatible model output instead
  of contaminating a reconstruction merely because inference completed.

## Validate the actual outcome

- Test the full pipeline on representative real input, not only synthetic unit
  tests. For reconstruction work, use the user's real capture whenever it is
  safely available.
- Combine automated tests with objective task metrics. Relevant reconstruction
  checks include camera-path continuity, registration coverage, reprojection or
  alignment error, held-out view quality, memory use, and runtime.
- Inspect the final visual artifact. A completed process, decreasing loss, or
  passing unit tests does not prove that a render is usable.
- Do not call work finished until the requested feature works end to end and the
  real output meets the stated quality bar. Clearly label diagnostic or failed
  artifacts so they cannot be mistaken for final output.
- When an approach fails, report the evidence, determine why, and continue with
  the strongest justified next approach rather than defending sunk work.

## Communicate judgment clearly

- Lead with conclusions and evidence. State important assumptions, quality
  criteria, constraints, and tradeoffs without requiring the user to discover
  them first.
- When "best quality" and "fastest" conflict, find the best quality-preserving
  Pareto point and explain it. Ask for a user decision only when the remaining
  tradeoff materially changes the requested outcome.
- Be candid about uncertainty and limitations. Agreement with user feedback
  must be supported by technical reasoning and followed by implementation and
  verification.
