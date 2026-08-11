# V2 release matrix and default selection

P19 is a release boundary, not a claim that the existing integration smokes cover production.
`scanlan-validation` owns a schema-1 matrix with ten independent scenarios: room-scale captures on
each supported RGB-D sensor, multi-scene photo/video/hybrid production, annotated material and
reflective/transmissive captures, an actual 12 GB CUDA run, and cancellation/resume stress.

## Promotion contract

Each scenario must provide a versioned evidence record with:

- representative real input and the exact source/sensor class required by that scenario;
- explicit release gates rather than a successful process exit;
- finite task metrics that meet the scenario's quality and latency thresholds;
- SHA-256-bound final artifacts; and
- an independent, timestamped visual inspection where geometry or appearance is judged.

One scene cannot stand in for another, a 16 GB run cannot stand in for the 12 GB target, synthetic
contract tests cannot stand in for annotated material captures, and installed CUDA packages cannot
stand in for executed kernels. Duplicate evidence is rejected as ambiguous. Artifact hashing is
streamed in bounded chunks so validating a large PLY, GLB, or checkpoint does not duplicate it in
memory.

The evaluator always writes the full scenario report. It exits with code 2 while any required case
is missing or failed. A separate default-promotion artifact can be written only after the complete
matrix passes; the library raises before creating it otherwise. This keeps ordinary adaptive
backend choices available as explicit, benchmark-gated options while preventing them from becoming
global defaults on partial evidence.

Run the canonical audit with:

```powershell
.\build\worker-venv\Scripts\python.exe -m pip install --no-deps .\validation
.\build\worker-venv\Scripts\scanlan-release-matrix.exe `
  --evidence validation\release-evidence\v2-p19-audit-2026-08-11.json `
  --report docs\v2-release-matrix-report.json
```

For a release candidate, supply reviewed evidence files for all scenarios and add
`--promote-defaults path\to\default-backends.json`. Do not use `--no-verify-artifacts` for a
release decision; it exists only to inspect incomplete declarations whose artifacts are not
available on the current machine.

## 2026-08-11 audit result

The first canonical audit is incomplete, so no defaults changed. The committed
`v2-release-matrix-report.json` records every failed gate. The decisive evidence is:

- Kinect v2 and Azure Kinect have no representative room-scale V2 run.
- The physical Femto capture validates the measurement path, bounded map, and production replay,
  but it is a short mostly-static sequence with no real relocalization/loop event and its 365.8 ms
  point-map p95 misses the 10 Hz target.
- The learned-first phone-photo solve registers 15/16 cameras, but it is one scene; its inspected
  Gaussian candidate also misses the PSNR and SSIM publication gates.
- The real 4K video solve registers 85/181 selected views (46.96%), below the release threshold.
- No multi-scene hybrid run, annotated material/optical-risk set, or real reflective/transmissive
  reconstruction has completed the required quality gates.
- CUDA inference was measured on a 16 GB RTX 5080 Laptop GPU, not an actual 12 GB device.
- A real Gaussian checkpoint resumed successfully, but output-equivalence and corrupted-cache
  recovery have not completed the full cancellation/resume cross-product.

Contract, unit, frozen-runtime, synthetic PBR, and real single-capture smokes remain valuable
regression evidence. They are deliberately not counted as P19 release acceptance.
