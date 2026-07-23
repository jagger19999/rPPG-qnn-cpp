# POS and CHROM C++ Port Implementation Plan

## Overview

Port the Python project's POS and CHROM traditional rPPG baselines into the
independent C++ runtime without changing deep inference or packaging boundaries.

## Architecture decisions

- Use one `TraditionalPredictor` with a fixed method per process.
- Preserve `GreenPredictor` as a default-GREEN compatibility alias.
- Share RGB timestamp resampling, spectral estimation, quality gates, and output.
- Freeze Python/SciPy 30 FPS filter coefficients and verify them with golden values.
- Keep OpenCV as the only numerical dependency.

## Tasks

### Task 1: RED contracts

Acceptance criteria:

- Config tests require `pos` and `chrom` to parse and continue rejecting unknown methods.
- Pure BVP and predictor tests fail before the new implementation exists.
- Pipeline tests require emitted method names to match the selected algorithm.

Verification: build the focused tests and record the expected failure.

### Task 2: Shared predictor and POS

Acceptance criteria:

- RGB history and timestamp resampling preserve existing GREEN behavior.
- POS raw BVP matches Python golden values within the frozen tolerance.
- POS returns a finite, valid 72 BPM result for the deterministic RGB signal.

Verification: config, GREEN compatibility, and POS focused tests pass.

### Task 3: CHROM

Acceptance criteria:

- CHROM raw BVP matches Python golden values and aggregate statistics.
- CHROM returns a finite, valid 72 BPM result.
- Degenerate input never produces NaN/Inf or silent GREEN fallback.

Verification: all traditional predictor tests pass.

### Task 4: Pipeline and operator documentation

Acceptance criteria:

- Pipeline constructs the configured method and emits matching CSV/JSON method values.
- README and bench runbook document `green|pos|chrom` and single-method execution.
- Four-file package whitelist remains unchanged.

Verification: pipeline and packaging tests pass.

### Task 5: Final gates and publication

Acceptance criteria:

- Native Release and UBSan test suites pass.
- No model/generated artifact becomes tracked.
- Original Python repository is unchanged.
- Atomic commits are pushed to public GitHub `main`.

Verification: compare remote `main` SHA and inspect the public tree.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| SciPy/C++ filtering differs | Freeze coefficients and selected golden waveform values. |
| POS detrending is expensive | Run once per second on 300 samples; record performance before optimizing. |
| Refactor changes GREEN | Preserve compatibility alias and run the full existing GREEN suite. |
| Invalid method silently falls back | Strict parser plus emitted method assertions. |
| Algorithm blocks capture | One window evaluation per second; measure and keep deep worker independent. |

## Scope boundary

LGI, CONSENSUS, multiple simultaneous traditional methods, QNN inference, and UI
changes remain separate work.
