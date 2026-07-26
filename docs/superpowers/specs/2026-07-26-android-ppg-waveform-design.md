# Android Traditional and TSCAN PPG Waveform Design

## Goal

Restore the two live PPG waveform panels from the Python local web UI in the
Android APK while preserving camera throughput and inference responsiveness.
The traditional and deep panels must remain independent: traditional data is
always presented when available, while the deep panel reports its own disabled,
sampling, inference, invalid, or valid state.

## Python parity target

Both platforms present the latest complete processing window rather than an
unbounded history. Waveform values are finite, centered, and normalized to
`[-1, 1]`. The horizontal axis is relative time ending at zero. A valid signal
uses green (`#2f8f74`); an invalid or low-quality signal uses orange
(`#b86b2b`). The chart labels identify the method and distinguish signal state
from the numeric BPM result.

The Android implementation intentionally excludes hover interaction, zooming,
and historical storage because those browser behaviors do not contribute to
the parity goal and would increase rendering cost.

## UI design

Add a `PPG 波形` section immediately below the heart-rate cards and above the
camera preview. It contains two vertically stacked native chart cards:

1. `传统 rPPG 波形` is always visible.
2. `深度 TSCAN 波形` always reserves its place so the page does not jump when
   deep inference is enabled.

Each card contains a title, a compact status label, and a custom
`PpgWaveformView`. The view draws a subtle zero line, optional horizontal grid
lines, the polyline, and relative-time endpoint labels. It does not allocate
objects inside `onDraw` and does not animate points.

Traditional states:

- Before a complete window: `正在采集传统算法数据`
- Valid waveform: green line plus method and window duration
- Invalid/low-quality waveform: orange line plus the existing invalid reason
- Camera stopped: retain the last waveform until a new session begins

Deep states:

- Deep unchecked: `未启用深度推理`
- Enabled but below 180 frames: `正在采集深度窗口 N / 180`
- Complete input awaiting output: `TSCAN 推理中`
- Invalid output: orange line plus the TSCAN invalid reason
- Valid output: green line plus `TSCAN · 180 帧 · 约 6 秒`
- Missing model/runtime failure: state text only; traditional chart continues

Starting a new camera session clears both previous-session waveforms. Stopping
the current session retains the most recent complete charts for inspection.

## Native data model

The camera session status owns two immutable latest-snapshot records:

- `traditional_waveform`: values, sample rate, validity, method, and a monotonically
  increasing revision
- `deep_waveform`: the raw 180-sample TSCAN model output, sample rate, validity,
  invalid reason, and its own revision

Traditional data is copied from the same predictor result that produces the
displayed traditional BPM. Deep data is copied from the same TSCAN result that
produces the displayed deep BPM. This guarantees that each chart and card refer
to the same completed window.

Normalization is performed once in native code when publishing a snapshot:
subtract the mean, divide by the largest absolute centered value, reject
non-finite or effectively constant data, and clamp numerical residue to
`[-1, 1]`. This mirrors the Python `RppgWaveformSnapshot` contract.

The session also exposes `deep_frames_collected`, `deep_frames_required`, and
`deep_inference_pending` so the UI can distinguish sampling from inference.

## JNI contract

Do not append hundreds of floating-point values to the existing once-per-second
status JSON. Add a dedicated JNI snapshot call that returns both waveform
metadata and `float[]` values only when requested.

Java polls lightweight status once per second as it does today. When a waveform
revision changes, it requests that waveform snapshot and updates the relevant
view. Unchanged revisions cause no array copy and no redraw. A 180-400 point
snapshot is therefore copied only once per completed processing window.

Snapshot reads take a short mutex only long enough to copy the latest immutable
record. Rendering and JSON parsing never occur on the camera or inference
threads.

## Android components

- `PpgWaveformSnapshot`: immutable Java value object containing values and
  display metadata.
- `PpgWaveformView`: native Canvas renderer with preallocated `Paint` and
  reusable `Path` objects.
- `PpgWaveformCard`: binds title, status, color, and snapshot to one layout.
- `MainActivity`: coordinates status polling and revision-based snapshot fetches;
  it does not contain waveform drawing logic.

The renderer maps sample index uniformly across the chart width. Because the
runtime uses a stable sample rate within one result window, index-based drawing
is equivalent to relative timestamps while avoiding a second timestamp array.
The labels derive start time as `-(sample_count - 1) / sample_rate_hz` and end
time as `0 s`.

## Error handling

- Empty, non-finite, constant, or mismatched data is rejected before publication.
- JNI returns an explicit unavailable snapshot rather than throwing for a normal
  sampling or disabled state.
- A malformed Java snapshot clears only the affected chart and displays its
  state; it does not stop camera capture.
- Deep runtime/model errors update only the deep card and chart.
- Activity destruction stops polling and releases references to waveform arrays.

## Performance constraints

- No per-frame JNI waveform calls.
- No per-frame UI invalidation.
- At most one redraw per new completed waveform revision.
- No WebView, JavaScript chart engine, bitmap chart generation, or new charting
  dependency.
- The renderer handles at most 400 traditional points and 180 TSCAN points.
- Existing camera frame orientation and ROI processing remain unchanged.

## Testing

Native unit tests cover normalization, invalid waveform rejection, revision
updates, session reset, and correspondence between BPM and waveform results.
JNI tests cover unavailable snapshots, metadata, and exact sample transfer.
Java unit tests cover state mapping, progress text, revision deduplication, and
relative-time label calculation. Renderer tests cover coordinate mapping through
a pure geometry helper rather than fragile screenshot assertions.

The final device check runs traditional-only and traditional-plus-TSCAN modes in
portrait. It verifies both chart states, valid waveform rendering, independent
failure behavior, correct camera orientation, and no material FPS regression.

## Acceptance criteria

- Traditional waveform appears after its first complete result without enabling
  deep inference.
- The deep panel progresses from disabled to sampling to inference to a valid or
  invalid waveform state.
- Each waveform is sourced from the same result window as its displayed BPM.
- Both charts use normalized `[-1, 1]` data, relative time ending at zero, and
  Python-compatible green/orange quality colors.
- Deep failure never hides or clears a valid traditional waveform.
- Starting a new session clears stale charts; stopping retains the last charts.
- Portrait camera preview remains upright.
- Unit tests and Android debug assembly pass, and the feature is verified on the
  connected phone.
