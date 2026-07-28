# Android ORT thread benchmark

Date: 2026-07-27 (Asia/Shanghai)

## Scope and fixed conditions

- Device: Redmi `23013RK75C`, Qualcomm `SM8475`, adb serial `50e584de`
- ABI: `arm64-v8a`
- ORT: Android 1.27.0, CPU execution provider, graph optimization `ORT_ENABLE_ALL`
- Inter-op threads: fixed at 1
- Candidate intra-op threads: 2, 4, 6
- Camera: Camera2 device 1, UI label `前置 (1)`, 30 FPS target
- Installation: `adb install -r`; application data was not cleared between candidates
- TSCAN model SHA-256: `342a3c8033dda9ab154e85d5a4e2a876a6461648b7fcb27c46a7023e662bcc64`
- EfficientPhys model SHA-256: `c1b321042db1335da70b0295cc84f653a2cfe90f75cff738b3045ea3c103257d`

The shared benchmark override was built with:

```text
./gradlew :app:assembleDebug -PrppgOrtIntraOpThreads=<2|4|6> --rerun-tasks
```

This sets both model thread counts for a candidate. The production build also
accepts `rppgTscanOrtIntraOpThreads` and
`rppgEfficientPhysOrtIntraOpThreads` independently. CMake and runtime validation
accept only 2, 4, or 6 intra-op threads; inter-op remains exactly 1.

## Candidate APK identity

| Intra-op | APK SHA-256 |
|---:|---|
| 2 | `048ab5c5022efe76d33003908dc760e6b0c0432060aa01d6d3f73f75143d524a` |
| 4 | `f8f82100bfb9a30580256ecccccb81e1e93aa447e697eed297103956517833d1` |
| 6 | `2a924523ceae9fe59791293718fc12bc70eaeb1d2372406a1cf1a9fa388eaab4` |

## Sustained measurements

Each completed cell requires at least three completed deep windows. The first
window is recorded as warm-up and excluded from the default-selection median.
Memory is `dumpsys meminfo` total PSS; battery temperature is reported in °C.

| Model | Intra-op | Warm-up runtime ms | Steady runtime ms (n/median) | Inference ms (n/median) | Capture FPS | Dropped/replaced | PSS MiB | Temperature °C | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TSCAN | 2 | pending | pending | pending | 29.0–30.4 before valid ROI | pending | 625.4 | 34.0→38.7 | blocked: black preview, no face |
| TSCAN | 4 | pending | pending | pending | pending | pending | pending | pending | not run |
| TSCAN | 6 | pending | pending | pending | pending | pending | pending | pending | not run |
| EfficientPhys | 2 | pending | pending | pending | pending | pending | pending | pending | not run |
| EfficientPhys | 4 | pending | pending | pending | pending | pending | pending | pending | not run |
| EfficientPhys | 6 | pending | pending | pending | pending | pending | pending | pending | not run |

## Current limitation and default decision

The 2-thread TSCAN candidate was installed without clearing application data,
and both app-private model hashes still matched. The app connected to front
camera device 1 and sustained roughly 29–30 FPS, but a device screenshot showed
an almost completely black preview. Every emitted frame-health event reported
`face_found=false`; consequently no 180-frame ROI window or genuine ORT timing
was produced during the approximately ten-minute session. The closed
`heart_rate.csv` contains only its header. These values must not be used to rank
thread counts. The session was then stopped to avoid further heating.

No fastest configuration has therefore been claimed. Model-specific defaults
remain the conservative existing value of 2 until the same phone is positioned
with an unobstructed, steadily lit face and all six cells above are collected.
A fresh test session can start as soon as a face can remain in the preview.

## Allocation guard

TSCAN now owns its difference and `[180,6,72,72]` output scratch tensors inside
one worker-local preprocessor. EfficientPhys similarly owns its
`[181,3,72,72]` model-input tensor inside one runtime. A→B→A repeated-call tests
prove exact tensor/result restoration and guard against cross-call pollution;
the compatibility TSCAN preprocessing wrapper remains available.
