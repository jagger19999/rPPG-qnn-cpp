# Android Live HR UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Phase 1 of the approved Live HR UI: three-card live BPM (traditional / deep / watch), collapsible config, ROI face thumbnail via JNI, and launcher icon from `logo.png`—without changing ORT/BLE contracts.

**Architecture:** Keep Camera2/rPPG/deep in native. Add an Android-only ROI processor wrapper that throttles JPEG thumbnails into session state; Java polls `nativeGetRoiJpeg` alongside status. Rebuild `MainActivity` around a scrollable layout with three metric cards and fold sections. Phase 2 (preview Surface + face box) is sequenced after Phase 1 gates pass.

**Tech Stack:** Android Java Views, JNI, OpenCV `imencode`, existing `WatchBleWorker` / `WatchAligner`, Gradle packaging gates.

**Spec:** `docs/superpowers/specs/2026-07-24-android-live-hr-ui-design.md`

---

## File map

| File | Responsibility |
|------|----------------|
| `android/app/src/main/res/mipmap-*/ic_launcher.webp` (+ adaptive XML) | Launcher from `logo.png` |
| `android/app/src/main/AndroidManifest.xml` | `android:icon` / `roundIcon` |
| `android/app/src/main/res/layout/activity_main.xml` | Scroll UI structure |
| `android/app/src/main/res/values/strings.xml` / `colors.xml` | Labels + palette |
| `android/app/src/main/java/.../MainActivity.java` | Wire controls, poll, watch lifecycle |
| `android/app/src/main/java/.../ui/HrMetricCard.java` | One BPM card binder |
| `android/app/src/main/java/.../NativeBridge.java` | `nativeGetRoiJpeg` |
| `android/app/src/main/cpp/native_bridge.cpp` | JNI for ROI bytes |
| `android/app/src/main/cpp/android_camera_session.{hpp,cpp}` | ROI JPEG cache + getter |
| `android/app/src/main/cpp/android_jni_handle.{hpp,cpp}` | Session API glue |
| `android/app/src/test/java/.../HrStatusFormatterTest.java` | Pure Java formatting of card text |
| `tests/test_android_packaging.sh` | Whitelist + icon/manifest gates |
| `docs/.../ANDROID_NEXT_STEPS.md` | Phase 1 device checklist note |

Logo source (external, copy during Task 1): `/Users/wangjie/Documents/keti/rPPG-qnn-cpp/logo.png`

---

### Task 1: Launcher icon from logo.png

**Files:**
- Create: `android/app/src/main/res/mipmap-hdpi/ic_launcher.webp` (and mdpi/xhdpi/xxhdpi/xxxhdpi)
- Create: `android/app/src/main/res/mipmap-anydpi-v26/ic_launcher.xml`
- Create: `android/app/src/main/res/mipmap-anydpi-v26/ic_launcher_round.xml`
- Create: `android/app/src/main/res/drawable/ic_launcher_foreground.xml` **or** mipmap foreground bitmaps
- Create: `android/branding/logo.png` (committed copy of source for rebuild)
- Modify: `android/app/src/main/AndroidManifest.xml`
- Modify: `tests/test_android_packaging.sh`

- [ ] **Step 1: Copy source logo into the repo**

```bash
mkdir -p android/branding
cp /Users/wangjie/Documents/keti/rPPG-qnn-cpp/logo.png android/branding/logo.png
test "$(shasum -a 256 android/branding/logo.png | awk '{print $1}')" = \
  "$(shasum -a 256 /Users/wangjie/Documents/keti/rPPG-qnn-cpp/logo.png | awk '{print $1}')"
```

- [ ] **Step 2: Generate density icons with `sips` (macOS)**

```bash
SRC=android/branding/logo.png
for pair in "mdpi:48" "hdpi:72" "xhdpi:96" "xxhdpi:144" "xxxhdpi:192"; do
  dens="${pair%%:*}"; px="${pair##*:}"
  mkdir -p "android/app/src/main/res/mipmap-${dens}"
  sips -z "$px" "$px" "$SRC" --out "/tmp/ic_${dens}.png"
  # Prefer PNG if webp tooling missing:
  cp "/tmp/ic_${dens}.png" "android/app/src/main/res/mipmap-${dens}/ic_launcher.png"
  cp "/tmp/ic_${dens}.png" "android/app/src/main/res/mipmap-${dens}/ic_launcher_round.png"
done
# Foreground for adaptive (~108dp safe zone): use 432px xxxhdpi-ish
mkdir -p android/app/src/main/res/mipmap-xxxhdpi
sips -z 432 432 "$SRC" --out android/app/src/main/res/mipmap-xxxhdpi/ic_launcher_foreground.png
```

- [ ] **Step 3: Adaptive icon XML**

Create `android/app/src/main/res/mipmap-anydpi-v26/ic_launcher.xml`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">
    <background android:drawable="@color/ic_launcher_background"/>
    <foreground android:drawable="@mipmap/ic_launcher_foreground"/>
</adaptive-icon>
```

Create matching `ic_launcher_round.xml`. Add `android/app/src/main/res/values/colors.xml`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<resources>
    <color name="ic_launcher_background">#1A0508</color>
    <color name="hr_card_bg">#FFFFFFF5</color>
    <color name="hr_watch_accent">#C9A227</color>
    <color name="hr_muted">#666666</color>
    <color name="page_bg">#F4F1EA</color>
</resources>
```

- [ ] **Step 4: Manifest application icon**

In `AndroidManifest.xml` `<application>`:

```xml
android:icon="@mipmap/ic_launcher"
android:roundIcon="@mipmap/ic_launcher_round"
```

- [ ] **Step 5: Packaging gate for icon**

In `tests/test_android_packaging.sh`, after Haar checks, require:

```bash
if [[ ! -f "$root/android/branding/logo.png" ]]; then
  echo "android packaging check: branding logo missing" >&2
  exit 1
fi
if ! grep -Fq 'android:icon="@mipmap/ic_launcher"' "$manifest"; then
  echo "android packaging check: application icon must be @mipmap/ic_launcher" >&2
  exit 1
fi
```

Update `expected_tracked` / source whitelists for new res files and `android/branding/logo.png`.

- [ ] **Step 6: Build APK and dump icon**

```bash
./scripts/build_android.sh
"$ANDROID_SDK_ROOT/build-tools/36.0.0/aapt2" dump xmltree \
  android/app/build/outputs/apk/debug/app-debug.apk --file AndroidManifest.xml \
  | rg 'icon|roundIcon'
```

Expected: icon attributes resolve to mipmap ic_launcher.

- [ ] **Step 7: Commit**

```bash
git add android/branding/logo.png android/app/src/main/res android/app/src/main/AndroidManifest.xml \
  tests/test_android_packaging.sh
git commit -m "feat: set APK launcher icon from branding logo"
```

---

### Task 2: Status → card text formatter (TDD, pure Java)

**Files:**
- Create: `android/app/src/main/java/com/jagger/rppgbench/ui/HrStatusFormatter.java`
- Create: `android/app/src/test/java/com/jagger/rppgbench/ui/HrStatusFormatterTest.java`
- Modify: packaging whitelist

- [ ] **Step 1: Write failing tests**

```java
package com.jagger.rppgbench.ui;

import org.json.JSONObject;
import org.junit.Test;
import static org.junit.Assert.*;

public class HrStatusFormatterTest {
    @Test
    public void traditionalShowsDashWhenInvalid() throws Exception {
        JSONObject json = new JSONObject(
                "{\"heart_rate_available\":true,\"heart_rate_valid\":false,"
                        + "\"bpm\":0,\"traditional_method\":\"green\"}");
        HrStatusFormatter.CardView traditional = HrStatusFormatter.traditional(json);
        assertEquals("--", traditional.primary);
        assertTrue(traditional.secondary.contains("不可用"));
    }

    @Test
    public void traditionalShowsBpmWhenValid() throws Exception {
        JSONObject json = new JSONObject(
                "{\"heart_rate_available\":true,\"heart_rate_valid\":true,"
                        + "\"bpm\":72.4,\"traditional_method\":\"pos\"}");
        HrStatusFormatter.CardView traditional = HrStatusFormatter.traditional(json);
        assertEquals("72", traditional.primary);
        assertTrue(traditional.secondary.toLowerCase().contains("pos"));
    }

    @Test
    public void deepShowsDashWhenDisabled() throws Exception {
        JSONObject json = new JSONObject(
                "{\"deep_enabled\":false,\"deep_result_available\":false}");
        assertEquals("--", HrStatusFormatter.deep(json).primary);
    }

    @Test
    public void watchShowsBpmAndStatus() {
        HrStatusFormatter.CardView watch =
                HrStatusFormatter.watch("STREAMING", 70, null);
        assertEquals("70", watch.primary);
        assertTrue(watch.secondary.contains("STREAMING"));
    }
}
```

- [ ] **Step 2: Run tests — expect compile/fail**

```bash
android/gradlew --no-daemon --project-dir android \
  :app:testDebugUnitTest --tests com.jagger.rppgbench.ui.HrStatusFormatterTest
```

Expected: FAIL (class missing).

- [ ] **Step 3: Implement formatter**

```java
package com.jagger.rppgbench.ui;

import org.json.JSONObject;
import java.util.Locale;

public final class HrStatusFormatter {
    public static final class CardView {
        public final String primary;
        public final String secondary;
        public CardView(String primary, String secondary) {
            this.primary = primary;
            this.secondary = secondary;
        }
    }

    private HrStatusFormatter() {}

    public static CardView traditional(JSONObject json) {
        if (json == null || !json.optBoolean("heart_rate_available", false)
                || !json.optBoolean("heart_rate_valid", false)) {
            String method = json == null ? "" : json.optString("traditional_method", "");
            return new CardView("--", method.isEmpty() ? "不可用" : method + " · 不可用");
        }
        int bpm = (int) Math.round(json.optDouble("bpm", Double.NaN));
        String method = json.optString("traditional_method", "rPPG");
        return new CardView(Integer.toString(bpm), method.toUpperCase(Locale.US) + " · 可信");
    }

    public static CardView deep(JSONObject json) {
        if (json == null || !json.optBoolean("deep_enabled", false)) {
            return new CardView("--", "未启用");
        }
        if (!json.optBoolean("deep_result_available", false)
                || !json.optBoolean("deep_result_valid", false)) {
            String reason = json.optString("deep_invalid_reason", "不可用");
            if (reason.isEmpty()) reason = "不可用";
            return new CardView("--", reason);
        }
        int bpm = (int) Math.round(json.optDouble("deep_bpm", Double.NaN));
        double ms = json.optDouble("deep_inference_ms", Double.NaN);
        String secondary = "ORT CPU";
        if (Double.isFinite(ms)) {
            secondary = String.format(Locale.US, "ORT CPU · %.0fms", ms);
        }
        return new CardView(Integer.toString(bpm), secondary);
    }

    public static CardView watch(String status, Integer bpm, String errorCode) {
        String st = status == null ? "DISCONNECTED" : status;
        if (bpm == null) {
            String sec = errorCode != null ? st + " · " + errorCode : st;
            return new CardView("--", sec);
        }
        return new CardView(Integer.toString(bpm), st + " · 实验参考");
    }

    public static String alignmentLine(
            String status, Double absError, Double coverage) {
        if (status == null) return "对齐 --";
        if (absError == null || coverage == null) {
            return "对齐 " + status;
        }
        return String.format(Locale.US, "对齐 %s · |误差| %.1f · 覆盖 %.0f%%",
                status, absError, coverage * 100.0);
    }
}
```

Note: Android unit tests include `org.json` via android.jar stubs in AGP; if missing, add `testImplementation 'org.json:json:20240303'`.

- [ ] **Step 4: Re-run tests — expect PASS**

- [ ] **Step 5: Update packaging whitelist + commit**

```bash
git add android/app/src/main/java/com/jagger/rppgbench/ui/HrStatusFormatter.java \
  android/app/src/test/java/com/jagger/rppgbench/ui/HrStatusFormatterTest.java \
  android/app/build.gradle tests/test_android_packaging.sh
git commit -m "feat: format live HR card text from status JSON"
```

---

### Task 3: Native ROI JPEG cache + JNI

**Files:**
- Modify: `android/app/src/main/cpp/android_camera_session.hpp`
- Modify: `android/app/src/main/cpp/android_camera_session.cpp`
- Modify: `android/app/src/main/cpp/android_jni_handle.hpp`
- Modify: `android/app/src/main/cpp/android_jni_handle.cpp`
- Modify: `android/app/src/main/cpp/native_bridge.cpp`
- Modify: `android/app/src/main/java/com/jagger/rppgbench/NativeBridge.java`
- Modify: packaging lifecycle token list to include `nativeGetRoiJpeg`

- [ ] **Step 1: Add session API**

In `CameraSessionStatus` / session Impl, add:

```cpp
std::vector<std::uint8_t> latest_roi_jpeg;  // guarded by status_mutex
double last_roi_jpeg_sec{0.0};
```

Public:

```cpp
[[nodiscard]] std::vector<std::uint8_t> latest_roi_jpeg() const;
```

- [ ] **Step 2: Wrap RoiProcessor to throttle JPEG**

Inside `start_processing` `make_roi` lambda, wrap:

```cpp
class ThumbnailRoi final : public IRoiProcessor {
 public:
  ThumbnailRoi(std::unique_ptr<IRoiProcessor> inner,
               AndroidCameraSession::Impl* owner)
      : inner_(std::move(inner)), owner_(owner) {}
  RoiPacket process(const FramePacket& frame) override {
    RoiPacket packet = inner_->process(frame);
    owner_->maybe_publish_roi_jpeg(packet);
    return packet;
  }
 private:
  std::unique_ptr<IRoiProcessor> inner_;
  AndroidCameraSession::Impl* owner_;
};
```

`maybe_publish_roi_jpeg`:

```cpp
void maybe_publish_roi_jpeg(const RoiPacket& packet) {
  if (packet.roi_bgr.empty() || !packet.face.has_value()) {
    std::lock_guard<std::mutex> lock(status_mutex);
    latest_roi_jpeg.clear();
    snapshot.face_found = false;
    return;
  }
  if (packet.timestamp_sec - last_roi_jpeg_sec < 0.25) {  // ~4 FPS max
    return;
  }
  cv::Mat small;
  cv::resize(packet.roi_bgr, small, cv::Size(160, 160));
  std::vector<std::uint8_t> buf;
  if (!cv::imencode(".jpg", small, buf, {cv::IMWRITE_JPEG_QUALITY, 80})) {
    return;
  }
  std::lock_guard<std::mutex> lock(status_mutex);
  latest_roi_jpeg = std::move(buf);
  last_roi_jpeg_sec = packet.timestamp_sec;
}
```

Link `imgcodecs` in Android CMake if not already (`opencv_imgcodecs` or module already pulled by imgproc static — verify OpenCV Android SDK components; add `imgcodecs` to `find_package` / link if encode fails at link time).

- [ ] **Step 3: JNI**

`NativeBridge.java`:

```java
public static native byte[] nativeGetRoiJpeg(long handle);
```

`native_bridge.cpp`: call `get_roi_jpeg(handle)` returning `jbyteArray` (null/empty if none).

`android_jni_handle.cpp`:

```cpp
std::vector<std::uint8_t> camera_session_roi_jpeg(std::int64_t handle) {
  return lookup(handle)->latest_roi_jpeg();
}
```

- [ ] **Step 4: Clear JPEG on stop/start**

In session `start()` reset block and `stop_processing`, clear `latest_roi_jpeg`.

- [ ] **Step 5: Host packaging + Android build**

```bash
bash tests/test_android_packaging.sh "$(pwd)"
./scripts/build_android.sh
```

Expected: build success; Activity still references new API in packaging grep list.

- [ ] **Step 6: Commit**

```bash
git add android/app/src/main/cpp android/app/src/main/java/com/jagger/rppgbench/NativeBridge.java \
  tests/test_android_packaging.sh
git commit -m "feat: expose throttled ROI JPEG thumbnail over JNI"
```

---

### Task 4: Layout XML + metric card views

**Files:**
- Create: `android/app/src/main/res/layout/activity_main.xml`
- Create: `android/app/src/main/res/layout/view_hr_metric_card.xml`
- Create: `android/app/src/main/java/com/jagger/rppgbench/ui/HrMetricCard.java`
- Modify: `strings.xml`
- Modify: packaging whitelist for layouts

- [ ] **Step 1: Card layout**

`view_hr_metric_card.xml`: vertical LinearLayout with `title`, `primary` (28–36sp bold), `secondary` (12sp muted). Background `@color/hr_card_bg`, padding 8dp.

- [ ] **Step 2: Activity layout**

`activity_main.xml` structure:

```text
ScrollView
  LinearLayout (vertical, padding 16dp, bg page_bg)
    title + disclaimer
    LinearLayout horizontal: cardTraditional | cardDeep | cardWatch
      (use include/merge three view_hr_metric_card)
    textAlignment
    ImageView roiImage (160dp) + textRoiPlaceholder
    (optional) TextureView preview — GONE in Phase 1, id reserved
    Expandable/buttons section Camera (initially expanded once)
    Watch section
    Diagnostic section (TextView mono, initially GONE; toggle button)
```

Phone narrow width: if three cards too tight, use `layout_weight=1` each and `primary` textSize 24sp.

- [ ] **Step 3: HrMetricCard binder**

```java
public final class HrMetricCard {
    private final TextView title;
    private final TextView primary;
    private final TextView secondary;
    public HrMetricCard(View root) { /* findViewById */ }
    public void bind(String titleText, HrStatusFormatter.CardView data) {
        title.setText(titleText);
        primary.setText(data.primary);
        secondary.setText(data.secondary);
    }
}
```

- [ ] **Step 4: Commit layouts**

```bash
git commit -m "feat: add live HR scroll layout and metric card views"
```

---

### Task 5: Rewire MainActivity to layout (Phase 1 UI)

**Files:**
- Modify: `MainActivity.java` (replace programmatic LinearLayout with `setContentView(R.layout.activity_main)`)
- Keep: watch worker lifecycle, CSV export, permissions

- [ ] **Step 1: onCreate inflate and bind**

```java
setContentView(R.layout.activity_main);
traditionalCard = new HrMetricCard(findViewById(R.id.card_traditional));
deepCard = new HrMetricCard(findViewById(R.id.card_deep));
watchCard = new HrMetricCard(findViewById(R.id.card_watch));
roiImage = findViewById(R.id.roi_image);
roiPlaceholder = findViewById(R.id.roi_placeholder);
diagnosticText = findViewById(R.id.diagnostic_text);
// existing spinners/buttons via findViewById
```

- [ ] **Step 2: Replace refreshCombinedStatus**

```java
private void refreshCombinedStatus() {
    String cameraJson = "{}";
    if (started && nativeHandle != 0) {
        cameraJson = NativeBridge.nativeGetStatus(nativeHandle);
        maybeAlignFromCameraStatus(cameraJson);
        byte[] jpeg = NativeBridge.nativeGetRoiJpeg(nativeHandle);
        if (jpeg != null && jpeg.length > 0) {
            Bitmap bmp = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.length);
            roiImage.setImageBitmap(bmp);
            roiImage.setVisibility(View.VISIBLE);
            roiPlaceholder.setVisibility(View.GONE);
        } else {
            roiImage.setVisibility(View.GONE);
            roiPlaceholder.setVisibility(View.VISIBLE);
        }
    }
    JSONObject json = new JSONObject(cameraJson); // try/catch
    traditionalCard.bind("传统 rPPG", HrStatusFormatter.traditional(json));
    deepCard.bind("深度 EfficientPhys", HrStatusFormatter.deep(json));
    // watch snapshot…
    watchCard.bind("广播心率", HrStatusFormatter.watch(...));
    alignmentView.setText(HrStatusFormatter.alignmentLine(...));
    if (diagnosticExpanded) {
        diagnosticText.setText(cameraJson + "\n" + formatWatchStatusLine());
    }
}
```

- [ ] **Step 3: Fold toggles**

Camera/Watch sections: `View.GONE`/`VISIBLE` on header click. Diagnostic default GONE.

- [ ] **Step 4: Manual sanity on device after install**

```bash
./scripts/build_android.sh
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.jagger.rppgbench/.MainActivity
```

Expected: three cards on top; JSON only in diagnostic when expanded; launcher shows logo.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: present three live HR cards and ROI thumbnail in MainActivity"
```

---

### Task 6: Docs + host verification (Phase 1 exit)

**Files:**
- Modify: `ANDROID_NEXT_STEPS.md`
- Optional: `README.md` one-liner on Live UI

- [ ] **Step 1: Document Phase 1 device checks** (cards, ROI, icon, folds)

- [ ] **Step 2: Full host verify**

```bash
cmake --build build-camera2-release -j && ctest --test-dir build-camera2-release --output-on-failure
android/gradlew --no-daemon --project-dir android :app:testDebugUnitTest
./scripts/build_android.sh
```

Expected: all green; APK has icon + no `.onnx`.

- [ ] **Step 3: Commit docs**

```bash
git commit -m "docs: note Live HR UI phase-1 device gates"
```

---

### Task 7 (Phase 2): Preview Surface + face box

**Files:**
- Modify: `android_camera_session` for dual output or frame push
- Modify: `activity_main.xml` — show `preview` TextureView
- Add: `face_rect` to status JSON (`x,y,w,h` normalized 0–1)
- Overlay: custom `FaceBoxOverlay` View

- [ ] **Step 1: Prefer Camera2 dual targets** (preview Surface + ImageReader); if session config blocked on device, fall back to 5–10 FPS JPEG full-frame push separate from ROI.

- [ ] **Step 2: Publish `face_rect` when face present; overlay draws stroke rect.

- [ ] **Step 3: Keep ROI ImageView; preview failure hides preview only.

- [ ] **Step 4: Device gate — FPS not collapsed; commit `feat: add camera preview with face box overlay`.

Do not start Task 7 until Phase 1 device checklist in Task 6 is acknowledged.

---

## Spec coverage checklist

| Spec item | Task |
|-----------|------|
| Layout A three cards | 4, 5 |
| Alignment line | 2, 5 |
| ROI 160 JPEG 2–5 FPS | 3, 5 |
| Fold camera/watch/diagnostic | 4, 5 |
| Launcher logo.png | 1 |
| Stop ≠ watch disconnect | 5 (preserve) |
| Phase 2 preview + box | 7 |
| Packaging / host tests | 1, 2, 3, 6 |
| No JSON wall on first screen | 5 |

## Placeholder / consistency review

- JNI name `nativeGetRoiJpeg` consistent across Java/C++/packaging.
- Formatter card titles match Chinese labels in layout.
- Phase 2 explicitly gated after Phase 1.
- No TBD steps; OpenCV imgcodecs link called out as verify-during-Task-3.
