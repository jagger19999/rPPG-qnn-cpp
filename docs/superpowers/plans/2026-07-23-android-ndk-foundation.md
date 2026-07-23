# Android NDK Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the first safe Android vertical slice: preserve the existing Linux runtime, add a cancellable portable pipeline seam, and create an `arm64-v8a` APK/NDK smoke application that reports native build identity without claiming camera or QNN inference.

**Architecture:** Keep the existing C++ algorithms and Linux executable intact. Add a minimal Java `Activity` and JNI shared library in `android/`; JNI initially exposes only immutable build/runtime capability status. Prepare `Pipeline` for Android lifecycle cancellation through an injected stop callback, while Camera2 and real QNN remain separate follow-on plans gated by bench facts.

**Tech Stack:** C++17, CMake 3.22+, OpenCV 4 host build, Android Gradle Plugin 9.0.1, Gradle 9.1.0, JDK 17, Android NDK 28.2.13676358, Java Android Activity, JNI, `arm64-v8a`, host CTest.

---

## Scope and file map

This plan implements the independently testable first half of design Slice 1.
It does not open a camera, request camera permission, link the complete native
pipeline into the APK, package QNN libraries, load a model, or expose fake deep
output. The Camera2 plan performs the root-CMake platform split when the first
Android frame source is available to exercise that boundary.

### Existing files modified

- `include/rppg_qnn/pipeline.hpp`: add a platform-neutral cooperative stop callback to `PipelineDependencies`.
- `src/pipeline.cpp`: stop capture at a frame boundary when the callback requests it.
- `tests/test_pipeline.cpp`: prove cancellation exits cleanly and publishes a successful summary.
- `CMakeLists.txt`: make host executable/test construction explicit and add a small portable build-identity source to the core.
- `.gitignore`: ignore Android/Gradle local state and APK build outputs.
- `README.md`: document the Android foundation boundary and exact local prerequisites.

### New portable files

- `include/rppg_qnn/build_identity.hpp`: stable native build identity API usable by JNI and host tests.
- `src/build_identity.cpp`: compile-time platform/ABI/capability identity with no Android headers.
- `tests/test_build_identity.cpp`: host contract tests.

### New Android files

- `android/settings.gradle`: plugin repositories and application module.
- `android/build.gradle`: pinned AGP plugin declaration.
- `android/gradle.properties`: conservative Gradle settings.
- `android/app/build.gradle`: API/ABI/NDK configuration and external CMake integration.
- `android/app/proguard-rules.pro`: empty, explicit application rules file.
- `android/app/src/main/AndroidManifest.xml`: launcher activity; no camera permission in this slice.
- `android/app/src/main/java/com/jagger/rppgbench/MainActivity.java`: minimal status UI.
- `android/app/src/main/java/com/jagger/rppgbench/NativeBridge.java`: loads JNI library and exposes build identity.
- `android/app/src/main/cpp/CMakeLists.txt`: builds `librppg_qnn_android.so` and the portable build identity.
- `android/app/src/main/cpp/native_bridge.cpp`: exception-safe JNI adapter.
- `android/app/src/main/res/values/strings.xml`: application strings.
- `scripts/build_android.sh`: fail-fast Android environment validation and debug APK build.
- `tests/test_android_packaging.sh`: repository-level contract for ABI, permissions, forbidden artifacts, and fake-deep exclusion.

## Task 1: Cooperative pipeline stop seam

**Files:**
- Modify: `include/rppg_qnn/pipeline.hpp`
- Modify: `src/pipeline.cpp`
- Modify: `tests/test_pipeline.cpp`

- [ ] **Step 1: Write the failing cancellation test**

Add a deterministic in-memory source to `tests/test_pipeline.cpp` that can produce more frames than the stop callback allows:

```cpp
class InfiniteCountingSource final : public rppg_qnn::FrameSource {
 public:
  explicit InfiniteCountingSource(int* reads) : reads_(reads) {}
  std::optional<rppg_qnn::FramePacket> read() override {
    rppg_qnn::FramePacket packet;
    packet.frame_id = static_cast<std::uint64_t>(*reads_);
    packet.timestamp_sec = static_cast<double>(*reads_) / 30.0;
    packet.bgr = cv::Mat(96, 96, CV_8UC3, cv::Scalar(20, 100, 20));
    ++*reads_;
    return packet;
  }
  bool eof() const override { return false; }
  double nominal_fps() const override { return 30.0; }

 private:
  int* reads_;
};
```

Add a test that injects `should_stop`, returns `true` after five accepted frames, and asserts `Pipeline::run()` returns `0`, the source read count is five, and `session_summary.json` records exit code zero:

```cpp
void pipeline_honors_cooperative_stop_at_frame_boundary() {
  ScopedDirectory output;
  int reads = 0;
  rppg_qnn::AppConfig config;
  config.video = "unused.avi";
  config.deep = "disabled";
  config.output = output.path() / "cooperative-stop";
  rppg_qnn::PipelineDependencies dependencies{
      [&] { return std::make_unique<InfiniteCountingSource>(&reads); },
      [] { return std::make_unique<FixedRoi>(); },
      {},
      {},
      [&] { return reads >= 5; }};

  rppg_qnn::Pipeline pipeline(config, std::move(dependencies));
  EXPECT_EQ(pipeline.run(), 0);
  EXPECT_EQ(reads, 5);
  EXPECT_TRUE(read_file(config.output / "session_summary.json").find(
                  "\"exit_code\":0") != std::string::npos);
}
```

Call `pipeline_honors_cooperative_stop_at_frame_boundary()` from the existing
`main()` before `return test_support::finish();`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
cmake -S . -B build-android-foundation-red -DCMAKE_BUILD_TYPE=Debug
cmake --build build-android-foundation-red --target test_pipeline --parallel
```

Expected: compilation fails because `PipelineDependencies` has no fifth `should_stop` member.

- [ ] **Step 3: Add the minimal stop dependency**

Append to `PipelineDependencies` in `include/rppg_qnn/pipeline.hpp`:

```cpp
std::function<bool()> should_stop;
```

At the start of the capture loop in `src/pipeline.cpp`, before `source->read()`:

```cpp
if (dependencies_.should_stop && dependencies_.should_stop()) {
  break;
}
```

Update every existing aggregate initialization of `PipelineDependencies` with a final `{}` where needed. Do not add signals, Android headers, global flags, or forced thread cancellation.

- [ ] **Step 4: Build and verify GREEN**

Run:

```bash
cmake --build build-android-foundation-red --target test_pipeline --parallel
ctest --test-dir build-android-foundation-red -R '^pipeline$' --output-on-failure
```

Expected: `pipeline` passes and the process exits without waiting for source EOF.

- [ ] **Step 5: Run the complete native suite**

Run:

```bash
ctest --test-dir build-android-foundation-red --output-on-failure
```

Expected: all existing tests plus the cancellation assertion pass.

- [ ] **Step 6: Commit the stop seam**

```bash
git add include/rppg_qnn/pipeline.hpp src/pipeline.cpp tests/test_pipeline.cpp
git commit -m "feat: add cooperative pipeline stop seam"
```

## Task 2: Portable native build identity

**Files:**
- Create: `include/rppg_qnn/build_identity.hpp`
- Create: `src/build_identity.cpp`
- Create: `tests/test_build_identity.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing host contract test**

Create `tests/test_build_identity.cpp`:

```cpp
#include "rppg_qnn/build_identity.hpp"
#include "test_support.hpp"

#include <string>

void build_identity_is_explicit_and_does_not_claim_runtime_features() {
  const rppg_qnn::BuildIdentity identity = rppg_qnn::build_identity();
  EXPECT_TRUE(!identity.platform.empty());
  EXPECT_TRUE(!identity.abi.empty());
  EXPECT_EQ(identity.camera_backend, "not_compiled");
  EXPECT_EQ(identity.deep_backend, "disabled");
  EXPECT_TRUE(!identity.qnn_ready);
}

void build_identity_text_is_stable_and_human_readable() {
  const std::string text = rppg_qnn::build_identity_text();
  EXPECT_TRUE(text.find("platform=") != std::string::npos);
  EXPECT_TRUE(text.find("abi=") != std::string::npos);
  EXPECT_TRUE(text.find("camera=not_compiled") != std::string::npos);
  EXPECT_TRUE(text.find("deep=disabled") != std::string::npos);
  EXPECT_TRUE(text.find("qnn_ready=false") != std::string::npos);
}

int main() {
  build_identity_is_explicit_and_does_not_claim_runtime_features();
  build_identity_text_is_stable_and_human_readable();
  return test_support::finish();
}
```

Register `test_build_identity` in `CMakeLists.txt` before implementation.

- [ ] **Step 2: Run the focused build and verify RED**

Run:

```bash
cmake -S . -B build-android-foundation-red -DCMAKE_BUILD_TYPE=Debug
cmake --build build-android-foundation-red --target test_build_identity --parallel
```

Expected: compilation fails because `rppg_qnn/build_identity.hpp` does not exist.

- [ ] **Step 3: Implement the platform-neutral identity**

Create `include/rppg_qnn/build_identity.hpp`:

```cpp
#pragma once

#include <string>

namespace rppg_qnn {

struct BuildIdentity {
  std::string platform;
  std::string abi;
  std::string camera_backend;
  std::string deep_backend;
  bool qnn_ready{false};
};

[[nodiscard]] BuildIdentity build_identity();
[[nodiscard]] std::string build_identity_text();

}  // namespace rppg_qnn
```

Create `src/build_identity.cpp` with compile-time platform and ABI selection. Android must report `platform=android`; host builds report `macos` or `linux`; unknown targets report `unknown`. ABI values are `arm64-v8a`, `aarch64`, `x86_64`, or `unknown`. This first slice always reports `camera_backend=not_compiled`, `deep_backend=disabled`, and `qnn_ready=false`.

Format text exactly as:

```text
platform=<value>;abi=<value>;camera=not_compiled;deep=disabled;qnn_ready=false
```

Add `src/build_identity.cpp` to `rppg_qnn_core`.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
cmake --build build-android-foundation-red --target test_build_identity --parallel
ctest --test-dir build-android-foundation-red -R '^build_identity$' --output-on-failure
```

Expected: `build_identity` passes.

- [ ] **Step 5: Run the complete native suite**

Run:

```bash
ctest --test-dir build-android-foundation-red --output-on-failure
```

Expected: all tests pass.

- [ ] **Step 6: Commit the identity contract**

```bash
git add CMakeLists.txt include/rppg_qnn/build_identity.hpp src/build_identity.cpp tests/test_build_identity.cpp
git commit -m "feat: add portable runtime build identity"
```

## Task 3: Android packaging contract before scaffold

**Files:**
- Create: `tests/test_android_packaging.sh`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write a packaging test that fails while Android files are absent**

Create an executable `tests/test_android_packaging.sh` that requires these exact properties:

```bash
#!/usr/bin/env bash
set -euo pipefail

root=${1:?repository root is required}
manifest="$root/android/app/src/main/AndroidManifest.xml"
app_gradle="$root/android/app/build.gradle"
bridge="$root/android/app/src/main/cpp/native_bridge.cpp"

test -f "$manifest"
test -f "$app_gradle"
test -f "$bridge"
grep -Fq "arm64-v8a" "$app_gradle"
grep -Fq "28.2.13676358" "$app_gradle"
grep -Fq "android.permission.CAMERA" "$manifest" && {
  echo "foundation APK must not request camera permission" >&2
  exit 1
}
grep -Fiq "fake_deep" "$bridge" && {
  echo "foundation JNI must not expose fake deep" >&2
  exit 1
}
if find "$root/android" -type f \( -name '*.pth' -o -name '*.pt' -o -name '*.onnx' \
     -o -name '*.dlc' -o -name '*.bin' \) -print -quit | grep -q .; then
  echo "Android source tree must not package model artifacts" >&2
  exit 1
fi
```

Register it in CTest:

```cmake
add_test(NAME android_packaging
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_android_packaging.sh
                      ${CMAKE_CURRENT_SOURCE_DIR})
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
chmod +x tests/test_android_packaging.sh
ctest --test-dir build-android-foundation-red -R '^android_packaging$' --output-on-failure
```

Expected: failure because `android/app/src/main/AndroidManifest.xml` is absent.

- [ ] **Step 3: Leave the packaging test uncommitted while it is RED**

Do not create a commit with a failing repository suite. Task 4 commits the test
and Android scaffold together after the contract turns green.

## Task 4: Minimal Android application and JNI smoke bridge

**Files:**
- Create: `android/settings.gradle`
- Create: `android/build.gradle`
- Create: `android/gradle.properties`
- Create: `android/app/build.gradle`
- Create: `android/app/proguard-rules.pro`
- Create: `android/app/src/main/AndroidManifest.xml`
- Create: `android/app/src/main/java/com/jagger/rppgbench/MainActivity.java`
- Create: `android/app/src/main/java/com/jagger/rppgbench/NativeBridge.java`
- Create: `android/app/src/main/cpp/CMakeLists.txt`
- Create: `android/app/src/main/cpp/native_bridge.cpp`
- Create: `android/app/src/main/res/values/strings.xml`
- Modify: `.gitignore`

- [ ] **Step 1: Add the pinned Gradle application configuration**

Use AGP `9.0.1`, Gradle `9.1.0`, compile/target SDK `36`, minimum SDK `26`, NDK `28.2.13676358`, and only `arm64-v8a`. `android/app/build.gradle` must use `externalNativeBuild.cmake` with C++17 and point to `src/main/cpp/CMakeLists.txt`.

The application ID and namespace are both:

```text
com.jagger.rppgbench
```

Do not add AndroidX, Compose, Kotlin, network, storage, camera, or foreground-service dependencies in this slice.

`android/settings.gradle`:

```groovy
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "rPPGBench"
include ":app"
```

`android/build.gradle`:

```groovy
plugins {
    id "com.android.application" version "9.0.1" apply false
}
```

`android/gradle.properties`:

```properties
org.gradle.jvmargs=-Xmx2048m -Dfile.encoding=UTF-8
org.gradle.daemon=false
android.useAndroidX=false
```

`android/app/build.gradle`:

```groovy
plugins {
    id "com.android.application"
}

android {
    namespace = "com.jagger.rppgbench"
    compileSdk = 36
    ndkVersion = "28.2.13676358"

    defaultConfig {
        applicationId = "com.jagger.rppgbench"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0-foundation"

        ndk {
            abiFilters += ["arm64-v8a"]
        }

        externalNativeBuild {
            cmake {
                cppFlags += ["-std=c++17"]
            }
        }
    }

    buildTypes {
        debug {
            debuggable = true
        }
        release {
            minifyEnabled = false
            proguardFiles getDefaultProguardFile("proguard-android-optimize.txt"),
                    "proguard-rules.pro"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        buildConfig = false
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}
```

`android/app/proguard-rules.pro` contains only a comment explaining that this
foundation has no custom keep rules.

- [ ] **Step 2: Add the launcher Activity and native bridge**

`NativeBridge.java`:

```java
package com.jagger.rppgbench;

public final class NativeBridge {
    static {
        System.loadLibrary("rppg_qnn_android");
    }

    private NativeBridge() {}

    public static native String nativeBuildIdentity();
}
```

`MainActivity.java` creates a vertical `LinearLayout`, a title, a status `TextView`, and a refresh `Button` using platform widgets only. On create and refresh, it sets the status to `NativeBridge.nativeBuildIdentity()`. If loading fails, it displays `NATIVE_LOAD_FAILED: <exception class>` and does not claim readiness.

Use this complete implementation:

```java
package com.jagger.rppgbench;

import android.app.Activity;
import android.os.Bundle;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

public final class MainActivity extends Activity {
    private TextView status;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        int padding = (int) (24 * getResources().getDisplayMetrics().density);
        root.setPadding(padding, padding, padding, padding);

        TextView title = new TextView(this);
        title.setText(R.string.app_title);
        title.setTextSize(24);
        root.addView(title, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));

        status = new TextView(this);
        root.addView(status, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));

        Button refresh = new Button(this);
        refresh.setText(R.string.refresh_native_status);
        refresh.setOnClickListener(view -> refreshStatus());
        root.addView(refresh, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));

        setContentView(root);
        refreshStatus();
    }

    private void refreshStatus() {
        try {
            status.setText(NativeBridge.nativeBuildIdentity());
        } catch (Throwable failure) {
            status.setText("NATIVE_LOAD_FAILED: " +
                    failure.getClass().getSimpleName());
        }
    }
}
```

- [ ] **Step 3: Add the exception-safe JNI adapter**

`android/app/src/main/cpp/native_bridge.cpp`:

```cpp
#include "rppg_qnn/build_identity.hpp"

#include <jni.h>

#include <exception>
#include <string>

extern "C" JNIEXPORT jstring JNICALL
Java_com_jagger_rppgbench_NativeBridge_nativeBuildIdentity(JNIEnv* env,
                                                            jclass) noexcept {
  try {
    const std::string identity = rppg_qnn::build_identity_text();
    return env->NewStringUTF(identity.c_str());
  } catch (const std::exception& error) {
    const std::string message = std::string("NATIVE_ERROR: ") + error.what();
    return env->NewStringUTF(message.c_str());
  } catch (...) {
    return env->NewStringUTF("NATIVE_ERROR: unknown");
  }
}
```

The Android CMake file imports only `src/build_identity.cpp`; it does not yet import OpenCV, Camera2, QNN, the full pipeline, or fake deep:

```cmake
cmake_minimum_required(VERSION 3.22.1)
project(rppg_qnn_android LANGUAGES CXX)

add_library(rppg_qnn_android SHARED
            native_bridge.cpp
            ../../../../../src/build_identity.cpp)
target_include_directories(rppg_qnn_android PRIVATE
                           ../../../../../include)
target_compile_features(rppg_qnn_android PRIVATE cxx_std_17)
target_compile_options(rppg_qnn_android PRIVATE
                       -Wall -Wextra -Wpedantic -Werror)
```

- [ ] **Step 4: Complete the manifest and resources**

The manifest declares only the launcher Activity, sets `android:exported="true"`, and contains no permissions or vendor libraries. Strings state that camera and QNN are not compiled in this foundation build.

`android/app/src/main/AndroidManifest.xml`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <application
        android:allowBackup="false"
        android:label="@string/app_name"
        android:supportsRtl="true">
        <activity
            android:name=".MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>
```

`android/app/src/main/res/values/strings.xml`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<resources>
    <string name="app_name">rPPG Bench</string>
    <string name="app_title">rPPG Android NDK Foundation</string>
    <string name="refresh_native_status">Refresh native status</string>
</resources>
```

- [ ] **Step 5: Update ignored local/build state**

Append:

```gitignore
android/.gradle/
android/local.properties
android/**/build/
android/.idea/
*.apk
*.aab
```

- [ ] **Step 6: Run the packaging contract and verify GREEN**

Run:

```bash
cmake -S . -B build-android-foundation-red -DCMAKE_BUILD_TYPE=Debug
ctest --test-dir build-android-foundation-red -R '^android_packaging$' --output-on-failure
```

Expected: `android_packaging` passes, and no model binary exists under `android/`.

- [ ] **Step 7: Commit the Android scaffold**

```bash
git add .gitignore CMakeLists.txt tests/test_android_packaging.sh android
git commit -m "feat: add Android NDK smoke application"
```

## Task 5: Android environment and build script

**Files:**
- Create: `scripts/build_android.sh`
- Create: `tests/test_android_build_script.sh`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write fail-fast script tests**

Create `tests/test_android_build_script.sh` that runs `scripts/build_android.sh` in three isolated failure cases and checks exact diagnostics:

```text
JAVA_HOME must point to JDK 17
ANDROID_SDK_ROOT must point to an Android SDK
Android NDK 28.2.13676358 is required
```

The test supplies temporary fake directories and stub executables so each failure reaches the intended gate without reading the developer's real machine state.

Use this complete test:

```bash
#!/usr/bin/env bash
set -euo pipefail

root=${1:?repository root is required}
script="$root/scripts/build_android.sh"
temp=$(mktemp -d)
trap 'rm -rf "$temp"' EXIT

expect_failure() {
  local expected=$1
  shift
  local output
  if output=$(env -i PATH=/usr/bin:/bin "$@" "$script" 2>&1); then
    echo "expected build script failure: $expected" >&2
    exit 1
  fi
  grep -Fq "$expected" <<<"$output" || {
    echo "missing diagnostic: $expected" >&2
    printf '%s\n' "$output" >&2
    exit 1
  }
}

expect_failure "JAVA_HOME must point to JDK 17"

mkdir -p "$temp/jdk/bin"
cat >"$temp/jdk/bin/java" <<'JAVA'
#!/usr/bin/env bash
echo 'openjdk version "17.0.12"' >&2
JAVA
chmod +x "$temp/jdk/bin/java"
expect_failure "ANDROID_SDK_ROOT must point to an Android SDK" \
  JAVA_HOME="$temp/jdk"

mkdir -p "$temp/sdk"
expect_failure "Android NDK 28.2.13676358 is required" \
  JAVA_HOME="$temp/jdk" ANDROID_SDK_ROOT="$temp/sdk"
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
chmod +x tests/test_android_build_script.sh
bash tests/test_android_build_script.sh "$PWD"
```

Expected: failure because `scripts/build_android.sh` does not exist.

- [ ] **Step 3: Implement the minimal build script**

Create `scripts/build_android.sh` with these gates, in order:

1. repository root resolved physically from the script location;
2. `JAVA_HOME/bin/java` exists and `java -version` reports major version 17;
3. `ANDROID_SDK_ROOT` is an absolute directory;
4. `$ANDROID_SDK_ROOT/ndk/28.2.13676358/build/cmake/android.toolchain.cmake` exists;
5. `android/gradlew` exists and is executable;
6. invoke `android/gradlew --no-daemon --project-dir android :app:assembleDebug`;
7. require `android/app/build/outputs/apk/debug/app-debug.apk` and print its absolute path.

The script must not download an SDK, JDK, NDK, Gradle, QNN, OpenCV, or model. It must not use `sudo`.

Use this complete script:

```bash
#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'build_android.sh: %s\n' "$1" >&2
  exit 2
}

script_dir=$(CDPATH= cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=$(CDPATH= cd -P -- "$script_dir/.." && pwd -P)

[[ -n ${JAVA_HOME:-} && -x "$JAVA_HOME/bin/java" ]] ||
  die "JAVA_HOME must point to JDK 17"
java_version=$($JAVA_HOME/bin/java -version 2>&1 | head -n 1)
[[ $java_version == *'version "17.'* ]] ||
  die "JAVA_HOME must point to JDK 17"

[[ -n ${ANDROID_SDK_ROOT:-} && ${ANDROID_SDK_ROOT} == /* &&
   -d ${ANDROID_SDK_ROOT} ]] ||
  die "ANDROID_SDK_ROOT must point to an Android SDK"

ndk_version=28.2.13676358
toolchain="$ANDROID_SDK_ROOT/ndk/$ndk_version/build/cmake/android.toolchain.cmake"
[[ -f $toolchain ]] || die "Android NDK $ndk_version is required"

gradlew="$root/android/gradlew"
[[ -x $gradlew ]] || die "android/gradlew is missing or not executable"

"$gradlew" --no-daemon --project-dir "$root/android" :app:assembleDebug

apk="$root/android/app/build/outputs/apk/debug/app-debug.apk"
[[ -f $apk ]] || die "debug APK was not produced"
printf '%s\n' "$apk"
```

- [ ] **Step 4: Generate and pin the Gradle wrapper on a provisioned machine**

With JDK 17 and Gradle 9.1.0 available:

```bash
cd android
gradle wrapper --gradle-version 9.1.0 --distribution-type bin
cd ..
```

Commit `android/gradlew`, `android/gradlew.bat`,
`android/gradle/wrapper/gradle-wrapper.jar`, and
`android/gradle/wrapper/gradle-wrapper.properties`. The distribution URL must be
`https\://services.gradle.org/distributions/gradle-9.1.0-bin.zip`.

- [ ] **Step 5: Run the script tests and verify GREEN**

Run:

```bash
bash tests/test_android_build_script.sh "$PWD"
```

Expected: all three negative environment cases pass.

- [ ] **Step 6: Register and run the host test suite**

Register the shell test as `android_build_script` in CTest, then run:

```bash
cmake -S . -B build-android-foundation-red -DCMAKE_BUILD_TYPE=Debug
cmake --build build-android-foundation-red --parallel
ctest --test-dir build-android-foundation-red --output-on-failure
```

Expected: the complete host suite passes without requiring an Android SDK.

- [ ] **Step 7: Build the debug APK when prerequisites are present**

Run:

```bash
JAVA_HOME=/absolute/path/to/jdk17 \
ANDROID_SDK_ROOT=/absolute/path/to/android-sdk \
./scripts/build_android.sh
```

Expected: exit zero and an absolute path ending in
`android/app/build/outputs/apk/debug/app-debug.apk`. If this Mac remains
unprovisioned, record the exact first failed gate and do not claim an APK build.

- [ ] **Step 8: Commit build tooling**

```bash
git add CMakeLists.txt scripts/build_android.sh tests/test_android_build_script.sh \
  android/gradlew android/gradlew.bat android/gradle/wrapper
git commit -m "build: add pinned Android foundation workflow"
```

## Task 6: Documentation and final foundation verification

**Files:**
- Modify: `README.md`
- Create: `ANDROID_NEXT_STEPS.md`

- [ ] **Step 1: Document the exact foundation boundary**

Update `README.md` to state:

- Android is now the primary target direction;
- the current Android screen proves only APK/JNI/native identity;
- camera, traditional live output, QNN conversion, and Adreno inference are not part of this foundation slice;
- the Linux/V4L2 reference remains available;
- fake deep is never an Android result.

Include links to the design and this implementation plan.

- [ ] **Step 2: Add a company-machine handoff checklist**

Create `ANDROID_NEXT_STEPS.md` with exact commands to capture:

```bash
adb shell getprop ro.build.version.release
adb shell getprop ro.build.version.sdk
adb shell getprop ro.product.cpu.abi
adb shell getprop ro.hardware
adb shell pm list features | grep -i camera
adb shell dumpsys media.camera > media-camera.txt
adb shell ls -l /vendor/lib64/libQnnGpu.so /vendor/lib64/libQnnSystem.so
adb shell readelf -h /vendor/lib64/libQnnGpu.so
```

Also require the company-provided Android NDK revision, QAIRT SDK version/root,
Android QNN sample name, and whether the application is normal, privileged, or
vendor-installed. Explain that library paths may differ and a missing path is
evidence to locate the vendor package, not permission to copy a Linux QNN
library.

- [ ] **Step 3: Run repository hygiene checks**

Run:

```bash
git diff --check
if git ls-files | grep -E '\.(pth|pt|onnx|dlc|bin|apk|aab)$'; then
  echo "forbidden generated/model artifact is tracked" >&2
  exit 1
fi
```

Expected: no whitespace errors and no forbidden artifacts tracked. The single
Gradle wrapper JAR is allowed and is not matched by this model/artifact rule.

- [ ] **Step 4: Run fresh native verification**

Run:

```bash
BUILD_DIR=build-android-foundation-release \
STAGE_DIR=stage/rppg-qnn-android-foundation-host \
./scripts/build_linux.sh native
```

Expected: Release build succeeds, every CTest passes, and the existing Linux
four-file stage whitelist remains unchanged.

- [ ] **Step 5: Run Android package verification if the SDK gate is available**

Run:

```bash
./scripts/build_android.sh
unzip -l android/app/build/outputs/apk/debug/app-debug.apk | \
  grep 'lib/arm64-v8a/librppg_qnn_android.so'
```

Expected: the APK contains the JNI library only under `arm64-v8a`. If the local
Android prerequisites are absent, preserve the build-script diagnostic as the
open external gate and report host verification separately.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md ANDROID_NEXT_STEPS.md
git commit -m "docs: hand off Android bench foundation"
```

## Follow-on plan gates

Do not start Camera2 implementation until `ANDROID_NEXT_STEPS.md` records the
bench Android API and confirms that the intended camera is visible through
Camera2. Do not start QNN C++ API implementation until the exact Android QAIRT
SDK, headers, sample, target libraries, and linker visibility are captured.

The next plans, in order, are:

1. Android Camera2/AImageReader frame source and YUV conversion.
2. Android OpenCV ROI and traditional live pipeline.
3. QAIRT converter evidence and QNN artifact contract.
4. Real Android QNN EfficientPhys runtime and concurrent acceptance.
