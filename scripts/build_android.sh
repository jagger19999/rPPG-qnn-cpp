#!/usr/bin/env bash

set -euo pipefail

die() {
  printf 'build_android.sh: %s\n' "$*" >&2
  exit 2
}

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=$(CDPATH= cd -- "$script_dir/.." && pwd -P)

if [[ -z ${JAVA_HOME:-} || ! -x "$JAVA_HOME/bin/java" ]]; then
  die 'JAVA_HOME must point to JDK 17'
fi

if ! java_version=$("$JAVA_HOME/bin/java" -version 2>&1); then
  die 'JAVA_HOME must point to JDK 17'
fi
java_version=${java_version%%$'\n'*}
if [[ "$java_version" =~ version[[:space:]]+\"([^\"]+)\" ]]; then
  java_version=${BASH_REMATCH[1]}
else
  die 'JAVA_HOME must point to JDK 17'
fi
java_major=${java_version%%.*}
if [[ "$java_major" != 17 ]]; then
  die 'JAVA_HOME must point to JDK 17'
fi

if [[ -z ${ANDROID_SDK_ROOT:-} || "$ANDROID_SDK_ROOT" != /* || ! -d "$ANDROID_SDK_ROOT" ]]; then
  die 'ANDROID_SDK_ROOT must point to an Android SDK'
fi

ndk_toolchain="$ANDROID_SDK_ROOT/ndk/28.2.13676358/build/cmake/android.toolchain.cmake"
if [[ ! -f "$ndk_toolchain" ]]; then
  die 'Android NDK 28.2.13676358 is required'
fi

if [[ -z ${RPPG_OPENCV_ANDROID_SDK:-} ||
      "$RPPG_OPENCV_ANDROID_SDK" != /* ||
      ! -f "$RPPG_OPENCV_ANDROID_SDK/sdk/native/jni/OpenCVConfig.cmake" ]]; then
  die 'RPPG_OPENCV_ANDROID_SDK must point to OpenCV Android SDK 4.13.0'
fi

ort_aar_name=onnxruntime-android-1.27.0.aar
ort_aar_sha256=077dec5e2d821234c7dc0aba584bec8f999854b546c754cab93a90741c56fbeb
if [[ -z ${RPPG_ONNXRUNTIME_ANDROID:-} ||
      "$RPPG_ONNXRUNTIME_ANDROID" != /* ||
      ! -f "$RPPG_ONNXRUNTIME_ANDROID/.aar.sha256" ||
      ! -f "$RPPG_ONNXRUNTIME_ANDROID/headers/onnxruntime_cxx_api.h" ||
      ! -f "$RPPG_ONNXRUNTIME_ANDROID/jni/arm64-v8a/libonnxruntime.so" ]]; then
  die 'RPPG_ONNXRUNTIME_ANDROID must point to ONNX Runtime Android 1.27.0'
fi
if [[ "$(<"$RPPG_ONNXRUNTIME_ANDROID/.aar.sha256")" != "$ort_aar_sha256" ]]; then
  die 'ONNX Runtime Android 1.27.0 checksum mismatch'
fi
if [[ -f "$RPPG_ONNXRUNTIME_ANDROID/$ort_aar_name" ]]; then
  actual_ort_sha256=$(
    shasum -a 256 "$RPPG_ONNXRUNTIME_ANDROID/$ort_aar_name" | awk '{print $1}'
  )
  if [[ "$actual_ort_sha256" != "$ort_aar_sha256" ]]; then
    die 'ONNX Runtime Android 1.27.0 checksum mismatch'
  fi
fi

gradlew="$root/android/gradlew"
if [[ ! -x "$gradlew" ]]; then
  die 'android/gradlew is missing or not executable'
fi

"$gradlew" --no-daemon --project-dir "$root/android" :app:assembleDebug

apk="$root/android/app/build/outputs/apk/debug/app-debug.apk"
if [[ ! -f "$apk" ]]; then
  die 'Android debug APK was not produced'
fi

printf '%s\n' "$apk"
