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
if [[ "$java_version" != *'version "17.'* ]]; then
  die 'JAVA_HOME must point to JDK 17'
fi

if [[ -z ${ANDROID_SDK_ROOT:-} || "$ANDROID_SDK_ROOT" != /* || ! -d "$ANDROID_SDK_ROOT" ]]; then
  die 'ANDROID_SDK_ROOT must point to an Android SDK'
fi

ndk_toolchain="$ANDROID_SDK_ROOT/ndk/28.2.13676358/build/cmake/android.toolchain.cmake"
if [[ ! -f "$ndk_toolchain" ]]; then
  die 'Android NDK 28.2.13676358 is required'
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
