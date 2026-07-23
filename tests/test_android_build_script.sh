#!/usr/bin/env bash

set -euo pipefail

root=${1:?repository root is required}
build_script="$root/scripts/build_android.sh"
workspace=$(mktemp -d "${TMPDIR:-/tmp}/rppg-android-build-test.XXXXXX")
trap 'rm -rf "$workspace"' EXIT

expect_failure() {
  local expected=$1
  shift

  local output_file="$workspace/output"
  if "$@" >"$output_file" 2>&1; then
    printf 'expected command to fail, but it succeeded: %s\n' "$*" >&2
    exit 1
  fi

  local actual
  actual=$(<"$output_file")
  if [[ "$actual" != "$expected" ]]; then
    printf 'missing expected diagnostic:\n  %s\nactual output:\n  %s\n' \
      "$expected" "$actual" >&2
    exit 1
  fi
}

expect_failure \
  'build_android.sh: JAVA_HOME must point to JDK 17' \
  env -i PATH=/usr/bin:/bin "$build_script"

fake_java_home="$workspace/jdk-17"
mkdir -p "$fake_java_home/bin"
printf '%s\n' \
  '#!/bin/sh' \
  'printf '\''openjdk version "17.0.12"\n'\''' \
  >"$fake_java_home/bin/java"
chmod +x "$fake_java_home/bin/java"

expect_failure \
  'build_android.sh: ANDROID_SDK_ROOT must point to an Android SDK' \
  env -i PATH=/usr/bin:/bin JAVA_HOME="$fake_java_home" "$build_script"

fake_sdk="$workspace/android-sdk"
mkdir -p "$fake_sdk"

expect_failure \
  'build_android.sh: Android NDK 28.2.13676358 is required' \
  env -i PATH=/usr/bin:/bin JAVA_HOME="$fake_java_home" \
    ANDROID_SDK_ROOT="$fake_sdk" "$build_script"

toolchain="$fake_sdk/ndk/28.2.13676358/build/cmake/android.toolchain.cmake"
mkdir -p "$(dirname "$toolchain")"
: >"$toolchain"

expect_failure \
  'build_android.sh: android/gradlew is missing or not executable' \
  env -i PATH=/usr/bin:/bin JAVA_HOME="$fake_java_home" \
    ANDROID_SDK_ROOT="$fake_sdk" "$build_script"
