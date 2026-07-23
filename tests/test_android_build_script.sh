#!/usr/bin/env bash

set -euo pipefail

root=${1:?repository root is required}
source_build_script="$root/scripts/build_android.sh"
workspace=$(mktemp -d "${TMPDIR:-/tmp}/rppg-android-build-test.XXXXXX")
trap 'rm -rf "$workspace"' EXIT

isolated_repo="$workspace/isolated repo"
mkdir -p "$isolated_repo/scripts" "$isolated_repo/android"
isolated_repo=$(CDPATH= cd -- "$isolated_repo" && pwd -P)
cp "$source_build_script" "$isolated_repo/scripts/build_android.sh"
build_script="$isolated_repo/scripts/build_android.sh"

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

make_fake_java_home() {
  local name=$1
  local version_line=$2
  local java_home="$workspace/$name"

  mkdir -p "$java_home/bin"
  printf '%s\n' \
    '#!/bin/sh' \
    "printf '%s\\n' '$version_line'" \
    >"$java_home/bin/java"
  chmod +x "$java_home/bin/java"
  printf '%s\n' "$java_home"
}

assert_file_equals() {
  local expected=$1
  local file=$2
  local actual
  actual=$(<"$file")
  if [[ "$actual" != "$expected" ]]; then
    printf 'unexpected file contents for %s:\nexpected:\n%s\nactual:\n%s\n' \
      "$file" "$expected" "$actual" >&2
    exit 1
  fi
}

expect_failure \
  'build_android.sh: JAVA_HOME must point to JDK 17' \
  env -i PATH=/usr/bin:/bin "$build_script"

java_17_ga_home=$(make_fake_java_home \
  jdk-17-ga 'openjdk version "17" 2021-09-14')
java_17_patch_home=$(make_fake_java_home \
  jdk-17-patch 'openjdk version "17.0.12" 2024-07-16')
java_21_home=$(make_fake_java_home \
  jdk-21 'openjdk version "21.0.2" 2024-01-16')

expect_failure \
  'build_android.sh: ANDROID_SDK_ROOT must point to an Android SDK' \
  env -i PATH=/usr/bin:/bin JAVA_HOME="$java_17_ga_home" "$build_script"

expect_failure \
  'build_android.sh: ANDROID_SDK_ROOT must point to an Android SDK' \
  env -i PATH=/usr/bin:/bin JAVA_HOME="$java_17_patch_home" "$build_script"

expect_failure \
  'build_android.sh: JAVA_HOME must point to JDK 17' \
  env -i PATH=/usr/bin:/bin JAVA_HOME="$java_21_home" "$build_script"

fake_sdk="$workspace/android-sdk"
mkdir -p "$fake_sdk"

expect_failure \
  'build_android.sh: Android NDK 28.2.13676358 is required' \
  env -i PATH=/usr/bin:/bin JAVA_HOME="$java_17_patch_home" \
    ANDROID_SDK_ROOT="$fake_sdk" "$build_script"

toolchain="$fake_sdk/ndk/28.2.13676358/build/cmake/android.toolchain.cmake"
mkdir -p "$(dirname "$toolchain")"
: >"$toolchain"

expect_failure \
  'build_android.sh: android/gradlew is missing or not executable' \
  env -i PATH=/usr/bin:/bin JAVA_HOME="$java_17_patch_home" \
    ANDROID_SDK_ROOT="$fake_sdk" "$build_script"

gradle_args="$workspace/gradle-args"
cat >"$isolated_repo/android/gradlew" <<'GRADLEW'
#!/bin/sh
printf '%s\n' "$@" >"$STUB_GRADLE_ARGS"
if [ "${STUB_CREATE_APK:-0}" = 1 ]; then
  mkdir -p "$(dirname "$STUB_APK")"
  : >"$STUB_APK"
fi
GRADLEW
chmod +x "$isolated_repo/android/gradlew"

isolated_apk="$isolated_repo/android/app/build/outputs/apk/debug/app-debug.apk"
expected_gradle_args=$(printf '%s\n' \
  '--no-daemon' \
  '--project-dir' \
  "$isolated_repo/android" \
  ':app:assembleDebug')

expect_failure \
  'build_android.sh: Android debug APK was not produced' \
  env -i PATH=/usr/bin:/bin JAVA_HOME="$java_17_patch_home" \
    ANDROID_SDK_ROOT="$fake_sdk" STUB_GRADLE_ARGS="$gradle_args" \
    STUB_APK="$isolated_apk" "$isolated_repo/scripts/build_android.sh"
assert_file_equals "$expected_gradle_args" "$gradle_args"

success_output="$workspace/success-output"
env -i PATH=/usr/bin:/bin JAVA_HOME="$java_17_patch_home" \
  ANDROID_SDK_ROOT="$fake_sdk" STUB_GRADLE_ARGS="$gradle_args" \
  STUB_CREATE_APK=1 STUB_APK="$isolated_apk" \
  "$isolated_repo/scripts/build_android.sh" >"$success_output"
assert_file_equals "$expected_gradle_args" "$gradle_args"
assert_file_equals "$isolated_apk" "$success_output"
[[ "$isolated_apk" == /* ]] || {
  echo "build script success must print an absolute APK path" >&2
  exit 1
}
