#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "android packaging check: expected repository root argument" >&2
  echo "usage: $0 <repository-root>" >&2
  exit 2
fi

root=$1
manifest="$root/android/app/src/main/AndroidManifest.xml"
app_gradle="$root/android/app/build.gradle"
native_bridge="$root/android/app/src/main/cpp/native_bridge.cpp"
gitignore="$root/.gitignore"

for required_file in "$manifest" "$app_gradle" "$native_bridge" "$gitignore"; do
  if [[ ! -f "$required_file" ]]; then
    echo "android packaging check: required file is missing: $required_file" >&2
    exit 1
  fi
done

if ! grep -Fxq "android/**/.cxx/" "$gitignore"; then
  echo "android packaging check: $gitignore must contain exact rule android/**/.cxx/" >&2
  exit 1
fi

if ! grep -Fxq "android/**/.externalNativeBuild/" "$gitignore"; then
  echo "android packaging check: $gitignore must contain exact rule android/**/.externalNativeBuild/" >&2
  exit 1
fi

if ! grep -Fq "arm64-v8a" "$app_gradle"; then
  echo "android packaging check: $app_gradle must restrict ABI packaging to arm64-v8a" >&2
  exit 1
fi

if ! grep -Fq "28.2.13676358" "$app_gradle"; then
  echo "android packaging check: $app_gradle must pin NDK version 28.2.13676358" >&2
  exit 1
fi

if grep -Fq "android.permission.CAMERA" "$manifest"; then
  echo "android packaging check: foundation manifest must not request android.permission.CAMERA" >&2
  exit 1
fi

if grep -Eqi "fake_deep" "$native_bridge"; then
  echo "android packaging check: $native_bridge must not contain fake_deep placeholders" >&2
  exit 1
fi

model_artifact=$(find "$root/android" -type f \( \
  -iname '*.pth' -o \
  -iname '*.pt' -o \
  -iname '*.onnx' -o \
  -iname '*.dlc' -o \
  -iname '*.bin' \
\) -print -quit)
if [[ -n "$model_artifact" ]]; then
  echo "android packaging check: model artifact is forbidden under android/: $model_artifact" >&2
  exit 1
fi
