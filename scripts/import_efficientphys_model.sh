#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'import_efficientphys_model.sh: %s\n' "$*" >&2
  exit 2
}

MODEL=${1:-}
EXPECTED=c1b321042db1335da70b0295cc84f653a2cfe90f75cff738b3045ea3c103257d
PACKAGE=com.jagger.rppgbench

[[ -n "$MODEL" && -f "$MODEL" ]] || die "usage: $0 /absolute/path/to/efficientphys_pure.onnx"

actual=$(shasum -a 256 "$MODEL" | awk '{print $1}')
[[ "$actual" == "$EXPECTED" ]] || die "checksum mismatch: $actual"

command -v adb >/dev/null || die "adb is required"
adb get-state >/dev/null || die "no adb device"

adb push "$MODEL" /data/local/tmp/efficientphys_pure.onnx
adb shell run-as "$PACKAGE" mkdir -p files/models
adb shell run-as "$PACKAGE" cp /data/local/tmp/efficientphys_pure.onnx \
  files/models/efficientphys_pure.onnx
adb shell run-as "$PACKAGE" sha256sum files/models/efficientphys_pure.onnx
adb shell rm /data/local/tmp/efficientphys_pure.onnx

printf 'imported into %s files/models/efficientphys_pure.onnx\n' "$PACKAGE"
