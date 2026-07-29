#!/usr/bin/env bash
set -euo pipefail

PACKAGE=com.jagger.rppgbench
APK_SHA256=9d60f36990894c8726657437c72af5c24a3c2cfcd53b42e5b991438e29dc625e
EFFICIENTPHYS_SHA256=c1b321042db1335da70b0295cc84f653a2cfe90f75cff738b3045ea3c103257d
TSCAN_SHA256=342a3c8033dda9ab154e85d5a4e2a876a6461648b7fcb27c46a7023e662bcc64

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APK="$SCRIPT_DIR/app-debug.apk"
EFFICIENTPHYS="$SCRIPT_DIR/efficientphys_pure.onnx"
TSCAN="$SCRIPT_DIR/ubfc_tscan_full_lr3e-5_Epoch10.onnx"
REMOTE_EFFICIENTPHYS=/data/local/tmp/rppg-efficientphys.onnx
REMOTE_TSCAN=/data/local/tmp/rppg-tscan.onnx

die() {
  printf '安装失败: %s\n' "$*" >&2
  exit 2
}

verify_sha256() {
  local path=$1
  local expected=$2
  [[ -f "$path" ]] || die "缺少文件: $path"
  local actual
  actual=$(shasum -a 256 "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] ||
    die "SHA-256 不匹配: $(basename "$path") ($actual)"
}

cleanup() {
  adb shell rm -f "$REMOTE_EFFICIENTPHYS" "$REMOTE_TSCAN" >/dev/null 2>&1 || true
}

command -v adb >/dev/null 2>&1 ||
  die "未找到 adb。请先安装 Android Platform Tools 并加入 PATH。"
verify_sha256 "$APK" "$APK_SHA256"
verify_sha256 "$EFFICIENTPHYS" "$EFFICIENTPHYS_SHA256"
verify_sha256 "$TSCAN" "$TSCAN_SHA256"

adb start-server >/dev/null
DEVICE_COUNT=$(adb devices | awk '$2 == "device" { count++ } END { print count + 0 }')
[[ "$DEVICE_COUNT" -eq 1 ]] ||
  die "必须且只能连接一台已授权设备，当前检测到 $DEVICE_COUNT 台。"

trap cleanup EXIT
adb install -r "$APK"
adb push "$EFFICIENTPHYS" "$REMOTE_EFFICIENTPHYS"
adb push "$TSCAN" "$REMOTE_TSCAN"
adb shell run-as "$PACKAGE" mkdir -p files/models
adb shell run-as "$PACKAGE" cp "$REMOTE_EFFICIENTPHYS" files/models/efficientphys_pure.onnx
adb shell run-as "$PACKAGE" cp "$REMOTE_TSCAN" files/models/ubfc_tscan_full_lr3e-5_Epoch10.onnx

DEVICE_HASHES=$(adb shell run-as "$PACKAGE" sha256sum \
  files/models/efficientphys_pure.onnx \
  files/models/ubfc_tscan_full_lr3e-5_Epoch10.onnx)
printf '%s\n' "$DEVICE_HASHES"
printf '%s\n' "$DEVICE_HASHES" | grep -q "$EFFICIENTPHYS_SHA256" ||
  die "手机中的 EfficientPhys 权重校验失败。"
printf '%s\n' "$DEVICE_HASHES" | grep -q "$TSCAN_SHA256" ||
  die "手机中的 TSCAN 权重校验失败。"

adb shell am start -n "$PACKAGE/.MainActivity"
printf '安装完成：APK 与两份模型均已校验并启动。\n'
