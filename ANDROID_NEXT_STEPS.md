# Android APK/NDK 公司机交接

本文档是 Android `arm64-v8a` 基础之后的现场交接清单。所有心率、波形和时延输出只能用于研究与工程验证，不得用于医疗诊断、治疗或其他高风险决策。

## 1. 冻结当前边界

在公司机 checkout 根目录现场记录实际 commit，不要在本文档填写伪造或过期的占位 hash：

```bash
git rev-parse HEAD
git status --short --branch
```

当前冻结边界包括：Gradle/`arm64-v8a`/NDK 构建、Java 相机权限、Camera2 NDK 枚举和采集、`AImageReader_acquireLatestImage`、`YUV_420_888` stride 校验及 BGR 转换、JNI opaque handle 生命周期、OpenCV Android 4.13.0、Haar ROI、GREEN/POS/CHROM latest-only worker、应用私有目录会话输出和状态 UI。主机已交叉构建 APK并以模拟输入验证三种传统算法和输出契约，但目标摄像头枚举、真实帧率/时间戳/颜色、ROI 稳定性和 Activity stop/start 仍需真机证据。它尚未转换/加载 QNN 模型，未证明 Adreno。

模型不进 Git。不得提交 `.pth`、`.pt`、`.onnx`、`.dlc`、`.bin` 或转换后的私有模型/context 产物；只记录可审计的版本、工具参数、hash 和外部存放位置。`--deep fake` 是 host 测试接线，永远不得作为 Android 结果。

## 2. 开发机前置和 Gradle wrapper 门禁

公司机需要精确准备：

- JDK 17，且 `JAVA_HOME` 指向该 JDK。
- Android SDK platform 36 与 Android SDK Build-Tools 36。
- Android NDK `28.2.13676358`，安装在 `$ANDROID_SDK_ROOT/ndk/28.2.13676358`。
- OpenCV Android SDK 4.13.0，且 `RPPG_OPENCV_ANDROID_SDK` 指向解压后的 `OpenCV-android-sdk`。
- ONNX Runtime Android 1.27.0，且 `RPPG_ONNXRUNTIME_ANDROID` 指向校验并解压后的 AAR；AAR SHA-256 必须为 `077dec5e2d821234c7dc0aba584bec8f999854b546c754cab93a90741c56fbeb`。

仓库已包含固定的 Gradle wrapper。执行前验证：

```bash
grep -Fx \
  'distributionSha256Sum=a17ddd85a26b6a7f5ddb71ff8b05fc5104c0202c6e64782429790c933686c806' \
  android/gradle/wrapper/gradle-wrapper.properties >/dev/null

WRAPPER_JAR_SHA256="$(shasum -a 256 android/gradle/wrapper/gradle-wrapper.jar | awk '{print $1}')"
test "$WRAPPER_JAR_SHA256" = \
  '76805e32c009c0cf0dd5d206bddc9fb22ea42e84db904b764f3047de095493f3'
```

Gradle 9.1.0 的 `-bin` ZIP SHA256 必须是 `a17ddd85a26b6a7f5ddb71ff8b05fc5104c0202c6e64782429790c933686c806`，wrapper JAR SHA256 必须是 `76805e32c009c0cf0dd5d206bddc9fb22ea42e84db904b764f3047de095493f3`；两者均来自 [Gradle 官方 checksum 参考](https://gradle.org/release-checksums/)。官方 Gradle Wrapper 文档分别定义了[下载 distribution 的 SHA256 验证](https://docs.gradle.org/current/userguide/gradle_wrapper.html#verification-of-downloaded-gradle-distributions)和[wrapper JAR 完整性校验](https://docs.gradle.org/current/userguide/gradle_wrapper.html#verifying-the-integrity-of-the-gradle-wrapper-jar)方式。

在任何 `./gradlew` 或 `scripts/build_android.sh` 执行前，上述 `grep -Fx` 必须确认 properties 中存在完全一致的 `distributionSha256Sum=...`，`test` 必须确认 wrapper JAR hash 完全一致。任一校验失败就停止，不执行 wrapper。`scripts/build_android.sh` 按 JDK 17 → SDK root → 精确 NDK → OpenCV 4.13.0 SDK → ONNX Runtime 1.27.0 与 checksum → wrapper 的顺序 fail-fast。

## 3. 台架证据采集

连接目标设备后保留以下原始输出：

```bash
adb shell getprop ro.build.version.release
adb shell getprop ro.build.version.sdk
adb shell getprop ro.product.cpu.abi
adb shell getprop ro.hardware
adb shell pm list features | grep -i camera
adb shell dumpsys media.camera > media-camera.txt
adb shell ls -l /vendor/lib64/libQnnGpu.so /vendor/lib64/libQnnSystem.so

mkdir -p qnn-device-evidence
collect_qnn_device_evidence() {
  if ! adb pull /vendor/lib64/libQnnGpu.so qnn-device-evidence/libQnnGpu.so \
    > qnn-device-evidence/libQnnGpu.pull.stdout.txt \
    2> qnn-device-evidence/libQnnGpu.pull.stderr.txt; then
    printf '%s\n' 'adb pull failed for /vendor/lib64/libQnnGpu.so' \
      | tee -a qnn-device-evidence/libQnnGpu.pull.stderr.txt >&2
    return 1
  fi

  local llvm_readelf
  llvm_readelf="$(find "$ANDROID_SDK_ROOT/ndk/28.2.13676358/toolchains/llvm/prebuilt" \
    -type f -path '*/bin/llvm-readelf' -print -quit)"
  if test -z "$llvm_readelf" || ! test -x "$llvm_readelf"; then
    printf '%s\n' 'executable llvm-readelf not found in Android NDK 28.2.13676358' >&2
    return 1
  fi

  "$llvm_readelf" -h -d qnn-device-evidence/libQnnGpu.so \
    > qnn-device-evidence/libQnnGpu.readelf.txt
}
collect_qnn_device_evidence
unset -f collect_qnn_device_evidence
```

同时收集并写入台架记录：

- 安装的 NDK revision；
- QAIRT SDK 精确版本和 SDK root；
- 对应版本 SDK 内选用的 Android QNN sample 名称；
- 部署身份是普通 APK、privileged APK、system APK 还是 vendor 组件；
- 目标摄像头是否能通过 Camera2 枚举。

`/vendor/lib64` 只是常见示例，供应商路径可能不同。如果路径不存在或 `adb pull` 被拒绝，保留 `ls`/拉取失败记录，然后在供应商镜像或对应 QAIRT 包中定位可审计的库副本。绝不把未验证的 vendor 库，或 Yocto/Linux QNN `.so`，复制进 APK。还要核对库的 AArch64 ELF 身份、依赖、许可和 Android linker namespace 可见性。

## 4. 配置后构建、安装与 JNI 身份冒烟

```bash
export JAVA_HOME=/absolute/path/to/jdk-17
export ANDROID_SDK_ROOT=/absolute/path/to/android-sdk
export RPPG_OPENCV_ANDROID_SDK=/absolute/path/to/OpenCV-android-sdk
export RPPG_ONNXRUNTIME_ANDROID=/absolute/path/to/onnxruntime-android-1.27.0

./scripts/build_android.sh

adb install -r android/app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.jagger.rppgbench/.MainActivity
```

EfficientPhys 模型不进入 Git 或 APK。拿到外部模型后先在 Mac 校验并导入应用私有目录：

```bash
MODEL=/absolute/path/to/efficientphys_pure.onnx
test "$(shasum -a 256 "$MODEL" | awk '{print $1}')" = \
  c1b321042db1335da70b0295cc84f653a2cfe90f75cff738b3045ea3c103257d
adb push "$MODEL" /data/local/tmp/efficientphys_pure.onnx
adb shell run-as com.jagger.rppgbench mkdir -p files/models
adb shell run-as com.jagger.rppgbench cp \
  /data/local/tmp/efficientphys_pure.onnx files/models/efficientphys_pure.onnx
adb shell run-as com.jagger.rppgbench sha256sum \
  files/models/efficientphys_pure.onnx
adb shell rm /data/local/tmp/efficientphys_pure.onnx
```

勾选 “Run EfficientPhys with ONNX Runtime CPU” 后，状态必须明确显示
`deep_backend=ONNX_RUNTIME_CPU`、deep BPM/quality、`deep_inference_ms` 或具体错误；
请求 CPU 后不得回退到 fake 或 QNN。

启动后页面先显示：

```text
platform=android;abi=arm64-v8a;camera=camera2;deep=onnxruntime_cpu;qnn_ready=false
```

点击 “List cameras” 时验证允许和拒绝权限两条路径；允许后页面必须显示 Camera2 NDK 返回的 ID。选择 GREEN/POS/CHROM 后点击 “Start camera”，状态 JSON 应持续增加 `accepted_frames`，`last_timestamp_sec` 严格递增，显示 measured FPS、`face_found` 和传统心率状态。切到后台后确认 `onStop()` 释放相机并生成 `events.jsonl`、`heart_rate.csv` 和 `session_summary.json`，再回到前台重复启动。按现场需要保留启动失败或 native loader 证据，例如：

```bash
adb logcat -c
adb shell am force-stop com.jagger.rppgbench
adb shell am start -n com.jagger.rppgbench/.MainActivity
APP_PID="$(adb shell pidof -s com.jagger.rppgbench | tr -d '\r')"
test -n "$APP_PID"
adb logcat -d --pid="$APP_PID" > rppg-android-foundation-logcat.txt
```

主机 APK 证明 Camera2/OpenCV/传统 pipeline 可交叉编译，主机测试证明 GREEN/POS/CHROM 和会话输出对模拟输入满足契约；只有上述真机操作和 logcat/ADB 记录才能证明目标 camera、ROI 和现场 rPPG。它仍不证明 QNN 或 Adreno。构建脚本成功时会打印上面的精确 debug APK 路径；路径中没有该文件就不得执行成功声明。

## 5. 部署身份决策门禁

1. 默认选择**普通 APK + Camera2**：申请 Android camera permission，通过受支持的 Camera2/Camera NDK 服务枚举和采集。
2. 只有在目标 camera、system camera 或 QNN vendor library 可见性确实要求时，才转为 privileged/system APK，并由设备供应商提供签名、镜像集成和策略证据。
3. 如果 camera 只存在于 `/dev/video*`，而 SELinux 或 Android 服务边界阻止应用访问，则 vendor native service 是一个需要单独设计和供应商配合的方案。

任何情况都不允许普通 APK 绕过 Android 服务直接强行访问 raw device，也不得把安全策略失败伪装成 camera 或 QNN 成功。

## 6. 后续实施顺序

1. Camera2/Camera NDK + `AImageReader`：完成 permission、camera 枚举、`YUV_420_888` row/pixel stride 转换测试、timestamp/FPS 和生命周期释放。
2. OpenCV Android ROI + traditional：仅导入 ABI 匹配的 Android `core`/`imgproc`/`objdetect`，加载 ROI 资源，然后在 APK 验证 GREEN/POS/CHROM 和结果落盘。
3. ONNX Runtime CPU：导入 hash 匹配的外部 EfficientPhys ONNX，验证固定 `[181,3,72,72]` → `[180,1]` 契约和冻结向量。
4. 并发验收：同时运行 camera、ROI/traditional、latest-only deep worker、结果输出和 UI，验证 FPS、丢帧、时延、stop/destroy、会话隔离及无旧结果窜入。
5. QNN/Adreno 仅作为后续可选优化；第一版手机演示不依赖 QAIRT。

## 7. 官方 Android 参考

- [Android NDK CMake](https://developer.android.com/ndk/guides/cmake)
- [Android ABI contract](https://developer.android.com/ndk/guides/abis)
- [Camera NDK API](https://developer.android.com/ndk/reference/group/camera)
- [`<uses-native-library>`](https://developer.android.com/guide/topics/manifest/uses-native-library-element)
- [Android linker namespaces](https://source.android.com/docs/core/architecture/vndk/linker-namespace)
- [Android SELinux](https://source.android.com/docs/security/features/selinux)

完整架构理由见 [Android NDK rPPG Runtime Design](docs/superpowers/specs/2026-07-23-android-ndk-rppg-runtime-design.md)，当前 foundation 步骤见 [Android NDK Foundation Plan](docs/superpowers/plans/2026-07-23-android-ndk-foundation.md)。
