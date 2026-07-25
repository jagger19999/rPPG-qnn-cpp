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

本机当前推荐环境变量（按实际路径调整）：

```bash
export JAVA_HOME=/Users/wangjie/.local/jdks/zulu17.68.17-ca-jdk17.0.20-macosx_aarch64/Contents/Home
export ANDROID_SDK_ROOT=/opt/homebrew/share/android-commandlinetools
export RPPG_OPENCV_ANDROID_SDK=/Users/wangjie/.local/android-deps/OpenCV-android-sdk
export RPPG_ONNXRUNTIME_ANDROID=/Users/wangjie/.local/android-deps/onnxruntime-android-1.27.0

./scripts/build_android.sh

adb install -r android/app/build/outputs/apk/debug/app-debug.apk

# import model (external ONNX; never commit)
./scripts/import_efficientphys_model.sh \
  /Users/wangjie/.config/superpowers/worktrees/rPPG-qnn-cpp/efficientphys-qnn-export/artifacts/model_export/efficientphys_pure/efficientphys_pure.onnx

adb shell am start -n com.jagger.rppgbench/.MainActivity
```

EfficientPhys 模型不进入 Git 或 APK。拿到外部模型后先在 Mac 校验并导入应用私有目录：

```bash
MODEL=/absolute/path/to/efficientphys_pure.onnx
# Preferred helper (verifies SHA-256 then adb run-as import):
./scripts/import_efficientphys_model.sh "$MODEL"
# Manual equivalent:
# test "$(shasum -a 256 "$MODEL" | awk '{print $1}')" = \
#   c1b321042db1335da70b0295cc84f653a2cfe90f75cff738b3045ea3c103257d
# adb push "$MODEL" /data/local/tmp/efficientphys_pure.onnx
# adb shell run-as com.jagger.rppgbench mkdir -p files/models
# adb shell run-as com.jagger.rppgbench cp \
#   /data/local/tmp/efficientphys_pure.onnx files/models/efficientphys_pure.onnx
# adb shell run-as com.jagger.rppgbench sha256sum \
#   files/models/efficientphys_pure.onnx
# adb shell rm /data/local/tmp/efficientphys_pure.onnx
```

勾选 “Run EfficientPhys with ONNX Runtime CPU” 后，状态必须明确显示
`deep_backend=ONNX_RUNTIME_CPU`、deep BPM/quality、`deep_inference_ms` 或具体错误；
请求 CPU 后不得回退到 fake 或 QNN。

### 手表 BLE 参考心率（GT 5 Pro 心率广播）

APK 额外请求 `BLUETOOTH_SCAN`（`neverForLocation`）与 `BLUETOOTH_CONNECT`。真机前请关闭 HyperRate 等占用手表连接的应用，并在手表上开启「心率广播」。

UI：`扫描心率设备` → 选择设备 → `连接手表`；`断开手表` 取消自动重连。相机 `Stop`/`onStop` **不断开**手表；只有显式断开或 Activity `onDestroy` 关闭 worker。

状态行应出现 `watch_status` / `watch_bpm`，合格窗口出现 `watch_alignment`、`watch_reference_bpm`、`watch_abs_error_bpm`、`watch_coverage`。会话目录在相机 stop 时追加：

- `watch_heart_rate_samples.csv`
- `watch_rppg_alignments.csv`
- `watch_export_notes.txt`（实验参考声明；接收时间为手机侧）

```bash
adb shell run-as com.jagger.rppgbench ls files/sessions
# 再按实际 session-* 目录拉取 CSV
```

设计与实现计划：

- [Android ORT CPU + Watch BLE Design](docs/superpowers/specs/2026-07-24-android-onnx-cpu-watch-ble-design.md)
- [Android ORT CPU + Watch BLE Plan](docs/superpowers/plans/2026-07-24-android-onnx-cpu-watch-ble.md)

### Phase 1 Live HR UI 真机门禁（一期退出条件）

主机已交叉构建三卡 Live UI、ROI 缩略图 JNI 与 launcher 图标；以下项**必须**在目标设备上目视确认，不可用主机测试代替。Phase 2（实时预览 + 人脸框）尚未开始，不在本清单内。

- [ ] **Launcher 图标**：安装后桌面显示 `logo.png` 衍生的品牌图标，而非系统默认机器人。
- [ ] **三卡 Live 心率**：应用顶部并排三个大号 BPM 卡片（传统 rPPG / 深度 EfficientPhys / 手表广播），数字清晰可读；**不得**把整段 status JSON 堆在顶部主区域。
- [ ] **ROI 脸图**：检测到人脸时 ROI 缩略图持续更新；未检测到人脸时显示占位（如「未检测到人脸」），不得空白或崩溃。
- [ ] **折叠分区**：相机、手表、诊断分别为可折叠区块；**诊断默认收起**；首次进入可展开相机一次。
- [ ] **Stop / onStop 生命周期**：点击 Stop 或 Activity `onStop()` 释放相机并写会话产物，但**不断开**手表 BLE；只有显式「断开手表」或 `onDestroy` 才关闭 BLE worker。
- [ ] **Phase 2 仍待**：实时 Camera2 预览 Surface + 人脸框 overlay 未交付；预览失败降级策略见 [Live HR UI 设计](docs/superpowers/specs/2026-07-24-android-live-hr-ui-design.md)。

设计与实现计划：

- [Android Live HR UI Design](docs/superpowers/specs/2026-07-24-android-live-hr-ui-design.md)
- [Android Live HR UI Plan](docs/superpowers/plans/2026-07-24-android-live-hr-ui.md)

### 仍待真机门禁（不可用主机测试代替）

1. 安装 debug APK；导入并校验 ONNX SHA-256。
2. 传统相机路径：人脸 ROI + GREEN/POS/CHROM BPM。
3. 勾选 EfficientPhys 后 deep 字段更新，采集 FPS 不明显塌陷。
4. GT 5 Pro 开心率广播 → ≤15s 扫到 → 连接后数秒内显示广播 BPM。
5. 合格窗口出现对齐误差/覆盖率；关闭广播后 ≤2s 进入 `STALE`。
6. 意外断线最多重连 3 次；用户断开后保持断开。
7. 会话目录写出手表样本与对齐 CSV。

启动后页面先显示：

```text
platform=android;abi=arm64-v8a;camera=camera2;deep=onnxruntime_cpu;qnn_ready=false
```

点击 “List cameras” 时验证允许和拒绝权限两条路径；允许后页面必须显示 Camera2 NDK 返回的 ID。选择 GREEN/POS/CHROM 后点击 “Start camera”，状态 JSON 应持续增加 `accepted_frames`，`last_timestamp_sec` 严格递增，显示 measured FPS、`face_found` 和传统心率状态（含 `window_start_sec` / `window_end_sec`）。切到后台后确认 `onStop()` 释放相机并生成会话产物，手表连接应仍保持，再回到前台可重复启动相机。按现场需要保留启动失败或 native loader 证据，例如：

```bash
adb logcat -c
adb shell am force-stop com.jagger.rppgbench
adb shell am start -n com.jagger.rppgbench/.MainActivity
APP_PID="$(adb shell pidof -s com.jagger.rppgbench | tr -d '\r')"
test -n "$APP_PID"
adb logcat -d --pid="$APP_PID" > rppg-android-foundation-logcat.txt
```

主机 APK 证明 Camera2/OpenCV/传统/ORT CPU/手表 Java BLE 可交叉编译；只有上述真机操作和 logcat/ADB 记录才能证明目标 camera、手表广播与现场对齐。它仍不证明 QNN 或 Adreno。构建脚本成功时会打印上面的精确 debug APK 路径；路径中没有该文件就不得执行成功声明。

## 5. 部署身份决策门禁

1. 默认选择**普通 APK + Camera2**：申请 Android camera permission，通过受支持的 Camera2/Camera NDK 服务枚举和采集。
2. 只有在目标 camera、system camera 或 QNN vendor library 可见性确实要求时，才转为 privileged/system APK，并由设备供应商提供签名、镜像集成和策略证据。
3. 如果 camera 只存在于 `/dev/video*`，而 SELinux 或 Android 服务边界阻止应用访问，则 vendor native service 是一个需要单独设计和供应商配合的方案。

任何情况都不允许普通 APK 绕过 Android 服务直接强行访问 raw device，也不得把安全策略失败伪装成 camera 或 QNN 成功。

## 6. 后续实施顺序

1. Camera2/Camera NDK + `AImageReader`：完成 permission、camera 枚举、`YUV_420_888` row/pixel stride 转换测试、timestamp/FPS 和生命周期释放。
2. OpenCV Android ROI + traditional：仅导入 ABI 匹配的 Android `core`/`imgproc`/`objdetect`，加载 ROI 资源，然后在 APK 验证 GREEN/POS/CHROM 和结果落盘。
3. ONNX Runtime CPU：导入 hash 匹配的外部 EfficientPhys ONNX，验证固定 `[181,3,72,72]` → `[180,1]` 契约和冻结向量。
4. 手表 BLE 参考心率：GT 5 Pro 心率广播扫描/连接/对齐/CSV（见 §4；真机门禁未完成前不得宣称设备成功）。
5. 并发验收：同时运行 camera、ROI/traditional、latest-only deep worker、手表参考与 UI，验证 FPS、丢帧、时延、stop/destroy、会话隔离及无旧结果窜入。
6. QNN/Adreno 仅作为后续可选优化；第一版手机演示不依赖 QAIRT。

## 7. 官方 Android 参考

- [Android NDK CMake](https://developer.android.com/ndk/guides/cmake)
- [Android ABI contract](https://developer.android.com/ndk/guides/abis)
- [Camera NDK API](https://developer.android.com/ndk/reference/group/camera)
- [`<uses-native-library>`](https://developer.android.com/guide/topics/manifest/uses-native-library-element)
- [Android linker namespaces](https://source.android.com/docs/core/architecture/vndk/linker-namespace)
- [Android SELinux](https://source.android.com/docs/security/features/selinux)

完整架构理由见 [Android NDK rPPG Runtime Design](docs/superpowers/specs/2026-07-23-android-ndk-rppg-runtime-design.md)，当前 foundation 步骤见 [Android NDK Foundation Plan](docs/superpowers/plans/2026-07-23-android-ndk-foundation.md)，ORT CPU + 手表 BLE 见 [2026-07-24 design](docs/superpowers/specs/2026-07-24-android-onnx-cpu-watch-ble-design.md)。
