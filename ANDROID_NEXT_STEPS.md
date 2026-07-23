# Android APK/NDK 公司机交接

本文档是 Android `arm64-v8a` 基础之后的现场交接清单。所有心率、波形和时延输出只能用于研究与工程验证，不得用于医疗诊断、治疗或其他高风险决策。

## 1. 冻结当前边界

在公司机 checkout 根目录现场记录实际 commit，不要在本文档填写伪造或过期的占位 hash：

```bash
git rev-parse HEAD
git status --short --branch
```

当前冻结边界仅包括：Gradle 源码脚手架、AGP 9.0.1、`arm64-v8a`、NDK 28.2.13676358、最小 Java Activity + JNI build identity、pipeline 协作式停止接缝和 Android 环境预检。它尚未请求/打开摄像头，未链接完整 rPPG pipeline，未导入 OpenCV Android，未在 APK 运行 POS/CHROM，未转换/加载 QNN 模型，未证明 Adreno。

模型不进 Git。不得提交 `.pth`、`.pt`、`.onnx`、`.dlc`、`.bin` 或转换后的私有模型/context 产物；只记录可审计的版本、工具参数、hash 和外部存放位置。`--deep fake` 是 host 测试接线，永远不得作为 Android 结果。

## 2. 开发机前置和 Gradle wrapper 门禁

公司机需要精确准备：

- JDK 17，且 `JAVA_HOME` 指向该 JDK。
- Android SDK platform 36 与 Android SDK Build-Tools 36。
- Android NDK `28.2.13676358`，安装在 `$ANDROID_SDK_ROOT/ndk/28.2.13676358`。
- 可用的 Gradle 9.1.0，仅用于首次生成 wrapper。

仓库目前没有已审阅的 Gradle wrapper。在已配置 Gradle 9.1.0 的公司机生成：

```bash
cd android
gradle wrapper --gradle-version 9.1.0 --distribution-type bin
cd ..
```

先审阅 `gradlew`、`gradlew.bat` 和 `gradle/wrapper/` 内容、distribution URL 及 checksum 策略，再单独提交 wrapper；不要未审即提交。`scripts/build_android.sh` 当前按 JDK 17 → SDK root → 精确 NDK → wrapper 的顺序 fail-fast，因此前三个门禁通过后，会在缺少 `android/gradlew` 处停止。

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
adb shell readelf -h /vendor/lib64/libQnnGpu.so
```

同时收集并写入台架记录：

- 安装的 NDK revision；
- QAIRT SDK 精确版本和 SDK root；
- 对应版本 SDK 内选用的 Android QNN sample 名称；
- 部署身份是普通 APK、privileged APK、system APK 还是 vendor 组件；
- 目标摄像头是否能通过 Camera2 枚举。

`/vendor/lib64` 只是常见示例，供应商路径可能不同。该路径缺少时，应从台架映像、供应商 QAIRT 包和对应 Android sample 定位真实库；绝不把 Yocto/Linux QNN `.so` 复制进 APK。还要核对库的 AArch64 ELF 身份、依赖、许可和 Android linker namespace 可见性。

## 4. 配置后构建、安装与 JNI 身份冒烟

```bash
export JAVA_HOME=/absolute/path/to/jdk-17
export ANDROID_SDK_ROOT=/absolute/path/to/android-sdk

./scripts/build_android.sh

adb install -r android/app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.jagger.rppgbench/.MainActivity
```

启动后页面预期显示：

```text
platform=android;abi=arm64-v8a;camera=not_compiled;deep=disabled;qnn_ready=false
```

按现场需要保留启动失败或 native loader 证据，例如：

```bash
adb logcat -c
adb shell am force-stop com.jagger.rppgbench
adb shell am start -n com.jagger.rppgbench/.MainActivity
adb logcat -d > rppg-android-foundation-logcat.txt
```

这个屏幕只证明 Activity 可进入且 JNI 返回了预期 build identity，不证明 camera、rPPG、QNN 或 Adreno。构建脚本成功时会打印上面的精确 debug APK 路径；路径中没有该文件就不得执行成功声明。

## 5. 部署身份决策门禁

1. 默认选择**普通 APK + Camera2**：申请 Android camera permission，通过受支持的 Camera2/Camera NDK 服务枚举和采集。
2. 只有在目标 camera、system camera 或 QNN vendor library 可见性确实要求时，才转为 privileged/system APK，并由设备供应商提供签名、镜像集成和策略证据。
3. 如果 camera 只存在于 `/dev/video*`，而 SELinux 或 Android 服务边界阻止应用访问，则 vendor native service 是一个需要单独设计和供应商配合的方案。

任何情况都不允许普通 APK 绕过 Android 服务直接强行访问 raw device，也不得把安全策略失败伪装成 camera 或 QNN 成功。

## 6. 后续实施顺序

1. Camera2/Camera NDK + `AImageReader`：完成 permission、camera 枚举、`YUV_420_888` row/pixel stride 转换测试、timestamp/FPS 和生命周期释放。
2. OpenCV Android ROI + traditional：仅导入 ABI 匹配的 Android `core`/`imgproc`/`objdetect`，加载 ROI 资源，然后在 APK 验证 GREEN/POS/CHROM 和结果落盘。
3. QAIRT converter 证据：用公司的精确 SDK 和 Android sample 转换外部 ONNX，保留命令、版本、输出、hash 或精确失败节点。
4. 真实 QNN runtime：接入版本匹配的 headers/libraries 和已验证 model/context，禁止 CPU、ORT 或 fake fallback，记录 GPU backend 初始化和 graph 执行证据。
5. 并发验收：同时运行 camera、ROI/traditional、latest-only deep worker、结果输出和 UI，验证 FPS、丢帧、时延、stop/destroy、会话隔离及无旧结果窜入。

## 7. 官方 Android 参考

- [Android NDK CMake](https://developer.android.com/ndk/guides/cmake)
- [Android ABI contract](https://developer.android.com/ndk/guides/abis)
- [Camera NDK API](https://developer.android.com/ndk/reference/group/camera)
- [`<uses-native-library>`](https://developer.android.com/guide/topics/manifest/uses-native-library-element)
- [Android linker namespaces](https://source.android.com/docs/core/architecture/vndk/linker-namespace)
- [Android SELinux](https://source.android.com/docs/security/features/selinux)

完整架构理由见 [Android NDK rPPG Runtime Design](docs/superpowers/specs/2026-07-23-android-ndk-rppg-runtime-design.md)，当前 foundation 步骤见 [Android NDK Foundation Plan](docs/superpowers/plans/2026-07-23-android-ndk-foundation.md)。
