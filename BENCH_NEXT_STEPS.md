# 高通 Linux 台架下一步操作手册

本文档用于把已经在 Mac 完成的 `rPPG-qnn-cpp` 工程移交到采用
Yocto/OpenEmbedded AArch64、`aarch64-oe-linux-gcc 11.2`、QAIRT/QNN 和
Adreno GPU 的台架环境。

当前目标不是立即宣称 EfficientPhys 已在台架运行，而是按顺序取得可追溯证据：

1. 冻结台架 SDK 和动态库版本。
2. 交叉编译并运行传统 GREEN、POS 和 CHROM rPPG。
3. 验证 V4L2 摄像头、ROI 和结果落盘。
4. 验证 EfficientPhys ONNX 能否被当前 QAIRT converter 接受。
5. converter 通过后，再开发并验证真实 QNN C++ Runtime。

所有输出仅用于研究和工程验证，不用于医学诊断、治疗或车辆安全闭环。

## 1. 当前基线

Mac 端已经完成并验证：

- 独立 C++17 工程，不修改 Python/Streamlit `rPPG` 仓库。
- V4L2/视频输入接口、人脸 ROI、传统 GREEN/POS/CHROM rPPG、异步深度 worker 接线。
- 终端状态、`events.jsonl`、`heart_rate.csv` 和 `session_summary.json`。
- 官方 `PURE_EfficientPhys.pth` 的严格加载和来源哈希校验。
- EfficientPhys 静态 ONNX：输入 `frames float32 [181,3,72,72]`，输出
  `pulse float32 [180,1]`，opset 17。
- PyTorch 与 ONNX Runtime CPU 数值对齐：最大绝对误差
  `1.3530254364013672e-05`，FFT BPM 误差 `0.0 bpm`。
- Mac 模型测试 `98 passed`、Release CTest `13/13 passed`、UBSan CTest
  `13/13 passed`。

关键版本：

```text
Git branch: codex/efficientphys-qnn-export
Phase 2A implementation baseline: 10978e4
Checkpoint SHA256: e65a962e07bcac32a668e6acb9f8ed43cdb1b01cfb97262654dc5b55c0cf3a49
ONNX SHA256: c1b321042db1335da70b0295cc84f653a2cfe90f75cff738b3045ea3c103257d
Manifest SHA256: cfa333bdcb8e88f22172bceaff452823b934476ab31c8eab48e7525f65f8ffdb
```

当前尚未完成：

- 没有执行真实 Yocto AArch64 交叉编译。
- 没有执行 QAIRT/QNN converter。
- C++ 中尚未实现真实 `IDeepRuntime` QNN adapter。
- `--deep fake` 只是调度测试，不是 EfficientPhys 输出。
- 没有在 Adreno GPU 或 Linux V4L2 摄像头上完成实机验收。

在 QNN adapter 完成前，台架正式运行必须使用 `--deep disabled`。

## 2. 开始前需要准备的材料

### 2.1 源码与参考模型

当前分支仍是本地开发分支。进入台架前，应先选择合并或推送该分支，再在 Linux
构建机取得完全相同的 commit。不要从 Mac 的临时 worktree 手工复制零散源码。

需要准备：

- `rPPG-qnn-cpp` 源码，commit 必须与交接记录一致。
- `model_specs/efficientphys_pure.json`。
- ignored 参考模型 `efficientphys_pure.onnx`，SHA256 必须为上面的固定值。
- 如需重新导出，再准备官方 rPPG-Toolbox 和 `PURE_EfficientPhys.pth`；台架运行包本身不需要 `.pth`。

模型和权重保持在 Git 外。不要把 `.pth`、`.onnx`、`.dlc`、QNN context binary
或公开数据集提交到仓库。

### 2.2 Yocto/OpenEmbedded SDK

向台架供应方取得以下内容：

- `environment-setup-aarch64-oe-linux*`。
- SDK 自带的 `OEToolchainConfig.cmake`。
- 与台架 rootfs 匹配的 target sysroot。
- AArch64 OpenCV 4：`core`、`imgproc`、`videoio`、`objdetect`。
- `pkg-config` 元数据和完整 SDK 许可/版本说明。

不要只取得一个裸 `aarch64-oe-linux-gcc`。Yocto 的 `CC`、`CXX` 往往包含
`--sysroot`、CPU 参数和安全编译 flags，必须完整保留 environment setup 提供的值。

### 2.3 QAIRT/QNN 与 GPU 运行库

向供应方确认并记录：

- 精确 QAIRT/QNN 版本。
- host converter 所在系统和路径。
- 与版本匹配的 QNN headers。
- AArch64 `libQnnGpu.so` 和 `libQnnSystem.so`。
- 台架的 `libOpenCL.so`、Adreno 驱动及其依赖。
- converter、runtime 和台架镜像是否来自同一发布版本。

不要把 Mac、x86_64 Linux 和 AArch64 动态库混用。

## 3. 冻结环境证据

在 Linux 构建机 source SDK 环境后执行。将路径替换成供应方实际路径：

```bash
source /path/to/environment-setup-aarch64-oe-linux

mkdir -p evidence/sdk
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  uname -a
  command -v aarch64-oe-linux-gcc
  aarch64-oe-linux-gcc --version
  aarch64-oe-linux-g++ --version
  cmake --version
  for name in CC CXX CFLAGS CXXFLAGS LDFLAGS SDKTARGETSYSROOT \
              OECORE_NATIVE_SYSROOT OECORE_TARGET_SYSROOT PKG_CONFIG_PATH; do
    printf '%s=%q\n' "$name" "${!name-}"
  done
} | tee evidence/sdk/yocto-sdk-env.txt
```

确认 compiler 主版本为 11.2，并确认 `SDKTARGETSYSROOT` 指向实际存在的目录：

```bash
test -d "$SDKTARGETSYSROOT"
test -d "$OECORE_NATIVE_SYSROOT"

export AARCH64_CMAKE_TOOLCHAIN_FILE="$OECORE_NATIVE_SYSROOT/usr/share/cmake/OEToolchainConfig.cmake"
test -r "$AARCH64_CMAKE_TOOLCHAIN_FILE"
```

如果供应方 SDK 使用不同布局，以其提供的 `OEToolchainConfig.cmake` 绝对路径为准，
不要自己拼接一个近似 toolchain 文件。

## 4. 检查目标库架构和依赖

先定义实际路径：

```bash
export QAIRT_SDK_ROOT=/path/to/qairt
export QAIRT_TARGET_LIB_DIR=/path/to/qairt/target/aarch64/lib
export QNN_GPU_LIBRARY="$QAIRT_TARGET_LIB_DIR/libQnnGpu.so"
export QNN_SYSTEM_LIBRARY="$QAIRT_TARGET_LIB_DIR/libQnnSystem.so"
export OPENCL_LIBRARY="$SDKTARGETSYSROOT/usr/lib/libOpenCL.so"
```

逐个检查：

```bash
for library in "$QNN_GPU_LIBRARY" "$QNN_SYSTEM_LIBRARY" "$OPENCL_LIBRARY"; do
  test -r "$library"
  file "$library"
  readelf -h "$library"
  readelf -d "$library"
  sha256sum "$library"
done | tee evidence/sdk/target-libraries.txt
```

通过标准：

- QNN 和 OpenCL 目标库都是 Linux ELF 64-bit AArch64。
- `readelf -d` 中的依赖能在 target sysroot 或台架 rootfs 找到。
- 记录 SHA256，后续替换任何库都视为新的验收版本。

## 5. Yocto AArch64 交叉编译

在源码根目录执行。必须使用全新的 build 和 stage 目录：

```bash
source /path/to/environment-setup-aarch64-oe-linux

export AARCH64_CMAKE_TOOLCHAIN_FILE="$OECORE_NATIVE_SYSROOT/usr/share/cmake/OEToolchainConfig.cmake"
export CMAKE_PREFIX_PATH="$SDKTARGETSYSROOT/usr"

BUILD_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
export BUILD_DIR="build-linux-aarch64-oe-$BUILD_ID"
export STAGE_DIR="stage/rppg-qnn-aarch64-oe-$BUILD_ID"
test ! -e "$BUILD_DIR"

set -o pipefail
RPPG_BUILD_VERBOSE=1 ./scripts/build_linux.sh aarch64 2>&1 \
  | tee "evidence/yocto-aarch64-build-$BUILD_ID.log"
```

构建脚本在交叉模式下只编译，不会在 x86_64 构建机上运行 AArch64 测试。

检查结果：

```bash
file "$BUILD_DIR/rppg_qnn_live"
readelf -h "$BUILD_DIR/rppg_qnn_live"
readelf -d "$BUILD_DIR/rppg_qnn_live"

grep -E 'CMAKE_(C|CXX)_COMPILER|CMAKE_SYSROOT' "$BUILD_DIR/CMakeCache.txt"
grep -E -- 'aarch64-oe-linux-(gcc|g\+\+)|--sysroot=|-mcpu=|-march=' \
  "evidence/yocto-aarch64-build-$BUILD_ID.log"
grep -F -- "$SDKTARGETSYSROOT" "evidence/yocto-aarch64-build-$BUILD_ID.log"

find "$STAGE_DIR" -type f | LC_ALL=C sort
```

发布目录必须严格只有：

```text
bin/rppg_qnn_live
bin/run_rppg_qnn.sh
share/rppg-qnn/README.md
share/rppg-qnn/config/runtime-defaults.env
```

如果 `file` 不是 `ELF 64-bit ... ARM aarch64`，立即停止，不要复制到台架。

## 6. 版本化传输到台架

不要复用固定目录。先在构建机生成清单：

```bash
RPPG_RELEASE_ID="0.1.0-$(git rev-parse --short HEAD)-$BUILD_ID"
REMOTE_RELEASE_DIR="/tmp/rppg-qnn-$RPPG_RELEASE_ID"

(
  cd "$STAGE_DIR"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "evidence/package-$RPPG_RELEASE_ID.sha256"

ssh bench "test ! -e '$REMOTE_RELEASE_DIR'"
ssh bench "mkdir '$REMOTE_RELEASE_DIR'"
scp -r "$STAGE_DIR"/. "bench:$REMOTE_RELEASE_DIR/"
scp "evidence/package-$RPPG_RELEASE_ID.sha256" "bench:$REMOTE_RELEASE_DIR/"
ssh bench "test -x '$REMOTE_RELEASE_DIR/bin/rppg_qnn_live'"
ssh bench "file '$REMOTE_RELEASE_DIR/bin/rppg_qnn_live'"
ssh bench "cd '$REMOTE_RELEASE_DIR' && sha256sum -c 'package-$RPPG_RELEASE_ID.sha256'"
```

保留 `RPPG_RELEASE_ID`、Git commit、SDK 版本和 package SHA256 的对应关系。

## 7. 台架动态库预检

在台架上使用绝对路径：

```bash
export RELEASE_DIR=/tmp/rppg-qnn-替换为本次版本
export QAIRT_TARGET_LIB_DIR=/path/to/qairt/aarch64/lib

"$RELEASE_DIR/bin/run_rppg_qnn.sh" \
  --preflight-only \
  --qnn-gpu-library "$QAIRT_TARGET_LIB_DIR/libQnnGpu.so" \
  --opencl-library /path/to/libOpenCL.so \
  --output "$RELEASE_DIR/outputs/preflight"
```

检查：

```bash
sed -n '1,200p' "$RELEASE_DIR/outputs/preflight/events.jsonl"
sed -n '1,200p' "$RELEASE_DIR/outputs/preflight/session_summary.json"
```

通过标准：QNN GPU 和 OpenCL 两个库都能解析为台架上的实际 AArch64 文件。
当前 `deferred_to_sdk_adapter` 只表示符号级检查尚未实现，不表示 QNN 推理成功。

## 8. V4L2 摄像头和传统算法验收

### 8.1 设备检查

```bash
v4l2-ctl --list-devices
v4l2-ctl --device /dev/video0 --list-formats-ext
ls -l /dev/video0
id
```

确认：

- 当前用户有设备访问权限。
- 摄像头支持目标分辨率、像素格式和接近 30 FPS 的模式。
- 没有其他进程占用设备。
- 正脸光照稳定，Haar cascade 文件存在。

```bash
export RPPG_HAAR_CASCADE=/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml
test -r "$RPPG_HAAR_CASCADE"
```

### 8.2 先跑视频回放

优先准备一段已知可读、有人脸、约 30 FPS 的短视频：

```bash
for method in green pos chrom; do
  "$RELEASE_DIR/bin/run_rppg_qnn.sh" \
    --video /data/face-test.avi \
    --traditional "$method" \
    --deep disabled \
    --output "$RELEASE_DIR/outputs/video-$method"
done
```

三次回放必须使用同一视频。每个进程只运行一种算法，输出方法必须分别为 `GREEN`、
`POS`、`CHROM`；不允许 POS/CHROM 静默产生 GREEN 记录。

### 8.3 再跑实时摄像头

```bash
"$RELEASE_DIR/bin/run_rppg_qnn.sh" \
  --camera /dev/video0 \
  --width 1280 --height 720 --fps 30 \
  --traditional green \
  --deep disabled \
  --qnn-gpu-library "$QAIRT_TARGET_LIB_DIR/libQnnGpu.so" \
  --opencl-library /path/to/libOpenCL.so \
  --output "$RELEASE_DIR/outputs/live-green-001"
```

检查输出：

```bash
head -n 5 "$RELEASE_DIR/outputs/live-green-001/heart_rate.csv"
tail -n 20 "$RELEASE_DIR/outputs/live-green-001/events.jsonl"
```

当前程序没有固定采集时长参数。实时进程被外部信号终止时，不应只依赖最终
`session_summary.json`；同时保留已实时 flush 的 JSONL/CSV 和终端日志。

实时摄像头示例保留 `green`。验收 POS/CHROM 时应分别重新启动命令，替换
`--traditional` 和输出目录；每次切换都会重新积累十秒窗口。

传统链路通过标准：

- 实际采集 FPS 稳定，没有持续 `LOW_CAPTURE_FPS`。
- 正常正脸条件下 ROI 连续，没有持续 `FACE_NOT_FOUND`。
- `heart_rate.csv` 中所选 GREEN/POS/CHROM 结果为有限数值，方法和 backend 字段正确。
- 输出目录、Git commit、摄像头格式和运行参数已记录。

## 9. EfficientPhys 的 QAIRT/QNN 转换门禁

这一步通常在 QAIRT 支持的 Linux host 上执行，不在 Mac 上假设完成。

### 9.1 冻结输入

```bash
sha256sum efficientphys_pure.onnx
sha256sum model_specs/efficientphys_pure.json
```

哈希必须与第 1 节一致。该 ONNX 已知包含：

- 12 个 `ScatterND` 节点。
- `215315552` bytes Constant tensor。
- 最大单个 Constant 为 `119439360` bytes。

这些是 QNN converter 风险，不是已经支持的证明。

### 9.2 找到本版本 converter

不同 QAIRT 版本的目录和参数可能不同，先记录实际工具，不要套用其他版本命令：

```bash
find "$QAIRT_SDK_ROOT" -type f \
  \( -name 'qnn-onnx-converter' -o -name 'qnn-model-lib-generator' \
     -o -name 'qnn-context-binary-generator' \) -print

/absolute/path/to/qnn-onnx-converter --help \
  | tee evidence/qnn-onnx-converter-help.txt
```

随后按照该 QAIRT 版本随附文档执行 converter，并完整保存：

- 命令行和环境变量。
- stdout/stderr 日志。
- converter 退出码。
- 生成文件清单和 SHA256。
- 不支持算子、shape、内存或 Constant folding 报错。

不要为了让转换“看起来成功”而静默修改输入 shape、预处理或输出语义。

### 9.3 转换结果判定

只有同时满足以下条件才进入 C++ QNN adapter 开发：

- converter 正常退出且无 fallback/custom CPU path。
- 输入仍为 `frames float32 [181,3,72,72]`。
- 输出仍为 `pulse float32 [180,1]`。
- 能生成当前台架版本可加载的 QNN model library 或 context binary。
- 固定参考输入上的 QNN 输出可以导出并与 ONNX 参考比较。

如果 converter 因 `ScatterND` 或大 Constant 失败，停止 Runtime 接入，先评估
TSM 的等价、可审计 graph rewrite。任何 rewrite 都必须重新执行 PyTorch/ONNX
数值对齐并产生新的模型 SHA256，不能覆盖当前参考模型。

## 10. 真实 QNN C++ Runtime 开发

converter 门禁通过后再实施这一阶段。当前代码中已有 `IDeepRuntime` 抽象和
latest-only worker，新增实现应保持摄像头、传统算法和深度推理解耦。

建议拆分为：

1. `QnnRuntime`：加载 `libQnnSystem.so`、`libQnnGpu.so` 和 model/context。
2. `EfficientPhysPreprocessor`：严格复现 180 帧 RGB、72×72、全窗口
   population mean/std、TCHW、末帧复制。
3. `QnnTensorBinding`：验证 dtype、shape、byte size 和连续内存。
4. `EfficientPhysPostprocessor`：输出 180 点脉搏波、FFT BPM 和质量字段。
5. 错误映射：使用已有稳定错误码，不静默回退 CPU 或 fake backend。

完成后增加真实模式，例如：

```text
--deep qnn --backend gpu
```

在实现并通过测试前，不要在文档或演示中使用该参数。

## 11. 最终验收

### 11.1 数值一致性

同一份固定 `frames_float32.npy` 分别运行：

- PyTorch 官方模型。
- ONNX Runtime CPU 参考。
- QNN/Adreno 实现。

至少记录：最大绝对误差、平均绝对误差、Pearson、FFT BPM 误差、输出是否有限、
输入/输出原始字节 SHA256。QNN 容差必须在看到实际精度模式和输出后冻结，不能事后
按结果放宽。

### 11.2 性能与稳定性

记录：

- 摄像头请求 FPS 和实际 FPS。
- 深度窗口长度和窗口覆盖秒数。
- 预处理、QNN execute、后处理耗时。
- 首次加载耗时、稳态 P50/P95/P99。
- CPU/GPU/内存占用和温度。
- latest-only 丢弃窗口数。
- 连续运行 30 分钟及更长压力测试结果。

传统算法和深度 worker 必须分别报告耗时。深度推理变慢时，不得阻塞摄像头采集，
也不得把 fake、上一次结果或无效结果显示为当前真实结果。

### 11.3 生理实验边界

工程数值对齐通过后，再做静坐受控实验：固定光照、正脸、稳定距离，并记录手表心率
及接收时间戳。先确认采样 FPS、同步和 ROI，再比较 GREEN/POS/CHROM 与 EfficientPhys；不要用
少量在线 BPM 直接重新训练模型，也不要把手表广播值当作医学真值。

## 12. 故障停止条件

遇到以下任一情况应停止当前阶段并保存证据：

- 编译器、sysroot 或目标库架构不一致。
- 构建命令丢失 Yocto SDK 提供的复合 flags。
- 发布包不是固定四文件，或误带模型/Python 文件。
- QNN/OpenCL 库无法加载或依赖缺失。
- 摄像头持续低 FPS 或人脸 ROI 持续丢失。
- converter 使用 fallback、改变接口或无法处理图。
- QNN 输出出现 NaN/Inf、shape 不符或与参考严重不一致。
- 深度 worker 阻塞采集线程。

每次失败都保留 Git commit、SDK/库 SHA256、完整命令、日志、输入数据和输出文件，
修复后用新的 release ID 重新验收，不覆盖旧记录。

## 13. 推荐的下一次工作顺序

```text
发布/合并当前 C++ 分支
        ↓
获取 Yocto SDK 与 QAIRT/QNN 精确版本
        ↓
完成第 3～4 节环境冻结
        ↓
交叉编译并生成四文件包
        ↓
台架 preflight
        ↓
V4L2 + GREEN/POS/CHROM + CSV/JSON 验收
        ↓
QAIRT converter 可行性门禁
        ↓
实现真实 QNN C++ Runtime
        ↓
固定向量数值对齐
        ↓
Adreno 性能、稳定性和受控实验验收
```

第一项外部依赖是完整 Yocto SDK 和精确 QAIRT/QNN 版本。在这两项到位前，不应把
Mac 上的 ONNX 验证描述为台架深度推理已经完成。
