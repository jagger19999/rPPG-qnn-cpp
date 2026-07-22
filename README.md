# rPPG QAIRT/QNN C++ 台架运行工程

这是面向 Linux AArch64、Qualcomm Adreno GPU 和 QAIRT/QNN 的独立 C++17 工程。它从 V4L2 摄像头或视频文件采集画面，运行传统 GREEN rPPG 基线，并把终端状态、JSONL 事件和 CSV 心率窗口保存到本地。

本项目只用于研究和工程验证，不是医疗器械，输出不得用于诊断、治疗、车辆安全闭环或其他高风险决策。

## 当前能力与边界

本仓库 `rPPG-qnn-cpp` 与 Python/Streamlit 仓库 `rPPG` 分离，拥有独立的 Git 历史、CMake 构建和发布目录。台架包不包含 Python、PyTorch、Streamlit、rPPG-Toolbox 或模型权重，也不会修改原仓库。

当前已完成：V4L2/视频输入、人脸 ROI、GREEN 基线、异步深度窗口接线、结果落盘、QNN/OpenCL 动态库预检，以及 Mac 开发机上的 EfficientPhys 静态 ONNX 参考导出和数值校验。QAIRT/QNN 模型转换、QNN context 和真实推理适配尚未接入，因此正式运行仍必须使用 `--deep disabled`。`--deep fake` 只是开发期调度测试，不能当作深度学习结果。

## Mac 离线 EfficientPhys ONNX 参考（仅开发机）

`tools/model_export/` 是隔离的 host-only 工具，不会安装进 CMake 发布包。下面的命令只能在源码 checkout 根目录执行；安装包携带的 README 只是边界和操作说明，四文件台架包不含 `tools/`、`scripts/setup_model_export_macos.sh` 或 Python 环境。工具严格读取官方 `PURE_EfficientPhys.pth`（SHA256 `e65a962e07bcac32a668e6acb9f8ed43cdb1b01cfb97262654dc5b55c0cf3a49`），按官方推理约定处理 180 帧、30 FPS、RGB、72 x 72 的窗口：对整个窗口做一次全局 population mean/std standardize，转为 `float32` TCHW，再复制末帧形成 181 帧输入。

静态 ONNX 接口固定为：输入 `frames`、`float32`、`[181,3,72,72]` TCHW；输出 `pulse`、`float32`、`[180,1]`；默认域 opset 17。没有动态维度、fallback 或 custom domain。

在 Mac 上准备可移植的 Python 3.12 隔离环境并执行完整流程：

```bash
export RPPG_TOOLBOX_PATH=/path/to/rPPG-Toolbox
export RPPG_EFFICIENTPHYS_CHECKPOINT="$RPPG_TOOLBOX_PATH/final_model_release/PURE_EfficientPhys.pth"
export RPPG_MODEL_ARTIFACT_DIR=artifacts/model_export/efficientphys_pure

PYTHON_BIN="$(command -v python3.12)" ./scripts/setup_model_export_macos.sh

.model-export-venv/bin/python tools/model_export/generate_test_vector.py \
  --artifact-dir "$RPPG_MODEL_ARTIFACT_DIR"

.model-export-venv/bin/python tools/model_export/export_efficientphys.py \
  --toolbox "$RPPG_TOOLBOX_PATH" \
  --checkpoint "$RPPG_EFFICIENTPHYS_CHECKPOINT" \
  --artifact-dir "$RPPG_MODEL_ARTIFACT_DIR"

.model-export-venv/bin/python tools/model_export/validate_efficientphys.py \
  --toolbox "$RPPG_TOOLBOX_PATH" \
  --checkpoint "$RPPG_EFFICIENTPHYS_CHECKPOINT" \
  --artifact-dir "$RPPG_MODEL_ARTIFACT_DIR" \
  --manifest model_specs/efficientphys_pure.json

.model-export-venv/bin/python -m pytest tools/model_export/tests -q \
  -m 'not integration'

.model-export-venv/bin/python -m pytest tools/model_export/tests -q \
  -m integration
```

最后一条 integration 命令继承上面的三个 `RPPG_*` 环境变量；三个被选中的集成测试都必须运行，不能以 skip 代替成功。

PyTorch 2.12.1 的 legacy exporter 原本不能导出官方模型中的 `aten::diff`。经用户明确授权，导出工具只把精确的 `aten::diff(n=1, dim=0, prepend=None, append=None)` 映射为 ONNX 标准域的两个 `Slice` 和一个 `Sub`。这只是已审计的算子 lowering，不是模型 rewrite：官方 EfficientPhys 模型文件、`forward`、trainer、配置和 checkpoint 均未修改；不允许其他 `n`/`dim`/`prepend`/`append` 组合，也没有动态图、fallback 或自定义算子域。TSM rewrite 仍未实施。

本次参考产物的实测结果如下：

- `efficientphys_pure.onnx`：224043138 bytes，SHA256 `c1b321042db1335da70b0295cc84f653a2cfe90f75cff738b3045ea3c103257d`。
- PyTorch/ORT parity：最大绝对误差 `1.3530254364013672e-05`，平均绝对误差 `1.1705689960055881e-06`，Pearson `0.9999999999936849`，FFT BPM 差 `0.0 bpm`。
- 可移植 manifest：`model_specs/efficientphys_pure.json`，SHA256 `cfa333bdcb8e88f22172bceaff452823b934476ab31c8eab48e7525f65f8ffdb`。
- ORT CPU 本机验证为数百毫秒级，但耗时不是门禁；当前实际值见 ignored 的 `validation_report.json`，且不进入可复现 manifest。

该 ONNX 包含 12 个 `ScatterND` 节点和 215315552 bytes Constant tensor，受 216000000 bytes 固定门限约束，存在明确的 QNN converter 风险；manifest 的 `qnn_conversion` 仍为 `not_run`。上述结果只证明 Mac 上 PyTorch/ORT 的实现等价性，不代表 QAIRT converter、Adreno GPU、Linux V4L2 或生理精度已经通过。C++ 运行仍使用 `--deep disabled`，`--deep fake` 的输出不能当成模型或生理结果。

## Ubuntu 依赖

建议使用 Ubuntu 22.04 或与台架系统 ABI 一致的 Linux 开发环境：

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build pkg-config file \
  libopencv-dev opencv-data v4l-utils
```

要求 CMake 3.20 以上、支持 C++17 的 GCC/Clang，以及 OpenCV 4 的 `core`、`imgproc`、`videoio`、`objdetect` 模块。台架运行还需要与系统版本匹配的 QAIRT/QNN 目标库及 OpenCL 库。

## 构建和安装包

原生 Linux 构建会编译 Release、运行全部主机测试，并把发布包安装到 `stage/rppg-qnn`：

```bash
./scripts/build_linux.sh native
```

在 macOS 上只做主机兼容验证时，可明确指定 Homebrew OpenCV 4：

```bash
CMAKE_PREFIX_PATH=/opt/homebrew/opt/opencv@4 ./scripts/build_linux.sh native
```

交叉编译前，准备 AArch64 交叉编译器、目标 sysroot 以及其中的 AArch64 OpenCV。前缀末尾通常保留连字符：

```bash
export AARCH64_TOOLCHAIN_PREFIX=aarch64-linux-gnu-
export AARCH64_SYSROOT=/opt/sysroots/bench-aarch64
export CMAKE_PREFIX_PATH="$AARCH64_SYSROOT/usr"
./scripts/build_linux.sh aarch64
```

上述通用 prefix+sysroot 模式使用仓库内的 `cmake/Toolchains/aarch64-linux.cmake`。也可以改用绝对路径 `AARCH64_CMAKE_TOOLCHAIN_FILE`；该路径必须存在、可读并最终指向 regular file，合法 symlink 可用。设置它后不再要求 `AARCH64_TOOLCHAIN_PREFIX`/`AARCH64_SYSROOT`。交叉构建只编译测试，不在宿主机运行 AArch64 测试；测试应在台架上补跑。打包前，脚本用 `file` 强制确认交叉产物是 Linux ELF AArch64；原生模式也会确认产物格式和架构与当前主机一致。

`RPPG_BUILD_VERBOSE` 只接受 `0` 或 `1`，默认 `0`；设为 `1` 时构建调用增加 `cmake --build ... --verbose`，便于审计真实编译器与 flags。

`native` 与 `aarch64` 必须使用不同的 `BUILD_DIR`。脚本会在构建目录记录模式；发现模式不符，或发现没有可信模式标记的旧 `CMakeCache.txt` 时会在配置前停止。升级自早期版本时请指定一个新的空构建目录。可通过 `BUILD_DIR` 和 `STAGE_DIR` 使用独立目录，路径中允许空格。

### 已确认目标台架：Yocto/OpenEmbedded AArch64

目标交叉工具链是 `aarch64-oe-linux-gcc 11.2`。必须先 source 目标 SDK 提供的 environment 文件，并完整保留其中的 `CC`、`CXX`、`CFLAGS`、`CXXFLAGS`、`LDFLAGS`、`SDKTARGETSYSROOT`、`PKG_CONFIG_*` 等变量；构建脚本原样传递这些环境值，不会尝试拆分可能包含编译参数的复合 `CC`/`CXX` 字符串。目标 sysroot 内还必须安装 AArch64 版本的 OpenCV 4，不能链接 Mac 或其他宿主架构的 OpenCV。可参照 [Yocto 4.2 SDK 官方手册](https://docs.yoctoproject.org/4.2/sdk-manual/working-projects.html) 核对 environment setup 的使用方式。

```bash
source /path/to/environment-setup-aarch64-oe-linux
command -v aarch64-oe-linux-gcc
aarch64-oe-linux-gcc --version   # 应确认11.2
aarch64-oe-linux-g++ --version

# 打印并保存 SDK contract；%q 保留空格和复合参数的边界。
for name in CC CXX CFLAGS CXXFLAGS LDFLAGS SDKTARGETSYSROOT; do
  printf '%s=%q\n' "$name" "${!name-}"
done | tee yocto-sdk-build-env.txt

export AARCH64_CMAKE_TOOLCHAIN_FILE="$OECORE_NATIVE_SYSROOT/usr/share/cmake/OEToolchainConfig.cmake"
test -f "$AARCH64_CMAKE_TOOLCHAIN_FILE"
# 若 SDK 布局不同，改用供应商给出的 OEToolchainConfig.cmake 绝对路径。

export CMAKE_PREFIX_PATH="$SDKTARGETSYSROOT/usr"
BUILD_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
BUILD_DIR="build-linux-aarch64-oe-$BUILD_ID"
STAGE_DIR="stage/rppg-qnn-aarch64-oe-$BUILD_ID"
test ! -e "$BUILD_DIR"   # 必须使用全新且尚不存在的目录
set -o pipefail

RPPG_BUILD_VERBOSE=1 BUILD_DIR="$BUILD_DIR" STAGE_DIR="$STAGE_DIR" \
  ./scripts/build_linux.sh aarch64 2>&1 | tee yocto-aarch64-build.log

file "$BUILD_DIR/rppg_qnn_live"
readelf -h "$BUILD_DIR/rppg_qnn_live"
readelf -d "$BUILD_DIR/rppg_qnn_live"

# 审计 CMake 选择与 verbose 命令；逐项对照上面保存的 SDK contract。
grep -E 'CMAKE_(C|CXX)_COMPILER|CMAKE_SYSROOT' "$BUILD_DIR/CMakeCache.txt"
grep -E -- 'aarch64-oe-linux-(gcc|g\+\+)|--sysroot=|-mcpu=|-march=' \
  yocto-aarch64-build.log
grep -F -- "$SDKTARGETSYSROOT" yocto-aarch64-build.log
```

确认台架必须走供应商 SDK toolchain 文件并审计 `CMakeCache.txt` 与 verbose 编译命令确实包含 SDK compiler、必要 flags 和目标 sysroot。上面的通用 prefix+sysroot 模式仍用于简单 GNU 工具链，但不能替代含复合 `CC`/`CXX` 参数的 Yocto SDK contract。上述命令是目标 SDK 到位后的操作手册，不是已通过声明；当前 Mac 没有该 Yocto SDK，因此本次没有执行 AArch64 交叉编译。

开始 QAIRT 接入前，需要从目标环境采集并冻结：精确 QAIRT 版本与 SDK root（SDK 提供的 `SDK_ROOT`，并按本项目约定设为 `QAIRT_SDK_ROOT`）、`$QAIRT_SDK_ROOT/include/QNN` headers、AArch64 `libQnnGpu.so` 和 `libQnnSystem.so`，以及台架匹配的 OpenCL loader/driver。对每个 `.so` 使用 `file` 和 `readelf -h/-d` 核对其确为 AArch64、依赖可由目标 sysroot/台架满足；Mac 动态库不能用于交叉链接或作为目标运行证据。当前 Mac 也不能据此宣称 QNN 转换成功。

构建成功后，把固定四文件 stage 复制到一个唯一版本目录，再从同一目录做动态库和摄像头预检。复制前必须确认目标不存在；不要复用固定目录，否则可能误跑旧 binary：

```bash
RPPG_RELEASE_ID="0.1.0-$(git rev-parse --short HEAD)-$BUILD_ID"
REMOTE_RELEASE_DIR="/tmp/rppg-qnn-$RPPG_RELEASE_ID"
ssh bench "test ! -e '$REMOTE_RELEASE_DIR'"
ssh bench "mkdir '$REMOTE_RELEASE_DIR'"
scp -r "$STAGE_DIR"/. "bench:$REMOTE_RELEASE_DIR/"
ssh bench "test -x '$REMOTE_RELEASE_DIR/bin/rppg_qnn_live'"

ssh bench "QAIRT_TARGET_LIB_DIR=/path/to/qairt/aarch64/lib \
  '$REMOTE_RELEASE_DIR/bin/run_rppg_qnn.sh' \
    --preflight-only \
    --qnn-gpu-library /path/to/qairt/aarch64/lib/libQnnGpu.so \
    --opencl-library /path/to/libOpenCL.so \
    --output '$REMOTE_RELEASE_DIR/preflight'"

ssh bench 'v4l2-ctl --list-devices'
ssh bench 'v4l2-ctl --device /dev/video0 --list-formats-ext'
ssh bench 'ls -l /dev/video0'
```

安装包布局：

```text
stage/rppg-qnn/
  bin/rppg_qnn_live
  bin/run_rppg_qnn.sh
  share/rppg-qnn/README.md
  share/rppg-qnn/config/runtime-defaults.env
```

发布包不应出现 `models/`、`.pth`、`.onnx`、`.dlc` 或 Python 文件。每次打包都先安装到 `STAGE_DIR` 同级的全新临时目录，核对固定文件白名单和架构，再替换最终目录；旧目录里的模型、Python 文件或其他污染不会遗留。为避免误删，脚本按物理绝对路径检查边界：`STAGE_DIR` 不允许是源码目录、构建目录或它们的任何祖先，也不能是符号链接；位于源码目录下的默认 `stage/rppg-qnn` 仍然合法。

## QAIRT/QNN 环境

开发机用 `QAIRT_SDK_ROOT` 指向与台架版本匹配的 QAIRT SDK。当前基础阶段还未编译 QNN API 适配层，该变量先作为后续模型接入的固定约定。台架用 `QAIRT_TARGET_LIB_DIR` 指向目标架构动态库目录，例如：

```bash
export QAIRT_SDK_ROOT=/opt/qairt/2.xx.x
export QAIRT_TARGET_LIB_DIR=/opt/qairt/2.xx.x/lib/aarch64-linux
```

启动脚本只把发布包的 `lib` 和可选的 `QAIRT_TARGET_LIB_DIR` 前置到当前 `LD_LIBRARY_PATH`，不会使用 `sudo`、复制文件到系统库目录或覆盖全局配置。

安装包附带可修改的环境默认值。文件中的变量带有 `export`，因此 source 后对启动的子进程可见：

```bash
source stage/rppg-qnn/share/rppg-qnn/config/runtime-defaults.env
```

先用绝对路径做动态库预检：

```bash
stage/rppg-qnn/bin/run_rppg_qnn.sh \
  --preflight-only \
  --qnn-gpu-library "$QAIRT_TARGET_LIB_DIR/libQnnGpu.so" \
  --opencl-library /usr/lib/aarch64-linux-gnu/libOpenCL.so.1 \
  --output outputs/preflight
```

成功报告必须记录两条库的实际解析路径。当前符号级 QNN API 校验会显示 `deferred_to_sdk_adapter`，直到下一阶段用目标 SDK 头文件锁定接口；这不是 EfficientPhys 已经运行的证明。缺库会以 `QNN_LIBRARY_NOT_FOUND` 和退出码 7 结束，并在输出目录保存预检事件和会话摘要。

## 摄像头准备

列出设备和格式：

```bash
v4l2-ctl --list-devices
v4l2-ctl --device /dev/video0 --list-formats-ext
ls -l /dev/video0
```

当前用户通常需要属于 `video` 组；修改组成员后要重新登录：

```bash
sudo usermod -aG video "$USER"
```

确认 OpenCV Haar 文件存在。如果系统路径不同，显式设置：

```bash
export RPPG_HAAR_CASCADE=/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml
```

## 运行

先用合成视频集成测试验证整条 C++ 接线，不需要摄像头：

```bash
BUILD_DIR=build-linux-native ./scripts/run_smoke.sh
```

用自己的有人脸视频做回放冒烟测试。此脚本会明确启用 fake deep，仅验证异步接线：

```bash
./scripts/run_smoke.sh /data/face-16s.avi outputs/video-smoke
```

台架正式采集时保持真实深度关闭：

```bash
stage/rppg-qnn/bin/run_rppg_qnn.sh \
  --camera /dev/video0 \
  --width 1280 --height 720 --fps 30 \
  --traditional green \
  --deep disabled \
  --qnn-gpu-library "$QAIRT_TARGET_LIB_DIR/libQnnGpu.so" \
  --opencl-library /usr/lib/aarch64-linux-gnu/libOpenCL.so.1 \
  --output outputs/live-001
```

也可以用 `--video /data/input.avi` 替代 `--camera`；二者不能同时设置。

## 输出文件

每个 `--output` 目录包含：

- `events.jsonl`：逐行 JSON 事件，包括预检、每秒帧健康、心率结果和运行错误。
- `heart_rate.csv`：每个心率窗口一行，记录方法、BPM、置信度、有效状态、输入 FPS、耗时、后端和模型哈希。
- `session_summary.json`：退出码及各类事件计数；程序安全结束后以原子重命名发布。

终端每秒显示帧率、人脸状态，并在窗口完成时显示心率。输出仅供实验分析。

## 稳定错误码

| 退出码 | 错误标识 | 处理建议 |
|---:|---|---|
| 2 | `CONFIG_INVALID` | 检查未知、重复、互斥参数和数值范围。 |
| 3 | `CAMERA_OPEN_FAILED` | 检查设备节点、权限、占用或视频是否完整。 |
| 4 | `CAMERA_FORMAT_UNSUPPORTED` | 用 `v4l2-ctl` 检查分辨率和像素格式。 |
| 5 | `LOW_CAPTURE_FPS` | 降低分辨率、检查 USB 带宽和曝光。 |
| 6 | `FACE_NOT_FOUND` | 调整正脸、距离、光照和 Haar 文件。 |
| 7 | `QNN_LIBRARY_NOT_FOUND` | 使用目标架构绝对路径，检查依赖和 `LD_LIBRARY_PATH`。 |
| 8 | `QNN_API_INCOMPATIBLE` | 核对 QAIRT SDK 与台架运行库版本。 |
| 9 | `QNN_GPU_INIT_FAILED` | 检查 Adreno/OpenCL 驱动、设备权限和后端日志。 |
| 10 | `MODEL_MANIFEST_INVALID` | 核对模型清单、形状和 SHA256。 |
| 11 | `MODEL_LOAD_FAILED` | 检查 QNN 模型产物、权限和版本。 |
| 12 | `INFERENCE_FAILED` | 保留日志与冻结输入，核对图执行错误。 |
| 13 | `OUTPUT_WRITE_FAILED` | 检查输出目录权限、空间和文件系统状态。 |
| 1 | `UNEXPECTED_EXCEPTION` | 保存完整输出并按未分类故障排查。 |

`FACE_NOT_FOUND` 和 `LOW_CAPTURE_FPS` 在采集过程中也可能作为可恢复状态记录，不一定立即退出。

## 版本化部署与回滚

不要覆盖正在运行的目录。每个构建解压到唯一版本目录，完成预检后再原子切换 `current` 软链接：

```bash
sudo mkdir -p /opt/rppg-qnn/releases
sudo cp -a stage/rppg-qnn /opt/rppg-qnn/releases/0.1.0-20260722
sudo ln -sfn /opt/rppg-qnn/releases/0.1.0-20260722 /opt/rppg-qnn/current.new
sudo mv -Tf /opt/rppg-qnn/current.new /opt/rppg-qnn/current
```

服务或人工命令始终调用 `/opt/rppg-qnn/current/bin/run_rppg_qnn.sh`。回滚时，把 `current.new` 指向上一个已验收目录并重复最后两步。部署步骤可以由受控运维账户执行；运行脚本本身不需要 root，也不会修改系统目录。

## 后续工作

下一阶段是在目标 SDK 到位后冻结精确 QAIRT 版本，运行 QAIRT converter 验证上述 `ScatterND`/Constant 风险，生成 QNN context 和目标模型清单，再实现真实 `IDeepRuntime`。在目标转换、台架预检和端到端推理通过前，任何 fake deep 的心率或波形都只属于测试数据。
