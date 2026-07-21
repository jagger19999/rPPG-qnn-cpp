# rPPG QAIRT/QNN C++ 台架运行工程

这是面向 Linux AArch64、Qualcomm Adreno GPU 和 QAIRT/QNN 的独立 C++17 工程。它从 V4L2 摄像头或视频文件采集画面，运行传统 GREEN rPPG 基线，并把终端状态、JSONL 事件和 CSV 心率窗口保存到本地。

本项目只用于研究和工程验证，不是医疗器械，输出不得用于诊断、治疗、车辆安全闭环或其他高风险决策。

## 当前能力与边界

本仓库 `rPPG-qnn-cpp` 与 Python/Streamlit 仓库 `rPPG` 分离，拥有独立的 Git 历史、CMake 构建和发布目录。台架包不包含 Python、PyTorch、Streamlit、rPPG-Toolbox 或模型权重，也不会修改原仓库。

当前已完成：V4L2/视频输入、人脸 ROI、GREEN 基线、异步深度窗口接线、结果落盘和 QNN/OpenCL 动态库预检。真实 EfficientPhys 的 QNN 模型转换、清单校验和推理适配尚未接入，因此正式运行必须使用 `--deep disabled`。`--deep fake` 只是开发期调度测试，不能当作深度学习结果。

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

脚本使用仓库内的 `cmake/Toolchains/aarch64-linux.cmake`。交叉构建只编译测试，不在宿主机运行 AArch64 测试；测试应在台架上补跑。打包前，脚本用 `file` 强制确认交叉产物是 Linux ELF AArch64；原生模式也会确认产物格式和架构与当前主机一致。

`native` 与 `aarch64` 必须使用不同的 `BUILD_DIR`。脚本会在构建目录记录模式；发现模式不符，或发现没有可信模式标记的旧 `CMakeCache.txt` 时会在配置前停止。升级自早期版本时请指定一个新的空构建目录。可通过 `BUILD_DIR` 和 `STAGE_DIR` 使用独立目录，路径中允许空格。

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

下一阶段会冻结目标 QAIRT 版本，导出并数值校验 EfficientPhys，生成 QNN GPU 产物及模型清单，随后实现真实 `IDeepRuntime`。在该阶段通过前，任何 fake deep 的心率或波形都只属于测试数据。
