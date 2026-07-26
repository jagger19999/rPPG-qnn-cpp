# TSCAN Android 数值对齐与运行时分层设计

日期：2026-07-26
状态：已与用户确认，待实施计划

## 1. 背景

Android 版 TSCAN 当前出现三个相关但应分开处理的问题：深度心率结果明显弱于 Python 版、推理耗时明显更长、心率曲线未正常显示。现有实现把窗口构建、传统算法 ROI、深度模型 ROI、TSCAN 前处理、ONNX 执行、心率后处理和 UI 展示连接在一起，多个实现差异会互相掩盖，无法通过端到端结果判断根因。

本设计采用“契约优先、逐层对齐”的方式。第一目标是让 Python 与 Android/C++ 对同一输入产生一致的预处理张量、模型波形、BPM 和置信度；在此基础上才比较性能、优化运行时和修复图表。QNN/NPU 接入不属于本轮正确性修复范围。

## 2. 目标与非目标

### 2.1 目标

- 建立可重复的 Python、ONNX Runtime 与 C++ 数值一致性基线。
- 将深度推理拆分为可独立测试的窗口、ROI、前处理、模型执行和后处理组件。
- 保持传统算法路径行为不变，同时为 TSCAN 提供与 Python 一致的完整人脸输入。
- 统一 Python 和 Android 的 BPM、confidence、有效性判断及计时定义。
- 建立可诊断的真实曲线，并提供明确隔离的可选模拟演示模式。
- 每个阶段都可独立测试、验证和回退。

### 2.2 非目标

- 不在数值基线建立前优化 NEON、线程数、NNAPI 或 QNN。
- 不重新训练或修改 TSCAN 权重。
- 不改变现有传统 GREEN/POS/CHROM 算法的 ROI 和计算契约。
- 不用模拟数据替代或美化真实测量结果。
- 不在同一增量中同时修复算法、性能和 UI。

## 3. 目标数据流

```text
Camera Frame
    |
    v
FaceDetector / FaceTracker
    |-------------------------------|
    v                               v
TraditionalRoiExtractor       DeepFaceExtractor
cheek ROI                     expanded full face (1.5x)
    |                               |
GREEN / POS / CHROM                 v
                              DeepWindowBuilder
                              uniform 180 frames / 6 s
                                    |
                                    v
                              TscanPreprocessor
                              RGB NHWC -> float32 NCHW
                                    |
                                    v
                              ITscanModelRuntime
                              PyTorch or ONNX Runtime
                                    |
                                    v
                              TscanPostprocessor
                              waveform -> BPM/confidence
                                    |
                                    v
                              Result + Timing + Diagnostics
                                    |
                                    v
                              Session sink / UI chart
```

共享边界只到人脸检测/跟踪结果。传统算法与深度模型拥有不同的 ROI 提取器和状态，避免传统算法所需的脸颊区域污染 TSCAN 输入。

## 4. 冻结参考契约

第一阶段生成并版本化一组小型、可复现的测试向量清单。大型二进制向量若不适合进入 Git，应由确定性脚本从固定输入生成，并通过 SHA-256 清单验证。

参考数据包括：

- 180 帧 RGB 输入，形状 `[180, 72, 72, 3]`，类型 `float32`。
- TSCAN 前处理张量，形状 `[180, 6, 72, 72]`，类型 `float32`。
- PyTorch 输出波形，形状 `[180, 1]`。
- ONNX Runtime 输出波形，形状 `[180, 1]`。
- 参考 BPM、confidence、有效状态和参数元数据。
- 模型 SHA-256、测试向量 SHA-256、采样率和实现版本。

验收阈值：

- Python/C++ 前处理最大绝对误差小于 `1e-5`。
- PyTorch/ONNX 波形最大绝对误差小于 `1e-4`。
- Python/C++ 波形 Pearson 相关系数大于 `0.9999`。
- BPM 相同；若以后引入不同 FFT 实现，误差不得超过一个 FFT bin。当前 180 点、30 Hz 下一个 bin 为 10 BPM。
- confidence 绝对误差小于 `1e-4`。

随机张量只能验证模型导出等价性，不能作为生理数据链路的唯一验收输入。至少包含一组固定真实或回放人脸窗口，以及一组边界输入。

## 5. 组件设计

### 5.1 `DeepWindowBuilder`

职责仅限于：

- 接收带单调时间戳的深度人脸 ROI。
- 保留足够覆盖 6 秒窗口的帧。
- 检查最低源 FPS、最大帧间隙和起始覆盖。
- 使用最近邻规则生成 `endpoint=false` 的 180 个目标采样点。
- 输出 RGB NHWC 张量及源 FPS、帧数、最大间隙等元数据。

它不执行颜色归一化、模型前处理或 BPM 计算。Python 与 C++ 必须用冻结时间戳测试验证选择了相同的源帧序号。

### 5.2 ROI 分离

`TraditionalRoiExtractor` 继续生成现有 cheek ROI。`DeepFaceExtractor` 根据检测/跟踪的人脸框生成以人脸中心为中心的 1.5 倍扩展矩形，并裁剪到图像边界。两者共享人脸框，但不共享裁剪图像。

深度 ROI 契约：

- Python/C++ 边界最多相差 1 像素。
- 无有效人脸框时不向深度窗口添加帧。
- 人脸框发生明显跳变时记录质量标志；本轮不引入复杂稳定器。
- UI 缩略图必须标注展示的是传统 ROI 还是深度 ROI。

### 5.3 `TscanPreprocessor`

输入为 `[180,72,72,3]` RGB `float32`，输出为连续的 `[180,6,72,72]` NCHW `float32`。

冻结公式：

```text
normalized_diff[t] = (frame[t+1] - frame[t])
                     / (frame[t+1] + frame[t] + 1e-7)
normalized_diff[0:179] /= population_std(normalized_diff[0:179])
normalized_diff[179] = 0

appearance = (frames - population_mean(frames))
             / population_std(frames)

model_input = concat(normalized_diff, appearance, channel_axis)
```

同时冻结：RGB 通道顺序、差分方向、最后一帧为零、population standard deviation、NHWC 到 NCHW 排列和 `float32` 输出。标准差过小应返回结构化 `PREPROCESS_ZERO_VARIANCE`，不能产生非有限值。

最初实现优先保证可读性和数值一致性。完成基线后才允许复用缓冲区或向量化，且优化前后必须通过同一测试向量。

### 5.4 `ITscanModelRuntime`

接口只接收已完成前处理的模型张量并返回原始波形。Android ONNX Runtime 不再内部解释 RGB、构建差分或计算 BPM。

运行时职责：

- 验证输入形状和类型。
- 执行模型 session。
- 验证输出为有限的 `[180,1]` `float32`。
- 记录纯 `model_run_ms`。
- 把 ONNX 异常转换为稳定错误码和详细诊断信息。

### 5.5 `TscanPostprocessor`

Python 与 C++ 使用同一契约：

```text
waveform
  -> subtract mean
  -> Hann window
  -> RFFT or mathematically equivalent DFT
  -> retain 0.75 to 2.5 Hz
  -> select maximum-power bin
  -> BPM = frequency * 60
  -> confidence = peak power / in-band total power
```

固定参数为采样率 30 Hz、心率范围 45–150 BPM 和共同的 confidence 有效阈值。常量、过短或包含非有限值的波形返回稳定错误状态，不抛出不可诊断的通用异常。

## 6. 计时与性能架构

废除含义模糊的单一推理计时作为比较依据，结果中分别记录：

- `window_build_ms`
- `preprocess_ms`
- `queue_wait_ms`
- `model_run_ms`
- `postprocess_ms`
- `pipeline_total_ms`
- `is_warmup`

Python 与 Android 使用相同边界。跨后端性能只比较 `model_run_ms`；用户等待和调度分析使用 `pipeline_total_ms` 与 `queue_wait_ms`。

正确性通过后按以下顺序优化：

1. 复用连续 `float32` 缓冲区，减少每次窗口的分配。
2. 预计算 Hann 窗和有效频率 bin。
3. 测量并选择 ONNX Runtime 线程参数。
4. 评估 NNAPI。
5. 单独立项评估 QNN/NPU。

深度 worker 保持 latest-only。若生产速度低于提交速度，新窗口替换旧待处理窗口，不允许无界排队。

## 7. 图表架构

图表具有互斥且明确标注的数据模式：

- `REAL`：传统、深度和手表的实际有效结果。
- `SIMULATION`：显式启用的演示数据，不写入真实会话 CSV，也不参与误差评估。

WebView 使用受支持的 Android asset 加载方式。页面生命周期和数据协议要求：

- 页面未加载完成时缓存最新事件，加载完成后补发。
- JavaScript 执行错误进入 `chart_status` 诊断。
- 横坐标始终以当前时间为右边界，保留最近 60 秒。
- 单个有效点也绘制标记。
- 无数据时显示“等待有效心率”，而不是空白画布。
- 无效结果产生断点，不重复旧值伪造连续曲线。
- Activity 销毁时停止更新并释放 WebView。

图表事件包含 `timestamp`、`source`、`bpm`、`confidence`、`valid` 和 `invalid_reason`。模拟事件额外标记 `simulated=true`。

## 8. 状态与错误传播

诊断按组件分层：

- `capture_status`
- `face_status`
- `deep_roi_status`
- `window_status`
- `preprocess_status`
- `model_status`
- `postprocess_status`
- `chart_status`

稳定错误码至少包括：

- `FACE_NOT_FOUND`
- `LOW_SOURCE_FPS`
- `CAPTURE_GAP`
- `START_COVERAGE_MISSING`
- `PREPROCESS_ZERO_VARIANCE`
- `MODEL_LOAD_FAILED`
- `MODEL_INFERENCE_FAILED`
- `MODEL_OUTPUT_INVALID`
- `LOW_CONFIDENCE`
- `CHART_ASSET_LOAD_FAILED`
- `CHART_SCRIPT_FAILED`

UI 展示简短中文说明；日志、状态 JSON 与会话输出保存稳定错误码和详细信息。错误不能跨层被统一折叠为“深度不可用”。

## 9. 分阶段实施与验收

### 阶段 1：参考基线

生成冻结测试向量和清单，验证 PyTorch 与 ONNX 导出等价性。此阶段不修改 Android 生产路径。

### 阶段 2：前处理组件

先写失败的 C++ 契约测试，再拆出 `TscanPreprocessor` 并达到参考阈值。保留现有运行时接线，直到测试通过。

### 阶段 3：ROI 分离

增加 `DeepFaceExtractor`，验证裁剪契约，同时证明传统算法测试和结果未改变。

### 阶段 4：后处理组件

先冻结 Python 输出，再实现 C++ `TscanPostprocessor`，对齐 BPM、confidence 和有效状态。

### 阶段 5：端到端一致性

让固定窗口依次通过 Python、ONNX 和 C++，记录张量、波形和最终结果差异。只有达到第 4 节阈值后，数值修复才算完成。

### 阶段 6：计时拆分

引入分段计时，分别报告 warm-up 和稳态结果。不得用旧 `inference_ms` 与 Python 新指标混合比较。

### 阶段 7：性能优化

每次只改变一个因素，记录优化前后相同输入、相同线程条件下的稳态分位数，并重新运行数值一致性测试。

### 阶段 8：真实图表

修复 asset 加载、页面握手、滑动时间轴、空状态和错误诊断；使用真实结果完成 Android UI 测试与真机验收。

### 阶段 9：模拟模式

增加显式开关和明显水印，验证模拟数据不会进入真实会话与对齐评估。

### 阶段 10：真机验收

在目标 Android 设备上记录：摄像头 FPS、丢帧、各阶段耗时、连续内存、传统/深度/手表曲线和会话文件。没有真机数据时不得声称移动端性能或生理精度已达标。

## 10. 测试策略

所有行为修改遵循测试先行：先写能稳定复现现有差异的失败测试，确认失败原因正确，再做最小实现。

测试层次：

- 单元测试：ROI、窗口重采样、前处理、后处理、错误码。
- 数值契约测试：Python 参考向量对 C++。
- 模型契约测试：ONNX 输入/输出名称、形状、类型和模型哈希。
- 集成测试：固定窗口到最终心率结果。
- Android UI 测试：asset 加载、页面握手、数据补发、60 秒滑窗、空状态。
- 真机测试：稳定性能、生命周期、摄像头与 WebView 行为。

旧 CMake 构建目录引用失效 worktree 时应重新配置到新的独立构建目录；不得把缓存路径错误误判为算法测试失败。

## 11. 回退与兼容性

- 每个阶段单独提交，不混合算法、性能和 UI。
- 新组件先由测试和受控接线验证，再替换旧逻辑。
- 会话 schema 新增计时与状态字段时提升 schema 版本，旧读取器对未知字段保持兼容。
- 在真实模式完成前，深度路径可保持默认关闭或明确标记实验状态。
- 任一优化导致数值阈值回归时，立即回退该优化，不调整参考阈值掩盖差异。

## 12. 完成定义

本轮工作只有同时满足以下条件才完成：

- 固定输入上的 Python、ONNX 和 C++ 达到数值阈值。
- 传统算法路径没有行为回归。
- Android 计时能区分窗口、前处理、排队、模型和后处理。
- 真实图表在 60 秒前后均持续可见，错误状态可诊断。
- 模拟模式显式标注并与真实会话隔离。
- 自动化测试通过，目标设备真机验收记录完整。
