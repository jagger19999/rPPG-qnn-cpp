# Android ONNX Runtime CPU + 手表 BLE 参考心率设计

## 1. 目标与范围

在已可交叉构建的 Android Camera2/NDK APK 上完成两项工作：

1. 使用已校验的 EfficientPhys ONNX，经 **ONNX Runtime Android 1.27.0 CPU** 与传统 GREEN/POS/CHROM 并行运行。
2. 接入 HUAWEI WATCH GT 5 Pro 的 **心率广播**（BLE 标准 Heart Rate Service），作为实验参考真值 `watch_reference`，与摄像头 rPPG 窗口对齐展示与导出。

第一版手机演示 **不依赖 QNN/Adreno**。请求的深度后端必须是 `ONNX_RUNTIME_CPU`；禁止静默回退到 fake 或 QNN。

手表数值只作实验参考，不作医学诊断，不得命名为 `clinical_ground_truth`。

### 范围内

- 复用本机已导出且哈希匹配的 `efficientphys_pure.onnx`（外部导入，不进 Git/APK）。
- Java Android BLE：手动扫描、选择设备、连接、订阅通知；意外断线最多自动重连 3 次。
- 与现有 10s 窗口 / 1s 步进的传统（及可选深度）结果对齐。
- UI 展示传统 BPM、深度 BPM、「广播心率」、对齐状态与误差；会话目录导出手表样本与对齐 CSV。

### 范围外

- QAIRT/QNN/Adreno 转换与推理。
- HyperRate / 华为健康云 / Android Intent 中转。
- 多手表并发、记住设备自动连接、自动连接第一个扫描结果。
- 浏览器 Web Bluetooth 或独立蓝牙微服务。

## 2. 方案选择

采用 **方案 A**：

- **模型**：直接使用 sibling 导出产物  
  `efficientphys-qnn-export/artifacts/model_export/efficientphys_pure/efficientphys_pure.onnx`  
  （SHA-256 `c1b321042db1335da70b0295cc84f653a2cfe90f75cff738b3045ea3c103257d`），通过 `adb run-as` 导入应用私有目录。不在本阶段重导 ONNX。
- **手表**：Java 层实现 Android BLE GATT（扫描/连接/通知/重连）；解析、有界历史、对齐规则与 Python `rppg_cabin.watch` 契约对齐；相机/rPPG 继续走现有 native pipeline。

不采用 NDK 直驱 BLE（平台支持弱、工期长），也不在本阶段重导模型（现有 ONNX 已通过 PyTorch↔ORT parity）。

Python 参考实现位于 `/Users/wangjie/Documents/keti/rPPG`：

- 设计：`docs/superpowers/specs/2026-07-14-huawei-watch-reference-heart-rate-design.md`
- 代码：`src/rppg_cabin/watch/*`
- 权重：`docs/code_repos/official/rPPG-Toolbox/final_model_release/PURE_EfficientPhys.pth`  
  （SHA-256 `e65a962e07bcac32a668e6acb9f8ed43cdb1b01cfb97262654dc5b55c0cf3a49`）

说明：口语中的「广播」指手表设置里的 **心率广播**（暴露标准 BLE HR 服务），不是 Android `BroadcastReceiver` Intent。

## 3. 架构

两条并行通道，互不阻塞：

```text
摄像头 → Camera2/YUV → OpenCV ROI
                       ├─ GREEN/POS/CHROM（native，已有）
                       └─ EfficientPhys ORT CPU（native，外部 ONNX）

手表 GT 5 Pro → Java BLE GATT (0x180D / 0x2A37)
              → 解析 → 180s 有界历史
              → 与 rPPG 窗口对齐 → UI / 会话 CSV
```

### 边界

| 单元 | 职责 | 依赖 |
|------|------|------|
| Camera / Pipeline | 采集、ROI、传统、深度 worker、会话落盘 | OpenCV、ORT、现有 C++ |
| WatchBleWorker（Java） | 扫描、连接、通知、重连、快照 | Android BLE API |
| HeartRateParser | 纯解析 0x2A37 | 无 BLE |
| WatchSampleStore | 线程安全有界历史与连接状态 | Parser 输出 |
| WatchAligner | 窗口对齐与误差统计 | Store + rPPG 窗口时钟 |
| MainActivity / Status | 权限、手表控件、合并状态展示 | Worker + native status |

相机 `onStop` 只释放相机；手表连接保持到用户断开或进程销毁。BLE 回调不得进入 Camera2 图像回调线程；深度推理保持 latest-only，不得阻塞采集。

## 4. 数据契约

### 4.1 EfficientPhys / ONNX Runtime CPU

| 项 | 值 |
|----|-----|
| 后端身份 | `ONNX_RUNTIME_CPU` |
| Runtime | onnxruntime-android 1.27.0（AAR SHA-256 `077dec5e2d821234c7dc0aba584bec8f999854b546c754cab93a90741c56fbeb`） |
| 模型路径 | 应用私有 `files/models/efficientphys_pure.onnx` |
| 模型 SHA-256 | `c1b321042db1335da70b0295cc84f653a2cfe90f75cff738b3045ea3c103257d` |
| 输入 | float32 `frames` `[181, 3, 72, 72]` TCHW |
| 输出 | float32 `pulse` `[180, 1]`，须有限 |
| 预处理 | 180 帧 RGB 72×72 → 全局 mean/std → 追加末帧 → TCHW |

未勾选深度或模型不可用时，传统与手表仍可运行。勾选深度但模型缺失/哈希错误时返回明确 `MODEL_LOAD_FAILED`，不得加载其他后端。

### 4.2 手表 BLE

| 项 | 值 |
|----|-----|
| 服务 UUID | `0000180d-0000-1000-8000-00805f9b34fb`（0x180D） |
| 特征 UUID | `00002a37-0000-1000-8000-00805f9b34fb`（0x2A37） |
| 交付方式 | GATT notifications |
| 扫描过滤 | 广播含 0x180D，或名称含 `huawei` / `heart` |
| 扫描超时 | 15 秒 |
| 时间戳 | 主机接收时刻的单调时钟（非手表内部采样时刻） |
| BPM 范围 | `[1, 300]` |
| 历史界 | 最近 180 秒（可另设样本数上限防失控） |
| 过期 | 超过 2 秒无新通知 → `STALE` |
| 重连 | 最多 3 次，间隔 0.25 / 0.5 / 1.0 秒；用户断开后禁止自动重连 |

连接状态机：

`DISCONNECTED → SCANNING → CONNECTING → STREAMING → STALE/RECONNECTING → STREAMING|ERROR`

样本字段至少包含：`received_monotonic_sec`、`bpm`、可选 `rr_intervals_sec`、`device_id`、`device_name`。

### 4.3 窗口对齐

与 Python 设计一致。对每个 rPPG 窗口 `[start_sec, end_sec]`（会话相对秒）：

1. 选取换算后时间戳落入窗口的手表样本。
2. 以有效 BPM 的中位数为 `watch_reference_bpm`。
3. 计算样本数、覆盖率、相邻样本最大间隔。
4. 同时满足时状态为 `ALIGNED`：样本 ≥3、覆盖率 ≥70%、最大间隔 ≤2s、手表 `STREAMING`、对应 rPPG 结果有效。
5. 仅 `ALIGNED` 窗口进入正式误差统计：  
   `signed_error_bpm = rppg_bpm - watch_reference_bpm`，`absolute_error_bpm = abs(signed_error_bpm)`。

对齐状态至少包含：`PENDING`、`ALIGNED`、`PARTIAL_COVERAGE`、`WATCH_STALE`、`RPPG_INVALID`、`DISCONNECTED`。

相机启动时记录 `session_start_monotonic`；手表接收时刻换算为会话相对秒。默认不做隐藏的手表算法延迟补偿；若后续增加偏移，必须为显式可配置项且默认 `0.0`。

## 5. UI、权限与导出

### UI

- 手表控件：扫描、设备列表、连接、断开。
- 指标：「广播心率」、手表连接状态、传统 BPM、深度 BPM（若启用）、对齐误差与覆盖率。
- 明确文案：实验参考设备，非医学诊断；时间戳为手机接收时刻。
- 现有相机 Start/Stop、传统方法选择、深度复选框保留。

### 权限

- Manifest：`BLUETOOTH_SCAN`、`BLUETOOTH_CONNECT`（API 31+）；`uses-feature android.hardware.bluetooth_le` 且 `required=false`。
- 旧 API 按系统要求兼容位置权限（仅扫描需要时）。
- 运行时蓝牙权限与相机权限分开申请与提示。
- 真机验收前应断开 HyperRate 或其他占用手表连接的应用。

### 导出

会话输出目录追加：

1. 原始手表样本 CSV：设备标识、接收单调时钟、会话相对时间、BPM、RR、解析状态。
2. 逐窗口对齐 CSV：窗口起止、方法、rPPG BPM、手表参考 BPM、误差、样本数、覆盖率、最大间隔、对齐状态。

汇总仅基于 `ALIGNED` 窗口（MAE/RMSE/平均偏差/有效窗口比例）。导出须注明设备类型为 HUAWEI WATCH GT 5 Pro（或实际连接名），并注明接收时间戳不是手表内部采集时间戳。

## 6. 错误处理

| 情况 | 行为 |
|------|------|
| 蓝牙权限拒绝 | `BLUETOOTH_PERMISSION_DENIED`，提示系统设置 |
| 扫描超时/空列表 | 可手动重试，不自动连接 |
| 缺少 Heart Rate Service | `INCOMPATIBLE_DEVICE`，不重连 |
| 畸形通知 | 丢弃并计数，保持连接 |
| 意外断线 | 最多重连 3 次；失败后 `ERROR`，等待用户手动连接 |
| 用户断开 | 取消重连，释放 GATT |
| ONNX 缺失/哈希不匹配 | `MODEL_LOAD_FAILED`；深度不可用；传统与手表可继续 |
| ORT 推理失败 | 深度结果无效并带具体原因；不回退 fake/QNN |
| 手表或相机单侧异常 | 互不传播；另一侧继续 |

## 7. 测试与验收

### 主机自动化

- 手表解析：8/16 位 BPM、RR、截断/非法包。
- Store：有界历史、`STALE`（2s）、快照不可变语义。
- 对齐：边界、样本数、覆盖率、间隔、中位数；无效窗口不进正式误差。
- Worker：重连上限、用户断开抑制、`INCOMPATIBLE_DEVICE` 不重连。
- 现有 CTest（含 android packaging / build script）不回退。
- 冻结向量：对已导出 ONNX 与 npy 验证 `[181,3,72,72] → [180,1]` 契约（向量与模型保持外部，不提交大文件进 Git）。

### 真机门禁（不可用模拟代替）

1. 安装 debug APK；导入并校验 ONNX SHA-256。
2. GT 5 Pro 开启心率广播 → 15s 内扫到 → 连接后数秒内显示「广播心率」。
3. 启动相机：传统与（可选）深度、手表并行；采集 FPS 不明显下降。
4. 合格窗口出现对齐误差与覆盖率；关闭手表广播后 ≤2s 进入 `STALE`。
5. 断线重连与手动断开行为符合设计；会话目录写出手表样本与对齐 CSV。

## 8. 完成标准

同时满足：

- 主机自动化测试通过，且不回归现有 Android 传统/ORT 契约。
- APK 含 `libonnxruntime.so` 与应用 native 库；不含模型/checkpoint；后端身份不为 QNN/fake。
- 真机完成：模型导入、传统心率、可选深度推理、手表扫描连接与对齐展示。
- 任何未完成的真机步骤必须明确标为待验证，不得用模拟结果宣称真机成功。

## 9. 参考

- Android NDK 运行时设计：`docs/superpowers/specs/2026-07-23-android-ndk-rppg-runtime-design.md`
- EfficientPhys ONNX 参考：`docs/superpowers/specs/2026-07-22-efficientphys-onnx-reference-design.md`
- 模型清单：`model_specs/efficientphys_pure.json`
- Python 手表设计：`/Users/wangjie/Documents/keti/rPPG/docs/superpowers/specs/2026-07-14-huawei-watch-reference-heart-rate-design.md`
- 台架操作手记：`ANDROID_NEXT_STEPS.md`
