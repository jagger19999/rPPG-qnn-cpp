# Android Live HR UI 设计（分期）

## 1. 目标与范围

把当前「单页堆叠 JSON + 按钮」的 `MainActivity` 改成接近 Python Cabin Live 的可读界面：

- 三路实时心率并排：**传统 rPPG / 深度 EfficientPhys / 手表广播**
- 配置折叠，不再压住 Live 数字
- **脸的呈现**：第一期 ROI 缩略图；第二期再加实时预览 + 人脸框
- 使用仓库旁路资源 `logo.png` 作为 APK **启动图标（launcher icon）**

本设计只改 UI 呈现与为 UI 服务的最小 native/Java 数据通路；不改变已批准的 ORT CPU、BLE 手表契约与会话 CSV 语义。

### 范围内

- 布局 A：三卡并排 Live 心率
- 折叠：相机 / 手表 / 诊断
- 第一期：ROI 脸图（降频更新）
- 第二期：实时预览 + 人脸框（与 ROI 并存）
- Launcher：由 `logo.png`（316×300 PNG）生成各密度 mipmap / adaptive icon

### 范围外

- Streamlit 波形大图、趋势图、完整表格导出 UI
- 医学诊断文案或把广播心率标成 clinical ground truth
- QNN/Adreno
- 第一期不做全幅预览与人脸框叠加

### 分期（已确认）

| 期 | 交付 |
|----|------|
| **一期** | 三卡 Live + 对齐行 + ROI 脸图 + 折叠配置/诊断 + launcher logo |
| **二期** | 实时预览 + 人脸框；ROI 保留；预览失败可降级为仅 ROI |

## 2. 视觉与信息架构（一期）

竖屏从上到下：

1. **标题行**：品牌/标题 + 「实验参考 · 非医学诊断」；可选小尺寸 logo 装饰（与 launcher 同源）
2. **三路 Live 心率卡**（等权大号 BPM）
3. **对齐摘要一行**：`ALIGNED` / `|误差|` / 覆盖率；未对齐为 `--`
4. **脸 ROI 区**：约正方形缩略图；无脸占位「未检测到人脸」
5. **折叠「相机」**：方法、深度勾选、List / Start / Stop、FPS 摘要
6. **折叠「手表」**：扫描 / 设备 / 连接 / 断开
7. **折叠「诊断」**（默认收起）：原始 status JSON

运行中「相机」「手表」默认收起；首次进入可展开相机一次。
Stop / `onStop` 只停相机；手表不断开（与现有 BLE 设计一致）。

### Launcher 图标

- **源文件**：`/Users/wangjie/Documents/keti/rPPG-qnn-cpp/logo.png`（实现时复制进 Android 资源并生成 mipmap）
- **用途**：`android:icon` / `android:roundIcon`（adaptive icon 前景裁切时保留圆形脸+波形主体）
- **实现**：由源 PNG 生成 `mipmap-*` 或 `mipmap-anydpi-v26` adaptive；不把未缩放原图直接当唯一密度资源
- 安装后桌面应显示该红黑科技风图标，而非系统默认机器人

## 3. 三卡 Live 规则

刷新约 **1s**（可与现有 status poll 合并）。

| 卡 | 主数字 | 副文案 | 显示 `--` 当 |
|----|--------|--------|----------------|
| 传统 rPPG | `bpm` | 方法名 + 可信/不可用 | 未 Start 或 `heart_rate_valid=false` |
| 深度 EfficientPhys | `deep_bpm` | `ORT CPU` + `inference_ms` | 未勾选、模型失败或 `deep_result_valid=false`（附短原因） |
| 广播心率 | 最新 watch BPM | 连接状态（STREAMING/STALE/…） | 未连接或无样本 |

对齐行：有 `WatchAligner` 结果时显示状态与误差/覆盖；否则 `--`。
禁止把整段 JSON 放进三卡区域。

文案对齐 Python 语义：广播为 **实验参考**；时间戳为手机接收时刻。

## 4. ROI 脸图（一期数据通路）

现状：native 仅有 `face_found`，无裁切图回传。

一期约定：

1. 在现有 ROI/处理路径上，**降频 2–5 FPS** 将当前脸部区域缩放到约 **160×160**
2. 以 JPEG `byte[]` 经 JNI 回传，或写入会话临时文件由 Java 读取（优先 JNI `byte[]` 以免文件抖动）
3. Java `ImageView` 显示最新帧；`face_found=false` 时占位文案
4. ROI 通路失败不得中断采集与心率计算

一期 **不上** 全幅预览、**不画**人脸框。

## 5. 二期：预览 + 人脸框

- **优先**：Camera2 双输出（预览 Surface + 现有 ImageReader）
- **降级**：分析帧 5–10 FPS 推 Java 画预览
- Native 暴露 `face_rect`（归一化或相对预览坐标系），UI 叠加矩形；无脸不画
- ROI 小图与预览并存
- 预览失败可降级为「仅 ROI」，不得拖垮采集

## 6. 错误与空态

| 情况 | UI |
|------|-----|
| 相机权限拒绝 | 诊断/状态条明确 `CAMERA_PERMISSION_DENIED`；三卡保持 `--` |
| 蓝牙权限拒绝 | 广播卡/手表面板 `BLUETOOTH_PERMISSION_DENIED` |
| 模型缺失 | 深度卡短错误；传统与广播可继续 |
| 手表 STALE | 广播卡保留最后数字或 `--`（与 store 快照一致）+ STALE 标签 |
| 无脸 | ROI 占位；传统可能不可用 |

## 7. 测试与验收

### 主机

- 现有 CTest / packaging / JVM unit tests / `build_android.sh` 不回退
- 新增或扩展：ROI JNI/契约的最小单测（若纯 Android View 难测，则测序列化/降频工具函数）
- APK 含自适应或各密度图标资源；`aapt`/`aapt2` dump 可见 `android:icon`

### 真机（一期）

1. 桌面图标为提供的 logo（非默认图标）
2. 打开 App：最上三路大号 BPM，非整屏 JSON
3. Start 后有脸时 ROI 更新；无脸占位
4. 手表连接后广播卡有数；合格窗口对齐行有误差/覆盖
5. 配置在折叠内；诊断默认收起
6. Stop/切后台：相机停、手表不断

### 真机（二期）

预览可见；有脸时框跟随；ROI 仍在；采集 FPS 不明显塌陷。

## 8. 实现边界提示（供后续 plan）

- UI：`MainActivity` 改为可滚动层级布局（或 `activity_main.xml`）；拆出三卡/折叠辅助类以免单文件过大
- Native：扩展 status 或独立 `nativeGetRoiJpeg` / face rect API；避免在 Camera2 回调线程做重 JPEG
- 资源：从 `logo.png` 生成 mipmap；更新 `AndroidManifest` `android:icon` / `roundIcon`
- 包装门禁：更新 `tests/test_android_packaging.sh` 白名单以包含新布局/资源文件

## 9. 决策记录

- 布局：**A 三卡并排**
- 脸：**都要**（ROI + 预览框），**分期**执行
- Launcher：`/Users/wangjie/Documents/keti/rPPG-qnn-cpp/logo.png`

## 10. Phase 2 增补（2026-07-25 真机反馈）

真机一期发现：默认后置、无法选前置、measured FPS 过低，导致 rPPG/深度结果不可信。二期必须同时交付：

1. **前置摄像头**
   - `list_cameras` JSON 包含每路 `id` + `facing`（`front`/`back`/`external`/`unknown`）
   - UI 摄像头下拉框；**默认选择 front**；切换后需 Stop/Start 生效
2. **采集硬目标 30 FPS**
   - CaptureRequest 设置 `AE_TARGET_FPS_RANGE` 为可用区间中覆盖 30 的最优（优先精确 `[30,30]`）
   - 状态常显 `measured_fps`；验收：**连续稳定运行时 measured_fps ≥ 29.0**（允许短暂抖动），否则标为未通过
   - 排查并削减拖垮采集的路径（ROI JPEG 不得阻塞 ImageReader 回调线程；预览双输出不得把分析降到远低于 30）
3. **实时预览 + 人脸框**（与 ROI 小图并存）
   - `TextureView` 必须在 `nativeStart` **之前**可见并拿到 Surface；`GONE` 的 TextureView 无 Surface，会导致 `preview_enabled:false` /「预览不可用」
   - 运行中禁止再设 preview Surface（须 Stop 后重建会话）
4. **界面清晰度**：三卡更大字号/对比；摄像头与 FPS 放在首屏可见区

