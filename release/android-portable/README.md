# rPPG Android 便携安装包

版本：`0.5.1-traditional-diagnostics`（versionCode 4）

这个目录应包含：

- `app-debug.apk`
- `efficientphys_pure.onnx`
- `ubfc_tscan_full_lr3e-5_Epoch10.onnx`
- `install-macos.sh`
- `install-windows.ps1`
- `install-windows.bat`
- `SHA256SUMS.txt`

## 使用条件

1. 新电脑已安装 Android Platform Tools，终端执行 `adb version` 有输出。
2. 手机已开启开发者选项和 USB 调试。
3. USB 连接后，手机上允许这台电脑进行调试。
4. 安装时只连接一台 Android 设备。

## macOS

解压后在该目录打开终端：

```bash
chmod +x install-macos.sh
./install-macos.sh
```

## Windows

解压后双击 `install-windows.bat`。如果窗口一闪而过，请在 PowerShell 中执行：

```powershell
.\install-windows.ps1
```

安装器会校验本地文件、覆盖安装 APK、把两份 ONNX 权重导入应用私有目录、再次校验手机端权重并启动应用。APK 是可调试演示包，不用于应用商店正式分发。
