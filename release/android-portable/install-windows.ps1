$ErrorActionPreference = "Stop"

$Package = "com.jagger.rppgbench"
$ApkSha256 = "9d60f36990894c8726657437c72af5c24a3c2cfcd53b42e5b991438e29dc625e"
$EfficientPhysSha256 = "c1b321042db1335da70b0295cc84f653a2cfe90f75cff738b3045ea3c103257d"
$TscanSha256 = "342a3c8033dda9ab154e85d5a4e2a876a6461648b7fcb27c46a7023e662bcc64"

$Apk = Join-Path $PSScriptRoot "app-debug.apk"
$EfficientPhys = Join-Path $PSScriptRoot "efficientphys_pure.onnx"
$Tscan = Join-Path $PSScriptRoot "ubfc_tscan_full_lr3e-5_Epoch10.onnx"
$RemoteEfficientPhys = "/data/local/tmp/rppg-efficientphys.onnx"
$RemoteTscan = "/data/local/tmp/rppg-tscan.onnx"

function Assert-FileHash([string]$Path, [string]$Expected) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "缺少文件: $Path"
    }
    $Actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne $Expected) {
        throw "SHA-256 不匹配: $Path ($Actual)"
    }
}

function Invoke-Adb([string[]]$Arguments) {
    & adb @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "adb 命令失败: adb $($Arguments -join ' ')"
    }
}

if (-not (Get-Command adb -ErrorAction SilentlyContinue)) {
    throw "未找到 adb。请先安装 Android Platform Tools 并加入 PATH。"
}
Assert-FileHash $Apk $ApkSha256
Assert-FileHash $EfficientPhys $EfficientPhysSha256
Assert-FileHash $Tscan $TscanSha256

Invoke-Adb -Arguments @("start-server")
$Devices = @(& adb devices | Select-String "\sdevice$")
if ($Devices.Count -ne 1) {
    throw "必须且只能连接一台已授权设备，当前检测到 $($Devices.Count) 台。"
}

try {
    Invoke-Adb -Arguments @("install", "-r", $Apk)
    Invoke-Adb -Arguments @("push", $EfficientPhys, $RemoteEfficientPhys)
    Invoke-Adb -Arguments @("push", $Tscan, $RemoteTscan)
    Invoke-Adb -Arguments @("shell", "run-as", $Package, "mkdir", "-p", "files/models")
    Invoke-Adb -Arguments @("shell", "run-as", $Package, "cp", $RemoteEfficientPhys,
        "files/models/efficientphys_pure.onnx")
    Invoke-Adb -Arguments @("shell", "run-as", $Package, "cp", $RemoteTscan,
        "files/models/ubfc_tscan_full_lr3e-5_Epoch10.onnx")

    $DeviceHashes = (& adb shell run-as $Package sha256sum `
        files/models/efficientphys_pure.onnx `
        files/models/ubfc_tscan_full_lr3e-5_Epoch10.onnx) -join "`n"
    if ($LASTEXITCODE -ne 0 -or
            -not $DeviceHashes.Contains($EfficientPhysSha256) -or
            -not $DeviceHashes.Contains($TscanSha256)) {
        throw "手机中的模型 SHA-256 校验失败。"
    }
    Write-Host $DeviceHashes
    Invoke-Adb -Arguments @("shell", "am", "start", "-n", "$Package/.MainActivity")
    Write-Host "安装完成：APK 与两份模型均已校验并启动。"
}
finally {
    & adb shell rm -f $RemoteEfficientPhys $RemoteTscan *> $null
}
