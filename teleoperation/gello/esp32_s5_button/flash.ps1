# 编译并烧录到 COM10。在本目录执行: powershell -ExecutionPolicy Bypass -File .\flash.ps1
$ErrorActionPreference = "Stop"
$Port = "COM10"
$SketchDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Fqbn = "esp32:esp32:esp32c3:CDCOnBoot=cdc,FlashMode=dio,UploadSpeed=921600"
$cli = Join-Path $env:LOCALAPPDATA "arduino-cli\arduino-cli.exe"
if (-not (Test-Path $cli)) {
    $cmd = Get-Command arduino-cli -ErrorAction SilentlyContinue
    if ($cmd) { $cli = $cmd.Source } else { Write-Error "未找到 arduino-cli" }
}
& $cli compile --fqbn $Fqbn $SketchDir
& $cli upload -p $Port --fqbn $Fqbn $SketchDir
Write-Host "烧录完成。串口监视器: & `"$cli`" monitor -p $Port -c baudrate=115200"
