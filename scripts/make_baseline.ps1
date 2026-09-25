#encoding: utf-8
# ============================================================
#  UmaAI 发行包完整性基线生成脚本
# ------------------------------------------------------------
#  用途: 记录各 exe 的 SHA256、大小、签名信息, 生成基线文件。
#  该脚本仅由打包/发布方使用, 不随用户验证流程运行。
#
#  用法:
#    powershell -ExecutionPolicy Bypass -File .\make_baseline.ps1
#  可选参数:
#    -Files "umaai.exe,UmamusumeResponseAnalyzer.exe"  指定要建档的文件(默认 umaai.exe)
#
#  生成文件: integrity_manifest.json(与脚本同目录)
# ============================================================

param(
    [string]$Manifest = "integrity_manifest.json",
    [string[]]$Files  = @("umaai.exe")
)

$ErrorActionPreference = "Stop"

# 发行包根目录 = gamedata 的上一级(脚本约定放在 gamedata/ 下)
$PackageRoot = Split-Path $PSScriptRoot -Parent

function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash
}

# 定位被检查的 exe: 优先发行包根目录, 其次是脚本所在目录
function Resolve-ExePath {
    param([string]$Name)
    $candidates = @(
        (Join-Path $PackageRoot $Name),
        (Join-Path $PSScriptRoot $Name)
    )
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c) { return $c }
    }
    return $candidates[0]
}

function New-SignatureInfo {
    param([string]$Path)
    $sig = Get-AuthenticodeSignature -FilePath $Path
    $info = [PSCustomObject]@{
        status       = $sig.Status.ToString()
        thumbprint   = ""
        subject      = ""
        not_before   = $null
        not_after    = $null
        time_stamped = ($null -ne $sig.TimeStamperCertificate)
    }
    if ($null -ne $sig.SignerCertificate) {
        $info.thumbprint = $sig.SignerCertificate.Thumbprint
        $info.subject    = $sig.SignerCertificate.Subject
        $info.not_before = $sig.SignerCertificate.NotBefore
        $info.not_after  = $sig.SignerCertificate.NotAfter
    }
    return $info
}

$manifestPath = Join-Path $PSScriptRoot $Manifest
Write-Host "> 生成完整性基线: $manifestPath" -ForegroundColor Cyan

$filesNode = @{}
foreach ($f in $Files) {
    $path = Resolve-ExePath $f
    if (-not (Test-Path -LiteralPath $path)) {
        Write-Host "  [跳过] 找不到文件: $path" -ForegroundColor Yellow
        continue
    }
    $sigInfo = New-SignatureInfo $path
    $filesNode[$f] = [PSCustomObject]@{
        sha256       = Get-Sha256 $path
        size         = (Get-Item -LiteralPath $path).Length
        status       = $sigInfo.status
        thumbprint   = $sigInfo.thumbprint
        subject      = $sigInfo.subject
        not_before   = $sigInfo.not_before
        not_after    = $sigInfo.not_after
        time_stamped = $sigInfo.time_stamped
    }
    $stamp = if ($sigInfo.time_stamped) { "已时间戳" } else { "无时间戳" }
    Write-Host ("  [OK] " + $f + "  | 签名者: " + $sigInfo.subject + "  | " + $stamp) -ForegroundColor Green
}

[PSCustomObject]@{
    created_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    files      = $filesNode
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Write-Host "> 基线已写入: $manifestPath" -ForegroundColor Green