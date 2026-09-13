<#
.SYNOPSIS
    按冻结的运行清单顺序执行 ramen_teacher_collect，并留存每次运行的完整证据。

.DESCRIPTION
    本驱动只做三件事：按清单启动采集、共享一个绝对墙钟截止时刻、把命令/参数/
    stdout/stderr/退出码/耗时逐条落盘。它**不**推导配方、不改参数、不算任何哈希。

    行为约定（与 scripts/bench/run_paired_game_bench.ps1 同一套纪律）：
      * 输出目录必须为空或不存在，拒绝覆盖既有证据；
      * 每条 run 的子目录名在**任何一条 run 启动之前**全部校验，避免先跑后拒；
      * 参数用 ProcessStartInfo.ArgumentList 逐个传入，由 .NET 负责转义；
        另存 args.json 作为参数边界的权威记录，command.txt 只是人读形态；
      * stdout/stderr 边跑边落盘（FileStream bufferSize=1，FileShare.ReadWrite），
        驱动被中断也不丢已产生的日志；
      * 等待退出 → 先存退出码 → 有界等待日志收尾；
      * 任一 run 非零退出即停止后续 run（除非 -ContinueOnError）；
      * 超过 -BudgetSeconds 即不再启动新的 run，正在跑的那条**不**中途杀掉
        （采集是逐根落盘 + manifest 续跑的，让它自然跑完比留半条更干净）。

.PARAMETER RunsJson
    运行清单，例如 scripts/collect/price0914_runs_local.json。

.PARAMETER OutDir
    证据目录，例如 logs/price0914_local。

.PARAMETER Exe
    ramen_teacher_collect 可执行文件路径。

.PARAMETER BudgetSeconds
    全部 run 合计的墙钟预算（秒）。达到即不再启动新 run。

.PARAMETER Threads
    RAYON_NUM_THREADS 的取值；必须与清单里的 threads 一致。

.PARAMETER ContinueOnError
    某条 run 非零退出时继续跑后面的 run（默认不继续）。

.EXAMPLE
    pwsh -NoProfile -File scripts/collect/run_price_batch.ps1 `
        -RunsJson scripts/collect/price0914_runs_local.json `
        -OutDir logs/price0914_local `
        -Exe target/release/ramen_teacher_collect `
        -BudgetSeconds 900 -Threads 16
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)][string]$RunsJson,
    [Parameter(Mandatory = $false)][string]$OutDir,
    [Parameter(Mandatory = $false)][string]$Exe,
    [Parameter(Mandatory = $false)][int]$BudgetSeconds = 900,
    [Parameter(Mandatory = $false)][int]$Threads = 16,
    [switch]$ContinueOnError
)

$ErrorActionPreference = 'Stop'

function Show-Usage {
    Write-Host 'usage: run_price_batch.ps1 -RunsJson <path> -OutDir <dir> -Exe <path> [-BudgetSeconds N] [-Threads N] [-ContinueOnError]'
}

if (-not $RunsJson -or -not $OutDir -or -not $Exe) {
    Show-Usage
    exit 2
}
if (-not (Test-Path -LiteralPath $RunsJson)) {
    Write-Host "运行清单不存在: $RunsJson"
    exit 2
}
if (-not (Test-Path -LiteralPath $Exe)) {
    Write-Host "可执行文件不存在: $Exe"
    exit 2
}
if ($BudgetSeconds -le 0) { Write-Host 'BudgetSeconds 必须为正'; exit 2 }
if ($Threads -le 0) { Write-Host 'Threads 必须为正'; exit 2 }

$plan = Get-Content -LiteralPath $RunsJson -Raw -Encoding utf8 | ConvertFrom-Json
if (-not $plan.runs -or $plan.runs.Count -eq 0) { Write-Host '清单里没有 runs'; exit 2 }
if ($plan.threads -ne $Threads) {
    Write-Host "清单要求 threads=$($plan.threads)，本次传入 $Threads —— 拒绝启动，避免两台机器口径不同"
    exit 2
}

# ---- 开跑前一次性校验全部 run 名，避免先跑后拒 ----------------------------
$seen = @{}
foreach ($r in $plan.runs) {
    if (-not $r.name) { Write-Host 'run 缺 name'; exit 2 }
    if ($r.name -notmatch '^[A-Za-z0-9._+-]+$') { Write-Host "run 名非法: $($r.name)"; exit 2 }
    if ($seen.ContainsKey($r.name)) { Write-Host "run 名重复: $($r.name)"; exit 2 }
    $seen[$r.name] = $true
    if (-not $r.dir) { Write-Host "run $($r.name) 缺 dir"; exit 2 }
    if (Test-Path -LiteralPath $r.dir) {
        Write-Host "采集输出目录已存在，拒绝覆盖: $($r.dir)"
        exit 2
    }
}

if (Test-Path -LiteralPath $OutDir) {
    if ((Get-ChildItem -LiteralPath $OutDir -Force | Measure-Object).Count -gt 0) {
        Write-Host "证据目录非空，拒绝覆盖: $OutDir"
        exit 2
    }
} else {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
}
$outFull = (Resolve-Path -LiteralPath $OutDir).Path

# 每条 run 的证据子目录必须解析成 OutDir 的直接子目录
foreach ($r in $plan.runs) {
    $child = Join-Path $outFull $r.name
    $parent = Split-Path -Path $child -Parent
    if ($parent -ne $outFull) { Write-Host "run 名逃出证据目录: $($r.name)"; exit 2 }
}

<#
.SYNOPSIS
    启动一条采集并等待它结束，返回该次运行的记录对象。
#>
function Invoke-CollectRun {
    param(
        [Parameter(Mandatory = $true)]$Run,
        [Parameter(Mandatory = $true)][string]$ExePath,
        [Parameter(Mandatory = $true)][string]$EvidenceDir,
        [Parameter(Mandatory = $true)]$Plan
    )

    New-Item -ItemType Directory -Path $EvidenceDir | Out-Null
    $argList = @(
        '--space-version', $Plan.space_version,
        '--start', "$($Run.index_start)",
        '--count', "$($Run.count)",
        '--search-n', "$($Plan.search_n)",
        '--shard-size', "$($Plan.shard_size)",
        '--output-dir', $Run.dir,
        '--region-quota-permille', $Run.quota_y2_y3,
        '--region-quota-permille-y1', "$($Run.quota_y1)",
        '--rollin', $Plan.rollin,
        '--model', $Plan.model,
        '--model-id', 'ens_R4_g123_30k'
    )
    $argList | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $EvidenceDir 'args.json') -Encoding utf8
    "$ExePath $($argList -join ' ')" | Set-Content -LiteralPath (Join-Path $EvidenceDir 'command.txt') -Encoding utf8

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = (Resolve-Path -LiteralPath $ExePath).Path
    foreach ($a in $argList) { $psi.ArgumentList.Add($a) }
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.EnvironmentVariables['RAYON_NUM_THREADS'] = "$($Plan.threads)"

    $outPath = Join-Path $EvidenceDir 'run.stdout.txt'
    $errPath = Join-Path $EvidenceDir 'run.stderr.txt'
    # bufferSize=1 + FileShare.ReadWrite：边跑边落盘，运行当中即可读
    $outFs = [System.IO.FileStream]::new($outPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite, 1, $false)
    $errFs = [System.IO.FileStream]::new($errPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite, 1, $false)

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $proc = [System.Diagnostics.Process]::new()
    $proc.StartInfo = $psi
    [void]$proc.Start()
    $copyOut = $proc.StandardOutput.BaseStream.CopyToAsync($outFs)
    $copyErr = $proc.StandardError.BaseStream.CopyToAsync($errFs)
    # 峰值内存必须在进程**还活着**时读：退出后 .NET 的这两个计数器会失效返回空。
    # 它们本身就是运行期峰值，故边等边取、留最后一次有效读数即可；
    # 采样间隔 200 ms，最后一次读数与真实退出点的间隔不超过一个采样周期。
    $peakWs = 0
    $peakPaged = 0
    while (-not $proc.WaitForExit(200)) {
        try {
            $proc.Refresh()
            $ws = $proc.PeakWorkingSet64
            $pg = $proc.PeakPagedMemorySize64
            if ($null -ne $ws -and $ws -gt $peakWs) { $peakWs = $ws }
            if ($null -ne $pg -and $pg -gt $peakPaged) { $peakPaged = $pg }
        } catch {
            # 进程刚好在两次读之间退出：保留已取到的读数，不当作失败
        }
    }
    $sw.Stop()
    $code = $proc.ExitCode
    "$code" | Set-Content -LiteralPath (Join-Path $EvidenceDir 'run.exitcode.txt') -Encoding utf8
    [void]$copyOut.Wait(5000)
    [void]$copyErr.Wait(5000)
    $outFs.Dispose()
    $errFs.Dispose()
    $proc.Dispose()

    return [pscustomobject]@{
        Name              = $Run.name
        Layer             = $Run.layer
        ExitCode          = $code
        ElapsedMs         = $sw.ElapsedMilliseconds
        IndexStart        = $Run.index_start
        Count             = $Run.count
        Dir               = $Run.dir
        PeakWorkingSet64  = $peakWs
        PeakPagedMemory64 = $peakPaged
    }
}

$deadline = (Get-Date).AddSeconds($BudgetSeconds)
Write-Host "运行清单 : $RunsJson"
Write-Host "证据目录 : $outFull"
Write-Host "线程     : $Threads"
Write-Host "预算     : $BudgetSeconds s，截止 $($deadline.ToString('o'))"
Write-Host ''

$results = @()
$skipped = @()
$failed = $false
foreach ($r in $plan.runs) {
    if ((Get-Date) -ge $deadline) {
        Write-Host "预算耗尽，未启动: $($r.name)"
        $skipped += $r.name
        continue
    }
    if ($failed -and -not $ContinueOnError) {
        Write-Host "前一条失败，未启动: $($r.name)"
        $skipped += $r.name
        continue
    }
    Write-Host "启动 $($r.name) (layer=$($r.layer) index=$($r.index_start) count=$($r.count))"
    $res = Invoke-CollectRun -Run $r -ExePath $Exe -EvidenceDir (Join-Path $outFull $r.name) -Plan $plan
    Write-Host "  退出码 $($res.ExitCode)，耗时 $([math]::Round($res.ElapsedMs / 1000.0, 2)) s"
    $results += $res
    if ($res.ExitCode -ne 0) { $failed = $true }
}

$summary = [pscustomobject]@{
    runs_json       = $RunsJson
    threads         = $Threads
    budget_seconds  = $BudgetSeconds
    started_runs    = $results.Count
    skipped_runs    = $skipped
    total_wall_ms   = ($results | Measure-Object -Property ElapsedMs -Sum).Sum
    results         = $results
}
$summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $outFull 'summary.json') -Encoding utf8

Write-Host ''
Write-Host "已启动 $($results.Count) 条，未启动 $($skipped.Count) 条"
Write-Host "合计进程墙钟 $([math]::Round(($results | Measure-Object -Property ElapsedMs -Sum).Sum / 1000.0, 2)) s"
Write-Host "汇总: $(Join-Path $outFull 'summary.json')"

if ($failed) { exit 1 }
if ($skipped.Count -gt 0) { exit 3 }
exit 0
