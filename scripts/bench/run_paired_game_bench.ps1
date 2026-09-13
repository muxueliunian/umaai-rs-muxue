<#
.SYNOPSIS
  两臂配对整局 benchmark 的最小可靠驱动（PowerShell 7 / pwsh）。

.DESCRIPTION
  顺序跑 A、B 两臂 `ramen_client_game_bench`，两臂共享**一个绝对截止时刻**，
  逐臂留存确切命令、stdout、stderr、退出码与耗时。

  设计约束（都是过去踩过的坑）：

  - **共享预算是绝对截止时刻**，不是「每臂各给 N 秒」：B 拿到的是
    `截止时刻 − 现在`，A 的对局时间、两臂之间的间隔、B 自己的初始化与模型加载
    全部计入。截止时刻在 A 启动前就定死。
  - **函数不打印裸字符串**：PowerShell 会把裸字符串并进函数返回值，
    曾导致返回值变成数组、`$budget - $elapsed` 抛 `op_Subtraction`。
    本脚本一律用 `Write-Host` / `Log` 打印，函数只返回一个 `[pscustomobject]`。
  - **`WaitForExit` 之后立刻保存 `ExitCode`**，两臂都保存，不留到事后补。
  - **非零退出码即失败**：默认不启动下一臂（`-ContinueOnFailure` 可放开）。
  - **超时只终止本驱动启动的进程**（按进程对象 `Kill($true)` 连同子进程），
    绝不按进程名全杀。
  - **拒绝覆盖既有证据**：输出目录已存在且非空时直接退出；两臂子目录名在 A 启动**之前**
    就校验：必须互不相同、必须是本次输出目录的直接子目录（不含路径分隔符与 `..`）、
    且各自不得已有内容。否则 B 会先覆盖 A 的 `command.txt` / stdout，
    等 benchmark 自己因 CSV 已存在而报错时，A 的证据已经没了。
    名字另外拒绝**尾随点**与 **Windows 保留设备名**（`CON` / `PRN` / `AUX` /
    `NUL` / `COM1-9` / `LPT1-9`）：`A` 与 `A.` 字符串不等，但普通 Windows 路径会
    解析到同一个目录，光靠 `GetFullPath` + 父目录检查拦不住。
    （代码里一并写了尾随空格，但白名单本就不允许空格，那一支是放宽白名单时的兜底。）
  - **参数按数组传递**，用 `ProcessStartInfo.ArgumentList` 逐个塞，不自己拼命令行：
    含空格的模型路径或输出路径不会被拆开。`command.args.json` 存结构化参数数组
    （权威），`command.txt` 只是给人看的近似形态。
  - **stdout / stderr 边跑边落盘**：两条流各自 `CopyToAsync` 到一个**不带内部缓冲**
    （`bufferSize=1`）、`FileShare.ReadWrite` 的 `FileStream`，运行当中就能读。
    驱动自己被中断时，已经产生的日志留在文件里，不会随内存一起丢。
    顺序是「等待退出 → **先存退出码** → 有界等待日志收尾」，退出码不排在日志之后。
  - **`-Arm?ExtraArgs` 不得覆盖共享字段**：正式模式下出现 `--seed` / `--cards` /
    `--search-n` / `--out` 等受保护参数即拒绝启动。benchmark 按「后出现的覆盖先出现的」
    解析，放任下去会出现「驱动打印相同条件、两臂实际不同」。
  - 不计算任何哈希或指纹；不改配置文件；不自动开跑正式任务
    （必需参数缺失时只打印用法并退出 2）。

.PARAMETER Exe
  `ramen_client_game_bench` 可执行文件路径（自行 Release 构建；本脚本不调 cargo）。

.PARAMETER OutDir
  本次运行的**全新**证据目录。已存在且非空时拒绝运行。

.PARAMETER BudgetSecs
  两臂**共享**的硬预算（秒）。从 A 启动前起表。

.PARAMETER Runs, Seed, RunOffset
  世界号段：`run_idx ∈ [RunOffset, RunOffset + Runs)`，基种子 `Seed`。两臂相同。

.PARAMETER Cards, ExtraCount, Threads, SearchN
  两臂共享的配置覆盖，原样转给 benchmark（只改它的内存配置，不写 game_config.toml）。

.PARAMETER ArmATrainer, ArmAModel, ArmBTrainer, ArmBModel, SpecialMode
  两臂的策略与模型。需要模型的策略缺 `-Arm?Model` 时由 benchmark 自己报错。

.PARAMETER ArmAName, ArmBName
  两臂在 `OutDir` 下的子目录名。

.PARAMETER ContinueOnFailure
  A 臂失败（非零退出码或超时）时仍然启动 B。默认关闭。

.PARAMETER ArmAExtraArgs, ArmBExtraArgs
  追加给对应臂的额外参数。

.PARAMETER NoBenchArgs
  **仅供驱动自检**：不生成任何 benchmark 参数，只用 `-Arm?ExtraArgs`，
  此时 `Runs/Seed/RunOffset/Cards/ExtraCount/Threads/SearchN` 都不再必需。
  用它配合假进程验证「成功 / A 失败不启动 B / 共享超时 / 拒绝覆盖 / 含空格参数 /
  同名臂 / 受保护参数」，不必启动真正的 benchmark。
  ❗此模式下**不做**受保护参数检查（本来就没有共享字段可保护）。

.EXAMPLE
  pwsh -NoProfile -File scripts/bench/run_paired_game_bench.ps1 `
    -Exe target/release/ramen_client_game_bench.exe `
    -OutDir logs/my_pair_run -BudgetSecs 2700 `
    -Runs 20 -Seed 61444 -RunOffset 170000 `
    -Cards 303124,303114,303044,303004,302894,303054 `
    -ExtraCount 0,20,20,40,40,40 -Threads 16 -SearchN 1024 `
    -ArmATrainer mcts `
    -ArmBTrainer mcts+region_nn -ArmBModel saved_models/arms/ens_R4_g123.onnx
#>
[CmdletBinding()]
param(
    [string]$Exe,
    [string]$OutDir,
    [int]$BudgetSecs = 0,

    [int]$Runs = 0,
    [string]$Seed,
    [string]$RunOffset,
    [string]$Cards,
    [string]$ExtraCount,
    [int]$Threads = 0,
    [int]$SearchN = 0,

    [string]$ArmATrainer = 'mcts',
    [string]$ArmAModel,
    [string]$ArmBTrainer,
    [string]$ArmBModel,
    [string]$SpecialMode = 'canonical',

    [string]$ArmAName = 'A',
    [string]$ArmBName = 'B',

    [switch]$ContinueOnFailure,
    [string[]]$ArmAExtraArgs = @(),
    [string[]]$ArmBExtraArgs = @(),
    [switch]$NoBenchArgs
)

$ErrorActionPreference = 'Stop'

$script:LogPath = $null

# 打印一行，同时追加到 driver.log。只写主机与文件，不产生返回值。
function Log {
    param([string]$Text)
    $stamp = (Get-Date).ToString('HH:mm:ss')
    Write-Host "[$stamp] $Text"
    if ($script:LogPath) {
        Add-Content -LiteralPath $script:LogPath -Value "[$stamp] $Text" -Encoding utf8
    }
}

# 参数不全时只打印用法并退出 2，绝不用默认值开跑正式任务。
function Show-UsageAndExit {
    param([string]$Why)
    Write-Host "用法错误：$Why"
    Write-Host ''
    Write-Host '最小必需参数：-Exe -OutDir -BudgetSecs -ArmBTrainer'
    Write-Host '正式跑还需要：-Runs -Seed -RunOffset -Cards -ExtraCount -Threads -SearchN'
    Write-Host '自检可加 -NoBenchArgs，用 -ArmAExtraArgs / -ArmBExtraArgs 直接给假进程参数。'
    Write-Host ''
    Write-Host 'Get-Help scripts/bench/run_paired_game_bench.ps1 -Detailed  # 完整说明与示例'
    exit 2
}

$missing = @()
if (-not $Exe) { $missing += '-Exe' }
if (-not $OutDir) { $missing += '-OutDir' }
if ($BudgetSecs -le 0) { $missing += '-BudgetSecs（正整数秒）' }
if (-not $ArmBTrainer) { $missing += '-ArmBTrainer' }
if (-not $NoBenchArgs) {
    if ($Runs -le 0) { $missing += '-Runs' }
    if (-not $Seed) { $missing += '-Seed' }
    if (-not $RunOffset) { $missing += '-RunOffset' }
    if (-not $Cards) { $missing += '-Cards' }
    if (-not $ExtraCount) { $missing += '-ExtraCount' }
    if ($Threads -le 0) { $missing += '-Threads' }
    if ($SearchN -le 0) { $missing += '-SearchN' }
}
if ($missing.Count -gt 0) {
    Show-UsageAndExit ("缺少参数 " + ($missing -join '、'))
}

if (-not (Test-Path -LiteralPath $Exe -PathType Leaf)) {
    Write-Host "找不到可执行文件：$Exe（请先自行 Release 构建）"
    exit 2
}
$exePath = (Resolve-Path -LiteralPath $Exe).Path

# 拒绝覆盖既有证据：目录已存在且非空就停
if (Test-Path -LiteralPath $OutDir) {
    $existing = @(Get-ChildItem -LiteralPath $OutDir -Force)
    if ($existing.Count -gt 0) {
        Write-Host "输出目录已存在且非空，拒绝覆盖既有证据：$OutDir"
        exit 2
    }
} else {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
}
$outRoot = (Resolve-Path -LiteralPath $OutDir).Path
$script:LogPath = Join-Path $outRoot 'driver.log'

# 受保护的共享参数：只能由驱动按两臂相同的方式生成，不许经 -Arm?ExtraArgs 再给一次。
# benchmark 是「后出现的覆盖先出现的」，放任下去两臂会在驱动毫不知情的情况下跑成不同条件。
$script:ProtectedArgs = @(
    '--runs', '--seed', '--run-offset', '--cards', '--extra-count',
    '--threads', '--search-n', '--out', '--trainer', '--model', '--special-mode'
)

# 校验并解析一臂的输出目录：必须是 $outRoot 的直接子目录，且不得已有内容。
# 返回绝对路径字符串；不打印任何东西（失败直接 exit）。
function Resolve-ArmDir {
    param([string]$Name, [string]$Which)
    if ([string]::IsNullOrWhiteSpace($Name)) {
        Write-Host "$Which 的目录名为空"
        exit 2
    }
    if ($Name -notmatch '^[A-Za-z0-9._+-]+$' -or $Name -eq '.' -or $Name -eq '..') {
        Write-Host "$Which 的目录名 '$Name' 非法：只允许字母、数字、. _ + -，且不能是 . 或 .."
        exit 2
    }
    # 尾随点：Windows 会把 'A.' 解析成 'A'，两臂名字串不等却指向同一个目录。
    # 空格分支在当前白名单下不可达（上面的 ^[A-Za-z0-9._+-]+$ 已先拒绝空格），
    # 留着是为了将来放宽白名单时不至于漏掉——这是兜底，不是当前生效的唯一屏障。
    if ($Name -match '[. ]$') {
        Write-Host "$Which 的目录名 '$Name' 非法：不允许以点或空格结尾（Windows 会把它解析成去掉尾随点的同一个目录）"
        exit 2
    }
    # Windows 保留设备名：连同扩展名形式一并拒绝
    $stem = $Name.Split('.')[0]
    if ($stem -match '^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$') {
        Write-Host "$Which 的目录名 '$Name' 非法：'$stem' 是 Windows 保留设备名"
        exit 2
    }
    $full = [System.IO.Path]::GetFullPath((Join-Path $outRoot $Name))
    $parent = [System.IO.Path]::GetFullPath((Split-Path -Parent $full))
    if ($parent.TrimEnd('\', '/') -ne $outRoot.TrimEnd('\', '/')) {
        Write-Host "$Which 的目录 '$Name' 解析后不在本次输出目录内：$full"
        exit 2
    }
    if (Test-Path -LiteralPath $full) {
        $items = @(Get-ChildItem -LiteralPath $full -Force)
        if ($items.Count -gt 0) {
            Write-Host "$Which 的目录已存在且非空，拒绝复用既有证据：$full"
            exit 2
        }
    }
    return $full
}

# 正式模式下检查额外参数没有覆盖共享字段；不打印，失败直接 exit。
function Assert-NoProtectedArgs {
    param([string[]]$Extra, [string]$Which)
    if ($NoBenchArgs -or -not $Extra -or $Extra.Count -eq 0) {
        return
    }
    foreach ($a in $Extra) {
        $flag = $a
        $eq = $a.IndexOf('=')
        if ($eq -gt 0) {
            $flag = $a.Substring(0, $eq)
        }
        if ($script:ProtectedArgs -contains $flag) {
            Write-Host "$Which 里出现受保护参数 '$flag'：它属于两臂共享配置，只能由驱动统一生成。"
            Write-Host "受保护参数：$($script:ProtectedArgs -join ' ')"
            exit 2
        }
    }
}

if ($ArmAName -eq $ArmBName) {
    Write-Host "两臂目录名相同（$ArmAName），B 会覆盖 A 的证据；请用 -ArmAName / -ArmBName 区分"
    exit 2
}
$armADir = Resolve-ArmDir -Name $ArmAName -Which 'A 臂'
$armBDir = Resolve-ArmDir -Name $ArmBName -Which 'B 臂'
Assert-NoProtectedArgs -Extra $ArmAExtraArgs -Which '-ArmAExtraArgs'
Assert-NoProtectedArgs -Extra $ArmBExtraArgs -Which '-ArmBExtraArgs'

# 组装一臂的参数表。返回 [string[]]，不打印任何东西。
function Build-ArmArgs {
    param([string]$Trainer, [string]$Model, [string[]]$Extra, [string]$ArmOut)
    $list = [System.Collections.Generic.List[string]]::new()
    if (-not $NoBenchArgs) {
        $list.AddRange([string[]]@(
                '--runs', "$Runs",
                '--seed', $Seed,
                '--run-offset', $RunOffset,
                '--cards', $Cards,
                '--extra-count', $ExtraCount,
                '--threads', "$Threads",
                '--search-n', "$SearchN",
                '--out', $ArmOut,
                '--trainer', $Trainer
            ))
        if ($Model) {
            $list.AddRange([string[]]@('--model', $Model, '--special-mode', $SpecialMode))
        }
    }
    if ($Extra -and $Extra.Count -gt 0) {
        $list.AddRange([string[]]$Extra)
    }
    return , $list.ToArray()
}

# 跑一臂，返回 [pscustomobject]（ExitCode / WallSeconds / TimedOut / Started）。
# 只用 Log 打印，绝不裸写字符串。
function Invoke-Arm {
    param(
        [string]$Name,
        [string]$ArmDir,
        [string]$Trainer,
        [string]$Model,
        [string[]]$Extra,
        [datetime]$Deadline
    )
    $armOut = $ArmDir
    New-Item -ItemType Directory -Path $armOut -Force | Out-Null
    $remain = ($Deadline - (Get-Date)).TotalSeconds
    if ($remain -le 0) {
        Log "[$Name] 共享预算已耗尽（剩余 $([math]::Round($remain,1)) s），本臂未启动"
        return [pscustomobject]@{ Name = $Name; Started = $false; ExitCode = $null; WallSeconds = 0.0; TimedOut = $false }
    }

    $argList = Build-ArmArgs -Trainer $Trainer -Model $Model -Extra $Extra -ArmOut $armOut
    # 权威记录是结构化数组：命令行拼接会丢掉参数边界（含空格的路径尤其明显）
    Set-Content -LiteralPath (Join-Path $armOut 'command.args.json') `
        -Value (ConvertTo-Json -InputObject @{ exe = $exePath; args = $argList } -Depth 3) -Encoding utf8
    $cmdText = "$exePath $(($argList | ForEach-Object { if ($_ -match '[\s"]') { '"' + $_ + '"' } else { $_ } }) -join ' ')"
    Set-Content -LiteralPath (Join-Path $armOut 'command.txt') -Value $cmdText -Encoding utf8
    Log "[$Name] 剩余预算 $([math]::Round($remain,1)) s（截止 $($Deadline.ToString('HH:mm:ss'))）"
    Log "[$Name] $cmdText"

    $t0 = Get-Date
    # 逐个 Add 到 ArgumentList：由 .NET 负责转义，不自己拼命令行。
    # CreateNoWindow + UseShellExecute=false：不弹可见窗口。
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $exePath
    foreach ($a in $argList) {
        $psi.ArgumentList.Add($a)
    }
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $p = [System.Diagnostics.Process]::Start($psi)
    # 先挂上异步搬运再等待：管道缓冲写满会让子进程阻塞。
    # 直接 BaseStream → FileStream 原样复制字节（不解码、不重编码）：
    # FileStream 的 bufferSize=1 表示不加内部缓冲，子进程 flush 出来的内容立刻进文件，
    # FileShare.ReadWrite 让运行当中也能读日志；驱动被中断时已产生的日志不会丢。
    $outPath = Join-Path $armOut 'run.stdout.txt'
    $errPath = Join-Path $armOut 'run.stderr.txt'
    $outFs = [System.IO.FileStream]::new(
        $outPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::ReadWrite, 1, $true)
    $errFs = [System.IO.FileStream]::new(
        $errPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::ReadWrite, 1, $true)
    $outCopy = $p.StandardOutput.BaseStream.CopyToAsync($outFs, 4096)
    $errCopy = $p.StandardError.BaseStream.CopyToAsync($errFs, 4096)

    $timedOut = $false
    if (-not $p.WaitForExit([int]([math]::Min($remain * 1000, [int]::MaxValue)))) {
        $timedOut = $true
        Log "[$Name] ❗到达共享截止时刻，终止本驱动启动的进程（PID $($p.Id)）"
        try { $p.Kill($true) } catch { Log "[$Name] 终止进程时报错：$_" }
        $p.WaitForExit()
    }
    # WaitForExit 之后**立刻**保存退出码，排在日志收尾之前
    $code = $p.ExitCode
    $wall = ((Get-Date) - $t0).TotalSeconds
    Set-Content -LiteralPath (Join-Path $armOut 'run.exitcode.txt') -Value "$code" -Encoding utf8
    Set-Content -LiteralPath (Join-Path $armOut 'run.wall_s.txt') -Value ('{0:N3}' -f $wall) -Encoding utf8
    # 日志搬运有界等待：管道已随进程结束而关闭，正常几毫秒内完成；
    # 卡住也不拖住驱动，文件里已经有到那一刻为止的全部内容。
    foreach ($t in @($outCopy, $errCopy)) {
        try {
            if (-not $t.Wait(5000)) {
                Log "[$Name] ❗日志搬运 5 s 内未收尾，按现有内容继续（文件已含到此刻的全部输出）"
            }
        } catch {
            Log "[$Name] 日志搬运收尾报错：$_"
        }
    }
    $outFs.Dispose()
    $errFs.Dispose()
    Log "[$Name] exit=$code wall=$([math]::Round($wall,1))s timedOut=$timedOut"
    return [pscustomobject]@{ Name = $Name; Started = $true; ExitCode = $code; WallSeconds = $wall; TimedOut = $timedOut }
}

Log "证据目录 $outRoot"
Log "A 臂目录 $armADir"
Log "B 臂目录 $armBDir"
Log "可执行文件 $exePath"
if ($NoBenchArgs) {
    Log '❗-NoBenchArgs：驱动自检模式，不生成 benchmark 参数'
} else {
    Log "世界 seed=$Seed run_idx ∈ [$RunOffset, +$Runs)；cards=$Cards extra=$ExtraCount threads=$Threads search_n=$SearchN"
}
Log "A 臂 trainer=$ArmATrainer model=$(if ($ArmAModel) { $ArmAModel } else { '—' })"
Log "B 臂 trainer=$ArmBTrainer model=$(if ($ArmBModel) { $ArmBModel } else { '—' })"

$start = Get-Date
$deadline = $start.AddSeconds($BudgetSecs)
Log "共享预算 $BudgetSecs s：起 $($start.ToString('HH:mm:ss')) 止 $($deadline.ToString('HH:mm:ss'))（两臂共用这一个截止时刻）"

$results = [System.Collections.Generic.List[object]]::new()
$armA = Invoke-Arm -Name $ArmAName -ArmDir $armADir -Trainer $ArmATrainer -Model $ArmAModel `
    -Extra $ArmAExtraArgs -Deadline $deadline
$results.Add($armA)

$aOk = $armA.Started -and (-not $armA.TimedOut) -and ($armA.ExitCode -eq 0)
if (-not $aOk -and -not $ContinueOnFailure) {
    Log "❗A 臂未成功（started=$($armA.Started) exit=$($armA.ExitCode) timedOut=$($armA.TimedOut)），按默认策略不启动 B 臂"
} else {
    if (-not $aOk) {
        Log '❗A 臂未成功，但 -ContinueOnFailure 已开启，仍启动 B 臂'
    }
    $armB = Invoke-Arm -Name $ArmBName -ArmDir $armBDir -Trainer $ArmBTrainer -Model $ArmBModel `
        -Extra $ArmBExtraArgs -Deadline $deadline
    $results.Add($armB)
}

$total = ((Get-Date) - $start).TotalSeconds
Log '=== 汇总 ==='
foreach ($r in $results) {
    if ($r.Started) {
        Log "  $($r.Name): exit=$($r.ExitCode) wall=$([math]::Round($r.WallSeconds,1))s timedOut=$($r.TimedOut)"
    } else {
        Log "  $($r.Name): 未启动（共享预算耗尽）"
    }
}
Log "两臂合计墙钟 $([math]::Round($total,1)) s / 预算 $BudgetSecs s"

$failed = @($results | Where-Object { (-not $_.Started) -or $_.TimedOut -or ($_.ExitCode -ne 0) })
$plannedArms = 2
if ($results.Count -lt $plannedArms) {
    Log "❗只跑了 $($results.Count) / $plannedArms 臂"
}
if ($failed.Count -gt 0 -or $results.Count -lt $plannedArms) {
    Log '❗本次配对未全部成功：已完成的记录保留在各臂目录，未完成项不补造'
    exit 1
}
Log '两臂均成功'
exit 0
