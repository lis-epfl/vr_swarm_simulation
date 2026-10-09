<#
.SYNOPSIS
Headless swarm bench: syncs a project copy to this working tree, flies a config in batchmode Unity,
and summarises the results. See README.md beside this script.

.DESCRIPTION
The user's editor holds the live project open, so batchmode runs on a COPY (default
D:\claude_obstacle_bench\vr_swarm_simulation). Every run first mirrors Assets, Packages and
ProjectSettings from the live working tree into it (robocopy /MIR: only changed files move), so the
bench always flies the code and scenes you have now, uncommitted edits included. The copy's Library is
kept, so only what changed is reimported.

Each run writes runs\<Tag>\ under the bench root: config.json, meta.json (live commit, dirty flag),
results.jsonl, unity.log, traj_*.csv.

Exit codes: 0 done; 1 bad arguments or setup; 2 config rejected (by the runner); 3 runner failed at
runtime; 4 compile errors; 5 timed out or Unity died.

.EXAMPLE
.\run_bench.ps1 -Config my_config.json -Tag diamond_baseline
.\run_bench.ps1 -CompileOnly
.\run_bench.ps1 -SyncOnly -LibrarySeed D:\claude_practice_goal\vr_swarm_simulation\Library
#>
param(
    [string]$Config,
    [string]$Tag,
    [string]$BenchRoot = 'D:\claude_obstacle_bench',
    # Only needed when the copy has no Library yet: another copy's Library (or the live one) to start from.
    [string]$LibrarySeed,
    [switch]$NoSync,
    [switch]$SyncOnly,
    [switch]$CompileOnly,
    [int]$TimeoutMin = 180,
    [string]$Python = "$env:LOCALAPPDATA\miniconda3\envs\stitching\python.exe",
    [string]$UnityExe
)
$ErrorActionPreference = 'Stop'

$live = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')).Path   # Assets\Scripts\swarm\Bench -> repo root
$proj = Join-Path $BenchRoot 'vr_swarm_simulation'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$TimedOut = -999999   # Invoke-Unity's result when the run outlived -TimeoutMin

function Fail([int]$code, [string]$message) {
    Write-Host "[bench] $message"
    exit $code
}

function Write-Utf8([string]$path, [string]$text) {
    # Windows PowerShell's utf8 writes a BOM, which Python's json module rejects.
    [System.IO.File]::WriteAllText($path, $text, (New-Object System.Text.UTF8Encoding $false))
}

if (-not ($SyncOnly -or $CompileOnly) -and -not $Config) { Fail 1 'give -Config <file> (or -SyncOnly / -CompileOnly)' }
if ($Config -and -not (Test-Path $Config)) { Fail 1 "config $Config does not exist" }
if (-not (Test-Path $Python)) { Fail 1 "no Python at $Python (the stitching env); pass -Python" }

if (-not $UnityExe) {
    $ver = (Select-String -Path (Join-Path $live 'ProjectSettings\ProjectVersion.txt') -Pattern 'm_EditorVersion:\s*(\S+)').Matches[0].Groups[1].Value
    $UnityExe = "C:\Program Files\Unity\Hub\Editor\$ver\Editor\Unity.exe"
}
if (-not (Test-Path $UnityExe)) { Fail 1 "no Unity at $UnityExe; pass -UnityExe" }

# ---- one Unity per copy: a second would wait on the copy's lock, or fight the first over Library
$holders = Get-CimInstance Win32_Process -Filter "Name='Unity.exe'" |
    Where-Object { $_.CommandLine -and $_.CommandLine -match [regex]::Escape($proj) }
if ($holders) { Fail 1 "a Unity (pid $(($holders | ForEach-Object ProcessId) -join ', ')) already has $proj open" }

# ---- sync the copy to the live working tree
if (-not $NoSync) {
    New-Item -ItemType Directory -Force $proj | Out-Null
    # The pre-2026-10 bench kept its runners in the copy only; they are superseded by
    # Assets\Scripts\swarm\Bench and would otherwise be deleted by the mirror.
    $old = Join-Path $proj 'Assets\ObstacleBench'
    if (Test-Path $old) {
        $legacy = Join-Path $BenchRoot 'legacy'
        New-Item -ItemType Directory -Force $legacy | Out-Null
        Move-Item $old (Join-Path $legacy "ObstacleBench_$stamp")
        Remove-Item "$old.meta" -ErrorAction SilentlyContinue
        Write-Host "[bench] moved the copy's old Assets\ObstacleBench to $legacy"
    }
    $syncLog = Join-Path $BenchRoot 'sync.log'
    foreach ($d in 'Assets', 'Packages', 'ProjectSettings') {
        # Not a git checkout: Assets\Modular City Pack and other large packs are gitignored but needed.
        & robocopy (Join-Path $live $d) (Join-Path $proj $d) /MIR /MT:16 /R:2 /W:2 /NP /NFL /NDL /XD __pycache__ "/LOG+:$syncLog" | Out-Null
        if ($LASTEXITCODE -ge 8) { Fail 1 "robocopy $d failed (exit $LASTEXITCODE); see $syncLog" }
    }
    if (-not (Test-Path (Join-Path $proj 'Library'))) {
        if (-not $LibrarySeed) {
            Fail 1 "$proj has no Library. Pass -LibrarySeed <a Library folder>: another D: copy's is best (the live one is held by the editor; a full import takes much longer)"
        }
        & robocopy $LibrarySeed (Join-Path $proj 'Library') /MIR /MT:16 /R:2 /W:2 /NP /NFL /NDL "/LOG+:$syncLog" | Out-Null
        if ($LASTEXITCODE -ge 8) { Fail 1 "seeding Library failed (exit $LASTEXITCODE); see $syncLog" }
    }
    Write-Host "[bench] synced $proj to $live"
}
if ($SyncOnly) { exit 0 }

function Invoke-Unity([string[]]$unityArgs, [string]$log, [string]$results) {
    for ($attempt = 1; $attempt -le 2; $attempt++) {
        # The copied Library can carry the live editor's IL post-processing runner PID, and a batchmode
        # start "kills the lingering runner" it names -- i.e. the user's. Never let it see one.
        Remove-Item (Join-Path $proj 'Library\ilpp.pid') -ErrorAction SilentlyContinue
        $p = Start-Process -FilePath $UnityExe -ArgumentList $unityArgs -PassThru -WindowStyle Hidden
        $null = $p.Handle   # without this, Windows PowerShell can lose the exit code
        if (-not $p.WaitForExit($TimeoutMin * 60 * 1000)) {
            $p.Kill()
            return $TimedOut
        }
        $code = $p.ExitCode
        # A start once died at "Initializing Unity extensions" with an access violation and no output;
        # a plain retry worked.
        $empty = (-not $results) -or (-not (Test-Path $results)) -or ((Get-Item $results).Length -eq 0)
        if ($code -eq -1073741819 -and $empty -and $attempt -eq 1) {
            Write-Host '[bench] Unity died at startup (0xC0000005); retrying once'
            continue
        }
        return $code
    }
}

function Show-CompileErrors([string]$log) {
    if (-not (Test-Path $log)) { return $false }
    $errs = Select-String -Path $log -Pattern 'error CS\d+|Scripts have compiler errors|executeMethod class .* could not be found' |
        Select-Object -First 20
    if ($errs) {
        Write-Host '[bench] the copy does not compile:'
        $errs | ForEach-Object { Write-Host "  $($_.Line.Trim())" }
        return $true
    }
    return $false
}

$logs = Join-Path $BenchRoot 'logs'
New-Item -ItemType Directory -Force $logs | Out-Null

if ($CompileOnly) {
    $log = Join-Path $logs "compile_$stamp.log"
    $code = Invoke-Unity @('-batchmode', '-quit', '-projectPath', "`"$proj`"", '-executeMethod', 'SwarmBenchLauncher.CompileCheck', '-logFile', "`"$log`"") $log $null
    if (Show-CompileErrors $log) { exit 4 }
    if ($code -eq $TimedOut) { Fail 5 "timed out after $TimeoutMin min; see $log" }
    if ((Test-Path $log) -and (Select-String -Path $log -Pattern '\[SwarmBench\] compiled' -Quiet)) { Write-Host "[bench] compiles ($log)"; exit 0 }
    Fail 5 "Unity exited $code without reaching CompileCheck; see $log"
}

# ---- one run
if (-not $Tag) { $Tag = $stamp }
$run = Join-Path $BenchRoot "runs\$Tag"
if (Test-Path (Join-Path $run 'results.jsonl')) { Fail 1 "$run already has results; pick another -Tag" }
New-Item -ItemType Directory -Force $run | Out-Null
$cfgCopy = Join-Path $run 'config.json'
Copy-Item $Config $cfgCopy -Force
$results = Join-Path $run 'results.jsonl'
$log = Join-Path $run 'unity.log'

$commit = (& git -C $live rev-parse HEAD).Trim()
$dirty = @(& git -C $live status --porcelain -- Assets Packages ProjectSettings).Count
$meta = [ordered]@{
    tag = $Tag; started = (Get-Date -Format 'o'); config = (Resolve-Path $Config).Path
    liveProject = $live; commit = $commit; uncommittedFiles = $dirty; synced = (-not $NoSync); copy = $proj
}
Write-Utf8 (Join-Path $run 'meta.json') ($meta | ConvertTo-Json)

$env:SWARM_BENCH_CONFIG = $cfgCopy
$env:SWARM_BENCH_OUT = $results
Write-Host "[bench] $Tag : flying $Config on $proj (commit $($commit.Substring(0, 8)), $dirty uncommitted file(s))"
try {
    $code = Invoke-Unity @('-batchmode', '-projectPath', "`"$proj`"", '-executeMethod', 'SwarmBenchLauncher.Run', '-logFile', "`"$log`"") $log $results
}
finally {
    Remove-Item Env:SWARM_BENCH_CONFIG, Env:SWARM_BENCH_OUT -ErrorAction SilentlyContinue
}

if (Show-CompileErrors $log) { exit 4 }
if ($code -eq $TimedOut) { Fail 5 "timed out after $TimeoutMin min; partial results in $results, log $log" }
if (Test-Path $results) { & $Python -B (Join-Path $PSScriptRoot 'bench.py') report $run }
if ($code -eq 2 -or $code -eq 3) { Fail $code "the runner stopped (exit $code); see the error line above and $log" }
if ($code -ne 0) { Fail 5 "Unity exited $code; see $log" }
Write-Host "[bench] done: $run"
exit 0
