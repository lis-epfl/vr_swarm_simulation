# launch_sim_scripts.ps1
# Launches the Python helper processes for the VR swarm simulation in a single
# Windows Terminal window, split into two panes:
#   left  : StitcherThreading.py  -> 'stitching' conda env (needs torch/StabStitch++)
#   right : readController.py      -> reads the joystick and streams to Unity over UDP
#
# Usage:  right-click > "Run with PowerShell", or from a shell:  .\launch_sim_scripts.ps1
#         .\launch_sim_scripts.ps1 -Asw Auto              # leave Oculus ASW on (see below)
param(
    [ValidateSet('Off', 'Auto', 'Keep')][string]$Asw = 'Off'
)

$ErrorActionPreference = 'Stop'

# Oculus Asynchronous SpaceWarp. Over Link the headset runs at 72 Hz, and with ASW on (the
# runtime's default, Auto) any stretch in which Unity misses the 13.9 ms frame budget gets the
# app locked to exactly half rate, 36 fps, until the runtime decides it has headroom again --
# which is why the recorded swarm runs sit at either 72 or 36 and nothing in between. With ASW
# off an overrun costs only the frames that miss (~60 fps rather than 36), and the compositor
# still reprojects head rotation for them. The setting belongs to the Oculus service and resets
# when it restarts, so it is re-applied on every launch. -Asw Auto restores the default;
# -Asw Keep leaves whatever is set.
if ($Asw -ne 'Keep') {
    $odt = @(
        'C:\Program Files\Meta Horizon\Support\oculus-diagnostics\OculusDebugToolCLI.exe',
        'C:\Program Files\Oculus\Support\oculus-diagnostics\OculusDebugToolCLI.exe'
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($odt) {
        try {
            $cmdFile = Join-Path $env:TEMP 'vr_swarm_asw.txt'
            Set-Content -Path $cmdFile -Value @("asw.$Asw", 'asw.Mode', 'exit') -Encoding ascii
            $proc = Start-Process -FilePath $odt -ArgumentList '-f', "`"$cmdFile`"" -NoNewWindow -PassThru
            if (-not $proc.WaitForExit(15000)) {
                $proc.Kill()
                Write-Warning "OculusDebugToolCLI did not finish; ASW left as it was (is the Oculus service running?)"
            }
        } catch {
            Write-Warning "Could not set Oculus ASW to $Asw ($($_.Exception.Message)); set it in the Oculus Debug Tool instead."
        }
    } else {
        Write-Warning "OculusDebugToolCLI.exe not found; set Asynchronous Spacewarp to $Asw in the Oculus Debug Tool."
    }
}

$stitcherDir = Join-Path $PSScriptRoot 'Assets\Scripts\ImageStitching'
$controlDir  = Join-Path $PSScriptRoot 'Assets\Scripts\Control'

# A fresh window does not know the `conda` command until the conda hook is
# sourced -- that is why a bare `conda activate` gives "not recognized".
# (Same hook the working dji-flocking.ps1 launcher uses.)
$CondaHook = "C:\Users\jarvis\AppData\Local\miniconda3\shell\condabin\conda-hook.ps1"
$EnvName   = 'stitching'

# Single Windows Terminal window, two vertical panes. `;` separates wt.exe
# sub-commands, so the `;` INSIDE each PowerShell -Command is escaped as `\;`.
wt.exe --size 200,50 `
  new-tab --title "stitcher" `
    -d "$stitcherDir" `
    PowerShell -NoExit -Command "& '$CondaHook' \; conda activate $EnvName \; python StitcherThreading.py" `
  `; split-pane -V --size 0.5 --title "controller" `
    -d "$controlDir" `
    PowerShell -NoExit -Command "& '$CondaHook' \; conda activate $EnvName \; python readController.py"
