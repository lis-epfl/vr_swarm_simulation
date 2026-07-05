# launch_sim_scripts.ps1
# Launches the Python helper processes for the VR swarm simulation in a single
# Windows Terminal window, split into two panes:
#   left  : StitcherThreading.py  -> 'stitching' conda env (needs torch/StabStitch++)
#   right : readController.py      -> reads the joystick and streams to Unity over UDP
#
# Usage:  right-click > "Run with PowerShell", or from a shell:  .\launch_sim_scripts.ps1

$ErrorActionPreference = 'Stop'

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
