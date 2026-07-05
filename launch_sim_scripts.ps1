# launch_sim_scripts.ps1
# Launches the Python helper processes for the VR swarm simulation, each in its own window:
#   1. StitcherThreading.py  -> runs in the 'stitching' conda env (needs torch/StabStitch++)
#   2. readController.py      -> reads the joystick and streams commands to Unity over UDP
#
# Usage:  right-click > "Run with PowerShell", or from a shell:  .\launch_sim_scripts.ps1

$ErrorActionPreference = 'Stop'

# Resolve paths relative to this script so it works regardless of the current directory.
$root       = $PSScriptRoot
$stitcherDir = Join-Path $root 'Assets\Scripts\ImageStitching'
$controlDir  = Join-Path $root 'Assets\Scripts\Control'

$condaEnv = 'stitching'

# --- Window 1: image stitcher in the 'stitching' conda env ---------------------
Start-Process powershell -ArgumentList @(
    '-NoExit',
    '-Command',
    "Set-Location '$stitcherDir'; conda activate $condaEnv; python StitcherThreading.py"
)

# --- Window 2: controller reader ----------------------------------------------
Start-Process powershell -ArgumentList @(
    '-NoExit',
    '-Command',
    "Set-Location '$controlDir'; python readController.py"
)

Write-Host "Launched StitcherThreading.py (env: $condaEnv) and readController.py in separate windows."
