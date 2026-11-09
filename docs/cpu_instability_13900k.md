# CPU instability on the dev workstation (i9-13900K)

Open issue, parked for later investigation. Heavy single-threaded Python/numpy on this machine crashes at random,
and the evidence points at the CPU, not the code.

## The machine

| | |
|---|---|
| CPU | Intel Core i9-13900K |
| Board / BIOS | ASUS ROG STRIX Z790-F GAMING WIFI, BIOS 0904 (2023-03-29) |
| Microcode | **0x113**, still as of 2026-10-19 (registry `HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0`, value `Update Revision`, 4 bytes little-endian: `19,1,0,0`) |
| RAM | 5600 kit at stock 4800 MT/s, XMP off, so not a memory overclock |

Microcode 0x113 predates every one of Intel's 13th/14th-gen instability fixes (0x125, 0x129, 0x12B, all 2024),
which address elevated core voltage at single-core boost (the "Vmin shift" degradation).

## Symptoms

- `Windows fatal exception: access violation` (exit code `-1073741819`, `0xC0000005`) or heap corruption
  (`-1073740940`, `0xC0000374`) from `python.exe`, at a different line each time.
- Impossible numpy errors just before the crash, for example `ValueError: operands could not be broadcast together
  with shapes (10,2) () (10,17)` from `np.where` in `swarm_replica.bump`, where all three operands come from one
  array and cannot disagree.
- A `multiprocessing` pool worker dies silently and its job is lost, so a sweep waits forever. `citysim.parallel`
  and `vissim`'s 15-minute no-result timeout exist because of this.
- Application log entries for `python310.dll` / `torch_cpu.dll` access violations at varying offsets
  (2026-09-14/15) fit the same pattern.

## Evidence that it is hardware

Found 2026-10-13. A 30-line loop with no project code faults on its own:

```python
np.maximum(np.abs(np.einsum("nbk,bjk->nbj", d, u)) - h, 0)
```

| configuration | calls | faults |
|---|---|---|
| one process alone (full single-core boost) | a few million | about 1 per few million calls |
| four copies pinned to four P-cores (lower all-core clocks) | 21 M | 0 |
| one copy on four E-cores | 8.6 M | 0 |

So the fault rate follows clock and voltage, not the code.

### 2026-10-19: worse than before

Rebuilding the density-study explorer (`python -m citysim.explorer --bare`) crashed **5 times out of 5** unpinned,
in the goal-scene step (replica numpy) and later in the build. On 2026-10-13 the same build failed about 1 run in 5.
Pinned to the E-cores it succeeded on the first try:

```bat
start "" /wait /b /affinity FFFF0000 "%LOCALAPPDATA%\miniconda3\envs\stitching\python.exe" -B -m citysim.explorer --bare --out <path>
```

(`FFFF0000` is logical processors 16–31, the 13900K's 16 E-cores. PowerShell's
`$p.ProcessorAffinity = [IntPtr]0xFFFF0000` was refused with "The parameter is incorrect"; `cmd`'s `start /affinity`
works.) The jump from 1-in-5 to 5-in-5 may mean the chip is degrading further. That is the main reason not to leave
this too long.

## Workarounds until it is fixed

- **Retry first.** Before chasing a segfault that does not reproduce, check the microcode. While it is 0x113, treat
  the crash as hardware and rerun.
- **Pin long single-threaded jobs to the E-cores** with `start /affinity FFFF0000` as above. Pool jobs that load all
  cores run at lower clocks and fault far less often.
- **Check results by rerunning bit-identically.** Every study job is deterministic in its seed, so a second run that
  matches the first shows the output was not silently corrupted. A crash is the visible failure; a wrong number that
  does not crash is the one to worry about.

## To do

1. Update the BIOS to a release carrying microcode **0x12B or later** (ASUS Z790-F BIOS 2024-08 or newer), then load
   Intel's default power profile in the BIOS.
2. Rerun the einsum loop above unpinned for ~10 M calls. Zero faults means the fix worked.
3. If it still faults, the chip has probably already degraded. Intel extended these chips' warranty by two years, so
   an RMA is the next step.
4. Rerun one already-recorded deterministic study job and check its output is unchanged from the committed result.
