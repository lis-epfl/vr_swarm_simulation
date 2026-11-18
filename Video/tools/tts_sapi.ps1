# Offline narration for build_video.py: Windows SAPI voices, one WAV per job.
# Usage: powershell -File tts_sapi.ps1 jobs.json   (jobs: [{text, voice, rate, out}, ...])
# List the voices on this machine:  powershell -File tts_sapi.ps1 -List
param([string]$JobsFile, [switch]$List)

Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

if ($List) {
    $synth.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name + "  (" + $_.VoiceInfo.Culture + ")" }
    exit 0
}

$jobs = Get-Content -Raw -Encoding UTF8 $JobsFile | ConvertFrom-Json
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(44100,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)

foreach ($j in $jobs) {
    try { $synth.SelectVoice($j.voice) } catch { Write-Warning "voice '$($j.voice)' not installed; using the default" }
    $synth.Rate = [int]$j.rate
    $synth.SetOutputToWaveFile($j.out, $fmt)
    if ($j.text.TrimStart().StartsWith("<speak")) { $synth.SpeakSsml($j.text) } else { $synth.Speak($j.text) }
    $synth.SetOutputToNull()
}
$synth.Dispose()
