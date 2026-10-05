# strategy_lab/scripts/setup_forward_task_scheduler.ps1
# -----------------------------------------------------
# Registers a Windows Scheduled Task to run the Phase 17 Forward Baseline Tracker daily at 00:10 UTC.

$TaskName = "XTrading_Phase17_Forward_Baseline_Tracker"
$ScriptDir = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptDir
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$TrackerScript = Join-Path $ScriptDir "paper\forward_baseline_tracker.py"

Write-Host "Registering Daily Task: $TaskName"
Write-Host "Target: $PythonExe $TrackerScript"

# Trigger daily at 00:10 UTC (convert to local time)
# For UTC+5:30 (IST), 00:10 UTC = 05:40 IST
$Action = New-ScheduledTaskAction -Execute $PythonExe -Argument $TrackerScript -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -Daily -At "05:40"
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# Operational Note:
# Registered to run under the active logged-on user account.
# Configured with -StartWhenAvailable and -AllowStartIfOnBatteries.
# If the machine is asleep/signed-out at 05:40 IST, the task triggers upon next login,
# and the forward tracker's chronological catch-up logic automatically processes all missed daily bars.
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "Daily Phase 17 Trend_EMA_50 Forward Paper-Trading Tracker" -Force

Write-Host "Task registered successfully. To run manually: schtasks /run /tn $TaskName"
