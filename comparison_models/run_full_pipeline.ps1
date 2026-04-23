$root = Split-Path -Parent $PSScriptRoot
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

$logDir = Join-Path $PSScriptRoot "pipeline_logs"
if (!(Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$logFile = Join-Path $logDir "pipeline_run_$timestamp.log"

function Invoke-Step {
    param(
        [string]$Name,
        [string]$Command
    )

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "Starting: $Name" -ForegroundColor Yellow
    Write-Host "Command: $Command"
    Write-Host "Start Time: $(Get-Date)"
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""

    Add-Content $logFile ""
    Add-Content $logFile "============================================================"
    Add-Content $logFile "Starting: $Name"
    Add-Content $logFile "Command: $Command"
    Add-Content $logFile "Start Time: $(Get-Date)"
    Add-Content $logFile "============================================================"

    $start = Get-Date

    Invoke-Expression $Command 2>&1 | Tee-Object -FilePath $logFile -Append

    $end = Get-Date
    $duration = $end - $start

    Write-Host ""
    Write-Host "[DONE] $Name completed in $($duration.ToString())" -ForegroundColor Green
    Write-Host ""

    Add-Content $logFile ""
    Add-Content $logFile "End Time: $end"
    Add-Content $logFile "Duration: $duration"
    Add-Content $logFile ""
}

Set-Location $root

Invoke-Step "compare_all_models" "python -u comparison_models/compare_all_models.py --verbose"
Invoke-Step "make_final_table" "python comparison_models/make_final_table.py"
Invoke-Step "make_comparison_graphs" "python comparison_models/make_comparison_graphs.py"

Write-Host ""
Write-Host "All tasks completed." -ForegroundColor Green
Write-Host "Log file saved to: $logFile" -ForegroundColor Yellow