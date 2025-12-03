# run_all_experiments.ps1
# Timestamped pipeline (uses fixed summarizer + fixed figure maker)
# Usage:
#   Activate your venv, then run:
#     .\run_all_experiments.ps1

$ErrorActionPreference = "Stop"

# ---------- CONFIG ----------
# Normfile lives in input_dataset/
$NORMFILE = Join-Path (Get-Location) "input_dataset\XML100_1111_01_merged_with_boxes_norm.json"

# Replicate settings (edit if desired)
$REPLICATES = 20
$SEED_START = 2000

# Project folders
$EXPERIMENTS_OUTPUT = Join-Path (Get-Location) "experiments_output"
$RESULTS_ROOT = Join-Path (Get-Location) "results"

# Scripts (fixed versions where appropriate)
$RUN_REPLICATES_PS1 = Join-Path (Get-Location) "experiments\run_replicates.ps1"
$SUMMARIZE_FIXED_PY = Join-Path (Get-Location) "scripts\summarize_replicates_fixed.py"
$MAKE_FIGURES_FIXED_PY = Join-Path (Get-Location) "scripts\make_figures_fixed.py"
$EXTRACT_ISOLATIONS_PY = Join-Path (Get-Location) "scripts\extract_isolations.py"
$ISOLATOR_PY = Join-Path (Get-Location) "byboxes_iterative_isolate.py"
# ---------- END CONFIG ----------

Write-Host "=== RUN ALL EXPERIMENTS (timestamped, fixed pipeline) ==="

# 0) Sanity checks
if (-not (Test-Path $RUN_REPLICATES_PS1)) {
    Write-Host "ERROR: cannot find $RUN_REPLICATES_PS1. Aborting."
    exit 1
}
if (-not (Test-Path $NORMFILE)) {
    Write-Host "ERROR: normfile not found at $NORMFILE. Aborting."
    exit 1
}
if (-not (Test-Path $SUMMARIZE_FIXED_PY)) {
    Write-Host "ERROR: fixed summarizer not found at $SUMMARIZE_FIXED_PY. Aborting."
    exit 1
}
if (-not (Test-Path $MAKE_FIGURES_FIXED_PY)) {
    Write-Host "WARNING: fixed figure maker not found at $MAKE_FIGURES_FIXED_PY. Figures step will be skipped if missing."
}

# 1) Build run id and run folders
$RUN_ID = (Get-Date).ToString("yyyyMMdd_HHmmss")
$RUN_DIR = Join-Path $RESULTS_ROOT ("runs\" + $RUN_ID)
$RUN_OUTPUT_DIR = Join-Path $RUN_DIR "experiments_output"
$RUN_FIGURES_DIR = Join-Path $RUN_DIR "figures"

New-Item -ItemType Directory -Path $RUN_DIR -Force | Out-Null
New-Item -ItemType Directory -Path $RUN_OUTPUT_DIR -Force | Out-Null
New-Item -ItemType Directory -Path $RUN_FIGURES_DIR -Force | Out-Null

Write-Host "Run ID: $RUN_ID"
Write-Host "Run dir: $RUN_DIR"
Write-Host "Using normfile: $NORMFILE"

# 2) Back up any existing experiments_output (move it aside)
$backupPath = $null
if (Test-Path $EXPERIMENTS_OUTPUT) {
    $itemCount = (Get-ChildItem $EXPERIMENTS_OUTPUT -Recurse -Force | Measure-Object).Count
    if ($itemCount -gt 0) {
        $backupPath = Join-Path (Get-Location) ("experiments_output_backup_" + $RUN_ID)
        Write-Host "Backing up existing experiments_output -> $backupPath"
        Move-Item -Path $EXPERIMENTS_OUTPUT -Destination $backupPath
    } else {
        # remove empty folder and recreate
        Remove-Item -Path $EXPERIMENTS_OUTPUT -Recurse -Force -ErrorAction SilentlyContinue
    }
}
# create fresh experiments_output
New-Item -ItemType Directory -Path $EXPERIMENTS_OUTPUT -Force | Out-Null

# 3) Run replicates (this writes into experiments_output)
Write-Host "Running replicates (this may take a while) ..."
& $RUN_REPLICATES_PS1 -normfile $NORMFILE -replicates $REPLICATES -seed_start $SEED_START

# 4) Move produced experiments_output into the run folder
if (Test-Path $EXPERIMENTS_OUTPUT) {
    Write-Host "Moving produced experiments_output -> $RUN_OUTPUT_DIR"
    Move-Item -Path $EXPERIMENTS_OUTPUT -Destination $RUN_OUTPUT_DIR -Force
} else {
    Write-Host "Warning: expected experiments_output not found after replicates."
}

# 5) Restore previous experiments_output if we backed it up
if ($backupPath) {
    Write-Host "Restoring previous experiments_output from backup: $backupPath"
    if (Test-Path $EXPERIMENTS_OUTPUT) {
        # remove any leftover empty experiments_output
        Remove-Item -Path $EXPERIMENTS_OUTPUT -Recurse -Force -ErrorAction SilentlyContinue
    }
    Move-Item -Path $backupPath -Destination $EXPERIMENTS_OUTPUT
} else {
    # ensure experiments_output exists for convenience
    New-Item -ItemType Directory -Path $EXPERIMENTS_OUTPUT -Force | Out-Null
}

# 6) Run isolator on route JSONs inside this run's output (if isolator exists)
Write-Host "Running isolator on route JSON files (if any)..."
$routeFiles = Get-ChildItem -Path $RUN_OUTPUT_DIR -Recurse -Filter "*route*.json" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
if ($routeFiles.Count -eq 0) {
    Write-Host "No route JSON files found in run output. Skipping isolator step."
} else {
    foreach ($r in $routeFiles) {
        $base = [System.IO.Path]::GetFileNameWithoutExtension($r)
        $logTarget = Join-Path $RUN_DIR ("isolate_" + $base + ".log")
        if (Test-Path $ISOLATOR_PY) {
            Write-Host "Isolating: $r -> $logTarget"
            python $ISOLATOR_PY $r 2>&1 | Out-File -FilePath $logTarget -Encoding utf8
        } else {
            Write-Host "WARNING: isolator script not found at $ISOLATOR_PY. Skipping."
        }
    }
}

# 7) Summarize replicates using the FIXED summarizer
$repSummaryTarget = Join-Path $RUN_DIR "replicates_summary.csv"
Write-Host "Summarizing replicates into: $repSummaryTarget"
python $SUMMARIZE_FIXED_PY --run-dir $RUN_OUTPUT_DIR --out $repSummaryTarget

if (-not (Test-Path $repSummaryTarget)) {
    Write-Host "ERROR: summarizer did not create $repSummaryTarget. Aborting figure creation."
} else {
    Write-Host "Summary CSV created: $repSummaryTarget"
}

# 8) Extract isolations (if the extractor exists)
$isolationCsv = Join-Path $RUN_DIR "isolation_summary.csv"
if (Test-Path $EXTRACT_ISOLATIONS_PY) {
    Write-Host "Extracting isolations to: $isolationCsv"
    python $EXTRACT_ISOLATIONS_PY --logs-dir $RUN_DIR --out $isolationCsv
} else {
    Write-Host "WARNING: extract_isolations.py not found at $EXTRACT_ISOLATIONS_PY. Skipping extraction."
}

# 9) Generate figures using the FIXED figure maker (into run-specific figures folder)
if ((Test-Path $MAKE_FIGURES_FIXED_PY) -and (Test-Path $repSummaryTarget)) {
    Write-Host "Generating figures into: $RUN_FIGURES_DIR"
    python $MAKE_FIGURES_FIXED_PY --summary $repSummaryTarget --out-dir $RUN_FIGURES_DIR
    Write-Host "Figures generated (if any)."
} else {
    Write-Host "Skipping figures: make_figures_fixed.py missing or summary CSV not present."
}

# 10) Save run-level summary text
$summaryTxt = Join-Path $RUN_DIR "summary_stats.txt"
"Run timestamp: $(Get-Date -Format u)" | Out-File -FilePath $summaryTxt -Encoding utf8
"Run ID: $RUN_ID" | Out-File -FilePath $summaryTxt -Append -Encoding utf8
"Normfile: $NORMFILE" | Out-File -FilePath $summaryTxt -Append -Encoding utf8
"Replicates: $REPLICATES  seed_start: $SEED_START" | Out-File -FilePath $summaryTxt -Append -Encoding utf8
if (Test-Path $repSummaryTarget) { "Replicates summary: $repSummaryTarget" | Out-File -FilePath $summaryTxt -Append -Encoding utf8 } else { "No replicates_summary.csv generated." | Out-File -FilePath $summaryTxt -Append -Encoding utf8 }
if (Test-Path $isolationCsv) { "Isolation summary: $isolationCsv" | Out-File -FilePath $summaryTxt -Append -Encoding utf8 } else { "No isolation_summary.csv generated." | Out-File -FilePath $summaryTxt -Append -Encoding utf8 }
"Run outputs moved to: $RUN_OUTPUT_DIR" | Out-File -FilePath $summaryTxt -Append -Encoding utf8
"Figures folder: $RUN_FIGURES_DIR" | Out-File -FilePath $summaryTxt -Append -Encoding utf8

Write-Host "=== RUN COMPLETE: Results saved in $RUN_DIR ==="
Write-Host "To inspect run files: Get-ChildItem -Recurse $RUN_DIR"
if (Test-Path $MAKE_FIGURES_FIXED_PY -and (Test-Path $repSummaryTarget)) {
    Write-Host "Generating figures into: $RUN_FIGURES_DIR"
    python $MAKE_FIGURES_FIXED_PY --summary $repSummaryTarget --out-dir $RUN_FIGURES_DIR
    Write-Host "Figures generated (if any)."
} else {
    Write-Host "Skipping figures: make_figures_fixed.py missing or summary CSV not present."
}
