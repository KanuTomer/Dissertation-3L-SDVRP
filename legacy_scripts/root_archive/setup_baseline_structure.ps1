# setup_baseline_structure.ps1
# Place this file in the ROOT of Dissertation-3L-SDVRP-main
#
# Example:
# Dissertation-3L-SDVRP-main/
# ├── dataset_generation/
# ├── scripts/
# ├── experiments/
# ├── run_all_experiments.ps1
# └── setup_baseline_structure.ps1
#
# Run with:
# powershell -ExecutionPolicy Bypass -File .\setup_baseline_structure.ps1

Write-Host ""
Write-Host "Creating baseline comparison structure..."
Write-Host ""

# -----------------------------
# Root folders
# -----------------------------
$root = "comparison_models"

$folders = @(
    "$root",
    "$root/common",
    "$root/common/algorithms",
    "$root/common/loaders",
    "$root/common/utils",
    "$root/common/scripts",

    "$root/baseline_a",
    "$root/baseline_b",
    "$root/baseline_c",
    "$root/proposed_model",

    "$root/outputs",
    "$root/outputs/baseline_a",
    "$root/outputs/baseline_b",
    "$root/outputs/baseline_c",
    "$root/outputs/proposed_model",

    "$root/plots",
    "$root/plots/baseline_a",
    "$root/plots/baseline_b",
    "$root/plots/baseline_c",
    "$root/plots/proposed_model"
)

foreach ($folder in $folders) {
    if (!(Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder | Out-Null
        Write-Host "Created folder: $folder"
    }
}

# -----------------------------
# Copy common GA/operator files
# -----------------------------
$copyMap = @{
    "dataset_generation/algorithms/ga/selection.py" = "$root/common/algorithms/selection.py"
    "dataset_generation/algorithms/ga/crossover.py" = "$root/common/algorithms/crossover.py"
    "dataset_generation/loaders/route_evaluator.py" = "$root/common/loaders/route_evaluator.py"
    "dataset_generation/utils/packer.py"            = "$root/common/utils/packer.py"
    "scripts/summarize_replicates_fixed.py"         = "$root/common/scripts/summarize_replicates_fixed.py"
    "scripts/make_figures_fixed.py"                 = "$root/common/scripts/make_figures_fixed.py"
}

foreach ($src in $copyMap.Keys) {
    $dest = $copyMap[$src]

    if (Test-Path $src) {
        Copy-Item $src $dest -Force
        Write-Host "Copied: $src -> $dest"
    }
    else {
        Write-Warning "Missing file: $src"
    }
}

# -----------------------------
# Copy GA runner + mutation + experiments into each model folder
# -----------------------------
$modelFolders = @(
    "$root/baseline_a",
    "$root/baseline_b",
    "$root/baseline_c",
    "$root/proposed_model"
)

$modelFiles = @(
    "dataset_generation/algorithms/ga/ga_runner.py",
    "dataset_generation/algorithms/ga/mutation.py",
    "dataset_generation/runners/run_experiments.py"
)

foreach ($modelFolder in $modelFolders) {
    foreach ($src in $modelFiles) {
        if (Test-Path $src) {
            $fileName = Split-Path $src -Leaf
            $dest = Join-Path $modelFolder $fileName
            Copy-Item $src $dest -Force
            Write-Host "Copied: $src -> $dest"
        }
        else {
            Write-Warning "Missing file: $src"
        }
    }
}

# -----------------------------
# Create config.py files
# -----------------------------
$configs = @{
    "$root/baseline_a/config.py"     = @"
USE_PACKING = False
PENALTY_ALPHA = 0
USE_ENHANCED_MUTATION = False
MODEL_NAME = "baseline_a"
MUTATION_TYPE = "swap"
"@

    "$root/baseline_b/config.py"     = @"
USE_PACKING = True
PENALTY_ALPHA = 100
USE_ENHANCED_MUTATION = False
MODEL_NAME = "baseline_b"
MUTATION_TYPE = "swap"
"@

    "$root/baseline_c/config.py"     = @"
USE_PACKING = True
PENALTY_ALPHA = 10000
USE_ENHANCED_MUTATION = False
MODEL_NAME = "baseline_c"
MUTATION_TYPE = "swap"
"@

    "$root/proposed_model/config.py" = @"
USE_PACKING = True
PENALTY_ALPHA = 10000
USE_ENHANCED_MUTATION = True
MODEL_NAME = "proposed_model"
MUTATION_TYPE = "hybrid"
"@
}

foreach ($path in $configs.Keys) {
    Set-Content -Path $path -Value $configs[$path]
    Write-Host "Created config file: $path"
}

# -----------------------------
# Create metrics logger
# -----------------------------
$metricsLogger = @"
import csv
import os

def save_metrics(output_csv, row_dict):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    file_exists = os.path.exists(output_csv)

    with open(output_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row_dict)
"@

Set-Content -Path "$root/common/metrics_logger.py" -Value $metricsLogger
Write-Host "Created metrics logger"

# -----------------------------
# Create placeholder compare script
# -----------------------------
$compareScript = @"
import pandas as pd
from pathlib import Path

base_dir = Path('comparison_models/outputs')

all_rows = []

for model_dir in base_dir.iterdir():
    if not model_dir.is_dir():
        continue

    csv_files = list(model_dir.glob('*.csv'))

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        summary = {
            'model': model_dir.name,
            'avg_score': df['best_score'].mean() if 'best_score' in df.columns else None,
            'avg_distance': df['best_distance'].mean() if 'best_distance' in df.columns else None,
            'avg_runtime': df['runtime_seconds'].mean() if 'runtime_seconds' in df.columns else None,
            'avg_feasibility_rate': df['feasibility_rate'].mean() if 'feasibility_rate' in df.columns else None,
            'std_dev_score': df['best_score'].std() if 'best_score' in df.columns else None
        }

        all_rows.append(summary)

summary_df = pd.DataFrame(all_rows)

print(summary_df)

summary_df.to_csv('comparison_models/model_comparison_summary.csv', index=False)
print('Saved comparison summary.')
"@

Set-Content -Path "$root/compare_all_models.py" -Value $compareScript
Write-Host "Created compare_all_models.py"

# -----------------------------
# Create README
# -----------------------------
$readme = @"
comparison_models/

common/
- Shared evaluator
- Shared packer
- Shared selection/crossover
- Shared metrics logger
- Shared plotting scripts

baseline_a/
- No packing
- No penalty

baseline_b/
- Packing enabled
- Weak penalty

baseline_c/
- Packing enabled
- Strong penalty

proposed_model/
- Packing enabled
- Strong penalty
- Enhanced mutation
"@

Set-Content -Path "$root/README.txt" -Value $readme
Write-Host "Created README.txt"

Write-Host ""
Write-Host "Baseline comparison structure setup complete."
Write-Host ""
Write-Host "Next files to modify:"
Write-Host "1. comparison_models/common/loaders/route_evaluator.py"
Write-Host "2. comparison_models/baseline_*/ga_runner.py"
Write-Host "3. comparison_models/baseline_*/mutation.py"
Write-Host "4. comparison_models/baseline_*/run_experiments.py"
Write-Host ""