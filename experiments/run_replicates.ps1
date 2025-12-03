param(
  [Parameter(Mandatory=$true)][string]$normfile,
  [int]$replicates = 10,
  [int]$seed_start = 1000
)

# Validate input file
if (-not (Test-Path $normfile)) {
    Write-Error "normfile not found: $normfile"
    exit 1
}

# Prepare output CSV
$baseName = (Split-Path $normfile -Leaf).Replace('.json','')
$outCsv = ".\experiments_output\{0}_replicates_summary.csv" -f $baseName
"seed,out_dir,best_score,unpacked,infeasible,duration" | Out-File $outCsv -Encoding utf8

# Load template config as PSCustomObject
$tpl = Get-Content .\experiments\vlr_ga_experiment.json -Raw | ConvertFrom-Json

for ($i = 0; $i -lt $replicates; $i++) {
    $seed = $seed_start + $i
    $outDir = "experiments_output/{0}_seed{1}" -f $baseName, $seed

    # Build a mutable hashtable/copy of the template properties
    $cfg_ht = @{}
    foreach ($p in $tpl.PSObject.Properties) { $cfg_ht[$p.Name] = $p.Value }

    # Set / override runtime fields
    $cfg_ht["dataset_path"] = $normfile
    $cfg_ht["seed"] = $seed
    $cfg_ht["output_dir"] = $outDir

    # If you want to override GA params for replicates, uncomment and edit:
    # $cfg_ht["population_size"] = 100
    # $cfg_ht["num_generations"] = 200
    # $cfg_ht["mut_prob"] = 0.6
    # $cfg_ht["cx_prob"]  = 0.8
    # $cfg_ht["use_isolator"] = $true

    # Write temp config
    $tempPath = ".\experiments\temp_run_{0}.json" -f $seed
    $cfg_ht | ConvertTo-Json -Depth 12 | Set-Content $tempPath -Encoding utf8

    Write-Host "Running seed $seed -> out: $outDir"
    # Run Python (do NOT pass -Verbose — use --verbose if main.py supports it)
    & $env:VIRTUAL_ENV\Scripts\python.exe .\main.py --config $tempPath --verbose

    # Read result (guarded)
    $resFile = ".\experiments_output\vlr_ga_experiment\result.json"
    if (Test-Path $resFile) {
        $r = Get-Content $resFile -Raw | ConvertFrom-Json
        $best = $r.best_score
        $routes = $r.best_info.routes
        $unp = ($routes | ForEach-Object { ($_.boxes_total - $_.boxes_packed) } | Measure-Object -Sum).Sum
        $inf = ($routes | Where-Object { -not $_.feasible } | Measure-Object).Count
        $dur = $r.duration
    } else {
        $best = "NA"; $unp = "NA"; $inf = "NA"; $dur = "NA"
    }

    # Append summary CSV
    '"{0}","{1}","{2}","{3}","{4}","{5}"' -f $seed, $outDir, $best, $unp, $inf, $dur | Out-File -Append $outCsv -Encoding utf8

    # Cleanup temp config
    if (Test-Path $tempPath) { Remove-Item $tempPath -Force -ErrorAction SilentlyContinue }
}

Write-Host "Replicates done. Summary at $outCsv"
