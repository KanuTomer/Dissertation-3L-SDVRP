param(
    [int]$replicates = 10,
    [string]$base_cfg = "experiments/vlr_ga_experiment.json",
    [string]$out_csv = "experiments_output/batch_summary.csv"
)

$base = Get-Content $base_cfg -Raw | ConvertFrom-Json
$rows = @()
for ($i = 0; $i -lt $replicates; $i++) {
    $seed = 1000 + $i
    $cfg = $base
    $cfg.seed = $seed
    $cfg.output_dir = "experiments_output/{$($base.name)}_seed$seed"
    # Write temp config
    $tmp = "experiments/temp_batch_cfg_$seed.json"
    $cfg | ConvertTo-Json -Depth 10 | Set-Content $tmp -Encoding utf8
    Write-Host "Running seed $seed -> $($cfg.output_dir)"
    python .\main.py --config $tmp
    # read result
    $res_path = Join-Path $cfg.output_dir "result.json"
    if (Test-Path $res_path) {
        $r = Get-Content $res_path -Raw | ConvertFrom-Json
        # compute unpacked & infeasible quickly
        $unpacked = 0
        $infeasible = 0
        foreach ($rr in $r.best_info.routes) {
            $bt = $rr.boxes_total
            $bp = $rr.boxes_packed
            $unpacked += [int]($bt - $bp)
            if (-not $rr.feasible) { $infeasible += 1 }
        }
        $rows += [PSCustomObject]@{
            seed = $seed
            out_dir = $cfg.output_dir
            best_score = $r.best_score
            unpacked = $unpacked
            infeasible = $infeasible
            duration = $r.duration
        }
    } else {
        $rows += [PSCustomObject]@{
            seed = $seed
            out_dir = $cfg.output_dir
            best_score = ""
            unpacked = ""
            infeasible = ""
            duration = ""
        }
    }
}

# write CSV
$rows | Export-Csv -Path $out_csv -NoTypeInformation -Force
Write-Host "Batch finished. Summary written to $out_csv"
