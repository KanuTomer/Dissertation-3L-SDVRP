# experiments/run_all_norm.ps1
$mergedDir = "C:\Kanu\Dissertation\Benchmark dataset and instance generator for Real-World 3dBPP\Output\merged"
$outCsv = ".\experiments_output\all_merged_summary_final.csv"
"file,best_score,unpacked,infeasible,duration" | Out-File $outCsv -Encoding utf8

Get-ChildItem $mergedDir -Filter "*_merged_norm.json" | ForEach-Object {
  $file = $_.FullName
  Write-Host "Running: $file"
  $cfgPath = ".\experiments\vlr_ga_experiment.json"
  $cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json
  $cfg.dataset_path = $file
  # tuned params (already in the file; override if desired)
  $cfg.population_size = 100
  $cfg.num_generations = 200
  $cfg.mut_prob = 0.6
  $cfg.cx_prob = 0.8
  $cfg.use_isolator = $true
  $temp = ".\experiments\temp_run.json"
  $cfg | ConvertTo-Json -Depth 12 | Set-Content $temp -Encoding utf8

  python .\main.py --config $temp --verbose

  $res = ".\experiments_output\vlr_ga_experiment\result.json"
  if (Test-Path $res) {
    $r = Get-Content $res -Raw | ConvertFrom-Json
    $best = $r.best_score
    $unp = ($r.best_info.routes | ForEach-Object { ($_.boxes_total - $_.boxes_packed) } | Measure-Object -Sum).Sum
    $inf = ($r.best_info.routes | Where-Object { -not $_.feasible } | Measure-Object).Count
    $dur = $r.duration
  } else {
    $best="NA"; $unp="NA"; $inf="NA"; $dur="NA"
  }
  '"{0}","{1}","{2}","{3}","{4}"' -f $file,$best,$unp,$inf,$dur | Out-File -Append $outCsv -Encoding utf8
  Remove-Item $temp -Force -ErrorAction SilentlyContinue
}
Write-Host "Done. Summary at $outCsv"
