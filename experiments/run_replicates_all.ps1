# experiments/run_replicates_all.ps1
param(
  [int]$replicates = 20,
  [int]$seed_start = 2000
)

$mergedDir = "C:\Kanu\Dissertation\Benchmark dataset and instance generator for Real-World 3dBPP\Output\merged"
$normFiles = Get-ChildItem $mergedDir -Filter "*_merged_with_boxes_norm.json" | Select-Object -ExpandProperty FullName
if (-not $normFiles) { Write-Host "No normalized files found in $mergedDir"; exit 1 }

$combined = ".\experiments_output\all_norm_replicates_summary.csv"
"instance,seed,out_dir,best_score,unpacked,infeasible,duration" | Out-File $combined -Encoding utf8

foreach ($f in $normFiles) {
  $base = (Split-Path $f -Leaf).Replace('.json','')
  Write-Host "`n== Running replicates for $base =="
  for ($i=0; $i -lt $replicates; $i++) {
    $seed = $seed_start + $i
    & .\experiments\run_replicates.ps1 -normfile $f -replicates 1 -seed_start $seed
    # read the produced CSV for this run and append to combined
    $singleCsv = ".\experiments_output\{0}_replicates_summary.csv" -f $base
    if (Test-Path $singleCsv) {
      $lines = Get-Content $singleCsv | Select-Object -Skip 1
      foreach ($ln in $lines) {
        # ln format: seed,out_dir,best_score,unpacked,infeasible,duration
        $parts = $ln -split ","
        $seedVal = $parts[0].Trim('"')
        $outdir = $parts[1].Trim('"')
        $best = $parts[2].Trim('"')
        $unp = $parts[3].Trim('"')
        $inf = $parts[4].Trim('"')
        $dur = $parts[5].Trim('"')
        '"{0}","{1}","{2}","{3}","{4}","{5}","{6}"' -f $base, $seedVal, $outdir, $best, $unp, $inf, $dur | Out-File -Append $combined -Encoding utf8
      }
      Remove-Item $singleCsv -Force -ErrorAction SilentlyContinue
    } else {
      Write-Host "Warning: expected summary $singleCsv missing"
    }
  }
}
Write-Host "`nAll replicates done. Combined summary at $combined"
