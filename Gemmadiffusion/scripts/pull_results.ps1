param(
    [string]$Remote = "vastai",
    [string]$Pair   = "en-ru"
)

$LocalResults = Resolve-Path "$PSScriptRoot\.." | Join-Path -ChildPath "results\$Pair"
New-Item -ItemType Directory -Force -Path $LocalResults | Out-Null

rsync -avz "${Remote}:/root/Gemmadiffusion/results/$Pair/" "$LocalResults/"

Write-Host "Results pulled to $LocalResults"
$report = Join-Path $LocalResults "bleu_report.txt"
if (Test-Path $report) {
    Write-Host ""
    Get-Content $report
}
