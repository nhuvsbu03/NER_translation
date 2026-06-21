param([string]$Remote = "vastai")

$LocalDir = (Resolve-Path "$PSScriptRoot\..").Path

rsync -avz `
    --exclude "results/" `
    --exclude "data/" `
    --exclude ".git/" `
    "$LocalDir/" "${Remote}:/root/Gemmadiffusion/"

Write-Host "Pushed Gemmadiffusion to ${Remote}:/root/Gemmadiffusion/"
