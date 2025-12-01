# Activates the medclipsamv2 conda environment for this repo
conda activate medclipsamv2
if ($LASTEXITCODE -eq 0) { Write-Host "Activated conda env: medclipsamv2" -ForegroundColor Green } else { Write-Host "Failed activating conda env: medclipsamv2" -ForegroundColor Red; exit 1 }