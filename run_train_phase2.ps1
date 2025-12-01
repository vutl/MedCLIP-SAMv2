param(
    [Parameter(Mandatory=$true)]
    [string]$Dataset,
    
    [Parameter(Mandatory=$false)]
    [int]$Epochs = 50,
    
    [Parameter(Mandatory=$false)]
    [int]$BatchSize = 8,
    
    [Parameter(Mandatory=$false)]
    [string]$VEnv = "C:/Users/Admin/anaconda3/envs/medclipsamv2/python.exe",

    [Parameter(Mandatory=$false)]
    [switch]$DryRun
)

Write-Host "Using Python: $VEnv" -ForegroundColor Cyan

Set-Location "D:\Documents\LMIS\MedCLIP-SAMv2"

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Starting FreqMedCLIP Phase 2 Training" -ForegroundColor Green
Write-Host "Dataset: $Dataset" -ForegroundColor Yellow
Write-Host "Epochs: $Epochs" -ForegroundColor Yellow
if ($DryRun) { Write-Host "Mode: DRY RUN" -ForegroundColor Yellow }
Write-Host "========================================`n" -ForegroundColor Green

$cmdArgs = @(
    "run", "-n", "medclipsamv2", "python", "train_freq_fusion.py",
    "--dataset", $Dataset,
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--save-dir", "checkpoints/$Dataset"
)

if ($DryRun) {
    $cmdArgs += "--dry-run"
}

conda @cmdArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nTraining Completed Successfully!" -ForegroundColor Green
    Write-Host "Checkpoints saved in checkpoints/$Dataset" -ForegroundColor Cyan
} else {
    Write-Host "`nTraining Failed!" -ForegroundColor Red
}
