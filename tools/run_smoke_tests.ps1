# Robust smoke test runner for MedCLIP-SAMv2
# Handles filenames with spaces and auto-generates prompt JSONs

$ErrorActionPreference = "Stop"
Set-Location 'D:\Documents\LMIS\MedCLIP-SAMv2'

# Dataset configurations: name, folder, prompt template, prompt json (if available)
$datasetConfigs = @(
    @{
        Name = 'breast_tumors'
        Folder = 'data\breast_tumors\train_images'
        PromptJson = 'saliency_maps\text_prompts\breast_tumors_testing.json'
        DefaultPrompt = 'A medical breast mammogram showing a mass suggestive of a breast tumor.'
    },
    @{
        Name = 'brain_tumor'
        Folder = 'data\brain_tumor\train_images'
        PromptJson = 'saliency_maps\text_prompts\brain_tumors_testing.json'
        DefaultPrompt = 'A brain imaging study revealing a mass indicative of a brain tumor.'
    },
    @{
        Name = 'lung_chest_xray'
        Folder = 'data\lung_chest_xray\train_images'
        PromptJson = 'saliency_maps\test_prompts.json'
        DefaultPrompt = 'Bilateral pulmonary infection, infected areas in lung.'
    },
    @{
        Name = 'lung_ct'
        Folder = 'data\lung_ct\train_images'
        PromptJson = $null
        DefaultPrompt = 'A medical CT scan displaying a clear and detailed image of the lung lobes.'
    }
)

foreach ($config in $datasetConfigs) {
    $datasetName = $config.Name
    $trainPath = $config.Folder
    
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Processing dataset: $datasetName" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    if (-not (Test-Path $trainPath)) {
        Write-Host "SKIP $datasetName : train_images not found at $trainPath" -ForegroundColor Yellow
        continue
    }
    
    # Get first valid image (non-zero size)
    $srcImage = Get-ChildItem -LiteralPath $trainPath -File -ErrorAction SilentlyContinue | 
                Where-Object { $_.Length -gt 0 -and $_.Extension -match '\.(png|jpg|jpeg)$' } | 
                Select-Object -First 1
    
    if ($null -eq $srcImage) {
        Write-Host "NO_IMAGE_FOUND in $datasetName" -ForegroundColor Yellow
        continue
    }
    
    Write-Host "Selected image: $($srcImage.Name)" -ForegroundColor Green
    
    # Setup output directories
    $outdir = "tmp_smoke\$datasetName"
    $inputDir = Join-Path $outdir 'input'
    $outputDir = Join-Path $outdir 'output'
    $postprocDir = Join-Path $outdir 'postproc'
    $samOutputDir = Join-Path $outdir 'sam_output'
    
    New-Item -ItemType Directory -Force -Path $inputDir | Out-Null
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    New-Item -ItemType Directory -Force -Path $postprocDir | Out-Null
    New-Item -ItemType Directory -Force -Path $samOutputDir | Out-Null
    
    # Copy image using -LiteralPath to handle spaces
    $destPath = Join-Path $inputDir $srcImage.Name
    Copy-Item -LiteralPath $srcImage.FullName -Destination $destPath -Force
    
    # Generate prompt JSON
    $promptJsonPath = Join-Path $outdir 'prompt.json'
    $promptJsonPathFull = Join-Path (Get-Location).Path $promptJsonPath
    $promptText = $config.DefaultPrompt
    
    # Try to use existing dataset JSON if available
    if ($config.PromptJson -and (Test-Path $config.PromptJson)) {
        try {
            $existingJson = Get-Content -LiteralPath $config.PromptJson -Encoding UTF8 | ConvertFrom-Json
            if ($existingJson.PSObject.Properties.Name -contains $srcImage.Name) {
                $promptText = $existingJson.($srcImage.Name)
                Write-Host "Using prompt from $($config.PromptJson): $promptText" -ForegroundColor Magenta
            } else {
                Write-Host "Image not found in JSON, using default prompt: $promptText" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "Could not load JSON, using default prompt: $promptText" -ForegroundColor Yellow
        }
    } else {
        Write-Host "Using default prompt: $promptText" -ForegroundColor Yellow
    }
    
    # Create prompt JSON for this run
    $promptMapping = @{ $srcImage.Name = $promptText }
    $jsonContent = $promptMapping | ConvertTo-Json
    [System.IO.File]::WriteAllText($promptJsonPathFull, $jsonContent, (New-Object System.Text.UTF8Encoding $false))
    
    # Run pipeline
    Write-Host "`n[1/3] Generating saliency map..." -ForegroundColor Blue
    & 'D:\anaconda3\envs\medclipsamv2\python.exe' saliency_maps\generate_saliency_maps.py `
        --input-path $inputDir `
        --output-path $outputDir `
        --model-name BiomedCLIP `
        --finetuned `
        --device cuda `
        --reproduce `
        --json-path $promptJsonPath
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Saliency generation failed for $datasetName" -ForegroundColor Red
        continue
    }
    
    Write-Host "[2/3] Postprocessing..." -ForegroundColor Blue
    & 'D:\anaconda3\envs\medclipsamv2\python.exe' postprocessing\postprocess_saliency_maps.py `
        --sal-path $outputDir `
        --output-path $postprocDir `
        --postprocess kmeans
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Postprocessing failed for $datasetName" -ForegroundColor Red
        continue
    }
    
    Write-Host "[3/3] Running SAM..." -ForegroundColor Blue
    & 'D:\anaconda3\envs\medclipsamv2\python.exe' segment-anything\prompt_sam.py `
        --input $inputDir `
        --mask-input $postprocDir `
        --output $samOutputDir `
        --model-type vit_h `
        --checkpoint segment-anything\sam_checkpoints\sam_vit_h_4b8939.pth `
        --prompts boxes `
        --device cuda
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: SAM failed for $datasetName" -ForegroundColor Red
        continue
    }
    
    # Report outputs
    Write-Host "`nOutputs for $datasetName :" -ForegroundColor Green
    Get-ChildItem $outdir -Recurse -File | ForEach-Object {
        Write-Host "  $($_.FullName.Replace((Get-Location).Path + '\', ''))  ($($_.Length) bytes)" -ForegroundColor Gray
    }
    
    Write-Host "[OK] $datasetName completed successfully" -ForegroundColor Green
}

Write-Host "`n========================================"
Write-Host "Smoke tests completed!" -ForegroundColor Cyan
Write-Host "========================================`n"

# Summary
Write-Host "Summary:" -ForegroundColor Cyan
Get-ChildItem tmp_smoke -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    $ds = $_.Name
    $samMask = Get-ChildItem (Join-Path $_.FullName 'sam_output') -Filter '*.png' -File -ErrorAction SilentlyContinue
    if ($samMask) {
        Write-Host "  [OK] $ds : SAM mask created ($($samMask.Name))" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] $ds : No SAM mask found" -ForegroundColor Red
    }
}
