@echo off
echo ========================================
echo FreqMedCLIP Phase 2 Training
echo ========================================
echo.
echo Activating conda environment: medclipsamv2
call conda activate medclipsamv2

echo.
echo Starting training...
echo Dataset: breast_tumors
echo Epochs: 20
echo Batch size: 4
echo.

python train_freq_fusion.py --dataset breast_tumors --epochs 20 --batch-size 4

echo.
echo ========================================
echo Training completed!
echo Check checkpoints/ folder for results
echo ========================================
pause
