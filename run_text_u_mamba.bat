@echo off
echo ========================================
echo Text-U-Mamba Training (Weakly Supervised)
echo ========================================
echo.
echo Activating conda environment: medclipsamv2
call conda activate medclipsamv2

echo.
echo Starting training...
echo Dataset: breast_tumors (Example)
echo Epochs: 50
echo.

python train_text_u_mamba.py ^
    --datasets breast_tumors ^
    --epochs 50 ^
    --batch-size 4 ^
    --lr 1e-4 ^
    --save-dir checkpoints

echo.
echo ========================================
echo Training completed!
echo ========================================
pause
