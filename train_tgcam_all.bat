@echo off
echo ========================================
echo TGCAM Fusion Training - All Datasets
echo ========================================
echo.
echo Activating conda environment: medclipsamv2
call conda activate medclipsamv2

echo.
echo Starting sequential training on 4 datasets...
echo - breast_tumors
echo - brain_tumors
echo - lung_CT
echo - lung_Xray
echo.
echo Epochs per dataset: 100
echo Batch size: 4
echo Learning rate: 1e-4
echo.

python train_tgcam_fusion.py ^
    --datasets breast_tumors brain_tumors lung_CT lung_Xray ^
    --epochs 100 ^
    --batch-size 4 ^
    --lr 0.0001 ^
    --mid-channels 512 ^
    --num-item-iterations 2 ^
    --save-dir checkpoints

echo.
echo ========================================
echo Training completed!
echo Checkpoints saved in:
echo   - checkpoints/tgcam_breast_tumors/
echo   - checkpoints/tgcam_brain_tumors/
echo   - checkpoints/tgcam_lung_CT/
echo   - checkpoints/tgcam_lung_Xray/
echo ========================================
pause
