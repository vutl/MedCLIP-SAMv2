@echo off
cd /d D:\Documents\LMIS\MedCLIP-SAMv2

echo ========================================
echo Running lung_chest_xray smoke test
echo ========================================
D:\anaconda3\envs\medclipsamv2\python.exe tools\run_one_smoke.py lung_chest_xray

echo ========================================
echo Running lung_ct smoke test
echo ========================================
D:\anaconda3\envs\medclipsamv2\python.exe tools\run_one_smoke.py lung_ct

echo ========================================
echo Done! Checking outputs...
echo ========================================
dir /s /b tmp_smoke\lung_chest_xray\sam_output\*.png
dir /s /b tmp_smoke\lung_ct\sam_output\*.png
pause
