@echo off
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
if not exist %PY% echo ERROR: .venv not found next to this script & pause & exit /b 1
echo ============================================================
echo  Run 1/2: Charbonnier-only control  (~2h, loss ablation)
echo ============================================================
%PY% src\train.py --data ..\train --out runs\charb --amp --batch 32 --crop 64 --ch 64 --nb 16 --epochs 40 --iters 500 --workers 6 --loss charbonnier
echo ============================================================
echo  Run 2/2: recentred degradation, combo loss  (~2h)
echo ============================================================
%PY% src\train.py --data ..\train --out runs\recenter --amp --batch 32 --crop 64 --ch 64 --nb 16 --epochs 40 --iters 500 --workers 6 --blur-p 0.15
echo.
echo ALL RUNS FINISHED.
pause
