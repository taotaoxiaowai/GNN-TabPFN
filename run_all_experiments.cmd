@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Always run relative to this script's directory
cd /d "%~dp0"

set "LOG_DIR=logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"
set "MASTER_LOG=%LOG_DIR%\run_%TS%.log"

echo =============================================== > "%MASTER_LOG%"
echo Run started at %date% %time% >> "%MASTER_LOG%"
echo Working directory: %cd% >> "%MASTER_LOG%"
echo =============================================== >> "%MASTER_LOG%"
echo.
echo [INFO] Master log: "%MASTER_LOG%"
echo.


call :RunOne 1 "python examples/train_gcn_planetoid.py --dataset Photo --encoder-fusion weighted --weighted-fusion-train-strategy two-stage --prediction-head tabpfn-ensemble-selection --embedding-branch branch2 --train-test-ratio 4 --val-size 0.25 --split-source random --pretrain-epochs 300 --fusion-epochs 200 --device cuda --tabpfn-device cuda --feature-normalization none --tabpfn-ens-candidates-per-table 2 --tabpfn-ens-colsample-min-rate 1.0 --tabpfn-ens-sources all-backbones"

call :RunOne 2 "python examples/train_gcn_planetoid.py --dataset PubMed --encoder-fusion weighted --weighted-fusion-train-strategy two-stage --prediction-head tabpfn-ensemble-selection --embedding-branch branch2 --train-test-ratio 4 --val-size 0.25 --split-source random --pretrain-epochs 300 --fusion-epochs 200 --device cuda --tabpfn-device cuda --feature-normalization none --tabpfn-ens-candidates-per-table 2 --tabpfn-ens-colsample-min-rate 1.0 --tabpfn-ens-sources all-backbones"


echo [INFO] All commands finished successfully.
echo [INFO] All commands finished successfully. >> "%MASTER_LOG%"
goto :final

:RunOne
set "IDX=%~1"
set "CMD=%~2"
set "STEP_LOG=%LOG_DIR%\run_%TS%_step%IDX%.log"

echo [INFO] Running step %IDX% ...
echo. >> "%MASTER_LOG%"
echo ---------- STEP %IDX% START %date% %time% ---------- >> "%MASTER_LOG%"
echo Command: !CMD! >> "%MASTER_LOG%"
echo [INFO] Step %IDX% log: "!STEP_LOG!"

cmd /c "!CMD!" > "!STEP_LOG!" 2>&1
set "RC=!ERRORLEVEL!"

if not "!RC!"=="0" (
  echo [ERROR] Step %IDX% failed with exit code !RC!.
  echo [ERROR] Step %IDX% failed with exit code !RC!. >> "%MASTER_LOG%"
) else (
  echo [INFO] Step %IDX% completed successfully.
  echo [INFO] Step %IDX% completed successfully. >> "%MASTER_LOG%"
)

echo ---------- STEP %IDX% OUTPUT BEGIN ---------- >> "%MASTER_LOG%"
type "!STEP_LOG!" >> "%MASTER_LOG%"
echo ---------- STEP %IDX% OUTPUT END ------------ >> "%MASTER_LOG%"
echo ---------- STEP %IDX% END %date% %time% ---------- >> "%MASTER_LOG%"

exit /b !RC!

:end
echo [WARN] Execution stopped because a step failed.
echo [WARN] Execution stopped because a step failed. >> "%MASTER_LOG%"

:final
echo.
echo [INFO] Finished. Master log: "%MASTER_LOG%"
echo Run finished at %date% %time% >> "%MASTER_LOG%"
echo =============================================== >> "%MASTER_LOG%"
endlocal
exit /b
