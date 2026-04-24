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

call :RunOne 1 "python examples/train_gcn_planetoid.py --dataset squirrel --base-model linkx --encoder-fusion none --encoder-pretrain-head linear --prediction-head tabpfn --embedding-branch branch1 --pre-agg-pca-dim 64 --num-layers 2 --dropout 0.5 --split-source random --train-test-ratio 4 --val-size 0.25 --pretrain-epochs 300  --device cuda --tabpfn-device cuda --feature-normalization none"

call :RunOne 2 "python examples/train_gcn_planetoid.py --dataset squirrel --base-model linkx --encoder-fusion none --encoder-pretrain-head linear --prediction-head linear --embedding-branch branch1 --pre-agg-pca-dim 64 --num-layers 2 --dropout 0.5 --split-source random --train-test-ratio 4 --val-size 0.25 --pretrain-epochs 300  --device cuda --tabpfn-device cuda --feature-normalization none"

call :RunOne 3 "python examples/train_gcn_planetoid.py --dataset squirrel --base-model fagcn --encoder-fusion none --encoder-pretrain-head linear --prediction-head tabpfn --embedding-branch branch1 --pre-agg-pca-dim 64 --num-layers 2 --dropout 0.5 --split-source random --train-test-ratio 4 --val-size 0.25 --pretrain-epochs 300  --device cuda --tabpfn-device cuda --feature-normalization none"

call :RunOne 4 "python examples/train_gcn_planetoid.py --dataset squirrel --base-model fagcn --encoder-fusion none --encoder-pretrain-head linear --prediction-head linear --embedding-branch branch1 --pre-agg-pca-dim 64 --num-layers 2 --dropout 0.5 --split-source random --train-test-ratio 4 --val-size 0.25 --pretrain-epochs 300  --device cuda --tabpfn-device cuda --feature-normalization none"

call :RunOne 5 "python examples/train_gcn_planetoid.py --dataset squirrel --base-model gprgnn --encoder-fusion none --encoder-pretrain-head linear --prediction-head tabpfn --embedding-branch branch1 --pre-agg-pca-dim 64 --num-layers 2 --dropout 0.5 --split-source random --train-test-ratio 4 --val-size 0.25 --pretrain-epochs 300  --device cuda --tabpfn-device cuda --feature-normalization none"

call :RunOne 6 "python examples/train_gcn_planetoid.py --dataset squirrel --base-model gprgnn --encoder-fusion none --encoder-pretrain-head linear --prediction-head linear --embedding-branch branch1 --pre-agg-pca-dim 64 --num-layers 2 --dropout 0.5 --split-source random --train-test-ratio 4 --val-size 0.25 --pretrain-epochs 300  --device cuda --tabpfn-device cuda --feature-normalization none"


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
