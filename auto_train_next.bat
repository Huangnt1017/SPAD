@echo off
setlocal enabledelayedexpansion

set "LOG_FILE=D:\PYproject\SPAD\logs\train_pointtransv2_20260528_005554_877505.log"
set "PYTHON=D:\anaconda3\envs\pytorch\python.exe"
set "TRAIN_PY=D:\PYproject\SPAD\scripts\train.py"
set "PYTHONPATH=D:\PYproject\SPAD"

echo [%date% %time%] Monitoring log: %LOG_FILE%
echo [%date% %time%] Will auto-start pointtransformer (B=32, epochs=100) when done.

:loop
REM Check if training finished
findstr /C:"Training finished" "%LOG_FILE%" >nul 2>&1
if %errorlevel% equ 0 (
    echo [%date% %time%] PointTransV2 training completed!
    goto start_next
)

REM Check if process still running
tasklist /FI "IMAGENAME eq python.exe" /FO CSV 2>nul | findstr /C:"python.exe" >nul 2>&1
if %errorlevel% neq 0 (
    echo [%date% %time%] WARNING: python.exe not found running, but no 'Training finished' yet.
    echo [%date% %time%] Checking log one more time...
    timeout /t 30 /nobreak >nul
    findstr /C:"Training finished" "%LOG_FILE%" >nul 2>&1
    if %errorlevel% equ 0 (
        echo [%date% %time%] Training finished detected on re-check.
        goto start_next
    )
    echo [%date% %time%] Training appears to have crashed or was killed. Exiting.
    exit /b 1
)

timeout /t 30 /nobreak >nul
goto loop

:start_next
echo.
echo ============================================================
echo [%date% %time%] Starting PointTransformer training...
echo [%date% %time%] Config: model=pointtransformer, batch_size=32, epochs=100
echo ============================================================
echo.
set "PYTHONPATH=%PYTHONPATH%"
"%PYTHON%" "%TRAIN_PY%" --model pointtransformer --batch-size 32 --epochs 100

echo.
echo [%date% %time%] PointTransformer training exited with code %errorlevel%
endlocal
