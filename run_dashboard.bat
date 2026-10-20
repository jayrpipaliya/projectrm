@echo off
title Project RM - Market Anomaly & Volatility Engine

REM Check if python is in PATH
python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python run_dashboard.py
    goto end
)

REM Check py launcher
py --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    py run_dashboard.py
    goto end
)

REM Fallback to detected Python 3.12 path
if exist "C:\Users\jayrp\AppData\Local\Programs\Python\Python312\python.exe" (
    "C:\Users\jayrp\AppData\Local\Programs\Python\Python312\python.exe" run_dashboard.py
    goto end
)

echo [!] Python executable not found. Please install Python or add it to PATH.
pause

:end
