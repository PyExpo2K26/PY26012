@echo off
setlocal
cd /d "%~dp0"
title EvidenX Forensic Unit - Startup

echo [1/4] Checking for existing process on port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| find "LISTENING" ^| find ":8000"') do (
    echo Found existing process PID %%a. Clearing...
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 1 /nobreak >nul

echo [2/4] Initializing Environment...
set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" (
    echo.  - Using virtual environment (.venv)
    set "PYTHON_EXE=.venv\Scripts\python.exe"
)

echo [3/4] Installing dependencies (if needed)...
%PYTHON_EXE% -m pip install -q -r requirements.txt

echo [4/4] Starting EvidenX Backend...
echo.
echo ==========================================================
echo  EvidenX v4.5 is starting...
echo  UI will launch automatically at http://localhost:8000
echo ==========================================================
echo.

if not exist mesonet_weights.pth (
    echo.  - Weights file missing. Downloading...
    %PYTHON_EXE% download_weights.py
)

:: Launch browser in background after a short delay
start "" "http://localhost:8000"

:: Start main server with warnings suppressed for cleaner forensics log
%PYTHON_EXE% -W ignore main.py
pause
