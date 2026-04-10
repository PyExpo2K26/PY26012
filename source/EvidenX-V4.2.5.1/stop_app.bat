@echo off
setlocal
title EvidenX Forensic Unit - Shutdown

echo Stopping EvidenX Backend (Port 8000)...
set "process_found=0"

for /f "tokens=5" %%a in ('netstat -aon ^| find "LISTENING" ^| find ":8000"') do (
    echo Found existing process PID %%a. Terminating...
    taskkill /F /PID %%a
    set "process_found=1"
)

if "%process_found%"=="0" (
    echo No process found running on port 8000.
) else (
    echo Successfully stopped the application.
)

pause 
