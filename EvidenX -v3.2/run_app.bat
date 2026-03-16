@echo off
echo Installing dependencies (if needed)...
python -m pip install -r requirements.txt
echo.
echo Starting EvidenX Backend...
echo Open your browser to: http://localhost:8000
echo.
if not exist mesonet_weights.pth (
    echo Weights file not found. Running initialization...
    python download_weights.py
)
echo.
python main.py
pause
