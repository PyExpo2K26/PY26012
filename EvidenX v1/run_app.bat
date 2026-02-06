@echo off
echo Installing dependencies (if needed)...
python -m pip install -r requirements.txt
echo.
echo Starting EvidenX Backend...
echo Open your browser to: http://localhost:8000
echo.
python main.py
pause
