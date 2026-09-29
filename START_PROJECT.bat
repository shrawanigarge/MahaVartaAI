@echo off
title Marathi News Classification - Web Server
cd /d "%~dp0"
echo =====================================================================
echo    Marathi News Classification Using Natural Language Processing
echo =====================================================================
echo.
echo [1/2] Opening web browser at http://127.0.0.1:5000 ...
start http://127.0.0.1:5000
echo.
echo [2/2] Starting Flask Web Server on http://127.0.0.1:5000 ...
echo Press Ctrl+C in this terminal window to stop the server anytime.
echo.
python app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Server stopped with error code %ERRORLEVEL%.
    pause
)
