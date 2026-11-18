@echo off
title PathSnap - Laptop 1 (Model & Inference API Server)
cls
echo ======================================================================
echo           PATHSNAP FASTAPI INFERENCE SERVER (LAPTOP 1)
echo ======================================================================
echo Checking Python dependencies...
python -c "import fastapi, uvicorn, sklearn, pydantic" 2>nul
if %errorlevel% neq 0 (
    echo [SETUP] Installing required dependencies...
    pip install fastapi uvicorn scikit-learn pydantic folium branca
)

set LAPTOP1_IP=10.30.204.40
echo.
echo [SERVER READY]
echo   Laptop 1 Wi-Fi IP : %LAPTOP1_IP%
echo   Local Endpoint    : http://localhost:8000
echo   Remote API Target : http://%LAPTOP1_IP%:8000
echo.
echo ----------------------------------------------------------------------
echo INSTRUCTIONS FOR TEAMMATES (LAPTOP 2):
echo 1. Ensure Laptop 2 is connected to the same Wi-Fi / Hotspot.
echo 2. Run 'run_ui.bat' on Laptop 2 - it will auto-connect directly!
echo ----------------------------------------------------------------------
echo.

cd /d "%~dp0"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
pause
