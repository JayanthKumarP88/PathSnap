@echo off
title PathSnap - Laptop 2 (Evaluator Dashboard UI)
cls
echo ======================================================================
echo            PATHSNAP EVALUATOR DASHBOARD (LAPTOP 2)
echo ======================================================================
echo Checking Python dependencies...
python -c "import streamlit, requests" 2>nul
if %errorlevel% neq 0 (
    echo [SETUP] Installing required dependencies (streamlit, requests)...
    pip install streamlit requests
)

set DEFAULT_IP=10.30.204.40
echo.
echo Connecting directly to Laptop 1: http://%DEFAULT_IP%:8000
echo Dashboard Port: 8501
echo.
echo Press ENTER to use default (http://%DEFAULT_IP%:8000)
echo or enter Laptop 1's IP if different:
set USER_IP=
set /p USER_IP="Target IP [%DEFAULT_IP%]: "
if "%USER_IP%"=="" set USER_IP=%DEFAULT_IP%

if not "%USER_IP:~0,4%"=="http" set LAPTOP1_API_URL=http://%USER_IP%:8000
if "%USER_IP:~0,4%"=="http" set LAPTOP1_API_URL=%USER_IP%

echo.
echo [CONNECTING] Target API set to: %LAPTOP1_API_URL%
echo Starting Streamlit Evaluator Dashboard...
echo ======================================================================
cd /d "%~dp0"
python -m streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
pause
