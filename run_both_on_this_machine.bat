@echo off
title PathSnap - Launch Both Servers Locally (Laptop 1 + Laptop 2)
cls
echo ======================================================================
echo           PATHSNAP: LAUNCHING BOTH SERVERS (LOCAL TEST MODE)
echo ======================================================================
echo Detecting Wi-Fi network configuration...

set CURRENT_IP=10.30.203.19
for /f "tokens=*" %%a in ('powershell -command "(Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias '*Wi-Fi*').IPAddress" 2^>nul') do (
    if not "%%a"=="" set CURRENT_IP=%%a
)

echo Active Wi-Fi IP : %CURRENT_IP%
echo Laptop 1 API    : http://localhost:8000 (and http://%CURRENT_IP%:8000)
echo Laptop 2 UI     : http://localhost:8501
echo ======================================================================
echo.

echo [1/2] Starting Laptop 1 FastAPI Inference Server in new window...
start "PathSnap - Laptop 1 (Model and Inference API Server)" cmd /k "cd /d %~dp0 && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"

timeout /t 2 /nobreak >nul

echo [2/2] Starting Laptop 2 Streamlit Dashboard UI in new window...
set LAPTOP1_API_URL=http://localhost:8000
start "PathSnap - Laptop 2 (Evaluator Dashboard UI)" cmd /k "cd /d %~dp0 && python -m streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0"

timeout /t 3 /nobreak >nul

echo.
echo ======================================================================
echo [SUCCESS] Both Laptop 1 and Laptop 2 are now running!
echo   Dashboard UI: http://localhost:8501
echo   FastAPI Engine: http://localhost:8000
echo.
echo Opening dashboard in your default browser...
start http://localhost:8501
echo ======================================================================
echo Press any key to close this launcher (servers will keep running in their own windows).
pause >nul
