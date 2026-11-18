@echo off
title PathSnap - Laptop 1 (Model and API Server)
echo ========================================================
echo Starting PathSnap FastAPI Inference Server on Laptop 1...
echo Host: 0.0.0.0  Port: 8000
echo ========================================================
cd /d "%~dp0"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
pause
