@echo off
title PathSnap - Laptop 2 (Evaluator User Interface)
echo ========================================================
echo Starting PathSnap Evaluator Dashboard on Laptop 2...
echo Connecting directly to Laptop 1: http://10.30.204.40:8000
echo Port: 8501
echo ========================================================
set LAPTOP1_API_URL=http://10.30.204.40:8000
cd /d "%~dp0"
python -m streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
pause
