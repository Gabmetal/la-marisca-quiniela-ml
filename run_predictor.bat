@echo off
title La Marisca - GPU Raffle Optimizer
echo ============================================================================
echo   LA MARISCA PROJECT - GPU RAFFLE OPTIMIZER ^& PREDICTOR (LOTBA / FORMOSA)
echo   Hardware Target: NVIDIA GeForce RTX 3080 (CUDA) / Hybrid Multi-Engine
echo ============================================================================
echo.

cd /d "%~dp0"

docker info >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [DOCKER DETECTED] Starting GPU-accelerated container on http://localhost:8000 ...
    start "" http://localhost:8000
    docker compose up --build
) else (
    echo [LOCAL RUNTIME] Docker daemon not active. Launching directly on Windows host at http://localhost:8000 ...
    start "" http://localhost:8000
    python src/app.py --host 0.0.0.0 --port 8000
)
pause
