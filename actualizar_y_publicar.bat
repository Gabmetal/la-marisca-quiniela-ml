@echo off
chcp 65001 >nul
echo ========================================================
echo   LA MARISCA - ACTUALIZADOR Y PUBLICADOR GPU RTX 3080
echo ========================================================
echo.

cd /d "%~dp0"

echo [1/3] Sincronizando últimos sorteos y re-entrenando en GPU RTX 3080...
docker exec la-marisca-gpu-predictor python -c "from src.gpu_engine import RaffleGPUPredictor; p = RaffleGPUPredictor(); added = p.sync_latest_draws(days_back=7); print(f'Nuevos sorteos agregados: {added}')"
docker exec la-marisca-gpu-predictor python src/export_static_state.py

echo.
echo [2/3] Verificando cambios en el repositorio...
git add data/ docs/
git diff --staged --quiet
if %errorlevel% equ 0 (
    echo No hay nuevos sorteos ni cambios pendientes. El modelo ya está al día.
    pause
    exit /b 0
)

echo.
echo [3/3] Publicando modelo actualizado a GitHub Pages...
git commit -m "chore(sync): actualizar sorteos y modelo GPU RTX 3080"
git push origin main

echo.
echo ========================================================
echo   ¡LISTO! Modelo publicado con éxito en:
echo   https://gabmetal.github.io/la-marisca-quiniela-ml/
echo ========================================================
pause
