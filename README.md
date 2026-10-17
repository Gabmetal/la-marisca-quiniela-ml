# 🎰 La Marisca — Quiniela Chaos & GPU Raffle Optimizer (`00`–`99`)

[![Live Serverless App](https://img.shields.io/badge/Live_App-GitHub_Pages_(Serverless)-10b981?style=for-the-badge&logo=github)](https://gabmetal.github.io/la-marisca-quiniela-ml/)
[![GPU Engine](https://img.shields.io/badge/GPU_Trained-NVIDIA_RTX_3080_(CUDA)-6366f1?style=for-the-badge&logo=nvidia)](./src/gpu_engine.py)

Sistema experimental de **Machine Learning, Inferencia Bayesiana, Cadenas de Markov, Simulación Monte Carlo (1,000,000 rollouts) y Optimización Combinatoria Ortogonal** diseñado para maximizar la cobertura de probabilidades en rifas basadas en **2 - LOTBA (Quiniela de la Ciudad Nocturna 21:00 ART)** y **Quiniela de Formosa**.

---

## 🌐 1. Versión 100% Serverless (Gratis y Sin Backend en GitHub Pages)

👉 **Abrir aplicación en vivo:** **[https://gabmetal.github.io/la-marisca-quiniela-ml/](https://gabmetal.github.io/la-marisca-quiniela-ml/)**

La edición web (`docs/index.html`) corre **100% en tu navegador sin necesitar servidor backend**:
- **Pesos GPU Pre-Entrenados (`docs/data/state.json`)**: Exportados desde `XGBoost 3.2.0 (CUDA GPU)` + `Deep ResNet MLP` sobre `NVIDIA GeForce RTX 3080` con los 185 sorteos históricos (`2026-07-01` al `2026-10-16`).
- **Clima en Vivo vía CORS (`Open-Meteo API`) + Efemérides Lunares Exactas**: El navegador consulta directamente la telemetría meteorológica a las 21:00 ART para cualquier fecha y calcula la fase sinódica lunar ($29.53058867$ días).
- **1,000,000 de Simulaciones Monte Carlo en el Navegador**: Ejecutadas en ~15–20 ms en `TypedArrays` (`Float64Array` / `Uint32Array`) cada vez que cambias pesos, tachas números vendidos en el grupo de WhatsApp o agregas un sorteo nuevo.
- **Base de Datos Local Persistente (`localStorage`) + GitHub Actions Nightly Cron**: Puedes cargar sorteos nuevos manualmente en la UI o dejar que el workflow automático de GitHub Actions ([`.github/workflows/deploy-pages.yml`](./.github/workflows/deploy-pages.yml)) sincronice los sorteos cada noche tras el sorteo de las 21:00 ART.

---

## 🐳 2. Ejecución Local con Docker + GPU NVIDIA (`RTX 3080`)

Si deseas correr el motor completo en tu PC con aceleración CUDA y base de datos SQLite en vivo (`data/quiniela_marisca.db`):

```bash
docker compose up -d --build
```

Luego abre **`http://localhost:8000`** en tu navegador.

---

## 📂 Estructura del Proyecto

```text
├── .github/workflows/
│   └── deploy-pages.yml       # Auto-Deploy a GitHub Pages + Cron nocturno de sorteos
├── data/
│   ├── quiniela_marisca.db    # Base de datos SQLite persistente (draws + predictions_log)
│   ├── raw_draws.csv          # Sorteos Nocturnos crudos (Ciudad y Formosa)
│   ├── dataset_enriched.csv   # Dataset enriquecido (Clima 21h + Luna + Lags + Atrasos)
│   └── state.json             # Estado exportado del ensamble GPU
├── docs/                      # App Web 100% Serverless (GitHub Pages)
│   ├── index.html             # Motor interactivo en navegador (1M MC + Rook Optimizer)
│   ├── data/state.json        # Vectores de probabilidad GPU + historial
│   └── charts/                # Gráficos científicos PNG
├── src/
│   ├── app.py                 # Servidor local Starlette/Uvicorn + REST API
│   ├── gpu_engine.py          # Motor Híbrido de 5 Pilares (CUDA XGBoost / PyTorch / MLP)
│   ├── database.py            # Gestor SQLite + Auto-Sync Web + Open-Meteo
│   ├── export_static_state.py # Exportador de pesos GPU hacia docs/data/state.json
│   ├── scrape_draws.py        # Scraper de sorteos Nocturnos (notitimba / ruta1000)
│   ├── enrich_features.py     # Ingeniería de variables meteorológicas, lunares y lags
│   ├── statistical_audit.py   # Auditoría Chi², Test de Rachas y Entropía de Shannon
│   └── train_and_predict.py   # Validación Walk-Forward TimeSeriesSplit y generación de gráficos
├── outputs/
│   ├── REPORT.md              # Informe técnico completo
│   └── charts/                # Gráficos de alta resolución
├── Dockerfile                 # Imagen con soporte CUDA GPU
└── docker-compose.yml         # Orquestación con passthrough NVIDIA GPU y volúmenes
```
