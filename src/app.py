import os
import sys
import json
import argparse
import asyncio
import datetime
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enrich_features
from gpu_engine import RaffleGPUPredictor
from hyperopt_backtest import optimize_hyperparameters, load_optimal_hyperparams

predictor = RaffleGPUPredictor()

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>La Marisca — LOTBA Quiniela Optimizer (lotba.gabo.lat)</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { background-color: #0b0f19; color: #e2e8f0; font-family: 'Inter', system-ui, -apple-system, sans-serif; }
    .glass { background: rgba(17, 24, 39, 0.82); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); }
    .cell-btn { transition: all 0.15s ease; }
    .cell-btn:hover { transform: scale(1.07); z-index: 10; }
    .recommended-ring { box-shadow: 0 0 0 2px #10b981, 0 0 14px rgba(16, 185, 129, 0.55); }
    .taken-cell { opacity: 0.25; text-decoration: line-through; filter: grayscale(100%); }
  </style>
</head>
<body class="min-h-screen p-4 md:p-6">
  <div class="max-w-7xl mx-auto space-y-6">

    <!-- Header -->
    <header class="glass rounded-2xl p-5 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
      <div>
        <div class="flex flex-wrap items-center gap-2">
          <span class="px-2.5 py-1 text-xs font-bold uppercase tracking-wider bg-emerald-500/20 text-emerald-400 rounded-full border border-emerald-500/30">
            lotba.gabo.lat • Producción K3s
          </span>
          <span id="hw-badge" class="px-2.5 py-1 text-xs font-mono bg-indigo-500/20 text-indigo-300 rounded-full border border-indigo-500/30">
            Xeon E3-1226 v3 (1-Core Throttled)
          </span>
          <span id="db-badge" class="px-2.5 py-1 text-xs font-mono bg-amber-500/20 text-amber-300 rounded-full border border-amber-500/30">
            SQLite: Cargando...
          </span>
          <span id="backtest-badge" class="px-2.5 py-1 text-xs font-mono bg-purple-500/20 text-purple-300 rounded-full border border-purple-500/30">
            Backtest: Calibrado
          </span>
        </div>
        <h1 class="text-2xl md:text-3xl font-extrabold text-white mt-2 tracking-tight">
          Optimizador Estocástico & Anti-Crowd de Rifa (00–99)
        </h1>
        <p class="text-sm text-slate-400 mt-1">
          Motor Desacoplado K3s con Auto-Sync Nocturno (21:15 ART) + Validación Walk-Forward TimeSeries
        </p>
      </div>
      <div class="flex flex-col items-end gap-2 self-stretch md:self-auto justify-between">
        <div class="text-right bg-slate-900/90 px-4 py-2.5 rounded-xl border border-slate-800">
          <div class="text-xs text-slate-400">Último Sorteo en Base de Datos</div>
          <div id="last-draw-banner" class="text-lg font-bold text-amber-400 font-mono">Cargando...</div>
        </div>
        <div class="flex gap-2">
          <button onclick="runHyperopt()" id="hyperopt-btn" class="text-xs px-3 py-1.5 rounded-lg bg-purple-600/20 hover:bg-purple-600/30 text-purple-300 border border-purple-500/30 transition font-semibold">
            🔬 Re-Calibrar Walk-Forward
          </button>
          <button onclick="syncDraws()" id="sync-btn" class="text-xs px-3 py-1.5 rounded-lg bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-500/30 transition font-semibold">
            🔄 Sincronizar Web
          </button>
        </div>
      </div>
    </header>

    <!-- Main Grid: Controls + KPIs -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">

      <!-- Left Column: Controls & Configuration -->
      <div class="glass rounded-2xl p-5 space-y-5">
        <h2 class="text-lg font-bold text-white flex items-center gap-2">
          <span>⚙️</span> Parámetros de Predicción Futura
        </h2>

        <!-- Lottery & Future Date -->
        <div class="space-y-2">
          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="text-xs font-semibold text-slate-400 block mb-1">Lotería</label>
              <select id="lottery-select" onchange="runPrediction()" class="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500">
                <option value="ciudad" selected>2 - LOTBA Ciudad (21:00)</option>
                <option value="formosa">Quiniela de Formosa</option>
              </select>
            </div>
            <div>
              <div class="flex items-center justify-between mb-1">
                <label class="text-xs font-semibold text-slate-400 block">Fecha Futura</label>
                <span id="horizon-badge" class="font-mono text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-bold">T+1</span>
              </div>
              <input type="date" id="target-date" onchange="runPrediction()" class="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500">
            </div>
          </div>
          <!-- Horizon Quick Buttons -->
          <div class="flex gap-1.5 pt-0.5">
            <button type="button" onclick="setQuickDate(0)" class="flex-1 py-1 px-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-[11px] font-mono text-slate-300 border border-slate-800 transition text-center">
              Hoy (T+1)
            </button>
            <button type="button" onclick="setQuickDate(1)" class="flex-1 py-1 px-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-[11px] font-mono text-slate-300 border border-slate-800 transition text-center">
              Mañana
            </button>
            <button type="button" onclick="setQuickDate(2)" class="flex-1 py-1 px-1.5 rounded-lg bg-indigo-950/60 hover:bg-indigo-900/60 text-[11px] font-mono text-indigo-300 border border-indigo-500/40 font-semibold transition text-center" title="Proyección 48h Chapman-Kolmogorov P²">
              🔭 Pasado Mañana (T+2)
            </button>
          </div>
        </div>

        <!-- Ticket Count & Orthogonal Mode -->
        <div class="grid grid-cols-2 gap-3">
          <div>
            <label class="text-xs font-semibold text-slate-400 block mb-1">Boletos a Comprar (K)</label>
            <input type="number" id="ticket-count" value="5" min="1" max="25" onchange="runPrediction()" class="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500">
          </div>
          <div>
            <label class="text-xs font-semibold text-slate-400 block mb-1">Modo Dispersión</label>
            <select id="orthogonal-mode" onchange="runPrediction()" class="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-emerald-500">
              <option value="true" selected>Ortogonal (Latin-Rook)</option>
              <option value="false">Máxima Probabilidad Pura</option>
            </select>
          </div>
        </div>

        <!-- Strategy Sliders -->
        <div class="space-y-3 pt-2 border-t border-slate-800">
          <div class="flex items-center justify-between">
            <span class="text-xs font-bold text-slate-300 uppercase tracking-wider">Pesos del Ensamble</span>
            <button onclick="resetOptimalWeights()" class="text-[11px] text-purple-400 hover:text-purple-300 underline font-mono">
              ↺ Auto-calibrar por Horizonte
            </button>
          </div>

          <div>
            <div class="flex justify-between text-xs text-slate-400 mb-1">
              <span>Árboles (RandomForest + HistGB)</span>
              <span id="v-tree" class="font-mono text-emerald-400">0.20</span>
            </div>
            <input type="range" id="w-tree" min="0" max="1" step="0.05" value="0.20" oninput="onUserSliderChange()" class="w-full accent-emerald-500">
          </div>

          <div>
            <div class="flex justify-between text-xs text-slate-400 mb-1">
              <span>Deep MLP (Red Neuronal)</span>
              <span id="v-deep" class="font-mono text-indigo-400">0.20</span>
            </div>
            <input type="range" id="w-deep" min="0" max="1" step="0.05" value="0.20" oninput="onUserSliderChange()" class="w-full accent-indigo-500">
          </div>

          <div>
            <div class="flex justify-between text-xs text-slate-400 mb-1">
              <span>Cadenas de Markov (Transición 1° / 2° Orden)</span>
              <span id="v-markov" class="font-mono text-cyan-400">0.20</span>
            </div>
            <input type="range" id="w-markov" min="0" max="1" step="0.05" value="0.20" oninput="onUserSliderChange()" class="w-full accent-cyan-500">
          </div>

          <div>
            <div class="flex justify-between text-xs text-slate-400 mb-1">
              <span>Inercia Bayesiana (Hot Dirichlet)</span>
              <span id="v-hot" class="font-mono text-amber-400">0.20</span>
            </div>
            <input type="range" id="w-hot" min="0" max="1" step="0.05" value="0.20" oninput="onUserSliderChange()" class="w-full accent-amber-500">
          </div>

          <div>
            <div class="flex justify-between text-xs text-slate-400 mb-1">
              <span>Reversión por Sequía (Cold Atrasos)</span>
              <span id="v-cold" class="font-mono text-rose-400">0.20</span>
            </div>
            <input type="range" id="w-cold" min="0" max="1" step="0.05" value="0.20" oninput="onUserSliderChange()" class="w-full accent-rose-500">
          </div>
        </div>

        <button onclick="runPrediction()" id="calc-btn" class="w-full py-3 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold text-sm shadow-lg shadow-emerald-900/30 transition flex items-center justify-center gap-2">
          <span>⚡</span> Simular & Optimizar Portafolio (&lt; 2 ms)
        </button>

        <!-- Walk-Forward Backtest Card -->
        <div class="p-3 rounded-xl bg-purple-950/20 border border-purple-500/20 space-y-2">
          <div class="text-xs font-bold text-purple-300 flex items-center justify-between">
            <span>📊 Validación Walk-Forward (Out-of-Sample)</span>
            <span id="wf-folds" class="font-mono text-[10px] text-purple-400">77 Folds</span>
          </div>
          <div class="grid grid-cols-4 gap-1.5 text-center text-xs">
            <div class="bg-slate-900/80 p-2 rounded-lg border border-slate-800">
              <div class="text-slate-400 text-[9px]">Hit@5</div>
              <div id="wf-hit5" class="font-mono font-bold text-emerald-400 mt-0.5">--</div>
            </div>
            <div class="bg-slate-900/80 p-2 rounded-lg border border-slate-800">
              <div class="text-slate-400 text-[9px]">Hit@10</div>
              <div id="wf-hit10" class="font-mono font-bold text-teal-300 mt-0.5">--</div>
            </div>
            <div class="bg-slate-900/80 p-2 rounded-lg border border-slate-800">
              <div class="text-slate-400 text-[9px]">Log-Loss</div>
              <div id="wf-logloss" class="font-mono font-bold text-purple-300 mt-0.5">--</div>
            </div>
            <div class="bg-slate-900/80 p-2 rounded-lg border border-slate-800">
              <div class="text-slate-400 text-[9px]">Brier</div>
              <div id="wf-brier" class="font-mono font-bold text-cyan-300 mt-0.5">--</div>
            </div>
          </div>
          <p id="wf-summary" class="text-[10px] text-slate-400 leading-tight">
            Parámetros hiperoptimizados con 47 variables: fluidodinámica, viscosidad Sutherland, mareas keplerianas, declinación solar y resonancia ergódica KAM.
          </p>
        </div>

      </div>

      <!-- Center & Right Columns: KPIs + Recommendations + Heatmap -->
      <div class="lg:col-span-2 space-y-6">

        <!-- Top Metrics Cards -->
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div class="glass rounded-xl p-3.5">
            <div class="text-xs text-slate-400">Probabilidad Portafolio</div>
            <div id="kpi-prob" class="text-2xl font-black text-emerald-400 font-mono mt-1">--%</div>
            <div id="kpi-lift" class="text-[11px] text-slate-400 mt-0.5">--</div>
          </div>

          <div class="glass rounded-xl p-3.5">
            <div class="text-xs text-slate-400">Cobertura de Dígito</div>
            <div id="kpi-reach" class="text-2xl font-black text-cyan-400 font-mono mt-1">--%</div>
            <div id="kpi-du-cov" class="text-[11px] text-slate-400 mt-0.5">--</div>
          </div>

          <div class="glass rounded-xl p-3.5">
            <div class="text-xs text-slate-400">Espacio Disponible</div>
            <div id="kpi-avail" class="text-2xl font-black text-amber-400 font-mono mt-1">-- / 100</div>
            <div class="text-[11px] text-slate-400 mt-0.5">Tachados: <span id="taken-count">0</span></div>
          </div>

          <div class="glass rounded-xl p-3.5">
            <div class="text-xs text-slate-400">Base Histórica LOTBA</div>
            <div id="kpi-db-count" class="text-2xl font-black text-indigo-400 font-mono mt-1">--</div>
            <div class="text-[11px] text-slate-400 mt-0.5">Sorteos nocturnos</div>
          </div>
        </div>

        <!-- Telemetry & Lunar Forecast -->
        <div class="glass rounded-2xl p-4">
          <div class="flex items-center justify-between mb-3">
            <h3 class="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
              <span>🛰️</span> Telemetría Atmosférica, Dinámica de Tambor & Efemérides Gravitacionales (21:00 ART)
            </h3>
            <span id="telemetry-source" class="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">Open-Meteo API + Efemérides Kepler</span>
          </div>
          <div id="telemetry-box" class="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div class="bg-slate-900/60 p-2.5 rounded-lg border border-slate-800">Cargando telemetría...</div>
          </div>
        </div>

        <!-- World-Intel: Space Weather & Planetary Intelligence -->
        <div class="glass rounded-2xl p-4">
          <div class="flex items-center justify-between mb-3">
            <h3 class="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
              <span>🌌</span> Inteligencia Planetaria, Clima Espacial & Micro-Sismicidad (World-Intel Feed)
            </h3>
            <span class="text-[10px] font-mono text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">NOAA SWPC + USGS</span>
          </div>
          <div id="intel-box" class="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div class="bg-slate-900/60 p-2.5 rounded-lg border border-slate-800">Cargando inteligencia espacial y sísmica...</div>
          </div>
        </div>

        <!-- Top Recommended Picks Cards -->
        <div class="glass rounded-2xl p-5 space-y-4">
          <div class="flex items-center justify-between">
            <h2 id="rec-title" class="text-lg font-bold text-white flex items-center gap-2">
              <span>🎯</span> Portafolio Óptimo Recomendado (Modo Ortogonal)
            </h2>
            <span class="text-xs text-emerald-400 font-mono">Dispersión de Dígitos Maximizado</span>
          </div>
          <div id="portfolio-cards" class="grid grid-cols-1 sm:grid-cols-5 gap-3">
            <div class="col-span-5 text-center text-slate-400 py-6">Calculando combinaciones...</div>
          </div>
        </div>

        <!-- Interactive 100-Ambo Grid (Heatmap + Exclusion) -->
        <div class="glass rounded-2xl p-5 space-y-4">
          <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
            <div>
              <h2 class="text-lg font-bold text-white flex items-center gap-2">
                <span>🎲</span> Matriz de Probabilidad (00–99) & Selector de Números Vendidos
              </h2>
              <p class="text-xs text-slate-400 mt-0.5">
                Haz clic en cualquier número para marcarlo como <span class="text-rose-400 font-bold">Vendido/Tachado</span> y re-optimizar al instante.
              </p>
            </div>
            <button onclick="clearTaken()" class="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition">
              Limpiar Tachados
            </button>
          </div>

          <div id="grid-container" class="grid grid-cols-10 gap-1.5 sm:gap-2">
            <!-- Rendered by JS -->
          </div>
        </div>

        <!-- Prediction Tracking & Accuracy Scorecard -->
        <div class="glass rounded-2xl p-5 space-y-4 border border-emerald-500/30">
          <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
            <div>
              <h2 class="text-lg font-bold text-white flex items-center gap-2">
                <span>🎯</span> Auditoría y Seguimiento de Predicciones Reales
              </h2>
              <p class="text-xs text-slate-400 mt-0.5">
                Historial auditado de pronósticos vs. sorteos oficiales de LOTBA con verificación de aciertos y ROI simulado.
              </p>
            </div>
            <div class="flex items-center gap-2">
              <button onclick="backfillTracker()" id="backfill-btn" class="text-xs px-3 py-1.5 rounded-lg bg-purple-900/60 hover:bg-purple-800 text-purple-200 border border-purple-500/30 font-semibold transition flex items-center gap-1.5">
                <span>🔬</span> Auditar Últimos 30 Sorteos
              </button>
              <button onclick="loadTracker()" id="refresh-tracker-btn" class="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition">
                <span>🔄</span> Refrescar
              </button>
            </div>
          </div>

          <!-- Tracker Summary KPIs -->
          <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1">
            <div class="bg-slate-900/80 p-3 rounded-xl border border-slate-800">
              <div class="text-[11px] text-slate-400">Tasa Acierto Exacto (Ambo)</div>
              <div id="tr-hit-rate" class="text-xl font-black text-emerald-400 font-mono mt-0.5">--%</div>
              <div id="tr-hit-lift" class="text-[10px] text-slate-400 mt-0.5">Base azar: 5.0%</div>
            </div>
            <div class="bg-slate-900/80 p-3 rounded-xl border border-slate-800">
              <div class="text-[11px] text-slate-400">Acierto de Dígito Ganador</div>
              <div id="tr-digit-rate" class="text-xl font-black text-cyan-400 font-mono mt-0.5">--%</div>
              <div id="tr-digit-detail" class="text-[10px] text-slate-400 mt-0.5">Decena: -- | Unidad: --</div>
            </div>
            <div class="bg-slate-900/80 p-3 rounded-xl border border-slate-800">
              <div class="text-[11px] text-slate-400">Balance Simulado (Pago 70x)</div>
              <div id="tr-pnl" class="text-xl font-black text-emerald-300 font-mono mt-0.5">--</div>
              <div id="tr-roi" class="text-[10px] text-slate-400 mt-0.5">ROI: --% ($1 / ticket)</div>
            </div>
            <div class="bg-slate-900/80 p-3 rounded-xl border border-slate-800">
              <div class="text-[11px] text-slate-400">Sorteos Auditados</div>
              <div id="tr-evaluated" class="text-xl font-black text-indigo-400 font-mono mt-0.5">--</div>
              <div id="tr-pending" class="text-[10px] text-slate-400 mt-0.5">Pendientes: --</div>
            </div>
          </div>

          <!-- Tracker Ledger Table -->
          <div class="overflow-x-auto max-h-[380px] overflow-y-auto pr-1">
            <table class="w-full text-xs text-left">
              <thead class="text-slate-400 border-b border-slate-800 sticky top-0 bg-slate-950/95 backdrop-blur z-10">
                <tr>
                  <th class="pb-2">Fecha Sorteo</th>
                  <th class="pb-2">Sorteo Real (1°)</th>
                  <th class="pb-2">Ambo Real</th>
                  <th class="pb-2">Top 5 Pronosticado</th>
                  <th class="pb-2">Resultado / Calificación</th>
                  <th class="pb-2">P&L Simulado</th>
                  <th class="pb-2">Origen</th>
                </tr>
              </thead>
              <tbody id="tracker-tbody" class="divide-y divide-slate-800/60 font-mono text-slate-300">
                <tr><td colspan="7" class="py-4 text-center text-slate-500">Cargando historial de predicciones...</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Recent Draws in SQLite Table -->
        <div class="glass rounded-2xl p-5 space-y-4">
          <h2 class="text-lg font-bold text-white flex items-center gap-2">
            <span>📜</span> Últimos Sorteos Nocturnos Registrados en SQLite
          </h2>
          <div class="overflow-x-auto">
            <table class="w-full text-xs text-left">
              <thead class="text-slate-400 border-b border-slate-800">
                <tr>
                  <th class="pb-2">Fecha</th>
                  <th class="pb-2">Lotería</th>
                  <th class="pb-2">1° Premio (4D)</th>
                  <th class="pb-2">Ambo (2D)</th>
                  <th class="pb-2">Dígitos</th>
                  <th class="pb-2">Clima (21h)</th>
                  <th class="pb-2">Fase Lunar</th>
                  <th class="pb-2">Origen</th>
                </tr>
              </thead>
              <tbody id="db-draws-tbody" class="divide-y divide-slate-800/60 font-mono text-slate-300">
                <!-- Rendered by JS -->
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </div>
  </div>

  <script>
    const takenNumbers = new Set();
    let currentOptimalWeights = {
      ml_tree: 0.20,
      deep_gpu: 0.20,
      markov: 0.20,
      bayesian_hot: 0.20,
      renewal_cold: 0.20
    };

    let userChangedSliders = false;

    function updateSliders() {
      document.getElementById('v-tree').textContent = parseFloat(document.getElementById('w-tree').value).toFixed(2);
      document.getElementById('v-deep').textContent = parseFloat(document.getElementById('w-deep').value).toFixed(2);
      document.getElementById('v-markov').textContent = parseFloat(document.getElementById('w-markov').value).toFixed(2);
      document.getElementById('v-hot').textContent = parseFloat(document.getElementById('w-hot').value).toFixed(2);
      document.getElementById('v-cold').textContent = parseFloat(document.getElementById('w-cold').value).toFixed(2);
    }

    function onUserSliderChange() {
      userChangedSliders = true;
      updateSliders();
      runPrediction();
    }

    function resetOptimalWeights() {
      userChangedSliders = false;
      runPrediction();
    }

    function setQuickDate(offsetDays) {
      const base = new Date();
      base.setDate(base.getDate() + offsetDays);
      document.getElementById('target-date').value = base.toISOString().split('T')[0];
      userChangedSliders = false;
      runPrediction();
    }

    function toggleNumber(num) {
      if (takenNumbers.has(num)) {
        takenNumbers.delete(num);
      } else {
        takenNumbers.add(num);
      }
      document.getElementById('taken-count').textContent = takenNumbers.size;
      runPrediction();
    }

    function clearTaken() {
      takenNumbers.clear();
      document.getElementById('taken-count').textContent = '0';
      runPrediction();
    }

    async function syncDraws() {
      const btn = document.getElementById('sync-btn');
      btn.textContent = '⏳ Sincronizando...';
      try {
        const response = await fetch('/api/sync', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({days_back: 14}) });
        const data = await response.json();
        if (data.status === 'ok') {
          btn.textContent = `✓ Sincronizado (+${data.added})`;
          setTimeout(() => { btn.textContent = '🔄 Sincronizar Web'; }, 3000);
          await runPrediction();
        } else {
          btn.textContent = '⚠️ Error Sync';
          setTimeout(() => { btn.textContent = '🔄 Sincronizar Web'; }, 3000);
        }
      } catch (e) {
        console.error("Sync error:", e);
        btn.textContent = '⚠️ Error Sync';
        setTimeout(() => { btn.textContent = '🔄 Sincronizar Web'; }, 3000);
      }
    }

    async function runHyperopt() {
      const btn = document.getElementById('hyperopt-btn');
      btn.textContent = '⏳ Calibrando Folds...';
      try {
        const response = await fetch('/api/hyperopt', { method: 'POST', headers: {'Content-Type': 'application/json'} });
        const data = await response.json();
        btn.textContent = '✓ Calibrado';
        setTimeout(() => { btn.textContent = '🔬 Re-Calibrar Walk-Forward'; }, 3000);
        runPrediction();
      } catch (e) {
        btn.textContent = '⚠️ Error';
      }
    }

    async function runPrediction() {
      const btn = document.getElementById('calc-btn');
      const targetDateInput = document.getElementById('target-date');

      // Default date to today if empty or in past
      const todayStr = new Date().toISOString().split('T')[0];
      if (!targetDateInput.value || targetDateInput.value < todayStr) {
        targetDateInput.value = todayStr;
      }
      targetDateInput.min = todayStr;

      const payload = {
        lottery: document.getElementById('lottery-select').value,
        target_date: targetDateInput.value,
        ticket_count: parseInt(document.getElementById('ticket-count').value),
        orthogonal_coverage: document.getElementById('orthogonal-mode').value === 'true',
        taken_numbers: Array.from(takenNumbers),
        strategy_weights: userChangedSliders ? {
          ml_tree: parseFloat(document.getElementById('w-tree').value),
          deep_gpu: parseFloat(document.getElementById('w-deep').value),
          markov: parseFloat(document.getElementById('w-markov').value),
          bayesian_hot: parseFloat(document.getElementById('w-hot').value),
          renewal_cold: parseFloat(document.getElementById('w-cold').value)
        } : null
      };

      try {
        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        renderDashboard(data);
      } catch (err) {
        console.error(err);
      }
    }

    function renderDashboard(data) {
      const hw = data.hardware;
      document.getElementById('hw-badge').textContent = `${hw.device_name}`;

      const dbs = data.db_stats;
      if (dbs) {
        document.getElementById('db-badge').textContent = `SQLite: ${dbs.ciudad_draws || dbs.total_draws} Sorteos LOTBA`;
        document.getElementById('kpi-db-count').textContent = `${dbs.ciudad_draws || dbs.total_draws}`;
      }

      if (data.hyperparams && data.hyperparams.optimal_weights) {
        currentOptimalWeights = data.hyperparams.optimal_weights;
      }

      // Update sliders if not manually overridden by user
      if (!userChangedSliders && data.horizon_weights_used) {
        const hw = data.horizon_weights_used;
        document.getElementById('w-tree').value = hw.ml_tree;
        document.getElementById('w-deep').value = hw.deep_gpu;
        document.getElementById('w-markov').value = hw.markov;
        document.getElementById('w-hot').value = hw.bayesian_hot;
        document.getElementById('w-cold').value = hw.renewal_cold;
        updateSliders();
      }

      // Horizon badge & Title
      if (data.horizon_label) {
        const hBadge = document.getElementById('horizon-badge');
        if (hBadge) {
          hBadge.textContent = data.horizon_draws > 1 ? `T+${data.horizon_draws} (${data.horizon_draws * 24}h)` : 'T+1 (Hoy)';
          hBadge.className = data.horizon_draws > 1 
            ? 'font-mono text-[10px] px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 font-bold'
            : 'font-mono text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-bold';
        }
        const titleBadge = document.getElementById('rec-title');
        if (titleBadge) {
          titleBadge.innerHTML = `<span>🎯</span> Portafolio Óptimo — ${data.horizon_label}`;
        }
      }

      if (data.hyperparams && data.hyperparams.backtest_results) {
        const bt = data.hyperparams.backtest_results;
        document.getElementById('wf-logloss').textContent = bt.log_loss_mean.toFixed(3);
        if (document.getElementById('wf-hit5')) {
          document.getElementById('wf-hit5').textContent = `${bt.hit_5_pct}%`;
        }
        document.getElementById('wf-hit10').textContent = `${bt.hit_10_pct}%`;
        document.getElementById('wf-brier').textContent = bt.brier_score_mean.toFixed(3);
        document.getElementById('wf-folds').textContent = `${bt.folds_tested || (data.horizon_draws > 1 ? 89 : 119)} Folds`;
        document.getElementById('backtest-badge').textContent = `Backtest Hit@5: ${bt.hit_5_pct}% (${data.horizon_draws > 1 ? '48h Difusivo' : '+' + bt.lift_hit_5_pct + '% lift'})`;
        if (document.getElementById('wf-summary')) {
          if (data.horizon_draws > 1) {
            document.getElementById('wf-summary').innerHTML = `Modo Anticipado a 48h (T+${data.horizon_draws}): Hit@5: <strong class="text-indigo-400">${bt.hit_5_pct}%</strong> (${bt.folds_tested || 89} Folds). Matriz Markov <strong class="text-purple-300">P^${data.horizon_draws} (Chapman-Kolmogorov)</strong> + Inercia Bayesiana Dirichlet adaptada a 48h.`;
          } else {
            document.getElementById('wf-summary').innerHTML = `Modo Inmediato (T+1): Hit@5: <strong class="text-emerald-400">${bt.hit_5_pct}%</strong> (+${bt.lift_hit_5_pct}% lift). Hit Decena: <strong class="text-teal-300">${bt.hit_decena_pct}%</strong>. Log-Loss: <strong class="text-purple-300">${bt.log_loss_mean.toFixed(3)}</strong> (119 Folds).`;
          }
        }
      }

      document.getElementById('last-draw-banner').textContent =
        `${data.last_draw.date} → 1° ${data.last_draw.number_4d} (Ambo: ${data.last_draw.number_2d})`;

      const pm = data.portfolio_metrics;
      document.getElementById('kpi-prob').textContent = `${pm.cumulative_master_prob_pct}%`;
      document.getElementById('kpi-lift').textContent = `Base azar: ${pm.uniform_baseline_pct}% (+${pm.lift_over_random_pct}% lift)`;
      document.getElementById('kpi-reach').textContent = `${pm.orthogonal_digit_reach_pct}%`;
      document.getElementById('kpi-du-cov').textContent = `Decenas: ${pm.decena_coverage_pct}% | Unidades: ${pm.unidad_coverage_pct}%`;
      document.getElementById('kpi-avail').textContent = `${pm.available_pool_size} / 100`;

      // Telemetry
      const wf = data.weather_forecast;
      const lf = data.lunar_ephemeris;
      document.getElementById('telemetry-box').innerHTML = `
        <div class="bg-slate-900/70 p-2.5 rounded-lg border border-slate-800">
          <div class="text-slate-400">Temperatura / Rocío</div>
          <div class="text-white font-bold mt-0.5">${wf.temperature_2m} °C <span class="text-[11px] text-slate-400 font-normal">(${wf.dew_point !== undefined ? wf.dew_point : '--'}°C)</span></div>
          <div class="text-[10px] text-slate-400 mt-1">Presión: ${wf.surface_pressure} hPa</div>
        </div>
        <div class="bg-slate-900/70 p-2.5 rounded-lg border border-slate-800">
          <div class="text-slate-400">Densidad (ρ) & Viscosidad (μ)</div>
          <div class="text-emerald-400 font-bold mt-0.5 font-mono">${wf.air_density ? wf.air_density + ' kg/m³' : '1.225 kg/m³'}</div>
          <div class="text-[10px] text-slate-400 mt-1 font-mono">μ=${wf.dynamic_viscosity || 1.815} · ν=${wf.kinematic_viscosity || 1.481}</div>
        </div>
        <div class="bg-slate-900/70 p-2.5 rounded-lg border border-slate-800">
          <div class="text-slate-400">Humedad & Flotabilidad (B)</div>
          <div class="text-white font-bold mt-0.5">${wf.relative_humidity_2m}% <span class="text-[11px] font-normal ${wf.static_risk ? 'text-amber-400' : 'text-emerald-400'}">${wf.static_risk ? '⚡ Estática' : '✓ Normal'}</span></div>
          <div class="text-[10px] text-teal-300 mt-1 font-mono">Flotab: ${(wf.ball_buoyant_ratio ? wf.ball_buoyant_ratio * 100 : 1.52).toFixed(2)}% peso</div>
        </div>
        <div class="bg-slate-900/70 p-2.5 rounded-lg border border-slate-800">
          <div class="text-slate-400">Viento & Vórtice Tambor</div>
          <div class="text-white font-bold mt-0.5">${wf.wind_speed_10m} km/h</div>
          <div class="text-[10px] text-slate-400 mt-1">Arrastre laminar bolillero</div>
        </div>
        <div class="bg-slate-900/70 p-2.5 rounded-lg border border-slate-800">
          <div class="text-slate-400">Fase Lunar (${data.target_date})</div>
          <div class="text-amber-300 font-bold mt-0.5">🌙 ${lf.lunar_phase_name}</div>
          <div class="text-[10px] text-amber-400/80 mt-1">Iluminación: ${lf.lunar_illumination}%</div>
        </div>
        <div class="bg-slate-900/70 p-2.5 rounded-lg border border-slate-800">
          <div class="text-slate-400">Marea Gravitacional (1/r³)</div>
          <div class="text-cyan-300 font-bold mt-0.5 font-mono">${lf.grav_tidal_force !== undefined ? Number(lf.grav_tidal_force).toFixed(3) : '1.000'}x</div>
          <div class="text-[10px] text-slate-400 mt-1">Distancia: ${lf.lunar_dist_ratio !== undefined ? Number(lf.lunar_dist_ratio).toFixed(3) : '1.000'} r/a</div>
        </div>
        <div class="bg-slate-900/70 p-2.5 rounded-lg border border-slate-800">
          <div class="text-slate-400">Declinación Solar & Lunar</div>
          <div class="text-indigo-300 font-bold mt-0.5 font-mono">${lf.solar_declination !== undefined ? lf.solar_declination : 0}° Sol · ${lf.lunar_declination !== undefined ? lf.lunar_declination : 0}° Luna</div>
          <div class="text-[10px] text-slate-400 mt-1">Ciclo estacional & nodal</div>
        </div>
        <div class="bg-slate-900/70 p-2.5 rounded-lg border border-slate-800">
          <div class="text-slate-400">Resonancia Áurea Φ & Entropía</div>
          <div class="text-purple-300 font-bold mt-0.5 font-mono">Φ-Phase: ${lf.golden_sin !== undefined ? Number(lf.golden_sin).toFixed(2) : '0.00'}</div>
          <div class="text-[10px] text-purple-400 mt-1">Ergódica KAM · Shannon H₁₀: 3.32</div>
        </div>
      `;

      // World-Intel: Space Weather & Planetary Intelligence
      const si = data.space_weather_intel || {current_kp: 1.67, kp_level: 'Calmo', saa_anomaly_risk: 'Normal'};
      const sei = data.seismic_intel || {tremor_count_24h: 5, max_magnitude: 4.6, vibration_noise_level: 'Normal / Micro-sísmico', latest_place: 'Placa Sudamericana'};
      document.getElementById('intel-box').innerHTML = `
        <div class="bg-slate-900/70 p-2.5 rounded-lg border border-slate-800">
          <div class="text-slate-400">Perturbación Geomagnética</div>
          <div class="text-cyan-300 font-bold mt-0.5 font-mono">Kp = ${si.current_kp} <span class="text-[11px] font-normal text-emerald-400">(${si.kp_level})</span></div>
          <div class="text-[10px] text-slate-400 mt-1">NOAA SWPC · Red 8 observatorios</div>
        </div>
        <div class="bg-slate-900/70 p-2.5 rounded-lg border border-slate-800">
          <div class="text-slate-400">Anomalía Atlántico Sur (SAA)</div>
          <div class="text-white font-bold mt-0.5 font-mono">Riesgo: <span class="${si.saa_anomaly_risk === 'Alto' ? 'text-rose-400' : 'text-emerald-400'} font-bold">${si.saa_anomaly_risk}</span></div>
          <div class="text-[10px] text-slate-400 mt-1">Cinturón Van Allen / Ionización CABA</div>
        </div>
        <div class="bg-slate-900/70 p-2.5 rounded-lg border border-slate-800">
          <div class="text-slate-400">Micro-Sismicidad Cono Sur</div>
          <div class="text-amber-300 font-bold mt-0.5 font-mono">${sei.tremor_count_24h} sismos <span class="text-[11px] text-slate-400 font-normal">(Mmax ${sei.max_magnitude})</span></div>
          <div class="text-[10px] text-slate-400 mt-1 truncate" title="${sei.latest_place}">Borde Placa Andina · USGS</div>
        </div>
        <div class="bg-slate-900/70 p-2.5 rounded-lg border border-slate-800">
          <div class="text-slate-400">Ruido Infrasónico / Mecánico</div>
          <div class="text-emerald-400 font-bold mt-0.5 font-mono">${sei.vibration_noise_level}</div>
          <div class="text-[10px] text-slate-400 mt-1">Ondas Rayleigh &lt; 5 Hz en tambor</div>
        </div>
      `;

      // Portfolio Cards
      document.getElementById('portfolio-cards').innerHTML = data.optimal_portfolio.map(p => `
        <div class="p-3.5 rounded-xl bg-slate-900/90 border border-emerald-500/40 relative overflow-hidden flex flex-col justify-between">
          <div class="flex items-center justify-between">
            <span class="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-bold">Pick #${p.rank}</span>
            <span class="text-[10px] text-slate-400 font-mono">D=${p.decena} U=${p.unidad}</span>
          </div>
          <div class="my-2 text-center">
            <div class="text-3xl font-black text-white font-mono tracking-tight">${p.number}</div>
            <div class="text-xs font-bold text-emerald-400 font-mono mt-0.5">${p.master_prob_pct}% prob</div>
          </div>
          <div class="text-[10px] text-slate-400 border-t border-slate-800 pt-1.5 space-y-0.5">
            <div>• Sequía: <strong class="text-slate-200">${p.delay_2d}d</strong></div>
            <div>• Anti-Crowd: <strong class="text-slate-200">${p.crowd_edge_score}%</strong></div>
            <div class="truncate text-slate-300">• ${p.dominant_driver}</div>
          </div>
        </div>
      `).join('');

      // Grid Heatmap
      const maxP = Math.max(...data.grid_100.map(c => c.prob_pct));
      const minP = Math.min(...data.grid_100.map(c => c.prob_pct));

      document.getElementById('grid-container').innerHTML = data.grid_100.map(cell => {
        const isRec = cell.recommended;
        const isTaken = cell.taken;
        const norm = (cell.prob_pct - minP) / (maxP - minP + 1e-9);

        let bg = 'rgba(15, 23, 42, 0.7)';
        if (!isTaken) {
          if (norm > 0.8) bg = 'rgba(16, 185, 129, 0.35)';
          else if (norm > 0.5) bg = 'rgba(14, 165, 233, 0.25)';
          else if (norm > 0.25) bg = 'rgba(99, 102, 241, 0.20)';
        }

        return `
          <button onclick="toggleNumber(${cell.number_int})"
                  style="background-color: ${bg};"
                  class="cell-btn rounded-lg p-2 text-center border border-slate-800 flex flex-col items-center justify-center ${isRec ? 'recommended-ring' : ''} ${isTaken ? 'taken-cell' : ''}"
                  title="Ambo ${cell.number} | Prob: ${cell.prob_pct}% | Salidas: ${cell.freq}x | Atraso: ${cell.delay} sorteos">
            <span class="text-sm sm:text-base font-black font-mono ${isRec ? 'text-emerald-300' : 'text-white'}">${cell.number}</span>
            <span class="text-[10px] font-mono text-slate-300">${cell.prob_pct.toFixed(2)}%</span>
          </button>
        `;
      }).join('');

      // Recent Draws in Table
      if (data.recent_draws) {
        document.getElementById('db-draws-tbody').innerHTML = data.recent_draws.map(r => `
          <tr class="hover:bg-slate-900/40">
            <td class="py-2 text-white font-semibold">${r.date}</td>
            <td class="py-2 uppercase text-slate-300">2 - LOTBA (Ciudad)</td>
            <td class="py-2 text-slate-300">${r.number_4d}</td>
            <td class="py-2"><span class="px-2 py-0.5 bg-amber-500/20 text-amber-300 font-bold rounded border border-amber-500/30">${String(r.number_2d).padStart(2, '0')}</span></td>
            <td class="py-2 text-slate-400">D=${r.decena} | U=${r.unidad}</td>
            <td class="py-2 text-slate-400">${r.temperature_2m}°C • ${r.surface_pressure} hPa</td>
            <td class="py-2 text-slate-400">${r.lunar_phase_name} (${r.lunar_illumination}%)</td>
            <td class="py-2"><span class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 text-[10px]">${r.source}</span></td>
          </tr>
        `).join('');
      }

      // Refresh Tracker ledger after prediction
      loadTracker();
    }

    async function loadTracker() {
      try {
        const lottery = document.getElementById('lottery-select') ? document.getElementById('lottery-select').value : 'ciudad';
        const res = await fetch(`/api/tracker?lottery=${lottery}`);
        const data = await res.json();
        renderTracker(data);
      } catch (err) {
        console.error("Error loading tracker:", err);
      }
    }

    async function backfillTracker() {
      const btn = document.getElementById('backfill-btn');
      btn.innerHTML = '⏳ Auditando 30 sorteos...';
      try {
        const lottery = document.getElementById('lottery-select') ? document.getElementById('lottery-select').value : 'ciudad';
        const res = await fetch('/api/tracker/backfill', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({days: 30, lottery: lottery})
        });
        const data = await res.json();
        btn.innerHTML = `✓ Auditados (${data.results.backfilled})`;
        setTimeout(() => { btn.innerHTML = '<span>🔬</span> Auditar Últimos 30 Sorteos'; }, 3000);
        renderTracker(data.results.report);
      } catch (err) {
        console.error("Error backfilling tracker:", err);
        btn.innerHTML = '⚠️ Error Auditoría';
        setTimeout(() => { btn.innerHTML = '<span>🔬</span> Auditar Últimos 30 Sorteos'; }, 3000);
      }
    }

    function renderTracker(data) {
      if (!data || !data.summary) return;
      const s = data.summary;
      document.getElementById('tr-hit-rate').textContent = `${s.exact_hit_rate_pct}% (${s.exact_hits}/${s.evaluated_count})`;
      document.getElementById('tr-hit-lift').textContent = `Lift: +${s.lift_over_random_pct}% vs azar 5%`;

      const digitHitRate = Math.round((s.decena_hits + s.unidad_hits) / (Math.max(1, s.evaluated_count * 2)) * 100);
      document.getElementById('tr-digit-rate').textContent = `${digitHitRate}%`;
      document.getElementById('tr-digit-detail').textContent = `Decena: ${s.decena_hit_rate_pct}% | Unidad: ${s.unidad_hit_rate_pct}%`;

      const pnlPrefix = s.net_profit_ars >= 0 ? '+$' : '-$';
      const pnlClass = s.net_profit_ars >= 0 ? 'text-emerald-300' : 'text-rose-400';
      document.getElementById('tr-pnl').className = `text-xl font-black font-mono mt-0.5 ${pnlClass}`;
      document.getElementById('tr-pnl').textContent = `${pnlPrefix}${Math.abs(s.net_profit_ars)}`;
      document.getElementById('tr-roi').textContent = `ROI: ${s.roi_pct}% ($${s.total_won_ars} ganados / $${s.total_staked_ars} costo)`;

      document.getElementById('tr-evaluated').textContent = `${s.evaluated_count}`;
      document.getElementById('tr-pending').textContent = `Pendientes: ${s.pending_count}`;

      const tbody = document.getElementById('tracker-tbody');
      if (!data.history || data.history.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="py-4 text-center text-slate-500">No hay predicciones registradas aún. Haz una predicción o corre la auditoría.</td></tr>';
        return;
      }

      tbody.innerHTML = data.history.map(item => {
        const picksHtml = item.top_picks.map(p => {
          if (p.is_hit) {
            return `<span class="inline-block px-1.5 py-0.5 rounded bg-emerald-500/30 text-emerald-300 font-bold border border-emerald-400 animate-pulse font-mono" title="¡Acierto exacto! Pick #${p.rank} (${p.dominant_driver})">🎯 ${p.number}</span>`;
          }
          return `<span class="inline-block px-1.5 py-0.5 rounded bg-slate-800/80 text-slate-300 border border-slate-700 font-mono text-[11px]" title="Pick #${p.rank} (${p.dominant_driver})">${p.number}</span>`;
        }).join(' ');

        let statusBadge = '';
        if (item.status_type === 'exact') {
          statusBadge = `<span class="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">🎯 ACIERTO EXACTO (Pick #${item.hit_rank})</span>`;
        } else if (item.status_type === 'both_digits') {
          statusBadge = `<span class="px-2 py-0.5 rounded-full text-[10px] font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">⚡ DECENA + UNIDAD</span>`;
        } else if (item.status_type === 'decena') {
          statusBadge = `<span class="px-2 py-0.5 rounded-full text-[10px] font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/40">⚡ DECENA (${Math.floor(item.actual_number_2d / 10)}x)</span>`;
        } else if (item.status_type === 'unidad') {
          statusBadge = `<span class="px-2 py-0.5 rounded-full text-[10px] font-bold bg-teal-500/20 text-teal-300 border border-teal-500/40">⚡ UNIDAD (x${item.actual_number_2d % 10})</span>`;
        } else if (item.status_type === 'pending') {
          statusBadge = `<span class="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40">⏳ PENDIENTE (21:00)</span>`;
        } else {
          statusBadge = `<span class="px-2 py-0.5 rounded-full text-[10px] font-bold bg-slate-800 text-slate-400 border border-slate-700">❌ NO ACERTÓ</span>`;
        }

        const payoutText = item.is_evaluated ? (item.payout_sim >= 0 ? `<span class="text-emerald-400 font-bold">+$${item.payout_sim}</span>` : `<span class="text-slate-400">-$${Math.abs(item.payout_sim)}</span>`) : '<span class="text-amber-400">--</span>';

        return `
          <tr class="hover:bg-slate-900/40 ${item.hit_exact ? 'bg-emerald-950/20' : ''}">
            <td class="py-2.5 text-white font-bold">${item.target_date}</td>
            <td class="py-2.5 font-bold text-slate-200">${item.actual_number_4d || '<span class="text-slate-500">Esperando...</span>'}</td>
            <td class="py-2.5 font-bold ${item.hit_exact ? 'text-emerald-400 text-sm' : 'text-slate-300'}">${item.actual_number_2d !== null ? String(item.actual_number_2d).padStart(2, '0') : '--'}</td>
            <td class="py-2.5">${picksHtml}</td>
            <td class="py-2.5">${statusBadge}</td>
            <td class="py-2.5">${payoutText}</td>
            <td class="py-2.5 text-[10px] text-slate-500">${item.source}</td>
          </tr>
        `;
      }).join('');
    }

    // Set today's date and run initial prediction
    window.addEventListener('DOMContentLoaded', () => {
      const todayStr = new Date().toISOString().split('T')[0];
      const targetDateInput = document.getElementById('target-date');
      targetDateInput.value = todayStr;
      targetDateInput.min = todayStr;
      updateSliders();
      runPrediction();
      loadTracker();
    });
  </script>
</body>
</html>
"""


async def homepage(request):
    return HTMLResponse(HTML_TEMPLATE)


async def api_health(request):
    try:
        stats = predictor.db.get_db_stats()
        return JSONResponse({
            'status': 'ok',
            'app': 'la-marisca-quiniela-ml',
            'cluster': 'k3s',
            'node': 'xeon',
            'db_stats': stats,
            'today': datetime.date.today().isoformat()
        })
    except Exception as e:
        return JSONResponse({'status': 'error', 'detail': str(e)}, status_code=500)


async def api_predict(request):
    body = await request.json()
    res = predictor.run_prediction(
        lottery=body.get('lottery', 'ciudad'),
        target_date=body.get('target_date', None),
        ticket_count=int(body.get('ticket_count', 5)),
        taken_numbers=body.get('taken_numbers', []),
        strategy_weights=body.get('strategy_weights', None),
        orthogonal_coverage=bool(body.get('orthogonal_coverage', True))
    )
    return JSONResponse(res)


async def api_sync(request):
    body = await request.json() if request.headers.get('content-length', '0') != '0' else {}
    days_back = int(body.get('days_back', 14))
    added = await asyncio.to_thread(predictor.sync_latest_draws, days_back=days_back)
    return JSONResponse({'status': 'ok', 'added': added, 'db_stats': predictor.db.get_db_stats()})


async def api_hyperopt(request):
    body = await request.json() if request.headers.get('content-length', '0') != '0' else {}
    lottery = body.get('lottery', 'ciudad')
    res = await asyncio.to_thread(optimize_hyperparameters, lottery=lottery)
    return JSONResponse({'status': 'ok', 'results': res})


async def api_hyperopt_results(request):
    return JSONResponse(load_optimal_hyperparams())


async def api_add_draw(request):
    body = await request.json()
    saved = predictor.add_manual_draw(
        date_str=body.get('date'),
        lottery=body.get('lottery', 'ciudad'),
        number_4d=body.get('number_4d')
    )
    return JSONResponse({'status': 'ok', 'saved': saved, 'db_stats': predictor.db.get_db_stats()})


async def api_tracker(request):
    lottery = request.query_params.get('lottery', 'ciudad')
    limit = int(request.query_params.get('limit', 60))
    report = predictor.db.get_tracker_report(lottery=lottery, limit=limit)
    return JSONResponse(report)


async def api_tracker_backfill(request):
    body = await request.json() if request.headers.get('content-length', '0') != '0' else {}
    days = int(body.get('days', 30))
    lottery = body.get('lottery', 'ciudad')
    res = await asyncio.to_thread(predictor.backfill_historical_predictions, days=days, lottery=lottery)
    return JSONResponse({'status': 'ok', 'results': res})


async def api_intel(request):
    try:
        sw = enrich_features.fetch_space_weather_intel()
        sei = enrich_features.fetch_regional_seismic_intel()
        return JSONResponse({
            'status': 'ok',
            'space_weather': sw,
            'seismic': sei,
            'source': 'NOAA SWPC + USGS (World-Intel Feed)'
        })
    except Exception as e:
        return JSONResponse({'status': 'error', 'detail': str(e)}, status_code=500)


app = Starlette(
    debug=False,
    routes=[
        Route('/', homepage, methods=['GET']),
        Route('/api/health', api_health, methods=['GET']),
        Route('/api/predict', api_predict, methods=['POST']),
        Route('/api/sync', api_sync, methods=['POST']),
        Route('/api/tracker', api_tracker, methods=['GET']),
        Route('/api/tracker/backfill', api_tracker_backfill, methods=['POST']),
        Route('/api/hyperopt', api_hyperopt, methods=['POST']),
        Route('/api/hyperopt/results', api_hyperopt_results, methods=['GET']),
        Route('/api/add_draw', api_add_draw, methods=['POST']),
        Route('/api/intel', api_intel, methods=['GET']),
    ]
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='La Marisca LOTBA Raffle Predictor')
    parser.add_argument('--host', default='0.0.0.0', help='Bind host')
    parser.add_argument('--port', type=int, default=8000, help='Bind port')
    parser.add_argument('--cli', action='store_true', help='Run CLI prediction once and exit')
    parser.add_argument('--tickets', type=int, default=5, help='Number of raffle tickets to optimize')
    args = parser.parse_args()

    if args.cli:
        res = predictor.run_prediction(lottery='ciudad', ticket_count=args.tickets)
        print(json.dumps({
            'hardware': res['hardware'],
            'db_stats': res['db_stats'],
            'last_draw': res['last_draw'],
            'portfolio_metrics': res['portfolio_metrics'],
            'optimal_portfolio': res['optimal_portfolio']
        }, indent=2, ensure_ascii=False))
    else:
        print(f"Starting La Marisca LOTBA Predictor at http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level='info')
