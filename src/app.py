import os
import sys
import json
import argparse
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_engine import RaffleGPUPredictor

predictor = RaffleGPUPredictor()

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>La Marisca — GPU Raffle Optimizer & SQLite Live Engine</title>
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
            La Marisca Project • 2 - LOTBA Primary
          </span>
          <span id="hw-badge" class="px-2.5 py-1 text-xs font-mono bg-indigo-500/20 text-indigo-300 rounded-full border border-indigo-500/30">
            Detectando GPU...
          </span>
          <span id="db-badge" class="px-2.5 py-1 text-xs font-mono bg-amber-500/20 text-amber-300 rounded-full border border-amber-500/30">
            SQLite: Cargando...
          </span>
        </div>
        <h1 class="text-2xl md:text-3xl font-extrabold text-white mt-2 tracking-tight">
          Optimizador Cuántico-Estocástico de Rifa (00–99)
        </h1>
        <p class="text-sm text-slate-400 mt-1">
          Motor Auto-Actualizable con Base de Datos SQLite + Ensamble GPU de 5 Pilares (LOTBA Ciudad & Formosa)
        </p>
      </div>
      <div class="flex flex-col items-end gap-2 self-stretch md:self-auto justify-between">
        <div class="text-right bg-slate-900/90 px-4 py-2.5 rounded-xl border border-slate-800">
          <div class="text-xs text-slate-400">Último Sorteo en Base de Datos</div>
          <div id="last-draw-banner" class="text-lg font-bold text-amber-400 font-mono">Cargando...</div>
        </div>
        <button onclick="syncDraws()" id="sync-btn" class="text-xs px-3 py-1.5 rounded-lg bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-500/30 transition font-semibold">
          ↻ Auto-Sincronizar Sorteos desde la Web
        </button>
      </div>
    </header>

    <!-- Main Grid: Controls + Optimal Portfolio -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

      <!-- Left Column: Controls & Weights (4 cols) -->
      <div class="lg:col-span-4 space-y-6">
        <div class="glass rounded-2xl p-5 space-y-4">
          <h2 class="text-lg font-bold text-white flex items-center justify-between">
            <span>⚙️ Parámetros de la Jugada</span>
          </h2>

          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="block text-xs text-slate-400 mb-1">Lotería Oficial</label>
              <select id="lottery-select" onchange="runPrediction()" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white font-semibold">
                <option value="ciudad" selected>2 - LOTBA (Ciudad Bs.As.)</option>
                <option value="formosa">Quiniela de Formosa</option>
              </select>
            </div>
            <div>
              <label class="block text-xs text-slate-400 mb-1">Fecha a Predecir</label>
              <input type="date" id="target-date" value="2026-10-17" onchange="runPrediction()" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white" />
            </div>
          </div>

          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="block text-xs text-slate-400 mb-1">Cantidad de Números (K)</label>
              <select id="ticket-count" onchange="runPrediction()" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white font-bold">
                <option value="1">1 Número (Bala de Plata)</option>
                <option value="2">2 Números (Dupla)</option>
                <option value="3">3 Números (Trío)</option>
                <option value="5" selected>5 Números (Portafolio Top 5)</option>
                <option value="8">8 Números (Cobertura Amplia)</option>
                <option value="10">10 Números (10x10 Latin Rook)</option>
              </select>
            </div>
            <div>
              <label class="block text-xs text-slate-400 mb-1">Modo de Cobertura</label>
              <select id="orthogonal-mode" onchange="runPrediction()" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-emerald-400 font-semibold">
                <option value="true" selected>Ortogonal (Sin repetir D/U)</option>
                <option value="false">Densidad Pura (Pico Máximo)</option>
              </select>
            </div>
          </div>

          <hr class="border-slate-800">

          <div class="space-y-3">
            <div class="text-xs font-semibold uppercase tracking-wider text-slate-400">Pesos del Ensamble Híbrido</div>

            <div>
              <div class="flex justify-between text-xs mb-1">
                <span>🌲 Árboles GBDT + Random Forest</span>
                <span id="w-tree-val" class="font-mono text-emerald-400">30%</span>
              </div>
              <input type="range" id="w-tree" min="0" max="100" value="30" oninput="updateWeightLabels()" onchange="runPrediction()" class="w-full accent-emerald-500">
            </div>

            <div>
              <div class="flex justify-between text-xs mb-1">
                <span>🧠 Deep ResNet + MC Dropout (GPU)</span>
                <span id="w-deep-val" class="font-mono text-indigo-400">25%</span>
              </div>
              <input type="range" id="w-deep" min="0" max="100" value="25" oninput="updateWeightLabels()" onchange="runPrediction()" class="w-full accent-indigo-500">
            </div>

            <div>
              <div class="flex justify-between text-xs mb-1">
                <span>⛓️ Transición de Markov (Post-Último)</span>
                <span id="w-markov-val" class="font-mono text-amber-400">15%</span>
              </div>
              <input type="range" id="w-markov" min="0" max="100" value="15" oninput="updateWeightLabels()" onchange="runPrediction()" class="w-full accent-amber-500">
            </div>

            <div>
              <div class="flex justify-between text-xs mb-1">
                <span>🔥 Inercia Bayesiana (Hot Streak)</span>
                <span id="w-hot-val" class="font-mono text-rose-400">15%</span>
              </div>
              <input type="range" id="w-hot" min="0" max="100" value="15" oninput="updateWeightLabels()" onchange="runPrediction()" class="w-full accent-rose-500">
            </div>

            <div>
              <div class="flex justify-between text-xs mb-1">
                <span>⏳ Tasa de Riesgo / Sequía (Atrasos)</span>
                <span id="w-cold-val" class="font-mono text-cyan-400">15%</span>
              </div>
              <input type="range" id="w-cold" min="0" max="100" value="15" oninput="updateWeightLabels()" onchange="runPrediction()" class="w-full accent-cyan-500">
            </div>
          </div>

          <button onclick="runPrediction()" id="calc-btn" class="w-full py-3 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-slate-950 font-extrabold text-sm shadow-lg shadow-emerald-500/20 transition">
            ⚡ Re-Entrenar y Simular en GPU (1M Rollouts)
          </button>
        </div>

        <!-- Telemetry & Audit Box -->
        <div class="glass rounded-2xl p-5 space-y-3">
          <h3 class="text-sm font-bold text-white uppercase tracking-wider">📡 Telemetría Física & Auditoría</h3>
          <div id="telemetry-box" class="grid grid-cols-2 gap-2 text-xs"></div>
          <hr class="border-slate-800">
          <div id="audit-box" class="text-xs text-slate-400 space-y-1"></div>
        </div>
      </div>

      <!-- Right Column: Optimal Portfolio & 5 Archetypes (8 cols) -->
      <div class="lg:col-span-8 space-y-6">

        <!-- KPI Metrics Banner -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div class="glass rounded-2xl p-4 border-l-4 border-emerald-500">
            <div class="text-xs text-slate-400">Masa Posterior Acumulada</div>
            <div id="kpi-prob" class="text-2xl font-black text-emerald-400 mt-1">--%</div>
            <div id="kpi-lift" class="text-[11px] text-slate-400">vs Base Uniforme</div>
          </div>
          <div class="glass rounded-2xl p-4 border-l-4 border-indigo-500">
            <div class="text-xs text-slate-400">Alcance Ortogonal D/U</div>
            <div id="kpi-reach" class="text-2xl font-black text-indigo-400 mt-1">--%</div>
            <div id="kpi-du-cov" class="text-[11px] text-slate-400">Cobertura de Bolilleros</div>
          </div>
          <div class="glass rounded-2xl p-4 border-l-4 border-amber-500">
            <div class="text-xs text-slate-400">Números Disponibles</div>
            <div id="kpi-avail" class="text-2xl font-black text-amber-400 mt-1">100 / 100</div>
            <div class="text-[11px] text-slate-400">En la grilla actual</div>
          </div>
          <div class="glass rounded-2xl p-4 border-l-4 border-cyan-500">
            <div class="text-xs text-slate-400">Base de Datos SQLite</div>
            <div id="kpi-db-count" class="text-2xl font-black text-cyan-400 mt-1">--</div>
            <div class="text-[11px] text-slate-400">Sorteos almacenados</div>
          </div>
        </div>

        <!-- Optimal Play Cards -->
        <div class="glass rounded-2xl p-5 space-y-4">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h2 class="text-xl font-extrabold text-white">🏆 Jugada Óptima Recomendada (Maximización de Probabilidad)</h2>
              <p class="text-xs text-slate-400">Seleccionada automáticamente entre los números libres maximizando verosimilitud + cobertura ortogonal de bolilleros</p>
            </div>
          </div>
          <div id="portfolio-cards" class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-3"></div>
        </div>

        <!-- 5 Classic Archetypes Table -->
        <div class="glass rounded-2xl p-5 space-y-3">
          <h3 class="text-sm font-bold text-white uppercase tracking-wider">🎯 Los 5 Perfiles Científicos Individuales (Mejor Libre por Categoría)</h3>
          <div class="overflow-x-auto">
            <table class="w-full text-left text-xs">
              <thead>
                <tr class="border-b border-slate-800 text-slate-400">
                  <th class="py-2">Perfil Estratégico</th>
                  <th class="py-2">Ambo</th>
                  <th class="py-2">Prob. Modelo</th>
                  <th class="py-2">Fundamento Técnico</th>
                </tr>
              </thead>
              <tbody id="archetypes-tbody" class="divide-y divide-slate-800/60"></tbody>
            </table>
          </div>
        </div>

      </div>
    </div>

    <!-- Interactive 00-99 Raffle Board -->
    <div class="glass rounded-2xl p-5 space-y-4">
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div>
          <h2 class="text-lg font-bold text-white">🔢 Tablero Interactivo de la Rifa (00 al 99) — Filtro de Números Vendidos</h2>
          <p class="text-xs text-slate-400">
            Haz clic en los números que <strong>ya fueron comprados por otros en el grupo de WhatsApp</strong> para tacharlos. El motor recalculará al instante los mejores números libres.
          </p>
        </div>
        <div class="flex items-center gap-2">
          <button onclick="clearTaken()" class="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700">
            Liberar Todos (0 Vendidos)
          </button>
        </div>
      </div>

      <div id="grid-container" class="grid grid-cols-10 gap-1.5 sm:gap-2"></div>
    </div>

    <!-- SQLite Database Manager & Live Ingestion Panel -->
    <div class="glass rounded-2xl p-5 space-y-4">
      <div class="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <h2 class="text-lg font-bold text-white">🗄️ Base de Datos SQLite (<code class="text-emerald-400">quiniela_marisca.db</code>) — Ingesta Continua y Actualización del Modelo</h2>
          <p class="text-xs text-slate-400 mt-0.5">
            Cada vez que sale un nuevo sorteo en <strong>2 - LOTBA (Ciudad)</strong> o <strong>Formosa</strong>, puedes auto-sincronizarlo desde la web o cargarlo manualmente aquí. El sistema descarga el clima/luna de esa fecha, actualiza los atrasos y re-entrena el modelo en el acto.
          </p>
        </div>

        <!-- Manual Quick Entry Form -->
        <div class="flex flex-wrap items-end gap-2 bg-slate-900/90 p-3 rounded-xl border border-slate-800">
          <div>
            <label class="block text-[11px] text-slate-400 mb-1">Fecha del Sorteo</label>
            <input type="date" id="manual-date" value="2026-10-16" class="bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white" />
          </div>
          <div>
            <label class="block text-[11px] text-slate-400 mb-1">Lotería</label>
            <select id="manual-lottery" class="bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white">
              <option value="ciudad" selected>2 - LOTBA (Ciudad)</option>
              <option value="formosa">Formosa</option>
            </select>
          </div>
          <div>
            <label class="block text-[11px] text-slate-400 mb-1">1° Premio (4 cifras)</label>
            <input type="text" id="manual-num4d" placeholder="Ej: 4264" maxlength="4" class="w-24 bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white font-mono" />
          </div>
          <button onclick="addManualDraw()" id="add-draw-btn" class="px-3.5 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs transition">
            ➕ Guardar en DB & Actualizar Modelo
          </button>
        </div>
      </div>

      <!-- Recent Draws Table from SQLite -->
      <div class="overflow-x-auto">
        <table class="w-full text-left text-xs">
          <thead>
            <tr class="border-b border-slate-800 text-slate-400">
              <th class="py-2">Fecha</th>
              <th class="py-2">Lotería</th>
              <th class="py-2">1° Premio (4D)</th>
              <th class="py-2">Ambo Ganador (2D)</th>
              <th class="py-2">Decena / Unidad</th>
              <th class="py-2">Clima (21:00 ART)</th>
              <th class="py-2">Fase Lunar</th>
              <th class="py-2">Fuente DB</th>
            </tr>
          </thead>
          <tbody id="db-draws-tbody" class="divide-y divide-slate-800/60 font-mono"></tbody>
        </table>
      </div>
    </div>

  </div>

  <script>
    let takenNumbers = new Set();

    function updateWeightLabels() {
      document.getElementById('w-tree-val').textContent = document.getElementById('w-tree').value + '%';
      document.getElementById('w-deep-val').textContent = document.getElementById('w-deep').value + '%';
      document.getElementById('w-markov-val').textContent = document.getElementById('w-markov').value + '%';
      document.getElementById('w-hot-val').textContent = document.getElementById('w-hot').value + '%';
      document.getElementById('w-cold-val').textContent = document.getElementById('w-cold').value + '%';
    }

    function toggleTaken(num) {
      if (takenNumbers.has(num)) {
        takenNumbers.delete(num);
      } else {
        takenNumbers.add(num);
      }
      runPrediction();
    }

    function clearTaken() {
      takenNumbers.clear();
      runPrediction();
    }

    async function syncDraws() {
      const btn = document.getElementById('sync-btn');
      btn.textContent = '⏳ Buscando nuevos sorteos en la web...';
      try {
        const res = await fetch('/api/sync', { method: 'POST' });
        const data = await res.json();
        btn.textContent = `✓ DB Sincronizada (+${data.added} nuevos)`;
        runPrediction();
      } catch (e) {
        btn.textContent = '⚠️ Error al sincronizar';
      }
    }

    async function addManualDraw() {
      const dateStr = document.getElementById('manual-date').value;
      const lottery = document.getElementById('manual-lottery').value;
      const num4d = document.getElementById('manual-num4d').value.trim();
      if (!/^\\d{1,4}$/.test(num4d)) {
        alert('Ingresa un número válido de hasta 4 cifras (ej: 4264 o 1856).');
        return;
      }
      const btn = document.getElementById('add-draw-btn');
      btn.textContent = '⏳ Guardando...';
      try {
        const res = await fetch('/api/add_draw', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ date: dateStr, lottery: lottery, number_4d: num4d })
        });
        const data = await res.json();
        document.getElementById('manual-num4d').value = '';
        btn.textContent = `✓ Guardado (${data.saved.number_4d})`;
        setTimeout(() => { btn.textContent = '➕ Guardar en DB & Actualizar Modelo'; }, 2500);
        runPrediction();
      } catch (e) {
        btn.textContent = '⚠️ Error';
      }
    }

    async function runPrediction() {
      const btn = document.getElementById('calc-btn');
      btn.textContent = '⏳ Calculando en GPU (1M Rollouts)...';

      const payload = {
        lottery: document.getElementById('lottery-select').value,
        target_date: document.getElementById('target-date').value,
        ticket_count: parseInt(document.getElementById('ticket-count').value),
        orthogonal_coverage: document.getElementById('orthogonal-mode').value === 'true',
        taken_numbers: Array.from(takenNumbers),
        strategy_weights: {
          ml_tree: parseFloat(document.getElementById('w-tree').value),
          deep_gpu: parseFloat(document.getElementById('w-deep').value),
          markov: parseFloat(document.getElementById('w-markov').value),
          bayesian_hot: parseFloat(document.getElementById('w-hot').value),
          renewal_cold: parseFloat(document.getElementById('w-cold').value)
        }
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
      } finally {
        btn.textContent = '⚡ Re-Entrenar y Simular en GPU (1M Rollouts)';
      }
    }

    function renderDashboard(data) {
      // Hardware badge
      const hw = data.hardware;
      document.getElementById('hw-badge').textContent = `${hw.device_name} (${hw.vram_mb} MB) • ${hw.backend}`;

      // DB badge
      if (data.db_stats) {
        const dbs = data.db_stats;
        document.getElementById('db-badge').textContent =
          `SQLite: ${dbs.ciudad_draws} LOTBA + ${dbs.formosa_draws} Formosa (${dbs.total_draws} total)`;
        document.getElementById('kpi-db-count').textContent = `${dbs.total_draws}`;
      }

      // Last draw
      document.getElementById('last-draw-banner').textContent =
        `${data.last_draw.date} → 1° ${data.last_draw.number_4d} (Ambo: ${data.last_draw.number_2d})`;

      // KPIs
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
          <div class="text-slate-400">Temp / Humedad (21h)</div>
          <div class="text-white font-bold mt-0.5">${wf.temperature_2m} °C • ${wf.relative_humidity_2m}%</div>
        </div>
        <div class="bg-slate-900/70 p-2.5 rounded-lg border border-slate-800">
          <div class="text-slate-400">Presión / Viento</div>
          <div class="text-white font-bold mt-0.5">${wf.surface_pressure} hPa • ${wf.wind_speed_10m} km/h</div>
        </div>
        <div class="bg-slate-900/70 p-2.5 rounded-lg border border-slate-800 col-span-2">
          <div class="text-slate-400">Fase Lunar (${data.target_date})</div>
          <div class="text-amber-300 font-bold mt-0.5">🌙 ${lf.lunar_phase_name} (${lf.lunar_illumination}% ilum. • φ=${lf.lunar_phase})</div>
        </div>
      `;

      // Audit
      const au = data.audit_summary;
      document.getElementById('audit-box').innerHTML = `
        <div>• <strong>Sorteos analizados (${data.lottery.toUpperCase()}):</strong> ${data.total_draws_analyzed}</div>
        <div>• <strong>Chi² Uniformidad:</strong> p(Dec)=${au.chi2_decena_p}, p(Uni)=${au.chi2_unidad_p}</div>
        <div>• <strong>Test de Rachas (Independencia):</strong> p=${au.runs_parity_p}</div>
        <div>• <strong>Entropía de Shannon:</strong> Dec=${au.shannon_entropy_decena} bits / 3.3219 max</div>
      `;

      // Optimal Portfolio Cards
      const cardsHtml = data.optimal_portfolio.map(item => `
        <div class="bg-gradient-to-b from-slate-900 to-slate-950 rounded-xl p-3.5 border border-emerald-500/40 relative overflow-hidden">
          <div class="flex items-center justify-between">
            <span class="text-[11px] font-bold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300">Pick #${item.rank}</span>
            <span class="text-xs font-mono text-emerald-400 font-bold">${item.master_prob_pct}%</span>
          </div>
          <div class="text-4xl font-black text-white text-center my-3 font-mono tracking-wider">
            ${item.number}
          </div>
          <div class="text-[11px] text-slate-300 space-y-1 border-t border-slate-800 pt-2">
            <div class="truncate" title="${item.dominant_driver}">⚡ <strong>Motor:</strong> ${item.dominant_driver}</div>
            <div>🎲 <strong>MC Hits:</strong> ${item.mc_sim_hits.toLocaleString()} / 1M</div>
            <div>📊 <strong>Salidas:</strong> ${item.freq_2d}x | <strong>Atraso:</strong> ${item.delay_2d}d</div>
            <div>🛡️ <strong>Edge Anti-Masa:</strong> ${item.crowd_edge_score}/100</div>
          </div>
        </div>
      `).join('');
      document.getElementById('portfolio-cards').innerHTML = cardsHtml;

      // Archetypes Table
      document.getElementById('archetypes-tbody').innerHTML = data.archetypes.map(a => `
        <tr class="hover:bg-slate-900/40">
          <td class="py-2 font-semibold text-white">${a.profile}</td>
          <td class="py-2"><span class="px-2 py-0.5 bg-indigo-500/20 text-indigo-300 font-mono font-bold rounded border border-indigo-500/30">${a.number}</span></td>
          <td class="py-2 font-mono text-emerald-400 font-bold">${a.prob_pct}%</td>
          <td class="py-2 text-slate-400">${a.desc}</td>
        </tr>
      `).join('');

      // 00-99 Grid
      const maxProb = Math.max(...data.grid_100.map(g => g.prob_pct), 1.5);
      const gridHtml = data.grid_100.map(cell => {
        const ratio = Math.min(1, cell.prob_pct / maxProb);
        const bgAlpha = (0.12 + ratio * 0.55).toFixed(2);
        const isRec = cell.recommended;
        const isTaken = cell.taken;
        const bgStyle = isTaken
          ? 'background: rgba(30, 41, 59, 0.4);'
          : isRec
          ? `background: rgba(16, 185, 129, 0.32);`
          : `background: rgba(79, 70, 229, ${bgAlpha});`;

        return `
          <button onclick="toggleTaken(${cell.number_int})"
                  style="${bgStyle}"
                  class="cell-btn rounded-lg p-2 text-center border border-slate-800 flex flex-col items-center justify-center ${isRec ? 'recommended-ring' : ''} ${isTaken ? 'taken-cell' : ''}"
                  title="Ambo ${cell.number} | Prob: ${cell.prob_pct}% | Salidas: ${cell.freq}x | Atraso: ${cell.delay} sorteos">
            <span class="text-sm sm:text-base font-black font-mono ${isRec ? 'text-emerald-300' : 'text-white'}">${cell.number}</span>
            <span class="text-[10px] font-mono text-slate-300">${cell.prob_pct.toFixed(2)}%</span>
          </button>
        `;
      }).join('');
      document.getElementById('grid-container').innerHTML = gridHtml;

      // Recent Draws in SQLite Table
      if (data.recent_draws) {
        document.getElementById('db-draws-tbody').innerHTML = data.recent_draws.map(r => `
          <tr class="hover:bg-slate-900/40">
            <td class="py-2 text-white font-semibold">${r.date}</td>
            <td class="py-2 uppercase text-slate-300">${r.lottery === 'ciudad' ? '2 - LOTBA (Ciudad)' : 'Formosa'}</td>
            <td class="py-2 text-slate-300">${r.number_4d}</td>
            <td class="py-2"><span class="px-2 py-0.5 bg-amber-500/20 text-amber-300 font-bold rounded border border-amber-500/30">${String(r.number_2d).padStart(2, '0')}</span></td>
            <td class="py-2 text-slate-400">D=${r.decena} | U=${r.unidad}</td>
            <td class="py-2 text-slate-400">${r.temperature_2m}°C • ${r.surface_pressure} hPa</td>
            <td class="py-2 text-slate-400">${r.lunar_phase_name} (${r.lunar_illumination}%)</td>
            <td class="py-2"><span class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 text-[10px]">${r.source}</span></td>
          </tr>
        `).join('');
      }
    }

    // Initial load
    window.addEventListener('DOMContentLoaded', () => {
      runPrediction();
    });
  </script>
</body>
</html>
"""


async def homepage(request):
    return HTMLResponse(HTML_TEMPLATE)


async def api_predict(request):
    body = await request.json()
    res = predictor.run_prediction(
        lottery=body.get('lottery', 'ciudad'),
        target_date=body.get('target_date', '2026-10-17'),
        ticket_count=int(body.get('ticket_count', 5)),
        taken_numbers=body.get('taken_numbers', []),
        strategy_weights=body.get('strategy_weights', None),
        orthogonal_coverage=bool(body.get('orthogonal_coverage', True))
    )
    return JSONResponse(res)


async def api_sync(request):
    added = predictor.sync_latest_draws(days_back=21)
    return JSONResponse({'status': 'ok', 'added': added, 'db_stats': predictor.db.get_db_stats()})


async def api_add_draw(request):
    body = await request.json()
    saved = predictor.add_manual_draw(
        date_str=body.get('date'),
        lottery=body.get('lottery', 'ciudad'),
        number_4d=body.get('number_4d')
    )
    return JSONResponse({'status': 'ok', 'saved': saved, 'db_stats': predictor.db.get_db_stats()})


app = Starlette(
    debug=False,
    routes=[
        Route('/', homepage, methods=['GET']),
        Route('/api/predict', api_predict, methods=['POST']),
        Route('/api/sync', api_sync, methods=['POST']),
        Route('/api/add_draw', api_add_draw, methods=['POST']),
    ]
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='La Marisca GPU Raffle Predictor Server & CLI')
    parser.add_argument('--host', default='0.0.0.0', help='Bind host')
    parser.add_argument('--port', type=int, default=8000, help='Bind port')
    parser.add_argument('--cli', action='store_true', help='Run CLI prediction once and exit')
    parser.add_argument('--tickets', type=int, default=5, help='Number of raffle tickets to optimize')
    args = parser.parse_args()

    if args.cli:
        res = predictor.run_prediction(lottery='ciudad', target_date='2026-10-17', ticket_count=args.tickets)
        print(json.dumps({
            'hardware': res['hardware'],
            'db_stats': res['db_stats'],
            'last_draw': res['last_draw'],
            'portfolio_metrics': res['portfolio_metrics'],
            'optimal_portfolio': res['optimal_portfolio']
        }, indent=2, ensure_ascii=False))
    else:
        print(f"Starting La Marisca GPU Raffle Optimizer at http://localhost:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level='info')
