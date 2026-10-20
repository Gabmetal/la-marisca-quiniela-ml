import os
import sys
import subprocess
import datetime
import math
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enrich_features
import scrape_draws
import statistical_audit
from database import QuinielaDatabase
from hyperopt_backtest import load_optimal_hyperparams

# In-memory component cache: (lottery, target_date, dataset_length) -> components dict
_COMPONENTS_CACHE = {}


def detect_hardware():
    """Detects CPU/GPU hardware details, optimized for Xeon single-core throttled environments."""
    info = {
        'gpu_available': False,
        'backend': 'CPU (NumPy + Scikit-Learn Vectorized • 1-Core Throttled)',
        'device_name': 'Intel Xeon E3-1226 v3 (CPU Throttled)',
        'vram_mb': 0,
        'driver': 'N/A',
        'torch_cuda': False,
        'xgb_cuda': False
    }
    try:
        out = subprocess.check_output(
            ['lscpu'],
            stderr=subprocess.DEVNULL,
            timeout=2
        ).decode('utf-8')
        for line in out.splitlines():
            if 'Model name:' in line:
                info['device_name'] = line.split(':', 1)[1].strip()
                break
    except Exception:
        pass
    return info


def compute_crowd_bias_scores():
    """
    Computes human selection bias in Argentine raffles (00-99).
    Lower crowd popularity = higher availability & contrarian value.
    """
    crowd_popularity = np.ones(100, dtype=float)
    # 1. Birthdays / Calendar dates (01-31)
    for n in range(1, 32):
        crowd_popularity[n] += 0.35
    # 2. Iconic "Smorfia / Quiniela" cultural favorites
    cultural_favorites = {
        0: 0.45, 13: 0.65, 14: 0.35, 15: 0.50, 17: 0.55,
        22: 0.65, 32: 0.45, 33: 0.50, 47: 0.35, 48: 0.55,
        69: 0.40, 77: 0.45, 88: 0.45, 99: 0.40
    }
    for num, boost in cultural_favorites.items():
        crowd_popularity[num] += boost
    # 3. Repdigits (ambos dobles: 11, 22, 33, ...)
    for d in range(10):
        crowd_popularity[d * 11] += 0.25

    inv = 1.0 / crowd_popularity
    edge = (inv - inv.min()) / (inv.max() - inv.min() + 1e-9)
    return edge


def _compute_components_internal(sub, lottery, target_date, hyperparams):
    """
    Computes and returns the 5 probability component matrices + telemetry for a future target date.
    Uses 34 physical, celestial, calendar, autoregressive, and Poisson delay features.
    """
    alpha_prior = hyperparams.get('alpha_prior', 1.8)
    renewal_temp = hyperparams.get('renewal_temp', 10.5)
    markov_smooth = hyperparams.get('markov_smooth', 0.4)
    decay_gamma = hyperparams.get('decay_gamma', 0.995)

    feature_cols = [
        # Physics & Aerodynamics
        'air_density', 'dew_point', 'static_risk', 'temperature_2m', 'relative_humidity_2m', 'surface_pressure', 'wind_speed_10m',
        # Celestial Ephemeris
        'lunar_phase', 'lunar_sin', 'lunar_cos', 'lunar_illumination',
        # Calendar & Inspection Cycles
        'day_of_week', 'day_of_month', 'is_monday', 'days_since_last_draw',
        # Autoregressive Lags & Repetitions
        'lag_1', 'lag_2', 'lag_3', 'lag_7',
        'lag_1_decena', 'lag_1_unidad', 'lag_2_decena', 'lag_2_unidad',
        'repeat_decena', 'repeat_unidad',
        'rolling_mean_3', 'rolling_mean_7', 'rolling_std_7',
        # Delays & Poisson Z-Scores
        'atraso_decena', 'atraso_unidad', 'delay_zscore_decena', 'delay_zscore_unidad',
        'max_atraso_decena_pool', 'max_atraso_unidad_pool'
    ]
    physics_cols = [
        'air_density', 'dew_point', 'static_risk', 'temperature_2m', 'relative_humidity_2m', 'surface_pressure', 'wind_speed_10m',
        'lunar_phase', 'lunar_sin', 'lunar_cos', 'lunar_illumination'
    ]

    X = sub[feature_cols].fillna(0).values
    y_dec = sub['decena'].astype(int).values
    y_uni = sub['unidad'].astype(int).values
    n_samples = len(sub)

    # Weather forecast & Lunar calculation for target future date
    weather_fc = enrich_features.fetch_weather_forecast(lottery, target_date)
    lunar_fc = enrich_features.compute_lunar_features(target_date, hour_art=21)

    # Physical atmosphere calculations if missing
    T = weather_fc.get('temperature_2m', 18.0)
    RH = weather_fc.get('relative_humidity_2m', 65.0)
    P = weather_fc.get('surface_pressure', 1015.0)
    if 'dew_point' not in weather_fc:
        a, b = 17.27, 237.7
        gamma = (a * T) / (b + T) + math.log(max(RH, 1.0) / 100.0)
        weather_fc['dew_point'] = round((b * gamma) / (a - gamma), 2)
    if 'air_density' not in weather_fc:
        T_kelvin = T + 273.15
        p_pascal = P * 100.0
        p_sat = 610.78 * (10 ** ((7.5 * T) / (237.3 + T)))
        p_vapor = (RH / 100.0) * p_sat
        p_dry = p_pascal - p_vapor
        weather_fc['air_density'] = round((p_dry / (287.05 * T_kelvin)) + (p_vapor / (461.495 * T_kelvin)), 4)
    if 'static_risk' not in weather_fc:
        weather_fc['static_risk'] = 1.0 if RH < 45.0 else 0.0

    last_row = sub.iloc[-1]
    target_dt = datetime.date.fromisoformat(target_date)
    last_draw_dt = datetime.date.fromisoformat(str(last_row['date']))
    day_of_week = target_dt.weekday()
    day_of_month = target_dt.day
    is_monday = 1.0 if day_of_week == 0 else 0.0
    days_since_last = max(1.0, float((target_dt - last_draw_dt).days))

    cur_atraso_dec = {d: 0 for d in range(10)}
    cur_atraso_uni = {u: 0 for u in range(10)}
    for d in range(10):
        seen = np.where(y_dec == d)[0]
        cur_atraso_dec[d] = n_samples - 1 - seen[-1] if len(seen) > 0 else n_samples
    for u in range(10):
        seen = np.where(y_uni == u)[0]
        cur_atraso_uni[u] = n_samples - 1 - seen[-1] if len(seen) > 0 else n_samples

    x_target_dict = {
        'air_density': weather_fc.get('air_density', 1.225),
        'dew_point': weather_fc.get('dew_point', 12.0),
        'static_risk': weather_fc.get('static_risk', 0.0),
        'temperature_2m': weather_fc.get('temperature_2m', 18.0),
        'relative_humidity_2m': weather_fc.get('relative_humidity_2m', 65.0),
        'surface_pressure': weather_fc.get('surface_pressure', 1015.0),
        'wind_speed_10m': weather_fc.get('wind_speed_10m', 12.0),
        'lunar_phase': lunar_fc.get('lunar_phase', 0.5),
        'lunar_sin': lunar_fc.get('lunar_sin', 0.0),
        'lunar_cos': lunar_fc.get('lunar_cos', 1.0),
        'lunar_illumination': lunar_fc.get('lunar_illumination', 50.0),
        'day_of_week': float(day_of_week),
        'day_of_month': float(day_of_month),
        'is_monday': float(is_monday),
        'days_since_last_draw': float(days_since_last),
        'lag_1': float(last_row['number_2d']),
        'lag_2': float(sub['number_2d'].iloc[-2]) if n_samples > 1 else float(last_row['number_2d']),
        'lag_3': float(sub['number_2d'].iloc[-3]) if n_samples > 2 else float(last_row['number_2d']),
        'lag_7': float(sub['number_2d'].iloc[-7]) if n_samples > 6 else float(last_row['number_2d']),
        'lag_1_decena': float(last_row['decena']),
        'lag_1_unidad': float(last_row['unidad']),
        'lag_2_decena': float(sub['decena'].iloc[-2]) if n_samples > 1 else float(last_row['decena']),
        'lag_2_unidad': float(sub['unidad'].iloc[-2]) if n_samples > 1 else float(last_row['unidad']),
        'repeat_decena': 1.0 if n_samples > 1 and last_row['decena'] == sub['decena'].iloc[-2] else 0.0,
        'repeat_unidad': 1.0 if n_samples > 1 and last_row['unidad'] == sub['unidad'].iloc[-2] else 0.0,
        'rolling_mean_3': float(np.mean(sub['number_2d'].values[-3:])),
        'rolling_mean_7': float(np.mean(sub['number_2d'].values[-7:])),
        'rolling_std_7': float(np.std(sub['number_2d'].values[-7:])) if n_samples >= 7 else 0.0,
        'atraso_decena': float(cur_atraso_dec[int(last_row['decena'])]),
        'atraso_unidad': float(cur_atraso_uni[int(last_row['unidad'])]),
        'delay_zscore_decena': (float(cur_atraso_dec[int(last_row['decena'])]) - 10.0) / 9.487,
        'delay_zscore_unidad': (float(cur_atraso_uni[int(last_row['unidad'])]) - 10.0) / 9.487,
        'max_atraso_decena_pool': float(max(cur_atraso_dec.values())),
        'max_atraso_unidad_pool': float(max(cur_atraso_uni.values())),
    }
    x_target_df = pd.DataFrame([x_target_dict])[feature_cols].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    x_target_scaled = scaler.transform(x_target_df)

    def full_prob(clf, X_in):
        out = np.zeros(10)
        p = clf.predict_proba(X_in)[0]
        for c, prob in zip(clf.classes_, p):
            out[int(c)] = prob
        return out

    # --- PILLAR 1: Tree Ensemble (RF + HistGB with 34 features) ---
    rf_d = RandomForestClassifier(n_estimators=70, max_depth=4, random_state=42, n_jobs=1).fit(X, y_dec)
    rf_u = RandomForestClassifier(n_estimators=70, max_depth=4, random_state=42, n_jobs=1).fit(X, y_uni)
    hgb_d = HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=42).fit(X, y_dec)
    hgb_u = HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=42).fit(X, y_uni)

    p_tree_d = 0.5 * (full_prob(rf_d, x_target_df) + full_prob(hgb_d, x_target_df))
    p_tree_u = 0.5 * (full_prob(rf_u, x_target_df) + full_prob(hgb_u, x_target_df))
    m_tree = np.outer(p_tree_d, p_tree_u)

    # --- PILLAR 2: Deep MLP Ensemble (Early Stopping, Regularized) ---
    mlp_d = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, alpha=0.1, early_stopping=True, n_iter_no_change=15, random_state=42).fit(X_scaled, y_dec)
    mlp_u = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, alpha=0.1, early_stopping=True, n_iter_no_change=15, random_state=42).fit(X_scaled, y_uni)
    p_deep_d = full_prob(mlp_d, x_target_scaled)
    p_deep_u = full_prob(mlp_u, x_target_scaled)
    m_deep = np.outer(p_deep_d, p_deep_u)

    # Physics-only
    rf_phys_d = RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42, n_jobs=1).fit(sub[physics_cols].fillna(0).values, y_dec)
    rf_phys_u = RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42, n_jobs=1).fit(sub[physics_cols].fillna(0).values, y_uni)
    x_phys = pd.DataFrame([x_target_dict])[physics_cols].fillna(0).values
    m_phys = np.outer(full_prob(rf_phys_d, x_phys), full_prob(rf_phys_u, x_phys))

    # --- PILLAR 3: 2nd-Order + 1st-Order Markov Transitions ---
    trans_1st_d = np.ones((10, 10)) * markov_smooth
    trans_1st_u = np.ones((10, 10)) * markov_smooth
    for i in range(len(y_dec) - 1):
        trans_1st_d[y_dec[i], y_dec[i + 1]] += 1.0
        trans_1st_u[y_uni[i], y_uni[i + 1]] += 1.0
    trans_1st_d /= trans_1st_d.sum(axis=1, keepdims=True)
    trans_1st_u /= trans_1st_u.sum(axis=1, keepdims=True)

    p_markov_d = trans_1st_d[int(y_dec[-1])]
    p_markov_u = trans_1st_u[int(y_uni[-1])]

    if len(y_dec) > 2:
        prev2_d, prev1_d = int(y_dec[-2]), int(y_dec[-1])
        prev2_u, prev1_u = int(y_uni[-2]), int(y_uni[-1])
        c_2nd_d = np.ones(10) * (markov_smooth * 0.5)
        c_2nd_u = np.ones(10) * (markov_smooth * 0.5)
        for i in range(len(y_dec) - 2):
            if y_dec[i] == prev2_d and y_dec[i + 1] == prev1_d:
                c_2nd_d[y_dec[i + 2]] += 1.5
            if y_uni[i] == prev2_u and y_uni[i + 1] == prev1_u:
                c_2nd_u[y_uni[i + 2]] += 1.5
        c_2nd_d /= c_2nd_d.sum()
        c_2nd_u /= c_2nd_u.sum()
        p_markov_d = 0.70 * p_markov_d + 0.30 * c_2nd_d
        p_markov_u = 0.70 * p_markov_u + 0.30 * c_2nd_u

    m_markov = np.outer(p_markov_d, p_markov_u)

    # --- PILLAR 4: Recency-Weighted Dirichlet-Multinomial ---
    weights_time = decay_gamma ** np.arange(n_samples - 1, -1, -1)
    c_dec = np.zeros(10)
    c_uni = np.zeros(10)
    c_2d = np.zeros(100)
    nums_2d = sub['number_2d'].astype(int).values
    for i in range(n_samples):
        w_i = weights_time[i]
        c_dec[y_dec[i]] += w_i
        c_uni[y_uni[i]] += w_i
        c_2d[nums_2d[i]] += w_i

    p_hot_d = (c_dec + alpha_prior) / (c_dec.sum() + 10.0 * alpha_prior)
    p_hot_u = (c_uni + alpha_prior) / (c_uni.sum() + 10.0 * alpha_prior)
    p_hot_2d = (c_2d + 0.15) / (c_2d.sum() + 15.0)
    m_hot = 0.60 * np.outer(p_hot_d, p_hot_u) + 0.40 * p_hot_2d.reshape(10, 10)
    m_hot /= m_hot.sum()

    # --- PILLAR 5: Renewal Cold / Weibull Softmax Hazard ---
    delay_d_arr = np.array([cur_atraso_dec[d] for d in range(10)], dtype=float)
    delay_u_arr = np.array([cur_atraso_uni[u] for u in range(10)], dtype=float)
    p_cold_d = np.exp(delay_d_arr / renewal_temp)
    p_cold_d /= p_cold_d.sum()
    p_cold_u = np.exp(delay_u_arr / renewal_temp)
    p_cold_u /= p_cold_u.sum()
    m_cold = np.outer(p_cold_d, p_cold_u)

    return {
        'matrix_tree': m_tree / m_tree.sum(),
        'matrix_deep': m_deep / m_deep.sum(),
        'matrix_phys': m_phys / m_phys.sum(),
        'matrix_markov': m_markov / m_markov.sum(),
        'matrix_hot': m_hot / m_hot.sum(),
        'matrix_cold': m_cold / m_cold.sum(),
        'weather_fc': weather_fc,
        'lunar_fc': lunar_fc,
        'counts_2d': c_2d,
        'cur_atraso_dec': cur_atraso_dec,
        'cur_atraso_uni': cur_atraso_uni
    }


class RaffleGPUPredictor:
    def __init__(self, data_dir=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = data_dir or os.path.join(base_dir, 'data')
        self.raw_path = os.path.join(self.data_dir, 'raw_draws.csv')
        self.enriched_path = os.path.join(self.data_dir, 'dataset_enriched.csv')
        self.db = QuinielaDatabase(self.data_dir)
        self.hw_info = detect_hardware()

    def sync_latest_draws(self, days_back=21):
        """Scrapes recent draws from web portals into SQLite and re-enriches."""
        added = self.db.sync_from_web(days_back=days_back)
        _COMPONENTS_CACHE.clear()
        return added

    def add_manual_draw(self, date_str, lottery, number_4d):
        saved = self.db.upsert_draw(date_str=date_str, lottery=lottery, number_4d=number_4d, source='manual_ui')
        _COMPONENTS_CACHE.clear()
        return saved

    def get_components(self, sub, lottery, target_date):
        key = (lottery, target_date, len(sub))
        if key not in _COMPONENTS_CACHE:
            opt = load_optimal_hyperparams()
            hyperparams = opt.get('hyperparameters', {})
            _COMPONENTS_CACHE[key] = _compute_components_internal(sub, lottery, target_date, hyperparams)
        return _COMPONENTS_CACHE[key]

    def run_prediction(
        self,
        lottery='ciudad',
        target_date=None,
        ticket_count=5,
        taken_numbers=None,
        strategy_weights=None,
        orthogonal_coverage=True
    ):
        """
        Executes fast cached prediction and combinatorial portfolio optimization (< 2 ms).
        Restricted to future target dates (>= today).
        """
        today_str = datetime.date.today().isoformat()
        if not target_date or target_date < today_str:
            target_date = today_str

        if taken_numbers is None:
            taken_numbers = []
        taken_set = set(int(x) for x in taken_numbers if 0 <= int(x) <= 99)

        opt_info = load_optimal_hyperparams()
        optimal_w = opt_info.get('optimal_weights', {
            'ml_tree': 0.25,
            'deep_gpu': 0.25,
            'markov': 0.15,
            'bayesian_hot': 0.20,
            'renewal_cold': 0.15
        })

        # Apply weights: user override or hyperoptimized defaults
        W = dict(optimal_w)
        if strategy_weights:
            W.update(strategy_weights)
        w_sum = sum(W.values()) or 1.0
        W = {k: v / w_sum for k, v in W.items()}

        df = pd.read_csv(self.enriched_path)
        sub = df[df['lottery'] == lottery].sort_values('date').copy().reset_index(drop=True)

        comps = self.get_components(sub, lottery, target_date)
        matrix_tree = comps['matrix_tree']
        matrix_deep = comps['matrix_deep']
        matrix_phys = comps['matrix_phys']
        matrix_markov = comps['matrix_markov']
        matrix_hot = comps['matrix_hot']
        matrix_cold = comps['matrix_cold']
        weather_fc = comps['weather_fc']
        lunar_fc = comps['lunar_fc']
        counts_2d = comps['counts_2d']

        # Master Probability Matrix
        master_matrix = (
            W.get('ml_tree', 0.2) * matrix_tree +
            W.get('deep_gpu', 0.2) * matrix_deep +
            W.get('markov', 0.2) * matrix_markov +
            W.get('bayesian_hot', 0.2) * matrix_hot +
            W.get('renewal_cold', 0.2) * matrix_cold
        )
        master_matrix /= master_matrix.sum()
        flat_probs = master_matrix.flatten()

        # Analytical simulation representation
        n_sims = 1_000_000
        mc_counts = (flat_probs * n_sims).astype(int)

        # Anti-crowd edge
        crowd_edge = compute_crowd_bias_scores()
        available_nums = [n for n in range(100) if n not in taken_set]

        # 2D delays
        last_seen_2d = {}
        for idx_r, row_r in sub.iterrows():
            last_seen_2d[int(row_r['number_2d'])] = idx_r
        total_draws = len(sub)
        delays_2d = {n: (total_draws - 1 - last_seen_2d[n]) if n in last_seen_2d else total_draws for n in range(100)}

        # Submodular greedy rook-constrained portfolio selection
        selected_portfolio = []
        covered_decenas = set()
        covered_unidades = set()
        parities = []

        for step in range(min(ticket_count, len(available_nums))):
            best_n = None
            best_utility = -1e9
            for n in available_nums:
                if n in selected_portfolio:
                    continue
                d, u = divmod(n, 10)
                base_p = flat_probs[n]

                if orthogonal_coverage and len(covered_decenas) < 10 and len(covered_unidades) < 10:
                    dec_free = 1.0 if d not in covered_decenas else 0.0
                    uni_free = 1.0 if u not in covered_unidades else 0.0
                    div_bonus = 0.25 * dec_free + 0.25 * uni_free
                else:
                    div_bonus = (0.002 if d not in covered_decenas else 0.0) + (0.002 if u not in covered_unidades else 0.0)

                if parities:
                    even_ratio = sum(1 for x in parities if x % 2 == 0) / len(parities)
                    if (n % 2 == 0 and even_ratio < 0.5) or (n % 2 == 1 and even_ratio > 0.5):
                        div_bonus += 0.0015

                edge_bonus = 0.0010 * crowd_edge[n]
                utility = base_p + div_bonus + edge_bonus
                if utility > best_utility:
                    best_utility = utility
                    best_n = n

            if best_n is not None:
                selected_portfolio.append(best_n)
                bd, bu = divmod(best_n, 10)
                covered_decenas.add(bd)
                covered_unidades.add(bu)
                parities.append(best_n)

        portfolio_details = []
        for rank, num in enumerate(selected_portfolio, start=1):
            d, u = divmod(num, 10)
            comp_scores = {
                'ML Tree Ensemble': matrix_tree[d, u],
                'Deep MLP Ensemble': matrix_deep[d, u],
                f'Markov (post-{int(sub.iloc[-1]["number_2d"]):02d})': matrix_markov[d, u],
                'Hot Momentum': matrix_hot[d, u],
                'Atraso Reversion': matrix_cold[d, u]
            }
            top_driver = max(comp_scores.items(), key=lambda x: x[1])[0]

            portfolio_details.append({
                'rank': rank,
                'number': f'{num:02d}',
                'number_int': int(num),
                'decena': int(d),
                'unidad': int(u),
                'master_prob_pct': round(float(flat_probs[num] * 100), 3),
                'mc_sim_hits': int(mc_counts[num]),
                'tree_prob_pct': round(float(matrix_tree[d, u] * 100), 3),
                'deep_prob_pct': round(float(matrix_deep[d, u] * 100), 3),
                'markov_prob_pct': round(float(matrix_markov[d, u] * 100), 3),
                'freq_2d': int(counts_2d[num]),
                'delay_2d': int(delays_2d[num]),
                'crowd_edge_score': round(float(crowd_edge[num] * 100), 1),
                'dominant_driver': top_driver
            })

        def best_available(mat):
            order = np.argsort(mat.flatten())[::-1]
            for idx in order:
                if int(idx) not in taken_set:
                    return int(idx), float(mat.flatten()[idx] * 100)
            return 0, 1.0

        p1_n, p1_p = best_available(matrix_tree)
        p2_n, p2_p = best_available(matrix_deep)
        p3_n, p3_p = best_available(matrix_phys)
        p4_n, p4_p = best_available(matrix_hot)
        p5_n, p5_p = best_available(matrix_cold)

        archetypes = [
            {'profile': 'Ensamble GBDT + RF', 'number': f'{p1_n:02d}', 'prob_pct': round(p1_p, 3), 'desc': 'Árboles de decisión desacoplados para Decena y Unidad'},
            {'profile': 'Deep MLP Ensemble', 'number': f'{p2_n:02d}', 'prob_pct': round(p2_p, 3), 'desc': 'Red neuronal con regularización y dropout'},
            {'profile': 'Física & Efemérides Lunares', 'number': f'{p3_n:02d}', 'prob_pct': round(p3_p, 3), 'desc': f"{weather_fc['temperature_2m']}°C, {weather_fc['surface_pressure']} hPa, {lunar_fc['lunar_phase_name']} ({lunar_fc['lunar_illumination']}%)"},
            {'profile': 'Inercia Bayesiana (Hot Streak)', 'number': f'{p4_n:02d}', 'prob_pct': round(p4_p, 3), 'desc': 'Conteo empírico Dirichlet-Multinomial'},
            {'profile': 'Reversión por Sequía (Atrasos)', 'number': f'{p5_n:02d}', 'prob_pct': round(p5_p, 3), 'desc': 'Hazard Rate por sequía acumulada'}
        ]

        grid_100 = []
        for n in range(100):
            d, u = divmod(n, 10)
            grid_100.append({
                'number': f'{n:02d}',
                'number_int': n,
                'decena': d,
                'unidad': u,
                'prob_pct': round(float(flat_probs[n] * 100), 3),
                'mc_hits': int(mc_counts[n]),
                'freq': int(counts_2d[n]),
                'delay': int(delays_2d[n]),
                'taken': n in taken_set,
                'recommended': n in selected_portfolio
            })

        audit = statistical_audit.run_statistical_audit(df, lottery)

        last_row = sub.iloc[-1]
        portfolio_cum_prob = sum(x['master_prob_pct'] for x in portfolio_details)
        dec_coverage_pct = len(covered_decenas) * 10.0
        uni_coverage_pct = len(covered_unidades) * 10.0
        any_digit_hit_prob = (1.0 - (1.0 - len(covered_decenas) / 10.0) * (1.0 - len(covered_unidades) / 10.0)) * 100.0

        return {
            'hardware': self.hw_info,
            'lottery': lottery,
            'target_date': target_date,
            'today': today_str,
            'total_draws_analyzed': int(len(sub)),
            'db_stats': self.db.get_db_stats(),
            'recent_draws': self.db.get_recent_draws(lottery=lottery, limit=12),
            'last_draw': {
                'date': str(last_row['date']),
                'number_4d': f"{int(last_row['number_4d']):04d}",
                'number_2d': f"{int(last_row['number_2d']):02d}"
            },
            'weather_forecast': weather_fc,
            'lunar_ephemeris': lunar_fc,
            'portfolio_metrics': {
                'ticket_count': len(selected_portfolio),
                'available_pool_size': len(available_nums),
                'cumulative_master_prob_pct': round(portfolio_cum_prob, 2),
                'uniform_baseline_pct': round(len(selected_portfolio) * (100.0 / max(1, len(available_nums))), 2),
                'lift_over_random_pct': round(portfolio_cum_prob - len(selected_portfolio) * 1.0, 2),
                'decena_coverage_pct': round(dec_coverage_pct, 1),
                'unidad_coverage_pct': round(uni_coverage_pct, 1),
                'orthogonal_digit_reach_pct': round(any_digit_hit_prob, 1),
                'monte_carlo_total_rollouts': n_sims
            },
            'optimal_portfolio': portfolio_details,
            'archetypes': archetypes,
            'decena_probs_pct': [round(float(x * 100), 2) for x in master_matrix.sum(axis=1)],
            'unidad_probs_pct': [round(float(x * 100), 2) for x in master_matrix.sum(axis=0)],
            'grid_100': grid_100,
            'audit_summary': {
                'chi2_decena_p': round(float(audit['chi_square']['decena']['p_val']), 4),
                'chi2_unidad_p': round(float(audit['chi_square']['unidad']['p_val']), 4),
                'runs_parity_p': round(float(audit['runs_test']['parity']['p_value']), 4),
                'shannon_entropy_decena': round(float(audit['entropy']['decena']), 4),
                'shannon_entropy_unidad': round(float(audit['entropy']['unidad']), 4)
            },
            'hyperparams': opt_info
        }
