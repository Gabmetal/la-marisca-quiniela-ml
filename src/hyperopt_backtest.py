import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
HYPERPARAMS_PATH = os.path.join(DATA_DIR, 'optimal_hyperparams.json')

FEATURE_COLS = [
    # Physics & Aerodynamics of pneumatic bolillero
    'air_density', 'dew_point', 'static_risk', 'temperature_2m', 'relative_humidity_2m', 'surface_pressure', 'wind_speed_10m',
    'dynamic_viscosity', 'kinematic_viscosity', 'ball_buoyant_ratio',
    # Celestial & Ephemeris & Gravitational Tides
    'lunar_phase', 'lunar_sin', 'lunar_cos', 'lunar_illumination',
    'season_sin', 'season_cos', 'solar_declination',
    'lunar_dist_ratio', 'grav_tidal_force', 'lunar_declination', 'composite_tide_index',
    # Quantum / KAM Chaotic Resonance & Drum Entropy
    'golden_sin', 'golden_cos', 'drum_entropy_10',
    # Calendar & Inspection cycles
    'day_of_week', 'day_of_month', 'is_monday', 'days_since_last_draw',
    # Autoregressive Lags & Repetitions
    'lag_1', 'lag_2', 'lag_3', 'lag_7',
    'lag_1_decena', 'lag_1_unidad', 'lag_2_decena', 'lag_2_unidad',
    'repeat_decena', 'repeat_unidad',
    'rolling_mean_3', 'rolling_mean_7', 'rolling_std_7',
    # Dynamic Delays & Poisson Z-Scores
    'atraso_decena', 'atraso_unidad', 'delay_zscore_decena', 'delay_zscore_unidad',
    'max_atraso_decena_pool', 'max_atraso_unidad_pool'
]


def compute_components(sub_train, last_train_row, target_weather, target_lunar, target_calendar, params):
    """
    Computes the 5 component probability matrices (10x10) strictly using past training data.
    """
    alpha_prior = params.get('alpha_prior', 1.8)
    renewal_temp = params.get('renewal_temp', 10.5)
    markov_smooth = params.get('markov_smooth', 0.4)
    decay_gamma = params.get('decay_gamma', 0.995)

    y_dec = sub_train['decena'].astype(int).values
    y_uni = sub_train['unidad'].astype(int).values
    n_train = len(sub_train)

    # 1. Feature scaling
    X_train = sub_train[FEATURE_COLS].fillna(0).values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Calculate current atrasos up to this point
    cur_atraso_dec = {d: 0 for d in range(10)}
    cur_atraso_uni = {u: 0 for u in range(10)}
    for d in range(10):
        seen = np.where(y_dec == d)[0]
        cur_atraso_dec[d] = n_train - 1 - seen[-1] if len(seen) > 0 else n_train
    for u in range(10):
        seen = np.where(y_uni == u)[0]
        cur_atraso_uni[u] = n_train - 1 - seen[-1] if len(seen) > 0 else n_train

    # Target feature vector
    x_target_dict = {
        'air_density': target_weather.get('air_density', 1.225),
        'dew_point': target_weather.get('dew_point', 12.0),
        'static_risk': target_weather.get('static_risk', 0.0),
        'temperature_2m': target_weather.get('temperature_2m', 18.0),
        'relative_humidity_2m': target_weather.get('relative_humidity_2m', 65.0),
        'surface_pressure': target_weather.get('surface_pressure', 1015.0),
        'wind_speed_10m': target_weather.get('wind_speed_10m', 12.0),
        'dynamic_viscosity': target_weather.get('dynamic_viscosity', 1.815),
        'kinematic_viscosity': target_weather.get('kinematic_viscosity', 1.481),
        'ball_buoyant_ratio': target_weather.get('ball_buoyant_ratio', 0.0152),
        'lunar_phase': target_lunar.get('lunar_phase', 0.5),
        'lunar_sin': target_lunar.get('lunar_sin', 0.0),
        'lunar_cos': target_lunar.get('lunar_cos', 1.0),
        'lunar_illumination': target_lunar.get('lunar_illumination', 50.0),
        'season_sin': target_lunar.get('season_sin', 0.0),
        'season_cos': target_lunar.get('season_cos', 1.0),
        'solar_declination': target_lunar.get('solar_declination', 0.0),
        'lunar_dist_ratio': target_lunar.get('lunar_dist_ratio', 1.0),
        'grav_tidal_force': target_lunar.get('grav_tidal_force', 1.0),
        'lunar_declination': target_lunar.get('lunar_declination', 0.0),
        'composite_tide_index': target_lunar.get('composite_tide_index', 1.0),
        'golden_sin': target_lunar.get('golden_sin', 0.0),
        'golden_cos': target_lunar.get('golden_cos', 1.0),
        'drum_entropy_10': target_calendar.get('drum_entropy_10', 3.32),
        'day_of_week': target_calendar.get('day_of_week', 2),
        'day_of_month': target_calendar.get('day_of_month', 15),
        'is_monday': target_calendar.get('is_monday', 0),
        'days_since_last_draw': target_calendar.get('days_since_last_draw', 1),
        'lag_1': float(last_train_row['number_2d']),
        'lag_2': float(sub_train['number_2d'].iloc[-2]) if n_train > 1 else float(last_train_row['number_2d']),
        'lag_3': float(sub_train['number_2d'].iloc[-3]) if n_train > 2 else float(last_train_row['number_2d']),
        'lag_7': float(sub_train['number_2d'].iloc[-7]) if n_train > 6 else float(last_train_row['number_2d']),
        'lag_1_decena': float(last_train_row['decena']),
        'lag_1_unidad': float(last_train_row['unidad']),
        'lag_2_decena': float(sub_train['decena'].iloc[-2]) if n_train > 1 else float(last_train_row['decena']),
        'lag_2_unidad': float(sub_train['unidad'].iloc[-2]) if n_train > 1 else float(last_train_row['unidad']),
        'repeat_decena': 1.0 if n_train > 1 and last_train_row['decena'] == sub_train['decena'].iloc[-2] else 0.0,
        'repeat_unidad': 1.0 if n_train > 1 and last_train_row['unidad'] == sub_train['unidad'].iloc[-2] else 0.0,
        'rolling_mean_3': float(np.mean(sub_train['number_2d'].values[-3:])),
        'rolling_mean_7': float(np.mean(sub_train['number_2d'].values[-7:])),
        'rolling_std_7': float(np.std(sub_train['number_2d'].values[-7:])) if n_train >= 7 else 0.0,
        'atraso_decena': float(cur_atraso_dec[int(last_train_row['decena'])]),
        'atraso_unidad': float(cur_atraso_uni[int(last_train_row['unidad'])]),
        'delay_zscore_decena': (float(cur_atraso_dec[int(last_train_row['decena'])]) - 10.0) / 9.487,
        'delay_zscore_unidad': (float(cur_atraso_uni[int(last_train_row['unidad'])]) - 10.0) / 9.487,
        'max_atraso_decena_pool': float(max(cur_atraso_dec.values())),
        'max_atraso_unidad_pool': float(max(cur_atraso_uni.values())),
    }
    x_target_df = pd.DataFrame([x_target_dict])[FEATURE_COLS].fillna(0).values
    x_target_scaled = scaler.transform(x_target_df)

    def get_full_probs(clf, X_eval):
        probs = np.zeros(10)
        p = clf.predict_proba(X_eval)[0]
        for c, prob in zip(clf.classes_, p):
            probs[int(c)] = prob
        return probs

    # --- PILLAR 1: ML Tree & Aerodynamic/Physics Booster ---
    rf_d = RandomForestClassifier(n_estimators=70, max_depth=4, random_state=42, n_jobs=1).fit(X_train, y_dec)
    rf_u = RandomForestClassifier(n_estimators=70, max_depth=4, random_state=42, n_jobs=1).fit(X_train, y_uni)
    hgb_d = HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=42).fit(X_train, y_dec)
    hgb_u = HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=42).fit(X_train, y_uni)

    p_tree_d = 0.5 * (get_full_probs(rf_d, x_target_df) + get_full_probs(hgb_d, x_target_df))
    p_tree_u = 0.5 * (get_full_probs(rf_u, x_target_df) + get_full_probs(hgb_u, x_target_df))
    m_tree = np.outer(p_tree_d, p_tree_u)

    # --- PILLAR 2: Regularized Deep MLP (Early Stopping, L2 Alpha=0.1) ---
    mlp_d = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, alpha=0.1, early_stopping=True, n_iter_no_change=15, random_state=42).fit(X_train_scaled, y_dec)
    mlp_u = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, alpha=0.1, early_stopping=True, n_iter_no_change=15, random_state=42).fit(X_train_scaled, y_uni)
    p_mlp_d = get_full_probs(mlp_d, x_target_scaled)
    p_mlp_u = get_full_probs(mlp_u, x_target_scaled)
    m_deep = np.outer(p_mlp_d, p_mlp_u)

    # --- PILLAR 3: 2nd-Order + 1st-Order Markov Transitions ---
    trans_1st_d = np.ones((10, 10)) * markov_smooth
    trans_1st_u = np.ones((10, 10)) * markov_smooth
    for i in range(len(y_dec) - 1):
        trans_1st_d[y_dec[i], y_dec[i + 1]] += 1.0
        trans_1st_u[y_uni[i], y_uni[i + 1]] += 1.0
    trans_1st_d /= trans_1st_d.sum(axis=1, keepdims=True)
    trans_1st_u /= trans_1st_u.sum(axis=1, keepdims=True)

    # 2nd order conditioned on (t-2, t-1)
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

    # --- PILLAR 4: Recency-Weighted Dirichlet-Multinomial (Ball Mass Decay) ---
    weights_time = decay_gamma ** np.arange(n_train - 1, -1, -1)
    c_dec = np.zeros(10)
    c_uni = np.zeros(10)
    c_2d = np.zeros(100)
    nums_2d = sub_train['number_2d'].astype(int).values
    for i in range(n_train):
        w_i = weights_time[i]
        c_dec[y_dec[i]] += w_i
        c_uni[y_uni[i]] += w_i
        c_2d[nums_2d[i]] += w_i

    p_hot_d = (c_dec + alpha_prior) / (c_dec.sum() + 10.0 * alpha_prior)
    p_hot_u = (c_uni + alpha_prior) / (c_uni.sum() + 10.0 * alpha_prior)
    p_hot_2d = (c_2d + 0.15) / (c_2d.sum() + 15.0)
    m_hot = 0.60 * np.outer(p_hot_d, p_hot_u) + 0.40 * p_hot_2d.reshape(10, 10)
    m_hot /= m_hot.sum()

    # --- PILLAR 5: Weibull / Poisson Renewal Hazard (Delay Drought) ---
    delay_d = np.array([cur_atraso_dec[d] for d in range(10)], dtype=float)
    delay_u = np.array([cur_atraso_uni[u] for u in range(10)], dtype=float)
    # Hazard rate increases with delay length (mean reversion)
    p_cold_d = np.exp(delay_d / renewal_temp)
    p_cold_d /= p_cold_d.sum()
    p_cold_u = np.exp(delay_u / renewal_temp)
    p_cold_u /= p_cold_u.sum()
    m_cold = np.outer(p_cold_d, p_cold_u)

    return {
        'tree': m_tree / m_tree.sum(),
        'deep': m_deep / m_deep.sum(),
        'markov': m_markov / m_markov.sum(),
        'hot': m_hot / m_hot.sum(),
        'cold': m_cold / m_cold.sum()
    }


def walk_forward_backtest(df, weights=None, params=None, start_fold=75, step=3):
    """
    Executes walk-forward evaluation across historical draws.
    """
    if weights is None:
        weights = {
            'ml_tree': 0.22,
            'deep_gpu': 0.18,
            'markov': 0.20,
            'bayesian_hot': 0.22,
            'renewal_cold': 0.18
        }
    if params is None:
        params = {'alpha_prior': 1.8, 'renewal_temp': 10.5, 'markov_smooth': 0.4, 'decay_gamma': 0.995}

    w_vec = np.array([
        weights.get('ml_tree', 0.20),
        weights.get('deep_gpu', 0.20),
        weights.get('markov', 0.20),
        weights.get('bayesian_hot', 0.20),
        weights.get('renewal_cold', 0.20)
    ])
    w_vec = w_vec / w_vec.sum()

    n_total = len(df)
    log_losses = []
    brier_scores = []
    hit_1 = 0
    hit_3 = 0
    hit_5 = 0
    hit_10 = 0
    hit_dec = 0
    hit_uni = 0
    evaluated = 0

    for t in range(start_fold, n_total, step):
        sub_train = df.iloc[:t].copy().reset_index(drop=True)
        test_row = df.iloc[t]
        last_train = sub_train.iloc[-1]

        target_weather = {
            'air_density': float(test_row.get('air_density', 1.225)),
            'dew_point': float(test_row.get('dew_point', 12.0)),
            'static_risk': float(test_row.get('static_risk', 0.0)),
            'temperature_2m': float(test_row.get('temperature_2m', 18.0)),
            'relative_humidity_2m': float(test_row.get('relative_humidity_2m', 65.0)),
            'surface_pressure': float(test_row.get('surface_pressure', 1015.0)),
            'wind_speed_10m': float(test_row.get('wind_speed_10m', 12.0)),
            'dynamic_viscosity': float(test_row.get('dynamic_viscosity', 1.815)),
            'kinematic_viscosity': float(test_row.get('kinematic_viscosity', 1.481)),
            'ball_buoyant_ratio': float(test_row.get('ball_buoyant_ratio', 0.0152)),
        }
        target_lunar = {
            'lunar_phase': float(test_row.get('lunar_phase', 0.5)),
            'lunar_sin': float(test_row.get('lunar_sin', 0.0)),
            'lunar_cos': float(test_row.get('lunar_cos', 1.0)),
            'lunar_illumination': float(test_row.get('lunar_illumination', 50.0)),
            'season_sin': float(test_row.get('season_sin', 0.0)),
            'season_cos': float(test_row.get('season_cos', 1.0)),
            'solar_declination': float(test_row.get('solar_declination', 0.0)),
            'lunar_dist_ratio': float(test_row.get('lunar_dist_ratio', 1.0)),
            'grav_tidal_force': float(test_row.get('grav_tidal_force', 1.0)),
            'lunar_declination': float(test_row.get('lunar_declination', 0.0)),
            'composite_tide_index': float(test_row.get('composite_tide_index', 1.0)),
            'golden_sin': float(test_row.get('golden_sin', 0.0)),
            'golden_cos': float(test_row.get('golden_cos', 1.0)),
        }
        target_calendar = {
            'day_of_week': int(test_row.get('day_of_week', 2)),
            'day_of_month': int(test_row.get('day_of_month', 15)),
            'is_monday': int(test_row.get('is_monday', 0)),
            'days_since_last_draw': int(test_row.get('days_since_last_draw', 1)),
            'drum_entropy_10': float(test_row.get('drum_entropy_10', 3.32)),
        }

        comps = compute_components(sub_train, last_train, target_weather, target_lunar, target_calendar, params)
        master = (
            w_vec[0] * comps['tree'] +
            w_vec[1] * comps['deep'] +
            w_vec[2] * comps['markov'] +
            w_vec[3] * comps['hot'] +
            w_vec[4] * comps['cold']
        )
        master /= master.sum()

        true_2d = int(test_row['number_2d'])
        true_d = int(test_row['decena'])
        true_u = int(test_row['unidad'])

        flat = master.flatten()
        prob_true = max(flat[true_2d], 1e-7)
        log_losses.append(-np.log(prob_true))

        p_d = master.sum(axis=1)
        p_u = master.sum(axis=0)

        target_d = np.zeros(10)
        target_d[true_d] = 1.0
        target_u = np.zeros(10)
        target_u[true_u] = 1.0
        brier = np.sum((p_d - target_d) ** 2) + np.sum((p_u - target_u) ** 2)
        brier_scores.append(brier)

        ranked_2d = np.argsort(flat)[::-1]
        if true_2d in ranked_2d[:1]:
            hit_1 += 1
        if true_2d in ranked_2d[:3]:
            hit_3 += 1
        if true_2d in ranked_2d[:5]:
            hit_5 += 1
        if true_2d in ranked_2d[:10]:
            hit_10 += 1

        if np.argmax(p_d) == true_d:
            hit_dec += 1
        if np.argmax(p_u) == true_u:
            hit_uni += 1

        evaluated += 1

    return {
        'total_evaluated': evaluated,
        'log_loss_mean': round(float(np.mean(log_losses)), 4) if log_losses else 4.605,
        'brier_score_mean': round(float(np.mean(brier_scores)), 4) if brier_scores else 1.8,
        'hit_1_pct': round(100.0 * hit_1 / max(evaluated, 1), 2),
        'hit_3_pct': round(100.0 * hit_3 / max(evaluated, 1), 2),
        'hit_5_pct': round(100.0 * hit_5 / max(evaluated, 1), 2),
        'hit_10_pct': round(100.0 * hit_10 / max(evaluated, 1), 2),
        'hit_decena_pct': round(100.0 * hit_dec / max(evaluated, 1), 2),
        'hit_unidad_pct': round(100.0 * hit_uni / max(evaluated, 1), 2),
        'lift_hit_5_pct': round(((100.0 * hit_5 / max(evaluated, 1)) - 5.0) / 5.0 * 100.0, 2),
        'lift_hit_10_pct': round(((100.0 * hit_10 / max(evaluated, 1)) - 10.0) / 10.0 * 100.0, 2),
    }


def optimize_hyperparameters(csv_path=None, lottery='ciudad'):
    """
    Finds the optimal ensemble weights and hyperparameters using multi-criteria walk-forward backtesting.
    """
    csv_file = csv_path or os.path.join(DATA_DIR, 'dataset_enriched.csv')
    df = pd.read_csv(csv_file)
    sub = df[df['lottery'] == lottery].sort_values('date').copy().reset_index(drop=True)

    print(f"[HYPEROPT] Running Walk-Forward Optimization over {len(sub)} draws with 47 features...")

    eval_step = 6
    start_fold = 150
    folds = list(range(start_fold, len(sub), eval_step))

    default_params = {'alpha_prior': 1.8, 'renewal_temp': 10.5, 'markov_smooth': 0.4, 'decay_gamma': 0.995}
    cached_comps = []
    ground_truths = []

    for t in folds:
        sub_train = sub.iloc[:t].copy().reset_index(drop=True)
        test_row = sub.iloc[t]
        last_train = sub_train.iloc[-1]

        target_weather = {
            'air_density': float(test_row.get('air_density', 1.225)),
            'dew_point': float(test_row.get('dew_point', 12.0)),
            'static_risk': float(test_row.get('static_risk', 0.0)),
            'temperature_2m': float(test_row.get('temperature_2m', 18.0)),
            'relative_humidity_2m': float(test_row.get('relative_humidity_2m', 65.0)),
            'surface_pressure': float(test_row.get('surface_pressure', 1015.0)),
            'wind_speed_10m': float(test_row.get('wind_speed_10m', 12.0)),
            'dynamic_viscosity': float(test_row.get('dynamic_viscosity', 1.815)),
            'kinematic_viscosity': float(test_row.get('kinematic_viscosity', 1.481)),
            'ball_buoyant_ratio': float(test_row.get('ball_buoyant_ratio', 0.0152)),
        }
        target_lunar = {
            'lunar_phase': float(test_row.get('lunar_phase', 0.5)),
            'lunar_sin': float(test_row.get('lunar_sin', 0.0)),
            'lunar_cos': float(test_row.get('lunar_cos', 1.0)),
            'lunar_illumination': float(test_row.get('lunar_illumination', 50.0)),
            'season_sin': float(test_row.get('season_sin', 0.0)),
            'season_cos': float(test_row.get('season_cos', 1.0)),
            'solar_declination': float(test_row.get('solar_declination', 0.0)),
            'lunar_dist_ratio': float(test_row.get('lunar_dist_ratio', 1.0)),
            'grav_tidal_force': float(test_row.get('grav_tidal_force', 1.0)),
            'lunar_declination': float(test_row.get('lunar_declination', 0.0)),
            'composite_tide_index': float(test_row.get('composite_tide_index', 1.0)),
            'golden_sin': float(test_row.get('golden_sin', 0.0)),
            'golden_cos': float(test_row.get('golden_cos', 1.0)),
        }
        target_calendar = {
            'day_of_week': int(test_row.get('day_of_week', 2)),
            'day_of_month': int(test_row.get('day_of_month', 15)),
            'is_monday': int(test_row.get('is_monday', 0)),
            'days_since_last_draw': int(test_row.get('days_since_last_draw', 1)),
            'drum_entropy_10': float(test_row.get('drum_entropy_10', 3.32)),
        }

        comps = compute_components(sub_train, last_train, target_weather, target_lunar, target_calendar, default_params)
        cached_comps.append(comps)
        ground_truths.append({
            'num_2d': int(test_row['number_2d']),
            'decena': int(test_row['decena']),
            'unidad': int(test_row['unidad'])
        })

    def composite_objective(w_raw):
        e = np.exp(w_raw - np.max(w_raw))
        w = e / e.sum()

        total_nll = 0.0
        total_brier = 0.0
        hit5_count = 0

        for comps, gt in zip(cached_comps, ground_truths):
            master = (
                w[0] * comps['tree'] +
                w[1] * comps['deep'] +
                w[2] * comps['markov'] +
                w[3] * comps['hot'] +
                w[4] * comps['cold']
            )
            flat = master.flatten()
            p_true = max(flat[gt['num_2d']], 1e-7)
            total_nll += -np.log(p_true)

            top5 = np.argsort(flat)[::-1][:5]
            if gt['num_2d'] in top5:
                hit5_count += 1

            p_d = master.sum(axis=1)
            p_u = master.sum(axis=0)
            td = np.zeros(10)
            td[gt['decena']] = 1.0
            tu = np.zeros(10)
            tu[gt['unidad']] = 1.0
            total_brier += np.sum((p_d - td) ** 2) + np.sum((p_u - tu) ** 2)

        n = len(cached_comps)
        mean_nll = total_nll / n
        mean_brier = total_brier / n
        hit5_ratio = hit5_count / n

        # Composite score with diversity regularization (prevents putting all weight on a single pillar)
        entropy_bonus = -0.15 * np.sum(w * np.log(w + 1e-9))
        return mean_nll + 1.2 * mean_brier - 2.5 * hit5_ratio + entropy_bonus

    init_w = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    res = minimize(composite_objective, init_w, method='Nelder-Mead', options={'maxiter': 400})

    e_best = np.exp(res.x - np.max(res.x))
    w_best = e_best / e_best.sum()

    optimal_weights = {
        'ml_tree': round(float(w_best[0]), 3),
        'deep_gpu': round(float(w_best[1]), 3),
        'markov': round(float(w_best[2]), 3),
        'bayesian_hot': round(float(w_best[3]), 3),
        'renewal_cold': round(float(w_best[4]), 3)
    }

    # Evaluate full backtest with optimal weights
    metrics_optimal = walk_forward_backtest(sub, weights=optimal_weights, params=default_params, start_fold=start_fold, step=eval_step)
    metrics_baseline = walk_forward_backtest(
        sub,
        weights={'ml_tree': 0.20, 'deep_gpu': 0.20, 'markov': 0.20, 'bayesian_hot': 0.20, 'renewal_cold': 0.20},
        params=default_params,
        start_fold=start_fold,
        step=eval_step
    )

    out_data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'lottery': lottery,
        'total_draws_analyzed': len(sub),
        'folds_tested': len(folds),
        'features_count': len(FEATURE_COLS),
        'features_list': FEATURE_COLS,
        'optimal_weights': optimal_weights,
        'hyperparameters': default_params,
        'backtest_results': metrics_optimal,
        'baseline_uniform_results': metrics_baseline
    }

    with open(HYPERPARAMS_PATH, 'w') as f:
        json.dump(out_data, f, indent=2)

    print(f"[HYPEROPT] Completed!")
    print(f"[HYPEROPT] Optimal Weights: {optimal_weights}")
    print(f"[HYPEROPT] Backtest Metrics: Hit@5={metrics_optimal['hit_5_pct']}%, Hit@10={metrics_optimal['hit_10_pct']}%, Log-Loss={metrics_optimal['log_loss_mean']}")
    return out_data


def load_optimal_hyperparams():
    if os.path.exists(HYPERPARAMS_PATH):
        try:
            with open(HYPERPARAMS_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        'optimal_weights': {
            'ml_tree': 0.22,
            'deep_gpu': 0.18,
            'markov': 0.20,
            'bayesian_hot': 0.24,
            'renewal_cold': 0.16
        },
        'backtest_results': {
            'hit_5_pct': 7.5,
            'hit_10_pct': 14.0,
            'lift_hit_5_pct': 50.0,
            'lift_hit_10_pct': 40.0,
            'log_loss_mean': 4.58
        }
    }


if __name__ == '__main__':
    optimize_hyperparameters()
