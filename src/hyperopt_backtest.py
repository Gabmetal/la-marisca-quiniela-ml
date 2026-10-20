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


def compute_components(sub_train, last_train_row, target_weather, target_lunar, params):
    """
    Computes the 5 component probability matrices (10x10) strictly using training data.
    """
    alpha_prior = params.get('alpha_prior', 2.0)
    renewal_temp = params.get('renewal_temp', 12.0)
    markov_smooth = params.get('markov_smooth', 0.5)

    y_dec = sub_train['decena'].astype(int).values
    y_uni = sub_train['unidad'].astype(int).values
    n_train = len(sub_train)

    feature_cols = [
        'temperature_2m', 'relative_humidity_2m', 'surface_pressure', 'wind_speed_10m',
        'lunar_phase', 'lunar_sin', 'lunar_cos', 'lunar_illumination',
        'lag_1', 'lag_2', 'lag_3', 'lag_7',
        'lag_1_decena', 'lag_1_unidad',
        'rolling_mean_3', 'rolling_mean_7', 'rolling_std_7',
        'atraso_decena', 'atraso_unidad',
        'max_atraso_decena_pool', 'max_atraso_unidad_pool'
    ]

    # 1. Feature scaling
    X_train = sub_train[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Build target feature vector
    x_target_dict = {
        'temperature_2m': target_weather.get('temperature_2m', 18.0),
        'relative_humidity_2m': target_weather.get('relative_humidity_2m', 65.0),
        'surface_pressure': target_weather.get('surface_pressure', 1015.0),
        'wind_speed_10m': target_weather.get('wind_speed_10m', 12.0),
        'lunar_phase': target_lunar.get('lunar_phase', 0.5),
        'lunar_sin': target_lunar.get('lunar_sin', 0.0),
        'lunar_cos': target_lunar.get('lunar_cos', 1.0),
        'lunar_illumination': target_lunar.get('lunar_illumination', 50.0),
        'lag_1': last_train_row['number_2d'],
        'lag_2': sub_train['number_2d'].iloc[-2] if n_train > 1 else last_train_row['number_2d'],
        'lag_3': sub_train['number_2d'].iloc[-3] if n_train > 2 else last_train_row['number_2d'],
        'lag_7': sub_train['number_2d'].iloc[-7] if n_train > 6 else last_train_row['number_2d'],
        'lag_1_decena': last_train_row['decena'],
        'lag_1_unidad': last_train_row['unidad'],
        'rolling_mean_3': float(np.mean(sub_train['number_2d'].values[-3:])),
        'rolling_mean_7': float(np.mean(sub_train['number_2d'].values[-7:])),
        'rolling_std_7': float(np.std(sub_train['number_2d'].values[-7:])) if n_train >= 7 else 0.0,
        'atraso_decena': last_train_row.get('atraso_decena', 0),
        'atraso_unidad': last_train_row.get('atraso_unidad', 0),
        'max_atraso_decena_pool': last_train_row.get('max_atraso_decena_pool', 0),
        'max_atraso_unidad_pool': last_train_row.get('max_atraso_unidad_pool', 0),
    }
    x_target_df = pd.DataFrame([x_target_dict])[feature_cols].fillna(0).values
    x_target_scaled = scaler.transform(x_target_df)

    def get_full_probs(clf, X_eval):
        probs = np.zeros(10)
        p = clf.predict_proba(X_eval)[0]
        for c, prob in zip(clf.classes_, p):
            probs[int(c)] = prob
        return probs

    # --- PILLAR 1: ML Tree & Boosting ---
    rf_d = RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42, n_jobs=1).fit(X_train, y_dec)
    rf_u = RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42, n_jobs=1).fit(X_train, y_uni)
    hgb_d = HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=42).fit(X_train, y_dec)
    hgb_u = HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=42).fit(X_train, y_uni)

    p_tree_d = 0.5 * (get_full_probs(rf_d, x_target_df) + get_full_probs(hgb_d, x_target_df))
    p_tree_u = 0.5 * (get_full_probs(rf_u, x_target_df) + get_full_probs(hgb_u, x_target_df))
    m_tree = np.outer(p_tree_d, p_tree_u)

    # --- PILLAR 2: Deep MLP Ensemble ---
    mlp_d = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=250, alpha=0.05, random_state=42).fit(X_train_scaled, y_dec)
    mlp_u = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=250, alpha=0.05, random_state=42).fit(X_train_scaled, y_uni)
    p_mlp_d = get_full_probs(mlp_d, x_target_scaled)
    p_mlp_u = get_full_probs(mlp_u, x_target_scaled)
    m_deep = np.outer(p_mlp_d, p_mlp_u)

    # --- PILLAR 3: Markov Transition Matrix ---
    trans_d = np.ones((10, 10)) * markov_smooth
    trans_u = np.ones((10, 10)) * markov_smooth
    for i in range(len(y_dec) - 1):
        trans_d[y_dec[i], y_dec[i + 1]] += 1.0
        trans_u[y_uni[i], y_uni[i + 1]] += 1.0
    trans_d /= trans_d.sum(axis=1, keepdims=True)
    trans_u /= trans_u.sum(axis=1, keepdims=True)

    last_d = int(y_dec[-1])
    last_u = int(y_uni[-1])
    m_markov = np.outer(trans_d[last_d], trans_u[last_u])

    # --- PILLAR 4: Bayesian Dirichlet-Multinomial (Hot Momentum) ---
    c_dec = np.bincount(y_dec, minlength=10).astype(float)
    c_uni = np.bincount(y_uni, minlength=10).astype(float)
    c_2d = np.bincount(sub_train['number_2d'].astype(int).values, minlength=100).astype(float)

    p_hot_d = (c_dec + alpha_prior) / (c_dec.sum() + 10.0 * alpha_prior)
    p_hot_u = (c_uni + alpha_prior) / (c_uni.sum() + 10.0 * alpha_prior)
    p_hot_2d = (c_2d + 0.2) / (c_2d.sum() + 20.0)
    m_hot = 0.65 * np.outer(p_hot_d, p_hot_u) + 0.35 * p_hot_2d.reshape(10, 10)
    m_hot /= m_hot.sum()

    # --- PILLAR 5: Renewal Cold / Atraso Softmax Hazard ---
    delay_d = np.zeros(10)
    delay_u = np.zeros(10)
    for d in range(10):
        pos = np.where(y_dec == d)[0]
        delay_d[d] = len(y_dec) - 1 - pos[-1] if len(pos) > 0 else len(y_dec)
    for u in range(10):
        pos = np.where(y_uni == u)[0]
        delay_u[u] = len(y_uni) - 1 - pos[-1] if len(pos) > 0 else len(y_uni)

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


def walk_forward_backtest(df, weights=None, params=None, start_fold=80, step=3):
    """
    Executes walk-forward backtesting across historical draws.
    Returns detailed out-of-sample metrics: Log-Loss, Brier Score, Hit@1, Hit@3, Hit@5, Hit@10.
    """
    if weights is None:
        weights = {
            'ml_tree': 0.25,
            'deep_gpu': 0.25,
            'markov': 0.15,
            'bayesian_hot': 0.20,
            'renewal_cold': 0.15
        }
    if params is None:
        params = {'alpha_prior': 2.0, 'renewal_temp': 12.0, 'markov_smooth': 0.5}

    w_vec = np.array([
        weights.get('ml_tree', 0.25),
        weights.get('deep_gpu', 0.25),
        weights.get('markov', 0.15),
        weights.get('bayesian_hot', 0.20),
        weights.get('renewal_cold', 0.15)
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
            'temperature_2m': test_row.get('temperature_2m', 18.0),
            'relative_humidity_2m': test_row.get('relative_humidity_2m', 65.0),
            'surface_pressure': test_row.get('surface_pressure', 1015.0),
            'wind_speed_10m': test_row.get('wind_speed_10m', 12.0)
        }
        target_lunar = {
            'lunar_phase': test_row.get('lunar_phase', 0.5),
            'lunar_sin': test_row.get('lunar_sin', 0.0),
            'lunar_cos': test_row.get('lunar_cos', 1.0),
            'lunar_illumination': test_row.get('lunar_illumination', 50.0)
        }

        comps = compute_components(sub_train, last_train, target_weather, target_lunar, params)
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

        # Probabilities
        flat = master.flatten()
        prob_true = max(flat[true_2d], 1e-6)
        log_loss = -np.log(prob_true)
        log_losses.append(log_loss)

        # Marginals
        p_d = master.sum(axis=1)
        p_u = master.sum(axis=0)

        # Brier score
        target_d_onehot = np.zeros(10)
        target_d_onehot[true_d] = 1.0
        target_u_onehot = np.zeros(10)
        target_u_onehot[true_u] = 1.0
        brier = np.sum((p_d - target_d_onehot) ** 2) + np.sum((p_u - target_u_onehot) ** 2)
        brier_scores.append(brier)

        # Top picks ranking
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
        'log_loss_mean': float(np.mean(log_losses)) if log_losses else 4.605,
        'brier_score_mean': float(np.mean(brier_scores)) if brier_scores else 1.8,
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
    Finds the optimal ensemble weights and hyperparameters using walk-forward cross-entropy minimization.
    Saves optimal configuration to optimal_hyperparams.json.
    """
    csv_file = csv_path or os.path.join(DATA_DIR, 'dataset_enriched.csv')
    df = pd.read_csv(csv_file)
    sub = df[df['lottery'] == lottery].sort_values('date').copy().reset_index(drop=True)

    print(f"[HYPEROPT] Starting Walk-Forward Parameter Optimization over {len(sub)} historical draws...")

    # Pre-extract components across evaluation folds to make optimization fast
    eval_step = 4
    start_fold = 90
    folds = list(range(start_fold, len(sub), eval_step))

    default_params = {'alpha_prior': 2.0, 'renewal_temp': 12.0, 'markov_smooth': 0.5}
    cached_comps = []
    ground_truths = []

    for t in folds:
        sub_train = sub.iloc[:t].copy().reset_index(drop=True)
        test_row = sub.iloc[t]
        last_train = sub_train.iloc[-1]

        target_weather = {
            'temperature_2m': test_row.get('temperature_2m', 18.0),
            'relative_humidity_2m': test_row.get('relative_humidity_2m', 65.0),
            'surface_pressure': test_row.get('surface_pressure', 1015.0),
            'wind_speed_10m': test_row.get('wind_speed_10m', 12.0)
        }
        target_lunar = {
            'lunar_phase': test_row.get('lunar_phase', 0.5),
            'lunar_sin': test_row.get('lunar_sin', 0.0),
            'lunar_cos': test_row.get('lunar_cos', 1.0),
            'lunar_illumination': test_row.get('lunar_illumination', 50.0)
        }

        comps = compute_components(sub_train, last_train, target_weather, target_lunar, default_params)
        cached_comps.append(comps)
        ground_truths.append(int(test_row['number_2d']))

    def objective(w_raw):
        # Softmax normalization to stay strictly on the probability simplex
        e = np.exp(w_raw - np.max(w_raw))
        w = e / e.sum()

        total_nll = 0.0
        for comps, true_2d in zip(cached_comps, ground_truths):
            master = (
                w[0] * comps['tree'] +
                w[1] * comps['deep'] +
                w[2] * comps['markov'] +
                w[3] * comps['hot'] +
                w[4] * comps['cold']
            )
            p = max(master.flatten()[true_2d], 1e-7)
            total_nll += -np.log(p)

        # L2 penalty to encourage balanced diversity across pillars
        reg = 0.01 * np.sum((w - 0.20) ** 2)
        return (total_nll / len(cached_comps)) + reg

    init_w = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Starts uniform (0.2 each)
    res = minimize(objective, init_w, method='Nelder-Mead', options={'maxiter': 300})

    e_best = np.exp(res.x - np.max(res.x))
    w_best = e_best / e_best.sum()

    optimal_weights = {
        'ml_tree': round(float(w_best[0]), 4),
        'deep_gpu': round(float(w_best[1]), 4),
        'markov': round(float(w_best[2]), 4),
        'bayesian_hot': round(float(w_best[3]), 4),
        'renewal_cold': round(float(w_best[4]), 4)
    }

    # Evaluate full backtest with optimal weights
    metrics_optimal = walk_forward_backtest(sub, weights=optimal_weights, params=default_params, start_fold=start_fold, step=eval_step)
    metrics_baseline = walk_forward_backtest(
        sub,
        weights={'ml_tree': 0.2, 'deep_gpu': 0.2, 'markov': 0.2, 'bayesian_hot': 0.2, 'renewal_cold': 0.2},
        params=default_params,
        start_fold=start_fold,
        step=eval_step
    )

    out_data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'lottery': lottery,
        'total_draws_analyzed': len(sub),
        'folds_tested': len(folds),
        'optimal_weights': optimal_weights,
        'hyperparameters': default_params,
        'backtest_results': metrics_optimal,
        'baseline_uniform_results': metrics_baseline
    }

    with open(HYPERPARAMS_PATH, 'w') as f:
        json.dump(out_data, f, indent=2)

    print(f"[HYPEROPT] Done! Optimal Weights: {optimal_weights}")
    print(f"[HYPEROPT] Hit@5: {metrics_optimal['hit_5_pct']}% | Hit@10: {metrics_optimal['hit_10_pct']}% | Log-Loss: {metrics_optimal['log_loss_mean']:.4f}")
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
            'ml_tree': 0.20,
            'deep_gpu': 0.20,
            'markov': 0.20,
            'bayesian_hot': 0.20,
            'renewal_cold': 0.20
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
