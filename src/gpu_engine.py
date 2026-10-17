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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enrich_features
import scrape_draws
import statistical_audit


def detect_hardware():
    """Detects GPU hardware via PyTorch CUDA, XGBoost CUDA, TensorFlow, or nvidia-smi."""
    info = {
        'gpu_available': False,
        'backend': 'CPU (NumPy + Scikit-Learn Vectorized)',
        'device_name': 'CPU',
        'vram_mb': 0,
        'driver': 'N/A',
        'torch_cuda': False,
        'xgb_cuda': False
    }
    # 1. Check nvidia-smi for GPU hardware details
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL,
            timeout=3
        ).decode('utf-8').strip()
        if out:
            parts = [p.strip() for p in out.split(',')]
            info['gpu_available'] = True
            info['device_name'] = parts[0]
            info['vram_mb'] = int(float(parts[1])) if len(parts) > 1 else 0
            info['driver'] = parts[2] if len(parts) > 2 else 'N/A'
    except Exception:
        pass

    # 2. Check PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            info['gpu_available'] = True
            info['torch_cuda'] = True
            info['device_name'] = torch.cuda.get_device_name(0)
            info['vram_mb'] = int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))
            info['backend'] = f'PyTorch {torch.__version__} (CUDA {torch.version.cuda})'
            return info
    except Exception:
        pass

    # 3. Check XGBoost CUDA GPU support
    if info['gpu_available']:
        try:
            import xgboost as xgb
            info['xgb_cuda'] = True
            info['backend'] = f'XGBoost {xgb.__version__} (CUDA GPU) + Deep MLP Ensemble'
            return info
        except Exception:
            pass
        try:
            import keras
            info['backend'] = f'Keras {keras.__version__} / TensorFlow + Host GPU ({info["device_name"]})'
        except Exception:
            info['backend'] = f'Accelerated Vector Engine ({info["device_name"]} Ready)'

    return info


def train_deep_bolillero(X_scaled, y_dec, y_uni, X_pred_scaled, hw_info):
    """
    Trains a Deep Residual / CUDA GPU Ensemble with Monte Carlo Bootstrap
    using PyTorch CUDA, XGBoost CUDA GPU, or Multi-Seed Deep MLP ensemble.
    Returns (p_dec_deep, p_uni_deep, epistemic_uncertainty_dec, epistemic_uncertainty_uni).
    """
    n_classes = 10
    if hw_info.get('torch_cuda', False):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        device = torch.device('cuda:0')
        torch.manual_seed(42)
        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        yd_t = torch.tensor(y_dec.values, dtype=torch.long, device=device)
        yu_t = torch.tensor(y_uni.values, dtype=torch.long, device=device)
        Xp_t = torch.tensor(X_pred_scaled, dtype=torch.float32, device=device)

        class ResidualBolilleroNet(nn.Module):
            def __init__(self, in_dim):
                super().__init__()
                self.fc1 = nn.Linear(in_dim, 64)
                self.bn1 = nn.BatchNorm1d(64)
                self.drop1 = nn.Dropout(0.25)
                self.fc2 = nn.Linear(64, 64)
                self.bn2 = nn.BatchNorm1d(64)
                self.drop2 = nn.Dropout(0.25)
                self.head_dec = nn.Linear(64, 10)
                self.head_uni = nn.Linear(64, 10)

            def forward(self, x):
                h1 = F.gelu(self.bn1(self.fc1(x)))
                h1 = self.drop1(h1)
                h2 = F.gelu(self.bn2(self.fc2(h1)))
                h2 = self.drop2(h2) + h1
                return self.head_dec(h2), self.head_uni(h2)

        model = ResidualBolilleroNet(X_scaled.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        model.train()
        for _ in range(250):
            optimizer.zero_grad()
            out_d, out_u = model(X_t)
            loss = criterion(out_d, yd_t) + criterion(out_u, yu_t)
            loss.backward()
            optimizer.step()

        model.train()
        with torch.no_grad():
            X_rep = Xp_t.repeat(500, 1)
            logits_d, logits_u = model(X_rep)
            probs_d = F.softmax(logits_d, dim=-1).cpu().numpy()
            probs_u = F.softmax(logits_u, dim=-1).cpu().numpy()

        return probs_d.mean(axis=0), probs_u.mean(axis=0), probs_d.std(axis=0), probs_u.std(axis=0)

    else:
        # Multi-seed Deep Neural Ensemble + optional XGBoost CUDA GPU booster
        seeds = [42, 101, 202, 303, 404, 505, 606, 707]
        probs_d_list, probs_u_list = [], []

        if hw_info.get('xgb_cuda', False):
            try:
                import xgboost as xgb
                for s in [42, 101, 202]:
                    xd = xgb.XGBClassifier(
                        n_estimators=80, max_depth=4, learning_rate=0.08,
                        subsample=0.85, colsample_bytree=0.85,
                        tree_method='hist', device='cuda', random_state=s, verbosity=0
                    )
                    xu = xgb.XGBClassifier(
                        n_estimators=80, max_depth=4, learning_rate=0.08,
                        subsample=0.85, colsample_bytree=0.85,
                        tree_method='hist', device='cuda', random_state=s + 500, verbosity=0
                    )
                    xd.fit(X_scaled, y_dec)
                    xu.fit(X_scaled, y_uni)
                    probs_d_list.append(xd.predict_proba(X_pred_scaled)[0])
                    probs_u_list.append(xu.predict_proba(X_pred_scaled)[0])
            except Exception:
                pass

        for s in seeds:
            mlp_d = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='relu', alpha=0.05, max_iter=350, random_state=s)
            mlp_u = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='relu', alpha=0.05, max_iter=350, random_state=s + 1000)
            mlp_d.fit(X_scaled, y_dec)
            mlp_u.fit(X_scaled, y_uni)

            pd_vec = np.zeros(n_classes)
            for c, p in zip(mlp_d.classes_, mlp_d.predict_proba(X_pred_scaled)[0]):
                pd_vec[int(c)] = p
            pu_vec = np.zeros(n_classes)
            for c, p in zip(mlp_u.classes_, mlp_u.predict_proba(X_pred_scaled)[0]):
                pu_vec[int(c)] = p

            probs_d_list.append(pd_vec)
            probs_u_list.append(pu_vec)

        probs_d_arr = np.array(probs_d_list)
        probs_u_arr = np.array(probs_u_list)
        return (
            probs_d_arr.mean(axis=0),
            probs_u_arr.mean(axis=0),
            probs_d_arr.std(axis=0),
            probs_u_arr.std(axis=0)
        )


def compute_crowd_bias_scores():
    """
    Computes human selection bias in Argentine raffles (00-99).
    Lower crowd popularity = higher availability & contrarian value.
    """
    crowd_popularity = np.ones(100, dtype=float)
    # 1. Birthdays / Calendar dates (01-31) are heavily over-picked
    for n in range(1, 32):
        crowd_popularity[n] += 0.35
    # 2. Iconic "Smorfia / Quiniela" cultural numbers
    cultural_favorites = {
        0: 0.45,   # Huevos
        13: 0.65,  # La Yeta
        14: 0.35,  # El Borracho
        15: 0.50,  # La Niña Bonita
        17: 0.55,  # La Desgracia
        22: 0.65,  # El Loco
        32: 0.45,  # El Dinero
        33: 0.50,  # Cristo
        47: 0.35,  # El Muerto
        48: 0.55,  # Muerto que habla
        69: 0.40,  # Vicios
        77: 0.45,  # Piernas de mujer
        88: 0.45,  # El Papa
        99: 0.40   # Los Hermanos
    }
    for num, boost in cultural_favorites.items():
        crowd_popularity[num] += boost
    # 3. Repdigits (ambos dobles: 11, 22, 33, ...)
    for d in range(10):
        crowd_popularity[d * 11] += 0.25

    # Normalize into anti-crowd edge in [0, 1]
    inv = 1.0 / crowd_popularity
    edge = (inv - inv.min()) / (inv.max() - inv.min() + 1e-9)
    return edge


from database import QuinielaDatabase


class RaffleGPUPredictor:
    def __init__(self, data_dir=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = data_dir or os.path.join(base_dir, 'data')
        self.raw_path = os.path.join(self.data_dir, 'raw_draws.csv')
        self.enriched_path = os.path.join(self.data_dir, 'dataset_enriched.csv')
        self.db = QuinielaDatabase(self.data_dir)
        self.hw_info = detect_hardware()

    def sync_latest_draws(self, days_back=21):
        """Scrapes any missing recent draws from web portals into SQLite and re-enriches the dataset."""
        return self.db.sync_from_web(days_back=days_back)

    def add_manual_draw(self, date_str, lottery, number_4d):
        """Inserts or updates a draw in SQLite, enriches telemetry/lags, and prepares for re-training."""
        return self.db.upsert_draw(date_str=date_str, lottery=lottery, number_4d=number_4d, source='manual_ui')

    def run_prediction(
        self,
        lottery='ciudad',
        target_date='2026-10-17',
        ticket_count=5,
        taken_numbers=None,
        strategy_weights=None,
        orthogonal_coverage=True
    ):
        """
        Executes the full 5-pillar hybrid GPU/ML prediction and combinatorial portfolio optimization.
        """
        if taken_numbers is None:
            taken_numbers = []
        taken_set = set(int(x) for x in taken_numbers if 0 <= int(x) <= 99)

        default_weights = {
            'ml_tree': 0.30,      # RF + HistGB + MLP
            'deep_gpu': 0.25,     # Deep ResNet / MC Dropout
            'markov': 0.15,       # 1st-Order Markov Transition from last draw
            'bayesian_hot': 0.15, # Dirichlet-Multinomial Momentum
            'renewal_cold': 0.15  # Survival Hazard / Atraso Mean Reversion
        }
        if strategy_weights:
            default_weights.update(strategy_weights)
        # Normalize weights
        w_sum = sum(default_weights.values()) or 1.0
        W = {k: v / w_sum for k, v in default_weights.items()}

        df = pd.read_csv(self.enriched_path)
        sub = df[df['lottery'] == lottery].sort_values('date').copy().reset_index(drop=True)

        feature_cols = [
            'temperature_2m', 'relative_humidity_2m', 'surface_pressure', 'wind_speed_10m',
            'lunar_phase', 'lunar_sin', 'lunar_cos', 'lunar_illumination',
            'lag_1', 'lag_2', 'lag_3', 'lag_7',
            'lag_1_decena', 'lag_1_unidad',
            'rolling_mean_3', 'rolling_mean_7', 'rolling_std_7',
            'atraso_decena', 'atraso_unidad',
            'max_atraso_decena_pool', 'max_atraso_unidad_pool'
        ]
        physics_cols = [
            'temperature_2m', 'relative_humidity_2m', 'surface_pressure', 'wind_speed_10m',
            'lunar_phase', 'lunar_sin', 'lunar_cos', 'lunar_illumination'
        ]

        X = sub[feature_cols].fillna(0).copy()
        y_dec = sub['decena'].astype(int).copy()
        y_uni = sub['unidad'].astype(int).copy()

        # Build target date feature vector
        from train_and_predict import build_tomorrow_features
        X_tom, weather_fc, lunar_fc, cur_atraso_dec, cur_atraso_uni = build_tomorrow_features(sub, lottery, target_date)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_tom_scaled = scaler.transform(X_tom[feature_cols].fillna(0))

        # --- PILLAR 1: DEEP LEARNING BOLILLERO (GPU / MC DROPOUT) ---
        p_deep_dec, p_deep_uni, unc_dec, unc_uni = train_deep_bolillero(
            X_scaled, y_dec, y_uni, X_tom_scaled, self.hw_info
        )
        matrix_deep = np.outer(p_deep_dec, p_deep_uni)

        # --- PILLAR 2: TREE & BOOSTING ENSEMBLE (RF + HistGB + MLP) ---
        rf_d = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42).fit(X, y_dec)
        rf_u = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42).fit(X, y_uni)
        hgb_d = HistGradientBoostingClassifier(max_iter=100, max_depth=3, random_state=42).fit(X, y_dec)
        hgb_u = HistGradientBoostingClassifier(max_iter=100, max_depth=3, random_state=42).fit(X, y_uni)
        mlp_d = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=350, random_state=42).fit(X_scaled, y_dec)
        mlp_u = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=350, random_state=42).fit(X_scaled, y_uni)

        def full_prob(clf, X_in):
            out = np.zeros(10)
            for c, p in zip(clf.classes_, clf.predict_proba(X_in)[0]):
                out[int(c)] = p
            return out

        p_tree_dec = (full_prob(rf_d, X_tom[feature_cols].fillna(0)) +
                      full_prob(hgb_d, X_tom[feature_cols].fillna(0)) +
                      full_prob(mlp_d, X_tom_scaled)) / 3.0
        p_tree_uni = (full_prob(rf_u, X_tom[feature_cols].fillna(0)) +
                      full_prob(hgb_u, X_tom[feature_cols].fillna(0)) +
                      full_prob(mlp_u, X_tom_scaled)) / 3.0
        matrix_tree = np.outer(p_tree_dec, p_tree_uni)

        # Physics-only model
        rf_phys_d = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42).fit(sub[physics_cols].fillna(0), y_dec)
        rf_phys_u = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42).fit(sub[physics_cols].fillna(0), y_uni)
        matrix_phys = np.outer(
            full_prob(rf_phys_d, X_tom[physics_cols].fillna(0)),
            full_prob(rf_phys_u, X_tom[physics_cols].fillna(0))
        )

        # --- PILLAR 3: 1ST-ORDER MARKOV TRANSITION MATRIX ---
        trans_dec = np.ones((10, 10)) * 0.5  # Laplace smoothing
        trans_uni = np.ones((10, 10)) * 0.5
        dec_seq = y_dec.values
        uni_seq = y_uni.values
        for i in range(len(dec_seq) - 1):
            trans_dec[dec_seq[i], dec_seq[i + 1]] += 1.0
            trans_uni[uni_seq[i], uni_seq[i + 1]] += 1.0
        trans_dec /= trans_dec.sum(axis=1, keepdims=True)
        trans_uni /= trans_uni.sum(axis=1, keepdims=True)

        last_dec = int(dec_seq[-1])
        last_uni = int(uni_seq[-1])
        p_markov_dec = trans_dec[last_dec]
        p_markov_uni = trans_uni[last_uni]
        matrix_markov = np.outer(p_markov_dec, p_markov_uni)

        # --- PILLAR 4: BAYESIAN DIRICHLET-MULTINOMIAL (HOT MOMENTUM) ---
        counts_dec = np.bincount(dec_seq, minlength=10).astype(float)
        counts_uni = np.bincount(uni_seq, minlength=10).astype(float)
        counts_2d = np.bincount(sub['number_2d'].astype(int).values, minlength=100).astype(float)
        alpha_prior = 2.0
        p_hot_dec = (counts_dec + alpha_prior) / (counts_dec.sum() + 10 * alpha_prior)
        p_hot_uni = (counts_uni + alpha_prior) / (counts_uni.sum() + 10 * alpha_prior)
        p_hot_2d = (counts_2d + 0.5) / (counts_2d.sum() + 50.0)
        matrix_hot = 0.65 * np.outer(p_hot_dec, p_hot_uni) + 0.35 * p_hot_2d.reshape(10, 10)
        matrix_hot /= matrix_hot.sum()

        # --- PILLAR 5: SURVIVAL HAZARD / ATRASO MEAN REVERSION ---
        delay_d_arr = np.array([cur_atraso_dec[d] for d in range(10)], dtype=float)
        delay_u_arr = np.array([cur_atraso_uni[u] for u in range(10)], dtype=float)
        # Softmax hazard weighting over normalized delays
        p_cold_dec = np.exp(delay_d_arr / 12.0)
        p_cold_dec /= p_cold_dec.sum()
        p_cold_uni = np.exp(delay_u_arr / 12.0)
        p_cold_uni /= p_cold_uni.sum()
        matrix_cold = np.outer(p_cold_dec, p_cold_uni)

        # --- HYBRID MASTER PROBABILITY MATRIX (10x10) ---
        master_matrix = (
            W['ml_tree'] * matrix_tree +
            W['deep_gpu'] * matrix_deep +
            W['markov'] * matrix_markov +
            W['bayesian_hot'] * matrix_hot +
            W['renewal_cold'] * matrix_cold
        )
        master_matrix /= master_matrix.sum()

        # --- GPU / VECTORIZED MONTE CARLO SIMULATION (1,000,000 ROLLOUTS) ---
        n_sims = 1_000_000
        flat_probs = master_matrix.flatten()
        flat_probs /= flat_probs.sum()
        if self.hw_info.get('torch_cuda', False):
            import torch
            device = torch.device('cuda:0')
            probs_t = torch.tensor(flat_probs, dtype=torch.float32, device=device)
            samples_t = torch.multinomial(probs_t, num_samples=n_sims, replacement=True)
            mc_counts = torch.bincount(samples_t, minlength=100).cpu().numpy()
        else:
            rng = np.random.default_rng(42)
            mc_counts = rng.multinomial(n_sims, flat_probs)

        # --- GAME-THEORY COMBINATORIAL PORTFOLIO OPTIMIZER ---
        crowd_edge = compute_crowd_bias_scores()
        available_nums = [n for n in range(100) if n not in taken_set]

        # Compute full 2D delays
        last_seen_2d = {}
        for idx_r, row_r in sub.iterrows():
            last_seen_2d[int(row_r['number_2d'])] = idx_r
        total_draws = len(sub)
        delays_2d = {n: (total_draws - 1 - last_seen_2d[n]) if n in last_seen_2d else total_draws for n in range(100)}

        # Submodular / Rook-Constrained Maximization for K-Ticket Optimal Raffle Play
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
                base_p = flat_probs[n]  # Master probability

                if orthogonal_coverage and len(covered_decenas) < 10 and len(covered_unidades) < 10:
                    # Strictly prefer uncovered Decena and Unidad first (Rook / Latin-Square coverage)
                    dec_free = 1.0 if d not in covered_decenas else 0.0
                    uni_free = 1.0 if u not in covered_unidades else 0.0
                    div_bonus = 0.25 * dec_free + 0.25 * uni_free
                else:
                    div_bonus = (0.002 if d not in covered_decenas else 0.0) + (0.002 if u not in covered_unidades else 0.0)

                # Parity balance bonus
                if parities:
                    even_ratio = sum(1 for x in parities if x % 2 == 0) / len(parities)
                    if (n % 2 == 0 and even_ratio < 0.5) or (n % 2 == 1 and even_ratio > 0.5):
                        div_bonus += 0.0015

                # Anti-crowd availability bonus
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

        # Build detailed portfolio cards
        portfolio_details = []
        for rank, num in enumerate(selected_portfolio, start=1):
            d, u = divmod(num, 10)
            # Identify dominant driver for this number
            comp_scores = {
                'ML Tree Ensemble': matrix_tree[d, u],
                'Deep ResNet GPU': matrix_deep[d, u],
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

        # Also compute the 5 Classic Archetype Picks (filtered by available_nums)
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
            {'profile': 'Ensamble GBDT + RF', 'number': f'{p1_n:02d}', 'prob_pct': round(p1_p, 3), 'desc': 'Máxima verosimilitud en árboles de decisión desacoplados'},
            {'profile': 'Deep ResNet + MC Dropout (GPU)', 'number': f'{p2_n:02d}', 'prob_pct': round(p2_p, 3), 'desc': 'Red neuronal residual profunda con muestreo Bayesiano'},
            {'profile': 'Física & Efemérides Lunares', 'number': f'{p3_n:02d}', 'prob_pct': round(p3_p, 3), 'desc': f"{weather_fc['temperature_2m']}°C, {weather_fc['surface_pressure']} hPa, {lunar_fc['lunar_phase_name']} ({lunar_fc['lunar_illumination']}%)"},
            {'profile': 'Inercia Bayesiana (Hot Streak)', 'number': f'{p4_n:02d}', 'prob_pct': round(p4_p, 3), 'desc': 'Conteo empírico Dirichlet-Multinomial desde Julio 2026'},
            {'profile': 'Reversión por Sequía (Atrasos)', 'number': f'{p5_n:02d}', 'prob_pct': round(p5_p, 3), 'desc': 'Tasa de riesgo (Hazard Rate) por demora acumulada'}
        ]

        # Full 00-99 grid data for UI heatmap & interactive selector
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

        # Statistical summary
        audit = statistical_audit.run_statistical_audit(df, lottery)

        # Log prediction in SQLite ledger
        try:
            self.db.log_prediction(target_date, lottery, selected_portfolio)
        except Exception:
            pass

        last_row = sub.iloc[-1]
        portfolio_cum_prob = sum(x['master_prob_pct'] for x in portfolio_details)
        dec_coverage_pct = len(covered_decenas) * 10.0
        uni_coverage_pct = len(covered_unidades) * 10.0
        any_digit_hit_prob = (1.0 - (1.0 - len(covered_decenas) / 10.0) * (1.0 - len(covered_unidades) / 10.0)) * 100.0

        return {
            'hardware': self.hw_info,
            'lottery': lottery,
            'target_date': target_date,
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
            }
        }


if __name__ == '__main__':
    engine = RaffleGPUPredictor()
    res = engine.run_prediction(lottery='ciudad', target_date='2026-10-17', ticket_count=5)
    print(f"\n[HARDWARE] {res['hardware']['backend']} | GPU: {res['hardware']['device_name']} ({res['hardware']['vram_mb']} MB)")
    print(f"[LAST DRAW] {res['last_draw']['date']} -> {res['last_draw']['number_4d']} (Ambo: {res['last_draw']['number_2d']})")
    print(f"[PORTFOLIO METRICS] {res['portfolio_metrics']}")
    print("\n=== TOP 5 OPTIMAL RAFFLE PLAYS (LOTBA CIUDAD - 17/10/2026) ===")
    for p in res['optimal_portfolio']:
        print(f" #{p['rank']} -> AMBO [{p['number']}] | Master Prob: {p['master_prob_pct']:.3f}% | MC Hits: {p['mc_sim_hits']:,}/1M | Driver: {p['dominant_driver']} | Delay: {p['delay_2d']}d")
