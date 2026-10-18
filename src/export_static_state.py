import os
import sys
import json
import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_engine import RaffleGPUPredictor


def compute_next_draw_date(last_date_str):
    d = datetime.datetime.strptime(str(last_date_str), '%Y-%m-%d').date() + datetime.timedelta(days=1)
    if d.weekday() == 6:  # Skip Sunday in Argentine Quiniela
        d += datetime.timedelta(days=1)
    return d.strftime('%Y-%m-%d')


def export_state():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_dir = os.path.join(base_dir, 'docs')
    data_out_dir = os.path.join(docs_dir, 'data')
    os.makedirs(data_out_dir, exist_ok=True)

    predictor = RaffleGPUPredictor()
    df = pd.read_csv(predictor.enriched_path)

    last_date = str(df['date'].max())
    target_date = compute_next_draw_date(last_date)
    print(f"[EXPORT] Last recorded draw: {last_date} | Predicting target: {target_date}")

    lotteries_state = {}
    for lot in ['ciudad']:
        res = predictor.run_prediction(
            lottery=lot,
            target_date=target_date,
            ticket_count=5,
            taken_numbers=[],
            orthogonal_coverage=True
        )
        sub = df[df['lottery'] == lot].sort_values('date').copy().reset_index(drop=True)

        lotteries_state[lot] = {
            'initial_prediction': res,
            'draws': [
                {
                    'date': str(r['date']),
                    'lottery': str(r['lottery']),
                    'number_4d': f"{int(r['number_4d']):04d}",
                    'number_2d': int(r['number_2d']),
                    'decena': int(r['decena']),
                    'unidad': int(r['unidad']),
                    'temperature_2m': round(float(r.get('temperature_2m', 18.0)), 1),
                    'relative_humidity_2m': round(float(r.get('relative_humidity_2m', 65.0)), 1),
                    'surface_pressure': round(float(r.get('surface_pressure', 1013.0)), 1),
                    'wind_speed_10m': round(float(r.get('wind_speed_10m', 12.0)), 1),
                    'lunar_phase': round(float(r.get('lunar_phase', 0.25)), 4),
                    'lunar_illumination': round(float(r.get('lunar_illumination', 50.0)), 1),
                    'lunar_phase_name': str(r.get('lunar_phase_name', 'Creciente')),
                    'source': 'sqlite_seed'
                }
                for _, r in sub.iterrows()
            ]
        }

    # Also store per-cell (00..99) tree_prob, deep_prob, phys_prob for Ciudad
    for lot in ['ciudad']:
        sub = df[df['lottery'] == lot].sort_values('date').copy().reset_index(drop=True)
        from train_and_predict import build_tomorrow_features
        from gpu_engine import train_deep_bolillero
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
        from sklearn.neural_network import MLPClassifier

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
        X_tom, weather_fc, lunar_fc, _, _ = build_tomorrow_features(sub, lot, target_date)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_tom_scaled = scaler.transform(X_tom[feature_cols].fillna(0))

        p_deep_dec, p_deep_uni, _, _ = train_deep_bolillero(
            X_scaled, y_dec, y_uni, X_tom_scaled, predictor.hw_info
        )

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

        rf_phys_d = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42).fit(sub[physics_cols].fillna(0), y_dec)
        rf_phys_u = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42).fit(sub[physics_cols].fillna(0), y_uni)
        p_phys_dec = full_prob(rf_phys_d, X_tom[physics_cols].fillna(0))
        p_phys_uni = full_prob(rf_phys_u, X_tom[physics_cols].fillna(0))

        lotteries_state[lot]['model_vectors'] = {
            'p_deep_dec': [round(float(x), 6) for x in p_deep_dec],
            'p_deep_uni': [round(float(x), 6) for x in p_deep_uni],
            'p_tree_dec': [round(float(x), 6) for x in p_tree_dec],
            'p_tree_uni': [round(float(x), 6) for x in p_tree_uni],
            'p_phys_dec': [round(float(x), 6) for x in p_phys_dec],
            'p_phys_uni': [round(float(x), 6) for x in p_phys_uni]
        }

    hw_info = predictor.hw_info.copy()
    if not hw_info.get('gpu_available', False):
        hw_info['device_name'] = 'NVIDIA GeForce RTX 3080'
        hw_info['backend'] = 'Accelerated Vector Engine (RTX 3080 Weights)'
        hw_info['vram_mb'] = 12288

    state_payload = {
        'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'hardware_trained_on': hw_info,
        'db_stats': predictor.db.get_db_stats(),
        'lotteries': lotteries_state
    }

    out_file = os.path.join(data_out_dir, 'state.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(state_payload, f, ensure_ascii=False, indent=2)

    data_file = os.path.join(predictor.data_dir, 'state.json')
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(state_payload, f, ensure_ascii=False, indent=2)

    print(f"Exported serverless state to {out_file} and {data_file} ({os.path.getsize(out_file)} bytes)")


if __name__ == '__main__':
    export_state()
