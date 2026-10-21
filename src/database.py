import os
import sys
import re
import sqlite3
import datetime
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enrich_features
import scrape_draws


class QuinielaDatabase:
    """
    Persistent SQLite Database Engine for La Marisca Project.
    Stores historical & live draws, weather/lunar telemetry, and prediction performance logs.
    Automatically synchronizes with CSV exports for full compatibility.
    """
    def __init__(self, data_dir=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = data_dir or os.path.join(base_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        self.db_path = os.path.join(self.data_dir, 'quiniela_marisca.db')
        self.raw_csv_path = os.path.join(self.data_dir, 'raw_draws.csv')
        self.enriched_csv_path = os.path.join(self.data_dir, 'dataset_enriched.csv')
        self._init_db()
        self._bootstrap_from_csv_if_needed()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS draws (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    lottery TEXT NOT NULL,
                    number_4d TEXT NOT NULL,
                    number_2d INTEGER NOT NULL,
                    decena INTEGER NOT NULL,
                    unidad INTEGER NOT NULL,
                    temperature_2m REAL,
                    relative_humidity_2m REAL,
                    surface_pressure REAL,
                    wind_speed_10m REAL,
                    air_density REAL,
                    dew_point REAL,
                    static_risk REAL,
                    lunar_phase REAL,
                    lunar_sin REAL,
                    lunar_cos REAL,
                    lunar_illumination REAL,
                    lunar_phase_name TEXT,
                    source TEXT DEFAULT 'scraper',
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, lottery)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_date TEXT NOT NULL,
                    lottery TEXT NOT NULL,
                    top_picks_json TEXT NOT NULL,
                    actual_number_4d TEXT,
                    actual_number_2d INTEGER,
                    hit_exact INTEGER DEFAULT 0,
                    hit_decena INTEGER DEFAULT 0,
                    hit_unidad INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(target_date, lottery)
                )
            """)
            # Auto-migrate columns if table already existed without them
            existing_cols = [r[1] for r in conn.execute("PRAGMA table_info(draws)").fetchall()]
            for col, col_type in [('air_density', 'REAL'), ('dew_point', 'REAL'), ('static_risk', 'REAL')]:
                if col not in existing_cols:
                    conn.execute(f"ALTER TABLE draws ADD COLUMN {col} {col_type}")
            conn.commit()

    def _bootstrap_from_csv_if_needed(self):
        """Imports existing CSV records into SQLite on first run."""
        with self._get_conn() as conn:
            cnt = conn.execute("SELECT COUNT(*) FROM draws").fetchone()[0]
            if cnt == 0 and os.path.exists(self.enriched_csv_path):
                df = pd.read_csv(self.enriched_csv_path)
                for _, r in df.iterrows():
                    conn.execute("""
                        INSERT OR IGNORE INTO draws (
                            date, lottery, number_4d, number_2d, decena, unidad,
                            temperature_2m, relative_humidity_2m, surface_pressure, wind_speed_10m,
                            lunar_phase, lunar_sin, lunar_cos, lunar_illumination, lunar_phase_name,
                            source
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        str(r['date']),
                        str(r['lottery']),
                        f"{int(r['number_4d']):04d}",
                        int(r['number_2d']),
                        int(r['decena']),
                        int(r['unidad']),
                        float(r.get('temperature_2m', 18.0)),
                        float(r.get('relative_humidity_2m', 60.0)),
                        float(r.get('surface_pressure', 1013.0)),
                        float(r.get('wind_speed_10m', 10.0)),
                        float(r.get('lunar_phase', 0.0)),
                        float(r.get('lunar_sin', 0.0)),
                        float(r.get('lunar_cos', 1.0)),
                        float(r.get('lunar_illumination', 50.0)),
                        str(r.get('lunar_phase_name', 'N/A')),
                        'initial_seed'
                    ))
                conn.commit()

    def upsert_draw(self, date_str, lottery, number_4d, source='manual'):
        """
        Inserts or updates a draw result in SQLite, automatically fetching weather & lunar telemetry
        and regenerating the enriched dataset.
        """
        num_str = f"{int(str(number_4d).strip()):04d}"
        n2 = int(num_str[-2:])
        dec = int(num_str[-2])
        uni = int(num_str[-1])

        lunar = enrich_features.compute_lunar_features(date_str, hour_art=21)
        try:
            w_hist = enrich_features.fetch_weather_historical(lottery, start_date=date_str, end_date=date_str)
            w = w_hist.get(date_str)
            if not w:
                w = enrich_features.fetch_weather_forecast(lottery, target_date=date_str)
        except Exception:
            w = {'temperature_2m': 18.0, 'relative_humidity_2m': 60.0, 'surface_pressure': 1014.0, 'wind_speed_10m': 12.0}

        t_val = float(w.get('temperature_2m', 18.0))
        h_val = float(w.get('relative_humidity_2m', 60.0))
        p_val = float(w.get('surface_pressure', 1014.0))
        w_val = float(w.get('wind_speed_10m', 12.0))
        air_rho = enrich_features.compute_air_density(t_val, p_val, h_val)
        dew_pt = enrich_features.compute_dew_point(t_val, h_val)
        st_risk = enrich_features.compute_static_risk(h_val)

        now_iso = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO draws (
                    date, lottery, number_4d, number_2d, decena, unidad,
                    temperature_2m, relative_humidity_2m, surface_pressure, wind_speed_10m,
                    air_density, dew_point, static_risk,
                    lunar_phase, lunar_sin, lunar_cos, lunar_illumination, lunar_phase_name,
                    source, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date, lottery) DO UPDATE SET
                    number_4d=excluded.number_4d,
                    number_2d=excluded.number_2d,
                    decena=excluded.decena,
                    unidad=excluded.unidad,
                    temperature_2m=excluded.temperature_2m,
                    relative_humidity_2m=excluded.relative_humidity_2m,
                    surface_pressure=excluded.surface_pressure,
                    wind_speed_10m=excluded.wind_speed_10m,
                    air_density=excluded.air_density,
                    dew_point=excluded.dew_point,
                    static_risk=excluded.static_risk,
                    source=excluded.source,
                    updated_at=excluded.updated_at
            """, (
                date_str, lottery, num_str, n2, dec, uni,
                t_val, h_val, p_val, w_val,
                air_rho, dew_pt, st_risk,
                lunar['lunar_phase'], lunar['lunar_sin'], lunar['lunar_cos'],
                lunar['lunar_illumination'], lunar['lunar_phase_name'],
                source, now_iso
            ))
            # Check if there was a logged prediction for this date and lottery to grade it
            pred_row = conn.execute(
                "SELECT id, top_picks_json FROM predictions_log WHERE target_date=? AND lottery=?",
                (date_str, lottery)
            ).fetchone()
            if pred_row:
                picks = json.loads(pred_row['top_picks_json'])
                pick_ints = [int(p) for p in picks]
                hit_exact = 1 if n2 in pick_ints else 0
                hit_dec = 1 if any(divmod(p, 10)[0] == dec for p in pick_ints) else 0
                hit_uni = 1 if any(divmod(p, 10)[1] == uni for p in pick_ints) else 0
                conn.execute("""
                    UPDATE predictions_log
                    SET actual_number_4d=?, actual_number_2d=?, hit_exact=?, hit_decena=?, hit_unidad=?
                    WHERE id=?
                """, (num_str, n2, hit_exact, hit_dec, hit_uni, pred_row['id']))
            conn.commit()

        self.rebuild_enriched_dataset()
        return {'date': date_str, 'lottery': lottery, 'number_4d': num_str, 'number_2d': f'{n2:02d}'}

    def log_prediction(self, target_date, lottery, top_picks_list):
        """Saves a prediction portfolio to SQLite and grades it if the draw already exists."""
        picks_json = json.dumps([f"{int(x):02d}" for x in top_picks_list])
        with self._get_conn() as conn:
            draw_row = conn.execute(
                "SELECT number_4d, number_2d, decena, unidad FROM draws WHERE date=? AND lottery=?",
                (target_date, lottery)
            ).fetchone()
            act_4d, act_2d, hit_ex, hit_d, hit_u = None, None, 0, 0, 0
            if draw_row:
                act_4d = draw_row['number_4d']
                act_2d = int(draw_row['number_2d'])
                dec = int(draw_row['decena'])
                uni = int(draw_row['unidad'])
                pick_ints = [int(p) for p in top_picks_list]
                hit_ex = 1 if act_2d in pick_ints else 0
                hit_d = 1 if any(divmod(p, 10)[0] == dec for p in pick_ints) else 0
                hit_u = 1 if any(divmod(p, 10)[1] == uni for p in pick_ints) else 0

            conn.execute("""
                INSERT INTO predictions_log (
                    target_date, lottery, top_picks_json, actual_number_4d, actual_number_2d,
                    hit_exact, hit_decena, hit_unidad
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(target_date, lottery) DO UPDATE SET
                    top_picks_json=excluded.top_picks_json,
                    actual_number_4d=excluded.actual_number_4d,
                    actual_number_2d=excluded.actual_number_2d,
                    hit_exact=excluded.hit_exact,
                    hit_decena=excluded.hit_decena,
                    hit_unidad=excluded.hit_unidad,
                    created_at=CURRENT_TIMESTAMP
            """, (target_date, lottery, picks_json, act_4d, act_2d, hit_ex, hit_d, hit_u))
            conn.commit()

    def rebuild_enriched_dataset(self):
        """Recomputes all autoregressive lags, rolling stats, delay Z-scores, and physical parameters."""
        with self._get_conn() as conn:
            df_db = pd.read_sql_query("SELECT * FROM draws ORDER BY lottery, date ASC", conn)

        if df_db.empty:
            return pd.DataFrame()

        # Export raw_draws.csv
        raw_cols = ['date', 'lottery', 'number_4d', 'number_2d', 'decena', 'unidad']
        df_raw = df_db[raw_cols].copy()
        df_raw['number_4d'] = df_raw['number_4d'].apply(lambda x: f"{int(x):04d}")
        df_raw.to_csv(self.raw_csv_path, index=False)

        enriched_parts = []
        for lot in ['ciudad']:
            sub = df_db[df_db['lottery'] == lot].copy().reset_index(drop=True)
            if sub.empty:
                continue

            auto_df = enrich_features.compute_autoregressive_features(sub[raw_cols])

            # Weather & Physical telemetry
            temps, hums, press, winds = [], [], [], []
            densities, dew_points, static_risks = [], [], []
            l_phase, l_sin, l_cos, l_illum, l_names = [], [], [], [], []

            for _, row in sub.iterrows():
                t = float(row['temperature_2m']) if pd.notnull(row.get('temperature_2m')) else 18.0
                h = float(row['relative_humidity_2m']) if pd.notnull(row.get('relative_humidity_2m')) else 65.0
                p = float(row['surface_pressure']) if pd.notnull(row.get('surface_pressure')) else 1015.0
                w = float(row['wind_speed_10m']) if pd.notnull(row.get('wind_speed_10m')) else 12.0

                rho = float(row['air_density']) if pd.notnull(row.get('air_density')) else enrich_features.compute_air_density(t, p, h)
                dew = float(row['dew_point']) if pd.notnull(row.get('dew_point')) else enrich_features.compute_dew_point(t, h)
                st = float(row['static_risk']) if pd.notnull(row.get('static_risk')) else enrich_features.compute_static_risk(h)

                temps.append(t)
                hums.append(h)
                press.append(p)
                winds.append(w)
                densities.append(rho)
                dew_points.append(dew)
                static_risks.append(st)

                l_phase.append(float(row.get('lunar_phase', 0.0)))
                l_sin.append(float(row.get('lunar_sin', 0.0)))
                l_cos.append(float(row.get('lunar_cos', 1.0)))
                l_illum.append(float(row.get('lunar_illumination', 50.0)))
                l_names.append(str(row.get('lunar_phase_name', 'N/A')))

            auto_df['temperature_2m'] = temps
            auto_df['relative_humidity_2m'] = hums
            auto_df['surface_pressure'] = press
            auto_df['wind_speed_10m'] = winds
            auto_df['air_density'] = densities
            auto_df['dew_point'] = dew_points
            auto_df['static_risk'] = static_risks
            auto_df['lunar_phase'] = l_phase
            auto_df['lunar_sin'] = l_sin
            auto_df['lunar_cos'] = l_cos
            auto_df['lunar_illumination'] = l_illum
            auto_df['lunar_phase_name'] = l_names

            # Forward-fill and backward-fill any rare missing values
            for col_fill in ['temperature_2m', 'relative_humidity_2m', 'surface_pressure', 'wind_speed_10m', 'air_density', 'dew_point', 'static_risk']:
                auto_df[col_fill] = auto_df[col_fill].bfill().ffill()

            enriched_parts.append(auto_df)

        df_all = pd.concat(enriched_parts, ignore_index=True)
        df_all.to_csv(self.enriched_csv_path, index=False)
        return df_all

    def sync_from_web(self, days_back=14):
        """
        Scrapes recent dates from web portals (loteriasmundiales & ruta1000) for Quiniela de la Ciudad
        and inserts any missing draws into SQLite.
        """
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        now_art = now_utc - datetime.timedelta(hours=3)
        # Sorteo Nocturno occurs at 21:00 ART. Results publish around 21:15-21:30 ART.
        # If current ART time is before 21:30, today's Nocturno has not been drawn yet!
        if now_art.hour < 21 or (now_art.hour == 21 and now_art.minute < 30):
            max_d = now_art.date() - datetime.timedelta(days=1)
        else:
            max_d = now_art.date()

        start_d = max_d - datetime.timedelta(days=days_back)

        with self._get_conn() as conn:
            existing = set(
                r['date']
                for r in conn.execute("SELECT date FROM draws WHERE lottery = 'ciudad' AND date >= ?", (start_d.strftime('%Y-%m-%d'),)).fetchall()
            )

        added = 0
        curr = start_d
        while curr <= max_d:
            if curr.weekday() != 6:
                ds = curr.strftime('%Y-%m-%d')
                if ds not in existing:
                    try:
                        c_num = scrape_draws.scrape_ciudad(curr)
                        if c_num and re.match(r'^\d{4}$', str(c_num)):
                            self.upsert_draw(ds, 'ciudad', c_num, source='auto_scraper')
                            added += 1
                    except Exception as e:
                        print(f"[WARN] Scraping failed for {curr}: {e}")
            curr += datetime.timedelta(days=1)

        return added

    def get_recent_draws(self, lottery='ciudad', limit=15):
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT date, lottery, number_4d, number_2d, decena, unidad,
                       temperature_2m, surface_pressure, lunar_phase_name, lunar_illumination, source
                FROM draws
                WHERE lottery = ?
                ORDER BY date DESC
                LIMIT ?
            """, (lottery, limit)).fetchall()
            return [dict(r) for r in rows]

    def get_db_stats(self):
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM draws").fetchone()[0]
            by_lot = dict(conn.execute("SELECT lottery, COUNT(*) FROM draws GROUP BY lottery").fetchall())
            min_max = conn.execute("SELECT MIN(date), MAX(date) FROM draws").fetchone()
            preds_count = conn.execute("SELECT COUNT(*) FROM predictions_log").fetchone()[0]
            return {
                'db_file': os.path.basename(self.db_path),
                'total_draws': total,
                'ciudad_draws': by_lot.get('ciudad', 0),
                'formosa_draws': by_lot.get('formosa', 0),
                'first_date': min_max[0],
                'last_date': min_max[1],
                'predictions_logged': preds_count
            }


if __name__ == '__main__':
    db = QuinielaDatabase()
    print("Database initialized:", db.get_db_stats())
    print("Recent LOTBA Ciudad draws:", db.get_recent_draws('ciudad', limit=5))
