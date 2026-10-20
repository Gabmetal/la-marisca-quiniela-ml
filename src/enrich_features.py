import os
import sys
import datetime
import math
import time
import requests
import numpy as np
import pandas as pd

COORDINATES = {
    'ciudad': {'lat': -34.6037, 'lon': -58.3816, 'tz': 'America/Argentina/Buenos_Aires'},
    'formosa': {'lat': -26.1849, 'lon': -58.1731, 'tz': 'America/Argentina/Buenos_Aires'}
}

SYNODIC_MONTH = 29.53058867
LUNAR_REF_UTC = datetime.datetime(2000, 1, 6, 18, 14, 0, tzinfo=datetime.timezone.utc)


def compute_air_density(temp_c, pressure_hpa, rh_pct):
    """
    Computes moist air density (kg/m3) using ideal gas law for pneumatic bolillero aerodynamics.
    Modulates aerodynamic drag and buoyancy on physical balls.
    """
    try:
        T_k = float(temp_c) + 273.15
        p_sat = 6.1078 * (10.0 ** ((7.5 * float(temp_c)) / (float(temp_c) + 237.3)))
        p_v = (float(rh_pct) / 100.0) * p_sat * 100.0
        p_d = (float(pressure_hpa) * 100.0) - p_v
        R_d = 287.058
        R_v = 461.495
        rho = (p_d / (R_d * T_k)) + (p_v / (R_v * T_k))
        return round(float(rho), 4)
    except Exception:
        return 1.2250  # Standard sea-level density fallback


def compute_dew_point(temp_c, rh_pct):
    """Computes dew point (deg C) using the Magnus formula."""
    try:
        a = 17.27
        b = 237.7
        alpha = ((a * float(temp_c)) / (b + float(temp_c))) + math.log(max(float(rh_pct), 1.0) / 100.0)
        return round(float((b * alpha) / (a - alpha)), 2)
    except Exception:
        return round(float(temp_c) - ((100.0 - float(rh_pct)) / 5.0), 2)


def compute_static_risk(rh_pct):
    """
    Triboelectric static electricity accumulation risk on cellulose/acrylic lottery balls.
    High when Relative Humidity < 45%.
    """
    try:
        return round(max(0.0, (45.0 - float(rh_pct)) / 45.0), 3)
    except Exception:
        return 0.0


def compute_lunar_features(date_str, hour_art=21):
    """
    Computes exact astronomical lunar features for a given date at hour_art ART (UTC-3).
    """
    dt_local = datetime.datetime.strptime(f"{date_str} {hour_art:02d}:00:00", "%Y-%m-%d %H:%M:%S")
    dt_utc = dt_local.replace(tzinfo=datetime.timezone(datetime.timedelta(hours=-3))).astimezone(datetime.timezone.utc)
    
    diff_days = (dt_utc - LUNAR_REF_UTC).total_seconds() / 86400.0
    phase = (diff_days % SYNODIC_MONTH) / SYNODIC_MONTH
    theta = 2.0 * math.pi * phase
    
    lunar_sin = math.sin(theta)
    lunar_cos = math.cos(theta)
    illumination = ((1.0 - math.cos(theta)) / 2.0) * 100.0
    
    if phase < 0.03 or phase >= 0.97:
        phase_name = 'Luna Nueva'
    elif phase < 0.22:
        phase_name = 'Creciente Iluminante'
    elif phase < 0.28:
        phase_name = 'Cuarto Creciente'
    elif phase < 0.47:
        phase_name = 'Gibosa Creciente'
    elif phase < 0.53:
        phase_name = 'Luna Llena'
    elif phase < 0.72:
        phase_name = 'Gibosa Menguante'
    elif phase < 0.78:
        phase_name = 'Cuarto Menguante'
    else:
        phase_name = 'Menguante'
        
    return {
        'lunar_phase': round(phase, 4),
        'lunar_sin': round(lunar_sin, 4),
        'lunar_cos': round(lunar_cos, 4),
        'lunar_illumination': round(illumination, 2),
        'lunar_phase_name': phase_name
    }


def fetch_weather_historical(lottery):
    coords = COORDINATES.get(lottery, COORDINATES['ciudad'])
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={coords['lat']}&longitude={coords['lon']}&"
        f"start_date=2025-10-15&end_date=2026-10-20&"
        f"hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m&"
        f"timezone={coords['tz'].replace('/', '%2F')}"
    )
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                data = r.json()
                hourly = data.get('hourly', {})
                times = hourly.get('time', [])
                temps = hourly.get('temperature_2m', [])
                humids = hourly.get('relative_humidity_2m', [])
                pressures = hourly.get('surface_pressure', [])
                winds = hourly.get('wind_speed_10m', [])
                
                weather_by_date = {}
                for t, temp, hum, pres, wnd in zip(times, temps, humids, pressures, winds):
                    if t.endswith('T21:00'):
                        d = t.split('T')[0]
                        weather_by_date[d] = {
                            'temperature_2m': temp,
                            'relative_humidity_2m': hum,
                            'surface_pressure': pres,
                            'wind_speed_10m': wnd
                        }
                return weather_by_date
        except Exception:
            time.sleep(1)
    return {}


def fetch_weather_forecast(lottery, target_date='2026-10-20'):
    coords = COORDINATES.get(lottery, COORDINATES['ciudad'])
    fallback = {
        'temperature_2m': 18.6,
        'relative_humidity_2m': 66.0,
        'surface_pressure': 1015.2,
        'wind_speed_10m': 14.5,
        'air_density': 1.2250,
        'dew_point': 12.1,
        'static_risk': 0.0
    }

    urls = [
        (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={coords['lat']}&longitude={coords['lon']}&"
            f"hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m&"
            f"forecast_days=16&"
            f"timezone={coords['tz'].replace('/', '%2F')}"
        ),
        (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={coords['lat']}&longitude={coords['lon']}&"
            f"start_date={target_date}&end_date={target_date}&"
            f"hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m&"
            f"timezone={coords['tz'].replace('/', '%2F')}"
        )
    ]

    for url in urls:
        for attempt in range(2):
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    hourly = data.get('hourly', {})
                    times = hourly.get('time', [])
                    temps = hourly.get('temperature_2m', [])
                    humids = hourly.get('relative_humidity_2m', [])
                    pressures = hourly.get('surface_pressure', [])
                    winds = hourly.get('wind_speed_10m', [])
                    
                    target_stamp = f"{target_date}T21:00"
                    for t, temp, hum, pres, wnd in zip(times, temps, humids, pressures, winds):
                        if t == target_stamp and temp is not None:
                            t_val = float(temp)
                            h_val = float(hum)
                            p_val = float(pres)
                            w_val = float(wnd)
                            return {
                                'temperature_2m': t_val,
                                'relative_humidity_2m': h_val,
                                'surface_pressure': p_val,
                                'wind_speed_10m': w_val,
                                'air_density': compute_air_density(t_val, p_val, h_val),
                                'dew_point': compute_dew_point(t_val, h_val),
                                'static_risk': compute_static_risk(h_val)
                            }
                    if temps and temps[-1] is not None:
                        t_val = float(temps[-1])
                        h_val = float(humids[-1])
                        p_val = float(pressures[-1])
                        w_val = float(winds[-1])
                        return {
                            'temperature_2m': t_val,
                            'relative_humidity_2m': h_val,
                            'surface_pressure': p_val,
                            'wind_speed_10m': w_val,
                            'air_density': compute_air_density(t_val, p_val, h_val),
                            'dew_point': compute_dew_point(t_val, h_val),
                            'static_risk': compute_static_risk(h_val)
                        }
            except Exception:
                time.sleep(0.5)

    return fallback


def compute_autoregressive_features(df_lottery):
    """
    Computes lag features, rolling statistics, delays (atrasos), parity, calendar cycles,
    and mechanical sequential properties for chronologically sorted draws.
    """
    df = df_lottery.sort_values('date').copy().reset_index(drop=True)
    dt_series = pd.to_datetime(df['date'])
    
    # Calendar cycles & maintenance inspections
    df['day_of_week'] = dt_series.dt.dayofweek
    df['day_of_month'] = dt_series.dt.day
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    days_diff = (dt_series - dt_series.shift(1)).dt.days
    df['days_since_last_draw'] = days_diff.fillna(1).astype(int)

    # Intrinsic digit properties
    df['parity'] = df['number_2d'] % 2
    df['decena_parity'] = df['decena'] % 2
    df['unidad_parity'] = df['unidad'] % 2
    df['sum_of_digits'] = df['decena'] + df['unidad']
    df['digit_diff'] = np.abs(df['decena'] - df['unidad'])
    df['is_repdigit'] = (df['decena'] == df['unidad']).astype(int)
    
    # Lags of number_2d
    df['lag_1'] = df['number_2d'].shift(1)
    df['lag_2'] = df['number_2d'].shift(2)
    df['lag_3'] = df['number_2d'].shift(3)
    df['lag_7'] = df['number_2d'].shift(7)
    
    # Lags of decena and unidad
    df['lag_1_decena'] = df['decena'].shift(1)
    df['lag_1_unidad'] = df['unidad'].shift(1)
    df['lag_2_decena'] = df['decena'].shift(2)
    df['lag_2_unidad'] = df['unidad'].shift(2)
    
    # Consecutive repetitions
    df['repeat_decena'] = (df['decena'] == df['decena'].shift(1)).astype(int)
    df['repeat_unidad'] = (df['unidad'] == df['unidad'].shift(1)).astype(int)

    # Rolling statistics (using past values only)
    df['rolling_mean_3'] = df['number_2d'].shift(1).rolling(window=3, min_periods=1).mean()
    df['rolling_mean_7'] = df['number_2d'].shift(1).rolling(window=7, min_periods=1).mean()
    df['rolling_std_7'] = df['number_2d'].shift(1).rolling(window=7, min_periods=2).std().fillna(0)
    
    # Dynamic delay tracking (atrasos)
    atraso_dec_history = {d: 0 for d in range(10)}
    atraso_uni_history = {u: 0 for u in range(10)}
    
    row_atraso_dec = []
    row_atraso_uni = []
    row_max_atraso_dec = []
    row_max_atraso_uni = []
    
    for idx, row in df.iterrows():
        d_val = row['decena']
        u_val = row['unidad']
        
        row_atraso_dec.append(atraso_dec_history[d_val])
        row_atraso_uni.append(atraso_uni_history[u_val])
        row_max_atraso_dec.append(max(atraso_dec_history.values()))
        row_max_atraso_uni.append(max(atraso_uni_history.values()))
        
        for d in range(10):
            if d == d_val:
                atraso_dec_history[d] = 0
            else:
                atraso_dec_history[d] += 1
                
        for u in range(10):
            if u == u_val:
                atraso_uni_history[u] = 0
            else:
                atraso_uni_history[u] += 1
                
    df['atraso_decena'] = row_atraso_dec
    df['atraso_unidad'] = row_atraso_uni
    df['max_atraso_decena_pool'] = row_max_atraso_dec
    df['max_atraso_unidad_pool'] = row_max_atraso_uni
    
    # Delay Z-Scores relative to Poisson / Geometric expected mean (mu=10, sigma=9.487)
    df['delay_zscore_decena'] = (df['atraso_decena'] - 10.0) / 9.487
    df['delay_zscore_unidad'] = (df['atraso_unidad'] - 10.0) / 9.487

    df.attrs['final_atraso_decena'] = atraso_dec_history
    df.attrs['final_atraso_unidad'] = atraso_uni_history
    
    return df


def enrich_dataset(raw_csv_path, output_csv_path):
    print(f"Loading raw draws from {raw_csv_path}...")
    df_raw = pd.read_csv(raw_csv_path)
    df_raw['date'] = df_raw['date'].astype(str)
    
    print("Fetching historical weather telemetry from Open-Meteo...")
    weather_data = {
        'ciudad': fetch_weather_historical('ciudad'),
        'formosa': fetch_weather_historical('formosa')
    }
    
    enriched_dfs = []
    for lot in ['ciudad', 'formosa']:
        sub = df_raw[df_raw['lottery'] == lot].copy()
        if len(sub) == 0:
            continue
        print(f"Processing mechanical and lag features for {lot} ({len(sub)} rows)...")
        sub_auto = compute_autoregressive_features(sub)
        
        temps, hums, press, winds = [], [], [], []
        densities, dew_points, static_risks = [], [], []
        l_phase, l_sin, l_cos, l_illum, l_names = [], [], [], [], []
        
        for idx, row in sub_auto.iterrows():
            d_str = row['date']
            w = weather_data[lot].get(d_str, {})
            t_val = w.get('temperature_2m', 18.0)
            h_val = w.get('relative_humidity_2m', 65.0)
            p_val = w.get('surface_pressure', 1015.0)
            w_val = w.get('wind_speed_10m', 12.0)
            
            temps.append(t_val)
            hums.append(h_val)
            press.append(p_val)
            winds.append(w_val)
            
            densities.append(compute_air_density(t_val, p_val, h_val))
            dew_points.append(compute_dew_point(t_val, h_val))
            static_risks.append(compute_static_risk(h_val))
            
            lunar = compute_lunar_features(d_str)
            l_phase.append(lunar['lunar_phase'])
            l_sin.append(lunar['lunar_sin'])
            l_cos.append(lunar['lunar_cos'])
            l_illum.append(lunar['lunar_illumination'])
            l_names.append(lunar['lunar_phase_name'])
            
        sub_auto['temperature_2m'] = temps
        sub_auto['relative_humidity_2m'] = hums
        sub_auto['surface_pressure'] = press
        sub_auto['wind_speed_10m'] = winds
        sub_auto['air_density'] = densities
        sub_auto['dew_point'] = dew_points
        sub_auto['static_risk'] = static_risks
        
        sub_auto['lunar_phase'] = l_phase
        sub_auto['lunar_sin'] = l_sin
        sub_auto['lunar_cos'] = l_cos
        sub_auto['lunar_illumination'] = l_illum
        sub_auto['lunar_phase_name'] = l_names
        
        # Forward-fill any minor missing values
        sub_auto['temperature_2m'] = sub_auto['temperature_2m'].bfill().ffill()
        sub_auto['relative_humidity_2m'] = sub_auto['relative_humidity_2m'].bfill().ffill()
        sub_auto['surface_pressure'] = sub_auto['surface_pressure'].bfill().ffill()
        sub_auto['wind_speed_10m'] = sub_auto['wind_speed_10m'].bfill().ffill()
        sub_auto['air_density'] = sub_auto['air_density'].bfill().ffill()
        sub_auto['dew_point'] = sub_auto['dew_point'].bfill().ffill()
        sub_auto['static_risk'] = sub_auto['static_risk'].bfill().ffill()
        
        enriched_dfs.append(sub_auto)
        
    df_all = pd.concat(enriched_dfs, ignore_index=True)
    df_all.to_csv(output_csv_path, index=False)
    print(f"Successfully generated enriched dataset at {output_csv_path} with {len(df_all)} records and {df_all.shape[1]} columns.")
    return df_all


if __name__ == '__main__':
    raw_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_draws.csv')
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset_enriched.csv')
    if os.path.exists(raw_path):
        enrich_dataset(raw_path, out_path)
    else:
        print(f"Raw draws file not found at {raw_path}.")
