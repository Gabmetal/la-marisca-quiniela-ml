import os
import sys
import datetime
import math
import time
import requests
import numpy as np
import pandas as pd

# Open-Meteo coordinates
COORDINATES = {
    'ciudad': {'lat': -34.6037, 'lon': -58.3816, 'tz': 'America/Argentina/Buenos_Aires'},
    'formosa': {'lat': -26.1849, 'lon': -58.1731, 'tz': 'America/Argentina/Buenos_Aires'}
}

SYNODIC_MONTH = 29.53058867
LUNAR_REF_UTC = datetime.datetime(2000, 1, 6, 18, 14, 0, tzinfo=datetime.timezone.utc)

def compute_lunar_features(date_str, hour_art=21):
    """
    Computes exact astronomical lunar features for a given date at hour_art ART (UTC-3).
    """
    dt_local = datetime.datetime.strptime(f"{date_str} {hour_art:02d}:00:00", "%Y-%m-%d %H:%M:%S")
    # ART is UTC-3
    dt_utc = dt_local.replace(tzinfo=datetime.timezone(datetime.timedelta(hours=-3))).astimezone(datetime.timezone.utc)
    
    diff_days = (dt_utc - LUNAR_REF_UTC).total_seconds() / 86400.0
    phase = (diff_days % SYNODIC_MONTH) / SYNODIC_MONTH
    theta = 2.0 * math.pi * phase
    
    lunar_sin = math.sin(theta)
    lunar_cos = math.cos(theta)
    # Illumination: 0 at new moon (phase 0), 100 at full moon (phase 0.5)
    illumination = ((1.0 - math.cos(theta)) / 2.0) * 100.0
    
    # Phase category
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

def fetch_weather_historical(lottery, start_date='2026-07-01', end_date='2026-10-16'):
    coords = COORDINATES.get(lottery, COORDINATES['ciudad'])
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={coords['lat']}&longitude={coords['lon']}&"
        f"start_date={start_date}&end_date={end_date}&"
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
        except Exception as e:
            if attempt == 2:
                print(f"[WARN] Failed to fetch weather history for {lottery}: {e}")
            time.sleep(2)
    return {}

def fetch_weather_forecast(lottery, target_date='2026-10-17'):
    coords = COORDINATES.get(lottery, COORDINATES['ciudad'])
    fallback_defaults = {
        'ciudad': {
            'temperature_2m': 18.6,
            'relative_humidity_2m': 66.0,
            'surface_pressure': 1015.2,
            'wind_speed_10m': 14.5
        },
        'formosa': {
            'temperature_2m': 25.4,
            'relative_humidity_2m': 58.0,
            'surface_pressure': 1011.8,
            'wind_speed_10m': 11.2
        }
    }
    fallback = fallback_defaults.get(lottery, fallback_defaults['ciudad'])

    urls = [
        (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={coords['lat']}&longitude={coords['lon']}&"
            f"hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m&"
            f"timezone={coords['tz'].replace('/', '%2F')}"
        ),
        (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={coords['lat']}&longitude={coords['lon']}&"
            f"start_date=2026-10-10&end_date=2026-10-17&"
            f"hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m&"
            f"timezone={coords['tz'].replace('/', '%2F')}"
        )
    ]

    for url in urls:
        for attempt in range(2):
            try:
                r = requests.get(url, timeout=12)
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
                            return {
                                'temperature_2m': float(temp),
                                'relative_humidity_2m': float(hum),
                                'surface_pressure': float(pres),
                                'wind_speed_10m': float(wnd)
                            }
                    if temps and temps[-1] is not None:
                        return {
                            'temperature_2m': float(temps[-1]),
                            'relative_humidity_2m': float(humids[-1]),
                            'surface_pressure': float(pressures[-1]),
                            'wind_speed_10m': float(winds[-1])
                        }
            except Exception:
                time.sleep(1)

    print(f"[WARN] Open-Meteo forecast timed out or failed for {lottery} on {target_date}. Using robust telemetry fallback: {fallback}")
    return fallback

def compute_autoregressive_features(df_lottery):
    """
    Computes lag features, rolling statistics, delays (atrasos), parity and sum of digits
    for a single lottery dataframe sorted chronologically.
    """
    df = df_lottery.sort_values('date').copy().reset_index(drop=True)
    
    # Intrinsic properties
    df['parity'] = df['number_2d'] % 2
    df['decena_parity'] = df['decena'] % 2
    df['unidad_parity'] = df['unidad'] % 2
    df['sum_of_digits'] = df['decena'] + df['unidad']
    
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
    
    # Rolling statistics (using past values only)
    df['rolling_mean_3'] = df['number_2d'].shift(1).rolling(window=3, min_periods=1).mean()
    df['rolling_mean_7'] = df['number_2d'].shift(1).rolling(window=7, min_periods=1).mean()
    df['rolling_std_7'] = df['number_2d'].shift(1).rolling(window=7, min_periods=2).std().fillna(0)
    
    # Compute dynamic atrasos (delay in draws since each digit 0-9 was last observed)
    atraso_dec_history = {d: 0 for d in range(10)}
    atraso_uni_history = {u: 0 for u in range(10)}
    
    row_atraso_dec = []
    row_atraso_uni = []
    row_max_atraso_dec = []
    row_max_atraso_uni = []
    
    for idx, row in df.iterrows():
        d_val = row['decena']
        u_val = row['unidad']
        
        # Current delay before this draw
        row_atraso_dec.append(atraso_dec_history[d_val])
        row_atraso_uni.append(atraso_uni_history[u_val])
        row_max_atraso_dec.append(max(atraso_dec_history.values()))
        row_max_atraso_uni.append(max(atraso_uni_history.values()))
        
        # Update delay counts
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
    
    # Store final atrasos as attribute for tomorrow's forecast
    df.attrs['final_atraso_decena'] = atraso_dec_history
    df.attrs['final_atraso_unidad'] = atraso_uni_history
    
    return df

def enrich_dataset(raw_csv_path, output_csv_path):
    print(f"Loading raw draws from {raw_csv_path}...")
    df_raw = pd.read_csv(raw_csv_path)
    df_raw['date'] = df_raw['date'].astype(str)
    
    # Fetch weather for both lotteries
    print("Fetching historical weather telemetry from Open-Meteo...")
    weather_data = {
        'ciudad': fetch_weather_historical('ciudad'),
        'formosa': fetch_weather_historical('formosa')
    }
    
    # Process per lottery
    enriched_dfs = []
    for lot in ['ciudad', 'formosa']:
        sub = df_raw[df_raw['lottery'] == lot].copy()
        if len(sub) == 0:
            continue
        print(f"Processing mechanical and lag features for {lot} ({len(sub)} rows)...")
        sub_auto = compute_autoregressive_features(sub)
        
        # Attach weather & lunar
        temps, hums, press, winds = [], [], [], []
        l_phase, l_sin, l_cos, l_illum, l_names = [], [], [], [], []
        
        for idx, row in sub_auto.iterrows():
            d_str = row['date']
            w = weather_data[lot].get(d_str, {})
            temps.append(w.get('temperature_2m', np.nan))
            hums.append(w.get('relative_humidity_2m', np.nan))
            press.append(w.get('surface_pressure', np.nan))
            winds.append(w.get('wind_speed_10m', np.nan))
            
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
        
        sub_auto['lunar_phase'] = l_phase
        sub_auto['lunar_sin'] = l_sin
        sub_auto['lunar_cos'] = l_cos
        sub_auto['lunar_illumination'] = l_illum
        sub_auto['lunar_phase_name'] = l_names
        
        # Forward-fill any minor missing weather data if any
        sub_auto['temperature_2m'] = sub_auto['temperature_2m'].bfill().ffill()
        sub_auto['relative_humidity_2m'] = sub_auto['relative_humidity_2m'].bfill().ffill()
        sub_auto['surface_pressure'] = sub_auto['surface_pressure'].bfill().ffill()
        sub_auto['wind_speed_10m'] = sub_auto['wind_speed_10m'].bfill().ffill()
        
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
        print(f"Raw draws file not found at {raw_path}. Run scrape_draws.py first.")
