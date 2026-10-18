import os
import sys
import time
import datetime
import requests
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scrape_draws
import enrich_features
from database import QuinielaDatabase


def fetch_open_meteo_year(start_date_str='2025-10-18', end_date_str='2026-10-17'):
    coords = enrich_features.COORDINATES['ciudad']
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={coords['lat']}&longitude={coords['lon']}&"
        f"start_date={start_date_str}&end_date={end_date_str}&"
        f"hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m&"
        f"timezone={coords['tz'].replace('/', '%2F')}"
    )
    print(f"Fetching 1-year historical weather from Open-Meteo for Buenos Aires ({start_date_str} to {end_date_str})...")
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Open-Meteo error: {r.status_code} {r.text}")
    data = r.json()
    hourly = data.get('hourly', {})
    times = hourly.get('time', [])
    temps = hourly.get('temperature_2m', [])
    humids = hourly.get('relative_humidity_2m', [])
    pressures = hourly.get('surface_pressure', [])
    winds = hourly.get('wind_speed_10m', [])

    weather_map = {}
    for t, temp, hum, pres, wnd in zip(times, temps, humids, pressures, winds):
        if t.endswith('T21:00'):
            d = t.split('T')[0]
            weather_map[d] = {
                'temperature_2m': round(float(temp), 1) if temp is not None else 18.0,
                'relative_humidity_2m': round(float(hum), 1) if hum is not None else 65.0,
                'surface_pressure': round(float(pres), 1) if pres is not None else 1015.0,
                'wind_speed_10m': round(float(wnd), 1) if wnd is not None else 12.0
            }
    print(f"Retrieved {len(weather_map)} daily 21:00 ART weather records from Open-Meteo.")
    return weather_map


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    db = QuinielaDatabase(data_dir)

    start_date = datetime.date(2025, 10, 18)
    end_date = datetime.date(2026, 10, 17)

    # 1. Fetch existing Ciudad draws from DB
    with db._get_conn() as conn:
        existing_rows = conn.execute("SELECT date, number_4d FROM draws WHERE lottery = 'ciudad'").fetchall()
    existing_draws = {r['date']: r['number_4d'] for r in existing_rows}
    print(f"Existing Ciudad draws in DB: {len(existing_draws)}")

    # 2. Identify missing draw dates (skipping Sundays)
    dates_to_scrape = []
    curr = start_date
    while curr <= end_date:
        if curr.weekday() != 6:  # Skip Sundays
            ds = curr.strftime('%Y-%m-%d')
            if ds not in existing_draws:
                dates_to_scrape.append(curr)
        curr += datetime.timedelta(days=1)

    print(f"Dates to scrape from historical portals: {len(dates_to_scrape)}")

    # 3. Scrape missing dates concurrently
    scraped_results = {}

    def fetch_date(d):
        num4d = scrape_draws.scrape_ciudad(d)
        return d.strftime('%Y-%m-%d'), num4d

    if dates_to_scrape:
        print("Starting concurrent scraping with 10 threads...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_date = {executor.submit(fetch_date, d): d for d in dates_to_scrape}
            completed = 0
            for future in as_completed(future_to_date):
                ds, num = future.result()
                if num:
                    scraped_results[ds] = num
                completed += 1
                if completed % 30 == 0 or completed == len(dates_to_scrape):
                    print(f"Scraped {completed}/{len(dates_to_scrape)} dates... ({len(scraped_results)} valid draws found)")

    # Combine existing + newly scraped draws
    all_draws = {}
    all_draws.update(existing_draws)
    all_draws.update(scraped_results)
    sorted_dates = sorted(all_draws.keys())
    print(f"Total Ciudad draws collected for 1-year period: {len(sorted_dates)}")

    # 4. Fetch 1-year 21:00 weather telemetry
    weather_map = fetch_open_meteo_year(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    # 5. Reset SQLite DB to contain ONLY 1-year Ciudad draws
    print("Writing 1-year Ciudad dataset to SQLite database...")
    now_iso = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with db._get_conn() as conn:
        conn.execute("DELETE FROM draws WHERE lottery != 'ciudad'")
        for ds in sorted_dates:
            num_str = f"{int(str(all_draws[ds]).strip()):04d}"
            n2 = int(num_str[-2:])
            dec = int(num_str[-2])
            uni = int(num_str[-1])
            w = weather_map.get(ds, {'temperature_2m': 18.0, 'relative_humidity_2m': 65.0, 'surface_pressure': 1015.0, 'wind_speed_10m': 12.0})
            lunar = enrich_features.compute_lunar_features(ds, hour_art=21)

            conn.execute("""
                INSERT INTO draws (
                    date, lottery, number_4d, number_2d, decena, unidad,
                    temperature_2m, relative_humidity_2m, surface_pressure, wind_speed_10m,
                    lunar_phase, lunar_sin, lunar_cos, lunar_illumination, lunar_phase_name,
                    source, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date, lottery) DO UPDATE SET
                    number_4d=excluded.number_4d,
                    number_2d=excluded.number_2d,
                    decena=excluded.decena,
                    unidad=excluded.unidad,
                    temperature_2m=excluded.temperature_2m,
                    relative_humidity_2m=excluded.relative_humidity_2m,
                    surface_pressure=excluded.surface_pressure,
                    wind_speed_10m=excluded.wind_speed_10m,
                    lunar_phase=excluded.lunar_phase,
                    lunar_sin=excluded.lunar_sin,
                    lunar_cos=excluded.lunar_cos,
                    lunar_illumination=excluded.lunar_illumination,
                    lunar_phase_name=excluded.lunar_phase_name,
                    source=excluded.source,
                    updated_at=excluded.updated_at
            """, (
                ds, 'ciudad', num_str, n2, dec, uni,
                w['temperature_2m'], w['relative_humidity_2m'], w['surface_pressure'], w['wind_speed_10m'],
                lunar['lunar_phase'], lunar['lunar_sin'], lunar['lunar_cos'],
                lunar['lunar_illumination'], lunar['lunar_phase_name'],
                '1year_expansion', now_iso
            ))
        conn.commit()

    # 6. Rebuild raw_draws.csv and dataset_enriched.csv
    print("Rebuilding dataset_enriched.csv and raw_draws.csv with full 1-year autoregressive lags...")
    df_enriched = db.rebuild_enriched_dataset()
    print("DB Stats after 1-year expansion:", db.get_db_stats())
    print(f"Enriched dataset shape: {df_enriched.shape}")


if __name__ == '__main__':
    main()
