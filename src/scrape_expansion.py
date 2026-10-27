import os
import sys
import re
import datetime
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scrape_draws

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_PATH = os.path.join(DATA_DIR, 'raw_draws.csv')

def expand_historical_dataset(start_date_str='2024-01-01', end_date_str='2025-10-17', max_workers=16):
    start = datetime.date.fromisoformat(start_date_str)
    end = datetime.date.fromisoformat(end_date_str)

    existing_df = pd.read_csv(RAW_PATH) if os.path.exists(RAW_PATH) else pd.DataFrame(columns=['date', 'lottery', 'number_4d', 'number_2d', 'decena', 'unidad'])
    existing_dates = set(existing_df[existing_df['lottery'] == 'ciudad']['date'].astype(str))

    dates_to_fetch = []
    curr = start
    while curr <= end:
        if curr.weekday() != 6: # Skip Sundays
            ds = curr.isoformat()
            if ds not in existing_dates:
                dates_to_fetch.append(curr)
        curr += datetime.timedelta(days=1)

    print(f"[EXPAND] Found {len(existing_dates)} existing draws. Need to scrape {len(dates_to_fetch)} draws from {start_date_str} to {end_date_str}...")

    if not dates_to_fetch:
        print("[EXPAND] No new dates needed.")
        return existing_df

    scraped_rows = []
    failed_dates = []

    def fetch_date(d):
        num = scrape_draws.scrape_ciudad(d)
        if num and re.match(r'^\d{4}$', str(num)):
            n_str = f"{int(num):04d}"
            n2 = int(n_str[-2:])
            return {
                'date': d.isoformat(),
                'lottery': 'ciudad',
                'number_4d': n_str,
                'number_2d': n2,
                'decena': int(n_str[-2]),
                'unidad': int(n_str[-1])
            }
        return None

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_date = {executor.submit(fetch_date, d): d for d in dates_to_fetch}
        for future in as_completed(future_to_date):
            d = future_to_date[future]
            try:
                row = future.result()
                if row:
                    scraped_rows.append(row)
                else:
                    failed_dates.append(d)
            except Exception as e:
                failed_dates.append(d)

    print(f"[EXPAND] Scraped {len(scraped_rows)} new historical draws in {time.time()-t0:.1f}s. Failed: {len(failed_dates)}")

    # Retry failed dates once with a slower concurrency
    if failed_dates:
        print(f"[EXPAND] Retrying {len(failed_dates)} dates...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_date = {executor.submit(fetch_date, d): d for d in failed_dates}
            for future in as_completed(future_to_date):
                try:
                    row = future.result()
                    if row:
                        scraped_rows.append(row)
                except Exception:
                    pass

    new_df = pd.DataFrame(scraped_rows)
    combined = pd.concat([new_df, existing_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['date', 'lottery']).sort_values('date').reset_index(drop=True)
    combined.to_csv(RAW_PATH, index=False)
    print(f"[EXPAND] Success! Total historical draws now: {len(combined)} (from {combined['date'].min()} to {combined['date'].max()})")
    return combined

if __name__ == '__main__':
    expand_historical_dataset()
