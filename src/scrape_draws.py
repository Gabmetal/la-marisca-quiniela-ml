import os
import sys
import time
import datetime
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

SPANISH_MONTHS = {
    1: 'enero', 2: 'febrero', 3: 'marzo', 4: 'abril', 5: 'mayo', 6: 'junio',
    7: 'julio', 8: 'agosto', 9: 'septiembre', 10: 'octubre', 11: 'noviembre', 12: 'diciembre'
}

def parse_page_date(soup):
    for tag in soup.find_all(['div', 'h2', 'h3', 'h4']):
        text = tag.get_text(' ', strip=True).lower()
        if 'resultados del d' in text:
            return text
    return ''

def is_valid_draw_date(soup, target_date):
    page_text = parse_page_date(soup)
    if not page_text:
        return True
    m_name = SPANISH_MONTHS[target_date.month]
    d_num = f'{target_date.day:02d}'
    d_single = f'{target_date.day}'
    if (d_num in page_text or d_single in page_text) and (m_name in page_text or 'setiembre' in page_text):
        return True
    return False

def scrape_loteriasmundiales_ciudad(target_date, session=None):
    ds = target_date.strftime('%Y-%m-%d')
    s = session or requests
    try:
        r = s.post(
            'https://www.loteriasmundiales.com.ar/Quinielas/ciudad',
            data={'Fecha': ds, 'QuinielaId': '26'},
            headers=HEADERS,
            timeout=8
        )
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            if is_valid_draw_date(soup, target_date):
                for card in soup.find_all('div', class_='w3-card-4'):
                    text = card.get_text(' ', strip=True).upper()
                    if 'NOCTURNA' in text and 'CIUDAD' in text:
                        td = card.find('td', id=re.compile(r'idQ\d+_\d+_N01'))
                        if td:
                            val = td.text.strip()
                            if re.match(r'^\d{4}$', val):
                                return val
    except Exception:
        pass
    return None

def scrape_ruta1000_ciudad(target_date, session=None):
    d_fmt = target_date.strftime('%Y_%m_%d')
    url = f'https://www.ruta1000.com.ar/index2008.php?FechaAlMinuto={d_fmt}'
    s = session or requests
    try:
        r = s.get(url, headers=HEADERS, timeout=8)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            text = soup.get_text('\n')
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            for i, l in enumerate(lines):
                if l == 'NOCTURNA' and any('CIUDAD' in lines[j].upper() for j in range(max(0, i-4), min(len(lines), i+4))):
                    for k in range(i, min(len(lines), i+10)):
                        if '1' in lines[k] and k+1 < len(lines) and re.match(r'^\d{4}$', lines[k+1]):
                            return lines[k+1]
    except Exception:
        pass
    return None

def scrape_ciudad(target_date, session=None):
    val = scrape_loteriasmundiales_ciudad(target_date, session)
    if val:
        return val
    return scrape_ruta1000_ciudad(target_date, session)

def scrape_formosa(target_date, session=None):
    ds = target_date.strftime('%Y-%m-%d')
    s = session or requests
    try:
        r = s.post(
            'https://www.loteriasmundiales.com.ar/Quinielas/formosena',
            data={'Fecha': ds, 'QuinielaId': '24'},
            headers=HEADERS,
            timeout=8
        )
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            if is_valid_draw_date(soup, target_date):
                for card in soup.find_all('div', class_='w3-card-4'):
                    text = card.get_text(' ', strip=True).upper()
                    if 'NOCTURNA' in text and 'FORMOSA' in text:
                        td = card.find('td', id=re.compile(r'idQ\d+_\d+_N01'))
                        if td:
                            val = td.text.strip()
                            if re.match(r'^\d{4}$', val):
                                return val
    except Exception:
        pass
    return None

def process_single_date(curr):
    ds = curr.strftime('%Y-%m-%d')
    with requests.Session() as s:
        c_num = scrape_ciudad(curr, s)
        f_num = scrape_formosa(curr, s)
    res = []
    if c_num and re.match(r'^\d{4}$', c_num):
        res.append({
            'date': ds,
            'lottery': 'ciudad',
            'number_4d': c_num,
            'number_2d': int(c_num[-2:]),
            'decena': int(c_num[-2]),
            'unidad': int(c_num[-1])
        })
    if f_num and re.match(r'^\d{4}$', f_num):
        res.append({
            'date': ds,
            'lottery': 'formosa',
            'number_4d': f_num,
            'number_2d': int(f_num[-2:]),
            'decena': int(f_num[-2]),
            'unidad': int(f_num[-1])
        })
    return ds, c_num, f_num, res

def main():
    start_date = datetime.date(2026, 7, 1)
    end_date = datetime.date(2026, 10, 16)
    
    dates_to_scrape = []
    curr = start_date
    while curr <= end_date:
        if curr.weekday() != 6: # Skip Sundays
            dates_to_scrape.append(curr)
        curr += datetime.timedelta(days=1)
        
    print(f"Scraping {len(dates_to_scrape)} dates with parallel threadpool...")
    sys.stdout.flush()
    
    all_rows = []
    completed = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_single_date, d): d for d in dates_to_scrape}
        for future in as_completed(futures):
            ds, c_num, f_num, res = future.result()
            all_rows.extend(res)
            completed += 1
            print(f"[{completed}/{len(dates_to_scrape)}] {ds}: Ciudad={c_num}, Formosa={f_num}", flush=True)
            
    df = pd.DataFrame(all_rows)
    df = df.sort_values(['lottery', 'date']).reset_index(drop=True)
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_draws.csv')
    df.to_csv(out_path, index=False)
    print(f"\nDone! Scraped {len(df)} total records saved to {out_path}", flush=True)
    print(df.groupby('lottery').size(), flush=True)

if __name__ == '__main__':
    main()
