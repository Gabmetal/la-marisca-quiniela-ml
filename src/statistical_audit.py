import os
import sys
import math
import numpy as np
import pandas as pd
from scipy import stats

def compute_entropy(series, base=2):
    """Computes Shannon entropy in bits (base 2)."""
    counts = series.value_counts()
    probs = counts / len(series)
    ent = -sum(p * math.log(p, base) for p in probs if p > 0)
    return ent

def wald_wolfowitz_runs_test(binary_seq):
    """
    Performs Wald-Wolfowitz runs test for randomness on a binary sequence (0s and 1s).
    """
    seq = np.array(binary_seq)
    n = len(seq)
    if n < 2:
        return {'runs': 0, 'z_stat': 0.0, 'p_value': 1.0, 'is_random': True}
        
    n1 = np.sum(seq == 1)
    n0 = np.sum(seq == 0)
    
    if n0 == 0 or n1 == 0:
        return {'runs': 1, 'z_stat': -999.0, 'p_value': 0.0, 'is_random': False}
        
    # Count runs
    runs = 1 + np.sum(seq[1:] != seq[:-1])
    
    mu_r = (2.0 * n0 * n1 / n) + 1.0
    var_r = (2.0 * n0 * n1 * (2.0 * n0 * n1 - n)) / (n * n * (n - 1.0))
    
    if var_r <= 0:
        z = 0.0
        p = 1.0
    else:
        # Continuity correction
        if runs > mu_r:
            z = (runs - 0.5 - mu_r) / math.sqrt(var_r)
        elif runs < mu_r:
            z = (runs + 0.5 - mu_r) / math.sqrt(var_r)
        else:
            z = 0.0
        p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
        
    return {
        'n': int(n),
        'n0': int(n0),
        'n1': int(n1),
        'runs': int(runs),
        'expected_runs': round(mu_r, 2),
        'z_stat': round(z, 4),
        'p_value': round(p, 4),
        'is_random': bool(p > 0.05)
    }

def run_statistical_audit(df, lottery_name=None, verbose=False):
    """
    Executes Chi-Square, Runs test, Shannon entropy, and Hot/Cold ranking.
    """
    if lottery_name:
        sub = df[df['lottery'] == lottery_name].copy()
        prefix = f"[{lottery_name.upper()}]"
    else:
        sub = df.copy()
        prefix = "[COMBINED]"
        
    n_samples = len(sub)
    if verbose:
        print(f"\n=======================================================")
        print(f" STATISTICAL AUDIT REPORT: {prefix} (N = {n_samples} draws)")
        print(f"=======================================================")
    
    # 1. CHI-SQUARE TESTS
    # Decena (0-9)
    obs_dec = sub['decena'].value_counts().reindex(range(10), fill_value=0)
    chi_dec, p_dec = stats.chisquare(obs_dec)
    
    # Unidad (0-9)
    obs_uni = sub['unidad'].value_counts().reindex(range(10), fill_value=0)
    chi_uni, p_uni = stats.chisquare(obs_uni)
    
    # Parity (0 vs 1)
    obs_par = (sub['number_2d'] % 2).value_counts().reindex([0, 1], fill_value=0)
    chi_par, p_par = stats.chisquare(obs_par)
    
    if verbose:
        print("\n--- 1. CHI-SQUARE UNIFORMITY TESTS ---")
        print(f"Decena (0-9):  chi2 = {chi_dec:6.3f}, df = 9, p-value = {p_dec:.4f} -> {'Uniform (p > 0.05)' if p_dec > 0.05 else 'Non-uniform anomaly (p <= 0.05)'}")
        print(f"Unidad (0-9):  chi2 = {chi_uni:6.3f}, df = 9, p-value = {p_uni:.4f} -> {'Uniform (p > 0.05)' if p_uni > 0.05 else 'Non-uniform anomaly (p <= 0.05)'}")
        print(f"Parity (Even/Odd): chi2 = {chi_par:6.3f}, df = 1, p-value = {p_par:.4f} -> {'Balanced (p > 0.05)' if p_par > 0.05 else 'Skewed (p <= 0.05)'}")
    
    # 2. WALD-WOLFOWITZ RUNS TEST (Temporal independence)
    runs_parity = wald_wolfowitz_runs_test((sub['number_2d'] % 2).values)
    median_val = sub['number_2d'].median()
    runs_median = wald_wolfowitz_runs_test((sub['number_2d'] >= median_val).astype(int).values)
    
    if verbose:
        print("\n--- 2. WALD-WOLFOWITZ RUNS TESTS (TEMPORAL INDEPENDENCE) ---")
        print(f"Parity sequence:   Observed Runs = {runs_parity['runs']}, Expected = {runs_parity['expected_runs']}, Z = {runs_parity['z_stat']}, p-value = {runs_parity['p_value']} -> {'Independent / Random' if runs_parity['is_random'] else 'Clustering/Persistence'}")
        print(f"Median sequence:   Observed Runs = {runs_median['runs']}, Expected = {runs_median['expected_runs']}, Z = {runs_median['z_stat']}, p-value = {runs_median['p_value']} -> {'Independent / Random' if runs_median['is_random'] else 'Clustering/Persistence'}")
    
    # 3. SHANNON ENTROPY
    ent_dec = compute_entropy(sub['decena'])
    max_ent_digit = math.log2(10)  # ~3.321928 bits
    ent_uni = compute_entropy(sub['unidad'])
    ent_num2d = compute_entropy(sub['number_2d'])
    max_ent_num2d = math.log2(100) # ~6.643856 bits
    
    if verbose:
        print("\n--- 3. SHANNON INFORMATION ENTROPY ---")
        print(f"Decena Entropy:   {ent_dec:.4f} bits (Max theoretical: {max_ent_digit:.4f} bits, Efficiency: {ent_dec/max_ent_digit*100:.2f}%)")
        print(f"Unidad Entropy:   {ent_uni:.4f} bits (Max theoretical: {max_ent_digit:.4f} bits, Efficiency: {ent_uni/max_ent_digit*100:.2f}%)")
        print(f"Number 2D Entropy:{ent_num2d:.4f} bits (Max theoretical: {max_ent_num2d:.4f} bits, Efficiency: {ent_num2d/max_ent_num2d*100:.2f}%)")
    
    # 4. HOT & COLD ANOMALY RANKING
    # Top 5 most frequent 2D numbers
    freq_2d = sub['number_2d'].value_counts()
    top5_hot_2d = freq_2d.head(5)
    
    # Delays (Atrasos)
    last_seen_2d = {}
    last_seen_dec = {}
    last_seen_uni = {}
    
    sub_sorted = sub.sort_values('date').reset_index(drop=True)
    total_draws = len(sub_sorted)
    
    for i, row in sub_sorted.iterrows():
        last_seen_2d[int(row['number_2d'])] = i
        last_seen_dec[int(row['decena'])] = i
        last_seen_uni[int(row['unidad'])] = i
        
    delay_2d = {num: (total_draws - 1 - last_seen_2d[num]) if num in last_seen_2d else total_draws for num in range(100)}
    delay_dec = {d: (total_draws - 1 - last_seen_dec[d]) if d in last_seen_dec else total_draws for d in range(10)}
    delay_uni = {u: (total_draws - 1 - last_seen_uni[u]) if u in last_seen_uni else total_draws for u in range(10)}
    
    top5_delay_2d = sorted(delay_2d.items(), key=lambda x: x[1], reverse=True)[:5]
    top5_delay_dec = sorted(delay_dec.items(), key=lambda x: x[1], reverse=True)[:5]
    top5_delay_uni = sorted(delay_uni.items(), key=lambda x: x[1], reverse=True)[:5]
    
    top5_hot_dec = sub['decena'].value_counts().head(5)
    top5_hot_uni = sub['unidad'].value_counts().head(5)
    
    if verbose:
        print("\n--- 4. HOT & COLD ANOMALIES (SINCE JULY 2026) ---")
        print("Top 5 Hot Numbers (2D): " + ", ".join([f"{num:02d} ({cnt}x)" for num, cnt in top5_hot_2d.items()]))
        print("Top 5 Cold/Delayed (2D): " + ", ".join([f"{num:02d} ({del_cnt} draws ago)" for num, del_cnt in top5_delay_2d]))
        print("Top 5 Hot Decenas:      " + ", ".join([f"Dec {d} ({cnt}x)" for d, cnt in top5_hot_dec.items()]))
        print("Top 5 Delayed Decenas:  " + ", ".join([f"Dec {d} ({del_cnt} draws)" for d, del_cnt in top5_delay_dec]))
        print("Top 5 Hot Unidades:     " + ", ".join([f"Uni {u} ({cnt}x)" for u, cnt in top5_hot_uni.items()]))
        print("Top 5 Delayed Unidades: " + ", ".join([f"Uni {u} ({del_cnt} draws)" for u, del_cnt in top5_delay_uni]))
    
    return {
        'lottery': lottery_name or 'combined',
        'n_samples': n_samples,
        'chi_square': {
            'decena': {'stat': chi_dec, 'p_val': p_dec},
            'unidad': {'stat': chi_uni, 'p_val': p_uni},
            'parity': {'stat': chi_par, 'p_val': p_par}
        },
        'runs_test': {
            'parity': runs_parity,
            'median': runs_median
        },
        'entropy': {
            'decena': ent_dec,
            'unidad': ent_uni,
            'number_2d': ent_num2d
        },
        'hot_2d': list(top5_hot_2d.index),
        'cold_2d': [x[0] for x in top5_delay_2d],
        'hot_decena': list(top5_hot_dec.index),
        'cold_decena': [x[0] for x in top5_delay_dec],
        'hot_unidad': list(top5_hot_uni.index),
        'cold_unidad': [x[0] for x in top5_delay_uni],
        'obs_dec': obs_dec.to_dict(),
        'obs_uni': obs_uni.to_dict(),
        'delay_dec': delay_dec,
        'delay_uni': delay_uni
    }

def main():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset_enriched.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_draws.csv')
        
    if not os.path.exists(data_path):
        print("Dataset not found. Please scrape draws first.")
        return
        
    df = pd.read_csv(data_path)
    audit_c = run_statistical_audit(df, 'ciudad')
    audit_f = run_statistical_audit(df, 'formosa')
    audit_all = run_statistical_audit(df, None)

if __name__ == '__main__':
    main()
