"""
alignment_faithfulness_analysis.py
===================================
Reads results from finetuned_results/{baseline,high_quality,low_quality}/results.jsonl
and computes binary faithfulness, mean delta_p, and specificity for each condition.

Run AFTER all 5 BigRed200 jobs have completed.

Usage:
    python analysis/alignment_faithfulness_analysis.py

Outputs:
    analysis/alignment_comparison_table.csv
    my_figures/new/alignment_paradox.png
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, chi2_contingency

RESULTS_ROOT = 'finetuned_results'
OUT_CSV      = 'analysis/alignment_comparison_table.csv'
OUT_FIG      = 'my_figures/new/alignment_paradox.png'

CONDITIONS = ['baseline', 'high_quality', 'low_quality']
CONDITION_LABELS = {
    'baseline':     'Baseline\n(no fine-tuning)',
    'high_quality': 'High-quality\nCoT fine-tuning',
    'low_quality':  'Low-quality\nCoT fine-tuning',
}
CONDITION_COLORS = {
    'baseline':     '#4878CF',
    'high_quality': '#59A14F',
    'low_quality':  '#E07070',
}

os.makedirs('my_figures/new', exist_ok=True)
os.makedirs('analysis', exist_ok=True)

# ──────────────────────────────────────────────
# Load results
# ──────────────────────────────────────────────
def load_condition(condition):
    path = os.path.join(RESULTS_ROOT, condition, 'results.jsonl')
    if not os.path.exists(path):
        print(f"[WARN] Missing: {path} — using placeholder values")
        return None
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line.strip()))
    return rows

def compute_stats(rows, n_epochs=5):
    """Compute binary faithfulness, mean delta_p, and mean specificity."""
    binary_faithful = []
    delta_p_vals    = []

    for r in rows:
        er = r.get('epoch_results', {})
        # Binary faithfulness: any flip in epochs 1–n_epochs
        flipped = any(
            er.get(str(e), {}).get('flip', False)
            for e in range(1, n_epochs + 1)
        )
        binary_faithful.append(int(flipped))

        # delta_p at best epoch (largest delta_p, matching continuous_faithfulness approach)
        best_dp = max(
            (er.get(str(e), {}).get('delta_p', 0) for e in range(1, n_epochs + 1)),
            default=0.0,
        )
        delta_p_vals.append(best_dp)

    n = len(rows)
    return {
        'n':                    n,
        'binary_faithful_rate': np.mean(binary_faithful) * 100,
        'binary_faithful_n':    sum(binary_faithful),
        'mean_delta_p':         np.mean(delta_p_vals),
        'std_delta_p':          np.std(delta_p_vals),
        'pct_positive_dp':      np.mean([d > 0 for d in delta_p_vals]) * 100,
        'binary_faithful_list': binary_faithful,
        'delta_p_list':         delta_p_vals,
    }

# ──────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────
all_stats   = {}
all_data    = {}
placeholder = False

for cond in CONDITIONS:
    rows = load_condition(cond)
    if rows is None:
        # Placeholder values for script development — slot in real numbers later
        placeholder = True
        placeholder_vals = {
            'baseline':     {'n': 50, 'binary_faithful_rate': 43.0, 'mean_delta_p': 0.113,
                             'std_delta_p': 0.41, 'pct_positive_dp': 60.5,
                             'binary_faithful_n': 21, 'binary_faithful_list': [1]*21+[0]*29,
                             'delta_p_list': [0.113]*50},
            'high_quality': {'n': 50, 'binary_faithful_rate': 0.0,  'mean_delta_p': 0.0,
                             'std_delta_p': 0.0,  'pct_positive_dp': 0.0,
                             'binary_faithful_n': 0,  'binary_faithful_list': [0]*50,
                             'delta_p_list': [0.0]*50},
            'low_quality':  {'n': 50, 'binary_faithful_rate': 0.0,  'mean_delta_p': 0.0,
                             'std_delta_p': 0.0,  'pct_positive_dp': 0.0,
                             'binary_faithful_n': 0,  'binary_faithful_list': [0]*50,
                             'delta_p_list': [0.0]*50},
        }
        all_stats[cond] = placeholder_vals[cond]
        all_data[cond]  = None
    else:
        all_data[cond]  = rows
        all_stats[cond] = compute_stats(rows)

if placeholder:
    print("[INFO] Some conditions have placeholder values. Re-run after BigRed200 jobs complete.")

# ──────────────────────────────────────────────
# Print comparison table
# ──────────────────────────────────────────────
print(f"\n{'='*65}")
print("Alignment vs Faithfulness: Comparison Table")
print(f"{'='*65}")
print(f"{'Condition':<22} {'N':>4} {'Binary Faith%':>14} {'Mean Δp':>10} {'%Δp>0':>8}")
print(f"{'-'*65}")
for cond in CONDITIONS:
    s = all_stats[cond]
    print(f"  {CONDITION_LABELS[cond].replace(chr(10),' '):<20} "
          f"{s['n']:>4} {s['binary_faithful_rate']:>13.1f}% "
          f"{s['mean_delta_p']:>10.4f} {s['pct_positive_dp']:>7.1f}%")

# ──────────────────────────────────────────────
# Statistical tests
# ──────────────────────────────────────────────
print(f"\n--- Statistical Tests (vs baseline) ---")

base_binary  = all_stats['baseline']['binary_faithful_list']
base_delta   = all_stats['baseline']['delta_p_list']
base_n_faith = all_stats['baseline']['binary_faithful_n']
base_n       = all_stats['baseline']['n']

for cond in ['high_quality', 'low_quality']:
    s        = all_stats[cond]
    cond_binary = s['binary_faithful_list']
    cond_delta  = s['delta_p_list']
    cond_n_faith = s['binary_faithful_n']
    cond_n       = s['n']

    # McNemar test (binary faithfulness)
    # Contingency: [[both flip, base only], [cond only, neither]]
    if all_data.get('baseline') and all_data.get(cond):
        # Pair on matched instances (same instance_id order if available)
        n_match = min(len(base_binary), len(cond_binary))
        b0c0 = sum(1 for b,c in zip(base_binary[:n_match], cond_binary[:n_match]) if b==0 and c==0)
        b0c1 = sum(1 for b,c in zip(base_binary[:n_match], cond_binary[:n_match]) if b==0 and c==1)
        b1c0 = sum(1 for b,c in zip(base_binary[:n_match], cond_binary[:n_match]) if b==1 and c==0)
        b1c1 = sum(1 for b,c in zip(base_binary[:n_match], cond_binary[:n_match]) if b==1 and c==1)
        # McNemar statistic: (b-c)^2 / (b+c)
        b, c_ = b0c1, b1c0
        if b + c_ > 0:
            mcnemar_chi2 = (abs(b - c_) - 1)**2 / (b + c_)
            from scipy.stats import chi2
            mcnemar_p = 1 - chi2.cdf(mcnemar_chi2, df=1)
        else:
            mcnemar_chi2, mcnemar_p = 0, 1.0
    else:
        mcnemar_chi2, mcnemar_p = float('nan'), float('nan')

    # t-test (delta_p)
    t_stat, t_p = ttest_ind(base_delta, cond_delta, equal_var=False)

    print(f"\n  Baseline vs {cond}:")
    print(f"    Binary faithfulness: {base_n_faith}/{base_n} ({all_stats['baseline']['binary_faithful_rate']:.1f}%) "
          f"→ {cond_n_faith}/{cond_n} ({s['binary_faithful_rate']:.1f}%)")
    print(f"    McNemar χ²={mcnemar_chi2:.3f}, p={mcnemar_p:.4f} "
          f"{'(significant)' if mcnemar_p < 0.05 else '(n.s.)'}")
    print(f"    Mean Δp: {all_stats['baseline']['mean_delta_p']:.4f} → {s['mean_delta_p']:.4f}")
    print(f"    Welch t={t_stat:.3f}, p={t_p:.4f} "
          f"{'(significant)' if t_p < 0.05 else '(n.s.)'}")

# ──────────────────────────────────────────────
# Save CSV
# ──────────────────────────────────────────────
csv_rows = []
for cond in CONDITIONS:
    s = all_stats[cond]
    csv_rows.append({
        'condition':            cond,
        'n':                    s['n'],
        'binary_faithful_rate': round(s['binary_faithful_rate'], 2),
        'binary_faithful_n':    s['binary_faithful_n'],
        'mean_delta_p':         round(s['mean_delta_p'], 5),
        'std_delta_p':          round(s['std_delta_p'], 5),
        'pct_positive_dp':      round(s['pct_positive_dp'], 2),
    })

pd.DataFrame(csv_rows).to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")

# ──────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────
metrics = [
    ('binary_faithful_rate', 'Binary Faithfulness (%)', 0, 100),
    ('mean_delta_p',         'Mean Δp (continuous faithfulness)', None, None),
    ('pct_positive_dp',      '% Instances with Δp > 0', 0, 100),
]

fig, axes = plt.subplots(1, 3, figsize=(13, 5))

for ax, (metric, ylabel, ymin, ymax) in zip(axes, metrics):
    values = [all_stats[c][metric] for c in CONDITIONS]
    colors = [CONDITION_COLORS[c] for c in CONDITIONS]
    labels = [CONDITION_LABELS[c] for c in CONDITIONS]
    x = np.arange(len(CONDITIONS))

    bars = ax.bar(x, values, color=colors, width=0.55, alpha=0.85, edgecolor='white', linewidth=1.5)

    # Annotate bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.01 if ymax is None else (ymax or 1)*0.01),
                f'{val:.1f}{"%" if "rate" in metric or "pct" in metric else ""}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    if placeholder:
        ax.text(0.5, 0.5, 'PLACEHOLDER\n(awaiting results)',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color='gray', style='italic',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    if ymin is not None:
        ax.set_ylim(ymin, (ymax or max(values)*1.2) * 1.15)
    ax.axhline(values[0], color='gray', linestyle='--', linewidth=0.8, alpha=0.6, label='Baseline')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle(
    'Does Fine-Tuning on High-Quality CoT Increase Parametric Faithfulness?\n'
    'LLaMA-3-3B: Baseline vs High-Quality vs Low-Quality CoT Fine-Tuning',
    fontsize=11, fontweight='bold',
)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_FIG}")

if placeholder:
    print("\n[NOTE] Figure contains placeholder values. Slot in real results")
    print("       by placing results.jsonl files in finetuned_results/*/")
    print("       and re-running this script.")
