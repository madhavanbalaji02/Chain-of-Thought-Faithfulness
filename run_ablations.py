"""
run_ablations.py
================
Standalone analysis script adapted from Ablations.ipynb.

Differences from the original notebook:
  - Path: 'final_results/' instead of 'results/' (our run output directory)
  - Filename pattern: s=False instead of s=True (how unlearn.py names stepwise outputs)
  - LR: all combinations use lr=3e-05 (single LR run; no ablation sweep)
  - LR ablation cells (depend on 'ablation/' directory) are skipped
  - All figures saved to my_figures/ with descriptive names
  - CSV tables saved alongside figures for easy comparison
"""

import os, sys, json, csv
import numpy as np
import scipy
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# -- local imports --
from stats import (instance_changed_prediction, efficacy_reduction_per_instance_scaled,
                   changed_prediction, compute_specificity, average_efficacy, make_stats)
from util import load_results, sort_key, group_results, filter_for_agreement
from const import datasets, models, LETTERS
from plotting import scatter_results, model_color, model_to_nice_model, dataset_to_nice_dataset

# ─────────────────────────────────────────────
# CONFIG — single LR run on final_results/
# ─────────────────────────────────────────────
PATH_ROOT   = 'final_results'
LR          = 3e-05         # only LR we ran
STEPWISE    = False         # matches s=False in our filenames
METHOD      = 'npo_KL'
RS          = 1001
POS         = True
FF2         = True
TYPE        = 'sentencize'

OUT_DIR = 'my_figures'
os.makedirs(OUT_DIR, exist_ok=True)

# Override best-LR map to always use our single LR
MY_BEST_LR = {dataset: {model: LR for model in models} for dataset in datasets}

# ─────────────────────────────────────────────
# 1. Load all results
# ─────────────────────────────────────────────
def load_my_results(path_root=PATH_ROOT, lr=LR):
    results = {}
    missing = []
    for dataset in datasets:
        for model in models:
            floc = (f'{path_root}/{dataset}/{model}/'
                    f'{METHOD}_{TYPE}_s={STEPWISE}_lr={lr}_rs={RS}'
                    f'_pos={POS}_ff2={FF2}.out')
            if not os.path.exists(floc):
                missing.append(floc)
                continue
            per_instance_results = load_results(floc)
            if per_instance_results:
                key = f"{dataset},{model},{METHOD},{lr}"
                results[key] = per_instance_results
    if missing:
        print(f"[WARN] Missing files ({len(missing)}):")
        for f in missing:
            print(f"  {f}")
    return results

print("Loading results from", PATH_ROOT, "...")
best_results = load_my_results()
print(f"Loaded {len(best_results)} result files.")
print("Keys:", sorted(best_results.keys()))

# ─────────────────────────────────────────────
# 2. Filter for agreement, group by question
# ─────────────────────────────────────────────
filtered_best_results = {}
for a_key, result in sorted(best_results.items(), key=lambda t: sort_key(t[0])):
    dataset, model, _, _ = a_key.split(",")
    filtered_result = filter_for_agreement(result)
    filtered_best_results[a_key] = filtered_result
    raw_n = len(result)
    filt_n = len(filtered_result)
    print(f"  {dataset}/{model}: {raw_n} steps → {filt_n} after agreement filter "
          f"({100*filt_n/raw_n:.1f}%)")

grouped_best_results = group_results(filtered_best_results)
grouped_results_all = group_results(best_results)    # unfiltered, for baseline accuracy

# ─────────────────────────────────────────────
# 3. Instance-level faithfulness table (Table 2 / main result)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("TABLE: Main faithfulness results (lr=3e-05, npo_KL, ff2, pos)")
print("="*60)

model_to_nice_model_local = {
    'Phi-3': 'Phi-3',
    'LLaMA-3': 'LLaMA-3-8B',
    'LLaMA-3-3B': 'LLaMA-3-3B',
    'Mistral-2': 'Mistral-2',
}
dataset_to_nice_local = {
    'arc-challenge': 'ARC-Challenge',
    'openbook': 'OpenBookQA',
    'sqa': 'StrategyQA',
    'sports': 'Sports',
}

faithfulness_stats = {}
for a_key, results in grouped_best_results.items():
    d, m, _, _ = a_key.split(",")
    k_s = ' '.join([d, m])
    faithfulness = []
    for idx, (q_key, q_results) in enumerate(results.items()):
        ff = False
        for unlearned_step in q_results:
            step_results = unlearned_step['unlearning_results']
            has_flip, flips = instance_changed_prediction(step_results)
            if has_flip:
                ff = True
                break
        faithfulness.append(ff)
    ff_mu = np.mean(faithfulness)
    N = len(faithfulness)
    if k_s in faithfulness_stats:
        if ff_mu > faithfulness_stats[k_s][0]:
            faithfulness_stats[k_s] = (ff_mu, N)
    else:
        faithfulness_stats[k_s] = (ff_mu, N)

print('\t\t' + '\t'.join(['ARC-Challenge', 'OpenBookQA', 'Sports', 'StrategyQA']))
table_rows = []
for model in ['LLaMA-3', 'LLaMA-3-3B', 'Mistral-2', 'Phi-3']:
    row = [model_to_nice_model_local[model]]
    print(f"{model_to_nice_model_local[model]:15s}", end=" | ")
    for dataset in ['arc-challenge', 'openbook', 'sports', 'sqa']:
        k = ' '.join([dataset, model])
        if k not in faithfulness_stats:
            print("  N/A  ", end="\t")
            row.append("N/A")
        else:
            val = faithfulness_stats[k][0] * 100.
            n = faithfulness_stats[k][1]
            print(f"{val:5.2f}% (n={n})", end="\t")
            row.append(f"{val:.2f}")
    print()
    table_rows.append(row)

# Save CSV
csv_path = f'{OUT_DIR}/table2_main_faithfulness.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'ARC-Challenge', 'OpenBookQA', 'Sports', 'StrategyQA'])
    writer.writerows(table_rows)
print(f"\nSaved: {csv_path}")

# ─────────────────────────────────────────────
# 4. Full stats (efficacy, specificity, faithfulness) per combo
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("TABLE: Efficacy / Specificity / Faithfulness (all 16 combos)")
print("="*60)
print(f"{'Model':15s} {'Dataset':15s} {'Efficacy':>10s} {'Specificity':>12s} {'Faithfulness':>13s} {'N':>5s}")
print("-"*70)

full_stats = {}
for a_key, results in sorted(filtered_best_results.items(), key=lambda t: sort_key(t[0])):
    d, m, _, _ = a_key.split(",")
    stats = make_stats(results)
    full_stats[a_key] = stats
    print(f"{model_to_nice_model_local[m]:15s} {dataset_to_nice_local[d]:15s} "
          f"{stats['efficacy']:10.2f} {stats['specificity']:12.2f} "
          f"{stats['faithfulness']:13.2f} {stats['n_instances']:5d}")

# Save CSV
csv_path2 = f'{OUT_DIR}/table_full_stats.csv'
with open(csv_path2, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'Dataset', 'Efficacy', 'Specificity', 'Faithfulness', 'N_instances'])
    for a_key, stats in sorted(full_stats.items()):
        d, m, _, _ = a_key.split(",")
        writer.writerow([model_to_nice_model_local[m], dataset_to_nice_local[d],
                         f"{stats['efficacy']:.4f}", f"{stats['specificity']:.4f}",
                         f"{stats['faithfulness']:.4f}", stats['n_instances']])
print(f"Saved: {csv_path2}")

# ─────────────────────────────────────────────
# 5. Baseline accuracy: no-CoT vs CoT (Table 1)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("TABLE: Baseline accuracy (no-CoT vs CoT)")
print("="*60)

nocot_stats = {}
cot_stats = {}
for a_key, results in grouped_results_all.items():
    nocot_stats[a_key] = []
    cot_stats[a_key] = []
    for idx, (q_key, q_results) in enumerate(results.items()):
        y_hat = LETTERS.index(q_results[0]['correct'])
        correct_nocot = q_results[0]['prediction'] == y_hat
        correct_cot   = q_results[0]['cot_prediction'] == y_hat
        nocot_stats[a_key].append(correct_nocot)
        cot_stats[a_key].append(correct_cot)

model_acc = {}
for k in sorted(nocot_stats):
    d, m, _, _ = k.split(",")
    k_s = ' '.join([d, m])
    if k_s in model_acc:
        continue
    model_acc[k_s] = {
        'nocot': np.mean(nocot_stats[k]),
        'cot':   np.mean(cot_stats[k]),
    }

print(f"{'Model':15s} {'Dataset':15s} {'No-CoT':>8s} {'CoT':>8s}")
print("-"*50)
acc_rows = []
for model in ['LLaMA-3', 'LLaMA-3-3B', 'Mistral-2', 'Phi-3']:
    for dataset in ['arc-challenge', 'openbook', 'sports', 'sqa']:
        k = ' '.join([dataset, model])
        if k not in model_acc:
            continue
        nocot = model_acc[k]['nocot']
        cot   = model_acc[k]['cot']
        print(f"{model_to_nice_model_local[model]:15s} {dataset_to_nice_local[dataset]:15s} "
              f"{nocot:8.3f} {cot:8.3f}")
        acc_rows.append([model_to_nice_model_local[model], dataset_to_nice_local[dataset],
                         f"{nocot:.4f}", f"{cot:.4f}"])

csv_path3 = f'{OUT_DIR}/table1_baseline_accuracy.csv'
with open(csv_path3, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'Dataset', 'NoCoT_Accuracy', 'CoT_Accuracy'])
    writer.writerows(acc_rows)
print(f"Saved: {csv_path3}")

# ─────────────────────────────────────────────
# 6. Scatter plot: Efficacy vs Specificity (Fig 2 style)
# ─────────────────────────────────────────────
print("\nGenerating efficacy–specificity scatter plot...")
plt.style.use('tableau-colorblind10')

fig, axs = plt.subplots(2, 2, figsize=(8, 6))
major_ticks = np.arange(0, 101, 20)

from matplotlib.lines import Line2D
model_handles = [
    Line2D([0], [0], color=model_color[m], linestyle='None', marker='s',
           markersize=8, label=model_to_nice_model[m])
    for m in model_color
]

for idx, dataset in enumerate(sorted(datasets)):
    row = idx // 2
    col = idx % 2
    ax = axs[row][col]
    ax.set_ylim(-5, 105)
    ax.set_xlim(-5, 105)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.grid()
    ax.set_title(dataset_to_nice_local.get(dataset, dataset), fontweight='bold')
    if col == 0: ax.set_ylabel('Specificity', fontsize=12)
    if row == 1: ax.set_xlabel('Efficacy', fontsize=12)

    for model in models:
        a_key = f"{dataset},{model},{METHOD},{LR}"
        if a_key not in full_stats:
            continue
        stats = full_stats[a_key]
        E = stats['efficacy']
        S = stats['specificity']
        F = stats['faithfulness']
        size = 50 + F / 100. * 150
        ax.scatter([E], [S], label=model, marker='*',
                   facecolors='none', edgecolors=model_color[model], s=size)

lgd = fig.legend(handles=model_handles, loc='lower left', ncol=2,
                 fancybox=True, bbox_to_anchor=(0.01, 0.01))
plt.tight_layout()
scatter_path = f'{OUT_DIR}/fig2_efficacy_specificity_scatter.png'
fig.savefig(scatter_path, dpi=150, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close()
print(f"Saved: {scatter_path}")

# ─────────────────────────────────────────────
# 7. Faithfulness bar chart by model and dataset
# ─────────────────────────────────────────────
print("Generating faithfulness bar chart...")
datasets_ordered  = ['arc-challenge', 'openbook', 'sports', 'sqa']
models_ordered    = ['LLaMA-3', 'LLaMA-3-3B', 'Mistral-2', 'Phi-3']
nice_datasets     = [dataset_to_nice_local[d] for d in datasets_ordered]
nice_models       = [model_to_nice_model_local[m] for m in models_ordered]

x = np.arange(len(datasets_ordered))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 5))
for i, model in enumerate(models_ordered):
    vals = []
    for dataset in datasets_ordered:
        k = ' '.join([dataset, model])
        vals.append(faithfulness_stats[k][0] * 100. if k in faithfulness_stats else 0.)
    bars = ax.bar(x + i * width, vals, width, label=nice_models[i],
                  color=model_color[model], alpha=0.85)

ax.set_xlabel('Dataset')
ax.set_ylabel('Faithfulness (%)')
ax.set_title('Faithfulness by Model and Dataset (lr=3e-05, npo_KL, ff2+pos)')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(nice_datasets)
ax.legend()
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.5)
plt.tight_layout()
bar_path = f'{OUT_DIR}/fig_faithfulness_by_model_dataset.png'
fig.savefig(bar_path, dpi=150)
plt.close()
print(f"Saved: {bar_path}")

# ─────────────────────────────────────────────
# 8. Correlation: Efficacy vs Faithfulness
# ─────────────────────────────────────────────
print("Generating correlation plot (efficacy vs faithfulness)...")
E_all, F_all = [], []
corr_by_model   = {m: [] for m in models}
corr_by_dataset = {d: [] for d in datasets}

for a_key, stats in full_stats.items():
    d, m, _, _ = a_key.split(",")
    e = stats['efficacy']
    f = stats['faithfulness']
    E_all.append(e)
    F_all.append(f)
    corr_by_model[m].append((e, f))
    corr_by_dataset[d].append((e, f))

frame = pd.DataFrame({'efficacy': E_all, 'faithfulness': F_all})
corr = pearsonr(frame['efficacy'], frame['faithfulness'])

fig, ax = plt.subplots(figsize=(4, 3))
ax.scatter(frame['efficacy'], frame['faithfulness'])
ax.set_title(f"Overall: Corr={corr.statistic:.3f}, p={corr.pvalue:.3g}")
ax.set_xlabel('Efficacy')
ax.set_ylabel('Faithfulness')
plt.tight_layout()
corr_path = f'{OUT_DIR}/fig_correlation_efficacy_faithfulness.png'
fig.savefig(corr_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {corr_path}")
print(f"  Overall Pearson r={corr.statistic:.3f}, p={corr.pvalue:.4f}")

# Per-model
N_sub = len(corr_by_model)
fig, axs = plt.subplots(1, N_sub, figsize=(12, 2.5))
for idx, (m, vals) in enumerate(sorted(corr_by_model.items())):
    if len(vals) < 2:
        continue
    E, F = zip(*vals)
    c = pearsonr(E, F)
    axs[idx].scatter(E, F)
    axs[idx].set_title(f"{model_to_nice_model_local[m]}\nr={c.statistic:.2f}, p={c.pvalue:.3g}")
    axs[idx].set_xlabel('Efficacy')
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/fig_correlation_by_model.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_DIR}/fig_correlation_by_model.png")

# Per-dataset
fig, axs = plt.subplots(1, len(corr_by_dataset), figsize=(12, 2.5))
for idx, (d, vals) in enumerate(sorted(corr_by_dataset.items())):
    if len(vals) < 2:
        continue
    E, F = zip(*vals)
    c = pearsonr(E, F)
    axs[idx].scatter(E, F)
    axs[idx].set_title(f"{dataset_to_nice_local[d]}\nr={c.statistic:.2f}, p={c.pvalue:.3g}")
    axs[idx].set_xlabel('Efficacy')
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/fig_correlation_by_dataset.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_DIR}/fig_correlation_by_dataset.png")

# ─────────────────────────────────────────────
# 9. Print LaTeX-style tables for report
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("LATEX: Main faithfulness table")
print("="*60)
print('\t\t' + '\t'.join(['arc-challenge', 'openbook', 'sports', '\tsqa']))
for model in ['LLaMA-3', 'LLaMA-3-3B', 'Mistral-2', 'Phi-3']:
    print(f"{model:10s}:", end=" => ")
    for dataset in ['arc-challenge', 'openbook', 'sports', 'sqa']:
        k = ' '.join([dataset, model])
        if k not in faithfulness_stats:
            print("N/A", end=" & ")
        else:
            print(f"{faithfulness_stats[k][0]*100.:.2f}\\%", end=" & ")
    print()

print("\n" + "="*60)
print("LATEX: Efficacy / Specificity per combo")
print("="*60)
for model in ['LLaMA-3', 'LLaMA-3-3B', 'Mistral-2', 'Phi-3']:
    print(f"--- {model_to_nice_model_local[model]} ---")
    for dataset in ['arc-challenge', 'openbook', 'sports', 'sqa']:
        a_key = f"{dataset},{model},{METHOD},{LR}"
        if a_key not in full_stats:
            print(f"  {dataset}: N/A")
        else:
            s = full_stats[a_key]
            print(f"  {dataset_to_nice_local[dataset]:15s}: "
                  f"E={s['efficacy']:.1f}  S={s['specificity']:.1f}  F={s['faithfulness']:.1f}  n={s['n_instances']}")

print("\nDone. All outputs written to:", OUT_DIR)
print("Files:")
for f in sorted(os.listdir(OUT_DIR)):
    fpath = os.path.join(OUT_DIR, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {f:50s} {size_kb:7.1f} KB")
