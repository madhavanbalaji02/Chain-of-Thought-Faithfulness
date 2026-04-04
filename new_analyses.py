"""
new_analyses.py
===============
Four novel analyses on the final_results/ data — potential contributions
beyond what Tutek et al. (2025) reported.

Analysis 5a — Step Position: Which CoT sentence position is unlearned,
              and how does CoT length correlate with first-step faithfulness?

Analysis 5b — Faithfulness × Accuracy: 2×2 cross-tabulation
              (faithful vs unfaithful) × (correct vs wrong answer).

Analysis 5c — Model Size vs Faithfulness: LLaMA-3-3B (3B) vs LLaMA-3-8B
              across all 4 datasets.

Analysis 5d — Dataset Difficulty vs Faithfulness: rank datasets by both
              CoT-accuracy and faithfulness, look for a relationship.

All figures saved to my_figures/new/.
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict, Counter
import scipy.stats as stats

# ─── paths ────────────────────────────────────────────────
PATH_ROOT = 'final_results'
OUT_DIR   = 'my_figures/new'
os.makedirs(OUT_DIR, exist_ok=True)

METHOD = 'npo_KL'
LR     = '3e-05'
RS     = 1001
POS    = True
FF2    = True
TYPE   = 'sentencize'
STEP   = False   # s=False

DATASETS = ['arc-challenge', 'openbook', 'sports', 'sqa']
MODELS   = ['LLaMA-3', 'LLaMA-3-3B', 'Mistral-2', 'Phi-3']

NICE_DATASET = {
    'arc-challenge': 'ARC-Challenge',
    'openbook':      'OpenBookQA',
    'sports':        'Sports',
    'sqa':           'StrategyQA',
}
NICE_MODEL = {
    'LLaMA-3':    'LLaMA-3-8B',
    'LLaMA-3-3B': 'LLaMA-3-3B',
    'Mistral-2':  'Mistral-7B',
    'Phi-3':      'Phi-3',
}
MODEL_COLOR = {
    'LLaMA-3':    'tab:red',
    'LLaMA-3-3B': 'tab:orange',
    'Mistral-2':  'tab:green',
    'Phi-3':      'tab:blue',
}
LETTERS = ['A', 'B', 'C', 'D', 'E']

def fpath(dataset, model):
    return (f'{PATH_ROOT}/{dataset}/{model}/'
            f'{METHOD}_{TYPE}_s={STEP}_lr={LR}_rs={RS}'
            f'_pos={POS}_ff2={FF2}.out')

def load(dataset, model):
    p = fpath(dataset, model)
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p)]

def flip_occurred(unlearning_results):
    """True if prediction changed in any epoch after epoch 0."""
    preds = [r['prediction'] for _, r in sorted(unlearning_results.items(), key=lambda x: int(x[0]))]
    return any(p != preds[0] for p in preds[1:])

def split_sentences(text):
    """Rough sentence splitter — split on '. ' and '\n'."""
    import re
    text = text.strip()
    # Split on sentence-ending punctuation followed by space or newline
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 8]

# ══════════════════════════════════════════════════════════
# Analysis 5a — Step Position & CoT Length
# ══════════════════════════════════════════════════════════
print("="*60)
print("5a: Step Position & CoT Length Analysis")
print("="*60)

# We always unlearn step_idx=0 (first sentence).
# Key questions:
#   1. What is the CoT length distribution across datasets?
#   2. Does CoT length predict first-step faithfulness?
#   3. What is the normalized position of the unlearned step
#      (always 0 from start, but varies 1/(N) to 1 from end)?

step_pos_data = {}   # (dataset, model) -> list of dicts

for dataset in DATASETS:
    for model in MODELS:
        rows = load(dataset, model)
        records = []
        for r in rows:
            cot = r['initial_cot']
            sents = split_sentences(cot)
            n_sents = max(1, len(sents))
            target = r['unlearning_results']['0']['target_cot_step'].strip()

            # Find position of target sentence from start
            pos_from_start = 0  # always step_idx=0
            # Position from end (last sentence = 0, second-to-last = 1, ...)
            pos_from_end = n_sents - 1

            faithful = flip_occurred(r['unlearning_results'])
            records.append({
                'n_sents': n_sents,
                'pos_from_start': pos_from_start,
                'pos_from_end': pos_from_end,
                'faithful': faithful,
                'question': r['question'],
            })
        step_pos_data[(dataset, model)] = records

# Plot 1: CoT length distribution by dataset
fig, axs = plt.subplots(1, 4, figsize=(14, 3.5), sharey=False)
for i, dataset in enumerate(DATASETS):
    # Pool all models for this dataset
    all_lengths = []
    for model in MODELS:
        all_lengths += [r['n_sents'] for r in step_pos_data[(dataset, model)]]
    c = Counter(all_lengths)
    lengths = sorted(c.keys())
    counts  = [c[l] for l in lengths]
    axs[i].bar(lengths, counts, color='steelblue', alpha=0.8)
    axs[i].set_title(NICE_DATASET[dataset], fontweight='bold')
    axs[i].set_xlabel('CoT sentences')
    if i == 0: axs[i].set_ylabel('Count')
    mean_l = np.mean(all_lengths)
    axs[i].axvline(mean_l, color='red', linestyle='--', label=f'mean={mean_l:.1f}')
    axs[i].legend(fontsize=8)
plt.suptitle('CoT Length Distribution by Dataset (all models pooled)', fontweight='bold')
plt.tight_layout()
p = f'{OUT_DIR}/5a_cot_length_distribution.png'
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p}")

# Plot 2: Faithfulness rate by CoT length
fig, axs = plt.subplots(1, 4, figsize=(14, 3.5))
for i, dataset in enumerate(DATASETS):
    length_faithful = defaultdict(list)
    for model in MODELS:
        for r in step_pos_data[(dataset, model)]:
            length_faithful[r['n_sents']].append(r['faithful'])
    lengths = sorted(k for k in length_faithful if len(length_faithful[k]) >= 5)
    faith_rates = [np.mean(length_faithful[l]) * 100 for l in lengths]
    ns = [len(length_faithful[l]) for l in lengths]
    axs[i].bar(lengths, faith_rates, color='tomato', alpha=0.85)
    axs[i].set_title(NICE_DATASET[dataset], fontweight='bold')
    axs[i].set_xlabel('CoT sentences')
    axs[i].set_ylim(0, 100)
    if i == 0: axs[i].set_ylabel('Faithfulness (%)')
    for l, r, n in zip(lengths, faith_rates, ns):
        if n >= 5:
            axs[i].text(l, r + 1, f'n={n}', ha='center', fontsize=7)

    # Correlation test
    all_lens = [r['n_sents'] for m in MODELS for r in step_pos_data[(dataset, m)]]
    all_faith = [int(r['faithful']) for m in MODELS for r in step_pos_data[(dataset, m)]]
    if len(set(all_lens)) > 2:
        corr, pval = stats.pointbiserialr(all_faith, all_lens)
        axs[i].set_xlabel(f'CoT sentences\nr={corr:.2f}, p={pval:.3f}')

plt.suptitle('First-Step Faithfulness Rate by CoT Length\n(longer CoT = first step less critical?)',
             fontweight='bold')
plt.tight_layout()
p = f'{OUT_DIR}/5a_faithfulness_by_cot_length.png'
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p}")

# Print summary
print("\nCoT length stats (mean sentences per instance):")
for dataset in DATASETS:
    for model in ['LLaMA-3', 'LLaMA-3-3B']:
        lens = [r['n_sents'] for r in step_pos_data[(dataset, model)]]
        faith = [r['faithful'] for r in step_pos_data[(dataset, model)]]
        print(f"  {NICE_DATASET[dataset]:15s} {NICE_MODEL[model]:12s}: "
              f"mean_len={np.mean(lens):.2f}  faithfulness={np.mean(faith)*100:.1f}%")

print("\nCorrelation: CoT length vs first-step faithfulness (all data):")
all_lens_global  = [r['n_sents']      for d in DATASETS for m in MODELS for r in step_pos_data[(d,m)]]
all_faith_global = [int(r['faithful']) for d in DATASETS for m in MODELS for r in step_pos_data[(d,m)]]
corr, pval = stats.pointbiserialr(all_faith_global, all_lens_global)
print(f"  r={corr:.3f}, p={pval:.4f} — {'significant' if pval < 0.05 else 'not significant'}")

# ══════════════════════════════════════════════════════════
# Analysis 5b — Faithfulness × Accuracy Cross-Tabulation
# ══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("5b: Faithfulness × Accuracy Cross-Tabulation")
print("="*60)

fa_data = {}   # (dataset, model) -> {FF, FW, UF, UW} counts

for dataset in DATASETS:
    for model in MODELS:
        rows = load(dataset, model)
        # Use only agreement instances (prediction == cot_prediction)
        ff = fw = uf = uw = 0
        for r in rows:
            if r['prediction'] != r.get('cot_prediction', r['prediction']):
                continue  # skip disagreement instances
            correct_idx = LETTERS.index(r['correct'])
            correct = (r['prediction'] == correct_idx)
            faithful = flip_occurred(r['unlearning_results'])
            if faithful and correct:     ff += 1
            elif faithful and not correct: fw += 1
            elif not faithful and correct: uf += 1
            else:                          uw += 1
        n = ff + fw + uf + uw
        fa_data[(dataset, model)] = {
            'FF': ff, 'FW': fw, 'UF': uf, 'UW': uw, 'N': n
        }
        if n > 0:
            print(f"  {dataset:15s} / {model:10s}: n={n:3d} | "
                  f"Faithful+Correct={ff/n*100:.1f}% | "
                  f"Faithful+Wrong={fw/n*100:.1f}% | "
                  f"Unfaithful+Correct={uf/n*100:.1f}% | "
                  f"Unfaithful+Wrong={uw/n*100:.1f}%")

# Plot: 4-cell stacked bar, one bar per model per dataset
fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
cell_colors = {
    'FF': '#2ca02c',   # faithful+correct  green
    'UF': '#98df8a',   # unfaithful+correct light green
    'FW': '#d62728',   # faithful+wrong    red
    'UW': '#ffbb78',   # unfaithful+wrong  light orange
}
cell_labels = {
    'FF': 'Faithful + Correct',
    'UF': 'Unfaithful + Correct',
    'FW': 'Faithful + Wrong',
    'UW': 'Unfaithful + Wrong',
}

for i, dataset in enumerate(DATASETS):
    bottoms = np.zeros(len(MODELS))
    for cell in ['FF', 'UF', 'FW', 'UW']:
        vals = []
        for model in MODELS:
            d = fa_data[(dataset, model)]
            vals.append(d[cell] / d['N'] * 100 if d['N'] > 0 else 0)
        axs[i].bar(range(len(MODELS)), vals, bottom=bottoms,
                   color=cell_colors[cell], label=cell_labels[cell] if i == 0 else '_')
        bottoms += np.array(vals)
    axs[i].set_title(NICE_DATASET[dataset], fontweight='bold')
    axs[i].set_xticks(range(len(MODELS)))
    axs[i].set_xticklabels([NICE_MODEL[m] for m in MODELS], rotation=30, ha='right', fontsize=8)
    axs[i].set_ylim(0, 100)
    if i == 0: axs[i].set_ylabel('% of instances')

fig.legend(loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.05))
plt.suptitle('Faithfulness × Accuracy: Where Do Models Fail?', fontweight='bold', y=1.02)
plt.tight_layout()
p = f'{OUT_DIR}/5b_faithfulness_accuracy_crosstab.png'
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {p}")

# Aggregate across datasets for each model
print("\nAggregated over all datasets:")
print(f"{'Model':12s} | {'Faith+Correct':>14s} | {'Unfaith+Correct':>16s} | {'Faith+Wrong':>12s} | {'Unfaith+Wrong':>14s}")
print("-"*75)
for model in MODELS:
    totals = {'FF': 0, 'FW': 0, 'UF': 0, 'UW': 0, 'N': 0}
    for dataset in DATASETS:
        for k in totals:
            totals[k] += fa_data[(dataset, model)][k]
    n = totals['N']
    if n > 0:
        print(f"{NICE_MODEL[model]:12s} | {totals['FF']/n*100:14.1f}% | "
              f"{totals['UF']/n*100:16.1f}% | {totals['FW']/n*100:12.1f}% | "
              f"{totals['UW']/n*100:14.1f}%  (n={n})")

# Compute "faithfulness given correct" and "faithfulness given wrong"
print("\nConditional faithfulness:")
for model in MODELS:
    total_correct_faith = total_correct = total_wrong_faith = total_wrong = 0
    for dataset in DATASETS:
        d = fa_data[(dataset, model)]
        total_correct_faith += d['FF']
        total_correct       += d['FF'] + d['UF']
        total_wrong_faith   += d['FW']
        total_wrong         += d['FW'] + d['UW']
    if total_correct > 0 and total_wrong > 0:
        fc = total_correct_faith / total_correct * 100
        fw = total_wrong_faith   / total_wrong   * 100
        print(f"  {NICE_MODEL[model]:12s}: Faith|Correct={fc:.1f}%  Faith|Wrong={fw:.1f}%  "
              f"Δ={fc-fw:+.1f}pp")

# ══════════════════════════════════════════════════════════
# Analysis 5c — Model Size vs Faithfulness (LLaMA 3B vs 8B)
# ══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("5c: Model Size vs Faithfulness (LLaMA-3-3B vs LLaMA-3-8B)")
print("="*60)

size_data = {}
for dataset in DATASETS:
    for model in ['LLaMA-3', 'LLaMA-3-3B']:
        rows = load(dataset, model)
        filtered = [r for r in rows if r['prediction'] == r.get('cot_prediction', r['prediction'])]
        faiths = [flip_occurred(r['unlearning_results']) for r in filtered]
        size_data[(dataset, model)] = {
            'faithfulness': np.mean(faiths) * 100 if faiths else 0,
            'n': len(faiths),
        }

# Plot: side-by-side bars for each dataset
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(DATASETS))
w = 0.3
bars3b = [size_data[(d, 'LLaMA-3-3B')]['faithfulness'] for d in DATASETS]
bars8b = [size_data[(d, 'LLaMA-3')]['faithfulness']    for d in DATASETS]
ns3b   = [size_data[(d, 'LLaMA-3-3B')]['n'] for d in DATASETS]
ns8b   = [size_data[(d, 'LLaMA-3')]['n']    for d in DATASETS]

b1 = ax.bar(x - w/2, bars3b, w, label='LLaMA-3-3B (3B)', color='tab:orange', alpha=0.85)
b2 = ax.bar(x + w/2, bars8b, w, label='LLaMA-3-8B (8B)', color='tab:red',    alpha=0.85)

for rect, n in zip(b1, ns3b):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.5,
            f'n={n}', ha='center', fontsize=7)
for rect, n in zip(b2, ns8b):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.5,
            f'n={n}', ha='center', fontsize=7)

# Annotate deltas
for i in range(len(DATASETS)):
    delta = bars8b[i] - bars3b[i]
    y = max(bars3b[i], bars8b[i]) + 4
    ax.annotate(f'Δ={delta:+.1f}pp', xy=(x[i], y), ha='center', fontsize=8, color='black')

ax.set_xticks(x)
ax.set_xticklabels([NICE_DATASET[d] for d in DATASETS])
ax.set_ylabel('Faithfulness (%)')
ax.set_ylim(0, 90)
ax.set_title('Model Size vs Faithfulness: LLaMA-3 3B vs 8B\n(same architecture, different scale)',
             fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
p = f'{OUT_DIR}/5c_model_size_faithfulness.png'
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p}")

avg_gap = np.mean([bars8b[i] - bars3b[i] for i in range(len(DATASETS))])
print(f"\nLLaMA-3-8B vs LLaMA-3-3B faithfulness:")
for i, d in enumerate(DATASETS):
    print(f"  {NICE_DATASET[d]:15s}: 3B={bars3b[i]:.1f}%  8B={bars8b[i]:.1f}%  Δ={bars8b[i]-bars3b[i]:+.1f}pp")
print(f"  Average gap: +{avg_gap:.1f}pp (8B over 3B)")
print(f"  Consistent direction across all datasets: "
      f"{'YES' if all(bars8b[i] > bars3b[i] for i in range(len(DATASETS))) else 'NO'}")

# Statistical test: paired t-test per dataset
instance_faith = {}
for model in ['LLaMA-3', 'LLaMA-3-3B']:
    for dataset in DATASETS:
        rows = load(dataset, model)
        rows = [r for r in rows if r['prediction'] == r.get('cot_prediction', r['prediction'])]
        instance_faith[(dataset, model)] = [int(flip_occurred(r['unlearning_results'])) for r in rows]

# Use McNemar-style: for matched questions
print("\nStatistical comparison (paired, matched questions):")
for dataset in DATASETS:
    f3b = instance_faith[(dataset, 'LLaMA-3-3B')]
    f8b = instance_faith[(dataset, 'LLaMA-3')]
    n = min(len(f3b), len(f8b))
    # Simple chi-square on 2x2 contingency
    a = sum(1 for i in range(n) if f3b[i]==1 and f8b[i]==1)
    b = sum(1 for i in range(n) if f3b[i]==0 and f8b[i]==1)
    c = sum(1 for i in range(n) if f3b[i]==1 and f8b[i]==0)
    d = sum(1 for i in range(n) if f3b[i]==0 and f8b[i]==0)
    # McNemar test on discordant pairs
    if b + c > 0:
        stat = (abs(b - c) - 1)**2 / (b + c)
        pval = 1 - stats.chi2.cdf(stat, 1)
        print(f"  {NICE_DATASET[dataset]:15s}: McNemar χ²={stat:.2f}, p={pval:.4f} "
              f"({'significant' if pval<0.05 else 'n.s.'})")

# ══════════════════════════════════════════════════════════
# Analysis 5d — Dataset Difficulty vs Faithfulness
# ══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("5d: Dataset Difficulty vs Faithfulness")
print("="*60)

# For each dataset, compute:
# 1. Average CoT accuracy (with CoT)
# 2. Average faithfulness (across all 4 models)
# 3. Rank both

dataset_stats = {}
for dataset in DATASETS:
    faith_vals = []
    acc_vals = []
    for model in MODELS:
        rows = load(dataset, model)
        correct_idx_list = [LETTERS.index(r['correct']) for r in rows]
        cot_correct = [r.get('cot_prediction', r['prediction']) == c for r, c in zip(rows, correct_idx_list)]
        acc_vals.append(np.mean(cot_correct) * 100)

        filtered = [r for r in rows if r['prediction'] == r.get('cot_prediction', r['prediction'])]
        faiths = [flip_occurred(r['unlearning_results']) for r in filtered]
        faith_vals.append(np.mean(faiths) * 100 if faiths else 0)

    dataset_stats[dataset] = {
        'mean_accuracy':    np.mean(acc_vals),
        'mean_faithfulness': np.mean(faith_vals),
        'acc_by_model':     dict(zip(MODELS, acc_vals)),
        'faith_by_model':   dict(zip(MODELS, faith_vals)),
    }

# Print rankings
print(f"\n{'Dataset':15s} {'CoT Accuracy':>14s} {'Faithfulness':>14s} {'Acc Rank':>10s} {'Faith Rank':>11s}")
print("-"*65)
acc_ranked   = sorted(DATASETS, key=lambda d: dataset_stats[d]['mean_accuracy'],    reverse=True)
faith_ranked = sorted(DATASETS, key=lambda d: dataset_stats[d]['mean_faithfulness'], reverse=True)

for dataset in DATASETS:
    acc_r   = acc_ranked.index(dataset)   + 1
    faith_r = faith_ranked.index(dataset) + 1
    print(f"{NICE_DATASET[dataset]:15s} "
          f"{dataset_stats[dataset]['mean_accuracy']:14.1f}% "
          f"{dataset_stats[dataset]['mean_faithfulness']:14.1f}% "
          f"{'#'+str(acc_r):>10s} "
          f"{'#'+str(faith_r):>11s}")

# Spearman correlation between accuracy and faithfulness
acc_scores   = [dataset_stats[d]['mean_accuracy']    for d in DATASETS]
faith_scores = [dataset_stats[d]['mean_faithfulness'] for d in DATASETS]
rho, pval = stats.spearmanr(acc_scores, faith_scores)
print(f"\nSpearman ρ (accuracy vs faithfulness): ρ={rho:.3f}, p={pval:.4f}")
print("(Positive ρ = harder datasets have less faithful CoT; negative = harder = more faithful)")

# Plot: scatter of accuracy vs faithfulness per dataset, labeled
fig, ax = plt.subplots(figsize=(6, 5))
scatter_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for i, dataset in enumerate(DATASETS):
    x_val = dataset_stats[dataset]['mean_accuracy']
    y_val = dataset_stats[dataset]['mean_faithfulness']
    ax.scatter([x_val], [y_val], s=150, color=scatter_colors[i],
               zorder=5, label=NICE_DATASET[dataset])
    ax.annotate(NICE_DATASET[dataset], (x_val, y_val),
                textcoords='offset points', xytext=(6, 4), fontsize=9)

ax.set_xlabel('Mean CoT Accuracy (%) — all models')
ax.set_ylabel('Mean Faithfulness (%) — all models')
ax.set_title(f'Dataset Difficulty vs Faithfulness\n(Spearman ρ={rho:.2f}, p={pval:.3f})',
             fontweight='bold')
ax.grid(alpha=0.4)

# Fit regression line
z = np.polyfit(acc_scores, faith_scores, 1)
p_line = np.poly1d(z)
xs = np.linspace(min(acc_scores)-2, max(acc_scores)+2, 100)
ax.plot(xs, p_line(xs), 'k--', alpha=0.5, label='trend')
ax.legend(fontsize=8)
plt.tight_layout()
p = f'{OUT_DIR}/5d_dataset_difficulty_faithfulness.png'
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p}")

# Per-model breakdown by dataset
print("\nPer-model accuracy and faithfulness by dataset:")
for model in ['LLaMA-3', 'LLaMA-3-3B']:
    print(f"\n  {NICE_MODEL[model]}:")
    for dataset in sorted(DATASETS, key=lambda d: dataset_stats[d]['acc_by_model'][model], reverse=True):
        acc = dataset_stats[dataset]['acc_by_model'][model]
        faith = dataset_stats[dataset]['faith_by_model'][model]
        print(f"    {NICE_DATASET[dataset]:15s}: acc={acc:.1f}%  faith={faith:.1f}%")

# Plot: dataset ranking comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
for ax, metric, title, ylabel in [
    (ax1, 'mean_accuracy',    'Dataset Ranked by CoT Accuracy', 'CoT Accuracy (%)'),
    (ax2, 'mean_faithfulness','Dataset Ranked by Faithfulness', 'Faithfulness (%)'),
]:
    ranked = sorted(DATASETS, key=lambda d: dataset_stats[d][metric], reverse=True)
    vals = [dataset_stats[d][metric] for d in ranked]
    bars = ax.barh(range(len(ranked)), vals, color=scatter_colors[:len(ranked)], alpha=0.85)
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels([NICE_DATASET[d] for d in ranked])
    ax.set_xlabel(ylabel)
    ax.set_title(title, fontweight='bold')
    for bar, val in zip(bars, vals):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9)

plt.tight_layout()
p = f'{OUT_DIR}/5d_dataset_ranking.png'
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p}")

# ══════════════════════════════════════════════════════════
# Summary of all figures
# ══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("DONE — figures saved to", OUT_DIR)
print("="*60)
for f in sorted(os.listdir(OUT_DIR)):
    fsize = os.path.getsize(f'{OUT_DIR}/{f}') / 1024
    print(f"  {f:50s} {fsize:7.1f} KB")

# ══════════════════════════════════════════════════════════
# Save summary CSV for report
# ══════════════════════════════════════════════════════════
import csv

# 5b summary CSV
with open(f'{OUT_DIR}/5b_faithfulness_accuracy_crosstab.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Dataset','Model','N','Faith+Correct%','Faith+Wrong%','Unfaith+Correct%','Unfaith+Wrong%',
                'Faith|Correct%','Faith|Wrong%'])
    for dataset in DATASETS:
        for model in MODELS:
            d = fa_data[(dataset, model)]
            n = d['N']
            if n == 0: continue
            fc_rate = (d['FF']/(d['FF']+d['UF'])*100) if (d['FF']+d['UF'])>0 else 0
            fw_rate = (d['FW']/(d['FW']+d['UW'])*100) if (d['FW']+d['UW'])>0 else 0
            w.writerow([NICE_DATASET[dataset], NICE_MODEL[model], n,
                        f"{d['FF']/n*100:.1f}", f"{d['FW']/n*100:.1f}",
                        f"{d['UF']/n*100:.1f}", f"{d['UW']/n*100:.1f}",
                        f"{fc_rate:.1f}", f"{fw_rate:.1f}"])

# 5c summary CSV
with open(f'{OUT_DIR}/5c_model_size.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Dataset','LLaMA-3-3B_Faith%','LLaMA-3-8B_Faith%','Delta_pp'])
    for i, dataset in enumerate(DATASETS):
        w.writerow([NICE_DATASET[dataset], f"{bars3b[i]:.2f}", f"{bars8b[i]:.2f}",
                    f"{bars8b[i]-bars3b[i]:+.2f}"])

# 5d summary CSV
with open(f'{OUT_DIR}/5d_dataset_difficulty.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Dataset','Mean_CoT_Accuracy%','Mean_Faithfulness%','Acc_Rank','Faith_Rank'])
    for dataset in DATASETS:
        w.writerow([NICE_DATASET[dataset],
                    f"{dataset_stats[dataset]['mean_accuracy']:.2f}",
                    f"{dataset_stats[dataset]['mean_faithfulness']:.2f}",
                    acc_ranked.index(dataset)+1,
                    faith_ranked.index(dataset)+1])

print("\nCSV summaries saved.")
