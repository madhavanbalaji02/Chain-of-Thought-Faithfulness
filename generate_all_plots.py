"""
generate_all_plots.py
=====================
Generate every possible plot from the available data for the research paper.
Outputs to paper_figures/generated/
"""

import os, json, glob, re, csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from collections import defaultdict, Counter
from scipy.stats import pearsonr, spearmanr, pointbiserialr, mannwhitneyu

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
RESULTS_DIR = 'final_results'
FT_DIR = 'finetuned_results'
ANALYSIS_DIR = 'analysis'
OUT_DIR = 'paper_figures/generated'
os.makedirs(OUT_DIR, exist_ok=True)

LETTERS = ['A', 'B', 'C', 'D', 'E']
EPOCHS = [str(i) for i in range(6)]

MODELS = ['LLaMA-3', 'LLaMA-3-3B', 'Mistral-2', 'Phi-3']
DATASETS = ['arc-challenge', 'openbook', 'sports', 'sqa']

MODEL_NICE = {
    'LLaMA-3': 'LLaMA-3-8B', 'LLaMA-3-3B': 'LLaMA-3-3B',
    'Mistral-2': 'Mistral-7B', 'Phi-3': 'Phi-3',
}
DATASET_NICE = {
    'arc-challenge': 'ARC-Challenge', 'openbook': 'OpenBookQA',
    'sports': 'Sports', 'sqa': 'StrategyQA',
}
MODEL_COLORS = {
    'LLaMA-3': '#d62728', 'LLaMA-3-3B': '#ff7f0e',
    'Mistral-2': '#2ca02c', 'Phi-3': '#1f77b4',
}
MODEL_MARKERS = {
    'LLaMA-3': 'o', 'LLaMA-3-3B': 's',
    'Mistral-2': 'D', 'Phi-3': '^',
}
DATASET_COLORS = {
    'arc-challenge': '#1f77b4', 'openbook': '#ff7f0e',
    'sports': '#2ca02c', 'sqa': '#d62728',
}

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def normalize(probs):
    s = sum(probs)
    if s < 1e-30:
        return [1.0/len(probs)]*len(probs)
    return [p/s for p in probs]

def count_sentences(text):
    text = text.strip()
    if not text: return 1
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    for s in sentences:
        sub = re.split(r'\n\d+\.\s+', s)
        result.extend(sub)
    return max(len([s for s in result if s.strip()]), 1)

def flip_occurred(unlearning_results):
    preds = [r['prediction'] for _, r in sorted(unlearning_results.items(), key=lambda x: int(x[0]))]
    return any(p != preds[0] for p in preds[1:])

def best_epoch(ul):
    best_e, best_val = '1', float('inf')
    for e in EPOCHS[1:]:
        if e not in ul: continue
        val = ul[e]['cot_step_prob']
        if isinstance(val, list): val = val[0]
        if val < best_val:
            best_val = val
            best_e = e
    return best_e

def savefig(fig, name):
    for fmt in ['png', 'pdf']:
        fig.savefig(f'{OUT_DIR}/{name}.{fmt}')
    plt.close(fig)
    print(f"  Saved: {OUT_DIR}/{name}.png/pdf")


# ──────────────────────────────────────────────
# LOAD ALL NPO DATA
# ──────────────────────────────────────────────
print("Loading NPO unlearning results...")
all_records = []
for fpath in sorted(glob.glob(f'{RESULTS_DIR}/**/*.out', recursive=True)):
    parts = fpath.replace('\\','/').split('/')
    dataset = parts[-3]
    model = parts[-2]
    with open(fpath) as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            all_records.append({'_raw': rec, 'model': model, 'dataset': dataset})

print(f"Loaded {len(all_records)} NPO records.")

# Build main DataFrame
rows = []
for entry in all_records:
    rec = entry['_raw']
    model, dataset = entry['model'], entry['dataset']
    ul = rec['unlearning_results']
    correct_idx = LETTERS.index(rec['correct'])
    p0 = normalize(rec['initial_probs'])
    be = best_epoch(ul)
    p_be = normalize(ul[be]['probs'])

    initial_correct_prob = p0[correct_idx] if correct_idx < len(p0) else np.nan
    post_correct_prob = p_be[correct_idx] if correct_idx < len(p_be) else np.nan
    delta_p = initial_correct_prob - post_correct_prob

    baseline_pred = ul['0']['prediction']
    binary_faithful = any(ul[e]['prediction'] != baseline_pred for e in EPOCHS[1:] if e in ul)

    n_steps = count_sentences(rec.get('initial_cot', ''))
    initial_pred_correct = (rec['prediction'] == correct_idx)

    csp0 = ul['0']['cot_step_prob']
    if isinstance(csp0, list): csp0 = csp0[0]

    rows.append({
        'model': model, 'dataset': dataset,
        'binary_faithful': binary_faithful, 'delta_p': delta_p,
        'initial_correct_prob': initial_correct_prob,
        'post_correct_prob': post_correct_prob,
        'initial_prediction_correct': initial_pred_correct,
        'n_steps': n_steps,
        'baseline_cot_step_prob': csp0,
    })

df = pd.DataFrame(rows)

# Build epoch-level DataFrame
epoch_rows = []
for entry in all_records:
    rec = entry['_raw']
    model, dataset = entry['model'], entry['dataset']
    ul = rec['unlearning_results']
    correct_idx = LETTERS.index(rec['correct'])
    p0 = normalize(rec['initial_probs'])

    for e in EPOCHS:
        if e not in ul: continue
        ep = ul[e]
        p = normalize(ep['probs'])
        csp0 = ul['0']['cot_step_prob']
        if isinstance(csp0, list): csp0 = csp0[0]
        cspe = ep['cot_step_prob']
        if isinstance(cspe, list): cspe = cspe[0]
        eff = csp0 - cspe

        base_preds = ul['0'].get('specificity_preds', [])
        curr_preds = ep.get('specificity_preds', [])
        if base_preds and curr_preds:
            n = min(len(base_preds), len(curr_preds))
            spec = sum(b == c for b, c in zip(base_preds[:n], curr_preds[:n])) / n if n > 0 else np.nan
        else:
            spec = np.nan

        base_pred = ul['0']['prediction']
        flip = (ep['prediction'] != base_pred) if e != '0' else False
        dp = p0[correct_idx] - p[correct_idx] if correct_idx < len(p) else np.nan

        epoch_rows.append({
            'model': model, 'dataset': dataset,
            'epoch': int(e), 'efficacy': eff, 'specificity': spec,
            'flip': flip, 'delta_p': dp,
        })

edf = pd.DataFrame(epoch_rows)


# ──────────────────────────────────────────────
# LOAD FINETUNED RESULTS
# ──────────────────────────────────────────────
print("Loading finetuned results...")
ft_records = []
for cond_dir in sorted(os.listdir(FT_DIR)):
    fpath = os.path.join(FT_DIR, cond_dir, 'results.jsonl')
    if not os.path.isfile(fpath): continue
    with open(fpath) as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            rec['_dir'] = cond_dir
            ft_records.append(rec)

print(f"Loaded {len(ft_records)} finetuned records.")

# ══════════════════════════════════════════════
# PLOT 1: Per-epoch trajectories (all models overlaid)
# ══════════════════════════════════════════════
print("\n[1] Per-epoch trajectories (all models overlaid)...")
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
cols = ['efficacy', 'specificity', 'flip', 'delta_p']
titles = ['Mean Efficacy\n(CoT step prob drop)', 'Mean Specificity',
          'Mean Binary Faithfulness\n(flip rate)', 'Mean Continuous Δp']

for model in MODELS:
    sub = edf[edf['model'] == model]
    traj = sub.groupby('epoch')[cols].mean().reset_index()
    color = MODEL_COLORS[model]
    marker = MODEL_MARKERS[model]
    for ax, col in zip(axes, cols):
        ax.plot(traj['epoch'], traj[col], f'{marker}-', color=color,
                linewidth=2, markersize=7, label=MODEL_NICE[model])

for ax, title in zip(axes, titles):
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_xticks(range(6))
    ax.grid(True, alpha=0.3)

axes[1].axhline(0.95, color='gray', ls='--', lw=1, label='95% thresh')
axes[1].set_ylim(0.5, 1.02)
axes[2].set_ylim(0, 0.5)
axes[0].legend(fontsize=8, loc='best')

fig.suptitle('Per-Epoch Unlearning Trajectories — All Models', fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'plot01_epoch_trajectories_all_models')


# ══════════════════════════════════════════════
# PLOT 2: Delta-p distribution histograms per model
# ══════════════════════════════════════════════
print("[2] Delta-p distribution histograms...")
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
for ax, model in zip(axes, MODELS):
    sub = df[df['model'] == model].dropna(subset=['delta_p'])
    ax.hist(sub['delta_p'], bins=50, color=MODEL_COLORS[model], alpha=0.7, edgecolor='white')
    ax.axvline(0, color='black', ls='--', lw=1)
    ax.axvline(sub['delta_p'].mean(), color='red', ls='-', lw=1.5,
               label=f"mean={sub['delta_p'].mean():.4f}")
    ax.axvline(sub['delta_p'].median(), color='blue', ls='-', lw=1.5,
               label=f"med={sub['delta_p'].median():.4f}")
    ax.set_title(MODEL_NICE[model], fontweight='bold')
    ax.set_xlabel('Δp (correct answer prob drop)')
    ax.legend(fontsize=7)
    if ax == axes[0]: ax.set_ylabel('Count')

fig.suptitle('Distribution of Continuous Faithfulness (Δp) by Model', fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'plot02_delta_p_distributions')


# ══════════════════════════════════════════════
# PLOT 3: Faithfulness heatmap (NPO baseline, by model × dataset)
# ══════════════════════════════════════════════
print("[3] NPO baseline faithfulness heatmap...")
heat_data = np.zeros((len(MODELS), len(DATASETS)))
for i, model in enumerate(MODELS):
    for j, dataset in enumerate(DATASETS):
        sub = df[(df['model'] == model) & (df['dataset'] == dataset)]
        heat_data[i, j] = sub['binary_faithful'].mean() * 100

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(heat_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=60)
ax.set_xticks(range(len(DATASETS)))
ax.set_xticklabels([DATASET_NICE[d] for d in DATASETS], rotation=30, ha='right')
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels([MODEL_NICE[m] for m in MODELS])
for i in range(len(MODELS)):
    for j in range(len(DATASETS)):
        ax.text(j, i, f'{heat_data[i,j]:.1f}%', ha='center', va='center',
                color='black' if heat_data[i,j] > 20 else 'white', fontweight='bold')
plt.colorbar(im, ax=ax, label='Faithfulness (%)')
ax.set_title('Parametric Faithfulness (%) — NPO Baseline\n(4 models × 4 datasets)', fontweight='bold')
plt.tight_layout()
savefig(fig, 'plot03_npo_faithfulness_heatmap')


# ══════════════════════════════════════════════
# PLOT 4: Efficacy vs Specificity scatter (validity zone)
# ══════════════════════════════════════════════
print("[4] Efficacy-Specificity validity zone scatter...")
combo_stats = []
for model in MODELS:
    for dataset in DATASETS:
        sub_df = df[(df['model'] == model) & (df['dataset'] == dataset)]
        sub_e = edf[(edf['model'] == model) & (edf['dataset'] == dataset)]
        if len(sub_df) == 0: continue
        ep5 = sub_e[sub_e['epoch'] == 5]
        combo_stats.append({
            'model': model, 'dataset': dataset,
            'efficacy': ep5['efficacy'].mean(),
            'specificity': ep5['specificity'].mean(),
            'faithfulness': sub_df['binary_faithful'].mean(),
        })
cdf = pd.DataFrame(combo_stats)

fig, ax = plt.subplots(figsize=(8, 6))
for model in MODELS:
    sub = cdf[cdf['model'] == model]
    sc = ax.scatter(sub['efficacy'], sub['specificity'],
                    c=sub['faithfulness'].values, cmap='RdYlGn', vmin=0, vmax=0.5,
                    s=150, zorder=5, edgecolors=MODEL_COLORS[model], linewidths=2.5,
                    marker=MODEL_MARKERS[model])
    for _, row in sub.iterrows():
        ax.annotate(DATASET_NICE[row['dataset']][:3],
                    (row['efficacy'], row['specificity']),
                    textcoords='offset points', xytext=(6, 4), fontsize=8)

ax.axhline(0.95, color='green', ls='--', lw=1.5, label='Spec=95%')
ax.axhline(0.70, color='orange', ls='--', lw=1.5, label='Spec=70%')
ax.axhspan(0.70, 1.0, alpha=0.05, color='green')
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Binary Faithfulness Rate')
handles = [Line2D([0], [0], marker=MODEL_MARKERS[m], color=MODEL_COLORS[m],
                  linestyle='None', markersize=10, label=MODEL_NICE[m]) for m in MODELS]
ax.legend(handles=handles, loc='lower left', fontsize=9)
ax.set_xlabel('Mean Efficacy (cot_step_prob drop at epoch 5)')
ax.set_ylabel('Mean Specificity at epoch 5')
ax.set_title('Efficacy–Specificity Validity Zone\n(16 model×dataset combos, color=faithfulness)')
plt.tight_layout()
savefig(fig, 'plot04_validity_zone_scatter')


# ══════════════════════════════════════════════
# PLOT 5: Faithfulness × Accuracy cross-tabulation
# ══════════════════════════════════════════════
print("[5] Faithfulness × Accuracy cross-tab...")
cell_colors = {'FF': '#2ca02c', 'UF': '#98df8a', 'FW': '#d62728', 'UW': '#ffbb78'}
cell_labels = {
    'FF': 'Faithful + Correct', 'UF': 'Unfaithful + Correct',
    'FW': 'Faithful + Wrong', 'UW': 'Unfaithful + Wrong',
}

fa_data = {}
for dataset in DATASETS:
    for model in MODELS:
        sub = df[(df['model'] == model) & (df['dataset'] == dataset)]
        ff = fw = uf = uw = 0
        for _, r in sub.iterrows():
            correct = r['initial_prediction_correct']
            faithful = r['binary_faithful']
            if faithful and correct: ff += 1
            elif faithful and not correct: fw += 1
            elif not faithful and correct: uf += 1
            else: uw += 1
        n = ff + fw + uf + uw
        fa_data[(dataset, model)] = {'FF': ff, 'FW': fw, 'UF': uf, 'UW': uw, 'N': n}

fig, axs = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
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
    axs[i].set_title(DATASET_NICE[dataset], fontweight='bold')
    axs[i].set_xticks(range(len(MODELS)))
    axs[i].set_xticklabels([MODEL_NICE[m] for m in MODELS], rotation=30, ha='right', fontsize=8)
    axs[i].set_ylim(0, 100)
    if i == 0: axs[i].set_ylabel('% of instances')

fig.legend(loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.05))
fig.suptitle('Faithfulness × Accuracy Cross-Tabulation', fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'plot05_faithfulness_accuracy_crosstab')


# ══════════════════════════════════════════════
# PLOT 6: Model size comparison (LLaMA 3B vs 8B)
# ══════════════════════════════════════════════
print("[6] Model size comparison (LLaMA 3B vs 8B)...")
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(DATASETS))
w = 0.3
bars3b = [df[(df['model']=='LLaMA-3-3B') & (df['dataset']==d)]['binary_faithful'].mean()*100 for d in DATASETS]
bars8b = [df[(df['model']=='LLaMA-3') & (df['dataset']==d)]['binary_faithful'].mean()*100 for d in DATASETS]
ns3b = [len(df[(df['model']=='LLaMA-3-3B') & (df['dataset']==d)]) for d in DATASETS]
ns8b = [len(df[(df['model']=='LLaMA-3') & (df['dataset']==d)]) for d in DATASETS]

b1 = ax.bar(x - w/2, bars3b, w, label='LLaMA-3-3B (3B)', color='tab:orange', alpha=0.85)
b2 = ax.bar(x + w/2, bars8b, w, label='LLaMA-3-8B (8B)', color='tab:red', alpha=0.85)
for rect, n in zip(b1, ns3b):
    ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.5, f'n={n}', ha='center', fontsize=7)
for rect, n in zip(b2, ns8b):
    ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.5, f'n={n}', ha='center', fontsize=7)
for i in range(len(DATASETS)):
    delta = bars8b[i] - bars3b[i]
    y = max(bars3b[i], bars8b[i]) + 5
    ax.annotate(f'Δ={delta:+.1f}pp', xy=(x[i], y), ha='center', fontsize=9, color='black')

ax.set_xticks(x)
ax.set_xticklabels([DATASET_NICE[d] for d in DATASETS])
ax.set_ylabel('Faithfulness (%)')
ax.set_title('Model Size vs Faithfulness: LLaMA-3 3B vs 8B', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
savefig(fig, 'plot06_model_size_comparison')


# ══════════════════════════════════════════════
# PLOT 7: CoT length vs faithfulness
# ══════════════════════════════════════════════
print("[7] CoT length vs faithfulness...")
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, dataset in enumerate(DATASETS):
    length_faithful = defaultdict(list)
    for model in MODELS:
        sub = df[(df['model']==model) & (df['dataset']==dataset)]
        for _, r in sub.iterrows():
            length_faithful[r['n_steps']].append(r['binary_faithful'])
    lengths = sorted(k for k in length_faithful if len(length_faithful[k]) >= 5)
    faith_rates = [np.mean(length_faithful[l]) * 100 for l in lengths]
    ns = [len(length_faithful[l]) for l in lengths]
    axes[i].bar(lengths, faith_rates, color='tomato', alpha=0.85)
    axes[i].set_title(DATASET_NICE[dataset], fontweight='bold')
    axes[i].set_xlabel('CoT sentences')
    axes[i].set_ylim(0, 100)
    if i == 0: axes[i].set_ylabel('Faithfulness (%)')
    for l, r, n in zip(lengths, faith_rates, ns):
        if n >= 5:
            axes[i].text(l, r + 1, f'n={n}', ha='center', fontsize=7)

fig.suptitle('First-Step Faithfulness Rate by CoT Length', fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'plot07_cot_length_vs_faithfulness')


# ══════════════════════════════════════════════
# PLOT 8: Subcritical faithfulness
# ══════════════════════════════════════════════
print("[8] Subcritical faithfulness breakdown...")
fig, ax = plt.subplots(figsize=(8, 5))
for i, model in enumerate(MODELS):
    sub = df[df['model'] == model].dropna(subset=['delta_p'])
    n = len(sub)
    pct_st = 100 * ((sub['delta_p'] > 0) & (sub['binary_faithful'])).mean()
    pct_sc = 100 * ((sub['delta_p'] > 0) & (~sub['binary_faithful'])).mean()
    pct_ml = 100 * (sub['delta_p'] < 0).mean()
    pct_z  = 100 * (sub['delta_p'] == 0).mean()

    ax.bar(i, pct_st, 0.5, color=MODEL_COLORS[model], alpha=0.9,
           label='Suprathreshold (Δp>0, flip)' if i == 0 else '')
    ax.bar(i, pct_sc, 0.5, bottom=pct_st, color=MODEL_COLORS[model], alpha=0.4,
           hatch='//', label='Subcritical (Δp>0, no flip)' if i == 0 else '')
    ax.bar(i, pct_ml, 0.5, bottom=pct_st+pct_sc, color='#999999', alpha=0.6,
           label='Misleading (Δp<0)' if i == 0 else '')
    ax.text(i, pct_st + pct_sc + pct_ml + 1, f'{pct_st+pct_sc:.0f}%\nΔp>0',
            ha='center', va='bottom', fontsize=8)

ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels([MODEL_NICE[m] for m in MODELS])
ax.set_ylabel('% of instances')
ax.set_title('Subcritical Faithfulness: Binary Metric Undercounts Causal Influence', fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.set_ylim(0, 110)
plt.tight_layout()
savefig(fig, 'plot08_subcritical_faithfulness')


# ══════════════════════════════════════════════
# PLOT 9: Specificity distribution boxplots
# ══════════════════════════════════════════════
print("[9] Specificity distribution boxplots...")
fig, ax = plt.subplots(figsize=(10, 5))
spec_data = []
spec_labels = []
positions = []
pos = 0
for model in MODELS:
    for dataset in DATASETS:
        sub = edf[(edf['model'] == model) & (edf['dataset'] == dataset) & (edf['epoch'] == 5)]
        spec_vals = sub['specificity'].dropna().values
        if len(spec_vals) > 0:
            spec_data.append(spec_vals)
            spec_labels.append(f"{MODEL_NICE[model][:6]}\n{DATASET_NICE[dataset][:4]}")
            positions.append(pos)
        pos += 1
    pos += 0.5

bp = ax.boxplot(spec_data, positions=positions, widths=0.6, patch_artist=True)
for i, (patch, model_idx) in enumerate(zip(bp['boxes'], [m for m in MODELS for _ in DATASETS])):
    patch.set_facecolor(MODEL_COLORS[model_idx])
    patch.set_alpha(0.6)
ax.set_xticks(positions)
ax.set_xticklabels(spec_labels, fontsize=7, rotation=45, ha='right')
ax.axhline(0.95, color='green', ls='--', lw=1, label='95% threshold')
ax.axhline(0.70, color='orange', ls='--', lw=1, label='70% threshold')
ax.set_ylabel('Specificity at Epoch 5')
ax.set_title('Specificity Distribution (epoch 5) by Model × Dataset', fontweight='bold')
ax.legend(fontsize=8)
plt.tight_layout()
savefig(fig, 'plot09_specificity_boxplots')


# ══════════════════════════════════════════════
# PLOT 10: CoT step probability decay curves
# ══════════════════════════════════════════════
print("[10] CoT step probability decay curves...")
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, dataset in enumerate(DATASETS):
    for model in MODELS:
        sub = edf[(edf['model'] == model) & (edf['dataset'] == dataset)]
        traj = sub.groupby('epoch')['efficacy'].mean().reset_index()
        axes[i].plot(traj['epoch'], traj['efficacy'], f'{MODEL_MARKERS[model]}-',
                     color=MODEL_COLORS[model], linewidth=2, markersize=6, label=MODEL_NICE[model])
    axes[i].set_title(DATASET_NICE[dataset], fontweight='bold')
    axes[i].set_xlabel('Epoch')
    axes[i].set_xticks(range(6))
    axes[i].grid(True, alpha=0.3)
    if i == 0:
        axes[i].set_ylabel('Mean Efficacy\n(cot_step_prob drop)')
        axes[i].legend(fontsize=7)

fig.suptitle('Unlearning Efficacy Per Epoch by Dataset', fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'plot10_efficacy_curves_by_dataset')


# ══════════════════════════════════════════════
# PLOT 11: Instance-level scatter (initial vs post prob)
# ══════════════════════════════════════════════
print("[11] Instance-level initial vs post correct prob scatter...")
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, model in zip(axes, MODELS):
    sub = df[df['model'] == model].dropna(subset=['delta_p'])
    faithful = sub[sub['binary_faithful']]
    unfaithful = sub[~sub['binary_faithful']]
    ax.scatter(unfaithful['initial_correct_prob'], unfaithful['post_correct_prob'],
               alpha=0.3, s=15, color='gray', label=f'Unfaithful (n={len(unfaithful)})')
    ax.scatter(faithful['initial_correct_prob'], faithful['post_correct_prob'],
               alpha=0.5, s=20, color=MODEL_COLORS[model], label=f'Faithful (n={len(faithful)})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
    ax.set_title(MODEL_NICE[model], fontweight='bold')
    ax.set_xlabel('Initial P(correct)')
    if ax == axes[0]: ax.set_ylabel('Post-unlearning P(correct)')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_aspect('equal')

fig.suptitle('Instance-Level Probability Shift: Before vs After Unlearning', fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'plot11_instance_prob_scatter')


# ══════════════════════════════════════════════
# PLOT 12: Dataset difficulty vs faithfulness
# ══════════════════════════════════════════════
print("[12] Dataset difficulty vs faithfulness scatter...")
dataset_stats = {}
for dataset in DATASETS:
    faith_vals, acc_vals = [], []
    for model in MODELS:
        sub = df[(df['model'] == model) & (df['dataset'] == dataset)]
        acc_vals.append(sub['initial_prediction_correct'].mean() * 100)
        faith_vals.append(sub['binary_faithful'].mean() * 100)
    dataset_stats[dataset] = {
        'mean_accuracy': np.mean(acc_vals), 'mean_faithfulness': np.mean(faith_vals),
    }

fig, ax = plt.subplots(figsize=(7, 5))
for i, dataset in enumerate(DATASETS):
    x_val = dataset_stats[dataset]['mean_accuracy']
    y_val = dataset_stats[dataset]['mean_faithfulness']
    ax.scatter([x_val], [y_val], s=200, color=DATASET_COLORS[dataset],
               zorder=5, label=DATASET_NICE[dataset])
    ax.annotate(DATASET_NICE[dataset], (x_val, y_val),
                textcoords='offset points', xytext=(8, 5), fontsize=10)

acc_scores = [dataset_stats[d]['mean_accuracy'] for d in DATASETS]
faith_scores = [dataset_stats[d]['mean_faithfulness'] for d in DATASETS]
rho, pval = spearmanr(acc_scores, faith_scores)
z = np.polyfit(acc_scores, faith_scores, 1)
xs = np.linspace(min(acc_scores)-5, max(acc_scores)+5, 100)
ax.plot(xs, np.poly1d(z)(xs), 'k--', alpha=0.5)
ax.set_xlabel('Mean CoT Accuracy (%) — across models')
ax.set_ylabel('Mean Faithfulness (%) — across models')
ax.set_title(f'Dataset Difficulty vs Faithfulness\n(Spearman ρ={rho:.2f}, p={pval:.3f})', fontweight='bold')
ax.grid(alpha=0.4)
ax.legend(fontsize=9)
plt.tight_layout()
savefig(fig, 'plot12_dataset_difficulty_vs_faithfulness')


# ══════════════════════════════════════════════
# PLOT 13: Faithfulness quartile analysis
# ══════════════════════════════════════════════
print("[13] Faithfulness quartiles × accuracy...")
df_q = df.dropna(subset=['delta_p']).copy()

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for ax, model in zip(axes, MODELS):
    sub = df_q[df_q['model'] == model].copy()
    if len(sub) < 20: continue
    sub['quartile'] = pd.qcut(sub['delta_p'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    grouped = sub.groupby('quartile', observed=True).agg(
        accuracy=('initial_prediction_correct', 'mean'),
        mean_dp=('delta_p', 'mean'),
        n=('delta_p', 'count'),
    ).reset_index()
    x = np.arange(len(grouped))
    bars = ax.bar(x, grouped['accuracy'], color=MODEL_COLORS[model], alpha=0.7, width=0.6)
    ax2 = ax.twinx()
    ax2.plot(x, grouped['mean_dp'], 'k--o', linewidth=1.5, markersize=5, label='Mean Δp')
    ax.set_xticks(x)
    ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    ax.set_xlabel('Faithfulness Quartile (by Δp)')
    ax.set_ylabel('Accuracy Rate', color=MODEL_COLORS[model])
    ax2.set_ylabel('Mean Δp')
    ax.set_title(MODEL_NICE[model], fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color='gray', ls=':', lw=0.8)
    ax2.legend(fontsize=8)
    for rect, n in zip(bars, grouped['n']):
        ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.01,
                f'n={n}', ha='center', fontsize=7)

fig.suptitle('Accuracy by Continuous Faithfulness Quartile\n(bars=accuracy, dashed=mean Δp)', fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'plot13_faithfulness_quartile_accuracy')


# ══════════════════════════════════════════════
# PLOT 14: Per-dataset per-model faithfulness bars
# ══════════════════════════════════════════════
print("[14] Per-dataset per-model faithfulness bars...")
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(DATASETS))
width = 0.18
for i, model in enumerate(MODELS):
    vals = [df[(df['model']==model) & (df['dataset']==d)]['binary_faithful'].mean()*100 for d in DATASETS]
    bars = ax.bar(x + i*width - 1.5*width, vals, width, label=MODEL_NICE[model],
                  color=MODEL_COLORS[model], alpha=0.85)
    for bar, val in zip(bars, vals):
        if val > 2:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{val:.0f}', ha='center', fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels([DATASET_NICE[d] for d in DATASETS])
ax.set_ylabel('Faithfulness (%)')
ax.set_title('Parametric Faithfulness by Dataset and Model (NPO Baseline)', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
savefig(fig, 'plot14_faithfulness_by_dataset_model')


# ══════════════════════════════════════════════
# PLOT 15: All-model FT comparison (baseline/high/low)
# ══════════════════════════════════════════════
print("[15] All-model fine-tuning comparison...")

ft_model_conditions = {
    'LLaMA-3-3B': {'baseline': 'baseline', 'high': 'high_quality', 'low': 'low_quality'},
    'Phi-3': {'baseline': 'phi3_baseline', 'high': 'phi3_high', 'low': 'phi3_low'},
    'Mistral-7B': {'baseline': 'mistral_baseline', 'high': 'mistral_high', 'low': 'mistral_low'},
    'LLaMA-3-8B': {'baseline': 'llama8b_baseline', 'high': 'llama8b_high', 'low': 'llama8b_low'},
}

ft_faith = {}
for model_nice, conds in ft_model_conditions.items():
    ft_faith[model_nice] = {}
    for cond_label, cond_dir in conds.items():
        fpath = os.path.join(FT_DIR, cond_dir, 'results.jsonl')
        if not os.path.isfile(fpath): continue
        records = [json.loads(l) for l in open(fpath) if l.strip()]
        flips = 0
        for rec in records:
            er = rec['epoch_results']
            baseline_pred = er['0']['prediction']
            if any(er[e]['prediction'] != baseline_pred for e in ['1','2','3','4','5'] if e in er):
                flips += 1
        ft_faith[model_nice][cond_label] = flips / len(records) * 100 if records else 0

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(ft_model_conditions))
width = 0.25
cond_colors = {'baseline': '#4878CF', 'high': '#d62728', 'low': '#2ca02c'}
cond_labels = {'baseline': 'Baseline', 'high': 'High-Quality FT', 'low': 'Low-Quality FT'}

for i, (cond, color) in enumerate(cond_colors.items()):
    vals = [ft_faith[m].get(cond, 0) for m in ft_model_conditions.keys()]
    bars = ax.bar(x + (i-1)*width, vals, width, label=cond_labels[cond], color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{val:.0f}%', ha='center', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(ft_model_conditions.keys())
ax.set_ylabel('Faithfulness (%)')
ax.set_title('Faithfulness Across All Fine-Tuning Conditions\n(All 4 Models)', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
savefig(fig, 'plot15_all_model_ft_comparison')


# ══════════════════════════════════════════════
# PLOT 16: Delta-p distributions under fine-tuning (bimodal)
# ══════════════════════════════════════════════
print("[16] Delta-p distributions under fine-tuning...")
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (model_nice, conds) in zip(axes, ft_model_conditions.items()):
    for cond_label, cond_dir in conds.items():
        fpath = os.path.join(FT_DIR, cond_dir, 'results.jsonl')
        if not os.path.isfile(fpath): continue
        records = [json.loads(l) for l in open(fpath) if l.strip()]
        dps = []
        for rec in records:
            er = rec['epoch_results']
            for e in ['1', '2', '3', '4', '5']:
                if e in er and 'delta_p' in er[e]:
                    dps.append(er[e]['delta_p'])
        if dps:
            ax.hist(dps, bins=30, alpha=0.5, color=cond_colors[cond_label],
                    label=f'{cond_labels[cond_label]} (mean={np.mean(dps):.4f})', density=True)
    ax.set_title(model_nice, fontweight='bold')
    ax.set_xlabel('Δp')
    ax.axvline(0, color='black', ls='--', lw=0.8)
    ax.legend(fontsize=6)
    if ax == axes[0]: ax.set_ylabel('Density')

fig.suptitle('Δp Distribution Under Fine-Tuning Conditions\n(Bimodal Pattern in High-Quality FT)', fontweight='bold', y=1.04)
plt.tight_layout()
savefig(fig, 'plot16_ft_delta_p_distributions')


# ══════════════════════════════════════════════
# PLOT 17: Correctness conditioning
# ══════════════════════════════════════════════
print("[17] Correctness conditioning visualization...")
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for ax, (model_nice, conds) in zip(axes, ft_model_conditions.items()):
    cond_data = {}
    for cond_label, cond_dir in conds.items():
        fpath = os.path.join(FT_DIR, cond_dir, 'results.jsonl')
        if not os.path.isfile(fpath): continue
        records = [json.loads(l) for l in open(fpath) if l.strip()]
        correct_faith, incorrect_faith = [], []
        for rec in records:
            er = rec['epoch_results']
            baseline_pred = er['0']['prediction']
            flipped = any(er[e]['prediction'] != baseline_pred for e in ['1','2','3','4','5'] if e in er)
            if rec['initial_prediction_correct']:
                correct_faith.append(flipped)
            else:
                incorrect_faith.append(flipped)
        cond_data[cond_label] = {
            'correct': np.mean(correct_faith)*100 if correct_faith else 0,
            'incorrect': np.mean(incorrect_faith)*100 if incorrect_faith else 0,
            'n_correct': len(correct_faith),
            'n_incorrect': len(incorrect_faith),
        }

    x = np.arange(len(cond_data))
    w = 0.35
    correct_vals = [cond_data[c]['correct'] for c in cond_data]
    incorrect_vals = [cond_data[c]['incorrect'] for c in cond_data]
    b1 = ax.bar(x - w/2, correct_vals, w, label='Initially Correct', color='#2ca02c', alpha=0.8)
    b2 = ax.bar(x + w/2, incorrect_vals, w, label='Initially Incorrect', color='#d62728', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([cond_labels[c] for c in cond_data], fontsize=7, rotation=15, ha='right')
    ax.set_title(model_nice, fontweight='bold')
    ax.set_ylim(0, 80)
    if ax == axes[0]:
        ax.set_ylabel('Faithfulness (%)')
        ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('Faithfulness Conditioned on Prediction Correctness\n(Incorrect predictions are consistently more faithful)', fontweight='bold', y=1.04)
plt.tight_layout()
savefig(fig, 'plot17_correctness_conditioning')


# ══════════════════════════════════════════════
# PLOT 18: Per-dataset FT collapse
# ══════════════════════════════════════════════
print("[18] Per-dataset fine-tuning collapse...")
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for ax, (model_nice, conds) in zip(axes, ft_model_conditions.items()):
    dataset_ft = defaultdict(lambda: defaultdict(list))
    for cond_label, cond_dir in conds.items():
        fpath = os.path.join(FT_DIR, cond_dir, 'results.jsonl')
        if not os.path.isfile(fpath): continue
        records = [json.loads(l) for l in open(fpath) if l.strip()]
        for rec in records:
            er = rec['epoch_results']
            baseline_pred = er['0']['prediction']
            flipped = any(er[e]['prediction'] != baseline_pred for e in ['1','2','3','4','5'] if e in er)
            dataset_ft[rec['dataset']][cond_label].append(flipped)

    datasets_present = sorted(dataset_ft.keys())
    x = np.arange(len(datasets_present))
    width = 0.25
    for i, (cond, color) in enumerate(cond_colors.items()):
        vals = [np.mean(dataset_ft[d].get(cond, [0]))*100 for d in datasets_present]
        ax.bar(x + (i-1)*width, vals, width, color=color, alpha=0.85,
               label=cond_labels[cond] if ax == axes[0] else '_')
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_NICE.get(d, d)[:6] for d in datasets_present], fontsize=8, rotation=30, ha='right')
    ax.set_title(model_nice, fontweight='bold')
    ax.set_ylim(0, 80)
    if ax == axes[0]:
        ax.set_ylabel('Faithfulness (%)')
        ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('Per-Dataset Faithfulness Collapse Under Fine-Tuning', fontweight='bold', y=1.04)
plt.tight_layout()
savefig(fig, 'plot18_per_dataset_ft_collapse')


# ══════════════════════════════════════════════
# PLOT 19: Model-specific vs cross-model comparison (Phi-3)
# ══════════════════════════════════════════════
print("[19] Model-specific vs cross-model comparison (Phi-3)...")
phi3_conditions = {
    'Cross-model\nBaseline': 'phi3_baseline',
    'Cross-model\nHigh-FT': 'phi3_high',
    'Cross-model\nLow-FT': 'phi3_low',
    'Model-specific\nHigh-FT': 'phi3_ms_high',
    'Model-specific\nLow-FT': 'phi3_ms_low',
}

fig, ax = plt.subplots(figsize=(9, 5))
vals = []
colors_list = ['#4878CF', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
for (label, cond_dir), color in zip(phi3_conditions.items(), colors_list):
    fpath = os.path.join(FT_DIR, cond_dir, 'results.jsonl')
    if not os.path.isfile(fpath):
        vals.append(0)
        continue
    records = [json.loads(l) for l in open(fpath) if l.strip()]
    flips = 0
    for rec in records:
        er = rec['epoch_results']
        baseline_pred = er['0']['prediction']
        if any(er[e]['prediction'] != baseline_pred for e in ['1','2','3','4','5'] if e in er):
            flips += 1
    vals.append(flips / len(records) * 100 if records else 0)

bars = ax.bar(range(len(vals)), vals, color=colors_list, alpha=0.85)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(range(len(phi3_conditions)))
ax.set_xticklabels(phi3_conditions.keys(), fontsize=9)
ax.set_ylabel('Faithfulness (%)')
ax.set_title('Phi-3: Cross-Model vs Model-Specific Fine-Tuning\n(Ruling out distribution shift)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
savefig(fig, 'plot19_phi3_crossmodel_vs_specific')


# ══════════════════════════════════════════════
# PLOT 20: Accuracy-faithfulness tradeoff scatter
# ══════════════════════════════════════════════
print("[20] Accuracy-faithfulness tradeoff scatter...")
fig, ax = plt.subplots(figsize=(8, 6))
for model_nice, conds in ft_model_conditions.items():
    acc_vals, faith_vals = [], []
    cond_names = []
    for cond_label, cond_dir in conds.items():
        fpath = os.path.join(FT_DIR, cond_dir, 'results.jsonl')
        if not os.path.isfile(fpath): continue
        records = [json.loads(l) for l in open(fpath) if l.strip()]
        acc = np.mean([r['initial_prediction_correct'] for r in records]) * 100
        flips = sum(1 for r in records if any(
            r['epoch_results'][e]['prediction'] != r['epoch_results']['0']['prediction']
            for e in ['1','2','3','4','5'] if e in r['epoch_results']
        ))
        faith = flips / len(records) * 100 if records else 0
        acc_vals.append(acc)
        faith_vals.append(faith)
        cond_names.append(cond_label)

    model_key = [k for k, v in MODEL_NICE.items() if v == model_nice]
    if not model_key:
        model_key = [k for k in MODELS if MODEL_NICE.get(k, k) == model_nice]
    color = MODEL_COLORS.get(model_key[0], 'gray') if model_key else 'gray'
    marker = MODEL_MARKERS.get(model_key[0], 'o') if model_key else 'o'

    ax.scatter(acc_vals, faith_vals, color=color, marker=marker, s=120, zorder=5, label=model_nice)
    for acc, faith, cond in zip(acc_vals, faith_vals, cond_names):
        ax.annotate(cond[:4], (acc, faith), textcoords='offset points', xytext=(5, 5), fontsize=7)
    if len(acc_vals) >= 2:
        ax.plot(acc_vals, faith_vals, color=color, alpha=0.3, lw=1)

ax.set_xlabel('Task Accuracy (%)')
ax.set_ylabel('Parametric Faithfulness (%)')
ax.set_title('Accuracy vs Faithfulness Tradeoff\n(arrows show effect of fine-tuning)', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
savefig(fig, 'plot20_accuracy_faithfulness_tradeoff')


# ══════════════════════════════════════════════
# PLOT 21: Quality score distribution (faithful vs non-faithful)
# ══════════════════════════════════════════════
print("[21] Quality score analysis...")
quality_csv = os.path.join(ANALYSIS_DIR, 'cot_quality_scores.csv')
if os.path.isfile(quality_csv):
    qdf = pd.read_csv(quality_csv)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: Distribution of composite quality scores
    ax = axes[0]
    ax.hist(qdf['composite_score'], bins=20, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(qdf['composite_score'].mean(), color='red', ls='--', lw=1.5,
               label=f"mean={qdf['composite_score'].mean():.2f}")
    ax.set_xlabel('Composite Quality Score (1-5)')
    ax.set_ylabel('Count')
    ax.set_title('Quality Score Distribution', fontweight='bold')
    ax.legend(fontsize=8)

    # Panel 2: Quality by correctness
    ax = axes[1]
    correct = qdf[qdf['initial_prediction_correct'] == True]['composite_score']
    incorrect = qdf[qdf['initial_prediction_correct'] == False]['composite_score']
    bp = ax.boxplot([correct, incorrect], labels=['Correct', 'Incorrect'],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#2ca02c')
    bp['boxes'][1].set_facecolor('#d62728')
    for box in bp['boxes']:
        box.set_alpha(0.6)
    ax.set_ylabel('Composite Quality Score')
    ax.set_title('Quality by Prediction Correctness', fontweight='bold')

    # Panel 3: Quality components
    ax = axes[2]
    components = ['coherence', 'plausibility', 'completeness']
    comp_means = [qdf[c].mean() for c in components]
    bars = ax.bar(components, comp_means, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    for bar, val in zip(bars, comp_means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f'{val:.2f}', ha='center', fontsize=10)
    ax.set_ylabel('Mean Score (1-5)')
    ax.set_title('Quality Components', fontweight='bold')
    ax.set_ylim(0, 5.5)

    fig.suptitle('CoT Quality Score Analysis (N=918)', fontweight='bold', y=1.02)
    plt.tight_layout()
    savefig(fig, 'plot21_quality_score_analysis')
else:
    print("  [SKIP] quality scores CSV not found")


# ══════════════════════════════════════════════
# PLOT 22: FF2 probe results
# ══════════════════════════════════════════════
print("[22] FF2 probe results visualization...")
ff2_json = os.path.join(ANALYSIS_DIR, 'ff2_probe_results.json')
if os.path.isfile(ff2_json):
    ff2 = json.load(open(ff2_json))
    layers = [8, 16, 24]
    conditions = ['Baseline', 'High_FT', 'Low_FT']
    cond_nice = {'Baseline': 'Baseline', 'High_FT': 'High-Quality FT', 'Low_FT': 'Low-Quality FT'}
    cond_colors_ff2 = {'Baseline': '#888888', 'High_FT': '#d62728', 'Low_FT': '#1f77b4'}

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(layers))
    width = 0.25
    for i, cond in enumerate(conditions):
        vals = [ff2[f'layer{l}_{cond}']['lift_over_majority'] for l in layers]
        bars = ax.bar(x + (i-1)*width, vals, width, label=cond_nice[cond],
                      color=cond_colors_ff2[cond], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height() + (0.003 if val >= 0 else -0.01),
                    f'{val:.3f}', ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'Layer {l}' for l in layers])
    ax.set_ylabel('Probe Accuracy Lift over Majority Baseline')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_title('FF2 Linear Probe: Faithfulness Encoding by Layer\n(High-FT collapses encoding at all layers)', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    savefig(fig, 'plot22_ff2_probe_results')
else:
    print("  [SKIP] FF2 probe results not found")


# ══════════════════════════════════════════════
# PLOT 23: REVEAL validation
# ══════════════════════════════════════════════
print("[23] REVEAL validation visualization...")
reveal_json = os.path.join(ANALYSIS_DIR, 'reveal_validation.json')
if os.path.isfile(reveal_json):
    rv = json.load(open(reveal_json))
    lc = rv['logical_correctness_vs_dp']
    attr = rv['attribution_majority_vs_dp']

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Panel 1: Logical correctness vs delta_p
    ax = axes[0]
    categories = ['Logically\nCorrect', 'Logically\nIncorrect']
    means = [lc['mean_dp_correct'], lc['mean_dp_incorrect']]
    ns = [lc['n_correct'], lc['n_incorrect']]
    colors_rv = ['#2ca02c', '#d62728']
    bars = ax.bar(categories, means, color=colors_rv, alpha=0.8)
    for bar, val, n in zip(bars, means, ns):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height() + (0.005 if val >= 0 else -0.015),
                f'{val:.3f}\n(n={n})', ha='center', fontsize=9)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('Mean Δp')
    pval_str = f'{lc["mannwhitney_p"]:.2e}'
    ax.set_title(f'Logical Correctness vs Δp\n(p = {pval_str})', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: Attribution vs delta_p
    ax = axes[1]
    ax.text(0.5, 0.6, f'Attribution vs Δp\nSpearman ρ = {attr["spearman_r"]:.3f}\np = {attr["p"]:.3f}',
            ha='center', va='center', transform=ax.transAxes, fontsize=14)
    ax.text(0.5, 0.3, 'Not significant\n(surface attribution ≠ causal influence)',
            ha='center', va='center', transform=ax.transAxes, fontsize=11, color='gray')
    ax.set_title('Evidence Attribution vs Δp', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    fig.suptitle(f'REVEAL External Validation (N={rv["n_matched_rows"]} steps, {rv["n_matched_questions"]} questions)',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    savefig(fig, 'plot23_reveal_validation')
else:
    print("  [SKIP] REVEAL validation not found")


# ══════════════════════════════════════════════
# PLOT 24: CoT length distribution
# ══════════════════════════════════════════════
print("[24] CoT length distribution...")
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=False)
for i, dataset in enumerate(DATASETS):
    all_lengths = df[df['dataset'] == dataset]['n_steps'].values
    c = Counter(all_lengths)
    lengths = sorted(c.keys())
    counts = [c[l] for l in lengths]
    axes[i].bar(lengths, counts, color='steelblue', alpha=0.8)
    axes[i].set_title(DATASET_NICE[dataset], fontweight='bold')
    axes[i].set_xlabel('CoT sentences')
    if i == 0: axes[i].set_ylabel('Count')
    mean_l = np.mean(all_lengths)
    axes[i].axvline(mean_l, color='red', ls='--', label=f'mean={mean_l:.1f}')
    axes[i].legend(fontsize=8)

fig.suptitle('CoT Length Distribution by Dataset (all models pooled)', fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'plot24_cot_length_distribution')


# ══════════════════════════════════════════════
# PLOT 25: Faithfulness by model (aggregated violin/bar)
# ══════════════════════════════════════════════
print("[25] Aggregated faithfulness by model...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Binary faithfulness
ax = axes[0]
model_faith = [df[df['model']==m]['binary_faithful'].mean()*100 for m in MODELS]
bars = ax.bar(range(len(MODELS)), model_faith,
              color=[MODEL_COLORS[m] for m in MODELS], alpha=0.85)
for bar, val in zip(bars, model_faith):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels([MODEL_NICE[m] for m in MODELS])
ax.set_ylabel('Binary Faithfulness (%)')
ax.set_title('Binary Faithfulness by Model', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Panel 2: Continuous delta_p
ax = axes[1]
model_dp = [df[df['model']==m]['delta_p'].mean() for m in MODELS]
bars = ax.bar(range(len(MODELS)), model_dp,
              color=[MODEL_COLORS[m] for m in MODELS], alpha=0.85)
for bar, val in zip(bars, model_dp):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height() + (0.001 if val >= 0 else -0.002),
            f'{val:.4f}', ha='center', fontsize=9)
ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels([MODEL_NICE[m] for m in MODELS])
ax.set_ylabel('Mean Δp')
ax.axhline(0, color='black', lw=0.8)
ax.set_title('Mean Continuous Faithfulness (Δp) by Model', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

fig.suptitle('Aggregated Faithfulness Measures', fontweight='bold', y=1.02)
plt.tight_layout()
savefig(fig, 'plot25_aggregated_faithfulness_by_model')


# ══════════════════════════════════════════════
# PLOT 26: Correlation scatter — binary faith vs delta_p
# ══════════════════════════════════════════════
print("[26] Binary vs continuous faithfulness correlation...")
fig, ax = plt.subplots(figsize=(7, 5))
for model in MODELS:
    sub = df[df['model'] == model].dropna(subset=['delta_p'])
    jitter = np.random.normal(0, 0.02, len(sub))
    faith_jittered = sub['binary_faithful'].astype(int) + jitter
    ax.scatter(sub['delta_p'], faith_jittered, alpha=0.15, s=15,
               color=MODEL_COLORS[model], label=MODEL_NICE[model])

r_all, p_all = pointbiserialr(df['binary_faithful'].astype(int), df['delta_p'].fillna(0))
ax.set_xlabel('Continuous Faithfulness (Δp)')
ax.set_ylabel('Binary Faithfulness (jittered)')
ax.set_title(f'Binary vs Continuous Faithfulness\n(r={r_all:.3f}, p={p_all:.2e})', fontweight='bold')
ax.legend(fontsize=8)
ax.axvline(0, color='black', ls='--', lw=0.8)
ax.grid(alpha=0.3)
plt.tight_layout()
savefig(fig, 'plot26_binary_vs_continuous_faithfulness')


# ══════════════════════════════════════════════
# PLOT 27: Faithfulness by prediction correctness (NPO data)
# ══════════════════════════════════════════════
print("[27] Faithfulness by prediction correctness (NPO)...")
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(MODELS))
width = 0.3

correct_faith = []
incorrect_faith = []
for model in MODELS:
    sub = df[df['model'] == model]
    c = sub[sub['initial_prediction_correct']]['binary_faithful'].mean() * 100
    ic = sub[~sub['initial_prediction_correct']]['binary_faithful'].mean() * 100
    correct_faith.append(c)
    incorrect_faith.append(ic)

b1 = ax.bar(x - width/2, correct_faith, width, label='Initially Correct', color='#2ca02c', alpha=0.8)
b2 = ax.bar(x + width/2, incorrect_faith, width, label='Initially Incorrect', color='#d62728', alpha=0.8)
for bars in [b1, b2]:
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{bar.get_height():.1f}%', ha='center', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([MODEL_NICE[m] for m in MODELS])
ax.set_ylabel('Faithfulness (%)')
ax.set_title('Faithfulness by Initial Prediction Correctness (NPO Baseline)\n(Incorrect predictions → more faithful CoTs)', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
savefig(fig, 'plot27_correctness_faithfulness_npo')


# ══════════════════════════════════════════════
# PLOT 28: Per-model per-dataset per-epoch detail grids
# ══════════════════════════════════════════════
print("[28] Per-model epoch detail grids...")
for model in MODELS:
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    metrics = ['efficacy', 'specificity', 'flip', 'delta_p']
    metric_titles = ['Efficacy', 'Specificity', 'Binary Faith', 'Δp']

    for j, dataset in enumerate(DATASETS):
        sub = edf[(edf['model'] == model) & (edf['dataset'] == dataset)]
        traj = sub.groupby('epoch')[metrics].mean().reset_index()
        for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[i][j]
            ax.plot(traj['epoch'], traj[metric], 'o-', color=MODEL_COLORS[model], lw=2, markersize=5)
            ax.set_xticks(range(6))
            ax.grid(True, alpha=0.3)
            if i == 0: ax.set_title(DATASET_NICE[dataset], fontweight='bold')
            if j == 0: ax.set_ylabel(title)
            if i == 3: ax.set_xlabel('Epoch')
            if metric == 'specificity':
                ax.axhline(0.95, color='gray', ls='--', lw=0.8)
                ax.set_ylim(0.5, 1.02)

    fig.suptitle(f'{MODEL_NICE[model]} — Per-Epoch Detail by Dataset', fontweight='bold', y=1.01)
    plt.tight_layout()
    safe = model.replace('-', '_')
    savefig(fig, f'plot28_detail_grid_{safe}')


# ══════════════════════════════════════════════
# PLOT 29: Faithfulness distribution (violin per dataset)
# ══════════════════════════════════════════════
print("[29] Δp violin plots by dataset...")
fig, ax = plt.subplots(figsize=(10, 5))
data_for_violin = []
positions = []
colors_violin = []
labels = []
pos = 0
for model in MODELS:
    for dataset in DATASETS:
        sub = df[(df['model']==model) & (df['dataset']==dataset)].dropna(subset=['delta_p'])
        if len(sub) > 0:
            data_for_violin.append(sub['delta_p'].values)
            positions.append(pos)
            colors_violin.append(MODEL_COLORS[model])
            labels.append(f"{MODEL_NICE[model][:6]}\n{DATASET_NICE[dataset][:4]}")
        pos += 1
    pos += 0.5

parts = ax.violinplot(data_for_violin, positions=positions, widths=0.7, showmeans=True, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors_violin[i])
    pc.set_alpha(0.6)

ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
ax.axhline(0, color='black', ls='--', lw=0.8)
ax.set_ylabel('Δp')
ax.set_title('Distribution of Δp by Model × Dataset', fontweight='bold')
plt.tight_layout()
savefig(fig, 'plot29_delta_p_violins')


# ══════════════════════════════════════════════
# PLOT 30: Summary dashboard
# ══════════════════════════════════════════════
print("[30] Summary dashboard...")
fig = plt.figure(figsize=(16, 10))

# Panel A: Faithfulness heatmap
ax1 = fig.add_subplot(2, 3, 1)
im = ax1.imshow(heat_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=60)
ax1.set_xticks(range(len(DATASETS)))
ax1.set_xticklabels([DATASET_NICE[d][:6] for d in DATASETS], fontsize=8, rotation=30, ha='right')
ax1.set_yticks(range(len(MODELS)))
ax1.set_yticklabels([MODEL_NICE[m] for m in MODELS], fontsize=8)
for i in range(len(MODELS)):
    for j in range(len(DATASETS)):
        ax1.text(j, i, f'{heat_data[i,j]:.0f}', ha='center', va='center', fontsize=9, fontweight='bold')
ax1.set_title('(A) Faithfulness %', fontweight='bold', fontsize=10)

# Panel B: Model comparison bar
ax2 = fig.add_subplot(2, 3, 2)
ax2.bar(range(len(MODELS)), model_faith, color=[MODEL_COLORS[m] for m in MODELS], alpha=0.85)
ax2.set_xticks(range(len(MODELS)))
ax2.set_xticklabels([MODEL_NICE[m][:8] for m in MODELS], fontsize=8)
ax2.set_ylabel('Faith %')
ax2.set_title('(B) By Model', fontweight='bold', fontsize=10)

# Panel C: Epoch trajectories
ax3 = fig.add_subplot(2, 3, 3)
for model in MODELS:
    sub = edf[edf['model'] == model]
    traj = sub.groupby('epoch')['flip'].mean().reset_index()
    ax3.plot(traj['epoch'], traj['flip'], f'{MODEL_MARKERS[model]}-',
             color=MODEL_COLORS[model], lw=2, markersize=5, label=MODEL_NICE[model][:8])
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Flip Rate')
ax3.set_title('(C) Epoch Trajectories', fontweight='bold', fontsize=10)
ax3.legend(fontsize=6)
ax3.set_xticks(range(6))

# Panel D: FT comparison
ax4 = fig.add_subplot(2, 3, 4)
for i, (cond, color) in enumerate(cond_colors.items()):
    vals = [ft_faith[m].get(cond, 0) for m in ft_model_conditions.keys()]
    ax4.bar(np.arange(4) + (i-1)*0.25, vals, 0.25, label=cond_labels[cond], color=color, alpha=0.85)
ax4.set_xticks(range(4))
ax4.set_xticklabels(list(ft_model_conditions.keys()), fontsize=7)
ax4.set_ylabel('Faith %')
ax4.set_title('(D) FT Conditions', fontweight='bold', fontsize=10)
ax4.legend(fontsize=6)

# Panel E: Correctness conditioning
ax5 = fig.add_subplot(2, 3, 5)
ax5.bar(np.arange(4)-0.15, correct_faith, 0.3, label='Correct', color='#2ca02c', alpha=0.8)
ax5.bar(np.arange(4)+0.15, incorrect_faith, 0.3, label='Incorrect', color='#d62728', alpha=0.8)
ax5.set_xticks(range(4))
ax5.set_xticklabels([MODEL_NICE[m][:8] for m in MODELS], fontsize=8)
ax5.set_ylabel('Faith %')
ax5.set_title('(E) By Correctness', fontweight='bold', fontsize=10)
ax5.legend(fontsize=7)

# Panel F: Subcritical
ax6 = fig.add_subplot(2, 3, 6)
for i, model in enumerate(MODELS):
    sub = df[df['model'] == model].dropna(subset=['delta_p'])
    pct_st = 100 * ((sub['delta_p'] > 0) & (sub['binary_faithful'])).mean()
    pct_sc = 100 * ((sub['delta_p'] > 0) & (~sub['binary_faithful'])).mean()
    pct_ml = 100 * (sub['delta_p'] < 0).mean()
    ax6.bar(i, pct_st, 0.5, color=MODEL_COLORS[model], alpha=0.9)
    ax6.bar(i, pct_sc, 0.5, bottom=pct_st, color=MODEL_COLORS[model], alpha=0.4, hatch='//')
    ax6.bar(i, pct_ml, 0.5, bottom=pct_st+pct_sc, color='#999999', alpha=0.6)
ax6.set_xticks(range(len(MODELS)))
ax6.set_xticklabels([MODEL_NICE[m][:8] for m in MODELS], fontsize=8)
ax6.set_ylabel('%')
ax6.set_title('(F) Subcritical', fontweight='bold', fontsize=10)

fig.suptitle('Faithfulness Research — Complete Dashboard', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
savefig(fig, 'plot30_summary_dashboard')


# ══════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"ALL PLOTS GENERATED in {OUT_DIR}/")
print("=" * 60)
for f in sorted(os.listdir(OUT_DIR)):
    fsize = os.path.getsize(f'{OUT_DIR}/{f}') / 1024
    print(f"  {f:55s} {fsize:7.1f} KB")
