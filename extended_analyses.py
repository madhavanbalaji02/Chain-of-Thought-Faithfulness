"""
extended_analyses.py
====================
Parts 2–8 from the extended analysis request.

Key data facts discovered in Part 1:
  - initial_probs == unlearning_results["0"]["probs"] (epoch 0 = pre-training baseline)
  - Mistral-2 and Phi-3: probs are unnormalized raw likelihoods (not softmax).
    LLaMA models: probs sum to ~1.0. Must normalize ALL before computing delta_p.
  - step_idx = 0 for every record in all 16 files (only first CoT sentence unlearned)
  - No "segmented_cot" field; n_steps inferred by splitting initial_cot into sentences
  - cot_step_prob: list with one float (log-prob of the unlearned step token sequence)
    epoch 0 = baseline; grows more negative as unlearning proceeds
  - "best epoch" = epoch where cot_step_prob is most negative (maximal unlearning)
"""

import os, json, glob, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr, spearmanr, pointbiserialr

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
RESULTS_DIR = 'final_results'
ANALYSIS_DIR = 'analysis'
FIG_DIR = 'my_figures/new'
LETTERS = ['A', 'B', 'C', 'D', 'E']
EPOCHS = [str(i) for i in range(6)]   # '0' = baseline, '1'–'5' = after training

MODEL_COLORS = {
    'LLaMA-3':    '#4878CF',   # blue
    'LLaMA-3-3B': '#6AC4C4',   # teal
    'Mistral-2':  '#E07070',   # coral
    'Phi-3':      '#F0A830',   # amber
}
MODEL_NICE = {
    'LLaMA-3':    'LLaMA-3-8B',
    'LLaMA-3-3B': 'LLaMA-3-3B',
    'Mistral-2':  'Mistral-7B',
    'Phi-3':      'Phi-3',
}
DATASET_NICE = {
    'arc-challenge': 'ARC-Challenge',
    'openbook':      'OpenBookQA',
    'sports':        'Sports',
    'sqa':           'StrategyQA',
}

os.makedirs(ANALYSIS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def normalize(probs):
    """Softmax-normalize a probability vector. Safe against all-zero."""
    s = sum(probs)
    if s < 1e-30:
        n = len(probs)
        return [1.0/n]*n
    return [p/s for p in probs]

def count_sentences(text):
    """Count sentences in a CoT string using simple regex splitting."""
    text = text.strip()
    if not text:
        return 1
    # Split on period/exclamation/question followed by whitespace or end
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Also split on numbered list items like "1. " "2. "
    result = []
    for s in sentences:
        sub = re.split(r'\n\d+\.\s+', s)
        result.extend(sub)
    result = [s.strip() for s in result if s.strip()]
    return max(len(result), 1)

def best_epoch(unlearning_results):
    """Return the epoch (str) with the most negative cot_step_prob (highest efficacy)."""
    best_e, best_val = '1', float('inf')
    for e in EPOCHS[1:]:   # skip epoch 0 (baseline)
        if e not in unlearning_results:
            continue
        val = unlearning_results[e]['cot_step_prob']
        if isinstance(val, list):
            val = val[0]
        if val < best_val:
            best_val = val
            best_e = e
    return best_e

def efficacy_at_epoch(unlearning_results, epoch):
    """
    Efficacy = drop in cot_step_prob from baseline (epoch 0).
    More negative cot_step_prob = better unlearning. Returned as positive number.
    """
    baseline = unlearning_results['0']['cot_step_prob']
    if isinstance(baseline, list): baseline = baseline[0]
    val = unlearning_results[epoch]['cot_step_prob']
    if isinstance(val, list): val = val[0]
    return baseline - val   # positive if unlearning worked

def specificity_at_epoch(unlearning_results, epoch):
    """
    Fraction of specificity instances where prediction did NOT change from epoch 0.
    """
    if epoch not in unlearning_results:
        return np.nan
    base_preds = unlearning_results['0']['specificity_preds']
    curr_preds = unlearning_results[epoch]['specificity_preds']
    if not base_preds or not curr_preds:
        return np.nan
    n = min(len(base_preds), len(curr_preds))
    if n == 0:
        return np.nan
    unchanged = sum(b == c for b, c in zip(base_preds[:n], curr_preds[:n]))
    return unchanged / n

# ──────────────────────────────────────────────
# LOAD ALL DATA
# ──────────────────────────────────────────────

print("Loading all result files...")
all_records = []   # list of dicts, one per instance

for fpath in sorted(glob.glob(f'{RESULTS_DIR}/**/*.out', recursive=True)):
    parts = fpath.replace('\\','/').split('/')
    dataset = parts[-3]
    model   = parts[-2]
    with open(fpath) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            all_records.append({
                '_raw': rec,
                'model': model,
                'dataset': dataset,
                'fpath': fpath,
            })

print(f"Loaded {len(all_records)} total records across 16 files.")

# ══════════════════════════════════════════════
# PART 2 — Continuous faithfulness CSV
# ══════════════════════════════════════════════
print("\n" + "="*60)
print("PART 2 — Building continuous_faithfulness.csv")
print("="*60)

rows = []
for entry in all_records:
    rec     = entry['_raw']
    model   = entry['model']
    dataset = entry['dataset']

    ul = rec['unlearning_results']
    correct_letter = rec['correct']
    correct_idx    = LETTERS.index(correct_letter)

    # Baseline (epoch 0) normalized probs
    p0_raw  = rec['initial_probs']    # same as ul['0']['probs']
    p0      = normalize(p0_raw)

    # Best epoch for this instance
    be      = best_epoch(ul)
    p_be_raw = ul[be]['probs']
    p_be    = normalize(p_be_raw)

    # Continuous faithfulness
    initial_correct_prob = p0[correct_idx]  if correct_idx < len(p0)  else np.nan
    post_correct_prob    = p_be[correct_idx] if correct_idx < len(p_be) else np.nan
    delta_p              = initial_correct_prob - post_correct_prob

    # Binary faithfulness: did prediction change at ANY epoch 1–5?
    baseline_pred = ul['0']['prediction']
    binary_faithful = False
    for e in EPOCHS[1:]:
        if e in ul and ul[e]['prediction'] != baseline_pred:
            binary_faithful = True
            break

    # n_steps from CoT
    n_steps = count_sentences(rec.get('initial_cot', ''))

    # Agreement filter flag
    initial_pred_correct = (rec['prediction'] == correct_idx)

    # efficacy epoch as int
    eff_epoch_int = int(be)

    rows.append({
        'model':                     model,
        'dataset':                   dataset,
        'instance_id':               rec.get('id', ''),
        'step_idx':                  rec.get('step_idx', 0),
        'n_steps':                   n_steps,
        'step_position_normalized':  0.0,   # always 0 since step_idx=0 always
        'binary_faithful':           binary_faithful,
        'delta_p':                   delta_p,
        'initial_correct_prob':      initial_correct_prob,
        'post_correct_prob':         post_correct_prob,
        'initial_prediction_correct': initial_pred_correct,
        'lr':                        3e-5,
        'efficacy_epoch':            eff_epoch_int,
    })

df = pd.DataFrame(rows)
out_csv = f'{ANALYSIS_DIR}/continuous_faithfulness.csv'
df.to_csv(out_csv, index=False)
print(f"Saved {len(df)} rows to {out_csv}")

# ──────────────────────────────────────────────
# Part 2 stats
# ──────────────────────────────────────────────
print("\n--- a) Correlation: binary faithfulness vs continuous delta_p ---")
for model in sorted(df['model'].unique()):
    sub = df[df['model'] == model].dropna(subset=['delta_p'])
    if len(sub) < 10: continue
    r, p = pointbiserialr(sub['binary_faithful'].astype(int), sub['delta_p'])
    print(f"  {MODEL_NICE[model]:15s}: r={r:.3f}  p={p:.2e}  n={len(sub)}")

print()
r_all, p_all = pointbiserialr(df['binary_faithful'].astype(int), df['delta_p'].fillna(0))
print(f"  ALL MODELS:      r={r_all:.3f}  p={p_all:.2e}  n={len(df)}")

print("\n--- b) Distribution of delta_p per model ---")
print(f"{'Model':15s} {'Mean':>8} {'Std':>8} {'%Pos':>8} {'%Neg':>8} {'N':>6}")
print("-"*55)
for model in ['LLaMA-3', 'LLaMA-3-3B', 'Mistral-2', 'Phi-3']:
    sub = df[df['model'] == model].dropna(subset=['delta_p'])
    mean_dp = sub['delta_p'].mean()
    std_dp  = sub['delta_p'].std()
    pct_pos = (sub['delta_p'] > 0).mean() * 100
    pct_neg = (sub['delta_p'] < 0).mean() * 100
    print(f"  {MODEL_NICE[model]:13s} {mean_dp:>8.4f} {std_dp:>8.4f} {pct_pos:>7.1f}% {pct_neg:>7.1f}% {len(sub):>6}")

print("\n--- c) Subcritical faithfulness (delta_p>0, no flip) ---")
subcrit  = df[(df['delta_p'] > 0) & (~df['binary_faithful'])]
suprathresh = df[(df['delta_p'] > 0) & (df['binary_faithful'])]
misleading  = df[df['delta_p'] < 0]

print(f"  delta_p > 0 AND no flip (subcritical):   {len(subcrit):4d}  ({100*len(subcrit)/len(df):.1f}%)")
print(f"  delta_p > 0 AND flip (suprathreshold):   {len(suprathresh):4d}  ({100*len(suprathresh)/len(df):.1f}%)")
print(f"  delta_p < 0 (step boosted correct ans):  {len(misleading):4d}  ({100*len(misleading)/len(df):.1f}%)")
print()
print("  Per-model subcritical breakdown:")
for model in ['LLaMA-3', 'LLaMA-3-3B', 'Mistral-2', 'Phi-3']:
    sub = df[df['model']==model].dropna(subset=['delta_p'])
    sc  = (sub['delta_p'] > 0) & (~sub['binary_faithful'])
    print(f"    {MODEL_NICE[model]:13s}: subcritical={sc.sum():3d}/{len(sub)} ({100*sc.mean():.1f}%)")

# ══════════════════════════════════════════════
# PART 3 — Per-epoch trajectories
# ══════════════════════════════════════════════
print("\n" + "="*60)
print("PART 3 — Per-epoch trajectories + validity zone")
print("="*60)

# Build epoch-level dataframe
epoch_rows = []
for entry in all_records:
    rec     = entry['_raw']
    model   = entry['model']
    dataset = entry['dataset']
    ul      = rec['unlearning_results']
    correct_idx = LETTERS.index(rec['correct'])

    p0_raw = rec['initial_probs']
    p0     = normalize(p0_raw)

    for e in EPOCHS:
        if e not in ul: continue
        ep = ul[e]
        p_raw = ep['probs']
        p     = normalize(p_raw)

        # efficacy: drop in cot_step_prob vs epoch 0
        csp0 = ul['0']['cot_step_prob']
        if isinstance(csp0, list): csp0 = csp0[0]
        cspe = ep['cot_step_prob']
        if isinstance(cspe, list): cspe = cspe[0]
        eff = csp0 - cspe   # positive = more unlearned

        # specificity
        spec = specificity_at_epoch(ul, e)

        # binary flip at this epoch (vs epoch 0 baseline)
        base_pred = ul['0']['prediction']
        flip_at_e = (ep['prediction'] != base_pred) if e != '0' else False

        # delta_p at this epoch
        dp = p0[correct_idx] - p[correct_idx] if correct_idx < len(p) else np.nan

        epoch_rows.append({
            'model': model, 'dataset': dataset,
            'epoch': int(e),
            'efficacy': eff, 'specificity': spec,
            'flip': flip_at_e, 'delta_p': dp,
        })

edf = pd.DataFrame(epoch_rows)

# Plot 4 trajectories per model
for model in ['LLaMA-3', 'LLaMA-3-3B', 'Mistral-2', 'Phi-3']:
    sub = edf[edf['model'] == model]
    mean_traj = sub.groupby('epoch')[['efficacy','specificity','flip','delta_p']].mean().reset_index()

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    color = MODEL_COLORS[model]
    titles = ['Mean Efficacy (cot_step_prob drop)', 'Mean Specificity',
              'Mean Binary Faithfulness', 'Mean Continuous Δp']
    cols   = ['efficacy', 'specificity', 'flip', 'delta_p']

    for ax, col, title in zip(axes, cols, titles):
        ax.plot(mean_traj['epoch'], mean_traj[col], 'o-', color=color, linewidth=2, markersize=6)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Epoch')
        ax.set_xticks(range(6))
        ax.grid(True, alpha=0.3)
        if col == 'specificity':
            ax.axhline(0.95, color='gray', linestyle='--', linewidth=1, label='95% threshold')
            ax.axhline(0.70, color='lightgray', linestyle='--', linewidth=1, label='70% threshold')
            ax.legend(fontsize=7)
        if col in ('flip', 'specificity'):
            ax.set_ylim(0, 1.05)

    fig.suptitle(f'Per-Epoch Trajectories — {MODEL_NICE[model]} (lr=3e-05)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    safe_model = model.replace('-', '_').replace('.', '')
    out = f'{FIG_DIR}/lr_sensitivity_trajectories_{safe_model}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

    # Print peak faithfulness epoch
    peak_faith_epoch = mean_traj.loc[mean_traj['flip'].idxmax(), 'epoch']
    peak_spec_epoch  = mean_traj.loc[mean_traj['specificity'].idxmax(), 'epoch']
    print(f"  {MODEL_NICE[model]:13s}: peak faithfulness at epoch {peak_faith_epoch}, "
          f"peak specificity at epoch {peak_spec_epoch}")
    # Does faithfulness degrade after peak?
    faith_at_peak = mean_traj.loc[mean_traj['epoch']==peak_faith_epoch, 'flip'].values[0]
    faith_at_end  = mean_traj.loc[mean_traj['epoch']==5, 'flip'].values[0]
    print(f"    Faithfulness: epoch {peak_faith_epoch}={faith_at_peak:.3f} vs epoch 5={faith_at_end:.3f} "
          f"({'DEGRADES' if faith_at_end < faith_at_peak - 0.01 else 'stable'})")

# ──────────────────────────────────────────────
# Part 3 step 5: validity zone scatter
# ──────────────────────────────────────────────
print("\nBuilding validity zone scatter (16 model/dataset combos)...")

# Compute per-(model,dataset) averages at best epoch
combo_stats = []
for model in ['LLaMA-3', 'LLaMA-3-3B', 'Mistral-2', 'Phi-3']:
    for dataset in ['arc-challenge', 'openbook', 'sports', 'sqa']:
        sub_df = df[(df['model']==model) & (df['dataset']==dataset)].dropna(subset=['delta_p'])
        sub_e  = edf[(edf['model']==model) & (edf['dataset']==dataset)]
        if len(sub_df) == 0: continue
        # Use epoch 5 avg for efficacy and specificity (matches REPRODUCTION_REPORT)
        ep5 = sub_e[sub_e['epoch']==5]
        mean_eff  = ep5['efficacy'].mean()
        mean_spec = ep5['specificity'].mean()
        mean_faith= sub_df['binary_faithful'].mean()
        combo_stats.append({
            'model': model, 'dataset': dataset,
            'efficacy': mean_eff, 'specificity': mean_spec, 'faithfulness': mean_faith,
        })

cdf = pd.DataFrame(combo_stats)

fig, ax = plt.subplots(figsize=(7, 5))
for model in ['LLaMA-3', 'LLaMA-3-3B', 'Mistral-2', 'Phi-3']:
    sub = cdf[cdf['model']==model]
    sc = ax.scatter(sub['efficacy'], sub['specificity'],
                    c=sub['faithfulness'].values,
                    cmap='RdYlGn', vmin=0, vmax=1,
                    s=120, zorder=5, edgecolors=MODEL_COLORS[model], linewidths=2,
                    label=MODEL_NICE[model])
    for _, row in sub.iterrows():
        ax.annotate(DATASET_NICE[row['dataset']][:3],
                    (row['efficacy'], row['specificity']),
                    textcoords='offset points', xytext=(5, 3), fontsize=7)

ax.axhline(0.95, color='green',  linestyle='--', linewidth=1.5, label='Spec=95% (paper threshold)', zorder=3)
ax.axhline(0.70, color='orange', linestyle='--', linewidth=1.5, label='Spec=70% (validity floor)', zorder=3)
# Shade the validity zone
ax.axhspan(0.70, 1.0, alpha=0.05, color='green', label='Validity zone')

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Binary Faithfulness Rate', fontsize=9)

# Legend for model colors
handles = [mpatches.Patch(edgecolor=MODEL_COLORS[m], facecolor='white',
                          linewidth=2, label=MODEL_NICE[m])
           for m in ['LLaMA-3', 'LLaMA-3-3B', 'Mistral-2', 'Phi-3']]
leg1 = ax.legend(handles=handles, loc='lower left', fontsize=8, title='Model')
ax.add_artist(leg1)
ax.legend(loc='lower right', fontsize=8)

ax.set_xlabel('Mean Efficacy (cot_step_prob drop at epoch 5)', fontsize=10)
ax.set_ylabel('Mean Specificity at epoch 5', fontsize=10)
ax.set_title('Efficacy–Specificity Validity Zone\n(color = faithfulness rate, 16 model×dataset combos)', fontsize=10)
plt.tight_layout()
out = f'{FIG_DIR}/validity_zone.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

# ══════════════════════════════════════════════
# PART 4 — Faithfulness quartiles × accuracy
# ══════════════════════════════════════════════
print("\n" + "="*60)
print("PART 4 — Faithfulness quartiles × accuracy")
print("="*60)

df_q = df.dropna(subset=['delta_p']).copy()
# Do NOT pre-filter — compute accuracy within each quartile across all instances
df_q['quartile_all'] = pd.qcut(df_q['delta_p'], q=4,
                                labels=['Q1 (bottom 25%)', 'Q2 (25-50%)',
                                        'Q3 (50-75%)', 'Q4 (top 25%)'])

print("\nAll models combined:")
qtable = df_q.groupby('quartile_all', observed=True).agg(
    n=('initial_prediction_correct','count'),
    accuracy=('initial_prediction_correct','mean'),
    mean_delta_p=('delta_p','mean'),
    binary_faith_rate=('binary_faithful','mean'),
).reset_index()
print(qtable.to_string(index=False))

# Per-model quartile table
quart_rows = []
for model in ['LLaMA-3', 'LLaMA-3-3B', 'Mistral-2', 'Phi-3']:
    sub = df_q[df_q['model']==model].copy()
    if len(sub) < 20: continue
    sub['quartile'] = pd.qcut(sub['delta_p'], q=4,
                               labels=['Q1','Q2','Q3','Q4'])
    for q, grp in sub.groupby('quartile', observed=True):
        quart_rows.append({
            'model': model,
            'quartile': str(q),
            'n': len(grp),
            'accuracy': grp['initial_prediction_correct'].mean(),
            'mean_delta_p': grp['delta_p'].mean(),
            'binary_faith_rate': grp['binary_faithful'].mean(),
        })

qdf = pd.DataFrame(quart_rows)
qdf.to_csv(f'{ANALYSIS_DIR}/faithfulness_accuracy_quartiles.csv', index=False)
print(f"\nSaved: {ANALYSIS_DIR}/faithfulness_accuracy_quartiles.csv")

# Plot
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
for ax, model in zip(axes, ['LLaMA-3', 'LLaMA-3-3B', 'Mistral-2', 'Phi-3']):
    sub = qdf[qdf['model']==model]
    if sub.empty:
        ax.set_visible(False)
        continue
    color = MODEL_COLORS[model]
    x = np.arange(len(sub))
    bars = ax.bar(x, sub['accuracy'], color=color, alpha=0.7, width=0.6)
    ax2 = ax.twinx()
    ax2.plot(x, sub['mean_delta_p'], 'k--o', linewidth=1.5, markersize=5, label='Mean Δp')
    ax.set_xticks(x)
    ax.set_xticklabels(['Q1','Q2','Q3','Q4'], fontsize=9)
    ax.set_xlabel('Faithfulness Quartile (by Δp)', fontsize=9)
    ax.set_ylabel('Accuracy Rate', fontsize=9, color=color)
    ax2.set_ylabel('Mean Δp', fontsize=9)
    ax.set_title(MODEL_NICE[model], fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
    ax2.legend(fontsize=8)

fig.suptitle('Accuracy by Continuous Faithfulness Quartile\n'
             '(bars=accuracy, dashed=mean Δp)', fontsize=11)
plt.tight_layout()
out = f'{FIG_DIR}/continuous_faithful_accuracy.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

# Print the key finding
print("\nKey finding — is accuracy monotonically increasing with quartile?")
for model in ['LLaMA-3', 'LLaMA-3-3B', 'Mistral-2', 'Phi-3']:
    sub = qdf[qdf['model']==model]['accuracy'].values
    if len(sub) < 4: continue
    mono = all(sub[i] <= sub[i+1] for i in range(len(sub)-1))
    print(f"  {MODEL_NICE[model]:13s}: Q1={sub[0]:.3f} Q2={sub[1]:.3f} Q3={sub[2]:.3f} Q4={sub[3]:.3f}  monotone={mono}")

# ══════════════════════════════════════════════
# PART 5 — Step position with CoT length
# ══════════════════════════════════════════════
print("\n" + "="*60)
print("PART 5 — Step position (CoT length as proxy, since step_idx=0 always)")
print("="*60)
print("NOTE: All step_idx=0 — so 'position' analysis becomes: does CoT LENGTH")
print("      (n_steps) moderate first-step faithfulness magnitude?")
print()

# Bin by n_steps (CoT length)
df_pos = df.dropna(subset=['delta_p']).copy()
df_pos['length_bin'] = pd.cut(df_pos['n_steps'],
                               bins=[0, 2, 4, 7, 100],
                               labels=['1-2 sent', '3-4 sent', '5-7 sent', '8+ sent'])

print(f"CoT length bins:")
print(df_pos.groupby('length_bin', observed=True)[['delta_p','binary_faithful','n_steps']].agg(
    n=('delta_p','count'), mean_dp=('delta_p','mean'),
    mean_faith=('binary_faithful','mean'), mean_n=('n_steps','mean')
).to_string())

fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
for ax, model in zip(axes, ['LLaMA-3', 'LLaMA-3-3B', 'Mistral-2', 'Phi-3']):
    sub = df_pos[df_pos['model']==model]
    grouped = sub.groupby('length_bin', observed=True)[['delta_p','binary_faithful']].mean().reset_index()
    if grouped.empty:
        ax.set_visible(False)
        continue

    x = np.arange(len(grouped))
    color = MODEL_COLORS[model]
    ax.bar(x, grouped['binary_faithful'], color=color, alpha=0.5, width=0.4, label='Binary faithfulness')
    ax2 = ax.twinx()
    ax2.plot(x, grouped['delta_p'], 's--', color=color, linewidth=2, markersize=7, label='Mean Δp')
    ax2.axhline(0, color='gray', linewidth=0.8, linestyle=':')

    ax.set_xticks(x)
    ax.set_xticklabels(grouped['length_bin'].astype(str), fontsize=8, rotation=15)
    ax.set_xlabel('CoT Length (n sentences)', fontsize=9)
    ax.set_ylabel('Binary Faithfulness Rate', fontsize=9, color='gray')
    ax2.set_ylabel('Mean Δp', fontsize=9, color=color)
    ax.set_title(MODEL_NICE[model], fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right', fontsize=7)
    ax2.legend(loc='upper left', fontsize=7)

fig.suptitle('First-Step Faithfulness vs CoT Length\n'
             '(step_idx=0 in all data; position analysis uses CoT length as proxy)',
             fontsize=10)
plt.tight_layout()
out = f'{FIG_DIR}/step_position_continuous.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out}")

# ══════════════════════════════════════════════
# PART 6 — Subcritical faithfulness
# ══════════════════════════════════════════════
print("\n" + "="*60)
print("PART 6 — Subcritical faithfulness")
print("="*60)

sub_rows = []
for model in ['LLaMA-3', 'LLaMA-3-3B', 'Mistral-2', 'Phi-3']:
    sub = df[df['model']==model].dropna(subset=['delta_p'])
    n   = len(sub)
    subcrit   = sub[(sub['delta_p'] > 0) & (~sub['binary_faithful'])]
    suprath   = sub[(sub['delta_p'] > 0) & (sub['binary_faithful'])]
    mislead   = sub[sub['delta_p'] < 0]
    neutral   = sub[sub['delta_p'] == 0]

    print(f"\n{MODEL_NICE[model]} (n={n}):")
    print(f"  Subcritical   (Δp>0, no flip): {len(subcrit):4d} ({100*len(subcrit)/n:.1f}%)  mean_Δp={subcrit['delta_p'].mean():.4f}")
    print(f"  Suprathreshold(Δp>0, flip):     {len(suprath):4d} ({100*len(suprath)/n:.1f}%)  mean_Δp={suprath['delta_p'].mean():.4f}")
    print(f"  Misleading    (Δp<0):           {len(mislead):4d} ({100*len(mislead)/n:.1f}%)  mean_Δp={mislead['delta_p'].mean():.4f}")
    print(f"  Binary faith undercounts by: {len(subcrit):d} instances = "
          f"{100*len(subcrit)/(len(subcrit)+len(suprath)):.1f}% of 'faithful' signal missed")

    sub_rows.append({
        'model': MODEL_NICE[model],
        'n_total': n,
        'subcritical_n': len(subcrit), 'subcritical_pct': 100*len(subcrit)/n,
        'subcritical_mean_dp': subcrit['delta_p'].mean() if len(subcrit)>0 else np.nan,
        'suprathreshold_n': len(suprath), 'suprathreshold_pct': 100*len(suprath)/n,
        'suprathreshold_mean_dp': suprath['delta_p'].mean() if len(suprath)>0 else np.nan,
        'misleading_n': len(mislead), 'misleading_pct': 100*len(mislead)/n,
        'misleading_mean_dp': mislead['delta_p'].mean() if len(mislead)>0 else np.nan,
    })

sc_df = pd.DataFrame(sub_rows)
sc_df.to_csv(f'{ANALYSIS_DIR}/subcritical_faithfulness.csv', index=False)
print(f"\nSaved: {ANALYSIS_DIR}/subcritical_faithfulness.csv")

# Stacked bar chart
fig, ax = plt.subplots(figsize=(8, 5))
models_ordered = ['LLaMA-3', 'LLaMA-3-3B', 'Mistral-2', 'Phi-3']
x = np.arange(len(models_ordered))
width = 0.5

for i, model in enumerate(models_ordered):
    sub = df[df['model']==model].dropna(subset=['delta_p'])
    n   = len(sub)
    pct_sc = 100*((sub['delta_p']>0)&(~sub['binary_faithful'])).mean()
    pct_st = 100*((sub['delta_p']>0)&(sub['binary_faithful'])).mean()
    pct_ml = 100*(sub['delta_p']<0).mean()

    ax.bar(i, pct_st, width, color=MODEL_COLORS[model], alpha=0.9, label='Suprathreshold (Δp>0, flip)' if i==0 else '')
    ax.bar(i, pct_sc, width, bottom=pct_st, color=MODEL_COLORS[model], alpha=0.4,
           label='Subcritical (Δp>0, no flip)' if i==0 else '', hatch='//')
    ax.bar(i, pct_ml, width, bottom=pct_st+pct_sc, color='#999999', alpha=0.6,
           label='Misleading (Δp<0)' if i==0 else '')

ax.set_xticks(x)
ax.set_xticklabels([MODEL_NICE[m] for m in models_ordered], fontsize=10)
ax.set_ylabel('% of instances', fontsize=11)
ax.set_ylim(0, 110)
ax.set_title('Subcritical Faithfulness: Binary Metric Undercounts Causal Influence\n'
             '(hatched = faithful in magnitude but below decision threshold)', fontsize=10)
ax.legend(fontsize=9, loc='upper right')
ax.axhline(50, color='gray', linestyle=':', linewidth=0.8)
for i, model in enumerate(models_ordered):
    sub = df[df['model']==model].dropna(subset=['delta_p'])
    pct_st = 100*((sub['delta_p']>0)&(sub['binary_faithful'])).mean()
    pct_sc = 100*((sub['delta_p']>0)&(~sub['binary_faithful'])).mean()
    ax.text(i, pct_st + pct_sc + 1, f"{pct_st+pct_sc:.0f}%\ntotal\nΔp>0",
            ha='center', va='bottom', fontsize=8)

plt.tight_layout()
out = f'{FIG_DIR}/subcritical_faithfulness.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

# ══════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════
print("\n" + "="*60)
print("All analyses complete.")
print(f"CSVs: {ANALYSIS_DIR}/")
for f in sorted(os.listdir(ANALYSIS_DIR)):
    fsize = os.path.getsize(f'{ANALYSIS_DIR}/{f}')
    print(f"  {f:<45} {fsize/1024:.1f} KB")
print(f"\nFigures: {FIG_DIR}/")
new_figs = [f for f in sorted(os.listdir(FIG_DIR)) if f.endswith('.png')]
for f in new_figs:
    fsize = os.path.getsize(f'{FIG_DIR}/{f}')
    print(f"  {f:<45} {fsize/1024:.1f} KB")
