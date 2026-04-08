"""
Figure 1: The Alignment-Faithfulness Paradox
Grouped bar chart: baseline, high-FT, low-FT for LLaMA-3-3B and Phi-3.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Data ─────────────────────────────────────────────────────────────────────
models = ['LLaMA-3-3B', 'Phi-3']
baseline = [30.0, 32.0]
high_ft  = [8.0,  14.0]
low_ft   = [28.0, 36.0]
rel_drop_high = ['-73%', '-56%']

# ── Layout ───────────────────────────────────────────────────────────────────
x = np.arange(len(models))
w = 0.25
fig, ax = plt.subplots(figsize=(5.5, 3.8))

# ── Bars ─────────────────────────────────────────────────────────────────────
bars_base = ax.bar(x - w, baseline, w, color='#999999', label='Baseline',      zorder=3)
bars_high = ax.bar(x,     high_ft,  w, color='#d62728', label='High-quality FT', zorder=3)
bars_low  = ax.bar(x + w, low_ft,   w, color='#1f77b4', label='Low-quality FT',  zorder=3)

# ── Relative-drop annotations on red bars ────────────────────────────────────
for bar, label in zip(bars_high, rel_drop_high):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.8,
        label,
        ha='center', va='bottom',
        fontsize=8, fontweight='bold', color='#d62728'
    )

# ── Axes formatting ──────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel('Parametric Faithfulness (%)', fontsize=10)
ax.set_ylim(0, 50)
ax.set_yticks([0, 10, 20, 30, 40, 50])
ax.tick_params(axis='both', labelsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_axisbelow(False)
ax.yaxis.grid(False)
ax.xaxis.grid(False)

# ── Legend ───────────────────────────────────────────────────────────────────
ax.legend(fontsize=8.5, frameon=False, loc='upper right')

plt.tight_layout(pad=0.5)

# ── Save ─────────────────────────────────────────────────────────────────────
out_dir = os.path.dirname(os.path.abspath(__file__))
for ext in ('pdf', 'png'):
    path = os.path.join(out_dir, f'fig1_paradox.{ext}')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f'Saved: {path}')

plt.close()
