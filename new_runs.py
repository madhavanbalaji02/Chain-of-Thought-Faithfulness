"""
new_runs.py
===========
Generates sbatch commands for the 16 new LR-sweep runs needed to make
Phi-3 and Mistral-7B results comparable to the original paper.

Results go to lr_sweep_results/{dataset}/{model}/ (separate from final_results/).

Usage:
    python new_runs.py          # print all 16 sbatch commands
    python new_runs.py | bash   # run directly on BigRed200
"""

import os

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

SMALL_JOB = 'ul_step_pos_ff2.job'       # 1× A100, 32GB RAM, 12h  (Phi-3)
BIG_JOB   = 'ul_step_pos_ff2_L2.job'    # 2× A100, 64GB RAM, 24h  (Mistral-7B)

# Output root — separate from final_results/ so we can compare
OUTPUT_ROOT = 'lr_sweep_results'

# New runs needed:
#   Phi-3:      arc/openbook at lr=1e-04 (paper best), all 4 at lr=5e-05 (intermediate)
#   Mistral-7B: all 4 at lr=5e-06 (paper best), all 4 at lr=1e-05 (intermediate)
NEW_RUNS = [
    # (model_hf_id, short_name, dataset, lr, job_script)
    # ── Phi-3 at lr=1e-04 (paper's best LR for arc/openbook) ──
    ('microsoft/Phi-3-mini-4k-instruct', 'Phi-3', 'arc-challenge', 1e-04, SMALL_JOB),
    ('microsoft/Phi-3-mini-4k-instruct', 'Phi-3', 'openbook',      1e-04, SMALL_JOB),
    # ── Phi-3 at lr=5e-05 (paper's best LR for sports/sqa, also intermediate for arc/openbook) ──
    ('microsoft/Phi-3-mini-4k-instruct', 'Phi-3', 'arc-challenge', 5e-05, SMALL_JOB),
    ('microsoft/Phi-3-mini-4k-instruct', 'Phi-3', 'openbook',      5e-05, SMALL_JOB),
    ('microsoft/Phi-3-mini-4k-instruct', 'Phi-3', 'sports',        5e-05, SMALL_JOB),
    ('microsoft/Phi-3-mini-4k-instruct', 'Phi-3', 'sqa',           5e-05, SMALL_JOB),
    # ── Phi-3 at lr=1e-04 for sports/sqa (full sweep) ──
    ('microsoft/Phi-3-mini-4k-instruct', 'Phi-3', 'sports',        1e-04, SMALL_JOB),
    ('microsoft/Phi-3-mini-4k-instruct', 'Phi-3', 'sqa',           1e-04, SMALL_JOB),
    # ── Mistral-7B at lr=5e-06 (paper's best LR) ──
    ('mistralai/Mistral-7B-Instruct-v0.2', 'Mistral-2', 'arc-challenge', 5e-06, BIG_JOB),
    ('mistralai/Mistral-7B-Instruct-v0.2', 'Mistral-2', 'openbook',      5e-06, BIG_JOB),
    ('mistralai/Mistral-7B-Instruct-v0.2', 'Mistral-2', 'sports',        5e-06, BIG_JOB),
    ('mistralai/Mistral-7B-Instruct-v0.2', 'Mistral-2', 'sqa',           5e-06, BIG_JOB),
    # ── Mistral-7B at lr=1e-05 (intermediate) ──
    ('mistralai/Mistral-7B-Instruct-v0.2', 'Mistral-2', 'arc-challenge', 1e-05, BIG_JOB),
    ('mistralai/Mistral-7B-Instruct-v0.2', 'Mistral-2', 'openbook',      1e-05, BIG_JOB),
    ('mistralai/Mistral-7B-Instruct-v0.2', 'Mistral-2', 'sports',        1e-05, BIG_JOB),
    ('mistralai/Mistral-7B-Instruct-v0.2', 'Mistral-2', 'sqa',           1e-05, BIG_JOB),
]

METHOD  = 'npo_KL'
SEED    = 1001

def lr_str(lr):
    """Format LR as a clean string for job names and filenames."""
    # e.g. 1e-04, 5e-05, 5e-06
    return f'{lr:.0e}'.replace('+0', '').replace('-0', '-')

def make_sbatch(model_id, short, dataset, lr, job_script):
    lr_s     = lr_str(lr)
    job_name = f'{short}-{dataset}-lr{lr_s}'
    # The job script expects positional args: model dataset lr method
    # Same interface as the original ul_step_pos_ff2.job
    # We pass OUTPUT_ROOT as an extra env var so the job writes to lr_sweep_results/
    cmd = (
        f'sbatch '
        f'--job-name={job_name} '
        f'--export=ALL,OUTPUT_ROOT={OUTPUT_ROOT} '
        f'{job_script} '
        f'{model_id} {dataset} {lr_s} {METHOD}'
    )
    return cmd

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

print("# ══════════════════════════════════════════")
print("# NEW LR-SWEEP RUNS — 16 sbatch commands")
print("# Run from: /N/scratch/madbala/parametric-faithfulness_run")
print("# Results go to: lr_sweep_results/{dataset}/{model}/")
print("# ══════════════════════════════════════════")
print()
print("# ── Phi-3 at lr=1e-04 (paper best for arc/openbook) ──")
for (model_id, short, dataset, lr, job) in NEW_RUNS:
    if short == 'Phi-3' and abs(lr - 1e-04) < 1e-20:
        print(make_sbatch(model_id, short, dataset, lr, job))

print()
print("# ── Phi-3 at lr=5e-05 (paper best for sports/sqa, intermediate for arc/openbook) ──")
for (model_id, short, dataset, lr, job) in NEW_RUNS:
    if short == 'Phi-3' and abs(lr - 5e-05) < 1e-20:
        print(make_sbatch(model_id, short, dataset, lr, job))

print()
print("# ── Mistral-7B at lr=5e-06 (paper best for all datasets) ──")
for (model_id, short, dataset, lr, job) in NEW_RUNS:
    if short == 'Mistral-2' and abs(lr - 5e-06) < 1e-20:
        print(make_sbatch(model_id, short, dataset, lr, job))

print()
print("# ── Mistral-7B at lr=1e-05 (intermediate) ──")
for (model_id, short, dataset, lr, job) in NEW_RUNS:
    if short == 'Mistral-2' and abs(lr - 1e-05) < 1e-20:
        print(make_sbatch(model_id, short, dataset, lr, job))

print()
print(f"# Total: {len(NEW_RUNS)} new runs")
print(f"# Note: job scripts must support OUTPUT_ROOT env var.")
print(f"# If they don't, add to each .job script:")
print(f"#   OUTPUT_ROOT=${{OUTPUT_ROOT:-final_results}}")
print(f"# and pass --output_root $OUTPUT_ROOT to unlearn.py")
