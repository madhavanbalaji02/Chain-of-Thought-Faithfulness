"""
make_finetune_splits.py
=======================
Reads analysis/cot_quality_scores.csv (produced by score_cot_quality.py)
and writes:
  finetune_data/high_quality.jsonl  — top 50% by composite_score
  finetune_data/low_quality.jsonl   — bottom 50%
  finetune_data/test_held_out.jsonl — 50 random instances excluded from both splits
                                       (used by evaluate_finetuned.py)

Format per line:
  {"prompt": "Question: ...\nLet's think step by step.",
   "completion": "{initial_cot}\nThe answer is {correct_letter}."}

Run AFTER score_cot_quality.py has finished.
"""

import csv, json, os, random

SCORES_CSV   = 'analysis/cot_quality_scores.csv'
OUT_DIR      = 'finetune_data'
HELD_OUT_N   = 50
RANDOM_SEED  = 42

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)

# ── Load scores ──────────────────────────────
rows = []
with open(SCORES_CSV, newline='', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        r['composite_score'] = float(r['composite_score'])
        rows.append(r)

print(f"Loaded {len(rows)} scored instances.")

# ── Hold out 50 for evaluation (stratified across datasets) ──────────────────
from collections import defaultdict
by_ds = defaultdict(list)
for r in rows:
    by_ds[r['dataset']].append(r)

held_out = []
for ds, ds_rows in by_ds.items():
    n = max(1, round(HELD_OUT_N * len(ds_rows) / len(rows)))
    held_out.extend(random.sample(ds_rows, min(n, len(ds_rows))))
held_out = held_out[:HELD_OUT_N]  # trim if over
held_out_ids = {(r['dataset'], r['instance_id']) for r in held_out}

remaining = [r for r in rows if (r['dataset'], r['instance_id']) not in held_out_ids]
print(f"Held out {len(held_out)} for evaluation; {len(remaining)} for training splits.")

# ── Split remaining by composite_score median ───────────────────────────────
remaining_sorted = sorted(remaining, key=lambda r: r['composite_score'])
mid = len(remaining_sorted) // 2
low_quality  = remaining_sorted[:mid]
high_quality = remaining_sorted[mid:]

def format_record(r):
    return {
        'prompt':     f"Question: {r['question']}\nLet's think step by step.",
        'completion': f"{r['initial_cot']}\nThe answer is {r['correct']}.",
        # metadata fields kept for reference (fine-tuning script ignores them)
        '_instance_id':    r['instance_id'],
        '_dataset':        r['dataset'],
        '_composite_score': r['composite_score'],
        '_correct':        r['correct'],
    }

def write_jsonl(path, records):
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

write_jsonl(f'{OUT_DIR}/high_quality.jsonl',  [format_record(r) for r in high_quality])
write_jsonl(f'{OUT_DIR}/low_quality.jsonl',   [format_record(r) for r in low_quality])
write_jsonl(f'{OUT_DIR}/test_held_out.jsonl', [format_record(r) for r in held_out])

# ── Print statistics ──────────────────────────────────────────────────────────
import statistics as stat

hq_scores = [r['composite_score'] for r in high_quality]
lq_scores = [r['composite_score'] for r in low_quality]
ho_scores = [r['composite_score'] for r in held_out]

print(f"\n{'='*55}")
print(f"Dataset splits")
print(f"{'='*55}")
print(f"{'Split':<25} {'N':>5} {'Mean score':>12} {'Min':>6} {'Max':>6}")
print(f"{'-'*55}")
print(f"{'HIGH quality':<25} {len(high_quality):>5} {stat.mean(hq_scores):>12.3f} {min(hq_scores):>6.3f} {max(hq_scores):>6.3f}")
print(f"{'LOW quality':<25} {len(low_quality):>5} {stat.mean(lq_scores):>12.3f} {min(lq_scores):>6.3f} {max(lq_scores):>6.3f}")
print(f"{'Held-out test':<25} {len(held_out):>5} {stat.mean(ho_scores):>12.3f} {min(ho_scores):>6.3f} {max(ho_scores):>6.3f}")

print(f"\nFiles written:")
for fname in ['high_quality.jsonl', 'low_quality.jsonl', 'test_held_out.jsonl']:
    size = os.path.getsize(f'{OUT_DIR}/{fname}') / 1024
    print(f"  {OUT_DIR}/{fname:<30} {size:.1f} KB")
