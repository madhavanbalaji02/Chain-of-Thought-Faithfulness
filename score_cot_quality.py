"""
score_cot_quality.py
====================
Calls claude-sonnet-4-20250514 to score LLaMA-3-3B CoT quality on
coherence, plausibility, and completeness (each 1–5).

Checkpointed: already-scored instances are loaded from the output CSV
at startup so the script can be safely re-run after interruption.

Usage:
    python3 score_cot_quality.py
Output:
    analysis/cot_quality_scores.csv
"""

import os, json, glob, time, csv, re
import anthropic

RESULTS_DIR   = 'final_results'
MODEL_FILTER  = 'LLaMA-3-3B'
OUT_CSV       = 'analysis/cot_quality_scores.csv'
API_MODEL     = 'claude-sonnet-4-5'   # current name on the API
LETTERS       = ['A', 'B', 'C', 'D', 'E']
RETRY_LIMIT   = 3
SLEEP_BETWEEN = 0.15   # seconds between calls to stay within rate limits

os.makedirs('analysis', exist_ok=True)

# ──────────────────────────────────────────────
# Load all LLaMA-3-3B instances
# ──────────────────────────────────────────────
instances = []
seen_keys = set()

for fpath in sorted(glob.glob(f'{RESULTS_DIR}/*/LLaMA-3-3B/*.out')):
    dataset = fpath.split('/')[1]
    with open(fpath) as fh:
        for line in fh:
            rec = json.loads(line.strip())
            uid = rec.get('id', '') or rec['question'][:80]
            key = (dataset, uid)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            correct_idx = LETTERS.index(rec['correct'])
            instances.append({
                'instance_id':              uid,
                'model':                    MODEL_FILTER,
                'dataset':                  dataset,
                'question':                 rec['question'],
                'initial_cot':              rec['initial_cot'],
                'correct':                  rec['correct'],
                'options':                  rec.get('options', []),
                'initial_prediction_correct': int(rec['prediction'] == correct_idx),
            })

print(f"Loaded {len(instances)} unique LLaMA-3-3B instances.")

# ──────────────────────────────────────────────
# Load checkpoint (already-scored rows)
# ──────────────────────────────────────────────
scored = {}   # key → row dict
if os.path.exists(OUT_CSV):
    with open(OUT_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = (row['dataset'], row['instance_id'])
            scored[k] = row
    print(f"Checkpoint: {len(scored)} already scored, {len(instances)-len(scored)} remaining.")

# ──────────────────────────────────────────────
# Scoring function
# ──────────────────────────────────────────────
client = anthropic.Anthropic()

SYSTEM = (
    "You are a precise evaluator of chain-of-thought reasoning. "
    "Respond only with valid JSON and nothing else."
)

def score_cot(question, cot, options=None):
    options_str = ''
    if options:
        options_str = '\nAnswer choices: ' + ', '.join(options)
    user_msg = f"""You are evaluating a chain-of-thought reasoning trace.
Question: {question}{options_str}
Chain of thought: {cot}

Score this CoT on:
1. Coherence (1-5): Is the reasoning logically connected step by step?
2. Plausibility (1-5): Does it sound like convincing human reasoning?
3. Completeness (1-5): Does it address all aspects needed to answer?

Respond ONLY with JSON: {{"coherence": X, "plausibility": X, "completeness": X}}"""

    for attempt in range(RETRY_LIMIT):
        try:
            resp = client.messages.create(
                model=API_MODEL,
                max_tokens=60,
                system=SYSTEM,
                messages=[{'role': 'user', 'content': user_msg}],
            )
            text = resp.content[0].text.strip()
            # Parse JSON — handle minor formatting issues
            text = re.sub(r'```json\s*|\s*```', '', text).strip()
            data = json.loads(text)
            c = int(data.get('coherence', data.get('Coherence', 3)))
            p = int(data.get('plausibility', data.get('Plausibility', 3)))
            k = int(data.get('completeness', data.get('Completeness', 3)))
            # Clamp to valid range
            c, p, k = max(1,min(5,c)), max(1,min(5,p)), max(1,min(5,k))
            return c, p, k
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            if attempt < RETRY_LIMIT - 1:
                time.sleep(1 + attempt)
            else:
                print(f"  [WARN] JSON parse failed after {RETRY_LIMIT} attempts: {e}  text={repr(text[:80])}")
                return 3, 3, 3   # fallback neutral score
        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"  [RATE LIMIT] waiting {wait}s...")
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            print(f"  [API ERROR] {e.status_code}: {e.message}")
            if attempt < RETRY_LIMIT - 1:
                time.sleep(2)
            else:
                return 3, 3, 3

# ──────────────────────────────────────────────
# Score all instances
# ──────────────────────────────────────────────
FIELDNAMES = [
    'instance_id', 'model', 'dataset',
    'coherence', 'plausibility', 'completeness', 'composite_score',
    'initial_prediction_correct', 'correct', 'question', 'initial_cot',
]

# Open CSV in append mode (checkpoint)
file_exists = os.path.exists(OUT_CSV) and len(scored) > 0
with open(OUT_CSV, 'a', newline='', encoding='utf-8') as csvf:
    writer = csv.DictWriter(csvf, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()

    to_score = [inst for inst in instances
                if (inst['dataset'], inst['instance_id']) not in scored]

    print(f"\nScoring {len(to_score)} instances with {API_MODEL}...")
    for i, inst in enumerate(to_score):
        c, p, k = score_cot(inst['question'], inst['initial_cot'], inst.get('options'))
        composite = round((c + p + k) / 3, 4)

        row = {
            'instance_id':              inst['instance_id'],
            'model':                    inst['model'],
            'dataset':                  inst['dataset'],
            'coherence':                c,
            'plausibility':             p,
            'completeness':             k,
            'composite_score':          composite,
            'initial_prediction_correct': inst['initial_prediction_correct'],
            'correct':                  inst['correct'],
            'question':                 inst['question'].replace('\n', ' '),
            'initial_cot':              inst['initial_cot'].replace('\n', ' '),
        }
        writer.writerow(row)
        csvf.flush()
        scored[(inst['dataset'], inst['instance_id'])] = row

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(to_score)}] dataset={inst['dataset']} "
                  f"composite={composite:.2f} (C={c} P={p} K={k})")

        time.sleep(SLEEP_BETWEEN)

print(f"\nDone. Total scored: {len(scored)}")
print(f"Output: {OUT_CSV}")

# ──────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────
import statistics as stat

all_rows = list(scored.values())
scores   = [float(r['composite_score']) for r in all_rows]
scores.sort()

n     = len(scores)
q25   = scores[n//4]
q50   = scores[n//2]
q75   = scores[3*n//4]
mean  = stat.mean(scores)
std   = stat.stdev(scores)

print(f"\n{'='*50}")
print(f"Composite score distribution (n={n})")
print(f"{'='*50}")
print(f"  Mean:   {mean:.3f}")
print(f"  Std:    {std:.3f}")
print(f"  Min:    {min(scores):.3f}")
print(f"  Q25:    {q25:.3f}")
print(f"  Median: {q50:.3f}")
print(f"  Q75:    {q75:.3f}")
print(f"  Max:    {max(scores):.3f}")

# Per-dataset breakdown
from collections import defaultdict
by_ds = defaultdict(list)
for r in all_rows:
    by_ds[r['dataset']].append(float(r['composite_score']))

print(f"\nPer-dataset mean composite score:")
for ds in sorted(by_ds):
    ds_scores = by_ds[ds]
    print(f"  {ds:<20}: {stat.mean(ds_scores):.3f} (n={len(ds_scores)})")

# Top 3 and bottom 3 examples
top3    = sorted(all_rows, key=lambda r: float(r['composite_score']), reverse=True)[:3]
bottom3 = sorted(all_rows, key=lambda r: float(r['composite_score']))[:3]

print(f"\n{'='*50}")
print("TOP 3 (highest composite score):")
print(f"{'='*50}")
for r in top3:
    print(f"\n  Score={r['composite_score']} (C={r['coherence']} P={r['plausibility']} K={r['completeness']})")
    print(f"  Dataset: {r['dataset']} | Correct: {r['correct']} | Pred correct: {r['initial_prediction_correct']}")
    print(f"  Q: {r['question'][:100]}")
    print(f"  CoT: {r['initial_cot'][:200]}")

print(f"\n{'='*50}")
print("BOTTOM 3 (lowest composite score):")
print(f"{'='*50}")
for r in bottom3:
    print(f"\n  Score={r['composite_score']} (C={r['coherence']} P={r['plausibility']} K={r['completeness']})")
    print(f"  Dataset: {r['dataset']} | Correct: {r['correct']} | Pred correct: {r['initial_prediction_correct']}")
    print(f"  Q: {r['question'][:100]}")
    print(f"  CoT: {r['initial_cot'][:200]}")
