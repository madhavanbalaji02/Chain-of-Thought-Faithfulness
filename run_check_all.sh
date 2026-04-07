#!/bin/bash
cd /N/scratch/madbala/parametric-faithfulness_run
source venv/bin/activate

# Write check_all_results.py inline
cat > check_all_results.py << 'PYEOF'
import json, os, math

def read_results(path):
    if not os.path.exists(path):
        return None, None, None
    lines = [json.loads(l) for l in open(path) if l.strip()]
    if not lines:
        return 0, 0, 0
    flips = 0
    deltas = []
    for l in lines:
        epoch_results = l.get('epoch_results', {})
        if not epoch_results:
            continue
        epochs = sorted(int(k) for k in epoch_results.keys())
        max_epoch = str(max(epochs))
        flipped = any(
            epoch_results.get(str(e), {}).get('flip', False)
            for e in epochs if e > 0
        )
        if flipped:
            flips += 1
        dp = epoch_results[max_epoch].get('delta_p', None)
        if dp is not None:
            deltas.append(dp)
    mean_dp = sum(deltas) / len(deltas) if deltas else 0
    return flips, len(lines), mean_dp

BASE = 'finetuned_results'
conditions = [
    ('LLaMA-3-3B',  'baseline',     f'{BASE}/baseline/results.jsonl'),
    ('LLaMA-3-3B',  'high-quality', f'{BASE}/high_quality/results.jsonl'),
    ('LLaMA-3-3B',  'low-quality',  f'{BASE}/low_quality/results.jsonl'),
    ('Phi-3',       'baseline',     f'{BASE}/phi3_baseline/results.jsonl'),
    ('LLaMA-3-8B',  'baseline',     f'{BASE}/llama8b_baseline/results.jsonl'),
    ('LLaMA-3-8B',  'high-quality', f'{BASE}/llama8b_high/results.jsonl'),
    ('LLaMA-3-8B',  'low-quality',  f'{BASE}/llama8b_low/results.jsonl'),
    ('Mistral-7B',  'baseline',     f'{BASE}/mistral_baseline/results.jsonl'),
    ('Mistral-7B',  'high-quality', f'{BASE}/mistral_high/results.jsonl'),
    ('Mistral-7B',  'low-quality',  f'{BASE}/mistral_low/results.jsonl'),
    ('Phi-3',       'high-quality', f'{BASE}/phi3_high/results.jsonl'),
    ('Phi-3',       'low-quality',  f'{BASE}/phi3_low/results.jsonl'),
]

print("=== RESULTS SUMMARY ===")
print()
print(f"{'Model':<12} {'Condition':<14} {'Flips/N':>8} {'%':>6} {'delta_p':>8} {'vs baseline':>12}")
print("-" * 68)

baselines = {}
for model, condition, path in conditions:
    flips, n, mean_dp = read_results(path)
    if n is None:
        print(f"{model:<12} {condition:<14} {'—':>22} NOT DONE")
        continue
    if n == 0:
        print(f"{model:<12} {condition:<14} {'—':>22} EMPTY")
        continue
    pct = flips / n * 100
    status = ' ✓' if n >= 50 else f' ({n}/50 partial)'
    if condition == 'baseline':
        baselines[model] = pct
        vs = '—'
    else:
        base = baselines.get(model)
        vs = f"{pct - base:+.1f}pp" if base is not None else 'no baseline yet'
    print(f"{model:<12} {condition:<14} {flips:>4}/{n:<4} {pct:>5.1f}% {mean_dp:>8.4f} {vs:>12}{status}")

print()
print("=== PARADOX CHECK ===")
for model in ['LLaMA-3-3B', 'LLaMA-3-8B', 'Mistral-7B', 'Phi-3']:
    base = baselines.get(model)
    if base is not None:
        print(f"  {model} baseline: {base:.1f}%")
PYEOF

python3 check_all_results.py
