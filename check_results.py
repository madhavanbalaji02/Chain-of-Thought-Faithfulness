import json

for condition in ['high_quality', 'low_quality']:
    path = f'/N/scratch/madbala/parametric-faithfulness_run/finetuned_results/{condition}/results.jsonl'
    try:
        with open(path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        flips = sum(1 for l in lines if any(
            l['epoch_results'].get(str(e), {}).get('flip', False)
            for e in range(1, 6)
        ))
        deltas = [l['epoch_results'].get('5', {}).get('delta_p', 0) for l in lines]
        mean_dp = sum(deltas)/len(deltas) if deltas else 0
        print(f'{condition}: {flips}/{len(lines)} = {flips/len(lines)*100:.1f}% binary faithful, mean delta_p(ep5)={mean_dp:.4f}')
    except Exception as e:
        print(f'{condition}: ERROR - {e}')
