import json, os
BASE='/N/scratch/madbala/parametric-faithfulness_run/finetuned_results'
for cond in ['mistral_baseline','mistral_high','mistral_low','llama8b_baseline','llama8b_high','llama8b_low']:
    path = f'{BASE}/{cond}/results.jsonl'
    if not os.path.exists(path):
        print(f'{cond}: missing')
        continue
    results = [json.loads(l) for l in open(path) if l.strip()]
    epochs = max(int(k) for r in results for k in r['epoch_results'])
    flips = [any(r['epoch_results'][str(e)]['flip'] for e in range(1,epochs+1) if str(e) in r['epoch_results']) for r in results]
    deltas = [r['epoch_results'].get(str(epochs),{}).get('delta_p',0) for r in results]
    print(f'{cond}: N={len(results)} binary_faith={sum(flips)/len(flips)*100:.1f}% mean_delta_p={sum(deltas)/len(deltas):.4f}')
