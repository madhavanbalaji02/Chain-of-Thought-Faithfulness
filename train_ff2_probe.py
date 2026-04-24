"""
Train linear probes on FF2 activations to predict:
  1. Binary faithfulness (flip at any epoch)
  2. Continuous delta_p

Compare probe accuracy across baseline vs high-FT vs low-FT.

Interpretation:
  - If probe accuracy >> majority baseline for baseline model:
      FF2 encodes faithfulness-predictive information
  - If accuracy DROPS for high-FT vs baseline:
      High-quality FT degrades FF2 step-specificity (this IS the mechanism)
  - If low-FT shows no drop:
      Quality is the causal variable, not fine-tuning in general

Usage:
    python train_ff2_probe.py \
        --baseline_acts ff2_activations/baseline.pkl \
        --high_acts     ff2_activations/high_ft.pkl \
        --low_acts      ff2_activations/low_ft.pkl \
        --layers 8,16,24
"""
import pickle, json, os, argparse
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler


def load_activations(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def run_binary_probe(acts_list, layer_idx):
    X, y = [], []
    for item in acts_list:
        if layer_idx not in item['activations']:
            continue
        X.append(item['activations'][layer_idx])
        y.append(int(item['binary_faithful']))
    if len(X) < 10 or len(set(y)) < 2:
        return None, None, len(X)
    X = StandardScaler().fit_transform(np.array(X))
    y = np.array(y)
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
    cv = StratifiedKFold(n_splits=min(5, int(min(np.bincount(y)))),
                         shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    baseline = max(y.mean(), 1 - y.mean())
    return float(scores.mean()), float(baseline), len(X)


def run_continuous_probe(acts_list, layer_idx):
    X, y = [], []
    for item in acts_list:
        if layer_idx not in item['activations']:
            continue
        X.append(item['activations'][layer_idx])
        y.append(float(item['delta_p']))
    if len(X) < 10:
        return None, len(X)
    X = StandardScaler().fit_transform(np.array(X))
    y = np.array(y)
    clf = Ridge(alpha=1.0)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='r2')
    return float(scores.mean()), len(X)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_acts', required=True)
    parser.add_argument('--high_acts',     required=True)
    parser.add_argument('--low_acts',      required=True)
    parser.add_argument('--layers',        default='8,16,24')
    parser.add_argument('--output',        default='analysis/ff2_probe_results.json')
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(',')]

    datasets = {
        'Baseline': load_activations(args.baseline_acts),
        'High-FT':  load_activations(args.high_acts),
        'Low-FT':   load_activations(args.low_acts),
    }

    for label, acts in datasets.items():
        n_faith = sum(a['binary_faithful'] for a in acts)
        print(f'{label}: N={len(acts)}  faithful={n_faith} ({n_faith/max(len(acts),1)*100:.1f}%)')
    print()

    results = {}
    header = f'{"Layer":<7} {"Condition":<12} {"N":>5} {"BinAcc":>8} {"MajBase":>9} {"Lift":>7} {"R2(Δp)":>9}'
    print(header)
    print('-' * len(header))

    for layer in layers:
        for label, acts in datasets.items():
            bin_acc, bin_base, n = run_binary_probe(acts, layer)
            r2, _ = run_continuous_probe(acts, layer)
            lift = (bin_acc - bin_base) if (bin_acc and bin_base) else None

            key = f'layer{layer}_{label.replace("-","_")}'
            results[key] = {
                'layer': layer,
                'condition': label,
                'n': n,
                'binary_accuracy':  bin_acc,
                'majority_baseline': bin_base,
                'lift_over_majority': lift,
                'delta_p_r2': r2,
            }

            if bin_acc is not None:
                print(f'{layer:<7} {label:<12} {n:>5} {bin_acc:>8.3f} '
                      f'{bin_base:>9.3f} {lift:>+7.3f} {(r2 or 0):>9.3f}')
        print()

    print('=== MECHANISTIC INTERPRETATION ===')
    for layer in layers:
        b  = results.get(f'layer{layer}_Baseline', {})
        h  = results.get(f'layer{layer}_High_FT',  {})
        l  = results.get(f'layer{layer}_Low_FT',   {})
        if b.get('binary_accuracy') and h.get('binary_accuracy'):
            drop = h['binary_accuracy'] - b['binary_accuracy']
            l_drop = (l['binary_accuracy'] - b['binary_accuracy']
                      if l.get('binary_accuracy') else None)
            l_drop_str = f'{l_drop:+.3f}' if l_drop is not None else 'N/A'
            print(f'Layer {layer}: Baseline→High drop = {drop:+.3f}  '
                  f'Baseline→Low drop = {l_drop_str}')
            if drop < -0.03:
                print(f'  ✓ High-FT degrades FF2 faithfulness encoding at layer {layer}')
            if l_drop is not None and abs(l_drop) < abs(drop):
                print(f'  ✓ Effect is quality-specific (Low-FT smaller drop)')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {args.output}')


if __name__ == '__main__':
    main()
