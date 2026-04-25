#!/usr/bin/env python3
"""
Preprocess research data into JSON files for the web dashboard.
Reads annotation results, LLM judge outputs, and mistake-experiment results.
"""

import os
import json
import csv
import glob
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

DATASETS = ['arc-challenge', 'openbook', 'sports', 'sqa']
MODELS = ['Phi-3', 'LLaMA-3', 'LLaMA-3-3B', 'Mistral-2']

DATASET_NICE = {
    'arc-challenge': 'ARC-Challenge',
    'openbook': 'OpenBookQA',
    'sqa': 'StrategyQA',
    'sports': 'Sports'
}

MODEL_NICE = {
    'Phi-3': 'Phi-3',
    'LLaMA-3': 'LLaMA-3-8B',
    'LLaMA-3-3B': 'LLaMA-3-3B',
    'Mistral-2': 'Mistral-2',
}


def load_jsonl(fpath):
    results = []
    with open(fpath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def process_annotation_data():
    """Process annotation study CSV with human ratings."""
    csv_path = os.path.join(BASE_DIR, 'annotation_results', 'reasoning-chain-study.csv')
    if not os.path.exists(csv_path):
        return {'summary': [], 'breakdown': [], 'instances': [], 'total': 0}

    rating_counts = defaultdict(int)
    dm_ratings = defaultdict(lambda: defaultdict(int))
    instances = []
    total = 0

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rating = row.get('rating', '')
            dataset = row.get('dataset', '')
            model = row.get('model', '')
            flip = row.get('flip', 'FALSE')
            dp = float(row.get('dp', 0))

            rating_counts[rating] += 1
            dm_ratings[f"{dataset}|||{model}"][rating] += 1
            total += 1

            # Parse segmented_cot 
            seg_cot_raw = row.get('segmented_cot', '')
            steps = [s.strip() for s in seg_cot_raw.split('||') if s.strip()]

            options_raw = row.get('options', '')
            options = [o.strip() for o in options_raw.split('||') if o.strip()]

            instances.append({
                'id': row.get('ID', ''),
                'question': row.get('Question', ''),
                'options': options,
                'steps': steps,
                'correct_answer': int(row.get('correct_answer', 0)),
                'predicted_answer': int(row.get('predicted_answer', 0)),
                'target_step_idx': int(row.get('target_step_idx', 0)),
                'dp': round(dp, 3),
                'flip': flip.upper() == 'TRUE',
                'dataset': dataset,
                'dataset_nice': DATASET_NICE.get(dataset, dataset),
                'model': model,
                'model_nice': MODEL_NICE.get(model, model),
                'rating': rating,
            })

    rating_order = ['Not Supportive At All', 'Slightly Supportive', 'Mostly Supportive', 'Very Supportive']
    summary = []
    for r in rating_order:
        c = rating_counts.get(r, 0)
        summary.append({
            'rating': r,
            'count': c,
            'percentage': round(c / total * 100, 1) if total > 0 else 0,
        })

    breakdown = []
    for key, ratings in dm_ratings.items():
        ds, mdl = key.split('|||')
        total_key = sum(ratings.values())
        entry = {
            'dataset': ds,
            'dataset_nice': DATASET_NICE.get(ds, ds),
            'model': mdl,
            'model_nice': MODEL_NICE.get(mdl, mdl),
            'total': total_key,
        }
        for r in rating_order:
            safe_key = r.replace(' ', '_').lower()
            entry[safe_key] = ratings.get(r, 0)
        breakdown.append(entry)

    return {
        'summary': summary,
        'breakdown': breakdown,
        'instances': instances,
        'total': total,
    }


def process_lm_judge():
    """Process LLM-as-judge results."""
    judge_dir = os.path.join(BASE_DIR, 'LM_judge_cot')
    if not os.path.exists(judge_dir):
        return []

    results = []
    for fname in sorted(os.listdir(judge_dir)):
        if not fname.endswith('.jsonl'):
            continue

        # Parse filename: Model_dataset_NPO_KL_lr_judgements.jsonl
        dataset = None
        for d in DATASETS:
            if d in fname:
                dataset = d
                break
        if not dataset:
            continue

        # Extract model name (everything before dataset)
        model = fname.split(f'_{dataset}')[0]

        fpath = os.path.join(judge_dir, fname)
        judgements = load_jsonl(fpath)

        yes, no, unclear = 0, 0, 0
        sample_judgements = []
        for j in judgements:
            resp = j.get('response', '').strip()
            first_line = resp.split('\n')[0].strip().lower()
            if first_line.startswith('yes'):
                yes += 1
                verdict = 'agree'
            elif first_line.startswith('no'):
                no += 1
                verdict = 'disagree'
            else:
                unclear += 1
                verdict = 'unclear'

            if len(sample_judgements) < 3:
                sample_judgements.append({
                    'instance_id': j.get('instance_id', ''),
                    'response': resp,
                    'verdict': verdict,
                })

        total = yes + no + unclear
        results.append({
            'model': model,
            'model_nice': MODEL_NICE.get(model, model),
            'dataset': dataset,
            'dataset_nice': DATASET_NICE.get(dataset, dataset),
            'total': total,
            'agree': yes,
            'disagree': no,
            'unclear': unclear,
            'agree_pct': round(yes / total * 100, 1) if total > 0 else 0,
            'disagree_pct': round(no / total * 100, 1) if total > 0 else 0,
            'samples': sample_judgements,
        })

    return results


def process_mistake_results():
    """Process add-mistake experiment results for the instance explorer."""
    results_dir = os.path.join(BASE_DIR, 'minimal_mistake_results')
    instances = []

    for dataset in DATASETS:
        for model in MODELS:
            pattern = os.path.join(results_dir, dataset, model, '*.out')
            files = glob.glob(pattern)
            if not files:
                continue
            data = load_jsonl(files[0])

            count = 0
            for item in data:
                if count >= 8:
                    break
                steps = item.get('segmented_cot', [])
                if not steps or len(steps) < 2:
                    continue
                instances.append({
                    'id': item.get('id', ''),
                    'question': item.get('question', ''),
                    'options': item.get('options', []),
                    'cot_step': item.get('cot_step', ''),
                    'mistake_cot_step': item.get('mistake_cot_step', ''),
                    'step_idx': item.get('step_idx', 0),
                    'segmented_cot': steps,
                    'dataset': dataset,
                    'dataset_nice': DATASET_NICE[dataset],
                    'model': model,
                    'model_nice': MODEL_NICE[model],
                })
                count += 1

    return instances


def build_aggregate_stats(annotations, judge_results):
    """Build aggregate dashboard stats from annotation + judge data."""
    stats = []

    # From annotation data: compute flip rates (faithfulness proxy) per model×dataset  
    dm_flip = defaultdict(lambda: {'flips': 0, 'total': 0, 'dp_sum': 0})
    for inst in annotations.get('instances', []):
        key = f"{inst['dataset']}|||{inst['model']}"
        dm_flip[key]['total'] += 1
        dm_flip[key]['dp_sum'] += abs(inst['dp'])
        if inst['flip']:
            dm_flip[key]['flips'] += 1

    # From judge: agreement rate per model×dataset
    judge_map = {}
    for j in judge_results:
        judge_map[f"{j['dataset']}|||{j['model']}"] = j

    for key, flip_data in dm_flip.items():
        ds, mdl = key.split('|||')
        faith_pct = round(flip_data['flips'] / flip_data['total'] * 100, 1) if flip_data['total'] > 0 else 0
        avg_dp = round(flip_data['dp_sum'] / flip_data['total'], 3) if flip_data['total'] > 0 else 0

        jd = judge_map.get(key, {})
        stats.append({
            'dataset': ds,
            'dataset_nice': DATASET_NICE.get(ds, ds),
            'model': mdl,
            'model_nice': MODEL_NICE.get(mdl, mdl),
            'flip_rate': faith_pct,
            'avg_dp': avg_dp,
            'n_annotation': flip_data['total'],
            'judge_agree_pct': jd.get('agree_pct', 0),
            'judge_total': jd.get('total', 0),
        })

    # Also include model×dataset combos only in judge data  
    for j in judge_results:
        key = f"{j['dataset']}|||{j['model']}"
        if key not in dm_flip:
            stats.append({
                'dataset': j['dataset'],
                'dataset_nice': j['dataset_nice'],
                'model': j['model'],
                'model_nice': j['model_nice'],
                'flip_rate': 0,
                'avg_dp': 0,
                'n_annotation': 0,
                'judge_agree_pct': j.get('agree_pct', 0),
                'judge_total': j.get('total', 0),
            })

    return stats


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Processing annotation results...")
    annotations = process_annotation_data()
    print(f"  → {annotations['total']} annotated instances")

    print("Processing LLM judge results...")
    judge = process_lm_judge()
    print(f"  → {len(judge)} model×dataset judge configs")

    print("Processing mistake results...")
    mistakes = process_mistake_results()
    print(f"  → {len(mistakes)} mistake instances")

    print("Building aggregate stats...")
    aggregate = build_aggregate_stats(annotations, judge)
    print(f"  → {len(aggregate)} model×dataset aggregate entries")

    # Write outputs
    with open(os.path.join(OUTPUT_DIR, 'annotations.json'), 'w') as f:
        json.dump(annotations, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, 'lm_judge.json'), 'w') as f:
        json.dump(judge, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, 'mistakes.json'), 'w') as f:
        json.dump(mistakes, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, 'aggregate.json'), 'w') as f:
        json.dump(aggregate, f, indent=2)

    meta = {
        'title': 'Measuring Faithfulness of Chains of Thought by Unlearning Reasoning Steps',
        'authors': 'Martin Tutek, Fateme Hashemi Chaleshtori, Ana Marasović, Yonatan Belinkov',
        'year': 2025,
        'arxiv': 'https://arxiv.org/abs/2502.14829',
        'venue': 'Preprint',
        'datasets': [{'id': d, 'name': DATASET_NICE[d]} for d in DATASETS],
        'models': [{'id': m, 'name': MODEL_NICE[m]} for m in MODELS],
    }
    with open(os.path.join(OUTPUT_DIR, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("Done! Output in:", OUTPUT_DIR)


if __name__ == '__main__':
    main()
