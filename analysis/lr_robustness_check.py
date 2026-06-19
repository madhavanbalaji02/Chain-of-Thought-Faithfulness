"""
lr_robustness_check.py
======================
Robustness analysis of the alignment-faithfulness paradox.

1. Documents which LR was actually used for each model's fine-tuning eval
   (cross-referencing new_experiment_jobs.py vs submit_multi_model.sh vs
   const.py:paper_best_lr).
2. Loads existing finetuned_results/ and computes binary faithfulness +
   two-proportion z-test for each model's baseline vs high-FT.
3. If calibrated-LR rerun data exists in finetuned_results_calibrated_lr/,
   produces a side-by-side comparison.
4. If LR-sweep data exists for LLaMA-3-3B, reports the sweep.
5. Writes analysis/LR_ROBUSTNESS_SUMMARY.md.

Usage:
    python analysis/lr_robustness_check.py
"""

import os, json, sys
import numpy as np
from scipy.stats import norm

LETTERS = ['A', 'B', 'C', 'D', 'E']

MODELS = {
    'LLaMA-3-3B': {
        'hf_name': 'meta-llama/Llama-3.2-3B-Instruct',
        'paper_lr': 3e-05,
        'eval_lr_used': 3e-05,
        'source_script': 'new_experiment_jobs.py',
        'dirs': {
            'baseline': 'finetuned_results/baseline',
            'high':     'finetuned_results/high_quality',
            'low':      'finetuned_results/low_quality',
        },
    },
    'LLaMA-3-8B': {
        'hf_name': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'paper_lr': 1e-05,
        'eval_lr_used': 1e-05,
        'source_script': 'submit_multi_model.sh',
        'dirs': {
            'baseline': 'finetuned_results/llama8b_baseline',
            'high':     'finetuned_results/llama8b_high',
            'low':      'finetuned_results/llama8b_low',
        },
    },
    'Mistral-7B': {
        'hf_name': 'mistralai/Mistral-7B-Instruct-v0.2',
        'paper_lr': 5e-06,
        'eval_lr_used': 1e-05,
        'source_script': 'sacct job 6815741-6815743 (NOT submit_multi_model.sh)',
        'dirs': {
            'baseline': 'finetuned_results/mistral_baseline',
            'high':     'finetuned_results/mistral_high',
            'low':      'finetuned_results/mistral_low',
        },
    },
    'Phi-3': {
        'hf_name': 'microsoft/Phi-3-mini-4k-instruct',
        'paper_lr': 1e-04,
        'eval_lr_used': 1e-04,
        'source_script': 'submit_multi_model.sh',
        'dirs': {
            'baseline': 'finetuned_results/phi3_baseline',
            'high':     'finetuned_results/phi3_high',
            'low':      'finetuned_results/phi3_low',
        },
    },
}

MISTRAL_CALIBRATED = {
    'baseline': 'finetuned_results_calibrated_lr/mistral_lr5e-06/baseline',
    'high':     'finetuned_results_calibrated_lr/mistral_lr5e-06/high_quality',
    'low':      'finetuned_results_calibrated_lr/mistral_lr5e-06/low_quality',
}

LLAMA3B_LR_SWEEP = {
    '1e-05': {
        'baseline': 'finetuned_results_calibrated_lr/llama3b_lr1e-05/baseline',
        'high':     'finetuned_results_calibrated_lr/llama3b_lr1e-05/high_quality',
        'low':      'finetuned_results_calibrated_lr/llama3b_lr1e-05/low_quality',
    },
    '3e-05': {
        'baseline': 'finetuned_results/baseline',
        'high':     'finetuned_results/high_quality',
        'low':      'finetuned_results/low_quality',
    },
    '5e-05': {
        'baseline': 'finetuned_results_calibrated_lr/llama3b_lr5e-05/baseline',
        'high':     'finetuned_results_calibrated_lr/llama3b_lr5e-05/high_quality',
        'low':      'finetuned_results_calibrated_lr/llama3b_lr5e-05/low_quality',
    },
}


def load_condition(dirpath):
    fpath = os.path.join(dirpath, 'results.jsonl')
    if not os.path.isfile(fpath):
        return None
    records = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def binary_faithfulness(records):
    flips = []
    for r in records:
        er = r['epoch_results']
        baseline_pred = er['0']['prediction']
        flipped = any(
            er[str(e)]['prediction'] != baseline_pred
            for e in range(1, 6) if str(e) in er
        )
        flips.append(flipped)
    return flips


def two_proportion_ztest(p1, n1, p2, n2):
    if n1 == 0 or n2 == 0:
        return float('nan'), float('nan')
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    if p_pool == 0 or p_pool == 1:
        return float('nan'), float('nan')
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if se < 1e-15:
        return float('nan'), float('nan')
    z = (p1 - p2) / se
    p_val = 2 * (1 - norm.cdf(abs(z)))
    return z, p_val


def analyze_model(model_name, info):
    results = {}
    for cond, dirpath in info['dirs'].items():
        records = load_condition(dirpath)
        if records is None:
            results[cond] = None
            continue
        flips = binary_faithfulness(records)
        n = len(flips)
        faith_rate = sum(flips) / n if n > 0 else 0
        results[cond] = {
            'n': n,
            'flips': sum(flips),
            'faithfulness': faith_rate * 100,
            'raw_flips': flips,
        }
    return results


def format_row(model, lr, results, z_val, p_val):
    b = results.get('baseline')
    h = results.get('high')
    l = results.get('low')
    b_str = f"{b['faithfulness']:.1f}% ({b['flips']}/{b['n']})" if b else "N/A"
    h_str = f"{h['faithfulness']:.1f}% ({h['flips']}/{h['n']})" if h else "N/A"
    l_str = f"{l['faithfulness']:.1f}% ({l['flips']}/{l['n']})" if l else "N/A"
    sig = ""
    if not np.isnan(p_val):
        if p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = "n.s."
    p_str = f"p={p_val:.4f} {sig}" if not np.isnan(p_val) else "N/A"
    drop = ""
    if b and h and b['faithfulness'] > 0:
        rel = (h['faithfulness'] - b['faithfulness']) / b['faithfulness'] * 100
        drop = f"{rel:+.0f}%"
    return f"| {model} | {lr} | {b_str} | {h_str} | {l_str} | {p_str} | {drop} |"


def main():
    os.makedirs('analysis', exist_ok=True)

    lines = []
    lines.append("# LR Robustness Check: Alignment-Faithfulness Paradox")
    lines.append("")
    lines.append("## 1. LR Calibration Audit")
    lines.append("")
    lines.append("**Question**: Were the fine-tuning evaluation runs at a single fixed LR,")
    lines.append("or were per-model calibrated LRs used?")
    lines.append("")
    lines.append("| Model | Paper Best LR | Eval LR Used | Match? | Source Script |")
    lines.append("|:------|:---:|:---:|:---:|:---|")

    all_match = True
    for model, info in MODELS.items():
        match = info['paper_lr'] == info['eval_lr_used']
        if not match:
            all_match = False
        lines.append(
            f"| {model} | {info['paper_lr']:.0e} | {info['eval_lr_used']:.0e} "
            f"| {'YES' if match else '**NO**'} | `{info['source_script']}` |"
        )

    lines.append("")
    if all_match:
        lines.append("**Finding**: All four models were evaluated at their paper-calibrated LRs.")
    else:
        lines.append("**Finding**: Not all models were evaluated at their paper-calibrated LRs.")
        lines.append("")
        lines.append("Evidence source: SLURM `sacct --format=SubmitLine` on BigRed200, which")
        lines.append("records the exact `sbatch --wrap=...` command for each job. This is")
        lines.append("authoritative — it reflects what SLURM actually executed, not what any")
        lines.append("script file currently contains.")
        lines.append("")
        lines.append("**Mistral-7B discrepancy**: `submit_multi_model.sh` specifies `--lr 5e-06`,")
        lines.append("but the SLURM jobs that produced the final results (6815741-6815743,")
        lines.append("submitted Apr 10) used `--lr 1e-05`. The initial Mistral runs (6754962-6754964,")
        lines.append("Apr 6) all CANCELLED or FAILED. The successful retries on Apr 10 used a")
        lines.append("different command (likely typed manually or from an edited script).")
        lines.append("")
        lines.append("The Tutek et al. ablation shows Mistral at lr=5e-06 gives ~57% efficacy")
        lines.append("with ~96% specificity; at lr=1e-05 efficacy rises to ~70% but specificity")
        lines.append("drops to ~86-91%. This means the Mistral results may have inflated flip")
        lines.append("rates from specificity degradation at the higher LR.")
        lines.append("")
        lines.append("**paper.tex line 348** states a single LR of 3×10⁻⁵ for all models.")
        lines.append("The actual LRs were: LLaMA-3-3B 3e-05, Phi-3 1e-04, Mistral 1e-05,")
        lines.append("LLaMA-8B 1e-05. The methods section must be corrected.")

    # Existing results
    lines.append("")
    lines.append("## 2. Existing Results (per-model calibrated LRs)")
    lines.append("")
    lines.append("| Model | Eval LR | Baseline | High-FT | Low-FT | z-test (base vs high) | Rel. Drop |")
    lines.append("|:------|:---:|:---|:---|:---|:---|:---:|")

    print("=" * 70)
    print("LR ROBUSTNESS CHECK — Alignment-Faithfulness Paradox")
    print("=" * 70)

    for model, info in MODELS.items():
        results = analyze_model(model, info)
        b = results.get('baseline')
        h = results.get('high')

        z_val, p_val = float('nan'), float('nan')
        if b and h:
            z_val, p_val = two_proportion_ztest(
                b['faithfulness']/100, b['n'],
                h['faithfulness']/100, h['n'],
            )

        lr_str = f"{info['eval_lr_used']:.0e}"
        lines.append(format_row(model, lr_str, results, z_val, p_val))

        print(f"\n{model} (lr={lr_str}):")
        for cond in ['baseline', 'high', 'low']:
            r = results.get(cond)
            if r:
                print(f"  {cond:12s}: {r['faithfulness']:.1f}% ({r['flips']}/{r['n']})")
            else:
                print(f"  {cond:12s}: no data")
        if not np.isnan(p_val):
            print(f"  z-test (baseline vs high): z={z_val:.3f}, p={p_val:.4f}")

    # LLaMA-3-3B LR sweep
    lines.append("")
    lines.append("## 3. LLaMA-3-3B LR Sensitivity Sweep")
    lines.append("")

    sweep_data = {}
    sweep_available = False
    for lr_str, dirs in LLAMA3B_LR_SWEEP.items():
        results = {}
        for cond, dirpath in dirs.items():
            records = load_condition(dirpath)
            if records is not None:
                flips = binary_faithfulness(records)
                n = len(flips)
                results[cond] = {
                    'n': n, 'flips': sum(flips),
                    'faithfulness': sum(flips)/n*100 if n > 0 else 0,
                }
                if lr_str != '3e-05':
                    sweep_available = True
            else:
                results[cond] = None
        sweep_data[lr_str] = results

    if sweep_available:
        lines.append("| LR | Baseline | High-FT | Low-FT | z-test (base vs high) | Paradox? |")
        lines.append("|:---|:---|:---|:---|:---|:---:|")
        for lr_str, results in sorted(sweep_data.items()):
            b = results.get('baseline')
            h = results.get('high')
            z_val, p_val = float('nan'), float('nan')
            if b and h:
                z_val, p_val = two_proportion_ztest(
                    b['faithfulness']/100, b['n'],
                    h['faithfulness']/100, h['n'],
                )
            paradox = ""
            if b and h:
                paradox = "YES" if h['faithfulness'] < b['faithfulness'] else "NO"
            b_str = f"{b['faithfulness']:.1f}%" if b else "N/A"
            h_str = f"{h['faithfulness']:.1f}%" if h else "N/A"
            l_str = f"{results['low']['faithfulness']:.1f}%" if results.get('low') else "N/A"
            p_str = f"p={p_val:.4f}" if not np.isnan(p_val) else "N/A"
            lines.append(f"| {lr_str} | {b_str} | {h_str} | {l_str} | {p_str} | {paradox} |")
    else:
        lines.append("**Status**: LR sweep data not yet available.")
        lines.append("Run `lr_robustness_jobs.sh` on BigRed200 to generate it.")
        lines.append("")
        lines.append("Expected output directories:")
        for lr_str, dirs in LLAMA3B_LR_SWEEP.items():
            if lr_str == '3e-05':
                continue
            for cond, dirpath in dirs.items():
                lines.append(f"  - `{dirpath}/results.jsonl`")

    # Mistral calibrated-LR check
    lines.append("")
    lines.append("## 3b. Mistral-7B Calibrated-LR Rerun (lr=5e-06)")
    lines.append("")

    mistral_cal = {}
    mistral_cal_available = False
    for cond, dirpath in MISTRAL_CALIBRATED.items():
        records = load_condition(dirpath)
        if records is not None:
            flips = binary_faithfulness(records)
            n = len(flips)
            mistral_cal[cond] = {
                'n': n, 'flips': sum(flips),
                'faithfulness': sum(flips)/n*100 if n > 0 else 0,
            }
            mistral_cal_available = True
        else:
            mistral_cal[cond] = None

    if mistral_cal_available:
        b_orig = analyze_model('Mistral-7B', MODELS['Mistral-7B'])
        lines.append("| Condition | lr=1e-05 (original) | lr=5e-06 (calibrated) |")
        lines.append("|:----------|:---:|:---:|")
        for cond in ['baseline', 'high', 'low']:
            orig = b_orig.get(cond)
            cal = mistral_cal.get(cond)
            o_str = f"{orig['faithfulness']:.1f}% ({orig['flips']}/{orig['n']})" if orig else "N/A"
            c_str = f"{cal['faithfulness']:.1f}% ({cal['flips']}/{cal['n']})" if cal else "N/A"
            lines.append(f"| {cond} | {o_str} | {c_str} |")

        if mistral_cal.get('baseline') and mistral_cal.get('high'):
            z, p = two_proportion_ztest(
                mistral_cal['baseline']['faithfulness']/100, mistral_cal['baseline']['n'],
                mistral_cal['high']['faithfulness']/100, mistral_cal['high']['n'],
            )
            sig = "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            lines.append(f"\nCalibrated z-test (baseline vs high): z={z:.3f}, p={p:.4f} {sig}")
    else:
        lines.append("**Status**: Calibrated-LR data not yet available.")
        lines.append("Run `lr_robustness_jobs.sh` to generate Mistral results at lr=5e-06.")

    # Conclusion
    lines.append("")
    lines.append("## 4. Conclusion")
    lines.append("")
    if all_match:
        lines.append("All models used calibrated LRs. No confound detected.")
    else:
        lines.append("### Status: Partially resolved, one confound identified")
        lines.append("")
        lines.append("**LLaMA-3-3B, Phi-3, LLaMA-3-8B**: Evaluated at their paper-calibrated LRs.")
        lines.append("Results for these three models are clean.")
        lines.append("")
        lines.append("**Mistral-7B**: Evaluated at lr=1e-05, which is 2× the paper-calibrated")
        lines.append("lr=5e-06. At lr=1e-05, the Tutek et al. ablation shows specificity drops")
        lines.append("to ~86-91% (below the 95% threshold used for LR selection). This means")
        lines.append("the Mistral result (16%→4%) **may be partially confounded** by over-")
        lines.append("aggressive unlearning. A rerun at lr=5e-06 is needed to confirm the")
        lines.append("paradox direction for Mistral specifically.")
        lines.append("")
        lines.append("**paper.tex methods section** claims a single LR of 3×10⁻⁵ for all models.")
        lines.append("This is wrong regardless of which scenario — the actual LRs varied by model.")
        lines.append("This must be corrected before submission.")
        lines.append("")
        lines.append("### Required actions")
        lines.append("1. Correct paper.tex §Faithfulness Evaluation to list per-model LRs")
        lines.append("2. Rerun Mistral-7B baseline/high/low at lr=5e-06 (calibrated LR)")
        lines.append("3. Run LLaMA-3-3B LR sweep (1e-05, 5e-05) for additional robustness")
        lines.append("4. Commit submit_multi_model.sh to git for reproducibility")

    summary_path = 'analysis/LR_ROBUSTNESS_SUMMARY.md'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nSaved: {summary_path}")


if __name__ == '__main__':
    main()
