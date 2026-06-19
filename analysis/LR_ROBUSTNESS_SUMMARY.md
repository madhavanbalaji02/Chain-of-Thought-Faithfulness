# LR Robustness Check: Alignment-Faithfulness Paradox

## 1. LR Calibration Audit

**Question**: Were the fine-tuning evaluation runs at a single fixed LR,
or were per-model calibrated LRs used?

| Model | Paper Best LR | Eval LR Used | Match? | Source Script |
|:------|:---:|:---:|:---:|:---|
| LLaMA-3-3B | 3e-05 | 3e-05 | YES | `new_experiment_jobs.py` |
| LLaMA-3-8B | 1e-05 | 1e-05 | YES | `submit_multi_model.sh` |
| Mistral-7B | 5e-06 | 1e-05 | **NO** | `sacct job 6815741-6815743 (NOT submit_multi_model.sh)` |
| Phi-3 | 1e-04 | 1e-04 | YES | `submit_multi_model.sh` |

**Finding**: Not all models were evaluated at their paper-calibrated LRs.

Evidence source: SLURM `sacct --format=SubmitLine` on BigRed200, which
records the exact `sbatch --wrap=...` command for each job. This is
authoritative — it reflects what SLURM actually executed, not what any
script file currently contains.

**Mistral-7B discrepancy**: `submit_multi_model.sh` specifies `--lr 5e-06`,
but the SLURM jobs that produced the final results (6815741-6815743,
submitted Apr 10) used `--lr 1e-05`. The initial Mistral runs (6754962-6754964,
Apr 6) all CANCELLED or FAILED. The successful retries on Apr 10 used a
different command (likely typed manually or from an edited script).

The Tutek et al. ablation shows Mistral at lr=5e-06 gives ~57% efficacy
with ~96% specificity; at lr=1e-05 efficacy rises to ~70% but specificity
drops to ~86-91%. This means the Mistral results may have inflated flip
rates from specificity degradation at the higher LR.

**paper.tex line 348** states a single LR of 3×10⁻⁵ for all models.
The actual LRs were: LLaMA-3-3B 3e-05, Phi-3 1e-04, Mistral 1e-05,
LLaMA-8B 1e-05. The methods section must be corrected.

## 2. Existing Results (per-model calibrated LRs)

| Model | Eval LR | Baseline | High-FT | Low-FT | z-test (base vs high) | Rel. Drop |
|:------|:---:|:---|:---|:---|:---|:---:|
| LLaMA-3-3B | 3e-05 | 30.0% (15/50) | 8.0% (4/50) | 28.0% (14/50) | p=0.0050 ** | -73% |
| LLaMA-3-8B | 1e-05 | 3.8% (2/53) | 4.0% (2/50) | 4.0% (2/50) | p=0.9526 n.s. | +6% |
| Mistral-7B | 1e-05 | 16.0% (8/50) | 4.0% (2/50) | 22.0% (11/50) | p=0.0455 * | -75% |
| Phi-3 | 1e-04 | 32.0% (16/50) | 14.0% (7/50) | 36.0% (18/50) | p=0.0325 * | -56% |

## 3. LLaMA-3-3B LR Sensitivity Sweep

**Status**: LR sweep data not yet available.
Run `lr_robustness_jobs.sh` on BigRed200 to generate it.

Expected output directories:
  - `finetuned_results_calibrated_lr/llama3b_lr1e-05/baseline/results.jsonl`
  - `finetuned_results_calibrated_lr/llama3b_lr1e-05/high_quality/results.jsonl`
  - `finetuned_results_calibrated_lr/llama3b_lr1e-05/low_quality/results.jsonl`
  - `finetuned_results_calibrated_lr/llama3b_lr5e-05/baseline/results.jsonl`
  - `finetuned_results_calibrated_lr/llama3b_lr5e-05/high_quality/results.jsonl`
  - `finetuned_results_calibrated_lr/llama3b_lr5e-05/low_quality/results.jsonl`

## 3b. Mistral-7B Calibrated-LR Rerun (lr=5e-06)

**Status**: Calibrated-LR data not yet available.
Run `lr_robustness_jobs.sh` to generate Mistral results at lr=5e-06.

## 4. Conclusion

### Status: Partially resolved, one confound identified

**LLaMA-3-3B, Phi-3, LLaMA-3-8B**: Evaluated at their paper-calibrated LRs.
Results for these three models are clean.

**Mistral-7B**: Evaluated at lr=1e-05, which is 2× the paper-calibrated
lr=5e-06. At lr=1e-05, the Tutek et al. ablation shows specificity drops
to ~86-91% (below the 95% threshold used for LR selection). This means
the Mistral result (16%→4%) **may be partially confounded** by over-
aggressive unlearning. A rerun at lr=5e-06 is needed to confirm the
paradox direction for Mistral specifically.

**paper.tex methods section** claims a single LR of 3×10⁻⁵ for all models.
This is wrong regardless of which scenario — the actual LRs varied by model.
This must be corrected before submission.

### Required actions
1. Correct paper.tex §Faithfulness Evaluation to list per-model LRs
2. Rerun Mistral-7B baseline/high/low at lr=5e-06 (calibrated LR)
3. Run LLaMA-3-3B LR sweep (1e-05, 5e-05) for additional robustness
4. Commit submit_multi_model.sh to git for reproducibility
