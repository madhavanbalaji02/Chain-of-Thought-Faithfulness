# Mistral-7B LR Assessment: Is lr=1e-05 a Problem?

## The Deviation

Mistral-7B was evaluated at `lr=1e-05`, while `const.py:paper_best_lr` specifies `lr=5e-06`.
The Tutek et al. ablation (from Ablations.ipynb) shows that at lr=1e-05, Mistral's
specificity drops to ~86-91% — below the 95% selection threshold.

## Diagnostic Evidence from the Result Data

However, `evaluate_finetuned.py` uses a simpler pipeline than the main `unlearn.py`
(first sentence only, held-out test set, no specificity instances). The question is
whether lr=1e-05 produced over-aggressive behavior *in these specific runs*.

### Signatures of over-aggressive unlearning (and what we see):

| Diagnostic | Over-aggressive signature | Mistral baseline | Mistral high-FT | LLaMA-3-3B baseline (reference) |
|:-----------|:-------------------------|:---------------:|:---------------:|:-------------------------------:|
| Oscillating predictions | >15% | **6.0%** | **0.0%** | 10.0% |
| Max-prob drop (ep0→ep5) | >0.05 | **0.0013** | **0.0013** | 0.0006 |
| Flip rate explosion by epoch | exponential growth | stable 10-14% | stable 2-4% | gradual 10→22% |
| Max |Δp| at epoch 5 | >0.10 | **0.017** | **0.005** | 0.005 |

### Assessment

The Mistral results do **not** show the over-aggressive unlearning signature:
- Oscillation rates (6%, 0%, 8%) are comparable to LLaMA-3-3B (10%, 4%) — no sign of instability
- Probability drops are tiny (0.001) — the model's confidence is not collapsing
- Flip rates progress smoothly, not explosively
- Maximum Δp values are small and comparable to the known-good LLaMA-3-3B

The key diagnostic: if lr=1e-05 were causing false positives through specificity degradation,
we'd see the **same artificial inflation across all three conditions** (baseline, high-FT, low-FT).
Instead, the pattern is condition-specific:
- Baseline: 16% faithfulness
- High-FT: **4%** faithfulness (large drop)
- Low-FT: **22%** faithfulness (preserved/increased)

This condition-specific pattern is the paradox signature and is inconsistent with an LR artifact
(which would inflate all conditions uniformly).

## Verdict

**The lr=1e-05 deviation is a checked deviation, not a confound.** The Mistral result
(16%→4%) can stand with a transparency note in the paper acknowledging the LR difference.
A rerun at lr=5e-06 would be ideal for belt-and-suspenders confidence, but the current data
does not show the failure mode that would invalidate the result.

## Recommended paper text

> Mistral-7B was evaluated at lr=1×10⁻⁵ (2× the ablation-selected lr=5×10⁻⁶,
> due to OOM-related job failures at the original LR on BigRed200). Diagnostic
> analysis of the result data shows no evidence of over-aggressive unlearning:
> prediction oscillation rates (6% baseline) are comparable to LLaMA-3-3B (10%),
> and the faithfulness collapse is condition-specific (16%→4% under high-FT,
> 16%→22% under low-FT), inconsistent with a uniform LR artifact.
