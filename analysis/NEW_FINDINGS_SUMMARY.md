# New Findings Summary

Generated from `extended_analyses.py` on the 16 model×dataset reproduction runs.  
All data: `analysis/continuous_faithfulness.csv` (3,699 instances).

> **Data caveat:** Mistral-2 and Phi-3 `initial_probs` are raw (unnormalized) token  
> likelihoods, not softmax probabilities. All delta_p computations normalize  
> `probs / sum(probs)` before differencing. LLaMA models are already normalized.

---

## Finding 1 — Continuous delta_p correlates with binary faithfulness, but weakly

**What we found:**  
Point-biserial correlation between binary faithfulness (flip/no-flip) and continuous  
delta_p (change in correct-answer probability after unlearning):

| Model | r | p |
|-------|:-:|:-:|
| LLaMA-3-8B | 0.318 | 5.2e-23 |
| LLaMA-3-3B | 0.231 | 7.5e-13 |
| Mistral-7B | 0.266 | 2.3e-16 |
| Phi-3 | 0.101 | 2.2e-03 |
| **All models** | **0.335** | **7.2e-98** |

The correlations are statistically significant but moderate (r ≈ 0.1–0.3). The gap  
between r=1 (perfect) and the observed values is meaningful: many instances have  
substantial delta_p without crossing the decision boundary (subcritical faithfulness,  
see Finding 3).

**Robust?** Yes — LLaMA-3-3B at correct LR (3e-05) shows r=0.231. The correlation  
holds across all four models including the two with LR mismatch.

**Novel?** Tutek et al. (2025) use only binary faithfulness. Yee et al. (arXiv:2405.15092)  
also use binary metrics. Lanham et al. (2023) measure token-level probability shifts  
but not this continuous delta_p framing. The moderate r is a new quantitative result  
showing binary faithfulness is a lossy summary of continuous causal influence.

---

## Finding 2 — Phi-3's delta_p is net negative (steps actively boost correct answer)

**What we found:**  
Distribution of delta_p (positive = step hurts correct answer = faithful):

| Model | Mean Δp | %Positive | %Negative |
|-------|:-:|:-:|:-:|
| LLaMA-3-8B | +0.248 | 69.2% | 30.8% |
| LLaMA-3-3B | +0.113 | 60.5% | 39.5% |
| Mistral-7B | +0.333 | 71.6% | 28.4% |
| **Phi-3** | **−0.004** | **32.2%** | **67.8%** |

Phi-3 is the only model where the majority of CoT steps (67.8%) *increase* the  
model's confidence in the correct answer when unlearned — meaning the steps were  
actively misleading, and removing them helps. Mean delta_p is essentially zero  
(−0.004), confirming Phi-3's CoT is near-random in its causal influence.

**Robust?** Yes for Phi-3's direction — but magnitudes are confounded by low LR  
(efficacy only 11–21% for Phi-3). At correct LR (1e-04), we'd expect larger  
delta_p values. The % positive/negative direction should hold.

**Novel?** Tutek et al. note Phi-3's low binary faithfulness but do not characterize  
the *direction* of causal influence. The finding that 67.8% of Phi-3's CoT steps  
actively mislead (boosting correct answer when removed) is new.

---

## Finding 3 — Binary faithfulness undercounts causal influence by 27–80%

**What we found (subcritical faithfulness):**

| Model | Subcritical | Suprathreshold | Misleading | Binary undercounts by |
|-------|:-:|:-:|:-:|:-:|
| LLaMA-3-8B | 25.1% | 44.1% | 30.8% | **36.3%** of faithful signal |
| LLaMA-3-3B | 31.6% | 28.9% | 39.5% | **52.2%** of faithful signal |
| Mistral-7B | 19.8% | 51.8% | 28.4% | **27.7%** of faithful signal |
| Phi-3 | 25.7% | 6.5% | 67.8% | **79.7%** of faithful signal |

"Subcritical" = delta_p > 0 (step does reduce correct-answer probability) but not  
enough to cross the decision boundary (no prediction flip).

For LLaMA-3-3B (correct LR), binary faithfulness captures only 47.8% of instances  
where the step demonstrably reduces correct-answer probability. The remaining 52.2%  
show real causal influence that binary metrics miss.

**Robust?** Strongly robust for LLaMA-3-3B. Confounded for Phi-3 (low LR → small  
magnitudes → more subcritical that would flip at higher LR) and Mistral (high LR →  
already pushes most faithful instances over threshold).

**Novel?** This framing does not appear in Tutek et al. (2025), Yee et al. (2024),  
Siegel et al. (2024), or Lanham et al. (2023). All use binary prediction-flip as  
the faithfulness criterion. Subcritical faithfulness is a new concept showing that  
binary metrics systematically undercount causal influence.

---

## Finding 4 — Faithfulness trajectories reveal per-model LR sensitivity patterns

**What we found:**  
Per-epoch trajectories of efficacy, specificity, binary faithfulness, and delta_p  
across epochs 0–5:

| Model | Peak faithfulness epoch | Specificity at epoch 5 | Degrades? |
|-------|:-:|:-:|:-:|
| LLaMA-3-8B | 5 | degrades steadily | Stable (no peak-then-drop) |
| LLaMA-3-3B | 5 | degrades slowly | Stable |
| Mistral-7B | **2** | collapses by epoch 5 | **YES** — peaks then degrades |
| Phi-3 | **3** | near-perfect throughout | Weak peak, trivially stable |

For Mistral-7B (LR 6× too high): faithfulness peaks at epoch 2 then slowly declines  
as specificity collapses — the LR sensitivity problem manifests *within a single run*,  
not just across different LR runs. This supports early stopping as a post-hoc fix.

For Phi-3 (LR 3× too low): faithfulness peaks at epoch 3 at ~10.9%, then barely  
declines. Efficacy never builds enough momentum to cause meaningful flips.

**Robust?** Mistral finding is robust — the trajectory shape is visible across all  
4 datasets. Phi-3 finding may understate the effect (at correct LR, we'd expect  
the peak to be much higher and possibly at an earlier epoch).

**Novel?** Tutek et al. select the best epoch via held-out LR sweep but do not  
report within-run epoch trajectories. The Mistral peak-at-epoch-2 pattern is a  
new observation suggesting early stopping on specificity is an alternative to  
LR calibration.

---

## Finding 5 — Accuracy is NOT monotone with faithfulness quartile (LLaMA-3-3B, Phi-3)

**What we found:**  
Splitting instances by delta_p quartile and reporting accuracy rate:

| Model | Q1 (least faithful) | Q2 | Q3 | Q4 (most faithful) | Monotone? |
|-------|:-:|:-:|:-:|:-:|:-:|
| LLaMA-3-8B | 36.5% | 65.7% | 85.2% | 100.0% | Yes |
| **LLaMA-3-3B** | **33.6%** | **63.0%** | **60.0%** | **95.3%** | **No (Q3 dip)** |
| Mistral-7B | 5.2% | 74.3% | 98.7% | 100.0% | Yes |
| **Phi-3** | **71.3%** | **93.9%** | **82.2%** | **49.1%** | **No (inverted!)** |

LLaMA-3-3B (correct LR): shows a non-monotone dip at Q3, confirming that  
faithfulness and correctness are dissociable even at continuous granularity.

Phi-3: the pattern is *inverted* — more faithful steps (higher delta_p) correspond  
to *lower* accuracy. This is the continuous-domain version of the earlier finding  
that Phi-3 answers correctly by bypassing its CoT: instances where CoT steps have  
high causal influence (large delta_p) are ones where the CoT is actively leading  
Phi-3 toward the wrong answer.

**Robust?** LLaMA-3-3B non-monotonicity is robust (correct LR). Phi-3 inversion  
is confounded by LR but the *direction* (high delta_p → lower accuracy) is  
consistent with Phi-3's unfaithful CoT hypothesis.

**Novel?** The inverted accuracy-faithfulness gradient for Phi-3 in the continuous  
domain is not reported in any of the four reference papers. It provides the  
strongest quantitative support yet for the hypothesis that Phi-3's CoT leads it  
astray rather than helping.

---

## Finding 6 — CoT length moderates first-step faithfulness magnitude

**What we found:**  
Binning by CoT length (all instances have step_idx=0, so this is the only  
positional analysis possible):

| CoT length | Mean Δp | Mean binary faithfulness | N |
|------------|:-:|:-:|:-:|
| 1–2 sentences | 0.271 | 56.2% | 523 |
| 3–4 sentences | 0.250 | 57.8% | 1035 |
| 5–7 sentences | 0.141 | 41.1% | 1081 |
| 8+ sentences  | 0.077 | 34.2% | 1060 |

Both magnitude (delta_p) and rate (binary) decrease monotonically as CoT length  
increases. For 8+ sentence CoTs, the first step has less than a third the causal  
influence (delta_p 0.077 vs 0.271) of first steps in short CoTs.

**Robust?** Yes — the trend holds across all models combined. LLaMA-3-8B  
generates shorter CoTs (avg 3.5 sentences) while LLaMA-3-3B generates longer  
ones (avg 6.1 sentences); this length difference partially explains the ~17pp  
faithfulness gap between the models.

**Novel?** Tutek et al. do not analyze CoT length as a moderator. Lanham et al.  
(2023) find that later steps matter more for mistake-adding, but do not study  
length effects. This is a new structural finding about how CoT density modulates  
causal faithfulness.

---

## What to Run Next on BigRed200

To make Findings 2, 3, and 5 fully robust, we need Phi-3 and Mistral runs at  
their correct LRs. See `new_runs.py` for the SLURM commands.

Priority order:
1. **Phi-3 at lr=1e-04** (ARC, OpenBook) — will clarify whether inverted  
   accuracy-faithfulness gradient holds at correct efficacy
2. **Phi-3 at lr=5e-05** (Sports, SQA) — correct LR for these datasets  
3. **Mistral-7B at lr=5e-06** (all 4) — will show whether early-stopping  
   finding holds or was an artifact of over-training
4. **Mistral-7B at lr=1e-05** (all 4) — intermediate for comparison
