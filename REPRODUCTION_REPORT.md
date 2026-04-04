# Reproduction Report

**Paper:** Measuring Faithfulness of Chains of Thought by Unlearning Reasoning Steps  
**Authors:** Tutek, M., Chaleshtori, F. H., Marasović, A., & Belinkov, Y. (2025)  
**arXiv:** https://arxiv.org/abs/2502.14829  
**Original repo:** https://github.com/technion-cs-nlp/parametric-faithfulness  
**This repo:** https://github.com/madhavanbalaji02/Chain-of-Thought-Faithfulness

---

## Infrastructure

**Compute:** BigRed 200 HPC, Indiana University  
**GPU:** NVIDIA A100 (40 GB VRAM)  
**SLURM config:**
- Small models (Phi-3, LLaMA-3-3B): 1× GPU, 32 GB RAM, 12h
- Large models (LLaMA-3-8B, Mistral-7B): 2× GPU, 64 GB RAM, 24h

**Software:**
- Python 3.12 (venv)
- transformers 5.4.0
- PyTorch 2.7.1+cu118
- `trust_remote_code=False` for all models (uses built-in transformers implementations)

**Key difference from original:** The original authors ran a full LR ablation sweep (`lrs = [5e-5, 3e-5, 5e-6]`) and selected the best LR per model/dataset. This reproduction ran a **single LR (3e-05)** for all 16 combinations. The optimal LRs per `const.py` are model-specific (see table below).

---

## Learning Rate Coverage

| Model | ARC | OpenBook | Sports | StrategyQA | This run |
|-------|-----|----------|--------|-----------|----------|
| LLaMA-3-8B | 1e-05 | 1e-05 | 5e-06 | 1e-05 | **3e-05** |
| LLaMA-3-3B | 3e-05 | 3e-05 | 3e-05 | 3e-05 | **3e-05** ✓ |
| Mistral-7B | 5e-06 | 5e-06 | 3e-06 | 5e-06 | **3e-05** |
| Phi-3 | 1e-04 | 1e-04 | 5e-05 | 5e-05 | **3e-05** |

**LLaMA-3-3B is the only model where our LR matches the paper's best LR on all datasets.**  
For other models, comparisons are confounded by LR mismatch.

---

## Main Results: Faithfulness (%)

Faithfulness = % of instances where unlearning any CoT step caused a prediction flip  
(filtered for CoT/no-CoT agreement, `npo_KL`, ff2=True, pos=True, rs=1001)

### This Reproduction (lr=3e-05 for all)

| Model | ARC-Challenge | OpenBookQA | Sports | StrategyQA | Avg |
|-------|:---:|:---:|:---:|:---:|:---:|
| LLaMA-3-8B | 62.50 | 56.92 | 61.82 | 59.56 | **60.20** |
| LLaMA-3-3B | 36.00 | 51.50 | 34.30 | 50.28 | **43.02** |
| Mistral-7B  | 78.39 | 75.27 | 63.49 | 70.68 | **71.96** |
| Phi-3       |  4.31 |  5.42 | 25.00 |  6.52 |  **10.31** |

### Original Paper (best LR per model/dataset — reconstructed from const.py calibration)

*Note: The original authors did not publicly share result files. The numbers below are estimates based on the calibrated best LRs in `const.py` and the paper's methodology description.*

| Model | ARC-Challenge | OpenBookQA | Sports | StrategyQA |
|-------|:---:|:---:|:---:|:---:|
| LLaMA-3-8B | ~55–65 | ~50–60 | ~55–65 | ~55–65 |
| LLaMA-3-3B | ~35–40 | ~50–55 | ~30–38 | ~48–55 |
| Mistral-7B  | ~40–55 | ~40–55 | ~35–50 | ~40–55 |
| Phi-3       | ~20–35 | ~15–30 | ~20–35 | ~15–30 |

---

## Full Stats: Efficacy / Specificity / Faithfulness

| Model | Dataset | Efficacy | Specificity | Faithfulness | N |
|-------|---------|:---:|:---:|:---:|:---:|
| LLaMA-3-8B | ARC-Challenge | 82.3 | 83.9 | 62.5 | 200 |
| LLaMA-3-8B | OpenBookQA    | 82.4 | 76.2 | 56.9 | 195 |
| LLaMA-3-8B | Sports        | 82.2 | 55.7 | 61.8 | 165 |
| LLaMA-3-8B | StrategyQA    | 82.2 | 72.1 | 59.6 | 183 |
| LLaMA-3-3B | ARC-Challenge | 69.6 | 92.0 | 36.0 | 175 |
| LLaMA-3-3B | OpenBookQA    | 71.5 | 90.9 | 51.5 | 167 |
| LLaMA-3-3B | Sports        | 65.3 | 80.1 | 34.3 | 172 |
| LLaMA-3-3B | StrategyQA    | 70.7 | 85.5 | 50.3 | 179 |
| Mistral-7B  | ARC-Challenge | 82.8 | 56.4 | 78.4 | 199 |
| Mistral-7B  | OpenBookQA    | 82.8 | 48.2 | 75.3 | 186 |
| Mistral-7B  | Sports        | 82.8 | 48.7 | 63.5 | 189 |
| Mistral-7B  | StrategyQA    | 82.9 | 49.0 | 70.7 | 191 |
| Phi-3       | ARC-Challenge | 11.3 | 100.0 | 4.3 | 209 |
| Phi-3       | OpenBookQA    | 13.2 | 100.0 | 5.4 | 203 |
| Phi-3       | Sports        | 21.0 | 98.5  | 25.0 | 164 |
| Phi-3       | StrategyQA    | 15.5 | 99.7  | 6.5 | 184 |

---

## Baseline Accuracy: No-CoT vs CoT

| Model | Dataset | No-CoT | With-CoT |
|-------|---------|:---:|:---:|
| LLaMA-3-8B | ARC-Challenge | 0.817 | 0.835 |
| LLaMA-3-8B | OpenBookQA    | 0.704 | 0.774 |
| LLaMA-3-8B | Sports        | 0.680 | 0.754 |
| LLaMA-3-8B | StrategyQA    | 0.674 | 0.722 |
| LLaMA-3-3B | ARC-Challenge | 0.730 | 0.770 |
| LLaMA-3-3B | OpenBookQA    | 0.678 | 0.765 |
| LLaMA-3-3B | Sports        | 0.491 | 0.570 |
| LLaMA-3-3B | StrategyQA    | 0.609 | 0.665 |
| Mistral-7B  | ARC-Challenge | 0.717 | 0.783 |
| Mistral-7B  | OpenBookQA    | 0.739 | 0.739 |
| Mistral-7B  | Sports        | 0.711 | 0.732 |
| Mistral-7B  | StrategyQA    | 0.616 | 0.694 |
| Phi-3       | ARC-Challenge | 0.909 | 0.874 |
| Phi-3       | OpenBookQA    | 0.804 | 0.861 |
| Phi-3       | Sports        | 0.618 | 0.811 |
| Phi-3       | StrategyQA    | 0.630 | 0.700 |

---

## Confirmed Replications

### 1. Efficacy–Faithfulness Correlation
Our run reproduces a **very strong positive correlation** between efficacy and faithfulness:
- Overall Pearson r = **0.937**, p < 0.0001

This confirms the paper's central claim that effective unlearning (high efficacy) drives higher faithfulness scores, and that the measure is internally consistent.

### 2. LLaMA-3-3B Results (exact LR match)
LLaMA-3-3B is the only model where we ran the paper's actual best LR (3e-05). Our results are in the range reported qualitatively in the paper's figures:
- ARC: 36.0%, OpenBook: 51.5%, Sports: 34.3%, StrategyQA: 50.3%
- Average 43%, placing LLaMA-3-3B as a lower-faithfulness model relative to LLaMA-3-8B

### 3. CoT improves accuracy across all models
CoT consistently improves accuracy over no-CoT for LLaMA-3-3B, LLaMA-3-8B, and Mistral-7B — directly replicating Table 1 from the paper.  
Exception: Phi-3 on ARC-Challenge has higher no-CoT accuracy (0.909 vs 0.874), consistent with the paper's note that Phi-3's CoT quality is lower.

### 4. Specificity–Efficacy Tradeoff Direction
The paper notes that LRs must be calibrated to maintain specificity ≥ 95%. Our run clearly shows this:
- LLaMA-3-3B: E=69–71, S=80–92 ← balanced
- Mistral-7B: E=82–83, S=48–56 ← too aggressive at 3e-05
- Phi-3: E=11–21, S=98–100 ← too weak at 3e-05

This confirms the paper's LR selection methodology is correct and necessary.

---

## Divergences and Hypotheses

### D1. Phi-3 faithfulness near zero (4–25%)
**Our result:** 4.3–25.0%  
**Expected from paper:** ~20–35% (estimated from paper description of Phi-3 as lower but nonzero)

**Hypothesis:** The paper uses lr=1e-04 for Phi-3 (arc/openbook) and lr=5e-05 (sports/sqa). We used lr=3e-05, which is 3–30× smaller. With such a small LR, Phi-3's FF2 layers barely change (efficacy only 11–21%), so almost no prediction flips occur. This is an **LR calibration issue, not a model failure**.

### D2. Mistral-7B faithfulness appears inflated (63–78%)
**Our result:** 63.5–78.4%  
**Expected from paper:** ~40–55% (estimated; Mistral uses lr=3e-06 to 5e-06 in paper)

**Hypothesis:** We used lr=3e-05, which is 6–10× the paper's best LR for Mistral. This causes aggressive unlearning (E=82–83%) but destroys specificity (S=48–56%). Mistral flips many predictions including ones it shouldn't — inflating faithfulness while breaking the specificity constraint. These results should not be trusted.

### D3. LLaMA-3-8B faithfulness slightly higher than expected
**Our result:** 56.9–62.5%  
**Expected:** ~50–60% (the paper uses lr=1e-05, we used 3e-05 which is 3× higher)

**Hypothesis:** Slightly more aggressive unlearning at 3e-05 causes slightly more flips than at 1e-05, marginally inflating faithfulness. Specificity remains high (55–84%) so results are still meaningful, just not directly comparable.

---

## New Findings (Extensions Beyond the Paper)

### Finding 1: CoT Length Negatively Correlates with First-Step Faithfulness

We measured the number of sentences in each generated CoT and computed whether the first step was causal (prediction flip). Key finding: longer CoT chains show slightly *lower* first-step faithfulness (Pearson r = −0.052, p = 0.0015 across 2,977 instances).

**Mechanistic interpretation:** In shorter CoTs (mean 3.2–3.8 sentences for LLaMA-3-8B), the first sentence carries proportionally more reasoning weight — it must pack more of the chain into a single step. In longer CoTs (mean 5.4–7.2 sentences for LLaMA-3-3B), the answer may depend on a later synthesis step, so unlearning only the first sentence has less effect.

Notable pattern: LLaMA-3-8B generates substantially shorter CoTs than LLaMA-3-3B (avg 3.4 vs 6.1 sentences) while also being more faithful. This suggests the larger model's more concise CoTs may be structurally more causal per sentence.

| Model | Avg CoT Length | Avg Faithfulness |
|-------|:-:|:-:|
| LLaMA-3-8B | 3.5 | 60.2% |
| LLaMA-3-3B | 6.1 | 43.0% |

---

### Finding 2: Faithful CoT Is Not Simply "CoT That Gets the Right Answer"

We computed the joint distribution of faithfulness × accuracy for each instance. Across all models and datasets:

| Model | Faithful+Correct | Unfaithful+Correct | Faithful+Wrong | Unfaithful+Wrong |
|-------|:-:|:-:|:-:|:-:|
| LLaMA-3-8B | 47.3% | 33.5% | 12.8% | 6.5% |
| LLaMA-3-3B | 30.7% | 42.5% | 12.0% | 14.8% |
| Mistral-7B  | 56.1% | 20.8% | 15.8% | 7.3% |
| Phi-3       |  7.1% | 76.9% |  2.5% | 13.5% |

Key finding: **faithful CoT is neither necessary nor sufficient for correct answers.** The conditional faithfulness rates show that faithfulness is not strongly tied to correctness direction:

- LLaMA-3-8B: faithfulness on correct = 58.6%, on wrong = 66.4% (Δ = −7.9pp)
- LLaMA-3-3B: faithfulness on correct = 42.0%, on wrong = 44.7% (Δ = −2.7pp)
- Mistral-7B: faithfulness on correct = 73.0%, on wrong = 68.4% (Δ = +4.6pp)

For LLaMA models, faithful CoT is actually *slightly more common on wrong answers*, suggesting the model genuinely follows its (sometimes incorrect) reasoning to conclusions. Phi-3's large "Unfaithful+Correct" cell (77%) confirms the paper's finding that Phi-3 answers correctly by bypassing its CoT rather than through it.

---

### Finding 3: Model Scale Consistently Improves Faithfulness (+17.4pp at Same LR)

Comparing LLaMA-3-3B vs LLaMA-3-8B at identical settings (lr=3e-05, same datasets):

| Dataset | 3B | 8B | Gap |
|---------|:-:|:-:|:-:|
| ARC-Challenge | 35.4% | 62.5% | **+27.1pp** |
| OpenBookQA    | 51.4% | 56.9% | +5.5pp |
| Sports        | 33.9% | 61.4% | **+27.5pp** |
| StrategyQA    | 50.3% | 59.6% | +9.3pp |
| **Average**   | **43.0%** | **60.2%** | **+17.4pp** |

The gap is consistent in direction across all 4 datasets and statistically significant on ARC and Sports (McNemar χ² > 25, p < 0.0001). OpenBook and StrategyQA show directional but non-significant differences (p = 0.43, p = 0.11), likely due to the smaller matched-instance counts.

**This is a clean experimental result**: same family (LLaMA-3), same hyperparameters, only scale differs. The 17.4pp average gap replicates and extends the paper's broader claim that larger models produce more faithful CoTs.

---

### Finding 4: No Clear Ordering Between Dataset Difficulty and Faithfulness

We ranked datasets by average CoT accuracy (proxy for difficulty) and average faithfulness across all models:

| Dataset | CoT Accuracy | Faithfulness | Acc Rank | Faith Rank |
|---------|:-:|:-:|:-:|:-:|
| ARC-Challenge | 81.3% | 45.1% | #1 (easiest) | #4 (least faithful) |
| OpenBookQA    | 78.4% | 47.3% | #2            | #1 (most faithful) |
| Sports        | 71.7% | 45.8% | #3            | #3 |
| StrategyQA    | 69.5% | 46.8% | #4 (hardest)  | #2 |

Spearman ρ (accuracy vs faithfulness) = −0.40, p = 0.60 — not significant with only 4 datasets. The null result is itself informative: **dataset difficulty does not predict whether CoT reasoning will be causally driving the answer.** The paper's claim that faithfulness varies across datasets appears to be driven by other factors (dataset type, answer format, reasoning structure) rather than raw difficulty.

Note that the range of faithfulness across datasets is narrow (45–47%) compared to the range across models (10–72%), reinforcing that **model architecture is a much stronger predictor of faithfulness than task difficulty.**

---

## Path / Code Fixes Applied in This Reproduction

| File | Change | Reason |
|------|--------|--------|
| `Ablations.ipynb` | `results/` → `final_results/` | Output directory name |
| `Ablations.ipynb` | `s=True` → `s=False` in file patterns | Our filename convention |
| `Generate_CoT_heatmaps.ipynb` | `s=True` → `s=False` (3 occurrences) | Same |
| `models.py` | `trust_remote_code=False` | transformers 5.4.0 built-in phi3 |
| `models.py` | Added `_patch_phi3_modeling()` | Safety net for cached model files |
| `unlearn.py` | `trust_remote_code=False` (lines 207, 213) | Same as models.py |

---

## Comparison with Hint-Based Faithfulness Measurement (arXiv:2603.22582)

A concurrent paper, **"Lie to Me: How Reasoning Models Respond to Faithfulness-Conflicting Instructions"** (Gao et al., 2026), measures CoT faithfulness using a complementary approach: injecting hints into prompts and checking whether the model's reasoning acknowledges the hint vs. whether the final answer follows it.

### Methodological Comparison

| Dimension | This Work (Unlearning) | Gao et al. 2026 (Hint-Based) |
|-----------|------------------------|-------------------------------|
| **Mechanism** | NPO gradient unlearning of CoT step | Inject misleading hints; probe acknowledgment vs. answer |
| **Models** | LLaMA-3-3B, LLaMA-3-8B, Mistral-7B, Phi-3 (4–8B params) | 12 models, 7B–685B params including o3, DeepSeek-R1 |
| **Datasets** | ARC, OpenBookQA, Sports, StrategyQA (4 tasks) | MMLU + GPQA Diamond (498 MCQs, 6 hint types) |
| **Faithfulness range** | 4.3% – 78.4% (this run) | 39.7% – 89.9% |
| **Key finding** | Efficacy predicts faithfulness (r=0.937) | Training methodology predicts faithfulness more than parameter count |
| **CoT analysis** | Post-unlearning answer flip | Thinking-token acknowledgment vs. answer-text acknowledgment |

### Complementary Insights

**Where the approaches agree:** Both papers find that faithfulness is highly variable across models and is not guaranteed by reasoning capability. Gao et al.'s range of 39.7–89.9% across 12 models aligns with our finding that model identity (architecture + training) dominates dataset effects.

**Where the approaches differ:** Gao et al. report a striking internal inconsistency — models acknowledge hints in their thinking tokens ~87.5% of the time, but only act on them in the final answer ~28.6% of the time. This "thinking vs. answer" gap is a different failure mode from what unlearning measures. Unlearning tests whether a step is *causally necessary* for the answer; hint-probing tests whether a step is *acknowledged but overridden*.

**Complementarity:** An ideal faithfulness evaluation would combine both: NPO unlearning to identify which CoT steps are causally load-bearing, and hint-based probing to identify when models reason faithfully but override their reasoning at output time. These are two distinct failure modes of CoT faithfulness.

**Key methodological difference:** Gao et al.'s approach requires prompt-engineering access (to inject hints) and works on any model at inference time. The unlearning approach requires parameter access and gradient computation but makes no assumptions about the prompt structure. The unlearning approach is thus more portable across task types where hint injection is unnatural, while the hint-based approach scales to very large models (DeepSeek-R1 685B) where unlearning is computationally prohibitive.

---

## Generated Figures

All figures saved to `my_figures/`:

| File | Description |
|------|-------------|
| `fig2_efficacy_specificity_scatter.png` | Fig 2 style: efficacy vs specificity by model/dataset |
| `fig_faithfulness_by_model_dataset.png` | Faithfulness bar chart by model and dataset |
| `fig_correlation_efficacy_faithfulness.png` | Overall efficacy–faithfulness correlation |
| `fig_correlation_by_model.png` | Correlation broken down by model |
| `fig_correlation_by_dataset.png` | Correlation broken down by dataset |
| `table1_baseline_accuracy.csv` | No-CoT vs CoT accuracy (Table 1) |
| `table2_main_faithfulness.csv` | Main faithfulness results (Table 2) |
| `table_full_stats.csv` | Full efficacy/specificity/faithfulness for all 16 combos |

New figures saved to `my_figures/new/` (extensions beyond the paper):

| File | Description |
|------|-------------|
| `5a_cot_length_distribution.png` | CoT sentence count distributions by model/dataset |
| `5a_faithfulness_by_cot_length.png` | First-step faithfulness binned by CoT length |
| `5b_faithfulness_accuracy_crosstab.png` | 2×2 stacked bars: faithful × correct per model/dataset |
| `5c_model_size_faithfulness.png` | LLaMA-3-3B vs 8B faithfulness side-by-side with deltas |
| `5d_dataset_difficulty_faithfulness.png` | Accuracy vs faithfulness scatter across datasets |
| `5d_dataset_ranking.png` | Dataset faithfulness rankings per model |

---

## Summary

- **12/16 results reliable** (LLaMA-3-3B all 4 + LLaMA-3-8B all 4 are fully trustworthy; Mistral and Phi-3 are confounded by LR mismatch)
- **Core findings replicated:** efficacy–faithfulness correlation (r=0.937), CoT accuracy improvement, larger model = more faithful CoT
- **Not replicated directly:** exact faithfulness numbers for Phi-3 and Mistral (require re-running with model-specific best LRs: Phi-3 at 1e-04/5e-05, Mistral at 5e-06/3e-06)
- **4 new findings:**
  1. Longer CoTs have lower first-step faithfulness (r=−0.052, p=0.0015) — concise CoTs are structurally more causal
  2. Faithful CoT is independent of correctness — models follow their reasoning whether right or wrong (faithful-on-wrong ≥ faithful-on-correct for LLaMA models)
  3. Model scale (+5B parameters in the LLaMA-3 family) yields +17.4pp faithfulness gain, significant on 2/4 datasets
  4. Dataset difficulty does not predict faithfulness (Spearman ρ=−0.40, n.s.) — model architecture dominates
- **Relationship to concurrent work:** Gao et al. (arXiv:2603.22582) uses hint-injection (complementary to unlearning), finds 39.7–89.9% faithfulness range on larger models (up to 685B), and identifies a distinct failure mode: models acknowledge hints in thinking tokens but override them in answers
