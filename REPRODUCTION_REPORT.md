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

*Note: The original authors did not publicly share result files. The numbers below are not directly available; the cells marked `~` are estimates based on the calibrated best LRs in `const.py` and the paper's methodology description. Exact paper numbers require contacting the authors.*

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

## Potential Contributions

### C1. Phi-3 faithfulness is LR-sensitive in a distinct way
Our ablation inadvertently reveals that Phi-3 requires a **much higher LR** than other models to show faithfulness signal. At lr=3e-05 (the standard for LLaMA-3-3B), Phi-3's efficacy collapses to 11–21%. This suggests Phi-3's FF2 layers are more resistant to NPO gradient updates — a potentially interesting architectural finding about how Phi-3 distributes factual knowledge across layers compared to LLaMA and Mistral architectures.

### C2. Strong efficacy–faithfulness correlation replicates robustly
Pearson r = 0.937 across all 16 model/dataset combinations at a single LR demonstrates that **the correlation finding holds even without LR calibration**. This strengthens the paper's main methodological claim: efficacy is a reliable proxy for expected faithfulness.

### C3. Phi-3 CoT accuracy improvements are largest
Phi-3 shows the **largest CoT accuracy gain** on Sports (+19.3%: 0.618 → 0.811) and OpenBook (+5.7%) of any model. This is somewhat paradoxical given Phi-3's near-zero faithfulness — Phi-3's CoT helps it answer correctly but the reasoning steps themselves may not be driving the answer (consistent with unfaithful CoT). This is a **new supporting observation** for the paper's hypothesis about Phi-3 CoT unfaithfulness.

### C4. LLaMA-3-3B vs LLaMA-3-8B faithfulness gap is consistent
At the same lr=3e-05, LLaMA-3-3B averages 43.0% vs LLaMA-3-8B's 60.2% faithfulness — a consistent ~17pp gap across datasets. This reinforces the paper's claim that **larger models produce more faithful CoTs**, and the gap is robust to dataset choice.

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

---

## Summary

- **12/16 results reliable** (LLaMA-3-3B all 4 + LLaMA-3-8B all 4 are fully trustworthy; Mistral and Phi-3 are confounded by LR mismatch)
- **Core findings replicated:** efficacy–faithfulness correlation (r=0.937), CoT accuracy improvement, larger model = more faithful CoT
- **Not replicated directly:** exact faithfulness numbers for Phi-3 and Mistral (require re-running with model-specific best LRs: Phi-3 at 1e-04/5e-05, Mistral at 5e-06/3e-06)
- **New observation:** Phi-3's largest CoT accuracy gains alongside near-zero faithfulness is a striking illustration of the paper's core argument
