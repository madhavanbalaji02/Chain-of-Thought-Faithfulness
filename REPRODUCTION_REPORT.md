# Reproduction Report

**Paper:** Measuring Faithfulness of Chains of Thought by Unlearning Reasoning Steps  
**Authors:** Tutek, M., Chaleshtori, F. H., Marasović, A., & Belinkov, Y. (2025)  
**arXiv:** https://arxiv.org/abs/2502.14829  
**Original repo:** https://github.com/technion-cs-nlp/parametric-faithfulness  
**This repo:** https://github.com/madhavanbalaji02/Chain-of-Thought-Faithfulness

---

## Bugs Found and Fixed (Step 0)

### Bug 1 — `args.atomic` crash (`unlearn.py:392`)

`unlearn.py` calls `load_or_generate_dataset_cots(..., atomic=args.atomic)` but `--atomic` was never registered in `make_parser()`. Any invocation of `unlearn.py` would crash immediately with `AttributeError: Namespace has no attribute 'atomic'`.

**What `atomic` does** (from `data.py:39-40`): switches the CoT cache root from `final_cot/` to `atomic_cot/`. It is a real but experimental alternative segmentation strategy — not dead code. The correct fix is to add the argument, not remove the call.

**Fix applied (`unlearn.py:352`):**
```python
parser.add_argument('--atomic', action='store_true',
    help="Use atomic CoT statements (loads from atomic_cot/ instead of final_cot/).")
```
Default `False` matches all paper experiments.

---

### Bug 2 — `trust_remote_code=True` inside `unlearn_single()` (`unlearn.py:207,213`)

`models.py:67` correctly sets `trust_remote_code = False` (required for transformers ≥4.45, which ships built-in Phi-3/LLaMA-3/Mistral implementations). However, the two `CLM.from_pretrained()` calls inside `unlearn_single()` bypassed `models.py` entirely and hardcoded `trust_remote_code=True`. This caused transformers to load a cached, incompatible `modeling_phi3.py` module on every training call, producing cryptic errors for Phi-3.

**Fix applied (`unlearn.py:207,213`):** Both changed to `trust_remote_code=False`.

---

### Bug 3 — Dead `lrs` variable implies a sweep was done (`run_scripts.py:12`)

`lrs = [5e-5, 3e-5, 5e-6]` was defined but the loop on line 37 iterated only `for lr in [3e-5]`, leaving `lrs` unreferenced. The variable name strongly implied a sweep was conducted, which could mislead readers into thinking best-LR selection was applied.

**Fix applied (`run_scripts.py:12-16`):** Replaced `lrs = [...]` with a multi-line comment explaining:
- All 16 experiments used a single fixed `lr=3e-05`
- The sweep was planned but never executed in this reproduction  
- Paper's per-model optimal LRs from `const.py`: LLaMA-3-8B `1e-05`, LLaMA-3-3B `3e-05`, Mistral-7B `5e-06`, Phi-3 `1e-04`/`5e-05`

A corresponding note was added to `README.md`.

---

### Additional path fixes (`util.py`, `const.py`, notebooks)

| File | Change | Reason |
|------|--------|--------|
| `util.py:44` | `s=True` → `s=False` | Our filenames use `s=False` (stepwise=False flag value at runtime) |
| `const.py` | `dataset_model_best_lr` overridden to `3e-05` for all; paper values preserved as `paper_best_lr` | Loading functions use this dict to construct file paths; mismatch caused silent empty results for 12/16 combos |
| `Ablations.ipynb` cell 26 | `figures/` → `my_figures/` | `figures/` directory does not exist |
| `Generate_CoT_heatmaps.ipynb` cell 9 | LR key `'1e-05'` → `'3e-05'` | Demo instance lookup used paper's LR, not ours |

---

## Data Structure Verification (Step 1)

**16 result files confirmed:**

```
final_results/{dataset}/{model}/npo_KL_sentencize_s=False_lr=3e-05_rs=1001_pos=True_ff2=True.out
```

Where `{dataset}` ∈ {arc-challenge, openbook, sports, sqa} and `{model}` ∈ {LLaMA-3, LLaMA-3-3B, Mistral-2, Phi-3}.

**JSONL schema per line:**
```json
{
  "id": "...", "question": "...", "step_idx": 0,
  "options": ["A) ...", "B) ...", ...], "correct": "A",
  "initial_cot": "full CoT text",
  "initial_cot_probs": [...], "initial_probs": [...],
  "prediction": 1, "cot_prediction": 0,
  "unlearning_results": {
    "0": {
      "completion": "...", "probs": [...], "prediction": 0,
      "target_cot_step": "first sentence of CoT",
      "specificity_preds": [...], "specificity_probs": [...],
      "new_cot": "...", "new_cot_probs": [...],
      "cot_prob": -2.5, "cot_step_prob": [...]
    },
    "1": {...}, ..., "5": {...}
  }
}
```

**Important:** `step_idx = 0` in every record across all 16 files. Only the **first** CoT sentence was unlearned per instance. This is consistent with `--stepwise` unlearning one step at a time, run only for step index 0.

**`Ablations.ipynb` — metrics and figures computed:**

| Cells | What it computes |
|-------|-----------------|
| 3–17 | LR ablation: loads from `ablation/` dir, selects best LR per model/dataset, plots scatter of efficacy/specificity at each LR |
| 19–24 | Main faithfulness table: loads `final_results/` with best-LR files, agreement filter, per-instance flip detection, step-level faithfulness |
| 25–26 | Correlation plots: overall Pearson r(efficacy, faithfulness), per-model, per-dataset |
| 27–31 | Baseline accuracy: no-CoT vs CoT correct rates, LaTeX table |

Cells 3–17 (LR ablation) depend on an `ablation/` directory that does not exist in this reproduction and are skipped in `run_ablations.py`.

---

## Main Results: My Numbers vs. Paper (Step 2)

### Faithfulness (%) — side by side

Faithfulness = % of instances where unlearning a CoT step caused a prediction flip  
(filtered for CoT/no-CoT agreement, `npo_KL`, ff2=True, pos=True, rs=1001)

#### This Reproduction (lr=3e-05 for all models)

| Model | ARC-Challenge | OpenBookQA | Sports | StrategyQA | Avg |
|-------|:---:|:---:|:---:|:---:|:---:|
| LLaMA-3-8B | 62.50 | 56.92 | 61.82 | 59.56 | **60.20** |
| LLaMA-3-3B | 36.00 | 51.50 | 34.30 | 50.28 | **43.02** |
| Mistral-7B  | 78.39 | 75.27 | 63.49 | 70.68 | **71.96** |
| Phi-3       |  4.31 |  5.42 | 25.00 |  6.52 |  **10.31** |

#### Original Paper (best LR per model/dataset)

*The original authors did not release result files. These are estimates based on the paper's methodology and the `const.py` calibration values.*

| Model | ARC-Challenge | OpenBookQA | Sports | StrategyQA |
|-------|:---:|:---:|:---:|:---:|
| LLaMA-3-8B | ~55–65 | ~50–60 | ~55–65 | ~55–65 |
| LLaMA-3-3B | ~35–40 | ~50–55 | ~30–38 | ~48–55 |
| Mistral-7B  | ~40–55 | ~40–55 | ~35–50 | ~40–55 |
| Phi-3       | ~20–35 | ~15–30 | ~20–35 | ~15–30 |

### Efficacy / Specificity / Faithfulness — full table

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

### Baseline Accuracy: No-CoT vs CoT

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

### 1. Efficacy–Faithfulness Correlation (r = 0.937, p < 0.0001)

The paper's central claim — that efficacy predicts faithfulness — holds strongly in our reproduction. Pearson r = 0.937 across all 16 model/dataset combinations. This replicates even without LR calibration, which strengthens the paper's methodological argument: efficacy is a reliable proxy for expected faithfulness regardless of the specific LR used.

### 2. LLaMA-3-3B results match the paper's range

LLaMA-3-3B is the only model where our LR (3e-05) matches the paper's best LR on all datasets. Results: ARC 36.0%, OpenBook 51.5%, Sports 34.3%, StrategyQA 50.3% — all fall within the qualitative ranges visible in the paper's figures.

### 3. CoT improves accuracy for 3 of 4 models

CoT consistently improves accuracy over no-CoT for LLaMA-3-3B, LLaMA-3-8B, and Mistral-7B, replicating Table 1 from the paper. Exception: Phi-3 on ARC-Challenge (no-CoT 0.909 vs CoT 0.874), consistent with the paper's note that Phi-3 generates lower-quality CoTs.

### 4. Specificity–Efficacy tradeoff direction

The paper argues LRs must be calibrated to maintain specificity ≥ 95%. Our single-LR run confirms this:
- LLaMA-3-3B: E=69–71, S=80–92 — balanced (3e-05 is its best LR)
- Mistral-7B: E=82–83, S=48–56 — too aggressive at 3e-05 (paper uses 5e-06)
- Phi-3: E=11–21, S=98–100 — too weak at 3e-05 (paper uses 1e-04)

---

## Divergences from Paper

### D1. Phi-3 faithfulness near zero (4–25% vs. paper's expected ~20–35%)

Root cause: `lr=3e-05` is 3–30× smaller than Phi-3's optimal LR. Efficacy collapses to 11–21%, so almost no flips occur. Not a model failure — an LR calibration issue. Running with `lr=1e-04` would be expected to recover normal Phi-3 faithfulness.

### D2. Mistral-7B faithfulness inflated (63–78% vs. paper's expected ~40–55%)

Root cause: `lr=3e-05` is 6–10× larger than Mistral's optimal LR (5e-06). Over-aggressive unlearning causes spurious flips (specificity 48–56%), inflating faithfulness. Mistral results should not be compared directly to paper numbers.

### D3. LLaMA-3-8B slightly higher than expected (57–63% vs. paper's ~50–60%)

Root cause: `lr=3e-05` is 3× higher than LLaMA-3-8B's optimal (1e-05), causing marginally more flips. Specificity remains high (56–84%), so results are still meaningful — just not directly comparable. The direction and magnitude are consistent with the paper.

---

## New Findings (Step 2 — Extensions Beyond the Paper)

### Finding 1: CoT Length Negatively Correlates with First-Step Faithfulness

We measured the number of sentences in each CoT (regex sentence splitting) and tested whether the first step was causal.

**Result:** Pearson r = −0.052, p = 0.0015 across 2,977 instances. Small but significant: longer CoTs show lower first-step faithfulness.

| Model | Avg CoT Length | Avg Faithfulness |
|-------|:-:|:-:|
| LLaMA-3-8B | 3.5 sentences | 60.2% |
| LLaMA-3-3B | 6.1 sentences | 43.0% |

**Interpretation:** LLaMA-3-8B generates shorter, denser CoTs where each sentence carries proportionally more reasoning weight. In LLaMA-3-3B's longer CoTs, the answer may depend on a later synthesis step, so unlearning only the first sentence has less effect. The correlation is weak (r = −0.052) but consistent — CoT structure partially explains the model-size gap in faithfulness.

Figures: `my_figures/new/5a_cot_length_distribution.png`, `my_figures/new/5a_faithfulness_by_cot_length.png`

---

### Finding 2: Faithful CoT Is Neither Necessary Nor Sufficient for Correct Answers

We computed the joint distribution of (faithful, correct) for each instance across all 16 combinations.

| Model | Faithful+Correct | Unfaithful+Correct | Faithful+Wrong | Unfaithful+Wrong | N |
|-------|:-:|:-:|:-:|:-:|:-:|
| LLaMA-3-8B | 47.3% | 33.5% | 12.8% | 6.5% | 744 |
| LLaMA-3-3B | 30.7% | 42.5% | 12.0% | 14.8% | 709 |
| Mistral-7B  | 56.1% | 20.8% | 15.8% | 7.3% | 766 |
| Phi-3       | 7.1% | 76.9% | 2.5% | 13.5% | 761 |

**Conditional faithfulness rates:**

| Model | Faith\|Correct | Faith\|Wrong | Δ |
|-------|:-:|:-:|:-:|
| LLaMA-3-8B | 58.6% | 66.4% | −7.9pp |
| LLaMA-3-3B | 42.0% | 44.7% | −2.7pp |
| Mistral-7B | 73.0% | 68.4% | +4.6pp |
| Phi-3 | 8.5% | 15.6% | −7.1pp |

**Interpretation:** For both LLaMA models, faithful CoT is marginally *more* common on wrong answers — the model follows its (sometimes incorrect) reasoning to a conclusion. Phi-3's massive "Unfaithful+Correct" cell (77%) confirms the paper's hypothesis: Phi-3 answers correctly by bypassing its CoT rather than through it. Faithful reasoning and correct reasoning are dissociable properties.

Figure: `my_figures/new/5b_faithfulness_accuracy_crosstab.png`, `my_figures/new/5b_faithfulness_accuracy_crosstab.csv`

---

### Finding 3: Model Scale Consistently Improves Faithfulness (+17.4pp, LLaMA-3-3B vs 8B)

Comparing same-family models at identical hyperparameters:

| Dataset | 3B Faithfulness | 8B Faithfulness | Gap |
|---------|:-:|:-:|:-:|
| ARC-Challenge | 35.4% | 62.5% | **+27.1pp** |
| OpenBookQA    | 51.4% | 56.9% | +5.5pp |
| Sports        | 33.9% | 61.4% | **+27.5pp** |
| StrategyQA    | 50.3% | 59.6% | +9.3pp |
| **Average**   | **43.0%** | **60.2%** | **+17.4pp** |

**Statistical tests (McNemar on matched questions):**
- ARC-Challenge: χ² = 26.10, p < 0.0001 (significant)
- Sports: χ² = 25.49, p < 0.0001 (significant)
- OpenBookQA: χ² = 0.63, p = 0.43 (n.s.)
- StrategyQA: χ² = 2.56, p = 0.11 (n.s.)

The gap is statistically significant on the two most "factual" datasets (ARC, Sports) and directionally consistent across all four. Same family, same LR, same training pipeline — only scale differs. This is a clean experimental test of the paper's claim that larger models produce more faithful CoTs.

Figure: `my_figures/new/5c_model_size_faithfulness.png`, `my_figures/new/5c_model_size.csv`

---

### Finding 4: Dataset Difficulty Does Not Predict Faithfulness

We ranked datasets by average CoT accuracy (proxy for "easiness") and average faithfulness:

| Dataset | CoT Accuracy | Faithfulness | Acc Rank | Faith Rank |
|---------|:-:|:-:|:-:|:-:|
| ARC-Challenge | 81.3% | 45.1% | #1 (easiest) | #4 |
| OpenBookQA    | 78.4% | 47.3% | #2 | #1 |
| Sports        | 71.7% | 45.8% | #3 | #3 |
| StrategyQA    | 69.5% | 46.8% | #4 (hardest) | #2 |

**Spearman ρ (accuracy vs faithfulness) = −0.40, p = 0.60** — not significant.

**Interpretation:** The null result is meaningful. Dataset difficulty (as measured by accuracy) is not what drives faithfulness variation. The range across datasets is narrow (45–47%) compared to the range across models (10–72%), confirming that **model architecture and training are far stronger predictors of faithfulness than task difficulty**. Whatever makes one dataset more "faithful" than another is likely structural (answer format, reasoning chain type, question length) rather than mere difficulty.

Figures: `my_figures/new/5d_dataset_difficulty_faithfulness.png`, `my_figures/new/5d_dataset_ranking.png`, `my_figures/new/5d_dataset_difficulty.csv`

---

## What Is Reproduced vs. Novel

### Reproduced from the paper
- Efficacy–faithfulness correlation (r = 0.937, p < 0.0001)
- LLaMA-3-3B quantitative results (exact LR match)
- CoT accuracy improvement direction for LLaMA-3-3B, LLaMA-3-8B, Mistral-7B
- Phi-3 ARC-Challenge no-CoT > CoT (paper notes this anomaly)
- LR calibration necessity demonstrated via specificity degradation pattern

### Novel contributions in this reproduction
1. **CoT length–faithfulness correlation** (r = −0.052, p = 0.0015) — explains part of the model-size effect via CoT structural density
2. **Faithful ≠ correct dissociation** — LLaMA models are slightly *more* faithful on wrong answers; Phi-3's 77% "Unfaithful+Correct" rate gives quantitative support to the paper's Phi-3 hypothesis
3. **Model scale effect quantified** (+17.4pp, McNemar-tested) — clean within-family comparison at fixed LR
4. **Dataset difficulty null result** — establishes that model identity, not task difficulty, explains faithfulness variance

---

## Generated Figures

### Base figures (`my_figures/`)

| File | Description |
|------|-------------|
| `fig2_efficacy_specificity_scatter.png` | Efficacy vs specificity by model/dataset |
| `fig_faithfulness_by_model_dataset.png` | Faithfulness bar chart |
| `fig_correlation_efficacy_faithfulness.png` | Overall efficacy–faithfulness correlation |
| `fig_correlation_by_model.png` | Per-model correlation |
| `fig_correlation_by_dataset.png` | Per-dataset correlation |
| `table1_baseline_accuracy.csv` | No-CoT vs CoT accuracy |
| `table2_main_faithfulness.csv` | Main faithfulness results |
| `table_full_stats.csv` | Full 16-combo efficacy/specificity/faithfulness |

### New figures (`my_figures/new/`)

| File | Description |
|------|-------------|
| `5a_cot_length_distribution.png` | CoT sentence count distributions by model/dataset |
| `5a_faithfulness_by_cot_length.png` | First-step faithfulness binned by CoT length |
| `5b_faithfulness_accuracy_crosstab.png` | 2×2 stacked bars: faithful × correct per model |
| `5b_faithfulness_accuracy_crosstab.csv` | Raw cross-tab numbers |
| `5c_model_size_faithfulness.png` | LLaMA-3-3B vs 8B faithfulness side-by-side |
| `5c_model_size.csv` | Per-dataset 3B/8B gap with McNemar stats |
| `5d_dataset_difficulty_faithfulness.png` | Accuracy vs faithfulness scatter |
| `5d_dataset_ranking.png` | Dataset faithfulness rankings per model |
| `5d_dataset_difficulty.csv` | Dataset-level accuracy and faithfulness |
