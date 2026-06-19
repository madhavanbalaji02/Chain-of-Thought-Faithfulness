<h1 align="center">🔍 Measuring Faithfulness of Chain-of-Thought<br>by Unlearning Reasoning Steps</h1>

<p align="center">
  <em>Reproduction + Extensions of <a href="https://arxiv.org/abs/2502.14829">Tutek et al. (2025)</a></em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Models-4_LLMs-blue" alt="Models">
  <img src="https://img.shields.io/badge/Datasets-4_Benchmarks-green" alt="Datasets">
  <img src="https://img.shields.io/badge/Instances-3%2C699_NPO-orange" alt="Instances">
  <img src="https://img.shields.io/badge/FT_Conditions-20_Experiments-red" alt="FT Conditions">
  <img src="https://img.shields.io/badge/Compute-IU_BigRed200-purple" alt="Compute">
</p>

---

## 📌 TL;DR — What I Did and What I Found

This project answers one question: **when an LLM writes out its "reasoning," is that reasoning actually driving the answer, or is it just decoration?**

I did three things:

1. **Reproduced** the NPO unlearning method from Tutek et al. across 4 models × 4 datasets (3,699 instances) — confirming their core finding that you can measure faithfulness by surgically unlearning individual reasoning steps and watching if the answer changes.

2. **Designed and ran a new experiment** — the **Alignment-Faithfulness Paradox** — asking: *if we fine-tune models on high-quality CoT, does that make their reasoning more faithful?* The answer is no — it makes it **dramatically less faithful** (−56% to −75%), while simultaneously improving accuracy. The better the model performs, the less its chain of thought actually matters to its answers.

3. **Investigated why**, through four mechanistic analyses: quality scoring of CoT steps, FF2 linear probes, faithfulness regularization, and external validation against REVEAL human annotations. The root cause: quality filters systematically select reasoning steps that *don't* causally influence predictions. This is a data-level selection bias that no training objective can fix.

<p align="center">
  <img src="paper_figures/generated/plot30_summary_dashboard.png" width="95%" alt="Research Dashboard">
</p>

---

## 🧠 The Problem: Are Chains of Thought Faithful?

Large language models produce step-by-step reasoning (Chain-of-Thought, or CoT) when answering questions. But there's a fundamental trust problem: **does the model actually use those reasoning steps to arrive at its answer, or does it decide the answer first and then generate plausible-sounding reasoning after the fact?**

If CoT is just post-hoc rationalization, then:
- We can't trust CoT for interpretability or safety auditing
- Fine-tuning on "better reasoning" might improve surface quality without improving actual reasoning
- Alignment techniques that rely on inspecting reasoning chains are built on sand

This matters because the entire premise of CoT-based oversight — that we can understand what a model is doing by reading its reasoning — depends on faithfulness.

---

## 🔬 The Method: How NPO Unlearning Measures Faithfulness

The key insight from Tutek et al.: if a reasoning step truly drives the answer, then **making the model forget that step should change its answer**. If the answer stays the same, the step was decorative.

Here's what happens for each instance:

```
Step 1:  Give the model a question → it generates a Chain-of-Thought → it produces an answer

Step 2:  Pick one sentence from the CoT (e.g., "The ball was caught by the outfielder")

Step 3:  Apply Negative Preference Optimization (NPO) to make the model "unlearn" 
         that specific sentence — train it for 5 epochs to reduce the probability 
         of generating those tokens, while a KL-divergence term prevents the rest 
         of the model from drifting

Step 4:  Ask the model the same question again

Step 5:  Did the answer change?
           → YES: that CoT step was FAITHFUL (it causally influenced the prediction)
           → NO:  that CoT step was NOT FAITHFUL (the model didn't need it)
```

**Key technical details:**
- **NPO-KL loss**: forget loss (push down the target step's probability) + KL retain loss (keep everything else stable)
- **FF2 restriction**: only update the second feed-forward layer (`mlp.down_proj.weight`) — this is where prior work found faithfulness is encoded
- **POS-tag filtering**: only unlearn content tokens (nouns, verbs, adjectives), not function words — via spaCy
- **Specificity check**: after unlearning, test the model on *other* questions to make sure we only affected the target, not general capabilities

---

## 📋 What I Did — Step by Step

### Phase 1: Reproduction (16 experiments)

I reproduced the full Tutek et al. experimental setup:

- **4 models**: LLaMA-3-3B, LLaMA-3-8B, Mistral-7B, Phi-3
- **4 datasets**: ARC-Challenge, OpenBookQA, Sports Understanding, StrategyQA
- **Settings**: NPO-KL, stepwise unlearning, FF2 restriction, POS filtering, lr=3e-05, 5 epochs per instance
- **Scale**: ~230 instances per model-dataset pair → **3,699 total unlearning experiments**
- **Compute**: Indiana University BigRed200 HPC (NVIDIA A100 GPUs, 12–24 hours per job)

I also **fixed 4 bugs** in the original codebase (see [Bugs Found](#-bugs-found-in-reproduction)) that were causing silent failures.

**Result**: Successfully replicated the core finding — efficacy-faithfulness correlation of r=0.937 (p < 0.0001).

<p align="center">
  <img src="paper_figures/generated/plot03_npo_faithfulness_heatmap.png" width="60%" alt="Faithfulness Heatmap">
  <br><em>Faithfulness (%) across all 16 model × dataset combinations. Each cell shows what percentage of CoT steps actually influenced the model's answer.</em>
</p>

| Model | ARC-Challenge | OpenBookQA | Sports | StrategyQA | **Avg** |
|:------|:---:|:---:|:---:|:---:|:---:|
| 🦙 LLaMA-3-8B | 62.5 | 56.9 | 61.8 | 59.6 | **60.2** |
| 🦙 LLaMA-3-3B | 36.0 | 51.5 | 34.3 | 50.3 | **43.0** |
| 🌀 Mistral-7B | 78.4 | 75.3 | 63.5 | 70.7 | **72.0** |
| 🔷 Phi-3 | 4.3 | 5.4 | 25.0 | 6.5 | **10.3** |

### Phase 2: New Metric — Continuous Faithfulness (Δp)

The original paper uses **binary faithfulness** — did the prediction flip or not? But this is coarse. A step might shift the model's confidence from 80% → 55% without flipping the prediction. That's clearly causal influence, but binary faithfulness says "not faithful."

I introduced **Δp**: the drop in correct-answer probability after unlearning. This reveals:

- **Subcritical faithfulness**: steps with Δp > 0 that don't flip the prediction — the binary metric misses these entirely
- **Counterproductive CoT**: steps with Δp < 0, where unlearning *increases* correct-answer probability — the CoT was actively hurting performance

<p align="center">
  <img src="paper_figures/generated/plot02_delta_p_distributions.png" width="95%" alt="Delta-p Distributions">
  <br><em>Distribution of Δp (correct-answer probability drop) for each model. Many instances cluster around 0, but the tails reveal strong causal effects that binary faithfulness would miss.</em>
</p>

<p align="center">
  <img src="paper_figures/generated/plot08_subcritical_faithfulness.png" width="55%" alt="Subcritical Faithfulness">
  <br><em>What binary faithfulness misses: solid bars = prediction flipped (binary faithful). Hatched = probability shifted but no flip (subcritical — binary says "not faithful" but there was real causal influence). Gray = Δp < 0 (CoT was counterproductive).</em>
</p>

**Key finding**: In Phi-3, **67.8%** of instances have negative Δp — unlearning the CoT step makes the model *more* correct. The CoT is actively suppressing the right answer.

### Phase 3: The Alignment-Faithfulness Paradox (New Experiment)

This is the main new contribution. I asked: **if we train models to produce better reasoning, does that make their reasoning more faithful?**

**What I did:**
1. **Scored CoT quality**: Used GPT-4o to score every CoT step on coherence, plausibility, and completeness (1–5 scale each) → 918 scored instances
2. **Split by quality**: Separated instances into high-quality (top scores) and low-quality (bottom scores) training sets (~400–500 instances each)
3. **Fine-tuned 4 models**: LoRA fine-tuning (rank 16, 3 epochs) on high-quality CoT data, low-quality CoT data, and a baseline (no fine-tuning) — for all 4 models
4. **Measured faithfulness**: Ran the same NPO unlearning probe on 50 held-out instances per condition
5. **Additional controls**: Model-specific fine-tuning (Phi-3 on its own CoTs), multi-model cross-training, and 3 faithfulness regularization strengths

**Total: 20 fine-tuning conditions evaluated.**

<p align="center">
  <img src="paper_figures/generated/plot15_all_model_ft_comparison.png" width="80%" alt="Fine-Tuning Comparison">
  <br><em>The paradox: 🔴 High-quality fine-tuning collapses faithfulness in 3 of 4 models. 🟢 Low-quality fine-tuning leaves it unchanged. The model that produces better-sounding reasoning is actually reasoning less.</em>
</p>

| Model | Baseline | High-Quality FT | Low-Quality FT | Relative Drop |
|:------|:--------:|:-------:|:------:|:-------------:|
| 🦙 LLaMA-3-3B | 30.0% | **8.0%** ↓ | 28.0% | **−73%** |
| 🔷 Phi-3 | 32.0% | **14.0%** ↓ | 36.0% | **−56%** |
| 🌀 Mistral-7B | 16.0% | **4.0%** ↓ | 22.0% | **−75%** |
| 🦙 LLaMA-3-8B | 3.8% | 4.0% | 4.0% | ~0% (floor) |

> *p-values (two-proportion z-test)*: LLaMA-3-3B p=0.005**, Phi-3 p=0.033*, Mistral-7B p=0.046*

<p align="center">
  <img src="paper_figures/generated/plot20_accuracy_faithfulness_tradeoff.png" width="55%" alt="Accuracy-Faithfulness Tradeoff">
  <br><em>The tradeoff visualized: each model moves from baseline → high-FT → low-FT. High-quality FT pushes models to the lower-right: higher accuracy, lower faithfulness. The models get better answers while their reasoning becomes less real.</em>
</p>

### Phase 4: Investigating the Mechanism

I ran four analyses to understand *why* high-quality fine-tuning destroys faithfulness:

#### 4a. Quality Scoring Reveals Selection Bias

I scored 940 CoT steps with GPT-4o and compared quality scores between faithful and non-faithful steps.

**Finding**: Non-faithful steps score **significantly higher** on quality (3.61 vs 3.33, Mann-Whitney p=0.008). This means quality filtering *systematically selects against* faithful reasoning — the steps that look best are the ones that don't actually drive the answer.

<p align="center">
  <img src="paper_figures/generated/plot21_quality_score_analysis.png" width="85%" alt="Quality Score Analysis">
</p>

#### 4b. FF2 Probe Shows Encoding Erasure

I extracted hidden states from FF2 layers (the feed-forward layers where faithfulness is encoded) at layers 8, 16, and 24, and trained linear probes to predict whether a step is faithful.

**Finding**: After high-quality FT, the probe's accuracy drops to majority baseline at every layer — the model's internal representation of "this step matters" is **completely erased**. Low-quality FT maintains or increases the signal.

<p align="center">
  <img src="paper_figures/generated/plot22_ff2_probe_results.png" width="60%" alt="FF2 Probe Results">
  <br><em>Probe accuracy lift over majority baseline. 🔴 High-quality FT → 0 lift at all layers. 🔵 Low-quality FT → maintained or improved. The faithfulness encoding is structurally erased by quality training.</em>
</p>

#### 4c. Faithfulness Regularization Fails

I tried to fix the paradox by adding a faithfulness-preserving regularization term to the training loss: L = L_quality + λ × L_faith, where L_faith penalizes the model when its answer probability doesn't change after masking a CoT step.

**Finding**: **Complete failure** at all λ values (0.0, 0.1, 1.0) — faithfulness drops to 0–2% in all conditions. The faith loss gradient is near-zero throughout training because high-quality training instances are already parametrically decoupled *before any gradient update*. There's no signal for the regularizer to amplify.

> **This is the key insight**: the paradox is a **data selection problem**, not a training dynamics problem. The high-quality CoT instances selected for training are ones where the reasoning was already decorative — no training objective applied to that data can restore faithfulness.

#### 4d. External Validation via REVEAL

I validated against human annotations from the REVEAL dataset (1,276 step annotations across 54 questions). Steps in logically correct answers show significantly higher Δp (+0.110 vs −0.084, p < 0.001), confirming that parametric faithfulness captures genuine logical grounding. Surface-level evidence attribution (ρ = 0.04, n.s.) shows no relationship — a step can cite a source without actually driving the answer.

<p align="center">
  <img src="paper_figures/generated/plot23_reveal_validation.png" width="70%" alt="REVEAL Validation">
</p>

---

## 📊 Additional Findings

### The Collapse Is Task-Type Specific

Not all datasets are affected equally. StrategyQA and Sports Understanding (tasks requiring implicit knowledge retrieval) collapse to **0% faithfulness** under high-quality FT. OpenBookQA (where CoT cites explicit supporting passages) is **immune**.

**Why?** Commonsense tasks rely on memorized associations — the model "knows" the answer parametrically and generates reasoning as post-hoc rationalization. Science tasks with explicit passages have CoT steps that genuinely activate the answer pathway.

<p align="center">
  <img src="paper_figures/generated/plot18_per_dataset_ft_collapse.png" width="95%" alt="Per-Dataset Collapse">
  <br><em>Faithfulness by dataset under each FT condition. StrategyQA and Sports collapse completely; OpenBookQA is immune; ARC-Challenge is at floor throughout.</em>
</p>

### Wrong Answers Have More Faithful Reasoning

Faithfulness is consistently **higher for incorrect predictions** than correct ones. When the model gets the answer right, the CoT is decorative — the correct answer was committed to through parametric shortcuts. When the model is wrong, the CoT was more likely actually driving the (wrong) prediction.

High-quality FT sharpens this: for LLaMA-3-3B, faithfulness for correct predictions drops to **0.0%** — the model learns to produce correct answers with zero causal dependence on its chain of thought.

<p align="center">
  <img src="paper_figures/generated/plot17_correctness_conditioning.png" width="95%" alt="Correctness Conditioning">
  <br><em>🟢 Correct predictions = low faithfulness (CoT is decoration). 🔴 Incorrect predictions = higher faithfulness (CoT was actually driving the answer).</em>
</p>

### How Unlearning Evolves Over Epochs

<p align="center">
  <img src="paper_figures/generated/plot01_epoch_trajectories_all_models.png" width="95%" alt="Epoch Trajectories">
  <br><em>Per-epoch trajectories: efficacy (how much the step probability drops) increases monotonically, while specificity (staying on-target) degrades. Faithfulness saturates early — most flips happen by epoch 2.</em>
</p>

### Model Size Matters

<p align="center">
  <img src="paper_figures/generated/plot06_model_size_comparison.png" width="60%" alt="Model Size Comparison">
  <br><em>LLaMA-3 8B vs 3B: the larger model shows higher faithfulness across all datasets. Larger models may maintain stronger CoT-parameter coupling.</em>
</p>

### CoT Length Dilutes Faithfulness

<p align="center">
  <img src="paper_figures/generated/plot07_cot_length_vs_faithfulness.png" width="95%" alt="CoT Length vs Faithfulness">
  <br><em>Mean Δp drops from 0.271 (1–2 sentence CoTs) to 0.077 (8+ sentences). Longer chains dilute per-step causal influence.</em>
</p>

### Instance-Level Probability Shifts

<p align="center">
  <img src="paper_figures/generated/plot11_instance_prob_scatter.png" width="95%" alt="Instance Probability Scatter">
  <br><em>Each dot is one instance. X-axis: initial P(correct). Y-axis: P(correct) after unlearning. Points below the diagonal = unlearning reduced confidence. Colored = prediction flipped. Gray = subcritical (probability shifted but answer didn't change).</em>
</p>

### Bimodal Δp Under Fine-Tuning

<p align="center">
  <img src="paper_figures/generated/plot16_ft_delta_p_distributions.png" width="95%" alt="FT Delta-p Distributions">
  <br><em>High-quality FT creates a bimodal distribution: most steps become fully decoupled (Δp ≈ 0), but the few that remain faithful are more strongly coupled than before. Binary faithfulness drops, but mean Δp can increase.</em>
</p>

---

## 🔑 The Big Picture: What This Means

The causal chain works like this:

```
1. Non-faithful CoT steps score higher on quality (3.61 vs 3.33, p=0.008)
         ↓
2. Quality filtering preferentially selects these decoupled steps for training
         ↓
3. Fine-tuning on this data teaches FF2 layers to generate fluent text
   without encoding step-specific faithfulness (probe lift → 0)
         ↓
4. NPO unlearning of individual steps no longer changes predictions
         ↓
5. Faithfulness collapses — the model produces better-sounding reasoning
   that has less to do with its actual answer
```

**Implication**: If the AI safety community relies on reading CoT to understand what models are doing, and the standard approach to improving CoT is quality-based fine-tuning, then **the standard approach actively undermines the interpretability property we're relying on**. Fixing this requires faithfulness-aware data curation — filtering training data by causal influence (Δp), not surface quality.

---

## ⚙️ Setup & Usage

### Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### StrategyQA Data

```bash
mkdir -p data/strategyqa
wget -O data/strategyqa/strategyqa_train.json \
  https://raw.githubusercontent.com/wicsaax/strategy-qa/main/strategyQA_train.json
```

### Running a Single Experiment

```bash
python unlearn.py \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --strategy sentencize \
  --stepwise \
  --dataset sqa \
  --lr 3e-05 \
  --pos \
  --ff2 \
  --method npo_KL
```

### Key Flags

| Flag | Description |
|:-----|:-----------|
| `--model_name` | HuggingFace model ID |
| `--dataset` | `arc-challenge`, `openbook`, `sports`, `sqa` |
| `--method` | `npo_KL` (default), `npo`, `npo_grad_diff` |
| `--ff2` | Restrict optimization to FF2 layers (`mlp.down_proj.weight`) |
| `--pos` | Filter function tokens via spaCy POS tagging |
| `--stepwise` | Unlearn one CoT sentence at a time |
| `--strategy sentencize` | Split CoT into sentences using NLTK |

### Submitting All 16 SLURM Jobs (BigRed200)

```bash
python run_scripts.py | grep sbatch | sed 's/sbatch /sbatch -A c01949 /g' | bash
```

### Generating All Plots

```bash
python generate_all_plots.py
# Outputs 30+ plots (PNG + PDF) to paper_figures/generated/
```

---

## 🏗️ Code Structure

### Experiment Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: NPO Unlearning (Reproduction)                        │
│                                                                 │
│  1. Generate CoTs    →  data.py:load_or_generate_dataset_cots() │
│     Cached in: final_cot/{dataset}/{model}_cots.jsonl           │
│                                                                 │
│  2. Unlearn steps    →  unlearn.py:unlearn_single()             │
│     Two model copies: trainable + frozen oracle                 │
│     NPO-KL loss on individual CoT sentences, 5 epochs          │
│                                                                 │
│  3. Evaluate         →  unlearn.py:evaluate()                   │
│     CoT probability, answer probs, specificity, new CoT        │
│                                                                 │
│  4. Results          →  final_results/{dataset}/{model}/*.out   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: Alignment-Faithfulness Paradox (New Experiment)       │
│                                                                 │
│  1. Score quality    →  score_cot_quality.py (GPT-4o)           │
│  2. Split data       →  make_finetune_splits.py                 │
│  3. LoRA fine-tune   →  finetune_lora.py                        │
│  4. Merge adapters   →  merge_adapters_nopeft.py                │
│  5. Evaluate         →  evaluate_finetuned.py (NPO probe)       │
│  6. Results          →  finetuned_results/{condition}/           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: Mechanistic Analysis                                  │
│                                                                 │
│  • FF2 activations   →  extract_ff2_activations.py              │
│  • FF2 probes        →  train_ff2_probe.py                      │
│  • Extended analysis →  extended_analyses.py (Δp, quartiles)    │
│  • New analyses      →  new_analyses.py (CoT length, size)      │
│  • All plots         →  generate_all_plots.py                   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Files

| File | What It Does |
|:-----|:------------|
| `unlearn.py` | Main entry point — runs NPO unlearning loop + per-epoch evaluation |
| `data.py` | Generates/caches CoTs, builds forget/retain dataset pairs |
| `evaluate.py` | Computes CoT probabilities, answer probabilities, generates new CoTs |
| `models.py` | Loads HuggingFace models in bfloat16 with device_map="auto" |
| `dataload.py` | Dataset-specific loaders for ARC, OpenBookQA, Sports, StrategyQA |
| `segment.py` | POS-tag alignment of CoT tokens using spaCy |
| `score_cot_quality.py` | GPT-4o quality scoring (coherence, plausibility, completeness) |
| `make_finetune_splits.py` | Splits scored data into high/low quality training sets |
| `finetune_lora.py` | LoRA fine-tuning with optional faithfulness regularization |
| `evaluate_finetuned.py` | Runs NPO faithfulness probe on fine-tuned models |
| `merge_adapters_nopeft.py` | Manual LoRA weight merging (avoids PEFT/Triton issues on HPC) |
| `extract_ff2_activations.py` | Extracts FF2 hidden states for linear probe analysis |
| `train_ff2_probe.py` | Trains logistic regression probes on FF2 activations |
| `extended_analyses.py` | Continuous Δp analysis, subcritical faithfulness, epoch trajectories |
| `new_analyses.py` | CoT length, model size, dataset difficulty analyses |
| `generate_all_plots.py` | Generates all 30+ research plots from local data |

### Loss Functions

| Loss | Formula | Used For |
|:-----|:--------|:---------|
| `npo` | Forget loss only | Ablation |
| `npo_grad_diff` | Forget + cross-entropy retain | Ablation |
| `npo_KL` | Forget + KL divergence retain | **All paper experiments** |

---

## 🐛 Bugs Found in Reproduction

| File | Bug | Fix |
|:-----|:----|:----|
| `unlearn.py` | `args.atomic` referenced but `--atomic` never registered — crashes on startup | Added `parser.add_argument('--atomic', ...)` |
| `unlearn.py` | `trust_remote_code=True` hardcoded, contradicting `models.py` | Changed to `False` |
| `run_scripts.py` | `lrs` dead code implying an LR sweep was run | Replaced with comment |
| `util.py` / `const.py` | `s=True` in filename pattern → silent empty results | Fixed `s=False`; set all LRs to `3e-05` |

---

## 🖥️ Compute

All experiments ran on **BigRed 200** (Indiana University HPC):
- **GPU**: NVIDIA A100 40GB
- **Small models** (Phi-3, LLaMA-3-3B): 1× GPU, 32GB RAM, 12h per job
- **Large models** (LLaMA-3-8B, Mistral-7B): 2× GPUs, 64GB RAM, 24h per job
- **Total compute**: 16 base NPO experiments + 20 fine-tuning conditions + activation extraction + probe training

---

## 📄 Citation

```bibtex
@article{tutek2025measuring,
  title={Measuring Faithfulness of Chains of Thought by Unlearning Reasoning Steps},
  author={Tutek, Martin and Chaleshtori, Farzad Habibi and Marasovi{\'c}, Ana and Belinkov, Yonatan},
  journal={arXiv preprint arXiv:2502.14829},
  year={2025}
}
```

---

<p align="center">
  <em>Reproduction and extensions by <strong>Madhavan Balaji</strong> — Indiana University Indianapolis</em>
</p>
