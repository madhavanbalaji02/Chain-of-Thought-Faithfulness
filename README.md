# Chain-of-Thought Faithfulness by Unlearning

This repository contains the code and results for reproducing and extending the experiments from:

> **Measuring Faithfulness of Chains of Thought by Unlearning Reasoning Steps**
> Tutek, M., Chaleshtori, F. H., Marasović, A., & Belinkov, Y. (2025)
> [[arXiv:2502.14829]](https://arxiv.org/abs/2502.14829)

![Faithfulness by Unlearning Reasoning Steps](figures/fig1_v2.png)

---

## Core Idea

A model's Chain-of-Thought (CoT) is **faithful** if the reasoning steps actually drive the final answer. This is measured by applying **Negative Preference Optimization (NPO)** to *unlearn* individual CoT sentences, then checking whether the model's answer changes. If unlearning a step causes the answer to change, that step was genuinely influencing the output.

---

## Models & Datasets

**Models evaluated:**
| Short name | HuggingFace ID |
|---|---|
| LLaMA-3-3B | `meta-llama/Llama-3.2-3B-Instruct` |
| LLaMA-3 | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Mistral-2 | `mistralai/Mistral-7B-Instruct-v0.2` |
| Phi-3 | `microsoft/Phi-3-mini-4k-instruct` |

**Datasets:**
- `arc-challenge` — ARC Challenge
- `openbook` — OpenBookQA
- `sports` — Sports Understanding (BIG-Bench)
- `sqa` — StrategyQA (requires `data/strategyqa/strategyqa_train.json`)

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Set your HuggingFace token in `unlearn.py` (line ~370):
```python
login("hf_YOUR_TOKEN_HERE")
```

StrategyQA data:
```bash
mkdir -p data/strategyqa
wget -O data/strategyqa/strategyqa_train.json \
  https://raw.githubusercontent.com/wicsaax/strategy-qa/main/strategyQA_train.json
```

---

## Compute

All 16 experiments (4 models × 4 datasets) were trained on **BigRed 200**, Indiana University's high-performance computing cluster. Each job ran on an NVIDIA A100 GPU (40GB VRAM) via SLURM — small models (Phi-3, LLaMA-3-3B) on a single GPU with 32GB RAM, and large models (LLaMA-3-8B, Mistral-7B) on 2 GPUs with 64GB RAM.

---

## Running Experiments

**Single run:**
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

**Key flags:**
| Flag | Description |
|---|---|
| `--model_name` | HuggingFace model ID |
| `--dataset` | `arc-challenge`, `openbook`, `sports`, `sqa` |
| `--method` | `npo_KL` (default), `npo`, `npo_grad_diff` |
| `--ff2` | Restrict optimization to FF2 layers (`mlp.down_proj.weight`) |
| `--pos` | Filter function tokens via spaCy POS tagging |
| `--stepwise` | Unlearn one CoT sentence at a time |
| `--strategy sentencize` | Split CoT into sentences using NLTK |
| `--new_cot` | Force regeneration of CoTs (otherwise cached in `final_cot/`) |

**Generate all 16 SLURM job scripts (BigRed200 / any SLURM cluster):**
```bash
python run_scripts.py
```
Uses `ul_step_pos_ff2.job` (32G, 12h) for small models and `ul_step_pos_ff2_L2.job` (64G, 24h) for LLaMA-3-8B and Mistral-7B.

---

## Experiment Pipeline

1. **CoT generation** (`data.py:load_or_generate_dataset_cots`) — generates or loads cached CoTs from `final_cot/{dataset}/{model}_s={seed}_t={temp}_cots.jsonl`
2. **Per-instance unlearning** (`unlearn.py:unlearn_single`) — for each instance and each CoT step, loads two model copies (trainable + frozen oracle) and applies NPO loss
3. **Evaluation after each epoch** (`unlearn.py:evaluate`) — measures CoT probability, answer probabilities (efficacy + specificity), and generates a new CoT
4. **Results** saved as JSONL to `final_results/{dataset}/{short_model}/`

### Loss Functions

- `npo` — forget loss only (NPO against frozen oracle)
- `npo_grad_diff` — forget loss + cross-entropy retain loss
- `npo_KL` — forget loss + KL divergence retain loss *(used in paper)*

---

## Results

The `final_results/` directory contains the completed experiment outputs for all 16 model × dataset combinations:

```
final_results/
├── arc-challenge/   {LLaMA-3-3B, LLaMA-3, Mistral-2, Phi-3}
├── openbook/        {LLaMA-3-3B, LLaMA-3, Mistral-2, Phi-3}
├── sports/          {LLaMA-3-3B, LLaMA-3, Mistral-2, Phi-3}
└── sqa/             {LLaMA-3-3B, LLaMA-3, Mistral-2, Phi-3}
```

Each `.out` file is a JSONL where each line is one instance:
```json
{
  "id": "...",
  "question": "...",
  "step_idx": 2,
  "correct": true,
  "initial_cot": "...",
  "initial_probs": {...},
  "unlearning_results": {
    "0": {"completion": "...", "probs": {...}, "prediction": "A", "new_cot": "...", "cot_prob": 0.42},
    "1": {"completion": "...", "probs": {...}, "prediction": "A", "new_cot": "...", "cot_prob": 0.11},
    ...
  }
}
```

---

## Code Structure

| File | Description |
|---|---|
| `unlearn.py` | Main entry point — NPO training loop, evaluation |
| `models.py` | Model loading (`load_model_and_tokenizer`), Phi-3 compatibility patch |
| `data.py` | CoT caching, `SegmentOTFDataset`, `FRCollator` |
| `dataload.py` | Dataset handlers for ARC, OpenBookQA, Sports, SQA |
| `evaluate.py` | CoT generation, completion/answer probabilities |
| `segment.py` | POS-tag based token filtering via spaCy |
| `const.py` | Model name → path mappings |
| `run_scripts.py` | Generates SLURM `sbatch` commands for all 16 jobs |
| `plotting.py` / `stats.py` | Analysis utilities |

**Notebooks:**
- `Ablations.ipynb` — paper plots and tables
- `Generate_CoT_heatmaps.ipynb` — CoT heatmap figures
- `Annotation analysis.ipynb` — human annotation study analysis
- `CoT LLM as judge.ipynb` — GPT-4o judge of post-unlearning CoT changes
- `Adding mistakes repro.ipynb` — Lanham et al. mistake-adding baseline

---

## Notes on Compatibility

- Requires `transformers>=4.45` for built-in Phi-3, LLaMA-3.2, and Mistral support
- `trust_remote_code=False` is set in `models.py` — uses the built-in transformers implementations rather than cached model code
- NPO method adapted from [licong-lin/negative-preference-optimization](https://github.com/licong-lin/negative-preference-optimization)

---

## Citation

```bibtex
@article{tutek2025measuring,
  title={Measuring Faithfulness of Chains of Thought by Unlearning Reasoning Steps},
  author={Tutek, Martin and Chaleshtori, Farzad Habibi and Marasovi{\'c}, Ana and Belinkov, Yonatan},
  journal={arXiv preprint arXiv:2502.14829},
  year={2025}
}
```
