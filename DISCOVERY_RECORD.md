# Discovery Record: The Alignment-Faithfulness Paradox

**Date discovered:** April 6, 2026
**Discovered by:** Madhavan Balaji
**Institution:** Indiana University Indianapolis
**Advisor:** Professor Hyeju Jang

## Core Finding

Fine-tuning LLaMA-3-3B on high-quality, human-preferred CoT outputs
reduces parametric faithfulness from 30.0% to 8.0% — a 73% relative
drop. Fine-tuning on low-quality CoTs leaves faithfulness nearly
unchanged (30.0% → 28.0%).

This is the first empirical demonstration that CoT quality fine-tuning
systematically reduces parametric faithfulness as measured by NPO
unlearning (Tutek et al., EMNLP 2025).

## The Paradox

Better-looking CoT → Less verifiable CoT.

Training models to produce more coherent, human-preferred reasoning
chains actively decouples those chains from the parametric knowledge
that actually drives model predictions. This has direct implications
for AI safety and CoT monitoring.

## Experimental Evidence

| Condition        | Binary Faithfulness | Mean delta_p |
|------------------|--------------------:|-------------:|
| Baseline         | 30.0%               | 0.0006       |
| High-quality FT  | 8.0%                | 0.0019       |
| Low-quality FT   | 28.0%               | 0.0003       |

Model: LLaMA-3-3B (meta-llama/Llama-3.2-3B-Instruct)
Test set: 50 held-out instances
Method: NPO unlearning (npo_KL, ff2=True, pos=True, lr=3e-05, 5 epochs)

## Continuous Score Observation

The continuous delta_p reveals a paradox within the paradox:
high-quality FT increases mean delta_p (0.0006 → 0.0019) while
simultaneously reducing binary faithfulness (30% → 8%). This suggests
high-quality FT creates a bimodal distribution: most steps become
completely decoupled, but the few remaining coupled steps are more
strongly coupled than before.

## Infrastructure

- Compute: BigRed200 HPC, Indiana University
- GPU: NVIDIA A100 40GB VRAM
- SLURM Job IDs: 6754442 (eval-high), 6754443 (eval-low)
- Fine-tuning: LoRA rank=16, alpha=32, lr=2e-4, 3 epochs
- Framework: NPO unlearning from Tutek et al. (EMNLP 2025)

## Replication Status

Experiments running on 3 additional models as of April 6, 2026:
- LLaMA-3-8B (eval jobs: 6754957, running)
- Mistral-7B (eval jobs: 6754962, running)
- Phi-3 (pending)
- Model-specific splits for all 4 models (pending)

## Build History

This work builds on a full reproduction of Tutek et al. (EMNLP 2025)
run on BigRed200. The reproduction found and fixed 3 bugs in the
original codebase, introduced continuous delta_p scoring, and
discovered subcritical faithfulness — all prior to this experiment.

## Repository

Private as of April 6, 2026.
Will be made public upon arXiv submission.
