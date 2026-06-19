#!/bin/bash
# lr_robustness_jobs.sh
# =====================
# LLaMA-3-3B LR sensitivity sweep for the alignment-faithfulness paradox.
#
# Tests whether the collapse (30% -> 8%) holds at two LRs bracketing
# the original 3e-05: lr=1e-05 (weaker unlearning) and lr=5e-05 (stronger).
#
# PREREQUISITE: LoRA adapters must exist. They were purged from scratch;
# Job 0a/0b recreate them using the same finetune_lora.py settings as
# the original new_experiment_jobs.py / submit_multi_model.sh.
#
# Run from: /N/scratch/madbala/parametric-faithfulness_run
# Account:  c02130

set -euo pipefail
ACCOUNT="c02130"
MODEL="meta-llama/Llama-3.2-3B-Instruct"
PRE="source venv/bin/activate && export HF_HOME=/N/scratch/madbala/hf_cache && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

echo "# ══════════════════════════════════════════════════════════════"
echo "# LR Robustness Check — LLaMA-3-3B Sensitivity Sweep"
echo "# ══════════════════════════════════════════════════════════════"
echo ""
echo "# ── Step 0: Recreate LoRA adapters (purged from scratch) ──────"
echo "# These use the SAME hyperparameters as the original experiment."
echo ""

JID_HI=$(sbatch --parsable -A $ACCOUNT --job-name=ft-hi-recreate \
  --nodes=1 --ntasks=1 --gpus-per-node=1 \
  --mem=32G --time=06:00:00 \
  --wrap="$PRE && python finetune_lora.py \
    --data_path finetune_data/high_quality.jsonl \
    --output_dir lora_adapters/high_quality \
    --model_name $MODEL \
    --epochs 3 --lr 2e-4 --batch_size 4 --grad_accum 4")
echo "Finetune high-quality adapter: $JID_HI"

JID_LO=$(sbatch --parsable -A $ACCOUNT --job-name=ft-lo-recreate \
  --nodes=1 --ntasks=1 --gpus-per-node=1 \
  --mem=32G --time=06:00:00 \
  --wrap="$PRE && python finetune_lora.py \
    --data_path finetune_data/low_quality.jsonl \
    --output_dir lora_adapters/low_quality \
    --model_name $MODEL \
    --epochs 3 --lr 2e-4 --batch_size 4 --grad_accum 4")
echo "Finetune low-quality adapter:  $JID_LO"

echo ""
echo "# ── LR = 1e-05 (weaker unlearning than original 3e-05) ───────"
echo ""

sbatch --parsable -A $ACCOUNT --job-name=ev-1e05-bs \
  --nodes=1 --ntasks=1 --gpus-per-node=1 \
  --mem=32G --time=12:00:00 \
  --wrap="$PRE && python evaluate_finetuned.py \
    --condition baseline \
    --output_dir finetuned_results_calibrated_lr/llama3b_lr1e-05/baseline \
    --model_name $MODEL \
    --lr 1e-05 --epochs 5"

sbatch --dependency=afterok:$JID_HI --parsable -A $ACCOUNT --job-name=ev-1e05-hi \
  --nodes=1 --ntasks=1 --gpus-per-node=1 \
  --mem=32G --time=12:00:00 \
  --wrap="$PRE && python evaluate_finetuned.py \
    --condition high_quality \
    --adapter_path lora_adapters/high_quality \
    --output_dir finetuned_results_calibrated_lr/llama3b_lr1e-05/high_quality \
    --model_name $MODEL \
    --lr 1e-05 --epochs 5"

sbatch --dependency=afterok:$JID_LO --parsable -A $ACCOUNT --job-name=ev-1e05-lo \
  --nodes=1 --ntasks=1 --gpus-per-node=1 \
  --mem=32G --time=12:00:00 \
  --wrap="$PRE && python evaluate_finetuned.py \
    --condition low_quality \
    --adapter_path lora_adapters/low_quality \
    --output_dir finetuned_results_calibrated_lr/llama3b_lr1e-05/low_quality \
    --model_name $MODEL \
    --lr 1e-05 --epochs 5"

echo ""
echo "# ── LR = 5e-05 (stronger unlearning than original 3e-05) ─────"
echo ""

sbatch --parsable -A $ACCOUNT --job-name=ev-5e05-bs \
  --nodes=1 --ntasks=1 --gpus-per-node=1 \
  --mem=32G --time=12:00:00 \
  --wrap="$PRE && python evaluate_finetuned.py \
    --condition baseline \
    --output_dir finetuned_results_calibrated_lr/llama3b_lr5e-05/baseline \
    --model_name $MODEL \
    --lr 5e-05 --epochs 5"

sbatch --dependency=afterok:$JID_HI --parsable -A $ACCOUNT --job-name=ev-5e05-hi \
  --nodes=1 --ntasks=1 --gpus-per-node=1 \
  --mem=32G --time=12:00:00 \
  --wrap="$PRE && python evaluate_finetuned.py \
    --condition high_quality \
    --adapter_path lora_adapters/high_quality \
    --output_dir finetuned_results_calibrated_lr/llama3b_lr5e-05/high_quality \
    --model_name $MODEL \
    --lr 5e-05 --epochs 5"

sbatch --dependency=afterok:$JID_LO --parsable -A $ACCOUNT --job-name=ev-5e05-lo \
  --nodes=1 --ntasks=1 --gpus-per-node=1 \
  --mem=32G --time=12:00:00 \
  --wrap="$PRE && python evaluate_finetuned.py \
    --condition low_quality \
    --adapter_path lora_adapters/low_quality \
    --output_dir finetuned_results_calibrated_lr/llama3b_lr5e-05/low_quality \
    --model_name $MODEL \
    --lr 5e-05 --epochs 5"

echo ""
echo "# ── Mistral-7B calibrated-LR rerun (lr=5e-06, the paper's LR) ───"
echo "# sacct shows the existing Mistral results used lr=1e-05, not 5e-06."
echo "# This reruns at the correct calibrated LR to check the paradox."
echo ""

MISTRAL="mistralai/Mistral-7B-Instruct-v0.2"

JID_MH=$(sbatch --parsable -A $ACCOUNT --job-name=ft-ms-hi-recreate \
  --nodes=1 --ntasks=1 --gpus-per-node=2 \
  --mem=64G --time=06:00:00 \
  --wrap="$PRE && python finetune_lora.py \
    --data_path finetune_data/high_quality.jsonl \
    --output_dir lora_adapters/mistral_high \
    --model_name $MISTRAL \
    --epochs 3 --lr 2e-4 --batch_size 2 --grad_accum 8")
echo "Finetune Mistral high-quality: $JID_MH"

JID_ML=$(sbatch --parsable -A $ACCOUNT --job-name=ft-ms-lo-recreate \
  --nodes=1 --ntasks=1 --gpus-per-node=2 \
  --mem=64G --time=06:00:00 \
  --wrap="$PRE && python finetune_lora.py \
    --data_path finetune_data/low_quality.jsonl \
    --output_dir lora_adapters/mistral_low \
    --model_name $MISTRAL \
    --epochs 3 --lr 2e-4 --batch_size 2 --grad_accum 8")
echo "Finetune Mistral low-quality:  $JID_ML"

sbatch --parsable -A $ACCOUNT --job-name=ev-ms-5e06-bs \
  --nodes=1 --ntasks=1 --gpus-per-node=2 \
  --mem=64G --time=24:00:00 \
  --wrap="$PRE && python evaluate_finetuned.py \
    --condition baseline \
    --output_dir finetuned_results_calibrated_lr/mistral_lr5e-06/baseline \
    --model_name $MISTRAL \
    --lr 5e-06 --epochs 5"

sbatch --dependency=afterok:$JID_MH --parsable -A $ACCOUNT --job-name=ev-ms-5e06-hi \
  --nodes=1 --ntasks=1 --gpus-per-node=2 \
  --mem=64G --time=24:00:00 \
  --wrap="$PRE && python evaluate_finetuned.py \
    --condition high_quality \
    --adapter_path lora_adapters/mistral_high \
    --output_dir finetuned_results_calibrated_lr/mistral_lr5e-06/high_quality \
    --model_name $MISTRAL \
    --lr 5e-06 --epochs 5"

sbatch --dependency=afterok:$JID_ML --parsable -A $ACCOUNT --job-name=ev-ms-5e06-lo \
  --nodes=1 --ntasks=1 --gpus-per-node=2 \
  --mem=64G --time=24:00:00 \
  --wrap="$PRE && python evaluate_finetuned.py \
    --condition low_quality \
    --adapter_path lora_adapters/mistral_low \
    --output_dir finetuned_results_calibrated_lr/mistral_lr5e-06/low_quality \
    --model_name $MISTRAL \
    --lr 5e-06 --epochs 5"

echo ""
echo "# ══════════════════════════════════════════════════════════════"
echo "# Total: 4 finetune jobs + 9 eval jobs = 13 jobs"
echo "#   - 2 LLaMA-3-3B adapter recreations + 6 LLaMA-3-3B evals"
echo "#   - 2 Mistral adapter recreations + 3 Mistral evals at lr=5e-06"
echo "# Eval jobs depend on finetune jobs via --dependency=afterok"
echo "# Baseline evals (no adapter) submit immediately."
echo "#"
echo "# When all complete, download results and rerun:"
echo "#   python analysis/lr_robustness_check.py"
echo "# ══════════════════════════════════════════════════════════════"
