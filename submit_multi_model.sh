#!/bin/bash
# Submit alignment-faithfulness jobs for LLaMA-3-8B, Mistral-7B, Phi-3
set -e
cd /N/scratch/madbala/parametric-faithfulness_run

VENV="source /N/scratch/madbala/parametric-faithfulness_run/venv/bin/activate"
ENVVARS="export HF_HOME=/N/scratch/madbala/hf_cache && export HF_TOKEN=REDACTED_HF_TOKEN && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
WD="cd /N/scratch/madbala/parametric-faithfulness_run"
PRE="$VENV && $ENVVARS && $WD"

# ── LLaMA-3-8B ──────────────────────────────────────────────
JID_L8_1=$(sbatch --parsable -A c01949 --job-name=ft-llama8b-hi \
  --nodes=1 --ntasks=1 --gpus-per-node=2 --mem=64G --time=06:00:00 -p gpu \
  --wrap="$PRE && python finetune_lora.py --data_path finetune_data/high_quality.jsonl --output_dir lora_adapters/llama8b_high --model_name meta-llama/Meta-Llama-3-8B-Instruct --epochs 3 --lr 2e-4 --batch_size 2 --grad_accum 8")
echo "LLaMA-8B ft-high: $JID_L8_1"

JID_L8_2=$(sbatch --parsable -A c01949 --job-name=ft-llama8b-lo \
  --nodes=1 --ntasks=1 --gpus-per-node=2 --mem=64G --time=06:00:00 -p gpu \
  --wrap="$PRE && python finetune_lora.py --data_path finetune_data/low_quality.jsonl --output_dir lora_adapters/llama8b_low --model_name meta-llama/Meta-Llama-3-8B-Instruct --epochs 3 --lr 2e-4 --batch_size 2 --grad_accum 8")
echo "LLaMA-8B ft-low:  $JID_L8_2"

sbatch -A c01949 --job-name=ev-llama8b-bs \
  --nodes=1 --ntasks=1 --gpus-per-node=2 --mem=64G --time=12:00:00 -p gpu \
  --wrap="$PRE && python evaluate_finetuned.py --condition baseline --output_dir finetuned_results/llama8b_baseline --model_name meta-llama/Meta-Llama-3-8B-Instruct --lr 1e-05 --epochs 5"

sbatch --dependency=afterok:$JID_L8_1 -A c01949 --job-name=ev-llama8b-hi \
  --nodes=1 --ntasks=1 --gpus-per-node=2 --mem=64G --time=12:00:00 -p gpu \
  --wrap="$PRE && python evaluate_finetuned.py --condition high_quality --adapter_path lora_adapters/llama8b_high --output_dir finetuned_results/llama8b_high --model_name meta-llama/Meta-Llama-3-8B-Instruct --lr 1e-05 --epochs 5"

sbatch --dependency=afterok:$JID_L8_2 -A c01949 --job-name=ev-llama8b-lo \
  --nodes=1 --ntasks=1 --gpus-per-node=2 --mem=64G --time=12:00:00 -p gpu \
  --wrap="$PRE && python evaluate_finetuned.py --condition low_quality --adapter_path lora_adapters/llama8b_low --output_dir finetuned_results/llama8b_low --model_name meta-llama/Meta-Llama-3-8B-Instruct --lr 1e-05 --epochs 5"

echo "=== LLaMA-3-8B: 5 jobs submitted ==="

# ── Mistral-7B ──────────────────────────────────────────────
JID_M1=$(sbatch --parsable -A c01949 --job-name=ft-mistral-hi \
  --nodes=1 --ntasks=1 --gpus-per-node=2 --mem=64G --time=06:00:00 -p gpu \
  --wrap="$PRE && python finetune_lora.py --data_path finetune_data/high_quality.jsonl --output_dir lora_adapters/mistral_high --model_name mistralai/Mistral-7B-Instruct-v0.2 --epochs 3 --lr 2e-4 --batch_size 2 --grad_accum 8")
echo "Mistral ft-high:  $JID_M1"

JID_M2=$(sbatch --parsable -A c01949 --job-name=ft-mistral-lo \
  --nodes=1 --ntasks=1 --gpus-per-node=2 --mem=64G --time=06:00:00 -p gpu \
  --wrap="$PRE && python finetune_lora.py --data_path finetune_data/low_quality.jsonl --output_dir lora_adapters/mistral_low --model_name mistralai/Mistral-7B-Instruct-v0.2 --epochs 3 --lr 2e-4 --batch_size 2 --grad_accum 8")
echo "Mistral ft-low:   $JID_M2"

sbatch -A c01949 --job-name=ev-mistral-bs \
  --nodes=1 --ntasks=1 --gpus-per-node=2 --mem=64G --time=12:00:00 -p gpu \
  --wrap="$PRE && python evaluate_finetuned.py --condition baseline --output_dir finetuned_results/mistral_baseline --model_name mistralai/Mistral-7B-Instruct-v0.2 --lr 5e-06 --epochs 5"

sbatch --dependency=afterok:$JID_M1 -A c01949 --job-name=ev-mistral-hi \
  --nodes=1 --ntasks=1 --gpus-per-node=2 --mem=64G --time=12:00:00 -p gpu \
  --wrap="$PRE && python evaluate_finetuned.py --condition high_quality --adapter_path lora_adapters/mistral_high --output_dir finetuned_results/mistral_high --model_name mistralai/Mistral-7B-Instruct-v0.2 --lr 5e-06 --epochs 5"

sbatch --dependency=afterok:$JID_M2 -A c01949 --job-name=ev-mistral-lo \
  --nodes=1 --ntasks=1 --gpus-per-node=2 --mem=64G --time=12:00:00 -p gpu \
  --wrap="$PRE && python evaluate_finetuned.py --condition low_quality --adapter_path lora_adapters/mistral_low --output_dir finetuned_results/mistral_low --model_name mistralai/Mistral-7B-Instruct-v0.2 --lr 5e-06 --epochs 5"

echo "=== Mistral-7B: 5 jobs submitted ==="

# ── Phi-3 ───────────────────────────────────────────────────
JID_P1=$(sbatch --parsable -A c01949 --job-name=ft-phi3-hi \
  --nodes=1 --ntasks=1 --gpus-per-node=1 --mem=32G --time=06:00:00 -p gpu \
  --wrap="$PRE && python finetune_lora.py --data_path finetune_data/high_quality.jsonl --output_dir lora_adapters/phi3_high --model_name microsoft/Phi-3-mini-4k-instruct --epochs 3 --lr 2e-4 --batch_size 4 --grad_accum 4")
echo "Phi-3 ft-high:    $JID_P1"

JID_P2=$(sbatch --parsable -A c01949 --job-name=ft-phi3-lo \
  --nodes=1 --ntasks=1 --gpus-per-node=1 --mem=32G --time=06:00:00 -p gpu \
  --wrap="$PRE && python finetune_lora.py --data_path finetune_data/low_quality.jsonl --output_dir lora_adapters/phi3_low --model_name microsoft/Phi-3-mini-4k-instruct --epochs 3 --lr 2e-4 --batch_size 4 --grad_accum 4")
echo "Phi-3 ft-low:     $JID_P2"

sbatch -A c01949 --job-name=ev-phi3-bs \
  --nodes=1 --ntasks=1 --gpus-per-node=1 --mem=32G --time=12:00:00 -p gpu \
  --wrap="$PRE && python evaluate_finetuned.py --condition baseline --output_dir finetuned_results/phi3_baseline --model_name microsoft/Phi-3-mini-4k-instruct --lr 1e-04 --epochs 5"

sbatch --dependency=afterok:$JID_P1 -A c01949 --job-name=ev-phi3-hi \
  --nodes=1 --ntasks=1 --gpus-per-node=1 --mem=32G --time=12:00:00 -p gpu \
  --wrap="$PRE && python evaluate_finetuned.py --condition high_quality --adapter_path lora_adapters/phi3_high --output_dir finetuned_results/phi3_high --model_name microsoft/Phi-3-mini-4k-instruct --lr 1e-04 --epochs 5"

sbatch --dependency=afterok:$JID_P2 -A c01949 --job-name=ev-phi3-lo \
  --nodes=1 --ntasks=1 --gpus-per-node=1 --mem=32G --time=12:00:00 -p gpu \
  --wrap="$PRE && python evaluate_finetuned.py --condition low_quality --adapter_path lora_adapters/phi3_low --output_dir finetuned_results/phi3_low --model_name microsoft/Phi-3-mini-4k-instruct --lr 1e-04 --epochs 5"

echo "=== Phi-3: 5 jobs submitted ==="
echo ""
echo "=== FULL QUEUE ==="
squeue -u madbala
echo ""
echo "Expected completion: ~12 hours from now"
echo "When done run: bash run_after_jobs.sh"
