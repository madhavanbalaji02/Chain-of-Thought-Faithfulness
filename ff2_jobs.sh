#!/bin/bash
# Submit FF2 extraction jobs for 3 conditions on BigRed200
# LLaMA-3-3B only (fits in 32GB with device_map=auto)

set -e
WORKDIR=/N/scratch/madbala/parametric-faithfulness_run
SBATCH="sbatch -A c01949 --nodes=1 --ntasks=1 --gres=gpu:1 --mem=32G --time=2:00:00 --partition=gpu"
PYTHON="$WORKDIR/venv/bin/python3"

# Baseline (no adapter)
J1=$($SBATCH --job-name=ff2-base \
  --output=$WORKDIR/slurm-ff2-base-%j.out \
  --wrap="cd $WORKDIR && $PYTHON extract_ff2_activations.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --results_file finetuned_results/baseline/results.jsonl \
    --test_data finetune_data/test_held_out.jsonl \
    --output_file ff2_activations/baseline.pkl \
    --n_instances 50 \
    --layers 8,16,24" | awk '{print $4}')
echo "ff2-baseline: $J1"

# High-quality FT
J2=$($SBATCH --job-name=ff2-high \
  --output=$WORKDIR/slurm-ff2-high-%j.out \
  --wrap="cd $WORKDIR && $PYTHON extract_ff2_activations.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --adapter_path lora_adapters/high_quality \
    --results_file finetuned_results/high_quality/results.jsonl \
    --test_data finetune_data/test_held_out.jsonl \
    --output_file ff2_activations/high_ft.pkl \
    --n_instances 50 \
    --layers 8,16,24" | awk '{print $4}')
echo "ff2-high: $J2"

# Low-quality FT
J3=$($SBATCH --job-name=ff2-low \
  --output=$WORKDIR/slurm-ff2-low-%j.out \
  --wrap="cd $WORKDIR && $PYTHON extract_ff2_activations.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --adapter_path lora_adapters/low_quality \
    --results_file finetuned_results/low_quality/results.jsonl \
    --test_data finetune_data/test_held_out.jsonl \
    --output_file ff2_activations/low_ft.pkl \
    --n_instances 50 \
    --layers 8,16,24" | awk '{print $4}')
echo "ff2-low: $J3"

echo ""
echo "All 3 FF2 extraction jobs submitted: $J1 $J2 $J3"
echo "Check: squeue -u madbala"
echo "Logs:  $WORKDIR/slurm-ff2-*.out"
