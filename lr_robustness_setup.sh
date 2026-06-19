#!/bin/bash
# lr_robustness_setup.sh
# ======================
# Run locally to upload files to BigRed200 and set up the environment
# for the LLaMA-3-3B LR sensitivity sweep.
#
# Usage: bash lr_robustness_setup.sh

set -euo pipefail
REMOTE="madbala@bigred200.uits.iu.edu"
WORKDIR="/N/scratch/madbala/parametric-faithfulness_run"

echo "=== Step 1: Upload required files ==="
# Use br200_scp_up.py pattern for Duo auth
python3 br200_scp_up.py 2 evaluate_finetuned.py "$WORKDIR/evaluate_finetuned.py"
python3 br200_scp_up.py 2 finetune_lora.py "$WORKDIR/finetune_lora.py"
python3 br200_scp_up.py 2 requirements.txt "$WORKDIR/requirements.txt"
python3 br200_scp_up.py 2 lr_robustness_jobs.sh "$WORKDIR/lr_robustness_jobs.sh"

echo "=== Step 2: Upload training data ==="
python3 br200_cmd.py 2 "mkdir -p $WORKDIR/finetune_data"
python3 br200_scp_up.py 2 finetune_data/high_quality.jsonl "$WORKDIR/finetune_data/high_quality.jsonl"
python3 br200_scp_up.py 2 finetune_data/low_quality.jsonl "$WORKDIR/finetune_data/low_quality.jsonl"
python3 br200_scp_up.py 2 finetune_data/test_held_out.jsonl "$WORKDIR/finetune_data/test_held_out.jsonl"

echo "=== Step 3: Create venv and install deps ==="
python3 br200_cmd.py 2 "cd $WORKDIR && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && python -m spacy download en_core_web_sm"

echo "=== Step 4: Create output directories ==="
python3 br200_cmd.py 2 "mkdir -p $WORKDIR/finetuned_results_calibrated_lr/llama3b_lr1e-05/{baseline,high_quality,low_quality} $WORKDIR/finetuned_results_calibrated_lr/llama3b_lr5e-05/{baseline,high_quality,low_quality} $WORKDIR/lora_adapters"

echo "=== Step 5: Submit jobs ==="
python3 br200_cmd.py 2 "cd $WORKDIR && bash lr_robustness_jobs.sh"

echo ""
echo "=== DONE ==="
echo "Monitor with: python3 br200_cmd.py 2 'squeue -u madbala'"
echo "When complete, download results with:"
echo "  python3 br200_scp_down.py 2 $WORKDIR/finetuned_results_calibrated_lr/ finetuned_results_calibrated_lr/"
echo "Then rerun: python analysis/lr_robustness_check.py"
