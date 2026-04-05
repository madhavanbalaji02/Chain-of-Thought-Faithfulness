"""
new_experiment_jobs.py
======================
Prints sbatch commands for the 5-job alignment-faithfulness experiment.

Pipeline:
  Job 1: fine-tune on HIGH quality CoT  → lora_adapters/high_quality/
  Job 2: fine-tune on LOW quality CoT   → lora_adapters/low_quality/
  Job 3: evaluate BASELINE (no adapter) → finetuned_results/baseline/
  Job 4: evaluate HIGH-quality model    → finetuned_results/high_quality/   [depends on Job 1]
  Job 5: evaluate LOW-quality model     → finetuned_results/low_quality/    [depends on Job 2]

Run from: /N/scratch/madbala/parametric-faithfulness_run
"""

BASE_MODEL  = 'meta-llama/Llama-3.2-3B-Instruct'
SMALL_JOB   = 'ul_step_pos_ff2.job'   # reuse for eval (1× A100, 32G, 12h)
ACCOUNT     = 'c01949'

print("# ══════════════════════════════════════════════════")
print("# Alignment-Faithfulness Experiment — 5 SLURM jobs")
print("# Run from: /N/scratch/madbala/parametric-faithfulness_run")
print("# ══════════════════════════════════════════════════")
print()
print("# ── Job 1: Fine-tune on HIGH quality CoT ──────────────────────────")
print("# Runtime: ~2–3h on 1× A100, 32G")
print(f"sbatch -A {ACCOUNT} \\")
print( "  --job-name=finetune-high \\")
print( "  --nodes=1 --ntasks=1 --gpus-per-node=1 \\")
print( "  --mem=32G --time=06:00:00 \\")
print( "  --wrap='python finetune_lora.py \\")
print(f"    --data_path finetune_data/high_quality.jsonl \\")
print(f"    --output_dir lora_adapters/high_quality \\")
print(f"    --model_name {BASE_MODEL} \\")
print( "    --epochs 3 --lr 2e-4 --batch_size 4 --grad_accum 4'")
print()
print("# ── Job 2: Fine-tune on LOW quality CoT ───────────────────────────")
print("# Runtime: ~2–3h on 1× A100, 32G")
print(f"sbatch -A {ACCOUNT} \\")
print( "  --job-name=finetune-low \\")
print( "  --nodes=1 --ntasks=1 --gpus-per-node=1 \\")
print( "  --mem=32G --time=06:00:00 \\")
print( "  --wrap='python finetune_lora.py \\")
print(f"    --data_path finetune_data/low_quality.jsonl \\")
print(f"    --output_dir lora_adapters/low_quality \\")
print(f"    --model_name {BASE_MODEL} \\")
print( "    --epochs 3 --lr 2e-4 --batch_size 4 --grad_accum 4'")
print()
print("# ── Job 3: Evaluate BASELINE (submit immediately) ─────────────────")
print("# Runtime: ~4–6h on 1× A100 (50 instances × 5 unlearn epochs × 2 model loads)")
print(f"sbatch -A {ACCOUNT} \\")
print( "  --job-name=eval-baseline \\")
print( "  --nodes=1 --ntasks=1 --gpus-per-node=1 \\")
print( "  --mem=32G --time=12:00:00 \\")
print( "  --wrap='python evaluate_finetuned.py \\")
print( "    --condition baseline \\")
print(f"    --output_dir finetuned_results/baseline \\")
print(f"    --model_name {BASE_MODEL} \\")
print( "    --lr 3e-05 --epochs 5'")
print()
print("# ── Job 4: Evaluate HIGH-quality model (submit after Job 1 completes) ──")
print("# Replace JOBID_1 with the actual job ID returned by Job 1's sbatch")
print("# sbatch --dependency=afterok:JOBID_1 \\")
print(f"sbatch -A {ACCOUNT} \\")
print( "  --job-name=eval-high \\")
print( "  --nodes=1 --ntasks=1 --gpus-per-node=1 \\")
print( "  --mem=32G --time=12:00:00 \\")
print( "  --wrap='python evaluate_finetuned.py \\")
print( "    --condition high_quality \\")
print(f"    --adapter_path lora_adapters/high_quality \\")
print(f"    --output_dir finetuned_results/high_quality \\")
print(f"    --model_name {BASE_MODEL} \\")
print( "    --lr 3e-05 --epochs 5'")
print()
print("# ── Job 5: Evaluate LOW-quality model (submit after Job 2 completes) ──")
print("# Replace JOBID_2 with the actual job ID returned by Job 2's sbatch")
print("# sbatch --dependency=afterok:JOBID_2 \\")
print(f"sbatch -A {ACCOUNT} \\")
print( "  --job-name=eval-low \\")
print( "  --nodes=1 --ntasks=1 --gpus-per-node=1 \\")
print( "  --mem=32G --time=12:00:00 \\")
print( "  --wrap='python evaluate_finetuned.py \\")
print( "    --condition low_quality \\")
print(f"    --adapter_path lora_adapters/low_quality \\")
print(f"    --output_dir finetuned_results/low_quality \\")
print(f"    --model_name {BASE_MODEL} \\")
print( "    --lr 3e-05 --epochs 5'")
print()
print("# ══════════════════════════════════════════════════")
print("# NOTE: Jobs 4 and 5 can also be submitted with SLURM dependencies:")
print("#   JID1=$(sbatch --parsable ... Job1 ...)")
print("#   sbatch --dependency=afterok:$JID1 ... Job4 ...")
print("# This avoids manual monitoring.")
print("# ══════════════════════════════════════════════════")
