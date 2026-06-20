# SACCT Provenance Archive

**Pulled**: 2026-06-19 ~21:00 UTC from BigRed200 via `sacct --format=SubmitLine`
**Purpose**: Preserve the only surviving evidence of which `--lr` was used for each
evaluation job feeding Table 1 of paper.tex, since SLURM job logs and scratch
filesystem results were purged.

> **Note**: HF_TOKEN values redacted below. The original sacct output contained
> plaintext tokens; those should be rotated.

---

## Summary Table

| Model | Condition | Job ID | LR Used | Submit Time | Status |
|:------|:----------|:------:|:-------:|:------------|:-------|
| LLaMA-3-3B | baseline | 6750454 | **3e-05** | 2026-04-05 13:53 | COMPLETED |
| LLaMA-3-3B | high-FT | 6754442 | **3e-05** | 2026-04-06 01:20 | COMPLETED |
| LLaMA-3-3B | low-FT | 6754443 | **3e-05** | 2026-04-06 01:20 | COMPLETED |
| Phi-3 | baseline | 6754967 | **1e-04** | 2026-04-06 01:35 | COMPLETED |
| Phi-3 | high-FT | 6754968 | **1e-04** | 2026-04-06 01:35 | COMPLETED |
| Phi-3 | low-FT | 6754969 | **1e-04** | 2026-04-06 01:35 | COMPLETED |
| Mistral-7B | baseline | 6815741 | **1e-05** | 2026-04-09 19:19 | COMPLETED |
| Mistral-7B | high-FT | 6815742 | **1e-05** | 2026-04-09 19:19 | COMPLETED |
| Mistral-7B | low-FT | 6815743 | **1e-05** | 2026-04-09 19:19 | COMPLETED |
| LLaMA-3-8B | baseline | 6815744 | **1e-05** | 2026-04-09 19:19 | COMPLETED |
| LLaMA-3-8B | high-FT | 6815745 | **1e-05** | 2026-04-09 19:19 | COMPLETED |
| LLaMA-3-8B | low-FT | 6815746 | **1e-05** | 2026-04-09 19:19 | COMPLETED |

### Cross-reference with `const.py:paper_best_lr`

| Model | LR Used | Paper Best LR | Match? |
|:------|:-------:|:-------------:|:------:|
| LLaMA-3-3B | 3e-05 | 3e-05 | YES |
| Phi-3 | 1e-04 | 1e-04 | YES |
| Mistral-7B | 1e-05 | 5e-06 | **NO** (2× calibrated value) |
| LLaMA-3-8B | 1e-05 | 1e-05 | YES |

---

## Verbatim sacct Records

### LLaMA-3-3B baseline — Job 6750454

```
JobID: 6750454 | JobName: eval-baseline | State: COMPLETED
Submit: 2026-04-05T13:53:39 | Start: 2026-04-05T21:48:25 | End: 2026-04-05T22:00:41

SubmitLine:
sbatch -A c01949 --job-name=eval-baseline --nodes=1 --ntasks=1 --gpus-per-node=1
  --mem=32G --time=12:00:00 -p gpu
  --wrap='... && python evaluate_finetuned.py
    --condition baseline
    --output_dir finetuned_results/baseline
    --model_name meta-llama/Llama-3.2-3B-Instruct
    --lr 3e-05 --epochs 5'
```

### LLaMA-3-3B high-FT — Job 6754442

```
JobID: 6754442 | JobName: eval-high | State: COMPLETED
Submit: 2026-04-06T01:20:25 | Start: 2026-04-06T07:49:05 | End: 2026-04-06T08:39:22

SubmitLine:
sbatch --parsable -A c01949 --job-name=eval-high --nodes=1 --ntasks=1 --gpus-per-node=1
  --mem=32G --time=12:00:00 -p gpu
  --wrap='... && python evaluate_finetuned.py
    --condition high_quality
    --adapter_path lora_adapters/high_quality
    --output_dir finetuned_results/high_quality
    --model_name meta-llama/Llama-3.2-3B-Instruct
    --lr 3e-05 --epochs 5'
```

### LLaMA-3-3B low-FT — Job 6754443

```
JobID: 6754443 | JobName: eval-low | State: COMPLETED
Submit: 2026-04-06T01:20:25 | Start: 2026-04-06T07:54:05 | End: 2026-04-06T08:39:22

SubmitLine:
sbatch --parsable -A c01949 --job-name=eval-low --nodes=1 --ntasks=1 --gpus-per-node=1
  --mem=32G --time=12:00:00 -p gpu
  --wrap='... && python evaluate_finetuned.py
    --condition low_quality
    --adapter_path lora_adapters/low_quality
    --output_dir finetuned_results/low_quality
    --model_name meta-llama/Llama-3.2-3B-Instruct
    --lr 3e-05 --epochs 5'
```

### Phi-3 baseline — Job 6754967

```
JobID: 6754967 | JobName: ev-phi3-bs | State: COMPLETED
Submit: 2026-04-06T01:35:31 | Start: 2026-04-06T15:59:50 | End: 2026-04-06T16:25:17

SubmitLine:
sbatch -A c01949 --job-name=ev-phi3-bs --nodes=1 --ntasks=1 --gpus-per-node=1
  --mem=32G --time=12:00:00 -p gpu
  --wrap='... && python evaluate_finetuned.py
    --condition baseline
    --output_dir finetuned_results/phi3_baseline
    --model_name microsoft/Phi-3-mini-4k-instruct
    --lr 1e-04 --epochs 5'
```

### Phi-3 high-FT — Job 6754968

```
JobID: 6754968 | JobName: ev-phi3-hi | State: COMPLETED
Submit: 2026-04-06T01:35:31 | Start: 2026-04-07T02:16:00 | End: 2026-04-07T02:28:42

SubmitLine:
sbatch --dependency=afterok:6754965 -A c01949 --job-name=ev-phi3-hi --nodes=1 --ntasks=1
  --gpus-per-node=1 --mem=32G --time=12:00:00 -p gpu
  --wrap='... && python evaluate_finetuned.py
    --condition high_quality
    --adapter_path lora_adapters/phi3_high
    --output_dir finetuned_results/phi3_high
    --model_name microsoft/Phi-3-mini-4k-instruct
    --lr 1e-04 --epochs 5'
```

### Phi-3 low-FT — Job 6754969

```
JobID: 6754969 | JobName: ev-phi3-lo | State: COMPLETED
Submit: 2026-04-06T01:35:31 | Start: 2026-04-07T02:31:01 | End: 2026-04-07T02:43:44

SubmitLine:
sbatch --dependency=afterok:6754966 -A c01949 --job-name=ev-phi3-lo --nodes=1 --ntasks=1
  --gpus-per-node=1 --mem=32G --time=12:00:00 -p gpu
  --wrap='... && python evaluate_finetuned.py
    --condition low_quality
    --adapter_path lora_adapters/phi3_low
    --output_dir finetuned_results/phi3_low
    --model_name microsoft/Phi-3-mini-4k-instruct
    --lr 1e-04 --epochs 5'
```

### Mistral-7B baseline — Job 6815741

```
JobID: 6815741 | JobName: ev-ms-bs-4bit | State: COMPLETED
Submit: 2026-04-09T19:19:17 | Start: 2026-04-10T07:03:08 | End: 2026-04-10T07:10:17

SubmitLine:
sbatch -A c01949 --nodes=1 --ntasks-per-node=1 --gpus-per-node=2 --mem=64G --time=24:00:00
  -p gpu --job-name=ev-ms-bs-4bit
  --wrap='... && python3 evaluate_finetuned.py
    --condition baseline
    --output_dir finetuned_results/mistral_baseline
    --test_data finetune_data/test_held_out.jsonl
    --model_name mistralai/Mistral-7B-Instruct-v0.2
    --lr 1e-05 --epochs 5'
```

Note: Earlier Mistral runs (jobs 6754962-6754964, submitted Apr 6 from
`submit_multi_model.sh` at lr=5e-06) CANCELLED or FAILED due to OOM.
The successful Apr 10 runs used lr=1e-05, submitted manually with
different resource specs (2 GPUs, 64G, `-4bit` suffix in job name).

### Mistral-7B high-FT — Job 6815742

```
JobID: 6815742 | JobName: ev-ms-hi-4bit | State: COMPLETED
Submit: 2026-04-09T19:19:17 | Start: 2026-04-10T07:10:21 | End: 2026-04-10T07:12:37

SubmitLine:
sbatch -A c01949 --nodes=1 --ntasks-per-node=1 --gpus-per-node=2 --mem=64G --time=24:00:00
  -p gpu --job-name=ev-ms-hi-4bit
  --wrap='... && python3 evaluate_finetuned.py
    --condition high_quality
    --adapter_path lora_adapters/mistral_high
    --output_dir finetuned_results/mistral_high
    --test_data finetune_data/test_held_out.jsonl
    --model_name mistralai/Mistral-7B-Instruct-v0.2
    --lr 1e-05 --epochs 5'
```

### Mistral-7B low-FT — Job 6815743

```
JobID: 6815743 | JobName: ev-ms-lo-4bit | State: COMPLETED
Submit: 2026-04-09T19:19:17 | Start: 2026-04-10T07:12:42 | End: 2026-04-10T07:14:31

SubmitLine:
sbatch -A c01949 --nodes=1 --ntasks-per-node=1 --gpus-per-node=2 --mem=64G --time=24:00:00
  -p gpu --job-name=ev-ms-lo-4bit
  --wrap='... && python3 evaluate_finetuned.py
    --condition low_quality
    --adapter_path lora_adapters/mistral_low
    --output_dir finetuned_results/mistral_low
    --test_data finetune_data/test_held_out.jsonl
    --model_name mistralai/Mistral-7B-Instruct-v0.2
    --lr 1e-05 --epochs 5'
```

### LLaMA-3-8B baseline — Job 6815744

```
JobID: 6815744 | JobName: ev-8b-bs-4bit | State: COMPLETED
Submit: 2026-04-09T19:19:17 | Start: 2026-04-10T07:14:36 | End: 2026-04-10T07:20:17

SubmitLine:
sbatch -A c01949 --nodes=1 --ntasks-per-node=1 --gpus-per-node=2 --mem=64G --time=24:00:00
  -p gpu --job-name=ev-8b-bs-4bit
  --wrap='... && python3 evaluate_finetuned.py
    --condition baseline
    --output_dir finetuned_results/llama8b_baseline
    --test_data finetune_data/test_held_out.jsonl
    --model_name meta-llama/Meta-Llama-3-8B-Instruct
    --lr 1e-05 --epochs 5'
```

### LLaMA-3-8B high-FT — Job 6815745

```
JobID: 6815745 | JobName: ev-8b-hi-4bit | State: COMPLETED
Submit: 2026-04-09T19:19:17 | Start: 2026-04-10T07:14:52 | End: 2026-04-10T07:17:48

SubmitLine:
sbatch -A c01949 --nodes=1 --ntasks-per-node=1 --gpus-per-node=2 --mem=64G --time=24:00:00
  -p gpu --job-name=ev-8b-hi-4bit
  --wrap='... && python3 evaluate_finetuned.py
    --condition high_quality
    --adapter_path lora_adapters/llama8b_high
    --output_dir finetuned_results/llama8b_high
    --test_data finetune_data/test_held_out.jsonl
    --model_name meta-llama/Meta-Llama-3-8B-Instruct
    --lr 1e-05 --epochs 5'
```

### LLaMA-3-8B low-FT — Job 6815746

```
JobID: 6815746 | JobName: ev-8b-lo-4bit | State: COMPLETED
Submit: 2026-04-09T19:19:17 | Start: 2026-04-10T07:14:52 | End: 2026-04-10T07:19:11

SubmitLine:
sbatch -A c01949 --nodes=1 --ntasks-per-node=1 --gpus-per-node=2 --mem=64G --time=24:00:00
  -p gpu --job-name=ev-8b-lo-4bit
  --wrap='... && python3 evaluate_finetuned.py
    --condition low_quality
    --adapter_path lora_adapters/llama8b_low
    --output_dir finetuned_results/llama8b_low
    --test_data finetune_data/test_held_out.jsonl
    --model_name meta-llama/Meta-Llama-3-8B-Instruct
    --lr 1e-05 --epochs 5'
```

---

## Failed/Cancelled/Superseded Runs (for completeness)

These are the earlier attempts that did NOT produce the final result files:

| Job ID | Name | Date | State | ExitCode | MaxRSS | LR | Notes |
|:------:|:-----|:-----|:------|:--------:|:------:|:--:|:------|
| 6748696 | eval-baseline | Apr 5 | FAILED | 1:0 | — | 3e-05 | LLaMA-3-3B, first attempt |
| 6750432 | eval-high | Apr 5-6 | FAILED | 1:0 | — | 3e-05 | LLaMA-3-3B |
| 6750433 | eval-low | Apr 6 | CANCELLED | 0:15 | — | 3e-05 | LLaMA-3-3B |
| 6754957 | ev-llama8b-bs | Apr 6 | TIMEOUT | 0:0 | — | 1e-05 | LLaMA-8B, hit 12h wall |
| 6754959 | ev-llama8b-lo | Apr 6-7 | TIMEOUT | 0:0 | — | 1e-05 | LLaMA-8B, hit 12h wall |
| 6754962 | ev-mistral-bs | Apr 6 | CANCELLED | 0:15 | 45GB | 5e-06 | Mistral, manually cancelled after 11.5h |
| 6754963 | ev-mistral-hi | Apr 6 | COMPLETED | 0:0 | 15GB | **5e-06** | Mistral, **completed at lr=5e-06** — results overwritten by job 6815742 |
| 6754964 | ev-mistral-lo | Apr 6 | COMPLETED | 0:0 | 15GB | **5e-06** | Mistral, **completed at lr=5e-06** — results overwritten by job 6815743 |
| 6790491 | ev-mistral-bs-bnb | Apr 9 | COMPLETED | 0:0 | 14GB | — | Mistral bnb retry, LR unknown |
| 6790492 | ev-mistral-hi-bnb | Apr 9 | COMPLETED | 0:0 | 10GB | — | Mistral bnb retry, LR unknown |
| 6790493 | ev-mistral-lo-bnb | Apr 9 | COMPLETED | 0:0 | 15GB | — | Mistral bnb retry, LR unknown |
| 6803535 | ev-ms-bs-4bit | Apr 9 | FAILED | 1:0 | <1GB | — | 4-bit attempt, Python error at startup |
| 6803536 | ev-ms-hi-4bit | Apr 9 | FAILED | 1:0 | <1GB | — | 4-bit attempt, Python error at startup |
| 6803537 | ev-ms-lo-4bit | Apr 9 | FAILED | 1:0 | <1GB | — | 4-bit attempt, Python error at startup |

### Mistral LR switch timeline

1. **Apr 6**: `submit_multi_model.sh` submitted baseline/high/low at lr=5e-06.
   High-FT (6754963) and low-FT (6754964) **completed successfully** at lr=5e-06.
   Baseline (6754962) was **manually cancelled** after 11.5h (not OOM — 45GB RSS within 64GB limit).
2. **Apr 9**: `bnb` retries completed; `4-bit` retries failed with Python errors.
3. **Apr 10**: Final runs (6815741-6815743) submitted at lr=1e-05, all completed.
   These overwrote the earlier lr=5e-06 results for high-FT and low-FT.

The reason for switching from lr=5e-06 to lr=1e-05 between the Apr 6 and Apr 10
runs is not recoverable from the SLURM records. The original lr=5e-06 results
for high-FT and low-FT existed briefly but were overwritten.
