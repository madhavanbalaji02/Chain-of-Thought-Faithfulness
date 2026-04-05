"""
finetune_lora.py
================
LoRA fine-tuning of meta-llama/Llama-3.2-3B-Instruct on CoT quality splits.

Usage (on BigRed200):
    python finetune_lora.py \
        --data_path finetune_data/high_quality.jsonl \
        --output_dir lora_adapters/high_quality \
        --model_name meta-llama/Llama-3.2-3B-Instruct

Arguments:
    --data_path     Path to .jsonl fine-tuning data (prompt/completion format)
    --output_dir    Directory to save LoRA adapter weights
    --model_name    HuggingFace model ID (default: meta-llama/Llama-3.2-3B-Instruct)
    --epochs        Number of training epochs (default: 3)
    --lr            Learning rate (default: 2e-4)
    --batch_size    Per-device batch size (default: 4)
    --grad_accum    Gradient accumulation steps (default: 4)
    --max_length    Maximum token length per example (default: 512)
"""

import os, json, argparse
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# ──────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path',   required=True)
    p.add_argument('--output_dir',  required=True)
    p.add_argument('--model_name',  default='meta-llama/Llama-3.2-3B-Instruct')
    p.add_argument('--epochs',      type=int,   default=3)
    p.add_argument('--lr',          type=float, default=2e-4)
    p.add_argument('--batch_size',  type=int,   default=4)
    p.add_argument('--grad_accum',  type=int,   default=4)
    p.add_argument('--max_length',  type=int,   default=512)
    return p.parse_args()

# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
def load_jsonl(path):
    records = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

# ──────────────────────────────────────────────
# Tokenize
# ──────────────────────────────────────────────
def make_tokenize_fn(tokenizer, max_length):
    def tokenize(example):
        full_text = example['prompt'] + ' ' + example['completion']
        prompt_ids = tokenizer(
            example['prompt'] + ' ',
            add_special_tokens=False,
        )['input_ids']

        full = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding=False,
            add_special_tokens=True,
        )
        input_ids      = full['input_ids']
        attention_mask = full['attention_mask']

        # Labels: mask the prompt tokens with -100 (only train on completion)
        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
        if len(labels) > max_length:
            labels = labels[:max_length]
        # Pad labels to match input_ids length
        labels = labels + [-100] * (len(input_ids) - len(labels))

        return {
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'labels':         labels,
        }
    return tokenize

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Model:      {args.model_name}")
    print(f"Data:       {args.data_path}")
    print(f"Output:     {args.output_dir}")
    print(f"Epochs:     {args.epochs}  LR: {args.lr}  Batch: {args.batch_size}  GradAccum: {args.grad_accum}")

    # ── Load tokenizer ──
    hf_token = os.environ.get('HF_TOKEN', '')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=False,
        token=hf_token or None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load model ──
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=False,
        token=hf_token or None,
    )
    model.config.use_cache = False   # required for gradient checkpointing

    # ── LoRA config ──
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias='none',
        target_modules=[
            'q_proj', 'v_proj', 'k_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ──
    raw = load_jsonl(args.data_path)
    print(f"Loaded {len(raw)} training examples.")
    dataset = Dataset.from_list(raw)
    tokenize_fn = make_tokenize_fn(tokenizer, args.max_length)
    tokenized = dataset.map(
        tokenize_fn,
        remove_columns=dataset.column_names,
        desc='Tokenizing',
    )

    # ── Training arguments ──
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy='epoch',
        save_total_limit=1,
        report_to='none',
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # ── Trainer ──
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("\nStarting training...")
    train_result = trainer.train()

    # Log per-epoch loss from training history
    print("\n--- Training loss per epoch ---")
    log_history = trainer.state.log_history
    epoch_losses = {}
    for entry in log_history:
        if 'loss' in entry:
            ep = int(entry.get('epoch', 0))
            epoch_losses[ep] = entry['loss']
    for ep, loss in sorted(epoch_losses.items()):
        print(f"  Epoch {ep}: loss={loss:.4f}")

    # ── Save adapter ──
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nLoRA adapter saved to: {args.output_dir}")

    # Save training summary
    summary = {
        'model_name':  args.model_name,
        'data_path':   args.data_path,
        'epochs':      args.epochs,
        'lr':          args.lr,
        'n_examples':  len(raw),
        'train_loss':  train_result.training_loss,
    }
    with open(os.path.join(args.output_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("Training summary saved.")

if __name__ == '__main__':
    main()
