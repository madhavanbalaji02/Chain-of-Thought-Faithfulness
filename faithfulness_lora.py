"""
Fine-tuning with faithfulness-preserving regularization.
Adds a penalty when model answers become independent of CoT steps.

L_total = L_quality + lambda * L_faithfulness
L_faithfulness = -mean_over_steps[ (P_full - P_masked)^2 ]
  Minimizing this maximizes the answer-probability change when a step
  is masked — i.e. forces the model to depend on each CoT step.

Usage:
  python faithfulness_lora.py \\
    --model_name meta-llama/Llama-3.2-3B-Instruct \\
    --train_data finetune_data/high_quality.jsonl \\
    --output_dir lora_adapters/faith_reg_lambda01 \\
    --lambda_faith 0.1

Lambda values to test:
  0.0  → standard quality FT (reproduces the paradox)
  0.1  → light regularization (expected: recovers faithfulness)
  1.0  → heavy regularization (may hurt quality)
"""
import os
os.environ.setdefault('HF_HOME', '/N/scratch/madbala/hf_cache')
os.environ.setdefault('TRANSFORMERS_CACHE', '/N/scratch/madbala/hf_cache')

# Stub bitsandbytes before PEFT import — prevents Triton/gcc -lcuda on HPC nodes.
# PEFT checks isinstance(layer, bnb.nn.Linear8bitLt/Linear4bit); stubs make
# those checks return False for normal bf16 layers. No quantization is used.
import sys, types, importlib.util as _ilu

class _BnbStub:
    pass

_bnb_nn = types.ModuleType('bitsandbytes.nn')
_bnb_nn.__spec__ = _ilu.ModuleSpec('bitsandbytes.nn', None)
for _n in ['Linear8bitLt', 'Linear4bit', 'Params4bit', 'Int8Params']:
    setattr(_bnb_nn, _n, _BnbStub)
_bnb = types.ModuleType('bitsandbytes')
_bnb.__spec__ = _ilu.ModuleSpec('bitsandbytes', None)
_bnb.nn = _bnb_nn
sys.modules.setdefault('bitsandbytes', _bnb)
sys.modules.setdefault('bitsandbytes.nn', _bnb_nn)

import torch
import json
import argparse
import re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset


# ── Dataset ──────────────────────────────────────────────────────────────────

class CoTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instances = []
        with open(data_path) as f:
            for line in f:
                if line.strip():
                    self.instances.append(json.loads(line))
        print(f'Loaded {len(self.instances)} training instances')

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        inst = self.instances[idx]
        prompt     = inst.get('prompt',     inst.get('question', ''))
        completion = inst.get('completion', inst.get('cot', ''))
        correct    = str(inst.get('correct', inst.get('answer', 'A')))

        text = prompt + ' ' + completion
        enc  = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        input_ids      = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        labels         = input_ids.clone()

        # CoT steps for faithfulness loss (up to 4)
        cot_steps = inst.get('cot_steps', [])
        if not cot_steps and completion:
            cot_steps = [s.strip() for s in
                         re.split(r'(?<=[.!?])\s+', completion) if s.strip()]

        return {
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'labels':         labels,
            # String fields — handled by custom collator, never passed to model
            'cot_steps': cot_steps[:4],
            'prompt':    prompt,
            'correct':   correct,
        }


# ── Custom collator — keeps string fields as lists, stacks tensors ────────────

def faith_collate_fn(batch):
    tensor_keys = ['input_ids', 'attention_mask', 'labels']
    string_keys = ['cot_steps', 'prompt', 'correct']
    collated = {k: torch.stack([b[k] for b in batch]) for k in tensor_keys}
    collated.update({k: [b[k] for b in batch] for k in string_keys})
    return collated


# ── Trainer subclass ──────────────────────────────────────────────────────────

class FaithfulnessTrainer(Trainer):
    """
    Overrides compute_loss to add faithfulness regularization.

    Faithfulness loss: for each training instance, forward the model
    with the full CoT and with each step masked, then penalize
    low dependence:
        L_faith = -mean_steps[ (P_full - P_masked)^2 ]
    Minimizing -MSE maximizes the gap → the model must attend to
    each step to predict the correct answer.
    """

    def __init__(self, lambda_faith: float = 0.1, tok=None, **kwargs):
        super().__init__(**kwargs)
        self.lambda_faith = lambda_faith
        self.tok = tok

    # ------------------------------------------------------------------
    def _answer_prob(self, model, text: str, answer_letter: str) -> torch.Tensor:
        """
        Forward pass (WITH gradients) and return the softmax probability
        of `answer_letter` at the last token position.
        """
        inputs = self.tok(
            text + ' The answer is:',
            return_tensors='pt',
            truncation=True,
            max_length=256,
        ).to(next(model.parameters()).device)

        out    = model(**inputs)
        logits = out.logits[0, -1, :]          # [vocab]
        probs  = torch.softmax(logits, dim=-1)

        tok_ids = self.tok.encode(' ' + answer_letter.upper(),
                                  add_special_tokens=False)
        if tok_ids:
            return probs[tok_ids[0]]
        return probs.mean()                     # fallback (should never happen)

    # ------------------------------------------------------------------
    def _faithfulness_loss(self, model, batch) -> torch.Tensor:
        """
        For up to 2 instances per batch (cost: ~5 forward passes each):
          1. Compute P_full  = P(correct | prompt + full_CoT)
          2. For each step s: P_masked = P(correct | prompt + CoT \ s)
          3. loss_i = -mean_s[ (P_full - P_masked)^2 ]
        Return mean over instances.
        """
        device     = next(model.parameters()).device
        faith_terms: list[torch.Tensor] = []

        # Limit to 2 instances per step to keep wall-clock manageable
        n = min(len(batch['prompt']), 2)

        for i in range(n):
            prompt  = batch['prompt'][i]
            steps   = batch['cot_steps'][i]
            correct = batch['correct'][i]

            if not steps or not prompt:
                continue

            full_cot    = ' '.join(steps)
            full_prompt = prompt + ' ' + full_cot

            try:
                p_full = self._answer_prob(model, full_prompt, correct)

                sq_diffs: list[torch.Tensor] = []
                for s_idx in range(len(steps)):
                    masked_cot    = ' '.join(s for j, s in enumerate(steps)
                                             if j != s_idx)
                    masked_prompt = prompt + ' ' + masked_cot
                    p_masked      = self._answer_prob(model, masked_prompt, correct)
                    sq_diffs.append((p_full - p_masked) ** 2)

                if sq_diffs:
                    # Negative: we MINIMIZE this, so the gap is MAXIMIZED
                    faith_terms.append(-torch.stack(sq_diffs).mean())

            except Exception as e:
                print(f'  [faith_loss] skipped instance {i}: {e}')

        if faith_terms:
            return torch.stack(faith_terms).mean()
        return torch.tensor(0.0, device=device, requires_grad=False)

    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Pop string fields before passing to model
        cot_steps = inputs.pop('cot_steps', None)
        prompts   = inputs.pop('prompt',    None)
        corrects  = inputs.pop('correct',   None)

        outputs      = model(**inputs)
        quality_loss = outputs.loss

        if self.lambda_faith > 0 and cot_steps is not None:
            aux_batch = {
                'cot_steps': cot_steps,
                'prompt':    prompts,
                'correct':   corrects,
            }
            faith_loss  = self._faithfulness_loss(model, aux_batch)
            total_loss  = quality_loss + self.lambda_faith * faith_loss
            # Log both components
            if self.state.global_step % 10 == 0:
                print(f'  step {self.state.global_step:5d} | '
                      f'L_quality={quality_loss.item():.4f}  '
                      f'L_faith={faith_loss.item():.4f}  '
                      f'L_total={total_loss.item():.4f}')
        else:
            total_loss = quality_loss

        return (total_loss, outputs) if return_outputs else total_loss


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',   required=True)
    parser.add_argument('--train_data',   required=True)
    parser.add_argument('--output_dir',   required=True)
    parser.add_argument('--lambda_faith', type=float, default=0.1)
    parser.add_argument('--epochs',       type=int,   default=3)
    parser.add_argument('--lr',           type=float, default=2e-4)
    args = parser.parse_args()

    hf_token = os.environ.get('HF_TOKEN') or None
    print(f'Training with lambda_faith={args.lambda_faith}')

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, token=hf_token,
        cache_dir=os.environ.get('HF_HOME'))
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        token=hf_token,
        cache_dir=os.environ.get('HF_HOME'),
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.05,
        bias='none',
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = CoTDataset(args.train_data, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy='epoch',
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        report_to='none',
        dataloader_num_workers=0,   # avoid fork issues with custom collator
    )

    trainer = FaithfulnessTrainer(
        lambda_faith=args.lambda_faith,
        tok=tokenizer,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=faith_collate_fn,
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f'Saved to {args.output_dir}')


if __name__ == '__main__':
    main()
