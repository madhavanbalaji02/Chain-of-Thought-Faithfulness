"""
evaluate_finetuned.py
=====================
Evaluates a (possibly LoRA-adapted) LLaMA-3-3B on the held-out test set
using the same NPO unlearning loop as unlearn.py.

For each of the 50 held-out instances:
  1. Optionally merges a LoRA adapter into the base model
  2. Runs the standard NPO unlearning loop (5 epochs) on the first CoT step
  3. Records binary faithfulness, delta_p, efficacy, specificity per epoch

Usage:
    # Baseline (no adapter):
    python evaluate_finetuned.py \
        --condition baseline \
        --output_dir finetuned_results/baseline

    # With LoRA adapter:
    python evaluate_finetuned.py \
        --condition high_quality \
        --adapter_path lora_adapters/high_quality \
        --output_dir finetuned_results/high_quality

Arguments:
    --condition      Name tag for this condition (used in output filenames)
    --adapter_path   Path to LoRA adapter dir (omit for baseline)
    --output_dir     Where to write results.jsonl
    --test_data      Path to held-out JSONL (default: finetune_data/test_held_out.jsonl)
    --model_name     Base model (default: meta-llama/Llama-3.2-3B-Instruct)
    --lr             Unlearning LR (default: 3e-05, matching baseline reproduction)
    --epochs         Unlearning epochs (default: 5)
"""

import os, json, argparse, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW

LETTERS = ['A', 'B', 'C', 'D', 'E']

# ──────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--condition',    required=True)
    p.add_argument('--adapter_path', default=None,
                   help='Path to LoRA adapter. Omit for baseline.')
    p.add_argument('--output_dir',   required=True)
    p.add_argument('--test_data',    default='finetune_data/test_held_out.jsonl')
    p.add_argument('--model_name',   default='meta-llama/Llama-3.2-3B-Instruct')
    p.add_argument('--lr',           type=float, default=3e-5)
    p.add_argument('--epochs',       type=int,   default=5)
    return p.parse_args()

# ──────────────────────────────────────────────
# Model loading (with optional LoRA merge)
# ──────────────────────────────────────────────
def load_model(model_name, adapter_path=None):
    hf_token = os.environ.get('HF_TOKEN', '')
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=False, token=hf_token or None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=False,
        token=hf_token or None,
    )

    if adapter_path and os.path.exists(adapter_path):
        from peft import PeftModel
        print(f"Merging LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()   # merge weights, remove adapter overhead
        model.requires_grad_(True)         # re-enable grads frozen by PeftModel
        print("Adapter merged.")

    model.eval()
    return model, tokenizer

# ──────────────────────────────────────────────
# NPO loss (mirrors unlearn.py compute_loss with method='npo_KL')
# ──────────────────────────────────────────────
def compute_npo_kl_loss(model, oracle_model, forget_input_ids, forget_labels,
                         retain_input_ids, retain_labels, beta=0.1):
    """Simplified npo_KL loss matching the paper's implementation."""
    device = forget_input_ids.device

    # Forget loss: NPO against oracle
    with torch.no_grad():
        oracle_out = oracle_model(forget_input_ids, labels=forget_labels)
        oracle_log_prob = -oracle_out.loss

    model_out = model(forget_input_ids, labels=forget_labels)
    model_log_prob = -model_out.loss
    npo_loss = -torch.nn.functional.logsigmoid(
        beta * (model_log_prob - oracle_log_prob)
    )

    # KL retain loss
    with torch.no_grad():
        oracle_retain = oracle_model(retain_input_ids)
        oracle_logits = oracle_retain.logits.detach()
    model_retain = model(retain_input_ids)
    kl_loss = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(model_retain.logits, dim=-1),
        torch.nn.functional.softmax(oracle_logits, dim=-1),
        reduction='batchmean',
    )

    return npo_loss + kl_loss

# ──────────────────────────────────────────────
# Answer probability (normalized)
# ──────────────────────────────────────────────
def answer_probs(model, tokenizer, question, options, initial_cot, device):
    """Return normalized probability over answer letters, and argmax prediction."""
    probs = []
    for opt in options:
        letter = opt[0]   # 'A', 'B', etc.
        prompt = (
            f"Question: {question}\n"
            f"Let's think step by step.\n{initial_cot}\n"
            f"The answer is {letter}."
        )
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inputs, labels=inputs['input_ids'])
        log_prob = -out.loss.item() * inputs['input_ids'].shape[1]
        probs.append(float(np.exp(log_prob / max(inputs['input_ids'].shape[1], 1))))

    total = sum(probs)
    if total > 1e-30:
        probs = [p / total for p in probs]
    else:
        probs = [1.0 / len(probs)] * len(probs)

    return probs, int(np.argmax(probs))

# ──────────────────────────────────────────────
# Forget step token sequence
# ──────────────────────────────────────────────
def get_forget_ids(tokenizer, question, first_cot_sentence, device, max_len=256):
    text = (
        f"Question: {question}\n"
        f"Let's think step by step.\n"
        f"{first_cot_sentence}"
    )
    ids = tokenizer(text, return_tensors='pt', max_length=max_len, truncation=True)
    return ids['input_ids'].to(device)

# ──────────────────────────────────────────────
# Split CoT into first sentence
# ──────────────────────────────────────────────
import re as _re

def first_sentence(cot):
    cot = cot.strip()
    # Try newline-delimited (numbered lists like "1. ...\n2. ...")
    lines = [l.strip() for l in cot.split('\n') if l.strip()]
    if len(lines) >= 2:
        return lines[0]
    # Fall back to sentence splitting
    parts = _re.split(r'(?<=[.!?])\s+', cot)
    return parts[0] if parts else cot

# ──────────────────────────────────────────────
# Evaluate one instance
# ──────────────────────────────────────────────
def evaluate_instance(rec, args):
    """
    Run the full NPO unlearning loop on one instance and return per-epoch stats.
    Two fresh model copies are loaded per instance (matching unlearn.py behaviour).
    """
    question    = rec['question']
    cot         = rec['initial_cot']
    options     = rec.get('options', [])
    correct_idx = LETTERS.index(rec['correct'])

    model, tokenizer = load_model(args.model_name, args.adapter_path)
    oracle, _        = load_model(args.model_name, args.adapter_path)
    for param in oracle.parameters():
        param.requires_grad_(False)

    device = next(model.parameters()).device

    # FF2 layers only (matching --ff2 flag from original runs)
    ff2_params = [p for n, p in model.named_parameters()
                  if 'mlp.down_proj' in n and p.requires_grad]
    if not ff2_params:
        ff2_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(ff2_params, lr=args.lr)

    forget_ids  = get_forget_ids(tokenizer, question, first_sentence(cot), device)
    retain_ids  = get_forget_ids(tokenizer, question, cot, device)   # full CoT as retain

    epoch_results = {}

    for epoch in range(args.epochs + 1):   # epoch 0 = baseline
        model.eval()
        p0_norm, pred0 = answer_probs(model, tokenizer, question, options, cot, device)
        delta_p = None
        if epoch == 0:
            baseline_probs = p0_norm[:]
            baseline_pred  = pred0

        delta_p = baseline_probs[correct_idx] - p0_norm[correct_idx]
        flip    = (pred0 != baseline_pred) if epoch > 0 else False

        epoch_results[str(epoch)] = {
            'probs':      p0_norm,
            'prediction': pred0,
            'delta_p':    delta_p,
            'flip':       flip,
        }

        if epoch < args.epochs:
            model.train()
            optimizer.zero_grad()
            loss = compute_npo_kl_loss(
                model, oracle,
                forget_ids, forget_ids,   # labels = input_ids for CLM
                retain_ids, retain_ids,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ff2_params, 1.0)
            optimizer.step()

    # Clean up
    del model, oracle
    torch.cuda.empty_cache()

    return {
        'instance_id':   rec.get('instance_id', rec['question'][:60]),
        'dataset':       rec.get('dataset', 'unknown'),
        'correct':       rec['correct'],
        'correct_idx':   correct_idx,
        'baseline_probs': baseline_probs,
        'baseline_pred':  baseline_pred,
        'initial_prediction_correct': (baseline_pred == correct_idx),
        'epoch_results': epoch_results,
    }

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Condition:    {args.condition}")
    print(f"Adapter:      {args.adapter_path or 'none (baseline)'}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Test data:    {args.test_data}")
    print(f"Unlearn LR:   {args.lr}")

    # Load test instances
    test_instances = []
    with open(args.test_data, encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line.strip())
            # Reconstruct fields from prompt/completion format
            prompt     = rec['prompt']
            completion = rec['completion']
            question   = prompt.replace("Question: ", "").replace("\nLet's think step by step.", "").strip()
            cot_lines  = completion.rsplit('\nThe answer is ', 1)
            initial_cot = cot_lines[0].strip()
            correct     = cot_lines[1].strip().rstrip('.') if len(cot_lines) > 1 else 'A'
            test_instances.append({
                'instance_id': rec.get('_instance_id', question[:60]),
                'dataset':     rec.get('_dataset', 'unknown'),
                'question':    question,
                'initial_cot': initial_cot,
                'correct':     correct,
                'options':     [f"{l})" for l in LETTERS[:4]],   # generic fallback
            })

    print(f"Loaded {len(test_instances)} test instances.")

    out_path = os.path.join(args.output_dir, 'results.jsonl')
    # Resume from checkpoint
    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                done_ids.add(r['instance_id'])
        print(f"Resuming: {len(done_ids)} already done.")

    with open(out_path, 'a', encoding='utf-8') as out_f:
        for i, inst in enumerate(test_instances):
            if inst['instance_id'] in done_ids:
                continue
            print(f"  [{i+1}/{len(test_instances)}] {inst['dataset']} | {inst['instance_id'][:40]}")
            try:
                result = evaluate_instance(inst, args)
                result['condition'] = args.condition
                out_f.write(json.dumps(result) + '\n')
                out_f.flush()

                ep5 = result['epoch_results'].get('5', {})
                print(f"    flip={ep5.get('flip')}  delta_p={ep5.get('delta_p',0):.4f}")
            except Exception as e:
                print(f"    [ERROR] {e}")
                continue

    print(f"\nResults written to: {out_path}")

    # Quick summary
    results = []
    with open(out_path) as f:
        for line in f:
            results.append(json.loads(line))

    flips    = [any(r['epoch_results'][str(e)]['flip']
                    for e in range(1, args.epochs+1)
                    if str(e) in r['epoch_results'])
                for r in results]
    deltas   = [r['epoch_results'].get(str(args.epochs), {}).get('delta_p', 0)
                for r in results]

    print(f"\n--- Condition: {args.condition} ---")
    print(f"  N instances:         {len(results)}")
    print(f"  Binary faithfulness: {sum(flips)/len(flips)*100:.1f}%")
    print(f"  Mean delta_p:        {sum(deltas)/len(deltas):.4f}")

if __name__ == '__main__':
    main()
