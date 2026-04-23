"""
evaluate_finetuned.py
=====================
Evaluates a (possibly LoRA-adapted) model on the held-out test set
using the same NPO unlearning loop as unlearn.py.

Each instance runs in a FRESH SUBPROCESS so the OS reclaims all GPU memory
between instances. This is the only reliable way to prevent CUDA memory
fragmentation/OOM when repeatedly loading 14-16GB models.

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
"""

import os, sys, json, argparse, gc, subprocess, tempfile, types
import numpy as np

# Must be set before any CUDA operation.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# bitsandbytes (a PEFT dependency) imports Triton at load time.
# Triton tries to compile a CUDA stub with gcc -lcuda, which fails on BigRed200
# compute nodes because libcuda.so is not in the linker search path.
# Mock bitsandbytes before PEFT imports it so the Triton compilation never runs.
# This is safe because we never use quantization (bf16 only).
def _make_mock(name):
    m = types.ModuleType(name)
    m.__path__ = []
    return m
for _mod in ['bitsandbytes', 'bitsandbytes.nn', 'bitsandbytes.optim',
             'bitsandbytes.functional', 'bitsandbytes.cuda_setup']:
    if _mod not in sys.modules:
        sys.modules[_mod] = _make_mock(_mod)

import torch

LETTERS = ['A', 'B', 'C', 'D', 'E']

# ──────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--condition',    required=True)
    p.add_argument('--adapter_path', default=None)
    p.add_argument('--output_dir',   required=True)
    p.add_argument('--test_data',    default='finetune_data/test_held_out.jsonl')
    p.add_argument('--model_name',   default='meta-llama/Llama-3.2-3B-Instruct')
    p.add_argument('--lr',           type=float, default=3e-5)
    p.add_argument('--epochs',       type=int,   default=5)
    p.add_argument('--max_instances', type=int,  default=None)
    # Internal: worker mode processes exactly one instance then exits
    p.add_argument('--_worker',      default=None,
                   help='Path to JSON file with a single instance. Internal use only.')
    p.add_argument('--_out_path',    default=None,
                   help='Output JSONL path. Internal use only.')
    return p.parse_args()

# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────
def load_model(model_name, adapter_path=None, oracle=False):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Disable caching_allocator_warmup — pre-allocates ~half model size as a
    # contiguous block, which fails on fragmented GPU memory.
    import transformers.modeling_utils as _tmu
    if hasattr(_tmu, 'caching_allocator_warmup'):
        _tmu.caching_allocator_warmup = lambda *a, **kw: None

    hf_token = os.environ.get('HF_TOKEN', '') or None
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=False, token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # device_map='balanced': splits layers evenly across all GPUs (50/50 for 2×40GB).
    # Avoids 'auto' which fills GPU 0 first and overloads it.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='balanced',
        trust_remote_code=False,
        token=hf_token,
    )

    if adapter_path and os.path.exists(adapter_path):
        from peft import PeftModel
        _cfg_path = os.path.join(adapter_path, 'adapter_config.json')
        if os.path.exists(_cfg_path):
            with open(_cfg_path, 'r') as _f:
                _cfg = json.load(_f)
            if 'target_modules' in _cfg and not isinstance(_cfg['target_modules'], list):
                _cfg['target_modules'] = sorted(list(_cfg['target_modules']))
                with open(_cfg_path, 'w') as _f:
                    json.dump(_cfg, _f, indent=2)
        model = PeftModel.from_pretrained(model, adapter_path)
        for _name in model.peft_config:
            _lora_cfg = model.peft_config[_name]
            if hasattr(_lora_cfg, 'target_modules') and isinstance(_lora_cfg.target_modules, set):
                _lora_cfg.target_modules = sorted(list(_lora_cfg.target_modules))
        # Always merge — removes PeftModel dual-weight overhead.
        model = model.merge_and_unload()

    if oracle:
        model.requires_grad_(False)
    else:
        model.requires_grad_(True)
        # Gradient checkpointing: recompute activations during backward, ~4-8x
        # activation memory reduction for 7B/8B models.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    model.eval()
    return model, tokenizer

# ──────────────────────────────────────────────
# NPO-KL loss
# ──────────────────────────────────────────────
def compute_npo_kl_loss(model, oracle_log_prob_cpu, oracle_logits_cpu,
                         forget_input_ids, forget_labels,
                         retain_input_ids, beta=0.1):
    import torch.nn.functional as F
    device = next(model.parameters()).device

    forget_ids      = forget_input_ids.to(device)
    forget_lbl      = forget_labels.to(device)
    retain_ids      = retain_input_ids.to(device)
    oracle_log_prob = oracle_log_prob_cpu.to(device)
    oracle_logits   = oracle_logits_cpu.to(device)

    model_out      = model(forget_ids, labels=forget_lbl)
    model_log_prob = -model_out.loss
    npo_loss = -F.logsigmoid(beta * (model_log_prob - oracle_log_prob))
    del model_out, oracle_log_prob

    model_retain = model(retain_ids)
    kl_loss = F.kl_div(
        F.log_softmax(model_retain.logits, dim=-1),
        F.softmax(oracle_logits, dim=-1),
        reduction='batchmean',
    )
    del model_retain, oracle_logits

    return npo_loss + kl_loss

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def answer_probs(model, tokenizer, question, options, cot, device):
    probs = []
    for opt in options:
        letter = opt[0]
        prompt = (
            f"Question: {question}\n"
            f"Let's think step by step.\n{cot}\n"
            f"The answer is {letter}."
        )
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inputs, labels=inputs['input_ids'])
        log_prob = -out.loss.item() * inputs['input_ids'].shape[1]
        probs.append(float(np.exp(log_prob / max(inputs['input_ids'].shape[1], 1))))
    total = sum(probs)
    probs = [p / total for p in probs] if total > 1e-30 else [1.0/len(probs)]*len(probs)
    return probs, int(np.argmax(probs))

def get_ids(tokenizer, question, cot_text, device, max_len=128):
    text = f"Question: {question}\nLet's think step by step.\n{cot_text}"
    ids = tokenizer(text, return_tensors='pt', max_length=max_len, truncation=True)
    return ids['input_ids'].to(device)

import re as _re
def first_sentence(cot):
    cot = cot.strip()
    lines = [l.strip() for l in cot.split('\n') if l.strip()]
    if len(lines) >= 2:
        return lines[0]
    parts = _re.split(r'(?<=[.!?])\s+', cot)
    return parts[0] if parts else cot

# ──────────────────────────────────────────────
# Worker: process exactly ONE instance, write result, exit
# ──────────────────────────────────────────────
def run_worker(args):
    with open(args._worker) as f:
        rec = json.load(f)

    question    = rec['question']
    cot         = rec['initial_cot']
    options     = rec.get('options', [])
    correct_idx = LETTERS.index(rec['correct'])

    # STEP 1: Oracle — precompute outputs, cache to CPU, free GPU completely
    oracle, tokenizer = load_model(args.model_name, args.adapter_path, oracle=True)
    oracle_device = next(oracle.parameters()).device

    forget_ids_tmp = get_ids(tokenizer, question, first_sentence(cot), oracle_device)
    retain_ids_tmp = get_ids(tokenizer, question, cot, oracle_device)

    with torch.no_grad():
        out = oracle(forget_ids_tmp, labels=forget_ids_tmp)
        oracle_log_prob_cpu = (-out.loss).cpu()
        ret = oracle(retain_ids_tmp)
        oracle_logits_cpu = ret.logits.detach().cpu()

    del out, ret, forget_ids_tmp, retain_ids_tmp, oracle
    gc.collect()
    torch.cuda.empty_cache()

    # STEP 2: Trainable model — use cached oracle outputs
    model, _ = load_model(args.model_name, args.adapter_path, oracle=False)
    device = next(model.parameters()).device

    ff2_params = [p for n, p in model.named_parameters()
                  if 'mlp.down_proj' in n and p.requires_grad]
    if not ff2_params:
        ff2_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(ff2_params, lr=args.lr)

    forget_ids = get_ids(tokenizer, question, first_sentence(cot), device)
    retain_ids = get_ids(tokenizer, question, cot, device)

    epoch_results = {}
    baseline_probs = None
    baseline_pred  = None

    for epoch in range(args.epochs + 1):
        model.eval()
        p0_norm, pred0 = answer_probs(model, tokenizer, question, options, cot, device)
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
                model, oracle_log_prob_cpu, oracle_logits_cpu,
                forget_ids, forget_ids, retain_ids,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ff2_params, 1.0)
            optimizer.step()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    result = {
        'instance_id':   rec.get('instance_id', rec['question'][:60]),
        'dataset':       rec.get('dataset', 'unknown'),
        'correct':       rec['correct'],
        'correct_idx':   correct_idx,
        'baseline_probs': baseline_probs,
        'baseline_pred':  baseline_pred,
        'initial_prediction_correct': (baseline_pred == correct_idx),
        'epoch_results': epoch_results,
        'condition':     args.condition,
    }

    with open(args._out_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result) + '\n')

    ep5 = epoch_results.get(str(args.epochs), {})
    print(f"  OK  flip={ep5.get('flip')}  delta_p={ep5.get('delta_p',0):.4f}  id={result['instance_id'][:40]}")

# ──────────────────────────────────────────────
# Orchestrator: spawn one subprocess per instance
# ──────────────────────────────────────────────
def main():
    args = parse_args()

    # Worker mode: process one instance and exit
    if args._worker:
        run_worker(args)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Condition:  {args.condition}")
    print(f"Adapter:    {args.adapter_path or 'none (baseline)'}")
    print(f"Model:      {args.model_name}")

    # Load test instances
    test_instances = []
    with open(args.test_data, encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line.strip())
            prompt      = rec['prompt']
            completion  = rec['completion']
            question    = prompt.replace("Question: ", "").replace("\nLet's think step by step.", "").strip()
            cot_lines   = completion.rsplit('\nThe answer is ', 1)
            initial_cot = cot_lines[0].strip()
            correct     = cot_lines[1].strip().rstrip('.') if len(cot_lines) > 1 else 'A'
            test_instances.append({
                'instance_id': rec.get('_instance_id', question[:60]),
                'dataset':     rec.get('_dataset', 'unknown'),
                'question':    question,
                'initial_cot': initial_cot,
                'correct':     correct,
                'options':     [f"{l})" for l in LETTERS[:4]],
            })
    print(f"Loaded {len(test_instances)} instances.")

    out_path = os.path.join(args.output_dir, 'results.jsonl')
    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)['instance_id'])
                except: pass
        print(f"Resuming: {len(done_ids)} already done.")

    new_done = 0
    for i, inst in enumerate(test_instances):
        if inst['instance_id'] in done_ids:
            continue
        if args.max_instances is not None and new_done >= args.max_instances:
            print(f"[max_instances={args.max_instances} reached, stopping]")
            break

        print(f"[{i+1}/{len(test_instances)}] {inst['dataset']} | {inst['instance_id'][:40]}")

        # Write instance to temp file, spawn fresh subprocess, then delete temp file.
        # When subprocess exits, OS reclaims ALL its GPU memory — no fragmentation.
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
            json.dump(inst, tf)
            tmp_path = tf.name

        cmd = [sys.executable, __file__,
               '--condition',   args.condition,
               '--output_dir',  args.output_dir,
               '--test_data',   args.test_data,
               '--model_name',  args.model_name,
               '--lr',          str(args.lr),
               '--epochs',      str(args.epochs),
               '--_worker',     tmp_path,
               '--_out_path',   out_path,
               ]
        if args.adapter_path:
            cmd += ['--adapter_path', args.adapter_path]

        try:
            result = subprocess.run(cmd, timeout=1800, capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout.strip())
                new_done += 1
            else:
                print(f"  [FAILED] exit code {result.returncode}")
                if result.stdout: print("STDOUT:", result.stdout[-2000:])
                if result.stderr: print("STDERR_TAIL:", result.stderr[-3000:])
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] instance took >30min, skipping")
        finally:
            try: os.unlink(tmp_path)
            except: pass

    # Summary
    results = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try: results.append(json.loads(line))
                except: pass

    if results:
        flips  = [any(r['epoch_results'][str(e)]['flip']
                      for e in range(1, args.epochs+1)
                      if str(e) in r['epoch_results'])
                  for r in results]
        deltas = [r['epoch_results'].get(str(args.epochs), {}).get('delta_p', 0)
                  for r in results]
        print(f"\n--- {args.condition} ---")
        print(f"  N:                   {len(results)}")
        print(f"  Binary faithfulness: {sum(flips)/len(flips)*100:.1f}%")
        print(f"  Mean delta_p:        {sum(deltas)/len(deltas):.4f}")

if __name__ == '__main__':
    main()
