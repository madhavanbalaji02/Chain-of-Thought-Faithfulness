"""
Extract FF2 (mlp.down_proj) hidden states at the last token
of the first CoT step for baseline, high-FT, and low-FT models.

Usage:
    python extract_ff2_activations.py \
        --model_name meta-llama/Llama-3.2-3B-Instruct \
        --results_file finetuned_results/baseline/results.jsonl \
        --test_data finetune_data/test_held_out.jsonl \
        --output_file ff2_activations/baseline.pkl \
        --layers 8,16,24
"""
import os
os.environ.setdefault('HF_HOME', '/N/scratch/madbala/hf_cache')
os.environ.setdefault('TRANSFORMERS_CACHE', '/N/scratch/madbala/hf_cache')


import torch, json, pickle, argparse, re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_ff2_hook(layer_idx, storage):
    def hook(module, input, output):
        storage[layer_idx] = output.detach().cpu().float()
    return hook


def extract_activations(model, tokenizer, text, layers_to_probe):
    storage = {}
    hooks = []
    for layer_idx in layers_to_probe:
        try:
            ff2 = model.model.layers[layer_idx].mlp.down_proj
        except AttributeError:
            print(f'Cannot find FF2 at layer {layer_idx}')
            continue
        hooks.append(ff2.register_forward_hook(get_ff2_hook(layer_idx, storage)))

    inputs = tokenizer(text, return_tensors='pt',
                       truncation=True, max_length=256).to(next(model.parameters()).device)
    with torch.no_grad():
        model(**inputs)
    for h in hooks:
        h.remove()

    return {idx: act[0, -1, :].numpy() for idx, act in storage.items()}


def first_sentence(cot):
    """Extract first sentence from CoT text."""
    cot = cot.strip()
    lines = [l.strip() for l in cot.split('\n') if l.strip()]
    if len(lines) >= 2:
        return lines[0]
    parts = re.split(r'(?<=[.!?])\s+', cot)
    return parts[0] if parts else cot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',   required=True)
    parser.add_argument('--adapter_path', default=None)
    parser.add_argument('--results_file', required=True,
                        help='finetuned_results/<cond>/results.jsonl')
    parser.add_argument('--test_data',    default='finetune_data/test_held_out.jsonl',
                        help='Held-out test set with CoT text')
    parser.add_argument('--output_file',  required=True)
    parser.add_argument('--n_instances',  type=int, default=50)
    parser.add_argument('--layers',       default='8,16,24')
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(',')]
    hf_token = os.environ.get('HF_TOKEN') or None

    # ── Build test_data map: instance_id → cot_text ───────────────
    test_map = {}
    with open(args.test_data) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            iid = rec.get('_instance_id', '')
            prompt = rec['prompt']
            completion = rec['completion']
            question = prompt.replace("Question: ", "").replace(
                "\nLet's think step by step.", "").strip()
            cot_lines = completion.rsplit('\nThe answer is ', 1)
            cot = cot_lines[0].strip()
            test_map[iid] = {'question': question, 'cot': cot}

    print(f'Loaded {len(test_map)} test instances')

    # ── Load results (faithfulness labels) ────────────────────────
    results = []
    with open(args.results_file) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    results = results[:args.n_instances]
    print(f'Loaded {len(results)} result instances')

    # ── Load model ────────────────────────────────────────────────
    print(f'Loading model: {args.model_name}')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, token=hf_token,
        cache_dir=os.environ.get('HF_HOME'))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        token=hf_token,
        cache_dir=os.environ.get('HF_HOME'),
    )

    if args.adapter_path and os.path.exists(args.adapter_path):
        # Manual LoRA merge — no PEFT, no bitsandbytes, no Triton.
        # Applies W += (lora_B @ lora_A) * (alpha / rank) for each adapted layer.
        from safetensors.torch import load_file as st_load
        cfg_path = os.path.join(args.adapter_path, 'adapter_config.json')
        with open(cfg_path) as _f:
            cfg = json.load(_f)
        rank  = cfg.get('r', 16)
        alpha = cfg.get('lora_alpha', 32)
        scale = alpha / rank

        adapter_file = os.path.join(args.adapter_path, 'adapter_model.safetensors')
        adapter_weights = st_load(adapter_file, device='cpu')

        merged = 0
        for name, param in model.named_parameters():
            base = name.replace('.weight', '')
            a_key = f'base_model.model.{base}.lora_A.weight'
            b_key = f'base_model.model.{base}.lora_B.weight'
            if a_key in adapter_weights and b_key in adapter_weights:
                lora_A = adapter_weights[a_key].to(param.dtype)
                lora_B = adapter_weights[b_key].to(param.dtype)
                param.data += (lora_B @ lora_A).to(param.device) * scale
                merged += 1
        print(f'Merged {merged} LoRA layers from: {args.adapter_path}')

    model.eval()
    print(f'Model loaded on: {next(model.parameters()).device}')
    print(f'Probing FF2 at layers: {layers}')

    # ── Extract activations ───────────────────────────────────────
    all_activations = []
    skipped = 0

    for idx, res in enumerate(results):
        iid = res.get('instance_id', '')
        test_rec = test_map.get(iid)
        if test_rec is None:
            skipped += 1
            continue

        cot = test_rec['cot']
        step0 = first_sentence(cot)

        # Faithfulness labels from epoch_results (epoch 5 = final)
        ep = res['epoch_results']
        # Binary: flip at any epoch 1-5
        binary_faithful = any(
            ep.get(str(e), {}).get('flip', False)
            for e in range(1, 6)
        )
        # Continuous: delta_p at epoch 5
        delta_p = ep.get('5', ep.get('4', ep.get('3', {}))).get('delta_p', 0.0)

        # Probe text: the full question + CoT context up to first step
        probe_text = (f"Question: {test_rec['question']}\n"
                      f"Let's think step by step.\n{step0}")

        try:
            acts = extract_activations(model, tokenizer, probe_text, layers)
            all_activations.append({
                'instance_id':      iid,
                'instance_idx':     idx,
                'binary_faithful':  binary_faithful,
                'delta_p':          float(delta_p),
                'activations':      acts,
            })
        except Exception as e:
            print(f'  SKIP instance {idx} ({iid[:30]}): {e}')
            skipped += 1

        if (idx + 1) % 10 == 0:
            print(f'  {idx+1}/{len(results)} done  '
                  f'(faithful={sum(a["binary_faithful"] for a in all_activations)})')

    print(f'\nDone: {len(all_activations)} activations extracted, {skipped} skipped')

    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    with open(args.output_file, 'wb') as f:
        pickle.dump(all_activations, f)
    print(f'Saved: {args.output_file}')


if __name__ == '__main__':
    main()
