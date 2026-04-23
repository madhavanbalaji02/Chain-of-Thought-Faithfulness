"""
merge_adapters_nopeft.py
========================
Pre-merges LoRA adapters into base model weights WITHOUT using PEFT.
This avoids bitsandbytes → Triton → gcc -lcuda entirely.

Manual LoRA merge: W_new = W_orig + (lora_B @ lora_A) * (alpha / r)
"""
# HF_HOME must be set BEFORE importing transformers/huggingface_hub,
# which read this env var at import time to determine cache directory.
import os
os.environ['HF_HOME'] = '/N/scratch/madbala/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/N/scratch/madbala/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/N/scratch/madbala/hf_cache'

import json, gc
import torch
from safetensors.torch import load_file as st_load
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_TOKEN = os.environ.get('HF_TOKEN', '') or None
HF_CACHE = '/N/scratch/madbala/hf_cache'

ADAPTERS = [
    ('mistralai/Mistral-7B-Instruct-v0.2',    'lora_adapters/mistral_high',  'merged_models/mistral_high'),
    ('mistralai/Mistral-7B-Instruct-v0.2',    'lora_adapters/mistral_low',   'merged_models/mistral_low'),
    ('meta-llama/Meta-Llama-3-8B-Instruct',   'lora_adapters/llama8b_high',  'merged_models/llama8b_high'),
    ('meta-llama/Meta-Llama-3-8B-Instruct',   'lora_adapters/llama8b_low',   'merged_models/llama8b_low'),
]

def apply_lora_weights(model, adapter_path):
    """Manually apply LoRA weights to model in-place."""
    cfg_path = os.path.join(adapter_path, 'adapter_config.json')
    with open(cfg_path) as f:
        cfg = json.load(f)

    r = cfg['r']
    alpha = cfg['lora_alpha']
    scale = alpha / r
    print(f"  LoRA config: r={r}, alpha={alpha}, scale={scale:.3f}")

    # Load adapter safetensors
    st_path = os.path.join(adapter_path, 'adapter_model.safetensors')
    if not os.path.exists(st_path):
        st_path = os.path.join(adapter_path, 'adapter_model.bin')
        adapter_weights = torch.load(st_path, map_location='cpu')
    else:
        adapter_weights = st_load(st_path, device='cpu')

    print(f"  Loaded {len(adapter_weights)} adapter tensors")

    lora_a_keys = [k for k in adapter_weights if k.endswith('lora_A.weight')]
    print(f"  Found {len(lora_a_keys)} LoRA pairs to merge")

    applied = 0
    for a_key in lora_a_keys:
        b_key = a_key.replace('lora_A.weight', 'lora_B.weight')
        if b_key not in adapter_weights:
            print(f"  WARNING: no lora_B for {a_key}, skipping")
            continue

        param_name = a_key
        if param_name.startswith('base_model.model.'):
            param_name = param_name[len('base_model.model.'):]
        param_name = param_name.replace('.lora_A.weight', '.weight')

        try:
            param = model
            for part in param_name.split('.'):
                param = getattr(param, part)
        except AttributeError:
            print(f"  WARNING: param not found: {param_name}, skipping")
            continue

        lora_A = adapter_weights[a_key].to(param.data.dtype)
        lora_B = adapter_weights[b_key].to(param.data.dtype)
        delta = (lora_B @ lora_A) * scale
        param.data += delta.to(param.data.device)
        applied += 1

    print(f"  Applied {applied} LoRA weight deltas")
    return model


for base_model, adapter_path, out_path in ADAPTERS:
    if os.path.exists(out_path) and os.path.exists(os.path.join(out_path, 'config.json')):
        print(f"Already exists: {out_path}, skipping.")
        continue

    print(f"\n=== Merging {adapter_path} -> {out_path} ===")
    os.makedirs(out_path, exist_ok=True)

    print(f"  Loading base model {base_model} from cache {HF_CACHE}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map='cpu',
        token=HF_TOKEN,
        cache_dir=HF_CACHE,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=HF_TOKEN, cache_dir=HF_CACHE)

    print(f"  Applying LoRA weights from {adapter_path}...")
    model = apply_lora_weights(model, adapter_path)

    print(f"  Saving merged model to {out_path}...")
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)

    del model
    gc.collect()
    print(f"  Done: {out_path}")

print("\nAll adapters merged successfully.")
