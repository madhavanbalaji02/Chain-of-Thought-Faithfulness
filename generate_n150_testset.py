"""
generate_n150_testset.py
========================
Generate a fresh N=150 test set by sampling new dataset instances not used
in training or the existing N=50 test set, then generating CoTs with the
baseline LLaMA-3-3B model.

Outputs:
  finetune_data/test_held_out_n150.jsonl

Usage:
  python generate_n150_testset.py
"""
import os, sys, json, random, glob

os.environ['HF_HOME'] = '/N/scratch/madbala/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/N/scratch/madbala/hf_cache'

import types
from importlib.machinery import ModuleSpec as _ModuleSpec

class _BnbStub: pass
_bnb_nn = types.ModuleType('bitsandbytes.nn')
_bnb_nn.__spec__ = _ModuleSpec('bitsandbytes.nn', None)
for _n in ['Linear8bitLt', 'Linear4bit', 'Params4bit', 'Int8Params']:
    setattr(_bnb_nn, _n, _BnbStub)
_bnb = types.ModuleType('bitsandbytes')
_bnb.__spec__ = _ModuleSpec('bitsandbytes', None)
_bnb.nn = _bnb_nn
for _mod_name, _mod_obj in [
    ('bitsandbytes', _bnb), ('bitsandbytes.nn', _bnb_nn),
    ('bitsandbytes.optim', types.ModuleType('bitsandbytes.optim')),
    ('bitsandbytes.functional', types.ModuleType('bitsandbytes.functional')),
    ('bitsandbytes.cuda_setup', types.ModuleType('bitsandbytes.cuda_setup')),
]:
    sys.modules.setdefault(_mod_name, _mod_obj)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

WORKDIR = '/N/scratch/madbala/parametric-faithfulness_run'
MODEL_NAME = 'meta-llama/Llama-3.2-3B-Instruct'
HF_TOKEN = os.environ.get('HF_TOKEN', '')
RANDOM_SEED = 456   # different from training (123 used there)
TARGETS = {'arc-challenge': 38, 'openbook': 38, 'sports': 37, 'sqa': 37}
OUT_PATH = os.path.join(WORKDIR, 'finetune_data/test_held_out_n150.jsonl')

# Build excluded_ids: prefer pre-built excluded_ids.json, fallback to JSONL glob
finetune_dir = os.path.join(WORKDIR, 'finetune_data')
excluded_ids_path = os.path.join(finetune_dir, 'excluded_ids.json')
excluded_ids = set()
if os.path.exists(excluded_ids_path):
    with open(excluded_ids_path) as f:
        excluded_ids = set(str(x) for x in json.load(f))
    print(f'Loaded {len(excluded_ids)} excluded IDs from excluded_ids.json')
else:
    for jsonl_path in glob.glob(os.path.join(finetune_dir, '*.jsonl')):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    iid = rec.get('_instance_id') or rec.get('instance_id')
                    if iid:
                        excluded_ids.add(str(iid))
                except json.JSONDecodeError:
                    pass
    print(f'Excluding {len(excluded_ids)} known IDs from existing finetune splits')

# Load DATASETS dict from dataload.py (pre-instantiated handlers)
sys.path.insert(0, WORKDIR)
from dataload import DATASETS

random.seed(RANDOM_SEED)

# Sample fresh instances per dataset
fresh_instances = []
for ds_name, target in TARGETS.items():
    handler = DATASETS[ds_name]
    splits = handler.get_dataset_splits()

    # Concatenate all non-None splits
    all_items = []
    for split in splits:
        if split is None:
            continue
        for item in split:
            all_items.append(item)

    id_key = handler.id_key
    pool = [item for item in all_items if str(item[id_key]) not in excluded_ids]
    print(f'{ds_name}: {len(all_items)} total, {len(pool)} available after exclusion')
    n = min(target, len(pool))
    if n < target:
        print(f'  WARNING: only {n} available (need {target})')
    sampled = random.sample(pool, n)

    for item in sampled:
        fresh_instances.append({
            '_ds_name':  ds_name,
            '_id':       str(item[id_key]),
            '_question': item[handler.q_key],
            '_options':  handler.get_answer_choices(item),   # ["A): text", ...]
            '_correct':  handler.correct_answer_letter(item),
        })
    print(f'  Sampled {n}')

print(f'\nTotal fresh instances: {len(fresh_instances)}')

# Generate CoTs with baseline LLaMA-3-3B
print('\nLoading baseline model...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map='balanced',
    token=HF_TOKEN,
)
model.eval()
print('Model loaded.')


def generate_cot(question, answer_choices):
    # answer_choices are already formatted as "A): text", "B): text", etc.
    opts_str = '\n'.join(answer_choices)
    prompt = (
        f"Question: {question}\nOptions:\n{opts_str}\n"
        f"Let's think step by step.\n"
    )
    inputs = tokenizer(prompt, return_tensors='pt').to(next(model.parameters()).device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:],
                                  skip_special_tokens=True)
    return generated.strip()


os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
records = []
for i, inst in enumerate(fresh_instances):
    print(f'[{i+1}/{len(fresh_instances)}] {inst["_ds_name"]} | {inst["_id"][:50]}')
    try:
        cot = generate_cot(inst['_question'], inst['_options'])
    except Exception as e:
        print(f'  CoT generation failed: {e}, using empty CoT')
        cot = 'I need to think about this carefully.'
    prompt     = f"Question: {inst['_question']}\nLet's think step by step."
    completion = f"{cot}\nThe answer is {inst['_correct']}."
    records.append({
        'prompt':       prompt,
        'completion':   completion,
        '_instance_id': inst['_id'],
        '_dataset':     inst['_ds_name'],
        '_correct':     inst['_correct'],
    })

with open(OUT_PATH, 'w') as f:
    for r in records:
        f.write(json.dumps(r) + '\n')

print(f'\nSaved {len(records)} instances to {OUT_PATH}')
from collections import Counter
print('Dataset breakdown:', dict(Counter(r['_dataset'] for r in records)))
