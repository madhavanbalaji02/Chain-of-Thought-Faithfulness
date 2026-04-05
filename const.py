datasets = ['arc-challenge', 'openbook', 'sports', 'sqa'] 
model_name_dict = {
    'Phi-3-mini-4k-instruct': 'Phi-3',
    'Meta-Llama-3-8B-Instruct': 'LLaMA-3',
    'Meta-Llama-3-3B-Instruct': 'LLaMA-3-3B',
    'Llama-3.2-3B-Instruct': 'LLaMA-3-3B',
    'Mistral-7B-Instruct-v0.2': 'Mistral-2',
}

model_name_to_path = {
    'Phi-3': 'microsoft/Phi-3-mini-4k-instruct',
    'LLaMA-3': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'LLaMA-3-3B':'meta-llama/Llama-3.2-3B-Instruct',
    'Mistral-2': 'mistralai/Mistral-7B-Instruct-v0.2',
}

LETTERS = ['A', 'B', 'C', 'D', 'E']

models = list(set(model_name_dict.values()))


# Paper's per-model best LRs (from the original authors' ablation sweep).
# Kept here for reference. NOT used by this reproduction — see dataset_model_best_lr below.
paper_best_lr = {
    'arc-challenge': {'Phi-3': 1e-04, 'LLaMA-3': 1e-05, 'LLaMA-3-3B': 3e-05, 'Mistral-2': 5e-06},
    'openbook':      {'Phi-3': 1e-04, 'LLaMA-3': 1e-05, 'LLaMA-3-3B': 3e-05, 'Mistral-2': 5e-06},
    'sports':        {'Phi-3': 5e-05, 'LLaMA-3': 5e-06, 'LLaMA-3-3B': 3e-05, 'Mistral-2': 3e-06},
    'sqa':           {'Phi-3': 5e-05, 'LLaMA-3': 1e-05, 'LLaMA-3-3B': 3e-05, 'Mistral-2': 5e-06},
}

# Reproduction used a single lr=3e-05 for all 16 combinations.
# This is what load_best_full_lrs() and the notebooks use to locate result files.
_REPRO_LR = 3e-05
dataset_model_best_lr = {
    dataset: {model: _REPRO_LR for model in ['Phi-3', 'LLaMA-3', 'LLaMA-3-3B', 'Mistral-2']}
    for dataset in ['arc-challenge', 'openbook', 'sports', 'sqa']
}

