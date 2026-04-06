import json, os

adapters = [
    'lora_adapters/mistral_high',
    'lora_adapters/mistral_low',
    'lora_adapters/llama8b_high',
    'lora_adapters/llama8b_low',
    'lora_adapters/phi3_high',
    'lora_adapters/phi3_low',
]

for adapter_path in adapters:
    config_path = os.path.join(adapter_path, 'adapter_config.json')
    if not os.path.exists(config_path):
        print(f'MISSING: {config_path}')
        continue
    with open(config_path) as f:
        config = json.load(f)
    tm = config.get('target_modules')
    print(f'{adapter_path}: target_modules type={type(tm).__name__}, value={tm}')
    if not isinstance(tm, list):
        config['target_modules'] = sorted(list(tm))
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f'  FIXED: converted to sorted list')
    else:
        print(f'  OK: already a list')
