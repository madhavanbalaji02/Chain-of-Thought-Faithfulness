import os
import glob
import random
import torch
import transformers
import numpy as np

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _patch_phi3_modeling():
    """
    Patch any cached modeling_phi3.py files that contain API calls incompatible
    with the installed transformers version.  Called before from_pretrained() so
    the fix is in place even if the cache was just populated.

    Primary fix: trust_remote_code=False (see load_model_and_tokenizer) causes
    transformers to use its built-in phi3 implementation instead of the cached
    file, making this function a no-op in normal operation.  It exists as a
    safety net in case the cached file is loaded for any reason.

    Bugs patched:
      1. rope_scaling["rope_type"] KeyError — newer transformers populates
         rope_scaling as {"rope_type": "default"}, not {"type": ...}.
      2. is_flash_attn_greater_or_equal_2_10() renamed to
         is_flash_attn_greater_or_equal('2.10.0') in newer flash-attn.
    """
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    pattern = os.path.join(hf_home, "**", "modeling_phi3.py")
    candidates = glob.glob(pattern, recursive=True)

    fixes = [
        # Bug 1: rope_scaling key — old code uses ["type"], new transformers uses "rope_type"
        (
            'scaling_type = self.config.rope_scaling["type"]',
            'scaling_type = self.config.rope_scaling.get("type") or self.config.rope_scaling.get("rope_type", "default")',
        ),
        # Bug 2: flash-attn helper renamed
        (
            "is_flash_attn_greater_or_equal_2_10()",
            "is_flash_attn_greater_or_equal('2.10.0')",
        ),
    ]

    for fpath in candidates:
        try:
            with open(fpath) as f:
                content = f.read()
            patched = False
            for old, new in fixes:
                if old in content:
                    content = content.replace(old, new)
                    patched = True
                    print(f"[phi3-patch] Applied fix to {fpath}: {old[:60]!r}")
            if patched:
                with open(fpath, "w") as f:
                    f.write(content)
        except OSError:
            pass  # file disappeared between glob and open — harmless


def load_model_and_tokenizer(model_name, half=True):
    # trust_remote_code=False: transformers 4.45+ has built-in phi3/llama3/mistral
    # support, so no remote code is needed. This also prevents loading the cached
    # modeling_phi3.py which is incompatible with transformers >=5.x.
    trust_remote_code = False

    if "phi" in model_name.lower():
        _patch_phi3_modeling()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    half_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=half_dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    return model, tokenizer
