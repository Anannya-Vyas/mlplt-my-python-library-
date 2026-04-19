import re
import os
import sys
import logging
import warnings
from typing import Optional

# NUCLEAR SILENCE: Suppress all HF/Transformers/TQDM noise at the absolute top
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Catch the specific UserWarning from huggingface_hub about unauthenticated requests
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

# Target specific loggers
for logger_name in ["huggingface_hub", "transformers", "huggingface_hub.utils._http"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Global cache to persist the model across different components
_MODEL_CACHE = {}

def call_builtin_llm(prompt: str, system_prompt: str = "You are a professional data scientist.") -> Optional[str]:
    """
    Universally shared zero-config fallback engine.
    Uses Qwen2.5-0.5B-Instruct (local-first).
    """
    global _MODEL_CACHE
    try:
        import torch
        import transformers
        # ABSOLUTE SILENCE: Force HF to be quiet even if env vars fail
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
        
        # Additional environmental silencing
        os.environ["TQDM_DISABLE"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        transformers.logging.set_verbosity_error()
        
        # Stricter instructions for the tiny model
        if "Python code" in system_prompt:
             system_prompt = (
                 "You are a Python expert. Output ONLY valid Python code. "
                 "Do NOT use markdown (no ```), NO introductory text, NO comments. "
                 "Start directly with the code."
             )

        if "pipeline" not in _MODEL_CACHE:
            model_id = "Qwen/Qwen2.5-0.5B-Instruct"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float32 if device == "cpu" else torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(device)
            
            _MODEL_CACHE["pipeline"] = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device
            )

        pipe = _MODEL_CACHE["pipeline"]
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        p = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        outputs = pipe(
            p,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        
        result = outputs[0]["generated_text"][len(p):].strip()
        # Double-check: if it STILL wrapped it in markdown, strip it
        if result.startswith("```"):
            result = re.sub(r"^```python\s?", "", result)
            result = re.sub(r"```$", "", result).strip()
        
        return result
    except Exception as e:
        # Standardize error reporting for zero-config engine
        print(f"\n  [WARN] Zero-Config AI engine failed to initialize: {e}")
        return None
