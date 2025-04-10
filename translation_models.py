# translation_models.py

"""
This module provides translation functionalities 
via Hugging Face transformers pipelines.
"""

from functools import lru_cache
import torch
from transformers import pipeline

# Map simple language names to ISO codes
LANGUAGE_CODES = {
    "English": "en",
    "Spanish": "es",
    "French":  "fr",
    "German":  "de",
    "Chinese": "zh",
    # Add or remove languages as needed
}

@lru_cache(maxsize=8)
def get_translation_pipeline(src_lang_code: str, tgt_lang_code: str):
    """
    Lazy-load a HF translation pipeline for src_lang_code -> tgt_lang_code.
    e.g. "en" -> "es" uses Helsinki-NLP/opus-mt-en-es
    If a relevant model does not exist, you may need a fallback or a custom model.
    """
    model_name = f"Helsinki-NLP/opus-mt-{src_lang_code}-{tgt_lang_code}"
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("translation", model=model_name, device=device)

def translate(text: str, tgt_lang_code: str, src_lang_code: str = "en") -> str:
    """
    Translate `text` from src_lang_code to tgt_lang_code.
    If tgt_lang_code == src_lang_code, returns text unchanged.
    """
    if not text.strip():
        return text  # no text to translate
    if tgt_lang_code == src_lang_code:
        return text

    # Attempt to get pipeline
    pipe = get_translation_pipeline(src_lang_code, tgt_lang_code)
    
    # The huggingface pipeline returns a list of dicts: [{"translation_text": "..."}]
    out = pipe(text, max_length=3000)  # Adjust max_length as needed
    return out[0]["translation_text"]
