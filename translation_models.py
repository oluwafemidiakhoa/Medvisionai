# translation_models.py

from functools import lru_cache
import torch
from transformers import pipeline

# Mapping for 10 languages:
LANGUAGE_CODES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Russian": "ru",
    "Portuguese": "pt"
}

@lru_cache(maxsize=8)
def get_local_translator(src_lang: str, tgt_lang: str):
    model_id = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("translation", model=model_id, device=device)

def translate(text: str, tgt_lang_name: str, src_lang_name: str = "English") -> str:
    if not text or tgt_lang_name == src_lang_name:
        return text
    src_code = LANGUAGE_CODES.get(src_lang_name, "en")
    tgt_code = LANGUAGE_CODES.get(tgt_lang_name, "en")
    translator = get_local_translator(src_code, tgt_code)
    result = translator(text, max_length=512, truncation=True)
    return result[0]["translation_text"]
