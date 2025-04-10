# translation_models.py

"""
Translation utilities using Helsinki-NLP MarianMT models.
Models follow the pattern: Helsinki-NLP/opus-mt-{src}-{tgt}
E.g. en→es uses Helsinki-NLP/opus-mt-en-es :contentReference[oaicite:0]{index=0}
"""

from functools import lru_cache
import torch
from transformers import pipeline

# User‑facing names → ISO codes
LANGUAGE_CODES = {
    "English": "en",
    "Spanish": "es",
    "French":  "fr",
    "German":  "de",
    "Chinese": "zh",
    # add more mappings as needed
}

@lru_cache(maxsize=8)
def get_local_translator(src_lang: str, tgt_lang: str):
    """
    Load (and cache) a local MarianMT pipeline for src→tgt.
    """
    model_id = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("translation", model=model_id, device=device)

def translate(text: str, tgt_lang_name: str, src_lang_name: str = "English") -> str:
    """
    Translate `text` from src_lang_name to tgt_lang_name.
    If tgt == src, returns text unchanged.
    """
    if not text or tgt_lang_name == src_lang_name:
        return text

    src_code = LANGUAGE_CODES.get(src_lang_name, "en")
    tgt_code = LANGUAGE_CODES.get(tgt_lang_name, "en")

    translator = get_local_translator(src_code, tgt_code)
    # pipeline returns list of dicts: [{"translation_text": "..."}]
    out = translator(text, max_length=3000)
    return out[0]["translation_text"]
