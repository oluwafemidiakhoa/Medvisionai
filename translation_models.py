# translation_models.py

from functools import lru_cache
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# Extend this dict for more languages if you want.
LANGUAGE_CODES = {
    "English": "en",
    "Spanish": "es",
    "French":  "fr",
    "German":  "de",
    "Chinese": "zh",
    "Japanese":"ja",
    "Korean":  "ko",
    "Arabic":  "ar",
    "Russian": "ru",
    "Portuguese":"pt"
}

@lru_cache(maxsize=8)
def get_translation_pipeline(src_code: str, tgt_code: str):
    """
    Returns a MarianMT translation pipeline for src_code→tgt_code.
    Example: "Helsinki-NLP/opus-mt-en-es"
    """
    model_id = f"Helsinki-NLP/opus-mt-{src_code}-{tgt_code}"
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("translation", model=model_id, device=device)

def translate(text: str, tgt_lang_name: str, src_lang_name: str = "English") -> str:
    """
    Translates `text` from src_lang_name to tgt_lang_name, preserving bullet points.
    You can wrap the text in a prompt (few-shot style) in app.py if you like.
    """
    if not text.strip() or tgt_lang_name == src_lang_name:
        return text  # Skip if empty or same language.

    # Convert user-facing language name to code
    src_code = LANGUAGE_CODES.get(src_lang_name, "en")
    tgt_code = LANGUAGE_CODES.get(tgt_lang_name, "en")

    translator = get_translation_pipeline(src_code, tgt_code)
    # Set a decent max_length to avoid truncation issues
    result = translator(
        text,
        max_length=1024,
        truncation=True,
        do_sample=False  # for deterministic output
    )
    return result[0]["translation_text"]

# OPTIONAL: If you'd like to auto-detect language using papluca/xlm-roberta-base-language-detection:

@lru_cache()
def get_language_detector():
    model_name = "papluca/xlm-roberta-base-language-detection"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

def detect_language(text: str) -> str:
    """
    Auto-detects language code from text using papluca/xlm-roberta-base-language-detection
    (e.g. 'en', 'es', 'fr', etc.). If detection is not recognized, returns 'unknown'.
    """
    if not text.strip():
        return "unknown"
    detector = get_language_detector()
    # Typically returns [{'label': 'en', 'score': 0.998}, ...]
    result = detector(text[:512])  # reduce length if text is very long
    if not result or "label" not in result[0]:
        return "unknown"
    return result[0]["label"].lower()  # e.g. 'en', 'es', 'fr'
