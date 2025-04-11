# translation_models.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from functools import lru_cache

# For your language codes (used by your translation pipeline)
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

# OPTIONAL: If you're using MarianMT or other translation models, define them here
# e.g., from huggingface/opus-mt-en-es, or from your local pipeline
# This example is for illustration. Adjust to your actual setup.

@lru_cache(maxsize=8)
def get_translation_pipeline(src_code: str, tgt_code: str):
    # for demonstration, using "Helsinki-NLP/opus-mt-en-es"
    model_id = f"Helsinki-NLP/opus-mt-{src_code}-{tgt_code}"
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("translation", model=model_id, device=device)

def translate(text: str, tgt_lang_name: str, src_lang_name: str = "English") -> str:
    if not text or tgt_lang_name == src_lang_name:
        return text
    src_code = LANGUAGE_CODES.get(src_lang_name, "en")
    tgt_code = LANGUAGE_CODES.get(tgt_lang_name, "en")
    translation_pipeline = get_translation_pipeline(src_code, tgt_code)
    result = translation_pipeline(text, max_length=1024, truncation=True, do_sample=False)
    return result[0]["translation_text"]

# --- Language Detection with xlm-roberta-base ---
@lru_cache()
def get_language_detection_model():
    model_name = "papluca/xlm-roberta-base-language-detection"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

def detect_language(text: str) -> str:
    """
    Detect the primary language code of the given text using
    papluca/xlm-roberta-base-language-detection pipeline.
    Returns a 2-letter language code like 'en', 'es', 'fr', etc.
    If the code is not recognized in your LANGUAGE_CODES, handle as needed.
    """
    if not text.strip():
        return "unknown"

    detection_pipeline = get_language_detection_model()
    # The model might return something like [{'label': 'es', 'score': 0.98}]
    # or a specific label naming approach. We'll parse it accordingly.
    result = detection_pipeline(text[:512])  # limit length if needed
    # We'll assume it returns a dict with "label"
    # For example: [{'label': 'es', 'score': 0.999...}]

    if not result or "label" not in result[0]:
        return "unknown"

    detected_label = result[0]["label"]
    # papluca/xlm-roberta-base-language-detection uses 2 or 3 letter codes 
    # for many languages, e.g., 'en', 'es', 'fr', 'zh', etc. 
    # We'll just return that code. 
    return detected_label.lower()  # ensure e.g. 'en'

