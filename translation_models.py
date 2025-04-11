# translation_models.py

from functools import lru_cache
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Define a mapping for 10 languages
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
    """
    Load and cache a MarianMT translation pipeline for src_lang -> tgt_lang.
    For example, for English to Spanish, the model ID is "Helsinki-NLP/opus-mt-en-es".
    """
    model_id = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("translation", model=model_id, device=device)

def translate(text: str, tgt_lang_name: str, src_lang_name: str = "English") -> str:
    """
    Translate the given text from src_lang_name to tgt_lang_name using MarianMT.
    Returns the translated text.
    """
    if not text or tgt_lang_name == src_lang_name:
        return text
    src_code = LANGUAGE_CODES.get(src_lang_name, "en")
    tgt_code = LANGUAGE_CODES.get(tgt_lang_name, "en")
    translator = get_local_translator(src_code, tgt_code)
    result = translator(text, max_length=1024, truncation=True, do_sample=False)
    return result[0]["translation_text"]

# --- Language Detection using papluca/xlm-roberta-base-language-detection ---
@lru_cache()
def get_language_detector():
    model_name = "papluca/xlm-roberta-base-language-detection"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

def detect_language(text: str) -> str:
    """
    Detect the language of the input text using papluca/xlm-roberta-base-language-detection.
    Returns a two-letter language code (e.g., "en", "es", etc.). Returns "unknown" if detection fails.
    """
    if not text.strip():
        return "unknown"
    detector = get_language_detector()
    result = detector(text[:512])
    if not result or "label" not in result[0]:
        return "unknown"
    return result[0]["label"].lower()
