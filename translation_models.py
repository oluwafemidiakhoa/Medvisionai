# translation_models.py

from functools import lru_cache
import torch
from transformers import pipeline

# Ensure sentencepiece is installed (required for MarianMT tokenization)
try:
    import sentencepiece  # noqa: F401
except ImportError as e:
    raise ImportError(
        "The MarianMT tokenizer requires 'sentencepiece'. "
        "Please install it using: pip install sentencepiece"
    ) from e

# --- Language Mapping ---
# Define a mapping for 10 languages (user-friendly names to ISO 639-1 codes)
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

# --- Translator Loader ---
@lru_cache(maxsize=8)
def get_local_translator(src_lang: str, tgt_lang: str):
    """
    Load and cache a MarianMT translation pipeline for the specified source and target languages.
    This function returns a translation pipeline from Hugging Face's model hub.
    
    Args:
        src_lang (str): Source language code (e.g., "en").
        tgt_lang (str): Target language code (e.g., "es").
    
    Returns:
        A Hugging Face translation pipeline.
    """
    model_id = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("translation", model=model_id, device=device)

# --- Translation Function ---
def translate(text: str, tgt_lang_name: str, src_lang_name: str = "English") -> str:
    """
    Translate the given text from src_lang_name to tgt_lang_name using a MarianMT model.
    
    This function uses an enhanced prompt to instruct the model to preserve all formatting
    (e.g., bullet points, numbering, and spacing) exactly as in the original text.
    
    Args:
        text (str): The text to be translated.
        tgt_lang_name (str): The target language (e.g., "Spanish").
        src_lang_name (str): The source language (defaults to "English").
    
    Returns:
        str: The translated text.
    """
    # If the text is empty or the target language is the same as the source, return the original text.
    if not text.strip() or tgt_lang_name == src_lang_name:
        return text

    src_code = LANGUAGE_CODES.get(src_lang_name, "en")
    tgt_code = LANGUAGE_CODES.get(tgt_lang_name, "en")
    translator = get_local_translator(src_code, tgt_code)

    # Construct an enhanced prompt that instructs the model to translate accurately while preserving formatting.
    enhanced_prompt = (
        "Please translate the following text from English to " + tgt_lang_name + ". "
        "Preserve all formatting elements such as bullet points, numbering, and spacing exactly as in the original text. "
        "Do not alter or add any details; only perform an accurate translation.\n\n" +
        text
    )

    try:
        result = translator(
            enhanced_prompt,
            max_length=1024,   # Adjust as necessary for long texts
            truncation=True,
            do_sample=False    # Use deterministic output for consistency
        )
    except Exception as e:
        raise RuntimeError(f"Translation pipeline error: {str(e)}") from e

    # Extract and return the translated text
    return result[0]["translation_text"]
