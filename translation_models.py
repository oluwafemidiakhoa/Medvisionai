"""
Translation utilities using Helsinki-NLP MarianMT models.

Models follow the pattern:
  Helsinki-NLP/opus-mt-{src}-{tgt}

For example:
  en → es uses: Helsinki-NLP/opus-mt-en-es
"""

from functools import lru_cache
import torch
from transformers import pipeline

# Ensure sentencepiece is installed
try:
    import sentencepiece  # noqa: F401
except ImportError as e:
    raise ImportError(
        "The MarianMT tokenizer requires 'sentencepiece'. Please install it using:\n\n   pip install sentencepiece\n"
    ) from e

# User-facing language names → ISO 639-1 codes
LANGUAGE_CODES = {
    "English": "en",
    "Spanish": "es",
    "French":  "fr",
    "German":  "de",
    "Chinese": "zh",
    # Add more mappings if needed
}

@lru_cache(maxsize=8)
def get_local_translator(src_lang: str, tgt_lang: str):
    """
    Load and cache a MarianMT translation pipeline for src_lang → tgt_lang.
    
    Example:
      src_lang='en', tgt_lang='es'
      => model_id = 'Helsinki-NLP/opus-mt-en-es'
    """
    model_id = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("translation", model=model_id, device=device)

def translate(text: str, tgt_lang_name: str, src_lang_name: str = "English") -> str:
    """
    Translate `text` from src_lang_name → tgt_lang_name using MarianMT.
    
    If the target language is the same as the source, or if `text` is empty,
    returns the original text unchanged.
    
    Args:
        text (str): Text to be translated.
        tgt_lang_name (str): Target language (e.g., "Spanish").
        src_lang_name (str): Source language (e.g., "English").
    
    Returns:
        str: The translated text (or original text if no translation is needed).
    
    Note:
        MarianMT models typically support up to ~512 tokens. To avoid index errors,
        this call uses max_length=512 and truncation=True.
    """
    if not text or tgt_lang_name == src_lang_name:
        return text

    src_code = LANGUAGE_CODES.get(src_lang_name, "en")
    tgt_code = LANGUAGE_CODES.get(tgt_lang_name, "en")
    translator = get_local_translator(src_code, tgt_code)
    result = translator(text, max_length=512, truncation=True)
    return result[0]["translation_text"]
