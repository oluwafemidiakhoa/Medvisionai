# translation_models.py

import torch
from transformers import pipeline

# Language codes
LANGUAGE_CODES = {
    "English": "en",
    "Spanish": "es",
    "French":  "fr",
    "German":  "de",
    "Chinese": "zh",
}

def get_local_translation_pipeline(src_lang_code: str, tgt_lang_code: str):
    """
    Load a Hugging Face model locally instead of using the Inference API.
    """
    model_name = f"Helsinki-NLP/opus-mt-{src_lang_code}-{tgt_lang_code}"
    
    # This automatically downloads the model to your local cache_dir
    # the first time you run it.
    # If you have a huggingface.co token, you can optionally pass: use_auth_token=HF_TOKEN
    pipe = pipeline("translation", model=model_name, device=0 if torch.cuda.is_available() else -1)
    return pipe

def translate(text: str, tgt_lang_code: str, src_lang_code: str = "en") -> str:
    if not text.strip():
        return text  # no text to translate
    if tgt_lang_code == src_lang_code:
        return text

    translator = get_local_translation_pipeline(src_lang_code, tgt_lang_code)
    # The pipeline returns [{"translation_text": "..."}]
    out = translator(text, max_length=3000)
    return out[0]["translation_text"]
