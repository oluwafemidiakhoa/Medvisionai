# translation_models.py

from functools import lru_cache
import torch
from transformers import pipeline

# 10-Language Mapping:
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
    You can switch to a different model if you want better translations,
    e.g. a bigger or specialized model from the Hugging Face Hub.
    """
    model_id = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("translation", model=model_id, device=device)

def translate(text: str, tgt_lang_name: str, src_lang_name: str = "English") -> str:
    """
    1) If the target language is the same as the source or the text is empty, return text as is.
    2) Otherwise, build a carefully structured prompt with an example (few-shot) + instructions.
    """
    if not text or tgt_lang_name == src_lang_name:
        return text

    src_code = LANGUAGE_CODES.get(src_lang_name, "en")
    tgt_code = LANGUAGE_CODES.get(tgt_lang_name, "en")

    # Example snippet for few-shot prompting (optional).
    # If your pipeline doesn't handle these well, you can skip this approach.
    # But it can help the translator see how you want bullet points, numbering, etc. handled.

    few_shot_example = f"""
Example of Format Preservation:

English:
1. Detailed Description:
   - Image: Chest X-ray.
   - View: Likely PA.

Spanish:
1. Descripción Detallada:
   - Imagen: Radiografía de tórax.
   - Vista: Probablemente PA.

-------------------------
Now, please translate the following text:
{text}
-------------------------
Remember to preserve bullet points, numbering, line breaks, and formatting exactly as in the original. 
Do not add or remove any details except for accurately translating the content.
"""

    translator = get_local_translator(src_code, tgt_code)

    # Perform the translation with a large max_length, no sampling, and truncation to handle big texts safely.
    result = translator(
        few_shot_example,
        max_length=2048,       # Larger to accommodate longer radiology texts
        truncation=True,
        do_sample=False
    )
    return result[0]["translation_text"]
