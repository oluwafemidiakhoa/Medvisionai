from functools import lru_cache
import torch
from transformers import pipeline

# translation_models.py
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
# ... along with the translate() function ...
