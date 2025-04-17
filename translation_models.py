# -*- coding: utf-8 -*-
"""
translation_models.py

Provides language detection and translation using the deep‑translator library
(Google Translate backend).

This module:
 - Attempts to import deep‑translator at load time; if unavailable, all functions
   become no‑ops (returning None or original text) without further import errors.
 - Defines a fixed set of human‑readable language names mapped to ISO codes.
 - Applies sensible defaults and length checks to guard against API limits.
 - Wraps exceptional cases to keep the UI from crashing.
"""

import logging
from typing import Dict, Optional, Type, Union

# --- Logging Setup ---
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_LANGUAGE_CODE = "en"
AUTO_DETECT_INDICATOR = "Auto-Detect"
DETECT_TEXT_SNIPPET_LENGTH = 500   # chars to send for detection
TRANSLATE_WARN_LENGTH = 4800       # chars threshold for warning

# --- Flags and Placeholders for deep-translator ---
DEEP_TRANSLATOR_AVAILABLE: bool = False
GoogleTranslator = None  # type: ignore
# Exceptions from deep-translator
TranslationNotFound: Type[Exception] = Exception
NotValidPayload: Type[Exception] = Exception
NotValidLength: Type[Exception] = Exception
RequestError: Type[Exception] = Exception
TooManyRequests: Type[Exception] = Exception

# --- Attempt Import ---
try:
    from deep_translator import GoogleTranslator as _GoogleTranslator
    from deep_translator.exceptions import (
        TranslationNotFound as _TranslationNotFound,
        NotValidPayload as _NotValidPayload,
        NotValidLength as _NotValidLength,
        RequestError as _RequestError,
        TooManyRequests as _TooManyRequests,
    )
    # Assign upon success
    GoogleTranslator = _GoogleTranslator  # type: ignore
    TranslationNotFound = _TranslationNotFound
    NotValidPayload = _NotValidPayload
    NotValidLength = _NotValidLength
    RequestError = _RequestError
    TooManyRequests = _TooManyRequests
    DEEP_TRANSLATOR_AVAILABLE = True
    logger.info("deep-translator imported successfully; translation enabled.")
except ImportError as e:
    logger.warning(
        "deep-translator not available; translation features disabled. "
        "Install with `pip install deep-translator`."
    )
except Exception as e:
    logger.error("Unexpected error loading deep-translator.", exc_info=True)

# --- Supported Languages ---
LANGUAGE_CODES: Dict[str, str] = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Japanese": "ja",
    "Chinese (Simplified)": "zh-CN",
    "Russian": "ru",
    "Arabic": "ar",
    "Hindi": "hi",
    "Korean": "ko",
    "Dutch": "nl",
    "Swedish": "sv",
    "Turkish": "tr",
    "Polish": "pl",
    "Vietnamese": "vi",
    "Thai": "th",
    # Extend as needed
}

# --- Public API ---

def detect_language(text: str) -> Optional[str]:
    """
    Detects the language of `text`.
    Returns the ISO code (e.g. 'en'), or DEFAULT_LANGUAGE_CODE on failure,
    or None if deep-translator is not available.
    """
    if not DEEP_TRANSLATOR_AVAILABLE or GoogleTranslator is None:
        return None

    if not text or not text.strip():
        return DEFAULT_LANGUAGE_CODE

    snippet = text[:DETECT_TEXT_SNIPPET_LENGTH]
    try:
        translator = GoogleTranslator(source="auto", target=DEFAULT_LANGUAGE_CODE)
        result = translator.detect(snippet)
        code = result[0] if isinstance(result, list) and result else result
        if not isinstance(code, str) or not code:
            raise ValueError(f"Invalid detection result: {result}")
        return code.lower()
    except (NotValidPayload, RequestError, TooManyRequests) as e:
        logger.error("Language detection API error: %s", e)
        return DEFAULT_LANGUAGE_CODE
    except Exception as e:
        logger.error("Unexpected error in detect_language: %s", e, exc_info=True)
        return DEFAULT_LANGUAGE_CODE

def translate(
    text: str,
    target_language: str,
    source_language: str = AUTO_DETECT_INDICATOR
) -> Optional[str]:
    """
    Translates `text` from `source_language` to `target_language`.
    - `target_language` and `source_language` must be keys in LANGUAGE_CODES,
      or source_language may be AUTO_DETECT_INDICATOR.
    Returns translated text, original text if no-op, or None on error.
    """
    if not DEEP_TRANSLATOR_AVAILABLE or GoogleTranslator is None:
        return None

    if not text or not text.strip():
        return text

    # Resolve codes
    tgt_code = LANGUAGE_CODES.get(target_language)
    if not tgt_code:
        logger.error("Unknown target language: %s", target_language)
        return None

    src_code = "auto"
    if source_language != AUTO_DETECT_INDICATOR:
        src_code = LANGUAGE_CODES.get(source_language, "auto")
        if src_code == "auto" and source_language != "Auto-Detect":
            logger.warning("Unknown source language: %s, falling back to auto", source_language)

    # Skip if identical
    if src_code != "auto" and src_code == tgt_code:
        return text

    if len(text) > TRANSLATE_WARN_LENGTH:
        logger.warning("Translating large text (%d chars)", len(text))

    try:
        translator = GoogleTranslator(source=src_code, target=tgt_code)
        result = translator.translate(text)
        if result is None:
            logger.warning("Translation API returned None")
            return text
        if not isinstance(result, str):
            logger.error("Non-str result from translator: %r", result)
            return None
        return result
    except TranslationNotFound:
        logger.error("TranslationNotFound for %s→%s", src_code, tgt_code)
        return None
    except NotValidPayload as e:
        logger.error("Invalid payload for translation: %s", e)
        return None
    except NotValidLength as e:
        logger.error("Invalid length for translation: %s", e)
        return None
    except (RequestError, TooManyRequests) as e:
        logger.error("Translation request error: %s", e)
        return None
    except Exception as e:
        logger.error("Unexpected error in translate(): %s", e, exc_info=True)
        return None

# --- Self‑Test Runner ---
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    if not DEEP_TRANSLATOR_AVAILABLE:
        logger.warning("Skipping self‑tests: deep-translator unavailable")
        sys.exit(0)

    sample = "Hello, world!"
    logger.info("Detecting language of English text: %s → %s", sample, detect_language(sample))
    logger.info("Translating to Spanish: %s → %s", sample, translate(sample, "Spanish", "English"))
    logger.info("Auto‑detect & translate to German: %s → %s", sample, translate(sample, "German"))
    logger.info("Preserve same language: %s → %s", sample, translate(sample, "English", "English"))
    logger.info("Unknown target: expects None → %r", translate(sample, "Klingon", "English"))
