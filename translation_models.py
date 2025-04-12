# -*- coding: utf-8 -*-
"""
translation_models.py

Provides language detection and translation functionalities using the
'deep-translator' library (Google Translate backend). This module assumes
that the deep-translator dependency is managed externally (via requirements.txt)
and does not attempt dynamic installation.

If the library is not available, the functions will log an error and return None.
"""

import logging
from typing import Dict, Optional, Union

# --- Constants ---
DEFAULT_LANGUAGE_CODE = "en"
AUTO_DETECT_INDICATOR = "Auto-Detect"
DETECT_TEXT_SNIPPET_LENGTH = 500  # Maximum characters for language detection
TRANSLATE_WARN_LENGTH = 4800      # Warn if text exceeds this length

# --- Dependency Check ---
try:
    from deep_translator import GoogleTranslator
    from deep_translator.exceptions import (
        TranslationNotFound, NotValidPayload, NotValidLength, RequestError, TooManyRequests, BadSourceLanguage, BadTargetLanguage
    )
    DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    DEEP_TRANSLATOR_AVAILABLE = False
    # Define dummy classes for type hinting if needed
    class GoogleTranslator:
        pass
    class TranslationNotFound(Exception):
        pass
    class NotValidPayload(Exception):
        pass
    class NotValidLength(Exception):
        pass
    class RequestError(Exception):
        pass
    class TooManyRequests(Exception):
        pass
    class BadSourceLanguage(Exception):
        pass
    class BadTargetLanguage(Exception):
        pass

# --- Logging Setup ---
logger = logging.getLogger(__name__)

# --- Language Configuration ---
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
    # Add additional languages as needed
}

# --- Core Functions ---

def detect_language(text: str) -> Optional[str]:
    """
    Detects the language of the input text using the Google Translate backend.

    Args:
        text: The text snippet for language detection.

    Returns:
        The detected language code (e.g., 'en', 'es') in lowercase if successful.
        Returns DEFAULT_LANGUAGE_CODE ('en') if the input is empty.
        Returns None if the deep-translator library is unavailable or detection fails.
    """
    if not DEEP_TRANSLATOR_AVAILABLE:
        logger.error("Language detection failed: deep-translator library not available.")
        return None

    if not text or not text.strip():
        logger.debug("Empty text provided for language detection; returning default language code.")
        return DEFAULT_LANGUAGE_CODE

    try:
        snippet = text[:DETECT_TEXT_SNIPPET_LENGTH]
        translator = GoogleTranslator(source='auto', target=DEFAULT_LANGUAGE_CODE)
        detected = translator.detect(snippet)

        lang_code: Optional[str] = None
        if isinstance(detected, list) and detected:
            lang_code = detected[0]
        elif isinstance(detected, str) and detected:
            lang_code = detected
        else:
            logger.warning(f"Unexpected detection result: {detected}")
            return None

        if not lang_code or not isinstance(lang_code, str):
            logger.warning(f"Invalid detected language code: '{lang_code}'")
            return None

        logger.info(f"Detected language: '{lang_code.lower()}' for text snippet: '{snippet[:50]}...'")
        return lang_code.lower()

    except (NotValidPayload, RequestError) as e:
        logger.error(f"Language detection API error: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error during language detection: {e}", exc_info=True)
        return None

def translate(
    text: str,
    target_language: str,
    source_language: str = AUTO_DETECT_INDICATOR
) -> Optional[str]:
    """
    Translates text using the deep-translator Google Translate backend.

    Args:
        text: The text to translate.
        target_language: The user-friendly target language name (e.g., "Spanish").
        source_language: The user-friendly source language name or "Auto-Detect".
                         Defaults to "Auto-Detect".

    Returns:
        Translated text if successful.
        The original text if source equals target.
        None if translation fails.
    """
    if not DEEP_TRANSLATOR_AVAILABLE:
        logger.error("Translation failed: deep-translator library not available.")
        return None

    if not text or not text.strip():
        logger.debug("Empty text provided for translation; returning original text.")
        return text

    target_code = LANGUAGE_CODES.get(target_language)
    if not target_code:
        logger.error(f"Target language '{target_language}' not found in LANGUAGE_CODES.")
        return None

    if source_language != AUTO_DETECT_INDICATOR:
        source_code = LANGUAGE_CODES.get(source_language, 'auto')
    else:
        source_code = 'auto'

    # Skip translation if source is explicitly the same as target.
    if source_code != 'auto' and source_code == target_code:
        logger.info(f"Source and target languages are the same ('{source_code}'); returning original text.")
        return text

    logger.info(f"Translating text (length {len(text)}) from '{source_code}' to '{target_code}'.")
    if len(text) > TRANSLATE_WARN_LENGTH:
        logger.warning(f"Translation text length ({len(text)}) may be too long for optimal performance.")

    try:
        translator = GoogleTranslator(source=source_code.lower(), target=target_code.lower())
        translated_text = translator.translate(text)

        if translated_text is None:
            logger.warning("Translation API returned None.")
            return None
        if not isinstance(translated_text, str):
            logger.error(f"Translation API returned a non-string result: {type(translated_text)}")
            return None
        if not translated_text.strip() and text.strip():
            logger.warning("Translation resulted in an empty string for non-empty input.")
            return translated_text

        logger.info(f"Translation successful. Output length: {len(translated_text)}")
        return translated_text

    except TranslationNotFound:
        logger.error("Translation not found for the provided text and language pair.")
        return None
    except NotValidPayload as e:
        logger.error(f"Invalid payload for translation: {e}", exc_info=True)
        return None
    except NotValidLength as e:
        logger.error(f"Text length issue for translation: {e}", exc_info=True)
        return None
    except (BadSourceLanguage, BadTargetLanguage) as e:
        logger.error(f"Invalid language code provided: {e}", exc_info=True)
        return None
    except (RequestError, TooManyRequests) as e:
        logger.error(f"API request error during translation: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error during translation: {e}", exc_info=True)
        return None

# --- Test Code (for direct execution) ---
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s',
        stream=sys.stdout
    )
    logger.info("--- Running Translation Module Self-Test ---")

    sample_text = "Bonjour tout le monde!"
    detected_lang = detect_language(sample_text)
    logger.info(f"Detected language for '{sample_text}': {detected_lang}")

    translated_text = translate(sample_text, target_language="English", source_language="French")
    if translated_text:
        logger.info(f"Translation result: {translated_text}")
    else:
        logger.error("Translation returned no result.")

    logger.info("--- Self-Test Complete ---")
