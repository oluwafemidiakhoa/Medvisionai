# -*- coding: utf-8 -*-
"""
translation_models.py

This module provides language detection and translation functionalities
using the 'deep-translator' library, specifically interfacing with the
Google Translate backend. It's designed to be used within the RadVision AI
application or similar projects requiring translation capabilities.
"""

import logging
from typing import Dict, Optional

# --- Dependency Check and Import ---
# Try importing the core dependency. If it fails, functionality will be disabled.
try:
    from deep_translator import GoogleTranslator
    from deep_translator.exceptions import TranslationNotFound, NotValidPayload, NotValidLength
    DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    DEEP_TRANSLATOR_AVAILABLE = False
    # Define dummy classes/functions if needed for type hinting or basic structure
    class GoogleTranslator: pass
    class TranslationNotFound(Exception): pass
    class NotValidPayload(Exception): pass
    class NotValidLength(Exception): pass

# --- Logging Setup ---
# Get the logger for this module. Configuration should be handled by the entry point (app.py).
logger = logging.getLogger(__name__)

# --- Language Codes ---
# Dictionary mapping user-friendly language names to Google Translate language codes.
# Add or remove languages as needed. Verify codes against Google Translate documentation.
# https://cloud.google.com/translate/docs/languages (Reference, though deep-translator might use slightly different aliases)
LANGUAGE_CODES: Dict[str, str] = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt", # General Portuguese (often defaults to European)
    # "Portuguese (Brazil)": "pt-BR", # Example if specific variant needed and supported
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
    # Add more languages here as required...
}

# --- Language Detection ---
def detect_language(text: str) -> str:
    """
    Detects the language of the input text using the Google Translate backend.

    Args:
        text: The text snippet for language detection (ideally > 3 words for accuracy).

    Returns:
        The detected language code (e.g., 'en', 'es') in lowercase,
        or defaults to 'en' on error, empty input, or if deep-translator is unavailable.
    """
    if not DEEP_TRANSLATOR_AVAILABLE:
        logger.warning("Language detection skipped: deep-translator library not available.")
        return "en" # Default if library is missing

    if not text or not text.strip():
        logger.warning("Detect language called with empty or whitespace-only text. Defaulting to 'en'.")
        return "en"

    try:
        # Limit detection length for efficiency and API limits. 500 chars is usually plenty.
        text_snippet = text[:500]
        # Instantiate translator for detection (target doesn't matter for detection)
        translator = GoogleTranslator(source='auto', target='en')
        detected_code = translator.detect(text_snippet)

        # deep-translator detect might return ['en'] or 'en'
        if isinstance(detected_code, list) and detected_code:
            lang_code = detected_code[0]
        elif isinstance(detected_code, str) and detected_code:
            lang_code = detected_code
        else:
             logger.warning(f"Detection returned unexpected result: {detected_code}. Defaulting to 'en'.")
             lang_code = "en" # Fallback if detection result is weird

        # Basic validation of the returned code format (e.g., 2-3 letters, maybe hyphen)
        if not (isinstance(lang_code, str) and (2 <= len(lang_code) <= 6) and lang_code.islower()):
             logger.warning(f"Detection returned potentially invalid code format: '{lang_code}'. Using it anyway.")
             # Keep the potentially odd code for now, translation might still work or fail informatively

        logger.info(f"Detected language code: '{lang_code}' for text snippet starting with: '{text_snippet[:50]}...'")
        return lang_code.lower() # Ensure lowercase

    except NotValidPayload:
        logger.error("Language detection failed: Invalid payload (possibly text format issue).", exc_info=True)
        return "en" # Fallback
    except Exception as e:
        # Catch other potential exceptions from the library or network issues
        logger.error(f"Language detection failed with unexpected error: {e}", exc_info=True)
        return "en" # Fallback to English on any error

# --- Translation Function ---
def translate(text: str, target_language: str, source_language: str = "Auto-Detect") -> str:
    """
    Translates text using the Google Translate backend via deep-translator.

    Args:
        text: The text string to translate.
        target_language: The user-friendly name (e.g., "Spanish") of the target language
                         as defined in LANGUAGE_CODES.
        source_language: The user-friendly name (e.g., "English") of the source language
                         as defined in LANGUAGE_CODES, or "Auto-Detect". Defaults to "Auto-Detect".

    Returns:
        The translated text string.
        If translation fails, returns a string indicating the error, prefixed with "[Translation Error: ...] ".
        If source and target languages are the same, returns the original text.
        If the deep-translator library is unavailable, returns an error message.
    """
    if not DEEP_TRANSLATOR_AVAILABLE:
        logger.error("Translation failed: deep-translator library not available.")
        return "[Translation Error: Library Unavailable] " + text

    if not text or not text.strip():
        logger.warning("Translate called with empty or whitespace-only text. Returning original.")
        return text # Return original text if input is empty

    # --- Get Language Codes ---
    target_code = LANGUAGE_CODES.get(target_language)
    if not target_code:
        logger.error(f"Target language name '{target_language}' not found in LANGUAGE_CODES.")
        return f"[Translation Error: Unsupported Target Language '{target_language}'] " + text

    source_code = 'auto' # Default to auto-detection
    # Map user-friendly source language name to code, unless it's 'Auto-Detect'
    if source_language != "Auto-Detect":
        resolved_source_code = LANGUAGE_CODES.get(source_language)
        if resolved_source_code:
            source_code = resolved_source_code
        else:
            # If source name not found, log warning and stick with 'auto'
            logger.warning(f"Source language name '{source_language}' not found in LANGUAGE_CODES. Falling back to 'auto' detection.")
            source_code = 'auto'

    # --- Check if Translation is Needed ---
    # If source was explicitly set and matches target, no translation needed.
    # If source is 'auto', we proceed as detection might still be useful or necessary.
    if source_code != 'auto' and source_code == target_code:
         logger.info(f"Source ('{source_language}') and target ('{target_language}') languages are the same ('{source_code}'). No translation needed.")
         return text

    logger.info(f"Attempting translation: source='{source_code}', target='{target_code}'")

    try:
        # Instantiate translator with resolved codes (use lowercase for library consistency)
        translator = GoogleTranslator(source=source_code.lower(), target=target_code.lower())

        # Perform translation
        # Note: Google Translate API has length limits (e.g., 5000 chars).
        # deep-translator might handle chunking, but be aware for very long texts.
        if len(text) > 4800: # Add a warning for potentially problematic lengths
            logger.warning(f"Input text length ({len(text)} chars) is large, translation might be slow or hit limits.")

        translated_text = translator.translate(text)

        # Validate the result
        if translated_text is None:
            logger.warning("Translation API returned None. This might indicate an issue.")
            return f"[Translation Error: Empty Result from API] " + text
        elif not isinstance(translated_text, str):
             logger.error(f"Translation API returned unexpected type: {type(translated_text)}")
             return f"[Translation Error: Invalid Result Type] " + text
        elif not translated_text.strip() and text.strip():
             logger.warning("Translation resulted in an empty string for non-empty input.")
             # Return the empty string as it might be valid in some edge cases, but log it.
             return translated_text
        else:
            logger.info("Translation successful.")
            return translated_text

    except TranslationNotFound:
        logger.error(f"Translation not found for the given text or language pair ('{source_code}' -> '{target_code}').")
        return f"[Translation Error: Not Found by API] " + text
    except NotValidPayload as e:
        logger.error(f"Translation failed: Invalid payload. Check input text format. Error: {e}", exc_info=True)
        return f"[Translation Error: Invalid Input Payload] " + text
    except NotValidLength as e:
         logger.error(f"Translation failed: Text length exceeds API limits. Error: {e}", exc_info=True)
         return f"[Translation Error: Text Too Long] " + text
    except Exception as e:
        # Catch other potential errors (network, library bugs, etc.)
        logger.error(f"Translation failed unexpectedly ('{source_code}' -> '{target_code}'): {e}", exc_info=True)
        return f"[Translation Error: Unexpected Failure ({type(e).__name__})] " + text

# --- Example Usage (for testing when run directly) ---
if __name__ == "__main__":
    # Configure basic logging ONLY when the script is run directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

    logger.info("--- Translation Module Direct Execution Test ---")

    if not DEEP_TRANSLATOR_AVAILABLE:
        logger.critical("Cannot run tests: 'deep-translator' library is not installed.")
    else:
        # Test Detection
        logger.info("\n--- Testing Detection ---")
        sample_text_en = "Hello, world! This is a test of the translation system."
        detected_en = detect_language(sample_text_en)
        logger.info(f"Detected ('{sample_text_en[:30]}...'): Expected ~'en', Got: '{detected_en}'")

        sample_text_es = "Hola Mundo. Esto es una prueba del sistema de traducción."
        detected_es = detect_language(sample_text_es)
        logger.info(f"Detected ('{sample_text_es[:30]}...'): Expected ~'es', Got: '{detected_es}'")

        detected_empty = detect_language("")
        logger.info(f"Detected ('')             : Expected 'en', Got: '{detected_empty}'")

        # Test Translation
        logger.info("\n--- Testing Translation ---")
        text_to_trans = "The quick brown fox jumps over the lazy dog."
        logger.info(f"Original Text: '{text_to_trans}'")

        # EN -> ES
        translated_es = translate(text_to_trans, target_language="Spanish", source_language="English")
        logger.info(f"EN -> ES: '{translated_es}'")

        # ES -> FR (using previous output)
        translated_fr = translate(translated_es, target_language="French", source_language="Spanish")
        logger.info(f"ES -> FR: '{translated_fr}'")

        # Auto Detect -> DE
        translated_de = translate(text_to_trans, target_language="German", source_language="Auto-Detect")
        logger.info(f"Auto -> DE: '{translated_de}'")

        # Same Language
        translated_en_same = translate(text_to_trans, target_language="English", source_language="English")
        logger.info(f"EN -> EN : '{translated_en_same}' (Should match original)")
        assert translated_en_same == text_to_trans

        # Unknown Target Name
        translated_unknown_tgt = translate(text_to_trans, target_language="Klingon", source_language="English")
        logger.info(f"EN -> Klingon: '{translated_unknown_tgt}' (Should start with [Translation Error...)")
        assert translated_unknown_tgt.startswith("[Translation Error:")

        # Unknown Source Name (should fallback to auto)
        translated_unknown_src = translate(text_to_trans, target_language="Japanese", source_language="Elvish")
        logger.info(f"Elvish -> JA: '{translated_unknown_src}' (Should attempt auto-detect)")

        # Empty text
        translated_empty = translate("", target_language="Spanish")
        logger.info(f"'' -> ES : '{translated_empty}' (Should be empty)")
        assert translated_empty == ""

    logger.info("\n--- Testing Complete ---")