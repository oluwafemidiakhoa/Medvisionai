# -*- coding: utf-8 -*-
"""
translation_models.py

Provides language detection and translation functionalities using the
'deep-translator' library (Google Translate backend). Designed for robustness
and integration into applications like RadVision AI.
"""

import logging
from typing import Dict, Optional, List, Union

# --- Constants ---
DEFAULT_LANGUAGE_CODE = "en"
AUTO_DETECT_INDICATOR = "Auto-Detect"
DETECT_TEXT_SNIPPET_LENGTH = 500  # Max chars for detection API call
TRANSLATE_WARN_LENGTH = 4800      # Warn if text exceeds this length

# --- Dependency Check ---
try:
    # Attempt to import necessary components
    from deep_translator import GoogleTranslator
    from deep_translator.exceptions import (
        TranslationNotFound, NotValidPayload, NotValidLength, RequestError, TooManyRequests, BadSourceLanguage, BadTargetLanguage
    )
    DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    # If import fails, set flag to False and define dummy exceptions for type hinting
    DEEP_TRANSLATOR_AVAILABLE = False
    class GoogleTranslator: pass # Dummy for type hinting
    class TranslationNotFound(Exception): pass
    class NotValidPayload(Exception): pass
    class NotValidLength(Exception): pass
    class RequestError(Exception): pass
    class TooManyRequests(Exception): pass
    class BadSourceLanguage(Exception): pass
    class BadTargetLanguage(Exception): pass

# --- Logging Setup ---
# Configure logging in the main application entry point (e.g., app.py)
logger = logging.getLogger(__name__)

# --- Language Configuration ---
# Maps user-friendly language names to Google Translate API language codes.
# Source of truth for supported languages. Verify codes:
# https://cloud.google.com/translate/docs/languages (or deep-translator docs)
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
    # Add more as needed
}

# --- Core Functions ---

def detect_language(text: str) -> Optional[str]:
    """
    Detects the language of the input text using the Google Translate backend.

    Args:
        text: The text snippet for language detection. More text generally yields
              higher accuracy.

    Returns:
        The detected language code (e.g., 'en', 'es') in lowercase if successful.
        Returns DEFAULT_LANGUAGE_CODE ('en') if the input text is empty or whitespace.
        Returns None if the deep-translator library is unavailable or if detection fails.
    """
    if not DEEP_TRANSLATOR_AVAILABLE:
        logger.warning("Language detection failed: deep-translator library not available.")
        return None

    if not text or not text.strip():
        logger.debug("Detect language called with empty text. Returning default.")
        # Returning default 'en' might be desired for empty strings, depending on use case.
        # Alternatively, could return None here too. Let's stick to 'en'.
        return DEFAULT_LANGUAGE_CODE

    try:
        # Use a snippet for efficiency and API limits
        text_snippet = text[:DETECT_TEXT_SNIPPET_LENGTH]
        # Target doesn't matter for detection with GoogleTranslator class here
        translator = GoogleTranslator(source='auto', target=DEFAULT_LANGUAGE_CODE)
        detected_code_result: Union[str, List[str]] = translator.detect(text_snippet)

        # Process the result which might be a string or a list
        lang_code: Optional[str] = None
        if isinstance(detected_code_result, list) and detected_code_result:
            lang_code = detected_code_result[0]
        elif isinstance(detected_code_result, str) and detected_code_result:
            lang_code = detected_code_result
        else:
             logger.warning(f"Detection returned unexpected result type or empty: {detected_code_result}. Cannot determine language.")
             return None # Indicate failure

        # Basic validation - ensure it's a non-empty string
        if not lang_code or not isinstance(lang_code, str):
             logger.warning(f"Detection resulted in invalid code: '{lang_code}'. Cannot determine language.")
             return None

        logger.info(f"Detected language code: '{lang_code}' for text snippet starting with: '{text_snippet[:50]}...'")
        # Ensure lowercase for consistency, though deep-translator usually handles it
        return lang_code.lower()

    except (NotValidPayload, RequestError) as e:
        logger.error(f"Language detection failed due to API/payload issue: {e}", exc_info=True)
        return None
    except Exception as e:
        # Catch unexpected errors from the library or network
        logger.error(f"Language detection failed with unexpected error: {e}", exc_info=True)
        return None

def translate(
    text: str,
    target_language: str,
    source_language: str = AUTO_DETECT_INDICATOR
) -> Optional[str]:
    """
    Translates text using the Google Translate backend via deep-translator.

    Args:
        text: The text string to translate.
        target_language: The user-friendly name (e.g., "Spanish") of the target
                         language as defined in LANGUAGE_CODES.
        source_language: The user-friendly name (e.g., "English") of the source
                         language, or "Auto-Detect" to let the API guess.
                         Defaults to "Auto-Detect".

    Returns:
        The translated text string if successful.
        The original text if the effective source and target languages are the same.
        None if translation fails (e.g., library unavailable, language not
        supported, API error, invalid input). Check logs for details on failure.
    """
    if not DEEP_TRANSLATOR_AVAILABLE:
        logger.error("Translation failed: deep-translator library not available.")
        return None

    if not text or not text.strip():
        logger.debug("Translate called with empty text. Returning original empty text.")
        return text # Return original (empty) text

    # --- Resolve Language Codes ---
    target_code = LANGUAGE_CODES.get(target_language)
    if not target_code:
        logger.error(f"Translation failed: Target language name '{target_language}' not found in LANGUAGE_CODES.")
        return None

    source_code = 'auto' # GoogleTranslator's code for auto-detection
    if source_language != AUTO_DETECT_INDICATOR:
        resolved_source_code = LANGUAGE_CODES.get(source_language)
        if resolved_source_code:
            source_code = resolved_source_code
        else:
            # If source name is provided but not found, log warning and stick with 'auto'
            logger.warning(f"Source language name '{source_language}' not found in LANGUAGE_CODES. Falling back to 'auto' detection.")
            source_code = 'auto' # Explicitly reset to auto

    # --- Check if Translation is Needed ---
    # If source was explicitly set (not 'auto') and matches target, return original.
    # If source is 'auto', we MUST proceed to let the API handle detection and translation.
    if source_code != 'auto' and source_code == target_code:
         logger.info(f"Source ('{source_language}') and target ('{target_language}') languages are the same ('{source_code}'). Skipping translation.")
         return text

    logger.info(f"Attempting translation: source='{source_code}', target='{target_code}' for text length {len(text)}")
    if len(text) > TRANSLATE_WARN_LENGTH:
        logger.warning(f"Input text length ({len(text)} chars) is large, translation might be slow or hit API limits.")

    try:
        # Instantiate translator. Ensure codes are lowercase if required by library.
        translator = GoogleTranslator(source=source_code.lower(), target=target_code.lower())

        # Perform translation
        translated_text: Optional[str] = translator.translate(text) # Type hint for clarity

        # --- Validate Result ---
        if translated_text is None:
            # Can happen if API returns null or empty for some reason
            logger.warning(f"Translation API returned None for source='{source_code}', target='{target_code}'.")
            return None # Indicate failure clearly
        elif not isinstance(translated_text, str):
             logger.error(f"Translation API returned unexpected type: {type(translated_text)}. Expected str.")
             return None # Indicate failure
        # Allow empty string result if input was non-empty, log as warning.
        elif not translated_text.strip() and text.strip():
             logger.warning(f"Translation resulted in an empty or whitespace-only string for non-empty input (source='{source_code}', target='{target_code}').")
             return translated_text # Return the empty string result
        else:
            logger.info(f"Translation successful (source='{source_code}', target='{target_code}'). Result length: {len(translated_text)}")
            return translated_text

    # --- Specific API/Library Error Handling ---
    except TranslationNotFound:
        logger.error(f"Translation not found by API for the given text or language pair ('{source_code}' -> '{target_code}').")
        return None
    except NotValidPayload as e:
        logger.error(f"Translation failed: Invalid payload. Check input text format. Error: {e}", exc_info=False) # exc_info=False optional here
        return None
    except NotValidLength as e:
         logger.error(f"Translation failed: Text length issue (likely exceeds API limits). Error: {e}", exc_info=False)
         return None
    except (BadSourceLanguage, BadTargetLanguage) as e:
         logger.error(f"Translation failed: Invalid source or target language code provided to API. Error: {e}", exc_info=False)
         return None
    except (RequestError, TooManyRequests) as e:
         logger.error(f"Translation failed: API request error (network, permissions, rate limits?). Error: {e}", exc_info=False)
         return None
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Translation failed unexpectedly ('{source_code}' -> '{target_code}'): {e}", exc_info=True)
        return None

# --- Example Usage (for testing when run directly) ---
if __name__ == "__main__":
    # Configure basic logging ONLY when the script is run directly for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'
    )

    logger.info("--- Translation Module Direct Execution Test ---")

    if not DEEP_TRANSLATOR_AVAILABLE:
        logger.critical("Cannot run tests: 'deep-translator' library is not installed.")
    else:
        # --- Test Detection ---
        logger.info("\n--- Testing Detection ---")
        sample_text_en = "Hello, world! This is a test of the translation system."
        detected_en = detect_language(sample_text_en)
        logger.info(f"Detected ('{sample_text_en[:30]}...'): Expected ~'en', Got: '{detected_en}'")

        sample_text_es = "Hola Mundo. Esto es una prueba del sistema de traducción."
        detected_es = detect_language(sample_text_es)
        logger.info(f"Detected ('{sample_text_es[:30]}...'): Expected ~'es', Got: '{detected_es}'")

        detected_empty = detect_language("")
        logger.info(f"Detected ('')             : Expected '{DEFAULT_LANGUAGE_CODE}', Got: '{detected_empty}'")

        detected_none = detect_language("? @#$ %^") # Test potentially problematic input
        logger.info(f"Detected ('? @#$ %^')       : Expected 'None' or lang code, Got: '{detected_none}'")


        # --- Test Translation ---
        logger.info("\n--- Testing Translation ---")
        text_to_trans = "The quick brown fox jumps over the lazy dog."
        logger.info(f"Original Text: '{text_to_trans}'")

        # EN -> ES (Successful Case)
        translated_es = translate(text_to_trans, target_language="Spanish", source_language="English")
        logger.info(f"EN -> ES: '{translated_es}'")
        assert isinstance(translated_es, str)

        # Auto Detect -> DE (Successful Case)
        translated_de = translate(text_to_trans, target_language="German", source_language=AUTO_DETECT_INDICATOR)
        logger.info(f"Auto -> DE: '{translated_de}'")
        assert isinstance(translated_de, str)

        # Same Language (Should return original)
        translated_en_same = translate(text_to_trans, target_language="English", source_language="English")
        logger.info(f"EN -> EN : '{translated_en_same}' (Should match original)")
        assert translated_en_same == text_to_trans

        # Unknown Target Name (Should fail gracefully -> None)
        translated_unknown_tgt = translate(text_to_trans, target_language="Klingon", source_language="English")
        logger.info(f"EN -> Klingon: '{translated_unknown_tgt}' (Should be None)")
        assert translated_unknown_tgt is None

        # Unknown Source Name (Should fallback to auto)
        translated_unknown_src = translate(text_to_trans, target_language="Japanese", source_language="Elvish")
        logger.info(f"Elvish -> JA: '{translated_unknown_src}' (Should attempt Auto-Detect, likely succeed)")
        assert isinstance(translated_unknown_src, str) # Assuming translation works

        # Empty text (Should return empty)
        translated_empty = translate("", target_language="Spanish")
        logger.info(f"'' -> ES : '{translated_empty}' (Should be empty string)")
        assert translated_empty == ""

    logger.info("\n--- Testing Complete ---")