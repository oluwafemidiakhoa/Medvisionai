# -*- coding: utf-8 -*-
"""
translation_models.py

Provides language detection and translation functionalities using the
'deep-translator' library (Google Translate backend).

Dependency Handling:
- This module attempts to import 'deep-translator' when loaded.
- If the import fails, a WARNING is logged once, and translation/detection
  functions will return None without further error messages about the missing library.
- Ensure 'deep-translator' is installed in the correct Python environment.

WORKAROUND APPLIED: Removed import/handling of BadSourceLanguage/BadTargetLanguage
due to persistent ImportError on the platform, even when the library version seems correct.
"""

import logging
from typing import Dict, Optional, Union, Type # Added Type for exception hinting

# --- Logging Setup ---
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_LANGUAGE_CODE = "en"
AUTO_DETECT_INDICATOR = "Auto-Detect"
DETECT_TEXT_SNIPPET_LENGTH = 500  # Maximum characters for language detection
TRANSLATE_WARN_LENGTH = 4800      # Warn if text exceeds this length

# --- Dependency Import and Check ---
DEEP_TRANSLATOR_AVAILABLE = False
GoogleTranslator = None
# Define base types first
TranslationNotFound: Type[Exception] = Exception
NotValidPayload: Type[Exception] = Exception
NotValidLength: Type[Exception] = Exception
RequestError: Type[Exception] = Exception
TooManyRequests: Type[Exception] = Exception
# WORKAROUND: Initialize BadSourceLanguage/BadTargetLanguage to base Exception
# as we won't import/catch them specifically due to the persistent ImportError.
BadSourceLanguage: Type[Exception] = Exception
BadTargetLanguage: Type[Exception] = Exception


try:
    # Attempt to import the necessary components
    from deep_translator import GoogleTranslator as _GoogleTranslator
    # WORKAROUND: Import only the exceptions known NOT to cause the ImportError
    from deep_translator.exceptions import (
        TranslationNotFound as _TranslationNotFound,
        NotValidPayload as _NotValidPayload,
        NotValidLength as _NotValidLength,
        RequestError as _RequestError,
        TooManyRequests as _TooManyRequests
        # EXCLUDED: BadSourceLanguage, BadTargetLanguage
    )

    # If import successful, assign to module-level variables and set flag
    GoogleTranslator = _GoogleTranslator # type: ignore
    TranslationNotFound = _TranslationNotFound
    NotValidPayload = _NotValidPayload
    NotValidLength = _NotValidLength
    RequestError = _RequestError
    TooManyRequests = _TooManyRequests
    # BadSourceLanguage/BadTargetLanguage remain as base Exception type
    DEEP_TRANSLATOR_AVAILABLE = True
    logger.info("Successfully imported 'deep-translator' (with workaround for language exceptions). Translation features enabled.")

# NOTE: The ImportError below should NO LONGER be triggered by BadSourceLanguage,
# but we keep it for general import failures of deep_translator itself.
except ImportError as import_error:
    # Log the specific import error once when the module is loaded
    logger.warning(
        f"Could not import 'deep-translator' library components. Translation features will be disabled. "
        f"Ensure it is installed in the correct environment. Error details: {import_error}"
    )
    # DEEP_TRANSLATOR_AVAILABLE remains False

except Exception as general_error:
     # Catch other potential issues during import setup
     logger.error(
         f"An unexpected error occurred during 'deep-translator' import/setup. "
         f"Translation features may be unstable or disabled. Error: {general_error}",
         exc_info=True # Log traceback for unexpected errors
     )
     # DEEP_TRANSLATOR_AVAILABLE remains False


# --- Language Configuration ---
# (Using user-friendly names as keys for easier UI integration)
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
        Returns DEFAULT_LANGUAGE_CODE ('en') if the input is empty or detection fails robustly.
        Returns None if the deep-translator library is unavailable.
    """
    if not DEEP_TRANSLATOR_AVAILABLE or GoogleTranslator is None:
        # Library not imported successfully, already logged at startup.
        return None

    if not text or not text.strip():
        logger.debug("Empty text provided for language detection; returning default lang code.")
        return DEFAULT_LANGUAGE_CODE

    try:
        # Use only a snippet for efficiency and API limits
        snippet = text[:DETECT_TEXT_SNIPPET_LENGTH]
        # Target doesn't matter much for detection, use default
        translator = GoogleTranslator(source='auto', target=DEFAULT_LANGUAGE_CODE)
        detected_result = translator.detect(snippet) # Can return list or string

        # Parse the potentially varied detection result
        lang_code: Optional[str] = None
        if isinstance(detected_result, list) and detected_result:
            lang_code = detected_result[0]
        elif isinstance(detected_result, str) and detected_result:
            lang_code = detected_result

        if not lang_code or not isinstance(lang_code, str):
            logger.warning(f"Detection returned invalid/empty code: '{detected_result}'. Using default.")
            return DEFAULT_LANGUAGE_CODE # Fallback if detection is weird

        final_code = lang_code.lower()
        logger.info(f"Detected language: '{final_code}' for text snippet: '{snippet[:50]}...'")
        return final_code

    except (NotValidPayload, RequestError, TooManyRequests) as e:
        logger.error(f"Language detection API error: {e}", exc_info=True)
        return DEFAULT_LANGUAGE_CODE # Fallback on API errors
    except Exception as e:
        logger.error(f"Unexpected error during language detection: {e}", exc_info=True)
        return DEFAULT_LANGUAGE_CODE # Fallback on other errors

def translate(
    text: str,
    target_language: str, # Expect user-friendly name (e.g., "Spanish")
    source_language: str = AUTO_DETECT_INDICATOR # Expect user-friendly name or "Auto-Detect"
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
        The original text if source equals target, or if text is empty.
        None if translation fails or the library is unavailable.
    """
    if not DEEP_TRANSLATOR_AVAILABLE or GoogleTranslator is None:
        # Library not imported successfully, already logged at startup.
        return None

    if not text or not text.strip():
        logger.debug("Empty text provided for translation; returning original.")
        return text

    # --- Resolve Language Codes ---
    target_code = LANGUAGE_CODES.get(target_language)
    if not target_code:
        logger.error(f"Target language name '{target_language}' not found in LANGUAGE_CODES.")
        return None # Cannot proceed without valid target

    source_code = 'auto' # Default to auto-detection
    if source_language != AUTO_DETECT_INDICATOR:
        resolved_source_code = LANGUAGE_CODES.get(source_language)
        if resolved_source_code:
            source_code = resolved_source_code
        else:
            logger.warning(
                f"Source language name '{source_language}' not found in LANGUAGE_CODES. "
                f"Falling back to 'auto'."
            )
            # Keep source_code as 'auto'

    # --- Skip if Source and Target Match ---
    # (Only skip if source was explicitly provided and matches target)
    if source_code != 'auto' and source_code == target_code:
        logger.info(f"Source language ('{source_language}') and target language ('{target_language}') "
                    f"resolve to the same code ('{source_code}'); skipping translation.")
        return text

    # --- Perform Translation ---
    logger.info(f"Attempting translation from '{source_code}' (resolved from '{source_language}') "
                f"to '{target_code}' (resolved from '{target_language}'). Input length: {len(text)}")

    if len(text) > TRANSLATE_WARN_LENGTH:
        logger.warning(
            f"Translation text length ({len(text)}) exceeds threshold ({TRANSLATE_WARN_LENGTH}). "
            "This may impact performance or encounter API limits."
        )

    try:
        # Instantiate translator with resolved codes (lowercase expected by lib)
        translator = GoogleTranslator(source=source_code.lower(), target=target_code.lower())
        translated_text = translator.translate(text)

        # --- Validate Result ---
        if translated_text is None:
            # This can happen, e.g., if translating empty strings after HTML stripping by the lib
            logger.warning("Translation API returned None. Input may have become empty after processing.")
            # Return original text if input was non-empty, otherwise empty string is fine
            return text if text.strip() else "" # Return original text if API gives None for non-empty input
        if not isinstance(translated_text, str):
            logger.error(f"Translation API returned a non-string result: {type(translated_text)}. Value: {translated_text!r}")
            return None # Indicate failure
        # It's possible valid translation results in an empty string for non-empty input
        # Log it, but return the result
        if not translated_text.strip() and text.strip():
            logger.warning("Translation resulted in an empty string for non-empty input.")

        logger.info(f"Translation successful. Output length: {len(translated_text)}")
        return translated_text

    # --- Handle Specific Translation Errors ---
    except TranslationNotFound:
        logger.error(f"Translation not found for the text between '{source_code}' and '{target_code}'.")
        return None
    except NotValidPayload as e:
        logger.error(f"Invalid payload sent to translation API: {e}", exc_info=True)
        return None
    except NotValidLength as e:
        logger.error(f"Text length issue during translation: {e}", exc_info=True)
        return None
    # WORKAROUND: Removed specific catch for BadSourceLanguage/BadTargetLanguage
    # except (BadSourceLanguage, BadTargetLanguage) as e:
    #     logger.error(f"Invalid source/target language code used for translation API: {e}", exc_info=True)
    #     return None
    except (RequestError, TooManyRequests) as e:
        logger.error(f"API request error during translation (network issue, quota exceeded, etc.): {e}", exc_info=True)
        return None
    except Exception as e:
        # Catch any other unexpected errors from the library or logic, including potentially
        # the underlying errors that BadSource/TargetLanguage would have represented.
        logger.error(f"Unexpected error during translation: {e}", exc_info=True)
        return None

# --- Test Code (for direct execution) ---
# (Self-test remains the same)
if __name__ == "__main__":
    import sys
    # Setup basic logging to console for testing
    logging.basicConfig(
        level=logging.DEBUG, # Show INFO and DEBUG messages for testing
        format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s',
        stream=sys.stdout
    )

    logger.info("--- Running Translation Module Self-Test ---")

    if not DEEP_TRANSLATOR_AVAILABLE:
        logger.warning("Self-test skipped: 'deep-translator' library is not available.")
    else:
        # Test detection
        sample_text_detect = "Bonjour tout le monde! Ceci est un test."
        logger.info(f"\nTesting detection for: '{sample_text_detect}'")
        detected_lang = detect_language(sample_text_detect)
        logger.info(f"-> Detected language code: {detected_lang}")

        # Test translation (French to English)
        sample_text_translate = "Le chat est assis sur le tapis."
        logger.info(f"\nTesting translation: '{sample_text_translate}' from French to English")
        translated_text = translate(sample_text_translate, target_language="English", source_language="French")
        if translated_text is not None:
            logger.info(f"-> Translation result: '{translated_text}'")
        else:
            logger.error("-> Translation failed.")

        # Test translation (Auto-detect Spanish to German)
        sample_text_auto = "Hola Mundo, cómo estás?"
        logger.info(f"\nTesting translation: '{sample_text_auto}' from Auto-Detect to German")
        translated_auto = translate(sample_text_auto, target_language="German", source_language=AUTO_DETECT_INDICATOR)
        if translated_auto is not None:
            logger.info(f"-> Translation result: '{translated_auto}'")
        else:
            logger.error("-> Translation failed.")

        # Test edge case: Empty string
        logger.info(f"\nTesting translation: Empty string")
        translated_empty = translate("", target_language="German", source_language="English")
        logger.info(f"-> Translation result: '{translated_empty}' (Expected: '')")

        # Test edge case: Source = Target
        logger.info(f"\nTesting translation: Source equals Target (English to English)")
        translated_same = translate("Hello", target_language="English", source_language="English")
        logger.info(f"-> Translation result: '{translated_same}' (Expected: 'Hello')")

        # Test edge case: Unknown target language name
        logger.info(f"\nTesting translation: Unknown target language name ('Klingon')")
        translated_bad_target = translate("Hello", target_language="Klingon", source_language="English")
        logger.info(f"-> Translation result: {translated_bad_target} (Expected: None)")

    logger.info("\n--- Self-Test Complete ---")