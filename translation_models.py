# translation_models.py

import logging
from deep_translator import GoogleTranslator
from deep_translator.exceptions import TranslationNotFound

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Language Codes ---
# Dictionary mapping user-friendly names to language codes used by the translator
# Add or remove languages as needed. Ensure codes are valid for the chosen backend (Google Translate).
LANGUAGE_CODES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt", # Note: 'pt' often defaults to European Portuguese. Specify 'pt-BR' if needed and supported.
    "Japanese": "ja",
    "Chinese (Simplified)": "zh-CN", # Common code for Simplified Chinese
    "Russian": "ru",
    "Arabic": "ar",
    "Hindi": "hi",
    "Korean": "ko",
    "Dutch": "nl",
    "Swedish": "sv",
    "Turkish": "tr",
    "Polish": "pl",
    # Add more languages here...
    # Example: "Vietnamese": "vi",
    # Example: "Thai": "th",
}

# --- Language Detection ---
def detect_language(text: str) -> str:
    """
    Detects the language of the input text using Google Translate backend.

    Args:
        text: The text snippet to detect the language for (preferably > 3 words).

    Returns:
        The detected language code (e.g., 'en', 'es') or defaults to 'en' on error or empty input.
    """
    if not text or not text.strip():
        logger.warning("Detect language called with empty text. Defaulting to 'en'.")
        return "en"

    try:
        # Limit detection length if needed, though deep-translator might handle it.
        # Using the first 500 chars is usually sufficient.
        detected_code = GoogleTranslator().detect(text[:500])
        # The detect method might return a list like ['en'], handle this:
        if isinstance(detected_code, list) and detected_code:
            lang_code = detected_code[0]
        elif isinstance(detected_code, str):
            lang_code = detected_code
        else:
             logger.warning(f"Unexpected detection result type: {type(detected_code)}. Defaulting to 'en'.")
             lang_code = "en"

        logger.info(f"Detected language code: {lang_code}")
        return lang_code.lower() # Return lowercase code for consistency
    except Exception as e:
        logger.error(f"Language detection failed: {e}", exc_info=True)
        return "en" # Fallback to English on any error

# --- Translation Function ---
def translate(text: str, target_language: str, source_language: str = "auto") -> str:
    """
    Translates text using the Google Translate backend via deep-translator.

    Args:
        text: The text to translate.
        target_language: The user-friendly name of the target language (e.g., "Spanish").
        source_language: The user-friendly name of the source language (e.g., "English").
                         Defaults to "auto" for auto-detection by the translator.

    Returns:
        The translated text, or the original text if translation fails or languages are the same.
    """
    if not text or not text.strip():
        logger.warning("Translate called with empty text.")
        return text # Return original text if input is empty

    # --- Get language codes from names ---
    target_code = LANGUAGE_CODES.get(target_language)
    if not target_code:
        logger.error(f"Target language '{target_language}' not found in LANGUAGE_CODES.")
        return f"[Error: Unsupported target language '{target_language}']"

    source_code = 'auto' # Default to auto-detect
    if source_language != "Auto-Detect" and source_language != "auto":
        source_code = LANGUAGE_CODES.get(source_language)
        if not source_code:
            logger.warning(f"Source language '{source_language}' not found in LANGUAGE_CODES. Falling back to 'auto'.")
            source_code = 'auto'
        elif source_code == target_code:
             logger.info(f"Source ({source_language}) and target ({target_language}) languages are the same. No translation needed.")
             return text # No translation needed

    logger.info(f"Attempting translation from '{source_language}' ({source_code}) to '{target_language}' ({target_code}).")

    try:
        # Instantiate translator with source and target codes
        translator = GoogleTranslator(source=source_code.lower(), target=target_code.lower())
        translated_text = translator.translate(text)

        if translated_text is None:
            logger.warning("Translation returned None.")
            return f"[Translation Error: Empty Result] {text}"

        logger.info("Translation successful.")
        return translated_text

    except TranslationNotFound:
        logger.error(f"Translation not found for the given text or language pair ({source_code} -> {target_code}).")
        return f"[Translation Error: Not Found] {text}"
    except Exception as e:
        logger.error(f"Translation failed ({source_code} -> {target_code}): {e}", exc_info=True)
        # Return original text with an error marker for clarity
        return f"[Translation Error] {text}"

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    print("--- Testing Translation Models ---")

    # Test Detection
    sample_text_en = "Hello, world! This is a test."
    detected_en = detect_language(sample_text_en)
    print(f"Detected ('{sample_text_en}'): {detected_en}")

    sample_text_es = "Hola Mundo. Esto es una prueba."
    detected_es = detect_language(sample_text_es)
    print(f"Detected ('{sample_text_es}'): {detected_es}")

    # Test Translation (EN -> ES)
    translated_es = translate(sample_text_en, target_language="Spanish", source_language="English")
    print(f"EN -> ES: {translated_es}")

    # Test Translation (ES -> FR)
    translated_fr = translate(sample_text_es, target_language="French", source_language="Spanish")
    print(f"ES -> FR: {translated_fr}")

    # Test Translation (Auto Detect -> DE)
    translated_de = translate(sample_text_en, target_language="German", source_language="Auto-Detect")
    print(f"Auto -> DE: {translated_de}")

    # Test Same Language
    translated_en_same = translate(sample_text_en, target_language="English", source_language="English")
    print(f"EN -> EN: {translated_en_same == sample_text_en} (Should be True)")

    # Test Unknown Target
    translated_unknown = translate(sample_text_en, target_language="Klingon", source_language="English")
    print(f"EN -> Klingon: {translated_unknown}")

    print("--- Testing Complete ---")