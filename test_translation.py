# test_translation.py
from translation_models import translate, LANGUAGE_CODES, detect_language

# Sample text
sample_text = (
    "1. Detailed Description:\n"
    "- Image: Chest X-ray.\n"
    "- View: Likely a posteroanterior (PA) view.\n\n"
    "2. Key Findings:\n"
    "- Cardiomegaly.\n"
    "- Consolidation in lung fields.\n\n"
    "3. Potential Differential Diagnoses:\n"
    "   1. Pulmonary Edema\n"
    "   2. Pneumonia\n\n"
    "4. Reasoning for Top Diagnosis:\n"
    "The findings suggest severe pneumonia."
)

# Test auto-detection
print("Detected language code:", detect_language(sample_text))

# Test translation from English to Spanish
translated = translate(sample_text, "Spanish", "English")
print("Translation (Spanish):\n", translated)
