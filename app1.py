# -*- coding: utf-8 -*-
"""
app.py - RadVision AI Advanced (Gemini Powered)

Main Streamlit application integrating Google Gemini for AI-assisted analysis.
Handles image uploading (DICOM, JPG, PNG), display, ROI selection,
Gemini-based analysis (initial, Q&A, condition focus, confidence assessment),
translation, and report generation. Includes monkey-patch for older Streamlit versions.

FIXED: Pass PIL Image directly to st_canvas background_image argument.
"""

import streamlit as st

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="RadVision AI Advanced (Gemini)",
    layout="wide",
    page_icon="⚕️",
    initial_sidebar_state="expanded"
)

# --- Core Python Libraries ---
import io
import os
import uuid
import logging
import base64
import hashlib
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple, List, Union
import copy
import random
import re

# --- Google Generative AI ---
try:
    import google.generativeai as genai
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    st.error("CRITICAL ERROR: Google Generative AI SDK not installed. Run `pip install google-generativeai`")
    print("CRITICAL ERROR: google-generativeai not found. App functionality severely impaired.")
    GOOGLE_GENAI_AVAILABLE = False
    st.stop() # Stop execution if core AI library is missing

# --- Ensure deep-translator is installed at runtime if not present ---
try:
    from deep_translator import GoogleTranslator
except ImportError:
    try:
        print("Attempting to install deep-translator...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "deep-translator"])
        from deep_translator import GoogleTranslator
        print("deep-translator installed successfully.")
    except Exception as e:
        print(f"CRITICAL: Could not install deep-translator: {e}")
        # Will gracefully disable translation later

# --- Logging Setup ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.info("--- RadVision AI (Gemini Powered) Application Start ---")
logger.info(f"Streamlit Version: {st.__version__}")
if GOOGLE_GENAI_AVAILABLE:
     try:
        # Use __version__ attribute if available, otherwise handle AttributeError
        logger.info(f"Google Generative AI SDK Version: {getattr(genai, '__version__', 'Unknown')}")
     except NameError:
        logger.warning("Could not retrieve google.generativeai version.")

logger.info(f"Logging Level: {LOG_LEVEL}")

# --- Streamlit Drawable Canvas ---
try:
    from streamlit_drawable_canvas import st_canvas
    import streamlit_drawable_canvas as st_canvas_module
    CANVAS_VERSION = getattr(st_canvas_module, '__version__', 'Unknown')
    logger.info(f"Streamlit Drawable Canvas Version: {CANVAS_VERSION}")
    DRAWABLE_CANVAS_AVAILABLE = True
except ImportError:
    st.error("CRITICAL ERROR: streamlit-drawable-canvas is not installed. Run pip install streamlit-drawable-canvas.")
    logger.critical("streamlit-drawable-canvas not found. App functionality impaired.")
    DRAWABLE_CANVAS_AVAILABLE = False
    st_canvas = None # Define st_canvas as None if import fails

# --- Pillow (PIL) ---
try:
    from PIL import Image, ImageDraw, UnidentifiedImageError
    import PIL
    PIL_VERSION = getattr(PIL, '__version__', 'Unknown')
    logger.info(f"Pillow (PIL) Version: {PIL_VERSION}")
    PIL_AVAILABLE = True
except ImportError:
    st.error("CRITICAL ERROR: Pillow (PIL) is not installed. Run pip install Pillow.")
    logger.critical("Pillow (PIL) not found. App functionality severely impaired.")
    PIL_AVAILABLE = False
    Image = None
    ImageDraw = None
    UnidentifiedImageError = None
    st.stop()

# --- pydicom & DICOM libraries ---
try:
    import pydicom
    import pydicom.errors
    PYDICOM_VERSION = getattr(pydicom, '__version__', 'Unknown')
    logger.info(f"Pydicom Version: {PYDICOM_VERSION}")
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_VERSION = 'Not Installed'
    logger.warning("pydicom not found. DICOM functionality will be disabled.")
    pydicom = None
    PYDICOM_AVAILABLE = False

if PYDICOM_AVAILABLE:
    try:
        import pylibjpeg
        logger.info("pylibjpeg found (for extended DICOM decompression).")
    except ImportError:
        logger.info("pylibjpeg not found. Some DICOM syntaxes may not be supported.")
    try:
        import gdcm
        logger.info("python-gdcm found (for improved DICOM compatibility).")
    except ImportError:
        logger.info("python-gdcm not found. Some DICOM functionalities may be reduced.")

# --- Custom Utilities & Backend Modules ---
# Assume these are available locally relative to app.py
# Create dummy files if they don't exist for basic testing
DUMMY_UTILS_CREATED = False
if not os.path.exists("dicom_utils.py"):
    with open("dicom_utils.py", "w") as f:
        f.write("""
import pydicom
from PIL import Image
import numpy as np

def parse_dicom(b, filename):
    try:
        from io import BytesIO
        return pydicom.dcmread(BytesIO(b))
    except Exception:
        return None

def extract_dicom_metadata(ds):
    return {str(elem.keyword): str(elem.value) for elem in ds if elem.keyword}

def dicom_to_image(ds, wc=None, ww=None, normalize=False):
    if ds is None or not hasattr(ds, 'pixel_array'): return None
    pixels = ds.pixel_array.astype(np.float32)
    if normalize:
        pixels = (pixels - np.min(pixels)) / (np.max(pixels) - np.min(pixels) + 1e-6) * 255.0
    elif wc is not None and ww is not None:
        wl = wc - ww / 2.0
        wh = wc + ww / 2.0
        pixels = np.clip(pixels, wl, wh)
        pixels = (pixels - wl) / (ww + 1e-6) * 255.0
    else: # Basic rescale if no W/L
        pixels = (pixels - np.min(pixels)) / (np.max(pixels) - np.min(pixels) + 1e-6) * 255.0

    pixels = np.uint8(pixels)
    return Image.fromarray(pixels).convert('RGB') # Ensure RGB

def get_default_wl(ds):
    wc = ds.get('WindowCenter', None)
    ww = ds.get('WindowWidth', None)
    if isinstance(wc, pydicom.multival.MultiValue): wc = wc[0]
    if isinstance(ww, pydicom.multival.MultiValue): ww = ww[0]
    try: return (float(wc), float(ww)) if wc is not None and ww is not None else (None, None)
    except (TypeError, ValueError): return (None, None)
        """)
    logger.info("Created dummy dicom_utils.py")
    DUMMY_UTILS_CREATED = True

if not os.path.exists("report_utils.py"):
    with open("report_utils.py", "w") as f:
        f.write("""
from PIL import Image
from io import BytesIO
def generate_pdf_report_bytes(**kwargs):
    # Dummy implementation - returns None
    print("WARN: Using dummy generate_pdf_report_bytes")
    return None
        """)
    logger.info("Created dummy report_utils.py")
    DUMMY_UTILS_CREATED = True

if not os.path.exists("ui_components.py"):
    with open("ui_components.py", "w") as f:
        f.write("""
import streamlit as st
def display_dicom_metadata(metadata):
    st.json({k: str(v)[:100] + '...' if len(str(v)) > 100 else str(v) for k, v in list(metadata.items())[:15]})
def dicom_wl_sliders(wc, ww):
    st.sidebar.caption("W/L sliders (dummy)")
    new_wc = st.sidebar.slider("Window Center (WC)", min_value=int(wc-1000), max_value=int(wc+1000), value=int(wc))
    new_ww = st.sidebar.slider("Window Width (WW)", min_value=1, max_value=int(ww*2+1000), value=int(ww))
    return float(new_wc), float(new_ww)
        """)
    logger.info("Created dummy ui_components.py")
    DUMMY_UTILS_CREATED = True

if not os.path.exists("translation_models.py"):
     with open("translation_models.py", "w") as f:
        f.write("""
from deep_translator import GoogleTranslator

LANGUAGE_CODES = {"English": "en", "Spanish": "es", "French": "fr", "German": "de"}
AUTO_DETECT_INDICATOR = "Auto-Detect"

def translate(text, target_language, source_language=None):
    try:
        target_code = LANGUAGE_CODES.get(target_language, 'en')
        source_code = 'auto' if source_language == AUTO_DETECT_INDICATOR else LANGUAGE_CODES.get(source_language, 'auto')
        return GoogleTranslator(source=source_code, target=target_code).translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return f"[Translation Error: {e}]"

def detect_language(text):
    try:
        return GoogleTranslator().detect(text) # Returns tuple (lang_code, lang_name) maybe? Adjust if needed
    except Exception as e:
        print(f"Language detection error: {e}")
        return None
        """)
     logger.info("Created dummy translation_models.py")
     DUMMY_UTILS_CREATED = True

# Reload modules if they were just created
if DUMMY_UTILS_CREATED:
    import importlib
    if 'dicom_utils' in sys.modules: importlib.reload(sys.modules['dicom_utils'])
    if 'report_utils' in sys.modules: importlib.reload(sys.modules['report_utils'])
    if 'ui_components' in sys.modules: importlib.reload(sys.modules['ui_components'])
    if 'translation_models' in sys.modules: importlib.reload(sys.modules['translation_models'])


try:
    from dicom_utils import (
        parse_dicom,
        extract_dicom_metadata,
        dicom_to_image,
        get_default_wl
    )
    DICOM_UTILS_AVAILABLE = True
    logger.info("dicom_utils imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import dicom_utils: {e}. DICOM features disabled.")
    DICOM_UTILS_AVAILABLE = False
    def parse_dicom(b, filename): return None
    def extract_dicom_metadata(ds): return {}
    def dicom_to_image(ds, wc=None, ww=None, normalize=False): return None
    def get_default_wl(ds): return (None, None)

try:
    from report_utils import generate_pdf_report_bytes
    REPORT_UTILS_AVAILABLE = True
    logger.info("report_utils imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import report_utils: {e}. PDF reporting disabled.")
    REPORT_UTILS_AVAILABLE = False
    def generate_pdf_report_bytes(**kwargs): return None # Fallback

try:
    from ui_components import display_dicom_metadata, dicom_wl_sliders
    UI_COMPONENTS_AVAILABLE = True
    logger.info("ui_components imported successfully.")
except ImportError as e:
    logger.warning(f"Failed to import ui_components: {e}. Custom UI elements might be missing.")
    UI_COMPONENTS_AVAILABLE = False
    def display_dicom_metadata(metadata): st.json({k: str(v)[:100] + '...' if len(str(v)) > 100 else str(v) for k, v in list(metadata.items())[:10]}) # Simple fallback
    def dicom_wl_sliders(wc, ww): return wc, ww

# --- Translation Setup ---
try:
    from translation_models import (
        translate, detect_language, LANGUAGE_CODES, AUTO_DETECT_INDICATOR
    )
    TRANSLATION_AVAILABLE = True
    logger.info("translation_models imported successfully. Translation is available.")
except ImportError as e:
    st.warning(f"Translation features disabled: {e}")
    logger.error(f"Could not import translation_models: {e}", exc_info=True)
    TRANSLATION_AVAILABLE = False
    translate = None
    detect_language = None
    LANGUAGE_CODES = {"English": "en"}
    AUTO_DETECT_INDICATOR = "Auto-Detect"

# --- Gemini Configuration ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
genai_client_configured = False
if GOOGLE_GENAI_AVAILABLE:
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            genai_client_configured = True
            logger.info("Google Generative AI client configured successfully.")
        except Exception as e:
            st.error(f"Fatal Error: Failed to configure Google Generative AI. Check API Key. Details: {e}", icon="🚨")
            logger.critical(f"Gemini configuration failed: {e}", exc_info=True)
            st.stop()
    else:
        # Attempt to get API key from environment variable if not in secrets
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
        if GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                genai_client_configured = True
                logger.info("Google Generative AI client configured successfully using environment variable.")
            except Exception as e:
                st.error(f"Fatal Error: Failed to configure Google Generative AI using env var. Check API Key. Details: {e}", icon="🚨")
                logger.critical(f"Gemini configuration failed using env var: {e}", exc_info=True)
                st.stop()
        else:
             st.error("⚠️ Gemini API Key not found. Please configure GEMINI_API_KEY in Streamlit secrets or environment variables.", icon="🔑")
             logger.critical("GEMINI_API_KEY not found.")
             st.stop()


# Initialize models using Session State
VISION_MODEL_NAME = 'gemini-1.5-flash' # Or 'gemini-pro-vision' if 1.5 not needed/available

if 'models_initialized' not in st.session_state:
    st.session_state.models_initialized = False
    st.session_state.vision_model = None

if genai_client_configured and not st.session_state.models_initialized:
    try:
        st.session_state.vision_model = genai.GenerativeModel(VISION_MODEL_NAME)
        st.session_state.models_initialized = True
        logger.info(f"Gemini models initialized: Vision='{VISION_MODEL_NAME}'")
    except Exception as e:
        st.error(f"Fatal Error: Failed to initialize Gemini models. Vision: {VISION_MODEL_NAME}. Details: {e}", icon="💥")
        logger.critical(f"Gemini model initialization failed: {e}", exc_info=True)
        st.stop()
elif not genai_client_configured and GOOGLE_GENAI_AVAILABLE:
    # This case should technically be handled by the API key check above, but double-checking
    st.error("AI Models could not be initialized due to configuration issues (API Key missing or invalid).", icon="🚫")
    st.stop()

# --- Gemini Prompt Templates ---
IMAGE_ANALYSIS_PROMPT_TEMPLATE = """
**Medical Image Analysis Request (Gemini Vision Model)**

**Role:** AI assistant providing objective visual observations on a medical image to support a healthcare professional. **This is NOT a diagnosis or radiological interpretation.**

**Context:** Analyze the provided medical image based *only* on visual information. User may provide additional context, a question, or a condition to focus on. An optional Region of Interest (ROI) may be described.

**Output Format:** Structure your response precisely using the following Markdown headings. Be factual, descriptive, and use cautious language.

**Task:**

1.  **## 1. Image Description:**
    *   Identify the likely imaging modality and view (e.g., PA Chest Radiograph, Axial CT slice of the abdomen).
    *   Briefly list the main anatomical structures clearly visible (e.g., ribs, heart silhouette, lung fields, diaphragm).
    *   Mention the Region of Interest (ROI) if provided: "{roi_description}"

2.  **## 2. Key Visual Findings / Observations:**
    *   Carefully examine the image (paying attention to the ROI if specified, but considering the whole image) for any areas that *appear* visually distinct or deviate from typical patterns.
    *   **Use extremely cautious, descriptive language.** Describe *what* you see visually (e.g., "area of increased opacity," "region of lucency," "asymmetry observed in X," "potential contour abnormality," "linear density," "patchy distribution").
    *   **Specify location accurately** using standard anatomical terms (e.g., "right lower lung zone," "left hilum," "hepatic flexure region," "within the described ROI").
    *   **Crucially, AVOID interpretive or diagnostic terms** (DO NOT use words like "pneumonia," "tumor," "fracture," "infection," "inflammation," "likely," "suggestive of"). Stick strictly to visual observation.
    *   If clearly discernible, mention the **absence** of certain major expected abnormalities (pertinent negatives, e.g., "No obvious large pneumothorax identified," "Bowel gas pattern appears unremarkable in visualized areas").
    *   Compare sides if applicable and relevant differences are seen (e.g., "Left lung field demonstrates greater transparency compared to the right").

3.  **## 3. Correlation with User Prompt/Focus (if provided):**
    *   Address the specific user question or condition focus based *strictly* on the visual information identifiable in the image and the findings noted in Step 2.
    *   If the user asked a question: Answer it based *only* on visual evidence. If the image cannot answer, state that clearly (e.g., "The image does not provide sufficient visual information to determine X.").
    *   If a condition focus was provided: Describe any visual findings from Step 2 that might be relevant to consider in the context of that condition, again using purely descriptive terms. State if no relevant visual signs are apparent.
    *   If no user prompt/focus was provided, state "N/A".

4.  **## 4. Limitations of this AI Analysis:**
    *   **Explicitly list the following limitations inherent to this AI visual observation:**
        *   Dependency on the **quality, resolution, and potential artifacts** of the single provided image.
        *   Analysis is restricted to the **single view/slice(s)** provided.
        *   **Complete lack of clinical context:** Patient history, symptoms, physical exam findings, and laboratory results are unknown and not considered.
        *   **Absence of prior imaging studies:** Comparisons over time are not possible.
        *   The AI functions purely on **visual pattern recognition**; it does not perform clinical reasoning or differential diagnosis.
        *   ROI focus (if used) might overlook findings outside the region.

5.  **## 5. Mandatory Disclaimer:**
    *   State clearly: This is an AI-generated visual observation intended for informational and demonstration purposes **ONLY**.
    *   It is **NOT** a radiological interpretation or medical diagnosis.
    *   It **CANNOT** substitute for a comprehensive evaluation and interpretation by a qualified radiologist or physician integrating full clinical information.
    *   Any potential observations noted herein **MUST** be correlated with clinical findings and reviewed/confirmed by qualified healthcare professionals.

**User's Specific Question / Condition Focus:**
---
{user_prompt}
---

**AI Visual Observation:**
"""

CONFIDENCE_ASSESSMENT_PROMPT_TEMPLATE = """
**AI Confidence Assessment Request (Gemini Vision Model)**

**Role:** AI assistant assessing the limitations and qualitative confidence of a *previous* AI-generated visual analysis of a medical image.

**Context:** You will be given the medical image, an optional Region of Interest (ROI) description, the previous AI analysis text, and potentially conversation history. Your task is to reflect on the certainty and potential weaknesses of that *prior* analysis based on the visual evidence in the image.

**Input:**
1.  The medical image.
2.  ROI Description (if any): "{roi_description}"
3.  The Previous AI Analysis Text:
    ---
    {previous_analysis}
    ---
4.  Relevant Conversation History (if any):
    ---
    {history_summary}
    ---

**Task:**

1.  **## 1. Review Previous Analysis:** Briefly acknowledge the main findings described in the provided `previous_analysis` text.
2.  **## 2. Assess Confidence & Limitations based on Visual Evidence:**
    *   Evaluate the visual clarity of the findings mentioned in the previous analysis *as seen in the image*.
    *   Comment on factors influencing confidence: Image quality (resolution, artifacts, noise), patient positioning, completeness of visible anatomy, subtlety or distinctness of findings.
    *   If an ROI was used, comment on how focusing on that region might impact confidence or potentially miss other findings.
    *   **Use qualitative descriptions:** "High confidence in observing the described opacity due to its clarity," "Lower confidence regarding subtle texture changes due to image noise," "The described asymmetry is clearly visible," "Assessment is limited by the single view provided," "Findings outside the ROI cannot be assessed with confidence."
    *   **DO NOT provide a numerical score.** Focus on *why* confidence might be higher or lower for specific observations based on the visual data.
3.  **## 3. Reiterate Key Limitations:** Briefly restate the most significant limitations identified (e.g., lack of clinical context, single view, image quality issues).
4.  **## 4. Mandatory Disclaimer (Confidence Assessment):**
    *   State clearly: This AI-generated confidence assessment is itself based on visual patterns and the provided text. It is **NOT** a guarantee of accuracy and is for informational purposes only.
    *   It **CANNOT** replace the judgment of a qualified healthcare professional who integrates all clinical data. The original analysis and this confidence assessment must be reviewed critically.

**AI Confidence Assessment:**
"""

# --- Gemini Interaction Functions ---
def generate_roi_description(roi_coords: Optional[Dict]) -> str:
    """Generates a textual description of the ROI for the prompt."""
    if not roi_coords:
        return "N/A"
    try:
        # Ensure coords are integers and handle potential floating point inputs
        left = int(roi_coords['left'])
        top = int(roi_coords['top'])
        width = int(roi_coords['width'])
        height = int(roi_coords['height'])
        # Basic description - could be enhanced with relative positioning later
        return (f"User highlighted a Region of Interest (ROI) at "
                f"Top-Left corner ({left},{top}) with "
                f"Width={width}px, Height={height}px. "
                f"Focus on findings within or related to this region, while maintaining overall context.")
    except (KeyError, TypeError, ValueError) as e:
        logger.warning(f"Could not format ROI coords {roi_coords}: {e}")
        return "ROI specified, but coordinates unclear."

def handle_gemini_response(response: Any) -> Tuple[Optional[str], Optional[str]]:
    """Safely extracts text from Gemini response or returns error."""
    try:
        # Attempt to access text directly first (common case)
        if hasattr(response, 'text'):
            final_text = response.text
        # Check candidates if direct text access fails or is empty
        elif hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            # Check content parts within the candidate
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                final_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text')) # Concatenate parts
            else:
                final_text = None # No text found in parts
        else:
            final_text = None # No text attribute or candidates found

        # Handle potential blocking or premature finish reasons
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
            reason = response.prompt_feedback.block_reason.name
            logger.warning(f"Gemini analysis blocked by safety filters: {reason}")
            return None, f"Analysis blocked by safety filters: {reason}. This might relate to sensitive content policies or image characteristics. Please review input or contact support."

        if hasattr(response, 'candidates') and response.candidates:
             candidate = response.candidates[0]
             if hasattr(candidate, 'finish_reason') and candidate.finish_reason != genai.types.FinishReason.STOP:
                 reason = candidate.finish_reason.name
                 logger.warning(f"Gemini analysis stopped prematurely: {reason}")
                 # Return any partial text collected along with the error
                 partial_text = final_text if final_text else "[Partial response not available]"
                 return partial_text, f"Analysis stopped prematurely. Reason: {reason}. Input might be too long, complex, or triggered other limits."


        # If we have text, add disclaimer if needed and return
        if final_text is not None:
            if "Mandatory Disclaimer" not in final_text and "NOT** a radiological interpretation" not in final_text:
                 # Add the general disclaimer if missing
                 final_text += "\n\n**## Mandatory Disclaimer:**\nThis is an AI-generated visual observation intended for informational and demonstration purposes ONLY. It is **NOT** a radiological interpretation or medical diagnosis. It **CANNOT** substitute for a comprehensive evaluation and interpretation by a qualified radiologist or physician integrating full clinical information. Any potential observations noted herein **MUST** be correlated with clinical findings and reviewed/confirmed by qualified healthcare professionals."
            return final_text, None
        else:
            # If no text and no block/stop reason identified above, it's an unexpected empty response
            logger.warning("Received an empty or unexpected response structure from Gemini.")
            return None, "Received an empty or unexpected response from the AI model."

    except AttributeError as e:
        logger.error(f"Error parsing Gemini response structure: {e}. Response object: {type(response)}", exc_info=True)
        return None, f"Internal error parsing AI response structure: {e}"
    except Exception as e:
        logger.error(f"Unexpected error handling Gemini response: {e}", exc_info=True)
        return None, f"Unexpected error processing AI response: {e}"


def run_gemini_image_analysis(image: Image.Image, roi_coords: Optional[Dict] = None) -> Tuple[Optional[str], Optional[str]]:
    """Performs initial visual observation using Gemini Vision."""
    if not st.session_state.models_initialized or not st.session_state.vision_model:
        return None, "Vision model not initialized."
    if not PIL_AVAILABLE or not isinstance(image, Image.Image):
         return None, "Invalid image provided for analysis."

    try:
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        roi_desc = generate_roi_description(roi_coords)
        prompt = IMAGE_ANALYSIS_PROMPT_TEMPLATE.format(
            user_prompt="N/A - Perform general visual observation.",
            roi_description=roi_desc
        )
        model_input = [prompt, image]
        logger.info("Sending request to Gemini for initial image analysis...")
        response = st.session_state.vision_model.generate_content(model_input, request_options={'timeout': 120}) # Increased timeout
        logger.info("Received response from Gemini for initial analysis.")
        return handle_gemini_response(response)

    except Exception as e:
        logger.error(f"ERROR in run_gemini_image_analysis: {e}", exc_info=True)
        return None, f"An internal error occurred: {e}"

def run_gemini_image_qa(image: Image.Image, question: str, history: List, roi_coords: Optional[Dict] = None) -> Tuple[Optional[str], Optional[str]]:
    """Answers a question about the image using Gemini Vision."""
    if not st.session_state.models_initialized or not st.session_state.vision_model:
        return None, "Vision model not initialized."
    if not PIL_AVAILABLE or not isinstance(image, Image.Image):
         return None, "Invalid image provided for Q&A."
    if not question or not question.strip():
        return None, "Question cannot be empty."

    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        roi_desc = generate_roi_description(roi_coords)
        prompt = IMAGE_ANALYSIS_PROMPT_TEMPLATE.format(
            user_prompt=question.strip(), # Use the actual question here
            roi_description=roi_desc
        )

        model_input = [prompt, image]
        logger.info(f"Sending request to Gemini for Q&A: '{question[:50]}...'")
        response = st.session_state.vision_model.generate_content(model_input, request_options={'timeout': 120}) # Increased timeout
        logger.info("Received response from Gemini for Q&A.")
        return handle_gemini_response(response)

    except Exception as e:
        logger.error(f"ERROR in run_gemini_image_qa: {e}", exc_info=True)
        return None, f"An internal error occurred: {e}"

def run_gemini_condition_analysis(image: Image.Image, condition: str, roi_coords: Optional[Dict] = None) -> Tuple[Optional[str], Optional[str]]:
    """Analyzes image focusing on a specific condition using Gemini Vision."""
    if not st.session_state.models_initialized or not st.session_state.vision_model:
        return None, "Vision model not initialized."
    if not PIL_AVAILABLE or not isinstance(image, Image.Image):
         return None, "Invalid image provided for condition analysis."
    if not condition or not condition.strip():
        return None, "Condition cannot be empty."

    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        roi_desc = generate_roi_description(roi_coords)
        condition_prompt = (f"Focus the visual observation specifically on findings potentially related to **{condition}**. "
                            f"Describe any relevant visual signs using objective, descriptive language as outlined in the main prompt instructions. "
                            f"State if no specific signs related to {condition} are visually apparent.")

        prompt = IMAGE_ANALYSIS_PROMPT_TEMPLATE.format(
            user_prompt=condition_prompt, # Specific instruction here
            roi_description=roi_desc
        )
        model_input = [prompt, image]
        logger.info(f"Sending request to Gemini for condition analysis: '{condition}'")
        response = st.session_state.vision_model.generate_content(model_input, request_options={'timeout': 120}) # Increased timeout
        logger.info("Received response from Gemini for condition analysis.")
        return handle_gemini_response(response)

    except Exception as e:
        logger.error(f"ERROR in run_gemini_condition_analysis: {e}", exc_info=True)
        return None, f"An internal error occurred: {e}"

def run_gemini_confidence_assessment(image: Image.Image, previous_analysis: str, history: List, roi_coords: Optional[Dict] = None) -> Tuple[Optional[str], Optional[str]]:
    """Assesses qualitative confidence of previous analysis using Gemini Vision."""
    if not st.session_state.models_initialized or not st.session_state.vision_model:
        return None, "Vision model not initialized."
    if not PIL_AVAILABLE or not isinstance(image, Image.Image):
         return None, "Invalid image provided for confidence assessment."
    if not previous_analysis or not previous_analysis.strip():
        # Need some prior analysis to assess confidence on
        return None, "No previous analysis text provided to assess confidence."

    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        roi_desc = generate_roi_description(roi_coords)
        history_summary = "\n".join([f"{role}: {text}" for role, text in history[-4:]]) # Last 4 interactions

        prompt = CONFIDENCE_ASSESSMENT_PROMPT_TEMPLATE.format(
            roi_description=roi_desc,
            previous_analysis=previous_analysis.strip(),
            history_summary=history_summary or "N/A"
        )
        model_input = [prompt, image]
        logger.info("Sending request to Gemini for confidence assessment...")
        response = st.session_state.vision_model.generate_content(model_input, request_options={'timeout': 120}) # Increased timeout
        logger.info("Received response from Gemini for confidence assessment.")

        # Use the general response handler, then manually add the specific confidence disclaimer
        result_text, error = handle_gemini_response(response)

        # Add the specific confidence disclaimer if analysis succeeded and it's missing
        if result_text and "Mandatory Disclaimer (Confidence Assessment)" not in result_text:
             result_text += "\n\n**## 4. Mandatory Disclaimer (Confidence Assessment):**\nThis AI-generated confidence assessment is itself based on visual patterns and the provided text. It is **NOT** a guarantee of accuracy and is for informational purposes only. It **CANNOT** replace the judgment of a qualified healthcare professional who integrates all clinical data. The original analysis and this confidence assessment must be reviewed critically."

        return result_text, error

    except Exception as e:
        logger.error(f"ERROR in run_gemini_confidence_assessment: {e}", exc_info=True)
        return None, f"An internal error occurred: {e}"

# --- Custom CSS ---
st.markdown(
    """
    <style>
      body {
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
          background-color: #f0f2f6; /* Light grey background */
      }
      .main .block-container {
          padding-top: 2rem;
          padding-bottom: 2rem;
          padding-left: 1.5rem;
          padding-right: 1.5rem;
          max-width: 1400px; /* Limit max width for very wide screens */
          margin: auto;
      }
      /* Sidebar styling */
      [data-testid="stSidebar"] { /* More robust selector */
          background-color: #ffffff;
          border-right: 1px solid #e0e0e0;
          padding: 1rem;
      }
      [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
          color: #333; /* Darker heading color */
      }
      /* Button styling */
      .stButton>button {
          border-radius: 8px;
          padding: 0.6rem 1.2rem; /* Slightly larger padding */
          font-weight: 500;
          border: 1px solid #007bff; /* Primary color border */
          background-color: #007bff; /* Primary color background */
          color: white;
          transition: background-color 0.2s ease-in-out, border-color 0.2s ease-in-out, transform 0.1s ease;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      }
      .stButton>button:hover {
        background-color: #0056b3; /* Darker shade on hover */
        border-color: #0056b3;
        filter: brightness(100%); /* Remove default dimming */
      }
      .stButton>button:active {
          transform: scale(0.98); /* Slight press effect */
      }
      .stButton>button:disabled {
          background-color: #cccccc;
          border-color: #cccccc;
          color: #666666;
          cursor: not-allowed;
          box-shadow: none;
      }
      /* Special button styling (e.g., clear ROI) */
      .stButton button[kind="secondary"] { /* Example for secondary buttons if needed */
           background-color: #6c757d;
           border-color: #6c757d;
      }
       .stButton button[kind="secondary"]:hover {
           background-color: #5a6268;
           border-color: #5a6268;
      }
      /* Tab styling */
      .stTabs [role="tablist"] {
            overflow-x: auto;
            white-space: nowrap;
            border-bottom: 2px solid #dee2e6; /* Thicker border */
            scrollbar-width: thin;
            scrollbar-color: #cccccc #f0f2f6;
      }
      .stTabs [role="tab"] {
          padding: 0.75rem 1.25rem; /* Adjust tab padding */
          font-weight: 500;
          color: #495057;
          transition: border-color 0.2s ease, color 0.2s ease;
          border-bottom: 2px solid transparent; /* Prepare for active border */
          margin-bottom: -2px; /* Align with bottom border */
      }
      .stTabs [role="tab"]:hover {
          color: #0056b3;
          border-bottom-color: #adb5bd; /* Light border on hover */
      }
       .stTabs [role="tab"][aria-selected="true"] {
           color: #007bff; /* Active tab color */
           border-bottom-color: #007bff; /* Active tab border */
      }
      /* Tab scrolling - Webkit browsers */
      .stTabs [role="tablist"]::-webkit-scrollbar { height: 6px; }
      .stTabs [role="tablist"]::-webkit-scrollbar-track { background: #f0f2f6; }
      .stTabs [role="tablist"]::-webkit-scrollbar-thumb { background-color: #cccccc; border-radius: 10px; border: 2px solid #f0f2f6; }
      /* Footer styling */
       footer {
          text-align: center;
          font-size: 0.85em; /* Slightly larger footer text */
          color: #6c757d;
          margin-top: 3rem; /* More space before footer */
          padding: 1.5rem 0;
          border-top: 1px solid #e0e0e0;
          background-color: #ffffff; /* Footer background */
      }
      footer p {
          margin-bottom: 0.5rem; /* Space between footer lines */
      }
      footer a { color: #007bff; text-decoration: none; }
      footer a:hover { text-decoration: underline; }
       /* Ensure canvas fits well */
      [data-testid="stDrawableCanvas"] {
          border: 1px solid #ced4da; /* Add a light border to the canvas */
          border-radius: 4px;
      }
      /* Make expander headers slightly bolder */
      .streamlit-expanderHeader {
          font-weight: 500;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Display Hero Logo ---
# Create dummy asset if it doesn't exist
ASSETS_DIR = "assets"
if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR)
logo_path = os.path.join(ASSETS_DIR, "radvisionai-hero.jpeg")
if not os.path.exists(logo_path) and PIL_AVAILABLE:
    try:
        img = Image.new('RGB', (350, 70), color = (73, 109, 137))
        d = ImageDraw.Draw(img)
        # You might need to install a font or use a default one
        try:
            # Try loading a common font, adjust path if needed
            # from PIL import ImageFont
            # font = ImageFont.truetype("arial.ttf", 20) # Example, requires font file
            d.text((10,10), "RadVision AI (Dummy Logo)", fill=(255,255,255))
        except IOError:
            d.text((10,10), "RadVision AI (Dummy Logo - Font Error)", fill=(255,255,0)) # Fallback text

        img.save(logo_path, "JPEG")
        logger.info(f"Created dummy logo at: {logo_path}")
    except Exception as e:
        logger.error(f"Could not create dummy logo: {e}")

if os.path.exists(logo_path):
    st.image(logo_path, width=350)
else:
    logger.warning(f"Hero logo not found and could not be created at: {logo_path}")
    st.subheader("RadVision AI Advanced (Gemini)") # Fallback text title


# --- Initialize Session State Defaults ---
DEFAULT_STATE = {
    "uploaded_file_info": None,
    "raw_image_bytes": None,
    "is_dicom": False,
    "dicom_dataset": None,
    "dicom_metadata": {},
    "processed_image": None, # Image ready for AI model (e.g., RGB)
    "display_image": None,   # Image for display (could have W/L applied)
    "session_id": None,
    "history": [], # List of tuples: ("User Question"/"AI Answer"/"System", message_text)
    "initial_analysis": "",
    "qa_answer": "",
    "disease_analysis": "",
    "confidence_score": "", # Stores qualitative assessment text
    "last_action": None,
    "pdf_report_bytes": None,
    "canvas_drawing": None, # Stores the state of the drawing canvas
    "roi_coords": None,     # Stores {'left': x, 'top': y, 'width': w, 'height': h}
    "current_display_wc": None,
    "current_display_ww": None,
    "clear_roi_feedback": False,
    "demo_loaded": False,
    "translation_result": None,
    "translation_error": None,
}

# Initialize session state only if keys don't exist
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value

if not isinstance(st.session_state.get("history", []), list):
    st.session_state.history = []

if not st.session_state.get("session_id"):
    st.session_state.session_id = str(uuid.uuid4())[:8]
logger.debug(f"Session state verified/initialized for session ID: {st.session_state.session_id}")

# --- Helper Functions ---
def format_translation(translated_text: Optional[str]) -> str:
    """Applies basic formatting to translated text."""
    if translated_text is None:
        return "Translation not available or failed."
    try:
        text_str = str(translated_text)
        # Basic formatting: Add newlines before numbered lists detected loosely
        formatted_text = re.sub(r'(?<=\S)\s+(\d+\.\s)', r'\n\n\1', text_str)
        # Ensure consistent spacing around markdown headings
        formatted_text = re.sub(r'\n*##\s*(\d+)\.\s*(.*)', r'\n\n## \1. \2\n', formatted_text)
        return formatted_text.strip()
    except Exception as e:
        logger.error(f"Error formatting translation: {e}", exc_info=True)
        return str(translated_text) # Return original if formatting fails

# --- Monkey-Patch (Conditional for older Streamlit versions) ---
import streamlit.elements.image as st_image # Import the image module
if not hasattr(st_image, "image_to_url"):
    logger.info("Streamlit version appears older. Applying monkey-patch for 'image_to_url'.")
    def image_to_url_monkey_patch(
        image: Any,
        width: int = -1,
        clamp: bool = False,
        channels: str = "RGB",
        output_format: str = "auto",
        image_id: str = "",
    ) -> str:
        """
        Monkey-patch implementation for st.elements.image.image_to_url.
        Converts PIL Images or numpy arrays to a data URL.
        """
        if PIL_AVAILABLE and isinstance(image, Image.Image):
            img_pil = image
            if channels == "BGR":
                 try:
                     import numpy as np
                     img_pil = Image.fromarray(np.array(img_pil)[:, :, ::-1])
                 except ImportError:
                     logger.error("NumPy needed for BGR conversion in image_to_url patch, but not installed.")
                     return ""
                 except Exception as e:
                     logger.error(f"Error during BGR conversion in image_to_url patch: {e}")
                     return ""

            if img_pil.mode not in ["RGB", "RGBA", "L"]:
                 logger.warning(f"Converting image mode {img_pil.mode} to RGB for URL generation.")
                 img_pil = img_pil.convert("RGB")

            format = output_format.upper()
            if format == "AUTO":
                format = "PNG"
            elif format not in ["PNG", "JPEG", "JPG"]:
                logger.warning(f"Unsupported output format '{format}'. Defaulting to PNG.")
                format = "PNG"
            if format == "JPG": format = "JPEG"

            try:
                buffered = io.BytesIO()
                img_pil.save(buffered, format=format)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/{format.lower()};base64,{img_str}"
            except Exception as e:
                logger.error(f"Failed to convert PIL Image to data URL: {e}", exc_info=True)
                return ""

        elif 'numpy' in sys.modules and isinstance(image, sys.modules['numpy'].ndarray):
             try:
                 import numpy as np
                 img_np = image
                 if channels == "BGR":
                     if img_np.ndim == 3 and img_np.shape[2] == 3:
                         img_np = img_np[..., ::-1]
                     else:
                         logger.warning("Received numpy array and channels='BGR' but array shape is not typical BGR.")

                 img_pil_from_np = Image.fromarray(np.uint8(img_np))
                 return image_to_url_monkey_patch(
                     img_pil_from_np, width, clamp, "RGB", output_format, image_id
                 )
             except ImportError:
                 logger.error("Numpy is required to handle numpy arrays in image_to_url patch, but not installed.")
                 return ""
             except Exception as e:
                 logger.error(f"Failed to convert Numpy array to data URL: {e}", exc_info=True)
                 return ""
        else:
             logger.error(f"Unsupported image type for image_to_url monkey-patch: {type(image)}")
             return ""

    # --- Perform the actual monkey-patch ---
    st_image.image_to_url = image_to_url_monkey_patch
    logger.info("Successfully applied monkey-patch for 'st.elements.image.image_to_url'.")
else:
    logger.debug("Streamlit version has 'image_to_url' natively. No monkey-patch needed.")
# --- End of Monkey-Patch Section ---


# --- Sidebar ---
with st.sidebar:
    st.header("⚕️ RadVision Controls")
    st.markdown("_(Powered by Gemini)_")
    st.markdown("---")

    TIPS = [
        "Draw a rectangle (ROI) on the image to focus the AI's observation.",
        "Ask specific questions about visual findings (e.g., 'Describe the opacity in the ROI').",
        "Use 'Analyze Condition' to look for visual signs relevant to a condition.",
        "AI output is observational, NOT diagnostic. Always correlate with clinical data.",
        "Use 'Assess AI Confidence' for a qualitative assessment of analysis limitations.",
        "Generate a PDF report to document the AI observations.",
        "Ensure uploaded images are clear and relevant for best results.",
        "Check the 'Limitations' section in the AI output.",
    ]
    st.info(f"💡 Tip: {random.choice(TIPS)}")
    st.markdown("---")

    # Upload
    st.subheader("1. Image Upload & Settings")
    uploaded_file = st.file_uploader(
        "Upload Image (JPG, PNG, DCM)",
        type=["jpg", "jpeg", "png", "dcm", "dicom"],
        key="file_uploader_widget",
        help="Upload a medical image file. Gemini will provide visual observations."
    )

    # Clear ROI
    if st.button("🗑️ Clear ROI Selection", help="Remove the selected ROI rectangle"):
        st.session_state.roi_coords = None
        st.session_state.canvas_drawing = None # Reset canvas drawing state too
        st.session_state.clear_roi_feedback = True
        logger.info("ROI Cleared by user.")
        st.rerun() # Rerun to update canvas immediately

    if st.session_state.get("clear_roi_feedback"):
        st.success("✅ ROI cleared!")
        st.session_state.clear_roi_feedback = False # Reset feedback flag

    # DICOM Window/Level
    if st.session_state.is_dicom and UI_COMPONENTS_AVAILABLE and st.session_state.display_image:
        st.markdown("---")
        st.caption("DICOM Display Settings")
        if st.session_state.current_display_wc is not None and st.session_state.current_display_ww is not None:
            new_wc, new_ww = dicom_wl_sliders(
                st.session_state.current_display_wc,
                st.session_state.current_display_ww
            )
            if new_wc != st.session_state.current_display_wc or new_ww != st.session_state.current_display_ww:
                logger.info(f"DICOM W/L changed via UI: WC={new_wc}, WW={new_ww}")
                st.session_state.current_display_wc = new_wc
                st.session_state.current_display_ww = new_ww
                if DICOM_UTILS_AVAILABLE and st.session_state.dicom_dataset:
                    with st.spinner("Applying new Window/Level..."):
                        new_display_img = dicom_to_image(
                            st.session_state.dicom_dataset,
                            wc=new_wc,
                            ww=new_ww
                        )
                        if isinstance(new_display_img, Image.Image):
                            if new_display_img.mode != 'RGB':
                                new_display_img = new_display_img.convert('RGB')
                            st.session_state.display_image = new_display_img
                            logger.info("DICOM display image updated with new W/L.")
                            st.rerun() # Rerun to update the image display
                        else:
                            st.error("Failed to update DICOM image with new W/L.")
                            logger.error("dicom_to_image returned invalid object during W/L update.")
                else:
                    st.warning("DICOM utilities not available to update W/L.")
        else:
            st.caption("Default W/L applied. Sliders available if values are detected/adjustable.")

    st.markdown("---")
    st.subheader("2. AI Analysis Actions")

    action_disabled = not (PIL_AVAILABLE and isinstance(st.session_state.get("processed_image"), Image.Image)) or not st.session_state.models_initialized

    if st.button("▶️ Run Initial Visual Observation", key="analyze_btn", disabled=action_disabled,
                 help="Ask Gemini for a general visual observation of the image or ROI."):
        st.session_state.last_action = "analyze"
        st.rerun()

    # --- Q&A Section ---
    with st.expander("❓ Ask AI About Image", expanded=True):
        question_input = st.text_area(
            "Enter your question:",
            height=100,
            key="question_input_widget",
            placeholder="E.g., 'Describe the structures within the ROI.' or 'Are there any linear opacities visible?'",
            disabled=action_disabled
        )
        if st.button("💬 Ask Gemini", key="ask_btn", disabled=action_disabled or not question_input.strip()):
            if question_input.strip():
                st.session_state.last_action = "ask"
                st.rerun()
            # No need for else warning as button is disabled if empty

    # --- Condition Focus Section ---
    with st.expander("🎯 Focus on Potential Condition Signs"):
        DISEASE_OPTIONS = [
            "Pneumonia", "Lung Cancer", "Nodule/Mass", "Effusion", "Fracture",
            "Stroke", "Appendicitis", "Bowel Obstruction", "Cardiomegaly",
            "Aortic Aneurysm", "Pulmonary Embolism", "Tuberculosis", "COVID-19",
            "Brain Tumor", "Arthritis", "Osteoporosis", "Other (Describe in Q&A)",
        ]
        disease_select = st.selectbox(
            "Select condition for focused visual search:",
            options=[""] + sorted(DISEASE_OPTIONS),
            key="disease_select_widget",
            disabled=action_disabled,
            help="Ask Gemini to look for visual signs relevant to this condition (descriptive only)."
        )
        if st.button("🩺 Analyze Visual Signs for Condition", key="disease_btn", disabled=action_disabled or not disease_select):
            if disease_select:
                st.session_state.last_action = "disease"
                st.rerun()
            # No need for else warning as button is disabled if empty

    st.markdown("---")
    st.subheader("3. Assessment & Reporting")

    prior_analysis_exists = bool(
        st.session_state.initial_analysis or
        (st.session_state.qa_answer and 'Error' not in st.session_state.qa_answer) or # Check for valid QA
        st.session_state.disease_analysis
    )
    can_estimate = prior_analysis_exists and not action_disabled

    if st.button("📈 Assess AI Confidence/Limitations", key="confidence_btn", disabled=not can_estimate,
                 help="Get AI's assessment of the certainty and limitations of its previous analysis."):
        if can_estimate:
            st.session_state.last_action = "confidence"
            st.rerun()
        # Button is disabled if cannot estimate

    report_generation_disabled = action_disabled or not REPORT_UTILS_AVAILABLE
    if st.button("📄 Generate PDF Report Data", key="generate_report_data_btn",
                 disabled=report_generation_disabled, help="Compile observations into data for PDF report."):
        st.session_state.last_action = "generate_report_data"
        st.rerun()

    # Download button appears only after PDF data is generated
    if st.session_state.get("pdf_report_bytes"):
        report_filename = f"RadVisionAI_Gemini_Report_{st.session_state.session_id or 'session'}.pdf"
        st.download_button(
            label="⬇️ Download PDF Report",
            data=st.session_state.pdf_report_bytes,
            file_name=report_filename,
            mime="application/pdf",
            key="download_pdf_button",
            help="Download the generated PDF report."
        )

# --- File Upload Logic ---
if uploaded_file is not None:
    try:
        uploaded_file.seek(0)
        file_content_hash = hashlib.sha256(uploaded_file.read()).hexdigest()[:16]
        uploaded_file.seek(0) # Reset pointer after reading
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{file_content_hash}"
    except Exception as e:
        logger.warning(f"Could not generate hash for uploaded file '{uploaded_file.name}': {e}")
        # Fallback to using a unique ID if hashing fails
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{uuid.uuid4().hex[:8]}"

    # Process only if it's a different file than the one currently loaded
    if new_file_info != st.session_state.get("uploaded_file_info"):
        logger.info(f"New file uploaded: {uploaded_file.name} ({uploaded_file.size} bytes), Info: {new_file_info}")
        st.toast(f"Processing '{uploaded_file.name}'...", icon="⏳")

        # --- Reset relevant session state for new file ---
        st.session_state.session_id = str(uuid.uuid4())[:8] # New session ID for new analysis
        logger.info(f"Resetting session state for new file, new Session ID: {st.session_state.session_id}")
        # Preserve essential states like models and potentially uploader widget state
        keys_to_preserve = {"session_id", "file_uploader_widget", "models_initialized", "vision_model"}
        for key, value in DEFAULT_STATE.items():
            if key not in keys_to_preserve:
                st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value
        # ------------------------------------

        st.session_state.uploaded_file_info = new_file_info
        st.session_state.demo_loaded = False

        st.session_state.raw_image_bytes = uploaded_file.getvalue()
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        # Determine if it's likely DICOM
        st.session_state.is_dicom = (
            PYDICOM_AVAILABLE and DICOM_UTILS_AVAILABLE and
            ("dicom" in uploaded_file.type.lower() or file_ext in (".dcm", ".dicom"))
        )

        with st.spinner("🔬 Analyzing file format..."):
            temp_display_img = None
            temp_processed_img = None # This will be the RGB image for Gemini
            processing_success = False
            error_msg = None

            if st.session_state.is_dicom:
                logger.info("Attempting to process as DICOM...")
                try:
                    dicom_dataset = parse_dicom(st.session_state.raw_image_bytes, filename=uploaded_file.name)
                    st.session_state.dicom_dataset = dicom_dataset # Store the dataset object
                    if dicom_dataset:
                        st.session_state.dicom_metadata = extract_dicom_metadata(dicom_dataset)
                        # Get default W/L for initial display
                        default_wc, default_ww = get_default_wl(dicom_dataset)
                        # Validate W/L before storing
                        if isinstance(default_wc, (int, float)) and isinstance(default_ww, (int, float)):
                             st.session_state.current_display_wc = default_wc
                             st.session_state.current_display_ww = default_ww
                             logger.info(f"Default DICOM W/L detected: WC={default_wc}, WW={default_ww}")
                        else:
                             logger.warning("Invalid or missing W/L values in DICOM, will attempt default scaling.")
                             st.session_state.current_display_wc = None
                             st.session_state.current_display_ww = None


                        # Generate display image with default W/L (or fallback scaling in dicom_to_image)
                        temp_display_img = dicom_to_image(dicom_dataset, wc=st.session_state.current_display_wc, ww=st.session_state.current_display_ww)
                        # Generate processed image (normalized, full dynamic range then converted to RGB) for AI
                        temp_processed_img = dicom_to_image(dicom_dataset, wc=None, ww=None, normalize=True)

                        if PIL_AVAILABLE and isinstance(temp_display_img, Image.Image) and isinstance(temp_processed_img, Image.Image):
                             # Ensure both are RGB
                            if temp_display_img.mode != 'RGB':
                                logger.debug(f"Converting display image from {temp_display_img.mode} to RGB")
                                temp_display_img = temp_display_img.convert('RGB')
                            if temp_processed_img.mode != 'RGB':
                                logger.debug(f"Converting processed image from {temp_processed_img.mode} to RGB")
                                temp_processed_img = temp_processed_img.convert('RGB')
                            processing_success = True
                            logger.info("DICOM parsed and converted to display/processed images successfully.")
                        else:
                            error_msg = "Failed to convert DICOM pixel data to displayable/processable image formats."
                            logger.error(f"{error_msg}. Display type: {type(temp_display_img)}, Processed type: {type(temp_processed_img)}")
                            st.session_state.dicom_dataset = None # Clear dataset if conversion failed
                            st.session_state.is_dicom = False # Treat as non-dicom if pixels unreadable
                    else:
                        error_msg = "Could not parse the DICOM file structure. It might be invalid or corrupted."
                        logger.error(error_msg)
                        st.session_state.is_dicom = False # Fallback
                except (pydicom.errors.InvalidDicomError, Exception) as e:
                    error_msg = f"Error processing DICOM file: {e}. Trying as standard image."
                    logger.warning(f"{error_msg} Filename: {uploaded_file.name}", exc_info=False) # Less verbose logs for common fallback
                    st.session_state.is_dicom = False # Fallback
                    st.session_state.dicom_dataset = None # Clear dataset on error

            # If not DICOM or DICOM processing failed, try standard image processing
            if not processing_success:
                logger.info("Attempting to process as standard image (JPG/PNG)...")
                st.session_state.is_dicom = False # Ensure flag is False
                st.session_state.dicom_dataset = None
                st.session_state.dicom_metadata = {}
                if not PIL_AVAILABLE:
                    error_msg = "Cannot process standard images: Pillow (PIL) library is missing."
                    logger.critical(error_msg)
                else:
                    try:
                        raw_img = Image.open(io.BytesIO(st.session_state.raw_image_bytes))
                        # Convert to RGB for consistency for display and AI
                        # Handle potential RGBA or P modes etc.
                        if raw_img.mode == 'RGBA':
                            logger.debug(f"Converting RGBA image '{uploaded_file.name}' to RGB.")
                            # Create a white background image
                            bg = Image.new("RGB", raw_img.size, (255, 255, 255))
                            bg.paste(raw_img, mask=raw_img.split()[3]) # Paste using alpha channel as mask
                            processed_img = bg
                        elif raw_img.mode == 'P':
                             logger.debug(f"Converting Palette image '{uploaded_file.name}' to RGB.")
                             processed_img = raw_img.convert("RGB")
                        elif raw_img.mode != 'RGB':
                             logger.debug(f"Converting image '{uploaded_file.name}' from mode {raw_img.mode} to RGB.")
                             processed_img = raw_img.convert("RGB")
                        else:
                            processed_img = raw_img # Already RGB

                        # For standard images, display and processed can be the same initially
                        temp_display_img = processed_img.copy()
                        temp_processed_img = processed_img.copy()
                        processing_success = True
                        logger.info(f"Standard image '{uploaded_file.name}' loaded and converted to RGB successfully.")
                    except UnidentifiedImageError:
                        error_msg = "Could not identify the image format. Please upload a valid JPG, PNG, or DICOM file."
                        logger.error(f"{error_msg} Filename: {uploaded_file.name}")
                    except Exception as e:
                        error_msg = f"An error occurred processing the standard image file: {e}"
                        logger.error(f"{error_msg} Filename: {uploaded_file.name}", exc_info=True)

            # Final check and state update
            if processing_success and PIL_AVAILABLE and isinstance(temp_display_img, Image.Image) and isinstance(temp_processed_img, Image.Image):
                st.session_state.display_image = temp_display_img
                st.session_state.processed_image = temp_processed_img
                # Reset ROI if a new image is loaded
                st.session_state.roi_coords = None
                st.session_state.canvas_drawing = None
                st.success(f"✅ '{uploaded_file.name}' loaded successfully!")
                logger.info(f"Image processing complete for: {uploaded_file.name}. Ready for display and AI.")
                st.rerun() # Rerun to update UI with the new image
            else:
                # Clear states if processing failed
                st.session_state.uploaded_file_info = None # Clear info so upload can be retried
                st.session_state.display_image = None
                st.session_state.processed_image = None
                st.session_state.is_dicom = False
                st.session_state.dicom_dataset = None
                st.session_state.dicom_metadata = {}
                st.session_state.raw_image_bytes = None
                st.error(f"Image loading failed: {error_msg or 'Unknown error during processing'}. Please try a different file or check format.", icon="❌")
                logger.error(f"Image processing ultimately failed for file: {uploaded_file.name}. Error: {error_msg}")
                # Do not rerun here, let the user see the error and upload again

# --- Main Page ---
st.markdown("---") # Separator after logo/title
# Title is handled by logo or fallback text above

# --- Main Layout ---
col1, col2 = st.columns([2, 3]) # Adjust ratio if needed (e.g., [1, 1] or [3, 2])

with col1:
    st.subheader("🖼️ Image Viewer")
    display_img = st.session_state.get("display_image")

    if PIL_AVAILABLE and isinstance(display_img, Image.Image):
        if DRAWABLE_CANVAS_AVAILABLE and st_canvas:
            st.caption("Draw a rectangle below to define a Region of Interest (ROI). Use sidebar to clear.")
            # --- Calculate canvas dimensions based on image aspect ratio ---
            MAX_CANVAS_WIDTH = 600
            MAX_CANVAS_HEIGHT = 500
            img_w, img_h = display_img.size

            # Calculate aspect ratio, handle potential division by zero
            aspect_ratio = (img_w / img_h) if img_h > 0 else 1

            canvas_width = MAX_CANVAS_WIDTH
            canvas_height = int(canvas_width / aspect_ratio) if aspect_ratio > 0 else MAX_CANVAS_HEIGHT

            if canvas_height > MAX_CANVAS_HEIGHT:
                canvas_height = MAX_CANVAS_HEIGHT
                canvas_width = int(canvas_height * aspect_ratio) if aspect_ratio > 0 else MAX_CANVAS_WIDTH

            # Ensure minimum size
            canvas_width = max(canvas_width, 150)
            canvas_height = max(canvas_height, 150)
            # ---------------------------------------------------------------

            # Retrieve the last drawing state if available
            initial_drawing = st.session_state.get("canvas_drawing", None)

            # --- Pass the PIL Image object directly to background_image ---
            try:
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.2)",  # Semi-transparent orange fill
                    stroke_width=2,
                    stroke_color="rgba(239, 83, 80, 0.8)", # Reddish border
                    background_image=display_img, # *** Pass the PIL Image object ***
                    update_streamlit=True, # Update Streamlit dynamically on drawing
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="rect", # Only allow rectangles
                    initial_drawing=initial_drawing, # Load previous drawing state
                    key="drawable_canvas" # Unique key for the canvas
                )
            except Exception as canvas_error:
                 logger.error(f"Error rendering drawable canvas: {canvas_error}", exc_info=True)
                 st.error(f"Could not display drawing canvas: {canvas_error}. Displaying static image.", icon="🖼️")
                 st.image(display_img, caption="Image Preview (Canvas Error)", use_container_width=True)
                 canvas_result = None # Ensure canvas_result is None if it failed


            # --- Process canvas result (only if canvas rendered successfully) ---
            if canvas_result and canvas_result.json_data is not None:
                if canvas_result.json_data.get("objects"):
                    # Get the last drawn object
                    if canvas_result.json_data["objects"]:
                        last_object = canvas_result.json_data["objects"][-1]
                        if last_object["type"] == "rect":
                            # Extract coordinates from canvas object
                            canvas_left = int(last_object["left"])
                            canvas_top = int(last_object["top"])
                            canvas_width_scaled = int(last_object["width"] * last_object.get("scaleX", 1))
                            canvas_height_scaled = int(last_object["height"] * last_object.get("scaleY", 1))

                            # --- Scale canvas coordinates back to original image dimensions ---
                            scale_x = img_w / canvas_width if canvas_width > 0 else 1
                            scale_y = img_h / canvas_height if canvas_height > 0 else 1
                            original_left = int(canvas_left * scale_x)
                            original_top = int(canvas_top * scale_y)
                            original_width = int(canvas_width_scaled * scale_x)
                            original_height = int(canvas_height_scaled * scale_y)

                            # --- Boundary checks ---
                            original_left = max(0, original_left)
                            original_top = max(0, original_top)
                            original_width = min(img_w - original_left, original_width) if img_w > original_left else 0
                            original_height = min(img_h - original_top, original_height) if img_h > original_top else 0
                            original_width = max(1, original_width) if original_width > 0 else 0 # Ensure min width 1 if possible
                            original_height = max(1, original_height) if original_height > 0 else 0
                            # ------------------------

                            # Only update if width and height are valid
                            if original_width > 0 and original_height > 0:
                                new_roi = {
                                    "left": original_left,
                                    "top": original_top,
                                    "width": original_width,
                                    "height": original_height
                                }
                                # Update session state only if ROI actually changed
                                if st.session_state.roi_coords != new_roi:
                                    st.session_state.roi_coords = new_roi
                                    st.session_state.canvas_drawing = canvas_result.json_data # Save canvas state
                                    logger.info(f"New ROI selected (original image coords): {new_roi}")
                                    st.info(f"ROI Set: Top-Left ({original_left},{original_top}), Size {original_width}x{original_height}", icon="🎯")
                                    # No rerun needed here, actions trigger reruns

                elif not canvas_result.json_data.get("objects"): # Check for empty objects list (user cleared drawing)
                    # If user clears drawing on canvas
                    if st.session_state.roi_coords is not None:
                        logger.info("Canvas drawing cleared by user action, removing ROI state.")
                        st.session_state.roi_coords = None
                        st.session_state.canvas_drawing = None # Clear saved drawing too
                        st.info("ROI cleared from canvas.", icon="🗑️")
                        # No rerun needed here, sidebar clear button handles rerun

        else: # Fallback if canvas library not available or PIL issue
            caption = "Image Preview"
            if not DRAWABLE_CANVAS_AVAILABLE:
                 caption += " (Install streamlit-drawable-canvas for ROI)"
            elif not PIL_AVAILABLE:
                 caption += " (PIL not available for display)"
            st.image(display_img, caption=caption, use_container_width=True)


        # Display current ROI coordinates if set
        if st.session_state.roi_coords:
            roi = st.session_state.roi_coords
            st.caption(f"Active ROI (Image Coords): ({roi['left']}, {roi['top']}), W:{roi['width']}, H:{roi['height']}")

        st.markdown("---")

        # DICOM Metadata Expander
        if st.session_state.is_dicom and st.session_state.dicom_metadata:
            with st.expander("📄 View DICOM Metadata", expanded=False):
                if UI_COMPONENTS_AVAILABLE:
                    display_dicom_metadata(st.session_state.dicom_metadata)
                else:
                    # Basic fallback display if ui_components missing
                    st.json({k: str(v)[:100] + '...' if len(str(v)) > 100 else str(v) for k, v in list(st.session_state.dicom_metadata.items())[:15]}) # Show first 15 keys
        elif st.session_state.is_dicom:
            st.caption("DICOM file loaded, but failed to extract metadata.")

    elif uploaded_file is not None:
        # This state might occur if processing failed after upload but before display obj created
        st.error("Image preview failed. The file might be corrupted or processing failed after upload.", icon="🖼️")
    else:
        st.info("⬅️ Please upload an image (JPG, PNG, DICOM) using the sidebar to begin.")


with col2:
    st.subheader("📊 Gemini AI Analysis & Results")

    # Check if any analysis has been performed yet
    analysis_performed = any([
        st.session_state.initial_analysis,
        st.session_state.qa_answer,
        st.session_state.disease_analysis,
        st.session_state.confidence_score
    ])

    if not isinstance(st.session_state.get("display_image"), Image.Image):
         st.info("Upload an image to enable AI analysis.")
    elif not analysis_performed:
         st.info("Use the sidebar controls to run an AI analysis on the loaded image.")


    tab_titles = [
        "🔬 Initial Observation",
        "💬 Q&A History",
        "🩺 Condition Focus",
        "📈 Confidence Assessment",
        "🌐 Translation"
    ]
    tabs = st.tabs(tab_titles)

    # --- Tab 1: Initial Observation ---
    with tabs[0]:
        st.markdown("**Gemini's General Visual Observation:**")
        if st.session_state.initial_analysis:
             st.markdown(st.session_state.initial_analysis, unsafe_allow_html=False) # Render markdown safely
        else:
             st.caption("_Run 'Initial Visual Observation' from the sidebar to get Gemini's description of the image._")


    # --- Tab 2: Q&A History ---
    with tabs[1]:
        st.markdown("**Latest AI Answer:**")
        if st.session_state.qa_answer:
             st.markdown(st.session_state.qa_answer, unsafe_allow_html=False)
        else:
            st.caption("_Ask a question about the image using the sidebar to see the AI's response here._")

        st.markdown("---")
        if st.session_state.history:
            with st.expander("Full Conversation History", expanded=False):
                # Display history, newest first
                for i, (q_type, message) in enumerate(reversed(st.session_state.history)):
                    cleaned_message = str(message).strip() # Ensure string and strip whitespace
                    if q_type.lower() == "user question":
                        st.markdown(f"**You:** {cleaned_message}")
                    elif q_type.lower() == "ai answer":
                        # Use markdown with proper line breaks for potentially long AI answers
                        st.markdown(f"**AI:**")
                        # Use code block for potentially long/formatted AI answers if markdown rendering is tricky
                        # st.code(cleaned_message, language=None)
                        st.markdown(cleaned_message, unsafe_allow_html=False) # Render AI's markdown response
                    elif q_type.lower() == "system":
                        st.info(f"*{cleaned_message}*", icon="ℹ️") # System messages like ROI cleared
                    else: # Fallback for unexpected types
                        st.markdown(f"**{q_type}:** {cleaned_message}")
                    if i < len(st.session_state.history) - 1: # Add separator between messages
                        st.markdown("---")
        else:
            st.caption("No questions asked yet in this session.")

    # --- Tab 3: Condition Focus ---
    with tabs[2]:
        st.markdown("**Visual Signs Related to Condition Focus:**")
        if st.session_state.disease_analysis:
            st.markdown(st.session_state.disease_analysis, unsafe_allow_html=False)
        else:
             st.caption("_Select a condition in the sidebar and run 'Analyze Visual Signs' to see relevant observations._")


    # --- Tab 4: Confidence Assessment ---
    with tabs[3]:
        st.markdown("**AI's Qualitative Confidence Assessment & Limitations:**")
        if st.session_state.confidence_score:
             st.markdown(st.session_state.confidence_score, unsafe_allow_html=False)
        else:
            st.caption("_Run 'Assess AI Confidence/Limitations' after an analysis to get the AI's reflection on certainty and potential issues._")


    # --- Tab 5: Translation ---
    with tabs[4]:
        st.subheader("🌐 Translate Analysis Text")

        if not TRANSLATION_AVAILABLE:
            st.warning("Translation features are unavailable. Ensure 'deep-translator' and dependencies are installed and `translation_models.py` is present.")
        else:
            st.caption("Select analysis text, choose target language, then click 'Translate'.")
            # Combine all potential analysis text into options for translation
            text_options = {
                "Initial Observation": st.session_state.initial_analysis,
                "Latest Q&A Answer": st.session_state.qa_answer if 'Error' not in st.session_state.qa_answer else "", # Exclude error messages
                "Condition Focus Analysis": st.session_state.disease_analysis,
                "Confidence Assessment": st.session_state.confidence_score,
                "(Enter Custom Text Below)": "" # Option for custom input
            }
            # Filter out options that are empty, except for the custom text option
            available_options = {
                label: txt for label, txt in text_options.items() if (txt and txt.strip()) or label == "(Enter Custom Text Below)"
            }

            if not available_options or all(not v for k, v in available_options.items() if k != "(Enter Custom Text Below)"):
                 st.info("No analysis text available to translate yet.")
            else:
                # Try to find the first non-empty option as default, else default to custom
                default_index = 0
                option_keys = list(available_options.keys())
                try:
                    first_valid_option = next(i for i, k in enumerate(option_keys) if k != "(Enter Custom Text Below)" and available_options[k])
                    default_index = first_valid_option
                except StopIteration:
                    default_index = option_keys.index("(Enter Custom Text Below)") if "(Enter Custom Text Below)" in option_keys else 0


                selected_label = st.selectbox(
                    "Select text to translate:",
                    option_keys,
                    index=default_index,
                    key="translate_source_selector"
                )
                text_to_translate_raw = available_options.get(selected_label, "")

                # Show custom text area only if that option is selected
                if selected_label == "(Enter Custom Text Below)":
                    custom_text = st.text_area(
                        "Enter or paste text to translate here:",
                        value=st.session_state.get("custom_translate_input_val", ""), # Persist custom text briefly
                        height=150,
                        key="custom_translate_input"
                    )
                    text_to_translate = custom_text
                    st.session_state.custom_translate_input_val = custom_text # Save for potential rerun
                else:
                     text_to_translate = text_to_translate_raw
                     if "custom_translate_input_val" in st.session_state:
                         del st.session_state.custom_translate_input_val # Clear if switching away


                # Display the selected text (read-only)
                st.text_area(
                    "Text selected/entered for translation:",
                    value=text_to_translate,
                    height=120, # Adjusted height
                    disabled=True,
                    key="translate_preview_area"
                )

                # Language selection columns
                col_lang1, col_lang2 = st.columns(2)
                with col_lang1:
                    # Source language (Auto-Detect is usually best)
                    source_language_options = [AUTO_DETECT_INDICATOR] + sorted(list(LANGUAGE_CODES.keys()))
                    source_language_name = st.selectbox(
                        "Source Language:",
                        source_language_options,
                        index=0, # Default to Auto-Detect
                        key="source_lang_selector"
                    )
                with col_lang2:
                    # Target language
                    target_language_options = sorted([lang for lang in LANGUAGE_CODES.keys() if lang != source_language_name or source_language_name == AUTO_DETECT_INDICATOR]) # Exclude source if selected
                    # Try to default to English or Spanish if available
                    default_target_index = 0
                    common_targets = ["English", "Spanish"] # Prioritize these common languages
                    try:
                        # Find the first common target present in the options
                        default_target_index = next(i for i, lang in enumerate(target_language_options) if lang in common_targets)
                    except StopIteration:
                        # If neither English nor Spanish is available, just use the first option
                        default_target_index = 0 if target_language_options else -1 # Handle empty options case

                    target_language_name = st.selectbox(
                        "Translate To:",
                        target_language_options,
                        index=default_target_index if default_target_index >= 0 else 0,
                        key="target_lang_selector",
                        disabled=(default_target_index == -1) # Disable if no target options
                    )

                # Translate button
                can_translate = (text_to_translate and text_to_translate.strip() and target_language_name)
                if st.button("🔄 Translate Now", key="translate_button", disabled=not can_translate):
                    st.session_state.translation_result = None # Clear previous result
                    st.session_state.translation_error = None  # Clear previous error

                    if not text_to_translate or not text_to_translate.strip():
                        # This case should be prevented by button disable logic, but check anyway
                        st.warning("Please select or enter some text to translate.")
                        st.session_state.translation_error = "Input text is empty."
                    elif source_language_name == target_language_name and source_language_name != AUTO_DETECT_INDICATOR:
                        st.info("Source and target languages are the same. No translation needed.")
                        st.session_state.translation_result = text_to_translate # Show original text
                    elif translate: # Check if translate function is available
                        with st.spinner(f"Translating from '{source_language_name}' to '{target_language_name}'..."):
                            try:
                                translation_output = translate(
                                    text=text_to_translate,
                                    target_language=target_language_name,
                                    source_language=source_language_name # Pass source name (might be "Auto-Detect")
                                )
                                if translation_output is not None and "[Translation Error:" not in translation_output:
                                    st.session_state.translation_result = translation_output
                                    st.success("Translation complete!")
                                    logger.info(f"Translation successful to {target_language_name}")
                                elif "[Translation Error:" in translation_output:
                                     st.error(f"Translation failed: {translation_output}")
                                     st.session_state.translation_error = translation_output
                                     logger.warning(f"Translation function returned error: {translation_output}")
                                else:
                                    st.error("Translation service returned no result. Please check logs or try again.")
                                    st.session_state.translation_error = "Translation service returned an empty result."
                                    logger.warning("Translation function returned None.")
                            except Exception as e:
                                st.error(f"Translation failed: {e}")
                                logger.error(f"Translation error during API call: {e}", exc_info=True)
                                st.session_state.translation_error = str(e)
                    else:
                         st.error("Translate function is not available.")
                         st.session_state.translation_error = "Translation module not loaded."

                # Display translation result or error
                if st.session_state.get("translation_result"):
                    formatted_result = format_translation(st.session_state.translation_result)
                    st.text_area("Translated Text:", value=formatted_result, height=200, key="translated_output_area")
                elif st.session_state.get("translation_error"):
                    # Show error message prominently if translation failed
                    st.error(f"Translation Error: {st.session_state.translation_error}", icon="❌")


# --- Button Action Handlers ---
current_action = st.session_state.get("last_action")
if current_action:
    logger.info(f"Handling action: '{current_action}' for session: {st.session_state.session_id}")

    # --- Pre-action Checks ---
    action_requires_image = current_action in ["analyze", "ask", "disease", "confidence"]
    action_requires_llm = current_action in ["analyze", "ask", "disease", "confidence"] # All core actions need LLM
    action_requires_report_util = (current_action == "generate_report_data")

    error_occurred = False
    # Ensure PIL Image object exists for actions requiring it
    if action_requires_image and not (PIL_AVAILABLE and isinstance(st.session_state.get("processed_image"), Image.Image)):
        st.error(f"Action '{current_action}' requires a processed image. Please upload a valid image first.", icon="🖼️")
        error_occurred = True
    if not st.session_state.session_id:
        st.error("Critical error: Session ID is missing. Cannot proceed.", icon="🆔")
        error_occurred = True
    if action_requires_llm and not st.session_state.models_initialized:
        st.error("Gemini AI models are not initialized. Check API key and configuration.", icon="🤖")
        error_occurred = True
    if action_requires_report_util and not REPORT_UTILS_AVAILABLE:
        st.error("Report generation utility is not available.", icon="📄")
        error_occurred = True

    if error_occurred:
        st.session_state.last_action = None # Clear action if prerequisites fail
        st.stop() # Stop further processing in this run
    # --- End Pre-action Checks ---


    img_for_llm = st.session_state.processed_image # Use the RGB image for Gemini
    roi_coords = st.session_state.roi_coords
    current_history = st.session_state.history # Get the current history list

    # Ensure history is a list (safety check)
    if not isinstance(current_history, list):
        current_history = []
        st.session_state.history = current_history

    try:
        analysis_result = None
        error_message = None

        # --- Execute Action ---
        if current_action == "analyze":
            st.info("🔬 Requesting initial visual observation from Gemini...")
            with st.spinner("AI analyzing image... This may take a minute."):
                analysis_result, error_message = run_gemini_image_analysis(img_for_llm, roi_coords)
            if analysis_result:
                st.session_state.initial_analysis = analysis_result
                st.session_state.qa_answer = "" # Clear Q&A if new initial analysis is run
                st.session_state.disease_analysis = "" # Clear disease focus too
                logger.info("Initial observation successful.")
                st.success("Initial visual observation complete!")
            else:
                st.error(f"Initial observation failed: {error_message or 'Unknown error from AI.'}", icon="❌")
                logger.error(f"Initial observation failed: {error_message}")

        elif current_action == "ask":
            question_text = st.session_state.question_input_widget.strip()
            if not question_text:
                st.warning("Question is empty. Please enter a question.")
                error_message = "Empty question" # Set internal error message
            else:
                st.info(f"❓ Asking Gemini about the image: '{question_text[:70]}...'")
                st.session_state.qa_answer = "" # Clear previous answer before new request
                with st.spinner("AI thinking... This may take a minute."):
                    analysis_result, error_message = run_gemini_image_qa(
                        img_for_llm,
                        question_text,
                        current_history, # Pass history
                        roi_coords=roi_coords # Pass ROI coords
                    )
                if analysis_result:
                    st.session_state.qa_answer = analysis_result
                    # Add interaction to history only if successful
                    st.session_state.history.append(("User Question", question_text))
                    st.session_state.history.append(("AI Answer", analysis_result))
                    logger.info("Q&A successful.")
                    st.success("AI answered your question!")
                else:
                    # Store the error message as the "answer" for visibility if needed
                    err_msg_display = f"[AI Error: {error_message or 'Unknown error from AI.'}]"
                    st.session_state.qa_answer = err_msg_display
                    # Optionally add error indication to history if needed
                    # st.session_state.history.append(("User Question", question_text))
                    # st.session_state.history.append(("System", f"Error getting AI answer: {error_message}"))
                    st.error(f"Failed to get answer: {error_message or 'Unknown error from AI.'}", icon="❌")
                    logger.error(f"Q&A failed: {error_message}")

        elif current_action == "disease":
            selected_disease = st.session_state.disease_select_widget
            if not selected_disease:
                st.warning("No condition selected. Please select a condition first.")
                error_message = "No condition selected"
            else:
                st.info(f"🩺 Asking Gemini to focus on visual signs related to '{selected_disease}'...")
                with st.spinner(f"AI analyzing for signs of {selected_disease}... This may take a minute."):
                    analysis_result, error_message = run_gemini_condition_analysis(
                        img_for_llm,
                        selected_disease,
                        roi_coords
                        )
                if analysis_result:
                    st.session_state.disease_analysis = analysis_result
                    # Optionally clear Q&A when running disease analysis? Depends on workflow.
                    # st.session_state.qa_answer = ""
                    logger.info(f"Condition focus analysis for '{selected_disease}' successful.")
                    st.success(f"Analysis focusing on '{selected_disease}' complete!")
                else:
                    st.error(f"Condition analysis failed: {error_message or 'Unknown error from AI.'}", icon="❌")
                    logger.error(f"Condition analysis for '{selected_disease}' failed: {error_message}")

        elif current_action == "confidence":
             # Combine previous analyses text to give context for confidence assessment
             # Use the most recent valid analysis as primary input
             primary_analysis = ""
             if st.session_state.disease_analysis: primary_analysis = st.session_state.disease_analysis
             elif st.session_state.qa_answer and 'Error' not in st.session_state.qa_answer: primary_analysis = st.session_state.qa_answer
             elif st.session_state.initial_analysis: primary_analysis = st.session_state.initial_analysis

             if not primary_analysis:
                 st.warning("No prior analysis available to assess confidence.")
                 error_message = "No prior analysis text"
             else:
                st.info("📊 Requesting Gemini's confidence assessment...")
                with st.spinner("AI assessing analysis confidence and limitations..."):
                    analysis_result, error_message = run_gemini_confidence_assessment(
                        img_for_llm,
                        previous_analysis=primary_analysis, # Pass the most relevant analysis text
                        history=current_history, # Pass history for broader context
                        roi_coords=roi_coords
                    )
                if analysis_result:
                    st.session_state.confidence_score = analysis_result # Store the qualitative text
                    logger.info("Confidence assessment successful.")
                    st.success("Confidence assessment complete!")
                else:
                    st.error(f"Confidence assessment failed: {error_message or 'Unknown error from AI.'}", icon="❌")
                    logger.error(f"Confidence assessment failed: {error_message}")

        elif current_action == "generate_report_data":
            st.info("📄 Compiling data for PDF report...")
            st.session_state.pdf_report_bytes = None # Clear previous report data

            image_for_report = st.session_state.get("display_image") # Use the display image (with W/L)
            if not (PIL_AVAILABLE and isinstance(image_for_report, Image.Image)):
                st.error("Cannot generate report: No valid display image found.", icon="🖼️")
                error_message = "No display image for report"
            elif not REPORT_UTILS_AVAILABLE:
                 st.error("Cannot generate report: Reporting utility not available.", icon="📄")
                 error_message = "Reporting utility missing"
            else:
                # --- Prepare data for the PDF ---
                final_image_for_pdf = image_for_report.copy().convert("RGB") # Ensure RGB

                # Draw ROI on the image copy for the report if ROI exists
                if roi_coords:
                    try:
                        draw = ImageDraw.Draw(final_image_for_pdf)
                        x0, y0 = int(roi_coords['left']), int(roi_coords['top'])
                        x1, y1 = x0 + int(roi_coords['width']), y0 + int(roi_coords['height'])
                        # Draw a noticeable rectangle (e.g., red, thickness relative to image size)
                        outline_color = "red"
                        outline_width = max(3, int(min(final_image_for_pdf.size) * 0.005)) # Adjust thickness based on image size
                        draw.rectangle(
                            [x0, y0, x1, y1],
                            outline=outline_color,
                            width=outline_width
                        )
                        logger.info("ROI bounding box drawn on the image for PDF report.")
                    except Exception as e:
                        logger.error(f"Error drawing ROI on PDF image copy: {e}", exc_info=True)
                        st.warning("Could not draw ROI rectangle on the report image.", icon="⚠️")

                # Format conversation history for the report
                formatted_history = "No Q&A interactions in this session."
                if current_history:
                    lines = []
                    for q_type, msg in current_history:
                        # Basic cleaning: remove potential HTML tags just in case
                        cleaned_msg = re.sub('<[^<]+?>', '', str(msg)).strip()
                        # Add bold formatting for types in the report
                        lines.append(f"**{q_type}:**\n{cleaned_msg}")
                    formatted_history = "\n\n".join(lines)

                # Consolidate all analysis outputs
                report_data = {
                    "Session ID": st.session_state.session_id,
                    "Image Filename": (st.session_state.uploaded_file_info or "N/A").split('-')[0], # Extract original filename
                    "Initial Visual Observation": st.session_state.initial_analysis or "Not Performed",
                    "Condition Focused Analysis": st.session_state.disease_analysis or "Not Performed",
                     "AI Confidence Assessment": st.session_state.confidence_score or "Not Performed",
                    "Conversation History": formatted_history, # Q&A consolidated here
                }

                # Add DICOM summary if available
                dicom_summary_for_report = None
                if st.session_state.is_dicom and st.session_state.dicom_metadata:
                    # Select key metadata fields for the report summary
                    keys_for_summary = [
                        'PatientName', 'PatientID', 'StudyDate', 'StudyTime', # DICOM keywords often lack spaces
                        'Modality', 'StudyDescription', 'SeriesDescription',
                        'Manufacturer', 'ManufacturerModelName'
                        ]
                    # Ensure values are strings for the report generator if it expects them
                    # Match exact DICOM keywords which might not have spaces
                    meta_summary = {k: str(st.session_state.dicom_metadata.get(k, ''))
                                    for k in keys_for_summary if st.session_state.dicom_metadata.get(k)} # Only include if key exists and has value
                    if meta_summary:
                        # Store the dictionary itself, let report_utils handle formatting
                        dicom_summary_for_report = meta_summary
                        report_data["DICOM Summary"] = meta_summary # Add to main data


                # --- Generate PDF Bytes ---
                with st.spinner("Generating PDF report..."):
                    pdf_bytes = generate_pdf_report_bytes(
                        session_id=st.session_state.session_id,
                        image=final_image_for_pdf, # Pass the image with ROI drawn
                        analysis_outputs=report_data, # Pass the consolidated data
                        dicom_metadata=st.session_state.dicom_metadata if st.session_state.is_dicom else None # Pass full metadata if needed by template
                    )

                if pdf_bytes:
                    st.session_state.pdf_report_bytes = pdf_bytes
                    st.success("PDF report data generated! Download button available in the sidebar.", icon="📄")
                    logger.info("PDF report generated successfully.")
                    st.balloons() # Celebrate!
                else:
                    st.error("Failed to generate PDF report data. The report utility might have encountered an error or is a dummy. Check logs.", icon="❌")
                    logger.error("PDF generation function returned None or empty bytes.")
                    error_message = "PDF generation failed"

        else:
            st.warning(f"Unknown action '{current_action}' triggered. No operation performed.")
            error_message = "Unknown action"

    except Exception as e:
        # Catch-all for unexpected errors during action execution
        st.error(f"An unexpected error occurred during action '{current_action}': {e}", icon="🔥")
        logger.critical(f"Critical error during action '{current_action}': {e}", exc_info=True)
        error_message = f"Unexpected error: {e}" # Store error message

    finally:
        # --- Post-action ---
        st.session_state.last_action = None # IMPORTANT: Clear the action flag
        logger.debug(f"Action '{current_action}' handling complete. Result: {'Success' if not error_message else 'Failed'}. Error: {error_message}")
        # Rerun Streamlit to update the UI reflecting the results (or errors)
        st.rerun()


# --- Footer ---
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced | Session ID: {st.session_state.get('session_id', 'N/A')}")
st.markdown(
    """
    <footer>
      <p>RadVision AI is for informational and educational purposes only. It is not a medical device and does not provide medical advice or diagnosis.</p>
      <p>Always consult qualified healthcare professionals. Output must be clinically correlated.</p>
      <!-- <p><a href="#" target="_blank">Privacy Policy</a> | <a href="#" target="_blank">Terms of Service</a></p> -->
    </footer>
    """,
    unsafe_allow_html=True
)
logger.info(f"--- Application render complete for session: {st.session_state.session_id} ---")

# --- End of Script ---
