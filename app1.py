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
        logger.info(f"Google Generative AI SDK Version: {genai.__version__}")
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
        st.error("⚠️ Gemini API Key not found. Please configure GEMINI_API_KEY in Streamlit secrets or environment variables.", icon="🔑")
        logger.critical("GEMINI_API_KEY not found.")
        st.stop()

# Initialize models using Session State
VISION_MODEL_NAME = 'gemini-1.5-flash'

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
    st.error("AI Models could not be initialized due to configuration issues.", icon="🚫")
    st.stop()

# --- Gemini Prompt Templates ---
# (Keep the existing IMAGE_ANALYSIS_PROMPT_TEMPLATE and CONFIDENCE_ASSESSMENT_PROMPT_TEMPLATE)
IMAGE_ANALYSIS_PROMPT_TEMPLATE = """
**Medical Image Analysis Request (Gemini Vision Model)**
... (template content unchanged) ...
"""
CONFIDENCE_ASSESSMENT_PROMPT_TEMPLATE = """
**AI Confidence Assessment Request (Gemini Vision Model)**
... (template content unchanged) ...
"""

# --- Gemini Interaction Functions ---
# (Keep the existing interaction functions - they were correct)
def generate_roi_description(roi_coords: Optional[Dict]) -> str:
    """Generates a textual description of the ROI for the prompt."""
    # ... (function content unchanged) ...
    if not roi_coords:
        return "N/A"
    try:
        left = int(roi_coords['left'])
        top = int(roi_coords['top'])
        width = int(roi_coords['width'])
        height = int(roi_coords['height'])
        return (f"User highlighted a Region of Interest (ROI) at "
                f"Top-Left corner ({left},{top}) with "
                f"Width={width}px, Height={height}px. "
                f"Focus on findings within or related to this region, while maintaining overall context.")
    except (KeyError, TypeError, ValueError) as e:
        logger.warning(f"Could not format ROI coords {roi_coords}: {e}")
        return "ROI specified, but coordinates unclear."

def handle_gemini_response(response: Any) -> Tuple[Optional[str], Optional[str]]:
    """Safely extracts text from Gemini response or returns error."""
    # ... (function content unchanged) ...
    try:
        if hasattr(response, 'text'):
            final_text = response.text
            if "Mandatory Disclaimer" not in final_text and "NOT** a radiological interpretation" not in final_text:
                 final_text += "\n\n**## Mandatory Disclaimer:**\nThis is an AI-generated visual observation intended for informational and demonstration purposes ONLY. It is **NOT** a radiological interpretation or medical diagnosis. It **CANNOT** substitute for a comprehensive evaluation and interpretation by a qualified radiologist or physician integrating full clinical information. Any potential observations noted herein **MUST** be correlated with clinical findings and reviewed/confirmed by qualified healthcare professionals."
            return final_text, None
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
            reason = response.prompt_feedback.block_reason.name
            logger.warning(f"Gemini analysis blocked by safety filters: {reason}")
            return None, f"Analysis blocked by safety filters: {reason}. This might relate to sensitive content policies or image characteristics. Please review input or contact support."
        elif hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason') and candidate.finish_reason != "STOP":
                 reason = candidate.finish_reason.name
                 logger.warning(f"Gemini analysis stopped prematurely: {reason}")
                 return None, f"Analysis stopped prematurely. Reason: {reason}. Input might be too long, complex, or triggered other limits."
            elif hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                 return candidate.content.parts[0].text, None # Assuming first part is the text
            else:
                 logger.warning("Received an empty or unexpected response structure from Gemini candidate.")
                 return None, "Received an empty or unexpected response from the AI model candidate."
        else:
            logger.warning(f"Received an unexpected response object from Gemini: {type(response)}")
            return None, "Received an unexpected response structure from the AI model."

    except AttributeError as e:
        logger.error(f"Error parsing Gemini response: {e}. Response object: {response}", exc_info=True)
        return None, f"Internal error parsing AI response structure: {e}"
    except Exception as e:
        logger.error(f"Unexpected error handling Gemini response: {e}", exc_info=True)
        return None, f"Unexpected error processing AI response: {e}"

def run_gemini_image_analysis(image: Image.Image, roi_coords: Optional[Dict] = None) -> Tuple[Optional[str], Optional[str]]:
    """Performs initial visual observation using Gemini Vision."""
    # ... (function content unchanged) ...
    if not st.session_state.models_initialized or not st.session_state.vision_model: return None, "Vision model not initialized."
    if not isinstance(image, Image.Image): return None, "Invalid image provided for analysis."
    try:
        if image.mode != 'RGB': image = image.convert('RGB')
        roi_desc = generate_roi_description(roi_coords)
        prompt = IMAGE_ANALYSIS_PROMPT_TEMPLATE.format(user_prompt="N/A - Perform general visual observation.", roi_description=roi_desc)
        model_input = [prompt, image]
        logger.info("Sending request to Gemini for initial image analysis...")
        response = st.session_state.vision_model.generate_content(model_input)
        logger.info("Received response from Gemini for initial analysis.")
        return handle_gemini_response(response)
    except Exception as e:
        logger.error(f"ERROR in run_gemini_image_analysis: {e}", exc_info=True)
        return None, f"An internal error occurred: {e}"

def run_gemini_image_qa(image: Image.Image, question: str, history: List, roi_coords: Optional[Dict] = None) -> Tuple[Optional[str], Optional[str]]:
    """Answers a question about the image using Gemini Vision."""
    # ... (function content unchanged) ...
    if not st.session_state.models_initialized or not st.session_state.vision_model: return None, "Vision model not initialized."
    if not isinstance(image, Image.Image): return None, "Invalid image provided for Q&A."
    if not question or not question.strip(): return None, "Question cannot be empty."
    try:
        if image.mode != 'RGB': image = image.convert('RGB')
        roi_desc = generate_roi_description(roi_coords)
        prompt = IMAGE_ANALYSIS_PROMPT_TEMPLATE.format(user_prompt=question.strip(), roi_description=roi_desc)
        model_input = [prompt, image]
        logger.info(f"Sending request to Gemini for Q&A: '{question[:50]}...'")
        response = st.session_state.vision_model.generate_content(model_input)
        logger.info("Received response from Gemini for Q&A.")
        return handle_gemini_response(response)
    except Exception as e:
        logger.error(f"ERROR in run_gemini_image_qa: {e}", exc_info=True)
        return None, f"An internal error occurred: {e}"

def run_gemini_condition_analysis(image: Image.Image, condition: str, roi_coords: Optional[Dict] = None) -> Tuple[Optional[str], Optional[str]]:
    """Analyzes image focusing on a specific condition using Gemini Vision."""
    # ... (function content unchanged) ...
    if not st.session_state.models_initialized or not st.session_state.vision_model: return None, "Vision model not initialized."
    if not isinstance(image, Image.Image): return None, "Invalid image provided for condition analysis."
    if not condition or not condition.strip(): return None, "Condition cannot be empty."
    try:
        if image.mode != 'RGB': image = image.convert('RGB')
        roi_desc = generate_roi_description(roi_coords)
        condition_prompt = (f"Focus the visual observation specifically on findings potentially related to **{condition}**. "
                            f"Describe any relevant visual signs using objective, descriptive language as outlined in the main prompt instructions. "
                            f"State if no specific signs related to {condition} are visually apparent.")
        prompt = IMAGE_ANALYSIS_PROMPT_TEMPLATE.format(user_prompt=condition_prompt, roi_description=roi_desc)
        model_input = [prompt, image]
        logger.info(f"Sending request to Gemini for condition analysis: '{condition}'")
        response = st.session_state.vision_model.generate_content(model_input)
        logger.info("Received response from Gemini for condition analysis.")
        return handle_gemini_response(response)
    except Exception as e:
        logger.error(f"ERROR in run_gemini_condition_analysis: {e}", exc_info=True)
        return None, f"An internal error occurred: {e}"

def run_gemini_confidence_assessment(image: Image.Image, previous_analysis: str, history: List, roi_coords: Optional[Dict] = None) -> Tuple[Optional[str], Optional[str]]:
    """Assesses qualitative confidence of previous analysis using Gemini Vision."""
    # ... (function content unchanged) ...
    if not st.session_state.models_initialized or not st.session_state.vision_model: return None, "Vision model not initialized."
    if not isinstance(image, Image.Image): return None, "Invalid image provided for confidence assessment."
    if not previous_analysis or not previous_analysis.strip(): return None, "No previous analysis text provided to assess confidence."
    try:
        if image.mode != 'RGB': image = image.convert('RGB')
        roi_desc = generate_roi_description(roi_coords)
        history_summary = "\n".join([f"{role}: {text}" for role, text in history[-4:]]) # Last 4 interactions
        prompt = CONFIDENCE_ASSESSMENT_PROMPT_TEMPLATE.format(roi_description=roi_desc, previous_analysis=previous_analysis.strip(), history_summary=history_summary or "N/A")
        model_input = [prompt, image]
        logger.info("Sending request to Gemini for confidence assessment...")
        response = st.session_state.vision_model.generate_content(model_input)
        logger.info("Received response from Gemini for confidence assessment.")
        result_text, error = handle_gemini_response(response)
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
      /* (Keep existing CSS rules) */
      ...
    </style>
    """,
    unsafe_allow_html=True
)

# --- Display Hero Logo ---
logo_path = os.path.join("assets", "radvisionai-hero.jpeg")
if os.path.exists(logo_path):
    st.image(logo_path, width=350)
else:
    logger.warning(f"Hero logo not found at: {logo_path}")

# --- Initialize Session State Defaults ---
DEFAULT_STATE = {
    "uploaded_file_info": None, "raw_image_bytes": None, "is_dicom": False,
    "dicom_dataset": None, "dicom_metadata": {}, "processed_image": None,
    "display_image": None, "session_id": None, "history": [],
    "initial_analysis": "", "qa_answer": "", "disease_analysis": "",
    "confidence_score": "", "last_action": None, "pdf_report_bytes": None,
    "canvas_drawing": None, "roi_coords": None, "current_display_wc": None,
    "current_display_ww": None, "clear_roi_feedback": False, "demo_loaded": False,
    "translation_result": None, "translation_error": None,
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value
if not isinstance(st.session_state.get("history", []), list): st.session_state.history = []
if not st.session_state.get("session_id"): st.session_state.session_id = str(uuid.uuid4())[:8]
logger.debug(f"Session state verified/initialized for session ID: {st.session_state.session_id}")

# --- Helper Functions ---
def format_translation(translated_text: Optional[str]) -> str:
    """Applies basic formatting to translated text."""
    # ... (function content unchanged) ...
    if translated_text is None: return "Translation not available or failed."
    try:
        text_str = str(translated_text)
        formatted_text = re.sub(r'(?<=\S)\s+(\d+\.\s)', r'\n\n\1', text_str)
        formatted_text = re.sub(r'\n*##\s*(\d+)\.\s*(.*)', r'\n\n## \1. \2\n', formatted_text)
        return formatted_text.strip()
    except Exception as e:
        logger.error(f"Error formatting translation: {e}", exc_info=True)
        return str(translated_text)

# --- Monkey-Patch (Conditional for older Streamlit versions) ---
import streamlit.elements.image as st_image # Import the image module
if not hasattr(st_image, "image_to_url"):
    logger.info("Streamlit version appears older. Applying monkey-patch for 'image_to_url'.")
    def image_to_url_monkey_patch(...): # Keep the full patch function definition here
        # ... (full function content unchanged) ...
        if PIL_AVAILABLE and isinstance(image, Image.Image):
            img_pil = image
            if channels == "BGR":
                 try:
                     import numpy as np
                     img_pil = Image.fromarray(np.array(img_pil)[:, :, ::-1])
                 except ImportError: logger.error("NumPy needed for BGR conversion..."); return ""
                 except Exception as e: logger.error(f"Error during BGR conversion...: {e}"); return ""
            if img_pil.mode not in ["RGB", "RGBA", "L"]:
                 logger.warning(f"Converting image mode {img_pil.mode} to RGB for URL generation.")
                 img_pil = img_pil.convert("RGB")
            format = output_format.upper();
            if format == "AUTO": format = "PNG"
            elif format not in ["PNG", "JPEG", "JPG"]: format = "PNG"; logger.warning(f"Unsupported output format '{format}'. Defaulting to PNG.")
            if format == "JPG": format = "JPEG"
            try:
                buffered = io.BytesIO(); img_pil.save(buffered, format=format)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/{format.lower()};base64,{img_str}"
            except Exception as e: logger.error(f"Failed to convert PIL Image to data URL: {e}", exc_info=True); return ""
        elif 'numpy' in sys.modules and isinstance(image, sys.modules['numpy'].ndarray):
             try:
                 import numpy as np; img_np = image
                 if channels == "BGR":
                     if img_np.ndim == 3 and img_np.shape[2] == 3: img_np = img_np[..., ::-1]
                     else: logger.warning("Received numpy array and channels='BGR' but shape not typical BGR.")
                 img_pil_from_np = Image.fromarray(np.uint8(img_np))
                 return image_to_url_monkey_patch(img_pil_from_np, width, clamp, "RGB", output_format, image_id)
             except ImportError: logger.error("Numpy required..."); return ""
             except Exception as e: logger.error(f"Failed to convert Numpy array...: {e}", exc_info=True); return ""
        else: logger.error(f"Unsupported image type...: {type(image)}"); return ""
    st_image.image_to_url = image_to_url_monkey_patch
    logger.info("Successfully applied monkey-patch for 'st.elements.image.image_to_url'.")
else:
    logger.debug("Streamlit version has 'image_to_url' natively. No monkey-patch needed.")
# --- End of Monkey-Patch Section ---

# --- Sidebar ---
with st.sidebar:
    # ... (Sidebar content unchanged - buttons, uploaders etc.) ...
    st.header("⚕️ RadVision Controls (Gemini)")
    st.markdown("---")
    TIPS = ["Tip: ...", ...] # Keep tips
    st.info(f"💡 {random.choice(TIPS)}")
    st.markdown("---")
    st.header("Image Upload & Settings")
    uploaded_file = st.file_uploader(...)
    if st.button("🗑️ Clear ROI", ...): ...
    if st.session_state.get("clear_roi_feedback"): ...
    if st.session_state.is_dicom and UI_COMPONENTS_AVAILABLE and st.session_state.display_image: ... # DICOM W/L
    st.markdown("---")
    st.header("🤖 Gemini AI Actions")
    action_disabled = ...
    if st.button("▶️ Run Initial Visual Observation", ...): ...
    st.subheader("❓ Ask AI About Image")
    question_input = st.text_area(...)
    if st.button("💬 Ask Gemini", ...): ...
    st.subheader("🎯 Focus on Potential Condition Signs")
    DISEASE_OPTIONS = [...]
    disease_select = st.selectbox(...)
    if st.button("🩺 Analyze Visual Signs for Condition", ...): ...
    st.markdown("---")
    st.header("📊 Assessment & Reporting")
    prior_analysis_exists = ...
    can_estimate = ...
    if st.button("📈 Assess AI Confidence/Limitations", ...): ...
    report_generation_disabled = ...
    if st.button("📄 Generate PDF Report Data", ...): ...
    if st.session_state.get("pdf_report_bytes"): st.download_button(...)


# --- File Upload Logic ---
if uploaded_file is not None:
    # ... (File upload and processing logic unchanged) ...
    try: # Hash generation
        uploaded_file.seek(0); file_content_hash = hashlib.sha256(uploaded_file.read()).hexdigest()[:16]; uploaded_file.seek(0)
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{file_content_hash}"
    except Exception as e: logger.warning(...); new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{uuid.uuid4().hex[:8]}"
    if new_file_info != st.session_state.get("uploaded_file_info"):
        logger.info(f"New file uploaded: {uploaded_file.name}...")
        st.toast(...)
        st.session_state.session_id = str(uuid.uuid4())[:8]
        logger.info(f"Resetting session state... new Session ID: {st.session_state.session_id}")
        keys_to_preserve = ...
        for key, value in DEFAULT_STATE.items(): ... # Reset state
        st.session_state.uploaded_file_info = new_file_info
        st.session_state.demo_loaded = False
        st.session_state.raw_image_bytes = uploaded_file.getvalue()
        file_ext = ...; st.session_state.is_dicom = ...
        with st.spinner("🔬 Analyzing file format..."):
            temp_display_img = None; temp_processed_img = None
            processing_success = False; error_msg = None
            if st.session_state.is_dicom:
                logger.info("Attempting DICOM...")
                try: # DICOM processing
                    dicom_dataset = ...; st.session_state.dicom_dataset = ...
                    if dicom_dataset:
                        st.session_state.dicom_metadata = ...
                        default_wc, default_ww = ...; st.session_state.current_display_wc = ...; st.session_state.current_display_ww = ...
                        logger.info(...)
                        temp_display_img = dicom_to_image(dicom_dataset, wc=default_wc, ww=default_ww)
                        temp_processed_img = dicom_to_image(dicom_dataset, wc=None, ww=None, normalize=True)
                        if isinstance(temp_display_img, Image.Image) and isinstance(temp_processed_img, Image.Image):
                            if temp_display_img.mode != 'RGB': temp_display_img = temp_display_img.convert('RGB')
                            if temp_processed_img.mode != 'RGB': temp_processed_img = temp_processed_img.convert('RGB')
                            processing_success = True; logger.info("DICOM parsed successfully.")
                        else: error_msg = ...; logger.error(...)
                    else: error_msg = ...; logger.error(...)
                except (pydicom.errors.InvalidDicomError, Exception) as e:
                    error_msg = ...; logger.warning(...); st.session_state.is_dicom = False
            if not processing_success:
                logger.info("Attempting standard image...")
                st.session_state.is_dicom = False; st.session_state.dicom_dataset = None; st.session_state.dicom_metadata = {}
                if not PIL_AVAILABLE: error_msg = ...; logger.critical(...)
                else:
                    try: # Standard image processing
                        raw_img = Image.open(...); processed_img = raw_img.convert("RGB")
                        temp_display_img = processed_img.copy(); temp_processed_img = processed_img.copy()
                        processing_success = True; logger.info(...)
                    except UnidentifiedImageError: error_msg = ...; logger.error(...)
                    except Exception as e: error_msg = ...; logger.error(...)
            if processing_success and isinstance(temp_display_img, Image.Image) and isinstance(temp_processed_img, Image.Image):
                st.session_state.display_image = temp_display_img; st.session_state.processed_image = temp_processed_img
                st.session_state.roi_coords = None; st.session_state.canvas_drawing = None
                st.success(...) ; logger.info(...)
                st.rerun()
            else: # Processing failed cleanup
                st.session_state.uploaded_file_info = None; st.session_state.display_image = None; ...
                st.error(...); logger.error(...)


# --- Main Page ---
st.markdown("---")
st.title("⚕️ RadVision AI Advanced (Powered by Google Gemini)")

# --- CRITICAL DISCLAIMER ---
st.warning(
    """
    **🔴 IMPORTANT SAFETY & USE DISCLAIMER 🔴**
    ... (Disclaimer content unchanged) ...
    """,
    icon="⚠️"
)
st.markdown("---")

# --- Main Layout ---
col1, col2 = st.columns([2, 3])

# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# THIS BLOCK CONTAINS THE FIX
with col1:
    st.subheader("🖼️ Image Viewer")
    display_img = st.session_state.get("display_image")

    if isinstance(display_img, Image.Image):
        if DRAWABLE_CANVAS_AVAILABLE and st_canvas:
            st.caption("Draw a rectangle below to define a Region of Interest (ROI).")
            # --- Calculate canvas dimensions ---
            MAX_CANVAS_WIDTH = 600; MAX_CANVAS_HEIGHT = 500
            img_w, img_h = display_img.size
            canvas_width = MAX_CANVAS_WIDTH
            canvas_height = int(canvas_width / (img_w / img_h)) if img_w > 0 and img_h > 0 else MAX_CANVAS_HEIGHT
            if canvas_height > MAX_CANVAS_HEIGHT:
                canvas_height = MAX_CANVAS_HEIGHT
                canvas_width = int(canvas_height * (img_w / img_h)) if img_w > 0 and img_h > 0 else MAX_CANVAS_WIDTH
            canvas_width = max(canvas_width, 150); canvas_height = max(canvas_height, 150)
            # ----------------------------------

            initial_drawing = st.session_state.get("canvas_drawing", None)

            # --- CORRECT FIX: Pass the PIL Image object directly ---
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.2)",
                stroke_width=2,
                stroke_color="rgba(239, 83, 80, 0.8)",
                background_image=display_img, # *** Pass the PIL Image object ***
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="rect",
                initial_drawing=initial_drawing,
                key="drawable_canvas"
            )

            # --- Process canvas result ---
            if canvas_result.json_data is not None and canvas_result.json_data.get("objects"):
                if canvas_result.json_data["objects"]:
                    last_object = canvas_result.json_data["objects"][-1]
                    if last_object["type"] == "rect":
                        canvas_left = int(last_object["left"])
                        canvas_top = int(last_object["top"])
                        canvas_width_scaled = int(last_object["width"] * last_object.get("scaleX", 1))
                        canvas_height_scaled = int(last_object["height"] * last_object.get("scaleY", 1))
                        scale_x = img_w / canvas_width; scale_y = img_h / canvas_height
                        original_left = int(canvas_left * scale_x); original_top = int(canvas_top * scale_y)
                        original_width = int(canvas_width_scaled * scale_x); original_height = int(canvas_height_scaled * scale_y)
                        original_left = max(0, original_left); original_top = max(0, original_top)
                        original_width = min(img_w - original_left, original_width); original_height = min(img_h - original_top, original_height)
                        original_width = max(1, original_width); original_height = max(1, original_height)
                        new_roi = {"left": original_left, "top": original_top, "width": original_width, "height": original_height}
                        if st.session_state.roi_coords != new_roi and original_width > 0 and original_height > 0:
                            st.session_state.roi_coords = new_roi
                            st.session_state.canvas_drawing = canvas_result.json_data
                            logger.info(f"New ROI selected (original image coords): {new_roi}")
                            st.info(f"ROI Set: Top-Left ({original_left},{original_top}), Size {original_width}x{original_height}", icon="🎯")
            elif canvas_result.json_data is not None and not canvas_result.json_data.get("objects"):
                 if st.session_state.roi_coords is not None:
                     logger.info("Canvas drawing cleared by user, removing ROI state.")
                     st.session_state.roi_coords = None; st.session_state.canvas_drawing = None
                     st.info("ROI cleared from canvas.", icon="🗑️")

        else: # Fallback if canvas library not available
            st.image(display_img, caption="Image Preview", use_container_width=True)
            if not DRAWABLE_CANVAS_AVAILABLE: st.warning("Drawable Canvas not available...")

        if st.session_state.roi_coords: # Display ROI coords
            roi = st.session_state.roi_coords
            st.caption(f"Active ROI (Image Coords): ({roi['left']}, {roi['top']}), W:{roi['width']}, H:{roi['height']}")

        st.markdown("---")

        if st.session_state.is_dicom and st.session_state.dicom_metadata: # DICOM Metadata
            with st.expander("📄 DICOM Metadata", expanded=False):
                if UI_COMPONENTS_AVAILABLE: display_dicom_metadata(...)
                else: st.json(...)
        elif st.session_state.is_dicom: st.caption("DICOM file loaded, metadata extraction failed.")

    elif uploaded_file is not None: st.error("Image preview failed...")
    else: st.info("⬅️ Please upload an image...")
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

with col2:
    # ... (Tabs and results display logic unchanged) ...
    st.subheader("📊 Gemini AI Analysis & Results")
    tab_titles = ["🔬 Initial Observation", "💬 Q&A History", "🩺 Condition Focus", "📈 Confidence Assessment", "🌐 Translation"]
    tabs = st.tabs(tab_titles)
    with tabs[0]: st.markdown("**Gemini's General Visual Observation:**"); st.markdown(st.session_state.initial_analysis or "...", unsafe_allow_html=False)
    with tabs[1]: # Q&A History
        st.markdown("**Latest AI Answer:**"); st.markdown(st.session_state.qa_answer or "...", unsafe_allow_html=False); st.markdown("---")
        if st.session_state.history:
            with st.expander("Full Conversation History", expanded=True):
                for i, (q_type, message) in enumerate(reversed(st.session_state.history)): ... # History display loop
        else: st.caption("No questions asked yet...")
    with tabs[2]: st.markdown("**Visual Signs Related to Condition Focus:**"); st.markdown(st.session_state.disease_analysis or "...", unsafe_allow_html=False)
    with tabs[3]: st.markdown("**AI's Qualitative Confidence Assessment & Limitations:**"); st.markdown(st.session_state.confidence_score or "...", unsafe_allow_html=False)
    with tabs[4]: # Translation Tab
        st.subheader("🌐 Translate Analysis Text")
        if not TRANSLATION_AVAILABLE: st.warning("Translation features unavailable...")
        else:
            st.caption("Select analysis text...")
            text_options = {...}; available_options = {...}
            if not available_options: st.info("No analysis text available...")
            else:
                selected_label = st.selectbox(...)
                text_to_translate_raw = ...
                if selected_label == "(Enter Custom Text Below)": custom_text = st.text_area(...); text_to_translate = custom_text
                else: text_to_translate = text_to_translate_raw
                st.text_area("Text selected/entered...", value=text_to_translate, ..., disabled=True)
                col_lang1, col_lang2 = st.columns(2)
                with col_lang1: source_language_name = st.selectbox(...)
                with col_lang2: target_language_options = ...; default_target_index = ...; target_language_name = st.selectbox(...)
                if st.button("🔄 Translate Now", ...):
                    st.session_state.translation_result = None; st.session_state.translation_error = None
                    if not text_to_translate or not text_to_translate.strip(): st.warning(...); st.session_state.translation_error = ...
                    elif source_language_name == target_language_name and ...: st.info(...); st.session_state.translation_result = ...
                    elif translate:
                        with st.spinner(...):
                            try: # Translation call
                                translation_output = translate(...)
                                if translation_output is not None: st.session_state.translation_result = ...; st.success(...); logger.info(...)
                                else: st.error(...); st.session_state.translation_error = ...; logger.warning(...)
                            except Exception as e: st.error(...); logger.error(...); st.session_state.translation_error = ...
                    else: st.error(...); st.session_state.translation_error = ...
                if st.session_state.get("translation_result"): formatted_result = format_translation(...); st.text_area("Translated Text:", ...)
                elif st.session_state.get("translation_error"): st.error(...)


# --- Button Action Handlers ---
current_action = st.session_state.get("last_action")
if current_action:
    # ... (Button action handling logic unchanged - pre-checks, action execution, post-action) ...
    logger.info(f"Handling action: '{current_action}'...")
    action_requires_image = ...; action_requires_llm = ...; action_requires_report_util = ...
    error_occurred = False
    if action_requires_image and not isinstance(...): error_occurred = True; st.error(...)
    if not st.session_state.session_id: error_occurred = True; st.error(...)
    if action_requires_llm and not st.session_state.models_initialized: error_occurred = True; st.error(...)
    if action_requires_report_util and not REPORT_UTILS_AVAILABLE: error_occurred = True; st.error(...)
    if error_occurred: st.session_state.last_action = None; st.stop()
    img_for_llm = ...; roi_coords = ...; current_history = ...
    if not isinstance(current_history, list): current_history = []; st.session_state.history = current_history
    try:
        analysis_result = None; error_message = None
        if current_action == "analyze": ...
        elif current_action == "ask": ...
        elif current_action == "disease": ...
        elif current_action == "confidence": ...
        elif current_action == "generate_report_data": ...
        else: st.warning(...); error_message = ...
    except Exception as e: st.error(...); logger.critical(...); error_message = ...
    finally:
        st.session_state.last_action = None
        logger.debug(...)
        st.rerun()

# --- Footer ---
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced (Gemini) | Session ID: {st.session_state.get('session_id', 'N/A')}")
st.markdown(
    """
    <footer>
      <p>RadVision AI is for informational and educational purposes only...</p>
      <p>Always consult qualified healthcare professionals...</p>
    </footer>
    """,
    unsafe_allow_html=True
)
logger.info(f"--- Application render complete for session: {st.session_state.session_id} ---")

# --- End of Script ---