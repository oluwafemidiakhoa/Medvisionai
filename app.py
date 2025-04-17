# -*- coding: utf-8 -*-
"""
app.py - Main Streamlit application for RadVision AI Advanced.

Handles image uploading (DICOM, JPG, PNG), display, ROI selection,
interaction with AI models for analysis and Q&A, translation,
UMLS lookup, and report generation.

IMPORTANT CHANGES:
- Ensured 'deep-translator' is installed on-the-fly if needed.
- Corrected session state management during file uploads to avoid errors.
- Integrated UMLS lookup functionality using umls_utils.py and ui_components.py.
- Added CSS to improve text contrast in sidebar input elements.
- Enhanced logging and error feedback for image loading.
"""

import streamlit as st

# --- Ensure page config is set early ---
st.set_page_config(
    page_title="RadVision AI Advanced",
    layout="wide",  # Use full width
    page_icon="⚕️", # Favicon
    initial_sidebar_state="expanded"  # Keep sidebar open initially
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

# --- Ensure deep-translator is installed at runtime if not present ---
try:
    from deep_translator import GoogleTranslator
except ImportError:
    st.warning("Trying to install 'deep-translator' for translation features...")
    try:
        # Use "--disable-pip-version-check" and "--quiet" for cleaner output
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--quiet",
            "--disable-pip-version-check", "deep-translator"
        ])
        from deep_translator import GoogleTranslator
        st.success("'deep-translator' installed successfully.")
        # Consider adding a st.rerun() here if installation is critical for the first run
    except Exception as e:
        print(f"CRITICAL: Could not install deep-translator: {e}")
        st.error(f"Failed to install 'deep-translator'. Translation disabled. Error: {e}")


# --- Logging Setup (Early) ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
log_format = '%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'
# Prevent duplicate handlers in Streamlit reruns
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=LOG_LEVEL,
    format=log_format,
    datefmt=date_format,
    stream=sys.stdout # Log to stdout for cloud environments
)
logger = logging.getLogger(__name__)
logger.info("--- RadVision AI Application Start ---")
logger.info(f"Streamlit Version: {st.__version__}")
logger.info(f"Logging Level: {LOG_LEVEL}")


# --- Streamlit Drawable Canvas ---
try:
    from streamlit_drawable_canvas import st_canvas
    import streamlit_drawable_canvas as st_canvas_module
    CANVAS_VERSION = getattr(st_canvas_module, '__version__', 'Unknown')
    logger.info(f"Streamlit Drawable Canvas Version: {CANVAS_VERSION}")
    DRAWABLE_CANVAS_AVAILABLE = True
except ImportError:
    st.error("CRITICAL ERROR: streamlit-drawable-canvas is not installed. Functionality impaired. Run `pip install streamlit-drawable-canvas`.")
    logger.critical("streamlit-drawable-canvas not found. App functionality impaired.")
    DRAWABLE_CANVAS_AVAILABLE = False
    st_canvas = None

# --- Pillow (PIL) ---
try:
    from PIL import Image, ImageDraw, UnidentifiedImageError
    import PIL
    PIL_VERSION = getattr(PIL, '__version__', 'Unknown')
    logger.info(f"Pillow (PIL) Version: {PIL_VERSION}")
    PIL_AVAILABLE = True
except ImportError:
    st.error("CRITICAL ERROR: Pillow (PIL) is not installed. Run `pip install Pillow`.")
    logger.critical("Pillow (PIL) not found. App functionality severely impaired.")
    PIL_AVAILABLE = False
    Image = None
    ImageDraw = None
    UnidentifiedImageError = Exception
    st.stop()

# --- pydicom & DICOM libraries ---
try:
    import pydicom
    import pydicom.errors
    import pydicom.valuerep
    # Required for pixel data handling if numpy is used within dicom_utils
    import numpy as np
    PYDICOM_VERSION = getattr(pydicom, '__version__', 'Unknown')
    logger.info(f"Pydicom Version: {PYDICOM_VERSION}")
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_VERSION = 'Not Installed'
    logger.warning("pydicom or numpy not found. DICOM functionality will be disabled.")
    pydicom = None
    np = None # Ensure numpy is None if import failed
    PYDICOM_AVAILABLE = False

if PYDICOM_AVAILABLE:
    try:
        import pylibjpeg
        logger.info("pylibjpeg found (for extended DICOM decompression).")
    except ImportError:
        logger.info("pylibjpeg not found. Some DICOM compression syntaxes may not be supported.")
    try:
        import gdcm
        logger.info("python-gdcm found (for improved DICOM compatibility).")
    except ImportError:
        logger.info("python-gdcm not found. Some DICOM functionalities may be reduced.")

# --- Custom Utilities & Backend Modules ---
# Use consistent try-except blocks with fallbacks for optional modules
try:
    from dicom_utils import (parse_dicom, extract_dicom_metadata, dicom_to_image, get_default_wl)
    DICOM_UTILS_AVAILABLE = True
    logger.info("dicom_utils imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import dicom_utils: {e}. DICOM features disabled.")
    DICOM_UTILS_AVAILABLE = False
    def parse_dicom(*args, **kwargs): logger.error("parse_dicom called but module unavailable."); return None
    def extract_dicom_metadata(*args, **kwargs): logger.error("extract_dicom_metadata called but module unavailable."); return {}
    def dicom_to_image(*args, **kwargs): logger.error("dicom_to_image called but module unavailable."); return None
    def get_default_wl(*args, **kwargs): logger.error("get_default_wl called but module unavailable."); return (None, None)

try:
    from llm_interactions import (run_initial_analysis, run_multimodal_qa, run_disease_analysis, estimate_ai_confidence)
    LLM_INTERACTIONS_AVAILABLE = True
    logger.info("llm_interactions imported successfully.")
except ImportError as e:
    st.error(f"Core AI module (llm_interactions) failed to import: {e}. Analysis functions disabled.")
    logger.critical(f"Failed to import llm_interactions: {e}", exc_info=True)
    LLM_INTERACTIONS_AVAILABLE = False
    def run_initial_analysis(*args, **kwargs): return "Error: AI Module Unavailable"
    def run_multimodal_qa(*args, **kwargs): return ("Error: AI Module Unavailable", False)
    def run_disease_analysis(*args, **kwargs): return "Error: AI Module Unavailable"
    def estimate_ai_confidence(*args, **kwargs): return "Error: AI Module Unavailable"
    # Consider st.stop() only if LLM is absolutely mandatory

try:
    from report_utils import generate_pdf_report_bytes
    REPORT_UTILS_AVAILABLE = True
    logger.info("report_utils imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import report_utils: {e}. PDF reporting disabled.")
    REPORT_UTILS_AVAILABLE = False
    def generate_pdf_report_bytes(*args, **kwargs): logger.error("generate_pdf_report_bytes called but module unavailable."); return None

try:
    from ui_components import display_dicom_metadata, dicom_wl_sliders, display_umls_concepts
    UI_COMPONENTS_AVAILABLE = True
    logger.info("ui_components imported successfully.")
except ImportError as e:
    logger.warning(f"Failed to import ui_components: {e}. Custom UI elements might be missing or use fallbacks.")
    UI_COMPONENTS_AVAILABLE = False
    def display_dicom_metadata(metadata): st.caption("Metadata display unavailable.")
    def dicom_wl_sliders(ds, current_wc, current_ww): st.caption("W/L sliders unavailable."); return current_wc, current_ww
    def display_umls_concepts(concepts): st.caption("UMLS display unavailable.")

try:
    from hf_models import query_hf_vqa_inference_api, HF_VQA_MODEL_ID
    HF_MODELS_AVAILABLE = True
    logger.info(f"hf_models imported successfully (Fallback VQA Model: {HF_VQA_MODEL_ID}).")
except ImportError:
    HF_VQA_MODEL_ID = "hf_model_not_found"
    HF_MODELS_AVAILABLE = False
    def query_hf_vqa_inference_api(img: Optional[Image.Image], question: str, roi: Optional[Dict] = None) -> Tuple[str, bool]:
        logger.warning("query_hf_vqa_inference_api called but hf_models module is unavailable.")
        return "[Fallback VQA Unavailable] HF module not found.", False
    logger.warning("hf_models not found. Fallback VQA disabled.")

try:
    from translation_models import (translate, detect_language, LANGUAGE_CODES, AUTO_DETECT_INDICATOR)
    TRANSLATION_AVAILABLE = True
    logger.info("translation_models imported successfully. Translation is available.")
except ImportError as e:
    st.warning(f"Translation features disabled: Could not import translation_models: {e}")
    logger.error(f"Could not import translation_models: {e}", exc_info=True)
    TRANSLATION_AVAILABLE = False
    def translate(*args, **kwargs): logger.error("translate called but module unavailable."); return "Translation Error: Module Unavailable"
    def detect_language(*args, **kwargs): logger.error("detect_language called but module unavailable."); return "en"
    LANGUAGE_CODES = {"English": "en"}
    AUTO_DETECT_INDICATOR = "Auto-Detect"

try:
    import umls_utils
    from umls_utils import UMLSAuthError, UMLSConcept
    UMLS_APIKEY = os.getenv("UMLS_APIKEY")
    if not UMLS_APIKEY:
        logger.warning("UMLS_APIKEY environment variable not set. UMLS features will be disabled.")
        UMLS_AVAILABLE = False
    else:
        logger.info("umls_utils imported successfully and UMLS_APIKEY found.")
        UMLS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import umls_utils: {e}. UMLS features disabled.")
    UMLS_AVAILABLE = False
    umls_utils = None
    UMLSAuthError = RuntimeError
    UMLSConcept = None
    UMLS_APIKEY = None
except Exception as e:
    logger.error(f"Error during UMLS setup: {e}", exc_info=True)
    UMLS_AVAILABLE = False
    umls_utils = None
    UMLSAuthError = RuntimeError
    UMLSConcept = None
    UMLS_APIKEY = None


# --- Custom CSS for Polished Look & Tab Scrolling ---
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
      }
      /* Sidebar Styling */
      .css-1d391kg { /* Adjust selector if Streamlit updates */
          background-color: #ffffff; /* White sidebar */
          border-right: 1px solid #e0e0e0;
      }
      .stButton>button {
          border-radius: 8px;
          padding: 0.5rem 1rem;
          font-weight: 500;
          transition: background-color 0.2s ease-in-out, border-color 0.2s ease-in-out;
          width: 100%;
          margin-bottom: 0.5rem;
      }
      .stButton>button:hover {
        filter: brightness(95%);
      }

      /* --- FIX for faint text in sidebar inputs --- */
      /* Target placeholder text in text_area within the sidebar */
      /* Note: Selectors might be fragile and depend on Streamlit's internal structure */
      .css-1d391kg .stTextArea textarea::placeholder {
          color: #6c757d !important; /* Use a standard muted text color, adjust if needed */
          opacity: 1; /* Ensure placeholder is not transparent */
      }
      /* Target selectbox default value display */
      /* This selector is very likely to change between versions */
      .css-1d391kg div[data-baseweb="select"] > div:first-child > div:first-child {
           color: #31333F !important; /* Use default text color or slightly lighter */
      }
      /* Target the dropdown arrow in selectbox */
       .css-1d391kg div[data-baseweb="select"] svg {
           fill: #31333F !important; /* Match text color */
       }
      /* --- End FIX --- */


      /* Tab Scrolling */
      div[role="tablist"] {
          overflow-x: auto;
          white-space: nowrap;
          border-bottom: 1px solid #e0e0e0;
          scrollbar-width: thin;
          scrollbar-color: #cccccc #f0f2f6;
      }
      div[role="tablist"]::-webkit-scrollbar { height: 6px; }
      div[role="tablist"]::-webkit-scrollbar-track { background: #f0f2f6; }
      div[role="tablist"]::-webkit-scrollbar-thumb {
          background-color: #cccccc; border-radius: 10px; border: 2px solid #f0f2f6;
      }
      /* Footer Styling */
      footer {
          text-align: center; font-size: 0.8em; color: #6c757d;
          margin-top: 2rem; padding: 1rem 0; border-top: 1px solid #e0e0e0;
      }
      footer a { color: #007bff; text-decoration: none; }
      footer a:hover { text-decoration: underline; }
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
    "translation_result": None, "translation_error": None, "umls_search_term": "",
    "umls_results": None, "umls_error": None,
}

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
    logger.info(f"New session started: {st.session_state.session_id}")

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value

if not isinstance(st.session_state.get("history"), list):
    st.session_state.history = []

logger.debug(f"Session state keys initialized/verified for session ID: {st.session_state.session_id}")


# --- Helper function for formatting translation ---
def format_translation(translated_text: Optional[str]) -> str:
    if translated_text is None: return "Translation not available or failed."
    try:
        text_str = str(translated_text)
        formatted_text = re.sub(r'\s+(\d+\.)', r'\n\n\1', text_str)
        return formatted_text.strip()
    except Exception as e:
        logger.error(f"Error formatting translation: {e}", exc_info=True)
        return str(translated_text)


# --- Monkey-Patch (Conditional) ---
# Keep for potential backward compatibility, but less likely needed now
import streamlit.elements.image as st_image
if not hasattr(st_image, "image_to_url"):
    logger.info("Attempting to apply monkey-patch for st.elements.image.image_to_url.")
    # (Include the monkey-patch function definition here if needed)
    # ... image_to_url_monkey_patch definition ...
    # st_image.image_to_url = image_to_url_monkey_patch
    # logger.info("Applied monkey-patch for st.elements.image.image_to_url.")
else:
     logger.debug("Monkey-patch for st.elements.image.image_to_url not needed.")


# --- Sidebar ---
with st.sidebar:
    st.header("⚕️ RadVision Controls")
    st.markdown("---")
    TIPS = [
        "Tip: Use 'Demo Mode' for a quick walkthrough with a sample chest X-ray.",
        "Tip: Draw a rectangle (ROI) on the image to focus the AI's attention.",
        "Tip: Adjust DICOM Window/Level sliders for optimal image contrast.",
        "Tip: Ask follow-up questions based on the initial analysis or previous answers.",
        "Tip: Generate a PDF report to document the AI findings and your interaction.",
        "Tip: Use the 'Translation' tab to understand findings in different languages.",
        "Tip: Clear the ROI using the button if you want the AI to consider the entire image again.",
        "Tip: Use the 'UMLS Lookup' tab to find standardized concepts for medical terms.",
    ]
    st.info(f"💡 {random.choice(TIPS)}")
    st.markdown("---")

    st.header("Image Upload & Settings")
    uploaded_file = st.file_uploader(
        "Upload Image (JPG, PNG, DCM)",
        type=["jpg", "jpeg", "png", "dcm", "dicom"],
        key="file_uploader_widget",
        help="Upload a medical image file for analysis. DICOM (.dcm) is preferred."
    )
    demo_mode = st.checkbox("🚀 Demo Mode", value=st.session_state.get("demo_loaded", False),
                            help="Load a sample chest X-ray image and analysis.")
    # Add Demo Mode implementation logic here if needed

    if st.button("🗑️ Clear ROI", help="Remove the selected ROI", key="clear_roi_btn"):
        st.session_state.roi_coords = None
        st.session_state.canvas_drawing = None
        st.session_state.clear_roi_feedback = True
        logger.info("ROI cleared by user.")
        st.rerun()

    if st.session_state.get("clear_roi_feedback"):
        st.success("✅ ROI cleared successfully!")
        st.balloons()
        st.session_state.clear_roi_feedback = False

    # DICOM Window/Level Section
    # Only show if it's a DICOM and the display/processing was successful
    if st.session_state.is_dicom and st.session_state.display_image and UI_COMPONENTS_AVAILABLE:
        st.markdown("---")
        st.subheader("DICOM Display")
        new_wc, new_ww = dicom_wl_sliders(
            st.session_state.dicom_dataset,
            st.session_state.current_display_wc,
            st.session_state.current_display_ww
        )
        if new_wc != st.session_state.current_display_wc or new_ww != st.session_state.current_display_ww:
            logger.info(f"DICOM W/L changed via sliders: WC={new_wc}, WW={new_ww}")
            st.session_state.current_display_wc = new_wc
            st.session_state.current_display_ww = new_ww
            if DICOM_UTILS_AVAILABLE and st.session_state.dicom_dataset and PIL_AVAILABLE:
                with st.spinner("Applying new Window/Level..."):
                    logger.debug("Attempting to regenerate display image with new W/L...")
                    new_display_img = dicom_to_image(st.session_state.dicom_dataset, wc=new_wc, ww=new_ww)
                    if isinstance(new_display_img, Image.Image):
                        if new_display_img.mode != 'RGB': new_display_img = new_display_img.convert('RGB')
                        st.session_state.display_image = new_display_img
                        logger.info("Display image updated successfully with new W/L.")
                        st.rerun()
                    else:
                        st.error("Failed to update DICOM image with new W/L.")
                        logger.error(f"dicom_to_image returned invalid type ({type(new_display_img)}) after W/L update.")
            elif not DICOM_UTILS_AVAILABLE:
                 st.warning("DICOM utilities not available to update W/L.")
                 logger.warning("W/L changed but DICOM utilities missing.")
            elif not st.session_state.dicom_dataset:
                 st.warning("DICOM dataset not loaded, cannot update W/L.")
                 logger.warning("W/L changed but DICOM dataset missing in state.")

    st.markdown("---")
    st.header("🤖 AI Analysis Actions")
    action_disabled = not isinstance(st.session_state.get("processed_image"), Image.Image)

    if st.button("▶️ Run Initial Analysis", key="analyze_btn", disabled=action_disabled, help="Perform a general analysis of the entire image or selected ROI."):
        st.session_state.last_action = "analyze"; st.rerun()
    st.subheader("❓ Ask AI a Question")
    question_input = st.text_area(
        "Enter your question:", height=100, key="question_input_widget",
        placeholder="E.g., 'Are there any nodules in the upper right lobe?'", disabled=action_disabled
    )
    if st.button("💬 Ask Question", key="ask_btn", disabled=action_disabled):
        if question_input.strip(): st.session_state.last_action = "ask"; st.rerun()
        else: st.warning("Please enter a question before submitting.")

    st.subheader("🎯 Condition-Specific Analysis")
    DISEASE_OPTIONS = [
        "Pneumonia", "Lung Cancer", "Nodule/Mass", "Effusion", "Fracture", "Stroke", "Appendicitis",
        "Bowel Obstruction", "Cardiomegaly", "Aortic Aneurysm", "Pulmonary Embolism", "Tuberculosis",
        "COVID-19", "Brain Tumor", "Arthritis", "Osteoporosis", "Other..."
    ]
    disease_select = st.selectbox(
        "Select condition to focus on:", options=[""] + sorted(DISEASE_OPTIONS),
        key="disease_select_widget", disabled=action_disabled
    )
    if st.button("🩺 Analyze Condition", key="disease_btn", disabled=action_disabled):
        if disease_select: st.session_state.last_action = "disease"; st.rerun()
        else: st.warning("Please select a condition first.")

    st.markdown("---")
    st.header("📊 Confidence & Reporting")
    can_estimate_confidence = bool(st.session_state.history or st.session_state.initial_analysis or st.session_state.disease_analysis) and not action_disabled
    if st.button("📈 Estimate AI Confidence", key="confidence_btn", disabled=not can_estimate_confidence):
        st.session_state.last_action = "confidence"; st.rerun()
    report_generation_disabled = action_disabled or not REPORT_UTILS_AVAILABLE
    if st.button("📄 Generate PDF Report Data", key="generate_report_data_btn", disabled=report_generation_disabled):
        st.session_state.last_action = "generate_report_data"; st.rerun()
    if st.session_state.get("pdf_report_bytes"):
        report_filename = f"RadVisionAI_Report_{st.session_state.session_id or 'session'}.pdf"
        st.download_button(
            label="⬇️ Download PDF Report", data=st.session_state.pdf_report_bytes,
            file_name=report_filename, mime="application/pdf", key="download_pdf_button",
            help="Download the generated PDF report."
        )


# --- File Upload Processing Logic ---
if uploaded_file is not None:
    try:
        uploaded_file.seek(0); file_content_hash = hashlib.sha256(uploaded_file.read()).hexdigest()[:16]; uploaded_file.seek(0)
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{file_content_hash}"
        logger.debug(f"Calculated new file info: {new_file_info}")
    except Exception as e:
        logger.warning(f"Could not generate hash for file {uploaded_file.name}: {e}")
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{uuid.uuid4().hex[:8]}" # Fallback
        logger.debug(f"Using fallback file info: {new_file_info}")

    if new_file_info != st.session_state.get("uploaded_file_info"):
        logger.info(f"New file detected: '{new_file_info}' (Previous: '{st.session_state.get('uploaded_file_info')}')")
        st.toast(f"Processing '{uploaded_file.name}'...", icon="⏳")
        # --- State Reset Logic for New File ---
        keys_to_preserve = {"session_id"} # Only preserve session ID by default
        preserved_values = {k: st.session_state.get(k) for k in keys_to_preserve if k in st.session_state}
        logger.debug("Resetting session state for new file (preserving only session_id)...")
        for key, default_value in DEFAULT_STATE.items():
            if key not in keys_to_preserve:
                st.session_state[key] = copy.deepcopy(default_value) if isinstance(default_value, (dict, list)) else default_value
        for k, v in preserved_values.items(): st.session_state[k] = v
        # --- End State Reset ---
        st.session_state.uploaded_file_info = new_file_info
        st.session_state.demo_loaded = False
        st.session_state.raw_image_bytes = uploaded_file.getvalue()
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        st.session_state.is_dicom = (PYDICOM_AVAILABLE and DICOM_UTILS_AVAILABLE and ("dicom" in uploaded_file.type.lower() or file_ext in (".dcm", ".dicom")))
        logger.info(f"File '{uploaded_file.name}' identified as DICOM: {st.session_state.is_dicom}")

        with st.spinner("🔬 Analyzing file... Please wait."):
            temp_display_img: Optional[Image.Image] = None
            temp_processed_img: Optional[Image.Image] = None
            processing_success = False
            error_message = "" # Store specific error message

            if st.session_state.is_dicom:
                logger.info("Processing as DICOM...")
                if not DICOM_UTILS_AVAILABLE or not PYDICOM_AVAILABLE:
                     error_message = "DICOM processing libraries (pydicom, dicom_utils) not available."
                     logger.error(error_message)
                     st.session_state.is_dicom = False # Cannot process as DICOM
                else:
                    try:
                        dicom_ds = parse_dicom(st.session_state.raw_image_bytes, filename=uploaded_file.name)
                        st.session_state.dicom_dataset = dicom_ds
                        if dicom_ds:
                            st.session_state.dicom_metadata = extract_dicom_metadata(dicom_ds)
                            logger.info(f"Extracted {len(st.session_state.dicom_metadata)} DICOM tags.")
                            default_wc, default_ww = get_default_wl(dicom_ds)
                            st.session_state.current_display_wc, st.session_state.current_display_ww = default_wc, default_ww
                            logger.info(f"Default DICOM W/L: WC={default_wc}, WW={default_ww}")
                            temp_display_img = dicom_to_image(dicom_ds, wc=default_wc, ww=default_ww)
                            temp_processed_img = dicom_to_image(dicom_ds, wc=None, ww=None, normalize=True)
                            if isinstance(temp_display_img, Image.Image) and isinstance(temp_processed_img, Image.Image):
                                processing_success = True
                                logger.info("DICOM converted to display/processed images.")
                            else:
                                error_message = "Failed to convert DICOM pixel data to image format (dicom_to_image error)."
                                logger.error(error_message)
                        else:
                            error_message = "Could not parse DICOM file structure (parse_dicom returned None)."
                            logger.error(error_message)
                            st.session_state.is_dicom = False
                    except pydicom.errors.InvalidDicomError as e:
                        error_message = f"Invalid DICOM file format: {e}"
                        logger.error(error_message)
                        st.session_state.is_dicom = False
                    except Exception as e:
                        error_message = f"Unexpected error during DICOM processing: {e}"
                        logger.error(error_message, exc_info=True)
                        st.session_state.is_dicom = False

            if not st.session_state.is_dicom: # Process as standard if not DICOM or DICOM failed
                logger.info("Attempting to process as standard image (JPG/PNG)...")
                if not PIL_AVAILABLE:
                    error_message = "Pillow (PIL) library not available for standard image processing."
                    logger.critical(error_message)
                else:
                    try:
                        raw_img = Image.open(io.BytesIO(st.session_state.raw_image_bytes))
                        processed_img = raw_img.convert("RGB") # Ensure RGB
                        temp_display_img = processed_img.copy()
                        temp_processed_img = processed_img.copy()
                        processing_success = True
                        logger.info(f"Standard image '{uploaded_file.name}' processed successfully.")
                    except UnidentifiedImageError:
                        error_message = "Cannot identify image file format. Please use JPG, PNG, or valid DICOM."
                        logger.error(f"{error_message} File: {uploaded_file.name}")
                    except Exception as e:
                        error_message = f"Error processing standard image: {e}"
                        logger.error(error_message, exc_info=True)

            # --- Finalize state update ---
            if processing_success and isinstance(temp_display_img, Image.Image) and isinstance(temp_processed_img, Image.Image):
                logger.info("Image processing successful. Updating session state.")
                st.session_state.display_image = temp_display_img.convert('RGB') if temp_display_img.mode != 'RGB' else temp_display_img
                st.session_state.processed_image = temp_processed_img
                st.success(f"✅ '{uploaded_file.name}' loaded and processed!")
                st.rerun()
            else:
                 logger.error(f"Image processing failed for {uploaded_file.name}. Error: {error_message}")
                 st.error(f"Failed to load image '{uploaded_file.name}'. Reason: {error_message or 'Unknown processing error.'}")
                 # Clear state associated with the failed load
                 st.session_state.uploaded_file_info = None
                 st.session_state.display_image = None
                 st.session_state.processed_image = None
                 st.session_state.is_dicom = False
                 st.session_state.dicom_dataset = None
                 st.session_state.dicom_metadata = {}
                 st.session_state.raw_image_bytes = None
                 st.session_state.current_display_wc = None
                 st.session_state.current_display_ww = None


# --- Main Page Layout ---
st.markdown("---")
st.title("⚕️ RadVision AI Advanced: AI-Assisted Image Analysis")
with st.expander("User Guide & Disclaimer", expanded=False):
    st.warning("⚠️ **Disclaimer**: This tool is for research/educational purposes only and is **NOT** a substitute for professional medical advice or diagnosis. AI analysis may contain errors.")
    # (Keep existing User Guide Markdown)
    st.markdown("""
    **Workflow:**
    1.  **Upload Image**: Use the sidebar to upload a DICOM, JPG, or PNG file (or activate Demo Mode).
    2.  **(Optional - DICOM)** Adjust Window/Level sliders in the sidebar for optimal contrast.
    3.  **(Optional)** Draw a rectangle (ROI) on the image viewer to focus the AI analysis. Use 'Clear ROI' to remove it.
    4.  **AI Analysis**: Use the action buttons in the sidebar:
        *   `Run Initial Analysis`: Get general findings.
        *   `Ask Question`: Ask specific questions about the image/ROI.
        *   `Analyze Condition`: Focus the AI on a specific condition.
    5.  **Explore Results**: View AI outputs in the tabs on the right (Initial Analysis, Q&A, Condition Focus).
    6.  **(Optional) UMLS Lookup**: Use the UMLS tab to search for standardized medical terms.
    7.  **(Optional) Translation**: Use the Translation tab to translate AI-generated text.
    8.  **(Optional) Confidence**: Estimate the AI's confidence based on the analyses performed.
    9.  **Generate Report**: Create a PDF summary of the session, including image, metadata (DICOM), and AI interactions.
    """)

st.markdown("---")
col1, col2 = st.columns([2, 3]) # Image column slightly smaller

with col1:
    st.subheader("🖼️ Image Viewer")
    display_img = st.session_state.get("display_image")

    # --- Display Logic ---
    if isinstance(display_img, Image.Image) and PIL_AVAILABLE:
        logger.debug("Displaying image using st_canvas (if available) or st.image.")
        if DRAWABLE_CANVAS_AVAILABLE and st_canvas:
            st.caption("Draw a rectangle below to select a Region of Interest (ROI).")
            MAX_CANVAS_WIDTH, MAX_CANVAS_HEIGHT = 600, 500
            img_w, img_h = display_img.size
            if img_w <= 0 or img_h <= 0:
                st.warning("Invalid image dimensions; cannot draw ROI.")
            else:
                aspect_ratio = img_w / img_h
                canvas_width = min(img_w, MAX_CANVAS_WIDTH)
                canvas_height = int(canvas_width / aspect_ratio)
                if canvas_height > MAX_CANVAS_HEIGHT: canvas_height = MAX_CANVAS_HEIGHT; canvas_width = int(canvas_height * aspect_ratio)
                canvas_width, canvas_height = max(canvas_width, 150), max(canvas_height, 150)
                logger.debug(f"Canvas dimensions: {canvas_width}x{canvas_height}")

                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.2)", stroke_width=2, stroke_color="rgba(239, 83, 80, 0.8)",
                    background_image=display_img, update_streamlit=True, height=canvas_height, width=canvas_width,
                    drawing_mode="rect", initial_drawing=st.session_state.get("canvas_drawing", None), key="drawable_canvas"
                )
                if canvas_result.json_data and canvas_result.json_data.get("objects"):
                    if canvas_result.json_data["objects"]:
                        last_object = canvas_result.json_data["objects"][-1]
                        if last_object["type"] == "rect":
                            # (ROI Calculation Logic - Keep as is)
                            canvas_left = int(last_object["left"]); canvas_top = int(last_object["top"])
                            canvas_width_scaled = int(last_object["width"] * last_object.get("scaleX", 1))
                            canvas_height_scaled = int(last_object["height"] * last_object.get("scaleY", 1))
                            scale_x = img_w / canvas_width; scale_y = img_h / canvas_height
                            original_left = max(0, int(canvas_left * scale_x)); original_top = max(0, int(canvas_top * scale_y))
                            original_width = max(1, min(img_w - original_left, int(canvas_width_scaled * scale_x)))
                            original_height = max(1, min(img_h - original_top, int(canvas_height_scaled * scale_y)))
                            new_roi = {"left": original_left, "top": original_top, "width": original_width, "height": original_height}
                            if st.session_state.roi_coords != new_roi:
                                st.session_state.roi_coords = new_roi
                                st.session_state.canvas_drawing = canvas_result.json_data
                                logger.info(f"New ROI selected (original coords): {new_roi}")
                                st.info(f"ROI Set: ({original_left},{original_top}), Size: {original_width}x{original_height}", icon="🎯")
        else:
             # Fallback if canvas is not available
             logger.debug("Drawable canvas not available, using st.image.")
             st.image(display_img, caption="Image Preview", use_container_width=True)

        if st.session_state.roi_coords:
            roi = st.session_state.roi_coords
            st.caption(f"Active ROI: ({roi['left']}, {roi['top']}) - W:{roi['width']}, H:{roi['height']}")
        st.markdown("---")
        if st.session_state.is_dicom:
            if st.session_state.dicom_metadata:
                display_dicom_metadata(st.session_state.dicom_metadata)
            # else: # Already handled by processing logic error messages
            #     st.caption("DICOM file loaded, but no metadata extracted.")

    elif st.session_state.get("uploaded_file_info") and not st.session_state.get("display_image"):
        st.error("❌ Image preview unavailable. Processing failed after upload.")
        logger.warning("Image display blocked because display_image is missing despite upload.")
    else:
        st.info("⬅️ Please upload an image or enable Demo Mode using the sidebar.")
        logger.debug("No valid image available for display.")


with col2: # Analysis Results Tabs
    st.subheader("📊 Analysis & Results")
    tab_titles = ["🔬 Initial Analysis", "💬 Q&A History", "🩺 Condition Focus", "📚 UMLS Lookup", "📈 Confidence", "🌐 Translation"]
    tabs = st.tabs(tab_titles)

    with tabs[0]: # Initial Analysis
        st.text_area("Overall Findings & Impressions", value=st.session_state.initial_analysis or "Run 'Initial Analysis' to see results.", height=400, disabled=True, key="initial_analysis_display")
    with tabs[1]: # Q&A History
        st.text_area("Latest AI Answer", value=st.session_state.qa_answer or "Ask a question to see the response.", height=150, disabled=True, key="qa_answer_display")
        st.markdown("---"); st.subheader("Full Conversation History")
        if st.session_state.history:
            for i, (q_type, message) in enumerate(reversed(st.session_state.history)):
                prefix = "👤 You:" if q_type.lower() == "user question" else \
                         "🤖 AI:" if q_type.lower() == "ai answer" else \
                         "👤 You (Fallback):" if q_type.lower() == "[fallback] user question" else \
                         "🤖 AI (Fallback):" if q_type.lower() == "[fallback] ai answer" else \
                         "ℹ️ System:" if q_type.lower() == "system" else f"**{q_type}:**"
                unsafe = "ai answer" in q_type.lower() # Allow markdown potentially in AI answers
                st.markdown(f"{prefix} {message}", unsafe_allow_html=unsafe)
                if i < len(st.session_state.history) - 1: st.markdown("---")
        else: st.caption("No questions asked yet.")
    with tabs[2]: # Condition Focus
        st.text_area("Condition-Specific Analysis", value=st.session_state.disease_analysis or "Select a condition and click 'Analyze Condition'.", height=400, disabled=True, key="disease_analysis_display")
    with tabs[3]: # UMLS Lookup
        st.subheader("📚 UMLS Concept Search")
        if not UMLS_AVAILABLE: st.warning("UMLS features unavailable (check API key / utils).")
        else:
            umls_search_term = st.text_input("Enter medical term to search:", value=st.session_state.get("umls_search_term", ""), key="umls_search_term_input", placeholder="e.g., lung nodule, cardiomegaly")
            if st.button("🔎 Search UMLS", key="umls_search_button"):
                if umls_search_term.strip():
                    st.session_state.last_action = "umls_search"; st.session_state.umls_search_term = umls_search_term.strip(); st.rerun()
                else: st.warning("Please enter a search term.")
            if st.session_state.get("umls_error"): st.error(f"UMLS Search Error: {st.session_state.umls_error}")
            if UI_COMPONENTS_AVAILABLE: display_umls_concepts(st.session_state.get("umls_results"))
            else: st.caption("UMLS display component unavailable.") # Fallback handled in component itself too
    with tabs[4]: # Confidence
        st.text_area("Estimated AI Confidence", value=st.session_state.confidence_score or "Run 'Estimate AI Confidence' after analysis.", height=400, disabled=True, key="confidence_score_display")
    with tabs[5]: # Translation
        st.subheader("🌐 Translate Analysis Text")
        if not TRANSLATION_AVAILABLE: st.warning("Translation features unavailable.")
        else:
            st.caption("Select text, choose languages, then click 'Translate'.")
            text_options = {"Initial Analysis": st.session_state.initial_analysis, "Latest Q&A Answer": st.session_state.qa_answer, "Condition Analysis": st.session_state.disease_analysis, "Confidence Estimation": st.session_state.confidence_score, "(Enter Custom Text Below)": ""}
            available_options = {label: txt for label, txt in text_options.items() if (txt and txt.strip()) or label == "(Enter Custom Text Below)"}
            if not available_options: st.info("No analysis text available to translate yet.")
            else:
                selected_label = st.selectbox("Select text to translate:", list(available_options.keys()), index=0, key="translate_text_select")
                text_to_translate = available_options.get(selected_label, "")
                if selected_label == "(Enter Custom Text Below)":
                    text_to_translate = st.text_area("Enter or paste text to translate:", value="", height=100, key="custom_translate_input")
                else:
                     st.text_area("Text selected for translation:", value=text_to_translate, height=100, disabled=True, key="selected_translate_display")
                col_lang1, col_lang2 = st.columns(2)
                with col_lang1:
                    source_language_options = [AUTO_DETECT_INDICATOR] + sorted(list(LANGUAGE_CODES.keys()))
                    source_language_name = st.selectbox("Source Language:", source_language_options, index=0, key="source_language_select")
                with col_lang2:
                    target_language_options = sorted(list(LANGUAGE_CODES.keys())); default_target_index = 0
                    common_targets = ["English", "Spanish", "French", "German"]
                    for i, lang in enumerate(target_language_options):
                        if lang in common_targets: default_target_index = i; break
                    target_language_name = st.selectbox("Translate To:", target_language_options, index=default_target_index, key="target_language_select")
                if st.button("🔄 Translate Now", key="translate_button"):
                    st.session_state.translation_result = None; st.session_state.translation_error = None
                    if not text_to_translate.strip(): st.warning("Please select or enter some text."); st.session_state.translation_error = "Input text is empty."
                    elif source_language_name == target_language_name and source_language_name != AUTO_DETECT_INDICATOR:
                        st.info("Source and target languages are the same."); st.session_state.translation_result = text_to_translate
                    else:
                         with st.spinner(f"Translating..."):
                             try:
                                 translation_output = translate(text=text_to_translate, target_language=target_language_name, source_language=source_language_name)
                                 if translation_output is not None: st.session_state.translation_result = translation_output; st.success("Translation complete!")
                                 else: st.error("Translation service returned no result."); logger.error("Translate function returned None."); st.session_state.translation_error = "Service returned None."
                             except Exception as e: st.error(f"Translation failed: {e}"); logger.error(f"Translation error: {e}", exc_info=True); st.session_state.translation_error = str(e)
                if st.session_state.get("translation_result"):
                    formatted_result = format_translation(st.session_state.translation_result)
                    st.text_area("Translated Text:", value=formatted_result, height=200, key="translation_output_display")


# --- Button Action Handlers ---
current_action = st.session_state.get("last_action")
if current_action:
    logger.info(f"Handling action: '{current_action}' for session: {st.session_state.session_id}")
    # --- Pre-action checks ---
    action_requires_image = current_action not in ["generate_report_data", "umls_search"]
    action_requires_llm = current_action in ["analyze", "ask", "disease", "confidence"]
    action_requires_report_util = (current_action == "generate_report_data")
    action_requires_umls = (current_action == "umls_search")
    valid_action = True
    if action_requires_image and not isinstance(st.session_state.get("processed_image"), Image.Image):
        st.error(f"Cannot perform '{current_action}': No valid image loaded."); valid_action = False
    if not st.session_state.session_id: st.error("Critical Error: No session ID."); valid_action = False
    if action_requires_llm and not LLM_INTERACTIONS_AVAILABLE: st.error(f"Cannot perform '{current_action}': AI module unavailable."); valid_action = False
    if action_requires_report_util and not REPORT_UTILS_AVAILABLE: st.error(f"Cannot perform '{current_action}': Report module unavailable."); valid_action = False
    if action_requires_umls and not UMLS_AVAILABLE: st.error(f"Cannot perform '{current_action}': UMLS module/key unavailable."); valid_action = False

    # --- Execute Action ---
    if valid_action:
        img_for_llm = st.session_state.processed_image
        roi_coords = st.session_state.roi_coords
        current_history = st.session_state.history
        try:
            # --- Action Implementations (Keep existing logic from previous full script) ---
            if current_action == "analyze":
                st.info("🔬 Performing initial analysis..."); logger.debug("Running initial analysis.")
                with st.spinner("AI analyzing image..."): analysis_result = run_initial_analysis(img_for_llm, roi=roi_coords)
                st.session_state.initial_analysis = analysis_result; st.session_state.qa_answer = ""; st.session_state.disease_analysis = ""
                logger.info("Initial analysis complete."); st.success("Initial analysis complete!")
            elif current_action == "ask":
                question_text = st.session_state.question_input_widget.strip()
                if not question_text: st.warning("Question input was empty.")
                else:
                    st.info(f"❓ Asking AI: '{question_text}'..."); logger.debug(f"Running Q&A for: {question_text}")
                    st.session_state.qa_answer = ""
                    with st.spinner("AI thinking..."): answer, success_flag = run_multimodal_qa(img=img_for_llm, question=question_text, history=current_history, roi=roi_coords)
                    if success_flag:
                        st.session_state.qa_answer = answer; st.session_state.history.append(("User Question", question_text)); st.session_state.history.append(("AI Answer", answer))
                        logger.info("Q&A successful."); st.success("AI answered!")
                    else: # Primary AI failed
                        primary_error_msg = f"Primary AI failed: {answer}"; st.session_state.qa_answer = primary_error_msg; st.error(primary_error_msg); logger.error(f"Primary Q&A failed: {answer}")
                        # Fallback Logic (keep existing)
                        hf_token = os.environ.get("HF_API_TOKEN")
                        if HF_MODELS_AVAILABLE and hf_token:
                            st.info(f"Attempting fallback: {HF_VQA_MODEL_ID}"); logger.debug("Attempting HF fallback.")
                            with st.spinner("Trying fallback..."): fallback_answer, fallback_success = query_hf_vqa_inference_api(img=img_for_llm, question=question_text, roi=roi_coords)
                            if fallback_success:
                                fallback_display = f"**[Fallback: {HF_VQA_MODEL_ID}]**\n{fallback_answer}"; st.session_state.qa_answer += "\n\n" + fallback_display
                                st.session_state.history.append(("[Fallback] User Question", question_text)); st.session_state.history.append(("[Fallback] AI Answer", fallback_display))
                                logger.info("Fallback Q&A successful."); st.success("Fallback AI answered.")
                            else: fallback_error_msg = f"[Fallback Error - {HF_VQA_MODEL_ID}]: {fallback_answer}"; st.session_state.qa_answer += f"\n\n{fallback_error_msg}"; logger.error(f"Fallback Q&A failed: {fallback_answer}"); st.error(fallback_error_msg)
                        elif HF_MODELS_AVAILABLE: st.session_state.qa_answer += "\n\n[Fallback Skipped: HF_API_TOKEN missing]"; logger.warning("Fallback skipped: token missing."); st.warning("HF Token missing for fallback.")
                        else: st.session_state.qa_answer += "\n\n[Fallback Unavailable]"; logger.warning("Fallback unavailable."); st.warning("Fallback AI unavailable.")
            elif current_action == "disease":
                selected_disease = st.session_state.disease_select_widget
                if not selected_disease: st.warning("No condition selected.")
                else:
                    st.info(f"🩺 Analyzing for '{selected_disease}'..."); logger.debug(f"Running disease analysis for: {selected_disease}")
                    with st.spinner(f"AI analyzing for {selected_disease}..."): disease_result = run_disease_analysis(img_for_llm, selected_disease, roi=roi_coords)
                    st.session_state.disease_analysis = disease_result; st.session_state.qa_answer = ""
                    logger.info(f"Disease analysis for '{selected_disease}' complete."); st.success(f"Analysis for '{selected_disease}' complete!")
            elif current_action == "umls_search":
                term_to_search = st.session_state.get("umls_search_term", "").strip()
                st.session_state.umls_results = None; st.session_state.umls_error = None
                if not term_to_search: st.warning("UMLS search term empty.")
                else:
                    st.info(f"🔎 Searching UMLS for: '{term_to_search}'..."); logger.debug(f"Running UMLS search for: {term_to_search}")
                    with st.spinner("Querying UMLS..."):
                        try:
                            results = umls_utils.search_umls(term_to_search, UMLS_APIKEY)
                            st.session_state.umls_results = results; logger.info(f"UMLS search returned {len(results)} result(s)."); st.success(f"UMLS search complete. Found {len(results)} concepts.")
                        except UMLSAuthError as e: err_msg = f"UMLS Auth Failed: {e}"; st.error(err_msg); logger.error(err_msg, exc_info=False); st.session_state.umls_error = f"Auth Error: {e}"
                        except RuntimeError as e: err_msg = f"UMLS Search Failed: {e}"; st.error(err_msg); logger.error(err_msg, exc_info=True); st.session_state.umls_error = f"Search Error: {e}"
                        except Exception as e: err_msg = f"Unexpected UMLS error: {e}"; st.error(err_msg); logger.critical(err_msg, exc_info=True); st.session_state.umls_error = f"Unexpected error: {e}"
            elif current_action == "confidence":
                if not (current_history or st.session_state.initial_analysis or st.session_state.disease_analysis): st.warning("No prior analysis/Q&A for confidence.")
                else:
                    st.info("📊 Estimating AI confidence..."); logger.debug("Running confidence estimation.")
                    with st.spinner("Calculating confidence..."): confidence_result = estimate_ai_confidence(img=img_for_llm, history=current_history, initial_analysis=st.session_state.initial_analysis, disease_analysis=st.session_state.disease_analysis, roi=roi_coords)
                    st.session_state.confidence_score = confidence_result; logger.info("Confidence estimation complete."); st.success("Confidence estimation complete!")
            elif current_action == "generate_report_data":
                st.info("📄 Generating PDF report data..."); logger.debug("Running report generation.")
                st.session_state.pdf_report_bytes = None
                image_for_report = st.session_state.get("display_image")
                if not isinstance(image_for_report, Image.Image): st.error("Cannot generate report: No valid image."); logger.error("Report gen failed: missing display image.")
                else:
                    final_image_for_pdf = image_for_report.copy().convert("RGB")
                    if roi_coords and ImageDraw:
                        try:
                            draw = ImageDraw.Draw(final_image_for_pdf); x0, y0, x1, y1 = roi_coords['left'], roi_coords['top'], roi_coords['left'] + roi_coords['width'], roi_coords['top'] + roi_coords['height']
                            draw.rectangle([x0, y0, x1, y1], outline="red", width=max(2, int(min(final_image_for_pdf.size) * 0.004)))
                            logger.info("Drew ROI on PDF image.")
                        except Exception as e: logger.error(f"Error drawing ROI on PDF image: {e}", exc_info=True); st.warning("Could not draw ROI on PDF image.")
                    formatted_history = "No Q&A interactions."
                    if current_history: lines = [f"{q_type}: {re.sub('<[^<]+?>', '', str(msg))}" for q_type, msg in current_history]; formatted_history = "\n\n".join(lines)
                    report_data = {"Session ID": st.session_state.session_id, "Image Filename": (st.session_state.uploaded_file_info or "N/A").split('-')[0], "Initial Analysis": st.session_state.initial_analysis or "Not Performed", "Conversation History": formatted_history, "Condition Analysis": st.session_state.disease_analysis or "Not Performed", "AI Confidence Estimation": st.session_state.confidence_score or "Not Performed"}
                    if st.session_state.is_dicom and st.session_state.dicom_metadata:
                         meta_keys = ['PatientName', 'PatientID', 'StudyDate', 'StudyTime', 'Modality', 'StudyDescription', 'SeriesDescription', 'Manufacturer', 'ManufacturerModelName']
                         meta_summary = {k: st.session_state.dicom_metadata.get(k, 'N/A') for k in meta_keys if st.session_state.dicom_metadata.get(k)}
                         if meta_summary: report_data["DICOM Summary"] = "\n".join([f"{k}: {v}" for k, v in meta_summary.items()])
                    with st.spinner("Generating PDF..."): pdf_bytes = generate_pdf_report_bytes(session_id=st.session_state.session_id, image=final_image_for_pdf, analysis_outputs=report_data, dicom_metadata=st.session_state.dicom_metadata if st.session_state.is_dicom else None)
                    if pdf_bytes: st.session_state.pdf_report_bytes = pdf_bytes; st.success("PDF data ready! Download in sidebar."); logger.info("PDF report generated."); st.balloons()
                    else: st.error("Failed to generate PDF."); logger.error("PDF generator returned no data.")
            else: st.warning(f"Unknown action '{current_action}'.")
        except Exception as e: st.error(f"Error during '{current_action}': {e}"); logger.critical(f"Action '{current_action}' error: {e}", exc_info=True)
        finally: st.session_state.last_action = None; logger.debug(f"Action '{current_action}' complete."); st.rerun()
    else: # Action prerequisites not met
        st.session_state.last_action = None # Reset action trigger if it was invalid
        logger.warning(f"Action '{current_action}' prerequisites not met. Action cancelled.")
        # No rerun needed, error message already shown


# --- Footer ---
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced | Session ID: {st.session_state.get('session_id', 'N/A')}")
st.markdown("""<footer> <p>RadVision AI is intended for informational and educational purposes only. It is not a substitute for professional medical judgment, diagnosis, or treatment.</p> <p> <a href="#" target="_blank">Privacy Policy</a> | <a href="#" target="_blank">Terms of Service</a> | <a href="https://github.com/mgbam/radvisionai" target="_blank">GitHub</a> </p></footer>""", unsafe_allow_html=True)
logger.info(f"--- Application render complete for session: {st.session_state.session_id} ---")