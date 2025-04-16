# -*- coding: utf-8 -*-
"""
app.py - Main Streamlit application for RadVision AI Advanced.

Handles image uploading, display, ROI selection, interaction with AI models
using an assumed agentic/structured approach for analysis and Q&A,
translation, and report generation. Focuses on responsible AI demonstration.
"""

import streamlit as st

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="RadVision AI Advanced",
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

# --- Ensure deep-translator is installed at runtime if not present ---
try:
    from deep_translator import GoogleTranslator
    DEEP_TRANSLATOR_INSTALLED = True
except ImportError:
    DEEP_TRANSLATOR_INSTALLED = False
    try:
        print("Attempting to install deep-translator...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "deep-translator"])
        from deep_translator import GoogleTranslator
        DEEP_TRANSLATOR_INSTALLED = True
        print("deep-translator installed successfully.")
    except Exception as e:
        print(f"CRITICAL: Could not install deep-translator: {e}")
        # Flag will remain False, handled later.

# --- Logging Setup ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.info("--- RadVision AI Application Start ---")
logger.info(f"Streamlit Version: {st.__version__}")
logger.info(f"Logging Level: {LOG_LEVEL}")

# --- Dependency Checks & Imports ---

# Streamlit Drawable Canvas
try:
    from streamlit_drawable_canvas import st_canvas
    import streamlit_drawable_canvas as st_canvas_module
    CANVAS_VERSION = getattr(st_canvas_module, '__version__', 'Unknown')
    logger.info(f"Streamlit Drawable Canvas Version: {CANVAS_VERSION}")
    DRAWABLE_CANVAS_AVAILABLE = True
except ImportError:
    logger.critical("streamlit-drawable-canvas not found. ROI functionality disabled.")
    DRAWABLE_CANVAS_AVAILABLE = False
    st_canvas = None # Define as None for checks later

# Pillow (PIL) - Essential
try:
    from PIL import Image, ImageDraw, UnidentifiedImageError
    import PIL
    PIL_VERSION = getattr(PIL, '__version__', 'Unknown')
    logger.info(f"Pillow (PIL) Version: {PIL_VERSION}")
    PIL_AVAILABLE = True
except ImportError:
    st.error("CRITICAL ERROR: Pillow (PIL) is not installed (`pip install Pillow`). Image processing disabled.")
    logger.critical("Pillow (PIL) not found. App functionality severely impaired.")
    PIL_AVAILABLE = False
    st.stop() # Stop execution if PIL is missing

# Pydicom & related libraries
try:
    import pydicom
    import pydicom.errors
    PYDICOM_VERSION = getattr(pydicom, '__version__', 'Unknown')
    logger.info(f"Pydicom Version: {PYDICOM_VERSION}")
    PYDICOM_AVAILABLE = True
    # Check for optional helpers
    try: import pylibjpeg; logger.info("pylibjpeg found.")
    except ImportError: logger.info("pylibjpeg not found (optional).")
    try: import gdcm; logger.info("python-gdcm found.")
    except ImportError: logger.info("python-gdcm not found (optional).")
except ImportError:
    PYDICOM_VERSION = 'Not Installed'
    logger.warning("pydicom not found. DICOM functionality will be disabled.")
    PYDICOM_AVAILABLE = False

# --- Custom Backend Modules (Crucial Dependencies) ---
# Assume these modules contain the actual AI interaction logic and prompts

try:
    # Assumed to handle DICOM parsing, metadata, and image conversion
    from dicom_utils import (
        parse_dicom, extract_dicom_metadata, dicom_to_image, get_default_wl
    )
    DICOM_UTILS_AVAILABLE = True
    logger.info("dicom_utils imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import dicom_utils: {e}. DICOM features disabled.")
    if PYDICOM_AVAILABLE: # Only warn if pydicom *was* available
        st.warning("DICOM utilities module missing. DICOM processing limited.")
    DICOM_UTILS_AVAILABLE = False

try:
    # **ASSUMPTION:** This module uses responsible, agentic prompts (like examples discussed)
    # for analysis functions, ensuring cautious language, structure, and limitation reporting.
    from llm_interactions import (
        run_initial_analysis, run_multimodal_qa, run_disease_analysis, estimate_ai_confidence
    )
    LLM_INTERACTIONS_AVAILABLE = True
    logger.info("llm_interactions imported successfully.")
except ImportError as e:
    st.error(f"Core AI module (llm_interactions) failed to import: {e}. Analysis functions disabled.")
    logger.critical(f"Failed to import llm_interactions: {e}", exc_info=True)
    LLM_INTERACTIONS_AVAILABLE = False
    st.stop() # Core functionality missing, stop the app

try:
    # **ASSUMPTION:** This module includes disclaimers in the generated PDF.
    from report_utils import generate_pdf_report_bytes
    REPORT_UTILS_AVAILABLE = True
    logger.info("report_utils imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import report_utils: {e}. PDF reporting disabled.")
    REPORT_UTILS_AVAILABLE = False

try:
    # Optional UI helpers
    from ui_components import display_dicom_metadata, dicom_wl_sliders
    UI_COMPONENTS_AVAILABLE = True
    logger.info("ui_components imported successfully.")
except ImportError as e:
    logger.warning(f"Failed to import ui_components: {e}. Using basic UI fallbacks.")
    UI_COMPONENTS_AVAILABLE = False
    def display_dicom_metadata(metadata): st.caption("Metadata Preview:"); st.json(dict(list(metadata.items())[:5])) # Basic fallback
    def dicom_wl_sliders(wc, ww): st.caption("W/L sliders unavailable."); return wc, ww # Basic fallback

# --- HF fallback for Q&A (Optional) ---
try:
    from hf_models import query_hf_vqa_inference_api, HF_VQA_MODEL_ID
    HF_MODELS_AVAILABLE = True
    logger.info(f"hf_models imported successfully (Fallback VQA Model: {HF_VQA_MODEL_ID}).")
except ImportError:
    HF_VQA_MODEL_ID = "hf_model_unavailable"
    HF_MODELS_AVAILABLE = False
    def query_hf_vqa_inference_api(img: Image.Image, question: str, roi: Optional[Dict] = None) -> Tuple[str, bool]:
        return "[Fallback VQA Unavailable] Module not found.", False
    logger.warning("hf_models not found. Fallback VQA disabled.")

# --- Translation Setup ---
try:
    # Assumes translation_models internally uses deep-translator if available
    from translation_models import (
        translate, detect_language, LANGUAGE_CODES, AUTO_DETECT_INDICATOR
    )
    # Check if the underlying library was actually installed
    TRANSLATION_AVAILABLE = DEEP_TRANSLATOR_INSTALLED
    if TRANSLATION_AVAILABLE:
        logger.info("translation_models imported successfully. Translation is available.")
    else:
        logger.error("translation_models imported, but deep-translator is missing. Translation disabled.")
        st.warning("Translation library (deep-translator) is missing or failed to install. Translation features disabled.")
except ImportError as e:
    logger.error(f"Could not import translation_models: {e}. Translation disabled.", exc_info=True)
    TRANSLATION_AVAILABLE = False
    if DEEP_TRANSLATOR_INSTALLED: # If library is there but model import failed
         st.warning(f"Translation module failed to load ({e}). Translation features disabled.")

# Define fallbacks if translation failed
if not TRANSLATION_AVAILABLE:
    translate = None
    detect_language = None
    LANGUAGE_CODES = {"English": "en"} # Minimal fallback
    AUTO_DETECT_INDICATOR = "Auto-Detect"

# --- Custom CSS ---
st.markdown(
    """
    <style>
      /* [Keep existing CSS from previous version] */
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji"; background-color: #f0f2f6; }
      .main .block-container { padding-top: 2rem; padding-bottom: 2rem; padding-left: 1.5rem; padding-right: 1.5rem; }
      .css-1d391kg { background-color: #ffffff; border-right: 1px solid #e0e0e0; } /* Sidebar */
      .stButton>button { border-radius: 8px; padding: 0.5rem 1rem; font-weight: 500; transition: background-color 0.2s, border-color 0.2s; }
      .stButton>button:hover { filter: brightness(95%); }
      div[role="tablist"] { overflow-x: auto; white-space: nowrap; border-bottom: 1px solid #e0e0e0; scrollbar-width: thin; scrollbar-color: #cccccc #f0f2f6; }
      div[role="tablist"]::-webkit-scrollbar { height: 6px; }
      div[role="tablist"]::-webkit-scrollbar-track { background: #f0f2f6; }
      div[role="tablist"]::-webkit-scrollbar-thumb { background-color: #cccccc; border-radius: 10px; border: 2px solid #f0f2f6; }
      footer { text-align: center; font-size: 0.8em; color: #6c757d; margin-top: 2rem; padding: 1rem 0; border-top: 1px solid #e0e0e0; }
      footer a { color: #007bff; text-decoration: none; } footer a:hover { text-decoration: underline; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Display Hero Logo ---
logo_path = os.path.join("assets", "radvisionai-hero.jpeg")
if os.path.exists(logo_path):
    st.image(logo_path, width=350) # Adjust width as needed
else:
    logger.warning(f"Hero logo not found at: {logo_path}. Displaying text title.")
    # Fallback to text if logo missing, can be removed if not desired
    # st.title("⚕️ RadVision AI Advanced") # Already set elsewhere

# --- Initialize Session State Defaults ---
DEFAULT_STATE = {
    "uploaded_file_info": None, "raw_image_bytes": None, "is_dicom": False,
    "dicom_dataset": None, "dicom_metadata": {}, "processed_image": None,
    "display_image": None, "session_id": None, "history": [],
    "initial_analysis": "", "qa_answer": "", "disease_analysis": "",
    "confidence_score": "", # Keeping state key, but UI label changes
    "last_action": None, "pdf_report_bytes": None, "canvas_drawing": None,
    "roi_coords": None, "current_display_wc": None, "current_display_ww": None,
    "clear_roi_feedback": False, "demo_loaded": False,
    "translation_result": None, "translation_error": None,
}
# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.session_id = str(uuid.uuid4())[:8]
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value
    logger.info(f"New session initialized: {st.session_state.session_id}")
# Ensure history is always a list
if not isinstance(st.session_state.get("history"), list):
    st.session_state.history = []

logger.debug(f"Session state verified for session ID: {st.session_state.session_id}")

# --- Utility Functions ---
def format_translation(translated_text: Optional[str]) -> str:
    """Applies basic formatting for readability, handles None."""
    if translated_text is None: return "Translation not available or failed."
    try:
        text_str = str(translated_text)
        # Add line breaks before numbered lists for clarity
        formatted_text = re.sub(r'\s+(\d+\.)', r'\n\n\1', text_str)
        return formatted_text.strip()
    except Exception as e:
        logger.error(f"Error formatting translation: {e}", exc_info=True)
        return str(translated_text) # Return original on error


# --- Sidebar ---
with st.sidebar:
    st.header("⚕️ RadVision Controls")
    st.markdown("---")

    # Tip of the Day
    TIPS = [
        "Tip: Use 'Demo Mode' for a quick look with a sample chest X-ray.",
        "Tip: Draw a rectangle (ROI) on the image to focus the AI's analysis.",
        "Tip: Adjust DICOM Window/Level sliders for better contrast if needed.",
        "Tip: Ask specific questions about the image or findings.",
        "Tip: Generate a PDF report summarizing the AI interaction.",
        "Tip: Use the 'Translation' tab for analysis in other languages.",
        "Tip: Click 'Clear ROI' to make the AI analyze the whole image again.",
    ]
    st.info(f"💡 {random.choice(TIPS)}")
    st.markdown("---")

    # Upload Section
    st.header("Image Upload & Settings")
    st.caption("🔒 Ensure all images are de-identified before uploading.") # PHI Warning
    uploaded_file = st.file_uploader(
        "Upload De-Identified Image (JPG, PNG, DCM)",
        type=["jpg", "jpeg", "png", "dcm", "dicom"],
        key="file_uploader_widget",
        help="Upload a de-identified medical image. DICOM (.dcm) preferred. DO NOT upload identifiable patient data unless permitted by privacy regulations."
    )

    # Demo Mode
    demo_mode = st.checkbox("🚀 Demo Mode", value=st.session_state.get("demo_loaded", False),
                            help="Load a sample chest X-ray image and analysis.")
    # Handle Demo Mode Activation (Example - Adapt path/logic as needed)
    if demo_mode and not st.session_state.demo_loaded:
        logger.info("Demo Mode activated.")
        # --- Placeholder for Demo Loading Logic ---
        # Example: Load a specific demo file, process it, maybe run initial analysis
        try:
            demo_file_path = os.path.join("assets", "demo_chest_xray.dcm") # Adjust path
            if os.path.exists(demo_file_path):
                 with open(demo_file_path, "rb") as f:
                    # Simulate upload process
                    # This part needs to replicate the logic in "File Upload Logic" section below
                    # Reset state, read bytes, determine type, process, set state vars
                    # ... (Add demo file processing logic here) ...
                    st.session_state.demo_loaded = True
                    st.success("Demo image loaded!")
                    st.rerun() # Rerun to reflect loaded state
            else:
                 st.warning("Demo file not found.")
                 st.session_state.demo_loaded = False # Uncheck if file missing
        except Exception as e:
             st.error(f"Error loading demo file: {e}")
             logger.error(f"Demo load error: {e}", exc_info=True)
             st.session_state.demo_loaded = False
        # -----------------------------------------
    elif not demo_mode and st.session_state.demo_loaded:
         logger.info("Demo Mode deactivated.")
         # Reset relevant state if demo is turned off
         # ... (Add reset logic if needed) ...
         st.session_state.demo_loaded = False

    # Clear ROI
    if DRAWABLE_CANVAS_AVAILABLE: # Only show if canvas is available
        if st.button("🗑️ Clear ROI", help="Remove the selected ROI rectangle"):
            st.session_state.roi_coords = None
            st.session_state.canvas_drawing = None # Clear drawing state too
            st.session_state.clear_roi_feedback = True
            logger.info("ROI cleared by user.")
            st.rerun()

        if st.session_state.get("clear_roi_feedback"):
            st.success("✅ ROI cleared!")
            st.session_state.clear_roi_feedback = False # Reset feedback flag

    # DICOM Window/Level Controls
    # Show only if it's DICOM, utils available, and image exists
    if st.session_state.is_dicom and DICOM_UTILS_AVAILABLE and UI_COMPONENTS_AVAILABLE and st.session_state.display_image:
        st.markdown("---")
        st.subheader("DICOM Display (W/L)")
        new_wc, new_ww = dicom_wl_sliders(
            st.session_state.current_display_wc,
            st.session_state.current_display_ww
        )
        # Check if W/L values actually changed
        if new_wc != st.session_state.current_display_wc or new_ww != st.session_state.current_display_ww:
            logger.info(f"DICOM W/L changed via UI: WC={new_wc}, WW={new_ww}")
            st.session_state.current_display_wc = new_wc
            st.session_state.current_display_ww = new_ww
            if st.session_state.dicom_dataset:
                # Update the display image with new W/L settings
                with st.spinner("Applying new Window/Level..."):
                    try:
                        new_display_img = dicom_to_image(
                            st.session_state.dicom_dataset, wc=new_wc, ww=new_ww
                        )
                        if isinstance(new_display_img, Image.Image):
                            # Ensure RGB format for display consistency
                            st.session_state.display_image = new_display_img.convert('RGB') if new_display_img.mode != 'RGB' else new_display_img
                            st.rerun() # Update the image viewer
                        else:
                            st.error("Failed to update DICOM image display.")
                            logger.error("dicom_to_image returned non-image for W/L update.")
                    except Exception as e:
                        st.error(f"Error applying W/L: {e}")
                        logger.error(f"W/L application error: {e}", exc_info=True)
            else:
                st.warning("DICOM dataset unavailable to update W/L.")

    st.markdown("---")
    st.header("🤖 AI Analysis Actions")

    # Disable actions if core AI module missing or no image loaded
    action_disabled = not LLM_INTERACTIONS_AVAILABLE or not isinstance(st.session_state.get("processed_image"), Image.Image)

    # Initial Analysis Button
    if st.button("🔬 Run Structured Initial Analysis", key="analyze_btn", disabled=action_disabled,
                 help="Perform a general, structured analysis (visual description, potential findings, limitations). Assumes backend uses agentic prompt."):
        st.session_state.last_action = "analyze"
        st.rerun()

    # Q&A Section
    st.subheader("❓ Ask AI a Question")
    question_input = st.text_area(
        "Enter your question about the image:",
        height=100, key="question_input_widget",
        placeholder="E.g., 'Describe the findings in the right lower lung zone.' or 'Is there evidence of cardiomegaly?'",
        disabled=action_disabled
    )
    if st.button("💬 Ask Question", key="ask_btn", disabled=action_disabled):
        if question_input.strip():
            st.session_state.last_action = "ask"
            st.rerun()
        else:
            st.warning("Please enter a question first.")

    # Condition-Specific Analysis Section
    st.subheader("🎯 Condition-Specific Analysis")
    # Example conditions - adjust as needed
    DISEASE_OPTIONS = [
        "Pneumonia", "Lung Cancer", "Nodule/Mass", "Effusion", "Fracture", "Stroke",
        "Appendicitis", "Bowel Obstruction", "Cardiomegaly", "Aortic Aneurysm",
        "Pulmonary Embolism", "Tuberculosis", "COVID-19 Findings", "Brain Tumor", "Arthritis",
    ]
    disease_select = st.selectbox(
        "Select condition for focused analysis:",
        options=[""] + sorted(DISEASE_OPTIONS),
        key="disease_select_widget",
        disabled=action_disabled,
        help="AI will analyze the image specifically for signs related to this condition. Assumes backend uses agentic prompt."
    )
    if st.button("🩺 Analyze for Condition", key="disease_btn", disabled=action_disabled):
        if disease_select:
            st.session_state.last_action = "disease"
            st.rerun()
        else:
            st.warning("Please select a condition.")

    st.markdown("---")
    st.header("📊 Reporting & Assessment")

    # Experimental Confidence Score (Renamed and Warned)
    can_estimate = bool(
        st.session_state.history or st.session_state.initial_analysis or st.session_state.disease_analysis
    )
    if st.button("🧪 Estimate LLM Self-Assessment (Experimental)", key="confidence_btn",
                 disabled=not can_estimate or action_disabled,
                 help="EXPERIMENTAL: Ask the LLM to assess its own response based on input. Not a clinical confidence score."):
        st.session_state.last_action = "confidence"
        st.rerun()

    # PDF Report Generation
    report_generation_disabled = action_disabled or not REPORT_UTILS_AVAILABLE
    if st.button("📄 Generate PDF Report Data", key="generate_report_data_btn",
                 disabled=report_generation_disabled,
                 help="Compile analysis results into PDF data. Download button will appear below."):
        st.session_state.last_action = "generate_report_data"
        st.rerun()

    # PDF Download Button (appears after generation)
    if st.session_state.get("pdf_report_bytes"):
        report_filename = f"RadVisionAI_Report_{st.session_state.session_id or 'session'}.pdf"
        st.download_button(
            label="⬇️ Download PDF Report", data=st.session_state.pdf_report_bytes,
            file_name=report_filename, mime="application/pdf",
            key="download_pdf_button", help="Download the generated PDF report."
        )

# --- File Upload Logic ---
if uploaded_file is not None and PIL_AVAILABLE: # Ensure PIL is available
    try:
        # Generate a unique identifier based on file content hash for change detection
        uploaded_file.seek(0)
        file_content_hash = hashlib.sha256(uploaded_file.read()).hexdigest()[:16]
        uploaded_file.seek(0) # Reset pointer
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{file_content_hash}"
    except Exception as e:
        logger.warning(f"Could not generate hash for file '{uploaded_file.name}': {e}")
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{uuid.uuid4().hex[:8]}" # Fallback ID

    # Check if a truly new file has been uploaded
    if new_file_info != st.session_state.get("uploaded_file_info"):
        logger.info(f"New file upload detected: {uploaded_file.name} ({uploaded_file.size} bytes)")
        st.toast(f"Processing '{uploaded_file.name}'...", icon="⏳")

        # --- Reset application state for the new file ---
        # Preserve only essential session info
        keys_to_preserve = {"session_id"} # Keep session ID across uploads
        st.session_state.session_id = st.session_state.get("session_id") or str(uuid.uuid4())[:8] # Ensure ID exists
        # Reset all other keys to defaults
        for key, value in DEFAULT_STATE.items():
            if key not in keys_to_preserve:
                st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value
        st.session_state.uploaded_file_info = new_file_info # Store new file info
        st.session_state.demo_loaded = False # Turn off demo mode on new upload
        # -------------------------------------------------

        st.session_state.raw_image_bytes = uploaded_file.getvalue()
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        # Determine if DICOM based on type/extension and availability of libraries
        st.session_state.is_dicom = (
            PYDICOM_AVAILABLE and DICOM_UTILS_AVAILABLE and
            ("dicom" in uploaded_file.type.lower() or file_ext in (".dcm", ".dicom"))
        )

        with st.spinner("🔬 Analyzing and preparing image..."):
            temp_display_img = None
            temp_processed_img = None # Image potentially pre-processed for LLM
            processing_success = False

            if st.session_state.is_dicom:
                logger.info("Processing as DICOM...")
                try:
                    dicom_dataset = parse_dicom(st.session_state.raw_image_bytes, filename=uploaded_file.name)
                    if dicom_dataset:
                        st.session_state.dicom_dataset = dicom_dataset
                        st.session_state.dicom_metadata = extract_dicom_metadata(dicom_dataset)
                        default_wc, default_ww = get_default_wl(dicom_dataset)
                        st.session_state.current_display_wc = default_wc
                        st.session_state.current_display_ww = default_ww
                        # Get image for display (with default W/L)
                        temp_display_img = dicom_to_image(dicom_dataset, wc=default_wc, ww=default_ww)
                        # Get image potentially processed for AI (e.g., normalized, no W/L)
                        temp_processed_img = dicom_to_image(dicom_dataset, wc=None, ww=None, normalize=True) # Adjust normalization as needed for backend
                        if isinstance(temp_display_img, Image.Image) and isinstance(temp_processed_img, Image.Image):
                            processing_success = True
                            logger.info("DICOM parsed and converted successfully.")
                        else:
                            st.error("Failed to convert DICOM pixel data to a displayable image format.")
                            logger.error("dicom_to_image returned invalid image type(s).")
                    else:
                        st.error("Could not parse DICOM file. It might be corrupted or incomplete.")
                        logger.error("parse_dicom returned None.")
                except pydicom.errors.InvalidDicomError:
                    st.error("Invalid DICOM file format detected. Please upload a valid DICOM (.dcm) file.")
                    logger.error("InvalidDicomError during parsing.")
                    st.session_state.is_dicom = False # Treat as non-DICOM if parsing fails
                except Exception as e:
                    st.error(f"An unexpected error occurred processing DICOM: {e}")
                    logger.error(f"DICOM processing error: {e}", exc_info=True)
                    st.session_state.is_dicom = False # Treat as non-DICOM on error

            # Fallback or primary path for standard image formats
            if not st.session_state.is_dicom and not processing_success:
                logger.info("Processing as standard image (JPG/PNG)...")
                try:
                    raw_img = Image.open(io.BytesIO(st.session_state.raw_image_bytes))
                    # Ensure RGB format for consistency with AI models and display
                    processed_img = raw_img.convert("RGB")
                    temp_display_img = processed_img.copy() # Use the same image for display and processing
                    temp_processed_img = processed_img.copy()
                    processing_success = True
                    logger.info("Standard image loaded and converted to RGB successfully.")
                except UnidentifiedImageError:
                    st.error("Cannot identify image format. Please upload a valid JPG, PNG, or DICOM file.")
                    logger.error(f"UnidentifiedImageError for file: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error processing standard image: {e}")
                    logger.error(f"Standard image processing error: {e}", exc_info=True)

            # Final state update after processing attempt
            if processing_success and isinstance(temp_display_img, Image.Image) and isinstance(temp_processed_img, Image.Image):
                # Ensure display image is RGB before storing
                st.session_state.display_image = temp_display_img.convert('RGB') if temp_display_img.mode != 'RGB' else temp_display_img
                st.session_state.processed_image = temp_processed_img # Store potentially different processed image
                st.success(f"✅ Image '{uploaded_file.name}' loaded successfully!")
                logger.info(f"Image processing complete for: {uploaded_file.name}")
                st.rerun() # Rerun to update the UI with the new image and state
            else:
                # Clear state if processing failed entirely
                st.error("Image loading failed. Please check the file format or try another image.")
                logger.error(f"Image processing failed for file: {uploaded_file.name}")
                st.session_state.uploaded_file_info = None # Clear file info so user can retry
                st.session_state.display_image = None
                st.session_state.processed_image = None
                st.session_state.is_dicom = False
                # No rerun here, let the error message stay

# --- Main Page Content ---

st.markdown("---")
# Moved Title and Disclaimer to the top after imports/config

col1, col2 = st.columns([2, 3], gap="large") # Adjust column ratio and gap as needed

# --- Column 1: Image Viewer & Metadata ---
with col1:
    st.subheader("🖼️ Image Viewer")
    display_img = st.session_state.get("display_image")

    if isinstance(display_img, Image.Image):
        # --- Drawable Canvas for ROI ---
        if DRAWABLE_CANVAS_AVAILABLE and st_canvas:
            st.caption("Draw a rectangle below to select a Region of Interest (ROI).")
            # Dynamic canvas size calculation
            MAX_CANVAS_WIDTH = 600 # Max width for the canvas container
            MAX_CANVAS_HEIGHT = 550 # Max height
            img_w, img_h = display_img.size

            # Basic validation for image dimensions
            if img_w <= 0 or img_h <= 0:
                st.warning("Image has invalid dimensions (<= 0). Cannot display canvas.")
            else:
                aspect_ratio = img_w / img_h
                # Calculate initial canvas size based on width constraint
                canvas_width = min(img_w, MAX_CANVAS_WIDTH)
                canvas_height = int(canvas_width / aspect_ratio)
                # If height exceeds max, recalculate based on height constraint
                if canvas_height > MAX_CANVAS_HEIGHT:
                    canvas_height = MAX_CANVAS_HEIGHT
                    canvas_width = int(canvas_height * aspect_ratio)
                # Ensure minimum practical size
                canvas_width = max(canvas_width, 150)
                canvas_height = max(canvas_height, 150)

                # Display the canvas
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.2)",  # Semi-transparent orange fill
                    stroke_width=2,
                    stroke_color="rgba(239, 83, 80, 0.8)", # Reddish stroke
                    background_image=display_img,
                    update_streamlit=True, # Update Streamlit dynamically on drawing
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="rect", # Only allow rectangles
                    initial_drawing=st.session_state.get("canvas_drawing", None), # Persist drawing state
                    key="drawable_canvas" # Unique key
                )

                # Process canvas results to extract ROI coordinates
                if canvas_result.json_data and canvas_result.json_data.get("objects"):
                    # Get the last drawn rectangle (assuming user draws one ROI)
                    last_object = canvas_result.json_data["objects"][-1]
                    if last_object["type"] == "rect":
                        # Extract coordinates from canvas (scaled relative to canvas size)
                        canvas_left = int(last_object["left"])
                        canvas_top = int(last_object["top"])
                        # Account for potential scaling within the canvas object itself
                        canvas_width_scaled = int(last_object["width"] * last_object.get("scaleX", 1))
                        canvas_height_scaled = int(last_object["height"] * last_object.get("scaleY", 1))

                        # Calculate scaling factors from original image to canvas size
                        scale_x = img_w / canvas_width
                        scale_y = img_h / canvas_height

                        # Convert canvas coordinates back to original image coordinates
                        original_left = int(canvas_left * scale_x)
                        original_top = int(canvas_top * scale_y)
                        original_width = int(canvas_width_scaled * scale_x)
                        original_height = int(canvas_height_scaled * scale_y)

                        # Ensure coordinates are within image bounds
                        original_left = max(0, original_left)
                        original_top = max(0, original_top)
                        original_width = min(img_w - original_left, original_width)
                        original_height = min(img_h - original_top, original_height)

                        # Store the ROI if it's valid and different from the last one
                        new_roi = {
                            "left": original_left, "top": original_top,
                            "width": original_width, "height": original_height
                        }
                        if st.session_state.roi_coords != new_roi and original_width > 0 and original_height > 0:
                            st.session_state.roi_coords = new_roi
                            st.session_state.canvas_drawing = canvas_result.json_data # Save drawing state
                            logger.info(f"New ROI selected (original coords): {new_roi}")
                            # Use st.toast for less intrusive feedback
                            st.toast(f"ROI set: ({original_left},{original_top}), {original_width}x{original_height}", icon="🎯")
                            # No rerun needed here as update_streamlit=True handles it

        else: # Fallback if canvas is not available
            st.image(display_img, caption="Image Preview (ROI drawing disabled)", use_container_width=True)

        # Display current ROI coordinates if set
        if st.session_state.roi_coords:
            roi = st.session_state.roi_coords
            st.caption(f"Current ROI: ({roi['left']}, {roi['top']}) Size: {roi['width']}x{roi['height']}")
        else:
            st.caption("No ROI selected. Analysis will cover the entire image.")

        st.markdown("---") # Separator

        # Display DICOM Metadata Expander
        if st.session_state.is_dicom and st.session_state.dicom_metadata:
            with st.expander("📄 View DICOM Metadata", expanded=False):
                if UI_COMPONENTS_AVAILABLE:
                    display_dicom_metadata(st.session_state.dicom_metadata)
                else: # Basic fallback if ui_components module failed
                    st.json(st.session_state.dicom_metadata)
        elif st.session_state.is_dicom:
            # Case where it's DICOM but metadata extraction failed or was empty
            st.caption("DICOM file loaded, but metadata could not be extracted or is empty.")

    elif uploaded_file is not None: # If upload happened but display_img is None
        st.error("Image preview failed. The file might be corrupted or unsupported after upload.")
    else: # Default message when no image is loaded
        st.info("⬅️ Please upload a de-identified image or enable Demo Mode in the sidebar.")

# --- Column 2: Analysis Results & Interaction Tabs ---
with col2:
    st.subheader("📊 Analysis & Interaction")
    tab_titles = [
        "🔬 Structured Analysis", # Renamed
        "💬 Q&A History",
        "🩺 Condition Focus",
        "🧪 LLM Self-Assessment", # Renamed
        "🌐 Translation"
    ]
    tabs = st.tabs(tab_titles)

    # Tab 1: Initial Analysis
    with tabs[0]:
        st.caption("Displays the AI's general structured analysis of the image/ROI.")
        analysis_text = st.session_state.initial_analysis or "Run 'Structured Initial Analysis' from the sidebar."
        # Use markdown to render potentially structured output from AI
        st.markdown(analysis_text)
        # Use a disabled text area as a container if markdown isn't enough or preferred
        # st.text_area("Findings & Impressions", value=analysis_text, height=450, disabled=True, key="initial_analysis_display")

    # Tab 2: Q&A History
    with tabs[1]:
        st.caption("Shows the latest answer and full conversation history.")
        st.markdown("**Latest AI Answer:**")
        latest_answer = st.session_state.qa_answer or "_Ask a question using the sidebar controls._"
        st.markdown(latest_answer) # Display latest answer using markdown
        # st.text_area("Latest AI Answer", value=latest_answer, height=200, disabled=True, key="qa_answer_display")
        st.markdown("---")
        # Display full history in an expander
        if st.session_state.history:
            with st.expander("Full Conversation History", expanded=True):
                # Display history chronologically (newest at the bottom typical for chat)
                for i, (q_type, message) in enumerate(st.session_state.history):
                    if q_type.lower() == "user question":
                        st.markdown(f"**You:** {message}")
                    elif q_type.lower() == "ai answer":
                        st.markdown(f"**AI:** {message}")
                    elif "[fallback]" in q_type.lower(): # Handle fallback display
                        st.markdown(f"**AI (Fallback):** {message.split('**')[-1]}") # Extract message part
                    elif q_type.lower() == "system":
                        st.info(f"*{message}*", icon="ℹ️")
                    else: # General case
                        st.markdown(f"**{q_type}:** {message}")
                    if i < len(st.session_state.history) - 1:
                        st.markdown("---") # Separator between messages
        else:
            st.caption("No questions asked in this session yet.")

    # Tab 3: Condition Focus
    with tabs[2]:
        st.caption("Displays the AI's analysis focused on the selected condition.")
        condition_text = st.session_state.disease_analysis or "Select a condition and run 'Analyze for Condition' from the sidebar."
        st.markdown(condition_text) # Use markdown for display
        # st.text_area("Condition-Specific Analysis", value=condition_text, height=450, disabled=True, key="disease_analysis_display")

    # Tab 4: LLM Self-Assessment (Experimental)
    with tabs[3]:
        st.caption("EXPERIMENTAL: Displays the AI's self-assessment score. Not clinical confidence.")
        st.warning("""
            **⚠️ Important Note:** This score reflects the AI model's internal assessment based on its training and the current interaction context.
            It is **highly experimental** and **DOES NOT represent clinical certainty or diagnostic accuracy.**
            Use this score for informational insight into the AI's perspective only, and **treat it with extreme caution.**
        """, icon="🧪")
        confidence_text = st.session_state.confidence_score or "Run 'Estimate LLM Self-Assessment' from the sidebar after performing analysis."
        st.markdown(confidence_text) # Use markdown for display
        # st.text_area("LLM Self-Assessment Score (Experimental)", value=confidence_text, height=350, disabled=True, key="confidence_display")

    # Tab 5: Translation
    with tabs[4]:
        st.subheader("🌐 Translate Analysis Text")
        if not TRANSLATION_AVAILABLE:
            st.warning("Translation features are unavailable. The required 'deep-translator' library might be missing or failed to install.", icon="🚫")
        else:
            st.caption("Select analysis text, choose languages, and click 'Translate'.")
            # Populate options dynamically based on available analysis results
            text_options = {
                "Structured Initial Analysis": st.session_state.initial_analysis,
                "Latest Q&A Answer": st.session_state.qa_answer,
                "Condition Analysis": st.session_state.disease_analysis,
                "LLM Self-Assessment": st.session_state.confidence_score,
                "(Enter Custom Text Below)": "" # Option for manual input
            }
            # Filter out options with no text yet, always include custom entry
            available_labels = [label for label, txt in text_options.items() if txt or label == "(Enter Custom Text Below)"]
            if not available_labels: available_labels = ["(Enter Custom Text Below)"] # Ensure custom is always there

            selected_label = st.selectbox(
                "Select text to translate:", options=available_labels, index=0,
                key="translate_source_select"
            )
            text_to_translate = text_options.get(selected_label, "")

            # Allow user to input custom text if that option is selected
            if selected_label == "(Enter Custom Text Below)":
                text_to_translate = st.text_area(
                    "Enter text to translate:", value="", height=150,
                    key="translate_custom_input"
                )

            # Display the text that will be translated (read-only)
            st.text_area(
                "Text selected/entered for translation:", value=text_to_translate, height=100,
                disabled=True, key="translate_preview"
            )

            # Language selection dropdowns
            col_lang1, col_lang2 = st.columns(2)
            with col_lang1:
                source_language_options = [AUTO_DETECT_INDICATOR] + sorted(list(LANGUAGE_CODES.keys()))
                source_language_name = st.selectbox(
                    "Source Language:", source_language_options, index=0,
                    key="translate_source_lang"
                )
            with col_lang2:
                target_language_options = sorted(list(LANGUAGE_CODES.keys()))
                # Try to default target to Spanish or English if available
                default_target_index = 0
                preferred_targets = ["Spanish", "English"]
                for target in preferred_targets:
                    if target in target_language_options:
                        default_target_index = target_language_options.index(target)
                        break
                target_language_name = st.selectbox(
                    "Translate To:", target_language_options, index=default_target_index,
                    key="translate_target_lang"
                )

            # Translate Button
            if st.button("🔄 Translate Now", key="translate_button"):
                st.session_state.translation_result = None # Clear previous results
                st.session_state.translation_error = None

                if not text_to_translate or not text_to_translate.strip():
                    st.warning("Please select or enter text to translate first.", icon="☝️")
                    st.session_state.translation_error = "Input text is empty."
                # Avoid translating if source and target are identical (and not auto-detect)
                elif source_language_name == target_language_name and source_language_name != AUTO_DETECT_INDICATOR:
                    st.info("Source and target languages are the same. No translation performed.", icon="✅")
                    st.session_state.translation_result = text_to_translate
                else:
                    # Perform translation using the backend function
                    with st.spinner(f"Translating from '{source_language_name}' to '{target_language_name}'..."):
                        try:
                            # Call the translation utility function
                            translation_output = translate(
                                text=text_to_translate,
                                target_language=target_language_name,
                                source_language=source_language_name
                            )
                            if translation_output is not None:
                                st.session_state.translation_result = translation_output
                                st.success("Translation complete!", icon="🎉")
                            else:
                                # Handle cases where translation returns None without an exception
                                st.error("Translation service returned an empty result. Please check the input or try again.", icon="❓")
                                logger.warning("Translation function returned None.")
                                st.session_state.translation_error = "Translation service returned no result."
                        except Exception as e:
                            # Catch potential errors from the translation library/service
                            st.error(f"Translation failed: {e}", icon="❌")
                            logger.error(f"Translation error during execution: {e}", exc_info=True)
                            st.session_state.translation_error = str(e)

            # Display Translation Result or Error
            if st.session_state.get("translation_result"):
                formatted_result = format_translation(st.session_state.translation_result)
                st.text_area("Translated Text:", value=formatted_result, height=200, key="translation_output_display")
            elif st.session_state.get("translation_error"):
                # Show error message if translation failed
                st.info(f"Translation Error: {st.session_state.translation_error}", icon="ℹ️")


# --- Button Action Handlers (Centralized Logic) ---
# This block runs *after* the main UI is drawn, responding to button clicks stored in last_action

current_action = st.session_state.get("last_action")
if current_action:
    logger.info(f"Handling action: '{current_action}' for session: {st.session_state.session_id}")

    # --- Pre-Action Checks ---
    action_requires_image = current_action in ["analyze", "ask", "disease", "confidence", "generate_report_data"] # Confidence might use image context too
    action_requires_llm = current_action in ["analyze", "ask", "disease", "confidence"]
    action_requires_report_util = (current_action == "generate_report_data")
    error_occurred = False

    if action_requires_image and not isinstance(st.session_state.get("processed_image"), Image.Image):
        st.error(f"Cannot perform '{current_action}': No valid image loaded.", icon="🖼️")
        error_occurred = True
    if not st.session_state.session_id:
        st.error("Critical Error: Session ID is missing. Please refresh.", icon="🆔")
        error_occurred = True
    if action_requires_llm and not LLM_INTERACTIONS_AVAILABLE:
        st.error(f"Cannot perform '{current_action}': Core AI interaction module is unavailable.", icon="🤖")
        error_occurred = True
    if action_requires_report_util and not REPORT_UTILS_AVAILABLE:
        st.error(f"Cannot perform '{current_action}': Report generation module is unavailable.", icon="📄")
        error_occurred = True

    if error_occurred:
        st.session_state.last_action = None # Reset action if checks fail
        st.stop() # Prevent further execution in this run

    # --- Execute Action ---
    img_for_llm = st.session_state.processed_image
    roi_coords = st.session_state.roi_coords
    current_history = st.session_state.history # Assumed to be a list

    try:
        if current_action == "analyze":
            st.toast("🔬 Performing initial structured analysis...", icon="⏳")
            with st.spinner("AI analyzing image structure and findings..."):
                # **ASSUMPTION:** run_initial_analysis uses a responsible, agentic prompt
                analysis_result = run_initial_analysis(img_for_llm, roi=roi_coords)
            st.session_state.initial_analysis = analysis_result
            # Clear other analysis fields when running initial analysis
            st.session_state.qa_answer = ""
            st.session_state.disease_analysis = ""
            logger.info("Initial analysis action completed.")
            st.success("Initial structured analysis complete!", icon="✅")

        elif current_action == "ask":
            question_text = st.session_state.question_input_widget.strip()
            if not question_text:
                st.warning("Question input was empty.", icon="❓")
            else:
                st.toast(f"Asking AI: '{question_text[:50]}...'")
                st.session_state.qa_answer = "" # Clear previous answer
                with st.spinner("AI formulating answer..."):
                    # **ASSUMPTION:** run_multimodal_qa handles context appropriately
                    answer, success_flag = run_multimodal_qa(
                        img_for_llm, question_text, current_history, roi=roi_coords
                    )
                if success_flag:
                    st.session_state.qa_answer = answer
                    # Append interaction to history
                    st.session_state.history.append(("User Question", question_text))
                    st.session_state.history.append(("AI Answer", answer))
                    st.success("AI answered your question!", icon="💬")
                else:
                    # --- Primary AI Failure + Fallback Logic ---
                    primary_error_msg = f"Primary AI failed to answer: {answer}"
                    st.session_state.qa_answer = primary_error_msg # Show primary error first
                    st.error(primary_error_msg, icon="⚠️")
                    logger.warning(f"Primary Q&A failed: {answer}")

                    hf_token = os.environ.get("HF_API_TOKEN") or st.secrets.get("HF_API_TOKEN") # Check secrets too
                    if HF_MODELS_AVAILABLE and hf_token:
                        st.info(f"Attempting fallback VQA with Hugging Face model: {HF_VQA_MODEL_ID}", icon="🔄")
                        with st.spinner(f"Trying fallback model ({HF_VQA_MODEL_ID})..."):
                            try:
                                fallback_answer, fallback_success = query_hf_vqa_inference_api(
                                    img_for_llm, question_text, roi=roi_coords
                                )
                            except Exception as hf_e:
                                fallback_success = False
                                fallback_answer = f"Error during fallback query: {hf_e}"
                                logger.error(f"Fallback VQA query error: {hf_e}", exc_info=True)

                        if fallback_success:
                            fallback_display = f"**[Fallback: {HF_VQA_MODEL_ID}]**\n\n{fallback_answer}"
                            st.session_state.qa_answer += f"\n\n---\n\n{fallback_display}" # Append fallback result
                            st.session_state.history.append(("[Fallback] User Question", question_text))
                            st.session_state.history.append(("[Fallback] AI Answer", fallback_display))
                            st.success("Fallback AI provided an answer.", icon="👍")
                        else:
                            fallback_error_msg = f"[Fallback Error - {HF_VQA_MODEL_ID}]: {fallback_answer}"
                            st.session_state.qa_answer += f"\n\n---\n\n{fallback_error_msg}" # Append fallback error
                            st.error("Fallback AI also failed.", icon="👎")
                            logger.warning(f"Fallback VQA failed: {fallback_answer}")
                    elif HF_MODELS_AVAILABLE and not hf_token:
                        # Only warn if module exists but token is missing
                        no_token_msg = "[Fallback Skipped: Hugging Face API Token (HF_API_TOKEN) not configured in environment/secrets]"
                        st.session_state.qa_answer += f"\n\n---\n\n{no_token_msg}"
                        st.warning("Hugging Face API token needed for fallback VQA.", icon="🔑")
                    else:
                        # If module itself is unavailable
                        no_fallback_msg = "[Fallback VQA Unavailable]"
                        st.session_state.qa_answer += f"\n\n---\n\n{no_fallback_msg}"
                        # Optional: Add a less prominent warning if desired
                        # st.caption("Note: Fallback Q&A model is not configured.")
                # -----------------------------------------

        elif current_action == "disease":
            selected_disease = st.session_state.disease_select_widget
            if not selected_disease:
                st.warning("No condition was selected.", icon="🏷️")
            else:
                st.toast(f"🩺 Analyzing image specifically for '{selected_disease}'...", icon="⏳")
                with st.spinner(f"AI analyzing for signs of {selected_disease}..."):
                    # **ASSUMPTION:** run_disease_analysis uses a responsible, agentic prompt
                    disease_result = run_disease_analysis(img_for_llm, selected_disease, roi=roi_coords)
                st.session_state.disease_analysis = disease_result
                st.session_state.qa_answer = "" # Clear Q&A answer
                logger.info(f"Disease-specific analysis action for '{selected_disease}' completed.")
                st.success(f"Analysis focused on '{selected_disease}' complete!", icon="✅")

        elif current_action == "confidence":
            # Check if there's anything to base confidence on
            if not (current_history or st.session_state.initial_analysis or st.session_state.disease_analysis):
                st.warning("Please perform at least one analysis (Initial, Q&A, or Condition) before estimating assessment.", icon="📊")
            else:
                st.toast("🧪 Estimating LLM self-assessment (Experimental)...", icon="⏳")
                with st.spinner("AI assessing its previous responses..."):
                    # **ASSUMPTION:** estimate_ai_confidence explains its basis and limitations
                    confidence_result = estimate_ai_confidence(
                        img_for_llm, history=current_history,
                        initial_analysis=st.session_state.initial_analysis,
                        disease_analysis=st.session_state.disease_analysis,
                        roi=roi_coords
                    )
                st.session_state.confidence_score = confidence_result
                st.success("LLM self-assessment estimation complete!", icon="✅")

        elif current_action == "generate_report_data":
            st.toast("📄 Generating PDF report data...", icon="⏳")
            st.session_state.pdf_report_bytes = None # Clear previous
            image_for_report = st.session_state.get("display_image") # Use the display image for context

            if not isinstance(image_for_report, Image.Image):
                st.error("Cannot generate report: No valid image currently loaded.", icon="🖼️")
            else:
                # Prepare image for PDF (copy, ensure RGB, draw ROI if present)
                final_image_for_pdf = image_for_report.copy().convert("RGB")
                if roi_coords:
                    try:
                        draw = ImageDraw.Draw(final_image_for_pdf)
                        x0, y0 = roi_coords['left'], roi_coords['top']
                        x1, y1 = x0 + roi_coords['width'], y0 + roi_coords['height']
                        # Draw a noticeable rectangle
                        draw.rectangle(
                            [x0, y0, x1, y1], outline="red",
                            width=max(3, int(min(final_image_for_pdf.size) * 0.005)) # Scale width slightly
                        )
                        logger.info("ROI bounding box drawn onto image for PDF report.")
                    except Exception as draw_e:
                        logger.error(f"Error drawing ROI box on PDF image: {draw_e}", exc_info=True)
                        st.warning("Could not draw the ROI box on the report image.", icon="✏️")

                # Format history for the report
                formatted_history = "No Q&A interactions recorded for this session."
                if current_history:
                    lines = []
                    for q_type, msg in current_history:
                        # Basic cleaning (remove potential HTML if accidentally included)
                        cleaned_msg = re.sub('<[^<]+?>', '', str(msg)).strip()
                        lines.append(f"[{q_type}]:\n{cleaned_msg}")
                    formatted_history = "\n\n---\n\n".join(lines)

                # Gather all data for the report
                report_data = {
                    "Session ID": st.session_state.session_id,
                    "Image Filename": (st.session_state.uploaded_file_info or "N/A").split('-')[0], # Extract original filename
                    "Structured Initial Analysis": st.session_state.initial_analysis or "Not Performed",
                    "Q&A History": formatted_history,
                    "Condition Specific Analysis": st.session_state.disease_analysis or "Not Performed",
                    "LLM Self-Assessment (Experimental)": st.session_state.confidence_score or "Not Performed",
                }
                # Add DICOM summary if available
                if st.session_state.is_dicom and st.session_state.dicom_metadata:
                    # Select key DICOM tags for summary
                    meta_summary = {
                        tag: st.session_state.dicom_metadata.get(tag, "N/A")
                        for tag in ['PatientName', 'PatientID', 'StudyDate', 'Modality', 'StudyDescription', 'InstitutionName']
                        if tag in st.session_state.dicom_metadata # Check if tag exists
                    }
                    if meta_summary:
                        lines = [f"{k.replace('PatientName', 'Patient Name').replace('PatientID', 'Patient ID').replace('StudyDate', 'Study Date').replace('StudyDescription', 'Study Desc.')}: {v}" for k, v in meta_summary.items()]
                        report_data["DICOM Summary"] = "\n".join(lines)

                # Generate the PDF bytes using the backend utility
                with st.spinner("Compiling PDF report..."):
                    # **ASSUMPTION:** generate_pdf_report_bytes includes necessary disclaimers
                    pdf_bytes = generate_pdf_report_bytes(
                        session_id=st.session_state.session_id,
                        image=final_image_for_pdf, # Image with ROI drawn if applicable
                        analysis_outputs=report_data,
                        dicom_metadata=st.session_state.dicom_metadata if st.session_state.is_dicom else None
                    )

                if pdf_bytes:
                    st.session_state.pdf_report_bytes = pdf_bytes
                    st.success("PDF report data generated! Download button available in the sidebar.", icon="📄")
                    logger.info("PDF report generation successful.")
                    st.balloons() # Fun feedback!
                else:
                    st.error("Failed to generate PDF report data. Check logs for details.", icon="❌")
                    logger.error("PDF generation function returned None or empty bytes.")

        else:
            st.warning(f"Unknown action '{current_action}' was triggered. No operation performed.", icon="❓")
            logger.warning(f"Unhandled action '{current_action}' encountered.")

    except Exception as e:
        # General catch-all for errors during action execution
        st.error(f"An unexpected error occurred while processing '{current_action}': {e}", icon="💥")
        logger.critical(f"Error during action '{current_action}': {e}", exc_info=True)

    finally:
        # --- Post-Action Cleanup ---
        st.session_state.last_action = None # IMPORTANT: Reset the action trigger
        logger.debug(f"Action '{current_action}' processing finished.")
        # Rerun to update the UI state based on the action's results
        st.rerun()

# --- Footer ---
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced | Session ID: {st.session_state.get('session_id', 'N/A')}")
st.markdown(
    """
    <footer>
      <p>RadVision AI is intended for research and educational demonstration purposes only. It is not a medical device and cannot substitute for professional medical evaluation.</p>
      <p>Ensure compliance with all applicable privacy laws (e.g., HIPAA, GDPR) when using this tool.</p>
      <!-- Add actual links if desired -->
      <!-- <p><a href="#" target="_blank">Privacy Policy</a> | <a href="#" target="_blank">Terms of Service</a></p> -->
    </footer>
    """,
    unsafe_allow_html=True
)
logger.info(f"--- Application render cycle complete for session: {st.session_state.session_id} ---")