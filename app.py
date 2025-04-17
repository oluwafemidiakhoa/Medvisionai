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
import hashlib    # Added for file content hashing
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple, List, Union
import copy
import random      # For Tip of the Day
import re         # For formatting the translation output

# --- Ensure deep-translator is installed at runtime if not present ---
# This fixes "Translation library (deep-translator) not found" on Spaces
# Note: This approach might have limitations in restricted environments.
try:
    from deep_translator import GoogleTranslator
except ImportError:
    st.warning("Trying to install 'deep-translator' for translation features...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "deep-translator"])
        from deep_translator import GoogleTranslator
        st.success("'deep-translator' installed successfully.")
    except Exception as e:
        # If fallback also fails, log it; we'll gracefully disable translation below.
        print(f"CRITICAL: Could not install deep-translator: {e}")
        st.error(f"Failed to install 'deep-translator'. Translation disabled. Error: {e}")


# --- Logging Setup (Early) ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
# Adjust logging for Streamlit Cloud if needed
log_format = '%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'
# Remove existing handlers if any to avoid duplicate logs, especially in Streamlit reruns
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=LOG_LEVEL,
    format=log_format,
    datefmt=date_format,
    stream=sys.stdout # Ensure logs go to stdout for Cloud environments
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
    st_canvas = None # Define as None so later checks don't raise NameError

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
    Image = None # Define fallbacks to prevent NameErrors later
    ImageDraw = None
    UnidentifiedImageError = Exception # Fallback exception type
    st.stop() # Stop execution if Pillow is missing

# --- pydicom & DICOM libraries ---
try:
    import pydicom
    import pydicom.errors
    import pydicom.valuerep # Needed for specific value representations
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
        import pylibjpeg # For JPEG compressed DICOMs
        logger.info("pylibjpeg found (for extended DICOM decompression).")
    except ImportError:
        logger.info("pylibjpeg not found. Some DICOM compression syntaxes may not be supported.")
    try:
        import gdcm # Alternative/complementary DICOM library
        logger.info("python-gdcm found (for improved DICOM compatibility).")
    except ImportError:
        logger.info("python-gdcm not found. Some DICOM functionalities may be reduced.")

# --- Custom Utilities & Backend Modules ---
try:
    from dicom_utils import (
        parse_dicom,
        extract_dicom_metadata,
        dicom_to_image,
        get_default_wl # Use this for initial W/L
    )
    DICOM_UTILS_AVAILABLE = True
    logger.info("dicom_utils imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import dicom_utils: {e}. DICOM features disabled.")
    DICOM_UTILS_AVAILABLE = False
    # Define dummy functions if module fails to load
    def parse_dicom(*args, **kwargs): return None
    def extract_dicom_metadata(*args, **kwargs): return {}
    def dicom_to_image(*args, **kwargs): return None
    def get_default_wl(*args, **kwargs): return (None, None)

try:
    from llm_interactions import (
        run_initial_analysis,
        run_multimodal_qa,
        run_disease_analysis,
        estimate_ai_confidence
    )
    LLM_INTERACTIONS_AVAILABLE = True
    logger.info("llm_interactions imported successfully.")
except ImportError as e:
    st.error(f"Core AI module (llm_interactions) failed to import: {e}. Analysis functions disabled.")
    logger.critical(f"Failed to import llm_interactions: {e}", exc_info=True)
    LLM_INTERACTIONS_AVAILABLE = False
    # Define dummy functions
    def run_initial_analysis(*args, **kwargs): return "Error: AI Module Unavailable"
    def run_multimodal_qa(*args, **kwargs): return ("Error: AI Module Unavailable", False)
    def run_disease_analysis(*args, **kwargs): return "Error: AI Module Unavailable"
    def estimate_ai_confidence(*args, **kwargs): return "Error: AI Module Unavailable"
    # Consider st.stop() here if LLM is absolutely critical

try:
    from report_utils import generate_pdf_report_bytes
    REPORT_UTILS_AVAILABLE = True
    logger.info("report_utils imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import report_utils: {e}. PDF reporting disabled.")
    REPORT_UTILS_AVAILABLE = False
    def generate_pdf_report_bytes(*args, **kwargs): return None

try:
    # Import all necessary UI components, including the new UMLS one
    from ui_components import display_dicom_metadata, dicom_wl_sliders, display_umls_concepts
    UI_COMPONENTS_AVAILABLE = True
    logger.info("ui_components imported successfully.")
except ImportError as e:
    logger.warning(f"Failed to import ui_components: {e}. Custom UI elements might be missing or use fallbacks.")
    UI_COMPONENTS_AVAILABLE = False
    # Define simple fallbacks for missing UI components
    def display_dicom_metadata(metadata): st.caption("Metadata display unavailable.")
    def dicom_wl_sliders(ds, current_wc, current_ww):
        st.caption("W/L sliders unavailable.")
        return current_wc, current_ww
    def display_umls_concepts(concepts): st.caption("UMLS display unavailable.")

# --- HF fallback for Q&A (Optional) ---
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

# --- Translation Setup ---
try:
    from translation_models import (
        translate, detect_language, LANGUAGE_CODES, AUTO_DETECT_INDICATOR
    )
    TRANSLATION_AVAILABLE = True
    logger.info("translation_models imported successfully. Translation is available.")
except ImportError as e:
    st.warning(f"Translation features disabled: Could not import translation_models: {e}")
    logger.error(f"Could not import translation_models: {e}", exc_info=True)
    TRANSLATION_AVAILABLE = False
    # Minimal fallbacks
    def translate(*args, **kwargs): return "Translation Error: Module Unavailable"
    def detect_language(*args, **kwargs): return "en" # Default to english
    LANGUAGE_CODES = {"English": "en"}
    AUTO_DETECT_INDICATOR = "Auto-Detect"

# --- UMLS Integration ---
try:
    import umls_utils
    # Import the specific exception and dataclass for type checking and handling
    from umls_utils import UMLSAuthError, UMLSConcept
    UMLS_APIKEY = os.getenv("UMLS_APIKEY") # Get API key from environment variables
    if not UMLS_APIKEY:
        logger.warning("UMLS_APIKEY environment variable not set. UMLS features will be disabled.")
        UMLS_AVAILABLE = False
    else:
        logger.info("umls_utils imported successfully and UMLS_APIKEY found.")
        UMLS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import umls_utils: {e}. UMLS features disabled.")
    UMLS_AVAILABLE = False
    # Define fallbacks for robustness if import fails
    umls_utils = None
    UMLSAuthError = RuntimeError # Fallback exception type
    UMLSConcept = None
    UMLS_APIKEY = None
except Exception as e: # Catch any other unexpected error during setup
    logger.error(f"Error during UMLS setup: {e}", exc_info=True)
    UMLS_AVAILABLE = False
    umls_utils = None
    UMLSAuthError = RuntimeError
    UMLSConcept = None
    UMLS_APIKEY = None


# --- Custom CSS for Polished Look & Tab Scrolling ---
# (Keep existing CSS as is)
st.markdown(
    """
    <style>
      body {
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
          background-color: #f0f2f6;
      }
      .main .block-container {
          padding-top: 2rem;
          padding-bottom: 2rem;
          padding-left: 1.5rem;
          padding-right: 1.5rem;
      }
      /* Sidebar Styling (Example) */
      .css-1d391kg { /* Specific class for Streamlit sidebar, might change */
          background-color: #ffffff;
          border-right: 1px solid #e0e0e0;
      }
      .stButton>button {
          border-radius: 8px;
          padding: 0.5rem 1rem;
          font-weight: 500;
          transition: background-color 0.2s ease-in-out, border-color 0.2s ease-in-out;
          width: 100%; /* Make buttons fill sidebar width */
          margin-bottom: 0.5rem; /* Add some space between buttons */
      }
      .stButton>button:hover {
        filter: brightness(95%);
      }
      /* Tab Scrolling */
      div[role="tablist"] {
          overflow-x: auto;
          white-space: nowrap;
          border-bottom: 1px solid #e0e0e0;
          scrollbar-width: thin; /* For Firefox */
          scrollbar-color: #cccccc #f0f2f6; /* For Firefox */
      }
      /* For Chrome, Edge, Safari */
      div[role="tablist"]::-webkit-scrollbar {
          height: 6px;
      }
      div[role="tablist"]::-webkit-scrollbar-track {
          background: #f0f2f6;
      }
      div[role="tablist"]::-webkit-scrollbar-thumb {
          background-color: #cccccc;
          border-radius: 10px;
          border: 2px solid #f0f2f6;
      }
      /* Footer Styling */
      footer {
          text-align: center;
          font-size: 0.8em;
          color: #6c757d;
          margin-top: 2rem;
          padding: 1rem 0;
          border-top: 1px solid #e0e0e0;
      }
      footer a {
          color: #007bff;
          text-decoration: none;
      }
      footer a:hover {
          text-decoration: underline;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Display Hero Logo ---
logo_path = os.path.join("assets", "radvisionai-hero.jpeg")
if os.path.exists(logo_path):
    # Use columns to center the logo slightly better if desired
    # col1_logo, col2_logo, col3_logo = st.columns([1,2,1])
    # with col2_logo:
    #     st.image(logo_path, width=350)
    st.image(logo_path, width=350) # Simpler layout
else:
    logger.warning(f"Hero logo not found at: {logo_path}")
    # Don't show a warning in the UI unless debugging
    # st.warning("Hero logo (radvisionai-hero.jpeg) not found in 'assets' folder.")

# --- Initialize Session State Defaults ---
DEFAULT_STATE = {
    "uploaded_file_info": None, # Stores "name-size-hash" to detect new uploads
    "raw_image_bytes": None,    # Raw bytes of the uploaded file
    "is_dicom": False,          # Flag indicating if the current file is DICOM
    "dicom_dataset": None,      # The parsed pydicom dataset object
    "dicom_metadata": {},       # Extracted key-value DICOM metadata for display
    "processed_image": None,    # PIL Image object ready for AI model input (e.g., normalized)
    "display_image": None,      # PIL Image object for display (potentially with W/L applied)
    "session_id": None,         # Unique ID for the current session
    "history": [],              # List to store Q&A interactions [(type, message), ...]
    "initial_analysis": "",     # Stores the result of the initial analysis action
    "qa_answer": "",            # Stores the latest answer from the Q&A action
    "disease_analysis": "",     # Stores the result of the disease-specific analysis
    "confidence_score": "",     # Stores the AI confidence estimation result
    "last_action": None,        # Tracks the last button clicked to handle actions post-rerun
    "pdf_report_bytes": None,   # Stores the generated PDF report bytes
    "canvas_drawing": None,     # Stores the JSON state of the drawable canvas (for ROI persistence)
    "roi_coords": None,         # Stores the calculated ROI coordinates {left, top, width, height}
    "current_display_wc": None, # Current window center for DICOM display
    "current_display_ww": None, # Current window width for DICOM display
    "clear_roi_feedback": False,# Flag to show feedback after clearing ROI
    "demo_loaded": False,       # Flag indicating if demo mode is active
    "translation_result": None, # Stores the result of the last translation
    "translation_error": None,  # Stores any error message from translation
    "umls_search_term": "",     # Stores the term entered for UMLS search
    "umls_results": None,       # Stores the list of UMLSConcept results from search
    "umls_error": None,         # Stores any error message from UMLS search
}

# Initialize session state using the defaults
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
    logger.info(f"New session started: {st.session_state.session_id}")

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        # Use deepcopy for mutable defaults like lists/dicts to avoid shared state issues
        st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value

# Ensure history is always a list
if not isinstance(st.session_state.get("history"), list):
    st.session_state.history = []

logger.debug(f"Session state keys initialized/verified for session ID: {st.session_state.session_id}")


# --- Helper function for formatting translation ---
def format_translation(translated_text: Optional[str]) -> str:
    """
    Applies basic formatting to translated text, primarily for readability.
    Handles potential None input gracefully.
    """
    if translated_text is None:
        return "Translation not available or failed."
    try:
        text_str = str(translated_text)
        # Basic formatting: Add newlines before numbered lists if preceded by space
        formatted_text = re.sub(r'\s+(\d+\.)', r'\n\n\1', text_str)
        return formatted_text.strip()
    except Exception as e:
        logger.error(f"Error formatting translation: {e}", exc_info=True)
        return str(translated_text) # Return original text if formatting fails


# --- Monkey-Patch (Conditional, less critical now with newer Streamlit versions) ---
# Keep for compatibility if needed, but might be removable depending on target Streamlit version
import streamlit.elements.image as st_image
if not hasattr(st_image, "image_to_url"):
    logger.info("Attempting to apply monkey-patch for st.elements.image.image_to_url (may not be needed).")
    def image_to_url_monkey_patch(img_obj: Any, width: int = -1, clamp: bool = False,
                                  channels: str = "RGB", output_format: str = "auto",
                                  image_id: str = "") -> str:
        if PIL_AVAILABLE and isinstance(img_obj, Image.Image):
            try:
                buffered = io.BytesIO()
                fmt = "PNG" if output_format.lower() == "auto" or img_obj.mode == 'RGBA' else "JPEG"
                temp_img = img_obj
                # Ensure image is in a savable mode (e.g., handle P mode, ensure RGB if requested)
                if temp_img.mode == 'P':
                     temp_img = temp_img.convert('RGBA') # Convert palette to RGBA for saving as PNG
                     fmt = "PNG"
                elif channels == "RGB" and temp_img.mode not in ['RGB', 'L']: # Allow grayscale 'L'
                    temp_img = temp_img.convert('RGB')

                temp_img.save(buffered, format=fmt)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/{fmt.lower()};base64,{img_str}"
            except Exception as e:
                logger.error(f"Monkey-patch image_to_url failed: {e}", exc_info=True)
                return "" # Return empty string on failure
        # Handle other types if necessary, or log warning
        logger.warning(f"Monkey-patch image_to_url: Unsupported type {type(img_obj)} or PIL unavailable.")
        return ""
    st_image.image_to_url = image_to_url_monkey_patch
    logger.info("Applied monkey-patch for st.elements.image.image_to_url.")
else:
     logger.info("Monkey-patch for st.elements.image.image_to_url not needed.")


# --- Sidebar ---
with st.sidebar:
    st.header("⚕️ RadVision Controls")
    st.markdown("---")

    # Tip of the Day
    TIPS = [
        "Tip: Use 'Demo Mode' for a quick walkthrough with a sample chest X-ray.",
        "Tip: Draw a rectangle (ROI) on the image to focus the AI's attention.",
        "Tip: Adjust DICOM Window/Level sliders for optimal image contrast.",
        "Tip: Ask follow-up questions based on the initial analysis or previous answers.",
        "Tip: Generate a PDF report to document the AI findings and your interaction.",
        "Tip: Use the 'Translation' tab to understand findings in different languages.",
        "Tip: Clear the ROI using the button if you want the AI to consider the entire image again.",
        "Tip: Use the 'UMLS Lookup' tab to find standardized concepts for medical terms.", # Added UMLS Tip
    ]
    st.info(f"💡 {random.choice(TIPS)}")
    st.markdown("---")

    # Upload Section
    st.header("Image Upload & Settings")
    uploaded_file = st.file_uploader(
        "Upload Image (JPG, PNG, DCM)",
        type=["jpg", "jpeg", "png", "dcm", "dicom"],
        key="file_uploader_widget", # Consistent key
        help="Upload a medical image file for analysis. DICOM (.dcm) is preferred."
    )

    # Demo Mode (Consider adding example image loading logic here if demo_mode is checked)
    demo_mode = st.checkbox("🚀 Demo Mode", value=st.session_state.get("demo_loaded", False),
                            help="Load a sample chest X-ray image and analysis.")
    if demo_mode and not st.session_state.demo_loaded:
         # Placeholder: Add logic here to load a demo DICOM or image file
         # e.g., load from a local path, set session state variables accordingly,
         # and set st.session_state.demo_loaded = True, then st.rerun()
         st.warning("Demo mode activation logic not yet implemented.")
         # To prevent re-triggering immediately:
         # st.session_state.demo_loaded = True # Or manage state carefully

    # ROI Control
    if st.button("🗑️ Clear ROI", help="Remove the selected ROI", key="clear_roi_btn"):
        st.session_state.roi_coords = None
        st.session_state.canvas_drawing = None # Clear canvas state too
        st.session_state.clear_roi_feedback = True
        logger.info("ROI cleared by user.")
        st.rerun()

    # Display feedback after clearing ROI
    if st.session_state.get("clear_roi_feedback"):
        st.success("✅ ROI cleared successfully!")
        st.balloons()
        st.session_state.clear_roi_feedback = False # Reset flag

    # DICOM Window/Level Section (conditional display)
    if st.session_state.is_dicom and st.session_state.display_image and UI_COMPONENTS_AVAILABLE:
        st.markdown("---")
        st.subheader("DICOM Display")
        # Pass the DICOM dataset and current W/L values to the UI component
        new_wc, new_ww = dicom_wl_sliders(
            st.session_state.dicom_dataset,
            st.session_state.current_display_wc,
            st.session_state.current_display_ww
        )

        # Check if W/L values have changed
        # Use a small tolerance for float comparison if necessary, but direct compare often ok here
        if new_wc != st.session_state.current_display_wc or new_ww != st.session_state.current_display_ww:
            logger.info(f"DICOM W/L changed via sliders: WC={new_wc}, WW={new_ww}")
            # Update session state with new values
            st.session_state.current_display_wc = new_wc
            st.session_state.current_display_ww = new_ww

            # Re-render the display image with new W/L
            if DICOM_UTILS_AVAILABLE and st.session_state.dicom_dataset:
                with st.spinner("Applying new Window/Level..."):
                    # Use dicom_to_image from utils to apply new W/L
                    new_display_img = dicom_to_image(
                        st.session_state.dicom_dataset,
                        wc=new_wc,
                        ww=new_ww
                    )
                    if isinstance(new_display_img, Image.Image):
                        # Ensure image is RGB for display consistency
                        if new_display_img.mode != 'RGB':
                            new_display_img = new_display_img.convert('RGB')
                        st.session_state.display_image = new_display_img
                        logger.info("Display image updated with new W/L.")
                        st.rerun() # Rerun to show the updated image
                    else:
                        st.error("Failed to update DICOM image with new W/L.")
                        logger.error("dicom_to_image returned invalid type after W/L update.")
            else:
                st.warning("DICOM utilities not available to update W/L.")
                logger.warning("W/L changed but DICOM utilities missing.")

    st.markdown("---")
    st.header("🤖 AI Analysis Actions")

    # Disable action buttons if no image is processed and ready
    action_disabled = not isinstance(st.session_state.get("processed_image"), Image.Image)

    # --- Action Buttons ---
    if st.button("▶️ Run Initial Analysis", key="analyze_btn", disabled=action_disabled,
                 help="Perform a general analysis of the entire image or selected ROI."):
        st.session_state.last_action = "analyze"
        st.rerun() # Rerun to trigger action handler

    # --- Q&A Section ---
    st.subheader("❓ Ask AI a Question")
    question_input = st.text_area(
        "Enter your question:",
        height=100,
        key="question_input_widget", # Consistent key
        placeholder="E.g., 'Are there any nodules in the upper right lobe?'",
        disabled=action_disabled
    )
    if st.button("💬 Ask Question", key="ask_btn", disabled=action_disabled):
        if question_input.strip():
            st.session_state.last_action = "ask"
            st.rerun() # Rerun to trigger action handler
        else:
            st.warning("Please enter a question before submitting.")

    # --- Disease Analysis Section ---
    st.subheader("🎯 Condition-Specific Analysis")
    # Consider making this list configurable or loading from a file
    DISEASE_OPTIONS = [
        "Pneumonia", "Lung Cancer", "Nodule/Mass", "Effusion", "Fracture",
        "Stroke", "Appendicitis", "Bowel Obstruction", "Cardiomegaly",
        "Aortic Aneurysm", "Pulmonary Embolism", "Tuberculosis", "COVID-19",
        "Brain Tumor", "Arthritis", "Osteoporosis", "Other..." # Add 'Other' if needed
    ]
    disease_select = st.selectbox(
        "Select condition to focus on:",
        options=[""] + sorted(DISEASE_OPTIONS), # Add empty option as default
        key="disease_select_widget", # Consistent key
        disabled=action_disabled
    )
    if st.button("🩺 Analyze Condition", key="disease_btn", disabled=action_disabled):
        if disease_select:
            st.session_state.last_action = "disease"
            st.rerun() # Rerun to trigger action handler
        else:
            st.warning("Please select a condition first.")

    st.markdown("---")
    st.header("📊 Confidence & Reporting")

    # Enable confidence button only if there's something to evaluate
    can_estimate_confidence = bool(
        st.session_state.history or
        st.session_state.initial_analysis or
        st.session_state.disease_analysis
    ) and not action_disabled

    if st.button("📈 Estimate AI Confidence", key="confidence_btn", disabled=not can_estimate_confidence):
        st.session_state.last_action = "confidence"
        st.rerun() # Rerun to trigger action handler

    # Enable report generation only if utils available and image loaded
    report_generation_disabled = action_disabled or not REPORT_UTILS_AVAILABLE
    if st.button("📄 Generate PDF Report Data", key="generate_report_data_btn",
                 disabled=report_generation_disabled):
        st.session_state.last_action = "generate_report_data"
        st.rerun() # Rerun to trigger action handler

    # Show download button only if PDF bytes exist in session state
    if st.session_state.get("pdf_report_bytes"):
        report_filename = f"RadVisionAI_Report_{st.session_state.session_id or 'session'}.pdf"
        st.download_button(
            label="⬇️ Download PDF Report",
            data=st.session_state.pdf_report_bytes,
            file_name=report_filename,
            mime="application/pdf",
            key="download_pdf_button",
            help="Download the generated PDF report."
        )


# --- File Upload Processing Logic ---
# This block handles detecting a new file upload and preparing it.
if uploaded_file is not None: # Check if the file uploader widget has a file this run
    # --- Calculate identifier for the currently uploaded file ---
    try:
        uploaded_file.seek(0) # Go to start of file bytes
        file_content_hash = hashlib.sha256(uploaded_file.read()).hexdigest()[:16] # Calculate hash
        uploaded_file.seek(0) # Reset pointer for subsequent reads
        # Define the unique identifier for this uploaded file
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{file_content_hash}"
        logger.debug(f"Calculated new file info: {new_file_info}")
    except Exception as e:
        logger.warning(f"Could not generate hash for file {uploaded_file.name}: {e}")
        # Define a fallback identifier if hashing fails
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{uuid.uuid4().hex[:8]}"
        logger.debug(f"Using fallback file info: {new_file_info}")
    # --- END Calculation ---

    # --- Compare with last processed file and reset state if it's NEW ---
    if new_file_info != st.session_state.get("uploaded_file_info"):
        logger.info(f"New file detected via info mismatch: '{new_file_info}' != '{st.session_state.get('uploaded_file_info')}'")
        st.toast(f"Processing '{uploaded_file.name}'...", icon="⏳")

        # --- State Reset Logic for New File ---
        # Define application state keys to PRESERVE across file uploads
        # DO NOT include widget keys (like 'file_uploader_widget')
        keys_to_preserve = {
            "session_id", # Always preserve session ID
            # Decide which other states to keep, e.g., maybe the UMLS search term?
            # "umls_search_term",
            # "umls_results", # Usually clear results for new image
            # "umls_error",   # Usually clear error for new image
        }
        logger.debug(f"Preserving state keys: {keys_to_preserve}")

        # Store the values of the keys we want to preserve *before* resetting
        preserved_values = {k: st.session_state.get(k) for k in keys_to_preserve if k in st.session_state}

        # Reset all keys defined in DEFAULT_STATE back to their defaults, EXCEPT those we preserve
        logger.debug("Resetting session state to defaults (excluding preserved keys)...")
        for key, default_value in DEFAULT_STATE.items():
            if key not in keys_to_preserve:
                st.session_state[key] = copy.deepcopy(default_value) if isinstance(default_value, (dict, list)) else default_value

        # Restore the explicitly preserved values
        for k, v in preserved_values.items():
            st.session_state[k] = v
        logger.debug("Restored preserved state keys.")
        # --- End State Reset ---

        # --- Update state for the newly uploaded file ---
        st.session_state.uploaded_file_info = new_file_info # Mark this file as the current one
        st.session_state.demo_loaded = False # Turn off demo mode

        # Store raw bytes and determine if it's DICOM
        st.session_state.raw_image_bytes = uploaded_file.getvalue()
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        # Check type and extension, requires pydicom and utils to be available
        st.session_state.is_dicom = (
            PYDICOM_AVAILABLE and DICOM_UTILS_AVAILABLE and
            ("dicom" in uploaded_file.type.lower() or file_ext in (".dcm", ".dicom"))
        )
        logger.info(f"File '{uploaded_file.name}' identified as DICOM: {st.session_state.is_dicom}")

        # --- Process the image (DICOM or Standard) ---
        with st.spinner("🔬 Analyzing file..."):
            temp_display_img: Optional[Image.Image] = None
            temp_processed_img: Optional[Image.Image] = None
            processing_success = False

            if st.session_state.is_dicom:
                logger.info("Processing as DICOM...")
                try:
                    # Parse DICOM using the utility function
                    dicom_ds = parse_dicom(st.session_state.raw_image_bytes, filename=uploaded_file.name)
                    st.session_state.dicom_dataset = dicom_ds # Store the dataset object
                    if dicom_ds:
                        # Extract metadata using the utility function
                        st.session_state.dicom_metadata = extract_dicom_metadata(dicom_ds)
                        logger.info(f"Extracted {len(st.session_state.dicom_metadata)} DICOM metadata tags.")
                        # Get default W/L using the utility function
                        default_wc, default_ww = get_default_wl(dicom_ds)
                        # Store initial W/L values in session state
                        st.session_state.current_display_wc = default_wc
                        st.session_state.current_display_ww = default_ww
                        logger.info(f"Default DICOM W/L: WC={default_wc}, WW={default_ww}")

                        # Create image for display (with default W/L)
                        temp_display_img = dicom_to_image(dicom_ds, wc=default_wc, ww=default_ww)
                        # Create image for processing (potentially normalized, full dynamic range)
                        temp_processed_img = dicom_to_image(dicom_ds, wc=None, ww=None, normalize=True)

                        if isinstance(temp_display_img, Image.Image) and isinstance(temp_processed_img, Image.Image):
                            processing_success = True
                            logger.info("DICOM parsed and converted to images successfully.")
                        else:
                            st.error("Failed to convert DICOM pixel data to image.")
                            logger.error("dicom_to_image did not return valid PIL Image objects.")
                    else:
                        st.error("Could not parse DICOM file structure.")
                        logger.error("parse_dicom returned None.")
                        st.session_state.is_dicom = False # Mark as not DICOM if parsing fails
                except pydicom.errors.InvalidDicomError:
                    st.error("Invalid DICOM file. Please upload a valid .dcm file.")
                    logger.error("InvalidDicomError during DICOM parsing.")
                    st.session_state.is_dicom = False
                except Exception as e:
                    st.error(f"Error processing DICOM file: {e}")
                    logger.error(f"DICOM processing error: {e}", exc_info=True)
                    st.session_state.is_dicom = False

            # --- Process as standard image if not DICOM or if DICOM processing failed ---
            if not st.session_state.is_dicom:
                logger.info("Processing as standard image (JPG/PNG)...")
                if not PIL_AVAILABLE:
                    st.error("Cannot process standard images: Pillow library is missing.")
                    logger.critical("Pillow missing, cannot process standard image.")
                else:
                    try:
                        # Open image from bytes using Pillow
                        raw_img = Image.open(io.BytesIO(st.session_state.raw_image_bytes))
                        # Convert to RGB for consistency (handles RGBA, P, L modes)
                        processed_img = raw_img.convert("RGB")
                        temp_display_img = processed_img.copy() # Use the same image for display and processing
                        temp_processed_img = processed_img.copy()
                        processing_success = True
                        logger.info(f"Standard image '{uploaded_file.name}' loaded and converted to RGB successfully.")
                    except UnidentifiedImageError:
                        st.error("Could not identify image format. Please upload JPG, PNG, or valid DICOM.")
                        logger.error(f"UnidentifiedImageError for file: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error processing standard image: {e}")
                        logger.error(f"Standard image processing error: {e}", exc_info=True)

            # --- Finalize state update after processing attempt ---
            if processing_success and isinstance(temp_display_img, Image.Image) and isinstance(temp_processed_img, Image.Image):
                # Ensure display image is RGB
                if temp_display_img.mode != 'RGB':
                    st.session_state.display_image = temp_display_img.convert('RGB')
                else:
                    st.session_state.display_image = temp_display_img
                # Store the processed image
                st.session_state.processed_image = temp_processed_img
                st.success(f"✅ '{uploaded_file.name}' loaded and processed successfully!")
                logger.info(f"Image processing complete for: {uploaded_file.name}")
                st.rerun() # Rerun to update the UI with the new image and cleared state
            else:
                 # If processing failed for any reason
                 st.error("Image loading failed. Please check the file format or try a different file.")
                 logger.error(f"Image processing failed for file: {uploaded_file.name}")
                 # Explicitly clear potentially partially set state to avoid confusion
                 st.session_state.uploaded_file_info = None # Mark that no file is successfully loaded
                 st.session_state.display_image = None
                 st.session_state.processed_image = None
                 st.session_state.is_dicom = False
                 st.session_state.dicom_dataset = None
                 st.session_state.dicom_metadata = {}
                 st.session_state.current_display_wc = None
                 st.session_state.current_display_ww = None
                 # Do not rerun here, let the user see the error message

    # --- If it's the SAME file (info matches) ---
    # No state reset is needed. The rest of the app will just use the existing image data in session state.
    # else: # Optional: log if the file is the same
    #    logger.debug(f"File '{st.session_state.uploaded_file_info}' is the same as previous run. No state reset.")


# --- Main Page Layout ---
st.markdown("---") # Separator
st.title("⚕️ RadVision AI Advanced: AI-Assisted Image Analysis")

# User Guide and Disclaimer
with st.expander("User Guide & Disclaimer", expanded=False):
    st.warning("⚠️ **Disclaimer**: This tool is for research/educational purposes only and is **NOT** a substitute for professional medical advice or diagnosis. AI analysis may contain errors.")
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

st.markdown("---") # Separator

# Main content area with two columns
col1, col2 = st.columns([2, 3]) # Adjust ratio as needed (e.g., [1, 1] for equal width)

# --- Column 1: Image Viewer and Metadata ---
with col1:
    st.subheader("🖼️ Image Viewer")
    display_img = st.session_state.get("display_image")

    if isinstance(display_img, Image.Image):
        # --- Drawable Canvas for ROI ---
        if DRAWABLE_CANVAS_AVAILABLE and st_canvas:
            st.caption("Draw a rectangle below to select a Region of Interest (ROI).")
            # --- Calculate canvas dimensions based on image aspect ratio ---
            MAX_CANVAS_WIDTH = 600  # Max width for the canvas container
            MAX_CANVAS_HEIGHT = 500 # Max height for the canvas container
            img_w, img_h = display_img.size

            # Basic validation for image dimensions
            if img_w <= 0 or img_h <= 0:
                st.warning("Invalid image dimensions; cannot draw ROI.")
                logger.warning(f"Cannot draw canvas, invalid image dimensions: {img_w}x{img_h}")
            else:
                aspect_ratio = img_w / img_h
                # Calculate width first, then height based on aspect ratio
                canvas_width = min(img_w, MAX_CANVAS_WIDTH)
                canvas_height = int(canvas_width / aspect_ratio)
                # If calculated height exceeds max, recalculate width based on max height
                if canvas_height > MAX_CANVAS_HEIGHT:
                    canvas_height = MAX_CANVAS_HEIGHT
                    canvas_width = int(canvas_height * aspect_ratio)
                # Ensure minimum dimensions
                canvas_width = max(canvas_width, 150)
                canvas_height = max(canvas_height, 150)
                logger.debug(f"Canvas dimensions set to: {canvas_width}x{canvas_height} for image {img_w}x{img_h}")

                # --- Instantiate the Drawable Canvas ---
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.2)",  # Semi-transparent orange fill for ROI
                    stroke_width=2,                       # Border width of drawn rectangle
                    stroke_color="rgba(239, 83, 80, 0.8)",# Reddish border color
                    background_image=display_img,         # The image to draw on
                    update_streamlit=True,                # Send updates back to Streamlit state
                    height=canvas_height,                 # Calculated canvas height
                    width=canvas_width,                   # Calculated canvas width
                    drawing_mode="rect",                  # Mode for drawing rectangles
                    initial_drawing=st.session_state.get("canvas_drawing", None), # Persist drawing state
                    key="drawable_canvas"                 # Unique key for the widget
                )

                # --- Process Canvas Results for ROI ---
                if canvas_result.json_data and canvas_result.json_data.get("objects"):
                    # Get the latest drawn object (usually the ROI rectangle)
                    if canvas_result.json_data["objects"]: # Check if list is not empty
                        last_object = canvas_result.json_data["objects"][-1]
                        if last_object["type"] == "rect":
                            # Extract coordinates from the canvas result (scaled to canvas size)
                            canvas_left = int(last_object["left"])
                            canvas_top = int(last_object["top"])
                            # Account for potential scaling within the canvas object itself
                            canvas_width_scaled = int(last_object["width"] * last_object.get("scaleX", 1))
                            canvas_height_scaled = int(last_object["height"] * last_object.get("scaleY", 1))

                            # --- Scale canvas coordinates back to original image dimensions ---
                            scale_x = img_w / canvas_width
                            scale_y = img_h / canvas_height
                            original_left = int(canvas_left * scale_x)
                            original_top = int(canvas_top * scale_y)
                            original_width = int(canvas_width_scaled * scale_x)
                            original_height = int(canvas_height_scaled * scale_y)

                            # --- Boundary checks to ensure ROI stays within image bounds ---
                            original_left = max(0, original_left)
                            original_top = max(0, original_top)
                            # Ensure width/height don't exceed image boundaries from the starting point
                            original_width = min(img_w - original_left, original_width)
                            original_height = min(img_h - original_top, original_height)
                            # Ensure width/height are positive
                            original_width = max(1, original_width)
                            original_height = max(1, original_height)


                            # Store the calculated original image ROI coordinates
                            new_roi = {
                                "left": original_left,
                                "top": original_top,
                                "width": original_width,
                                "height": original_height
                            }

                            # Update session state only if ROI has changed
                            if st.session_state.roi_coords != new_roi:
                                st.session_state.roi_coords = new_roi
                                # Store the raw canvas data to redraw it on rerun
                                st.session_state.canvas_drawing = canvas_result.json_data
                                logger.info(f"New ROI selected (original coords): {new_roi}")
                                st.info(f"ROI Set: ({original_left}, {original_top}), Size: {original_width}x{original_height}", icon="🎯")
                                # Note: No rerun needed here, canvas update handles it
        else:
             # Fallback if canvas is not available - just display the image
             st.image(display_img, caption="Image Preview", use_container_width=True)

        # Display current ROI coordinates if set
        if st.session_state.roi_coords:
            roi = st.session_state.roi_coords
            st.caption(f"Active ROI: ({roi['left']}, {roi['top']}) - W:{roi['width']}, H:{roi['height']}")

        st.markdown("---") # Separator

        # --- Display DICOM Metadata ---
        if st.session_state.is_dicom:
            if st.session_state.dicom_metadata:
                 # Use the dedicated UI component
                 display_dicom_metadata(st.session_state.dicom_metadata)
            else:
                 st.caption("DICOM file loaded, but no metadata could be extracted.")
        # elif st.session_state.processed_image: # Optional: Could show basic image info for non-DICOM
        #     st.caption(f"Image Type: {display_img.format}, Mode: {display_img.mode}, Size: {display_img.size}")

    # --- Placeholder if no image is loaded ---
    elif st.session_state.get("uploaded_file_info") and not st.session_state.get("display_image"):
        # Case where upload happened but processing failed
        st.error("Image preview unavailable due to a processing error.")
    else:
        # Default message when app starts or no file is loaded
        st.info("⬅️ Please upload an image or enable Demo Mode using the sidebar.")


# --- Column 2: Analysis Results and Tabs ---
with col2:
    st.subheader("📊 Analysis & Results")

    # Define Tab Titles
    tab_titles = [
        "🔬 Initial Analysis",
        "💬 Q&A History",
        "🩺 Condition Focus",
        "📚 UMLS Lookup", # Added UMLS Tab
        "📈 Confidence",
        "🌐 Translation"
    ]

    # Create Tabs
    tabs = st.tabs(tab_titles)

    # --- Tab Content ---
    with tabs[0]: # Initial Analysis
        st.text_area(
            "Overall Findings & Impressions",
            value=st.session_state.initial_analysis or "Run 'Initial Analysis' from the sidebar to see results here.",
            height=400, # Adjust height as needed
            disabled=True, # Read-only display
            key="initial_analysis_display"
        )

    with tabs[1]: # Q&A History
        st.text_area(
            "Latest AI Answer",
            value=st.session_state.qa_answer or "Ask a question using the sidebar to see the AI's response here.",
            height=150, # Shorter height for the latest answer
            disabled=True,
            key="qa_answer_display"
        )
        st.markdown("---")
        st.subheader("Full Conversation History")
        if st.session_state.history:
            # Display history in reverse chronological order (newest first)
            for i, (q_type, message) in enumerate(reversed(st.session_state.history)):
                 # Use markdown for better formatting possibilities
                 if q_type.lower() == "user question":
                     st.markdown(f"**👤 You:** {message}")
                 elif q_type.lower() == "ai answer":
                     # Process potential markdown in AI answer
                     st.markdown(f"**🤖 AI:** {message}", unsafe_allow_html=True) # Be cautious with unsafe_allow_html
                 elif q_type.lower() == "[fallback] user question":
                      st.markdown(f"**👤 You (Fallback):** {message}")
                 elif q_type.lower() == "[fallback] ai answer":
                      st.markdown(f"**🤖 AI (Fallback):** {message}", unsafe_allow_html=True)
                 elif q_type.lower() == "system":
                     st.info(f"*{message}*", icon="ℹ️") # System messages
                 else: # Generic display for other types
                     st.markdown(f"**{q_type}:** {message}")

                 # Add separator between messages, except for the last one
                 if i < len(st.session_state.history) - 1:
                     st.markdown("---")
        else:
            st.caption("No questions asked yet in this session.")

    with tabs[2]: # Condition Focus
        st.text_area(
            "Condition-Specific Analysis",
            value=st.session_state.disease_analysis or "Select a condition in the sidebar and click 'Analyze Condition'.",
            height=400,
            disabled=True,
            key="disease_analysis_display"
        )

    with tabs[3]: # UMLS Lookup
        st.subheader("📚 UMLS Concept Search")
        if not UMLS_AVAILABLE:
            st.warning("UMLS features are unavailable. Ensure 'umls_utils.py' is present and the 'UMLS_APIKEY' environment variable is set.")
        else:
            # Input field for the search term
            umls_search_term = st.text_input(
                "Enter medical term to search:",
                value=st.session_state.get("umls_search_term", ""),
                key="umls_search_term_input", # Unique key
                placeholder="e.g., lung nodule, cardiomegaly"
            )
            # Search button
            if st.button("🔎 Search UMLS", key="umls_search_button"):
                if umls_search_term.strip():
                    st.session_state.last_action = "umls_search"
                    # Store the term being searched in session state
                    st.session_state.umls_search_term = umls_search_term.strip()
                    st.rerun() # Rerun to trigger the UMLS search action handler
                else:
                    st.warning("Please enter a search term.")

            # Display UMLS error message if it exists
            if st.session_state.get("umls_error"):
                st.error(f"UMLS Search Error: {st.session_state.umls_error}")

            # Display results using the dedicated UI component
            # Pass the list of UMLSConcept objects (or None) from session state
            if UI_COMPONENTS_AVAILABLE:
                 display_umls_concepts(st.session_state.get("umls_results"))
            else:
                 # Fallback display if ui_components failed to load
                 st.caption("UMLS display component unavailable.")
                 if st.session_state.get("umls_results"):
                     st.json([r.to_dict() if hasattr(r, 'to_dict') else r for r in st.session_state.umls_results])


    with tabs[4]: # Confidence
        st.text_area(
            "Estimated AI Confidence",
            value=st.session_state.confidence_score or "Run 'Estimate AI Confidence' from the sidebar after performing an analysis.",
            height=400,
            disabled=True,
            key="confidence_score_display"
        )

    with tabs[5]: # Translation
        st.subheader("🌐 Translate Analysis Text")

        if not TRANSLATION_AVAILABLE:
            st.warning("Translation features are unavailable. Ensure 'deep-translator' and 'translation_models' are functional.")
        else:
            st.caption("Select analysis text, choose languages, then click 'Translate'.")
            # --- Text Selection ---
            text_options = {
                "Initial Analysis": st.session_state.initial_analysis,
                "Latest Q&A Answer": st.session_state.qa_answer, # Use latest QA answer
                "Condition Analysis": st.session_state.disease_analysis,
                "Confidence Estimation": st.session_state.confidence_score,
                "(Enter Custom Text Below)": "" # Option for custom input
            }
            # Filter out options that are empty (unless it's the custom text option)
            available_options = {
                label: txt for label, txt in text_options.items() if (txt and txt.strip()) or label == "(Enter Custom Text Below)"
            }
            if not available_options:
                st.info("No analysis text available to translate yet.")
            else:
                selected_label = st.selectbox(
                    "Select text to translate:",
                    list(available_options.keys()),
                    index=0, # Default to the first available option
                    key="translate_text_select"
                )

                text_to_translate = available_options.get(selected_label, "")

                # Show custom text area only if selected
                if selected_label == "(Enter Custom Text Below)":
                    text_to_translate = st.text_area(
                        "Enter or paste text to translate:",
                        value="", # Start empty for custom input
                        height=100,
                        key="custom_translate_input"
                    )
                else:
                    # Display the selected text (read-only)
                     st.text_area(
                        "Text selected for translation:",
                        value=text_to_translate,
                        height=100,
                        disabled=True,
                        key="selected_translate_display"
                    )


                # --- Language Selection ---
                col_lang1, col_lang2 = st.columns(2)
                with col_lang1:
                    # Source Language (Allow Auto-Detect)
                    source_language_options = [AUTO_DETECT_INDICATOR] + sorted(list(LANGUAGE_CODES.keys()))
                    source_language_name = st.selectbox(
                        "Source Language:",
                        source_language_options,
                        index=0, # Default to Auto-Detect
                        key="source_language_select"
                    )
                with col_lang2:
                    # Target Language
                    target_language_options = sorted(list(LANGUAGE_CODES.keys()))
                    # Try to find a sensible default like English or Spanish
                    default_target_index = 0
                    common_targets = ["English", "Spanish", "French", "German"]
                    for i, lang in enumerate(target_language_options):
                        if lang in common_targets:
                             default_target_index = i
                             if lang == "English": break # Prefer English if available


                    target_language_name = st.selectbox(
                        "Translate To:",
                        target_language_options,
                        index=default_target_index,
                        key="target_language_select"
                    )

                # --- Translate Button ---
                if st.button("🔄 Translate Now", key="translate_button"):
                    st.session_state.translation_result = None # Clear previous results/errors
                    st.session_state.translation_error = None

                    if not text_to_translate.strip():
                        st.warning("Please select or enter some text to translate.")
                        st.session_state.translation_error = "Input text is empty."
                    # Avoid translating if source and target are explicitly the same
                    elif source_language_name == target_language_name and source_language_name != AUTO_DETECT_INDICATOR:
                        st.info("Source and target languages are the same. No translation needed.")
                        st.session_state.translation_result = text_to_translate
                    else:
                         # Perform translation
                         with st.spinner(f"Translating from '{source_language_name}' to '{target_language_name}'..."):
                             try:
                                 # Call the translation function from translation_models
                                 translation_output = translate(
                                     text=text_to_translate,
                                     target_language=target_language_name,
                                     source_language=source_language_name # Pass selected source
                                 )
                                 if translation_output is not None:
                                     st.session_state.translation_result = translation_output
                                     st.success("Translation complete!")
                                     logger.info("Translation successful.")
                                 else:
                                     # Handle cases where the translation function returns None unexpectedly
                                     st.error("Translation service returned no result. Please check logs or try again.")
                                     logger.error("Translation function returned None.")
                                     st.session_state.translation_error = "Translation service returned None."
                             except Exception as e:
                                 # Catch potential errors from the translation library or network issues
                                 st.error(f"Translation failed: {e}")
                                 logger.error(f"Translation error: {e}", exc_info=True)
                                 st.session_state.translation_error = str(e)

                # --- Display Translation Result or Error ---
                if st.session_state.get("translation_result"):
                    formatted_result = format_translation(st.session_state.translation_result)
                    st.text_area("Translated Text:", value=formatted_result, height=200, key="translation_output_display")
                elif st.session_state.get("translation_error"):
                    # Display error informatively, already shown via st.error/warning above
                    # st.info(f"Translation Error: {st.session_state.translation_error}")
                    pass


# --- Button Action Handlers ---
# This block executes when a button press sets 'last_action' and triggers a rerun
current_action = st.session_state.get("last_action")
if current_action:
    logger.info(f"Handling action: '{current_action}' for session: {st.session_state.session_id}")

    # --- Pre-action checks ---
    action_requires_image = current_action not in ["generate_report_data", "umls_search"] # These actions don't need a processed image
    action_requires_llm = current_action in ["analyze", "ask", "disease", "confidence"]
    action_requires_report_util = (current_action == "generate_report_data")
    action_requires_umls = (current_action == "umls_search")

    valid_action = True # Flag to track if prerequisites are met

    # Check for processed image if required
    if action_requires_image and not isinstance(st.session_state.get("processed_image"), Image.Image):
        st.error(f"Cannot perform '{current_action}': No valid image loaded and processed.")
        logger.warning(f"Action '{current_action}' blocked: Missing processed image.")
        valid_action = False

    # Check for session ID (should always exist but good practice)
    if not st.session_state.session_id:
        st.error("Critical Error: No session ID available.")
        logger.critical("Action blocked: Missing session ID.")
        valid_action = False

    # Check for LLM module availability if required
    if action_requires_llm and not LLM_INTERACTIONS_AVAILABLE:
        st.error(f"Cannot perform '{current_action}': Core AI module is unavailable.")
        logger.error(f"Action '{current_action}' blocked: LLM module unavailable.")
        valid_action = False

    # Check for Report utility availability if required
    if action_requires_report_util and not REPORT_UTILS_AVAILABLE:
        st.error(f"Cannot perform '{current_action}': Report generation module unavailable.")
        logger.error(f"Action '{current_action}' blocked: Report module unavailable.")
        valid_action = False

    # Check for UMLS availability if required
    if action_requires_umls and not UMLS_AVAILABLE:
        st.error(f"Cannot perform '{current_action}': UMLS module unavailable or API key missing.")
        logger.error(f"Action '{current_action}' blocked: UMLS module/key unavailable.")
        valid_action = False

    # --- Execute Action if Valid ---
    if valid_action:
        # Prepare common inputs for actions
        img_for_llm = st.session_state.processed_image # Use the processed image
        roi_coords = st.session_state.roi_coords       # Current ROI coordinates
        current_history = st.session_state.history     # Q&A history list

        try:
            # --- Initial Analysis Action ---
            if current_action == "analyze":
                st.info("🔬 Performing initial analysis...")
                with st.spinner("AI analyzing image... Please wait."):
                    analysis_result = run_initial_analysis(img_for_llm, roi=roi_coords)
                st.session_state.initial_analysis = analysis_result
                # Optionally clear other results when running a new analysis
                st.session_state.qa_answer = ""
                st.session_state.disease_analysis = ""
                logger.info("Initial analysis complete.")
                st.success("Initial analysis complete!")

            # --- Ask Question Action ---
            elif current_action == "ask":
                question_text = st.session_state.question_input_widget.strip() # Get question from widget state
                if not question_text:
                    st.warning("Question input was empty.")
                    logger.warning("Ask action triggered with empty question.")
                else:
                    st.info(f"❓ Asking AI: '{question_text}'...")
                    st.session_state.qa_answer = "" # Clear previous answer display
                    with st.spinner("AI thinking... Please wait."):
                        # Run the multimodal Q&A function
                        answer, success_flag = run_multimodal_qa(
                            img=img_for_llm,
                            question=question_text,
                            history=current_history, # Pass conversation history
                            roi=roi_coords
                        )
                    if success_flag:
                        st.session_state.qa_answer = answer
                        # Append interaction to history
                        st.session_state.history.append(("User Question", question_text))
                        st.session_state.history.append(("AI Answer", answer))
                        logger.info(f"Q&A successful for: '{question_text}'")
                        st.success("AI answered your question!")
                    else:
                        # Handle primary AI failure
                        primary_error_msg = f"Primary AI failed to answer. Reason: {answer}"
                        st.session_state.qa_answer = primary_error_msg # Display error
                        st.error(primary_error_msg)
                        logger.error(f"Primary Q&A failed for '{question_text}'. Reason: {answer}")

                        # --- Attempt Fallback HF Model ---
                        hf_token = os.environ.get("HF_API_TOKEN")
                        if HF_MODELS_AVAILABLE and hf_token:
                            st.info(f"Attempting fallback Hugging Face model: {HF_VQA_MODEL_ID}")
                            with st.spinner("Trying fallback model..."):
                                fallback_answer, fallback_success = query_hf_vqa_inference_api(
                                    img=img_for_llm, question=question_text, roi=roi_coords
                                )
                            if fallback_success:
                                fallback_display = f"**[Fallback: {HF_VQA_MODEL_ID}]**\n{fallback_answer}"
                                # Append fallback answer to the display and history
                                st.session_state.qa_answer += "\n\n" + fallback_display
                                st.session_state.history.append(("[Fallback] User Question", question_text))
                                st.session_state.history.append(("[Fallback] AI Answer", fallback_display))
                                logger.info(f"Fallback Q&A successful for: '{question_text}'")
                                st.success("Fallback AI provided an answer.")
                            else:
                                fallback_error_msg = f"Fallback AI ({HF_VQA_MODEL_ID}) also failed. Reason: {fallback_answer}"
                                st.session_state.qa_answer += f"\n\n{fallback_error_msg}"
                                logger.error(f"Fallback Q&A failed for '{question_text}'. Reason: {fallback_answer}")
                                st.error(fallback_error_msg)
                        elif HF_MODELS_AVAILABLE and not hf_token:
                             # Fallback available but token missing
                             no_token_msg = "[Fallback Skipped: Hugging Face API token (HF_API_TOKEN) not found in environment variables]"
                             st.session_state.qa_answer += f"\n\n{no_token_msg}"
                             logger.warning("Fallback Q&A skipped: HF_API_TOKEN missing.")
                             st.warning(no_token_msg)
                        else:
                            # Fallback module not available
                            no_fallback_msg = "[Fallback Unavailable: No fallback model configured]"
                            st.session_state.qa_answer += f"\n\n{no_fallback_msg}"
                            logger.warning("Fallback Q&A unavailable: hf_models not loaded.")
                            st.warning(no_fallback_msg)

            # --- Disease Analysis Action ---
            elif current_action == "disease":
                selected_disease = st.session_state.disease_select_widget # Get from widget state
                if not selected_disease:
                    st.warning("No condition was selected.")
                    logger.warning("Disease analysis action triggered with no condition selected.")
                else:
                    st.info(f"🩺 Analyzing for '{selected_disease}'...")
                    with st.spinner(f"AI analyzing for {selected_disease}... Please wait."):
                        disease_result = run_disease_analysis(img_for_llm, selected_disease, roi=roi_coords)
                    st.session_state.disease_analysis = disease_result
                    # Optionally clear other results
                    st.session_state.qa_answer = ""
                    logger.info(f"Disease analysis for '{selected_disease}' complete.")
                    st.success(f"Analysis for '{selected_disease}' complete!")

            # --- UMLS Search Action ---
            elif current_action == "umls_search":
                term_to_search = st.session_state.get("umls_search_term", "").strip()
                st.session_state.umls_results = None # Clear previous results
                st.session_state.umls_error = None   # Clear previous error
                if not term_to_search:
                    st.warning("UMLS search term is empty.")
                    logger.warning("UMLS search action triggered with empty term.")
                else:
                    st.info(f"🔎 Searching UMLS for: '{term_to_search}'...")
                    with st.spinner("Querying UMLS Metathesaurus..."):
                        try:
                            # Ensure utils and key are available (already checked by valid_action)
                            results: List[UMLSConcept] = umls_utils.search_umls(term_to_search, UMLS_APIKEY)
                            # Store the list of UMLSConcept objects (or empty list)
                            st.session_state.umls_results = results
                            logger.info(f"UMLS search for '{term_to_search}' returned {len(results)} result(s).")
                            st.success(f"UMLS search complete. Found {len(results)} concepts.")
                        except UMLSAuthError as auth_err:
                            err_msg = f"UMLS Authentication Failed: {auth_err}. Check your API key."
                            st.error(err_msg)
                            logger.error(f"UMLS Auth Error: {auth_err}", exc_info=False) # Don't need full traceback usually
                            st.session_state.umls_error = f"Authentication Error: {auth_err}"
                        except RuntimeError as search_err: # Catch generic runtime errors from search
                            err_msg = f"UMLS Search Failed: {search_err}. The service might be down or the query invalid."
                            st.error(err_msg)
                            logger.error(f"UMLS Search Runtime Error: {search_err}", exc_info=True)
                            st.session_state.umls_error = f"Search Error: {search_err}"
                        except Exception as e: # Catch any other unexpected errors
                            err_msg = f"An unexpected error occurred during UMLS search: {e}"
                            st.error(err_msg)
                            logger.critical(f"Unexpected UMLS error: {e}", exc_info=True)
                            st.session_state.umls_error = f"Unexpected error: {e}"

            # --- Confidence Estimation Action ---
            elif current_action == "confidence":
                # Double-check if there's actually content to analyze
                if not (current_history or st.session_state.initial_analysis or st.session_state.disease_analysis):
                    st.warning("No prior analysis or Q&A found to estimate confidence from.")
                    logger.warning("Confidence estimation skipped: No prior interactions.")
                else:
                    st.info("📊 Estimating AI confidence...")
                    with st.spinner("Calculating confidence score... Please wait."):
                        confidence_result = estimate_ai_confidence(
                            img=img_for_llm, # Pass the image
                            history=current_history,
                            initial_analysis=st.session_state.initial_analysis,
                            disease_analysis=st.session_state.disease_analysis,
                            roi=roi_coords
                        )
                    st.session_state.confidence_score = confidence_result
                    logger.info("Confidence estimation complete.")
                    st.success("Confidence estimation complete!")

            # --- Generate Report Data Action ---
            elif current_action == "generate_report_data":
                st.info("📄 Generating PDF report data...")
                st.session_state.pdf_report_bytes = None # Clear previous report
                image_for_report = st.session_state.get("display_image") # Use the display image

                if not isinstance(image_for_report, Image.Image):
                    st.error("Cannot generate report: No valid image is currently loaded.")
                    logger.error("Report generation failed: Missing display image.")
                else:
                    # --- Prepare Image for PDF (draw ROI if exists) ---
                    final_image_for_pdf = image_for_report.copy()
                    # Ensure it's RGB for PDF compatibility
                    if final_image_for_pdf.mode != "RGB":
                        final_image_for_pdf = final_image_for_pdf.convert("RGB")

                    if roi_coords and ImageDraw: # Check if ROI exists and ImageDraw is available
                        try:
                            draw = ImageDraw.Draw(final_image_for_pdf)
                            x0, y0 = roi_coords['left'], roi_coords['top']
                            x1, y1 = x0 + roi_coords['width'], y0 + roi_coords['height']
                            # Draw a red rectangle, adjust line width based on image size
                            line_width = max(2, int(min(final_image_for_pdf.size) * 0.004))
                            draw.rectangle(
                                [x0, y0, x1, y1],
                                outline="red",
                                width=line_width
                            )
                            logger.info("ROI bounding box drawn on image for PDF report.")
                        except Exception as e:
                            logger.error(f"Error drawing ROI on PDF image: {e}", exc_info=True)
                            st.warning("Could not draw ROI outline on the PDF image.")

                    # --- Format History for PDF ---
                    formatted_history = "No Q&A interactions recorded for this session."
                    if current_history:
                        lines = []
                        for q_type, msg in current_history:
                             # Simple text formatting, remove potential HTML/Markdown for basic PDF
                             cleaned_msg = re.sub('<[^<]+?>', '', str(msg)) # Basic HTML tag removal
                             lines.append(f"{q_type}: {cleaned_msg}")
                        formatted_history = "\n\n".join(lines)

                    # --- Gather Data for Report ---
                    report_data = {
                        "Session ID": st.session_state.session_id,
                        "Image Filename": (st.session_state.uploaded_file_info or "N/A").split('-')[0], # Extract filename
                        "Initial Analysis": st.session_state.initial_analysis or "Not Performed",
                        "Conversation History": formatted_history,
                        "Condition Analysis": st.session_state.disease_analysis or "Not Performed",
                        "AI Confidence Estimation": st.session_state.confidence_score or "Not Performed",
                    }

                    # --- Add DICOM Summary if available ---
                    dicom_summary_for_report = None
                    if st.session_state.is_dicom and st.session_state.dicom_metadata:
                         # Select key metadata fields for the summary
                         meta_keys_for_summary = [
                             'PatientName', 'PatientID', 'StudyDate', 'StudyTime',
                             'Modality', 'StudyDescription', 'SeriesDescription',
                             'Manufacturer', 'ManufacturerModelName'
                         ]
                         # Use get method to handle missing keys gracefully
                         meta_summary = {
                             k: st.session_state.dicom_metadata.get(k, 'N/A')
                             for k in meta_keys_for_summary
                             if st.session_state.dicom_metadata.get(k) # Only include if value exists
                         }
                         if meta_summary:
                             lines = [f"{k}: {v}" for k, v in meta_summary.items()]
                             report_data["DICOM Summary"] = "\n".join(lines)
                             dicom_summary_for_report = meta_summary # Pass structured data if needed

                    # --- Generate PDF Bytes ---
                    with st.spinner("Generating PDF document..."):
                        pdf_bytes = generate_pdf_report_bytes(
                            session_id=st.session_state.session_id,
                            image=final_image_for_pdf, # Image with ROI drawn
                            analysis_outputs=report_data, # Dictionary of text sections
                            # Pass full metadata if needed by report generator, else pass summary
                            dicom_metadata=st.session_state.dicom_metadata if st.session_state.is_dicom else None
                        )

                    if pdf_bytes:
                        st.session_state.pdf_report_bytes = pdf_bytes
                        st.success("PDF report data generated! Download available in the sidebar.")
                        logger.info("PDF report generated successfully.")
                        st.balloons() # Fun feedback!
                    else:
                        st.error("Failed to generate PDF report. Check logs for details.")
                        logger.error("PDF generator function returned None or empty bytes.")

            # --- Unknown Action ---
            else:
                st.warning(f"Action '{current_action}' is recognized but not implemented.")
                logger.warning(f"Unhandled action '{current_action}' triggered.")

        # --- Catch any unexpected errors during action execution ---
        except Exception as e:
            st.error(f"An unexpected error occurred during action '{current_action}': {e}")
            logger.critical(f"Action '{current_action}' failed with unexpected error: {e}", exc_info=True)

        # --- Cleanup after action handling ---
        finally:
            st.session_state.last_action = None # Reset the action trigger
            logger.debug(f"Action '{current_action}' handling complete.")
            st.rerun() # Rerun to reflect changes made by the action handler

# --- Footer ---
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced | Session ID: {st.session_state.get('session_id', 'N/A')}")
st.markdown(
    """
    <footer>
      <p>RadVision AI is intended for informational and educational purposes only. It is not a substitute for professional medical judgment, diagnosis, or treatment.</p>
      <p>
         <!-- Replace # with actual links if available -->
         <a href="#" target="_blank">Privacy Policy</a> |
         <a href="#" target="_blank">Terms of Service</a> |
         <a href="https://github.com/your_repo" target="_blank">GitHub</a> <!-- Example Link -->
      </p>
    </footer>
    """,
    unsafe_allow_html=True
)
logger.info(f"--- Application render complete for session: {st.session_state.session_id} ---")