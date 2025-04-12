# -*- coding: utf-8 -*-
"""
RadVision AI Advanced: AI-Assisted Medical Image Analysis Application

This Streamlit application allows users to upload medical images (JPG, PNG, DICOM),
perform AI-driven analysis (initial findings, Q&A, condition-specific checks),
define Regions of Interest (ROI), translate results, and generate PDF reports.
"""

# --- Streamlit Import and Configuration (MUST be the first Streamlit command) ---
import streamlit as st # Updated import

st.set_page_config(
    page_title="RadVision AI Advanced",
    layout="wide",
    page_icon="⚕️",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help', # Replace with your help link
        'Report a bug': "https://www.example.com/bug", # Replace with your bug report link
        'About': """
        **RadVision AI Advanced**

        AI-powered medical image analysis assistant.
        *For research and informational purposes only.*
        Version: 1.1 (Improved)
        """
    }
)

# --- Core Libraries ---
import io
import os
import uuid
import logging
import base64
from typing import Any, Dict, Optional, Tuple, List, Union
import copy
import random
import re
import hashlib
from pathlib import Path # Use pathlib for cleaner path operations

# --- UI/Interaction Libraries ---
import streamlit.elements.image as st_image # For potential monkey-patching check
try:
    from streamlit_drawable_canvas import st_canvas
    import streamlit_drawable_canvas as st_canvas_module
    CANVAS_AVAILABLE = True
    # Use __version__ if available, otherwise fallback
    CANVAS_VERSION = getattr(st_canvas_module, '__version__', getattr(st_canvas_module, 'version', 'Unknown'))
except ImportError:
    CANVAS_AVAILABLE = False
    CANVAS_VERSION = 'Not Installed'
    # No st.error here yet, handle downstream where canvas is used

# --- Image Processing ---
try:
    from PIL import Image, ImageDraw, UnidentifiedImageError
    import PIL
    PIL_AVAILABLE = True
    # Use __version__ if available, otherwise fallback
    PIL_VERSION = getattr(PIL, '__version__', getattr(PIL, 'version', 'Unknown'))
except ImportError:
    st.error("CRITICAL ERROR: Pillow (PIL) is not installed. Run `pip install Pillow`.")
    st.stop() # Pillow is essential

# --- DICOM Processing (Optional but Core Feature) ---
try:
    import pydicom
    import pydicom.errors
    PYDICOM_AVAILABLE = True
    # Use __version__ if available, otherwise fallback
    PYDICOM_VERSION = getattr(pydicom, '__version__', getattr(pydicom, 'version', 'Unknown'))
except ImportError:
    PYDICOM_AVAILABLE = False
    PYDICOM_VERSION = 'Not Installed'
    pydicom = None # Define pydicom as None if not installed

# --- Constants ---
APP_NAME = "RadVision AI Advanced"
ASSETS_DIR = Path("assets")
LOGO_PATH = ASSETS_DIR / "radvisionai-hero.jpeg"
DEMO_IMG_PATH = ASSETS_DIR / "demo.png"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
SESSION_ID_PREFIX = "radv-"
DEMO_SESSION_ID_PREFIX = "demo-"

# --- Logging Setup (Configure BEFORE first use) ---
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__) # Get logger for this module

# Log initial critical info
logger.info(f"--- {APP_NAME} App Start ---")
logger.info(f"Logging level set to: {LOG_LEVEL}")
# Use st.__version__ for streamlit version
logger.info(f"Streamlit version: {st.__version__}")
logger.info(f"Pillow (PIL) version: {PIL_VERSION}")

if CANVAS_AVAILABLE:
    logger.info(f"streamlit_drawable_canvas version: {CANVAS_VERSION}")
else:
    logger.warning("streamlit-drawable-canvas not installed. ROI functionality will be disabled.")

if PYDICOM_AVAILABLE:
    logger.info(f"pydicom version: {PYDICOM_VERSION}")
    # Check for optional DICOM handlers only if pydicom is present
    try:
        import pylibjpeg
        logger.info("pylibjpeg found (Enhanced DICOM compression support).")
    except ImportError:
        logger.info("pylibjpeg not found. Some DICOM compressions may not be supported.")
    try:
        import gdcm
        logger.info("python-gdcm found (Enhanced DICOM transfer syntax support).")
    except ImportError:
        logger.info("python-gdcm not found. Some DICOM transfer syntaxes may not be supported.")
else:
    logger.warning("pydicom not installed. DICOM functionality will be disabled.")


# --- Custom Application Modules (Import with Error Handling) ---
# Encapsulate imports to handle failures gracefully
try:
    from dicom_utils import parse_dicom, extract_dicom_metadata, dicom_to_image, get_default_wl
    from llm_interactions import run_initial_analysis, run_multimodal_qa, run_disease_analysis, estimate_ai_confidence
    from report_utils import generate_pdf_report_bytes
    from ui_components import display_dicom_metadata, dicom_wl_sliders
    APP_MODULES_AVAILABLE = True
    logger.info("Core custom utility modules imported successfully.")

    # HF Fallback (Optional, depends on llm_interactions structure)
    try:
        from hf_models import query_hf_vqa_inference_api, HF_VQA_MODEL_ID
        HF_FALLBACK_AVAILABLE = True
        logger.info(f"HF Fallback model configured: {HF_VQA_MODEL_ID}")
    except (ImportError, AttributeError) as e:
        HF_FALLBACK_AVAILABLE = False
        HF_VQA_MODEL_ID = None
        def query_hf_vqa_inference_api(img: Image.Image, question: str, roi: Optional[Dict] = None) -> Tuple[str, bool]:
            logger.warning(f"HF Fallback VQA called but not available/configured: {e}")
            return "[Fallback Unavailable] HF module or function not found/configured.", False
        logger.warning(f"HF VQA fallback disabled (hf_models.py import/attribute error: {e}).")

except ImportError as e:
    APP_MODULES_AVAILABLE = False
    HF_FALLBACK_AVAILABLE = False # If core modules fail, fallback likely won't work either
    st.error(f"CRITICAL ERROR: Failed to import core application module: {e}. Please check installation and file structure.")
    logger.critical(f"Failed to import core application module: {e}", exc_info=True)
    # We don't st.stop() immediately, allow basic UI rendering with error message.

# --- Translation Module (Optional Feature) ---
DEFAULT_LANGUAGES = { # Keep defaults even if module fails
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Japanese": "ja",
    "Chinese (Simplified)": "zh-CN", "Russian": "ru", "Arabic": "ar", "Hindi": "hi"
}
LANGUAGE_CODES = None
translate = None
detect_language = None
TRANSLATION_AVAILABLE = False

try:
    if not APP_MODULES_AVAILABLE:
        raise ImportError("Skipping translation import due to core module failure.") # Avoid cascading errors

    # Ensure the translation module itself is available
    from translation_models import translate as translate_func, LANGUAGE_CODES as lang_codes_from_module, detect_language as detect_func, DEEP_TRANSLATOR_AVAILABLE
    if not DEEP_TRANSLATOR_AVAILABLE:
        raise ImportError("deep-translator library not found within translation_models.")

    # Validate imported variables
    if callable(translate_func) and isinstance(lang_codes_from_module, dict) and lang_codes_from_module and callable(detect_func):
        translate = translate_func
        LANGUAGE_CODES = lang_codes_from_module
        detect_language = detect_func
        TRANSLATION_AVAILABLE = True
        logger.info("Translation module imported and validated successfully.")
    else:
        raise TypeError("Imported translation components are invalid (not callable or incorrect type).")

except (ImportError, TypeError, AttributeError) as e:
    LANGUAGE_CODES = DEFAULT_LANGUAGES # Use defaults for UI consistency
    logger.warning(f"Translation features disabled: {e}", exc_info=(LOG_LEVEL == "DEBUG")) # Only show full trace in DEBUG
    # Don't show st.warning here, handle it contextually in the translation tab
    translate = None
    detect_language = None
    TRANSLATION_AVAILABLE = False

except Exception as e: # Catch other unexpected errors
    LANGUAGE_CODES = DEFAULT_LANGUAGES
    logger.error(f"An unexpected error occurred importing translation_models: {e}", exc_info=True)
    translate = None
    detect_language = None
    TRANSLATION_AVAILABLE = False


# --- Function to Post-Process Translated Text ---
def format_translation(translated_text: str) -> str:
    """Attempts to restore basic list formatting possibly lost during translation."""
    if not isinstance(translated_text, str):
        return ""
    try:
        # Ensure numbered lists start on a new line with proper spacing
        # Ensure bullet points (* or -) start on a new line
        # Use non-capturing groups (?:...) to avoid altering the delimiter itself unnecessarily
        formatted = re.sub(r'\s*(?:(\d+\.\s+)|([-*]\s+))', r'\n\n\1\2', translated_text.strip())
        # Clean up potential multiple newlines introduced
        formatted = re.sub(r'\n{3,}', '\n\n', formatted)
        return formatted
    except Exception as e:
        logger.error(f"Error formatting translation: {e}", exc_info=True)
        return translated_text # Return original on error


# --- Monkey-Patch st.elements.image.image_to_url if missing (Less likely needed now) ---
# This might not be necessary in modern Streamlit versions. Kept for compatibility.
if not hasattr(st_image, "image_to_url"):
    logger.warning("Attempting to apply monkey-patch for st.elements.image.image_to_url.")
    def image_to_url_monkey_patch(img_obj: Any, *args, **kwargs) -> str:
        if isinstance(img_obj, Image.Image):
            try:
                buffered = io.BytesIO()
                fmt = "PNG" # PNG is generally safest for data URLs
                # Convert complex modes before saving
                temp_img = img_obj
                if img_obj.mode == 'P': # Palette mode often needs RGBA conversion for transparency
                    temp_img = img_obj.convert('RGBA')
                elif img_obj.mode not in ['RGB', 'L', 'RGBA']: # Convert other modes to RGB as a fallback
                    temp_img = img_obj.convert('RGB')

                temp_img.save(buffered, format=fmt)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/{fmt.lower()};base64,{img_str}"
            except Exception as e:
                logger.error(f"Monkey-patch failed during image conversion: {e}", exc_info=True)
                return "" # Return empty string on failure
        else:
            # This case should ideally not be hit if Streamlit internals are consistent
            logger.warning(f"Unsupported object type passed to image_to_url monkey-patch: {type(img_obj)}")
            return ""
    try:
        st_image.image_to_url = image_to_url_monkey_patch
        logger.info("Applied monkey-patch for st.elements.image.image_to_url successfully.")
    except Exception as patch_err:
         logger.error(f"Failed to apply monkey-patch for image_to_url: {patch_err}", exc_info=True)


# --- Utility Functions ---
def safe_image_to_data_url(img: Image.Image) -> str:
    """Converts a PIL Image to a base64 PNG data URL safely."""
    if not isinstance(img, Image.Image):
        logger.warning(f"Attempted to convert non-PIL Image to data URL: {type(img)}")
        return ""
    try:
        buffered = io.BytesIO()
        fmt = "PNG" # Always use PNG for data URLs for broad compatibility
        # Create a copy to avoid modifying the original object, especially during conversion
        temp_img = img.copy()

        # Ensure image mode compatibility for saving as PNG
        if temp_img.mode == 'P':
            temp_img = temp_img.convert('RGBA')
        elif temp_img.mode == 'CMYK' or temp_img.mode == 'I': # Convert complex modes to RGB
             temp_img = temp_img.convert('RGB')
        elif temp_img.mode not in ['RGB', 'L', 'RGBA']: # Fallback for other modes
             temp_img = temp_img.convert('RGB') # Convert to RGB if not already compatible

        temp_img.save(buffered, format=fmt)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{fmt.lower()};base64,{img_str}"
    except Exception as e:
        logger.error(f"Failed converting image to data URL: {e}", exc_info=True)
        return ""

def generate_file_hash(file_bytes: bytes) -> str:
    """Generates an MD5 hash for file content."""
    hasher = hashlib.md5()
    hasher.update(file_bytes)
    return hasher.hexdigest()

# --- Session State Initialization ---
def get_default_session_state() -> Dict[str, Any]:
    """Returns a dictionary with default session state values."""
    return {
        "session_id": None,
        "uploaded_file_info": None, # Stores "name-size-hash" to detect new uploads
        "raw_image_bytes": None,
        "is_dicom": False,
        "dicom_dataset": None,
        "dicom_metadata": {},
        "processed_image": None, # Image for AI analysis (normalized/consistent format)
        "display_image": None,   # Image for UI display (W/L applied for DICOM)
        "history": [],           # List of tuples: (question, answer, timestamp)
        "initial_analysis": "",
        "qa_answer": "",
        "disease_analysis": "",
        "confidence_score": "",
        "last_action": None,     # Tracks the last button press to handle actions
        "pdf_report_bytes": None,
        "canvas_drawing": None,  # Raw JSON data from the canvas
        "roi_coords": None,      # Processed ROI dict {left, top, width, height} in original image coords
        "current_display_wc": None,
        "current_display_ww": None,
        "translation_output": "",
        "translation_src_lang": "Auto-Detect", # Default source lang for UI
        "translation_tgt_lang": "Spanish",     # Default target lang for UI
        "demo_loaded": False,
        "clear_roi_triggered": False,          # Flag for showing ROI cleared message
        "error_message": None                  # Store persistent error messages if needed
    }

# Initialize session state if keys are missing or have incorrect types
default_state = get_default_session_state()
for key, value in default_state.items():
    if key not in st.session_state or type(st.session_state[key]) != type(value):
        # Deepcopy for mutable defaults like lists/dicts to avoid shared references
        st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value

# Ensure history is always a list (extra safety)
if not isinstance(st.session_state.history, list):
    logger.warning("Session state history was not a list, resetting.")
    st.session_state.history = []

# Ensure a unique Session ID exists
if not st.session_state.get("session_id"):
    st.session_state.session_id = f"{SESSION_ID_PREFIX}{str(uuid.uuid4())[:8]}"
    logger.info(f"New session started: {st.session_state.session_id}")


# --- Custom CSS (Includes Tab Scrolling Fix) ---
st.markdown(
    """
    <style>
      /* Base styling */
      body {
          font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; /* Modern font stack */
          background-color: #f4f4f8; /* Slightly softer background */
      }
      .stApp {
           background-color: #f4f4f8;
      }
      /* Sidebar styling - Adjust selector based on Streamlit version if needed */
      /* Common selector: div[data-testid="stSidebarNav"] + div[data-testid="stSidebar"] > div:first-child */
      div[data-testid="stSidebar"] > div:first-child {
          background-color: #ffffff;
          border-right: 1px solid #e0e0e0;
      }
      /* Footer styling */
      footer {
          text-align: center;
          font-size: 0.8em;
          color: #777777; /* Slightly darker grey */
          margin-top: 3em;
          padding-bottom: 1em;
      }
      /* Tab styling for horizontal scrolling on overflow */
      div[role="tablist"] {
          overflow-x: auto;
          white-space: nowrap;
          padding-bottom: 10px; /* Space below tabs */
          border-bottom: 1px solid #e0e0e0; /* Subtle separator */
          scrollbar-width: thin; /* Firefox */
          scrollbar-color: #cccccc #f0f0f0; /* Firefox */
      }
      /* Webkit scrollbar styling for tabs */
      div[role="tablist"]::-webkit-scrollbar {
          height: 8px;
      }
      div[role="tablist"]::-webkit-scrollbar-track {
          background: #f0f0f0;
          border-radius: 4px;
      }
      div[role="tablist"]::-webkit-scrollbar-thumb {
          background-color: #cccccc;
          border-radius: 4px;
          border: 2px solid #f0f0f0;
      }
       /* Style for individual tabs to prevent wrapping (redundant but safe) */
      div[role="tablist"] button {
          white-space: nowrap;
          margin-right: 5px; /* Add a small gap between tabs */
      }
      /* Style disabled text areas for better readability */
      .stTextArea textarea[disabled] {
            background-color: #f9f9f9; /* Slightly different background */
            color: #333; /* Ensure text is readable */
            /* Consider adding opacity: 0.9; */
            border: 1px solid #eee; /* Subtle border */
       }
    </style>
    """, unsafe_allow_html=True
)

# --- Display Header Logo ---
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=400) # Use str() for Path object
else:
    st.warning(f"Hero logo not found at expected location: {LOGO_PATH}")
    logger.warning(f"Hero logo file missing: {LOGO_PATH}")


# --- Sidebar ---
with st.sidebar:
    st.header("Image & Controls")

    # --- Tip of the Day ---
    TIPS = [
        "Use Demo Mode to quickly explore features.",
        "Draw an ROI rectangle on the image viewer to focus the AI.",
        "Adjust DICOM W/L sliders for optimal contrast.",
        "Review the Q&A History tab for the full conversation.",
        "Generate a PDF report to save your analysis session.",
        "Translate results using the Translation tab (if available).",
        "Clear the ROI selection using the 'Clear ROI' button.",
        "Check the DICOM Metadata expander for image details.",
    ]
    st.info(f"💡 {random.choice(TIPS)}")
    st.markdown("---")

    # --- Demo Mode ---
    demo_mode_active = st.checkbox(
        "🚀 Demo Mode",
        value=st.session_state.demo_loaded,
        help="Load a demo image and sample analysis. Disables file upload."
    )

    # Handle Demo Mode state change *precisely*
    if demo_mode_active != st.session_state.demo_loaded:
        if demo_mode_active: # Just Toggled ON
            if DEMO_IMG_PATH.exists():
                try:
                    logger.info("Activating Demo Mode...")
                    st.toast("Loading Demo Mode...", icon="⏳")
                    demo_img = Image.open(DEMO_IMG_PATH).convert("RGB")

                    # Reset state, preserving only specific UI elements if needed
                    preserved_keys = {"translation_src_lang", "translation_tgt_lang"} # Keep language prefs
                    default_state_values = get_default_session_state()
                    for key, value in default_state_values.items():
                        if key not in preserved_keys:
                            st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value

                    # Set demo-specific state
                    st.session_state.display_image = demo_img
                    st.session_state.processed_image = demo_img
                    st.session_state.session_id = f"{DEMO_SESSION_ID_PREFIX}{str(uuid.uuid4())[:4]}"
                    # Add timestamp to demo history entry
                    from datetime import datetime
                    ts_demo = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.history = [("System", "Demo mode activated. Sample analysis loaded.", ts_demo)]
                    st.session_state.initial_analysis = "This is a sample AI analysis for demonstration purposes.\n\n**Findings:**\n1. A subtle opacity is noted in the upper right lung field.\n2. The cardiac silhouette appears normal.\n3. No clear signs of pneumothorax or significant effusion.\n\n**Impression:**\nPossible early-stage infiltrate or artifact in the RUL, recommend comparison with prior studies or follow-up imaging if clinically indicated."
                    st.session_state.demo_loaded = True
                    st.session_state.uploaded_file_info = f"demo.png-{DEMO_IMG_PATH.stat().st_size}-demo_hash" # Simulate file info
                    st.success("Demo Mode Activated!")
                    logger.info(f"Demo mode activated successfully. Session: {st.session_state.session_id}")
                    st.rerun() # Rerun to reflect changes immediately
                except Exception as e:
                    st.sidebar.error(f"Error loading demo image: {e}")
                    logger.error(f"Failed to load demo image '{DEMO_IMG_PATH}': {e}", exc_info=True)
                    st.session_state.demo_loaded = False # Ensure flag matches reality
                    # No rerun, let user see the error
            else:
                st.sidebar.warning(f"Demo image ({DEMO_IMG_PATH.name}) not found in assets.")
                logger.warning(f"Demo image file missing: {DEMO_IMG_PATH}")
                st.session_state.demo_loaded = False # Keep checkbox state consistent
                # No rerun needed here

        else: # Just Toggled OFF
            logger.info("Deactivating Demo Mode.")
            st.toast("Exiting Demo Mode...", icon="🚪")
            # Reset state back to defaults, preserving specific settings
            preserved_keys = {"translation_src_lang", "translation_tgt_lang"}
            default_state_values = get_default_session_state()
            for key, value in default_state_values.items():
                 if key not in preserved_keys:
                     st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value

            # Explicitly set demo_loaded to False and generate a new real session ID
            st.session_state.demo_loaded = False
            st.session_state.session_id = f"{SESSION_ID_PREFIX}{str(uuid.uuid4())[:8]}"
            st.info("Demo mode deactivated. Upload an image to begin.")
            logger.info(f"Demo mode deactivated. New session: {st.session_state.session_id}")
            st.rerun()

    # --- File Uploader ---
    uploaded_file = st.file_uploader(
        "Upload Image (JPG, PNG, DCM)",
        type=["jpg", "jpeg", "png", "dcm", "dicom"],
        key="file_uploader_widget", # Consistent key helps maintain state if needed
        help="Upload your medical image file. Disabled in Demo Mode.",
        disabled=st.session_state.demo_loaded # Disable when demo is active
    )

    # --- File Processing Logic ---
    if uploaded_file is not None and not st.session_state.demo_loaded:
        process_file = False
        try:
            # Read file bytes ONCE
            raw_bytes = uploaded_file.getvalue()
            file_hash = generate_file_hash(raw_bytes)
            new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{file_hash}"

            # Check if this is genuinely a new file upload
            if new_file_info != st.session_state.get("uploaded_file_info"):
                process_file = True
                logger.info(f"New file detected: {uploaded_file.name} (Size: {uploaded_file.size}, Hash: {file_hash[:8]}...)")
                st.toast(f"Processing '{uploaded_file.name}'...", icon="⏳")

                # Reset application state for the new file
                logger.debug("Resetting session state for new file.")
                preserved_keys = {"file_uploader_widget", "translation_src_lang", "translation_tgt_lang"}
                default_state_values = get_default_session_state()
                for key, value in default_state_values.items():
                    if key not in preserved_keys:
                        st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value

                # Store info about the newly uploaded file
                st.session_state.uploaded_file_info = new_file_info
                st.session_state.session_id = f"{SESSION_ID_PREFIX}{str(uuid.uuid4())[:8]}" # New session ID
                st.session_state.raw_image_bytes = raw_bytes
                st.session_state.demo_loaded = False # Explicitly ensure demo is off
                logger.info(f"Processing new file. New session: {st.session_state.session_id}")

            else:
                logger.debug("Uploaded file widget triggered, but file info hasn't changed. No reprocessing.")

        except Exception as e:
            st.error(f"Error reading or identifying file: {e}")
            logger.error(f"Failed during file upload pre-processing: {e}", exc_info=True)
            # Clear potentially corrupted state
            st.session_state.uploaded_file_info = None
            st.session_state.raw_image_bytes = None
            process_file = False

        # Proceed only if it's a new file and reading was successful
        if process_file:
            file_ext = Path(uploaded_file.name).suffix.lower()
            # Check if DICOM processing is possible and likely
            is_dicom_possible = PYDICOM_AVAILABLE and pydicom is not None
            is_dicom_likely = is_dicom_possible and (file_ext in (".dcm", ".dicom") or "dicom" in uploaded_file.type.lower())

            with st.spinner("🔬 Analyzing image format..."):
                temp_display_img: Optional[Image.Image] = None
                temp_processed_img: Optional[Image.Image] = None
                is_dicom_confirmed = False
                processing_successful = False

                # 1. Attempt DICOM Processing (if likely and possible)
                if is_dicom_likely:
                    logger.info(f"Attempting DICOM processing for {uploaded_file.name}.")
                    try:
                        # Ensure dicom_utils are available before calling
                        if not APP_MODULES_AVAILABLE:
                           raise ImportError("Cannot process DICOM, core modules unavailable.")

                        ds = parse_dicom(st.session_state.raw_image_bytes, uploaded_file.name)
                        if ds:
                            metadata = extract_dicom_metadata(ds)
                            wc, ww = get_default_wl(ds) # Get windowing from metadata or defaults
                            logger.debug(f"Default DICOM W/L: WC={wc}, WW={ww}")

                            # Generate initial display image with default W/L
                            img_display = dicom_to_image(ds, wc, ww, normalize=False)
                            # Generate processed image (maybe normalized or using default W/L depending on AI needs)
                            # For consistency, let's use a normalized version for the AI if possible,
                            # or fallback to default W/L if normalization fails.
                            img_processed = dicom_to_image(ds, wc, ww, normalize=True) # Try normalized first
                            if not isinstance(img_processed, Image.Image):
                                logger.warning("DICOM normalization failed, using default W/L for processed image.")
                                img_processed = dicom_to_image(ds, wc, ww, normalize=False) # Fallback


                            if isinstance(img_display, Image.Image) and isinstance(img_processed, Image.Image):
                                temp_display_img = img_display.convert("RGB") # Ensure RGB for display
                                # Ensure processed image is also suitable (e.g., RGB or Grayscale L)
                                temp_processed_img = img_processed.convert("RGB") if img_processed.mode != 'L' else img_processed

                                st.session_state.dicom_dataset = ds
                                st.session_state.dicom_metadata = metadata
                                st.session_state.current_display_wc = wc
                                st.session_state.current_display_ww = ww
                                is_dicom_confirmed = True
                                processing_successful = True
                                logger.info("DICOM processing successful.")
                            else:
                                logger.warning("DICOM parsing succeeded, but image conversion failed. Treating as non-DICOM.")
                                is_dicom_likely = False # Force fallback
                        else:
                            logger.warning("parse_dicom returned None. File might not be DICOM or is unreadable.")
                            is_dicom_likely = False # Force fallback to standard image processing

                    except pydicom.errors.InvalidDicomError:
                        logger.warning(f"'{uploaded_file.name}' is not a valid DICOM file (InvalidDicomError). Attempting standard image processing.")
                        is_dicom_likely = False
                    except ImportError as ie: # Catch if dicom_utils failed import earlier
                        st.error(f"DICOM processing skipped: {ie}")
                        logger.error(f"DICOM processing skipped due to import error: {ie}")
                        is_dicom_likely = False
                    except Exception as e:
                        st.error(f"Error processing as DICOM: {e}")
                        logger.error(f"DICOM processing failed unexpectedly: {e}", exc_info=True)
                        is_dicom_likely = False # Attempt standard processing as fallback

                # 2. Attempt Standard Image Processing (if not DICOM or DICOM failed)
                if not is_dicom_confirmed:
                    st.session_state.is_dicom = False
                    logger.info(f"Attempting standard image processing for {uploaded_file.name}.")
                    try:
                        img = Image.open(io.BytesIO(st.session_state.raw_image_bytes))
                        # Convert to RGB immediately after opening for broad compatibility
                        # Let's keep grayscale images as 'L' if they are originally grayscale
                        if img.mode == 'L':
                            processed_img_fmt = img.copy()
                            display_img_fmt = img.convert("RGB") # Display needs RGB generally
                        else:
                             processed_img_fmt = img.convert("RGB")
                             display_img_fmt = processed_img_fmt.copy()

                        temp_display_img = display_img_fmt
                        temp_processed_img = processed_img_fmt
                        processing_successful = True
                        logger.info("Standard image processing successful.")
                    except UnidentifiedImageError:
                        st.error("Unsupported or corrupted image format. Please upload JPG, PNG, or a valid DICOM file.")
                        logger.error(f"UnidentifiedImageError for file {uploaded_file.name}.")
                        processing_successful = False
                    except Exception as e:
                        st.error(f"Error processing image file: {e}")
                        logger.error(f"Standard image processing failed: {e}", exc_info=True)
                        processing_successful = False

                # 3. Finalize State Update
                if processing_successful and temp_display_img and temp_processed_img:
                    st.session_state.is_dicom = is_dicom_confirmed
                    st.session_state.display_image = temp_display_img
                    st.session_state.processed_image = temp_processed_img
                    file_type = "DICOM" if is_dicom_confirmed else "Image"
                    st.success(f"✅ {file_type} '{uploaded_file.name}' processed!")
                    logger.info(f"Image processing finalized successfully. Is DICOM: {is_dicom_confirmed}")
                    st.rerun() # Rerun to update the UI immediately with the new image/state
                else:
                    st.error("Image loading and processing failed. Please check the file or try a different one.")
                    # Reset relevant state variables on failure to prevent inconsistent state
                    st.session_state.uploaded_file_info = None
                    st.session_state.raw_image_bytes = None
                    st.session_state.display_image = None
                    st.session_state.processed_image = None
                    st.session_state.is_dicom = False
                    st.session_state.dicom_dataset = None
                    logger.error("Image processing failed overall. State reset.")
                    # No rerun here, let the user see the error message clearly

    # --- DICOM Window/Level Sliders ---
    if st.session_state.is_dicom and st.session_state.dicom_dataset is not None and isinstance(st.session_state.display_image, Image.Image):
        st.markdown("---")
        st.subheader("DICOM Windowing")
        if not APP_MODULES_AVAILABLE:
             st.warning("DICOM UI components unavailable (module import failed).")
        else:
            try:
                # Use the imported UI component function
                wc_new, ww_new = dicom_wl_sliders(
                    st.session_state.dicom_dataset,
                    st.session_state.current_display_wc,
                    st.session_state.current_display_ww
                )

                # Check if W/L values *actually* changed before reprocessing
                if wc_new != st.session_state.current_display_wc or ww_new != st.session_state.current_display_ww:
                    logger.info(f"DICOM W/L changed via UI: WC={wc_new}, WW={ww_new}")
                    st.session_state.current_display_wc = wc_new
                    st.session_state.current_display_ww = ww_new
                    with st.spinner("Updating DICOM view..."):
                        # Ensure dicom_utils is available
                        if not APP_MODULES_AVAILABLE:
                             raise ImportError("Cannot update DICOM view, core modules unavailable.")

                        new_display_img = dicom_to_image(st.session_state.dicom_dataset, wc_new, ww_new, normalize=False)
                        if isinstance(new_display_img, Image.Image):
                            st.session_state.display_image = new_display_img.convert("RGB")
                            # Decide if processed_image should also change. Generally, no, unless AI should adapt to view settings.
                            # logger.debug("Display image updated with new W/L settings.")
                            st.rerun() # Rerun needed to show the updated image
                        else:
                            st.warning("Failed to update DICOM view with new W/L settings.")
                            logger.warning("dicom_to_image returned non-image during W/L update.")
            except ImportError as ie:
                 st.error(f"DICOM update skipped: {ie}")
                 logger.error(f"DICOM update skipped due to import error: {ie}")
            except Exception as e:
                st.error(f"Error updating DICOM view or sliders: {e}")
                logger.error(f"Error in DICOM W/L slider section: {e}", exc_info=True)

    st.markdown("---")

    # --- Clear ROI Button ---
    if CANVAS_AVAILABLE: # Only show if canvas is installed
        if st.button("🗑️ Clear ROI", help="Clear the current Region of Interest selection", disabled=st.session_state.roi_coords is None):
            st.session_state.roi_coords = None
            st.session_state.canvas_drawing = None # Reset raw canvas state too
            st.session_state.clear_roi_triggered = True # Use flag for feedback after rerun
            logger.info("Clear ROI button clicked.")
            st.rerun() # Rerun immediately to clear canvas visually and update UI state

    st.markdown("---")
    st.header("AI Actions")

    # Determine if core AI actions can be performed
    img_available_for_ai = isinstance(st.session_state.get("processed_image"), Image.Image)
    actions_enabled = APP_MODULES_AVAILABLE and img_available_for_ai

    # --- AI Action Buttons ---
    if st.button("▶️ Run Initial Analysis", key="analyze_btn", help="Perform a general AI analysis of the image.", disabled=not actions_enabled):
        st.session_state.last_action = "analyze"
        st.rerun() # Rerun now to trigger action handling logic at the end

    st.subheader("❓ Ask AI a Question")
    question_input = st.text_area(
        "Your Question:", height=80, key="question_input_widget",
        placeholder="e.g., 'Are there any signs of fracture?' or 'Describe the highlighted region.'",
        disabled=not actions_enabled
    )
    if st.button("💬 Ask AI", key="ask_btn", help="Submit your question about the image/ROI to the AI.", disabled=not actions_enabled):
        if question_input.strip():
            st.session_state.last_action = "ask"
            st.rerun() # Rerun to handle action
        else:
            st.warning("Please enter a question.")
        # No automatic rerun if question is empty

    st.subheader("🎯 Condition Analysis")
    # Consider making this list configurable or loaded from a file
    DISEASE_OPTIONS = [
        "", # Add empty option for placeholder
        "Pneumonia", "Lung Cancer", "Nodule/Mass", "Effusion", "Pneumothorax",
        "Fracture", "Arthritis", "Osteoporosis",
        "Stroke", "Brain Tumor", "Hemorrhage",
        "Appendicitis", "Bowel Obstruction", "Cardiomegaly", "Aortic Aneurysm",
        "Tuberculosis", "COVID-19", "Pulmonary Embolism", "Other (Specify in Q&A)"
    ]
    disease_select = st.selectbox(
        "Select Condition:", sorted(DISEASE_OPTIONS), key="disease_select_widget",
        help="Select a potential condition for focused AI analysis.",
        disabled=not actions_enabled
    )
    if st.button("🩺 Run Condition Analysis", key="disease_btn", help="Analyze the image specifically for the selected condition.", disabled=not actions_enabled):
        if disease_select:
            st.session_state.last_action = "disease"
            st.rerun() # Rerun to handle action
        else:
            st.warning("Please select a condition.")
        # No rerun if condition is empty

    st.subheader("📊 Confidence & Report")
    # Confidence estimation typically relies on previous interactions (history)
    can_estimate_confidence = actions_enabled and bool(st.session_state.history)
    if st.button("📈 Estimate Confidence", key="confidence_btn", disabled=not can_estimate_confidence, help="Estimate the AI's confidence based on the conversation history."):
        st.session_state.last_action = "confidence"
        st.rerun() # Rerun to handle action

    # PDF generation only needs an image and some results
    can_generate_report = img_available_for_ai and APP_MODULES_AVAILABLE
    if st.button("📄 Generate PDF Data", key="generate_report_data_btn", help="Compile the analysis results into data for PDF report.", disabled=not can_generate_report):
        st.session_state.last_action = "generate_report_data"
        st.rerun() # Rerun to handle action

    # Download button appears *only after* PDF bytes are generated
    if st.session_state.get("pdf_report_bytes"):
        pdf_fname = f"RadVisionAI_Report_{st.session_state.session_id or 'session'}.pdf"
        st.download_button(
            label="⬇️ Download PDF Report",
            data=st.session_state.pdf_report_bytes,
            file_name=pdf_fname,
            mime="application/pdf",
            key="download_pdf_button",
            help="Click to download the generated PDF report."
        )
        # Option: Add a button to clear the generated report state
        # if st.button("Clear Generated Report", key="clear_report_btn"):
        #     st.session_state.pdf_report_bytes = None
        #     st.rerun()


# --- Main Content Area ---
st.title(f"⚕️ {APP_NAME}")

# Show message if critical modules failed
if not APP_MODULES_AVAILABLE:
    st.error("Core application features are unavailable due to module import errors. Please check the logs and setup.")
    # Consider st.stop() if the app is unusable without core modules
    # st.stop()

# Show ROI Cleared Confirmation Message (if triggered)
if st.session_state.get("clear_roi_triggered", False):
    st.success("Region of Interest (ROI) cleared!")
    st.balloons() # Optional visual feedback
    st.session_state.clear_roi_triggered = False # Reset flag immediately


with st.expander("User Guide & Disclaimer", expanded=False):
    st.warning("⚠️ **Disclaimer:** This tool is intended for research, educational, and informational purposes **only**. It is **not** a substitute for professional medical advice, diagnosis, or treatment. AI-generated insights **must** be critically reviewed and verified by qualified healthcare professionals before being considered in any clinical decision-making process.")
    st.markdown(
        """
        **Workflow:**
        1.  **Upload/Demo:** Use the sidebar to upload JPG, PNG, or DICOM images, OR activate Demo Mode.
        2.  **(Optional) Adjust DICOM:** If a DICOM image is loaded, use the 'DICOM Windowing' sliders in the sidebar to optimize contrast (WC/WW).
        3.  **(Optional) Select ROI:** If `streamlit-drawable-canvas` is installed, draw a rectangle on the 'Image Viewer' below to define a Region of Interest for focused analysis. Use the 'Clear ROI' button in the sidebar to remove it.
        4.  **Analyze:** Use the 'AI Actions' buttons in the sidebar (`Run Initial Analysis`, `Ask AI`, `Run Condition Analysis`) to interact with the AI. Actions will consider the ROI if one is defined.
        5.  **Review Results:** Check the analysis outputs in the corresponding tabs on the right ('Initial Analysis', 'Q&A History', etc.).
        6.  **(Optional) Translate:** Navigate to the 'Translation' tab to translate analysis text into different languages (if the translation module is available).
        7.  **Confidence & Report:** Use 'Estimate Confidence' (after Q&A) and 'Generate PDF Data'/'Download PDF Report' for summarization.
        """
    )
st.markdown("---")

# --- Two-Column Layout ---
col1, col2 = st.columns([2, 3]) # Image/Controls on left, Results on right

with col1:
    st.subheader("🖼️ Image Viewer")
    display_img = st.session_state.get("display_image")

    if isinstance(display_img, Image.Image):
        # --- Display Image ---
        img_caption = "Current View"
        if st.session_state.is_dicom:
            wc_disp = st.session_state.current_display_wc
            ww_disp = st.session_state.current_display_ww
            img_caption += f" (DICOM | WC: {wc_disp if wc_disp is not None else 'N/A'}, WW: {ww_disp if ww_disp is not None else 'N/A'})"
        if st.session_state.roi_coords:
            img_caption += " [ROI Active]"
        st.image(display_img, caption=img_caption, use_container_width=True)

        # --- ROI Drawing Canvas (Conditional) ---
        if CANVAS_AVAILABLE:
            st.markdown("---")
            st.caption("Draw a rectangle below to select a Region of Interest (ROI).")

            # Dynamically calculate canvas size based on image aspect ratio, bounded by max dimensions
            MAX_CANVAS_WIDTH = 600
            MAX_CANVAS_HEIGHT = 500
            img_w, img_h = display_img.size

            if img_w > 0 and img_h > 0:
                aspect_ratio = img_w / img_h
                canvas_width = min(img_w, MAX_CANVAS_WIDTH)
                canvas_height = int(canvas_width / aspect_ratio)

                if canvas_height > MAX_CANVAS_HEIGHT:
                    canvas_height = MAX_CANVAS_HEIGHT
                    canvas_width = int(canvas_height * aspect_ratio)

                # Ensure minimum dimensions
                canvas_width = max(int(canvas_width), 150)
                canvas_height = max(int(canvas_height), 150)

                logger.debug(f"Canvas dimensions calculated: {canvas_width}x{canvas_height}")

                try:
                    # Use BytesIO for background image for potential reliability improvements
                    buffered_bg = io.BytesIO()
                    bg_img_for_canvas = display_img.copy()
                    # Ensure the background image is in a format st_canvas handles well (RGB/RGBA)
                    if bg_img_for_canvas.mode not in ['RGB', 'RGBA']:
                         bg_img_for_canvas = bg_img_for_canvas.convert('RGB')
                    bg_img_for_canvas.save(buffered_bg, format="PNG")
                    buffered_bg.seek(0) # Reset buffer position

                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.25)", # Semi-transparent orange fill
                        stroke_width=2,
                        stroke_color="rgba(255, 0, 0, 0.8)",  # Brighter red border
                        background_image=Image.open(buffered_bg), # Load from buffer
                        update_streamlit=True, # Trigger Streamlit updates on drawing actions
                        height=canvas_height,
                        width=canvas_width,
                        drawing_mode="rect", # Mode for drawing rectangles
                        # Restore previous state if exists; handle potential malformed data
                        initial_drawing=st.session_state.get("canvas_drawing") if isinstance(st.session_state.get("canvas_drawing"), dict) else None,
                        key="drawable_canvas" # Essential unique key
                    )

                    # --- Process Canvas ROI Result ---
                    # Check if canvas_result and json_data exist before proceeding
                    if canvas_result and canvas_result.json_data and isinstance(canvas_result.json_data.get("objects"), list):
                        current_roi = None
                        # If objects exist, process the last one (assuming single ROI)
                        if canvas_result.json_data["objects"]:
                            # Ensure the last object is a dictionary before accessing keys
                            last_obj = canvas_result.json_data["objects"][-1]
                            if isinstance(last_obj, dict) and last_obj.get("type") == "rect":
                                # Extract rect properties safely using .get() with defaults
                                scaleX = last_obj.get("scaleX", 1.0)
                                scaleY = last_obj.get("scaleY", 1.0)
                                left = last_obj.get("left", 0)
                                top = last_obj.get("top", 0)
                                width = last_obj.get("width", 0)
                                height = last_obj.get("height", 0)

                                # Ensure coordinates are numeric before calculation
                                if all(isinstance(n, (int, float)) for n in [left, top, width, height, scaleX, scaleY]):
                                    left_scaled = max(0, int(left))
                                    top_scaled = max(0, int(top))
                                    width_scaled = max(0, int(width * scaleX))
                                    height_scaled = max(0, int(height * scaleY))

                                    # Calculate scaling factor from canvas display size to original image size
                                    scale_x_img = img_w / canvas_width
                                    scale_y_img = img_h / canvas_height

                                    # Calculate ROI coordinates in the *original* image dimensions
                                    orig_left = max(0, int(left_scaled * scale_x_img))
                                    orig_top = max(0, int(top_scaled * scale_y_img))
                                    orig_width = int(width_scaled * scale_x_img)
                                    orig_height = int(height_scaled * scale_y_img)

                                    # Clamp width/height to ensure ROI stays within image boundaries
                                    orig_width = max(0, min(orig_width, img_w - orig_left))
                                    orig_height = max(0, min(orig_height, img_h - orig_top))

                                    # Define a minimum valid ROI size (e.g., 10x10 pixels)
                                    MIN_ROI_DIM = 10
                                    if orig_width >= MIN_ROI_DIM and orig_height >= MIN_ROI_DIM:
                                        current_roi = {"left": orig_left, "top": orig_top, "width": orig_width, "height": orig_height}
                                    else:
                                        logger.debug(f"Ignoring drawn ROI, too small or invalid: {orig_width}x{orig_height}")
                                        current_roi = None # Treat as no ROI if too small
                                else:
                                    logger.warning(f"Invalid numeric type found in ROI object: {last_obj}")
                                    current_roi = None
                            else:
                                logger.debug("Last canvas object was not a valid rectangle dictionary.")
                        # else: No objects drawn or present

                        # --- Update Session State Only If ROI Changed ---
                        # Use json comparison for potentially nested dicts, though simple != should work for this structure
                        import json
                        current_roi_json = json.dumps(current_roi, sort_keys=True) if current_roi else None
                        prev_roi_json = json.dumps(st.session_state.roi_coords, sort_keys=True) if st.session_state.roi_coords else None

                        if prev_roi_json != current_roi_json:
                            st.session_state.roi_coords = current_roi
                            st.session_state.canvas_drawing = canvas_result.json_data # Store the raw canvas state
                            if current_roi:
                                logger.info(f"ROI updated: {current_roi}")
                                st.toast("ROI Set!", icon="🎯")
                            else:
                                # This handles both clearing by drawing tiny rect and clearing by removing objects
                                logger.info("ROI cleared or became invalid.")
                                st.toast("ROI Cleared.", icon="🗑️")
                            st.rerun() # Rerun is necessary to reflect ROI state change (e.g., in caption, AI actions)

                except Exception as e:
                    st.error(f"Error initializing or processing drawing canvas: {e}")
                    logger.error(f"Canvas initialization/processing error: {e}", exc_info=True)

            else:
                st.warning("Cannot enable ROI drawing: Image has invalid dimensions.")
                logger.warning("Skipping canvas setup due to zero image dimensions.")
        elif not CANVAS_AVAILABLE:
             st.info("Install `streamlit-drawable-canvas` for ROI selection capability.")

        # --- DICOM Metadata Display ---
        if st.session_state.is_dicom and st.session_state.dicom_metadata:
            st.markdown("---")
            with st.expander("DICOM Metadata", expanded=False):
                 if not APP_MODULES_AVAILABLE:
                     st.warning("DICOM UI components unavailable (module import failed).")
                 else:
                    try:
                        # Use the cleaner ui_components function
                        display_dicom_metadata(st.session_state.dicom_metadata)
                    except Exception as e:
                        st.error(f"Error displaying DICOM metadata: {e}")
                        logger.error(f"Error calling display_dicom_metadata: {e}", exc_info=True)

    else:
        # Message when no image is loaded or processed
        st.info("👋 Welcome! Upload an image using the sidebar or activate Demo Mode to begin analysis.")


with col2:
    st.subheader("📊 Analysis & Results")

    # Define tab titles dynamically based on feature availability
    tab_titles = ["🔬 Initial Analysis", "💬 Q&A History", "🩺 Disease Focus", "📈 Confidence"]
    if TRANSLATION_AVAILABLE:
        tab_titles.append("🌐 Translation")
    else:
         # Optionally add a disabled placeholder tab or just omit it
         # tab_titles.append("🌐 Translation (N/A)")
         logger.info("Translation tab omitted as feature is unavailable.")

    try:
        tabs = st.tabs(tab_titles)
    except Exception as e:
        st.error(f"Fatal Error: Failed to create Streamlit tabs: {e}")
        logger.critical(f"Error creating st.tabs: {e}", exc_info=True)
        tabs = [] # Assign empty list to prevent NameError downstream, though app is likely broken
        st.stop() # Stop if essential UI element fails

    # --- Tab Contents ---
    # Ensure tabs list is not empty before trying to access elements
    if tabs:
        # Tab 0: Initial Analysis
        with tabs[0]:
            st.text_area(
                "Overall Findings & Impressions",
                value=st.session_state.initial_analysis or "Run 'Initial Analysis' from the sidebar.",
                height=400, # Adjust height as needed
                key="output_initial",
                disabled=True, # Display only
                help="Results from the 'Run Initial Analysis' action."
            )

        # Tab 1: Q&A History
        with tabs[1]:
            st.text_area(
                "Latest AI Answer",
                value=st.session_state.qa_answer or "Ask a question using the sidebar.",
                height=150, # Shorter area for the latest answer
                key="output_qa",
                disabled=True,
                help="The most recent answer from the 'Ask AI' action."
            )
            st.markdown("---")
            st.subheader("Conversation History")
            if st.session_state.history:
                # Display history in reverse chronological order (newest first)
                # Use st.container for better layout control within the loop
                history_container = st.container()
                with history_container:
                    # Apply max height and scrollbar if history gets long
                    st.markdown('<div style="max-height: 300px; overflow-y: auto; border: 1px solid #e0e0e0; padding: 10px; border-radius: 5px; background-color: #ffffff;">', unsafe_allow_html=True)
                    for i, entry in enumerate(reversed(st.session_state.history)):
                        # Unpack entry, handling potential missing timestamp if old format
                        q, a, ts = entry if len(entry) == 3 else (*entry, "N/A")
                        entry_num = len(st.session_state.history) - i
                        # Use markdown quote for questions for visual distinction
                        st.markdown(f"> **You ({entry_num}):** `{q}`")
                        # Use markdown with potentially unsafe HTML if answers contain formatting
                        answer_display = a if a else "[No Answer Recorded]"
                        st.markdown(f"**AI ({entry_num}):**\n{answer_display}", unsafe_allow_html=True)
                        # Optional: Display timestamp if available and meaningful
                        if ts != "N/A": st.caption(f"Timestamp: {ts}")
                        if i < len(st.session_state.history) - 1:
                            st.markdown("---") # Separator
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.caption("No questions asked yet in this session.")

        # Tab 2: Disease Focus
        with tabs[2]:
            st.text_area(
                "Disease-Specific Analysis",
                value=st.session_state.disease_analysis or "Run 'Condition Analysis' from the sidebar.",
                height=400,
                key="output_disease",
                disabled=True,
                help="Results from the 'Run Condition Analysis' action."
            )

        # Tab 3: Confidence
        with tabs[3]:
            st.text_area(
                "AI Confidence Estimation",
                value=st.session_state.confidence_score or "Run 'Estimate Confidence' from the sidebar.",
                height=400,
                key="output_confidence",
                disabled=True,
                help="An estimation of the AI's confidence based on the analysis context."
            )

        # Tab 4: Translation (Conditional)
        if TRANSLATION_AVAILABLE and len(tabs) > 4: # Check index exists
            with tabs[4]:
                st.subheader("🌐 Translate Analysis Results")
                st.caption("Translate analysis text or enter custom text.")

                # 1. Select Text Source
                text_sources = {
                    "Initial Analysis": st.session_state.initial_analysis,
                    "Latest Q&A Answer": st.session_state.qa_answer,
                    "Disease Analysis": st.session_state.disease_analysis,
                    "Confidence Estimation": st.session_state.confidence_score,
                    "(Enter Custom Text Below)": "" # Placeholder for custom
                }
                # Filter out empty sources unless it's the custom option
                available_sources = {k: v for k, v in text_sources.items() if v or k == "(Enter Custom Text Below)"}
                if not available_sources: # Handle case where all analyses are empty
                     available_sources = {"(Enter Custom Text Below)": ""}
                     st.info("No analysis text available yet to translate. Use custom input.")

                selected_label = st.selectbox(
                    "Select text to translate:",
                    options=list(available_sources.keys()),
                    key="translate_source_select",
                    index=0 # Default to first available option
                )

                # 2. Determine Text Input
                is_custom_text = (selected_label == "(Enter Custom Text Below)")
                text_to_translate = ""

                if is_custom_text:
                    text_to_translate_input = st.text_area(
                        "Enter text to translate here:", "", height=150, key="translate_custom_input"
                    )
                    text_to_translate = text_to_translate_input
                else:
                    selected_text_value = available_sources.get(selected_label, "")
                    st.text_area(
                        f"Selected text ({selected_label}):",
                        value=selected_text_value, # Should not be empty due to filtering above
                        height=150,
                        key="translate_selected_display",
                        disabled=True
                    )
                    text_to_translate = selected_text_value

                # 3. Language Selection (using validated LANGUAGE_CODES)
                if not isinstance(LANGUAGE_CODES, dict) or not LANGUAGE_CODES:
                     st.error("Internal Error: Language codes configuration is missing or invalid.")
                     logger.critical("Translation Widget Error: LANGUAGE_CODES is invalid.")
                else:
                    lang_names = sorted(list(LANGUAGE_CODES.keys()))
                    try:
                        # Ensure persisted target language is valid, otherwise use default
                        default_tgt_name = st.session_state.translation_tgt_lang
                        if default_tgt_name not in lang_names:
                            default_tgt_name = "Spanish" if "Spanish" in lang_names else lang_names[0] # Sensible default
                        default_tgt_index = lang_names.index(default_tgt_name)
                    except (ValueError, IndexError):
                         default_tgt_index = 0 # Fallback to first language

                    col_lang1, col_lang2 = st.columns(2)
                    with col_lang1:
                        src_lang_options = ["Auto-Detect"] + lang_names
                        try:
                            # Ensure persisted source language is valid, otherwise use default
                            default_src_name = st.session_state.translation_src_lang
                            if default_src_name not in src_lang_options:
                                default_src_name = "Auto-Detect" # Sensible default
                            default_src_index = src_lang_options.index(default_src_name)
                        except ValueError:
                            default_src_index = 0 # Fallback to Auto-Detect
                        selected_src_lang_name = st.selectbox(
                            "Source Language:",
                            options=src_lang_options,
                            index=default_src_index,
                            key="translate_source_lang_select"
                        )
                    with col_lang2:
                        selected_tgt_lang_name = st.selectbox(
                            "Target Language:",
                            options=lang_names,
                            index=default_tgt_index,
                            key="translate_target_lang_select"
                        )

                    # 4. Translate Button & Logic
                    if st.button("Translate Now", key="translate_button_go", disabled=not text_to_translate.strip()):
                        # Persist selections
                        st.session_state.translation_src_lang = selected_src_lang_name
                        st.session_state.translation_tgt_lang = selected_tgt_lang_name
                        st.session_state.translation_output = "" # Clear previous output

                        if selected_src_lang_name == selected_tgt_lang_name and selected_src_lang_name != "Auto-Detect":
                            st.info("Source and Target languages are the same. No translation needed.")
                            st.session_state.translation_output = text_to_translate
                        else:
                            with st.spinner(f"Translating to {selected_tgt_lang_name}..."):
                                try:
                                    # Use the imported translate function which handles code lookup etc.
                                    raw_translation = translate(
                                        text=text_to_translate,
                                        target_language=selected_tgt_lang_name,
                                        source_language=selected_src_lang_name # Pass "Auto-Detect" or the name
                                    )

                                    # --- Process Result ---
                                    if isinstance(raw_translation, str):
                                        # Check if the function returned an error message
                                        if raw_translation.startswith("[Translation Error:"):
                                             st.error(f"Translation failed. {raw_translation}")
                                             logger.error(f"Translation failed for {selected_src_lang_name} -> {selected_tgt_lang_name}. Reason: {raw_translation}")
                                             st.session_state.translation_output = raw_translation # Show error in output
                                        else:
                                            final_translation = format_translation(raw_translation)
                                            st.session_state.translation_output = final_translation
                                            st.success("Translation complete!")
                                            logger.info(f"Translation successful: {selected_src_lang_name} -> {selected_tgt_lang_name}")
                                    else:
                                         # Should not happen if translate function is robust, but handle anyway
                                         st.error("Translation failed: Received invalid result type from translation function.")
                                         logger.error(f"Translation function returned non-string type: {type(raw_translation)}")
                                         st.session_state.translation_output = "[Translation Error: Invalid Result Type]"

                                except Exception as e:
                                    st.error(f"An unexpected error occurred during translation: {e}")
                                    logger.error(f"Translation process exception: {e}", exc_info=True)
                                    st.session_state.translation_output = f"[Translation Error: Unexpected Exception - {e}]"

                    # 5. Display Translation Output (Always show the area)
                    st.text_area(
                        "Translated Text:",
                        value=st.session_state.translation_output or "", # Display stored output
                        height=250,
                        key="translate_output_area",
                        disabled=True, # Display only, user can copy
                        help="The translation result will appear here."
                    )
        elif not TRANSLATION_AVAILABLE and len(tabs) > 4:
             with tabs[4]:
                 st.info("Translation features are currently unavailable. Please check the application configuration and logs.")


# --- Action Handling Logic (Runs AFTER UI render, based on flags set by buttons) ---
if st.session_state.get("last_action"):
    current_action = st.session_state.last_action
    st.session_state.last_action = None # Reset action flag immediately
    logger.info(f"Handling action: {current_action}")

    # --- Action Pre-checks ---
    img_for_action = st.session_state.get("processed_image")
    if not isinstance(img_for_action, Image.Image):
        # Allow report generation even without image if desired? Check dependencies.
        # report_utils likely requires an image, so fail here for all actions.
        st.error(f"Cannot perform '{current_action}': No valid image loaded or processed.")
        logger.error(f"Action '{current_action}' skipped: processed_image is invalid or None.")
        st.stop() # Stop if image is required but missing

    if not st.session_state.get("session_id"):
        st.error(f"Cannot perform '{current_action}': Session ID is missing. Please reload.")
        logger.error(f"Action '{current_action}' aborted: missing session ID.")
        st.stop()

    # Ensure core modules are available for actions needing them
    if not APP_MODULES_AVAILABLE:
         st.error(f"Cannot perform '{current_action}': Core application modules are unavailable.")
         logger.error(f"Action '{current_action}' skipped: APP_MODULES_AVAILABLE is False.")
         st.stop()

    # Prepare common variables for AI calls
    roi_for_action = st.session_state.get("roi_coords")
    roi_str_log = " (with ROI)" if roi_for_action else ""
    history_for_action = st.session_state.history if isinstance(st.session_state.history, list) else []
    # Ensure history type safety (should be handled at init, but belt-and-suspenders)
    st.session_state.history = history_for_action

    action_status_placeholder = st.empty() # Placeholder for status messages/spinners

    try:
        # --- Perform Action ---
        if current_action == "analyze":
            action_status_placeholder.info(f"🔬 Performing initial analysis{roi_str_log}...")
            with st.spinner("AI is analyzing the image... This may take a moment."):
                # Pass ROI to the analysis function
                result = run_initial_analysis(img_for_action, roi=roi_for_action)
            st.session_state.initial_analysis = result
            # Clear potentially outdated related results
            st.session_state.qa_answer = ""
            st.session_state.disease_analysis = ""
            st.session_state.confidence_score = ""
            logger.info(f"Initial analysis action completed{roi_str_log}.")
            # Check if the result indicates failure (based on llm_interactions convention)
            if result and any(err in result for err in ["Error:", "Failed:", "Unavailable", "Blocked"]):
                 action_status_placeholder.error(f"Initial Analysis Failed: {result.split(':', 1)[-1].strip()}")
            else:
                 action_status_placeholder.success("Initial Analysis Complete.")
            st.rerun()

        elif current_action == "ask":
            question = st.session_state.question_input_widget.strip()
            # Question validity was checked before setting flag, so proceed
            action_status_placeholder.info(f"❓ Asking AI: \"{question[:50]}...\"{roi_str_log}...")
            st.session_state.qa_answer = "" # Clear previous answer display immediately
            with st.spinner("AI is processing your question..."):
                # Pass history and ROI
                answer, ok = run_multimodal_qa(img_for_action, question, history_for_action, roi=roi_for_action)

            st.session_state.qa_answer = answer # Store result/error message
            if ok:
                # Add timestamp to history entry
                from datetime import datetime
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.history.append((question, answer, ts))
                logger.info(f"Q&A successful for: '{question[:50]}...' {roi_str_log}")
                action_status_placeholder.success("AI Answer Received.")
            else:
                # Primary AI failed, 'answer' contains the error message
                action_status_placeholder.error(f"Q&A Failed: {answer}")
                logger.warning(f"Primary AI Q&A failed for '{question[:50]}...'. Reason: {answer}")

                # --- Attempt Fallback (if configured and available) ---
                if HF_FALLBACK_AVAILABLE and os.environ.get("HF_API_TOKEN"):
                    logger.info(f"Attempting HF fallback using {HF_VQA_MODEL_ID}.")
                    action_status_placeholder.info(f"Attempting fallback with {HF_VQA_MODEL_ID}...")
                    with st.spinner(f"Trying fallback: {HF_VQA_MODEL_ID}..."):
                        fb_ans, fb_ok = query_hf_vqa_inference_api(img_for_action, question, roi=roi_for_action)
                    if fb_ok:
                        fb_disp = f"**[Fallback Result: {HF_VQA_MODEL_ID}]**\n\n{fb_ans}"
                        st.session_state.qa_answer += "\n\n" + fb_disp # Append to primary failure message
                        # Add fallback result distinctly to history
                        from datetime import datetime
                        ts_fb = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.history.append((f"[Fallback Query] {question}", fb_disp, ts_fb))
                        action_status_placeholder.info("Fallback AI provided an answer.")
                        logger.info(f"HF fallback successful for '{question[:50]}...'.")
                    else:
                        fb_fail_msg = f"**[Fallback Failed - {HF_VQA_MODEL_ID}]:** {fb_ans}"
                        st.session_state.qa_answer += "\n\n" + fb_fail_msg # Append failure message
                        action_status_placeholder.error(f"Fallback AI also failed: {fb_ans}")
                        logger.error(f"HF fallback failed for '{question[:50]}...'. Reason: {fb_ans}")
                elif HF_FALLBACK_AVAILABLE:
                    logger.warning("HF fallback skipped: HF_API_TOKEN environment variable not set.")
                    st.session_state.qa_answer += "\n\n**[Fallback Skipped: API Token Missing]**"
                    action_status_placeholder.warning("Fallback skipped (API token missing).")
                else:
                     logger.info("HF fallback skipped: No fallback model configured.")
                     st.session_state.qa_answer += "\n\n**[Fallback Unavailable]**"
            st.rerun() # Rerun to display the answer/updated history

        elif current_action == "disease":
            disease = st.session_state.disease_select_widget
            # Condition validity checked before setting flag
            action_status_placeholder.info(f"🩺 Running focused analysis for '{disease}'{roi_str_log}...")
            with st.spinner(f"AI is analyzing for signs of {disease}..."):
                 # Pass ROI to the analysis function
                result = run_disease_analysis(img_for_action, disease, roi=roi_for_action)
            st.session_state.disease_analysis = result
            # Clear potentially outdated related results
            st.session_state.qa_answer = ""
            st.session_state.confidence_score = ""
            logger.info(f"Disease analysis action completed for '{disease}'{roi_str_log}.")
            if result and any(err in result for err in ["Error:", "Failed:", "Unavailable", "Blocked"]):
                action_status_placeholder.error(f"Disease Analysis Failed: {result.split(':', 1)[-1].strip()}")
            else:
                action_status_placeholder.success(f"Analysis for '{disease}' Complete.")
            st.rerun()

        elif current_action == "confidence":
            # Check if history exists (required by current implementation)
            if not history_for_action:
                action_status_placeholder.warning("Cannot estimate confidence without prior Q&A history.")
                logger.warning("Confidence estimation skipped: no history available.")
                # No rerun needed as button should be disabled
            else:
                action_status_placeholder.info(f"📊 Estimating AI confidence based on context{roi_str_log}...")
                context_summary = f"History entries: {len(history_for_action)}. ROI used: {bool(roi_for_action)}"
                logger.info(f"Running confidence estimation. Context: {context_summary}")
                with st.spinner("AI is assessing its confidence..."):
                    # Pass relevant context to the confidence function
                    result = estimate_ai_confidence(
                        image=img_for_action,
                        history=history_for_action,
                        initial_analysis=st.session_state.initial_analysis,
                        disease_analysis=st.session_state.disease_analysis,
                        roi=roi_for_action # Pass ROI used during the interaction being evaluated
                    )
                st.session_state.confidence_score = result
                logger.info("Confidence estimation action completed.")
                if result and any(err in result for err in ["Error:", "Failed:", "Unavailable", "Blocked"]):
                     action_status_placeholder.error(f"Confidence Estimation Failed: {result.split(':', 1)[-1].strip()}")
                else:
                    action_status_placeholder.success("Confidence Estimation Complete.")
            st.rerun()

        elif current_action == "generate_report_data":
            action_status_placeholder.info("📄 Generating PDF report data...")
            st.session_state.pdf_report_bytes = None # Clear previous bytes

            img_for_report = st.session_state.get("display_image") # Use the *display* image for the report
            if not isinstance(img_for_report, Image.Image):
                action_status_placeholder.error("Cannot generate report: Invalid or missing display image.")
                logger.error("PDF generation aborted: invalid display_image in session state.")
            else:
                img_final_report = img_for_report.copy() # Work on a copy

                # Draw ROI on the report image if ROI is valid
                if roi_for_action and isinstance(roi_for_action, dict) and all(k in roi_for_action for k in ['left', 'top', 'width', 'height']):
                    try:
                        draw = ImageDraw.Draw(img_final_report)
                        x0 = int(roi_for_action['left'])
                        y0 = int(roi_for_action['top'])
                        w = int(roi_for_action['width'])
                        h = int(roi_for_action['height'])
                        x1 = x0 + w
                        y1 = y0 + h
                        # Ensure coordinates are within image bounds before drawing
                        x0, y0 = max(0, x0), max(0, y0)
                        x1, y1 = min(img_final_report.width, x1), min(img_final_report.height, y1)
                        # Only draw if valid rectangle remains after clamping
                        if x1 > x0 and y1 > y0:
                             draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
                             logger.info("ROI drawn on image for PDF report.")
                        else:
                             logger.warning("ROI coordinates invalid after clamping, not drawn on report.")
                    except Exception as e:
                        logger.error(f"Error drawing ROI on image for PDF: {e}", exc_info=True)
                        action_status_placeholder.warning("Could not draw ROI on report image due to an error.")
                        # Continue without the drawn ROI

                # Compile text outputs for the report
                # Use the timestamped history format
                history_text = "\n\n".join([f"Q ({i+1}): {q}\nA ({i+1}): {a if a else '[No Answer]'}" for i, (q, a, *_) in enumerate(history_for_action)]) \
                               if history_for_action else "No conversation history recorded."
                report_outputs = {
                    "Session ID": st.session_state.session_id or "N/A",
                    "Image Type": "DICOM" if st.session_state.is_dicom else "Standard Image",
                    "Initial Analysis": st.session_state.initial_analysis or "Not Performed",
                    "Conversation History": history_text,
                    "Condition Analysis": st.session_state.disease_analysis or "Not Performed",
                    "Confidence Estimation": st.session_state.confidence_score or "Not Estimated",
                }

                # Add DICOM metadata summary if available
                dicom_meta_report = None
                if st.session_state.is_dicom and st.session_state.dicom_metadata:
                    dicom_meta_report = st.session_state.dicom_metadata
                    # Create a simple summary string or pass the dict, depending on report_utils
                    summary_keys = ['PatientName', 'PatientID', 'StudyDescription', 'Modality', 'StudyDate', 'WindowCenter', 'WindowWidth']
                    meta_summary = {k: v for k, v in dicom_meta_report.items() if k in summary_keys and v}
                    report_outputs["DICOM Summary"] = "\n".join([f"{k}: {v}" for k,v in meta_summary.items()]) if meta_summary else "Basic DICOM metadata available."

                # Generate the PDF bytes using the utility function
                with st.spinner("Generating PDF document..."):
                    # Ensure report_utils is available
                    if not APP_MODULES_AVAILABLE:
                        raise ImportError("Cannot generate report, report_utils unavailable.")

                    pdf_bytes = generate_pdf_report_bytes(
                        session_id=st.session_state.session_id,
                        image=img_final_report, # Image with ROI drawn (if applicable)
                        analysis_outputs=report_outputs, # Dict of text results
                        dicom_metadata=dicom_meta_report # Pass full metadata if needed
                    )

                if pdf_bytes:
                    st.session_state.pdf_report_bytes = pdf_bytes
                    action_status_placeholder.success("PDF report generated! Download button available in sidebar.")
                    logger.info("PDF generation successful.")
                    st.balloons() # Visual feedback for success
                else:
                    action_status_placeholder.error("Failed to generate PDF report. Check logs for details from report_utils.")
                    logger.error("PDF generation failed: generate_pdf_report_bytes returned None or empty data.")
            st.rerun() # Rerun to show download button or clear status messages

        else:
            st.warning(f"Unknown action '{current_action}' encountered.")
            logger.warning(f"Attempted to handle unknown action: '{current_action}'")
            # No rerun usually needed for unknown actions

    except ImportError as e:
         # Catch potential errors if optional modules needed for the action weren't loaded
         st.error(f"Action '{current_action}' failed due to missing module: {e}. Please check installation.")
         logger.error(f"ImportError during action '{current_action}': {e}", exc_info=True)
         st.rerun() # Rerun to clear state maybe
    except Exception as e:
        # Catch-all for unexpected errors during action execution
        st.error(f"A critical error occurred while performing '{current_action}': {e}")
        logger.critical(f"Unhandled exception during action '{current_action}': {e}", exc_info=True)
        # Optionally clear the status placeholder here
        action_status_placeholder.empty()
        st.rerun() # Rerun to try and recover or clear state

# --- Footer ---
st.markdown("---")
# Dynamic Footer Content
footer_session_id = st.session_state.get('session_id', 'N/A')
footer_text = f"""
<div style="text-align: center; font-size: 0.85em; color: #777; margin-top: 2em; padding-bottom: 1em;">
    <p>Session ID: <span style="font-family: monospace; background-color: #eee; padding: 2px 4px; border-radius: 3px;">{footer_session_id}</span></p>
    <p>
        <a href="#" target="_blank" rel="noopener noreferrer" style="color: #555;">Privacy Policy</a> |
        <a href="#" target="_blank" rel="noopener noreferrer" style="color: #555;">Terms of Service</a> |
        <a href="#" target="_blank" rel="noopener noreferrer" style="color: #555;">Documentation</a>
    </p>
    <p style="font-size: 0.9em; margin-top: 0.5em;">
      ⚠️ {APP_NAME} is for informational purposes only. Always consult qualified healthcare professionals for medical diagnosis and decisions.
    </p>
    <p style="font-size: 0.8em; color: #aaa;">
        Powered by Streamlit | v{st.__version__}
    </p>
</div>
"""
st.markdown(footer_text, unsafe_allow_html=True)

logger.info("--- App Render Complete ---")