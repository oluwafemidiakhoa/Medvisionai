# -*- coding: utf-8 -*-
"""
app.py - Main Streamlit application for RadVision AI Advanced.
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
import logging # Import logging first
import base64
import hashlib
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple, List, Union
import copy
import random
import re

# --- v v v --- LOGGING SETUP (MOVED EARLIER) --- v v v ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
# Ensure logging is configured before any logger calls are made
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__) # Define the main logger
logger.info("--- RadVision AI Application Start (Logger Initialized) ---")
logger.info(f"Streamlit Version: {st.__version__}")
logger.info(f"Logging Level: {LOG_LEVEL}")
# --- ^ ^ ^ --- END OF LOGGING SETUP --- ^ ^ ^ ---


# --- Pillow (PIL) - Essential (Import Early for Patch) ---
try:
    from PIL import Image, ImageDraw, UnidentifiedImageError
    import PIL
    PIL_VERSION = getattr(PIL, '__version__', 'Unknown')
    PIL_AVAILABLE = True
    logger.info(f"Pillow (PIL) Version: {PIL_VERSION}") # Log PIL version now
except ImportError:
    logger.critical("Pillow (PIL) is not installed (`pip install Pillow`). Cannot continue.") # Use logger
    st.error("CRITICAL ERROR: Pillow (PIL) is not installed (`pip install Pillow`). Image processing disabled.")
    PIL_AVAILABLE = False
    st.stop()

# --- v v v --- MONKEY PATCH FOR st_image.image_to_url --- v v v ---
# Apply this after core imports, PIL, and logging setup
try:
    import streamlit.elements.image as st_image

    if not hasattr(st_image, "image_to_url"):
        logger.info("Applying monkey-patch for missing 'streamlit.elements.image.image_to_url'...") # Use logger

        def image_to_url_monkey_patch(
            image: Any,
            width: int = -1,
            clamp: bool = False,
            channels: str = "RGB",
            output_format: str = "auto",
            image_id: str = "",
        ) -> str:
            """Simplified image_to_url implementation for compatibility."""
            patch_logger = logging.getLogger(__name__ + ".monkey_patch") # Can use sub-logger if desired
            patch_logger.debug(f"Monkey patch image_to_url called with type: {type(image)}")

            if isinstance(image, Image.Image):
                try:
                    fmt = output_format.upper(); fmt = "PNG" if fmt == "AUTO" else fmt
                    if fmt not in ["PNG", "JPEG", "GIF", "WEBP"]:
                         patch_logger.warning(f"Image format {fmt} converting to PNG for data URL.")
                         fmt = "PNG"
                    img_to_save = image
                    if channels == "RGB" and image.mode not in ['RGB', 'L']:
                        patch_logger.debug(f"Converting image mode {image.mode} to RGB.")
                        img_to_save = image.convert("RGB")
                    elif image.mode == 'P':
                        patch_logger.debug(f"Converting image mode P to RGBA.")
                        img_to_save = image.convert("RGBA"); fmt = "PNG"
                    buffered = io.BytesIO()
                    img_to_save.save(buffered, format=fmt)
                    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    result_url = f"data:image/{fmt.lower()};base64,{img_b64}"
                    patch_logger.debug(f"Monkey patch generated data URL (len: {len(result_url)}).")
                    return result_url
                except Exception as e:
                    patch_logger.error(f"ERROR in monkey-patch image_to_url: {e}", exc_info=True)
                    return ""
            else:
                 patch_logger.warning(f"Monkey-patch image_to_url received unsupported type: {type(image)}")
                 return ""

        st_image.image_to_url = image_to_url_monkey_patch
        logger.info("Monkey-patch applied successfully.")
    else:
        logger.info("'streamlit.elements.image.image_to_url' already exists. No patch needed.")

except ImportError:
    logger.warning("Could not import streamlit.elements.image. Skipping monkey-patch.")
except Exception as e:
    logger.error(f"An error occurred during monkey-patch setup: {e}", exc_info=True) # Use logger
# --- ^ ^ ^ --- END OF MONKEY PATCH --- ^ ^ ^ ---


# --- Ensure deep-translator is installed ---
try:
    from deep_translator import GoogleTranslator
    DEEP_TRANSLATOR_INSTALLED = True
except ImportError:
    DEEP_TRANSLATOR_INSTALLED = False
    try:
        logger.info("Attempting to install deep-translator...") # Use logger
        subprocess.check_call([sys.executable, "-m", "pip", "install", "deep-translator"])
        from deep_translator import GoogleTranslator
        DEEP_TRANSLATOR_INSTALLED = True
        logger.info("deep-translator installed successfully.") # Use logger
    except Exception as e:
        logger.critical(f"Could not install deep-translator: {e}", exc_info=True) # Use logger

# --- Dependency Checks & Imports (Continued) ---

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
    st_canvas = None

# Pydicom & related libraries
try:
    import pydicom
    import pydicom.errors
    PYDICOM_VERSION = getattr(pydicom, '__version__', 'Unknown')
    logger.info(f"Pydicom Version: {PYDICOM_VERSION}")
    PYDICOM_AVAILABLE = True
    try: import pylibjpeg; logger.info("pylibjpeg found.")
    except ImportError: logger.info("pylibjpeg not found (optional).")
    try: import gdcm; logger.info("python-gdcm found.")
    except ImportError: logger.info("python-gdcm not found (optional).")
except ImportError:
    PYDICOM_VERSION = 'Not Installed'
    logger.warning("pydicom not found. DICOM functionality will be disabled.")
    PYDICOM_AVAILABLE = False

# --- Custom Backend Modules ---
try:
    from dicom_utils import (
        parse_dicom, extract_dicom_metadata, dicom_to_image, get_default_wl
    )
    DICOM_UTILS_AVAILABLE = True
    logger.info("dicom_utils imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import dicom_utils: {e}. DICOM features disabled.")
    if PYDICOM_AVAILABLE:
        st.warning("DICOM utilities module missing. DICOM processing limited.")
    DICOM_UTILS_AVAILABLE = False

try:
    from llm_interactions import (
        run_initial_analysis,
        run_multimodal_qa,
        run_disease_analysis,
        run_llm_self_assessment # Using the corrected name
    )
    LLM_INTERACTIONS_AVAILABLE = True
    logger.info("llm_interactions imported successfully.")
except ImportError as e:
    st.error(f"Core AI module (llm_interactions) failed to import: {e}. Analysis functions disabled.")
    logger.critical(f"Failed to import llm_interactions: {e}", exc_info=True)
    LLM_INTERACTIONS_AVAILABLE = False
    st.stop()

try:
    from report_utils import generate_pdf_report_bytes
    REPORT_UTILS_AVAILABLE = True
    logger.info("report_utils imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import report_utils: {e}. PDF reporting disabled.")
    REPORT_UTILS_AVAILABLE = False

try:
    from ui_components import display_dicom_metadata, dicom_wl_sliders
    UI_COMPONENTS_AVAILABLE = True
    logger.info("ui_components imported successfully.")
except ImportError as e:
    logger.warning(f"Failed to import ui_components: {e}. Using basic UI fallbacks.")
    UI_COMPONENTS_AVAILABLE = False
    def display_dicom_metadata(metadata): st.caption("Metadata Preview:"); st.json(dict(list(metadata.items())[:5]))
    def dicom_wl_sliders(wc, ww): st.caption("W/L sliders unavailable."); return wc, ww

# --- HF fallback for Q&A ---
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
    from translation_models import (
        translate, detect_language, LANGUAGE_CODES, AUTO_DETECT_INDICATOR
    )
    TRANSLATION_AVAILABLE = DEEP_TRANSLATOR_INSTALLED
    if TRANSLATION_AVAILABLE:
        logger.info("translation_models imported successfully. Translation is available.")
    else:
        logger.error("translation_models imported, but deep-translator is missing. Translation disabled.")
        st.warning("Translation library (deep-translator) is missing or failed to install. Translation features disabled.")
except ImportError as e:
    logger.error(f"Could not import translation_models: {e}. Translation disabled.", exc_info=True)
    TRANSLATION_AVAILABLE = False
    if DEEP_TRANSLATOR_INSTALLED:
         st.warning(f"Translation module failed to load ({e}). Translation features disabled.")

if not TRANSLATION_AVAILABLE:
    translate = None
    detect_language = None
    LANGUAGE_CODES = {"English": "en"}
    AUTO_DETECT_INDICATOR = "Auto-Detect"

# --- Custom CSS ---
st.markdown(
    """
    <style>
        /* Improve spacing for generated report button */
        div[data-testid="stDownloadButton"] button {
            margin-top: 10px; /* Add some space above the download button */
            width: 100%; /* Make button full width */
        }
        /* Ensure tabs are spaced reasonably */
        .stTabs [data-baseweb="tab-list"] {
          gap: 2px;
          padding-bottom: 10px;
        }
        .stTabs [data-baseweb="tab"] {
          height: 40px;
          white-space: pre-wrap;
          background-color: #f0f2f6; /* Light grey background for tabs */
          border-radius: 4px 4px 0 0;
          margin: 0 !important;
          padding: 10px 15px;
          font-size: 0.9rem;
        }
        .stTabs [aria-selected="true"] {
          background-color: #FFFFFF; /* White background for selected tab */
          font-weight: bold;
        }
        /* Style the main content area */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }
        /* Sidebar specific styles */
        [data-testid="stSidebar"] {
            padding: 1rem;
            background-color: #f8f9fa; /* Slightly different sidebar color */
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #004085; /* Dark blue headers in sidebar */
        }
        [data-testid="stSidebar"] .stButton button {
            border-color: #007bff; /* Blue border for sidebar buttons */
            color: #007bff; /* Blue text for sidebar buttons */
        }
        [data-testid="stSidebar"] .stButton button:hover {
            border-color: #0056b3;
            color: #0056b3;
            background-color: #e7f3ff; /* Light blue background on hover */
        }
        /* General button styling */
        .stButton button {
            border-radius: 5px;
        }
        /* Improve text area contrast */
        .stTextArea textarea {
            border: 1px solid #ced4da;
        }
        /* Footer styling */
        footer {
            visibility: hidden; /* Hide default Streamlit footer */
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            color: #6c757d; /* Gray footer text */
            text-align: center;
            padding: 5px 0;
            font-size: 0.8rem;
            z-index: 1000; /* Ensure footer stays on top */
            border-top: 1px solid #dee2e6; /* Light border top */
        }
    </style>
    """,
    unsafe_allow_html=True
)


# --- Display Hero Logo ---
logo_path = os.path.join("assets", "radvisionai-hero.jpeg") # Assuming logo is in assets/
if os.path.exists(logo_path):
    st.image(logo_path, width=350)
else:
    logger.warning(f"Hero logo not found at: {logo_path}. Displaying title instead.")
    st.title("RadVision AI Advanced") # Fallback if logo missing

# --- Initialize Session State Defaults ---
DEFAULT_STATE = {
    "uploaded_file_info": None,      # Stores info about the current file (name, size, hash)
    "raw_image_bytes": None,         # Raw bytes of the uploaded file
    "is_dicom": False,               # Flag indicating if the file is DICOM
    "dicom_dataset": None,           # Parsed pydicom dataset object
    "dicom_metadata": {},            # Extracted relevant DICOM metadata dict
    "processed_image": None,         # PIL Image object prepared for LLM (e.g., normalized)
    "display_image": None,           # PIL Image object for display (e.g., with W/L applied)
    "session_id": None,              # Unique ID for the session
    "history": [],                   # List to store Q&A history (tuples: (type, message))
    "initial_analysis": "",          # Stores the result of the initial structured analysis
    "qa_answer": "",                 # Stores the latest Q&A answer from the AI
    "disease_analysis": "",          # Stores the result of disease-specific analysis
    "confidence_score": "",          # Stores the LLM self-assessment score/text
    "last_action": None,             # Tracks the last button clicked to trigger processing
    "pdf_report_bytes": None,        # Bytes of the generated PDF report
    "canvas_drawing": None,          # Raw result from streamlit-drawable-canvas
    "roi_coords": None,              # Processed ROI coordinates (dict: {left, top, width, height})
    "current_display_wc": None,      # Current DICOM Window Center for display
    "current_display_ww": None,      # Current DICOM Window Width for display
    "clear_roi_feedback": False,     # Flag to show a temporary "ROI cleared" message
    "demo_loaded": False,            # Flag indicating if demo mode content is loaded
    "translation_result": None,      # Stores the last successful translation
    "translation_error": None,       # Stores the last translation error message
}
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.session_id = str(uuid.uuid4())[:8] # Generate unique session ID
    logger.info(f"New session initialized: {st.session_state.session_id}")
# Ensure all keys exist in session state, initializing if needed
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value
# Ensure history is always a list
if not isinstance(st.session_state.get("history"), list):
    st.session_state.history = []
logger.debug(f"Session state verified for session ID: {st.session_state.session_id}")

# --- Utility Functions ---
def format_translation(translated_text: Optional[str]) -> str:
    """Formats translated text, trying to preserve simple list structures."""
    if translated_text is None:
        return "Translation not available or failed."
    try:
        text_str = str(translated_text)
        # Attempt to add line breaks before numbered list items
        formatted_text = re.sub(r'\s+(\d+\.)', r'\n\n\1', text_str)
        return formatted_text.strip()
    except Exception as e:
        logger.error(f"Error formatting translation: {e}", exc_info=True)
        return str(translated_text) # Fallback to original string

# --- Sidebar ---
with st.sidebar:
    st.header("⚕️ RadVision Controls")
    st.markdown("---")
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

    st.header("Image Upload & Settings")
    st.caption("🔒 Ensure all images are de-identified before uploading.")

    uploaded_file = st.file_uploader(
        "Upload De-Identified Image (JPG, PNG, DCM)",
        type=["jpg", "jpeg", "png", "dcm", "dicom"],
        key="file_uploader_widget",
        help="Upload a de-identified medical image. DICOM (.dcm) preferred. DO NOT upload identifiable patient data unless permitted by privacy regulations."
    )

    demo_mode = st.checkbox(
        "🚀 Demo Mode",
        value=st.session_state.get("demo_loaded", False),
        help="Load a sample chest X-ray image and analysis."
    )
    if demo_mode and not st.session_state.demo_loaded:
        logger.info("Demo Mode activated.")
        # NOTE: Actual loading logic for demo mode is missing here.
        # This needs implementation (e.g., load a default image and pre-fill analysis).
        st.warning("Demo mode selected, but loading logic needs implementation.")
        # Set demo_loaded to True once loading is done
        # st.session_state.demo_loaded = True
        # Reset relevant state if needed
    elif not demo_mode and st.session_state.demo_loaded:
        logger.info("Demo Mode deactivated.")
        # Clear demo state? Or just allow user to upload over it?
        st.session_state.demo_loaded = False

    # ROI Clearing Button (Only if canvas is available)
    if DRAWABLE_CANVAS_AVAILABLE:
        if st.button("🗑️ Clear ROI", help="Remove the selected ROI rectangle"):
            st.session_state.roi_coords = None
            st.session_state.canvas_drawing = None # Clear canvas state too
            st.session_state.clear_roi_feedback = True # Trigger feedback message
            logger.info("ROI cleared by user.")
            st.rerun() # Update the UI immediately

        # Display temporary feedback after clearing ROI
        if st.session_state.get("clear_roi_feedback"):
            st.success("✅ ROI cleared!")
            st.session_state.clear_roi_feedback = False # Reset the flag

    # DICOM Window/Level Controls (Only if DICOM is loaded and utils available)
    if st.session_state.is_dicom and DICOM_UTILS_AVAILABLE and UI_COMPONENTS_AVAILABLE and st.session_state.display_image:
        st.markdown("---")
        st.subheader("DICOM Display (W/L)")
        # Use the UI component for sliders
        new_wc, new_ww = dicom_wl_sliders(
            st.session_state.current_display_wc,
            st.session_state.current_display_ww
        )
        # Check if values changed
        if new_wc != st.session_state.current_display_wc or new_ww != st.session_state.current_display_ww:
            logger.info(f"DICOM W/L changed via UI: WC={new_wc}, WW={new_ww}")
            st.session_state.current_display_wc = new_wc
            st.session_state.current_display_ww = new_ww
            # Re-apply W/L if dataset is available
            if st.session_state.dicom_dataset:
                with st.spinner("Applying new Window/Level..."):
                    try:
                        # --- v v v CORRECTED BLOCK START v v v ---
                        new_display_img = dicom_to_image(
                            st.session_state.dicom_dataset,
                            wc=new_wc,
                            ww=new_ww
                        )
                        # Check if image conversion was successful INSIDE the try block
                        if isinstance(new_display_img, Image.Image):
                            # Update the display image state
                            st.session_state.display_image = new_display_img.convert('RGB') if new_display_img.mode != 'RGB' else new_display_img
                            st.rerun() # Rerun to update the image display
                        else:
                            # Handle case where dicom_to_image succeeded but returned wrong type
                            st.error("Failed to update DICOM image display (Invalid conversion result).")
                            logger.error("dicom_to_image returned non-image for W/L update.")
                        # --- ^ ^ ^ CORRECTED BLOCK END ^ ^ ^ ---
                    except Exception as e:
                        # Catch errors during dicom_to_image call
                        st.error(f"Error applying W/L: {e}")
                        logger.error(f"W/L application error: {e}", exc_info=True)
            else:
                st.warning("DICOM dataset unavailable to update W/L.")

    # --- AI Analysis Actions ---
    st.markdown("---")
    st.header("🤖 AI Analysis Actions")
    action_disabled = not LLM_INTERACTIONS_AVAILABLE or not isinstance(st.session_state.get("processed_image"), Image.Image)

    if st.button("🔬 Run Structured Initial Analysis", key="analyze_btn", disabled=action_disabled, help="Perform a general, structured analysis of the image using the AI."):
        st.session_state.last_action = "analyze"
        st.rerun()

    st.subheader("❓ Ask AI a Question")
    question_input = st.text_area(
        "Enter your question about the image:",
        height=100,
        key="question_input_widget",
        placeholder="E.g., 'Describe the findings in the upper right lung field.' or 'Is there evidence of cardiomegaly?'",
        disabled=action_disabled
    )
    if st.button("💬 Ask Question", key="ask_btn", disabled=action_disabled):
        if question_input.strip():
            st.session_state.last_action = "ask"
            st.rerun()
        else:
            st.warning("Please enter a question first.")

    st.subheader("🎯 Condition-Specific Analysis")
    DISEASE_OPTIONS = [
        "Pneumonia", "Lung Cancer", "Nodule/Mass", "Effusion", "Fracture",
        "Stroke", "Appendicitis", "Bowel Obstruction", "Cardiomegaly",
        "Aortic Aneurysm", "Pulmonary Embolism", "Tuberculosis",
        "COVID-19 Findings", "Brain Tumor", "Arthritis",
        # Add more relevant conditions
    ]
    disease_select = st.selectbox(
        "Select condition for focused analysis:",
        options=[""] + sorted(DISEASE_OPTIONS), # Add empty option
        key="disease_select_widget",
        disabled=action_disabled,
        help="The AI will analyze the image specifically looking for signs related to the selected condition."
    )
    if st.button("🩺 Analyze for Condition", key="disease_btn", disabled=action_disabled):
        if disease_select:
            st.session_state.last_action = "disease"
            st.rerun()
        else:
            st.warning("Please select a condition first.")

    # --- Reporting & Assessment ---
    st.markdown("---")
    st.header("📊 Reporting & Assessment")

    # Only allow confidence estimate if there's history (something to assess)
    can_estimate = bool(st.session_state.history)
    if st.button("🧪 Estimate LLM Self-Assessment (Experimental)", key="confidence_btn", disabled=not can_estimate or action_disabled, help="EXPERIMENTAL: Ask the LLM to assess the confidence or quality of its most recent Q&A response based on the image."):
        st.session_state.last_action = "confidence"
        st.rerun()

    # Disable report generation if core AI or report utils are missing
    report_generation_disabled = action_disabled or not REPORT_UTILS_AVAILABLE
    if st.button("📄 Generate PDF Report Data", key="generate_report_data_btn", disabled=report_generation_disabled, help="Compile the current analysis, Q&A, and image into PDF data ready for download."):
        st.session_state.last_action = "generate_report_data"
        st.rerun()

    # Display download button only if PDF bytes are available
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

# --- File Upload Logic ---
if uploaded_file is not None and PIL_AVAILABLE:
    # Generate a unique identifier for the uploaded file content
    try:
        uploaded_file.seek(0) # Go to start
        file_content_hash = hashlib.sha256(uploaded_file.read()).hexdigest()[:16]
        uploaded_file.seek(0) # Reset pointer for further use
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{file_content_hash}"
    except Exception as e:
        logger.warning(f"Could not generate hash for uploaded file: {e}")
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{uuid.uuid4().hex[:8]}" # Fallback ID

    # Check if this is a new file upload (or the first one)
    if new_file_info != st.session_state.get("uploaded_file_info"):
        logger.info(f"New file upload detected: {uploaded_file.name} ({uploaded_file.size} bytes)")
        st.toast(f"Processing '{uploaded_file.name}'...", icon="⏳")

        # --- Reset Session State for New Image ---
        keys_to_preserve = {"session_id"} # Keep the session ID
        st.session_state.session_id = st.session_state.get("session_id") or str(uuid.uuid4())[:8] # Ensure ID exists

        # Reset all other relevant state variables to defaults
        for key, value in DEFAULT_STATE.items():
            if key not in keys_to_preserve:
                st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value

        # Store info about the new file and its raw bytes
        st.session_state.uploaded_file_info = new_file_info
        st.session_state.demo_loaded = False # Turn off demo mode if a file is uploaded
        st.session_state.raw_image_bytes = uploaded_file.getvalue()

        # Determine if the file is DICOM
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        is_dicom_type = "dicom" in uploaded_file.type.lower() or file_ext in (".dcm", ".dicom")
        st.session_state.is_dicom = (PYDICOM_AVAILABLE and DICOM_UTILS_AVAILABLE and is_dicom_type)

        # --- Process Image (DICOM or Standard) ---
        with st.spinner("🔬 Analyzing and preparing image..."):
            temp_display_img = None    # For UI display (e.g., with W/L)
            temp_processed_img = None  # For AI model (e.g., normalized)
            processing_success = False

            if st.session_state.is_dicom:
                logger.info("Processing uploaded file as DICOM...")
                try:
                    # Parse DICOM file using utility function
                    dicom_dataset = parse_dicom(st.session_state.raw_image_bytes, filename=uploaded_file.name)
                    if dicom_dataset:
                        st.session_state.dicom_dataset = dicom_dataset
                        st.session_state.dicom_metadata = extract_dicom_metadata(dicom_dataset)
                        # Get default W/L values and store them
                        default_wc, default_ww = get_default_wl(dicom_dataset)
                        st.session_state.current_display_wc = default_wc
                        st.session_state.current_display_ww = default_ww
                        # Generate display image with default W/L
                        temp_display_img = dicom_to_image(dicom_dataset, wc=default_wc, ww=default_ww)
                        # Generate processed image (e.g., normalized, no W/L)
                        temp_processed_img = dicom_to_image(dicom_dataset, wc=None, ww=None, normalize=True)

                        if isinstance(temp_display_img, Image.Image) and isinstance(temp_processed_img, Image.Image):
                            processing_success = True
                            logger.info("DICOM file processed successfully into display and processed images.")
                        else:
                            st.error("Failed to convert DICOM pixel data into displayable images.")
                            logger.error("dicom_to_image returned None or non-Image objects during initial processing.")
                    else:
                        st.error("Could not parse the uploaded DICOM file.")
                        logger.error("parse_dicom utility returned None.")
                except pydicom.errors.InvalidDicomError:
                    st.error("Invalid DICOM file format.")
                    logger.error("pydicom raised InvalidDicomError during parsing.")
                    st.session_state.is_dicom = False # Fallback to standard image processing if possible
                except Exception as e:
                    st.error(f"An unexpected error occurred during DICOM processing: {e}")
                    logger.error(f"DICOM processing failed: {e}", exc_info=True)
                    st.session_state.is_dicom = False # Fallback if error occurs

            # If not DICOM or DICOM processing failed, try standard image processing
            if not st.session_state.is_dicom and not processing_success:
                logger.info("Processing uploaded file as a standard image (JPG, PNG)...")
                try:
                    raw_img = Image.open(io.BytesIO(st.session_state.raw_image_bytes))
                    # Ensure image is in RGB format for consistency
                    processed_img = raw_img.convert("RGB")
                    temp_display_img = processed_img.copy()   # Use the same image for display and processing initially
                    temp_processed_img = processed_img.copy()
                    processing_success = True
                    logger.info("Standard image loaded and converted to RGB.")
                except UnidentifiedImageError:
                    st.error("Cannot identify image file format. Please upload a valid JPG, PNG, or DICOM file.")
                    logger.error(f"UnidentifiedImageError for file: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error processing standard image: {e}")
                    logger.error(f"Standard image processing error: {e}", exc_info=True)

            # --- Finalize State Update ---
            logger.info(f"Image processing completion status: {processing_success}")
            if processing_success and isinstance(temp_display_img, Image.Image) and isinstance(temp_processed_img, Image.Image):
                # Convert display image to RGB if needed (PIL requirement for some Streamlit elements)
                st.session_state.display_image = temp_display_img.convert('RGB') if temp_display_img.mode != 'RGB' else temp_display_img
                st.session_state.processed_image = temp_processed_img
                st.success(f"✅ Image '{uploaded_file.name}' loaded successfully!")
                logger.info(f"Session state updated with processed images for file: {uploaded_file.name}")
                st.rerun() # Rerun to display the new image and clear processing spinners
            else:
                # Explicitly clear state if processing failed entirely
                st.error("Image loading failed. Please try a different file or check format.")
                logger.error(f"Image processing failed for file: {uploaded_file.name} (success_flag={processing_success})")
                st.session_state.uploaded_file_info = None # Reset file info so user can retry
                st.session_state.display_image = None
                st.session_state.processed_image = None
                st.session_state.is_dicom = False

# --- Main Page Content ---

st.markdown("---")
# Main Disclaimer added near top after logo/title
st.warning(
    """
    **🔴 IMPORTANT: For Research & Educational Use Only 🔴**

    RadVision AI Advanced is an experimental tool demonstrating AI capabilities in medical image analysis.
    **It is NOT a medical device and has NOT been approved for clinical diagnosis or patient care.**

    *   **Do NOT use for primary diagnosis or treatment decisions.**
    *   AI analysis may contain inaccuracies or omissions.
    *   Always rely on the judgment of qualified healthcare professionals.
    *   Ensure compliance with all privacy regulations (e.g., HIPAA) regarding image de-identification.
    *   By using this tool, you acknowledge these limitations.
    """,
    icon="⚠️"
)

st.title("⚕️ RadVision AI Advanced: AI-Assisted Image Analysis")

with st.expander("View User Guide & Workflow", expanded=False):
    st.markdown(
        """
        **Workflow:**

        1.  **Upload Image:** Use the sidebar to upload a de-identified medical image (JPG, PNG, or preferably DICOM `.dcm`). Or, select "Demo Mode".
        2.  **View Image:** The uploaded image appears in the left panel.
            *   For DICOM, adjust Window/Level sliders (in sidebar) for optimal contrast if needed.
            *   View DICOM metadata (if available) in the expander below the image.
        3.  **(Optional) Draw ROI:** Use the drawing tool on the image to select a Region of Interest (ROI). This focuses the AI's analysis. Click "Clear ROI" in the sidebar to remove it.
        4.  **Perform Analysis (Sidebar Actions):**
            *   **Initial Analysis:** Get a general, structured report.
            *   **Ask Question:** Enter a specific question about the image (with or without ROI).
            *   **Analyze Condition:** Select a condition for targeted analysis.
        5.  **Review Results (Right Panel Tabs):**
            *   **Structured Analysis:** View the initial report.
            *   **Q&A History:** See your questions and the AI's answers.
            *   **Condition Focus:** View results of the condition-specific analysis.
            *   **LLM Self-Assessment:** View the AI's experimental self-rating (use with caution).
            *   **Translation:** Translate analysis text into different languages.
        6.  **(Optional) Generate Report:** Click "Generate PDF Report Data" in the sidebar, then "Download PDF Report".
        7.  **Repeat:** Ask more questions or upload a new image.

        **Tips:**
        *   De-identify all images *before* upload.
        *   Specific questions often yield better results than vague ones.
        *   ROI helps pinpoint areas of interest for the AI.
        """
    )
st.markdown("---")

col1, col2 = st.columns([2, 3], gap="large") # Adjust column ratio if needed (e.g., [1, 1] or [3, 2])

# --- Column 1: Image Viewer & Metadata ---
with col1:
    st.subheader("🖼️ Image Viewer")
    display_img = st.session_state.get("display_image")

    if isinstance(display_img, Image.Image):
        # Attempt to display using st.image first (for simplicity or fallback)
        # Comment out the simple st.image if canvas is primary goal
        # st.image(display_img, caption="Image Preview", use_container_width=True)

        if DRAWABLE_CANVAS_AVAILABLE and st_canvas:
            st.caption("Draw a rectangle (ROI) below to focus analysis, or use sidebar actions directly.")

            # --- Canvas Setup ---
            # Adjust canvas size dynamically based on image aspect ratio
            # Max dimensions to prevent excessive size
            MAX_CANVAS_WIDTH = 600
            MAX_CANVAS_HEIGHT = 550
            img_width, img_height = display_img.size
            aspect_ratio = img_width / img_height
            canvas_width = MAX_CANVAS_WIDTH
            canvas_height = int(canvas_width / aspect_ratio)
            if canvas_height > MAX_CANVAS_HEIGHT:
                canvas_height = MAX_CANVAS_HEIGHT
                canvas_width = int(canvas_height * aspect_ratio)

            # Ensure dimensions are positive
            canvas_width = max(1, canvas_width)
            canvas_height = max(1, canvas_height)

            # --- Define Canvas Tool Properties ---
            drawing_mode = "rect" # Mode for drawing rectangles (ROI)
            stroke_width = 3
            stroke_color = "red"
            bg_color = "#eee" # Background for the canvas area itself

            # --- Initialize Canvas ---
            # Use previous drawing if available, otherwise None
            initial_drawing = st.session_state.get("canvas_drawing")

            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.1)",  # Semi-transparent fill for ROI
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=display_img, # The crucial part: display the image
                update_streamlit=True, # Send updates back to Streamlit
                height=canvas_height,
                width=canvas_width,
                drawing_mode=drawing_mode,
                initial_drawing=initial_drawing, # Load previous state if exists
                key="roi_canvas",
            )

            # --- Process Canvas Result (ROI Detection) ---
            if canvas_result and canvas_result.json_data and canvas_result.json_data["objects"]:
                # Get the last drawn object (assuming user draws one ROI)
                # If multiple objects allowed, might need different logic
                last_object = canvas_result.json_data["objects"][-1]
                if last_object["type"] == "rect":
                    # Store the full drawing state (for potential reloading)
                    st.session_state.canvas_drawing = canvas_result.json_data
                    # Extract coordinates and store them
                    st.session_state.roi_coords = {
                        "left": int(last_object["left"]),
                        "top": int(last_object["top"]),
                        "width": int(last_object["width"]),
                        "height": int(last_object["height"])
                    }
                    logger.info(f"ROI updated: {st.session_state.roi_coords}")
                # Handle potential clearing or modification (canvas may return empty objects list)
            elif st.session_state.roi_coords is not None and (not canvas_result or not canvas_result.json_data or not canvas_result.json_data["objects"]):
                 # If ROI existed but canvas is now empty, user likely cleared/undid it
                 # However, the "Clear ROI" button is the explicit way to clear
                 # We might not need this logic if button is primary clearing mechanism
                 # logger.info("ROI appears to have been cleared via canvas interaction.")
                 # st.session_state.roi_coords = None
                 # st.session_state.canvas_drawing = None
                 pass

            # Display ROI info if it exists
            if st.session_state.roi_coords:
                roi = st.session_state.roi_coords
                st.caption(f"Current ROI: ({roi['left']}, {roi['top']}) Size: {roi['width']}x{roi['height']}")
            else:
                st.caption("No ROI selected. Draw on the image or use 'Clear ROI' button.")

        elif not DRAWABLE_CANVAS_AVAILABLE:
            # Fallback if canvas is not installed - just display the image
            st.image(display_img, caption="Image Preview (Drawable Canvas not available)", use_container_width=True)
            st.warning("ROI selection disabled: `streamlit-drawable-canvas` not installed.", icon="⚠️")
        else:
            # Should not happen if display_img is Image and DRAWABLE_CANVAS_AVAILABLE is True
             st.error("Image viewer component failed to load.")
             logger.error("Drawable canvas available but st_canvas object is None.")

        st.markdown("---")

        # Display DICOM Metadata if it exists
        if st.session_state.is_dicom and st.session_state.dicom_metadata:
            with st.expander("📄 View DICOM Metadata", expanded=False):
                if UI_COMPONENTS_AVAILABLE:
                    display_dicom_metadata(st.session_state.dicom_metadata)
                else:
                    # Basic fallback display if ui_components module failed
                    st.json(dict(list(st.session_state.dicom_metadata.items())[:15])) # Show first 15 tags
                    st.caption("Basic metadata view. Full view requires `ui_components`.")
        elif st.session_state.is_dicom:
            st.caption("DICOM file loaded, but metadata extraction failed or was empty.")

    elif uploaded_file is not None:
        # If a file was uploaded but display_img is None, means processing failed
        st.error("Image preview failed. Check logs or try uploading again.")
        logger.warning("display_img is None even though a file was uploaded, indicating a processing error.")
    else:
        # Initial state before upload
        st.info("⬅️ Upload a de-identified image or use Demo Mode in the sidebar to begin.")


# --- Column 2: Analysis Results & Interaction Tabs ---
with col2:
    st.subheader("📊 Analysis & Interaction")

    # Define tabs
    tab_titles = [
        "🔬 Structured Analysis",
        "💬 Q&A History",
        "🩺 Condition Focus",
        "🧪 LLM Self-Assessment",
        "🌐 Translation"
    ]
    tabs = st.tabs(tab_titles)

    # Tab 1: Structured Analysis
    with tabs[0]:
        st.caption("AI's general analysis based on the image (and ROI if selected).")
        analysis_text = st.session_state.initial_analysis or "No analysis performed yet. Use the 'Run Structured Initial Analysis' button in the sidebar."
        st.markdown(analysis_text) # Use markdown to render potential formatting

    # Tab 2: Q&A History
    with tabs[1]:
        st.caption("Your questions and the AI's answers.")
        st.markdown("**Latest AI Answer:**")
        latest_answer = st.session_state.qa_answer or "_Ask a question using the sidebar._"
        st.markdown(latest_answer) # Display latest answer prominently
        st.markdown("---")

        if st.session_state.history:
            with st.expander("Full Interaction History", expanded=True):
                # Display history in reverse chronological order? Or chronological? Chronological seems better.
                for i, (q_type, message) in enumerate(st.session_state.history):
                    if "user" in q_type.lower():
                        st.markdown(f"**You:** {message}")
                    elif "ai" in q_type.lower() and "[fallback]" not in q_type.lower():
                         st.markdown(f"**AI:**\n{message}")
                    elif "[fallback]" in q_type.lower():
                         # Extract model name if present in q_type
                         fallback_model = q_type.split(":")[-1].strip() if ":" in q_type else "Fallback"
                         st.markdown(f"**AI ({fallback_model}):**\n{message.split('**')[-1].strip()}") # Clean up potential markdown formatting in message
                    else: # Handle other types if needed
                         st.markdown(f"**{q_type}:** {message}")

                    if i < len(st.session_state.history) - 1:
                        st.markdown("---") # Separator between entries
        else:
            st.caption("No questions asked yet in this session.")

    # Tab 3: Condition Focus
    with tabs[2]:
        st.caption("AI's analysis focused on the selected condition.")
        condition_text = st.session_state.disease_analysis or "No condition-specific analysis performed. Select a condition and use the 'Analyze for Condition' button in the sidebar."
        st.markdown(condition_text)

    # Tab 4: LLM Self-Assessment
    with tabs[3]:
        st.caption("EXPERIMENTAL: AI's self-assessment of its last Q&A response quality/confidence.")
        st.warning("""**⚠️ Note:** This score is experimental and generated by the AI itself. It should not be considered a clinical measure of confidence or accuracy. Use with extreme caution.""", icon="🧪")
        confidence_text = st.session_state.confidence_score or "No self-assessment performed. Use the 'Estimate LLM Self-Assessment' button in the sidebar after asking a question."
        st.markdown(confidence_text)

    # Tab 5: Translation
    with tabs[4]:
        st.subheader("🌐 Translate Analysis Text")
        if not TRANSLATION_AVAILABLE:
            st.warning("Translation features are unavailable. The `deep-translator` library might be missing or failed to install.", icon="🚫")
        else:
            st.caption("Translate generated analysis text into different languages.")
            # Options for text selection
            text_options = {
                "Structured Initial Analysis": st.session_state.initial_analysis,
                "Latest Q&A Answer": st.session_state.qa_answer,
                "Condition Analysis": st.session_state.disease_analysis,
                "LLM Self-Assessment": st.session_state.confidence_score,
                "(Enter Custom Text Below)": "" # Option for custom input
            }
            # Filter out options with no text, but always keep custom option
            available_labels = [lbl for lbl, txt in text_options.items() if txt or lbl=="(Enter Custom Text Below)"]
            if not available_labels: # Ensure custom option is always there
                 available_labels = ["(Enter Custom Text Below)"]

            selected_label = st.selectbox("Select text to translate:", options=available_labels, key="ts_select")
            text_to_translate = text_options.get(selected_label, "")

            # Show custom text area if selected
            if selected_label == "(Enter Custom Text Below)":
                text_to_translate = st.text_area("Enter text to translate:", value="", height=100, key="ts_custom")

            # Display the text that will be translated (read-only)
            st.text_area("Selected text:", value=text_to_translate or " ", height=80, disabled=True, key="ts_preview")

            # Language selection dropdowns
            cl1, cl2 = st.columns(2)
            with cl1:
                src_opts = [AUTO_DETECT_INDICATOR] + sorted(list(LANGUAGE_CODES.keys()))
                src_lang_display = st.selectbox("From Language:", src_opts, key="ts_src", help="Select source language or use Auto-Detect.")
            with cl2:
                tgt_opts = sorted(list(LANGUAGE_CODES.keys()))
                # Try to pre-select common target like Spanish or keep English as default
                tgt_idx=0
                preferred_targets=["Spanish", "English"] # Can change order or add more
                try:
                    for pref in preferred_targets:
                        if pref in tgt_opts:
                            tgt_idx = tgt_opts.index(pref)
                            break
                except ValueError: pass # If preferred not found, default to index 0 (usually English if sorted)

                tgt_lang_display = st.selectbox("To Language:", tgt_opts, index=tgt_idx, key="ts_tgt")

            # Translate button
            if st.button("🔄 Translate", key="ts_btn"):
                # Reset previous results/errors
                st.session_state.translation_result = None
                st.session_state.translation_error = None

                if not text_to_translate or not text_to_translate.strip():
                    st.warning("No text provided to translate.", icon="☝️")
                    st.session_state.translation_error = "Input text is empty."
                elif src_lang_display == tgt_lang_display and src_lang_display != AUTO_DETECT_INDICATOR:
                     st.info("Source and target languages are the same.", icon="✅")
                     st.session_state.translation_result = text_to_translate # No translation needed
                else:
                    # Get language codes from display names
                    src_code = AUTO_DETECT_INDICATOR if src_lang_display == AUTO_DETECT_INDICATOR else LANGUAGE_CODES.get(src_lang_display)
                    tgt_code = LANGUAGE_CODES.get(tgt_lang_display)

                    if not tgt_code:
                        st.error(f"Invalid target language selected: {tgt_lang_display}", icon="❌")
                        st.session_state.translation_error = "Invalid target language."
                    else:
                        with st.spinner(f"Translating from '{src_lang_display}' to '{tgt_lang_display}'..."):
                            try:
                                # Call the translation function from translation_models.py
                                translated_output = translate(
                                    text=text_to_translate,
                                    target_language=tgt_lang_display, # Pass display name or code based on function needs
                                    source_language=src_lang_display # Pass display name or code
                                )
                                if translated_output is not None:
                                    st.session_state.translation_result = translated_output
                                    st.success("Translation successful!", icon="🎉")
                                else:
                                    st.error("Translation service returned an empty result.", icon="❓")
                                    logger.warning("translate function returned None.")
                                    st.session_state.translation_error = "Translation service returned empty result."
                            except Exception as e:
                                st.error(f"Translation failed: {e}", icon="❌")
                                logger.error(f"Translation call failed: {e}", exc_info=True)
                                st.session_state.translation_error = str(e)

            # Display translation result or error
            if st.session_state.get("translation_result"):
                formatted_result = format_translation(st.session_state.translation_result)
                st.text_area("Translation:", value=formatted_result, height=150, key="ts_out")
            elif st.session_state.get("translation_error"):
                st.info(f"Translation Status: {st.session_state.translation_error}", icon="ℹ️")


# --- Button Action Handlers (Centralized Logic) ---
current_action = st.session_state.get("last_action")

if current_action:
    logger.info(f"Handling action: '{current_action}' for session: {st.session_state.session_id}")

    # --- Pre-Action Checks ---
    action_requires_image = current_action in ["analyze", "ask", "disease", "confidence", "generate_report_data"]
    action_requires_llm = current_action in ["analyze", "ask", "disease", "confidence"]
    action_requires_report_util = (current_action == "generate_report_data")
    error_occurred = False

    if action_requires_image and not isinstance(st.session_state.get("processed_image"), Image.Image):
        st.error(f"Cannot perform '{current_action}': No valid image loaded.", icon="🖼️")
        error_occurred = True
    if not st.session_state.session_id:
        st.error("Critical error: Session ID is missing.", icon="🆔")
        error_occurred = True # Should not happen but check anyway
    if action_requires_llm and not LLM_INTERACTIONS_AVAILABLE:
        st.error(f"Cannot perform '{current_action}': Core AI module (llm_interactions) is unavailable.", icon="🤖")
        error_occurred = True
    if action_requires_report_util and not REPORT_UTILS_AVAILABLE:
        st.error(f"Cannot perform '{current_action}': PDF reporting module (report_utils) is unavailable.", icon="📄")
        error_occurred = True

    if error_occurred:
        st.session_state.last_action = None # Reset action if checks fail
        st.stop() # Stop script execution for this run

    # --- Execute Action ---
    img_for_llm = st.session_state.processed_image
    roi_coords = st.session_state.roi_coords # Get current ROI, if any
    current_history = st.session_state.history # Get current history

    try:
        if current_action == "analyze":
            st.toast("🔬 Running initial analysis...", icon="⏳")
            with st.spinner("AI is performing structured analysis..."):
                analysis_result = run_initial_analysis(img_for_llm, roi=roi_coords)
                st.session_state.initial_analysis = analysis_result
                st.session_state.qa_answer = "" # Clear other results
                st.session_state.disease_analysis = ""
                logger.info("Initial analysis completed.")
                st.success("Structured analysis complete!", icon="✅")

        elif current_action == "ask":
            question_text = st.session_state.question_input_widget.strip()
            if not question_text:
                st.warning("Question input was empty.", icon="❓")
            else:
                st.toast(f"💬 Asking AI: '{question_text[:50]}...'")
                st.session_state.qa_answer = "" # Clear previous answer
                with st.spinner("AI is thinking..."):
                    # Run primary Q&A model
                    answer, success_flag = run_multimodal_qa(
                        image=img_for_llm,
                        question=question_text,
                        history=current_history,
                        roi=roi_coords
                    )

                    if success_flag:
                        st.session_state.qa_answer = answer
                        # Add interaction to history
                        st.session_state.history.append(("User Question", question_text))
                        st.session_state.history.append(("AI Answer", answer))
                        logger.info(f"Q&A successful for question: '{question_text[:50]}...'")
                        st.success("AI has answered!", icon="💬")
                    else:
                        # Primary AI failed
                        primary_error_msg = f"Primary AI failed to answer. Error: {answer}"
                        st.session_state.qa_answer = primary_error_msg
                        st.error(primary_error_msg, icon="⚠️")
                        logger.warning(f"Primary Q&A failed for question '{question_text[:50]}...': {answer}")

                        # --- Attempt Fallback ---
                        hf_token = os.environ.get("HF_API_TOKEN") or st.secrets.get("HF_API_TOKEN")
                        if HF_MODELS_AVAILABLE and hf_token:
                            st.info(f"Attempting fallback analysis with Hugging Face model: {HF_VQA_MODEL_ID}", icon="🔄")
                            with st.spinner(f"Running fallback model ({HF_VQA_MODEL_ID})..."):
                                try:
                                    fallback_answer, fallback_success = query_hf_vqa_inference_api(
                                        img=img_for_llm, # Ensure correct image passed
                                        question=question_text,
                                        roi=roi_coords # Pass ROI if available
                                    )
                                except Exception as hf_e:
                                    fallback_success = False
                                    fallback_answer = f"Exception during fallback: {hf_e}"
                                    logger.error(f"Exception calling Hugging Face fallback API: {hf_e}", exc_info=True)

                                if fallback_success:
                                    fallback_display = f"**[Fallback: {HF_VQA_MODEL_ID}]**\n\n{fallback_answer}"
                                    # Append fallback result to the display message
                                    st.session_state.qa_answer += f"\n\n---\n\n{fallback_display}"
                                    # Add fallback interaction to history with clear labels
                                    st.session_state.history.append(("[Fallback] User Question", question_text))
                                    st.session_state.history.append((f"[Fallback] AI Answer:{HF_VQA_MODEL_ID}", fallback_display)) # Store model in type
                                    st.success(f"Fallback model ({HF_VQA_MODEL_ID}) provided an answer.", icon="👍")
                                    logger.info(f"Fallback Q&A successful with {HF_VQA_MODEL_ID}.")
                                else:
                                    fallback_error_msg = f"[Fallback Error - {HF_VQA_MODEL_ID}]: {fallback_answer}"
                                    st.session_state.qa_answer += f"\n\n---\n\n{fallback_error_msg}"
                                    st.error(f"Fallback model ({HF_VQA_MODEL_ID}) also failed.", icon="👎")
                                    logger.warning(f"Fallback Q&A failed with {HF_VQA_MODEL_ID}: {fallback_answer}")
                        elif HF_MODELS_AVAILABLE and not hf_token:
                            no_token_msg = "[Fallback Skipped: Hugging Face API Token (HF_API_TOKEN) is missing in environment or secrets.]"
                            st.session_state.qa_answer += f"\n\n---\n\n{no_token_msg}"
                            st.warning("Could not attempt fallback analysis: HF_API_TOKEN is not configured.", icon="🔑")
                            logger.warning("Fallback skipped due to missing HF_API_TOKEN.")
                        else:
                            no_fallback_msg = "[Fallback Not Attempted: Fallback module (hf_models) is not available.]"
                            st.session_state.qa_answer += f"\n\n---\n\n{no_fallback_msg}"
                            logger.warning("Fallback skipped because hf_models module is not available.")

        elif current_action == "disease":
            selected_disease = st.session_state.disease_select_widget
            if not selected_disease:
                st.warning("No condition was selected from the dropdown.", icon="🏷️")
            else:
                st.toast(f"🩺 Analyzing for signs of '{selected_disease}'...", icon="⏳")
                with st.spinner(f"AI is analyzing for {selected_disease}..."):
                    disease_result = run_disease_analysis(
                        image=img_for_llm,
                        disease=selected_disease,
                        roi=roi_coords
                    )
                    st.session_state.disease_analysis = disease_result
                    st.session_state.qa_answer = "" # Clear other results
                    logger.info(f"Disease-specific analysis for '{selected_disease}' completed.")
                    st.success(f"Analysis for '{selected_disease}' complete!", icon="✅")

        elif current_action == "confidence":
            if not current_history:
                 # Should be prevented by button disabling, but double-check
                st.warning("Cannot estimate assessment: No Q&A history exists yet.", icon="📊")
            else:
                st.toast("🧪 Estimating LLM self-assessment...", icon="⏳")
                with st.spinner("AI is assessing its previous response..."):
                    # Using the corrected function name
                    assessment_result = run_llm_self_assessment(
                        image=img_for_llm,
                        history=current_history,
                        roi=roi_coords
                    )
                    st.session_state.confidence_score = assessment_result
                    logger.info("LLM self-assessment completed.")
                    st.success("Self-assessment estimation complete!", icon="✅")

        elif current_action == "generate_report_data":
            st.toast("📄 Generating PDF report data...", icon="⏳")
            st.session_state.pdf_report_bytes = None # Clear previous report data
            image_for_report = st.session_state.get("display_image") # Use the display image (e.g., with W/L)

            if not isinstance(image_for_report, Image.Image):
                st.error("Cannot generate report: No valid image available for the report.", icon="🖼️")
            else:
                # Prepare image: ensure RGB, draw ROI if present
                final_image_for_pdf = image_for_report.copy().convert("RGB")
                if roi_coords:
                    try:
                        draw = ImageDraw.Draw(final_image_for_pdf)
                        x0, y0 = roi_coords['left'], roi_coords['top']
                        w, h = roi_coords['width'], roi_coords['height']
                        # Ensure ROI is within image bounds (optional, good practice)
                        x1, y1 = min(x0 + w, final_image_for_pdf.width), min(y0 + h, final_image_for_pdf.height)
                        x0, y0 = max(0, x0), max(0, y0)
                        # Draw rectangle with dynamic thickness based on image size
                        outline_width = max(3, int(min(final_image_for_pdf.size) * 0.005))
                        draw.rectangle([x0, y0, x1, y1], outline="red", width=outline_width)
                        logger.info("ROI drawn on image for PDF report.")
                    except Exception as draw_e:
                        logger.error(f"Failed to draw ROI on report image: {draw_e}", exc_info=True)
                        st.warning("Could not draw the ROI rectangle onto the report image.", icon="✏️")

                # Compile text data for the report
                # Format history nicely
                formatted_history = "No Q&A interactions recorded."
                if current_history:
                    lines = []
                    for q_type, message in current_history:
                        # Clean up potential HTML/Markdown tags from messages if needed
                        cleaned_message = re.sub('<[^<]+?>', '', str(message)).strip()
                        lines.append(f"[{q_type}]:\n{cleaned_message}")
                    formatted_history = "\n\n---\n\n".join(lines)

                report_data = {
                    "Session ID": st.session_state.session_id,
                    "Image Filename": (st.session_state.uploaded_file_info or "N/A").split('-')[0], # Extract filename
                    "Structured Initial Analysis": st.session_state.initial_analysis or "Not Performed",
                    "Q&A History": formatted_history,
                    "Condition Specific Analysis": st.session_state.disease_analysis or "Not Performed",
                    "LLM Self-Assessment (Experimental)": st.session_state.confidence_score or "Not Performed",
                }

                # Add summary of key DICOM tags if available
                if st.session_state.is_dicom and st.session_state.dicom_metadata:
                    # Select key tags for summary (customize as needed)
                    meta_tags_for_summary = [
                        'PatientName', 'PatientID', 'StudyDate', 'Modality',
                        'StudyDescription', 'InstitutionName', 'Manufacturer'
                    ]
                    meta_summary = {
                        tag: st.session_state.dicom_metadata.get(tag, "N/A")
                        for tag in meta_tags_for_summary if tag in st.session_state.dicom_metadata
                    }
                    if meta_summary:
                        # Format summary nicely (e.g., shorter labels)
                        label_map = {'PatientName':'Pt Name', 'PatientID':'Pt ID', 'StudyDate':'Date', 'StudyDescription':'Desc.', 'InstitutionName': 'Institution'}
                        summary_lines = [f"{label_map.get(k, k)}: {v}" for k, v in meta_summary.items()]
                        report_data["DICOM Summary"] = "\n".join(summary_lines)

                # Generate PDF using utility function
                with st.spinner("Compiling PDF document..."):
                    pdf_bytes = generate_pdf_report_bytes(
                        session_id=st.session_state.session_id,
                        image=final_image_for_pdf,
                        analysis_outputs=report_data,
                        dicom_metadata=st.session_state.dicom_metadata if st.session_state.is_dicom else None
                    )
                    if pdf_bytes:
                        st.session_state.pdf_report_bytes = pdf_bytes
                        st.success("PDF report data generated successfully!", icon="📄")
                        logger.info("PDF report data generated and stored in session state.")
                        st.balloons() # Fun success indicator
                    else:
                        st.error("PDF generation failed. Check logs for details.", icon="❌")
                        logger.error("generate_pdf_report_bytes returned None.")

        else:
            # Should not happen if buttons are defined correctly
            st.warning(f"Unknown action '{current_action}' encountered.", icon="❓")
            logger.warning(f"Unhandled action type received: '{current_action}'.")

    except Exception as e:
        st.error(f"An unexpected error occurred while processing action '{current_action}': {e}", icon="💥")
        logger.critical(f"Error during execution of action '{current_action}': {e}", exc_info=True)
        # Optionally clear specific results if action failed mid-way
        # e.g., st.session_state.qa_answer = "Action failed."

    finally:
        # --- Post-Action Cleanup ---
        st.session_state.last_action = None # Reset the action trigger
        logger.debug(f"Action '{current_action}' handling complete.")
        st.rerun() # Rerun to update the UI based on state changes

# --- Footer ---
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced | Session ID: {st.session_state.get('session_id', 'N/A')} | For Research & Education Only")

# Custom Footer HTML
st.markdown(
    """
    <div class="footer">
        <p>RadVision AI Advanced - Experimental Tool - Not for Clinical Use - © 2024</p>
    </div>
    """,
    unsafe_allow_html=True
)

logger.info(f"--- Application render cycle complete for session: {st.session_state.session_id} ---")
# --- End of app.py ---