# -*- coding: utf-8 -*-
"""
app.py - Main Streamlit application for RadVision AI Advanced.

Handles image uploading (DICOM, JPG, PNG), display, ROI selection,
interaction with AI models for analysis and Q&A, translation,
and report generation.
"""

# Ensure this is the very first command in the script
import streamlit as st
st.set_page_config(
    page_title="RadVision AI Advanced",
    layout="wide", # Use full width
    page_icon="⚕️", # Favicon
    initial_sidebar_state="expanded" # Keep sidebar open initially
)

# --- Core Libraries ---
import io
import os
import uuid
import logging
import base64
from typing import Any, Dict, Optional, Tuple, List, Union # Added Union
import copy
import random  # For Tip of the Day
import re      # For formatting the translation output

# --- Logging Setup (Early) ---
# Set level from environment variable or default to INFO
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__) # Get logger for this module

# --- Dependency Checks & Version Logging (Informative) ---
logger.info("--- RadVision AI Application Start ---")
logger.info(f"Streamlit Version: {st.__version__}")
logger.info(f"Logging Level: {LOG_LEVEL}")

# Drawable Canvas
try:
    from streamlit_drawable_canvas import st_canvas
    import streamlit_drawable_canvas as st_canvas_module
    CANVAS_VERSION = getattr(st_canvas_module, '__version__', 'Unknown') # Prefer __version__
    logger.info(f"Streamlit Drawable Canvas Version: {CANVAS_VERSION}")
    DRAWABLE_CANVAS_AVAILABLE = True
except ImportError:
    st.error("CRITICAL ERROR: streamlit-drawable-canvas is not installed. Functionality will be limited. Run `pip install streamlit-drawable-canvas`.")
    logger.critical("streamlit-drawable-canvas not found. App functionality impaired.")
    DRAWABLE_CANVAS_AVAILABLE = False
    st_canvas = None # Define as None to avoid NameErrors later if used conditionally

# Pillow (PIL)
try:
    from PIL import Image, ImageDraw, UnidentifiedImageError
    import PIL
    PIL_VERSION = getattr(PIL, '__version__', 'Unknown') # Prefer __version__
    logger.info(f"Pillow (PIL) Version: {PIL_VERSION}")
    PIL_AVAILABLE = True
except ImportError:
    st.error("CRITICAL ERROR: Pillow (PIL) is not installed. Image processing is disabled. Run `pip install Pillow`.")
    logger.critical("Pillow (PIL) not found. App functionality severely impaired.")
    PIL_AVAILABLE = False
    # Define dummy classes if needed downstream, although graceful failure is better
    Image = None
    ImageDraw = None
    UnidentifiedImageError = None
    st.stop() # Stop execution if PIL is missing

# Pydicom and related libraries
try:
    import pydicom
    import pydicom.errors
    PYDICOM_VERSION = getattr(pydicom, '__version__', 'Unknown') # Prefer __version__
    logger.info(f"Pydicom Version: {PYDICOM_VERSION}")
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_VERSION = 'Not Installed'
    logger.warning("pydicom not found. DICOM functionality will be disabled.")
    pydicom = None # Define as None
    PYDICOM_AVAILABLE = False

# Optional DICOM libraries
if PYDICOM_AVAILABLE:
    try:
        import pylibjpeg
        logger.info("pylibjpeg found (for extended DICOM decompression).")
    except ImportError:
        logger.info("pylibjpeg not found. DICOM support might be limited for certain transfer syntaxes.")
    try:
        import gdcm
        logger.info("python-gdcm found (for improved DICOM compatibility).")
    except ImportError:
        logger.info("python-gdcm not found. DICOM compatibility might be reduced.")

# --- Custom Utilities & Backend Modules ---
# Wrap imports in try-except blocks for graceful failure and clear logging
try:
    from dicom_utils import (parse_dicom, extract_dicom_metadata,
                             dicom_to_image, get_default_wl)
    DICOM_UTILS_AVAILABLE = True
    logger.info("dicom_utils imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import dicom_utils: {e}. DICOM features disabled.")
    DICOM_UTILS_AVAILABLE = False
    # Define dummy functions or rely on PYDICOM_AVAILABLE checks later

try:
    from llm_interactions import (run_initial_analysis, run_multimodal_qa,
                                  run_disease_analysis, estimate_ai_confidence)
    LLM_INTERACTIONS_AVAILABLE = True
    logger.info("llm_interactions imported successfully.")
except ImportError as e:
    st.error(f"Core AI module (llm_interactions) failed to import: {e}. Analysis functions disabled.")
    logger.critical(f"Failed to import llm_interactions: {e}", exc_info=True)
    LLM_INTERACTIONS_AVAILABLE = False
    st.stop() # Stop if core AI module is missing

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
    logger.warning(f"Failed to import ui_components: {e}. Custom UI elements might be missing.")
    UI_COMPONENTS_AVAILABLE = False
    # Define dummy functions if they are critical and used without checks
    def display_dicom_metadata(metadata): st.json(metadata[:5]) # Simple fallback
    def dicom_wl_sliders(wc, ww): return wc, ww # No-op fallback

# HF fallback for Q&A (Optional)
try:
    from hf_models import query_hf_vqa_inference_api, HF_VQA_MODEL_ID
    HF_MODELS_AVAILABLE = True
    logger.info(f"hf_models imported successfully (Fallback VQA Model: {HF_VQA_MODEL_ID}).")
except ImportError:
    HF_VQA_MODEL_ID = "hf_model_not_found"
    HF_MODELS_AVAILABLE = False
    # Define dummy fallback function
    def query_hf_vqa_inference_api(img: Image.Image, question: str, roi: Optional[Dict] = None) -> Tuple[str, bool]:
        logger.warning("query_hf_vqa_inference_api called but hf_models module is unavailable.")
        return "[Fallback VQA Unavailable] HF module not found.", False
    logger.warning("hf_models not found. Fallback VQA disabled.")


# --- Translation Module Import & Setup ---
# Use the updated module returning Optional[str]
try:
    # Import the core functions and constants
    from translation_models import translate, detect_language, LANGUAGE_CODES, AUTO_DETECT_INDICATOR
    # Check if the library itself was available *within* translation_models
    from translation_models import DEEP_TRANSLATOR_AVAILABLE
    if not DEEP_TRANSLATOR_AVAILABLE:
        st.warning("Translation library (deep-translator) not found. Translation features will be disabled.")
        logger.warning("deep-translator library not found by translation_models module.")
        TRANSLATION_AVAILABLE = False
        translate = None # Explicitly set to None
        detect_language = None # Explicitly set to None
    else:
        TRANSLATION_AVAILABLE = True
        logger.info("translation_models imported successfully.")
except ImportError as e:
    st.warning(f"Could not import translation_models: {e}. Translation features disabled.")
    logger.error(f"Failed to import translation_models: {e}", exc_info=True)
    TRANSLATION_AVAILABLE = False
    translate = None
    detect_language = None
    LANGUAGE_CODES = {"English": "en"} # Minimal fallback
    AUTO_DETECT_INDICATOR = "Auto-Detect"

# --- Custom CSS for Polished Look & Tab Scrolling ---
st.markdown(
    """
    <style>
      /* General Styling */
      body {
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
          background-color: #f0f2f6; /* Light gray background */
      }
      /* Main content area styling */
      .main .block-container {
          padding-top: 2rem;
          padding-bottom: 2rem;
          padding-left: 1.5rem;
          padding-right: 1.5rem;
      }
      /* Sidebar styling */
      .css-1d391kg { /* Specific class for Streamlit sidebar, might change */
          background-color: #ffffff; /* White sidebar */
          border-right: 1px solid #e0e0e0; /* Subtle border */
      }
      /* Enhance button appearance */
      .stButton>button {
          border-radius: 8px;
          padding: 0.5rem 1rem;
          font-weight: 500;
          transition: background-color 0.2s ease-in-out, border-color 0.2s ease-in-out;
          /* Add specific styles for primary/secondary if needed */
      }
      .stButton>button:hover {
        filter: brightness(95%);
      }
      /* Style for tabs to allow scrolling on smaller screens */
      div[role="tablist"] {
          overflow-x: auto; /* Enable horizontal scrolling */
          white-space: nowrap; /* Prevent tabs from wrapping */
          border-bottom: 1px solid #e0e0e0; /* Add a bottom border */
          scrollbar-width: thin; /* For Firefox */
          scrollbar-color: #cccccc #f0f2f6; /* For Firefox */
      }
      /* Webkit scrollbar styling */
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
      /* Footer styling */
      footer {
          text-align: center;
          font-size: 0.8em;
          color: #6c757d; /* Muted gray color */
          margin-top: 2rem;
          padding: 1rem 0;
          border-top: 1px solid #e0e0e0;
      }
      footer a {
          color: #007bff; /* Standard link blue */
          text-decoration: none;
      }
      footer a:hover {
          text-decoration: underline;
      }
    </style>
    """, unsafe_allow_html=True
)

# --- Display Hero Logo ---
# Use os.path.join for cross-platform compatibility
logo_path = os.path.join("assets", "radvisionai-hero.jpeg")
if os.path.exists(logo_path):
    st.image(logo_path, width=350) # Slightly smaller width
else:
    logger.warning(f"Hero logo not found at expected path: {logo_path}")
    st.warning("Hero logo image (radvisionai-hero.jpeg) not found in the 'assets' folder.")

# --- Initialize Session State Defaults ---
# Define default values for all session state keys used
DEFAULT_STATE = {
    "uploaded_file_info": None,       # Stores identifier for the uploaded file (name-size-mtime/hash)
    "raw_image_bytes": None,          # Raw bytes of the uploaded file
    "is_dicom": False,                # Flag indicating if the file is DICOM
    "dicom_dataset": None,            # Stores the parsed pydicom dataset object
    "dicom_metadata": {},             # Extracted, display-friendly DICOM metadata
    "processed_image": None,          # Image object (PIL) preprocessed for AI models
    "display_image": None,            # Image object (PIL) formatted for display (e.g., with W/L applied)
    "session_id": None,               # Unique ID for the current analysis session
    "history": [],                    # List of (question, answer) tuples for Q&A
    "initial_analysis": "",           # Stores the result of the initial analysis AI call
    "qa_answer": "",                  # Stores the latest answer from Q&A AI call
    "disease_analysis": "",           # Stores the result of the disease-specific AI call
    "confidence_score": "",           # Stores the estimated AI confidence score
    "last_action": None,              # Tracks the last button clicked to trigger logic
    "pdf_report_bytes": None,         # Stores the generated PDF report as bytes
    "canvas_drawing": None,           # Stores the JSON state of the drawable canvas
    "roi_coords": None,               # Stores calculated ROI coordinates {left, top, width, height} in original image space
    "current_display_wc": None,       # Current DICOM window center for display
    "current_display_ww": None,       # Current DICOM window width for display
    "clear_roi_feedback": False,      # Flag to show feedback after clearing ROI
    "demo_loaded": False,             # Flag to indicate if demo mode is active and loaded
    "translation_result": None,       # Stores the result of the last translation attempt
    "translation_error": None,        # Stores any error message from translation
}

# Ensure session state keys exist, initializing only if needed
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        # Use deepcopy for mutable defaults like lists or dicts
        st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value

# Ensure history is always a list (safety check)
if not isinstance(st.session_state.history, list):
    st.session_state.history = []

# Ensure a Session ID Exists for logging and tracking
if not st.session_state.get("session_id"):
    st.session_state.session_id = str(uuid.uuid4())[:8] # Generate a short UUID
logger.debug(f"Session state initialized/verified for session ID: {st.session_state.session_id}")


# --- Helper Functions ---

def format_translation(translated_text: Optional[str]) -> str:
    """
    Applies basic formatting to translated text, primarily for readability.
    Handles potential None input gracefully.
    """
    if translated_text is None:
        return "Translation not available or failed."

    # Example: Add line breaks before numbered list items (adjust regex as needed)
    # This attempts to restore some list formatting if the translation model merges lines.
    try:
        # Ensure text is treated as a string
        text_str = str(translated_text)
        # Replace space(s) before a number followed by a period with a newline
        formatted_text = re.sub(r'\s+(\d+\.)', r'\n\n\1', text_str)
        # Add more formatting rules here if necessary
        return formatted_text.strip()
    except Exception as e:
        logger.error(f"Error formatting translation: {e}", exc_info=True)
        return str(translated_text) # Return original on error


# --- Monkey-Patch (Conditional) ---
# Apply only if necessary (check Streamlit version or attribute existence)
# Note: Newer Streamlit versions might handle this better.
import streamlit.elements.image as st_image
if not hasattr(st_image, "image_to_url"):
    def image_to_url_monkey_patch(img_obj: Any, width: int = -1, clamp: bool = False,
                                  channels: str = "RGB", output_format: str = "auto",
                                  image_id: str = "") -> str:
        """Basic monkey-patch for image_to_url if missing."""
        if PIL_AVAILABLE and isinstance(img_obj, Image.Image):
            try:
                buffered = io.BytesIO()
                fmt = "PNG" if output_format.lower() == "auto" else output_format.upper()
                if fmt not in ["PNG", "JPEG"]: fmt = "PNG" # Default to PNG

                temp_img = img_obj
                # Handle color modes (simplified)
                if channels == "RGB" and temp_img.mode not in ['RGB', 'L', 'RGBA']:
                    temp_img = temp_img.convert('RGB')
                elif temp_img.mode == 'P': # Palette mode
                    temp_img = temp_img.convert('RGBA') # Convert to RGBA first

                temp_img.save(buffered, format=fmt)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/{fmt.lower()};base64,{img_str}"
            except Exception as e:
                logger.error(f"Monkey-patch image_to_url failed: {e}", exc_info=True)
                return ""
        else:
            logger.warning(f"Monkey-patch image_to_url: Unsupported type {type(img_obj)} or PIL unavailable.")
            return ""
    st_image.image_to_url = image_to_url_monkey_patch
    logger.info("Applied compatibility monkey-patch for st.elements.image.image_to_url.")


# --- Sidebar UI Elements ---
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
        "Tip: Clear the ROI using the button if you want the AI to consider the whole image again.",
    ]
    st.info(f"💡 {random.choice(TIPS)}")
    st.markdown("---")

    # --- Upload & DICOM Section ---
    st.header("Image Upload & Settings")
    uploaded_file = st.file_uploader(
        "Upload Image (JPG, PNG, DCM)",
        type=["jpg", "jpeg", "png", "dcm", "dicom"],
        key="file_uploader_widget", # Unique key for the widget
        help="Upload a medical image file for analysis. DICOM (.dcm) is preferred for metadata."
    )

    # Demo Mode Checkbox
    demo_mode = st.checkbox("🚀 Demo Mode", value=st.session_state.get("demo_loaded", False),
                            help="Load a sample chest X-ray image and analysis.")

    # ROI Clear Button
    if st.button("🗑️ Clear ROI", help="Remove the selected Region of Interest (ROI)"):
        st.session_state.roi_coords = None
        st.session_state.canvas_drawing = None # Reset canvas state as well
        st.session_state.clear_roi_feedback = True # Flag to show confirmation
        st.rerun() # Rerun to reflect the cleared ROI

    # Show feedback if ROI was just cleared
    if st.session_state.get("clear_roi_feedback"):
        st.success("✅ ROI cleared successfully!")
        st.balloons()
        st.session_state.clear_roi_feedback = False # Reset feedback flag


    # --- DICOM Window/Level Sliders (Conditional) ---
    if st.session_state.is_dicom and UI_COMPONENTS_AVAILABLE and st.session_state.display_image:
        st.markdown("---")
        st.subheader("DICOM Display")
        new_wc, new_ww = dicom_wl_sliders(
            st.session_state.current_display_wc,
            st.session_state.current_display_ww
        )
        # Update display image only if W/L values change
        if new_wc != st.session_state.current_display_wc or new_ww != st.session_state.current_display_ww:
            logger.info(f"DICOM W/L changed: WC={new_wc}, WW={new_ww}")
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
                         # Ensure RGB for display consistency
                        if new_display_img.mode != 'RGB':
                            new_display_img = new_display_img.convert('RGB')
                        st.session_state.display_image = new_display_img
                        st.rerun() # Rerun to update the image viewer
                    else:
                        st.error("Failed to update DICOM image with new W/L.")
                        logger.error("dicom_to_image failed to return a valid PIL Image after W/L change.")
            else:
                st.warning("DICOM utilities not available to update W/L.")

    # --- AI Actions Section ---
    st.markdown("---")
    st.header("🤖 AI Analysis Actions")

    # Disable buttons if no image is loaded, unless in demo mode
    action_disabled = not isinstance(st.session_state.get("processed_image"), Image.Image)

    # Initial Analysis Button
    if st.button("▶️ Run Initial Analysis", key="analyze_btn", disabled=action_disabled,
                 help="Perform a general analysis of the entire image or selected ROI."):
        st.session_state.last_action = "analyze"
        st.rerun()

    # Q&A Section
    st.subheader("❓ Ask AI a Question")
    question_input = st.text_area(
        "Enter your question:",
        height=100,
        key="question_input_widget",
        placeholder="E.g., 'Are there any nodules in the upper right lobe?', 'Describe the abnormality near the center.'",
        disabled=action_disabled,
        help="Ask a specific question about the image or the ROI."
    )
    if st.button("💬 Ask Question", key="ask_btn", disabled=action_disabled,
                 help="Submit your question to the AI for an answer based on the image/ROI."):
        if question_input.strip():
            st.session_state.last_action = "ask"
            st.rerun()
        else:
            st.warning("Please enter a question before submitting.")

    # Condition Analysis Section
    st.subheader("🎯 Condition-Specific Analysis")
    # Predefined list of common conditions for user convenience
    DISEASE_OPTIONS = [
        "Pneumonia", "Lung Cancer", "Nodule/Mass", "Effusion", "Fracture",
        "Stroke", "Appendicitis", "Bowel Obstruction", "Cardiomegaly",
        "Aortic Aneurysm", "Pulmonary Embolism", "Tuberculosis", "COVID-19",
        "Brain Tumor", "Arthritis", "Osteoporosis",
        # Add more relevant conditions
    ]
    # Add an empty option for default/no selection
    disease_select = st.selectbox(
        "Select condition to focus on:",
        options=[""] + sorted(DISEASE_OPTIONS),
        key="disease_select_widget",
        disabled=action_disabled,
        help="Ask the AI to specifically analyze the image for signs of the selected condition."
    )
    if st.button("🩺 Analyze Condition", key="disease_btn", disabled=action_disabled,
                 help="Run the AI analysis focused on the selected medical condition."):
        if disease_select:
            st.session_state.last_action = "disease"
            st.rerun()
        else:
            st.warning("Please select a condition from the list first.")

    # Confidence & Report Section
    st.markdown("---")
    st.header("📊 Confidence & Reporting")

    # Enable confidence estimation only if there's some analysis context
    can_estimate = bool(st.session_state.history or st.session_state.initial_analysis or st.session_state.disease_analysis)
    if st.button("📈 Estimate AI Confidence", key="confidence_btn", disabled=not can_estimate or action_disabled,
                 help="Estimate the AI's confidence based on the analysis performed so far."):
        if can_estimate:
            st.session_state.last_action = "confidence"
            st.rerun()
        else:
            st.warning("Perform at least one analysis (Initial, Q&A, or Condition) before estimating confidence.")

    # PDF Report Generation
    report_generation_disabled = action_disabled or not REPORT_UTILS_AVAILABLE
    if st.button("📄 Generate PDF Report Data", key="generate_report_data_btn",
                 disabled=report_generation_disabled,
                 help="Compile the current analysis, Q&A history, and image into data for a PDF report."):
        st.session_state.last_action = "generate_report_data"
        st.rerun()

    # PDF Download Button (appears only after data is generated)
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
        # Optionally clear the bytes after download button is shown to prevent repeated large state
        # Consider if user might want to download multiple times vs. state size
        # st.session_state.pdf_report_bytes = None


# --- File Upload and Demo Mode Logic ---

# Handle Demo Mode Activation
if demo_mode and not st.session_state.get("demo_loaded"):
    logger.info("Activating Demo Mode...")
    demo_img_path = os.path.join("assets", "demo.png") # Assuming demo image is PNG
    if os.path.exists(demo_img_path) and PIL_AVAILABLE:
        try:
            # Load demo image
            demo_img = Image.open(demo_img_path).convert("RGB")

            # Reset relevant state variables
            for key, value in DEFAULT_STATE.items():
                 # Don't reset flags like demo_loaded or sidebar state if needed
                if key not in {"file_uploader_widget", "demo_loaded"}:
                    st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value

            # Set demo-specific state
            st.session_state.display_image = demo_img.copy()
            st.session_state.processed_image = demo_img.copy()
            st.session_state.session_id = "demo-session" # Specific ID for demo
            # Provide sample history and analysis for demo
            st.session_state.history = [
                ("System", "Demo mode activated. Sample Chest X-ray loaded."),
                ("User Question", "Are there any signs of pneumonia?"),
                ("AI Answer", "Based on the demo image, there appears to be consolidation in the lower right lung field, which could be indicative of pneumonia. However, correlation with clinical findings is recommended. This is a simulated response for demonstration purposes.")
            ]
            st.session_state.initial_analysis = "Demo Initial Analysis: The simulated chest X-ray shows potential consolidation in the right lower lobe. The heart size appears normal. No clear signs of pneumothorax or significant pleural effusion are noted in this simulated view."
            st.session_state.qa_answer = st.session_state.history[-1][1] # Show last demo answer
            st.session_state.demo_loaded = True # Set flag
            st.success("🚀 Demo mode activated! Sample image and analysis loaded.")
            logger.info("Demo mode successfully loaded.")
            st.rerun() # Rerun to reflect demo state immediately

        except FileNotFoundError:
            st.sidebar.error("Demo image file ('demo.png') not found in 'assets' folder.")
            logger.error("Demo image file not found.")
            st.session_state.demo_loaded = False # Ensure flag is false
        except UnidentifiedImageError:
            st.sidebar.error("Demo image file ('demo.png') is corrupted or not a valid image.")
            logger.error("Demo image file is invalid.")
            st.session_state.demo_loaded = False
        except Exception as e:
            st.sidebar.error(f"An error occurred loading the demo image: {e}")
            logger.error(f"Error loading demo image: {e}", exc_info=True)
            st.session_state.demo_loaded = False
    elif not PIL_AVAILABLE:
         st.sidebar.error("Cannot load demo image because Pillow (PIL) is not installed.")
    else:
        st.sidebar.warning("Demo image ('demo.png') not found in 'assets' folder.")
        logger.warning("Demo image file not found.")

# Handle File Upload Processing
if uploaded_file is not None:
    # Generate a unique identifier for the uploaded file to detect changes
    try:
        # Use hash of content for robustness if mtime is unreliable or unavailable
        uploaded_file.seek(0) # Ensure reading from the start
        file_content_hash = hashlib.sha256(uploaded_file.read()).hexdigest()[:16]
        uploaded_file.seek(0) # Reset pointer after reading
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{file_content_hash}"
    except Exception as e:
        logger.warning(f"Could not generate hash for file info, using random ID: {e}")
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{uuid.uuid4().hex[:8]}"

    # Check if it's a new file compared to the last one processed
    if new_file_info != st.session_state.get("uploaded_file_info"):
        logger.info(f"New file uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
        st.toast(f"Processing '{uploaded_file.name}'...", icon="⏳")

        # --- Reset Application State for New File ---
        logger.debug("Resetting application state for new file upload.")
        # Store keys to preserve (like widget keys, maybe session_id if desired)
        keys_to_preserve = {"file_uploader_widget", "session_id", "uploaded_file_info", "demo_loaded"}
        # Preserve session ID or generate new one? Let's generate a new one for new file.
        st.session_state.session_id = str(uuid.uuid4())[:8]

        for key, value in DEFAULT_STATE.items():
            if key not in keys_to_preserve:
                st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value

        st.session_state.uploaded_file_info = new_file_info # Store info of the current file
        st.session_state.demo_loaded = False # Ensure demo mode is off if a file is uploaded

        # --- Process the Uploaded File ---
        st.session_state.raw_image_bytes = uploaded_file.getvalue()
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()

        # Determine if DICOM based on extension or MIME type (if available)
        st.session_state.is_dicom = (
            PYDICOM_AVAILABLE and DICOM_UTILS_AVAILABLE and
            ("dicom" in uploaded_file.type.lower() or file_ext in (".dcm", ".dicom"))
        )

        with st.spinner("🔬 Analyzing file structure and loading image..."):
            temp_display_img = None
            temp_processed_img = None
            processing_success = False

            if st.session_state.is_dicom:
                logger.info("Processing uploaded file as DICOM...")
                try:
                    # Parse DICOM using utility function
                    dicom_dataset = parse_dicom(st.session_state.raw_image_bytes, filename=uploaded_file.name)
                    st.session_state.dicom_dataset = dicom_dataset

                    if dicom_dataset:
                        # Extract metadata
                        st.session_state.dicom_metadata = extract_dicom_metadata(dicom_dataset)

                        # Get default window/level settings
                        default_wc, default_ww = get_default_wl(dicom_dataset)
                        st.session_state.current_display_wc = default_wc
                        st.session_state.current_display_ww = default_ww
                        logger.info(f"DICOM default W/L: WC={default_wc}, WW={default_ww}")

                        # Convert DICOM to PIL Image for display (with default W/L)
                        temp_display_img = dicom_to_image(dicom_dataset, wc=default_wc, ww=default_ww)

                        # Convert DICOM to PIL Image for processing (potentially normalized, full range)
                        # Adjust normalization as needed by AI models
                        temp_processed_img = dicom_to_image(dicom_dataset, wc=None, ww=None, normalize=True)

                        # Check if images were created successfully
                        if isinstance(temp_display_img, Image.Image) and isinstance(temp_processed_img, Image.Image):
                            processing_success = True
                            logger.info("DICOM parsed and converted to images successfully.")
                        else:
                            st.error("Failed to convert DICOM pixel data to an image.")
                            logger.error("dicom_to_image did not return valid PIL Images.")

                    else:
                        st.error("Could not parse the DICOM file structure.")
                        logger.error("parse_dicom returned None.")

                except pydicom.errors.InvalidDicomError:
                    st.error("Invalid DICOM file format. Please upload a valid DICOM file.")
                    logger.error("InvalidDicomError during parsing.")
                    st.session_state.is_dicom = False # Treat as non-DICOM if parsing fails badly
                except Exception as e:
                    st.error(f"An error occurred during DICOM processing: {e}")
                    logger.error(f"Error processing DICOM file '{uploaded_file.name}': {e}", exc_info=True)
                    st.session_state.is_dicom = False # Fallback

            # Process as standard image (if not DICOM or DICOM processing failed)
            if not st.session_state.is_dicom and not processing_success:
                logger.info("Processing uploaded file as standard image (JPG/PNG)...")
                if not PIL_AVAILABLE:
                    st.error("Cannot process standard images: Pillow (PIL) library is missing.")
                else:
                    try:
                        # Open image using Pillow
                        raw_img = Image.open(io.BytesIO(st.session_state.raw_image_bytes))
                        # Ensure image is in RGB format for consistency
                        processed_img = raw_img.convert("RGB")

                        temp_display_img = processed_img.copy()
                        temp_processed_img = processed_img.copy()
                        processing_success = True
                        logger.info("Standard image loaded and converted to RGB successfully.")

                    except UnidentifiedImageError:
                        st.error("Could not identify image format. Please upload a valid JPG, PNG, or DICOM file.")
                        logger.error(f"UnidentifiedImageError for file: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"An error occurred processing the image: {e}")
                        logger.error(f"Error processing standard image '{uploaded_file.name}': {e}", exc_info=True)

            # --- Finalize Processing ---
            if processing_success and isinstance(temp_display_img, Image.Image) and isinstance(temp_processed_img, Image.Image):
                # Ensure display image is RGB
                if temp_display_img.mode != 'RGB':
                    st.session_state.display_image = temp_display_img.convert('RGB')
                else:
                    st.session_state.display_image = temp_display_img

                st.session_state.processed_image = temp_processed_img
                st.success(f"✅ Image '{uploaded_file.name}' loaded and processed successfully!")
                logger.info(f"Successfully processed and loaded image: {uploaded_file.name}")
                st.rerun() # Rerun to update the UI with the new image
            else:
                # Clear state if processing failed
                st.error("Image loading failed. Please check the file format or try another file.")
                logger.error(f"Image processing failed for file: {uploaded_file.name}")
                st.session_state.uploaded_file_info = None # Reset file info so user can try again
                # Clear image states
                st.session_state.display_image = None
                st.session_state.processed_image = None
                st.session_state.is_dicom = False


# --- Main Page Layout ---
st.markdown("---") # Separator

# Page Title & Usage Guide
st.title("⚕️ RadVision QA Advanced: AI-Assisted Image Analysis")
with st.expander("User Guide & Disclaimer", expanded=False):
    st.warning("⚠️ **Disclaimer:** This tool is intended for research, educational, and informational purposes only. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider regarding any medical conditions or concerns. AI outputs may contain errors and must be critically evaluated by a qualified professional before being considered in any clinical context.")
    st.markdown(
        """
        **Workflow:**
        1.  **Upload Image:** Use the sidebar to upload a JPG, PNG, or DICOM file, or activate Demo Mode.
        2.  **(DICOM)** If a DICOM image is loaded, adjust Window/Level sliders in the sidebar for optimal visualization.
        3.  **(Optional) Select ROI:** Draw a rectangle on the image in the 'Image Viewer' to focus the AI's analysis on a specific Region of Interest. Clear the ROI using the sidebar button if needed.
        4.  **Run Analysis:** Use the 'AI Analysis Actions' in the sidebar:
            *   `Run Initial Analysis`: Get a general overview.
            *   `Ask Question`: Pose specific questions about the image/ROI.
            *   `Analyze Condition`: Focus the AI on signs of a selected condition.
        5.  **Review Results:** Check the tabs in the right panel ('Analysis & Results') for AI outputs.
        6.  **Translate (Optional):** Use the 'Translation' tab to translate analysis text into different languages.
        7.  **Estimate Confidence:** Use the 'Estimate AI Confidence' button (sidebar) after performing analysis.
        8.  **Generate Report:** Use the 'Generate PDF Report Data' button (sidebar) and then download the report.
        """
    )
st.markdown("---") # Separator

# --- Main Content Area: Two Columns ---
col1, col2 = st.columns([2, 3]) # Adjust ratio as needed (e.g., 1:1, 2:3)

# --- Column 1: Image Viewer, ROI, Metadata ---
with col1:
    st.subheader("🖼️ Image Viewer")
    display_img = st.session_state.get("display_image")

    if isinstance(display_img, Image.Image):
        # --- ROI Drawing Canvas ---
        if DRAWABLE_CANVAS_AVAILABLE and st_canvas:
            st.caption("Draw a rectangle on the image below to select a Region of Interest (ROI).")
            # Calculate canvas dimensions dynamically based on image aspect ratio
            MAX_CANVAS_WIDTH = 600 # Max width for the canvas column
            MAX_CANVAS_HEIGHT = 500 # Max height
            img_w, img_h = display_img.size

            if img_w <= 0 or img_h <= 0:
                st.warning("Image has invalid dimensions (0 or negative). Cannot draw ROI.")
                logger.warning("Cannot draw ROI: Image has invalid dimensions.")
            else:
                aspect_ratio = img_w / img_h
                # Calculate width first, constrained by max width
                canvas_width = min(img_w, MAX_CANVAS_WIDTH)
                # Calculate corresponding height
                canvas_height = int(canvas_width / aspect_ratio)

                # If calculated height exceeds max height, recalculate width based on max height
                if canvas_height > MAX_CANVAS_HEIGHT:
                    canvas_height = MAX_CANVAS_HEIGHT
                    canvas_width = int(canvas_height * aspect_ratio)

                # Ensure minimum dimensions
                canvas_width = max(canvas_width, 150)
                canvas_height = max(canvas_height, 150)

                logger.debug(f"Canvas dimensions set to: {canvas_width}x{canvas_height} for image {img_w}x{img_h}")

                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.2)",  # Semi-transparent orange fill
                    stroke_width=2,                      # Border width of the rectangle
                    stroke_color="rgba(239, 83, 80, 0.8)", # Reddish border
                    background_image=display_img,        # The image to draw on
                    update_streamlit=True,               # Send updates back to Streamlit state
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="rect",                 # Allow drawing rectangles
                    initial_drawing=st.session_state.get("canvas_drawing", None), # Restore previous drawing state if available
                    key="drawable_canvas"                # Unique key for the canvas widget
                )

                # Process canvas results to update ROI coordinates if a new rectangle was drawn
                if canvas_result.json_data is not None and canvas_result.json_data.get("objects"):
                    # Assuming the last drawn object is the intended ROI
                    # Note: For multi-object drawings, logic might need adjustment
                    last_object = canvas_result.json_data["objects"][-1]
                    if last_object["type"] == "rect":
                        # Extract coordinates and dimensions from the canvas object
                        # These are relative to the canvas size
                        canvas_left = int(last_object["left"])
                        canvas_top = int(last_object["top"])
                        canvas_width_scaled = int(last_object["width"] * last_object.get("scaleX", 1))
                        canvas_height_scaled = int(last_object["height"] * last_object.get("scaleY", 1))

                        # Scale canvas coordinates back to original image dimensions
                        scale_x = img_w / canvas_width
                        scale_y = img_h / canvas_height

                        original_left = int(canvas_left * scale_x)
                        original_top = int(canvas_top * scale_y)
                        original_width = int(canvas_width_scaled * scale_x)
                        original_height = int(canvas_height_scaled * scale_y)

                        # Ensure coordinates are within image bounds
                        original_left = max(0, original_left)
                        original_top = max(0, original_top)
                        original_width = min(img_w - original_left, original_width)
                        original_height = min(img_h - original_top, original_height)

                        # Store the calculated ROI in session state if it has changed
                        new_roi = {
                            "left": original_left,
                            "top": original_top,
                            "width": original_width,
                            "height": original_height
                        }

                        # Update state and rerun only if ROI actually changed
                        if st.session_state.roi_coords != new_roi:
                            st.session_state.roi_coords = new_roi
                            st.session_state.canvas_drawing = canvas_result.json_data # Save canvas state
                            logger.info(f"New ROI selected: {new_roi}")
                            st.info(f"ROI selected: ({original_left},{original_top}) size {original_width}x{original_height}", icon="🎯")
                            # No automatic rerun here - let user trigger actions with the new ROI
        elif not DRAWABLE_CANVAS_AVAILABLE:
             st.warning("Drawing canvas library not available. ROI selection is disabled.")
             st.image(display_img, caption="Image Preview (ROI disabled)", use_container_width=True)
        else: # Canvas available but st_canvas is None (shouldn't happen with current checks)
            st.error("Internal Error: Canvas component failed to load.")


        # Display current ROI info if selected
        if st.session_state.roi_coords:
            roi = st.session_state.roi_coords
            st.caption(f"Current ROI: ({roi['left']},{roi['top']}) - {roi['width']}x{roi['height']}px")

        st.markdown("---") # Separator

        # --- DICOM Metadata Display ---
        if st.session_state.is_dicom and st.session_state.dicom_metadata:
            with st.expander("📄 DICOM Metadata", expanded=False):
                if UI_COMPONENTS_AVAILABLE:
                     # Use the custom display function if available
                     display_dicom_metadata(st.session_state.dicom_metadata)
                else:
                    # Fallback to simple JSON display
                    st.json(st.session_state.dicom_metadata, expanded=False)
        elif st.session_state.is_dicom:
            st.caption("DICOM file loaded, but no metadata extracted or available.")

    elif uploaded_file is not None:
        # Show if file uploaded but image failed to load
        st.error("Image preview failed. The uploaded file might be corrupted or in an unsupported format.")
    else:
        # Default message when no image is loaded
        st.info("⬅️ Please upload an image or activate Demo Mode using the sidebar.")

# --- Column 2: Analysis Results Tabs ---
with col2:
    st.subheader("📊 Analysis & Results")

    # Define tab titles
    tab_titles = [
        "🔬 Initial Analysis",
        "💬 Q&A History",
        "🩺 Condition Focus",
        "📈 Confidence",
        "🌐 Translation"
    ]
    # Create tabs
    tabs = st.tabs(tab_titles)

    # --- Tab 0: Initial Analysis ---
    with tabs[0]:
        st.text_area(
            "Overall Findings & Impressions",
            value=st.session_state.initial_analysis or "Run 'Initial Analysis' from the sidebar to generate findings.",
            height=450, # Adjust height as needed
            key="output_initial",
            disabled=True, # Display only
            help="Displays the AI's general analysis of the image or ROI."
        )

    # --- Tab 1: Q&A History ---
    with tabs[1]:
        st.text_area(
            "Latest AI Answer",
            value=st.session_state.qa_answer or "Ask a question using the sidebar to get an AI response.",
            height=200, # Adjust height
            key="output_qa",
            disabled=True,
            help="Displays the AI's answer to your most recent question."
        )
        st.markdown("---")
        if st.session_state.history:
            # Display conversation history in an expander
            with st.expander("Full Conversation History", expanded=True):
                for i, (q_type, message) in enumerate(reversed(st.session_state.history)): # Show newest first
                    if q_type.lower() == "user question":
                        st.markdown(f"**You:** {message}")
                    elif q_type.lower() == "ai answer":
                         st.markdown(f"**AI:**")
                         st.markdown(f"{message}", unsafe_allow_html=True) # Allow basic markdown in answers
                    elif q_type.lower() == "system":
                         st.info(f"*{message}*", icon="ℹ️")
                    else: # Fallback for unexpected types
                         st.markdown(f"**{q_type}:** {message}")

                    if i < len(st.session_state.history) - 1:
                        st.markdown("---") # Separator between Q&A pairs
        else:
            st.caption("No questions asked yet in this session.")

    # --- Tab 2: Disease Focus ---
    with tabs[2]:
        st.text_area(
            "Condition-Specific Analysis",
            value=st.session_state.disease_analysis or "Select a condition and run 'Analyze Condition' from the sidebar.",
            height=450, # Adjust height
            key="output_disease",
            disabled=True,
            help="Displays the AI's analysis focused on the selected medical condition."
        )

    # --- Tab 3: Confidence ---
    with tabs[3]:
        st.text_area(
            "Estimated AI Confidence",
            value=st.session_state.confidence_score or "Run 'Estimate AI Confidence' from the sidebar after performing analysis.",
            height=450, # Adjust height
            key="output_confidence",
            disabled=True,
            help="Displays the AI's estimated confidence level based on the performed analysis."
        )

    # --- Tab 4: Translation ---
    with tabs[4]:
        st.subheader("🌐 Translate Analysis Text")

        if not TRANSLATION_AVAILABLE:
            st.warning("Translation features are currently unavailable. Please ensure the 'deep-translator' library is installed and functional.")
        else:
            st.caption("Select the analysis text you want to translate, choose the target language, and optionally specify the source language (or use Auto-Detect).")

            # Options for text selection
            text_options = {
                "Initial Analysis": st.session_state.initial_analysis,
                "Latest Q&A Answer": st.session_state.qa_answer,
                "Condition Analysis": st.session_state.disease_analysis,
                "Confidence Estimation": st.session_state.confidence_score,
                "(Enter Custom Text Below)": "" # Placeholder for custom input
            }
            # Filter out options with empty content to avoid clutter
            available_options = {label: text for label, text in text_options.items() if text or label == "(Enter Custom Text Below)"}

            selected_label = st.selectbox(
                "Select text to translate:",
                options=list(available_options.keys()),
                index=0,
                key="translate_text_select"
            )

            text_to_translate = available_options.get(selected_label, "")

            # Show custom text area only if "(Custom Text)" is selected
            if selected_label == "(Enter Custom Text Below)":
                text_to_translate = st.text_area(
                    "Enter text to translate here:",
                    value="", # Start empty
                    height=150,
                    key="custom_translate_input"
                )

            # Display the selected/entered text for confirmation (read-only)
            st.text_area("Text selected for translation:", value=text_to_translate, height=100, disabled=True, key="translate_source_preview")

            # Language selection dropdowns
            col_lang1, col_lang2 = st.columns(2)
            with col_lang1:
                # Source language: Include Auto-Detect option
                # Use the constant imported from translation_models
                source_language_options = [AUTO_DETECT_INDICATOR] + sorted(list(LANGUAGE_CODES.keys()))
                source_language_name = st.selectbox(
                    "Source Language:",
                    options=source_language_options,
                    index=0, # Default to Auto-Detect
                    key="translate_source_lang",
                    help="Select the original language of the text, or 'Auto-Detect'."
                )
            with col_lang2:
                # Target language
                target_language_options = sorted(list(LANGUAGE_CODES.keys()))
                 # Try to default to Spanish or English if available
                default_target_index = 0
                if "Spanish" in target_language_options:
                    default_target_index = target_language_options.index("Spanish")
                elif "English" in target_language_options:
                     default_target_index = target_language_options.index("English")

                target_language_name = st.selectbox(
                    "Translate To:",
                    options=target_language_options,
                    index=default_target_index,
                    key="translate_target_lang",
                    help="Select the language you want to translate the text into."
                )

            # Translate button
            if st.button("🔄 Translate Now", key="translate_button"):
                # Clear previous results/errors
                st.session_state.translation_result = None
                st.session_state.translation_error = None

                if not text_to_translate or not text_to_translate.strip():
                    st.warning("Please select or enter some text to translate.")
                    st.session_state.translation_error = "Input text is empty."
                elif source_language_name == target_language_name and source_language_name != AUTO_DETECT_INDICATOR:
                     st.info("Source and target languages are the same. No translation needed.")
                     st.session_state.translation_result = text_to_translate # Show original text as result
                else:
                    with st.spinner(f"Translating from '{source_language_name}' to '{target_language_name}'..."):
                        try:
                            # Call the updated translate function from translation_models
                            # It now returns Optional[str]
                            logger.info(f"Calling translate: text_len={len(text_to_translate)}, src='{source_language_name}', tgt='{target_language_name}'")
                            translation_output = translate(
                                text=text_to_translate,
                                target_language=target_language_name,
                                source_language=source_language_name # Pass name directly, module handles 'Auto-Detect'
                            )

                            # --- Handle the Optional[str] result ---
                            if translation_output is not None:
                                st.session_state.translation_result = translation_output
                                logger.info(f"Translation successful. Result length: {len(translation_output)}")
                                st.success("Translation complete!")
                            else:
                                # Translation failed, translate() returned None
                                st.error("Translation failed. The translation service might be unavailable, the language pair unsupported, or the input invalid. Please check the application logs for more details.")
                                logger.error(f"Translation function returned None for src='{source_language_name}', tgt='{target_language_name}'. Check previous logs.")
                                st.session_state.translation_error = "Translation service failed or returned no result."

                        except Exception as e:
                            # Catch unexpected errors during the call
                            st.error(f"An unexpected error occurred during translation: {e}")
                            logger.critical(f"Unexpected error calling translate function: {e}", exc_info=True)
                            st.session_state.translation_error = f"Unexpected error: {e}"

            # Display translation result or error message
            if st.session_state.get("translation_result") is not None:
                formatted_result = format_translation(st.session_state.translation_result)
                st.text_area("Translated Text:", value=formatted_result, height=200, key="translate_output_area")
            elif st.session_state.get("translation_error"):
                # Display error message if translation failed and result is None
                st.info(f"Translation Error: {st.session_state.translation_error}")


# --- Backend Action Handling Logic ---
# This block runs when a button sets st.session_state.last_action and st.rerun() is called

current_action: Optional[str] = st.session_state.get("last_action")

if current_action:
    logger.info(f"Handling action triggered: '{current_action}' for session: {st.session_state.session_id}")

    # --- Pre-Action Checks ---
    action_requires_image = current_action not in ["generate_report_data"] # List actions not needing an image
    action_requires_llm = current_action in ["analyze", "ask", "disease", "confidence"]
    action_requires_report_util = current_action == "generate_report_data"
    action_requires_hf_fallback = current_action == "ask" # Only needed if primary fails

    if action_requires_image and not isinstance(st.session_state.get("processed_image"), Image.Image):
        st.error(f"Cannot perform '{current_action}': No valid image has been processed. Please upload an image.")
        logger.error(f"Action '{current_action}' aborted: 'processed_image' is not a valid PIL Image.")
        st.session_state.last_action = None # Reset action
        st.stop() # Stop further execution in this run

    if not st.session_state.session_id:
        st.error(f"Cannot perform '{current_action}': Critical error - Session ID is missing.")
        logger.critical(f"Action '{current_action}' aborted: Session ID is None.")
        st.session_state.last_action = None
        st.stop()

    if action_requires_llm and not LLM_INTERACTIONS_AVAILABLE:
         st.error(f"Cannot perform '{current_action}': Core AI interaction module is unavailable.")
         logger.critical(f"Action '{current_action}' aborted: LLM_INTERACTIONS_AVAILABLE is False.")
         st.session_state.last_action = None
         st.stop()

    if action_requires_report_util and not REPORT_UTILS_AVAILABLE:
         st.error(f"Cannot perform '{current_action}': Report generation module is unavailable.")
         logger.critical(f"Action '{current_action}' aborted: REPORT_UTILS_AVAILABLE is False.")
         st.session_state.last_action = None
         st.stop()


    # --- Execute Action ---
    img_for_llm = st.session_state.processed_image # Use the preprocessed image for AI
    roi_coords = st.session_state.roi_coords # Get current ROI
    roi_info_str = " (ROI applied)" if roi_coords else "" # For logging/display
    current_history = st.session_state.history # Get current history list

    # Ensure history is always a list (belt-and-suspenders check)
    if not isinstance(current_history, list):
        logger.warning("Session state history was not a list, resetting to empty list.")
        current_history = []
        st.session_state.history = current_history

    try:
        # --- Initial Analysis Action ---
        if current_action == "analyze":
            st.info(f"🔬 Performing initial analysis{roi_info_str}...")
            with st.spinner("AI is analyzing the image... Please wait."):
                analysis_result = run_initial_analysis(img_for_llm, roi=roi_coords) # Pass ROI if available
            st.session_state.initial_analysis = analysis_result
            # Clear other potentially conflicting results
            st.session_state.qa_answer = ""
            st.session_state.disease_analysis = ""
            # st.session_state.confidence_score = "" # Optionally clear confidence too
            logger.info(f"Initial analysis completed successfully{roi_info_str}.")
            st.success("Initial analysis complete!")

        # --- Ask Question Action ---
        elif current_action == "ask":
            question_text = st.session_state.question_input_widget.strip()
            if not question_text:
                st.warning("Question field is empty. Please enter a question.")
                logger.warning("Ask action triggered with empty question input.")
            else:
                st.info(f"❓ Asking AI: '{question_text}'{roi_info_str}...")
                st.session_state.qa_answer = "" # Clear previous answer display
                with st.spinner("Thinking... AI is processing your question."):
                    # Run primary Q&A model
                    answer, success_flag = run_multimodal_qa(
                        img_for_llm,
                        question_text,
                        current_history,
                        roi=roi_coords
                    )
                if success_flag:
                    st.session_state.qa_answer = answer
                    # Append to history correctly
                    st.session_state.history.append(("User Question", question_text))
                    st.session_state.history.append(("AI Answer", answer))
                    logger.info(f"Primary Q&A successful for question: '{question_text}'{roi_info_str}")
                    st.success("AI answered your question!")
                else:
                    # --- Handle Primary AI Failure ---
                    primary_error_msg = f"Primary AI failed to answer. Reason: {answer}"
                    st.session_state.qa_answer = primary_error_msg # Show error in UI
                    st.error(primary_error_msg)
                    logger.warning(f"Primary AI Q&A failed for question '{question_text}'. Reason: {answer}")
                    # Attempt Fallback if available and configured
                    hf_token = os.environ.get("HF_API_TOKEN")
                    if HF_MODELS_AVAILABLE and hf_token:
                        st.info(f"Attempting fallback using Hugging Face model ({HF_VQA_MODEL_ID})...")
                        logger.info(f"Attempting HF fallback VQA ({HF_VQA_MODEL_ID}).")
                        with st.spinner(f"Trying fallback AI ({HF_VQA_MODEL_ID})..."):
                            fallback_answer, fallback_success = query_hf_vqa_inference_api(
                                img_for_llm, question_text, roi=roi_coords
                            )
                        if fallback_success:
                            fallback_display = f"**[Fallback: {HF_VQA_MODEL_ID}]**\n\n{fallback_answer}"
                            st.session_state.qa_answer += "\n\n" + fallback_display # Append fallback answer
                            # Append fallback Q&A to history distinctly
                            st.session_state.history.append(("[Fallback] User Question", question_text))
                            st.session_state.history.append(("[Fallback] AI Answer", fallback_display))
                            logger.info("HF fallback VQA successful.")
                            st.success("Fallback AI provided an answer.")
                        else:
                            fallback_error_msg = f"**[Fallback Error - {HF_VQA_MODEL_ID}]:** {fallback_answer}"
                            st.session_state.qa_answer += f"\n\n{fallback_error_msg}"
                            logger.error(f"HF fallback VQA failed. Reason: {fallback_answer}")
                            st.error("Fallback AI also failed to provide an answer.")
                    elif HF_MODELS_AVAILABLE and not hf_token:
                         missing_token_msg = "\n\n**[Fallback Skipped]:** Hugging Face API token (HF_API_TOKEN) not configured."
                         st.session_state.qa_answer += missing_token_msg
                         logger.warning("HF fallback VQA skipped: HF_API_TOKEN environment variable not set.")
                         st.warning("Fallback AI requires configuration (API Token).")
                    else:
                         unavailable_msg = "\n\n**[Fallback Unavailable]:** No fallback model configured or available."
                         st.session_state.qa_answer += unavailable_msg
                         logger.warning("HF fallback VQA skipped: hf_models module not available.")

        # --- Disease Analysis Action ---
        elif current_action == "disease":
            selected_disease = st.session_state.disease_select_widget
            if not selected_disease:
                st.warning("No condition selected. Please choose a condition from the dropdown.")
                logger.warning("Disease analysis action triggered with no condition selected.")
            else:
                st.info(f"🩺 Running focused analysis for '{selected_disease}'{roi_info_str}...")
                with st.spinner(f"AI is analyzing for signs of '{selected_disease}'..."):
                    disease_result = run_disease_analysis(img_for_llm, selected_disease, roi=roi_coords)
                st.session_state.disease_analysis = disease_result
                # Clear other potentially conflicting results
                st.session_state.qa_answer = ""
                # st.session_state.initial_analysis = "" # Decide if initial should be cleared
                # st.session_state.confidence_score = "" # Optionally clear confidence
                logger.info(f"Disease-specific analysis completed for '{selected_disease}'{roi_info_str}.")
                st.success(f"Analysis for '{selected_disease}' complete!")

        # --- Confidence Estimation Action ---
        elif current_action == "confidence":
            # Check again if there's context (redundant with sidebar check, but safe)
            if not (current_history or st.session_state.initial_analysis or st.session_state.disease_analysis):
                st.warning("Cannot estimate confidence without prior analysis or Q&A.")
                logger.warning("Confidence estimation action skipped: No analysis context available.")
            else:
                st.info(f"📊 Estimating AI confidence based on current context{roi_info_str}...")
                with st.spinner("Calculating confidence score..."):
                    # Pass relevant context to the confidence estimation function
                    confidence_result = estimate_ai_confidence(
                        img_for_llm, # Image might be needed for context
                        history=current_history,
                        initial_analysis=st.session_state.initial_analysis,
                        disease_analysis=st.session_state.disease_analysis,
                        roi=roi_coords
                    )
                st.session_state.confidence_score = confidence_result
                logger.info(f"Confidence estimation completed{roi_info_str}.")
                st.success("Confidence estimation complete!")

        # --- Generate Report Data Action ---
        elif current_action == "generate_report_data":
            st.info("📄 Preparing data for PDF report generation...")
            st.session_state.pdf_report_bytes = None # Clear previous report data

            # Use the display image for the report (might have W/L applied)
            image_for_report = st.session_state.get("display_image")

            if not isinstance(image_for_report, Image.Image):
                st.error("Cannot generate report: The image to include is invalid or missing.")
                logger.error("PDF report generation aborted: 'display_image' is not a valid PIL Image.")
            else:
                final_image_for_pdf = image_for_report.copy().convert("RGB") # Ensure RGB copy

                # Draw ROI on the image if selected
                if roi_coords:
                    try:
                        draw = ImageDraw.Draw(final_image_for_pdf)
                        x0, y0 = roi_coords['left'], roi_coords['top']
                        x1, y1 = x0 + roi_coords['width'], y0 + roi_coords['height']
                        # Draw a noticeable rectangle (adjust color/width)
                        draw.rectangle([x0, y0, x1, y1], outline="red", width=max(3, int(min(final_image_for_pdf.size)*0.005))) # Scale width slightly
                        logger.info("ROI bounding box drawn on image for PDF report.")
                    except Exception as e:
                        logger.error(f"Error drawing ROI rectangle on image for PDF: {e}", exc_info=True)
                        st.warning("Could not draw ROI on the report image, proceeding without it.")

                # Compile all text outputs for the report
                # Format history for readability
                formatted_history = "No Q&A history available."
                if current_history:
                    history_lines = []
                    for q_type, msg in current_history:
                         # Clean up potential markdown/html for plain text report section
                         cleaned_msg = re.sub('<[^<]+?>', '', str(msg)) # Basic HTML tag removal
                         history_lines.append(f"{q_type}: {cleaned_msg}")
                    formatted_history = "\n\n".join(history_lines)


                report_data = {
                    "Session ID": st.session_state.session_id,
                    "Image Filename": st.session_state.uploaded_file_info.split('-')[0] if st.session_state.uploaded_file_info else "N/A",
                    "Initial Analysis": st.session_state.initial_analysis or "Not Performed",
                    "Conversation History": formatted_history,
                    "Condition Analysis": st.session_state.disease_analysis or "Not Performed",
                    "AI Confidence Estimation": st.session_state.confidence_score or "Not Performed",
                }

                # Add DICOM metadata summary if available
                if st.session_state.is_dicom and st.session_state.dicom_metadata:
                    # Include a few key fields or a summary string
                    meta_summary = {k: v for k, v in st.session_state.dicom_metadata.items() if k in ['Patient Name', 'Patient ID', 'Study Date', 'Modality', 'Study Description']}
                    report_data["DICOM Summary"] = "\n".join([f"{k}: {v}" for k,v in meta_summary.items()]) if meta_summary else "Basic metadata available."


                # Generate PDF bytes using the utility function
                with st.spinner("Generating PDF document... This may take a moment."):
                    pdf_bytes = generate_pdf_report_bytes(
                        session_id=st.session_state.session_id,
                        image=final_image_for_pdf,
                        analysis_outputs=report_data,
                        # Pass metadata separately if needed by the report function
                        dicom_metadata=st.session_state.dicom_metadata if st.session_state.is_dicom else None
                    )

                if pdf_bytes:
                    st.session_state.pdf_report_bytes = pdf_bytes
                    st.success("PDF report data generated successfully! Click 'Download PDF Report' in the sidebar.")
                    logger.info("PDF report generation successful.")
                    st.balloons() # Fun confirmation
                else:
                    st.error("Failed to generate the PDF report. An internal error occurred.")
                    logger.error("PDF generation function (generate_pdf_report_bytes) returned None or empty bytes.")

        # --- Unknown Action ---
        else:
            st.warning(f"An unknown action '{current_action}' was triggered.")
            logger.warning(f"Attempted to handle unknown action: '{current_action}'")

    # --- Error Handling for Actions ---
    except Exception as e:
        # Catch any unexpected errors during the action execution
        st.error(f"An unexpected error occurred while performing '{current_action}': {e}")
        logger.critical(f"Unexpected error during action '{current_action}' execution: {e}", exc_info=True)
        # Optionally reset parts of the state if the error is severe

    # --- Post-Action Cleanup ---
    finally:
        # ALWAYS reset the last_action flag after handling it
        st.session_state.last_action = None
        logger.debug(f"Action '{current_action}' handling complete. Resetting last_action.")
        # Rerun to update the UI based on state changes made during the action
        st.rerun()


# --- Footer ---
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced | Session ID: {st.session_state.get('session_id', 'N/A')}")
# Basic footer example
st.markdown(
    """
    <footer style="text-align: center; padding-top: 1rem; font-size: 0.8em; color: #6c757d;">
      <p>RadVision AI is for informational and research purposes only. Not a substitute for professional medical evaluation.</p>
      <p><a href="#" target="_blank">Privacy Policy</a> | <a href="#" target="_blank">Terms of Service</a></p>
      <!-- <p>Consider adding version number here -->
    </footer>
    """, unsafe_allow_html=True
)

logger.info(f"--- Application render complete for session: {st.session_state.session_id} ---")