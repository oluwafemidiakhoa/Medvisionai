# app.py

import streamlit as st

# Ensure this is the very first command
st.set_page_config(
    page_title="RadVision AI Advanced",
    layout="wide",
    page_icon="⚕️",
    initial_sidebar_state="expanded"
)

# --- Core Libraries ---
import io
import os
import uuid
import logging
import base64
from typing import Any, Dict, Optional, Tuple, List
import copy
import random  # For Tip of the Day
import re      # For formatting the translation

# --- Drawable Canvas ---
try:
    from streamlit_drawable_canvas import st_canvas
    import streamlit_drawable_canvas as st_canvas_module
    CANVAS_VERSION = getattr(st_canvas_module, '__version__', 'Unknown') # Use __version__
except ImportError:
    st.error("CRITICAL ERROR: streamlit-drawable-canvas is not installed. Run `pip install streamlit-drawable-canvas`.")
    st.stop()

# --- Custom CSS for a Polished Look & Tab Scrolling ---
st.markdown(
    """
    <style>
      body {
          font-family: 'Helvetica', sans-serif;
          background-color: #f9f9f9;
      }
      .stApp { /* Target the main Streamlit app container */
           background-color: #f9f9f9; /* Or your desired background */
      }
      .css-1d391kg { background-color: #ffffff; } /* May need adjustment based on Streamlit version */
      footer {
          text-align: center;
          font-size: 0.8em;
          color: #888888;
          margin-top: 2em; /* Add space above footer */
      }
      /* Ensure tab scrolling on overflow */
      div[role="tablist"] {
          overflow-x: auto;
          white-space: nowrap;
          /* Add some padding for better visual */
          padding-bottom: 10px;
          border-bottom: 1px solid #e0e0e0; /* Optional: add border */
      }
       /* Style for individual tabs to prevent wrapping */
      div[role="tablist"] button {
          white-space: nowrap;
      }
    </style>
    """, unsafe_allow_html=True
)

# --- Display Hero Logo ---
logo_path = os.path.join("assets", "radvisionai-hero.jpeg")
if os.path.exists(logo_path):
    st.image(logo_path, width=400)
else:
    st.warning("Hero logo (assets/radvisionai-hero.jpeg) not found.")

# --- Tip of the Day in Sidebar ---
TIPS = [
    "Tip: Use Demo Mode to quickly see how the analysis works.",
    "Tip: Draw an ROI rectangle on the image viewer to focus the AI.",
    "Tip: Adjust DICOM window/level sliders for optimal image contrast.",
    "Tip: Review Q&A History tab to see the full conversation.",
    "Tip: Generate a PDF report to save your analysis session.",
    "Tip: Use the Translation tab to view results in other languages."
]
st.sidebar.info(random.choice(TIPS))

# --- Image & DICOM Processing ---
try:
    from PIL import Image, ImageDraw, UnidentifiedImageError
    import PIL
    PIL_VERSION = getattr(PIL, '__version__', 'Unknown') # Use __version__
except ImportError:
    st.error("CRITICAL ERROR: Pillow (PIL) is not installed. Run `pip install Pillow`.")
    st.stop()

try:
    import pydicom
    import pydicom.errors
    PYDICOM_VERSION = getattr(pydicom, '__version__', 'Unknown') # Use __version__
except ImportError:
    PYDICOM_VERSION = 'Not Installed'
    pydicom = None

# --- Set Up Logging ---
# Configure logging BEFORE using the logger
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s', # Use %(name)s
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Get the logger for the current module
logger = logging.getLogger(__name__) # Use __name__ for module-specific logger

# Log initial info
logger.info(f"--- App Start ---")
logger.info(f"Logging level set to: {LOG_LEVEL}")
logger.info(f"Streamlit version: {st.__version__}") # Use st.__version__
logger.info(f"Pillow (PIL) version: {PIL_VERSION}")
logger.info(f"streamlit_drawable_canvas version: {CANVAS_VERSION}")
if pydicom:
    logger.info(f"pydicom version: {PYDICOM_VERSION}")
else:
    logger.warning("pydicom not installed. DICOM functionality disabled.")

# Check for optional DICOM handlers
try:
    import pylibjpeg
    logger.info("pylibjpeg found.")
except ImportError:
    logger.info("pylibjpeg not found. Some DICOM compressions may not be supported.")

try:
    import gdcm
    logger.info("python-gdcm found.")
except ImportError:
    logger.info("python-gdcm not found. Some DICOM transfer syntaxes may not be supported.")


# --- Function to Post-Process Translated Text ---
def format_translation(translated_text: str) -> str:
    """Attempts to restore some list formatting that might be lost in translation."""
    if not isinstance(translated_text, str):
        return ""
    # Simple replacements: Finds digits followed by a dot and ensures they start on a new line.
    formatted_text = re.sub(r'\s*(\d+\.)\s*', r'\n\n\1 ', translated_text.strip())
    # Simple replacements: Finds hyphens/asterisks used as bullets
    formatted_text = re.sub(r'\s*([-*]\s+)', r'\n\1', formatted_text)
    return formatted_text

# --- Monkey-Patch st.elements.image.image_to_url if missing ---
# (This might not be needed in recent Streamlit versions, but kept for compatibility)
import streamlit.elements.image as st_image
if not hasattr(st_image, "image_to_url"):
    def image_to_url_monkey_patch(img_obj: Any, *args, **kwargs) -> str:
        if isinstance(img_obj, Image.Image):
            try:
                buffered = io.BytesIO()
                fmt = "PNG"
                temp_img = img_obj
                # Ensure compatibility with different modes before saving
                if img_obj.mode == 'P': # Palette mode often needs RGBA conversion for transparency
                    temp_img = img_obj.convert('RGBA')
                elif img_obj.mode not in ['RGB', 'L', 'RGBA']: # Convert other modes to RGB as a fallback
                    temp_img = img_obj.convert('RGB')

                temp_img.save(buffered, format=fmt)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/{fmt.lower()};base64,{img_str}"
            except Exception as e:
                logger.error(f"Monkey-patch failed during image conversion: {e}", exc_info=True)
                return ""
        else:
            logger.warning(f"Unsupported type for image_to_url monkey-patch: {type(img_obj)}")
            return ""
    st_image.image_to_url = image_to_url_monkey_patch
    logger.info("Applied monkey-patch for st.elements.image.image_to_url (if needed).")


# --- Import Custom Utilities & Fallbacks ---
try:
    # Ensure these utils use logger = logging.getLogger(__name__) for proper naming
    from dicom_utils import parse_dicom, extract_dicom_metadata, dicom_to_image, get_default_wl
    from llm_interactions import run_initial_analysis, run_multimodal_qa, run_disease_analysis, estimate_ai_confidence
    from report_utils import generate_pdf_report_bytes
    from ui_components import display_dicom_metadata, dicom_wl_sliders
    logger.info("Custom utility modules imported successfully.")

    # HF fallback for Q&A (Optional)
    try:
        from hf_models import query_hf_vqa_inference_api, HF_VQA_MODEL_ID
        logger.info(f"HF Fallback model configured: {HF_VQA_MODEL_ID}")
    except ImportError:
        HF_VQA_MODEL_ID = None # Explicitly set to None if not found
        def query_hf_vqa_inference_api(img: Image.Image, question: str, roi: Optional[Dict] = None) -> Tuple[str, bool]:
            logger.info("HF Fallback called but module not found.")
            return "[Fallback Unavailable] HF module not found.", False
        logger.warning("hf_models.py not found. HF VQA fallback disabled.")
    except AttributeError:
        HF_VQA_MODEL_ID = None # Handle case where module exists but variables don't
        def query_hf_vqa_inference_api(img: Image.Image, question: str, roi: Optional[Dict] = None) -> Tuple[str, bool]:
            logger.info("HF Fallback called but required functions/variables not found in module.")
            return "[Fallback Unavailable] HF module incomplete.", False
        logger.warning("HF functions/variables not found in hf_models.py. HF VQA fallback disabled.")

except ImportError as e:
    # Be specific about which module failed
    st.error(f"CRITICAL ERROR importing application modules: {e}. Check file existence and dependencies.")
    logger.critical(f"Failed import: {e}", exc_info=True)
    st.stop()

# --- Import the Translation Module (with detection) ---
# Define default languages here - can be replaced by LANGUAGE_CODES from module
DEFAULT_LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Japanese": "ja",
    "Chinese (Simplified)": "zh-CN", "Russian": "ru", "Arabic": "ar", "Hindi": "hi"
}
LANGUAGE_CODES = None
translate = None
detect_language = None
try:
    # Ensure translation_models uses logger = logging.getLogger(__name__)
    from translation_models import translate, LANGUAGE_CODES, detect_language
    logger.info("Translation module imported successfully.")
    if not isinstance(LANGUAGE_CODES, dict) or not LANGUAGE_CODES:
        LANGUAGE_CODES = DEFAULT_LANGUAGES
        logger.warning("LANGUAGE_CODES not found or invalid in translation_models, using defaults.")
except ImportError as e:
    st.warning(f"Could not import translation_models: {e}. Translation features will be disabled.")
    logger.error("Translation module not found. Translation features disabled.", exc_info=True)
    # Use defaults for UI elements even if backend fails, but disable functionality later
    LANGUAGE_CODES = DEFAULT_LANGUAGES
    translate = None
    detect_language = None
except Exception as e: # Catch other potential errors during import
    st.error(f"An unexpected error occurred importing translation_models: {e}")
    logger.critical(f"Unexpected error importing translation_models: {e}", exc_info=True)
    LANGUAGE_CODES = DEFAULT_LANGUAGES
    translate = None
    detect_language = None


# --- Initialize Session State Defaults ---
# Encapsulate default state definition
def get_default_session_state() -> Dict[str, Any]:
    return {
        "uploaded_file_info": None,
        "raw_image_bytes": None,
        "is_dicom": False,
        "dicom_dataset": None,
        "dicom_metadata": {},
        "processed_image": None, # Image used for AI analysis (potentially normalized)
        "display_image": None,   # Image shown in viewer (can have W/L applied)
        "session_id": None,
        "history": [],           # List of tuples (question, answer)
        "initial_analysis": "",
        "qa_answer": "",         # Last Q&A answer
        "disease_analysis": "",
        "confidence_score": "",
        "last_action": None,     # Track button clicks
        "pdf_report_bytes": None,
        "canvas_drawing": None,  # Store canvas state for ROI
        "roi_coords": None,      # Processed ROI coordinates {left, top, width, height}
        "current_display_wc": None, # DICOM window center
        "current_display_ww": None, # DICOM window width
        "translation_output": "", # Store the last translation result
        "translation_src_lang": "Auto-Detect", # Default source language
        "translation_tgt_lang": "Spanish", # Default target language
        "demo_loaded": False,
        "clear_roi_triggered": False,
    }

# Initialize session state if keys are missing
default_state = get_default_session_state()
for key, value in default_state.items():
    if key not in st.session_state:
        # Use deepcopy for mutable defaults like lists/dicts
        st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value

# Ensure history is always a list (in case state gets corrupted somehow)
if not isinstance(st.session_state.history, list):
    logger.warning("Session state history was not a list, resetting.")
    st.session_state.history = []

logger.debug("Session state initialized/verified.")

# --- Ensure a Session ID Exists ---
if not st.session_state.get("session_id"):
    st.session_state.session_id = str(uuid.uuid4())[:8]
    logger.info(f"New session started: {st.session_state.session_id}")


# --- Additional UI: Clear ROI ---
if st.sidebar.button("🗑️ Clear ROI", help="Clear the current Region of Interest selection"):
    st.session_state.roi_coords = None
    st.session_state.canvas_drawing = None # Reset canvas drawing state too
    st.session_state.clear_roi_triggered = True # Use a dedicated flag
    logger.info("Clear ROI button clicked.")
    st.rerun() # Rerun immediately to clear canvas visually

# Handle ROI clear confirmation after rerun
if st.session_state.get("clear_roi_triggered", False):
    st.success("ROI cleared!")
    st.balloons()
    st.session_state.clear_roi_triggered = False # Reset flag


# --- Demo Mode Logic ---
demo_mode_active = st.sidebar.checkbox("Demo Mode", value=st.session_state.demo_loaded, help="Load a demo image & sample analysis.")

# Check if demo mode checkbox state *changed*
if demo_mode_active != st.session_state.demo_loaded:
    if demo_mode_active: # Checkbox was just checked
        demo_path = os.path.join("assets", "demo.png")
        if os.path.exists(demo_path):
            try:
                demo_img = Image.open(demo_path).convert("RGB")
                # Reset relevant states when demo mode is activated
                logger.info("Activating Demo Mode.")
                default_state = get_default_session_state()
                keys_to_preserve = {"file_uploader_widget", "translation_src_lang", "translation_tgt_lang"}
                for key, value in default_state.items():
                     if key not in keys_to_preserve:
                        st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value

                st.session_state.display_image = demo_img
                st.session_state.processed_image = demo_img
                st.session_state.session_id = "demo-" + str(uuid.uuid4())[:4] # Unique demo session
                st.session_state.history = [("System", "Demo mode activated. Sample analysis loaded.")]
                st.session_state.initial_analysis = "This is a demonstration analysis.\n\nFindings:\n1. Potential area of interest noted in the upper lobe.\n2. No other significant abnormalities detected.\n\nImpression: Suspicious finding requires further investigation."
                st.session_state.demo_loaded = True
                st.success("Demo mode activated! Demo image loaded.")
                logger.info("Demo mode activated successfully.")
                st.rerun() # Rerun to update UI with demo content
            except Exception as e:
                st.sidebar.error(f"Error loading demo image: {e}")
                logger.error(f"Failed to load demo image: {e}", exc_info=True)
                st.session_state.demo_loaded = False # Ensure flag is reset on error
        else:
            st.sidebar.warning("Demo image (assets/demo.png) not found.")
            st.session_state.demo_loaded = False # Keep checkbox state consistent
            # We don't rerun here, just show the warning
    else: # Checkbox was just unchecked
        logger.info("Deactivating Demo Mode.")
        # Reset state back to defaults, preserving widget state
        default_state = get_default_session_state()
        keys_to_preserve = {"file_uploader_widget", "translation_src_lang", "translation_tgt_lang"}
        for key, value in default_state.items():
            if key not in keys_to_preserve:
                 st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value
        st.session_state.demo_loaded = False
        st.session_state.session_id = str(uuid.uuid4())[:8] # New real session ID
        st.info("Demo mode deactivated. Upload an image to begin.")
        st.rerun()


# --- Helper: Convert PIL Image to Data URL Safely ---
# (Used for background image in st_canvas if needed)
def safe_image_to_data_url(img: Image.Image) -> str:
    if not isinstance(img, Image.Image):
        logger.warning(f"Attempted to convert non-PIL Image to data URL: {type(img)}")
        return ""
    try:
        buffered = io.BytesIO()
        fmt = "PNG" # Always use PNG for data URLs for broad compatibility
        temp_img = img

        # Handle different image modes gracefully
        if img.mode == 'P': # Palette mode often needs RGBA for transparency
            temp_img = img.convert('RGBA')
        elif img.mode == 'CMYK' or img.mode == 'I': # Convert complex modes to RGB
             temp_img = img.convert('RGB')
        elif img.mode not in ['RGB', 'L', 'RGBA']: # Fallback for other modes
             temp_img = img.convert('RGB') # Convert to RGB if not already compatible

        temp_img.save(buffered, format=fmt)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{fmt.lower()};base64,{img_str}"
    except Exception as e:
        logger.error(f"Failed converting image to data URL: {e}", exc_info=True)
        return ""


# --- Page Title & Usage Guide ---
st.title("⚕️ RadVision QA Advanced: AI-Assisted Image Analysis")
with st.expander("Usage Guide", expanded=False):
    st.info("⚠️ Disclaimer: This tool is for research/informational purposes only. AI-generated insights should always be verified by a qualified medical professional before making any clinical decisions.")
    st.markdown(
        "**Workflow:**\n"
        "1.  **Upload:** Use the sidebar to upload a JPG, PNG, or DICOM image, or activate Demo Mode.\n"
        "2.  **(Optional) DICOM Adjust:** If DICOM, use sliders (sidebar) to adjust Window Center (WC) / Window Width (WW).\n"
        "3.  **(Optional) ROI:** Draw a rectangle on the 'Image Viewer' (left panel) to define a Region of Interest.\n"
        "4.  **Analyze:** Use the 'AI Actions' in the sidebar (`Run Initial Analysis`, `Ask AI`, `Run Condition Analysis`).\n"
        "5.  **Review:** Check results in the tabs (right panel).\n"
        "6.  **(Optional) Translate:** Use the 'Translation' tab to translate results.\n"
        "7.  **Confidence & Report:** Estimate AI confidence and generate/download a PDF summary."
    )
st.markdown("---")

# --- Sidebar: Upload, DICOM W/L, and AI Actions ---
with st.sidebar:
    st.header("Image Upload & Controls")
    # Disable upload if demo mode is active to avoid confusion
    uploaded_file = st.file_uploader(
        "Upload Image (JPG, PNG, DCM)",
        type=["jpg", "jpeg", "png", "dcm", "dicom"],
        key="file_uploader_widget", # Consistent key
        help="Upload a JPG, PNG, or DICOM file. Disabled in Demo Mode.",
        disabled=st.session_state.demo_loaded # Disable when demo is loaded
    )

    # --- File Processing Logic ---
    if uploaded_file is not None and not st.session_state.demo_loaded:
        # Create a unique identifier for the uploaded file to detect changes
        try:
            # Use modification time if available, otherwise hash content
            file_mtime = getattr(uploaded_file, 'last_modified', None)
            if file_mtime is None:
                import hashlib
                hasher = hashlib.md5()
                file_content = uploaded_file.getvalue() # Read content once
                hasher.update(file_content)
                file_unique_id = hasher.hexdigest()
                uploaded_file.seek(0) # Reset cursor after reading
                raw_bytes = file_content # Use already read content
            else:
                file_unique_id = str(file_mtime)
                raw_bytes = uploaded_file.getvalue() # Read content
                uploaded_file.seek(0) # Always reset cursor

            new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{file_unique_id}"

        except Exception as err:
            logger.error(f"Error generating file info: {err}", exc_info=True)
            new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{str(uuid.uuid4())[:8]}" # Fallback ID
            raw_bytes = uploaded_file.getvalue()
            uploaded_file.seek(0)

        # Check if it's a new file compared to the last one processed
        if new_file_info != st.session_state.get("uploaded_file_info"):
            logger.info(f"Processing new file: {uploaded_file.name} (Size: {uploaded_file.size})")
            st.toast(f"Processing '{uploaded_file.name}'...", icon="⏳")

            # --- State Reset for New File ---
            default_state = get_default_session_state()
            keys_to_preserve = {"file_uploader_widget", "translation_src_lang", "translation_tgt_lang"}
            for key, default_value in default_state.items():
                if key not in keys_to_preserve: # Only reset if key is NOT preserved
                    st.session_state[key] = copy.deepcopy(default_value) if isinstance(default_value, (dict, list)) else default_value
            # --- End State Reset ---

            st.session_state.uploaded_file_info = new_file_info
            st.session_state.session_id = str(uuid.uuid4())[:8] # New session for new file
            st.session_state.raw_image_bytes = raw_bytes
            # demo_loaded should already be false here, but set explicitly just in case
            st.session_state.demo_loaded = False

            # Determine if DICOM
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            # Check pydicom availability before proceeding
            is_dicom_possible = pydicom is not None
            is_dicom_likely = is_dicom_possible and ("dicom" in uploaded_file.type.lower() or file_ext in (".dcm", ".dicom"))

            with st.spinner("🔬 Processing image..."):
                temp_display = None
                temp_processed = None
                is_dicom_confirmed = False
                success = False

                if is_dicom_likely:
                    logger.info(f"Attempting DICOM processing for {uploaded_file.name}")
                    try:
                        ds = parse_dicom(st.session_state.raw_image_bytes, uploaded_file.name)
                        if ds:
                            st.session_state.dicom_dataset = ds
                            st.session_state.dicom_metadata = extract_dicom_metadata(ds)
                            wc, ww = get_default_wl(ds)
                            st.session_state.current_display_wc, st.session_state.current_display_ww = wc, ww
                            temp_display = dicom_to_image(ds, wc, ww) # Image for display with initial W/L
                            # Processed image: Use default W/L or normalized for AI consistency
                            temp_processed = dicom_to_image(ds, wc, ww, normalize=False) # Keep closer to original for AI
                            if isinstance(temp_display, Image.Image) and isinstance(temp_processed, Image.Image):
                                is_dicom_confirmed = True
                                success = True
                                logger.info("DICOM processing successful.")
                            else:
                                logger.warning("DICOM parsing succeeded but image conversion failed.")
                        else:
                             logger.warning("parse_dicom returned None, treating as non-DICOM.")
                    except pydicom.errors.InvalidDicomError:
                        logger.warning(f"File '{uploaded_file.name}' is not a valid DICOM based on pydicom check. Attempting standard image processing.")
                        is_dicom_likely = False # Fallback to standard image processing
                    except Exception as e:
                        st.error(f"Error processing DICOM file: {e}")
                        logger.error(f"DICOM processing failed: {e}", exc_info=True)

                # If not DICOM or DICOM processing failed, try standard image processing
                if not is_dicom_confirmed:
                    st.session_state.is_dicom = False
                    logger.info(f"Attempting standard image processing for {uploaded_file.name}")
                    try:
                        raw_img = Image.open(io.BytesIO(st.session_state.raw_image_bytes))
                        # Convert to RGB AFTER opening to handle various modes like RGBA, P, L etc.
                        rgb_img = raw_img.convert("RGB")
                        temp_display = rgb_img.copy()
                        temp_processed = rgb_img.copy()
                        success = True
                        logger.info("Standard image processing successful.")
                    except UnidentifiedImageError:
                        st.error("Unsupported image format. Please upload JPG, PNG, or a valid DICOM file.")
                        logger.error("UnidentifiedImageError during standard image processing.")
                        success = False
                    except Exception as e:
                        st.error(f"Error processing standard image file: {e}")
                        logger.error(f"Standard image processing failed: {e}", exc_info=True)
                        success = False

                # Finalize state update if processing was successful
                if success and isinstance(temp_display, Image.Image) and isinstance(temp_processed, Image.Image):
                    st.session_state.is_dicom = is_dicom_confirmed
                    # Ensure images are in RGB for display consistency if needed (already done above)
                    # if temp_display.mode != 'RGB': temp_display = temp_display.convert('RGB')
                    # if temp_processed.mode != 'RGB': temp_processed = temp_processed.convert('RGB')

                    st.session_state.display_image = temp_display
                    st.session_state.processed_image = temp_processed
                    st.success(f"✅ Image '{uploaded_file.name}' processed successfully!")
                    logger.info(f"Image processing finalized. DICOM: {st.session_state.is_dicom}")
                    st.rerun() # Rerun to update the UI with the new image
                else:
                    st.error("Image loading failed. Please check the file format or integrity.")
                    # Reset relevant state variables on failure
                    st.session_state.uploaded_file_info = None
                    st.session_state.raw_image_bytes = None
                    st.session_state.display_image = None
                    st.session_state.processed_image = None
                    logger.error("Image processing failed overall. State reset.")
                    # No rerun here, let the user see the error

    # --- DICOM Window/Level Sliders ---
    # Show only if a DICOM image is successfully loaded
    if st.session_state.is_dicom and st.session_state.dicom_dataset is not None and isinstance(st.session_state.display_image, Image.Image):
        st.markdown("---")
        st.subheader("DICOM Windowing")
        try:
            # Use the ui_components function for sliders
            wc, ww = dicom_wl_sliders(
                st.session_state.dicom_dataset,
                st.session_state.current_display_wc,
                st.session_state.current_display_ww
            )
            # Check if values changed
            if wc != st.session_state.current_display_wc or ww != st.session_state.current_display_ww:
                logger.info(f"DICOM W/L changed: WC={wc}, WW={ww}")
                st.session_state.current_display_wc = wc
                st.session_state.current_display_ww = ww
                with st.spinner("Updating DICOM view..."):
                    new_display_img = dicom_to_image(st.session_state.dicom_dataset, wc, ww)
                    if isinstance(new_display_img, Image.Image):
                        if new_display_img.mode != 'RGB': new_display_img = new_display_img.convert('RGB')
                        st.session_state.display_image = new_display_img
                        # Note: We usually don't update st.session_state.processed_image here,
                        # as the AI should ideally work on a consistent representation unless intended.
                    else:
                         st.warning("Failed to update DICOM view with new W/L settings.")
                         logger.warning("dicom_to_image failed during W/L update.")
                st.rerun() # Rerun to show the updated image
        except Exception as e:
            st.error(f"Error rendering DICOM sliders: {e}")
            logger.error(f"Error in DICOM slider section: {e}", exc_info=True)

    st.markdown("---")
    st.header("AI Actions")

    # --- AI Action Buttons ---
    img_available = isinstance(st.session_state.get("processed_image"), Image.Image)

    if st.button("▶️ Run Initial Analysis", key="analyze_btn", help="Perform a general analysis of the image.", disabled=not img_available):
        st.session_state.last_action = "analyze"
        st.rerun()

    st.subheader("❓ Ask AI a Question")
    question_input = st.text_area(
        "Your Question:", height=80, key="question_input_widget",
        placeholder="e.g., 'Are there any signs of fracture?' or 'Describe the highlighted region.'",
        disabled=not img_available
    )
    if st.button("💬 Ask AI", key="ask_btn", help="Submit your question about the image/ROI to the AI.", disabled=not img_available):
        if question_input.strip():
            st.session_state.last_action = "ask"
            st.rerun()
        else:
            st.warning("Please enter a question before submitting.")

    st.subheader("🎯 Condition Analysis")
    # Common conditions - you might want to customize this list
    DISEASE_OPTIONS = [
        "Pneumonia", "Lung Cancer", "Nodule/Mass", "Effusion", "Pneumothorax",
        "Fracture", "Arthritis", "Osteoporosis",
        "Stroke", "Brain Tumor", "Hemorrhage",
        "Appendicitis", "Bowel Obstruction", "Cardiomegaly", "Aortic Aneurysm",
        "Tuberculosis", "COVID-19", "Pulmonary Embolism", "Other (Specify in Q&A)"
    ]
    disease_select = st.selectbox(
        "Select Condition:", [""] + sorted(DISEASE_OPTIONS), key="disease_select_widget",
        help="Select a potential condition for focused analysis.",
        disabled=not img_available
    )
    if st.button("🩺 Run Condition Analysis", key="disease_btn", help="Analyze the image specifically for the selected condition.", disabled=not img_available):
        if disease_select:
            st.session_state.last_action = "disease"
            st.rerun()
        else:
            st.warning("Please select a condition to analyze.")

    st.subheader("📊 Confidence & Report")
    # Enable confidence if there's any text output from the AI Q&A (as per llm_interactions logic)
    can_estimate = bool(st.session_state.history and img_available)
    if st.button("📈 Estimate Confidence", key="confidence_btn", disabled=not can_estimate, help="Estimate the AI's confidence based on the last Q&A interaction."):
        if can_estimate:
            st.session_state.last_action = "confidence"
            st.rerun()
        # No warning needed here as button is disabled if no history

    # Enable PDF generation if an image has been processed
    if st.button("📄 Generate PDF Data", key="generate_report_data_btn", help="Compile the analysis results into a PDF report.", disabled=not img_available):
        st.session_state.last_action = "generate_report_data"
        st.rerun()

    # Download button appears only after PDF bytes are generated
    if st.session_state.get("pdf_report_bytes"):
        fname = f"RadVisionAI_Report_{st.session_state.session_id or 'session'}.pdf"
        st.download_button(
            label="⬇️ Download PDF Report",
            data=st.session_state.pdf_report_bytes,
            file_name=fname,
            mime="application/pdf",
            key="download_pdf_button",
            help="Click to download the generated PDF report."
        )
        # Optionally clear the bytes after showing the button once to avoid clutter
        # Consider adding a button to explicitly clear the generated report state if needed
        # st.session_state.pdf_report_bytes = None

# --- Main Content: Two-Column Layout ---
col1, col2 = st.columns([2, 3]) # Adjust ratio if needed (e.g., [1, 1] or [3, 2])

with col1:
    st.subheader("🖼️ Image Viewer")
    display_img = st.session_state.get("display_image")

    if isinstance(display_img, Image.Image):
        # Display the image first
        st.image(display_img, caption="Current View" + (" (DICOM)" if st.session_state.is_dicom else ""), use_container_width=True)

        # ROI Drawing Canvas Section
        st.markdown("---")
        st.caption("Draw a rectangle below to select a Region of Interest (ROI). Use sidebar button to clear.")
        # Adjust canvas size dynamically but within limits
        MAX_CANVAS_WIDTH = 600 # Max width in pixels
        MAX_CANVAS_HEIGHT = 500 # Max height in pixels
        img_w, img_h = display_img.size

        if img_w > 0 and img_h > 0:
            aspect_ratio = img_w / img_h
            # Calculate width first based on container width (col1)
            canvas_width = min(img_w, MAX_CANVAS_WIDTH) # Simple approach, consider container width if needed
            canvas_height = int(canvas_width / aspect_ratio)

            # Fit to max height if necessary
            if canvas_height > MAX_CANVAS_HEIGHT:
                canvas_height = MAX_CANVAS_HEIGHT
                canvas_width = int(canvas_height * aspect_ratio)

            # Ensure minimum dimensions
            canvas_width = max(int(canvas_width), 150)
            canvas_height = max(int(canvas_height), 150)

            logger.debug(f"Canvas dimensions set to: {canvas_width}x{canvas_height}")

            try:
                # Use a BytesIO object for the background image for reliability
                buffered_bg = io.BytesIO()
                bg_img_display = display_img.copy()
                if bg_img_display.mode != 'RGB': bg_img_display=bg_img_display.convert('RGB')
                bg_img_display.save(buffered_bg, format="PNG")

                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.2)", # Semi-transparent orange fill
                    stroke_width=2,
                    stroke_color="rgba(230, 50, 50, 0.9)", # Red border
                    # background_image=display_img, # Use the image object directly
                    background_image=Image.open(buffered_bg), # More reliable?
                    update_streamlit=True, # Update Streamlit state on drawing events
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="rect", # Only allow rectangle drawing
                    initial_drawing=st.session_state.get("canvas_drawing", None), # Restore previous drawing if available
                    key="drawable_canvas" # Unique key for the canvas
                )

                # Process canvas results if drawing exists and has objects
                if canvas_result.json_data and canvas_result.json_data.get("objects"):
                    # Get the last drawn object (assuming only one ROI rectangle is intended)
                    if isinstance(canvas_result.json_data["objects"], list) and canvas_result.json_data["objects"]:
                        last_obj = canvas_result.json_data["objects"][-1]
                        if isinstance(last_obj, dict) and last_obj.get("type") == "rect":
                            # Extract rectangle properties, considering potential scaling factors
                            scaleX = last_obj.get("scaleX", 1.0)
                            scaleY = last_obj.get("scaleY", 1.0)
                            left_val = int(last_obj.get("left", 0))
                            top_val = int(last_obj.get("top", 0))
                            width_val = int(last_obj.get("width", 0) * scaleX)
                            height_val = int(last_obj.get("height", 0) * scaleY)

                            # Calculate scaling factor from displayed canvas size to original image size
                            scale_x_img = img_w / canvas_width
                            scale_y_img = img_h / canvas_height

                            # Calculate ROI coordinates in the original image dimensions
                            orig_left = max(0, int(left_val * scale_x_img))
                            orig_top = max(0, int(top_val * scale_y_img))
                            # Clamp width/height to image boundaries
                            orig_width = min(img_w - orig_left, int(width_val * scale_x_img))
                            orig_height = min(img_h - orig_top, int(height_val * scale_y_img))
                            orig_width = max(0, orig_width) # Ensure non-negative
                            orig_height = max(0, orig_height) # Ensure non-negative


                            # Store the calculated ROI coordinates if they are valid (min size 5x5?)
                            if orig_width > 5 and orig_height > 5:
                                new_roi = {"left": orig_left, "top": orig_top, "width": orig_width, "height": orig_height}
                                # Update session state only if ROI changed to avoid unnecessary reruns
                                if st.session_state.roi_coords != new_roi:
                                    st.session_state.roi_coords = new_roi
                                    st.session_state.canvas_drawing = canvas_result.json_data # Save canvas state
                                    logger.info(f"ROI updated: {new_roi}")
                                    st.rerun() # Rerun to reflect potential ROI-based changes elsewhere
                            else:
                                 # If calculated ROI is invalid (too small), clear existing ROI
                                 if st.session_state.roi_coords is not None:
                                     st.session_state.roi_coords = None
                                     logger.info("ROI cleared due to invalid dimensions (too small).")
                                     st.rerun()
                        else:
                             logger.debug("Last canvas object was not a dictionary or not a rect.")
                    else:
                        logger.debug("Canvas objects data is not a list or is empty.")


                # Handle case where drawing exists but has no objects (e.g., cleared by double click?)
                elif canvas_result.json_data and isinstance(canvas_result.json_data.get("objects"), list) and not canvas_result.json_data.get("objects"):
                     if st.session_state.roi_coords is not None:
                         st.session_state.roi_coords = None
                         st.session_state.canvas_drawing = None # Clear canvas state as well
                         logger.info("ROI cleared via canvas interaction (no objects).")
                         st.rerun()

            except Exception as e:
                st.error(f"Error initializing drawing canvas: {e}")
                logger.error(f"Canvas error: {e}", exc_info=True)

        else:
            st.warning("Image has invalid dimensions (width or height is zero), cannot enable ROI drawing.")

        # --- DICOM Metadata Display ---
        if st.session_state.is_dicom and st.session_state.dicom_metadata:
            st.markdown("---")
            with st.expander("DICOM Metadata", expanded=False):
                try:
                    # Use the ui_components function to display metadata
                    display_dicom_metadata(st.session_state.dicom_metadata)
                except Exception as e:
                    st.error(f"Error displaying DICOM metadata: {e}")
                    logger.error(f"Error calling display_dicom_metadata: {e}", exc_info=True)

    else:
        # Message when no image is loaded
        st.info("Upload an image using the sidebar or activate Demo Mode to begin analysis.")


with col2:
    st.subheader("📊 Analysis & Results")

    # Define tab titles including the new Translation tab
    tab_titles = ["🔬 Initial Analysis", "💬 Q&A History", "🩺 Disease Focus", "📈 Confidence", "🌐 Translation"]
    try:
        tabs = st.tabs(tab_titles)
    except Exception as e:
        st.error(f"Failed to create tabs: {e}")
        logger.error(f"Error creating st.tabs: {e}", exc_info=True)
        st.stop() # Stop if tabs fail

    # Tab 0: Initial Analysis
    with tabs[0]:
        analysis_value = st.session_state.initial_analysis or "Run 'Initial Analysis' from the sidebar."
        st.text_area(
            "Overall Findings & Impressions",
            value=analysis_value,
            height=450,
            key="output_initial",
            disabled=True, # Display only
            help="Results from the 'Run Initial Analysis' action."
        )

    # Tab 1: Q&A History
    with tabs[1]:
        qa_value = st.session_state.qa_answer or "Ask a question using the sidebar."
        st.text_area(
            "Latest AI Answer",
            value=qa_value,
            height=200,
            key="output_qa",
            disabled=True, # Display only
            help="The most recent answer from the 'Ask AI' action."
        )
        st.markdown("---")
        if st.session_state.history:
            with st.expander("Full Conversation History", expanded=True):
                # Display history in reverse chronological order (newest first)
                for i, (q, a) in enumerate(reversed(st.session_state.history)):
                    st.markdown(f"**You ({len(st.session_state.history) - i}):** {q}")
                    # Use markdown with unsafe_allow_html=True if answers might contain HTML/Markdown formatting
                    answer_display = a if a else "[No Answer Recorded]"
                    st.markdown(f"**AI ({len(st.session_state.history) - i}):**\n{answer_display}", unsafe_allow_html=True)
                    if i < len(st.session_state.history) - 1:
                        st.markdown("---") # Separator between entries
        else:
            st.caption("No questions asked yet in this session.")

    # Tab 2: Disease Focus
    with tabs[2]:
        disease_value = st.session_state.disease_analysis or "Run 'Condition Analysis' from the sidebar."
        st.text_area(
            "Disease-Specific Analysis",
            value=disease_value,
            height=450,
            key="output_disease",
            disabled=True, # Display only
            help="Results from the 'Run Condition Analysis' action for a selected condition."
        )

    # Tab 3: Confidence
    with tabs[3]:
        confidence_value = st.session_state.confidence_score or "Run 'Estimate Confidence' from the sidebar."
        st.text_area(
            "AI Confidence Estimation",
            value=confidence_value,
            height=450,
            key="output_confidence",
            disabled=True, # Display only
            help="An estimation of the AI's confidence based on the last Q&A."
        )

    # Tab 4: Translation - NEW TAB
    with tabs[4]:
        st.subheader("🌐 Translate Analysis Results")

        # Check if translation functions are available
        translation_enabled = bool(translate and detect_language and LANGUAGE_CODES)
        if not translation_enabled:
            st.warning("Translation features are currently unavailable. Please check the application setup and logs.")
            logger.warning("Translation tab accessed but translation functions/codes are missing or failed to import.")
        else:
            st.caption("Translate analysis text or enter custom text.")

            # 1. Select text source
            # Use helper function or dict comprehension for cleaner access
            def get_text_option(label):
                if label == "Initial Analysis": return st.session_state.initial_analysis
                if label == "Latest Q&A Answer": return st.session_state.qa_answer
                if label == "Disease Analysis": return st.session_state.disease_analysis
                if label == "Confidence Estimation": return st.session_state.confidence_score
                return ""

            text_option_labels = [
                "Initial Analysis", "Latest Q&A Answer", "Disease Analysis",
                "Confidence Estimation", "(Enter Custom Text Below)"
            ]
            selected_label = st.selectbox(
                "Select text to translate:",
                options=text_option_labels,
                key="translate_source_select",
                index=0 # Default to first option
            )

            # 2. Text Area for custom input or display selected text
            if selected_label == "(Enter Custom Text Below)":
                text_to_translate_input = st.text_area(
                    "Enter text here:", "", height=150, key="translate_custom_input"
                )
            else:
                # Display the selected text in a disabled text area
                selected_text_value = get_text_option(selected_label)
                text_to_translate_input = st.text_area(
                    f"Selected text ({selected_label}):",
                    value=selected_text_value or f"'{selected_label}' is empty.",
                    height=150,
                    key="translate_selected_display",
                    disabled=True
                )

            # Determine the actual text to use for translation
            text_to_translate = text_to_translate_input if selected_label == "(Enter Custom Text Below)" else get_text_option(selected_label)

            # 3. Language Selection
            # Ensure LANGUAGE_CODES is usable
            if not isinstance(LANGUAGE_CODES, dict) or not LANGUAGE_CODES:
                 st.error("Internal Error: Language codes are not available for translation.")
                 logger.error("Translation Widget Error: LANGUAGE_CODES is not a valid dictionary.")
                 # Optionally disable the rest of the translation UI here
            else:
                lang_names = sorted(list(LANGUAGE_CODES.keys())) # Sort for better UI
                try:
                     # Find the index of the stored target language, default to 0 if not found or invalid
                     default_tgt_index = lang_names.index(st.session_state.translation_tgt_lang)
                except (ValueError, AttributeError, KeyError):
                     default_tgt_index = lang_names.index("Spanish") if "Spanish" in lang_names else 0 # Fallback

                col_lang1, col_lang2 = st.columns(2)
                with col_lang1:
                    src_lang_options = ["Auto-Detect"] + lang_names
                    try:
                        # Find index of stored source lang, default to 0 (Auto-Detect)
                        default_src_index = src_lang_options.index(st.session_state.translation_src_lang)
                    except (ValueError, AttributeError, KeyError):
                        default_src_index = 0 # Default to Auto-Detect
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

                # 4. Translate Button
                if st.button("Translate Now", key="translate_button_go", disabled=not translation_enabled):
                    st.session_state.translation_src_lang = selected_src_lang_name # Save selection
                    st.session_state.translation_tgt_lang = selected_tgt_lang_name # Save selection

                    if not text_to_translate or not text_to_translate.strip():
                        st.warning("Please select or enter text to translate.")
                    elif selected_src_lang_name == selected_tgt_lang_name and selected_src_lang_name != "Auto-Detect":
                        st.info("Source and Target languages are the same. No translation needed.")
                        st.session_state.translation_output = text_to_translate
                    else:
                        with st.spinner(f"Translating to {selected_tgt_lang_name}..."):
                            try:
                                actual_src_lang_name = selected_src_lang_name
                                detected_code = None

                                # Auto-Detection Logic
                                if selected_src_lang_name == "Auto-Detect":
                                    detected_code = detect_language(text_to_translate[:500]) # Detect based on first 500 chars
                                    logger.info(f"Attempting auto-detection, detected code: {detected_code}")
                                    detected_lang_found = False
                                    if detected_code and isinstance(detected_code, str):
                                        # Try to match detected code (or its base) with codes in our list
                                        detected_code_lower = detected_code.lower()
                                        detected_base_code = detected_code_lower.split('-')[0]
                                        for name, code in LANGUAGE_CODES.items():
                                            code_lower = code.lower()
                                            if detected_code_lower == code_lower or detected_base_code == code_lower.split('-')[0]:
                                                actual_src_lang_name = name
                                                st.info(f"Auto-detected source language: **{name}** ({detected_code})")
                                                detected_lang_found = True
                                                break
                                    if not detected_lang_found:
                                        st.warning("Could not reliably map auto-detected source language to known list. Assuming English.")
                                        logger.warning(f"Auto-detection failed or code '{detected_code}' not in LANGUAGE_CODES. Defaulting to English.")
                                        actual_src_lang_name = "English" # Fallback if detection fails or not in our list
                                    # Handle case where detect_language itself failed/returned non-string
                                    elif not detected_code:
                                         st.warning("Auto-detection failed. Assuming English.")
                                         logger.warning("Auto-detection function returned empty/None. Defaulting to English.")
                                         actual_src_lang_name = "English"


                                # Check again if source and target are now the same after detection
                                if actual_src_lang_name == selected_tgt_lang_name:
                                    st.info(f"Detected source language ({actual_src_lang_name}) is the same as the target. No translation needed.")
                                    st.session_state.translation_output = text_to_translate
                                else:
                                    # --- Call your translation function ---
                                    # This uses the function imported from translation_models.py
                                    raw_translation = translate(
                                        text_to_translate,
                                        target_language=selected_tgt_lang_name,
                                        source_language=actual_src_lang_name # Pass detected/selected source name
                                    )
                                    # ---

                                    if raw_translation and not raw_translation.startswith("[Translation Error"):
                                        # Apply post-processing formatting
                                        final_trans = format_translation(raw_translation)
                                        st.session_state.translation_output = final_trans
                                        st.success("Translation complete!")
                                        logger.info(f"Translation successful: {actual_src_lang_name} -> {selected_tgt_lang_name}")
                                    else:
                                        # Handle errors returned explicitly by the translate function or empty results
                                        error_msg = raw_translation if (raw_translation and isinstance(raw_translation, str)) else "[Error: Translation service returned an empty or invalid result.]"
                                        st.error(f"Translation failed. {error_msg}")
                                        logger.error(f"Translation failed for: {actual_src_lang_name} -> {selected_tgt_lang_name}. Reason: {error_msg}")
                                        st.session_state.translation_output = error_msg

                            except Exception as e:
                                st.error(f"An unexpected error occurred during translation: {e}")
                                logger.error(f"Translation exception: {e}", exc_info=True)
                                st.session_state.translation_output = f"[Error: Exception during translation - {e}]"

                # 5. Display Translation Output
                st.text_area(
                    "Translated Text:",
                    value=st.session_state.translation_output or "",
                    height=250,
                    key="translate_output_area",
                    disabled=True, # Usually display only, user can copy
                    help="The result of the translation will appear here."
                )


# --- ACTION HANDLING LOGIC ---
# (This section processes button clicks stored in st.session_state.last_action)
if 'last_action' in st.session_state and st.session_state.last_action:
    current_action = st.session_state.last_action
    logger.info(f"Handling action: {current_action}")

    # Reset action flag immediately to prevent re-triggering on subsequent reruns
    action_to_perform = st.session_state.last_action
    st.session_state.last_action = None

    # Ensure image exists for actions that require it
    img_llm = st.session_state.get("processed_image")
    if action_to_perform != "generate_report_data" and not isinstance(img_llm, Image.Image):
        st.error(f"Cannot perform '{action_to_perform}': No valid image loaded or processed.")
        logger.error(f"Action '{action_to_perform}' skipped: processed_image is invalid or None.")
        st.stop() # Stop processing this action

    # Ensure session ID exists
    if not st.session_state.get("session_id"):
        st.error(f"Cannot perform '{action_to_perform}': Session ID is missing. Please reload.")
        logger.error(f"Action '{action_to_perform}' aborted: missing session ID.")
        st.stop() # Stop processing this action

    # Prepare common variables
    roi = st.session_state.get("roi_coords") # Use .get for safety
    roi_str = " (with ROI)" if roi else ""
    history = st.session_state.history if isinstance(st.session_state.history, list) else []
    # Ensure history is assigned back if corrected type
    st.session_state.history = history

    try:
        if action_to_perform == "analyze":
            st.info(f"🔬 Performing initial analysis{roi_str}...")
            with st.spinner("AI is analyzing the image... This may take a moment."):
                # Pass ROI to the updated function definition
                result = run_initial_analysis(img_llm, roi=roi)
            st.session_state.initial_analysis = result
            # Clear other analysis fields when starting a new initial analysis
            st.session_state.qa_answer = ""
            st.session_state.disease_analysis = ""
            st.session_state.confidence_score = ""
            logger.info(f"Initial analysis action completed{roi_str}.")
            if result and result.startswith("Error:") or "Failed:" in result:
                 st.error(f"Initial Analysis Error: {result}")
            st.rerun() # Rerun to display results/errors in the correct tab

        elif action_to_perform == "ask":
            q = st.session_state.question_input_widget.strip() # Get question from widget state
            if not q:
                st.warning("Question field was empty.")
                logger.warning("Attempted 'ask' action with empty question.")
            else:
                st.info(f"❓ Asking AI: \"{q}\"{roi_str}...")
                st.session_state.qa_answer = "" # Clear previous answer display
                with st.spinner("AI is processing your question... This may take a moment."):
                    # Pass history and ROI to the updated function
                    answer, ok = run_multimodal_qa(img_llm, q, history, roi=roi)

                st.session_state.qa_answer = answer # Store answer/error message
                if ok:
                    st.session_state.history.append((q, answer)) # Add successful Q&A to history
                    logger.info(f"Q&A successful for: '{q}'{roi_str}")
                else:
                    # Handle primary AI failure (error message is already in 'answer')
                    st.error(f"Q&A Failed: {answer}")
                    logger.warning(f"Primary AI Q&A failed for '{q}'. Reason: {answer}")

                    # --- Attempt Fallback (Optional - requires hf_models.py) ---
                    if HF_VQA_MODEL_ID and os.environ.get("HF_API_TOKEN"):
                        logger.info(f"Attempting HF fallback using {HF_VQA_MODEL_ID}.")
                        st.info(f"Attempting fallback with {HF_VQA_MODEL_ID}...") # User feedback
                        with st.spinner(f"Trying fallback: {HF_VQA_MODEL_ID}..."):
                            fb_ans, fb_ok = query_hf_vqa_inference_api(img_llm, q, roi=roi)
                        if fb_ok:
                            fb_disp = f"**[Fallback Result: {HF_VQA_MODEL_ID}]**\n\n{fb_ans}"
                            st.session_state.qa_answer += "\n\n" + fb_disp # Append fallback answer
                            st.session_state.history.append((f"[Fallback Query] {q}", fb_disp)) # Add fallback to history
                            st.info("Fallback AI provided an answer.")
                            logger.info(f"HF fallback successful for '{q}'.")
                        else:
                            fb_fail_msg = f"**[Fallback Failed - {HF_VQA_MODEL_ID}]:** {fb_ans}"
                            st.session_state.qa_answer += "\n\n" + fb_fail_msg
                            st.error(f"Fallback AI also failed: {fb_ans}")
                            logger.error(f"HF fallback failed for '{q}'. Reason: {fb_ans}")
                    elif HF_VQA_MODEL_ID:
                        logger.warning("HF fallback skipped: HF_API_TOKEN secret not set.")
                        st.session_state.qa_answer += "\n\n**[Fallback Skipped: API Token Missing]**"
                    else:
                         logger.info("HF fallback skipped: No fallback model configured.")
                         # Optionally add message: "\n\n**[Fallback Unavailable]**"
                    # --- End Fallback ---
            st.rerun() # Rerun to display answer/errors

        elif action_to_perform == "disease":
            d = st.session_state.disease_select_widget # Get selected disease from widget state
            if not d:
                st.warning("No condition was selected for analysis.")
                logger.warning("Attempted 'disease' action with no condition selected.")
            else:
                st.info(f"🩺 Running focused analysis for '{d}'{roi_str}...")
                with st.spinner(f"AI is analyzing for signs of {d}... This may take a moment."):
                    # Pass ROI to the updated function
                    result = run_disease_analysis(img_llm, d, roi=roi)
                st.session_state.disease_analysis = result
                # Clear other analysis fields
                st.session_state.qa_answer = ""
                st.session_state.confidence_score = ""
                logger.info(f"Disease analysis action completed for '{d}'{roi_str}.")
                if result and result.startswith("Error:") or "Failed:" in result:
                    st.error(f"Disease Analysis Error: {result}")
            st.rerun() # Rerun to display results

        elif action_to_perform == "confidence":
            # Check again if there is context, though button should be disabled if not
            if not history: # Confidence estimation requires history based on current llm_interactions
                st.warning("Cannot estimate confidence without prior Q&A history.")
                logger.warning("Confidence estimation skipped: no history available.")
            else:
                st.info(f"📊 Estimating AI confidence based on last interaction{roi_str}...")
                context_summary = f"History: {len(history)} entries. ROI used: {bool(roi)}"
                logger.info(f"Running confidence estimation with context: {context_summary}")
                with st.spinner("AI is assessing its confidence..."):
                    # Pass relevant context to the confidence function
                    # Note: Assuming estimate_ai_confidence mainly uses the *last* history item
                    res = estimate_ai_confidence(
                        img_llm,
                        history=history,
                        initial_analysis=st.session_state.initial_analysis, # Pass for potential context
                        disease_analysis=st.session_state.disease_analysis, # Pass for potential context
                        roi=roi # Pass ROI if it influenced the analyses being evaluated
                    )
                st.session_state.confidence_score = res
                logger.info("Confidence estimation action completed.")
                if res and res.startswith("Error:") or "Failed:" in res:
                     st.error(f"Confidence Estimation Error: {res}")
            st.rerun() # Rerun to display score

        elif action_to_perform == "generate_report_data":
            st.info("📄 Generating PDF report data...")
            st.session_state.pdf_report_bytes = None # Clear previous report bytes
            img_for_report = st.session_state.get("display_image") # Use the currently displayed image

            if not isinstance(img_for_report, Image.Image):
                st.error("Cannot generate report: Invalid or missing display image.")
                logger.error("PDF generation aborted: invalid display_image.")
            else:
                img_final = img_for_report.copy() # Work on a copy
                # Draw ROI on the image copy if ROI exists and is valid
                if roi and isinstance(roi, dict) and all(k in roi for k in ['left', 'top', 'width', 'height']):
                    try:
                        draw = ImageDraw.Draw(img_final)
                        x0, y0 = roi['left'], roi['top']
                        x1 = x0 + roi['width']
                        y1 = y0 + roi['height']
                        # Draw a distinct rectangle (e.g., red, thickness 3)
                        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
                        logger.info("ROI drawn on image for PDF report.")
                    except Exception as e:
                        logger.error(f"Error drawing ROI on image for PDF: {e}", exc_info=True)
                        st.warning("Could not draw ROI on report image.")
                        # Continue report generation without the drawn ROI

                # Compile all text outputs for the report
                full_history = "\n\n".join([f"Q: {q}\nA: {a if a else '[No Answer]'}" for q, a in history]) if history else "No conversation history recorded."
                outputs = {
                    "Session ID": st.session_state.session_id or "N/A",
                    "Initial Analysis": st.session_state.initial_analysis or "Not Performed",
                    "Conversation History": full_history,
                    "Condition Analysis": st.session_state.disease_analysis or "Not Performed",
                    "Confidence Estimation": st.session_state.confidence_score or "Not Estimated",
                }

                # Add DICOM metadata summary if available
                dicom_meta_for_report = None
                if st.session_state.is_dicom and st.session_state.dicom_metadata:
                    dicom_meta_for_report = st.session_state.dicom_metadata
                    # Optionally create a summary string if the report function expects that
                    meta_summary = {k: v for k, v in st.session_state.dicom_metadata.items() if k in ['PatientName', 'PatientID', 'StudyDescription', 'Modality']}
                    outputs["DICOM Summary"] = str(meta_summary) if meta_summary else "Basic DICOM metadata available."

                # Generate the PDF bytes
                with st.spinner("Generating PDF document..."):
                    # Pass metadata if generate_pdf_report_bytes expects it
                    pdf_bytes = generate_pdf_report_bytes(
                        st.session_state.session_id,
                        img_final, # Image with ROI drawn (if applicable)
                        outputs, # Dictionary of text results
                        dicom_meta_for_report # Pass full metadata if needed by report function
                    )

                if pdf_bytes:
                    st.session_state.pdf_report_bytes = pdf_bytes
                    st.success("PDF report generated successfully! Download button available in the sidebar.")
                    logger.info("PDF generation successful.")
                    st.balloons() # Fun success indicator!
                else:
                    st.error("Failed to generate PDF report. The report generation function might have encountered an issue.")
                    logger.error("PDF generation failed: generate_pdf_report_bytes returned None or empty.")
            st.rerun() # Rerun to show download button or error

        else:
            st.warning(f"Unknown action '{action_to_perform}' encountered.")
            logger.warning(f"Attempted to handle unknown action: '{action_to_perform}'")
            # No rerun needed if action is unknown

    except Exception as e:
        # Catchall for unexpected errors during action handling
        st.error(f"An critical error occurred while handling action '{action_to_perform}': {e}")
        logger.critical(f"Unhandled exception during action '{action_to_perform}': {e}", exc_info=True)
        # Rerun to clear the action state and potentially show error message correctly
        st.rerun()

# --- Footer ---
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced | Session ID: {st.session_state.get('session_id', 'N/A')}")

# Simple Footer Example (Update links as needed)
st.markdown(
    """
    <div style="text-align: center; font-size: 0.8em; color: #888; margin-top: 2em;">
      <a href="#" target="_blank" rel="noopener noreferrer">Privacy Policy</a> |
      <a href="#" target="_blank" rel="noopener noreferrer">Terms of Service</a> |
      <a href="#" target="_blank" rel="noopener noreferrer">Documentation</a>
      <br>
      RadVision AI is intended for informational purposes only. Consult a qualified healthcare professional for diagnosis.
    </div>
    """, unsafe_allow_html=True
)

logger.info("--- App Render Complete ---")