# -*- coding: utf-8 -*-
"""
app.py - Main Streamlit application for RadVision AI Advanced.

Includes debugging steps for image display issue and monkey-patch fix.
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

# --- Pillow (PIL) - Essential (Import Early for Patch) ---
try:
    from PIL import Image, ImageDraw, UnidentifiedImageError
    import PIL
    PIL_VERSION = getattr(PIL, '__version__', 'Unknown')
    PIL_AVAILABLE = True
except ImportError:
    print("CRITICAL ERROR: Pillow (PIL) is not installed (`pip install Pillow`). Cannot continue.")
    st.error("CRITICAL ERROR: Pillow (PIL) is not installed (`pip install Pillow`). Image processing disabled.")
    PIL_AVAILABLE = False
    st.stop()

# --- v v v --- MONKEY PATCH FOR st_image.image_to_url --- v v v ---
# Apply this early, after core imports and PIL, before other components
try:
    import streamlit.elements.image as st_image

    if not hasattr(st_image, "image_to_url"):
        print("Applying monkey-patch for missing 'streamlit.elements.image.image_to_url'...")

        def image_to_url_monkey_patch(
            image: Any,
            width: int = -1,
            clamp: bool = False,
            channels: str = "RGB",
            output_format: str = "auto",
            image_id: str = "",
        ) -> str:
            """Simplified image_to_url implementation for compatibility."""
            # Add logging inside the patch
            patch_logger = logging.getLogger(__name__ + ".monkey_patch")
            patch_logger.debug(f"Monkey patch image_to_url called with type: {type(image)}")

            if isinstance(image, Image.Image):
                try:
                    fmt = output_format.upper()
                    if fmt == "AUTO":
                        fmt = image.format if image.format else "PNG"

                    if fmt not in ["PNG", "JPEG", "GIF", "WEBP"]:
                         patch_logger.warning(f"Image format {fmt} may not be ideal for data URL. Converting to PNG.")
                         fmt = "PNG"

                    img_to_save = image
                    if channels == "RGB" and image.mode != "RGB":
                        if image.mode not in ['L', 'RGB']:
                             patch_logger.debug(f"Converting image mode {image.mode} to RGB.")
                             img_to_save = image.convert("RGB")
                    elif image.mode == 'P':
                        patch_logger.debug(f"Converting image mode P to RGBA.")
                        img_to_save = image.convert("RGBA")
                        fmt = "PNG"

                    buffered = io.BytesIO()
                    img_to_save.save(buffered, format=fmt)
                    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    result_url = f"data:image/{fmt.lower()};base64,{img_b64}"
                    patch_logger.debug(f"Monkey patch successfully generated data URL (len: {len(result_url)}).")
                    return result_url

                except Exception as e:
                    patch_logger.error(f"ERROR in monkey-patch image_to_url: {e}", exc_info=True)
                    return ""
            else:
                 patch_logger.warning(f"Monkey-patch image_to_url received unsupported type: {type(image)}")
                 return ""

        st_image.image_to_url = image_to_url_monkey_patch
        print("Monkey-patch applied successfully.")
        logger.info("Applied monkey-patch for st.elements.image.image_to_url.") # Use main logger
    else:
        print("'streamlit.elements.image.image_to_url' already exists. No patch needed.")
        logger.info("'streamlit.elements.image.image_to_url' already exists. No patch needed.")

except ImportError:
    print("Could not import streamlit.elements.image. Skipping monkey-patch.")
    logger.warning("Could not import streamlit.elements.image. Skipping monkey-patch.")
except Exception as e:
    print(f"An error occurred during monkey-patch setup: {e}")
    logger.error(f"An error occurred during monkey-patch setup: {e}", exc_info=True)
# --- ^ ^ ^ --- END OF MONKEY PATCH --- ^ ^ ^ ---


# --- Ensure deep-translator is installed ---
# (Keeping this logic as it is)
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

# --- Logging Setup ---
# (Keeping this logic as it is)
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
logger.info(f"Pillow (PIL) Version: {PIL_VERSION}")


# --- Dependency Checks & Imports (Continued) ---
# (Keeping imports as they were)

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
# (Keeping this as is)
st.markdown(
    """ <style> ... [Existing CSS Rules] ... </style> """,
    unsafe_allow_html=True
)

# --- Display Hero Logo ---
# (Keeping this as is)
logo_path = os.path.join("assets", "radvisionai-hero.jpeg")
if os.path.exists(logo_path): st.image(logo_path, width=350)
else: logger.warning(f"Hero logo not found at: {logo_path}.")

# --- Initialize Session State Defaults ---
# (Keeping this as is)
DEFAULT_STATE = {
    # ... [Existing keys/values] ...
    "uploaded_file_info": None, "raw_image_bytes": None, "is_dicom": False,
    "dicom_dataset": None, "dicom_metadata": {}, "processed_image": None,
    "display_image": None, "session_id": None, "history": [],
    "initial_analysis": "", "qa_answer": "", "disease_analysis": "",
    "confidence_score": "",
    "last_action": None, "pdf_report_bytes": None, "canvas_drawing": None,
    "roi_coords": None, "current_display_wc": None, "current_display_ww": None,
    "clear_roi_feedback": False, "demo_loaded": False,
    "translation_result": None, "translation_error": None,
}
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.session_id = str(uuid.uuid4())[:8]
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value
    logger.info(f"New session initialized: {st.session_state.session_id}")
if not isinstance(st.session_state.get("history"), list):
    st.session_state.history = []
logger.debug(f"Session state verified for session ID: {st.session_state.session_id}")

# --- Utility Functions ---
def format_translation(translated_text: Optional[str]) -> str:
    # (Keeping this as is)
    if translated_text is None: return "Translation not available or failed."
    try:
        text_str = str(translated_text); formatted_text = re.sub(r'\s+(\d+\.)', r'\n\n\1', text_str)
        return formatted_text.strip()
    except Exception as e:
        logger.error(f"Error formatting translation: {e}", exc_info=True); return str(translated_text)

# --- REMOVED pil_to_base64_url HELPER FUNCTION ---

# --- Sidebar ---
# (Keeping this as is)
with st.sidebar:
    # ... [Existing Sidebar Code] ...
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

    demo_mode = st.checkbox("🚀 Demo Mode", value=st.session_state.get("demo_loaded", False),
                            help="Load a sample chest X-ray image and analysis.")
    if demo_mode and not st.session_state.demo_loaded:
        logger.info("Demo Mode activated.")
        st.warning("Demo mode selected, but loading logic needs implementation.")
    elif not demo_mode and st.session_state.demo_loaded:
         logger.info("Demo Mode deactivated.")
         st.session_state.demo_loaded = False

    if DRAWABLE_CANVAS_AVAILABLE:
        if st.button("🗑️ Clear ROI", help="Remove the selected ROI rectangle"):
            st.session_state.roi_coords = None
            st.session_state.canvas_drawing = None
            st.session_state.clear_roi_feedback = True
            logger.info("ROI cleared by user.")
            st.rerun()
        if st.session_state.get("clear_roi_feedback"):
            st.success("✅ ROI cleared!")
            st.session_state.clear_roi_feedback = False

    if st.session_state.is_dicom and DICOM_UTILS_AVAILABLE and UI_COMPONENTS_AVAILABLE and st.session_state.display_image:
        st.markdown("---")
        st.subheader("DICOM Display (W/L)")
        new_wc, new_ww = dicom_wl_sliders(
            st.session_state.current_display_wc,
            st.session_state.current_display_ww
        )
        if new_wc != st.session_state.current_display_wc or new_ww != st.session_state.current_display_ww:
            logger.info(f"DICOM W/L changed via UI: WC={new_wc}, WW={new_ww}")
            st.session_state.current_display_wc = new_wc
            st.session_state.current_display_ww = new_ww
            if st.session_state.dicom_dataset:
                with st.spinner("Applying new Window/Level..."):
                    try:
                        new_display_img = dicom_to_image(
                            st.session_state.dicom_dataset, wc=new_wc, ww=new_ww
                        )
                        if isinstance(new_display_img, Image.Image):
                            st.session_state.display_image = new_display_img.convert('RGB') if new_display_img.mode != 'RGB' else new_display_img
                            st.rerun()
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
    action_disabled = not LLM_INTERACTIONS_AVAILABLE or not isinstance(st.session_state.get("processed_image"), Image.Image)

    if st.button("🔬 Run Structured Initial Analysis", key="analyze_btn", disabled=action_disabled,
                 help="Perform a general, structured analysis (visual description, potential findings, limitations). Assumes backend uses agentic prompt."):
        st.session_state.last_action = "analyze"
        st.rerun()

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

    st.subheader("🎯 Condition-Specific Analysis")
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
    can_estimate = bool(st.session_state.history)
    if st.button("🧪 Estimate LLM Self-Assessment (Experimental)", key="confidence_btn",
                 disabled=not can_estimate or action_disabled,
                 help="EXPERIMENTAL: Ask the LLM to assess its last Q&A response. Not a clinical confidence score."):
        st.session_state.last_action = "confidence"
        st.rerun()

    report_generation_disabled = action_disabled or not REPORT_UTILS_AVAILABLE
    if st.button("📄 Generate PDF Report Data", key="generate_report_data_btn",
                 disabled=report_generation_disabled,
                 help="Compile analysis results into PDF data. Download button will appear below."):
        st.session_state.last_action = "generate_report_data"
        st.rerun()

    if st.session_state.get("pdf_report_bytes"):
        report_filename = f"RadVisionAI_Report_{st.session_state.session_id or 'session'}.pdf"
        st.download_button(
            label="⬇️ Download PDF Report", data=st.session_state.pdf_report_bytes,
            file_name=report_filename, mime="application/pdf",
            key="download_pdf_button", help="Download the generated PDF report."
        )


# --- File Upload Logic ---
# (Keeping this as is, but adding logging for processing_success)
if uploaded_file is not None and PIL_AVAILABLE:
    try:
        uploaded_file.seek(0)
        file_content_hash = hashlib.sha256(uploaded_file.read()).hexdigest()[:16]
        uploaded_file.seek(0)
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{file_content_hash}"
    except Exception as e:
        logger.warning(f"Could not generate hash for file '{uploaded_file.name}': {e}")
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{uuid.uuid4().hex[:8]}"

    if new_file_info != st.session_state.get("uploaded_file_info"):
        logger.info(f"New file upload detected: {uploaded_file.name} ({uploaded_file.size} bytes)")
        st.toast(f"Processing '{uploaded_file.name}'...", icon="⏳")
        keys_to_preserve = {"session_id"}
        st.session_state.session_id = st.session_state.get("session_id") or str(uuid.uuid4())[:8]
        for key, value in DEFAULT_STATE.items():
            if key not in keys_to_preserve:
                st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value
        st.session_state.uploaded_file_info = new_file_info
        st.session_state.demo_loaded = False

        st.session_state.raw_image_bytes = uploaded_file.getvalue()
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        st.session_state.is_dicom = (
            PYDICOM_AVAILABLE and DICOM_UTILS_AVAILABLE and
            ("dicom" in uploaded_file.type.lower() or file_ext in (".dcm", ".dicom"))
        )

        with st.spinner("🔬 Analyzing and preparing image..."):
            temp_display_img = None
            temp_processed_img = None
            processing_success = False # Initialize flag

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
                        temp_display_img = dicom_to_image(dicom_dataset, wc=default_wc, ww=default_ww)
                        temp_processed_img = dicom_to_image(dicom_dataset, wc=None, ww=None, normalize=True)
                        if isinstance(temp_display_img, Image.Image) and isinstance(temp_processed_img, Image.Image):
                            processing_success = True # Set flag on success
                            logger.info("DICOM parsed and converted successfully.")
                        else:
                            st.error("Failed to convert DICOM pixel data to a displayable image format.")
                            logger.error("dicom_to_image returned invalid image type(s).")
                    else:
                        st.error("Could not parse DICOM file. It might be corrupted or incomplete.")
                        logger.error("parse_dicom returned None.")
                except pydicom.errors.InvalidDicomError:
                    st.error("Invalid DICOM file format detected.")
                    logger.error("InvalidDicomError during parsing.")
                    st.session_state.is_dicom = False
                except Exception as e:
                    st.error(f"An unexpected error occurred processing DICOM: {e}")
                    logger.error(f"DICOM processing error: {e}", exc_info=True)
                    st.session_state.is_dicom = False

            if not st.session_state.is_dicom and not processing_success:
                logger.info("Processing as standard image (JPG/PNG)...")
                try:
                    raw_img = Image.open(io.BytesIO(st.session_state.raw_image_bytes))
                    processed_img = raw_img.convert("RGB")
                    temp_display_img = processed_img.copy()
                    temp_processed_img = processed_img.copy()
                    processing_success = True # Set flag on success
                    logger.info("Standard image loaded and converted to RGB successfully.")
                except UnidentifiedImageError:
                    st.error("Cannot identify image format. Please upload a valid JPG, PNG, or DICOM file.")
                    logger.error(f"UnidentifiedImageError for file: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error processing standard image: {e}")
                    logger.error(f"Standard image processing error: {e}", exc_info=True)

            # Log the final outcome of processing
            logger.info(f"Image processing completion status: {processing_success}")

            if processing_success and isinstance(temp_display_img, Image.Image) and isinstance(temp_processed_img, Image.Image):
                st.session_state.display_image = temp_display_img.convert('RGB') if temp_display_img.mode != 'RGB' else temp_display_img
                st.session_state.processed_image = temp_processed_img
                st.success(f"✅ Image '{uploaded_file.name}' loaded successfully!")
                logger.info(f"Image processing state updated for: {uploaded_file.name}")
                st.rerun()
            else:
                st.error("Image loading failed after processing attempt. Check file or logs.")
                logger.error(f"Image processing failed for file: {uploaded_file.name} (processing_success={processing_success})")
                st.session_state.uploaded_file_info = None
                st.session_state.display_image = None
                st.session_state.processed_image = None
                st.session_state.is_dicom = False

# --- Main Page Content ---

st.markdown("---")
# Main Disclaimer moved near top

st.warning(
    """
    **🔴 IMPORTANT: For Research & Educational Use Only 🔴**
    *   This tool **demonstrates** AI capabilities and is **NOT** a substitute for professional medical evaluation, diagnosis, or treatment advice.
    *   AI analysis is based on patterns and **may be inaccurate or incomplete.** It lacks full clinical context.
    *   **ALWAYS consult qualified healthcare professionals** for any medical concerns or decisions.
    *   **PRIVACY:** Do **NOT** upload identifiable patient information (PHI) unless you fully comply with all privacy regulations (e.g., HIPAA, GDPR) and have necessary consents.
    """,
    icon="⚠️"
)
st.title("⚕️ RadVision AI Advanced: AI-Assisted Image Analysis")
# Renamed expander for clarity
with st.expander("View User Guide & Workflow", expanded=False):
    st.markdown("""
    **Workflow:**
    1. **Upload Image**: Use sidebar to upload DICOM, JPG, or PNG (ensure de-identification!).
    2. **(DICOM)** Adjust Window/Level sliders in sidebar if needed.
    3. **ROI (Optional)**: Draw a rectangle on the image viewer below to focus the AI's attention. Clear with sidebar button.
    4. **AI Analysis**: Use sidebar buttons ('Structured Initial Analysis', 'Ask Question', 'Analyze for Condition'). Results appear in the right panel.
    5. **Translation**: Use the 'Translation' tab to translate AI-generated text.
    6. **Self-Assessment**: Use the 'LLM Self-Assessment' tab (and sidebar button) for an *experimental* AI reflection on its last answer.
    7. **Generate Report**: Use sidebar button to compile session data into a PDF, then download.
    """)

st.markdown("---")

col1, col2 = st.columns([2, 3], gap="large")

# --- Column 1: Image Viewer & Metadata ---
with col1:
    st.subheader("🖼️ Image Viewer")
    display_img = st.session_state.get("display_image")

    # --- v v v --- DEBUGGING IMAGE DISPLAY --- v v v ---
    if isinstance(display_img, Image.Image):
        st.success("DEBUG: `display_img` is a valid PIL Image object in session state.") # Add success message for debugging
        try:
            # --- TEMPORARILY USE st.image INSTEAD OF st_canvas ---
            st.image(display_img, caption="Image Preview (Debug Mode - Canvas Disabled)", use_container_width=True)
            st.info("Debug Info: Displaying with `st.image`. If this works, the issue is likely with `st_canvas` or the monkey patch.", icon="ℹ️")
            # --- END OF TEMPORARY st.image CALL ---

            # --- Commented out original st_canvas block for debugging ---
            '''
            if DRAWABLE_CANVAS_AVAILABLE and st_canvas:
                st.caption("Draw a rectangle below to select a Region of Interest (ROI).")
                # Relying on monkey-patch now, pass PIL image directly
                background_image_to_pass = display_img

                # (Canvas size calculation remains the same)
                MAX_CANVAS_WIDTH = 600
                MAX_CANVAS_HEIGHT = 550
                img_w, img_h = background_image_to_pass.size
                if img_w <= 0 or img_h <= 0:
                    st.warning("Image has invalid dimensions (<= 0). Cannot display canvas.")
                else:
                    aspect_ratio = img_w / img_h
                    canvas_width = min(img_w, MAX_CANVAS_WIDTH)
                    canvas_height = int(canvas_width / aspect_ratio)
                    if canvas_height > MAX_CANVAS_HEIGHT:
                        canvas_height = MAX_CANVAS_HEIGHT
                        canvas_width = int(canvas_height * aspect_ratio)
                    canvas_width = max(canvas_width, 150)
                    canvas_height = max(canvas_height, 150)

                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.2)",
                        stroke_width=2,
                        stroke_color="rgba(239, 83, 80, 0.8)",
                        background_image=background_image_to_pass, # Pass PIL Image
                        update_streamlit=True,
                        height=canvas_height,
                        width=canvas_width,
                        drawing_mode="rect",
                        initial_drawing=st.session_state.get("canvas_drawing", None),
                        key="drawable_canvas"
                    )

                    # (ROI processing logic remains the same)
                    if canvas_result.json_data and canvas_result.json_data.get("objects"):
                        # ... [ROI extraction logic] ...
                        pass # Placeholder for brevity
            else: # Fallback if canvas is not available
                st.image(display_img, caption="Image Preview (ROI drawing disabled)", use_container_width=True)
            '''
            # --- End of commented out st_canvas block ---

        except Exception as display_e:
             st.error(f"Error during image display attempt: {display_e}")
             logger.error(f"Error in display block: {display_e}", exc_info=True)

        # Display current ROI coordinates (remains the same)
        if st.session_state.roi_coords:
            roi = st.session_state.roi_coords
            st.caption(f"Current ROI: ({roi['left']}, {roi['top']}) Size: {roi['width']}x{roi['height']}")
        else:
            st.caption("No ROI selected. Analysis will cover the entire image.")

        st.markdown("---")

        # Display DICOM Metadata (remains the same)
        if st.session_state.is_dicom and st.session_state.dicom_metadata:
            with st.expander("📄 View DICOM Metadata", expanded=False):
                if UI_COMPONENTS_AVAILABLE:
                    display_dicom_metadata(st.session_state.dicom_metadata)
                else:
                    st.json(st.session_state.dicom_metadata)
        elif st.session_state.is_dicom:
            st.caption("DICOM file loaded, but metadata could not be extracted or is empty.")

    elif uploaded_file is not None:
        # This suggests the file was uploaded but processing failed to set display_image
        st.error("Image preview failed. Processing may have encountered an error. Check logs.")
        logger.warning("display_img is None even though uploaded_file exists. Checking processing logs is advised.")
    else: # Default message when no image is loaded
        st.info("⬅️ Please upload a de-identified image or enable Demo Mode in the sidebar.")
    # --- ^ ^ ^ --- END OF DEBUGGING IMAGE DISPLAY --- ^ ^ ^ ---


# --- Column 2: Analysis Results & Interaction Tabs ---
# (Keeping this logic as it is)
with col2:
    # ... [Existing Tabs Code] ...
    st.subheader("📊 Analysis & Interaction")
    tab_titles = [
        "🔬 Structured Analysis",
        "💬 Q&A History",
        "🩺 Condition Focus",
        "🧪 LLM Self-Assessment",
        "🌐 Translation"
    ]
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        st.caption("Displays the AI's general structured analysis of the image/ROI.")
        analysis_text = st.session_state.initial_analysis or "Run 'Structured Initial Analysis' from the sidebar."
        st.markdown(analysis_text)

    with tabs[1]:
        st.caption("Shows the latest answer and full conversation history.")
        st.markdown("**Latest AI Answer:**")
        latest_answer = st.session_state.qa_answer or "_Ask a question using the sidebar controls._"
        st.markdown(latest_answer)
        st.markdown("---")
        if st.session_state.history:
            with st.expander("Full Conversation History", expanded=True):
                for i, (q_type, message) in enumerate(st.session_state.history):
                    if q_type.lower() == "user question":
                        st.markdown(f"**You:** {message}")
                    elif q_type.lower() == "ai answer":
                        st.markdown(f"**AI:** {message}")
                    elif "[fallback]" in q_type.lower():
                        st.markdown(f"**AI (Fallback):** {message.split('**')[-1]}")
                    elif q_type.lower() == "system":
                        st.info(f"*{message}*", icon="ℹ️")
                    else:
                        st.markdown(f"**{q_type}:** {message}")
                    if i < len(st.session_state.history) - 1:
                        st.markdown("---")
        else:
            st.caption("No questions asked in this session yet.")

    with tabs[2]:
        st.caption("Displays the AI's analysis focused on the selected condition.")
        condition_text = st.session_state.disease_analysis or "Select a condition and run 'Analyze for Condition' from the sidebar."
        st.markdown(condition_text)

    with tabs[3]:
        st.caption("EXPERIMENTAL: Displays the AI's self-assessment score. Not clinical confidence.")
        st.warning("""
            **⚠️ Important Note:** This score reflects the AI model's internal assessment based on its training and the current interaction context.
            It is **highly experimental** and **DOES NOT represent clinical certainty or diagnostic accuracy.**
            Use this score for informational insight into the AI's perspective only, and **treat it with extreme caution.**
        """, icon="🧪")
        confidence_text = st.session_state.confidence_score or "Run 'Estimate LLM Self-Assessment' from the sidebar after performing analysis."
        st.markdown(confidence_text)

    with tabs[4]:
        st.subheader("🌐 Translate Analysis Text")
        if not TRANSLATION_AVAILABLE:
            st.warning("Translation features are unavailable. The required 'deep-translator' library might be missing or failed to install.", icon="🚫")
        else:
            # (Translation UI logic remains the same)
            st.caption("Select analysis text, choose languages, and click 'Translate'.")
            text_options = {
                "Structured Initial Analysis": st.session_state.initial_analysis,
                "Latest Q&A Answer": st.session_state.qa_answer,
                "Condition Analysis": st.session_state.disease_analysis,
                "LLM Self-Assessment": st.session_state.confidence_score,
                "(Enter Custom Text Below)": ""
            }
            available_labels = [label for label, txt in text_options.items() if txt or label == "(Enter Custom Text Below)"]
            if not available_labels: available_labels = ["(Enter Custom Text Below)"]

            selected_label = st.selectbox(
                "Select text to translate:", options=available_labels, index=0,
                key="translate_source_select"
            )
            text_to_translate = text_options.get(selected_label, "")

            if selected_label == "(Enter Custom Text Below)":
                text_to_translate = st.text_area(
                    "Enter text to translate:", value="", height=150,
                    key="translate_custom_input"
                )

            st.text_area(
                "Text selected/entered for translation:", value=text_to_translate, height=100,
                disabled=True, key="translate_preview"
            )

            col_lang1, col_lang2 = st.columns(2)
            with col_lang1:
                source_language_options = [AUTO_DETECT_INDICATOR] + sorted(list(LANGUAGE_CODES.keys()))
                source_language_name = st.selectbox(
                    "Source Language:", source_language_options, index=0,
                    key="translate_source_lang"
                )
            with col_lang2:
                target_language_options = sorted(list(LANGUAGE_CODES.keys()))
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

            if st.button("🔄 Translate Now", key="translate_button"):
                st.session_state.translation_result = None
                st.session_state.translation_error = None

                if not text_to_translate or not text_to_translate.strip():
                    st.warning("Please select or enter text to translate first.", icon="☝️")
                    st.session_state.translation_error = "Input text is empty."
                elif source_language_name == target_language_name and source_language_name != AUTO_DETECT_INDICATOR:
                    st.info("Source and target languages are the same. No translation performed.", icon="✅")
                    st.session_state.translation_result = text_to_translate
                else:
                    with st.spinner(f"Translating from '{source_language_name}' to '{target_language_name}'..."):
                        try:
                            translation_output = translate(
                                text=text_to_translate,
                                target_language=target_language_name,
                                source_language=source_language_name
                            )
                            if translation_output is not None:
                                st.session_state.translation_result = translation_output
                                st.success("Translation complete!", icon="🎉")
                            else:
                                st.error("Translation service returned an empty result. Please check the input or try again.", icon="❓")
                                logger.warning("Translation function returned None.")
                                st.session_state.translation_error = "Translation service returned no result."
                        except Exception as e:
                            st.error(f"Translation failed: {e}", icon="❌")
                            logger.error(f"Translation error during execution: {e}", exc_info=True)
                            st.session_state.translation_error = str(e)

            if st.session_state.get("translation_result"):
                formatted_result = format_translation(st.session_state.translation_result)
                st.text_area("Translated Text:", value=formatted_result, height=200, key="translation_output_display")
            elif st.session_state.get("translation_error"):
                st.info(f"Translation Error: {st.session_state.translation_error}", icon="ℹ️")


# --- Button Action Handlers (Centralized Logic) ---
# (Keeping this logic as it is, including the corrected call to run_llm_self_assessment)
current_action = st.session_state.get("last_action")
if current_action:
    # ... [ Pre-Action Checks remain the same ] ...
    logger.info(f"Handling action: '{current_action}' for session: {st.session_state.session_id}")
    action_requires_image = current_action in ["analyze", "ask", "disease", "confidence", "generate_report_data"]
    action_requires_llm = current_action in ["analyze", "ask", "disease", "confidence"]
    action_requires_report_util = (current_action == "generate_report_data")
    error_occurred = False

    if action_requires_image and not isinstance(st.session_state.get("processed_image"), Image.Image):
        st.error(f"Cannot perform '{current_action}': No valid image loaded.", icon="🖼️"); error_occurred = True
    if not st.session_state.session_id:
        st.error("Critical Error: Session ID is missing. Please refresh.", icon="🆔"); error_occurred = True
    if action_requires_llm and not LLM_INTERACTIONS_AVAILABLE:
        st.error(f"Cannot perform '{current_action}': Core AI interaction module unavailable.", icon="🤖"); error_occurred = True
    if action_requires_report_util and not REPORT_UTILS_AVAILABLE:
        st.error(f"Cannot perform '{current_action}': Report generation module unavailable.", icon="📄"); error_occurred = True

    if error_occurred:
        st.session_state.last_action = None; st.stop()

    img_for_llm = st.session_state.processed_image
    roi_coords = st.session_state.roi_coords
    current_history = st.session_state.history

    try:
        # ... [ Action Execution logic for analyze, ask, disease remains the same ] ...
        if current_action == "analyze":
            st.toast("🔬 Performing initial structured analysis...", icon="⏳")
            with st.spinner("AI analyzing image structure and findings..."):
                analysis_result = run_initial_analysis(img_for_llm, roi=roi_coords)
            st.session_state.initial_analysis = analysis_result; st.session_state.qa_answer = ""; st.session_state.disease_analysis = ""
            logger.info("Initial analysis action completed."); st.success("Initial structured analysis complete!", icon="✅")
        elif current_action == "ask":
            question_text = st.session_state.question_input_widget.strip()
            if not question_text: st.warning("Question input was empty.", icon="❓")
            else:
                st.toast(f"Asking AI: '{question_text[:50]}...'"); st.session_state.qa_answer = ""
                with st.spinner("AI formulating answer..."):
                    answer, success_flag = run_multimodal_qa(img_for_llm, question_text, current_history, roi=roi_coords)
                if success_flag:
                    st.session_state.qa_answer = answer; st.session_state.history.append(("User Question", question_text)); st.session_state.history.append(("AI Answer", answer))
                    st.success("AI answered your question!", icon="💬")
                else:
                    primary_error_msg = f"Primary AI failed to answer: {answer}"; st.session_state.qa_answer = primary_error_msg; st.error(primary_error_msg, icon="⚠️"); logger.warning(f"Primary Q&A failed: {answer}")
                    hf_token = os.environ.get("HF_API_TOKEN") or st.secrets.get("HF_API_TOKEN")
                    if HF_MODELS_AVAILABLE and hf_token:
                        st.info(f"Attempting fallback VQA with Hugging Face model: {HF_VQA_MODEL_ID}", icon="🔄")
                        with st.spinner(f"Trying fallback model ({HF_VQA_MODEL_ID})..."):
                            try: fallback_answer, fallback_success = query_hf_vqa_inference_api(img_for_llm, question_text, roi=roi_coords)
                            except Exception as hf_e: fallback_success = False; fallback_answer = f"Error during fallback query: {hf_e}"; logger.error(f"Fallback VQA query error: {hf_e}", exc_info=True)
                        if fallback_success:
                            fallback_display = f"**[Fallback: {HF_VQA_MODEL_ID}]**\n\n{fallback_answer}"; st.session_state.qa_answer += f"\n\n---\n\n{fallback_display}"
                            st.session_state.history.append(("[Fallback] User Question", question_text)); st.session_state.history.append(("[Fallback] AI Answer", fallback_display)); st.success("Fallback AI provided an answer.", icon="👍")
                        else:
                            fallback_error_msg = f"[Fallback Error - {HF_VQA_MODEL_ID}]: {fallback_answer}"; st.session_state.qa_answer += f"\n\n---\n\n{fallback_error_msg}"; st.error("Fallback AI also failed.", icon="👎"); logger.warning(f"Fallback VQA failed: {fallback_answer}")
                    elif HF_MODELS_AVAILABLE and not hf_token: no_token_msg = "[Fallback Skipped: Hugging Face API Token (HF_API_TOKEN) not configured in environment/secrets]"; st.session_state.qa_answer += f"\n\n---\n\n{no_token_msg}"; st.warning("Hugging Face API token needed for fallback VQA.", icon="🔑")
                    else: no_fallback_msg = "[Fallback VQA Unavailable]"; st.session_state.qa_answer += f"\n\n---\n\n{no_fallback_msg}"
        elif current_action == "disease":
            selected_disease = st.session_state.disease_select_widget
            if not selected_disease: st.warning("No condition was selected.", icon="🏷️")
            else:
                st.toast(f"🩺 Analyzing image specifically for '{selected_disease}'...", icon="⏳")
                with st.spinner(f"AI analyzing for signs of {selected_disease}..."):
                    disease_result = run_disease_analysis(img_for_llm, selected_disease, roi=roi_coords)
                st.session_state.disease_analysis = disease_result; st.session_state.qa_answer = ""
                logger.info(f"Disease-specific analysis action for '{selected_disease}' completed."); st.success(f"Analysis focused on '{selected_disease}' complete!", icon="✅")

        elif current_action == "confidence":
            if not current_history: st.warning("Please ask at least one question before estimating the AI's self-assessment.", icon="📊")
            else:
                st.toast("🧪 Estimating LLM self-assessment (Experimental)...", icon="⏳")
                with st.spinner("AI assessing its previous responses..."):
                    assessment_result = run_llm_self_assessment(image=img_for_llm, history=current_history, roi=roi_coords)
                st.session_state.confidence_score = assessment_result
                st.success("LLM self-assessment estimation complete!", icon="✅")

        elif current_action == "generate_report_data":
            # (Report generation logic remains the same)
            st.toast("📄 Generating PDF report data...", icon="⏳")
            st.session_state.pdf_report_bytes = None
            image_for_report = st.session_state.get("display_image")
            if not isinstance(image_for_report, Image.Image): st.error("Cannot generate report: No valid image currently loaded.", icon="🖼️")
            else:
                final_image_for_pdf = image_for_report.copy().convert("RGB")
                if roi_coords:
                    try:
                        draw = ImageDraw.Draw(final_image_for_pdf); x0, y0 = roi_coords['left'], roi_coords['top']; x1, y1 = x0 + roi_coords['width'], y0 + roi_coords['height']
                        draw.rectangle([x0, y0, x1, y1], outline="red", width=max(3, int(min(final_image_for_pdf.size) * 0.005)))
                        logger.info("ROI bounding box drawn onto image for PDF report.")
                    except Exception as draw_e: logger.error(f"Error drawing ROI box on PDF image: {draw_e}", exc_info=True); st.warning("Could not draw the ROI box on the report image.", icon="✏️")
                formatted_history = "No Q&A interactions recorded for this session."
                if current_history: lines = []; [lines.append(f"[{q_type}]:\n{re.sub('<[^<]+?>', '', str(msg)).strip()}") for q_type, msg in current_history]; formatted_history = "\n\n---\n\n".join(lines)
                report_data = { "Session ID": st.session_state.session_id, "Image Filename": (st.session_state.uploaded_file_info or "N/A").split('-')[0], "Structured Initial Analysis": st.session_state.initial_analysis or "Not Performed", "Q&A History": formatted_history, "Condition Specific Analysis": st.session_state.disease_analysis or "Not Performed", "LLM Self-Assessment (Experimental)": st.session_state.confidence_score or "Not Performed" }
                if st.session_state.is_dicom and st.session_state.dicom_metadata:
                    meta_summary = { tag: st.session_state.dicom_metadata.get(tag, "N/A") for tag in ['PatientName', 'PatientID', 'StudyDate', 'Modality', 'StudyDescription', 'InstitutionName'] if tag in st.session_state.dicom_metadata }
                    if meta_summary: lines = [f"{k.replace('PatientName', 'Patient Name').replace('PatientID', 'Patient ID').replace('StudyDate', 'Study Date').replace('StudyDescription', 'Study Desc.')}: {v}" for k, v in meta_summary.items()]; report_data["DICOM Summary"] = "\n".join(lines)
                with st.spinner("Compiling PDF report..."): pdf_bytes = generate_pdf_report_bytes( session_id=st.session_state.session_id, image=final_image_for_pdf, analysis_outputs=report_data, dicom_metadata=st.session_state.dicom_metadata if st.session_state.is_dicom else None )
                if pdf_bytes: st.session_state.pdf_report_bytes = pdf_bytes; st.success("PDF report data generated! Download button available in the sidebar.", icon="📄"); logger.info("PDF report generation successful."); st.balloons()
                else: st.error("Failed to generate PDF report data. Check logs for details.", icon="❌"); logger.error("PDF generation function returned None or empty bytes.")

        else:
            st.warning(f"Unknown action '{current_action}' was triggered.", icon="❓"); logger.warning(f"Unhandled action '{current_action}' encountered.")

    except Exception as e:
        st.error(f"An unexpected error occurred while processing '{current_action}': {e}", icon="💥")
        logger.critical(f"Error during action '{current_action}': {e}", exc_info=True)

    finally:
        st.session_state.last_action = None
        logger.debug(f"Action '{current_action}' processing finished.")
        st.rerun()

# --- Footer ---
# (Keeping this as is)
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced | Session ID: {st.session_state.get('session_id', 'N/A')}")
st.markdown(
    """ <footer> ... [Existing Footer HTML] ... </footer> """,
    unsafe_allow_html=True
)
logger.info(f"--- Application render cycle complete for session: {st.session_state.session_id} ---")
# --- End of app.py ---