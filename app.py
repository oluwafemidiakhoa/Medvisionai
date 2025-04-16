# -*- coding: utf-8 -*-
"""
app.py - Main Streamlit application for RadVision AI Advanced.

Includes debugging steps for image display issue and monkey-patch fix.
Ensures logger is defined before monkey-patch attempts to use it.
Corrected try-except block in translation logic.
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
import logging  # Import logging first
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
logger = logging.getLogger(__name__)  # Define the main logger
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
    logger.info(f"Pillow (PIL) Version: {PIL_VERSION}")  # Log PIL version now
except ImportError:
    logger.critical("Pillow (PIL) is not installed (`pip install Pillow`). Cannot continue.")  # Use logger
    st.error("CRITICAL ERROR: Pillow (PIL) is not installed (`pip install Pillow`). Image processing disabled.")
    PIL_AVAILABLE = False
    st.stop()

# --- v v v --- MONKEY PATCH FOR st_image.image_to_url --- v v v ---
# Apply this after core imports, PIL, and logging setup
try:
    import streamlit.elements.image as st_image

    if not hasattr(st_image, "image_to_url"):
        logger.info("Applying monkey-patch for missing 'streamlit.elements.image.image_to_url'...")  # Use logger

        def image_to_url_monkey_patch(
            image: Any,
            width: int = -1,
            clamp: bool = False,
            channels: str = "RGB",
            output_format: str = "auto",
            image_id: str = "",
        ) -> str:
            """Simplified image_to_url implementation for compatibility."""
            patch_logger = logging.getLogger(__name__ + ".monkey_patch")  # Can use sub-logger if desired
            patch_logger.debug(f"Monkey patch image_to_url called with type: {type(image)}")

            if isinstance(image, Image.Image):
                try:
                    fmt = output_format.upper()
                    fmt = "PNG" if fmt == "AUTO" else fmt
                    if fmt not in ["PNG", "JPEG", "GIF", "WEBP"]:
                        patch_logger.warning(f"Image format {fmt} converting to PNG for data URL.")
                        fmt = "PNG"
                    img_to_save = image
                    if channels == "RGB" and image.mode not in ['RGB', 'L']:
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
    logger.error(f"An error occurred during monkey-patch setup: {e}", exc_info=True)  # Use logger
# --- ^ ^ ^ --- END OF MONKEY PATCH --- ^ ^ ^ ---


# --- Ensure deep-translator is installed ---
# (Keeping this logic as it is)
try:
    from deep_translator import GoogleTranslator
    DEEP_TRANSLATOR_INSTALLED = True
except ImportError:
    DEEP_TRANSLATOR_INSTALLED = False
    try:
        logger.info("Attempting to install deep-translator...")  # Use logger
        subprocess.check_call([sys.executable, "-m", "pip", "install", "deep-translator"])
        from deep_translator import GoogleTranslator
        DEEP_TRANSLATOR_INSTALLED = True
        logger.info("deep-translator installed successfully.")  # Use logger
    except Exception as e:
        logger.critical(f"Could not install deep-translator: {e}", exc_info=True)  # Use logger

# --- Dependency Checks & Imports (Continued) ---
# (Imports remain the same)

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
    try:
        import pylibjpeg
        logger.info("pylibjpeg found.")
    except ImportError:
        logger.info("pylibjpeg not found (optional).")
    try:
        import gdcm
        logger.info("python-gdcm found.")
    except ImportError:
        logger.info("python-gdcm not found (optional).")
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
        run_llm_self_assessment  # Using the corrected name
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
    def display_dicom_metadata(metadata):
        st.caption("Metadata Preview:")
        st.json(dict(list(metadata.items())[:5]))
    def dicom_wl_sliders(wc, ww):
        st.caption("W/L sliders unavailable.")
        return wc, ww

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
st.markdown(""" <style> ... [Existing CSS] ... </style> """, unsafe_allow_html=True)

# --- Display Hero Logo ---
logo_path = os.path.join("assets", "radvisionai-hero.jpeg")
if os.path.exists(logo_path):
    st.image(logo_path, width=350)
else:
    logger.warning(f"Hero logo not found at: {logo_path}.")

# --- Initialize Session State Defaults ---
DEFAULT_STATE = {  # ... [Existing keys/values] ...
    "uploaded_file_info": None,
    "raw_image_bytes": None,
    "is_dicom": False,
    "dicom_dataset": None,
    "dicom_metadata": {},
    "processed_image": None,
    "display_image": None,
    "session_id": None,
    "history": [],
    "initial_analysis": "",
    "qa_answer": "",
    "disease_analysis": "",
    "confidence_score": "",
    "last_action": None,
    "pdf_report_bytes": None,
    "canvas_drawing": None,
    "roi_coords": None,
    "current_display_wc": None,
    "current_display_ww": None,
    "clear_roi_feedback": False,
    "demo_loaded": False,
    "translation_result": None,
    "translation_error": None,
}
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.session_id = str(uuid.uuid4())[:8]
    logger.info(f"New session initialized: {st.session_state.session_id}")
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value
if not isinstance(st.session_state.get("history"), list):
    st.session_state.history = []
logger.debug(f"Session state verified for session ID: {st.session_state.session_id}")

# --- Utility Functions ---
def format_translation(translated_text: Optional[str]) -> str:
    if translated_text is None:
        return "Translation not available or failed."
    try:
        text_str = str(translated_text)
        formatted_text = re.sub(r'\s+(\d+\.)', r'\n\n\1', text_str)
        return formatted_text.strip()
    except Exception as e:
        logger.error(f"Error formatting translation: {e}", exc_info=True)
        return str(translated_text)

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
    demo_mode = st.checkbox("🚀 Demo Mode", value=st.session_state.get("demo_loaded", False), help="Load a sample chest X-ray image and analysis.")
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
        new_wc, new_ww = dicom_wl_sliders(st.session_state.current_display_wc, st.session_state.current_display_ww)
        if new_wc != st.session_state.current_display_wc or new_ww != st.session_state.current_display_ww:
            logger.info(f"DICOM W/L changed via UI: WC={new_wc}, WW={new_ww}")
            st.session_state.current_display_wc = new_wc
            st.session_state.current_display_ww = new_ww
            if st.session_state.dicom_dataset:
                with st.spinner("Applying new Window/Level..."):
                    try:
                        new_display_img = dicom_to_image(st.session_state.dicom_dataset, wc=new_wc, ww=new_ww)
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
    if st.button("🔬 Run Structured Initial Analysis", key="analyze_btn", disabled=action_disabled, help="Perform a general, structured analysis..."):
        st.session_state.last_action = "analyze"
        st.rerun()
    st.subheader("❓ Ask AI a Question")
    question_input = st.text_area("Enter your question about the image:", height=100, key="question_input_widget", placeholder="E.g., 'Describe findings...'", disabled=action_disabled)
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
    disease_select = st.selectbox("Select condition for focused analysis:", options=[""] + sorted(DISEASE_OPTIONS), key="disease_select_widget", disabled=action_disabled, help="AI will analyze for signs related to this condition...")
    if st.button("🩺 Analyze for Condition", key="disease_btn", disabled=action_disabled):
        if disease_select:
            st.session_state.last_action = "disease"
            st.rerun()
        else:
            st.warning("Please select a condition.")
    st.markdown("---")
    st.header("📊 Reporting & Assessment")
    can_estimate = bool(st.session_state.history)
    if st.button("🧪 Estimate LLM Self-Assessment (Experimental)", key="confidence_btn", disabled=not can_estimate or action_disabled, help="EXPERIMENTAL: Ask LLM to assess its last Q&A response..."):
        st.session_state.last_action = "confidence"
        st.rerun()
    report_generation_disabled = action_disabled or not REPORT_UTILS_AVAILABLE
    if st.button("📄 Generate PDF Report Data", key="generate_report_data_btn", disabled=report_generation_disabled, help="Compile analysis into PDF data..."):
        st.session_state.last_action = "generate_report_data"
        st.rerun()
    if st.session_state.get("pdf_report_bytes"):
        report_filename = f"RadVisionAI_Report_{st.session_state.session_id or 'session'}.pdf"
        st.download_button(label="⬇️ Download PDF Report", data=st.session_state.pdf_report_bytes, file_name=report_filename, mime="application/pdf", key="download_pdf_button", help="Download the generated PDF report.")

# --- File Upload Logic ---
if uploaded_file is not None and PIL_AVAILABLE:
    try:
        uploaded_file.seek(0)
        file_content_hash = hashlib.sha256(uploaded_file.read()).hexdigest()[:16]
        uploaded_file.seek(0)
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{file_content_hash}"
    except Exception as e:
        logger.warning(f"Could not generate hash: {e}")
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{uuid.uuid4().hex[:8]}"
    if new_file_info != st.session_state.get("uploaded_file_info"):
        logger.info(f"New file upload: {uploaded_file.name} ({uploaded_file.size} bytes)")
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
        st.session_state.is_dicom = (PYDICOM_AVAILABLE and DICOM_UTILS_AVAILABLE and
                                      ("dicom" in uploaded_file.type.lower() or file_ext in (".dcm", ".dicom")))
        with st.spinner("🔬 Analyzing and preparing image..."):
            temp_display_img = None
            temp_processed_img = None
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
                        temp_display_img = dicom_to_image(dicom_dataset, wc=default_wc, ww=default_ww)
                        temp_processed_img = dicom_to_image(dicom_dataset, wc=None, ww=None, normalize=True)
                        if isinstance(temp_display_img, Image.Image) and isinstance(temp_processed_img, Image.Image):
                            processing_success = True
                            logger.info("DICOM processed successfully.")
                        else:
                            st.error("Failed to convert DICOM pixels.")
                            logger.error("dicom_to_image invalid return.")
                    else:
                        st.error("Could not parse DICOM.")
                        logger.error("parse_dicom returned None.")
                except pydicom.errors.InvalidDicomError:
                    st.error("Invalid DICOM format.")
                    logger.error("InvalidDicomError.")
                    st.session_state.is_dicom = False
                except Exception as e:
                    st.error(f"DICOM processing error: {e}")
                    logger.error(f"DICOM error: {e}", exc_info=True)
                    st.session_state.is_dicom = False
            if not st.session_state.is_dicom and not processing_success:
                logger.info("Processing as standard image...")
                try:
                    raw_img = Image.open(io.BytesIO(st.session_state.raw_image_bytes))
                    processed_img = raw_img.convert("RGB")
                    temp_display_img = processed_img.copy()
                    temp_processed_img = processed_img.copy()
                    processing_success = True
                    logger.info("Standard image loaded.")
                except UnidentifiedImageError:
                    st.error("Cannot identify image format.")
                    logger.error(f"UnidentifiedImageError: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    logger.error(f"Standard image error: {e}", exc_info=True)
            logger.info(f"Image processing completion status: {processing_success}")
            if processing_success and isinstance(temp_display_img, Image.Image) and isinstance(temp_processed_img, Image.Image):
                st.session_state.display_image = temp_display_img.convert('RGB') if temp_display_img.mode != 'RGB' else temp_display_img
                st.session_state.processed_image = temp_processed_img
                st.success(f"✅ Image '{uploaded_file.name}' loaded!")
                logger.info(f"State updated for: {uploaded_file.name}")
                st.rerun()
            else:
                st.error("Image loading failed.")
                logger.error(f"Processing failed: {uploaded_file.name} (success={processing_success})")
                st.session_state.uploaded_file_info = None
                st.session_state.display_image = None
                st.session_state.processed_image = None
                st.session_state.is_dicom = False

# --- Button Action Handlers (Centralized Logic) ---
current_action = st.session_state.get("last_action")
if current_action:
    # Pre-Action Checks
    logger.info(f"Handling action: '{current_action}' for session: {st.session_state.session_id}")
    action_requires_image = current_action in ["analyze", "ask", "disease", "confidence", "generate_report_data"]
    action_requires_llm = current_action in ["analyze", "ask", "disease", "confidence"]
    action_requires_report_util = (current_action == "generate_report_data")
    error_occurred = False
    if action_requires_image and not isinstance(st.session_state.get("processed_image"), Image.Image):
        st.error(f"No valid image for '{current_action}'.", icon="🖼️")
        error_occurred = True
    if not st.session_state.session_id:
        st.error("Session ID missing.", icon="🆔")
        error_occurred = True
    if action_requires_llm and not LLM_INTERACTIONS_AVAILABLE:
        st.error(f"AI module unavailable for '{current_action}'.", icon="🤖")
        error_occurred = True
    if action_requires_report_util and not REPORT_UTILS_AVAILABLE:
        st.error(f"Report module unavailable for '{current_action}'.", icon="📄")
        error_occurred = True
    if error_occurred:
        st.session_state.last_action = None
        st.stop()

    img_for_llm = st.session_state.processed_image
    roi_coords = st.session_state.roi_coords
    current_history = st.session_state.history
    try:
        if current_action == "analyze":
            st.toast("🔬 Analyzing...", icon="⏳")
            with st.spinner("AI analyzing..."):
                analysis_result = run_initial_analysis(img_for_llm, roi=roi_coords)
                st.session_state.initial_analysis = analysis_result
                st.session_state.qa_answer = ""
                st.session_state.disease_analysis = ""
                logger.info("Initial analysis done.")
                st.success("Analysis complete!", icon="✅")
        elif current_action == "ask":
            question_text = st.session_state.question_input_widget.strip()
            if not question_text:
                st.warning("Question empty.", icon="❓")
            else:
                st.toast(f"Asking: '{question_text[:50]}...'", icon="⏳")
                st.session_state.qa_answer = ""
                with st.spinner("Thinking..."):
                    answer, success_flag = run_multimodal_qa(img_for_llm, question_text, current_history, roi=roi_coords)
                if success_flag:
                    st.session_state.qa_answer = answer
                    st.session_state.history.append(("User Question", question_text))
                    st.session_state.history.append(("AI Answer", answer))
                    st.success("AI answered!", icon="💬")
                else:
                    primary_error_msg = f"Primary AI failed: {answer}"
                    st.session_state.qa_answer = primary_error_msg
                    st.error(primary_error_msg, icon="⚠️")
                    logger.warning(f"Q&A failed: {answer}")
                    hf_token = os.environ.get("HF_API_TOKEN") or st.secrets.get("HF_API_TOKEN")
                    if HF_MODELS_AVAILABLE and hf_token:
                        st.info(f"Trying fallback: {HF_VQA_MODEL_ID}", icon="🔄")
                        with st.spinner("Fallback..."):
                            try:
                                fallback_answer, fallback_success = query_hf_vqa_inference_api(img_for_llm, question_text, roi=roi_coords)
                            except Exception as hf_e:
                                fallback_success = False
                                fallback_answer = f"Err: {hf_e}"
                                logger.error(f"Fallback err: {hf_e}", exc_info=True)
                        if fallback_success:
                            fallback_display = f"**[Fallback: {HF_VQA_MODEL_ID}]**\n\n{fallback_answer}"
                            st.session_state.qa_answer += f"\n\n---\n\n{fallback_display}"
                            st.session_state.history.append(("[Fallback] User Q", question_text))
                            st.session_state.history.append(("[Fallback] AI A", fallback_display))
                            st.success("Fallback answered.", icon="👍")
                        else:
                            fallback_error_msg = f"[Fallback Err - {HF_VQA_MODEL_ID}]: {fallback_answer}"
                            st.session_state.qa_answer += f"\n\n---\n\n{fallback_error_msg}"
                            st.error("Fallback failed.", icon="👎")
                            logger.warning(f"Fallback failed: {fallback_answer}")
                    elif HF_MODELS_AVAILABLE and not hf_token:
                        no_token_msg = "[Fallback Skip: HF_TOKEN missing]"
                        st.session_state.qa_answer += f"\n\n---\n\n{no_token_msg}"
                        st.warning("HF token needed.", icon="🔑")
                    else:
                        no_fallback_msg = "[Fallback N/A]"
                        st.session_state.qa_answer += f"\n\n---\n\n{no_fallback_msg}"
        elif current_action == "disease":
            selected_disease = st.session_state.disease_select_widget
            if not selected_disease:
                st.warning("No condition selected.", icon="🏷️")
            else:
                st.toast(f"🩺 Analyzing for '{selected_disease}'...", icon="⏳")
                with st.spinner("AI analyzing..."):
                    disease_result = run_disease_analysis(img_for_llm, selected_disease, roi=roi_coords)
                st.session_state.disease_analysis = disease_result
                st.session_state.qa_answer = ""
                logger.info(f"Disease analysis '{selected_disease}' done.")
                st.success(f"Analysis for '{selected_disease}' complete!", icon="✅")
        elif current_action == "confidence":
            if not current_history:
                st.warning("Ask a question first.", icon="📊")
            else:
                st.toast("🧪 Estimating assessment...", icon="⏳")
                with st.spinner("AI assessing..."):
                    assessment_result = run_llm_self_assessment(image=img_for_llm, history=current_history, roi=roi_coords)
                st.session_state.confidence_score = assessment_result
                st.success("Self-assessment complete!", icon="✅")
        elif current_action == "generate_report_data":
            st.toast("📄 Generating PDF...", icon="⏳")
            st.session_state.pdf_report_bytes = None
            image_for_report = st.session_state.get("display_image")
            if not isinstance(image_for_report, Image.Image):
                st.error("No image for report.", icon="🖼️")
            else:
                final_image_for_pdf = image_for_report.copy().convert("RGB")
                if roi_coords:
                    try:
                        draw = ImageDraw.Draw(final_image_for_pdf)
                        x0, y0, w, h = roi_coords['left'], roi_coords['top'], roi_coords['width'], roi_coords['height']
                        draw.rectangle([x0, y0, x0+w, y0+h], outline="red", width=max(3, int(min(final_image_for_pdf.size)*0.005)))
                        logger.info("ROI drawn.")
                    except Exception as draw_e:
                        logger.error(f"ROI draw err: {draw_e}", exc_info=True)
                        st.warning("Could not draw ROI.", icon="✏️")
                formatted_history = "No Q&A."
                if current_history:
                    lines = []
                    for qt, m in current_history:
                        lines.append(f"[{qt}]:\n{re.sub('<[^<]+?>','',str(m)).strip()}")
                    formatted_history = "\n\n---\n\n".join(lines)
                report_data = {
                    "Session ID": st.session_state.session_id,
                    "Image Filename": (st.session_state.uploaded_file_info or "N/A").split('-')[0],
                    "Structured Initial Analysis": st.session_state.initial_analysis or "N/P",
                    "Q&A History": formatted_history,
                    "Condition Specific Analysis": st.session_state.disease_analysis or "N/P",
                    "LLM Self-Assessment (Experimental)": st.session_state.confidence_score or "N/P"
                }
                if st.session_state.is_dicom and st.session_state.dicom_metadata:
                    meta_tags = ['PatientName','PatientID','StudyDate','Modality','StudyDescription','InstitutionName']
                    meta_summary = {t: st.session_state.dicom_metadata.get(t, "N/A") for t in meta_tags if t in st.session_state.dicom_metadata}
                    if meta_summary:
                        lines = [f"{k.replace('PatientName','Pt Name').replace('PatientID','Pt ID').replace('StudyDate','Date').replace('StudyDescription','Desc.')}: {v}" for k, v in meta_summary.items()]
                        report_data["DICOM Summary"] = "\n".join(lines)
                with st.spinner("Compiling PDF..."):
                    pdf_bytes = generate_pdf_report_bytes(
                        session_id=st.session_state.session_id,
                        image=final_image_for_pdf,
                        analysis_outputs=report_data,
                        dicom_metadata=st.session_state.dicom_metadata if st.session_state.is_dicom else None
                    )
                if pdf_bytes:
                    st.session_state.pdf_report_bytes = pdf_bytes
                    st.success("PDF ready!", icon="📄")
                    logger.info("PDF generated.")
                    st.balloons()
                else:
                    st.error("PDF generation failed.", icon="❌")
                    logger.error("PDF gen returned None.")
        else:
            st.warning(f"Unknown action '{current_action}'.", icon="❓")
            logger.warning(f"Unhandled action '{current_action}'.")
    except Exception as e:
        st.error(f"Error processing '{current_action}': {e}", icon="💥")
        logger.critical(f"Action '{current_action}' error: {e}", exc_info=True)
    finally:
        st.session_state.last_action = None
        logger.debug(f"Action '{current_action}' complete.")
        st.rerun()

# --- Footer ---
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced | Session ID: {st.session_state.get('session_id', 'N/A')}")
st.markdown(""" <footer> ... [Existing Footer HTML] ... </footer> """, unsafe_allow_html=True)
logger.info(f"--- Application render cycle complete for session: {st.session_state.session_id} ---")
# --- End of app.py ---
