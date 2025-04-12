# -*- coding: utf-8 -*-
"""
app.py - Main Streamlit application for RadVision AI Advanced.

Handles image uploading (DICOM, JPG, PNG), display, ROI selection,
interaction with AI models for analysis and Q&A, translation,
and report generation.

IMPORTANT CHANGES:
- Ensured 'deep-translator' is installed on-the-fly if needed.
- Removed extra warnings about 'deep-translator' not found unless fallback also fails.
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
try:
    from deep_translator import GoogleTranslator
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "deep-translator"])
        from deep_translator import GoogleTranslator
    except Exception as e:
        # If fallback also fails, log it; we'll gracefully disable translation below.
        print(f"CRITICAL: Could not install deep-translator: {e}")

# --- Logging Setup (Early) ---
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

# --- Streamlit Drawable Canvas ---
try:
    from streamlit_drawable_canvas import st_canvas
    import streamlit_drawable_canvas as st_canvas_module
    CANVAS_VERSION = getattr(st_canvas_module, '__version__', 'Unknown')
    logger.info(f"Streamlit Drawable Canvas Version: {CANVAS_VERSION}")
    DRAWABLE_CANVAS_AVAILABLE = True
except ImportError:
    st.error("CRITICAL ERROR: streamlit-drawable-canvas is not installed. Run `pip install streamlit-drawable-canvas`.")
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
    logger.warning(f"Failed to import ui_components: {e}. Custom UI elements might be missing.")
    UI_COMPONENTS_AVAILABLE = False
    def display_dicom_metadata(metadata): st.json(metadata[:5])  # Simple fallback
    def dicom_wl_sliders(wc, ww): return wc, ww

# --- HF fallback for Q&A (Optional) ---
try:
    from hf_models import query_hf_vqa_inference_api, HF_VQA_MODEL_ID
    HF_MODELS_AVAILABLE = True
    logger.info(f"hf_models imported successfully (Fallback VQA Model: {HF_VQA_MODEL_ID}).")
except ImportError:
    HF_VQA_MODEL_ID = "hf_model_not_found"
    HF_MODELS_AVAILABLE = False
    def query_hf_vqa_inference_api(img: Image.Image, question: str, roi: Optional[Dict] = None) -> Tuple[str, bool]:
        logger.warning("query_hf_vqa_inference_api called but hf_models module is unavailable.")
        return "[Fallback VQA Unavailable] HF module not found.", False
    logger.warning("hf_models not found. Fallback VQA disabled.")

# --- Translation Setup ---
# If 'deep-translator' import worked above, let's confirm it with a quick boolean
try:
    # Attempt to import your translation helper module if it depends on deep-translator
    from translation_models import (
        translate, detect_language, LANGUAGE_CODES, AUTO_DETECT_INDICATOR
    )
    # We assume your translation_models also tries `from deep_translator import GoogleTranslator`.
    # If that fails, it should raise ImportError.
    TRANSLATION_AVAILABLE = True
    logger.info("translation_models imported successfully. Translation is available.")
except ImportError as e:
    st.warning(f"Translation features disabled: {e}")
    logger.error(f"Could not import translation_models: {e}", exc_info=True)
    TRANSLATION_AVAILABLE = False
    # Minimal fallback
    translate = None
    detect_language = None
    LANGUAGE_CODES = {"English": "en"}
    AUTO_DETECT_INDICATOR = "Auto-Detect"

# --- Custom CSS for Polished Look & Tab Scrolling ---
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
      .css-1d391kg {
          background-color: #ffffff;
          border-right: 1px solid #e0e0e0;
      }
      .stButton>button {
          border-radius: 8px;
          padding: 0.5rem 1rem;
          font-weight: 500;
          transition: background-color 0.2s ease-in-out, border-color 0.2s ease-in-out;
      }
      .stButton>button:hover {
        filter: brightness(95%);
      }
      div[role="tablist"] {
          overflow-x: auto;
          white-space: nowrap;
          border-bottom: 1px solid #e0e0e0;
          scrollbar-width: thin;
          scrollbar-color: #cccccc #f0f2f6;
      }
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
    st.image(logo_path, width=350)
else:
    logger.warning(f"Hero logo not found at: {logo_path}")
    st.warning("Hero logo (radvisionai-hero.jpeg) not found in 'assets' folder.")

# --- Initialize Session State Defaults ---
DEFAULT_STATE = {
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

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value

if not isinstance(st.session_state.history, list):
    st.session_state.history = []

if not st.session_state.get("session_id"):
    st.session_state.session_id = str(uuid.uuid4())[:8]
logger.debug(f"Session state verified for session ID: {st.session_state.session_id}")

def format_translation(translated_text: Optional[str]) -> str:
    """
    Applies basic formatting to translated text, primarily for readability.
    Handles potential None input gracefully.
    """
    if translated_text is None:
        return "Translation not available or failed."
    try:
        text_str = str(translated_text)
        formatted_text = re.sub(r'\s+(\d+\.)', r'\n\n\1', text_str)
        return formatted_text.strip()
    except Exception as e:
        logger.error(f"Error formatting translation: {e}", exc_info=True)
        return str(translated_text)

# --- Monkey-Patch (Conditional) ---
import streamlit.elements.image as st_image
if not hasattr(st_image, "image_to_url"):
    def image_to_url_monkey_patch(img_obj: Any, width: int = -1, clamp: bool = False,
                                  channels: str = "RGB", output_format: str = "auto",
                                  image_id: str = "") -> str:
        if PIL_AVAILABLE and isinstance(img_obj, Image.Image):
            try:
                buffered = io.BytesIO()
                fmt = "PNG" if output_format.lower() == "auto" else output_format.upper()
                if fmt not in ["PNG", "JPEG"]:
                    fmt = "PNG"
                temp_img = img_obj
                if channels == "RGB" and temp_img.mode not in ['RGB', 'L', 'RGBA']:
                    temp_img = temp_img.convert('RGB')
                elif temp_img.mode == 'P':
                    temp_img = temp_img.convert('RGBA')
                temp_img.save(buffered, format=fmt)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/{fmt.lower()};base64,{img_str}"
            except Exception as e:
                logger.error(f"Monkey-patch image_to_url failed: {e}", exc_info=True)
                return ""
        else:
            logger.warning(f"Unsupported type {type(img_obj)} or PIL unavailable.")
            return ""
    st_image.image_to_url = image_to_url_monkey_patch
    logger.info("Applied monkey-patch for st.elements.image.image_to_url.")

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
    ]
    st.info(f"💡 {random.choice(TIPS)}")
    st.markdown("---")

    # Upload
    st.header("Image Upload & Settings")
    uploaded_file = st.file_uploader(
        "Upload Image (JPG, PNG, DCM)",
        type=["jpg", "jpeg", "png", "dcm", "dicom"],
        key="file_uploader_widget",
        help="Upload a medical image file for analysis. DICOM (.dcm) is preferred."
    )

    # Demo Mode
    demo_mode = st.checkbox("🚀 Demo Mode", value=st.session_state.get("demo_loaded", False),
                            help="Load a sample chest X-ray image and analysis.")

    # Clear ROI
    if st.button("🗑️ Clear ROI", help="Remove the selected ROI"):
        st.session_state.roi_coords = None
        st.session_state.canvas_drawing = None
        st.session_state.clear_roi_feedback = True
        st.rerun()

    if st.session_state.get("clear_roi_feedback"):
        st.success("✅ ROI cleared successfully!")
        st.balloons()
        st.session_state.clear_roi_feedback = False

    # DICOM Window/Level
    if st.session_state.is_dicom and UI_COMPONENTS_AVAILABLE and st.session_state.display_image:
        st.markdown("---")
        st.subheader("DICOM Display")
        new_wc, new_ww = dicom_wl_sliders(
            st.session_state.current_display_wc,
            st.session_state.current_display_ww
        )
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
                        if new_display_img.mode != 'RGB':
                            new_display_img = new_display_img.convert('RGB')
                        st.session_state.display_image = new_display_img
                        st.rerun()
                    else:
                        st.error("Failed to update DICOM image with new W/L.")
                        logger.error("dicom_to_image returned invalid image.")
            else:
                st.warning("DICOM utilities not available to update W/L.")

    st.markdown("---")
    st.header("🤖 AI Analysis Actions")

    action_disabled = not isinstance(st.session_state.get("processed_image"), Image.Image)

    if st.button("▶️ Run Initial Analysis", key="analyze_btn", disabled=action_disabled,
                 help="Perform a general analysis of the entire image or selected ROI."):
        st.session_state.last_action = "analyze"
        st.rerun()

    st.subheader("❓ Ask AI a Question")
    question_input = st.text_area(
        "Enter your question:",
        height=100,
        key="question_input_widget",
        placeholder="E.g., 'Are there any nodules in the upper right lobe?'",
        disabled=action_disabled
    )
    if st.button("💬 Ask Question", key="ask_btn", disabled=action_disabled):
        if question_input.strip():
            st.session_state.last_action = "ask"
            st.rerun()
        else:
            st.warning("Please enter a question before submitting.")

    st.subheader("🎯 Condition-Specific Analysis")
    DISEASE_OPTIONS = [
        "Pneumonia", "Lung Cancer", "Nodule/Mass", "Effusion", "Fracture",
        "Stroke", "Appendicitis", "Bowel Obstruction", "Cardiomegaly",
        "Aortic Aneurysm", "Pulmonary Embolism", "Tuberculosis", "COVID-19",
        "Brain Tumor", "Arthritis", "Osteoporosis",
    ]
    disease_select = st.selectbox(
        "Select condition to focus on:",
        options=[""] + sorted(DISEASE_OPTIONS),
        key="disease_select_widget",
        disabled=action_disabled
    )
    if st.button("🩺 Analyze Condition", key="disease_btn", disabled=action_disabled):
        if disease_select:
            st.session_state.last_action = "disease"
            st.rerun()
        else:
            st.warning("Please select a condition first.")

    st.markdown("---")
    st.header("📊 Confidence & Reporting")

    can_estimate = bool(
        st.session_state.history or
        st.session_state.initial_analysis or
        st.session_state.disease_analysis
    )
    if st.button("📈 Estimate AI Confidence", key="confidence_btn", disabled=not can_estimate or action_disabled):
        if can_estimate:
            st.session_state.last_action = "confidence"
            st.rerun()
        else:
            st.warning("Perform at least one analysis before estimating confidence.")

    report_generation_disabled = action_disabled or not REPORT_UTILS_AVAILABLE
    if st.button("📄 Generate PDF Report Data", key="generate_report_data_btn",
                 disabled=report_generation_disabled):
        st.session_state.last_action = "generate_report_data"
        st.rerun()

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
if uploaded_file is not None:
    try:
        uploaded_file.seek(0)
        file_content_hash = hashlib.sha256(uploaded_file.read()).hexdigest()[:16]
        uploaded_file.seek(0)
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{file_content_hash}"
    except Exception as e:
        logger.warning(f"Could not generate hash for file: {e}")
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{uuid.uuid4().hex[:8]}"

    if new_file_info != st.session_state.get("uploaded_file_info"):
        logger.info(f"New file uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
        st.toast(f"Processing '{uploaded_file.name}'...", icon="⏳")

        keys_to_preserve = {"file_uploader_widget", "session_id", "uploaded_file_info", "demo_loaded"}
        st.session_state.session_id = str(uuid.uuid4())[:8]
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

        with st.spinner("🔬 Analyzing file..."):
            temp_display_img = None
            temp_processed_img = None
            processing_success = False

            if st.session_state.is_dicom:
                logger.info("Processing as DICOM...")
                try:
                    dicom_dataset = parse_dicom(st.session_state.raw_image_bytes, filename=uploaded_file.name)
                    st.session_state.dicom_dataset = dicom_dataset
                    if dicom_dataset:
                        st.session_state.dicom_metadata = extract_dicom_metadata(dicom_dataset)
                        default_wc, default_ww = get_default_wl(dicom_dataset)
                        st.session_state.current_display_wc = default_wc
                        st.session_state.current_display_ww = default_ww
                        temp_display_img = dicom_to_image(dicom_dataset, wc=default_wc, ww=default_ww)
                        temp_processed_img = dicom_to_image(dicom_dataset, wc=None, ww=None, normalize=True)
                        if isinstance(temp_display_img, Image.Image) and isinstance(temp_processed_img, Image.Image):
                            processing_success = True
                            logger.info("DICOM parsed and converted successfully.")
                        else:
                            st.error("Failed to convert DICOM pixel data to image.")
                    else:
                        st.error("Could not parse DICOM file structure.")
                except pydicom.errors.InvalidDicomError:
                    st.error("Invalid DICOM file. Please upload a valid .dcm file.")
                    logger.error("InvalidDicomError while parsing.")
                    st.session_state.is_dicom = False
                except Exception as e:
                    st.error(f"Error processing DICOM: {e}")
                    logger.error(f"DICOM processing error: {e}", exc_info=True)
                    st.session_state.is_dicom = False

            if not st.session_state.is_dicom and not processing_success:
                logger.info("Processing as standard image...")
                if not PIL_AVAILABLE:
                    st.error("Cannot process standard images: Pillow missing.")
                else:
                    try:
                        raw_img = Image.open(io.BytesIO(st.session_state.raw_image_bytes))
                        processed_img = raw_img.convert("RGB")
                        temp_display_img = processed_img.copy()
                        temp_processed_img = processed_img.copy()
                        processing_success = True
                        logger.info("Standard image loaded successfully.")
                    except UnidentifiedImageError:
                        st.error("Could not identify image format. Please upload JPG, PNG, or DICOM.")
                    except Exception as e:
                        st.error(f"Error processing image: {e}")
                        logger.error(f"Standard image processing error: {e}", exc_info=True)

            if processing_success and isinstance(temp_display_img, Image.Image) and isinstance(temp_processed_img, Image.Image):
                if temp_display_img.mode != 'RGB':
                    st.session_state.display_image = temp_display_img.convert('RGB')
                else:
                    st.session_state.display_image = temp_display_img
                st.session_state.processed_image = temp_processed_img
                st.success(f"✅ '{uploaded_file.name}' loaded successfully!")
                logger.info(f"Image processed: {uploaded_file.name}")
                st.rerun()
            else:
                st.error("Image loading failed. Check format or try again.")
                logger.error(f"Image processing failed for file: {uploaded_file.name}")
                st.session_state.uploaded_file_info = None
                st.session_state.display_image = None
                st.session_state.processed_image = None
                st.session_state.is_dicom = False

# --- Display Uploaded Image (Minimal Addition) ---
# If an image has been processed, immediately display it.
if st.session_state.get("display_image") is not None:
    st.image(st.session_state.display_image, caption="Uploaded Image", use_column_width=True)

# --- Main Page ---
st.markdown("---")
st.title("⚕️ RadVision QA Advanced: AI-Assisted Image Analysis")
with st.expander("User Guide & Disclaimer", expanded=False):
    st.warning("⚠️ **Disclaimer**: This tool is for research/educational purposes only and is **NOT** a substitute for professional medical advice or diagnosis.")
    st.markdown("""
    **Workflow:**
    1. **Upload Image**: DICOM, JPG, or PNG (or activate Demo Mode).
    2. **(DICOM)** Adjust Window/Level if needed.
    3. **ROI**: Draw a rectangle to focus the AI if desired.
    4. **AI Analysis**: Use sidebar buttons (Initial Analysis, Ask Question, Condition Analysis).
    5. **Translation**: Translate AI text if needed.
    6. **Confidence**: Estimate AI confidence.
    7. **Generate Report**: Compile PDF with your interactions.
    """)

st.markdown("---")

col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("🖼️ Image Viewer")
    display_img = st.session_state.get("display_image")

    if isinstance(display_img, Image.Image):
        if DRAWABLE_CANVAS_AVAILABLE and st_canvas:
            st.caption("Draw a rectangle below to set an ROI.")
            MAX_CANVAS_WIDTH = 600
            MAX_CANVAS_HEIGHT = 500
            img_w, img_h = display_img.size

            if img_w <= 0 or img_h <= 0:
                st.warning("Invalid image dimensions; cannot draw ROI.")
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
                    background_image=display_img,
                    update_streamlit=True,
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="rect",
                    initial_drawing=st.session_state.get("canvas_drawing", None),
                    key="drawable_canvas"
                )

                if canvas_result.json_data and canvas_result.json_data.get("objects"):
                    last_object = canvas_result.json_data["objects"][-1]
                    if last_object["type"] == "rect":
                        canvas_left = int(last_object["left"])
                        canvas_top = int(last_object["top"])
                        canvas_width_scaled = int(last_object["width"] * last_object.get("scaleX", 1))
                        canvas_height_scaled = int(last_object["height"] * last_object.get("scaleY", 1))
                        scale_x = img_w / canvas_width
                        scale_y = img_h / canvas_height
                        original_left = int(canvas_left * scale_x)
                        original_top = int(canvas_top * scale_y)
                        original_width = int(canvas_width_scaled * scale_x)
                        original_height = int(canvas_height_scaled * scale_y)
                        original_left = max(0, original_left)
                        original_top = max(0, original_top)
                        original_width = min(img_w - original_left, original_width)
                        original_height = min(img_h - original_top, original_height)
                        new_roi = {
                            "left": original_left,
                            "top": original_top,
                            "width": original_width,
                            "height": original_height
                        }
                        if st.session_state.roi_coords != new_roi:
                            st.session_state.roi_coords = new_roi
                            st.session_state.canvas_drawing = canvas_result.json_data
                            logger.info(f"New ROI selected: {new_roi}")
                            st.info(f"ROI: ({original_left},{original_top}) size {original_width}x{original_height}", icon="🎯")
        else:
            st.image(display_img, caption="Image Preview", use_container_width=True)

        if st.session_state.roi_coords:
            roi = st.session_state.roi_coords
            st.caption(f"Current ROI: ({roi['left']}, {roi['top']}) - {roi['width']}x{roi['height']}")

        st.markdown("---")

        if st.session_state.is_dicom and st.session_state.dicom_metadata:
            with st.expander("📄 DICOM Metadata", expanded=False):
                if UI_COMPONENTS_AVAILABLE:
                    display_dicom_metadata(st.session_state.dicom_metadata)
                else:
                    st.json(st.session_state.dicom_metadata)
        elif st.session_state.is_dicom:
            st.caption("DICOM file loaded, but no metadata available.")
    elif uploaded_file is not None:
        st.error("Image preview failed. The file might be corrupted.")
    else:
        st.info("⬅️ Please upload an image or enable Demo Mode in the sidebar.")

with col2:
    st.subheader("📊 Analysis & Results")
    tab_titles = [
        "🔬 Initial Analysis",
        "💬 Q&A History",
        "🩺 Condition Focus",
        "📈 Confidence",
        "🌐 Translation"
    ]
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        st.text_area(
            "Overall Findings & Impressions",
            value=st.session_state.initial_analysis or "Run 'Initial Analysis' to see results here.",
            height=450,
            disabled=True
        )

    with tabs[1]:
        st.text_area(
            "Latest AI Answer",
            value=st.session_state.qa_answer or "Ask a question to see AI's response here.",
            height=200,
            disabled=True
        )
        st.markdown("---")
        if st.session_state.history:
            with st.expander("Full Conversation History", expanded=True):
                for i, (q_type, message) in enumerate(reversed(st.session_state.history)):
                    if q_type.lower() == "user question":
                        st.markdown(f"**You:** {message}")
                    elif q_type.lower() == "ai answer":
                        st.markdown(f"**AI:** {message}")
                    elif q_type.lower() == "system":
                        st.info(f"*{message}*", icon="ℹ️")
                    else:
                        st.markdown(f"**{q_type}:** {message}")
                    if i < len(st.session_state.history) - 1:
                        st.markdown("---")
        else:
            st.caption("No questions asked yet.")

    with tabs[2]:
        st.text_area(
            "Condition-Specific Analysis",
            value=st.session_state.disease_analysis or "Select a condition and click 'Analyze Condition'.",
            height=450,
            disabled=True
        )

    with tabs[3]:
        st.text_area(
            "Estimated AI Confidence",
            value=st.session_state.confidence_score or "Run 'Estimate AI Confidence' after analysis.",
            height=450,
            disabled=True
        )

    with tabs[4]:
        st.subheader("🌐 Translate Analysis Text")

        if not TRANSLATION_AVAILABLE:
            st.warning("Translation features are unavailable. Ensure 'deep-translator' is installed.")
        else:
            st.caption("Select or enter text to translate, choose source/target languages, then click 'Translate'.")
            text_options = {
                "Initial Analysis": st.session_state.initial_analysis,
                "Latest Q&A Answer": st.session_state.qa_answer,
                "Condition Analysis": st.session_state.disease_analysis,
                "Confidence Estimation": st.session_state.confidence_score,
                "(Enter Custom Text Below)": ""
            }
            available_options = {
                label: txt for label, txt in text_options.items() if txt or label == "(Enter Custom Text Below)"
            }
            selected_label = st.selectbox(
                "Select text to translate:",
                list(available_options.keys()),
                index=0
            )
            text_to_translate = available_options.get(selected_label, "")
            if selected_label == "(Enter Custom Text Below)":
                text_to_translate = st.text_area(
                    "Enter text to translate:",
                    value="",
                    height=150
                )
            st.text_area(
                "Text selected for translation:",
                value=text_to_translate,
                height=100,
                disabled=True
            )

            col_lang1, col_lang2 = st.columns(2)
            with col_lang1:
                source_language_options = [AUTO_DETECT_INDICATOR] + sorted(list(LANGUAGE_CODES.keys()))
                source_language_name = st.selectbox(
                    "Source Language:",
                    source_language_options,
                    index=0
                )
            with col_lang2:
                target_language_options = sorted(list(LANGUAGE_CODES.keys()))
                default_target_index = 0
                if "Spanish" in target_language_options:
                    default_target_index = target_language_options.index("Spanish")
                elif "English" in target_language_options:
                    default_target_index = target_language_options.index("English")

                target_language_name = st.selectbox(
                    "Translate To:",
                    target_language_options,
                    index=default_target_index
                )

            if st.button("🔄 Translate Now"):
                st.session_state.translation_result = None
                st.session_state.translation_error = None

                if not text_to_translate.strip():
                    st.warning("Please select or enter some text first.")
                    st.session_state.translation_error = "Input text is empty."
                elif source_language_name == target_language_name and source_language_name != AUTO_DETECT_INDICATOR:
                    st.info("Source and target are the same; no translation needed.")
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
                                st.success("Translation complete!")
                            else:
                                st.error("Translation returned no result. Check logs.")
                                st.session_state.translation_error = "Service returned None."
                        except Exception as e:
                            st.error(f"Unexpected error: {e}")
                            logger.error(f"Translation error: {e}", exc_info=True)
                            st.session_state.translation_error = str(e)

                if st.session_state.get("translation_result"):
                    formatted_result = format_translation(st.session_state.translation_result)
                    st.text_area("Translated Text:", value=formatted_result, height=200)
                elif st.session_state.get("translation_error"):
                    st.info(f"Translation Error: {st.session_state.translation_error}")

# --- Button Action Handlers ---
current_action = st.session_state.get("last_action")
if current_action:
    logger.info(f"Handling action: '{current_action}' for session: {st.session_state.session_id}")
    action_requires_image = current_action not in ["generate_report_data"]
    action_requires_llm = current_action in ["analyze", "ask", "disease", "confidence"]
    action_requires_report_util = (current_action == "generate_report_data")

    if action_requires_image and not isinstance(st.session_state.get("processed_image"), Image.Image):
        st.error(f"No valid image for '{current_action}'. Please upload an image.")
        st.session_state.last_action = None
        st.stop()
    if not st.session_state.session_id:
        st.error("No session ID available—cannot continue.")
        st.session_state.last_action = None
        st.stop()
    if action_requires_llm and not LLM_INTERACTIONS_AVAILABLE:
        st.error("Core AI module unavailable.")
        st.session_state.last_action = None
        st.stop()
    if action_requires_report_util and not REPORT_UTILS_AVAILABLE:
        st.error("Report generation module unavailable.")
        st.session_state.last_action = None
        st.stop()

    img_for_llm = st.session_state.processed_image
    roi_coords = st.session_state.roi_coords
    current_history = st.session_state.history
    if not isinstance(current_history, list):
        current_history = []
        st.session_state.history = current_history

    try:
        if current_action == "analyze":
            st.info("🔬 Performing initial analysis...")
            with st.spinner("AI analyzing..."):
                analysis_result = run_initial_analysis(img_for_llm, roi=roi_coords)
            st.session_state.initial_analysis = analysis_result
            st.session_state.qa_answer = ""
            st.session_state.disease_analysis = ""
            logger.info("Initial analysis complete.")
            st.success("Initial analysis complete!")

        elif current_action == "ask":
            question_text = st.session_state.question_input_widget.strip()
            if not question_text:
                st.warning("Question is empty.")
            else:
                st.info(f"Asking AI: '{question_text}'...")
                st.session_state.qa_answer = ""
                with st.spinner("Thinking..."):
                    answer, success_flag = run_multimodal_qa(
                        img_for_llm,
                        question_text,
                        current_history,
                        roi=roi_coords
                    )
                if success_flag:
                    st.session_state.qa_answer = answer
                    st.session_state.history.append(("User Question", question_text))
                    st.session_state.history.append(("AI Answer", answer))
                    st.success("AI answered your question!")
                else:
                    primary_error_msg = f"Primary AI failed: {answer}"
                    st.session_state.qa_answer = primary_error_msg
                    st.error(primary_error_msg)
                    hf_token = os.environ.get("HF_API_TOKEN")
                    if HF_MODELS_AVAILABLE and hf_token:
                        st.info(f"Attempting fallback HF model: {HF_VQA_MODEL_ID}")
                        with st.spinner("Trying fallback..."):
                            fallback_answer, fallback_success = query_hf_vqa_inference_api(
                                img_for_llm, question_text, roi=roi_coords
                            )
                        if fallback_success:
                            fallback_display = f"**[Fallback: {HF_VQA_MODEL_ID}]**\n\n{fallback_answer}"
                            st.session_state.qa_answer += "\n\n" + fallback_display
                            st.session_state.history.append(("[Fallback] User Question", question_text))
                            st.session_state.history.append(("[Fallback] AI Answer", fallback_display))
                            st.success("Fallback AI answered.")
                        else:
                            fallback_error_msg = f"[Fallback Error - {HF_VQA_MODEL_ID}]: {fallback_answer}"
                            st.session_state.qa_answer += f"\n\n{fallback_error_msg}"
                            st.error("Fallback AI also failed.")
                    elif HF_MODELS_AVAILABLE and not hf_token:
                        st.session_state.qa_answer += "\n\n[Fallback Skipped: HF_API_TOKEN missing]"
                        st.warning("Hugging Face API token needed for fallback.")
                    else:
                        st.session_state.qa_answer += "\n\n[Fallback Unavailable]"
                        st.warning("No fallback AI is configured.")

        elif current_action == "disease":
            selected_disease = st.session_state.disease_select_widget
            if not selected_disease:
                st.warning("No condition selected.")
            else:
                st.info(f"Analyzing for '{selected_disease}'...")
                with st.spinner("AI analyzing condition..."):
                    disease_result = run_disease_analysis(img_for_llm, selected_disease, roi=roi_coords)
                st.session_state.disease_analysis = disease_result
                st.session_state.qa_answer = ""
                logger.info(f"Disease analysis for {selected_disease} complete.")
                st.success(f"Analysis for '{selected_disease}' complete!")

        elif current_action == "confidence":
            if not (current_history or st.session_state.initial_analysis or st.session_state.disease_analysis):
                st.warning("No prior analysis to estimate confidence.")
            else:
                st.info("📊 Estimating confidence...")
                with st.spinner("Calculating confidence..."):
                    confidence_result = estimate_ai_confidence(
                        img_for_llm,
                        history=current_history,
                        initial_analysis=st.session_state.initial_analysis,
                        disease_analysis=st.session_state.disease_analysis,
                        roi=roi_coords
                    )
                st.session_state.confidence_score = confidence_result
                st.success("Confidence estimation complete!")

        elif current_action == "generate_report_data":
            st.info("📄 Generating PDF report data...")
            st.session_state.pdf_report_bytes = None
            image_for_report = st.session_state.get("display_image")
            if not isinstance(image_for_report, Image.Image):
                st.error("Cannot generate report: No valid image in memory.")
            else:
                final_image_for_pdf = image_for_report.copy().convert("RGB")
                if roi_coords:
                    try:
                        draw = ImageDraw.Draw(final_image_for_pdf)
                        x0, y0 = roi_coords['left'], roi_coords['top']
                        x1, y1 = x0 + roi_coords['width'], y0 + roi_coords['height']
                        draw.rectangle(
                            [x0, y0, x1, y1],
                            outline="red",
                            width=max(3, int(min(final_image_for_pdf.size) * 0.005))
                        )
                        logger.info("ROI box drawn on PDF image.")
                    except Exception as e:
                        logger.error(f"Error drawing ROI on PDF image: {e}", exc_info=True)
                        st.warning("Could not draw ROI on the PDF image.")

                formatted_history = "No Q&A history available."
                if current_history:
                    lines = []
                    for q_type, msg in current_history:
                        cleaned_msg = re.sub('<[^<]+?>', '', str(msg))
                        lines.append(f"{q_type}: {cleaned_msg}")
                    formatted_history = "\n\n".join(lines)

                report_data = {
                    "Session ID": st.session_state.session_id,
                    "Image Filename": (st.session_state.uploaded_file_info or "N/A").split('-')[0],
                    "Initial Analysis": st.session_state.initial_analysis or "Not Performed",
                    "Conversation History": formatted_history,
                    "Condition Analysis": st.session_state.disease_analysis or "Not Performed",
                    "AI Confidence Estimation": st.session_state.confidence_score or "Not Performed",
                }
                if st.session_state.is_dicom and st.session_state.dicom_metadata:
                    meta_summary = {k: v for k, v in st.session_state.dicom_metadata.items() if k in [
                        'Patient Name', 'Patient ID', 'Study Date', 'Modality', 'Study Description'
                    ]}
                    if meta_summary:
                        lines = [f"{k}: {v}" for k, v in meta_summary.items()]
                        report_data["DICOM Summary"] = "\n".join(lines)

                with st.spinner("Generating PDF..."):
                    pdf_bytes = generate_pdf_report_bytes(
                        session_id=st.session_state.session_id,
                        image=final_image_for_pdf,
                        analysis_outputs=report_data,
                        dicom_metadata=st.session_state.dicom_metadata if st.session_state.is_dicom else None
                    )
                if pdf_bytes:
                    st.session_state.pdf_report_bytes = pdf_bytes
                    st.success("PDF data ready! Download in sidebar.")
                    logger.info("PDF report generated.")
                    st.balloons()
                else:
                    st.error("Failed to generate PDF.")
                    logger.error("PDF generator returned no data.")

        else:
            st.warning(f"Unknown action '{current_action}' triggered.")
    except Exception as e:
        st.error(f"Error during '{current_action}': {e}")
        logger.critical(f"Action '{current_action}' error: {e}", exc_info=True)
    finally:
        st.session_state.last_action = None
        logger.debug(f"Action '{current_action}' complete.")
        st.rerun()

# --- Footer ---
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced | Session ID: {st.session_state.get('session_id', 'N/A')}")
st.markdown(
    """
    <footer>
      <p>RadVision AI is for informational purposes only. Not a substitute for professional evaluation.</p>
      <p><a href="#" target="_blank">Privacy Policy</a> | <a href="#" target="_blank">Terms of Service</a></p>
    </footer>
    """,
    unsafe_allow_html=True
)
logger.info(f"--- Application render complete for session: {st.session_state.session_id} ---")
