# -*- coding: utf-8 -*-
"""
app.py - Main Streamlit application for RadVision AI Advanced.
Advanced UI/UX implementation with a modern, responsive design.
Includes enhanced sidebar navigation, custom CSS styling, and refined layouts.
"""

import streamlit as st
import io, os, uuid, logging, base64, hashlib, subprocess, sys, copy, random, re
from typing import Any, Dict, Optional, Tuple, List, Union

# ====================================================================
# CUSTOM CSS FOR ADVANCED UI/UX DESIGN
# ====================================================================
CUSTOM_CSS = """
<style>
/* Global Body and Font Settings */
body {
    background-color: #f5f5f5;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Header Styling */
header {
    background: linear-gradient(90deg, #004d40, #00796b);
    color: #fff;
    padding: 20px 0;
    text-align: center;
    border-bottom: 4px solid #004d40;
}
header h1 {
    margin: 0;
    font-size: 2.5rem;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #002F34;
    color: #fff;
}
[data-testid="stSidebar"] .sidebar-content {
    padding: 20px;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #fff;
}

/* Button Styling */
.stButton>button {
    border-radius: 8px;
    background-color: #00796b;
    color: #fff;
    font-weight: bold;
    border: none;
}
.stButton>button:hover {
    background-color: #005f56;
}

/* Card and Container Styling */
.card {
    background-color: #ffffff;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.container-fluid {
    padding: 20px;
}

/* Footer Styling */
.footer {
    text-align: center;
    padding: 12px;
    font-size: 0.85rem;
    color: #555;
    border-top: 1px solid #ccc;
    margin-top: 20px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ====================================================================
# PAGE CONFIGURATION
# ====================================================================
st.set_page_config(
    page_title="RadVision AI Advanced",
    layout="wide",
    page_icon="⚕️",
    initial_sidebar_state="expanded"
)

# ====================================================================
# LOGGING SETUP
# ====================================================================
LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.info("RadVision AI Application Starting...")
logger.info(f"Streamlit Version: {st.__version__}")
logger.info(f"Logging Level: {LOG_LEVEL}")

# ====================================================================
# PILLLOW (PIL) IMPORT AND CHECK
# ====================================================================
try:
    from PIL import Image, ImageDraw, UnidentifiedImageError, ImageOps
    import PIL
    PIL_VERSION = getattr(PIL, '__version__', 'Unknown')
    PIL_AVAILABLE = True
    logger.info(f"Pillow Version: {PIL_VERSION}")
except ImportError:
    logger.critical("Pillow (PIL) is not installed.")
    st.error("Pillow (PIL) missing. Please install it via 'pip install Pillow'.")
    PIL_AVAILABLE = False
    st.stop()

# ====================================================================
# MONKEY PATCH (if necessary)
# ====================================================================
try:
    import streamlit.elements.image as st_image
    if not hasattr(st_image, "image_to_url"):
        logger.info("Applying monkey-patch for st_image.image_to_url...")
        def image_to_url_monkey_patch(image: Any, width: int = -1, clamp: bool = False,
                                      channels: str = "RGB", output_format: str = "auto",
                                      image_id: str = "") -> str:
            patch_logger = logging.getLogger(__name__ + ".monkey_patch")
            patch_logger.debug(f"Monkey patch image_to_url, type: {type(image)}")
            if isinstance(image, Image.Image):
                try:
                    fmt = output_format.upper()
                    fmt = "PNG" if fmt == "AUTO" else fmt
                    if fmt not in ["PNG", "JPEG", "GIF", "WEBP"]:
                        patch_logger.warning(f"Unsupported format {fmt}. Converting to PNG.")
                        fmt = "PNG"
                    img_to_save = image
                    if channels == "RGB" and image.mode not in ['RGB', 'L']:
                        img_to_save = image.convert("RGB")
                    elif channels == "RGBA" and image.mode != 'RGBA':
                        img_to_save = image.convert("RGBA")
                        fmt = "PNG"
                    elif image.mode == 'P':
                        img_to_save = image.convert("RGBA")
                        fmt = "PNG"
                    elif channels == "L" and image.mode != 'L':
                        img_to_save = image.convert("L")
                    elif image.mode == 'L' and channels != "L":
                        img_to_save = image.convert(channels)
                    
                    buffered = io.BytesIO()
                    img_to_save.save(buffered, format=fmt)
                    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    return f"data:image/{fmt.lower()};base64,{img_b64}"
                except Exception as e:
                    patch_logger.error(f"Error in monkey-patch: {e}", exc_info=True)
                    return ""
            else:
                patch_logger.warning(f"Unsupported type: {type(image)}")
                return ""
        st_image.image_to_url = image_to_url_monkey_patch
        logger.info("Monkey-patch applied.")
    else:
        logger.info("Monkey-patch not needed. 'image_to_url' exists.")
except ImportError:
    logger.warning("streamlit.elements.image not found. Skipping monkey-patch.")
except Exception as e:
    logger.error(f"Error during monkey-patch setup: {e}", exc_info=True)

# ====================================================================
# DEPENDENCY IMPORTS & MODULE CHECKS
# ====================================================================
# Deep Translator
try:
    from deep_translator import GoogleTranslator
    DEEP_TRANSLATOR_INSTALLED = True
except ImportError:
    DEEP_TRANSLATOR_INSTALLED = False
    try:
        logger.info("Installing deep-translator...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "deep-translator"])
        from deep_translator import GoogleTranslator
        DEEP_TRANSLATOR_INSTALLED = True
        logger.info("deep-translator installed successfully.")
    except Exception as e:
        logger.critical(f"Installation error for deep-translator: {e}", exc_info=True)

# Streamlit Drawable Canvas
try:
    from streamlit_drawable_canvas import st_canvas
    import streamlit_drawable_canvas as st_canvas_module
    CANVAS_VERSION = getattr(st_canvas_module, '__version__', 'Unknown')
    logger.info(f"Drawable Canvas Version: {CANVAS_VERSION}")
    DRAWABLE_CANVAS_AVAILABLE = True
except ImportError:
    logger.critical("streamlit-drawable-canvas missing. ROI features disabled.")
    DRAWABLE_CANVAS_AVAILABLE = False
    st_canvas = None

# Pydicom & Related Libraries
try:
    import pydicom
    import pydicom.errors
    PYDICOM_VERSION = getattr(pydicom, '__version__', 'Unknown')
    logger.info(f"Pydicom Version: {PYDICOM_VERSION}")
    PYDICOM_AVAILABLE = True
    try:
        import pylibjpeg
        logger.info("pylibjpeg detected.")
    except ImportError:
        logger.info("pylibjpeg not detected (optional).")
    try:
        import gdcm
        logger.info("python-gdcm detected.")
    except ImportError:
        logger.info("python-gdcm not detected (optional).")
except ImportError:
    logger.warning("Pydicom not installed. DICOM features disabled.")
    PYDICOM_AVAILABLE = False

# Custom Backend Modules
try:
    from dicom_utils import parse_dicom, extract_dicom_metadata, dicom_to_image, get_default_wl
    DICOM_UTILS_AVAILABLE = True
    logger.info("dicom_utils imported successfully.")
except ImportError as e:
    logger.error(f"dicom_utils error: {e}")
    DICOM_UTILS_AVAILABLE = False
    if PYDICOM_AVAILABLE:
        st.warning("DICOM utilities missing. Some features limited.")
        
try:
    from llm_interactions import run_initial_analysis, run_multimodal_qa, run_disease_analysis, run_llm_self_assessment
    LLM_INTERACTIONS_AVAILABLE = True
    logger.info("llm_interactions imported successfully.")
except ImportError as e:
    st.error(f"LLM module import error: {e}")
    logger.critical(f"LLM module error: {e}", exc_info=True)
    LLM_INTERACTIONS_AVAILABLE = False
    st.stop()

try:
    from report_utils import generate_pdf_report_bytes
    REPORT_UTILS_AVAILABLE = True
    logger.info("report_utils imported successfully.")
except ImportError as e:
    logger.error(f"report_utils error: {e}")
    REPORT_UTILS_AVAILABLE = False

try:
    from ui_components import display_dicom_metadata, dicom_wl_sliders
    UI_COMPONENTS_AVAILABLE = True
    logger.info("ui_components imported successfully.")
except ImportError as e:
    logger.warning(f"ui_components error: {e}")
    UI_COMPONENTS_AVAILABLE = False
    def display_dicom_metadata(metadata):
        st.caption("Metadata Preview:")
        st.json(dict(list(metadata.items())[:5]))
    def dicom_wl_sliders(wc, ww):
        st.caption("W/L sliders unavailable.")
        return wc, ww

# HF fallback for VQA
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

# Translation Setup
try:
    from translation_models import translate, detect_language, LANGUAGE_CODES, AUTO_DETECT_INDICATOR
    TRANSLATION_AVAILABLE = DEEP_TRANSLATOR_INSTALLED
    if TRANSLATION_AVAILABLE:
        logger.info("translation_models imported successfully.")
    else:
        logger.error("translation_models imported but deep-translator missing.")
        st.warning("Translation features disabled.")
except ImportError as e:
    logger.error(f"translation_models error: {e}", exc_info=True)
    TRANSLATION_AVAILABLE = False
    if DEEP_TRANSLATOR_INSTALLED:
        st.warning(f"Translation module error: {e}")

if not TRANSLATION_AVAILABLE:
    translate = None
    detect_language = None
    LANGUAGE_CODES = {"English": "en"}
    AUTO_DETECT_INDICATOR = "Auto-Detect"

# ====================================================================
# SESSION STATE DEFAULTS
# ====================================================================
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
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.session_id = str(uuid.uuid4())[:8]
    logger.info(f"Session initialized: {st.session_state.session_id}")
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value
if not isinstance(st.session_state.get("history"), list):
    st.session_state.history = []
logger.debug(f"Session state verified for: {st.session_state.session_id}")

# ====================================================================
# UTILITY FUNCTION FOR FORMATTING TRANSLATION
# ====================================================================
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

# ====================================================================
# ADVANCED SIDEBAR NAVIGATION
# ====================================================================
with st.sidebar:
    st.image(os.path.join("assets", "radvisionai-hero.jpeg"), width=180)
    st.markdown("<h2 style='text-align: center;'>RadVision AI</h2>", unsafe_allow_html=True)
    
    # Navigation Radio for Main Sections
    nav_option = st.radio("Navigation", options=[
        "Image Upload & Settings", 
        "AI Analysis Actions",
        "Translation & Reports"
    ], index=0, help="Select the section you wish to view")
    
    st.markdown("---")
    
    # Common Sidebar Hints and Tips (dynamically updated)
    TIPS = [
        "Use Demo Mode for a quick sample chest X-ray analysis.",
        "Draw a ROI on the image to focus the AI analysis.",
        "Adjust DICOM Window/Level sliders for enhanced contrast.",
        "Ask specific questions to get detailed responses.",
        "Generate a PDF report summarizing the interactions."
    ]
    st.info(f"💡 {random.choice(TIPS)}", icon="💡")
    
    # Conditional Sidebar Items Based on Navigation
    if nav_option == "Image Upload & Settings":
        st.header("Upload & Settings")
        st.caption("🔒 Ensure images are de-identified before uploading.")
        uploaded_file = st.file_uploader(
            "Upload Image (JPG, PNG, DCM)",
            type=["jpg", "jpeg", "png", "dcm", "dicom"],
            key="file_uploader_widget",
            help="Upload your de-identified medical image. (Preferred: DICOM)"
        )
        demo_mode = st.checkbox("🚀 Demo Mode", value=st.session_state.get("demo_loaded", False),
                                help="Load a sample chest X-ray with analysis.")
        if demo_mode and not st.session_state.demo_loaded:
            logger.info("Demo Mode activated.")
            st.warning("Demo mode selected, but sample loading logic not implemented yet.")
        elif not demo_mode and st.session_state.demo_loaded:
            logger.info("Demo Mode deactivated.")
            st.session_state.demo_loaded = False

        if DRAWABLE_CANVAS_AVAILABLE:
            if st.button("🗑️ Clear ROI", help="Remove the current ROI selection"):
                st.session_state.roi_coords = None
                st.session_state.canvas_drawing = None
                st.session_state.clear_roi_feedback = True
                logger.info("ROI cleared by user.")
                st.experimental_rerun()
            if st.session_state.get("clear_roi_feedback"):
                st.success("ROI cleared!", icon="✅")
                st.session_state.clear_roi_feedback = False

        # DICOM W/L controls when applicable
        if st.session_state.is_dicom and DICOM_UTILS_AVAILABLE and UI_COMPONENTS_AVAILABLE and st.session_state.display_image:
            st.markdown("---")
            st.subheader("Adjust DICOM Window/Level")
            new_wc, new_ww = dicom_wl_sliders(st.session_state.current_display_wc, st.session_state.current_display_ww)
            if new_wc != st.session_state.current_display_wc or new_ww != st.session_state.current_display_ww:
                logger.info(f"DICOM W/L updated: WC={new_wc}, WW={new_ww}")
                st.session_state.current_display_wc, st.session_state.current_display_ww = new_wc, new_ww
                if st.session_state.dicom_dataset:
                    with st.spinner("Applying new W/L settings..."):
                        try:
                            new_display_img = dicom_to_image(st.session_state.dicom_dataset, wc=new_wc, ww=new_ww)
                            if isinstance(new_display_img, Image.Image):
                                st.session_state.display_image = new_display_img.convert('RGB')
                                st.experimental_rerun()
                            else:
                                st.error("Failed to update DICOM image display.")
                                logger.error("dicom_to_image did not return a valid image.")
                        except Exception as e:
                            st.error(f"Error applying W/L: {e}")
                            logger.error(f"W/L update error: {e}", exc_info=True)
                else:
                    st.warning("DICOM dataset not available.")
                    
    elif nav_option == "AI Analysis Actions":
        st.header("AI Analysis")
        action_disabled = not LLM_INTERACTIONS_AVAILABLE or not isinstance(st.session_state.get("processed_image"), Image.Image)
        if st.button("🔬 Run Structured Analysis", key="analyze_btn", disabled=action_disabled,
                     help="Perform a general structured analysis."):
            st.session_state.last_action = "analyze"
            st.experimental_rerun()
        st.subheader("Ask AI a Question")
        question_input = st.text_area("Your question about the image:", height=100, key="question_input_widget",
                                      placeholder="E.g., 'Describe the findings...'", disabled=action_disabled)
        if st.button("💬 Ask Question", key="ask_btn", disabled=action_disabled):
            if question_input.strip():
                st.session_state.last_action = "ask"
                st.experimental_rerun()
            else:
                st.warning("Please enter a question first.")
        st.subheader("Condition-Focused Analysis")
        DISEASE_OPTIONS = [
            "Pneumonia", "Lung Cancer", "Nodule/Mass", "Effusion", "Fracture", "Stroke",
            "Appendicitis", "Bowel Obstruction", "Cardiomegaly", "Aortic Aneurysm",
            "Pulmonary Embolism", "Tuberculosis", "COVID-19 Findings", "Brain Tumor", "Arthritis",
        ]
        disease_select = st.selectbox("Select condition:", options=[""] + sorted(DISEASE_OPTIONS),
                                      key="disease_select_widget", disabled=action_disabled,
                                      help="AI will focus its analysis on this condition.")
        if st.button("🩺 Analyze for Condition", key="disease_btn", disabled=action_disabled):
            if disease_select:
                st.session_state.last_action = "disease"
                st.experimental_rerun()
            else:
                st.warning("Please select a condition.")

    elif nav_option == "Translation & Reports":
        st.header("Translation & Reporting")
        st.subheader("Translation")
        if not TRANSLATION_AVAILABLE:
            st.warning("Translation service unavailable.", icon="🚫")
        else:
            text_options = {
                "Structured Analysis": st.session_state.initial_analysis,
                "Latest Answer": st.session_state.qa_answer,
                "Condition Analysis": st.session_state.disease_analysis,
                "LLM Self-Assessment": st.session_state.confidence_score,
                "(Enter Custom Text)": ""
            }
            available_labels = [lbl for lbl, txt in text_options.items() if txt or lbl == "(Enter Custom Text)"]
            selected_label = st.selectbox("Select source text:", options=available_labels, key="ts_select")
            text_to_translate = text_options.get(selected_label, "")
            if selected_label == "(Enter Custom Text)":
                text_to_translate = st.text_area("Enter text:", value="", height=100, key="ts_custom")
            st.text_area("Source Text:", value=text_to_translate, height=80, disabled=True, key="ts_preview")
            col1_ts, col2_ts = st.columns(2)
            with col1_ts:
                src_opts = [AUTO_DETECT_INDICATOR] + sorted(list(LANGUAGE_CODES.keys()))
                src_lang = st.selectbox("From:", src_opts, key="ts_src")
            with col2_ts:
                tgt_opts = sorted(list(LANGUAGE_CODES.keys()))
                tgt_idx = 0
                pref = ["Spanish", "English"]
                [tgt_idx := tgt_opts.index(t) for t in pref if t in tgt_opts]
                tgt_lang = st.selectbox("To:", tgt_opts, index=tgt_idx, key="ts_tgt")
            if st.button("🔄 Translate", key="ts_btn"):
                st.session_state.translation_result = None
                st.session_state.translation_error = None
                if not text_to_translate.strip():
                    st.warning("No text provided.", icon="☝️")
                    st.session_state.translation_error = "Empty input."
                elif src_lang == tgt_lang and src_lang != AUTO_DETECT_INDICATOR:
                    st.info("Source and target languages are the same.", icon="✅")
                    st.session_state.translation_result = text_to_translate
                else:
                    with st.spinner("Translating..."):
                        try:
                            t_out = translate(text=text_to_translate, target_language=tgt_lang, source_language=src_lang)
                            if t_out:
                                st.session_state.translation_result = t_out
                                st.success("Translation complete!", icon="🎉")
                            else:
                                st.error("Translation returned empty result.", icon="❓")
                                st.session_state.translation_error = "Empty result."
                        except Exception as e:
                            st.error(f"Translation failed: {e}", icon="❌")
                            st.session_state.translation_error = str(e)
            if st.session_state.get("translation_result"):
                fmt_res = format_translation(st.session_state.translation_result)
                st.text_area("Translation:", value=fmt_res, height=150, key="ts_out")
            elif st.session_state.get("translation_error"):
                st.info(f"Translation Error: {st.session_state.translation_error}", icon="ℹ️")
        st.markdown("---")
        st.subheader("Generate PDF Report")
        report_generation_disabled = (not LLM_INTERACTIONS_AVAILABLE or not isinstance(st.session_state.get("processed_image"), Image.Image)) or not REPORT_UTILS_AVAILABLE
        if st.button("📄 Generate PDF Report", key="generate_report_data_btn", disabled=report_generation_disabled,
                     help="Compile analysis into a PDF report."):
            st.session_state.last_action = "generate_report_data"
            st.experimental_rerun()
        if st.session_state.get("pdf_report_bytes"):
            report_filename = f"RadVisionAI_Report_{st.session_state.session_id or 'session'}.pdf"
            st.download_button(label="⬇️ Download PDF Report",
                               data=st.session_state.pdf_report_bytes,
                               file_name=report_filename,
                               mime="application/pdf",
                               key="download_pdf_button",
                               help="Download the generated PDF report.")

# ====================================================================
# FILE UPLOAD & IMAGE PROCESSING LOGIC
# ====================================================================
if nav_option == "Image Upload & Settings":
    if 'uploaded_file' not in locals():
        uploaded_file = None

    if uploaded_file is not None and PIL_AVAILABLE:
        try:
            uploaded_file.seek(0)
            file_content_hash = hashlib.sha256(uploaded_file.read()).hexdigest()[:16]
            uploaded_file.seek(0)
            new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{file_content_hash}"
        except Exception as e:
            logger.warning(f"Hash generation failed: {e}")
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
            is_dicom_type = "dicom" in uploaded_file.type.lower() or file_ext in (".dcm", ".dicom")
            st.session_state.is_dicom = (PYDICOM_AVAILABLE and DICOM_UTILS_AVAILABLE and is_dicom_type)
            logger.debug(f"Is DICOM: {st.session_state.is_dicom}")
    
            with st.spinner("Analyzing and preparing image..."):
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
                            st.session_state.current_display_wc, st.session_state.current_display_ww = default_wc, default_ww
                            temp_display_img = dicom_to_image(dicom_dataset, wc=default_wc, ww=default_ww)
                            temp_processed_img = dicom_to_image(dicom_dataset, wc=None, ww=None, normalize=True)
    
                            if isinstance(temp_display_img, Image.Image):
                                if temp_display_img.mode != 'RGB':
                                    try:
                                        temp_display_img = temp_display_img.convert('RGB')
                                    except Exception as conv_e:
                                        logger.error(f"Conversion to RGB failed: {conv_e}")
                                        temp_display_img = None
                            else:
                                temp_display_img = None
    
                            if isinstance(temp_processed_img, Image.Image):
                                if temp_processed_img.mode != 'RGB':
                                    try:
                                        temp_processed_img = temp_processed_img.convert('RGB')
                                    except Exception as conv_e:
                                        logger.error(f"Processed image RGB conversion failed: {conv_e}")
                                        temp_processed_img = None
                            else:
                                temp_processed_img = None
    
                            if temp_display_img and temp_processed_img:
                                processing_success = True
                                logger.info("DICOM processed into display and processed images.")
                        else:
                            st.error("Failed to parse DICOM file.")
                    except pydicom.errors.InvalidDicomError:
                        st.error("Invalid DICOM format.")
                        st.session_state.is_dicom = False
                    except Exception as e:
                        st.error(f"DICOM processing error: {e}")
                        st.session_state.is_dicom = False
    
                if not st.session_state.is_dicom and not processing_success:
                    logger.info("Processing as standard image...")
                    try:
                        raw_img = Image.open(io.BytesIO(st.session_state.raw_image_bytes))
                        processed_img = raw_img.convert("RGB")
                        temp_display_img = processed_img.copy()
                        temp_processed_img = processed_img.copy()
                        processing_success = True
                    except UnidentifiedImageError:
                        st.error("Cannot identify image format.")
                    except Exception as e:
                        st.error(f"Error processing image: {e}")
    
                if processing_success and isinstance(temp_display_img, Image.Image) and isinstance(temp_processed_img, Image.Image):
                    if temp_display_img.mode != 'RGB':
                        try:
                            st.session_state.display_image = temp_display_img.convert('RGB')
                        except Exception as final_conv_e:
                            st.error("Final image conversion failed.")
                            processing_success = False
                    else:
                        st.session_state.display_image = temp_display_img
    
                    if processing_success and st.session_state.display_image:
                        st.session_state.processed_image = temp_processed_img
                        st.success(f"✅ Image '{uploaded_file.name}' loaded successfully!")
                        st.experimental_rerun()
                    else:
                        st.error("Final image preparation failed.")
                        st.session_state.uploaded_file_info = None
                        st.session_state.display_image = None
                        st.session_state.processed_image = None
                        st.session_state.is_dicom = False
                else:
                    st.error("Image loading failed. Please try a different file.")
                    st.session_state.uploaded_file_info = None
                    st.session_state.display_image = None
                    st.session_state.processed_image = None
                    st.session_state.is_dicom = False

# ====================================================================
# MAIN PAGE CONTENT
# ====================================================================
# Header Section
st.markdown("<header><h1>RadVision AI Advanced: AI-Assisted Image Analysis</h1></header>", unsafe_allow_html=True)
st.markdown("<div class='container-fluid'>", unsafe_allow_html=True)

# Main Content Layout: Two Columns for Viewer and Analysis
col1, col2 = st.columns([2, 3], gap="large")

with col1:
    with st.container():
        st.subheader("🖼️ Image Viewer")
        display_img = st.session_state.get("display_image")
        st.write(f"DEBUG: Display image type: {type(display_img)}")
        if isinstance(display_img, Image.Image):
            st.info(f"Image Mode: {display_img.mode}, Size: {display_img.size}", icon="✅")
            try:
                st.image(display_img, caption="Image Preview", use_column_width=True)
                logger.info("Image displayed successfully in viewer.")
            except Exception as e:
                st.error(f"Error displaying image: {e}")
        elif 'uploaded_file_info' in st.session_state:
            st.error("Image processing failed. Please check logs or try a different file.")
        else:
            st.info("⬅️ Please upload an image or activate Demo Mode.")
        if st.session_state.roi_coords:
            roi = st.session_state.roi_coords
            st.caption(f"Current ROI: ({roi['left']}, {roi['top']}) Size: {roi['width']} x {roi['height']}")
        if st.session_state.is_dicom and st.session_state.dicom_metadata:
            with st.expander("📄 View DICOM Metadata", expanded=False):
                if UI_COMPONENTS_AVAILABLE:
                    display_dicom_metadata(st.session_state.dicom_metadata)
                else:
                    st.json(st.session_state.dicom_metadata)
                    
with col2:
    with st.container():
        st.subheader("📊 Analysis & Interaction")
        tab_titles = ["🔬 Structured Analysis", "💬 Q&A History", "🩺 Condition Focus", "🧪 LLM Self-Assessment", "🌐 Translation"]
        tabs = st.tabs(tab_titles)
        with tabs[0]:
            st.caption("AI's General Analysis")
            analysis_text = st.session_state.initial_analysis or "Run 'Structured Analysis'."
            st.markdown(analysis_text)
        with tabs[1]:
            st.caption("Latest Answer & History")
            st.markdown("**Latest AI Answer:**")
            latest_answer = st.session_state.qa_answer or "_Awaiting your question..._"
            st.markdown(latest_answer)
            st.markdown("---")
            if st.session_state.history:
                with st.expander("Full History", expanded=True):
                    for i, (q_type, message) in enumerate(st.session_state.history):
                        if "user" in q_type.lower():
                            st.markdown(f"**You:** {message}")
                        elif "ai" in q_type.lower():
                            st.markdown(f"**AI:** {message}")
                        elif "[fallback]" in q_type.lower():
                            st.markdown(f"**AI (Fallback):** {message}")
                        else:
                            st.markdown(f"**{q_type}:** {message}")
                        if i < len(st.session_state.history) - 1:
                            st.markdown("---")
            else:
                st.caption("No history available yet.")
        with tabs[2]:
            st.caption("Condition-Specific Analysis")
            condition_text = st.session_state.disease_analysis or "Select a condition and run analysis."
            st.markdown(condition_text)
        with tabs[3]:
            st.caption("LLM Self-Assessment (Experimental)")
            st.warning("⚠️ Experimental score. Use with caution.", icon="🧪")
            confidence_text = st.session_state.confidence_score or "Run self-assessment."
            st.markdown(confidence_text)
        with tabs[4]:
            st.subheader("🌐 Translate Analysis")
            if not TRANSLATION_AVAILABLE:
                st.warning("Translation feature is unavailable.", icon="🚫")
            else:
                st.caption("Translate the analysis text below.")
                text_options = {
                    "Structured Analysis": st.session_state.initial_analysis,
                    "Latest Answer": st.session_state.qa_answer,
                    "Condition Analysis": st.session_state.disease_analysis,
                    "LLM Self-Assessment": st.session_state.confidence_score,
                    "(Enter Custom Text)": ""
                }
                available_labels = [lbl for lbl, txt in text_options.items() if txt or lbl == "(Enter Custom Text)"]
                selected_label = st.selectbox("Select text:", options=available_labels, key="ts_select")
                text_to_translate = text_options.get(selected_label, "")
                if selected_label == "(Enter Custom Text)":
                    text_to_translate = st.text_area("Enter text:", value="", height=100, key="ts_custom")
                st.text_area("Source Text:", value=text_to_translate, height=80, disabled=True, key="ts_preview")
                cl1, cl2 = st.columns(2)
                with cl1:
                    src_opts = [AUTO_DETECT_INDICATOR] + sorted(list(LANGUAGE_CODES.keys()))
                    src_lang = st.selectbox("From:", src_opts, key="ts_src")
                with cl2:
                    tgt_opts = sorted(list(LANGUAGE_CODES.keys()))
                    tgt_idx = 0
                    pref = ["Spanish", "English"]
                    [tgt_idx := tgt_opts.index(t) for t in pref if t in tgt_opts]
                    tgt_lang = st.selectbox("To:", tgt_opts, index=tgt_idx, key="ts_tgt")
                if st.button("🔄 Translate", key="ts_btn"):
                    st.session_state.translation_result = None
                    st.session_state.translation_error = None
                    if not text_to_translate.strip():
                        st.warning("No text provided.", icon="☝️")
                        st.session_state.translation_error = "Empty input."
                    elif src_lang == tgt_lang and src_lang != AUTO_DETECT_INDICATOR:
                        st.info("Source and target languages are the same.", icon="✅")
                        st.session_state.translation_result = text_to_translate
                    else:
                        with st.spinner("Translating..."):
                            try:
                                t_out = translate(text=text_to_translate, target_language=tgt_lang, source_language=src_lang)
                                if t_out:
                                    st.session_state.translation_result = t_out
                                    st.success("Translation complete!", icon="🎉")
                                else:
                                    st.error("Empty translation result.", icon="❓")
                                    st.session_state.translation_error = "Empty result."
                            except Exception as e:
                                st.error(f"Translation error: {e}", icon="❌")
                                st.session_state.translation_error = str(e)
                if st.session_state.get("translation_result"):
                    fmt_res = format_translation(st.session_state.translation_result)
                    st.text_area("Translation:", value=fmt_res, height=150, key="ts_out")
                elif st.session_state.get("translation_error"):
                    st.info(f"Error: {st.session_state.translation_error}", icon="ℹ️")

st.markdown("</div>", unsafe_allow_html=True)

# ====================================================================
# BUTTON ACTION HANDLERS (CENTRAL LOGIC)
# ====================================================================
current_action = st.session_state.get("last_action")
if current_action:
    logger.info(f"Processing action '{current_action}' for session {st.session_state.session_id}")
    action_requires_image = current_action in ["analyze", "ask", "disease", "confidence", "generate_report_data"]
    action_requires_llm = current_action in ["analyze", "ask", "disease", "confidence"]
    action_requires_report_util = (current_action == "generate_report_data")
    error_occurred = False
    if action_requires_image and not isinstance(st.session_state.get("processed_image"), Image.Image):
        st.error(f"No valid image for action '{current_action}'.", icon="🖼️")
        error_occurred = True
    if not st.session_state.session_id:
        st.error("Session ID missing.", icon="🆔")
        error_occurred = True
    if action_requires_llm and not LLM_INTERACTIONS_AVAILABLE:
        st.error(f"AI module unavailable for '{current_action}'.", icon="🤖")
        error_occurred = True
    if action_requires_report_util and not REPORT_UTILS_AVAILABLE:
        st.error(f"PDF report module unavailable for '{current_action}'.", icon="📄")
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
                logger.info("Initial analysis complete.")
                st.success("Analysis complete!", icon="✅")
        elif current_action == "ask":
            question_text = st.session_state.question_input_widget.strip()
            if not question_text:
                st.warning("Question field is empty.", icon="❓")
            else:
                st.toast(f"Asking: '{question_text[:50]}...'", icon="⏳")
                st.session_state.qa_answer = ""
                with st.spinner("Thinking..."):
                    answer, success_flag = run_multimodal_qa(img_for_llm, question_text, current_history, roi=roi_coords)
                if success_flag:
                    st.session_state.qa_answer = answer
                    st.session_state.history.extend([("User Question", question_text), ("AI Answer", answer)])
                    st.success("AI responded successfully!", icon="💬")
                else:
                    primary_error_msg = f"Primary AI failed: {answer}"
                    st.session_state.qa_answer = primary_error_msg
                    st.error(primary_error_msg, icon="⚠️")
                    logger.warning(f"Q&A failure: {answer}")
        elif current_action == "disease":
            selected_disease = st.session_state.disease_select_widget
            if not selected_disease:
                st.warning("Please select a condition.", icon="🏷️")
            else:
                st.toast(f"🩺 Analyzing for '{selected_disease}'...", icon="⏳")
                with st.spinner("AI analyzing..."):
                    disease_result = run_disease_analysis(img_for_llm, selected_disease, roi=roi_coords)
                st.session_state.disease_analysis = disease_result
                st.session_state.qa_answer = ""
                logger.info(f"Disease analysis for '{selected_disease}' complete.")
                st.success(f"Condition analysis complete for '{selected_disease}'!", icon="✅")
        elif current_action == "confidence":
            if not current_history:
                st.warning("No Q&A history available to assess.", icon="📊")
            else:
                st.toast("🧪 Estimating self-assessment...", icon="⏳")
                with st.spinner("LLM self-assessing..."):
                    assessment_result = run_llm_self_assessment(image=img_for_llm, history=current_history, roi=roi_coords)
                st.session_state.confidence_score = assessment_result
                st.success("Self-assessment complete!", icon="✅")
        elif current_action == "generate_report_data":
            st.toast("📄 Generating PDF Report...", icon="⏳")
            st.session_state.pdf_report_bytes = None
            image_for_report = st.session_state.get("display_image")
            if not isinstance(image_for_report, Image.Image):
                st.error("No image available for report.", icon="🖼️")
            else:
                final_image_for_pdf = image_for_report.copy().convert("RGB")
                if roi_coords:
                    try:
                        draw = ImageDraw.Draw(final_image_for_pdf)
                        x0, y0, w, h = roi_coords['left'], roi_coords['top'], roi_coords['width'], roi_coords['height']
                        draw.rectangle([x0, y0, x0+w, y0+h], outline="red", width=max(3, int(min(final_image_for_pdf.size)*0.005)))
                        logger.info("ROI drawn for PDF.")
                    except Exception as draw_e:
                        logger.error(f"Error drawing ROI: {draw_e}", exc_info=True)
                        st.warning("Could not draw ROI.", icon="✏️")
                formatted_history = "No Q&A history available."
                if current_history:
                    lines = [f"[{qt}]:\n{re.sub('<[^<]+?>', '', str(m)).strip()}" for qt, m in current_history]
                    formatted_history = "\n\n---\n\n".join(lines)
                report_data = {
                    "Session ID": st.session_state.session_id,
                    "Image Filename": (st.session_state.uploaded_file_info or "N/A").split('-')[0],
                    "Structured Analysis": st.session_state.initial_analysis or "N/P",
                    "Q&A History": formatted_history,
                    "Condition Analysis": st.session_state.disease_analysis or "N/P",
                    "LLM Self-Assessment": st.session_state.confidence_score or "N/P"
                }
                if st.session_state.is_dicom and st.session_state.dicom_metadata:
                    meta_tags = ['PatientName', 'PatientID', 'StudyDate', 'Modality', 'StudyDescription', 'InstitutionName']
                    meta_summary = {t: st.session_state.dicom_metadata.get(t, "N/A") for t in meta_tags if t in st.session_state.dicom_metadata}
                    if meta_summary:
                        lines = [f"{k.replace('PatientName', 'Pt Name').replace('PatientID', 'Pt ID').replace('StudyDate', 'Date').replace('StudyDescription', 'Desc.')}: {v}" for k, v in meta_summary.items()]
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
                    st.success("PDF generated successfully!", icon="📄")
                    st.balloons()
                    logger.info("PDF report generated.")
                else:
                    st.error("PDF generation failed.", icon="❌")
                    logger.error("PDF generation returned None.")
        else:
            st.warning(f"Unknown action '{current_action}'.", icon="❓")
            logger.warning(f"Unhandled action: {current_action}")
    except Exception as e:
        st.error(f"Error processing '{current_action}': {e}", icon="💥")
        logger.critical(f"Action '{current_action}' error: {e}", exc_info=True)
    finally:
        st.session_state.last_action = None
        logger.debug(f"Action '{current_action}' complete.")
        st.experimental_rerun()

# ====================================================================
# FOOTER SECTION
# ====================================================================
st.markdown("<div class='footer'>⚕️ RadVision AI Advanced | Session ID: " +
            f"{st.session_state.get('session_id', 'N/A')}" +
            " | © 2025 RadVision Inc.</div>", unsafe_allow_html=True)
logger.info(f"Application render cycle complete for session: {st.session_state.session_id}")
