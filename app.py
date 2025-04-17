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
- Added CSS to improve text contrast in sidebar input elements.
- Enhanced logging and error feedback for image loading.
- Restored monkey-patch for streamlit_drawable_canvas compatibility.
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
import io         # Required for BytesIO (used in file processing and monkey-patch)
import os
import uuid
import logging
import base64     # Required for monkey-patch
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
except ImportError:
    st.warning("Trying to install 'deep-translator' for translation features...")
    try:
        # Use "--disable-pip-version-check" and "--quiet" for cleaner output
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--quiet",
            "--disable-pip-version-check", "deep-translator"
        ])
        from deep_translator import GoogleTranslator
        st.success("'deep-translator' installed successfully.")
    except Exception as e:
        print(f"CRITICAL: Could not install deep-translator: {e}")
        st.error(f"Failed to install 'deep-translator'. Translation disabled. Error: {e}")


# --- Logging Setup (Early) ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
log_format = '%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler) # Prevent duplicate logs
logging.basicConfig(
    level=LOG_LEVEL, format=log_format, datefmt=date_format, stream=sys.stdout
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
    st.error("CRITICAL ERROR: streamlit-drawable-canvas not installed. Functionality impaired. Run `pip install streamlit-drawable-canvas`.")
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
    Image = None; ImageDraw = None; UnidentifiedImageError = Exception
    st.stop()

# --- pydicom & DICOM libraries ---
try:
    import pydicom
    import pydicom.errors
    import pydicom.valuerep
    import numpy as np
    PYDICOM_VERSION = getattr(pydicom, '__version__', 'Unknown')
    logger.info(f"Pydicom Version: {PYDICOM_VERSION}")
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_VERSION = 'Not Installed'
    logger.warning("pydicom or numpy not found. DICOM functionality will be disabled.")
    pydicom = None; np = None; PYDICOM_AVAILABLE = False

if PYDICOM_AVAILABLE:
    try: import pylibjpeg; logger.info("pylibjpeg found.")
    except ImportError: logger.info("pylibjpeg not found.")
    try: import gdcm; logger.info("python-gdcm found.")
    except ImportError: logger.info("python-gdcm not found.")

# --- Custom Utilities & Backend Modules ---
try:
    from dicom_utils import (parse_dicom, extract_dicom_metadata, dicom_to_image, get_default_wl)
    DICOM_UTILS_AVAILABLE = True; logger.info("dicom_utils imported.")
except ImportError as e:
    logger.error(f"Failed to import dicom_utils: {e}. DICOM features disabled.")
    DICOM_UTILS_AVAILABLE = False
    def parse_dicom(*a, **k): logger.error("parse_dicom stub called."); return None
    def extract_dicom_metadata(*a, **k): logger.error("extract_dicom_metadata stub called."); return {}
    def dicom_to_image(*a, **k): logger.error("dicom_to_image stub called."); return None
    def get_default_wl(*a, **k): logger.error("get_default_wl stub called."); return (None, None)

try:
    from llm_interactions import (run_initial_analysis, run_multimodal_qa, run_disease_analysis, estimate_ai_confidence)
    LLM_INTERACTIONS_AVAILABLE = True; logger.info("llm_interactions imported.")
except ImportError as e:
    st.error(f"Core AI module (llm_interactions) failed: {e}. Analysis disabled.")
    logger.critical(f"Failed: {e}", exc_info=True)
    LLM_INTERACTIONS_AVAILABLE = False
    def run_initial_analysis(*a, **k): return "Error: AI Module Unavailable"
    def run_multimodal_qa(*a, **k): return ("Error: AI Module Unavailable", False)
    def run_disease_analysis(*a, **k): return "Error: AI Module Unavailable"
    def estimate_ai_confidence(*a, **k): return "Error: AI Module Unavailable"

try:
    from report_utils import generate_pdf_report_bytes
    REPORT_UTILS_AVAILABLE = True; logger.info("report_utils imported.")
except ImportError as e:
    logger.error(f"Failed: {e}. PDF reporting disabled.")
    REPORT_UTILS_AVAILABLE = False
    def generate_pdf_report_bytes(*a, **k): logger.error("generate_pdf_report_bytes stub called."); return None

try:
    from ui_components import display_dicom_metadata, dicom_wl_sliders, display_umls_concepts
    UI_COMPONENTS_AVAILABLE = True; logger.info("ui_components imported.")
except ImportError as e:
    logger.warning(f"Failed: {e}. Using UI fallbacks.")
    UI_COMPONENTS_AVAILABLE = False
    def display_dicom_metadata(m): st.caption("Metadata display unavailable.")
    def dicom_wl_sliders(d, wc, ww): st.caption("W/L sliders unavailable."); return wc, ww
    def display_umls_concepts(c): st.caption("UMLS display unavailable.")

try:
    from hf_models import query_hf_vqa_inference_api, HF_VQA_MODEL_ID
    HF_MODELS_AVAILABLE = True; logger.info(f"hf_models imported (Fallback: {HF_VQA_MODEL_ID}).")
except ImportError:
    HF_VQA_MODEL_ID = "hf_model_not_found"; HF_MODELS_AVAILABLE = False
    def query_hf_vqa_inference_api(i, q, r=None): logger.warning("HF VQA stub called."); return ("[Fallback VQA Unavailable]", False)
    logger.warning("hf_models not found. Fallback VQA disabled.")

try:
    from translation_models import (translate, detect_language, LANGUAGE_CODES, AUTO_DETECT_INDICATOR)
    TRANSLATION_AVAILABLE = True; logger.info("translation_models imported.")
except ImportError as e:
    st.warning(f"Translation features disabled: {e}")
    logger.error(f"Failed: {e}", exc_info=True); TRANSLATION_AVAILABLE = False
    def translate(*a, **k): logger.error("translate stub called."); return "Translation Error"
    def detect_language(*a, **k): logger.error("detect_language stub called."); return "en"
    LANGUAGE_CODES = {"English": "en"}; AUTO_DETECT_INDICATOR = "Auto-Detect"

try:
    import umls_utils
    from umls_utils import UMLSAuthError, UMLSConcept
    UMLS_APIKEY = os.getenv("UMLS_APIKEY")
    if not UMLS_APIKEY: logger.warning("UMLS_APIKEY not set. UMLS disabled."); UMLS_AVAILABLE = False
    else: logger.info("umls_utils imported & API key found."); UMLS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed: {e}. UMLS disabled."); UMLS_AVAILABLE = False
    umls_utils=None; UMLSAuthError=RuntimeError; UMLSConcept=None; UMLS_APIKEY=None
except Exception as e:
    logger.error(f"Error during UMLS setup: {e}", exc_info=True); UMLS_AVAILABLE = False
    umls_utils=None; UMLSAuthError=RuntimeError; UMLSConcept=None; UMLS_APIKEY=None


# --- Custom CSS ---
st.markdown(
    """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f0f2f6; }
      .main .block-container { padding: 2rem 1.5rem; }
      .css-1d391kg { background-color: #ffffff; border-right: 1px solid #e0e0e0; } /* Sidebar */
      .stButton>button { border-radius: 8px; padding: 0.5rem 1rem; font-weight: 500; width: 100%; margin-bottom: 0.5rem; }
      .stButton>button:hover { filter: brightness(95%); }
      /* --- FIX for faint text in sidebar inputs --- */
      .css-1d391kg .stTextArea textarea::placeholder { color: #6c757d !important; opacity: 1; }
      .css-1d391kg div[data-baseweb="select"] > div:first-child > div:first-child { color: #31333F !important; }
      .css-1d391kg div[data-baseweb="select"] svg { fill: #31333F !important; }
      /* --- End FIX --- */
      div[role="tablist"] { overflow-x: auto; white-space: nowrap; border-bottom: 1px solid #e0e0e0; scrollbar-width: thin; scrollbar-color: #ccc #f0f2f6; }
      div[role="tablist"]::-webkit-scrollbar { height: 6px; }
      div[role="tablist"]::-webkit-scrollbar-track { background: #f0f2f6; }
      div[role="tablist"]::-webkit-scrollbar-thumb { background-color: #ccc; border-radius: 10px; border: 2px solid #f0f2f6; }
      footer { text-align: center; font-size: 0.8em; color: #6c757d; margin-top: 2rem; padding: 1rem 0; border-top: 1px solid #e0e0e0; }
      footer a { color: #007bff; text-decoration: none; }
      footer a:hover { text-decoration: underline; }
    </style>
    """, unsafe_allow_html=True
)

# --- Display Hero Logo ---
logo_path = os.path.join("assets", "radvisionai-hero.jpeg")
if os.path.exists(logo_path): st.image(logo_path, width=350)
else: logger.warning(f"Hero logo not found: {logo_path}")

# --- Initialize Session State Defaults ---
DEFAULT_STATE = {
    "uploaded_file_info": None, "raw_image_bytes": None, "is_dicom": False,
    "dicom_dataset": None, "dicom_metadata": {}, "processed_image": None,
    "display_image": None, "session_id": None, "history": [],
    "initial_analysis": "", "qa_answer": "", "disease_analysis": "",
    "confidence_score": "", "last_action": None, "pdf_report_bytes": None,
    "canvas_drawing": None, "roi_coords": None, "current_display_wc": None,
    "current_display_ww": None, "clear_roi_feedback": False, "demo_loaded": False,
    "translation_result": None, "translation_error": None, "umls_search_term": "",
    "umls_results": None, "umls_error": None,
}
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]; logger.info(f"New session: {st.session_state.session_id}")
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state: st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value
if not isinstance(st.session_state.get("history"), list): st.session_state.history = []
logger.debug(f"Session state verified: {st.session_state.session_id}")

# --- Helper function for formatting translation ---
def format_translation(t: Optional[str]) -> str:
    if t is None: return "Translation unavailable."
    try: return re.sub(r'\s+(\d+\.)', r'\n\n\1', str(t)).strip()
    except Exception as e: logger.error(f"Fmt err: {e}", exc_info=True); return str(t)

# --- Monkey-Patch for streamlit_drawable_canvas compatibility ---
# streamlit-drawable-canvas relies on st_image.image_to_url, which might be
# missing in some Streamlit versions. This patch provides the function if needed.
import streamlit.elements.image as st_image
# Ensure io and base64 are imported (already done globally, but safe here)

logger.debug(f"Checking for st_image.image_to_url. Available: {hasattr(st_image, 'image_to_url')}")

if not hasattr(st_image, "image_to_url"):
    logger.info("Applying monkey-patch for st.elements.image.image_to_url as it is missing.")

    def image_to_url_monkey_patch(
        img_obj: Any, width: int = -1, clamp: bool = False, channels: str = "RGB",
        output_format: str = "auto", image_id: str = "",
    ) -> str:
        """Monkey-patch implementation to convert image object to data URL."""
        if PIL_AVAILABLE and isinstance(img_obj, Image.Image):
            try:
                buffered = io.BytesIO()
                fmt = img_obj.format or "PNG"
                if output_format.lower() != "auto": fmt = output_format.upper()
                if fmt not in ["PNG", "JPEG", "GIF", "WEBP"]: fmt = "PNG"
                if img_obj.mode == 'RGBA' and fmt == 'JPEG': fmt = 'PNG'
                temp_img = img_obj
                if temp_img.mode == 'P': temp_img = temp_img.convert('RGBA'); fmt = "PNG"
                elif channels == "RGB" and temp_img.mode not in ['RGB', 'L']: temp_img = temp_img.convert('RGB')
                elif temp_img.mode == 'CMYK': temp_img = temp_img.convert('RGB')
                logger.debug(f"Saving image for data URL with format: {fmt}")
                temp_img.save(buffered, format=fmt)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/{fmt.lower()};base64,{img_str}"
            except Exception as e: logger.error(f"Monkey-patch failed: {e}", exc_info=True); return ""
        else:
            if not PIL_AVAILABLE: logger.warning("Monkey-patch: PIL unavailable.")
            else: logger.warning(f"Monkey-patch: Unsupported type {type(img_obj)}.")
            return ""

    st_image.image_to_url = image_to_url_monkey_patch
    logger.info("Monkey-patch for st.elements.image.image_to_url applied.")
else:
     logger.info("Streamlit has built-in st_image.image_to_url. Patch not needed.")
# --- End Monkey-Patch Section ---


# --- Sidebar ---
with st.sidebar:
    st.header("⚕️ RadVision Controls")
    st.markdown("---")
    TIPS = ["Tip: Use 'Demo Mode' for a quick walkthrough.", "Tip: Draw an ROI rectangle.", "Tip: Adjust DICOM W/L.", "Tip: Ask follow-up questions.", "Tip: Generate a PDF report.", "Tip: Use 'Translation' tab.", "Tip: Clear the ROI.", "Tip: Use 'UMLS Lookup'."]
    st.info(f"💡 {random.choice(TIPS)}")
    st.markdown("---")
    st.header("Image Upload & Settings")
    uploaded_file = st.file_uploader("Upload Image (JPG, PNG, DCM)", type=["jpg","jpeg","png","dcm","dicom"], key="file_uploader_widget", help="Upload medical image. DICOM preferred.")
    demo_mode = st.checkbox("🚀 Demo Mode", value=st.session_state.get("demo_loaded", False), help="Load sample X-ray.")
    if st.button("🗑️ Clear ROI", help="Remove selected ROI", key="clear_roi_btn"): st.session_state.roi_coords = None; st.session_state.canvas_drawing = None; st.session_state.clear_roi_feedback = True; logger.info("ROI cleared."); st.rerun()
    if st.session_state.get("clear_roi_feedback"): st.success("✅ ROI cleared!"); st.balloons(); st.session_state.clear_roi_feedback = False

    if st.session_state.is_dicom and st.session_state.display_image and UI_COMPONENTS_AVAILABLE:
        st.markdown("---"); st.subheader("DICOM Display")
        new_wc, new_ww = dicom_wl_sliders(st.session_state.dicom_dataset, st.session_state.current_display_wc, st.session_state.current_display_ww)
        if new_wc != st.session_state.current_display_wc or new_ww != st.session_state.current_display_ww:
            logger.info(f"W/L changed: WC={new_wc}, WW={new_ww}")
            st.session_state.current_display_wc = new_wc; st.session_state.current_display_ww = new_ww
            if DICOM_UTILS_AVAILABLE and st.session_state.dicom_dataset and PIL_AVAILABLE:
                with st.spinner("Applying W/L..."):
                    logger.debug("Regenerating display image with new W/L...")
                    new_display_img = dicom_to_image(st.session_state.dicom_dataset, wc=new_wc, ww=new_ww)
                    if isinstance(new_display_img, Image.Image):
                        st.session_state.display_image = new_display_img.convert('RGB') if new_display_img.mode != 'RGB' else new_display_img
                        logger.info("Display image updated with new W/L."); st.rerun()
                    else: st.error("Failed to update DICOM image."); logger.error(f"dicom_to_image invalid type ({type(new_display_img)}).")
            elif not DICOM_UTILS_AVAILABLE: st.warning("DICOM utils unavailable."); logger.warning("W/L changed; DICOM utils missing.")
            elif not st.session_state.dicom_dataset: st.warning("DICOM dataset missing."); logger.warning("W/L changed; DICOM dataset missing.")

    st.markdown("---"); st.header("🤖 AI Analysis Actions")
    action_disabled = not isinstance(st.session_state.get("processed_image"), Image.Image)
    if st.button("▶️ Run Initial Analysis", key="analyze_btn", disabled=action_disabled, help="General analysis of image/ROI."): st.session_state.last_action = "analyze"; st.rerun()
    st.subheader("❓ Ask AI a Question")
    question_input = st.text_area("Enter your question:", height=100, key="question_input_widget", placeholder="E.g., 'Any nodules?'", disabled=action_disabled)
    if st.button("💬 Ask Question", key="ask_btn", disabled=action_disabled):
        if question_input.strip(): st.session_state.last_action = "ask"; st.rerun()
        else: st.warning("Please enter a question.")
    st.subheader("🎯 Condition-Specific Analysis")
    DISEASE_OPTIONS = ["Pneumonia", "Lung Cancer", "Nodule/Mass", "Effusion", "Fracture", "Stroke", "Appendicitis", "Bowel Obstruction", "Cardiomegaly", "Aortic Aneurysm", "Pulmonary Embolism", "Tuberculosis", "COVID-19", "Brain Tumor", "Arthritis", "Osteoporosis", "Other..."]
    disease_select = st.selectbox("Select condition:", options=[""] + sorted(DISEASE_OPTIONS), key="disease_select_widget", disabled=action_disabled)
    if st.button("🩺 Analyze Condition", key="disease_btn", disabled=action_disabled):
        if disease_select: st.session_state.last_action = "disease"; st.rerun()
        else: st.warning("Please select a condition.")
    st.markdown("---"); st.header("📊 Confidence & Reporting")
    can_estimate = bool(st.session_state.history or st.session_state.initial_analysis or st.session_state.disease_analysis) and not action_disabled
    if st.button("📈 Estimate AI Confidence", key="confidence_btn", disabled=not can_estimate): st.session_state.last_action = "confidence"; st.rerun()
    report_disabled = action_disabled or not REPORT_UTILS_AVAILABLE
    if st.button("📄 Generate PDF Report Data", key="generate_report_data_btn", disabled=report_disabled): st.session_state.last_action = "generate_report_data"; st.rerun()
    if st.session_state.get("pdf_report_bytes"):
        report_filename = f"RadVisionAI_Report_{st.session_state.session_id or 'session'}.pdf"
        st.download_button("⬇️ Download PDF Report", st.session_state.pdf_report_bytes, report_filename, "application/pdf", key="download_pdf_button", help="Download generated PDF.")


# --- File Upload Processing Logic ---
if uploaded_file is not None:
    try: uploaded_file.seek(0); h = hashlib.sha256(uploaded_file.read()).hexdigest()[:16]; uploaded_file.seek(0); new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{h}"; logger.debug(f"File info: {new_file_info}")
    except Exception as e: logger.warning(f"Hash fail: {e}"); new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{uuid.uuid4().hex[:8]}"; logger.debug(f"Fallback info: {new_file_info}")
    if new_file_info != st.session_state.get("uploaded_file_info"):
        logger.info(f"New file: '{new_file_info}' (Prev: '{st.session_state.get('uploaded_file_info')}')")
        st.toast(f"Processing '{uploaded_file.name}'...", icon="⏳")
        keys_to_preserve = {"session_id"}; preserved = {k: st.session_state.get(k) for k in keys_to_preserve if k in st.session_state}; logger.debug("Resetting state...")
        for k, v_def in DEFAULT_STATE.items():
            if k not in keys_to_preserve: st.session_state[k] = copy.deepcopy(v_def) if isinstance(v_def, (dict, list)) else v_def
        for k, v_pres in preserved.items(): st.session_state[k] = v_pres
        st.session_state.uploaded_file_info = new_file_info; st.session_state.demo_loaded = False
        st.session_state.raw_image_bytes = uploaded_file.getvalue()
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        st.session_state.is_dicom = (PYDICOM_AVAILABLE and DICOM_UTILS_AVAILABLE and ("dicom" in uploaded_file.type.lower() or file_ext in (".dcm", ".dicom")))
        logger.info(f"Is DICOM: {st.session_state.is_dicom}")
        with st.spinner("🔬 Analyzing file..."):
            img_disp, img_proc, success, err_msg = None, None, False, ""
            if st.session_state.is_dicom:
                logger.info("Processing DICOM...");
                if not DICOM_UTILS_AVAILABLE or not PYDICOM_AVAILABLE: err_msg="DICOM libs unavailable."; logger.error(err_msg); st.session_state.is_dicom = False
                else:
                    try:
                        ds = parse_dicom(st.session_state.raw_image_bytes, uploaded_file.name); st.session_state.dicom_dataset = ds
                        if ds:
                            st.session_state.dicom_metadata = extract_dicom_metadata(ds); logger.info(f"{len(st.session_state.dicom_metadata)} tags."); wc, ww = get_default_wl(ds); st.session_state.current_display_wc, st.session_state.current_display_ww = wc, ww; logger.info(f"Def W/L: {wc}, {ww}")
                            img_disp = dicom_to_image(ds, wc=wc, ww=ww); img_proc = dicom_to_image(ds, wc=None, ww=None, normalize=True)
                            if isinstance(img_disp, Image.Image) and isinstance(img_proc, Image.Image): success = True; logger.info("DICOM converted.")
                            else: err_msg = "DICOM conversion failed."; logger.error(err_msg)
                        else: err_msg = "DICOM parse failed."; logger.error(err_msg); st.session_state.is_dicom = False
                    except pydicom.errors.InvalidDicomError as e: err_msg = f"Invalid DICOM: {e}"; logger.error(err_msg); st.session_state.is_dicom = False
                    except Exception as e: err_msg = f"DICOM proc error: {e}"; logger.error(err_msg, exc_info=True); st.session_state.is_dicom = False
            if not st.session_state.is_dicom: # Process standard
                logger.info("Processing standard image...")
                if not PIL_AVAILABLE: err_msg = "PIL unavailable."; logger.critical(err_msg)
                else:
                    try:
                        raw = Image.open(io.BytesIO(st.session_state.raw_image_bytes)); proc = raw.convert("RGB")
                        img_disp, img_proc = proc.copy(), proc.copy(); success = True; logger.info("Standard image processed.")
                    except UnidentifiedImageError: err_msg = "Cannot ID format."; logger.error(f"{err_msg} File: {uploaded_file.name}")
                    except Exception as e: err_msg = f"Std image err: {e}"; logger.error(err_msg, exc_info=True)
            if success and isinstance(img_disp, Image.Image) and isinstance(img_proc, Image.Image):
                logger.info("Image proc OK."); st.session_state.display_image = img_disp.convert('RGB') if img_disp.mode != 'RGB' else img_disp; st.session_state.processed_image = img_proc
                st.success(f"✅ '{uploaded_file.name}' loaded!"); st.rerun()
            else:
                 logger.error(f"Image proc FAILED: {err_msg}"); st.error(f"Load failed: {err_msg or 'Unknown error.'}")
                 st.session_state.uploaded_file_info=None; st.session_state.display_image=None; st.session_state.processed_image=None; st.session_state.is_dicom=False; st.session_state.dicom_dataset=None; st.session_state.dicom_metadata={}; st.session_state.raw_image_bytes=None; st.session_state.current_display_wc=None; st.session_state.current_display_ww=None

# --- Main Page Layout ---
st.markdown("---"); st.title("⚕️ RadVision AI Advanced: AI-Assisted Image Analysis")
with st.expander("User Guide & Disclaimer", expanded=False):
    st.warning("⚠️ **Disclaimer**: For research/educational use ONLY. NOT medical advice.")
    st.markdown("""**Workflow:** 1.Upload Image 2.(DICOM) Adjust W/L 3.(Optional) Draw ROI 4.AI Analysis (Sidebar) 5.Explore Results (Tabs) 6.(Optional) UMLS Lookup 7.(Optional) Translation 8.(Optional) Confidence 9.Generate Report""")
st.markdown("---")
col1, col2 = st.columns([2, 3])

with col1: # Image Viewer
    st.subheader("🖼️ Image Viewer")
    display_img = st.session_state.get("display_image")
    if isinstance(display_img, Image.Image) and PIL_AVAILABLE:
        logger.debug("Displaying image...")
        if DRAWABLE_CANVAS_AVAILABLE and st_canvas:
            st.caption("Draw rectangle for ROI.")
            MAX_W, MAX_H = 600, 500; img_w, img_h = display_img.size
            if img_w <= 0 or img_h <= 0: st.warning("Invalid image size.")
            else:
                ar = img_w / img_h; can_w = min(img_w, MAX_W); can_h = int(can_w / ar)
                if can_h > MAX_H: can_h = MAX_H; can_w = int(can_h * ar)
                can_w, can_h = max(can_w, 150), max(can_h, 150); logger.debug(f"Canvas: {can_w}x{can_h}")
                # --- st_canvas call ---
                canvas_result = st_canvas(fill_color="rgba(255,165,0,0.2)", stroke_width=2, stroke_color="rgba(239,83,80,0.8)", background_image=display_img, update_streamlit=True, height=can_h, width=can_w, drawing_mode="rect", initial_drawing=st.session_state.get("canvas_drawing"), key="drawable_canvas")
                # --- ROI Processing ---
                if canvas_result.json_data and canvas_result.json_data.get("objects"):
                    if canvas_result.json_data["objects"]:
                        lobj = canvas_result.json_data["objects"][-1]
                        if lobj["type"] == "rect":
                            cl, ct = int(lobj["left"]), int(lobj["top"]); cws = int(lobj["width"]*lobj.get("scaleX",1)); chs = int(lobj["height"]*lobj.get("scaleY",1))
                            scx, scy = img_w/can_w, img_h/can_h; ol = max(0, int(cl*scx)); ot = max(0, int(ct*scy))
                            ow = max(1, min(img_w-ol, int(cws*scx))); oh = max(1, min(img_h-ot, int(chs*scy)))
                            new_roi = {"left":ol, "top":ot, "width":ow, "height":oh}
                            if st.session_state.roi_coords != new_roi: st.session_state.roi_coords=new_roi; st.session_state.canvas_drawing=canvas_result.json_data; logger.info(f"ROI set: {new_roi}"); st.info(f"ROI Set: ({ol},{ot}), Size: {ow}x{oh}", icon="🎯")
        else: st.image(display_img, caption="Preview", use_container_width=True)
        if st.session_state.roi_coords: r = st.session_state.roi_coords; st.caption(f"ROI: ({r['left']},{r['top']})-W:{r['width']},H:{r['height']}")
        st.markdown("---")
        if st.session_state.is_dicom and st.session_state.dicom_metadata: display_dicom_metadata(st.session_state.dicom_metadata)
    elif st.session_state.get("uploaded_file_info") and not display_img: st.error("❌ Image preview unavailable (processing failed).")
    else: st.info("⬅️ Upload an image or use Demo Mode.")

with col2: # Analysis Tabs
    st.subheader("📊 Analysis & Results")
    tabs = st.tabs(["🔬 Initial", "💬 Q&A", "🩺 Condition", "📚 UMLS", "📈 Confidence", "🌐 Translate"])
    with tabs[0]: st.text_area("Findings", value=st.session_state.initial_analysis or "Run Initial Analysis.", height=400, disabled=True, key="init_disp")
    with tabs[1]:
        st.text_area("Latest Answer", value=st.session_state.qa_answer or "Ask a question.", height=150, disabled=True, key="qa_disp")
        st.markdown("---"); st.subheader("History")
        if st.session_state.history:
            for i, (qt, m) in enumerate(reversed(st.session_state.history)):
                pfx = "👤:" if qt.lower().startswith("user") else "🤖:" if qt.lower().startswith("ai") else "ℹ️:" if qt.lower().startswith("sys") else f"**{qt}:**"
                unsafe = "ai" in qt.lower()
                st.markdown(f"{pfx} {m}", unsafe_allow_html=unsafe)
                if i < len(st.session_state.history)-1: st.markdown("---")
        else: st.caption("No Q&A yet.")
    with tabs[2]: st.text_area("Condition Analysis", value=st.session_state.disease_analysis or "Run Condition Analysis.", height=400, disabled=True, key="dis_disp")
    with tabs[3]: # UMLS
        st.subheader("📚 UMLS Search")
        if not UMLS_AVAILABLE: st.warning("UMLS unavailable.")
        else:
            term = st.text_input("Search term:", value=st.session_state.get("umls_search_term", ""), key="umls_in", placeholder="e.g., lung nodule")
            if st.button("🔎 Search UMLS", key="umls_btn"):
                if term.strip(): st.session_state.last_action="umls_search"; st.session_state.umls_search_term=term.strip(); st.rerun()
                else: st.warning("Enter search term.")
            if st.session_state.get("umls_error"): st.error(f"UMLS Error: {st.session_state.umls_error}")
            if UI_COMPONENTS_AVAILABLE: display_umls_concepts(st.session_state.get("umls_results"))
            else: st.caption("UMLS display unavailable.")
    with tabs[4]: st.text_area("Confidence", value=st.session_state.confidence_score or "Run Confidence Estimation.", height=400, disabled=True, key="conf_disp")
    with tabs[5]: # Translation
        st.subheader("🌐 Translate")
        if not TRANSLATION_AVAILABLE: st.warning("Translation unavailable.")
        else:
            st.caption("Select text, languages, translate.")
            txt_opts = {"Initial": st.session_state.initial_analysis, "Q&A": st.session_state.qa_answer, "Condition": st.session_state.disease_analysis, "Confidence": st.session_state.confidence_score, "(Custom)": ""}
            avail_opts = {lbl: txt for lbl, txt in txt_opts.items() if (txt and txt.strip()) or lbl == "(Custom)"}
            if not avail_opts: st.info("No text to translate.")
            else:
                sel_lbl = st.selectbox("Translate:", list(avail_opts.keys()), index=0, key="trans_sel")
                txt_to_trans = avail_opts.get(sel_lbl, "")
                if sel_lbl == "(Custom)": txt_to_trans = st.text_area("Custom text:", value="", height=100, key="trans_cust_in")
                else: st.text_area("Selected:", value=txt_to_trans, height=100, disabled=True, key="trans_sel_disp")
                cl1, cl2 = st.columns(2)
                with cl1: src_opts = [AUTO_DETECT_INDICATOR] + sorted(LANGUAGE_CODES.keys()); src_lang = st.selectbox("From:", src_opts, 0, key="trans_src")
                with cl2: tgt_opts = sorted(LANGUAGE_CODES.keys()); tgt_idx=0; common=["English","Spanish"];
                for i, l in enumerate(tgt_opts):
                    if l in common: tgt_idx=i; break
                tgt_lang = st.selectbox("To:", tgt_opts, tgt_idx, key="trans_tgt")
                if st.button("🔄 Translate", key="trans_btn"):
                    st.session_state.translation_result=None; st.session_state.translation_error=None
                    if not txt_to_trans.strip(): st.warning("No text."); st.session_state.translation_error="Empty input."
                    elif src_lang == tgt_lang and src_lang != AUTO_DETECT_INDICATOR: st.info("Same language."); st.session_state.translation_result=txt_to_trans
                    else:
                        with st.spinner("Translating..."):
                            try: out=translate(text=txt_to_trans, target_language=tgt_lang, source_language=src_lang)
                            if out is not None: st.session_state.translation_result=out; st.success("Translated!")
                            else: st.error("No result."); logger.error("Translate returned None."); st.session_state.translation_error="Service None."
                            except Exception as e: st.error(f"Failed: {e}"); logger.error(f"Trans err: {e}", exc_info=True); st.session_state.translation_error=str(e)
                if st.session_state.get("translation_result"): st.text_area("Result:", value=format_translation(st.session_state.translation_result), height=200, key="trans_out_disp")

# --- Button Action Handlers ---
current_action = st.session_state.get("last_action")
if current_action:
    logger.info(f"Handling action: '{current_action}'")
    req_img = current_action not in ["generate_report_data", "umls_search"]
    req_llm = current_action in ["analyze", "ask", "disease", "confidence"]
    req_rpt = (current_action == "generate_report_data")
    req_umls = (current_action == "umls_search")
    valid = True
    if req_img and not isinstance(st.session_state.get("processed_image"), Image.Image): st.error(f"No valid image for {current_action}."); valid = False
    if not st.session_state.session_id: st.error("No session ID."); valid = False
    if req_llm and not LLM_INTERACTIONS_AVAILABLE: st.error(f"AI module unavailable for {current_action}."); valid = False
    if req_rpt and not REPORT_UTILS_AVAILABLE: st.error(f"Report module unavailable."); valid = False
    if req_umls and not UMLS_AVAILABLE: st.error(f"UMLS module/key unavailable."); valid = False

    if valid:
        img = st.session_state.processed_image; roi = st.session_state.roi_coords; hist = st.session_state.history
        try: # --- Action Execution Logic (Keep as is) ---
            if current_action == "analyze":
                st.info("🔬 Analyzing..."); with st.spinner("AI analyzing..."): res = run_initial_analysis(img, roi)
                st.session_state.initial_analysis = res; st.session_state.qa_answer = ""; st.session_state.disease_analysis = ""; logger.info("Analyze OK."); st.success("Analysis complete!")
            elif current_action == "ask":
                q = st.session_state.question_input_widget.strip()
                if not q: st.warning("Question empty.")
                else:
                    st.info(f"❓ Asking: '{q}'..."); st.session_state.qa_answer = "";
                    with st.spinner("AI thinking..."): ans, ok = run_multimodal_qa(img=img, question=q, history=hist, roi=roi)
                    if ok: st.session_state.qa_answer = ans; hist.append(("User", q)); hist.append(("AI", ans)); logger.info("Ask OK."); st.success("AI answered!")
                    else: # Primary Fail + Fallback (Keep as is)
                        err = f"Primary AI fail: {ans}"; st.session_state.qa_answer = err; st.error(err); logger.error(f"Ask fail: {ans}")
                        tok=os.getenv("HF_API_TOKEN")
                        if HF_MODELS_AVAILABLE and tok:
                            st.info(f"Trying fallback: {HF_VQA_MODEL_ID}"); logger.debug("Trying HF fallback.")
                            with st.spinner("Fallback..."): fb_ans, fb_ok = query_hf_vqa_inference_api(img=img, question=q, roi=roi)
                            if fb_ok: fb_disp=f"**[Fallback]**\n{fb_ans}"; st.session_state.qa_answer+=f"\n\n{fb_disp}"; hist.append(("[FB] User",q)); hist.append(("[FB] AI", fb_disp)); logger.info("Fallback OK."); st.success("Fallback answered.")
                            else: fb_err=f"[FB Error]: {fb_ans}"; st.session_state.qa_answer+=f"\n\n{fb_err}"; logger.error(f"Fallback fail: {fb_ans}"); st.error(fb_err)
                        elif HF_MODELS_AVAILABLE: st.session_state.qa_answer+="\n\n[FB Skip: Token missing]"; logger.warning("FB skip: token."); st.warning("HF Token missing.")
                        else: st.session_state.qa_answer+="\n\n[FB Unavailable]"; logger.warning("FB unavailable."); st.warning("Fallback unavailable.")
            elif current_action == "disease":
                d = st.session_state.disease_select_widget
                if not d: st.warning("No condition selected.")
                else:
                    st.info(f"🩺 Analyzing for '{d}'..."); with st.spinner(f"Analyzing {d}..."): res = run_disease_analysis(img, d, roi)
                    st.session_state.disease_analysis = res; st.session_state.qa_answer = ""; logger.info(f"Disease '{d}' OK."); st.success(f"Analysis for '{d}' complete!")
            elif current_action == "umls_search":
                t = st.session_state.get("umls_search_term","").strip(); st.session_state.umls_results=None; st.session_state.umls_error=None
                if not t: st.warning("UMLS term empty.")
                else:
                    st.info(f"🔎 UMLS: '{t}'..."); with st.spinner("Querying..."):
                        try: res = umls_utils.search_umls(t, UMLS_APIKEY); st.session_state.umls_results=res; logger.info(f"UMLS OK ({len(res)})."); st.success(f"Found {len(res)} concepts.")
                        except UMLSAuthError as e: err=f"UMLS Auth Fail: {e}"; st.error(err); logger.error(err); st.session_state.umls_error=f"Auth: {e}"
                        except RuntimeError as e: err=f"UMLS Search Fail: {e}"; st.error(err); logger.error(err,exc_info=True); st.session_state.umls_error=f"Search: {e}"
                        except Exception as e: err=f"UMLS Unexpected: {e}"; st.error(err); logger.critical(err,exc_info=True); st.session_state.umls_error=f"Unexpected: {e}"
            elif current_action == "confidence":
                if not (hist or st.session_state.initial_analysis or st.session_state.disease_analysis): st.warning("No analysis for confidence.")
                else: st.info("📊 Estimating confidence..."); with st.spinner("Calculating..."): res=estimate_ai_confidence(img=img,history=hist,initial_analysis=st.session_state.initial_analysis,disease_analysis=st.session_state.disease_analysis,roi=roi)
                st.session_state.confidence_score=res; logger.info("Confidence OK."); st.success("Confidence estimated!")
            elif current_action == "generate_report_data":
                st.info("📄 Generating report..."); st.session_state.pdf_report_bytes=None; img_rep=st.session_state.get("display_image")
                if not isinstance(img_rep, Image.Image): st.error("No valid image for report."); logger.error("Report fail: no image.")
                else: # Report Generation (Keep logic as is)
                    pdf_img = img_rep.copy().convert("RGB")
                    if roi and ImageDraw:
                        try: draw=ImageDraw.Draw(pdf_img); x0,y0,x1,y1=roi['left'],roi['top'],roi['left']+roi['width'],roi['top']+roi['height']; draw.rectangle([x0,y0,x1,y1],outline="red",width=max(2,int(min(pdf_img.size)*0.004))); logger.info("Drew ROI for PDF.")
                        except Exception as e: logger.error(f"ROI draw err: {e}"); st.warning("Could not draw ROI.")
                    hist_fmt="No Q&A.";
                    if hist: hist_fmt="\n\n".join([f"{qt}: {re.sub('<[^<]+?>','',str(m))}" for qt, m in hist])
                    rep_data={"Session ID":st.session_state.session_id, "Image Filename":(st.session_state.uploaded_file_info or "N/A").split('-')[0], "Initial Analysis":st.session_state.initial_analysis or "N/P", "Conversation History":hist_fmt, "Condition Analysis":st.session_state.disease_analysis or "N/P", "AI Confidence":st.session_state.confidence_score or "N/P"}
                    if st.session_state.is_dicom and st.session_state.dicom_metadata: meta_keys=['PatientName','PatientID','StudyDate','Modality','StudyDescription']; meta_sum={k:st.session_state.dicom_metadata.get(k,'N/A') for k in meta_keys if st.session_state.dicom_metadata.get(k)}; if meta_sum: rep_data["DICOM Summary"]="\n".join([f"{k}: {v}" for k,v in meta_sum.items()])
                    with st.spinner("Generating PDF..."): pdf_bytes=generate_pdf_report_bytes(session_id=st.session_state.session_id, image=pdf_img, analysis_outputs=rep_data, dicom_metadata=st.session_state.dicom_metadata if st.session_state.is_dicom else None)
                    if pdf_bytes: st.session_state.pdf_report_bytes=pdf_bytes; st.success("PDF ready!"); logger.info("PDF OK."); st.balloons()
                    else: st.error("PDF generation failed."); logger.error("PDF gen fail.")
            else: st.warning(f"Unknown action '{current_action}'.")
        except Exception as e: st.error(f"Error during '{current_action}': {e}"); logger.critical(f"Action '{current_action}' error: {e}", exc_info=True)
        finally: st.session_state.last_action = None; logger.debug(f"Action '{current_action}' complete."); st.rerun()
    else: st.session_state.last_action = None; logger.warning(f"Action '{current_action}' cancelled (invalid).")

# --- Footer ---
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced | Session ID: {st.session_state.get('session_id', 'N/A')}")
st.markdown("""<footer> <p>RadVision AI is for informational/educational use ONLY. Not medical advice.</p> <p> <a href="#" target="_blank">Privacy</a> | <a href="#" target="_blank">Terms</a> | <a href="https://github.com/mgbam/radvisionai" target="_blank">GitHub</a> </p></footer>""", unsafe_allow_html=True)
logger.info(f"--- Render complete: {st.session_state.session_id} ---")