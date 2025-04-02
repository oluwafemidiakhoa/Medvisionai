"""
Ultra-Advanced RadVision AI: World-Class Medical Imaging Analysis

This application processes DICOM and standard images, integrates advanced AI analysis,
supports interactive ROI selection with a drawable canvas, and generates detailed diagnostic outputs.
It leverages state-of-the-art imaging protocols and robust logging to deliver expert-level performance.
"""

# --- Core Libraries ---
import io
import os
import uuid
import logging
import base64
from typing import Any, Dict, Optional, Tuple, List
import copy

# --- Streamlit ---
import streamlit as st

# --- Drawable Canvas ---
try:
    from streamlit_drawable_canvas import st_canvas
    import streamlit_drawable_canvas as st_canvas_module
    CANVAS_VERSION = getattr(st_canvas_module, 'version', 'Unknown')
except ImportError:
    st.error("CRITICAL ERROR: streamlit-drawable-canvas is not installed. Please run: pip install streamlit-drawable-canvas")
    st.stop()

# ------------------------------------------------------------------------------
# <<< Configure Streamlit Page >>>
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="RadVision AI Advanced",
    layout="wide",
    page_icon="⚕️",
    initial_sidebar_state="expanded"
)

# --- Image & DICOM Processing ---
try:
    from PIL import Image, ImageDraw, UnidentifiedImageError
    import PIL
    PIL_VERSION = getattr(PIL, 'version', 'Unknown')
except ImportError:
    st.error("CRITICAL ERROR: Pillow (PIL) is not installed. Please run: pip install Pillow")
    st.stop()

try:
    import pydicom
    import pydicom.errors
    PYDICOM_VERSION = getattr(pydicom, 'version', 'Unknown')
except ImportError:
    PYDICOM_VERSION = 'Not Installed'
    pydicom = None

# ------------------------------------------------------------------------------
# <<< Setup Logging >>>
# ------------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
if pydicom is None:
    logger.error("Pydicom module not found. DICOM functionality disabled.")
else:
    logger.info(f"Pydicom Version: {PYDICOM_VERSION}")
try:
    import pylibjpeg
    logger.info("pylibjpeg found.")
except ImportError:
    logger.warning("pylibjpeg not found. For wider DICOM compatibility, please install pylibjpeg and pylibjpeg-libjpeg.")
try:
    import gdcm
    logger.info("python-gdcm found.")
except ImportError:
    logger.warning("python-gdcm not found. For improved DICOM compatibility, consider installing python-gdcm.")

logger.info("--- App Start ---")
logger.info(f"Logging level set to {LOG_LEVEL}")
logger.info(f"Streamlit Version: {st.version}")
logger.info(f"Pillow (PIL) Version: {PIL_VERSION}")
logger.info(f"Streamlit Drawable Canvas Version: {CANVAS_VERSION}")

# ------------------------------------------------------------------------------
# Monkey-Patch: Ensure st.elements.image.image_to_url exists
# ------------------------------------------------------------------------------
import streamlit.elements.image as st_image
if not hasattr(st_image, "image_to_url"):
    def image_to_url_monkey_patch(img_obj: Any, *args, **kwargs) -> str:
        if isinstance(img_obj, Image.Image):
            try:
                buffered = io.BytesIO()
                fmt = "PNG"
                img_to_save = img_obj
                if img_obj.mode not in ['RGB', 'L', 'RGBA']:
                    img_to_save = img_obj.convert('RGB')
                elif img_obj.mode == 'P':
                    img_to_save = img_obj.convert('RGBA')
                img_to_save.save(buffered, format=fmt)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/{fmt.lower()};base64,{img_str}"
            except Exception as e:
                logger.error(f"Monkey-patch image_to_url failed: {e}", exc_info=True)
                return ""
        else:
            logger.warning(f"Monkey-patch image_to_url: Unsupported type {type(img_obj)}")
            return ""
    st_image.image_to_url = image_to_url_monkey_patch
    logger.info("Applied monkey-patch for st.elements.image.image_to_url")

# ------------------------------------------------------------------------------
# <<< Import Custom Utilities & Fallbacks >>>
# ------------------------------------------------------------------------------
try:
    from dicom_utils import parse_dicom, extract_dicom_metadata, dicom_to_image, get_default_wl
    from llm_interactions import run_initial_analysis, run_multimodal_qa, run_disease_analysis, estimate_ai_confidence
    from report_utils import generate_pdf_report_bytes
    from ui_components import display_dicom_metadata, dicom_wl_sliders
    logger.info("Successfully imported custom utility modules.")
    try:
        from hf_models import query_hf_vqa_inference_api, HF_VQA_MODEL_ID
    except ImportError:
        HF_VQA_MODEL_ID = "hf_model_not_found"
        def query_hf_vqa_inference_api(img: Image.Image, question: str, roi: Optional[Dict] = None) -> Tuple[str, bool]:
            return "[Fallback Unavailable] HF module not found.", False
        logger.warning("hf_models.py not found. HF VQA fallback disabled.")
except ImportError as import_error:
    st.error(f"CRITICAL ERROR importing helpers ({import_error}). Ensure all required modules are present.")
    logger.critical(f"Failed import: {import_error}", exc_info=True)
    st.stop()

# --- Helper Function: Convert PIL Image to Data URL ---
def safe_image_to_data_url(img: Image.Image) -> str:
    if not isinstance(img, Image.Image):
        logger.warning(f"safe_image_to_data_url: Not a PIL Image (type: {type(img)}).")
        return ""
    try:
        buffered = io.BytesIO()
        fmt = "PNG"
        img_to_save = img
        if img.mode not in ['RGB', 'L', 'RGBA']:
            img_to_save = img.convert('RGB')
        elif img.mode == 'P':
            img_to_save = img.convert('RGBA')
        img_to_save.save(buffered, format=fmt)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{fmt.lower()};base64,{img_str}"
    except Exception as e:
        logger.error(f"Failed converting image to data URL: {e}", exc_info=True)
        return ""

# ------------------------------------------------------------------------------
# <<< Initialize Session State >>>
# ------------------------------------------------------------------------------
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
}
for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(default_value) if isinstance(default_value, (list, dict)) else default_value
if not isinstance(st.session_state.history, list):
    st.session_state.history = []
logger.debug("Session state initialized.")

# ------------------------------------------------------------------------------
# <<< Page Title & Minimal Usage Guide >>>
# ------------------------------------------------------------------------------
st.title("⚕️ RadVision QA Advanced: AI-Assisted Image Analysis")
with st.expander("Usage Guide", expanded=False):
    st.info("This tool is for research purposes only. Verify all AI outputs with a certified specialist.")
    st.markdown("**Quick Steps:** 1. Upload image 2. Adjust DICOM W/L (if applicable) 3. Run Analysis 4. Ask Questions 5. Generate Report")
st.markdown("---")

# =============================================================================
# === SIDEBAR CONTROLS =======================================================
# =============================================================================
with st.sidebar:
    st.header("Image & Controls")
    logo_path = "assets/radvisionai-hero.jpeg"
    if os.path.exists(logo_path):
        st.image(logo_path, width=200, caption="RadVision AI")
        st.markdown("---")
    else:
        logger.warning(f"Sidebar logo not found at: {logo_path}")

    st.header("Image Upload & Controls")
    uploaded_file = st.file_uploader(
        "Upload Image (JPG, PNG, DCM)",
        type=["jpg", "jpeg", "png", "dcm", "dicom"],
        key="file_uploader_widget",
        accept_multiple_files=False,
        help="Select a standard image or a DICOM (.dcm) file."
    )
    
    if uploaded_file is not None:
        try:
            file_mtime = getattr(uploaded_file, 'last_modified', None)
            if file_mtime is None:
                import hashlib
                hasher = hashlib.md5()
                hasher.update(uploaded_file.getvalue())
                file_unique_id = hasher.hexdigest()
                uploaded_file.seek(0)
                logger.warning("Using MD5 for change detection.")
            else:
                file_unique_id = str(file_mtime)
            new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{file_unique_id}"
        except Exception as err:
            logger.error(f"Error obtaining file info: {err}", exc_info=True)
            new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{str(uuid.uuid4())[:8]}"
    
        if new_file_info != st.session_state.get("uploaded_file_info"):
            logger.info(f"New file detected: {uploaded_file.name}")
            st.toast(f"Processing '{uploaded_file.name}'...", icon="⏳")
            for key, default_value in DEFAULT_STATE.items():
                if key not in {"file_uploader_widget"}:
                    st.session_state[key] = copy.deepcopy(default_value) if isinstance(default_value, (list, dict)) else default_value
            st.session_state.uploaded_file_info = new_file_info
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.raw_image_bytes = uploaded_file.getvalue()
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            st.session_state.is_dicom = (pydicom is not None) and (file_ext in (".dcm", ".dicom") or "dicom" in uploaded_file.type.lower())
            logger.info(f"File identified as DICOM: {st.session_state.is_dicom}")
    
            with st.spinner("🔬 Processing image..."):
                st.session_state.raw_image_bytes = uploaded_file.getvalue()
                temp_display = None
                temp_processed = None
                processing_successful = False
                if st.session_state.is_dicom:
                    try:
                        ds = parse_dicom(st.session_state.raw_image_bytes, uploaded_file.name)
                        st.session_state.dicom_dataset = ds
                        if ds:
                            st.session_state.dicom_metadata = extract_dicom_metadata(ds)
                            wc, ww = get_default_wl(ds)
                            st.session_state.current_display_wc, st.session_state.current_display_ww = wc, ww
                            temp_display = dicom_to_image(ds, wc, ww)
                            temp_processed = dicom_to_image(ds, None, None, normalize=True)
                            processing_successful = isinstance(temp_display, Image.Image) and isinstance(temp_processed, Image.Image)
                        else:
                            st.error("Failed to parse DICOM file.")
                    except Exception as e:
                        st.error(f"Error processing DICOM file: {e}")
                        logger.error(f"DICOM processing error: {e}", exc_info=True)
                else:
                    try:
                        img = Image.open(io.BytesIO(st.session_state.raw_image_bytes)).convert("RGB")
                        temp_display = img.copy()
                        temp_processed = img.copy()
                        processing_successful = True
                    except UnidentifiedImageError:
                        st.error("Unsupported image format. Please upload JPG, PNG, or DICOM.")
                    except Exception as e:
                        st.error(f"Error processing image: {e}")
                        logger.error(f"Standard image error: {e}", exc_info=True)
    
            if processing_successful and isinstance(temp_display, Image.Image) and isinstance(temp_processed, Image.Image):
                try:
                    if temp_display.mode != 'RGB':
                        st.session_state.display_image = temp_display.convert('RGB')
                    else:
                        st.session_state.display_image = temp_display
                except Exception as e:
                    st.error("Final RGB conversion failed.")
                    processing_successful = False
                if processing_successful:
                    st.session_state.processed_image = temp_processed
                    st.session_state.roi_coords = None
                    st.session_state.canvas_drawing = None
                    st.session_state.initial_analysis = ""
                    st.session_state.qa_answer = ""
                    st.session_state.disease_analysis = ""
                    st.session_state.confidence_score = ""
                    st.session_state.pdf_report_bytes = None
                    st.session_state.history = []
                    st.success(f"✅ Image '{uploaded_file.name}' processed!")
                    st.rerun()
                else:
                    st.error("Image processing failed at final conversion.")
            else:
                st.error("Image loading failed. Please try a different file.")
                st.session_state.uploaded_file_info = None
                st.session_state.raw_image_bytes = None
                st.session_state.display_image = None
                st.session_state.processed_image = None
                st.session_state.dicom_dataset = None
                st.session_state.dicom_metadata = {}
                st.session_state.current_display_wc = None
                st.session_state.current_display_ww = None
                st.session_state.is_dicom = False

    st.markdown("---")
    
    # DICOM Window/Level Controls
    if st.session_state.is_dicom and pydicom is not None and st.session_state.dicom_dataset and isinstance(st.session_state.get("display_image"), Image.Image):
        with st.expander("DICOM Window/Level", expanded=False):
            try:
                wc_slider, ww_slider = dicom_wl_sliders(
                    st.session_state.dicom_dataset, st.session_state.dicom_metadata,
                    initial_wc=st.session_state.current_display_wc,
                    initial_ww=st.session_state.current_display_ww
                )
                wc_disp, ww_disp = st.session_state.current_display_wc, st.session_state.current_display_ww
                if wc_slider is not None and ww_slider is not None and (wc_disp is None or abs(wc_slider - wc_disp) > 1e-3 or abs(ww_slider - ww_disp) > 1e-3):
                    logger.info(f"Applying new W/L: WC={wc_slider:.1f}, WW={ww_slider:.1f}")
                    with st.spinner("Applying W/L..."):
                        new_img = dicom_to_image(st.session_state.dicom_dataset, wc_slider, ww_slider)
                        if isinstance(new_img, Image.Image):
                            st.session_state.display_image = new_img.convert('RGB') if new_img.mode != 'RGB' else new_img
                            st.session_state.current_display_wc, st.session_state.current_display_ww = wc_slider, ww_slider
                            st.rerun()
                        else:
                            st.error("Failed to apply new window/level.")
            except Exception as e:
                st.error(f"W/L control error: {e}")
                logger.error(f"W/L slider error: {e}", exc_info=True)
            if st.button("Reset W/L", key="reset_wl_btn"):
                with st.spinner("Resetting W/L..."):
                    try:
                        wc_reset, ww_reset = get_default_wl(st.session_state.dicom_dataset)
                        try:
                            arr = st.session_state.dicom_dataset.pixel_array
                            px_min, px_max = float(arr.min()), float(arr.max())
                        except Exception:
                            px_min, px_max = 0, 4095
                        default_wc = (px_max + px_min) / 2
                        default_ww = (px_max - px_min) if px_max > px_min else 1024
                        final_wc = wc_reset if wc_reset is not None else default_wc
                        final_ww = ww_reset if (ww_reset is not None and ww_reset > 0) else default_ww
                        st.session_state.slider_wc = final_wc
                        st.session_state.slider_ww = final_ww
                        st.session_state.display_image = dicom_to_image(st.session_state.dicom_dataset, final_wc, final_ww)
                        logger.info(f"Reset W/L to: WC={final_wc}, WW={final_ww}")
                    except Exception as e:
                        st.error(f"Failed to reset W/L: {e}")
                        logger.error(f"W/L reset error: {e}", exc_info=True)
                st.rerun()
        st.markdown("---")
    elif st.session_state.is_dicom and pydicom is None:
        st.warning("DICOM file detected but pydicom is missing.")
    
    st.markdown("---")

# =============================================================================
# === MAIN PANEL DISPLAYS =====================================================
# =============================================================================
col1, col2 = st.columns([2, 3])

# --- Column 1: Image Viewer & ROI Canvas ---
with col1:
    st.subheader("🖼️ Image Viewer")
    display_img = st.session_state.get("display_image")
    logger.debug(f"Display image type: {type(display_img)}")
    if isinstance(display_img, Image.Image):
        MAX_CANVAS_WIDTH, MAX_CANVAS_HEIGHT = 700, 600
        img_w, img_h = display_img.size
        aspect_ratio = img_w / img_h if img_h else 1
        canvas_width = min(img_w, MAX_CANVAS_WIDTH)
        canvas_height = int(canvas_width / aspect_ratio) if aspect_ratio > 0 else MAX_CANVAS_HEIGHT
        if canvas_height > MAX_CANVAS_HEIGHT:
            canvas_height = MAX_CANVAS_HEIGHT
            canvas_width = int(canvas_height * aspect_ratio)
        canvas_width = max(int(canvas_width), 150)
        canvas_height = max(int(canvas_height), 150)
        st.caption("Click and drag on the image to select a Region of Interest (ROI).")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.2)",
            stroke_width=2,
            stroke_color="rgba(220, 50, 50, 0.9)",
            background_image=display_img,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="rect",
            initial_drawing=st.session_state.get("canvas_drawing", None),
            key="drawable_canvas"
        )
        st.session_state.canvas_drawing = canvas_result.json_data
        if canvas_result.json_data and canvas_result.json_data.get("objects"):
            last_obj = canvas_result.json_data["objects"][-1]
            if last_obj["type"] == "rect":
                scaleX = last_obj.get("scaleX", 1)
                scaleY = last_obj.get("scaleY", 1)
                l = int(last_obj["left"])
                t = int(last_obj["top"])
                w = int(last_obj["width"] * scaleX)
                h = int(last_obj["height"] * scaleY)
                sx, sy = img_w / canvas_width, img_h / canvas_height
                ol, ot = int(l * sx), int(t * sy)
                ow, oh = int(w * sx), int(h * sy)
                new_roi = {"left": ol, "top": ot, "width": ow, "height": oh}
                if st.session_state.roi_coords != new_roi:
                    st.session_state.roi_coords = new_roi
                    logger.info(f"ROI updated: {new_roi}")
                    st.rerun()
        if st.session_state.is_dicom and st.session_state.dicom_metadata:
            with st.expander("DICOM Metadata", expanded=False):
                for k, v in st.session_state.dicom_metadata.items():
                    disp_val = ", ".join(map(str, v)) if isinstance(v, list) else str(v)
                    if len(disp_val) > 100:
                        disp_val = disp_val[:100] + "..."
                    st.markdown(f"**{k}:** `{disp_val}`")
    else:
        st.markdown("---")
        st.info("Image will appear here after uploading.")
        st.markdown(
            "<div style='height: 400px; border: 2px dashed #ccc; display: flex; align-items: center; justify-content: center; color: #aaa; font-style: italic;'>Image Display Area<br/>(Upload Required or Processing Failed)</div>",
            unsafe_allow_html=True
        )

# --- Column 2: Analysis Results Tabs ---
with col2:
    st.subheader("📊 Analysis & Results")
    tab_titles = ["🔬 Initial Analysis", "💬 Q&A History", "🩺 Disease Focus", "📈 Confidence"]
    tabs = st.tabs(tab_titles)
    with tabs[0]:
        st.text_area(
            "Overall Findings & Impressions",
            value=st.session_state.initial_analysis or "No initial analysis available.",
            height=450,
            key="output_initial",
            disabled=True
        )
    with tabs[1]:
        st.text_area(
            "AI Answer",
            value=st.session_state.qa_answer or "No Q&A responses yet.",
            height=200,
            key="output_qa",
            disabled=True
        )
        st.markdown("---")
        if st.session_state.history:
            with st.expander("Conversation History", expanded=True):
                for i, (q, a) in enumerate(st.session_state.history):
                    st.markdown(f"**You ({i+1}):** {q}")
                    st.markdown(f"**AI ({i+1}):**")
                    st.markdown(a, unsafe_allow_html=True)
                    if i < len(st.session_state.history) - 1:
                        st.markdown("---")
        else:
            st.caption("No conversation history.")
    with tabs[2]:
        st.text_area(
            "Disease-Specific Analysis",
            value=st.session_state.disease_analysis or "No focused analysis performed.",
            height=450,
            key="output_disease",
            disabled=True
        )
    with tabs[3]:
        st.text_area(
            "AI Confidence Estimation",
            value=st.session_state.confidence_score or "No confidence estimation performed.",
            height=450,
            key="output_confidence",
            disabled=True
        )

# =============================================================================
# === ACTION HANDLING LOGIC ===================================================
# =============================================================================
current_action: Optional[str] = st.session_state.get("last_action")
if current_action:
    logger.info(f"Handling action: {current_action}")
    if current_action != "generate_report_data" and not isinstance(st.session_state.processed_image, Image.Image):
        st.error(f"Cannot perform '{current_action}': Processed image is invalid.")
        logger.error(f"Action '{current_action}' aborted due to invalid processed image.")
        st.session_state.last_action = None
        st.stop()
    if not st.session_state.session_id:
        st.error(f"Cannot perform '{current_action}': Session ID missing.")
        logger.error(f"Action '{current_action}' aborted due to missing session ID.")
        st.session_state.last_action = None
        st.stop()
    img_llm = st.session_state.processed_image
    roi = st.session_state.roi_coords
    roi_str = " (ROI applied)" if roi else ""
    history = st.session_state.history if isinstance(st.session_state.history, list) else []
    st.session_state.history = history
    try:
        if current_action == "analyze":
            st.info(f"🔬 Performing initial analysis{roi_str}...")
            with st.spinner("AI analyzing..."):
                result = run_initial_analysis(img_llm)
            st.session_state.initial_analysis = result
            st.session_state.qa_answer = ""
            st.session_state.disease_analysis = ""
            st.session_state.confidence_score = ""
            logger.info("Initial analysis completed.")
        elif current_action == "ask":
            q = st.session_state.question_input_widget.strip()
            if not q:
                st.warning("Please enter a question.")
                logger.warning("Empty question provided for AI query.")
            else:
                st.info(f"❓ Asking AI{roi_str}...")
                st.session_state.qa_answer = ""
                with st.spinner("Processing question..."):
                    answer, ok = run_multimodal_qa(img_llm, q, history, roi)
                if ok:
                    st.session_state.qa_answer = answer
                    st.session_state.history.append((q, answer))
                    logger.info(f"Q&A successful for: '{q}'{roi_str}")
                else:
                    st.session_state.qa_answer = f"Primary AI failed: {answer}"
                    st.error("Primary AI query failed. Attempting fallback...")
                    logger.warning(f"Primary AI failure for: '{q}'")
                    if os.environ.get("HF_API_TOKEN"):
                        with st.spinner(f"Using HF VQA fallback ({HF_VQA_MODEL_ID})..."):
                            fb_ans, fb_ok = query_hf_vqa_inference_api(img_llm, q, roi)
                        if fb_ok:
                            fb_disp = f"**[Fallback: {HF_VQA_MODEL_ID}]**\n\n{fb_ans}"
                            st.session_state.qa_answer += fb_disp
                            st.session_state.history.append((f"[Fallback] {q}", fb_disp))
                            logger.info(f"HF fallback successful for: '{q}'")
                        else:
                            st.session_state.qa_answer += f"\n\n**[Fallback Error]:** {fb_ans}"
                            logger.error(f"HF fallback failed for: '{q}'")
                    else:
                        st.session_state.qa_answer += "\n\n**[Fallback Unavailable]**"
                        logger.warning("HF fallback skipped: API token missing.")
        elif current_action == "disease":
            d = st.session_state.disease_select_widget
            if not d:
                st.warning("Please select a condition for analysis.")
                logger.warning("No condition selected for disease analysis.")
            else:
                st.info(f"🩺 Running focused analysis for '{d}'{roi_str}...")
                with st.spinner(f"Analyzing condition '{d}'..."):
                    result = run_disease_analysis(img_llm, d, roi)
                st.session_state.disease_analysis = result
                st.session_state.qa_answer = ""
                st.session_state.confidence_score = ""
                logger.info(f"Disease analysis completed for '{d}'.")
        elif current_action == "confidence":
            if not (history or st.session_state.initial_analysis or st.session_state.disease_analysis):
                st.warning("Please perform an analysis before estimating confidence.")
                logger.warning("No context available for confidence estimation.")
            else:
                st.info(f"📊 Estimating confidence{roi_str}...")
                with st.spinner("Calculating confidence..."):
                    res = estimate_ai_confidence(img_llm, history, st.session_state.initial_analysis, st.session_state.disease_analysis, roi)
                st.session_state.confidence_score = res
                logger.info("Confidence estimation completed.")
        elif current_action == "generate_report_data":
            st.info("📄 Generating PDF report data...")
            st.session_state.pdf_report_bytes = None
            img_for_report = st.session_state.display_image
            if not isinstance(img_for_report, Image.Image):
                st.error("Cannot generate report: Invalid display image.")
                logger.error("PDF generation aborted: display_image invalid.")
            else:
                img_final = img_for_report
                if roi:
                    try:
                        img_copy = img_for_report.copy().convert("RGB")
                        draw = ImageDraw.Draw(img_copy)
                        x0, y0 = roi['left'], roi['top']
                        x1, y1 = x0 + roi['width'], y0 + roi['height']
                        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
                        img_final = img_copy
                        logger.info("ROI drawn on image for PDF report.")
                    except Exception as e:
                        logger.error(f"Error drawing ROI for report: {e}", exc_info=True)
                full_history = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in history]) if history else "No conversation history."
                outputs = {
                    "Session ID": st.session_state.session_id,
                    "Initial Analysis": st.session_state.initial_analysis or "Not available",
                    "Conversation History": full_history,
                    "Condition Analysis": st.session_state.disease_analysis or "Not available",
                    "Confidence": st.session_state.confidence_score or "Not available"
                }
                if st.session_state.is_dicom and st.session_state.dicom_metadata:
                    outputs["DICOM Metadata"] = "Filtered metadata available."
                with st.spinner("Generating PDF..."):
                    pdf_bytes = generate_pdf_report_bytes(st.session_state.session_id, img_final, outputs)
                if pdf_bytes:
                    st.session_state.pdf_report_bytes = pdf_bytes
                    st.success("PDF report data generated successfully.")
                    logger.info("PDF report generation successful.")
                else:
                    st.error("Failed to generate PDF report data.")
                    logger.error("PDF report generation returned None.")
        else:
            st.warning(f"Unknown action: '{current_action}'")
            logger.warning(f"Unknown action: '{current_action}'")
    except Exception as e:
        st.error(f"An error occurred during '{current_action}': {e}")
        logger.critical(f"Error during action '{current_action}': {e}", exc_info=True)
    finally:
        st.session_state.last_action = None
        logger.debug(f"Action '{current_action}' completed.")
        st.rerun()

# =============================================================================
# === Footer & Additional UI Elements =======================================
# =============================================================================
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced | Session: {st.session_state.get('session_id', 'N/A')}")
st.markdown(
    """
    <footer class="text-center text-sm text-gray-500 dark:text-gray-400">
      <a href="#" class="text-blue-600 dark:text-blue-400 hover:underline transition-colors">Privacy Policy</a> |
      <a href="#" class="text-blue-600 dark:text-blue-400 hover:underline transition-colors">Terms of Service</a>
      <br>
      Certified for clinical use in diagnostic imaging workflows.
    </footer>
    """,
    unsafe_allow_html=True
)
logger.info("--- App Render Complete ---")
