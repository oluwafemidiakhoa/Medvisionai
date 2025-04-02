# main_app.py (Revision - Re-enable Monkey-Patch)

# --- Core Libraries ---
# ... (imports for io, os, uuid, logging, base64, typing, copy) ...
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
    CANVAS_VERSION = getattr(st_canvas_module, '__version__', 'Unknown')
except ImportError:
    st.error("CRITICAL ERROR: streamlit-drawable-canvas is not installed. `pip install streamlit-drawable-canvas`")
    st.stop()

# ------------------------------------------------------------------------------
# <<< --- Configure Streamlit Page (MUST BE FIRST st COMMAND) --- >>>
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
    PIL_VERSION = getattr(PIL, '__version__', 'Unknown')
except ImportError:
    st.error("CRITICAL ERROR: Pillow (PIL) is not installed. `pip install Pillow`")
    st.stop()
try:
    import pydicom
    import pydicom.errors
    PYDICOM_VERSION = getattr(pydicom, '__version__', 'Unknown')
except ImportError:
    PYDICOM_VERSION = 'Not Installed'
    pydicom = None


# ------------------------------------------------------------------------------
# <<< --- Setup Logging (After set_page_config) --- >>>
# ------------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper() # Default back to INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ... (logging setup continues) ...
if pydicom is None: logger.info("Pydicom module not found. DICOM functionality disabled.")
else:
    logger.info(f"Pydicom Version: {PYDICOM_VERSION}")
    # ... (pylibjpeg, gdcm checks) ...

logger.info(f"--- App Start ---")
logger.info(f"Logging level set to {LOG_LEVEL}")
logger.info(f"Streamlit Version: {st.__version__}")
logger.info(f"Pillow (PIL) Version: {PIL_VERSION}")
logger.info(f"Streamlit Drawable Canvas Version: {CANVAS_VERSION}")


# ------------------------------------------------------------------------------
# <<< --- Monkey-Patch to Add image_to_url if Missing --- >>>
# Ensure this block is UNCOMMENTED and PRESENT
# ------------------------------------------------------------------------------
import streamlit.elements.image as st_image
if not hasattr(st_image, "image_to_url"):
    logger.warning("Streamlit version seems to be missing 'image_to_url'. Applying monkey-patch.")
    def image_to_url_monkey_patch(
        image: Any,
        width: int,
        clamp: bool,
        channels: str,
        output_format: str,
        image_id: str,
        allow_emoji: bool = False,
        ) -> str:
        """
        Monkey-patch implementation for Streamlit's internal image_to_url.
        Converts PIL Image to a data URL.
        Note: Parameters match expected internal signature; some might not be used in basic conversion.
        """
        try:
            if isinstance(image, Image.Image):
                # Simple conversion for the patch
                buffered = io.BytesIO()
                img_format = output_format.upper() if output_format else "PNG"
                if img_format == "JPEG": img_format = "JPEG" # Common synonym

                # Ensure image is in a savable mode (like RGB/RGBA for PNG/JPEG)
                img_to_save = image
                if image.mode not in ['RGB', 'RGBA', 'L'] and img_format in ['PNG', 'JPEG']:
                    logger.debug(f"Patch: Converting image mode {image.mode} to RGB for saving as {img_format}")
                    img_to_save = image.convert("RGB")
                elif image.mode == 'P' and img_format == 'PNG':
                     logger.debug(f"Patch: Converting image mode P to RGBA for saving as PNG")
                     img_to_save = image.convert("RGBA")


                img_to_save.save(buffered, format=img_format)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                data_url = f"data:image/{img_format.lower()};base64,{img_str}"
                # logger.debug(f"Patch: Successfully created data URL (len: {len(data_url)})")
                return data_url
            else:
                # If it's not a PIL image, we can't handle it here easily.
                # The original function might handle URLs, paths, etc.
                # This patch focuses on the PIL Image case for st_canvas.
                logger.warning(f"Patch: image_to_url received non-PIL image type: {type(image)}. Returning empty string.")
                return ""
        except Exception as e:
            logger.error(f"Monkey-patch image_to_url failed: {e}", exc_info=True)
            return "" # Return empty string on error

    # Apply the patch
    st_image.image_to_url = image_to_url_monkey_patch
    logger.info("Successfully applied monkey-patch for streamlit.elements.image.image_to_url.")
else:
    logger.info("Streamlit version has 'image_to_url' attribute. No patch needed.")


# ------------------------------------------------------------------------------
# <<< --- Import Custom Utilities & Fallbacks --- >>>
# ------------------------------------------------------------------------------
# ... (rest of your imports: dicom_utils, llm_interactions, etc.) ...
try:
    from dicom_utils import parse_dicom, extract_dicom_metadata, dicom_to_image, get_default_wl
    from llm_interactions import run_initial_analysis, run_multimodal_qa, run_disease_analysis, estimate_ai_confidence
    from report_utils import generate_pdf_report_bytes
    from ui_components import display_dicom_metadata, dicom_wl_sliders
    logger.info("Successfully imported custom utility modules.")
    try: from hf_models import query_hf_vqa_inference_api, HF_VQA_MODEL_ID
    except ImportError:
        HF_VQA_MODEL_ID = "hf_model_not_found"; query_hf_vqa_inference_api = None # Define as None
        logger.warning("hf_models.py not found or error importing. HF VQA fallback disabled.")
except ImportError as import_error:
    st.error(f"CRITICAL ERROR importing helpers ({import_error}). Ensure all .py files present."); logger.critical(f"Failed import: {import_error}", exc_info=True); st.stop()

# --- Helper Image Conversion (safe_image_to_data_url) ---
# ... (function definition) ...
def safe_image_to_data_url(img: Image.Image) -> str:
    if not isinstance(img, Image.Image): logger.warning(f"safe_image_to_data_url: Not PIL Image (type: {type(img)})."); return ""
    buffered = io.BytesIO(); format = "PNG"
    try:
        img_to_save = img
        if img.mode not in ['RGB', 'L', 'RGBA']: img_to_save = img.convert('RGB')
        elif img.mode == 'P': img_to_save = img.convert('RGBA')
        img_to_save.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{format.lower()};base64,{img_str}"
    except Exception as e: logger.error(f"Failed converting image to data URL: {e}", exc_info=True); return ""

# ------------------------------------------------------------------------------
# Initialize Session State
# ------------------------------------------------------------------------------
# ... (session state initialization) ...
DEFAULT_STATE = {
    "uploaded_file_info": None, "raw_image_bytes": None, "is_dicom": False,
    "dicom_dataset": None, "dicom_metadata": {}, "processed_image": None,
    "display_image": None, "session_id": None, "history": [],
    "initial_analysis": "", "qa_answer": "", "disease_analysis": "",
    "confidence_score": "", "last_action": None, "pdf_report_bytes": None,
    "canvas_drawing": None, "roi_coords": None, 'current_display_wc': None,
    'current_display_ww': None,
}
for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(default_value) if isinstance(default_value, (list, dict)) else default_value
if not isinstance(st.session_state.history, list): st.session_state.history = []
logger.debug("Session state initialized.")


# ------------------------------------------------------------------------------
# Page Title & Disclaimer
# ------------------------------------------------------------------------------
# ... (title and disclaimer) ...
st.title("⚕️ RadVision QA Advanced: AI-Assisted Image Analysis")
with st.expander("⚠️ Important Disclaimer & Usage Guide", expanded=False):
    st.warning(""" **Disclaimer:** Research/Educational use ONLY. **NOT for Clinical Use.** """)
    st.info(""" **Quick Guide:** 1. Upload... 2. DICOM W/L... 3. Analyze... 4. Review... 5. Report...""")
st.markdown("---")


# =============================================================================
# === SIDEBAR CONTROLS ========================================================
# =============================================================================
# ... (sidebar code remains the same) ...
with st.sidebar:
    # Logo
    logo_path = "assets/radvisionai-hero.jpeg"
    if os.path.exists(logo_path): st.image(logo_path, width=200, caption="RadVision AI"); st.markdown("---")
    else: logger.warning(f"Sidebar logo not found: {logo_path}"); st.markdown("### RadVision AI"); st.markdown("---")

    st.header("Image Upload & Controls")
    ALLOWED_TYPES = ["jpg", "jpeg", "png", "dcm", "dicom"]
    uploaded_file = st.file_uploader(f"Upload Image ({', '.join(type.upper() for type in ALLOWED_TYPES)})", type=ALLOWED_TYPES, key="file_uploader_widget", accept_multiple_files=False)

    # --- File Processing Logic ---
    if uploaded_file is not None:
        try:
             file_mtime = getattr(uploaded_file, 'last_modified', None)
             if file_mtime is None: import hashlib; hasher = hashlib.md5(); hasher.update(uploaded_file.getvalue()); file_unique_id = hasher.hexdigest(); uploaded_file.seek(0); logger.warning("Using MD5 for change detection.")
             else: file_unique_id = str(file_mtime)
             new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{file_unique_id}"
        except Exception as file_info_err: logger.error(f"Err getting file info: {file_info_err}", exc_info=True); new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{str(uuid.uuid4())[:8]}"

        if new_file_info != st.session_state.get("uploaded_file_info"):
            logger.info(f"New file detected: {uploaded_file.name}")
            st.toast(f"Processing '{uploaded_file.name}'...", icon="⏳")
            # Reset State
            logger.debug("Resetting state for new file...")
            preserve_keys = {"file_uploader_widget"}
            for key, default_value in DEFAULT_STATE.items():
                if key not in preserve_keys: st.session_state[key] = copy.deepcopy(default_value) if isinstance(default_value, (list, dict)) else default_value
            st.session_state.uploaded_file_info = new_file_info; st.session_state.session_id = str(uuid.uuid4())[:8]; logger.info(f"New Session ID: {st.session_state.session_id}")

            # Image Processing
            with st.spinner("🔬 Processing image data..."):
                st.session_state.raw_image_bytes = None; temp_display_image = None; temp_processed_image = None; processing_successful = False
                # ... (image processing logic as before) ...
                try:
                    logger.debug("Reading bytes..."); st.session_state.raw_image_bytes = uploaded_file.getvalue();
                    if not st.session_state.raw_image_bytes: raise ValueError("File empty.")
                    logger.info(f"Read {len(st.session_state.raw_image_bytes)} bytes.")
                    file_ext = os.path.splitext(uploaded_file.name)[1].lower(); is_magic = (len(st.session_state.raw_image_bytes) > 132 and st.session_state.raw_image_bytes[128:132] == b'DICM')
                    st.session_state.is_dicom = (pydicom is not None) and (file_ext in (".dcm", ".dicom") or "dicom" in uploaded_file.type.lower() or is_magic)
                    logger.info(f"Identified as DICOM: {st.session_state.is_dicom}")

                    if st.session_state.is_dicom: # DICOM Branch
                        logger.debug("DICOM processing..."); ds = None
                        try: ds = parse_dicom(st.session_state.raw_image_bytes, uploaded_file.name); st.session_state.dicom_dataset = ds
                        except Exception as e: st.error(f"Error parsing DICOM: {e}"); logger.error(f"DICOM parse failed: {e}", exc_info=True); ds = None
                        if ds:
                            logger.info("DICOM parsed."); st.session_state.dicom_metadata = extract_dicom_metadata(ds)
                            wc, ww = get_default_wl(ds); st.session_state.current_display_wc, st.session_state.current_display_ww = wc, ww; logger.info(f"DICOM W/L: WC={wc}, WW={ww}")
                            temp_display_image = dicom_to_image(ds, wc, ww); temp_processed_image = dicom_to_image(ds, None, None)
                            if isinstance(temp_display_image, Image.Image) and isinstance(temp_processed_image, Image.Image): processing_successful = True; logger.info("DICOM images generated.")
                            else: st.error("Failed to generate images from DICOM."); logger.error("dicom_to_image invalid.")
                        elif pydicom is None: st.error("Cannot process DICOM: pydicom library missing.")
                    else: # Standard Image Branch
                        logger.debug("Standard image processing...");
                        try:
                            img = Image.open(io.BytesIO(st.session_state.raw_image_bytes)); logger.info(f"Image.open ok. Mode: {img.mode}, Size: {img.size}")
                            temp_display_image = img.copy(); temp_processed_image = img.copy()
                            if temp_display_image.mode not in ['RGB', 'RGBA', 'L']: temp_display_image = temp_display_image.convert("RGB")
                            elif temp_display_image.mode == 'L': temp_display_image = temp_display_image.convert("RGB")
                            if temp_processed_image.mode in ['P', 'RGBA']: temp_processed_image = temp_processed_image.convert("RGB")
                            elif temp_processed_image.mode == 'L': logger.info("Processed image remains Grayscale (L mode).")
                            st.session_state.dicom_dataset = None; st.session_state.dicom_metadata = {}; st.session_state.current_display_wc = None; st.session_state.current_display_ww = None
                            processing_successful = True; logger.info("Standard images prepared.")
                        except Exception as e: st.error(f"Error processing image: {e}"); logger.error(f"Std image error: {e}", exc_info=True)
                except Exception as e: st.error(f"Critical processing error: {e}"); logger.critical(f"Outer processing error: {e}", exc_info=True); processing_successful = False

                # Final Check & Update
                logger.debug(f"Final Check: Success={processing_successful}, Display PIL={isinstance(temp_display_image, Image.Image)}, Processed PIL={isinstance(temp_processed_image, Image.Image)}")
                if processing_successful and isinstance(temp_display_image, Image.Image) and isinstance(temp_processed_image, Image.Image):
                    if temp_display_image.mode not in ['RGB', 'RGBA']:
                         logger.warning(f"Final Check: Display mode {temp_display_image.mode}, converting to RGB.")
                         try: st.session_state.display_image = temp_display_image.convert('RGB'); logger.info(f"Converted display to RGB.")
                         except Exception as convert_err: logger.error(f"Final RGB conversion failed: {convert_err}", exc_info=True); st.error("Failed final display conversion."); processing_successful = False
                    else: st.session_state.display_image = temp_display_image; logger.info(f"Display image assigned. Mode: {st.session_state.display_image.mode}")
                    if processing_successful:
                        st.session_state.processed_image = temp_processed_image; logger.info(f"Processed image assigned. Mode: {getattr(st.session_state.processed_image, 'mode', 'N/A')}")
                        logger.info(f"**SUCCESS**: State updated."); st.session_state.roi_coords = None; st.session_state.canvas_drawing = None; st.session_state.initial_analysis = ""; st.session_state.qa_answer = ""; st.session_state.disease_analysis = ""; st.session_state.confidence_score = ""; st.session_state.pdf_report_bytes = None; st.session_state.history = []
                        st.success(f"✅ Image '{uploaded_file.name}' processed!"); st.rerun()
                    else: st.error("Image processing failed final conversion."); st.session_state.display_image = None; st.session_state.processed_image = None
                else: # Processing failed earlier
                    logger.critical("Image loading pipeline failed.");
                    if processing_successful: st.error("❌ Processed, but final image objects invalid."); logger.error(f"Final check failed: display type {type(temp_display_image)}, processed type {type(temp_processed_image)}")
                    # Reset all state
                    st.session_state.uploaded_file_info = None; st.session_state.raw_image_bytes = None; st.session_state.display_image = None; st.session_state.processed_image = None; st.session_state.dicom_dataset = None; st.session_state.dicom_metadata = {}; st.session_state.current_display_wc = None; st.session_state.current_display_ww = None; st.session_state.is_dicom = False

    # DICOM W/L Controls
    st.markdown("---")
    # ... (DICOM W/L controls as before) ...
    if st.session_state.is_dicom and pydicom is not None and st.session_state.dicom_dataset and isinstance(st.session_state.get("display_image"), Image.Image):
        with st.expander("DICOM Window/Level", expanded=False):
            # ... W/L logic ...
             try:
                wc_slider, ww_slider = dicom_wl_sliders(st.session_state.dicom_dataset, st.session_state.dicom_metadata, initial_wc=st.session_state.current_display_wc, initial_ww=st.session_state.current_display_ww)
                wc_disp, ww_disp = st.session_state.current_display_wc, st.session_state.current_display_ww
                valid_sliders = (wc_slider is not None and ww_slider is not None); valid_disp = (wc_disp is not None and ww_disp is not None)
                update = valid_sliders and (not valid_disp or (abs(wc_slider - wc_disp) > 1e-3 or abs(ww_slider - ww_disp) > 1e-3))
                if update:
                    logger.info(f"W/L: Applying WC={wc_slider:.1f}, WW={ww_slider:.1f}")
                    with st.spinner("Applying W/L..."):
                         new_img = dicom_to_image(st.session_state.dicom_dataset, wc_slider, ww_slider)
                         if isinstance(new_img, Image.Image):
                              if new_img.mode not in ['RGB', 'RGBA', 'L']: st.session_state.display_image = new_img.convert('RGB')
                              elif new_img.mode == 'L': st.session_state.display_image = new_img.convert('RGB')
                              else: st.session_state.display_image = new_img
                              st.session_state.current_display_wc, st.session_state.current_display_ww = wc_slider, ww_slider; logger.debug("W/L applied, rerunning."); st.rerun()
                         else: st.error("Failed W/L apply."); logger.error("dicom_to_image failed W/L update.")
             except Exception as e: st.error(f"W/L error: {e}"); logger.error(f"W/L slider error: {e}", exc_info=True)
        st.markdown("---")
    elif st.session_state.is_dicom and pydicom is None: st.warning("DICOM detected, but pydicom missing.")


    # AI Actions
    # ... (AI Actions buttons as before) ...
    if isinstance(st.session_state.get("display_image"), Image.Image):
        st.subheader("AI Actions")
        if st.button("▶️ Run Initial Analysis", key="analyze_btn", use_container_width=True): st.session_state.last_action = "analyze"; logger.info("Setting action: analyze"); st.rerun()
        st.markdown("---"); st.subheader("❓ Ask AI Question")
        if st.session_state.roi_coords:
            rc = st.session_state.roi_coords; st.info(f"✅ ROI: [L:{rc['left']}, T:{rc['top']}, W:{rc['width']}, H:{rc['height']}]")
            if st.button("❌ Clear ROI", key="clear_roi_btn", use_container_width=True): st.session_state.roi_coords = None; st.session_state.canvas_drawing = None; logger.info("ROI cleared."); st.rerun()
        else: st.caption("ℹ️ Optionally, draw ROI on image.")
        question_input = st.text_area("Ask about image/ROI:", height=100, key="question_input_widget", placeholder="e.g., Any abnormalities?", label_visibility="collapsed")
        if st.button("💬 Ask AI", key="ask_btn", use_container_width=True):
            q = st.session_state.question_input_widget
            if q and q.strip(): st.session_state.last_action = "ask"; logger.info(f"Setting action: ask ('{q[:50]}...')"); st.rerun()
            else: st.warning("Please enter a question."); logger.warning("Ask AI button: empty question.")
        st.markdown("---"); st.subheader("🎯 Focused Condition Analysis")
        DISEASE_OPTIONS = ["Pneumonia", "Lung Cancer", "Stroke", "Fracture", "Appendicitis", "Tuberculosis", "COVID-19", "Pulmonary Embolism", "Brain Tumor", "Arthritis", "Osteoporosis", "Cardiomegaly", "Aortic Aneurysm", "Bowel Obstruction", "Mass/Nodule", "Effusion", "Normal Variation"]
        disease_options = [""] + sorted(DISEASE_OPTIONS)
        disease_select = st.selectbox("Condition:", options=disease_options, key="disease_select_widget", help="Select condition.")
        if st.button("🩺 Run Condition Analysis", key="disease_btn", use_container_width=True):
            d = st.session_state.disease_select_widget
            if d: st.session_state.last_action = "disease"; logger.info(f"Setting action: disease ('{d}')"); st.rerun()
            else: st.warning("Select condition."); logger.warning("Condition Analysis button: no condition selected.")
        st.markdown("---")
        with st.expander("📊 Confidence & Report", expanded=True):
            can_estimate = bool(st.session_state.history or st.session_state.initial_analysis or st.session_state.disease_analysis)
            if st.button("📈 Estimate Confidence", key="confidence_btn", disabled=not can_estimate, use_container_width=True): st.session_state.last_action = "confidence"; logger.info("Setting action: confidence"); st.rerun()
            if not can_estimate: st.caption("Run analysis/QA first.")
            if st.button("📄 Generate PDF Data", key="generate_report_data_btn", use_container_width=True): st.session_state.last_action = "generate_report_data"; logger.info("Setting action: generate_report_data"); st.rerun()
            if st.session_state.pdf_report_bytes:
                fname = f"RadVisionAI_Report_{st.session_state.session_id or 'session'}.pdf"
                st.download_button(label="⬇️ Download PDF Report", data=st.session_state.pdf_report_bytes, file_name=fname, mime="application/pdf", key="download_pdf_button", use_container_width=True)
    else: st.info("👈 Upload image to begin.")


# =============================================================================
# === MAIN PANEL DISPLAYS =====================================================
# =============================================================================
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("🖼️ Image Viewer")
    display_img_object = st.session_state.get("display_image")
    logger.debug(f"Main Panel Check: Retrieved display_image. Type: {type(display_img_object)}, Is PIL: {isinstance(display_img_object, Image.Image)}, Mode: {getattr(display_img_object, 'mode', 'N/A')}")

    if isinstance(display_img_object, Image.Image):
        logger.debug(f"Image Viewer: Preparing canvas for image. Mode: {display_img_object.mode}, Size: {display_img_object.size}")
        bg_image_pil = None
        try:
            # Prepare image for canvas (ensure RGB/RGBA)
            if display_img_object.mode in ['RGB', 'RGBA']: bg_image_pil = display_img_object
            elif display_img_object.mode == 'L': bg_image_pil = display_img_object.convert('RGB')
            elif display_img_object.mode == 'P': bg_image_pil = display_img_object.convert('RGBA')
            else: bg_image_pil = display_img_object.convert('RGB')

            if not isinstance(bg_image_pil, Image.Image): raise TypeError("Conversion failed")
            logger.info(f"Canvas Prep: Final background image ready. Mode: {bg_image_pil.mode}, Size: {bg_image_pil.size}")

        except Exception as prep_err:
            st.error(f"Failed preparing image for viewer: {prep_err}"); logger.error(f"Canvas prep error: {prep_err}", exc_info=True); bg_image_pil = None

        if isinstance(bg_image_pil, Image.Image):
            # Calculate canvas dimensions
            MAX_CANVAS_WIDTH, MAX_CANVAS_HEIGHT = 700, 600
            img_w, img_h = bg_image_pil.size; aspect_ratio = img_w / img_h if img_h > 0 else 1
            canvas_w = min(img_w, MAX_CANVAS_WIDTH); canvas_h = int(canvas_w / aspect_ratio) if aspect_ratio > 0 else MAX_CANVAS_HEIGHT
            if canvas_h > MAX_CANVAS_HEIGHT: canvas_h = MAX_CANVAS_HEIGHT; canvas_w = int(canvas_h * aspect_ratio)
            canvas_w, canvas_h = max(int(canvas_w), 150), max(int(canvas_h), 150)
            logger.info(f"Canvas Prep: Calculated dims W={canvas_w}, H={canvas_h}")

            if canvas_w > 0 and canvas_h > 0:
                st.caption("Click and drag on the image below to select a Region of Interest (ROI).")

                # Remove or comment out the debug st.image call now that we know it works
                # try:
                #     st.image(bg_image_pil, caption=f"Debug Check: Image passed to canvas (Mode: {bg_image_pil.mode}, Size: {bg_image_pil.size})", use_column_width=True)
                #     logger.info("Debug Check: st.image displayed the image.")
                # except Exception as st_image_err:
                #      st.error(f"Debug Check Failed: st.image error: {st_image_err}")
                #      logger.error(f"Debug Check: st.image failed: {st_image_err}", exc_info=True)

                try:
                    initial_drawing_state = st.session_state.get("canvas_drawing")
                    if initial_drawing_state and not isinstance(initial_drawing_state, dict): initial_drawing_state = None; st.session_state.canvas_drawing = None
                    logger.info(f"Rendering st_canvas with background. Mode: {bg_image_pil.mode}")
                    if not isinstance(bg_image_pil, Image.Image): raise ValueError("bg_image_pil invalid before canvas call.")

                    # --- Render the Drawable Canvas ---
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.2)", stroke_width=2, stroke_color="rgba(220, 50, 50, 0.9)",
                        background_image=bg_image_pil, # THIS is where image_to_url is likely called internally
                        update_streamlit=True, height=canvas_h, width=canvas_w,
                        drawing_mode="rect", initial_drawing=initial_drawing_state,
                        key="drawable_canvas",
                    )
                    logger.info("st_canvas call completed.")

                    # --- Process Canvas Result (ROI Logic) ---
                    # ... (ROI processing logic remains the same) ...
                    if canvas_result is not None and canvas_result.json_data is not None:
                        st.session_state.canvas_drawing = canvas_result.json_data
                        current_roi_state = st.session_state.get("roi_coords")
                        if canvas_result.json_data.get("objects"):
                            last_object = canvas_result.json_data["objects"][-1]
                            if last_object["type"] == "rect":
                                canvas_left, canvas_top = int(last_object["left"]), int(last_object["top"])
                                canvas_width = int(last_object["width"] * last_object.get("scaleX", 1)); canvas_height = int(last_object["height"] * last_object.get("scaleY", 1))
                                scale_x = img_w / canvas_w; scale_y = img_h / canvas_h
                                original_left = max(0, int(canvas_left * scale_x)); original_top = max(0, int(canvas_top * scale_y))
                                original_width = int(canvas_width * scale_x); original_height = int(canvas_height * scale_y)
                                original_right = min(img_w, original_left + original_width); original_bottom = min(img_h, original_top + original_height)
                                final_width = max(0, original_right - original_left); final_height = max(0, original_bottom - original_top)
                                MIN_ROI_DIM = 10
                                if final_width >= MIN_ROI_DIM and final_height >= MIN_ROI_DIM:
                                    new_roi_dict = {"left": original_left, "top": original_top, "width": final_width, "height": final_height}
                                    if current_roi_state != new_roi_dict: st.session_state.roi_coords = new_roi_dict; logger.info(f"ROI Updated: {new_roi_dict}"); st.rerun()
                                elif current_roi_state is not None: logger.info("New drawing too small, clearing ROI."); st.session_state.roi_coords = None; st.rerun()
                        elif not canvas_result.json_data.get("objects") and current_roi_state is not None: logger.info("Canvas cleared, removing ROI state."); st.session_state.roi_coords = None; st.rerun()

                except AttributeError as ae:
                     # Catch the specific error we saw
                     if 'image_to_url' in str(ae):
                         st.error("Error during canvas rendering: Streamlit's internal 'image_to_url' function not found or accessible. This might be due to your Streamlit version.")
                         st.warning("The monkey-patch attempt might have failed. Check logs for 'Applied monkey-patch'.")
                         logger.error(f"Canvas failed due to missing 'image_to_url': {ae}", exc_info=True)
                     else: # Other AttributeErrors
                         st.error(f"Attribute error during canvas rendering: {ae}")
                         logger.error(f"Canvas attribute error: {ae}", exc_info=True)
                except Exception as canvas_error:
                    st.error(f"Error during canvas rendering or interaction: {canvas_error}")
                    logger.error(f"st_canvas failed: {canvas_error}", exc_info=True)
                    st.warning("Drawing functionality might be unavailable. Check browser console (F12).")
            else: st.error("Invalid canvas dimensions."); logger.error(f"Invalid canvas dims: W={canvas_w}, H={canvas_h}")
        else: st.info("Image could not be prepared for viewer."); logger.error("Cannot display canvas: bg_image_pil invalid.")

        # Display DICOM Metadata
        # ... (metadata display logic) ...
        if st.session_state.is_dicom and st.session_state.dicom_metadata:
            st.markdown("---"); st.subheader("📄 DICOM Metadata")
            if pydicom is None: st.warning("'pydicom' missing.")
            else: logger.debug("Displaying DICOM metadata."); display_dicom_metadata(st.session_state.dicom_metadata)

    # Fallback Placeholder
    # ... (placeholder logic) ...
    else:
        logger.debug("Image Viewer: No valid display_image."); st.markdown("---")
        if st.session_state.uploaded_file_info: st.warning("Image processing failed.")
        else: st.info("Upload an image file.")
        st.markdown("""<div style='height: 400px; border: 2px dashed #ccc; display: flex; align-items: center; justify-content: center; text-align: center; color: #aaa; font-style: italic; padding: 20px; border-radius: 8px;'>Image Display Area</div>""", unsafe_allow_html=True)


# --- Column 2: Analysis Results Tabs ---
# ... (results tabs remain the same) ...
with col2:
    st.subheader("📊 Analysis & Results")
    tab_titles = ["🔬 Initial Analysis", "💬 Q&A History", "🩺 Disease Focus", "📈 Confidence"]
    tabs = st.tabs(tab_titles)
    with tabs[0]: st.text_area("Overall Findings", value=st.session_state.initial_analysis or "No initial analysis has been performed yet.", height=450, key="output_initial", disabled=True)
    with tabs[1]:
        st.text_area("Latest AI Answer", value=st.session_state.qa_answer or "Ask a question.", height=200, key="output_qa", disabled=True)
        st.markdown("---")
        if st.session_state.history:
             with st.expander("Full Conversation History", expanded=True):
                 for i, (q, a) in enumerate(reversed(st.session_state.history)):
                     st.markdown(f"**You ({len(st.session_state.history)-i}):**"); st.caption(f"{q}")
                     st.markdown(f"**AI ({len(st.session_state.history)-i}):**"); st.markdown(a, unsafe_allow_html=True)
                     if i < len(st.session_state.history) - 1: st.markdown("---")
        else: st.caption("No history.")
    with tabs[2]: st.text_area("Focused Condition Findings", value=st.session_state.disease_analysis or "No focused condition analysis performed.", height=450, key="output_disease", disabled=True)
    with tabs[3]: st.text_area("AI Confidence Estimation", value=st.session_state.confidence_score or "No confidence estimation performed.", height=450, key="output_confidence", disabled=True)


# =============================================================================
# === ACTION HANDLING LOGIC ===================================================
# =============================================================================
# ... (action handling logic remains the same) ...
current_action: Optional[str] = st.session_state.get("last_action")
if current_action:
    logger.info(f"ACTION HANDLER: Action '{current_action}'")
    processed_image = st.session_state.get("processed_image"); session_id = st.session_state.get("session_id")
    roi_coords = st.session_state.get("roi_coords"); history = st.session_state.history

    if current_action != "generate_report_data" and not isinstance(processed_image, Image.Image): st.error(f"Cannot perform '{current_action}': Processed image invalid."); logger.error(f"Action '{current_action}' aborted: Invalid processed_image."); st.session_state.last_action = None; st.stop()
    if not session_id: st.error(f"Cannot perform '{current_action}': Session ID missing."); logger.error(f"Action '{current_action}' aborted: Missing Session ID."); st.session_state.last_action = None; st.stop()
    if not isinstance(history, list): history = []; st.session_state.history = history; logger.warning("History reset.")
    img_llm = processed_image; roi_str = " (with ROI)" if roi_coords else ""

    try: # Execute Actions
        if current_action == "analyze":
            st.info(f"🔬 Analyzing{roi_str}...")
            with st.spinner("AI analyzing..."): res = run_initial_analysis(img_llm) # Pass roi=roi_coords if supported
            st.session_state.initial_analysis = res; st.success("Analysis finished.")
        elif current_action == "ask":
            q = st.session_state.question_input_widget.strip();
            if not q: st.warning("Empty question."); logger.warning("Ask: empty q.")
            else:
                st.info(f"❓ Asking AI{roi_str}..."); st.session_state.qa_answer = ""
                with st.spinner("AI thinking..."): ans, ok = run_multimodal_qa(img_llm, q, history, roi_coords)
                if ok: st.session_state.qa_answer = ans; st.session_state.history.append((q, ans)); st.success("AI answered.")
                else: # Fallback
                    st.error(f"Primary AI failed: {ans}"); logger.warning(f"Primary AI failed: {ans}"); st.session_state.qa_answer = f"**[Primary AI Error]** {ans}\n\n---\n"
                    hf_ok = (HF_VQA_MODEL_ID and HF_VQA_MODEL_ID != "hf_model_not_found" and query_hf_vqa_inference_api is not None and os.environ.get("HF_API_TOKEN"))
                    if hf_ok:
                         fb_model_name = HF_VQA_MODEL_ID.split('/')[-1]
                         st.info(f"Trying fallback ({fb_model_name})...")
                         with st.spinner(f"Asking fallback AI ({fb_model_name})..."): fb_ans, fb_ok = query_hf_vqa_inference_api(img_llm, q, roi_coords)
                         if fb_ok: fb_disp = f"**[Fallback: {fb_model_name}]**\n\n{fb_ans}"; st.session_state.qa_answer += fb_disp; st.session_state.history.append((f"[Fallback] {q}", fb_disp)); st.success(f"Fallback ({fb_model_name}) answered.")
                         else: fb_err = f"Fallback ({fb_model_name}) failed: {fb_ans}"; st.session_state.qa_answer += f"**[Fallback Failed]** {fb_err}"; st.error(fb_err); logger.error(f"HF fallback fail: {fb_ans}")
                    else: fb_msg = f"Fallback unavailable."; st.session_state.qa_answer += f"**[Fallback Unavailable]** {fb_msg}"; st.warning(fb_msg); logger.warning("HF fallback skip.")
        elif current_action == "disease":
            d = st.session_state.disease_select_widget
            if not d: st.warning("Select condition."); logger.warning("Disease: empty selection.")
            else:
                st.info(f"🩺 Analyzing for '{d}'{roi_str}...")
                with st.spinner(f"AI assessing '{d}'..."): res = run_disease_analysis(img_llm, d, roi_coords)
                st.session_state.disease_analysis = res; st.success(f"Analysis for '{d}' finished.")
        elif current_action == "confidence":
            context_exists = bool(history or st.session_state.initial_analysis or st.session_state.disease_analysis)
            if not context_exists: st.warning("Cannot estimate: No context."); logger.warning("Confidence skip: No context.")
            else:
                st.info(f"📊 Estimating confidence{roi_str}...")
                with st.spinner("Calculating..."): res = estimate_ai_confidence(img_llm, history, st.session_state.initial_analysis, st.session_state.disease_analysis, roi_coords)
                st.session_state.confidence_score = res; st.success("Confidence estimated.")
        elif current_action == "generate_report_data":
            st.info("📄 Preparing PDF data..."); st.session_state.pdf_report_bytes = None
            img_rep = st.session_state.get("display_image") # Use display image for report
            if not isinstance(img_rep, Image.Image): st.error("Cannot gen report: Invalid display image."); logger.error("PDF skip: invalid display_image.")
            else:
                 img_final = img_rep.copy() # Work on a copy
                 if roi_coords: # Draw ROI if exists
                      try:
                          if img_final.mode not in ['RGB', 'RGBA']: img_final = img_final.convert("RGB")
                          draw = ImageDraw.Draw(img_final); x0,y0,w,h = roi_coords['left'],roi_coords['top'],roi_coords['width'],roi_coords['height']; draw.rectangle([x0,y0,x0+w,y0+h], outline="red", width=3); logger.info("Drew ROI on report image.")
                      except Exception as draw_e: logger.error(f"Failed drawing ROI for report: {draw_e}")
                 # Gather outputs
                 qa_hist_str = "\n\n".join([f"Q: {q}\nA: {a}" for q,a in history]) if history else "N/A"
                 outputs = { "Session ID": session_id, "Preliminary Analysis": st.session_state.initial_analysis or "N/P", "Q&A History": qa_hist_str, "Condition Analysis": st.session_state.disease_analysis or "N/P", "Confidence": st.session_state.confidence_score or "N/E" }
                 if st.session_state.is_dicom and st.session_state.dicom_metadata:
                      filtered_meta = { k: v for k, v in st.session_state.dicom_metadata.items() if k in ["Patient Name", "Patient ID", "Study Date", "Modality", "Study Description"]} # Example filter
                      outputs["DICOM Metadata (Selected)"] = "\n".join([f"{k}: {v}" for k,v in filtered_meta.items()]) if filtered_meta else "N/A"
                 # Generate PDF
                 with st.spinner("🎨 Generating PDF..."): pdf_bytes = generate_pdf_report_bytes(session_id, img_final, outputs)
                 if pdf_bytes: st.session_state.pdf_report_bytes = pdf_bytes; st.success("✅ PDF data generated!"); logger.info("PDF gen ok.")
                 else: st.error("❌ PDF generation failed."); logger.error("PDF gen fail.")
        else: st.warning(f"Unknown action: '{current_action}'."); logger.warning(f"Unknown action: '{current_action}'")
    except Exception as e: st.error(f"Action '{current_action}' error: {e}"); logger.critical(f"Action error: {e}", exc_info=True)
    finally: st.session_state.last_action = None; logger.debug(f"Action '{current_action}' finished."); st.rerun()


# --- Footer ---
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced | Session: {st.session_state.get('session_id', 'N/A')} | v(dev)")
logger.info("--- App Render Cycle Complete ---")