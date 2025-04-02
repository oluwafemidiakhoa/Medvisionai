# main_app.py (Revision - Force RGB, Simplify Canvas, Check Console)

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
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ... (logging setup continues) ...
if pydicom is None: logger.info("Pydicom module not found. DICOM functionality disabled.")
else: logger.info(f"Pydicom Version: {PYDICOM_VERSION}") # Check dependencies...
logger.info(f"--- App Start ---")
logger.info(f"Logging level set to {LOG_LEVEL}")
logger.info(f"Streamlit Version: {st.__version__}")
logger.info(f"Pillow (PIL) Version: {PIL_VERSION}")
logger.info(f"Streamlit Drawable Canvas Version: {CANVAS_VERSION}")


# ------------------------------------------------------------------------------
# <<< --- Monkey-Patch to Add image_to_url if Missing --- >>>
# ------------------------------------------------------------------------------
import streamlit.elements.image as st_image
logger.debug("Checking for 'image_to_url' attribute in streamlit.elements.image")
if not hasattr(st_image, "image_to_url"):
    logger.warning("Attribute 'image_to_url' NOT FOUND in streamlit.elements.image. Applying monkey-patch.")
    # Define the patch function (ensure it handles PIL Images correctly)
    def image_to_url_monkey_patch(
        image: Any, width: int, clamp: bool, channels: str,
        output_format: str, image_id: str, allow_emoji: bool = False,
    ) -> str:
        patch_func_name = "image_to_url_monkey_patch" # For logging
        try:
            if isinstance(image, Image.Image):
                buffered = io.BytesIO()
                img_format = output_format.upper() if output_format else "PNG"
                if img_format not in ["PNG", "JPEG"]: img_format = "PNG" # Default to PNG

                img_to_save = image
                # Ensure compatibility for saving
                if image.mode not in ['RGB', 'RGBA', 'L']:
                     logger.debug(f"[{patch_func_name}] Converting mode {image.mode} to RGB for saving.")
                     img_to_save = image.convert("RGB")
                elif image.mode == 'P' and img_format == 'PNG':
                     logger.debug(f"[{patch_func_name}] Converting mode P to RGBA for PNG saving.")
                     img_to_save = image.convert("RGBA") # RGBA often better for palette conversion
                elif image.mode != 'RGB' and img_format == 'JPEG':
                     logger.debug(f"[{patch_func_name}] Converting mode {image.mode} to RGB for JPEG saving.")
                     img_to_save = image.convert("RGB") # JPEG needs RGB

                img_to_save.save(buffered, format=img_format)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                data_url = f"data:image/{img_format.lower()};base64,{img_str}"
                logger.debug(f"[{patch_func_name}] Successfully created data URL (format: {img_format}).")
                return data_url
            else:
                logger.warning(f"[{patch_func_name}] Received non-PIL image type: {type(image)}. Cannot create data URL.")
                return "" # Cannot handle non-PIL images with this basic patch
        except Exception as e:
            logger.error(f"[{patch_func_name}] Failed: {e}", exc_info=True)
            return ""
    # Apply the patch
    try:
        st_image.image_to_url = image_to_url_monkey_patch
        logger.info("Monkey-patch for 'image_to_url' applied successfully.")
    except Exception as patch_apply_err:
        logger.error(f"Failed to apply monkey-patch: {patch_apply_err}", exc_info=True)
else:
    logger.info("Attribute 'image_to_url' FOUND in streamlit.elements.image. No patch needed.")


# ------------------------------------------------------------------------------
# <<< --- Import Custom Utilities & Fallbacks --- >>>
# ------------------------------------------------------------------------------
# ... (imports for dicom_utils, llm_interactions, etc.) ...
try:
    from dicom_utils import parse_dicom, extract_dicom_metadata, dicom_to_image, get_default_wl
    from llm_interactions import run_initial_analysis, run_multimodal_qa, run_disease_analysis, estimate_ai_confidence
    from report_utils import generate_pdf_report_bytes
    from ui_components import display_dicom_metadata, dicom_wl_sliders
    logger.info("Successfully imported custom utility modules.")
    try: from hf_models import query_hf_vqa_inference_api, HF_VQA_MODEL_ID
    except ImportError: HF_VQA_MODEL_ID = "hf_model_not_found"; query_hf_vqa_inference_api = None; logger.warning("hf_models.py not found. HF VQA fallback disabled.")
except ImportError as import_error: st.error(f"CRITICAL ERROR importing helpers ({import_error})."); logger.critical(f"Failed import: {import_error}", exc_info=True); st.stop()

# ... (safe_image_to_data_url function) ...
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
    if key not in st.session_state: st.session_state[key] = copy.deepcopy(default_value) if isinstance(default_value, (list, dict)) else default_value
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
# ... (Sidebar code remains largely the same, ensure image processing creates display_image) ...
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
                            temp_display_image = dicom_to_image(ds, wc, ww); temp_processed_image = dicom_to_image(ds, None, None) # Raw pixels
                            if isinstance(temp_display_image, Image.Image) and isinstance(temp_processed_image, Image.Image):
                                # Ensure display is RGB for consistency
                                if temp_display_image.mode != 'RGB':
                                     logger.info(f"Converting DICOM display image from {temp_display_image.mode} to RGB.")
                                     temp_display_image = temp_display_image.convert('RGB')
                                processing_successful = True; logger.info("DICOM images generated and display converted to RGB.")
                            else: st.error("Failed to generate images from DICOM."); logger.error("dicom_to_image invalid.")
                        elif pydicom is None: st.error("Cannot process DICOM: pydicom library missing.")
                    else: # Standard Image Branch
                        logger.debug("Standard image processing...");
                        try:
                            img = Image.open(io.BytesIO(st.session_state.raw_image_bytes)); logger.info(f"Image.open ok. Mode: {img.mode}, Size: {img.size}")
                            temp_display_image = img.copy(); temp_processed_image = img.copy()
                            # Ensure display image is RGB
                            if temp_display_image.mode != 'RGB':
                                logger.info(f"Converting display image from {temp_display_image.mode} to RGB.")
                                temp_display_image = temp_display_image.convert("RGB")
                            # Processed image can remain L, convert others to RGB
                            if temp_processed_image.mode not in ['L', 'RGB']:
                                logger.info(f"Converting processed image from {temp_processed_image.mode} to RGB.")
                                temp_processed_image = temp_processed_image.convert("RGB")

                            st.session_state.dicom_dataset = None; st.session_state.dicom_metadata = {}; st.session_state.current_display_wc = None; st.session_state.current_display_ww = None
                            processing_successful = True; logger.info("Standard images prepared, display forced RGB.")
                        except Exception as e: st.error(f"Error processing image: {e}"); logger.error(f"Std image error: {e}", exc_info=True)
                except Exception as e: st.error(f"Critical processing error: {e}"); logger.critical(f"Outer processing error: {e}", exc_info=True); processing_successful = False

                # Final Check & Update
                logger.debug(f"Final Check: Success={processing_successful}, Display PIL={isinstance(temp_display_image, Image.Image)}, Processed PIL={isinstance(temp_processed_image, Image.Image)}")
                if processing_successful and isinstance(temp_display_image, Image.Image) and temp_display_image.mode == 'RGB' and isinstance(temp_processed_image, Image.Image):
                    st.session_state.display_image = temp_display_image
                    st.session_state.processed_image = temp_processed_image
                    logger.info(f"**SUCCESS**: State updated. Display Mode: {st.session_state.display_image.mode}, Processed Mode: {st.session_state.processed_image.mode}")
                    # Reset analysis state
                    st.session_state.roi_coords = None; st.session_state.canvas_drawing = None; st.session_state.initial_analysis = ""; st.session_state.qa_answer = ""; st.session_state.disease_analysis = ""; st.session_state.confidence_score = ""; st.session_state.pdf_report_bytes = None; st.session_state.history = []
                    st.success(f"✅ Image '{uploaded_file.name}' processed!"); st.rerun()
                else: # Processing failed or final checks failed
                    logger.critical("Image loading pipeline failed or final checks inconsistent.")
                    if processing_successful: # Check why it failed if logic thought it worked
                        if not isinstance(temp_display_image, Image.Image): logger.error("Final Check Fail: temp_display_image is not PIL Image.")
                        elif temp_display_image.mode != 'RGB': logger.error(f"Final Check Fail: temp_display_image mode is {temp_display_image.mode}, not RGB.")
                        if not isinstance(temp_processed_image, Image.Image): logger.error("Final Check Fail: temp_processed_image is not PIL Image.")
                        st.error("❌ Processed, but final image data is invalid or in wrong format.")
                    # Reset all state
                    st.session_state.uploaded_file_info = None; st.session_state.raw_image_bytes = None; st.session_state.display_image = None; st.session_state.processed_image = None; st.session_state.dicom_dataset = None; st.session_state.dicom_metadata = {}; st.session_state.current_display_wc = None; st.session_state.current_display_ww = None; st.session_state.is_dicom = False


    # DICOM W/L Controls
    st.markdown("---")
    if st.session_state.is_dicom and pydicom is not None and st.session_state.dicom_dataset and isinstance(st.session_state.get("display_image"), Image.Image):
        with st.expander("DICOM Window/Level", expanded=False):
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
                              # Ensure RGB after W/L adjustment
                              if new_img.mode != 'RGB':
                                   logger.info(f"Converting W/L adjusted image from {new_img.mode} to RGB.")
                                   st.session_state.display_image = new_img.convert('RGB')
                              else:
                                   st.session_state.display_image = new_img
                              st.session_state.current_display_wc, st.session_state.current_display_ww = wc_slider, ww_slider
                              logger.debug("W/L applied, rerunning."); st.rerun()
                         else: st.error("Failed W/L apply."); logger.error("dicom_to_image failed W/L update.")
             except Exception as e: st.error(f"W/L error: {e}"); logger.error(f"W/L slider error: {e}", exc_info=True)
        st.markdown("---")
    elif st.session_state.is_dicom and pydicom is None: st.warning("DICOM detected, but pydicom missing.")

    # AI Actions
    if isinstance(st.session_state.get("display_image"), Image.Image):
        # ... (AI Actions buttons remain the same) ...
        st.subheader("AI Actions")
        if st.button("▶️ Run Initial Analysis", key="analyze_btn", use_container_width=True): st.session_state.last_action = "analyze"; logger.info("Setting action: analyze"); st.rerun()
        st.markdown("---"); st.subheader("❓ Ask AI Question")
        if st.session_state.roi_coords: rc = st.session_state.roi_coords; st.info(f"✅ ROI: [L:{rc['left']}, T:{rc['top']}, W:{rc['width']}, H:{rc['height']}]"); # Clear button...
        else: st.caption("ℹ️ Optionally, draw ROI on image.")
        # ... rest of AI actions buttons ...
        question_input = st.text_area("Ask about image/ROI:", height=100, key="question_input_widget", placeholder="e.g., Any abnormalities?", label_visibility="collapsed")
        if st.button("💬 Ask AI", key="ask_btn", use_container_width=True): q = st.session_state.question_input_widget; # Ask logic...
        st.markdown("---"); st.subheader("🎯 Focused Condition Analysis"); # Disease logic...
        st.markdown("---"); # Confidence/Report logic...
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
        # Add explicit warning to check browser console if issues persist
        st.warning("ℹ️ If the image viewer below appears blank or drawing fails, please check the Browser Developer Console (press F12) for JavaScript errors.", icon="⚠️")

        bg_image_pil = None
        try:
            # --- Force to RGB for Canvas Background ---
            if display_img_object.mode == 'RGB':
                bg_image_pil = display_img_object
                logger.info("Canvas Prep: Image is already RGB.")
            else:
                logger.warning(f"Canvas Prep: Forcing conversion of display image (Mode: {display_img_object.mode}) to RGB for canvas.")
                bg_image_pil = display_img_object.convert('RGB')

            if not isinstance(bg_image_pil, Image.Image) or bg_image_pil.mode != 'RGB':
                raise TypeError(f"Image conversion to RGB failed or resulted in wrong mode: {getattr(bg_image_pil, 'mode', 'Invalid Type')}")
            logger.info(f"Canvas Prep: Final background image ready. Type: {type(bg_image_pil)}, Mode: {bg_image_pil.mode}, Size: {bg_image_pil.size}")

        except Exception as prep_err:
            st.error(f"Fatal Error: Failed to prepare image for viewer: {prep_err}")
            logger.critical(f"Canvas background image preparation FAILED: {prep_err}", exc_info=True)
            bg_image_pil = None # Prevent trying to use it

        # --- Render Canvas only if background is valid RGB ---
        if isinstance(bg_image_pil, Image.Image) and bg_image_pil.mode == 'RGB':
            # Calculate canvas dimensions
            MAX_CANVAS_WIDTH, MAX_CANVAS_HEIGHT = 700, 600
            img_w, img_h = bg_image_pil.size; aspect_ratio = img_w / img_h if img_h > 0 else 1
            canvas_w = min(img_w, MAX_CANVAS_WIDTH); canvas_h = int(canvas_w / aspect_ratio) if aspect_ratio > 0 else MAX_CANVAS_HEIGHT
            if canvas_h > MAX_CANVAS_HEIGHT: canvas_h = MAX_CANVAS_HEIGHT; canvas_w = int(canvas_h * aspect_ratio)
            canvas_w, canvas_h = max(int(canvas_w), 150), max(int(canvas_h), 150)
            logger.info(f"Canvas Prep: Calculated dims W={canvas_w}, H={canvas_h}")

            if canvas_w > 0 and canvas_h > 0:
                st.caption("Click and drag on the image below to select a Region of Interest (ROI).")
                try:
                    logger.info(f"Attempting to render st_canvas with RGB background image.")
                    # Simplify call slightly for testing - remove initial_drawing temporarily
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.2)",
                        stroke_width=2,
                        stroke_color="rgba(220, 50, 50, 0.9)",
                        background_image=bg_image_pil, # MUST be RGB PIL Image here
                        update_streamlit=True,
                        height=canvas_h,
                        width=canvas_w,
                        drawing_mode="rect",
                        # initial_drawing=st.session_state.get("canvas_drawing"), # Temporarily commented out
                        key="drawable_canvas",
                    )
                    logger.info("st_canvas call completed without Python exception.")

                    # Process Canvas Result (ROI Logic) - Re-enable initial_drawing storage if call succeeds
                    if canvas_result is not None and canvas_result.json_data is not None:
                        # Store the drawing state if needed (re-enable if removing initial_drawing worked)
                        # st.session_state.canvas_drawing = canvas_result.json_data
                        # ... (rest of ROI processing logic as before) ...
                        current_roi_state = st.session_state.get("roi_coords")
                        if canvas_result.json_data.get("objects"): # Check for drawings
                           # ... (ROI calculation) ...
                           pass # Placeholder for brevity
                        elif not canvas_result.json_data.get("objects") and current_roi_state is not None:
                            logger.info("Canvas cleared, removing ROI state."); st.session_state.roi_coords = None; # Rerun might be needed
                            pass # Placeholder for brevity

                except AttributeError as ae:
                     if 'image_to_url' in str(ae):
                         st.error("Error: Streamlit's 'image_to_url' function issue persists. Check patch & logs.")
                         logger.error(f"Canvas failed: 'image_to_url' missing/failed despite patch checks: {ae}", exc_info=True)
                     else: st.error(f"Attribute error during canvas rendering: {ae}"); logger.error(f"Canvas attr error: {ae}", exc_info=True)
                except Exception as canvas_error:
                    st.error(f"Error during canvas rendering: {canvas_error}")
                    logger.error(f"st_canvas failed unexpectedly: {canvas_error}", exc_info=True)
            else: st.error("Invalid canvas dimensions."); logger.error(f"Invalid canvas dims: W={canvas_w}, H={canvas_h}")
        else:
            if isinstance(st.session_state.get("display_image"), Image.Image): # Check if display_image existed but prep failed
                 st.error("Image Viewer Error: Could not prepare the uploaded image for the interactive viewer. Check logs.")
            # Don't show the placeholder if prep failed, just the error.

        # Display DICOM Metadata (if applicable)
        if st.session_state.is_dicom and st.session_state.dicom_metadata:
             # ... (metadata display) ...
             pass # Placeholder for brevity

    # Fallback Placeholder (if no image was ever loaded successfully)
    elif not st.session_state.get("uploaded_file_info"): # Only show if no upload attempted or initial load failed
        logger.debug("Image Viewer: No valid display_image and no upload info.")
        st.markdown("---")
        st.info("Upload an image file using the sidebar.")
        st.markdown("""<div style='height: 400px; border: 2px dashed #ccc; ...'>Image Display Area</div>""", unsafe_allow_html=True)
    # If upload happened but display_image is None, errors from processing should be visible.

# --- Column 2: Analysis Results Tabs ---
# ... (results tabs remain the same) ...
with col2:
    st.subheader("📊 Analysis & Results")
    tab_titles = ["🔬 Initial Analysis", "💬 Q&A History", "🩺 Disease Focus", "📈 Confidence"]
    tabs = st.tabs(tab_titles)
    # ... tab content ...


# =============================================================================
# === ACTION HANDLING LOGIC ===================================================
# =============================================================================
# ... (action handling logic remains the same) ...
current_action: Optional[str] = st.session_state.get("last_action")
if current_action:
    # ... action execution ...
    pass # Placeholder for brevity


# --- Footer ---
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced | Session: {st.session_state.get('session_id', 'N/A')} | v(dev)")
logger.info("--- App Render Cycle Complete ---")