# main_app.py

# --- Core Libraries ---
import io
import os
import uuid
import logging
import base64
from typing import Any, Dict, Optional, Tuple, List

# --- Streamlit ---
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ------------------------------------------------------------------------------
# <<< --- 4) Configure Streamlit Page (MOVED HERE - MUST BE FIRST st COMMAND) --- >>>
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="RadVision AI Advanced",
    layout="wide", # Use wide layout for more space
    page_icon="⚕️", # Standard emoji icon
    initial_sidebar_state="expanded"
)

# --- Image & DICOM Processing ---
from PIL import Image, ImageDraw, UnidentifiedImageError
import pydicom # Keep import for type checking if needed, even if helpers use it

# ------------------------------------------------------------------------------
# <<< --- 3) Setup Logging (MOVED HERE - After set_page_config is safe) --- >>>
# ------------------------------------------------------------------------------
# Configure logging (consider moving to a separate config file or using environment variables)
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper() # Default to INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.info(f"Logging level set to {LOG_LEVEL}")
# Optionally suppress verbose logs from libraries
# logging.getLogger("PIL").setLevel(logging.WARNING)
# logging.getLogger("fpdf").setLevel(logging.WARNING)


# ------------------------------------------------------------------------------
# 1) Monkey-Patch: Add image_to_url (Can stay here)
# ------------------------------------------------------------------------------
import streamlit.elements.image as st_image
if not hasattr(st_image, "image_to_url"):
    # ... (monkey-patch code as before) ...
    def image_to_url_monkey_patch(img_obj: Any, *args, **kwargs) -> str:
        # (Implementation as provided in the original code)
        if isinstance(img_obj, Image.Image):
            buffered = io.BytesIO()
            format = "PNG"
            try:
                img_obj.save(buffered, format=format)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/{format.lower()};base64,{img_str}"
            except Exception as e:
                 logging.error(f"Monkey-patched image_to_url failed: {e}")
                 return ""
        else:
            logging.warning(f"Monkey-patched image_to_url: Unsupported type {type(img_obj)}")
            return ""
    st_image.image_to_url = image_to_url_monkey_patch
    logging.info("Monkey-patched streamlit.elements.image.image_to_url (if missing)")


# ------------------------------------------------------------------------------
# <<< --- 2) Import Custom Utilities & Fallbacks (NOW SAFE TO RUN st.error) --- >>>
# ------------------------------------------------------------------------------
# --- Attempt to import all custom utility modules ---
try:
    # Ensure these files exist in the same directory as this script!
    from dicom_utils import parse_dicom, extract_dicom_metadata, dicom_to_image, get_default_wl
    from llm_interactions import (
        run_initial_analysis, run_multimodal_qa, run_disease_analysis,
        estimate_ai_confidence
    )
    from report_utils import generate_pdf_report_bytes
    from ui_components import display_dicom_metadata, dicom_wl_sliders # Ensure ui_components.py exists!

    try:
        # Keep HF optional
        from hf_models import query_hf_vqa_inference_api, HF_VQA_MODEL_ID
    except ImportError:
        HF_VQA_MODEL_ID = "hf_model_not_found"
        def query_hf_vqa_inference_api(img: Image.Image, question: str, roi: Optional[Dict] = None) -> Tuple[str, bool]:
            logging.warning("Using MOCK Hugging Face VQA fallback (hf_models.py not found).")
            return f"[Fallback Unavailable] Hugging Face module not found.", False
        logging.warning("hf_models.py not found. Hugging Face VQA fallback is disabled.")

except ImportError as import_error:
    # Display a clear error in the UI and log it
    # This st.error() call is now SAFE because set_page_config ran first.
    st.error(f"CRITICAL ERROR: Failed to import required code module ({import_error}). The application cannot start. Please ensure all `.py` helper files (like dicom_utils.py, ui_components.py, etc.) are present in the same directory as the main app script.")
    logger.critical(f"Failed to import required utility modules: {import_error}", exc_info=True)
    # Stop execution completely if critical code is missing
    st.stop()

# --- Helper Functions (If any are defined directly in the main script) ---
# e.g., image_to_data_url function (can stay here)
def image_to_data_url(img: Image.Image) -> str:
    # ... (implementation as before) ...
    buffered = io.BytesIO()
    try:
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Failed to convert image to data URL: {e}")
        return ""

# ------------------------------------------------------------------------------
# 6) Initialize Session State (Remains the same)
# ------------------------------------------------------------------------------
DEFAULT_STATE = {
    # ... (state dictionary as before) ...
    "uploaded_file_info": None,     # Tracks the current file to detect changes
    "raw_image_bytes": None,        # Stores the raw bytes of the uploaded file
    "is_dicom": False,              # Flag indicating if the file is DICOM
    "dicom_dataset": None,          # Stores the parsed pydicom.Dataset
    "dicom_metadata": {},           # Extracted technical DICOM metadata
    "processed_image": None,        # Image object used for AI analysis (potentially normalized)
    "display_image": None,          # Image object shown in UI (potentially with W/L applied)
    "session_id": None,             # Unique ID for the current session/upload
    "history": [],                  # List of (question, answer) tuples for conversation
    "initial_analysis": "",         # Stores result of initial analysis
    "qa_answer": "",                # Stores result of the latest Q&A
    "disease_analysis": "",         # Stores result of disease-specific analysis
    "confidence_score": "",         # Stores AI confidence estimation result
    "last_action": None,            # Tracks the last button clicked to trigger actions
    "pdf_report_bytes": None,       # Stores the generated PDF report bytes
    "canvas_drawing": None,         # Stores the state of the drawable canvas (JSON)
    "roi_coords": None,             # Stores the extracted ROI coordinates {'left', 'top', 'width', 'height'}
    'current_display_wc': None,     # Stores WC value used for current display_image
    'current_display_ww': None,     # Stores WW value used for current display_image
}
# Initialize session state keys if they don't exist
for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value
# Ensure history is always a list
if not isinstance(st.session_state.history, list):
    st.session_state.history = []


# ------------------------------------------------------------------------------
# 7) Page Title & Disclaimer (Remains the same)
# ------------------------------------------------------------------------------
st.title("⚕️ RadVision QA Advanced: AI-Assisted Image Analysis")
# ... (disclaimer expander as before) ...
with st.expander("⚠️ Important Disclaimer & Usage Guide", expanded=False):
    st.warning(
        """
        **Disclaimer:** This tool uses AI for medical image analysis and is intended strictly
        for **research, informational, and educational purposes ONLY.**

        *   **NOT for Clinical Use:** Do NOT use this tool for primary diagnosis, treatment planning,
            or any decisions impacting patient care. It is not a substitute for professional
            medical evaluation by qualified healthcare providers.
        *   **AI Limitations:** AI analysis may be inaccurate or incomplete. Results require
            validation by experts. Image quality, artifacts, and atypical presentations can
            significantly affect performance.
        *   **Data Privacy:** While efforts are made to handle data appropriately within the session,
            avoid uploading identifiable patient information unless explicitly permitted by your
            institution's policies and necessary for your research purpose. DICOM metadata containing
            Protected Health Information (PHI) should be anonymized *before* upload if possible.
            The generated report attempts to filter common PHI tags, but this filtering is **not guaranteed**
            to be exhaustive or compliant with all regulations (e.g., HIPAA, GDPR). Verify output.
        *   **No Liability:** The creators and providers of this tool assume no liability for its use
            or interpretation of its results. Use at your own risk.

        **By using this tool, you acknowledge you have read, understood, and agree to these terms.**
        """
    )
    st.info(
        """
        **Quick Guide:**
        1.  **Upload:** Use the sidebar to upload a JPG, PNG, or DICOM file.
        2.  **DICOM W/L:** If DICOM, adjust Window/Level sliders in the sidebar for optimal viewing.
        3.  **Analyze:** Use sidebar buttons to run initial analysis, ask specific questions (optionally draw an ROI first), or analyze for specific conditions.
        4.  **Review:** Results appear in the tabs on the right.
        5.  **Report:** Generate and download a PDF summary (use with caution regarding PHI).
        """
    )
st.markdown("---") # Visual separator

# =============================================================================
# === SIDEBAR CONTROLS ========================================================
# =============================================================================
# ... (Sidebar code remains the same as in the previous corrected version) ...
with st.sidebar:
    st.header("Image Upload & Controls")

    uploaded_file = st.file_uploader(
        "Upload Image (JPG, PNG, DCM)",
        type=["jpg", "jpeg", "png", "dcm", "dicom"], # Allow various extensions/types
        key="file_uploader_widget", # Use a distinct key for the widget itself
        accept_multiple_files=False,
        help="Select a standard image (JPG, PNG) or a DICOM file (.dcm). Max file size configured by Streamlit deployment."
    )

    if uploaded_file is not None:
        try:
             last_modified = getattr(uploaded_file, 'last_modified', '') # Robust access
             new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{last_modified}"
        except Exception: # Fallback if attributes change
             new_file_info = f"{uploaded_file.name}-{uploaded_file.size}"

        if new_file_info != st.session_state.get("uploaded_file_info"):
            logger.info(f"New file upload detected: {uploaded_file.name} ({uploaded_file.type}, {uploaded_file.size} bytes)")
            st.toast(f"Processing '{uploaded_file.name}'...", icon="⏳")
            preserve_keys = {"file_uploader_widget"} # Add other keys if necessary
            logger.debug("Resetting session state for new file upload...")
            for key, default_value in DEFAULT_STATE.items():
                if key not in preserve_keys: st.session_state[key] = default_value
            st.session_state.history = [] # Explicitly reset history
            logger.debug("Session state reset.")
            st.session_state.uploaded_file_info = new_file_info
            st.session_state.session_id = str(uuid.uuid4())[:8] # New session ID
            logger.info(f"Generated new Session ID: {st.session_state.session_id}")

            with st.spinner("🔬 Processing image... Please wait."):
                st.session_state.raw_image_bytes = uploaded_file.getvalue()
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                st.session_state.is_dicom = file_ext in (".dcm", ".dicom") or "dicom" in uploaded_file.type.lower()
                logger.info(f"File '{uploaded_file.name}' identified as DICOM: {st.session_state.is_dicom}")
                processing_successful = False; temp_display_image = None; temp_processed_image = None

                if st.session_state.is_dicom:
                    try: # DICOM Branch
                        logger.debug("Attempting to parse DICOM..."); ds = parse_dicom(st.session_state.raw_image_bytes, uploaded_file.name); st.session_state.dicom_dataset = ds
                        if ds:
                            logger.info("DICOM parsed successfully."); st.session_state.dicom_metadata = extract_dicom_metadata(ds); logger.debug(f"Extracted {len(st.session_state.dicom_metadata)} metadata tags.")
                            wc_default, ww_default = get_default_wl(ds); st.session_state.current_display_wc = wc_default; st.session_state.current_display_ww = ww_default; logger.info(f"Default W/L from DICOM tags: WC={wc_default}, WW={ww_default}")
                            logger.debug("Generating display image from DICOM using default W/L..."); temp_display_image = dicom_to_image(ds, wc_default, ww_default)
                            logger.debug("Generating processed image from DICOM using basic scaling..."); temp_processed_image = dicom_to_image(ds, window_center=None, window_width=None)
                            if temp_display_image and temp_processed_image: logger.info("DICOM images generated."); processing_successful = True
                            else: st.error("DICOM processing failed: Could not generate images."); logger.error("dicom_to_image returned None.")
                        else: logger.error("parse_dicom returned None.")
                    except Exception as e: st.error(f"Unexpected DICOM processing error: {e}"); logger.critical(f"DICOM pipeline error: {e}", exc_info=True); processing_successful = False
                else: # Standard Image Branch
                    try:
                        logger.debug("Attempting to open standard image..."); img = Image.open(io.BytesIO(st.session_state.raw_image_bytes)); temp_processed_image = img.copy()
                        if img.mode != 'RGB': logger.info(f"Converting {img.mode} to RGB."); temp_display_image = img.convert("RGB")
                        else: temp_display_image = img
                        st.session_state.dicom_dataset = None; st.session_state.dicom_metadata = {}; st.session_state.current_display_wc = None; st.session_state.current_display_ww = None
                        processing_successful = True; logger.info("Standard image opened.")
                    except UnidentifiedImageError: st.error(f"Cannot identify format for '{uploaded_file.name}'."); logger.error(f"UnidentifiedImageError: {uploaded_file.name}"); processing_successful = False
                    except Exception as e: st.error(f"Error processing image '{uploaded_file.name}': {e}"); logger.error(f"Std image processing error: {e}", exc_info=True); processing_successful = False
                if processing_successful and isinstance(temp_display_image, Image.Image) and isinstance(temp_processed_image, Image.Image): # Final Check
                    if temp_display_image.mode != 'RGB': st.session_state.display_image = temp_display_image.convert('RGB')
                    else: st.session_state.display_image = temp_display_image
                    st.session_state.processed_image = temp_processed_image
                    logger.info(f"Processing complete. Display: {st.session_state.display_image.mode} {st.session_state.display_image.size}, Processed: {st.session_state.processed_image.mode} {st.session_state.processed_image.size}")
                    st.session_state.roi_coords = None; st.session_state.canvas_drawing = None
                    st.success(f"✅ Image '{uploaded_file.name}' loaded!"); st.rerun()
                else: # Final Failure Cleanup
                    logger.critical("Image loading failed post-processing.");
                    if processing_successful: st.error("❌ Image loading failed unexpectedly post-processing.")
                    st.session_state.uploaded_file_info = None; st.session_state.raw_image_bytes = None; st.session_state.display_image = None; st.session_state.processed_image = None; st.session_state.dicom_dataset = None; st.session_state.dicom_metadata = {}; st.session_state.current_display_wc = None; st.session_state.current_display_ww = None
    st.markdown("---")
    if st.session_state.is_dicom and st.session_state.dicom_dataset and st.session_state.display_image: # W/L Controls
        with st.expander("DICOM Window/Level", expanded=False):
            try:
                wc_current_slider, ww_current_slider = dicom_wl_sliders(st.session_state.dicom_dataset, st.session_state.dicom_metadata)
                wc_displayed = st.session_state.get('current_display_wc'); ww_displayed = st.session_state.get('current_display_ww')
                # Initialize state if first run for this image
                if wc_displayed is None and ww_displayed is None and wc_current_slider is not None and ww_current_slider is not None:
                    st.session_state.current_display_wc = wc_current_slider; st.session_state.current_display_ww = ww_current_slider; logger.debug("Init display W/L state.")
                    wc_displayed = wc_current_slider # Update local vars for comparison below
                    ww_displayed = ww_current_slider
                # Check for changes if values are valid
                wc_changed = False; ww_changed = False
                if wc_current_slider is not None and wc_displayed is not None: wc_changed = abs(wc_current_slider - wc_displayed) > 1e-3
                if ww_current_slider is not None and ww_displayed is not None: ww_changed = abs(ww_current_slider - ww_displayed) > 1e-3

                if (wc_changed or ww_changed):
                    logger.info(f"Applying W/L: WC={wc_current_slider:.1f}, WW={ww_current_slider:.1f}")
                    with st.spinner("Applying Window/Level..."):
                         new_display_image = dicom_to_image(st.session_state.dicom_dataset, wc_current_slider, ww_current_slider)
                         if new_display_image:
                              if new_display_image.mode != 'RGB': st.session_state.display_image = new_display_image.convert('RGB')
                              else: st.session_state.display_image = new_display_image
                              st.session_state.current_display_wc = wc_current_slider; st.session_state.current_display_ww = ww_current_slider
                              st.rerun()
                         else: st.error("Failed to apply W/L settings."); logger.error("dicom_to_image failed for W/L.")
            except Exception as e: st.error(f"W/L control error: {e}"); logger.error(f"W/L slider error: {e}", exc_info=True)
        st.markdown("---")
    if isinstance(st.session_state.get("display_image"), Image.Image): # AI Actions
        st.subheader("AI Actions")
        if st.button("▶️ Run Initial Analysis", key="analyze_btn", help="General analysis.", use_container_width=True): st.session_state.last_action = "analyze"; st.rerun()
        st.markdown("---"); st.subheader("❓ Ask AI Question")
        if st.session_state.get("roi_coords"): rc = st.session_state.roi_coords; roi_summary = f"L:{rc['left']}, T:{rc['top']}, W:{rc['width']}, H:{rc['height']}"; st.info(f"✅ ROI Selected: [{roi_summary}].")
        if st.button("❌ Clear ROI", key="clear_roi_btn", help="Remove ROI.", use_container_width=True): st.session_state.roi_coords = None; st.session_state.canvas_drawing = None; logger.info("ROI cleared."); st.rerun()
        else: st.caption("ℹ️ Optionally, draw ROI on image.")
        question_input = st.text_area("Ask about the image or ROI:", height=100, key="question_input_widget", placeholder="e.g., Any abnormalities?", label_visibility="collapsed")
        if st.button("💬 Ask AI", key="ask_btn", use_container_width=True): user_question = st.session_state.question_input_widget; (st.session_state.last_action := "ask", st.rerun()) if user_question and user_question.strip() else st.warning("Enter question.")
        st.markdown("---"); st.subheader("🎯 Focused Condition Analysis")
        DISEASE_OPTIONS = ["Pneumonia", "Lung Cancer", "Stroke", "Fracture", "Appendicitis", "Tuberculosis", "COVID-19", "Pulm Emb", "Brain Tumor", "Arthritis", "Osteoporosis", "Cardiomegaly", "Aortic Aneurysm", "Bowel Obstruction"] # Shorter names
        disease_options_sorted = [""] + sorted(DISEASE_OPTIONS)
        disease_select = st.selectbox("Condition to analyze:", options=disease_options_sorted, key="disease_select_widget", help="Select condition.")
        if st.button("🩺 Run Condition Analysis", key="disease_btn", use_container_width=True): selected_disease = st.session_state.disease_select_widget; (st.session_state.last_action := "disease", st.rerun()) if selected_disease else st.warning("Select condition.")
        st.markdown("---")
        with st.expander("📊 Confidence & Report", expanded=True):
            can_estimate = bool(st.session_state.history or st.session_state.initial_analysis or st.session_state.disease_analysis)
            if st.button("📈 Estimate Confidence", key="confidence_btn", help="Estimate confidence.", disabled=not can_estimate, use_container_width=True): st.session_state.last_action = "confidence"; st.rerun()
            if not can_estimate: st.caption("Run analysis/QA first.")
            if st.button("📄 Generate PDF Data", key="generate_report_data_btn", help="Generate PDF data.", use_container_width=True): st.session_state.last_action = "generate_report_data"; st.rerun()
            if st.session_state.get("pdf_report_bytes"): report_filename = f"RadVisionAI_Report_{st.session_state.session_id}.pdf"; st.download_button(label="⬇️ Download PDF", data=st.session_state.pdf_report_bytes, file_name=report_filename, mime="application/pdf", key="download_pdf_button", help=f"Download report ({report_filename})", use_container_width=True); st.success("PDF ready.")
    else: st.info("👈 Upload image to begin.")

# =============================================================================
# === MAIN PANEL DISPLAYS =====================================================
# =============================================================================
col1, col2 = st.columns([2, 3])

# --- Column 1: Image Viewer, Canvas, Metadata ---
with col1:
    st.subheader("🖼️ Image Viewer")
    display_img_object = st.session_state.get("display_image")

    # Check if the display_image object exists and is a PIL Image
    if isinstance(display_img_object, Image.Image):
        logger.debug(f"Preparing viewer. Image details - Type: {type(display_img_object)}, Mode: {display_img_object.mode}, Size: {display_img_object.size}")

        bg_image_pil = None # Initialize
        try:
            # Prepare image for canvas background (ensure RGB)
            if display_img_object.mode == 'RGB':
                bg_image_pil = display_img_object
                logger.debug("Canvas Prep: Image is already RGB.")
            else:
                logger.info(f"Canvas Prep: Converting display image from {display_img_object.mode} to RGB.")
                bg_image_pil = display_img_object.convert('RGB')
                logger.debug("Canvas Prep: Conversion to RGB successful.")

            # Check type *after* potential conversion
            if not isinstance(bg_image_pil, Image.Image):
                 # Log detailed info if conversion failed unexpectedly
                 logger.critical(f"Image object became invalid after RGB conversion attempt! Original type: {type(display_img_object)}, Type after convert: {type(bg_image_pil)}")
                 raise TypeError(f"Image object invalid after RGB conversion attempt.")

            # <<< --- ADDED DEBUG STEP --- >>>
            # Optional: Keep this section commented out unless actively debugging canvas issues
            # st.markdown("--- Debug Image Check ---")
            # st.write(f"Type of image object BEFORE canvas: `{type(bg_image_pil)}`")
            # st.write(f"Mode: `{bg_image_pil.mode}` | Size: `{bg_image_pil.size}`")
            # # Try displaying the exact object with st.image
            # st.image(bg_image_pil, caption="Debug: This is the image being sent to the canvas", use_column_width=True)
            # st.markdown("--- End Debug Check ---")
            # <<< --- END DEBUG STEP --- >>>

        except Exception as prep_err:
             # Catch errors during preparation (conversion, type checks)
             st.error(f"Failed to prepare image for canvas display: {prep_err}")
             logger.error(f"Image preparation error for canvas background: {prep_err}", exc_info=True)
             bg_image_pil = None # Ensure it's None on failure

        # Proceed only if background image preparation was successful AND it's a valid Image
        if isinstance(bg_image_pil, Image.Image): # Re-check after try-except
            # --- Calculate Canvas Dimensions (Keep as before) ---
            MAX_CANVAS_WIDTH = 600; MAX_CANVAS_HEIGHT = 550
            img_w, img_h = bg_image_pil.size
            aspect_ratio = img_w / img_h if img_h > 0 else 1
            canvas_width = min(img_w, MAX_CANVAS_WIDTH); canvas_height = int(canvas_width / aspect_ratio) if aspect_ratio > 0 else MAX_CANVAS_HEIGHT
            if canvas_height > MAX_CANVAS_HEIGHT: canvas_height = MAX_CANVAS_HEIGHT; canvas_width = int(canvas_height * aspect_ratio)
            canvas_width = max(canvas_width, 150); canvas_height = max(canvas_height, 150)
            logger.info(f"Canvas Prep: Image size={img_w}x{img_h}, Calculated canvas size W={canvas_width}, H={canvas_height}")

            # --- Drawable Canvas ---
            if canvas_width > 0 and canvas_height > 0:
                st.caption("Click and drag on the image below to select a Region of Interest (ROI).")
                try:
                    initial_drawing = st.session_state.get("canvas_drawing")
                    logger.info(f"Attempting to render st_canvas with background image type: {type(bg_image_pil)}, mode: {bg_image_pil.mode}, size: {bg_image_pil.size}")
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.2)", stroke_width=2, stroke_color="rgba(220, 50, 50, 0.9)",
                        background_image=bg_image_pil, # Use the prepared RGB image
                        update_streamlit=True, height=int(canvas_height), width=int(canvas_width),
                        drawing_mode="rect", initial_drawing=initial_drawing,
                        key="drawable_canvas",
                    )
                    logger.info("st_canvas rendered (or attempted).")

                    # --- ROI Processing Logic (Keep as before) ---
                    if canvas_result and canvas_result.json_data is not None and canvas_result.json_data.get("objects"):
                        logger.debug(f"Canvas updated. Raw JSON data: {canvas_result.json_data}")
                        last_object = canvas_result.json_data["objects"][-1]
                        if last_object["type"] == "rect":
                             scale_x = img_w / canvas_width; scale_y = img_h / canvas_height
                             raw_left = int(last_object["left"]); raw_top = int(last_object["top"]); raw_width = int(last_object["width"] * last_object.get("scaleX", 1)); raw_height = int(last_object["height"] * last_object.get("scaleY", 1))
                             orig_left = int(raw_left * scale_x); orig_top = int(raw_top * scale_y); orig_width = int(raw_width * scale_x); orig_height = int(raw_height * scale_y)
                             logger.debug(f"Canvas Rect: L={raw_left}, T={raw_top}, W={raw_width}, H={raw_height} | Scaled ROI: L={orig_left}, T={orig_top}, W={orig_width}, H={orig_height}")
                             if orig_width > 10 and orig_height > 10:
                                 new_roi = {"left": orig_left, "top": orig_top, "width": orig_width, "height": orig_height}
                                 if st.session_state.roi_coords != new_roi: st.session_state.roi_coords = new_roi; st.session_state.canvas_drawing = canvas_result.json_data; logger.info(f"ROI coords updated: {new_roi}"); st.rerun()
                             else: logger.debug("Rect too small post-scaling.")

                except Exception as canvas_error:
                    st.error(f"Error rendering/processing the drawing canvas: {canvas_error}")
                    logger.error(f"st_canvas failed: {canvas_error}", exc_info=True)
                    # Check browser console for frontend errors
                    st.warning("Drawing feature may be unavailable. Check Browser's Developer Console (F12).")
            else:
                st.error("Calculated canvas dimensions invalid (<= 0). Cannot display drawing canvas.")
                logger.error(f"Invalid canvas dimensions: W={canvas_width}, H={canvas_height}")
        else:
             # Error message for invalid bg_image_pil already shown above
             st.info("Image could not be prepared for canvas.")
             logger.error("Cannot display canvas because bg_image_pil is invalid.")

        # --- DICOM Metadata Display (Keep as before) ---
        if st.session_state.is_dicom and st.session_state.dicom_metadata:
             display_dicom_metadata(st.session_state.dicom_metadata)

    # --- Fallback if display_image is NOT valid ---
    else:
        # This block remains the same - displays placeholder if no valid image
        logger.debug(f"Viewer: No valid display_image in state (Type: {type(display_img_object)}).")
        st.markdown("---")
        if st.session_state.get("uploaded_file_info"): st.warning("Image processing may have failed. Check logs or try again.")
        else: st.info("Image will appear here after uploading.")
        st.markdown("<div style='height: 400px; border: 2px dashed #ccc; display: flex; align-items: center; justify-content: center; color: #aaa;'>Image Display Area</div>", unsafe_allow_html=True) # Truncated style


# --- Column 2: Analysis Results Tabs ---
# ... (Code from lines 467 to 511 - Keep as before) ...
with col2:
    st.subheader("📊 Analysis & Results")
    tabs = st.tabs(["🔬 Initial Analysis", "💬 Q&A History", "🩺 Disease Focus", "📈 Confidence"])
    with tabs[0]: st.text_area("Overall Findings & Impressions", value=st.session_state.get("initial_analysis", "No initial analysis performed yet."), height=450, key="output_initial_text", disabled=True, help="General analysis results.")
    with tabs[1]:
        st.text_area("AI Answer to Last Question", value=st.session_state.get("qa_answer", "Ask a question using the sidebar."), height=200, key="output_qa_text", disabled=True, help="AI's answer to last question.")
        st.markdown("---")
        if st.session_state.history:
             with st.expander("View Full Conversation History", expanded=True):
                 for i, (q, a) in enumerate(reversed(st.session_state.history)): st.markdown(f"**You ({len(st.session_state.history)-i}):**"); st.caption(q); st.markdown(f"**AI ({len(st.session_state.history)-i}):**"); st.markdown(a, unsafe_allow_html=True);
                 if i < len(st.session_state.history) - 1: st.markdown("---")
        else: st.caption("No conversation history.")
    with tabs[2]: st.text_area("Disease-Specific Findings", value=st.session_state.get("disease_analysis", "No focused condition analysis performed yet."), height=450, key="output_disease_text", disabled=True, help="Analysis for selected condition.")
    with tabs[3]: st.text_area("AI Confidence Estimation", value=st.session_state.get("confidence_score", "No confidence estimation performed yet."), height=450, key="output_confidence_text", disabled=True, help="AI's estimated confidence.")


# --- Action Handling ---
# ... (Code from lines 514 to 684 - Keep full rewritten block as before) ...
current_action: Optional[str] = st.session_state.get("last_action")
if current_action:
    logger.info(f"Initiating action handling for: '{current_action}'")
    processed_image = st.session_state.get("processed_image"); session_id = st.session_state.get("session_id")
    if not isinstance(processed_image, Image.Image) or not session_id: error_msg = f"Cannot perform action '{current_action}': Processed image invalid."; st.error(error_msg); processed_image_type = type(processed_image).__name__ if processed_image else "None"; logger.error(f"Action '{current_action}' aborted. Img Type: {processed_image_type}, Session: {bool(session_id)}"); st.session_state.last_action = None; st.stop()
    img_for_llm: Image.Image = processed_image; roi: Optional[Dict[str, int]] = st.session_state.get("roi_coords"); roi_context_str = " (focused on ROI)" if roi else ""
    try:
        if current_action != "ask": st.session_state.qa_answer = ""
        if current_action != "analyze": st.session_state.initial_analysis = ""
        if current_action != "disease": st.session_state.disease_analysis = ""
        if current_action != "confidence": st.session_state.confidence_score = ""
        if current_action != "generate_report_data": st.session_state.pdf_report_bytes = None
        if current_action == "analyze": st.info("🔬 Performing preliminary analysis..."); with st.spinner("AI analyzing..."): analysis_result = run_initial_analysis(img_for_llm); st.session_state.initial_analysis = analysis_result; logger.info("Initial analysis completed."); st.success("Analysis finished.")
        elif current_action == "ask":
            question = st.session_state.get("question_input_widget", "").strip()
            if not question: st.warning("Please enter question."); logger.warning("Ask: empty question.")
            else:
                st.info(f"❓ Asking AI{roi_context_str}..."); st.session_state.qa_answer = ""; primary_model_name = "Gemini"
                with st.spinner(f"{primary_model_name} thinking..."): gemini_answer, success = run_multimodal_qa(img_for_llm, question, st.session_state.history, roi)
                if success: st.session_state.qa_answer = gemini_answer; st.session_state.history.append((question, gemini_answer)); logger.info(f"QA successful: '{question}'{roi_context_str}"); st.success(f"{primary_model_name} answered.")
                else:
                    error_message = f"Primary AI ({primary_model_name}) failed: {gemini_answer}"; st.error(error_message); logger.warning(f"Primary AI failed: '{question}'. Reason: {gemini_answer}"); st.session_state.qa_answer = f"**[Primary AI Error]** {gemini_answer}\n\n---\n"
                    hf_token = bool(os.environ.get("HF_API_TOKEN")); hf_model = bool(HF_VQA_MODEL_ID and HF_VQA_MODEL_ID not in ["hf_model_not_found", "unavailable"])
                    if hf_token and hf_model:
                        st.info(f"Attempting fallback ({HF_VQA_MODEL_ID})..."); with st.spinner(f"Fallback AI thinking..."): hf_answer, hf_success = query_hf_vqa_inference_api(img_for_llm, question, roi)
                        if hf_success: fallback_display = f"**[Fallback ({HF_VQA_MODEL_ID})]**\n\n{hf_answer}"; st.session_state.qa_answer += fallback_display; st.session_state.history.append((question, fallback_display)); logger.info(f"HF fallback ok: '{question}'"); st.success("Fallback AI answered.")
                        else: fallback_error = f"Fallback ({HF_VQA_MODEL_ID}) failed: {hf_answer}"; st.session_state.qa_answer += f"**[Fallback Failed]** {fallback_error}"; st.error(fallback_error); logger.error(f"HF fallback fail: '{question}'. Reason: {hf_answer}")
                    else: missing = [c for c, available in [("HF Token", hf_token), ("HF Model/Module", hf_model)] if not available]; fallback_msg = f"Fallback unavailable (Missing: {', '.join(missing)})."; st.session_state.qa_answer += f"**[Fallback Unavailable]** {fallback_msg}"; st.warning(fallback_msg); logger.warning(f"HF fallback skip: '{question}'. Missing: {missing}")
        elif current_action == "disease":
            selected_disease = st.session_state.get("disease_select_widget")
            if not selected_disease: st.warning("Select condition."); logger.warning("Disease: no selection.")
            else:
                st.info(f"🩺 Analyzing for '{selected_disease}'{roi_context_str}..."); with st.spinner(f"AI assessing '{selected_disease}'..."): disease_result = run_disease_analysis(img_for_llm, selected_disease, roi);
                st.session_state.disease_analysis = disease_result; logger.info(f"Disease analysis ok: '{selected_disease}'."); st.success(f"Analysis for '{selected_disease}' finished.")
        elif current_action == "confidence":
            st.info(f"📊 Estimating AI confidence{roi_context_str}..."); history = st.session_state.get("history", [])
            if not isinstance(history, list) or not history: st.warning("Cannot estimate: No history."); logger.warning("Confidence skip: No history.")
            else: with st.spinner("Calculating confidence..."): confidence_result = estimate_ai_confidence(img_for_llm, history);
            st.session_state.confidence_score = confidence_result; logger.info("Confidence estimation ok."); st.success("Confidence estimated.")
        elif current_action == "generate_report_data":
            st.info("📄 Preparing PDF data..."); st.session_state.pdf_report_bytes = None; img_for_report: Optional[Image.Image] = st.session_state.get("display_image"); img_with_roi = None
            if isinstance(img_for_report, Image.Image):
                roi_coords = st.session_state.get("roi_coords")
                if roi_coords and isinstance(roi_coords, dict) and all(k in roi_coords for k in ['left', 'top', 'width', 'height']):
                    try: # Draw ROI
                        img_copy = img_for_report.copy().convert("RGB"); draw = ImageDraw.Draw(img_copy); x0, y0 = int(roi_coords['left']), int(roi_coords['top']); x1, y1 = x0 + int(roi_coords['width']), y0 + int(roi_coords['height']); img_w, img_h = img_copy.size; x0=max(0,min(x0,img_w-1)); y0=max(0,min(y0,img_h-1)); x1=max(0,min(x1,img_w)); y1=max(0,min(y1,img_h))
                        if x1>x0 and y1>y0: draw.rectangle([x0,y0,x1,y1],outline="red",width=3); img_with_roi = img_copy; logger.info("Drew ROI on report image.")
                        else: logger.warning("Invalid ROI dims post-clamp."); img_with_roi = img_for_report
                    except Exception as e: logger.error(f"Failed to draw ROI: {e}", exc_info=True); st.warning("Could not draw ROI."); img_with_roi = img_for_report
                else: img_with_roi = img_for_report
                history = st.session_state.get("history", []); qa_hist = "\n\n".join([f"User Q: {q}\n\nAI A: {a}" for q, a in history]) if history else "No Q&A."
                outputs = {"Preliminary Analysis": st.session_state.get("initial_analysis","N/P"), "Conversation History": qa_hist, "Condition-Specific Analysis": st.session_state.get("disease_analysis","N/P"), "Last Confidence": st.session_state.get("confidence_score","N/E")}
                if st.session_state.get("is_dicom") and st.session_state.get("dicom_metadata"): # Add Filtered DICOM Meta
                    logger.info("Processing DICOM meta for report."); filtered_list = []
                    PHI_EXCLUDE = ["PatientName", "PatientID", "PatientBirthDate", "PatientSex", "OtherPatientIDs", "OtherPatientNames", "PatientAddress", "PatientTelephoneNumbers", "ReferringPhysicianName", "InstitutionName", "InstitutionAddress", "PhysicianOfRecord", "OperatorsName", "IssuerOfPatientID", "PatientBirthTime", "PatientComments", "PerformedProcedureStepStartDate", "PerformedProcedureStepStartTime", "RequestingPhysician", "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate", "StudyTime", "SeriesTime", "AcquisitionTime", "ContentTime", "AccessionNumber", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",] # Use full list
                    meta = st.session_state.dicom_metadata
                    for k, v in meta.items():
                        if k in PHI_EXCLUDE: logger.debug(f"Excluding tag from report: {k}"); continue
                        dv = "";
                        try:
                            if isinstance(v, list): dv = ", ".join(map(str, v))
                            elif isinstance(v, pydicom.uid.UID): dv = v.name
                            elif isinstance(v, bytes): dv = f"[{len(v)} bytes]"
                            elif isinstance(v, pydicom.valuerep.PersonName): dv = "[Person Name]"
                            else: dv = str(v).strip()
                            if dv and len(dv)<512: filtered_list.append(f"{k}: {dv}")
                        except Exception as tag_err: logger.warning(f"Err proc meta tag '{k}': {tag_err}"); filtered_list.append(f"{k}: [Error]")
                    outputs["DICOM Metadata (Filtered)"] = "\n".join(filtered_list) if filtered_list else "N/A"; logger.info(f"Incl {len(filtered_list)} filt tags.")
                with st.spinner("🎨 Generating PDF..."): pdf_bytes = generate_pdf_report_bytes(session_id, img_with_roi, outputs)
                if pdf_bytes: st.session_state.pdf_report_bytes = pdf_bytes; st.success("✅ PDF data generated!"); logger.info("PDF gen ok.")
                else: st.error("❌ PDF generation failed."); logger.error("PDF gen fail.")
            else: st.error("Cannot gen report: Invalid image."); logger.error("PDF skip: invalid image.")
        else: st.warning(f"Unknown action: '{current_action}'."); logger.warning(f"Unknown action: '{current_action}'")
    except Exception as e: st.error(f"Unexpected error during action '{current_action}'."); logger.critical(f"Unhandled action error: {e}", exc_info=True)
    finally: st.session_state.last_action = None; logger.debug(f"Action '{current_action}' finished."); st.rerun()

# --- Footer ---
# ... (Code from line 687 - Keep as before) ...
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced ")