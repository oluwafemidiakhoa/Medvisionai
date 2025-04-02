# main_app.py (Revision for Debugging Image Display)

# --- Core Libraries ---
import io
import os
import uuid
import logging
import base64
from typing import Any, Dict, Optional, Tuple, List
import copy # For deepcopy of default state

# --- Streamlit ---
import streamlit as st
# --- Drawable Canvas (Import early, check version) ---
try:
    from streamlit_drawable_canvas import st_canvas
    import streamlit_drawable_canvas as st_canvas_module
    CANVAS_VERSION = getattr(st_canvas_module, '__version__', 'Unknown')
except ImportError:
    st.error("CRITICAL ERROR: streamlit-drawable-canvas is not installed. Cannot run app. `pip install streamlit-drawable-canvas`")
    st.stop() # Stop if canvas is missing

# ------------------------------------------------------------------------------
# <<< --- Configure Streamlit Page (MUST BE FIRST st COMMAND) --- >>>
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="RadVision AI Debug", # Changed title for clarity
    layout="wide",
    page_icon="⚕️",
    initial_sidebar_state="expanded"
)

# --- Image & DICOM Processing (Import early, check versions) ---
try:
    from PIL import Image, ImageDraw, UnidentifiedImageError
    import PIL
    PIL_VERSION = getattr(PIL, '__version__', 'Unknown')
except ImportError:
    st.error("CRITICAL ERROR: Pillow (PIL) is not installed. Cannot run app. `pip install Pillow`")
    st.stop()
try:
    import pydicom
    import pydicom.errors
    PYDICOM_VERSION = getattr(pydicom, '__version__', 'Unknown')
except ImportError:
    st.error("CRITICAL ERROR: pydicom is not installed. Needed for DICOM support. `pip install pydicom`")
    # Don't stop immediately, allow non-DICOM use, but log warning
    PYDICOM_VERSION = 'Not Installed'
    pydicom = None # Set to None to allow checks later

# ------------------------------------------------------------------------------
# <<< --- Setup Logging (After set_page_config) --- >>>
# ------------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG").upper() # Default to DEBUG for this version
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s', # Added line number
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.info(f"--- App Start ---")
logger.info(f"Logging level set to {LOG_LEVEL}")
logger.info(f"Streamlit Version: {st.__version__}")
logger.info(f"Pillow (PIL) Version: {PIL_VERSION}")
logger.info(f"Pydicom Version: {PYDICOM_VERSION}")
logger.info(f"Streamlit Drawable Canvas Version: {CANVAS_VERSION}")

# ------------------------------------------------------------------------------
# Monkey-Patch (Optional, often not needed with modern Streamlit, keep for now)
# ------------------------------------------------------------------------------
import streamlit.elements.image as st_image
if not hasattr(st_image, "image_to_url"):
    # ... (monkey-patch code as before, slightly enhanced logging) ...
    def image_to_url_monkey_patch(img_obj: Any, *args, **kwargs) -> str:
        if isinstance(img_obj, Image.Image):
            buffered = io.BytesIO()
            format = "PNG"
            try:
                img_to_save = img_obj
                if img_obj.mode == 'RGBA': img_to_save = img_obj.convert('RGB')
                elif img_obj.mode == 'P': img_to_save = img_obj.convert('RGB')
                elif img_obj.mode not in ['RGB', 'L']: img_to_save = img_obj.convert('RGB')
                img_to_save.save(buffered, format=format)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/{format.lower()};base64,{img_str}"
            except Exception as e:
                 logger.error(f"Monkey-patch image_to_url failed: {e}", exc_info=True)
                 return ""
        else:
            logger.warning(f"Monkey-patch image_to_url: Unsupported type {type(img_obj)}")
            return ""
    st_image.image_to_url = image_to_url_monkey_patch
    logging.info("Applied monkey-patch for streamlit.elements.image.image_to_url (if missing)")

# ------------------------------------------------------------------------------
# <<< --- Import Custom Utilities & Fallbacks --- >>>
# ------------------------------------------------------------------------------
try:
    from dicom_utils import parse_dicom, extract_dicom_metadata, dicom_to_image, get_default_wl
    from llm_interactions import run_initial_analysis, run_multimodal_qa, run_disease_analysis, estimate_ai_confidence
    from report_utils import generate_pdf_report_bytes
    from ui_components import display_dicom_metadata, dicom_wl_sliders
    logger.info("Successfully imported custom utility modules.")
    # Optional HF Fallback
    try:
        from hf_models import query_hf_vqa_inference_api, HF_VQA_MODEL_ID
    except ImportError:
        HF_VQA_MODEL_ID = "hf_model_not_found"
        def query_hf_vqa_inference_api(img: Image.Image, question: str, roi: Optional[Dict] = None) -> Tuple[str, bool]:
            return "[Fallback Unavailable] Hugging Face module not found.", False
        logger.warning("hf_models.py not found or import failed. HF VQA fallback disabled.")
except ImportError as import_error:
    st.error(f"CRITICAL ERROR importing helpers ({import_error}). App cannot function correctly. Ensure all .py files are present.")
    logger.critical(f"Failed import of required utils: {import_error}", exc_info=True)
    st.stop() # Stop if core helpers are missing

# --- Helper Image Conversion (Redundant if monkey-patch works, but safe fallback) ---
def safe_image_to_data_url(img: Image.Image) -> str:
    if not isinstance(img, Image.Image):
        logger.warning(f"safe_image_to_data_url: Input is not a PIL Image (type: {type(img)}).")
        return ""
    buffered = io.BytesIO()
    format = "PNG"
    try:
        img_to_save = img
        # Aggressive conversion for safety, though PNG supports many modes
        if img.mode not in ['RGB', 'L', 'RGBA']:
             logger.warning(f"safe_image_to_data_url: Converting image mode {img.mode} to RGB for data URL.")
             img_to_save = img.convert('RGB')
        elif img.mode == 'P': # Palette needs conversion for some uses
             img_to_save = img.convert('RGB')

        img_to_save.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{format.lower()};base64,{img_str}"
    except Exception as e:
        logger.error(f"Failed to convert image to data URL: {e}", exc_info=True)
        return ""

# ------------------------------------------------------------------------------
# Initialize Session State
# ------------------------------------------------------------------------------
DEFAULT_STATE = {
    "uploaded_file_info": None, "raw_image_bytes": None, "is_dicom": False,
    "dicom_dataset": None, "dicom_metadata": {},
    "processed_image": None, "display_image": None, # Crucial states for display
    "session_id": None, "history": [], "initial_analysis": "", "qa_answer": "",
    "disease_analysis": "", "confidence_score": "", "last_action": None,
    "pdf_report_bytes": None, "canvas_drawing": None, "roi_coords": None,
    'current_display_wc': None, 'current_display_ww': None,
}
for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        # Use deepcopy for mutable defaults like lists/dicts
        if isinstance(default_value, (list, dict)):
             st.session_state[key] = copy.deepcopy(default_value)
        else:
             st.session_state[key] = default_value
if not isinstance(st.session_state.history, list): st.session_state.history = []
logger.debug("Session state initialized.")

# ------------------------------------------------------------------------------
# Page Title & Disclaimer
# ------------------------------------------------------------------------------
st.title("⚕️ RadVision QA Advanced: AI-Assisted Image Analysis")
with st.expander("⚠️ Important Disclaimer & Usage Guide", expanded=False):
    # ... (Disclaimer content as before) ...
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
st.markdown("---")

# =============================================================================
# === SIDEBAR CONTROLS ========================================================
# =============================================================================
with st.sidebar:
    st.header("Image Upload & Controls")

    uploaded_file = st.file_uploader(
        "Upload Image (JPG, PNG, DCM)",
        type=["jpg", "jpeg", "png", "dcm", "dicom"],
        key="file_uploader_widget",
        accept_multiple_files=False,
        help="Select standard image or DICOM file. Check logs if processing fails."
    )

    # --- File Processing Logic ---
    if uploaded_file is not None:
        # --- Change Detection ---
        try:
             file_mtime = getattr(uploaded_file, 'last_modified', None)
             if file_mtime is None: # Fallback hash
                 import hashlib; hasher = hashlib.md5(); file_content_peek = uploaded_file.read(1024*1024); hasher.update(file_content_peek); file_unique_id = hasher.hexdigest(); uploaded_file.seek(0)
                 logger.warning("Using MD5 hash for change detection (last_modified missing).")
             else: file_unique_id = str(file_mtime)
             new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{file_unique_id}"
        except Exception as file_info_err:
             logger.error(f"Error getting file info: {file_info_err}", exc_info=True)
             new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{str(uuid.uuid4())[:8]}"

        # --- Process if New File ---
        if new_file_info != st.session_state.get("uploaded_file_info"):
            logger.info(f"New file detected: {uploaded_file.name} ({uploaded_file.type}, {uploaded_file.size} bytes)")
            st.toast(f"Processing '{uploaded_file.name}'...", icon="⏳")

            # --- Reset State ---
            logger.debug("Resetting session state for new file...")
            preserve_keys = {"file_uploader_widget"}
            current_state_snapshot = {k: v for k, v in st.session_state.items()} # Snapshot before reset
            for key, default_value in DEFAULT_STATE.items():
                if key not in preserve_keys:
                    if isinstance(default_value, (list, dict)): st.session_state[key] = copy.deepcopy(default_value)
                    else: st.session_state[key] = default_value
            logger.debug(f"State reset. Previous keys: {list(current_state_snapshot.keys())}, New keys: {list(st.session_state.keys())}")
            st.session_state.uploaded_file_info = new_file_info
            st.session_state.session_id = str(uuid.uuid4())[:8]
            logger.info(f"New Session ID: {st.session_state.session_id}")

            # --- Image Reading and Processing ---
            with st.spinner("🔬 Processing image... Please wait."):
                st.session_state.raw_image_bytes = None # Ensure clean slate
                temp_display_image = None
                temp_processed_image = None
                processing_successful = False

                try:
                    logger.debug("Reading file bytes...")
                    st.session_state.raw_image_bytes = uploaded_file.getvalue()
                    if not st.session_state.raw_image_bytes: raise ValueError("File is empty or could not be read.")
                    logger.info(f"Read {len(st.session_state.raw_image_bytes)} bytes successfully.")

                    # --- Determine File Type (DICOM or Standard) ---
                    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                    is_likely_dicom_magic = (len(st.session_state.raw_image_bytes) > 132 and st.session_state.raw_image_bytes[128:132] == b'DICM')
                    st.session_state.is_dicom = (pydicom is not None) and (file_ext in (".dcm", ".dicom") or "dicom" in uploaded_file.type.lower() or is_likely_dicom_magic)
                    logger.info(f"Identified as DICOM: {st.session_state.is_dicom} (pydicom available: {pydicom is not None})")

                    # --- DICOM Processing Branch ---
                    if st.session_state.is_dicom:
                        logger.debug("Attempting DICOM processing...")
                        ds = None
                        try:
                            ds = parse_dicom(st.session_state.raw_image_bytes, uploaded_file.name)
                            st.session_state.dicom_dataset = ds
                        except pydicom.errors.InvalidDicomError as dicom_err:
                            st.error(f"Invalid DICOM file: {dicom_err}")
                            logger.error(f"InvalidDicomError: {dicom_err}", exc_info=True)
                            ds = None # Ensure ds is None on parse failure
                        except Exception as parse_err:
                            st.error(f"Error parsing DICOM: {parse_err}")
                            logger.error(f"DICOM parsing failed: {parse_err}", exc_info=True)
                            ds = None

                        if ds:
                            logger.info("DICOM parsed successfully.")
                            st.session_state.dicom_metadata = extract_dicom_metadata(ds)
                            wc_default, ww_default = get_default_wl(ds, fallback_wc=50, fallback_ww=400)
                            st.session_state.current_display_wc, st.session_state.current_display_ww = wc_default, ww_default
                            logger.info(f"Using DICOM W/L: WC={wc_default}, WW={ww_default}")

                            logger.debug("Generating display image from DICOM (using W/L)...")
                            temp_display_image = dicom_to_image(ds, wc_default, ww_default)
                            if not isinstance(temp_display_image, Image.Image): logger.error("dicom_to_image (W/L) did not return a valid PIL Image.")

                            logger.debug("Generating processed image from DICOM (auto-scaled)...")
                            temp_processed_image = dicom_to_image(ds, window_center=None, window_width=None)
                            if not isinstance(temp_processed_image, Image.Image): logger.error("dicom_to_image (auto) did not return a valid PIL Image.")

                            if isinstance(temp_display_image, Image.Image) and isinstance(temp_processed_image, Image.Image):
                                logger.info("DICOM image objects generated successfully.")
                                processing_successful = True
                            else:
                                st.error("Failed to generate usable image objects from DICOM data.")
                                logger.error("One or both image generation steps failed for DICOM.")
                        else:
                            logger.error("DICOM processing aborted because parsing failed.")
                            # Error message already shown above

                    # --- Standard Image Processing Branch ---
                    else:
                        logger.debug("Attempting standard image processing...")
                        try:
                            img = Image.open(io.BytesIO(st.session_state.raw_image_bytes))
                            logger.info(f"Image.open successful. Original mode: {img.mode}, Size: {img.size}")

                            # Create copies for display and processing
                            img_copy_for_display = img.copy()
                            img_copy_for_processing = img.copy()

                            # Ensure display image is RGB
                            if img_copy_for_display.mode != 'RGB':
                                logger.info(f"Converting display image from {img_copy_for_display.mode} to RGB.")
                                temp_display_image = img_copy_for_display.convert("RGB")
                            else:
                                temp_display_image = img_copy_for_display
                            logger.debug(f"Display image prepared. Mode: {temp_display_image.mode}")

                            # Prepare processed image (can keep original mode or convert as needed by AI)
                            # Example: Keep original unless it's palette etc.
                            if img_copy_for_processing.mode in ['P', 'RGBA']: # Modes often needing conversion
                                logger.info(f"Converting processed image from {img_copy_for_processing.mode} to RGB for AI.")
                                temp_processed_image = img_copy_for_processing.convert("RGB")
                            else:
                                temp_processed_image = img_copy_for_processing # Use as-is (L, RGB etc)
                            logger.debug(f"Processed image prepared. Mode: {temp_processed_image.mode}")

                            st.session_state.dicom_dataset = None; st.session_state.dicom_metadata = {}
                            st.session_state.current_display_wc = None; st.session_state.current_display_ww = None
                            processing_successful = True
                            logger.info("Standard image objects generated successfully.")

                        except UnidentifiedImageError:
                            st.error(f"Cannot identify image format for '{uploaded_file.name}'. Check file type.")
                            logger.error(f"UnidentifiedImageError: {uploaded_file.name}", exc_info=True)
                        except Exception as img_err:
                            st.error(f"Error processing image '{uploaded_file.name}': {img_err}")
                            logger.error(f"Standard image processing error: {img_err}", exc_info=True)

                except Exception as outer_proc_err:
                    # Catch errors during file reading or type determination
                    st.error(f"Critical error during image processing setup: {outer_proc_err}")
                    logger.critical(f"Error in outer processing block: {outer_proc_err}", exc_info=True)
                    processing_successful = False # Ensure failure state


                # --- Final Check and State Update ---
                logger.debug(f"Final Check: processing_successful={processing_successful}, is display_image PIL={isinstance(temp_display_image, Image.Image)}, is processed_image PIL={isinstance(temp_processed_image, Image.Image)}")
                if processing_successful and isinstance(temp_display_image, Image.Image) and isinstance(temp_processed_image, Image.Image):
                    # Ensure display image is RGB (should be, but final check)
                    if temp_display_image.mode != 'RGB':
                        logger.warning(f"Final Check: Display image mode is {temp_display_image.mode}, forcing RGB conversion.")
                        try:
                            st.session_state.display_image = temp_display_image.convert('RGB')
                        except Exception as final_convert_err:
                             logger.error(f"Final RGB conversion for display image failed: {final_convert_err}", exc_info=True)
                             st.error("Failed final conversion to RGB for display.")
                             # Mark as failure if conversion fails
                             processing_successful = False # Override success
                    else:
                        st.session_state.display_image = temp_display_image

                    # Only proceed if display image is set
                    if processing_successful:
                        st.session_state.processed_image = temp_processed_image
                        logger.info(f"**SUCCESS**: State updated. Display Img: {st.session_state.display_image.mode} {st.session_state.display_image.size}, Processed Img: {st.session_state.processed_image.mode} {st.session_state.processed_image.size}")
                        # Reset ROI/Canvas/Results for new image
                        st.session_state.roi_coords = None; st.session_state.canvas_drawing = None
                        st.session_state.initial_analysis = ""; st.session_state.qa_answer = ""; st.session_state.disease_analysis = ""; st.session_state.confidence_score = ""; st.session_state.pdf_report_bytes = None; st.session_state.history = []
                        logger.debug("Reset results state for new image.")
                        st.success(f"✅ Image '{uploaded_file.name}' processed!")
                        st.rerun() # Trigger UI update with the new image state
                    else:
                         # Handle case where final conversion failed
                         logger.error("Processing marked as failed during final checks (likely RGB conversion).")
                         st.error("Image processing failed during final conversion steps.")
                         # Explicitly clear potentially partially set state
                         st.session_state.display_image = None
                         st.session_state.processed_image = None


                else: # Processing failed earlier or final objects were invalid
                    logger.critical("Image loading pipeline failed. Check previous logs for specific errors.")
                    if processing_successful: # This implies the PIL check failed
                        st.error("❌ Image processed, but final image objects are invalid. Check logs.")
                        logger.error(f"Final check failed: display type {type(temp_display_image)}, processed type {type(temp_processed_image)}")
                    # else: # Error message likely already shown by specific try/except blocks
                    #    st.error("❌ Image processing failed. Check logs for details.")

                    # --- Cleanup State on Failure ---
                    logger.warning("Clearing potentially inconsistent state due to processing failure.")
                    st.session_state.uploaded_file_info = None # Allow re-upload attempt
                    st.session_state.raw_image_bytes = None; st.session_state.display_image = None; st.session_state.processed_image = None; st.session_state.dicom_dataset = None; st.session_state.dicom_metadata = {}; st.session_state.current_display_wc = None; st.session_state.current_display_ww = None; st.session_state.is_dicom = False

            # End `with st.spinner`
        # End `if new_file_info != ...`
    # End `if uploaded_file is not None`


    # --- DICOM W/L Controls ---
    # (Keep W/L logic as in previous corrected version - seems okay)
    st.markdown("---")
    if st.session_state.is_dicom and st.session_state.dicom_dataset and isinstance(st.session_state.get("display_image"), Image.Image): # W/L Controls + check display_image
        with st.expander("DICOM Window/Level", expanded=False):
            try:
                wc_current_slider, ww_current_slider = dicom_wl_sliders(
                    st.session_state.dicom_dataset, st.session_state.dicom_metadata,
                    initial_wc=st.session_state.get('current_display_wc'), initial_ww=st.session_state.get('current_display_ww')
                )
                wc_displayed = st.session_state.get('current_display_wc'); ww_displayed = st.session_state.get('current_display_ww')
                slider_values_valid = (wc_current_slider is not None and ww_current_slider is not None)
                displayed_values_exist = (wc_displayed is not None and ww_displayed is not None)
                update_needed = False
                if slider_values_valid:
                    if not displayed_values_exist: update_needed = True
                    else:
                        wc_changed = abs(wc_current_slider - wc_displayed) > 1e-3
                        ww_changed = abs(ww_current_slider - ww_displayed) > 1e-3
                        if wc_changed or ww_changed: update_needed = True

                if update_needed:
                    logger.info(f"W/L: Applying WC={wc_current_slider:.1f}, WW={ww_current_slider:.1f}")
                    with st.spinner("Applying Window/Level..."):
                         new_display_image = dicom_to_image(st.session_state.dicom_dataset, wc_current_slider, ww_current_slider)
                         if isinstance(new_display_image, Image.Image): # Check result
                              # Ensure RGB for display consistency
                              if new_display_image.mode != 'RGB':
                                   logger.debug(f"W/L: Converting new display image from {new_display_image.mode} to RGB")
                                   st.session_state.display_image = new_display_image.convert('RGB')
                              else:
                                   st.session_state.display_image = new_display_image
                              st.session_state.current_display_wc = wc_current_slider
                              st.session_state.current_display_ww = ww_current_slider
                              logger.debug("W/L applied, rerunning.")
                              st.rerun()
                         else:
                              st.error("Failed to apply W/L settings (image generation failed)."); logger.error("dicom_to_image returned None/invalid for W/L update.")
            except Exception as e: st.error(f"W/L control error: {e}"); logger.error(f"W/L slider/update error: {e}", exc_info=True)
        st.markdown("---")


    # --- AI Actions ---
    # (Keep AI Action buttons logic as in previous corrected version - seems okay)
    if isinstance(st.session_state.get("display_image"), Image.Image): # AI Actions available only if display_image is valid
        st.subheader("AI Actions")
        # --- Initial Analysis ---
        if st.button("▶️ Run Initial Analysis", key="analyze_btn", help="Perform a general analysis of the image.", use_container_width=True):
            st.session_state.last_action = "analyze"; st.rerun()
        st.markdown("---")
        # --- Q&A ---
        st.subheader("❓ Ask AI Question")
        roi_coords = st.session_state.get("roi_coords")
        if roi_coords:
            rc = roi_coords; roi_summary = f"L:{rc['left']}, T:{rc['top']}, W:{rc['width']}, H:{rc['height']}"
            st.info(f"✅ ROI Selected: [{roi_summary}]")
            if st.button("❌ Clear ROI", key="clear_roi_btn", help="Remove ROI.", use_container_width=True):
                st.session_state.roi_coords = None; st.session_state.canvas_drawing = None; logger.info("ROI cleared."); st.rerun()
        else:
            st.caption("ℹ️ Optionally, draw ROI on image.")
        question_input = st.text_area("Ask about the image or ROI:", height=100, key="question_input_widget", placeholder="e.g., Any abnormalities?", label_visibility="collapsed")
        if st.button("💬 Ask AI", key="ask_btn", use_container_width=True):
            user_question = st.session_state.question_input_widget
            if user_question and user_question.strip():
                st.session_state.last_action = "ask"; logger.info(f"Ask AI clicked: '{user_question[:50]}...'"); st.rerun()
            else: st.warning("Enter question."); logger.warning("Ask AI clicked with empty question.")
        st.markdown("---")
        # --- Condition Analysis ---
        st.subheader("🎯 Focused Condition Analysis")
        DISEASE_OPTIONS = ["Pneumonia", "Lung Cancer", "Stroke", "Fracture", "Appendicitis", "Tuberculosis", "COVID-19", "Pulmonary Embolism", "Brain Tumor", "Arthritis", "Osteoporosis", "Cardiomegaly", "Aortic Aneurysm", "Bowel Obstruction", "Mass/Nodule", "Effusion"]
        disease_options_sorted = [""] + sorted(DISEASE_OPTIONS)
        disease_select = st.selectbox("Condition to analyze:", options=disease_options_sorted, key="disease_select_widget", help="Select condition.")
        if st.button("🩺 Run Condition Analysis", key="disease_btn", use_container_width=True):
            selected_disease = st.session_state.disease_select_widget
            if selected_disease: st.session_state.last_action = "disease"; logger.info(f"Condition Analysis clicked: '{selected_disease}'"); st.rerun()
            else: st.warning("Select condition."); logger.warning("Condition Analysis clicked empty.")
        st.markdown("---")
        # --- Confidence & Report ---
        with st.expander("📊 Confidence & Report", expanded=True):
            can_estimate = bool(st.session_state.history or st.session_state.initial_analysis or st.session_state.disease_analysis)
            if st.button("📈 Estimate Confidence", key="confidence_btn", help="Estimate confidence.", disabled=not can_estimate, use_container_width=True):
                st.session_state.last_action = "confidence"; logger.info("Estimate Confidence clicked."); st.rerun()
            if not can_estimate: st.caption("Run analysis/QA first.")
            if st.button("📄 Generate PDF Data", key="generate_report_data_btn", help="Generate PDF data.", use_container_width=True):
                st.session_state.last_action = "generate_report_data"; logger.info("Generate PDF Data clicked."); st.rerun()
            if st.session_state.get("pdf_report_bytes"):
                report_filename = f"RadVisionAI_Report_{st.session_state.session_id or 'session'}.pdf"
                st.download_button(label="⬇️ Download PDF Report", data=st.session_state.pdf_report_bytes, file_name=report_filename, mime="application/pdf", key="download_pdf_button", help=f"Download report ({report_filename})", use_container_width=True)
    else: # No valid image loaded yet
        st.info("👈 Upload an image file to begin analysis.")


# =============================================================================
# === MAIN PANEL DISPLAYS =====================================================
# =============================================================================
col1, col2 = st.columns([2, 3])

# --- Column 1: Image Viewer, Canvas, Metadata ---
with col1:
    st.subheader("🖼️ Image Viewer")
    # Get the display image object from state
    display_img_object = st.session_state.get("display_image")
    logger.debug(f"Main Panel: Checking display_image from state. Type: {type(display_img_object)}")

    # --- Original Canvas Logic (Proceed only if display_img_object is still valid PIL Image) ---
    if isinstance(display_img_object, Image.Image):
        logger.debug(f"Viewer: Proceeding to canvas setup. Image Mode: {display_img_object.mode}, Size: {display_img_object.size}")
        # Prepare background image for canvas (ensure RGB)
        bg_image_pil = None
        try:
            if display_img_object.mode == 'RGB':
                bg_image_pil = display_img_object
                logger.debug("Canvas Prep: Image is already RGB.")
            else:
                logger.info(f"Canvas Prep: Converting display image from {display_img_object.mode} to RGB for canvas.")
                bg_image_pil = display_img_object.convert('RGB')
                logger.debug(f"Canvas Prep: Conversion result type {type(bg_image_pil)}, mode {getattr(bg_image_pil, 'mode', 'N/A')}")
            # Final check after conversion
            if not isinstance(bg_image_pil, Image.Image):
                raise TypeError(f"Image object invalid after RGB conversion attempt (became {type(bg_image_pil)}).")
        except Exception as prep_err:
             st.error(f"Failed to prepare image for canvas display: {prep_err}")
             logger.error(f"Canvas Prep: Image preparation error: {prep_err}", exc_info=True)
             bg_image_pil = None # Ensure it's None on failure

        # Proceed only if background image preparation was successful
        if isinstance(bg_image_pil, Image.Image):
            # --- Calculate Canvas Dimensions ---
            MAX_CANVAS_WIDTH = 700; MAX_CANVAS_HEIGHT = 600
            img_w, img_h = bg_image_pil.size
            aspect_ratio = img_w / img_h if img_h > 0 else 1
            container_width = MAX_CANVAS_WIDTH; canvas_width = min(img_w, container_width); canvas_height = int(canvas_width / aspect_ratio) if aspect_ratio > 0 else MAX_CANVAS_HEIGHT
            if canvas_height > MAX_CANVAS_HEIGHT: canvas_height = MAX_CANVAS_HEIGHT; canvas_width = int(canvas_height * aspect_ratio)
            canvas_width = max(int(canvas_width), 150); canvas_height = max(int(canvas_height), 150)
            logger.info(f"Canvas Prep: Calculated canvas size W={canvas_width}, H={canvas_height}")

            # --- Drawable Canvas ---
            if canvas_width > 0 and canvas_height > 0:
                st.caption("Draw ROI below. (If image doesn't appear here but did above, canvas has an issue).")
                try:
                    initial_drawing = st.session_state.get("canvas_drawing")
                    if initial_drawing and not isinstance(initial_drawing, dict): initial_drawing = None # Safety check
                    logger.info(f"Rendering st_canvas. BG img mode: {bg_image_pil.mode}, size: {bg_image_pil.size}. Initial drawing: {'Set' if initial_drawing else 'None'}")

                    # Make sure bg_image_pil is still valid before passing
                    if not isinstance(bg_image_pil, Image.Image):
                         raise ValueError("Background image became invalid just before st_canvas call.")

                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.2)", stroke_width=2, stroke_color="rgba(220, 50, 50, 0.9)",
                        background_image=bg_image_pil, # Use the prepared RGB image
                        update_streamlit=True, height=canvas_height, width=canvas_width,
                        drawing_mode="rect", initial_drawing=initial_drawing,
                        key="drawable_canvas",
                    )
                    logger.info("st_canvas rendered (or attempted).")

                    # --- ROI Processing Logic ---
                    if canvas_result is not None and canvas_result.json_data is not None:
                        st.session_state.canvas_drawing = canvas_result.json_data # Store latest state
                        if canvas_result.json_data.get("objects"):
                            last_object = canvas_result.json_data["objects"][-1]
                            if last_object["type"] == "rect":
                                 scale_x = img_w / canvas_width; scale_y = img_h / canvas_height
                                 raw_left = int(last_object["left"]); raw_top = int(last_object["top"]); raw_width = int(last_object["width"] * last_object.get("scaleX", 1)); raw_height = int(last_object["height"] * last_object.get("scaleY", 1))
                                 orig_left = max(0, int(raw_left * scale_x)); orig_top = max(0, int(raw_top * scale_y))
                                 orig_width = int(raw_width * scale_x); orig_height = int(raw_height * scale_y)
                                 orig_right = min(img_w, orig_left + orig_width); orig_bottom = min(img_h, orig_top + orig_height)
                                 orig_width = max(0, orig_right - orig_left); orig_height = max(0, orig_bottom - orig_top)
                                 MIN_ROI_DIM = 10
                                 if orig_width >= MIN_ROI_DIM and orig_height >= MIN_ROI_DIM:
                                     new_roi = {"left": orig_left, "top": orig_top, "width": orig_width, "height": orig_height}
                                     if st.session_state.roi_coords != new_roi:
                                         st.session_state.roi_coords = new_roi; logger.info(f"ROI updated: {new_roi}"); st.rerun()
                        elif not canvas_result.json_data.get("objects") and st.session_state.roi_coords is not None:
                                logger.info("Canvas cleared, removing ROI coordinates."); st.session_state.roi_coords = None; # Removed rerun here

                except Exception as canvas_error:
                    st.error(f"Error rendering/processing drawing canvas: {canvas_error}")
                    logger.error(f"st_canvas interaction failed: {canvas_error}", exc_info=True)
                    st.warning("Drawing feature unavailable. Check Browser's Developer Console (F12) for frontend errors (WebGL, image loading etc.).")
            else:
                st.error("Canvas dimensions invalid (<= 0). Cannot display drawing canvas.")
                logger.error(f"Invalid canvas dimensions: W={canvas_width}, H={canvas_height}")
        else:
             st.info("Image could not be prepared for canvas background.")
             logger.error("Cannot display canvas because bg_image_pil is invalid or prep failed.")

        # --- DICOM Metadata Display ---
        if st.session_state.is_dicom and st.session_state.dicom_metadata:
             logger.debug("Displaying DICOM metadata.")
             display_dicom_metadata(st.session_state.dicom_metadata)

    # --- Fallback if display_image is NOT valid or st.image failed ---
    else:
        logger.debug("Viewer: No valid display_image object available for rendering.")
        st.markdown("---")
        if st.session_state.get("uploaded_file_info"):
            st.warning("Image processing failed or resulted in invalid data. Check logs above or in terminal.")
        else: st.info("Upload an image using the sidebar to start.")
        st.markdown("<div style='height: 400px; border: 2px dashed #ccc; display: flex; align-items: center; justify-content: center; text-align: center; color: #aaa; font-style: italic;'>Image Display Area<br/>(Upload Required or Processing Failed)</div>", unsafe_allow_html=True)


# --- Column 2: Analysis Results Tabs ---
# (Keep Tabs logic as in previous corrected version - seems okay)
with col2:
    st.subheader("📊 Analysis & Results")
    tab_titles = ["🔬 Initial Analysis", "💬 Q&A History", "🩺 Disease Focus", "📈 Confidence"]
    tabs = st.tabs(tab_titles)
    with tabs[0]:
        initial_analysis_content = st.session_state.get("initial_analysis", "").strip()
        initial_analysis_display = initial_analysis_content if initial_analysis_content else "No initial analysis performed yet."
        st.text_area("Overall Findings & Impressions", value=initial_analysis_display, height=450, key="output_initial_text", disabled=True, help="General analysis results.")
    with tabs[1]:
        qa_answer_content = st.session_state.get("qa_answer", "").strip()
        qa_answer_display = qa_answer_content if qa_answer_content else "Ask a question using the sidebar."
        st.text_area("AI Answer to Last Question", value=qa_answer_display, height=200, key="output_qa_text", disabled=True, help="AI's answer to last question.")
        st.markdown("---")
        if st.session_state.history:
             with st.expander("View Full Conversation History", expanded=True):
                 for i, (q, a) in enumerate(st.session_state.history):
                     st.markdown(f"**You ({i+1}):**"); st.caption(q); st.markdown(f"**AI ({i+1}):**"); st.markdown(a, unsafe_allow_html=True);
                     if i < len(st.session_state.history) - 1: st.markdown("---")
        else: st.caption("No conversation history.")
    with tabs[2]:
        disease_analysis_content = st.session_state.get("disease_analysis", "").strip()
        disease_analysis_display = disease_analysis_content if disease_analysis_content else "No focused condition analysis performed yet."
        st.text_area("Disease-Specific Findings", value=disease_analysis_display, height=450, key="output_disease_text", disabled=True, help="Analysis for selected condition.")
    with tabs[3]:
        confidence_score_content = st.session_state.get("confidence_score", "").strip()
        confidence_score_display = confidence_score_content if confidence_score_content else "No confidence estimation performed yet."
        st.text_area("AI Confidence Estimation", value=confidence_score_display, height=450, key="output_confidence_text", disabled=True, help="AI's estimated confidence.")


# =============================================================================
# === ACTION HANDLING LOGIC ===================================================
# =============================================================================
# (Keep Action Handling logic as in previous corrected version - seems okay)
current_action: Optional[str] = st.session_state.get("last_action")
if current_action:
    logger.info(f"ACTION HANDLER: Initiating action '{current_action}'")
    processed_image = st.session_state.get("processed_image")
    session_id = st.session_state.get("session_id")

    # Pre-Action Checks
    action_requires_processed_image = current_action not in ["generate_report_data"] # Report uses display_image
    if action_requires_processed_image and not isinstance(processed_image, Image.Image):
        error_msg = f"Cannot perform '{current_action}': Processed image invalid/missing."; st.error(error_msg); logger.error(f"Action '{current_action}' aborted. Invalid processed_image (type: {type(processed_image)})."); st.session_state.last_action = None; st.stop()
    if not session_id:
        error_msg = f"Cannot perform '{current_action}': Session ID missing."; st.error(error_msg); logger.error(f"Action '{current_action}' aborted. Missing Session ID."); st.session_state.last_action = None; st.stop()

    # Prepare Common Vars
    img_for_llm: Image.Image = processed_image
    roi: Optional[Dict[str, int]] = st.session_state.get("roi_coords")
    roi_context_str = " (focused on ROI)" if roi else ""
    history: List[Tuple[str, str]] = st.session_state.get("history", [])
    if not isinstance(history, list): history = []; st.session_state.history = history; logger.warning("History was not list, reset.")

    try:
        # --- Execute Actions --- (Simplified logging here, detailed logs in functions)
        if current_action == "analyze":
            st.info(f"🔬 Performing analysis{roi_context_str}...");
            with st.spinner("AI analyzing..."): analysis_result = run_initial_analysis(img_for_llm) # Add roi=roi if function supports
            st.session_state.initial_analysis = analysis_result; st.success("Analysis finished.")
        elif current_action == "ask":
            question = st.session_state.get("question_input_widget", "").strip()
            if not question: st.warning("Enter question."); logger.warning("Ask: empty question.")
            else:
                st.info(f"❓ Asking AI{roi_context_str}..."); st.session_state.qa_answer = ""
                primary_model_name = "Gemini"
                with st.spinner(f"{primary_model_name} thinking..."): gemini_answer, success = run_multimodal_qa(img_for_llm, question, history, roi)
                if success:
                    st.session_state.qa_answer = gemini_answer; st.session_state.history.append((question, gemini_answer)); st.success(f"{primary_model_name} answered.")
                else: # Primary failed, try fallback
                    error_message = f"Primary AI ({primary_model_name}) failed: {gemini_answer}"; st.error(error_message); logger.warning(f"Primary AI failed. Reason: {gemini_answer}")
                    st.session_state.qa_answer = f"**[Primary AI Error]** {gemini_answer}\n\n---\n"
                    # Simplified Fallback Check
                    hf_token = bool(os.environ.get("HF_API_TOKEN")); hf_model_ok = (HF_VQA_MODEL_ID and HF_VQA_MODEL_ID not in ["hf_model_not_found", "unavailable"] and 'query_hf_vqa_inference_api' in globals())
                    if hf_token and hf_model_ok:
                        st.info(f"Attempting fallback ({HF_VQA_MODEL_ID})...");
                        with st.spinner(f"Fallback AI thinking..."): hf_answer, hf_success = query_hf_vqa_inference_api(img_for_llm, question, roi)
                        if hf_success: fallback_display = f"**[Fallback ({HF_VQA_MODEL_ID})]**\n\n{hf_answer}"; st.session_state.qa_answer += fallback_display; st.session_state.history.append((f"[Fallback] {question}", fallback_display)); st.success("Fallback AI answered.")
                        else: fallback_error = f"Fallback ({HF_VQA_MODEL_ID}) failed: {hf_answer}"; st.session_state.qa_answer += f"**[Fallback Failed]** {fallback_error}"; st.error(fallback_error); logger.error(f"HF fallback fail. Reason: {hf_answer}")
                    else: fallback_msg = f"Fallback unavailable (HF Token: {hf_token}, HF Model/Func: {hf_model_ok})."; st.session_state.qa_answer += f"**[Fallback Unavailable]** {fallback_msg}"; st.warning(fallback_msg); logger.warning(f"HF fallback skipped. Missing: Token={not hf_token}, Model/Func={not hf_model_ok}")
        elif current_action == "disease":
            selected_disease = st.session_state.get("disease_select_widget")
            if not selected_disease: st.warning("Select condition."); logger.warning("Disease: no selection.")
            else:
                st.info(f"🩺 Analyzing for '{selected_disease}'{roi_context_str}...");
                with st.spinner(f"AI assessing '{selected_disease}'..."): disease_result = run_disease_analysis(img_for_llm, selected_disease, roi)
                st.session_state.disease_analysis = disease_result; st.success(f"Analysis for '{selected_disease}' finished.")
        elif current_action == "confidence":
            if not history and not st.session_state.initial_analysis and not st.session_state.disease_analysis: st.warning("Cannot estimate: No history/analysis."); logger.warning("Confidence skip: No context.")
            else:
                st.info(f"📊 Estimating AI confidence{roi_context_str}...");
                with st.spinner("Calculating confidence..."): confidence_result = estimate_ai_confidence(img_for_llm, history, st.session_state.initial_analysis, st.session_state.disease_analysis, roi) # Pass context
                st.session_state.confidence_score = confidence_result; st.success("Confidence estimated.")
        elif current_action == "generate_report_data":
            st.info("📄 Preparing PDF data..."); st.session_state.pdf_report_bytes = None;
            img_for_report: Optional[Image.Image] = st.session_state.get("display_image")
            if not isinstance(img_for_report, Image.Image): st.error("Cannot gen report: Invalid display image."); logger.error("PDF skip: invalid display_image.")
            else:
                 # Simplified Report Gen Logic (details in previous versions)
                 img_with_roi = img_for_report # Placeholder - Add drawing logic back if needed
                 roi_coords = st.session_state.get("roi_coords")
                 if roi_coords: # Add drawing logic here if desired (copy image, draw rect)
                    logger.info("Report: ROI detected, add drawing logic if needed.")
                    # Example placeholder: just log it for now
                    # img_copy = img_for_report.copy().convert("RGB"); draw = ImageDraw.Draw(img_copy); ... draw.rectangle ...; img_with_roi = img_copy

                 qa_hist = "\n\n".join([f"User Q: {q}\n\nAI A: {a}" for q, a in history]) if history else "No Q&A."
                 outputs = { "Session ID": session_id, "Preliminary Analysis": st.session_state.get("initial_analysis","N/P"), "Conversation History": qa_hist, "Condition-Specific Analysis": st.session_state.get("disease_analysis","N/P"), "Last Confidence": st.session_state.get("confidence_score","N/E")}
                 # Add filtered DICOM Meta if needed (logic from previous version)
                 if st.session_state.get("is_dicom") and st.session_state.get("dicom_metadata"):
                      logger.info("Report: Including filtered DICOM metadata.")
                      # Add filtering logic here...
                      outputs["DICOM Metadata (Filtered)"] = "DICOM Metadata Placeholder - Add filtering logic"

                 with st.spinner("🎨 Generating PDF..."): pdf_bytes = generate_pdf_report_bytes(session_id, img_with_roi, outputs)
                 if pdf_bytes: st.session_state.pdf_report_bytes = pdf_bytes; st.success("✅ PDF data generated!"); logger.info("PDF gen ok.")
                 else: st.error("❌ PDF generation failed."); logger.error("PDF gen fail.")
        else: st.warning(f"Unknown action: '{current_action}'."); logger.warning(f"Unknown action: '{current_action}'")
    except Exception as e: st.error(f"Unexpected error during action '{current_action}'."); logger.critical(f"Unhandled action error: {e}", exc_info=True)
    finally: st.session_state.last_action = None; logger.debug(f"Action '{current_action}' finished."); st.rerun()


# --- Footer ---
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced | Session: {st.session_state.get('session_id', 'N/A')}")

logger.info("--- App Render Complete ---")
