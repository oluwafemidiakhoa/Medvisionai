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
            format = "PNG" # Explicitly PNG for web display
            try:
                # Handle RGBA or other modes that might not save as JPEG easily
                if img_obj.mode == 'RGBA':
                    img_obj = img_obj.convert('RGB')
                elif img_obj.mode == 'P': # Palette mode, convert to RGB
                     img_obj = img_obj.convert('RGB')
                elif img_obj.mode not in ['RGB', 'L']: # Grayscale 'L' is fine for PNG
                    logger.warning(f"Monkey-patch: Converting image mode {img_obj.mode} to RGB for PNG.")
                    img_obj = img_obj.convert('RGB')

                img_obj.save(buffered, format=format)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/{format.lower()};base64,{img_str}"
            except Exception as e:
                 logging.error(f"Monkey-patched image_to_url failed: {e}", exc_info=True)
                 return ""
        # Add handling for numpy arrays if needed, converting them to PIL Images first
        # elif isinstance(img_obj, np.ndarray):
        #    try:
        #        img = Image.fromarray(...) # Add logic to handle numpy array formats
        #        # ... rest of the PIL image handling ...
        #    except Exception as e:
        #        logging.error(f"Monkey-patched image_to_url (numpy) failed: {e}")
        #        return ""
        else:
            # Log the actual type for better debugging
            logging.warning(f"Monkey-patched image_to_url: Unsupported type {type(img_obj)}. Input object was: {str(img_obj)[:100]}...") # Log first part of object string
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
# NOTE: The monkey-patch above serves a similar purpose for internal Streamlit use.
# This function might be redundant unless used explicitly elsewhere.
def image_to_data_url(img: Image.Image) -> str:
    # Re-using logic similar to the monkey patch for consistency
    buffered = io.BytesIO()
    format = "PNG" # Use PNG for data URLs
    try:
        # Ensure RGB for broad compatibility if needed, though PNG supports more modes
        img_to_save = img
        if img.mode == 'RGBA':
            img_to_save = img.convert('RGB') # Example: Convert RGBA to RGB
        elif img.mode == 'P':
             img_to_save = img.convert('RGB')

        img_to_save.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{format.lower()};base64,{img_str}"
    except Exception as e:
        logger.error(f"Failed to convert image to data URL: {e}", exc_info=True)
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
# ... (Sidebar code remains the same as in the previous corrected version, up to the AI Actions section) ...
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
             # Use creation time or hash as fallback if last_modified isn't reliable/present
             file_mtime = getattr(uploaded_file, 'last_modified', None)
             if file_mtime is None: # Fallback for environments where last_modified might not exist
                 import hashlib
                 hasher = hashlib.md5()
                 hasher.update(uploaded_file.getvalue())
                 file_unique_id = hasher.hexdigest()
                 uploaded_file.seek(0) # Reset pointer after getvalue()
                 logger.warning("File object missing 'last_modified'. Using MD5 hash for change detection.")
             else:
                 file_unique_id = str(file_mtime)

             new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{file_unique_id}"

        except Exception as file_info_err: # Fallback if attributes change or hashing fails
             logger.error(f"Error getting file info: {file_info_err}", exc_info=True)
             new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{str(uuid.uuid4())[:8]}" # Use random part as last resort

        if new_file_info != st.session_state.get("uploaded_file_info"):
            logger.info(f"New file upload detected: {uploaded_file.name} ({uploaded_file.type}, {uploaded_file.size} bytes)")
            st.toast(f"Processing '{uploaded_file.name}'...", icon="⏳")
            preserve_keys = {"file_uploader_widget"} # Add other keys if necessary
            logger.debug("Resetting session state for new file upload...")
            # Store current state before reset if needed for comparison or transfer
            # current_state_backup = {k: v for k, v in st.session_state.items()}
            for key, default_value in DEFAULT_STATE.items():
                if key not in preserve_keys:
                    # Use deepcopy for mutable defaults like lists/dicts if necessary
                    if isinstance(default_value, (list, dict)):
                         import copy
                         st.session_state[key] = copy.deepcopy(default_value)
                    else:
                         st.session_state[key] = default_value
            # Explicitly reset history (already covered by loop if default is [])
            # st.session_state.history = []
            logger.debug("Session state reset.")
            st.session_state.uploaded_file_info = new_file_info
            st.session_state.session_id = str(uuid.uuid4())[:8] # New session ID
            logger.info(f"Generated new Session ID: {st.session_state.session_id}")

            with st.spinner("🔬 Processing image... Please wait."):
                # Read file bytes ONCE
                try:
                    st.session_state.raw_image_bytes = uploaded_file.getvalue()
                    if not st.session_state.raw_image_bytes:
                        raise ValueError("Uploaded file is empty.")
                except Exception as read_err:
                    st.error(f"Failed to read uploaded file '{uploaded_file.name}': {read_err}")
                    logger.critical(f"File read error: {read_err}", exc_info=True)
                    # Clear state to prevent partial processing
                    st.session_state.uploaded_file_info = None
                    st.session_state.raw_image_bytes = None
                    st.stop() # Stop further processing for this upload attempt

                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                # More robust DICOM check: check magic number 'DICM' at byte 128
                is_likely_dicom_magic = False
                if len(st.session_state.raw_image_bytes) > 132:
                     try:
                         magic_word = st.session_state.raw_image_bytes[128:132].decode('ascii')
                         is_likely_dicom_magic = (magic_word == "DICM")
                         logger.debug(f"DICOM magic word check: Found '{magic_word}', Result: {is_likely_dicom_magic}")
                     except UnicodeDecodeError:
                         logger.debug("DICOM magic word check: Bytes 128-131 are not ASCII.")

                st.session_state.is_dicom = (file_ext in (".dcm", ".dicom") or "dicom" in uploaded_file.type.lower() or is_likely_dicom_magic)
                logger.info(f"File '{uploaded_file.name}' identified as DICOM: {st.session_state.is_dicom}")

                processing_successful = False; temp_display_image = None; temp_processed_image = None

                if st.session_state.is_dicom:
                    try: # DICOM Branch
                        logger.debug("Attempting to parse DICOM..."); ds = parse_dicom(st.session_state.raw_image_bytes, uploaded_file.name); st.session_state.dicom_dataset = ds
                        if ds:
                            logger.info("DICOM parsed successfully."); st.session_state.dicom_metadata = extract_dicom_metadata(ds); logger.debug(f"Extracted {len(st.session_state.dicom_metadata)} metadata tags.")
                            # Use fallback W/L if DICOM tags are missing/invalid
                            wc_default, ww_default = get_default_wl(ds, fallback_wc=50, fallback_ww=400) # Provide reasonable fallbacks
                            st.session_state.current_display_wc = wc_default; st.session_state.current_display_ww = ww_default; logger.info(f"Default/Fallback W/L: WC={wc_default}, WW={ww_default}")
                            logger.debug("Generating display image from DICOM using default W/L..."); temp_display_image = dicom_to_image(ds, wc_default, ww_default)
                            # Generate processed image (often needs normalization/specific W/L for AI)
                            # Option 1: Use default W/L as well (simpler)
                            # temp_processed_image = temp_display_image.copy() if temp_display_image else None
                            # Option 2: Use a standard scaling or AI-specific W/L (potentially better for AI)
                            logger.debug("Generating processed image from DICOM using auto-scaling..."); temp_processed_image = dicom_to_image(ds, window_center=None, window_width=None) # Or use specific values if known good ones exist
                            if temp_display_image and temp_processed_image: logger.info("DICOM images generated."); processing_successful = True
                            else: st.error("DICOM processing failed: Could not generate images."); logger.error("dicom_to_image returned None for display or processed image.")
                        else: st.error("Failed to parse DICOM file. It might be corrupted or not a valid DICOM format."); logger.error("parse_dicom returned None.")
                    except pydicom.errors.InvalidDicomError as dicom_err:
                        st.error(f"Invalid DICOM file: {dicom_err}. Please ensure it's a valid DICOM format."); logger.error(f"InvalidDicomError: {dicom_err}", exc_info=True); processing_successful = False
                    except Exception as e: st.error(f"Unexpected DICOM processing error: {e}"); logger.critical(f"DICOM pipeline error: {e}", exc_info=True); processing_successful = False
                else: # Standard Image Branch
                    try:
                        logger.debug("Attempting to open standard image..."); img = Image.open(io.BytesIO(st.session_state.raw_image_bytes))
                        # Ensure we have a copy for processing that doesn't affect display version initially
                        temp_processed_image = img.copy()
                        # Convert display image to RGB for consistency in display widgets
                        if img.mode != 'RGB': logger.info(f"Converting display image from {img.mode} to RGB."); temp_display_image = img.convert("RGB")
                        else: temp_display_image = img # It's already RGB
                        # AI might prefer/require RGB, Grayscale ('L'), or specific formats. Adjust temp_processed_image if needed.
                        # Example: If AI needs RGB:
                        # if temp_processed_image.mode != 'RGB':
                        #     logger.info(f"Converting processed image from {temp_processed_image.mode} to RGB for AI."); temp_processed_image = temp_processed_image.convert("RGB")

                        st.session_state.dicom_dataset = None; st.session_state.dicom_metadata = {}; st.session_state.current_display_wc = None; st.session_state.current_display_ww = None
                        processing_successful = True; logger.info("Standard image opened and prepared.")
                    except UnidentifiedImageError: st.error(f"Cannot identify image format for '{uploaded_file.name}'. Please upload a valid JPG, PNG, or DICOM file."); logger.error(f"UnidentifiedImageError: {uploaded_file.name}"); processing_successful = False
                    except Exception as e: st.error(f"Error processing image '{uploaded_file.name}': {e}"); logger.error(f"Std image processing error: {e}", exc_info=True); processing_successful = False

                # --- Final Check and State Update ---
                if processing_successful and isinstance(temp_display_image, Image.Image) and isinstance(temp_processed_image, Image.Image): # Final Check
                    # Ensure display image is RGB before storing (already done above, but double-check)
                    if temp_display_image.mode != 'RGB':
                        logger.warning(f"Display image was not RGB ({temp_display_image.mode}) before final state update. Converting.")
                        st.session_state.display_image = temp_display_image.convert('RGB')
                    else:
                        st.session_state.display_image = temp_display_image

                    # Store the processed image (could be L, RGB, etc., depending on processing steps)
                    st.session_state.processed_image = temp_processed_image

                    logger.info(f"Processing complete. Display Img: {st.session_state.display_image.mode} {st.session_state.display_image.size}, Processed Img: {st.session_state.processed_image.mode} {st.session_state.processed_image.size}")

                    # Reset ROI and drawing state for the new image
                    st.session_state.roi_coords = None; st.session_state.canvas_drawing = None
                    # Reset AI results
                    st.session_state.initial_analysis = ""; st.session_state.qa_answer = ""; st.session_state.disease_analysis = ""; st.session_state.confidence_score = ""; st.session_state.pdf_report_bytes = None; st.session_state.history = []

                    st.success(f"✅ Image '{uploaded_file.name}' loaded successfully!");
                    st.rerun() # Rerun to update UI with the new image and state
                else: # Final Failure Cleanup
                    logger.critical("Image loading failed post-processing or final checks failed.");
                    if processing_successful: # This means the processing *thought* it worked, but the images are invalid
                         st.error("❌ Image loading failed unexpectedly after processing. The resulting image data might be invalid.")
                    # Clear potentially partially set state on failure
                    st.session_state.uploaded_file_info = None; st.session_state.raw_image_bytes = None; st.session_state.display_image = None; st.session_state.processed_image = None; st.session_state.dicom_dataset = None; st.session_state.dicom_metadata = {}; st.session_state.current_display_wc = None; st.session_state.current_display_ww = None; st.session_state.is_dicom = False
                    # Don't rerun here, let the error message be displayed

    st.markdown("---")
    # --- DICOM W/L Controls ---
    if st.session_state.is_dicom and st.session_state.dicom_dataset and st.session_state.display_image: # W/L Controls
        with st.expander("DICOM Window/Level", expanded=False):
            try:
                # Get current slider values based on state or defaults from dataset
                wc_current_slider, ww_current_slider = dicom_wl_sliders(
                    st.session_state.dicom_dataset,
                    st.session_state.dicom_metadata,
                    # Pass current state values back to sliders for initialization
                    initial_wc=st.session_state.get('current_display_wc'),
                    initial_ww=st.session_state.get('current_display_ww')
                )

                # Get the WC/WW values currently used for the displayed image
                wc_displayed = st.session_state.get('current_display_wc')
                ww_displayed = st.session_state.get('current_display_ww')

                # Check if sliders returned valid numbers and if they changed
                slider_values_valid = (wc_current_slider is not None and ww_current_slider is not None)
                displayed_values_exist = (wc_displayed is not None and ww_displayed is not None)

                # Determine if an update is needed
                update_needed = False
                if slider_values_valid:
                    if not displayed_values_exist:
                        # First time setting W/L for this image
                        update_needed = True
                        logger.debug("W/L Initializing display image.")
                    else:
                        # Check for significant change (avoid floating point noise)
                        wc_changed = abs(wc_current_slider - wc_displayed) > 1e-3
                        ww_changed = abs(ww_current_slider - ww_displayed) > 1e-3
                        if wc_changed or ww_changed:
                            update_needed = True
                            logger.info(f"W/L change detected: WC {wc_displayed:.1f} -> {wc_current_slider:.1f}, WW {ww_displayed:.1f} -> {ww_current_slider:.1f}")

                if update_needed:
                    logger.info(f"Applying W/L: WC={wc_current_slider:.1f}, WW={ww_current_slider:.1f}")
                    with st.spinner("Applying Window/Level..."):
                         new_display_image = dicom_to_image(st.session_state.dicom_dataset, wc_current_slider, ww_current_slider)
                         if new_display_image:
                              # Ensure the new display image is RGB
                              if new_display_image.mode != 'RGB':
                                   st.session_state.display_image = new_display_image.convert('RGB')
                              else:
                                   st.session_state.display_image = new_display_image
                              # Update the state to reflect the applied W/L values
                              st.session_state.current_display_wc = wc_current_slider
                              st.session_state.current_display_ww = ww_current_slider
                              logger.debug("W/L applied, rerunning.")
                              st.rerun()
                         else:
                              st.error("Failed to apply W/L settings (image generation failed).");
                              logger.error("dicom_to_image returned None when applying new W/L.")
                # else: logger.debug("W/L: No update needed.") # Can uncomment for verbose debugging

            except Exception as e: st.error(f"W/L control error: {e}"); logger.error(f"W/L slider/update error: {e}", exc_info=True)
        st.markdown("---")

    # --- AI Actions ---
    if isinstance(st.session_state.get("display_image"), Image.Image): # AI Actions available only if image is loaded
        st.subheader("AI Actions")
        if st.button("▶️ Run Initial Analysis", key="analyze_btn", help="Perform a general analysis of the image.", use_container_width=True):
            st.session_state.last_action = "analyze"
            st.rerun()

        st.markdown("---")
        st.subheader("❓ Ask AI Question")

        # ROI Display and Clear Button
        roi_coords = st.session_state.get("roi_coords")
        if roi_coords:
            rc = roi_coords; roi_summary = f"L:{rc['left']}, T:{rc['top']}, W:{rc['width']}, H:{rc['height']}"
            st.info(f"✅ ROI Selected: [{roi_summary}]")
            if st.button("❌ Clear ROI", key="clear_roi_btn", help="Remove the selected Region of Interest.", use_container_width=True):
                st.session_state.roi_coords = None
                st.session_state.canvas_drawing = None # Reset canvas state too
                logger.info("ROI cleared by button click.")
                st.rerun()
        else:
            st.caption("ℹ️ Optionally, draw a rectangle on the image to define a Region of Interest (ROI) for your question.")

        # Question Input and Ask Button
        question_input = st.text_area(
            "Ask about the image or ROI:",
            height=100,
            key="question_input_widget",
            placeholder="e.g., Are there any signs of fracture in the selected region?",
            label_visibility="collapsed"
        )

        # --- CORRECTED BUTTON LOGIC ---
        if st.button("💬 Ask AI", key="ask_btn", use_container_width=True):
            user_question = st.session_state.question_input_widget # Get value from state
            if user_question and user_question.strip():
                # If question is valid, set action and rerun
                st.session_state.last_action = "ask"
                logger.info(f"Ask AI button clicked with question: '{user_question[:50]}...'")
                st.rerun()
            else:
                # If question is empty/whitespace, show warning
                st.warning("Please enter a question before clicking 'Ask AI'.")
                logger.warning("Ask AI button clicked with empty question.")
        # --- END CORRECTION ---

        st.markdown("---")
        st.subheader("🎯 Focused Condition Analysis")
        # Consider making disease list configurable or longer
        DISEASE_OPTIONS = ["Pneumonia", "Lung Cancer", "Stroke", "Fracture", "Appendicitis", "Tuberculosis", "COVID-19", "Pulmonary Embolism", "Brain Tumor", "Arthritis", "Osteoporosis", "Cardiomegaly", "Aortic Aneurysm", "Bowel Obstruction", "Mass/Nodule", "Effusion"] # Added more common ones
        disease_options_sorted = [""] + sorted(DISEASE_OPTIONS) # Add blank option for placeholder
        disease_select = st.selectbox(
            "Select a specific condition to analyze for:",
            options=disease_options_sorted,
            key="disease_select_widget",
            help="Choose a condition for the AI to focus its analysis on."
        )

        if st.button("🩺 Run Condition Analysis", key="disease_btn", use_container_width=True):
            selected_disease = st.session_state.disease_select_widget # Get value from state
            if selected_disease:
                st.session_state.last_action = "disease"
                logger.info(f"Condition Analysis button clicked for: '{selected_disease}'")
                st.rerun()
            else:
                st.warning("Please select a condition from the dropdown list first.")
                logger.warning("Condition Analysis button clicked without selecting a disease.")

        st.markdown("---")
        # --- Confidence & Report ---
        with st.expander("📊 Confidence & Report", expanded=True):
            # Enable confidence estimation if there's *any* prior AI output
            can_estimate = bool(
                st.session_state.get("history") or
                st.session_state.get("initial_analysis") or
                st.session_state.get("disease_analysis")
            )
            if st.button("📈 Estimate Confidence", key="confidence_btn", help="Estimate the AI's confidence based on recent interactions.", disabled=not can_estimate, use_container_width=True):
                st.session_state.last_action = "confidence"
                logger.info("Estimate Confidence button clicked.")
                st.rerun()
            if not can_estimate:
                st.caption("Run an initial analysis, ask a question, or run condition analysis first to enable confidence estimation.")

            # Generate PDF Data Button (separate from download)
            if st.button("📄 Generate PDF Data", key="generate_report_data_btn", help="Prepare the data for the PDF report. Download button will appear below.", use_container_width=True):
                st.session_state.last_action = "generate_report_data"
                logger.info("Generate PDF Data button clicked.")
                st.rerun()

            # Download Button (appears only after data is generated)
            if st.session_state.get("pdf_report_bytes"):
                report_filename = f"RadVisionAI_Report_{st.session_state.session_id or 'session'}.pdf"
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=st.session_state.pdf_report_bytes,
                    file_name=report_filename,
                    mime="application/pdf",
                    key="download_pdf_button",
                    help=f"Download the generated PDF report ({report_filename})",
                    use_container_width=True
                )
                # Optional: Add a success message or clear the bytes after download if desired
                # st.success("PDF report data ready for download.")
            elif st.session_state.get("last_action") == "generate_report_data":
                 # If the action was just run but failed, this won't show, handled by error messages
                 # If it succeeded, the download button appears above on the *next* rerun triggered by the action handler
                 pass # Avoid showing "Click generate" right after clicking it

    else: # No image loaded yet
        st.info("👈 Upload an image file using the panel on the left to begin analysis.")


# =============================================================================
# === MAIN PANEL DISPLAYS =====================================================
# =============================================================================
col1, col2 = st.columns([2, 3]) # Adjust ratio if needed (e.g., [5, 4] for larger image)

# --- Column 1: Image Viewer, Canvas, Metadata ---
with col1:
    st.subheader("🖼️ Image Viewer")
    display_img_object = st.session_state.get("display_image")

    # Check if the display_image object exists and is a PIL Image
    if isinstance(display_img_object, Image.Image):
        logger.debug(f"Viewer: Preparing display. Image details - Type: {type(display_img_object)}, Mode: {display_img_object.mode}, Size: {display_img_object.size}")

        # Use the already prepared RGB display_image directly for the canvas background
        bg_image_pil = display_img_object
        if bg_image_pil.mode != 'RGB':
            logger.warning(f"Viewer: Display image in state is not RGB ({bg_image_pil.mode}), attempting conversion for canvas.")
            try:
                bg_image_pil = bg_image_pil.convert('RGB')
            except Exception as convert_err:
                 st.error(f"Failed to convert image to RGB for canvas: {convert_err}")
                 logger.error(f"Canvas Prep: Failed to convert {display_img_object.mode} to RGB: {convert_err}", exc_info=True)
                 bg_image_pil = None # Prevent using unconverted image

        # Proceed only if background image preparation was successful AND it's a valid Image
        if isinstance(bg_image_pil, Image.Image): # Re-check after potential conversion
            # --- Calculate Canvas Dimensions ---
            # Make canvas slightly larger relative to the column width if possible
            MAX_CANVAS_WIDTH = 700 # Increased max width
            MAX_CANVAS_HEIGHT = 600 # Increased max height
            img_w, img_h = bg_image_pil.size
            aspect_ratio = img_w / img_h if img_h > 0 else 1

            # Calculate width based on container, respecting image aspect ratio and max dimensions
            # Use a hypothetical container width (Streamlit doesn't expose this easily) or fixed max
            container_width = MAX_CANVAS_WIDTH # Assume max width is the constraint initially
            canvas_width = min(img_w, container_width)
            canvas_height = int(canvas_width / aspect_ratio) if aspect_ratio > 0 else MAX_CANVAS_HEIGHT

            # If calculated height exceeds max height, recalculate width based on max height
            if canvas_height > MAX_CANVAS_HEIGHT:
                canvas_height = MAX_CANVAS_HEIGHT
                canvas_width = int(canvas_height * aspect_ratio)

            # Ensure minimum dimensions
            canvas_width = max(int(canvas_width), 150)
            canvas_height = max(int(canvas_height), 150)

            logger.info(f"Canvas Prep: Image size={img_w}x{img_h}, Aspect Ratio={aspect_ratio:.2f}. Calculated canvas size W={canvas_width}, H={canvas_height}")

            # --- Drawable Canvas ---
            if canvas_width > 0 and canvas_height > 0:
                # Use st.container to potentially better control layout/sizing if needed
                # with st.container():
                st.caption("Click and drag on the image below to select a Region of Interest (ROI). The last drawn rectangle is used.")
                try:
                    # Retrieve the last known drawing state to maintain the rectangle visually
                    initial_drawing = st.session_state.get("canvas_drawing") # This should be JSON
                    # Add safety check: ensure initial_drawing is None or dict-like
                    if initial_drawing and not isinstance(initial_drawing, dict):
                        logger.warning(f"Canvas: Invalid initial_drawing type ({type(initial_drawing)}), resetting.")
                        initial_drawing = None
                        st.session_state.canvas_drawing = None # Clear bad state

                    # logger.info(f"Rendering st_canvas. BG img type: {type(bg_image_pil)}, mode: {bg_image_pil.mode}, size: {bg_image_pil.size}. Initial drawing: {'Set' if initial_drawing else 'None'}")
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.2)",  # Semi-transparent orange fill
                        stroke_width=2,                       # Border width
                        stroke_color="rgba(220, 50, 50, 0.9)", # Red border
                        background_image=bg_image_pil,        # Use the prepared RGB image
                        update_streamlit=True,                # Send updates back to Streamlit
                        height=canvas_height,                 # Use calculated height
                        width=canvas_width,                   # Use calculated width
                        drawing_mode="rect",                  # Allow drawing rectangles
                        initial_drawing=initial_drawing,      # Load previous drawing if available
                        key="drawable_canvas",                # Unique key for the widget
                    )
                    # logger.info("st_canvas rendered (or attempted).")

                    # --- ROI Processing Logic ---
                    # Process canvas result ONLY IF it's not None and has updated data
                    if canvas_result is not None and canvas_result.json_data is not None:
                        # Store the latest drawing state regardless of objects (clears if empty)
                        st.session_state.canvas_drawing = canvas_result.json_data

                        if canvas_result.json_data.get("objects"):
                            # logger.debug(f"Canvas updated. Raw JSON data: {canvas_result.json_data}")
                            # Always use the LAST object drawn as the ROI
                            last_object = canvas_result.json_data["objects"][-1]
                            if last_object["type"] == "rect":
                                 # Calculate scaling factors from canvas size to original image size
                                 scale_x = img_w / canvas_width
                                 scale_y = img_h / canvas_height

                                 # Extract raw coordinates and dimensions from canvas object
                                 # Note: st_canvas applies scaling internally, so 'width'/'height' might already be scaled. Check documentation/behavior.
                                 # Assuming width/height need scaling by object's scaleX/scaleY
                                 raw_left = int(last_object["left"])
                                 raw_top = int(last_object["top"])
                                 # Use .get() with default 1 for scaleX/scaleY for robustness
                                 raw_width = int(last_object["width"] * last_object.get("scaleX", 1))
                                 raw_height = int(last_object["height"] * last_object.get("scaleY", 1))

                                 # Scale coordinates back to the original image dimensions
                                 orig_left = max(0, int(raw_left * scale_x))
                                 orig_top = max(0, int(raw_top * scale_y))
                                 orig_width = int(raw_width * scale_x)
                                 orig_height = int(raw_height * scale_y)

                                 # Ensure width/height are positive and coords are within image bounds
                                 orig_right = min(img_w, orig_left + orig_width)
                                 orig_bottom = min(img_h, orig_top + orig_height)
                                 orig_width = max(0, orig_right - orig_left)
                                 orig_height = max(0, orig_bottom - orig_top)

                                 # logger.debug(f"Canvas Rect: L={raw_left}, T={raw_top}, W={raw_width}, H={raw_height} | Scaled ROI: L={orig_left}, T={orig_top}, W={orig_width}, H={orig_height}")

                                 # Define a minimum ROI size to be considered valid (e.g., 10x10 pixels)
                                 MIN_ROI_DIM = 10
                                 if orig_width >= MIN_ROI_DIM and orig_height >= MIN_ROI_DIM:
                                     new_roi = {"left": orig_left, "top": orig_top, "width": orig_width, "height": orig_height}
                                     # Update state and rerun ONLY if the ROI coordinates have actually changed
                                     if st.session_state.roi_coords != new_roi:
                                         st.session_state.roi_coords = new_roi
                                         logger.info(f"ROI coordinates updated from canvas: {new_roi}")
                                         st.rerun() # Rerun to update the sidebar ROI info text
                                 # else: logger.debug("Scaled ROI rect too small or invalid.") # No need to log excessively

                        # If there are no objects, but there was an ROI before, clear it
                        elif not canvas_result.json_data.get("objects") and st.session_state.roi_coords is not None:
                                logger.info("Canvas cleared, removing ROI coordinates.")
                                st.session_state.roi_coords = None
                                # No need to rerun here unless the sidebar info needs explicit clearing

                except Exception as canvas_error:
                    st.error(f"Error rendering or processing the drawing canvas: {canvas_error}")
                    logger.error(f"st_canvas interaction failed: {canvas_error}", exc_info=True)
                    # Check browser console for frontend errors
                    st.warning("The drawing feature encountered an error and may be unavailable. Check the Browser's Developer Console (usually F12) for potential frontend issues (e.g., related to WebGL or image loading).")
            else:
                st.error("Calculated canvas dimensions are invalid (width or height <= 0). Cannot display drawing canvas.")
                logger.error(f"Invalid canvas dimensions prevented rendering: W={canvas_width}, H={canvas_height}")
        else:
             # Error message for invalid bg_image_pil already shown above if conversion failed
             st.info("Image could not be prepared for the drawing canvas.")
             logger.error("Cannot display canvas because the background image object is invalid or failed conversion.")

        # --- DICOM Metadata Display ---
        if st.session_state.is_dicom and st.session_state.dicom_metadata:
             # Use the imported function from ui_components
             display_dicom_metadata(st.session_state.dicom_metadata)

    # --- Fallback if display_image is NOT valid ---
    else:
        # This block displays a placeholder if no valid image is in the session state
        logger.debug(f"Viewer: No valid display_image in session state (Type: {type(display_img_object)}). Displaying placeholder.")
        st.markdown("---")
        if st.session_state.get("uploaded_file_info"):
            # If a file was uploaded but processing failed
            st.warning("Image processing may have failed or the file might be unsupported. Please check the file or try uploading again.")
        else:
            # If no file has been uploaded yet
            st.info("The uploaded image will appear here.")
        # Simple placeholder div
        st.markdown("<div style='height: 400px; border: 2px dashed #ccc; display: flex; align-items: center; justify-content: center; text-align: center; color: #aaa; font-style: italic;'>Image Display Area<br/>(Upload Required)</div>", unsafe_allow_html=True)


# --- Column 2: Analysis Results Tabs ---
# ... (Code from lines 467 to 511 - Keep as before, seems okay) ...
with col2:
    st.subheader("📊 Analysis & Results")

    # Define tab labels
    tab_titles = ["🔬 Initial Analysis", "💬 Q&A History", "🩺 Disease Focus", "📈 Confidence"]
    tabs = st.tabs(tab_titles)

    # --- Initial Analysis Tab ---
    with tabs[0]:
        initial_analysis_content = st.session_state.get("initial_analysis", "").strip()
        if not initial_analysis_content:
             initial_analysis_display = "No initial analysis has been performed yet. Click 'Run Initial Analysis' in the sidebar."
        else:
             initial_analysis_display = initial_analysis_content
        st.text_area(
             "Overall Findings & Impressions",
             value=initial_analysis_display,
             height=450, # Adjust height as needed
             key="output_initial_text",
             disabled=True, # Make read-only
             help="Shows the general analysis results provided by the AI."
        )

    # --- Q&A History Tab ---
    with tabs[1]:
        qa_answer_content = st.session_state.get("qa_answer", "").strip()
        if not qa_answer_content and not st.session_state.history:
            qa_answer_display = "Ask a question using the input in the sidebar. The AI's answer will appear here."
        elif not qa_answer_content and st.session_state.history:
            qa_answer_display = "Most recent answer will appear here after you ask a new question."
        else:
            qa_answer_display = qa_answer_content
        st.text_area(
            "AI Answer to Last Question",
            value=qa_answer_display,
            height=200, # Shorter height for the latest answer
            key="output_qa_text",
            disabled=True,
            help="Displays the AI's most recent answer to your question."
        )
        st.markdown("---")

        # Display full conversation history in an expander
        if st.session_state.history:
             with st.expander("View Full Conversation History", expanded=True):
                 # Display history in chronological order (oldest first)
                 for i, (q, a) in enumerate(st.session_state.history):
                     st.markdown(f"**You ({i+1}):**")
                     st.caption(q) # Use caption for questions for visual distinction
                     st.markdown(f"**AI ({i+1}):**")
                     # Allow markdown formatting in answers (e.g., lists, bold)
                     st.markdown(a, unsafe_allow_html=True) # Be cautious with unsafe_allow_html if AI output isn't trusted
                     if i < len(st.session_state.history) - 1:
                         st.markdown("---") # Separator between Q&A pairs
        else:
             st.caption("No conversation history yet.")

    # --- Disease Focus Tab ---
    with tabs[2]:
        disease_analysis_content = st.session_state.get("disease_analysis", "").strip()
        if not disease_analysis_content:
             disease_analysis_display = "No focused condition analysis has been performed yet. Select a condition and click 'Run Condition Analysis' in the sidebar."
        else:
             disease_analysis_display = disease_analysis_content
        st.text_area(
            "Disease-Specific Findings",
            value=disease_analysis_display,
            height=450,
            key="output_disease_text",
            disabled=True,
            help="Shows the AI's analysis focused on the selected medical condition."
        )

    # --- Confidence Tab ---
    with tabs[3]:
        confidence_score_content = st.session_state.get("confidence_score", "").strip()
        if not confidence_score_content:
             confidence_score_display = "No confidence estimation has been performed yet. Run analysis or Q&A, then click 'Estimate Confidence' in the sidebar."
        else:
             confidence_score_display = confidence_score_content
        st.text_area(
            "AI Confidence Estimation",
            value=confidence_score_display,
            height=450,
            key="output_confidence_text",
            disabled=True,
            help="Displays the AI's estimated confidence in its recent analysis or answers. This is an experimental feature."
        )

# =============================================================================
# === ACTION HANDLING LOGIC ===================================================
# =============================================================================
# ... (Code from lines 514 to 684 - Keep full rewritten block as before, seems okay) ...
# This block processes actions triggered by button clicks (set in `last_action`)

current_action: Optional[str] = st.session_state.get("last_action")

# Only proceed if an action was actually set
if current_action:
    logger.info(f"ACTION HANDLER: Initiating action '{current_action}'")

    # --- Pre-Action Checks ---
    processed_image = st.session_state.get("processed_image")
    session_id = st.session_state.get("session_id")

    # Most actions require a processed image
    if current_action not in ["generate_report_data"]: # Report generation uses display_image
        if not isinstance(processed_image, Image.Image):
            error_msg = f"Cannot perform action '{current_action}': The processed image is missing or invalid. Please ensure an image is successfully uploaded and processed."
            st.error(error_msg)
            processed_image_type = type(processed_image).__name__ if processed_image else "None"
            logger.error(f"Action '{current_action}' aborted. Required 'processed_image' is missing or invalid (Type: {processed_image_type}). Session ID: {session_id}")
            st.session_state.last_action = None # Clear the invalid action
            st.stop() # Stop further execution in this script run

    # All actions require a session ID for context/logging
    if not session_id:
        error_msg = f"Cannot perform action '{current_action}': Session ID is missing. This might indicate an issue with application initialization."
        st.error(error_msg)
        logger.error(f"Action '{current_action}' aborted. Session ID is missing.")
        st.session_state.last_action = None # Clear the invalid action
        st.stop()

    # --- Prepare Common Variables ---
    img_for_llm: Image.Image = processed_image # Use the 'processed' version for AI consistency
    roi: Optional[Dict[str, int]] = st.session_state.get("roi_coords")
    roi_context_str = " (focused on defined ROI)" if roi else ""
    history: List[Tuple[str, str]] = st.session_state.get("history", [])
    if not isinstance(history, list): # Ensure history is a list
        logger.warning("Session state 'history' was not a list, resetting to empty list.")
        history = []
        st.session_state.history = history

    # --- Execute Action ---
    try:
        # Clear previous results of *other* actions for clarity (optional, depends on desired UI behavior)
        # If you want results to persist across actions, comment these out.
        # if current_action != "ask": st.session_state.qa_answer = "" # Keep last answer unless asking new Q
        # if current_action != "analyze": st.session_state.initial_analysis = ""
        # if current_action != "disease": st.session_state.disease_analysis = ""
        # if current_action != "confidence": st.session_state.confidence_score = ""
        # Clear report bytes unless specifically generating them
        # if current_action != "generate_report_data": st.session_state.pdf_report_bytes = None

        # --- ANALYZE Action ---
        if current_action == "analyze":
            st.info(f"🔬 Performing preliminary analysis{roi_context_str}...")
            with st.spinner("AI is analyzing the image... Please wait."):
                # Pass ROI to analysis function if applicable (modify function signature if needed)
                # analysis_result = run_initial_analysis(img_for_llm, roi=roi) # Example if function supports ROI
                analysis_result = run_initial_analysis(img_for_llm) # Assuming it doesn't use ROI for now
            st.session_state.initial_analysis = analysis_result
            logger.info(f"Action 'analyze' completed successfully{roi_context_str}.")
            st.success("Initial analysis finished.")

        # --- ASK Action ---
        elif current_action == "ask":
            question = st.session_state.get("question_input_widget", "").strip()
            if not question:
                st.warning("Question input was empty. Please type a question in the sidebar.")
                logger.warning("Action 'ask' skipped: question was empty.")
            else:
                st.info(f"❓ Asking AI: \"{question[:60]}...\" {roi_context_str}")
                st.session_state.qa_answer = "" # Clear previous answer before getting new one
                primary_model_name = "Gemini" # Or your primary model's name
                with st.spinner(f"{primary_model_name} is thinking about your question..."):
                    # Pass image, question, history, and ROI to the QA function
                    gemini_answer, success = run_multimodal_qa(img_for_llm, question, history, roi)

                if success:
                    st.session_state.qa_answer = gemini_answer
                    # Append the new Q&A pair to history
                    st.session_state.history.append((question, gemini_answer))
                    logger.info(f"Action 'ask' successful for question: '{question[:60]}...'{roi_context_str}")
                    st.success(f"{primary_model_name} answered your question.")
                    # Clear the text input widget after successful question
                    # st.session_state.question_input_widget = "" # Optional: clear input field
                else:
                    # Handle primary AI failure
                    error_message = f"Primary AI ({primary_model_name}) failed to answer. Reason: {gemini_answer}" # Gemini_answer contains error msg on failure
                    st.error(error_message)
                    logger.warning(f"Action 'ask' failed (Primary AI: {primary_model_name}). Q: '{question[:60]}...'. Reason: {gemini_answer}")
                    st.session_state.qa_answer = f"**[Primary AI Error]** {gemini_answer}\n\n---\n" # Display error in the answer box

                    # --- Fallback Logic (Example using Hugging Face) ---
                    hf_token = bool(os.environ.get("HF_API_TOKEN"))
                    # Check if model ID is valid and module was imported
                    hf_model_available = (HF_VQA_MODEL_ID and HF_VQA_MODEL_ID not in ["hf_model_not_found", "unavailable"] and 'query_hf_vqa_inference_api' in globals())

                    if hf_token and hf_model_available:
                        st.info(f"Attempting fallback using Hugging Face model ({HF_VQA_MODEL_ID})...")
                        with st.spinner(f"Fallback AI ({HF_VQA_MODEL_ID}) is thinking..."):
                            hf_answer, hf_success = query_hf_vqa_inference_api(img_for_llm, question, roi)

                        if hf_success:
                            fallback_display = f"**[Fallback Answer ({HF_VQA_MODEL_ID})]**\n\n{hf_answer}"
                            st.session_state.qa_answer += fallback_display # Append fallback answer
                            # Also add fallback answer to history, clearly marked
                            st.session_state.history.append((f"[Fallback Attempt] {question}", fallback_display))
                            logger.info(f"Action 'ask': HF fallback successful for Q: '{question[:60]}...'")
                            st.success("Fallback AI provided an answer.")
                        else:
                            fallback_error = f"Fallback AI ({HF_VQA_MODEL_ID}) also failed. Reason: {hf_answer}"
                            st.session_state.qa_answer += f"**[Fallback Failed]** {fallback_error}" # Append fallback failure
                            st.error(fallback_error)
                            logger.error(f"Action 'ask': HF fallback failed for Q: '{question[:60]}...'. Reason: {hf_answer}")
                    else:
                        missing_details = []
                        if not hf_token: missing_details.append("Hugging Face API Token")
                        if not hf_model_available: missing_details.append(f"Hugging Face Model ({HF_VQA_MODEL_ID}) or function")
                        fallback_msg = f"Fallback AI is unavailable (Missing: {', '.join(missing_details)})."
                        st.session_state.qa_answer += f"**[Fallback Unavailable]** {fallback_msg}" # Append unavailability message
                        st.warning(fallback_msg)
                        logger.warning(f"Action 'ask': HF fallback skipped for Q: '{question[:60]}...'. Missing: {missing_details}")
                    # --- End Fallback Logic ---

        # --- DISEASE Action ---
        elif current_action == "disease":
            selected_disease = st.session_state.get("disease_select_widget")
            if not selected_disease:
                st.warning("No condition was selected. Please choose a condition from the list in the sidebar.")
                logger.warning("Action 'disease' skipped: no disease selected.")
            else:
                st.info(f"🩺 Analyzing image specifically for signs of '{selected_disease}'{roi_context_str}...")
                with st.spinner(f"AI is assessing '{selected_disease}'..."):
                    # Pass image, disease, and ROI to the analysis function
                    disease_result = run_disease_analysis(img_for_llm, selected_disease, roi)
                st.session_state.disease_analysis = disease_result
                logger.info(f"Action 'disease' completed for condition: '{selected_disease}'{roi_context_str}.")
                st.success(f"Analysis for '{selected_disease}' finished.")

        # --- CONFIDENCE Action ---
        elif current_action == "confidence":
            st.info(f"📊 Estimating AI confidence based on recent interactions{roi_context_str}...")
            # Confidence estimation might need the image, history, and potentially other results
            # Check if there's enough context to estimate confidence
            if not history and not st.session_state.get("initial_analysis") and not st.session_state.get("disease_analysis"):
                 st.warning("Cannot estimate confidence: No prior analysis or Q&A found in this session.")
                 logger.warning("Action 'confidence' skipped: No history or prior analysis results available.")
            else:
                 with st.spinner("Calculating confidence score..."):
                    # Pass relevant context to the confidence function
                    confidence_result = estimate_ai_confidence(
                        img_for_llm,
                        history=history,
                        initial_analysis=st.session_state.get("initial_analysis"),
                        disease_analysis=st.session_state.get("disease_analysis"),
                        roi=roi # Pass ROI if the function uses it
                    )
                 st.session_state.confidence_score = confidence_result
                 logger.info("Action 'confidence' estimation completed.")
                 st.success("Confidence estimation finished.")

        # --- GENERATE REPORT DATA Action ---
        elif current_action == "generate_report_data":
            st.info("📄 Preparing data for the PDF report...")
            st.session_state.pdf_report_bytes = None # Clear any old report data

            # Use the 'display_image' for the report, as it includes W/L adjustments
            img_for_report: Optional[Image.Image] = st.session_state.get("display_image")

            if not isinstance(img_for_report, Image.Image):
                 st.error("Cannot generate report: The display image is missing or invalid.")
                 logger.error("Action 'generate_report_data' failed: 'display_image' is invalid.")
            else:
                img_with_roi = None # Initialize variable for image potentially with ROI drawn
                # Check if ROI exists and is valid
                roi_coords = st.session_state.get("roi_coords")
                if roi_coords and isinstance(roi_coords, dict) and all(k in roi_coords for k in ['left', 'top', 'width', 'height']):
                    try:
                        # Create a copy to draw on, ensure it's RGB for drawing
                        img_copy = img_for_report.copy()
                        if img_copy.mode != 'RGB':
                             img_copy = img_copy.convert("RGB")
                        draw = ImageDraw.Draw(img_copy)

                        # Define ROI box coordinates, clamping to image boundaries
                        x0, y0 = int(roi_coords['left']), int(roi_coords['top'])
                        x1, y1 = x0 + int(roi_coords['width']), y0 + int(roi_coords['height'])
                        img_w, img_h = img_copy.size
                        x0 = max(0, min(x0, img_w - 1))
                        y0 = max(0, min(y0, img_h - 1))
                        x1 = max(0, min(x1, img_w)) # Right/bottom bounds can be equal to size
                        y1 = max(0, min(y1, img_h))

                        # Draw rectangle only if dimensions are valid after clamping
                        if x1 > x0 and y1 > y0:
                             draw.rectangle([x0, y0, x1, y1], outline="red", width=3) # Adjust color/width as needed
                             img_with_roi = img_copy
                             logger.info("Drew ROI rectangle on the image for the PDF report.")
                        else:
                             logger.warning("ROI dimensions became invalid after clamping to image boundaries. Not drawing on report image.")
                             img_with_roi = img_for_report # Use original image without drawing
                    except Exception as draw_err:
                         logger.error(f"Failed to draw ROI on report image: {draw_err}", exc_info=True)
                         st.warning(f"Could not draw the ROI on the report image due to an error: {draw_err}")
                         img_with_roi = img_for_report # Fallback to original image
                else:
                     # No ROI defined or invalid, use the original display image
                     img_with_roi = img_for_report

                # --- Gather Text Outputs for Report ---
                # Format Q&A history nicely
                qa_hist_str = ""
                if history:
                    qa_entries = []
                    for i, (q, a) in enumerate(history):
                        # Basic cleaning: replace multiple newlines, strip whitespace
                        q_clean = ' '.join(q.split())
                        a_clean = '\n'.join(line.strip() for line in a.split('\n') if line.strip()) # Keep structure but clean lines
                        qa_entries.append(f"Interaction {i+1}:\nQ: {q_clean}\n\nA:\n{a_clean}")
                    qa_hist_str = "\n\n---\n\n".join(qa_entries)
                else:
                    qa_hist_str = "No questions were asked in this session."

                # Consolidate all text sections
                report_outputs = {
                    "Session ID": session_id or "N/A",
                    "Preliminary Analysis": st.session_state.get("initial_analysis","Not Performed"),
                    "Conversation History": qa_hist_str,
                    "Condition-Specific Analysis": st.session_state.get("disease_analysis","Not Performed"),
                    "Last Confidence Estimation": st.session_state.get("confidence_score","Not Estimated")
                }

                # --- Add Filtered DICOM Metadata if available ---
                if st.session_state.get("is_dicom") and st.session_state.get("dicom_metadata"):
                    logger.info("Processing DICOM metadata for the report (filtering PHI).")
                    filtered_meta_list = []
                    # Expand this list based on common PHI tags
                    PHI_TAG_KEYWORDS = [
                        "Patient", "Person", "Address", "Telephone", "BirthDate", "Sex",
                        "Physician", "Operator", "Institution", "Issuer", "Record",
                        "UID", # Exclude all UIDs by default for caution
                        "Date", "Time", # Exclude specific dates/times that might be identifying
                        "Accession", "Referring", "Requesting", "Performing", "Scheduled",
                    ]
                    # Tags to explicitly ALLOW if they don't contain obvious PHI keywords
                    ALLOWED_TAGS = {"Modality", "StudyDescription", "SeriesDescription", "ProtocolName", "BodyPartExamined", "PixelSpacing", "SliceThickness", "WindowCenter", "WindowWidth", "Manufacturer", "ManufacturerModelName", "SoftwareVersions"}

                    meta_dict = st.session_state.dicom_metadata
                    for key, value in meta_dict.items():
                         # Skip if key is in allowed list (already checked)
                         if key in ALLOWED_TAGS:
                             pass
                         # Skip if any PHI keyword is part of the tag name (case-insensitive)
                         elif any(keyword.lower() in key.lower() for keyword in PHI_TAG_KEYWORDS):
                             logger.debug(f"Excluding potentially identifying DICOM tag from report: {key}")
                             continue

                         # If not explicitly allowed or excluded by keyword, proceed to format
                         display_value = ""
                         try:
                             if isinstance(value, list): display_value = ", ".join(map(str, value))
                             elif isinstance(value, pydicom.uid.UID): display_value = value.name # Show name not number
                             elif isinstance(value, bytes): display_value = f"<{len(value)} bytes of binary data>"
                             elif isinstance(value, pydicom.valuerep.PersonName): display_value = "[Person Name Redacted]" # Explicitly redact PersonName objects
                             else: display_value = str(value).strip()

                             # Limit value length and skip empty values
                             MAX_VALUE_LEN = 100
                             if display_value and len(display_value) < MAX_VALUE_LEN:
                                  filtered_meta_list.append(f"{key}: {display_value}")
                             elif len(display_value) >= MAX_VALUE_LEN:
                                 filtered_meta_list.append(f"{key}: {display_value[:MAX_VALUE_LEN]}...")


                         except Exception as tag_proc_err:
                             logger.warning(f"Error processing DICOM tag '{key}' for report: {tag_proc_err}")
                             filtered_meta_list.append(f"{key}: [Error processing value]")

                    if filtered_meta_list:
                         report_outputs["DICOM Metadata (Filtered)"] = "\n".join(filtered_meta_list)
                         logger.info(f"Included {len(filtered_meta_list)} filtered DICOM tags in report data.")
                    else:
                         report_outputs["DICOM Metadata (Filtered)"] = "N/A or all tags filtered."

                # --- Generate the PDF Bytes ---
                with st.spinner("🎨 Generating PDF document..."):
                     pdf_bytes = generate_pdf_report_bytes(
                         session_id,
                         img_with_roi, # Pass the image (potentially with ROI drawn)
                         report_outputs # Pass the dictionary of text outputs
                     )

                if pdf_bytes:
                     st.session_state.pdf_report_bytes = pdf_bytes
                     st.success("✅ PDF report data generated successfully! Click the download button below.")
                     logger.info("Action 'generate_report_data' completed successfully.")
                else:
                     st.error("❌ PDF generation failed. Please check the application logs for errors.")
                     logger.error("Action 'generate_report_data' failed: 'generate_pdf_report_bytes' returned None or empty.")


        # --- Unknown Action ---
        else:
            st.warning(f"An unknown action '{current_action}' was triggered. Please report this issue.")
            logger.warning(f"Unknown action '{current_action}' encountered in action handler.")

    # --- Error Handling for Actions ---
    except Exception as action_err:
        st.error(f"An unexpected error occurred while performing action '{current_action}': {action_err}")
        logger.critical(f"Unhandled error during action '{current_action}': {action_err}", exc_info=True)

    # --- Post-Action Cleanup ---
    finally:
        # ALWAYS clear the last_action flag to prevent re-execution on next rerun
        st.session_state.last_action = None
        logger.debug(f"Action handler finished processing '{current_action}'. Cleared last_action flag.")
        # Trigger a final rerun AFTER the action is fully processed (success or fail)
        # This ensures the UI updates reflect the results (text areas, download button, etc.)
        st.rerun()


# --- Footer ---
# ... (Code from line 687 - Keep as before) ...
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced | Session ID: {st.session_state.get('session_id', 'N/A')}")
# Add version number or other info if desired
# APP_VERSION = "1.1.0"
# st.caption(f"⚕️ RadVision AI Advanced v{APP_VERSION} | Session ID: {st.session_state.get('session_id', 'N/A')}")