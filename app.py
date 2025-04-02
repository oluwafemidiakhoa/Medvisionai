# main_app.py (Example Name)

# --- Core Libraries ---
import io
import os
import uuid
import logging
import base64
from typing import Any, Dict, Optional, Tuple, List # Ensure all needed types are here

# --- Streamlit ---
import streamlit as st # Streamlit import first
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

    # --- 1) Upload Image ---
    uploaded_file = st.file_uploader(
        "Upload Image (JPG, PNG, DCM)",
        type=["jpg", "jpeg", "png", "dcm", "dicom"], # Allow various extensions/types
        key="file_uploader_widget", # Use a distinct key for the widget itself
        accept_multiple_files=False,
        help="Select a standard image (JPG, PNG) or a DICOM file (.dcm). Max file size configured by Streamlit deployment."
    )

    # --- Process Uploaded File (Only if a new file is uploaded) ---
    if uploaded_file is not None:
        # Create a unique identifier for the uploaded file instance
        # Using name, size, and last modified time (if available) for better detection
        try:
             last_modified = getattr(uploaded_file, 'last_modified', '') # Robust access
             new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{last_modified}"
        except Exception: # Fallback if attributes change
             new_file_info = f"{uploaded_file.name}-{uploaded_file.size}"

        # Check if this file is different from the one currently in session state
        if new_file_info != st.session_state.get("uploaded_file_info"):
            logger.info(f"New file upload detected: {uploaded_file.name} ({uploaded_file.type}, {uploaded_file.size} bytes)")
            st.toast(f"Processing '{uploaded_file.name}'...", icon="⏳")

            # --- Reset Session State for New File ---
            # Preserve keys that shouldn't be reset (like widget keys if needed)
            preserve_keys = {"file_uploader_widget"} # Add other keys if necessary
            logger.debug("Resetting session state for new file upload...")
            for key, default_value in DEFAULT_STATE.items():
                if key not in preserve_keys:
                    st.session_state[key] = default_value
            # Ensure history is explicitly reset to an empty list
            st.session_state.history = []
            logger.debug("Session state reset.")

            # Store info about the new file
            st.session_state.uploaded_file_info = new_file_info
            st.session_state.session_id = str(uuid.uuid4())[:8] # New session ID
            logger.info(f"Generated new Session ID: {st.session_state.session_id}")

            # --- Read and Process File Bytes ---
            with st.spinner("🔬 Processing image... Please wait."):
                st.session_state.raw_image_bytes = uploaded_file.getvalue()
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()

                # Determine if it's DICOM based on extension or MIME type
                st.session_state.is_dicom = file_ext in (".dcm", ".dicom") or "dicom" in uploaded_file.type.lower()
                logger.info(f"File '{uploaded_file.name}' identified as DICOM: {st.session_state.is_dicom}")

                processing_successful = False
                temp_display_image = None
                temp_processed_image = None

                # --- DICOM Processing Branch ---
                if st.session_state.is_dicom:
                    try:
                        logger.debug("Attempting to parse DICOM...")
                        # Use the refined parse_dicom function
                        ds = parse_dicom(st.session_state.raw_image_bytes, uploaded_file.name)
                        st.session_state.dicom_dataset = ds # Store dataset or None

                        if ds:
                            logger.info("DICOM parsed successfully.")
                            # Extract technical metadata using the helper
                            st.session_state.dicom_metadata = extract_dicom_metadata(ds)
                            logger.debug(f"Extracted {len(st.session_state.dicom_metadata)} metadata tags.")

                            # Get default W/L (helper handles missing tags)
                            wc_default, ww_default = get_default_wl(ds)
                            st.session_state.current_display_wc = wc_default # Store initial W/L used
                            st.session_state.current_display_ww = ww_default # Store initial W/L used
                            logger.info(f"Default W/L from DICOM tags: WC={wc_default}, WW={ww_default}")

                            # Generate display image using default W/L
                            logger.debug("Generating display image from DICOM using default W/L...")
                            temp_display_image = dicom_to_image(ds, wc_default, ww_default)

                            # Generate processed image (e.g., for AI) - using basic scaling
                            logger.debug("Generating processed image from DICOM using basic scaling...")
                            temp_processed_image = dicom_to_image(ds, window_center=None, window_width=None)

                            if temp_display_image and temp_processed_image:
                                logger.info("DICOM images (display & processed) generated successfully.")
                                processing_successful = True
                            else:
                                st.error("DICOM processing failed: Could not generate required image objects from pixel data.")
                                logger.error("dicom_to_image returned None for display and/or processed image.")
                        else:
                            logger.error("parse_dicom returned None.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during DICOM processing: {e}")
                        logger.critical(f"DICOM processing pipeline error: {e}", exc_info=True)
                        processing_successful = False

                # --- Standard Image Processing Branch ---
                else:
                    try:
                        logger.debug("Attempting to open standard image file...")
                        img = Image.open(io.BytesIO(st.session_state.raw_image_bytes))
                        temp_processed_image = img.copy()
                        if img.mode != 'RGB':
                            logger.info(f"Original image mode: {img.mode}. Converting to RGB for display.")
                            temp_display_image = img.convert("RGB")
                        else: temp_display_image = img
                        st.session_state.dicom_dataset = None
                        st.session_state.dicom_metadata = {}
                        st.session_state.current_display_wc = None # No W/L for standard images
                        st.session_state.current_display_ww = None
                        processing_successful = True
                        logger.info("Standard image opened and prepared for display.")
                    except UnidentifiedImageError:
                        st.error(f"Cannot identify image file format for '{uploaded_file.name}'. Please upload JPG, PNG, or DICOM.")
                        logger.error(f"UnidentifiedImageError for file: {uploaded_file.name}")
                        processing_successful = False
                    except Exception as e:
                        st.error(f"Error processing standard image file '{uploaded_file.name}': {e}")
                        logger.error(f"Standard image processing error: {e}", exc_info=True)
                        processing_successful = False

                # --- Final Check and State Update ---
                if processing_successful and isinstance(temp_display_image, Image.Image) and isinstance(temp_processed_image, Image.Image):
                    if temp_display_image.mode != 'RGB':
                         st.session_state.display_image = temp_display_image.convert('RGB')
                    else: st.session_state.display_image = temp_display_image
                    st.session_state.processed_image = temp_processed_image
                    logger.info(f"Image processing complete. Display: {st.session_state.display_image.mode} {st.session_state.display_image.size}, Processed: {st.session_state.processed_image.mode} {st.session_state.processed_image.size}")
                    st.session_state.roi_coords = None
                    st.session_state.canvas_drawing = None
                    st.success(f"✅ Image '{uploaded_file.name}' loaded successfully!")
                    st.rerun()
                else:
                    logger.critical("Image loading failed after processing step.")
                    if processing_successful:
                         st.error("❌ Image loading failed unexpectedly after processing. Please check logs.")
                    st.session_state.uploaded_file_info = None
                    st.session_state.raw_image_bytes = None
                    st.session_state.display_image = None
                    st.session_state.processed_image = None
                    st.session_state.dicom_dataset = None
                    st.session_state.dicom_metadata = {}
                    st.session_state.current_display_wc = None
                    st.session_state.current_display_ww = None

    st.markdown("---")

    # --- 2) DICOM Window/Level Controls ---
    if st.session_state.is_dicom and st.session_state.dicom_dataset and st.session_state.display_image:
        with st.expander("DICOM Window/Level", expanded=False):
            try:
                # Use helper for sliders; it reads dataset/metadata and returns current slider values
                wc_current_slider, ww_current_slider = dicom_wl_sliders(
                    st.session_state.dicom_dataset,
                    st.session_state.dicom_metadata
                )

                # Check if slider values changed compared to what's currently displayed
                wc_displayed = st.session_state.get('current_display_wc')
                ww_displayed = st.session_state.get('current_display_ww')

                # Handle None comparison carefully
                wc_changed = wc_displayed is None or abs(wc_current_slider - wc_displayed) > 1e-3
                ww_changed = ww_displayed is None or abs(ww_current_slider - ww_displayed) > 1e-3

                # Regenerate image only if W/L values *actually changed*
                # (Slider interaction itself causes a rerun, but we only recalculate if value differs)
                if (wc_changed or ww_changed) and wc_current_slider is not None and ww_current_slider is not None:
                    # Check against potential None values from the helper if dataset was bad
                    logger.info(f"W/L Sliders changed. Applying WC={wc_current_slider:.1f}, WW={ww_current_slider:.1f}")
                    with st.spinner("Applying Window/Level..."):
                         new_display_image = dicom_to_image(st.session_state.dicom_dataset, wc_current_slider, ww_current_slider)
                         if new_display_image:
                              if new_display_image.mode != 'RGB':
                                   st.session_state.display_image = new_display_image.convert('RGB')
                              else:
                                   st.session_state.display_image = new_display_image
                              # Update the state tracking the displayed W/L
                              st.session_state.current_display_wc = wc_current_slider
                              st.session_state.current_display_ww = ww_current_slider
                              st.rerun() # Update main panel image
                         else:
                              st.error("Failed to apply new Window/Level settings.")
                              logger.error("dicom_to_image returned None when applying new W/L.")
                # Initialize if first run after load
                elif wc_displayed is None and ww_displayed is None and wc_current_slider is not None and ww_current_slider is not None:
                      st.session_state.current_display_wc = wc_current_slider
                      st.session_state.current_display_ww = ww_current_slider
                      logger.debug("Initialized current display W/L state from initial slider values.")

            except Exception as e:
                 st.error(f"Error in W/L control logic: {e}")
                 logger.error(f"W/L slider integration error: {e}", exc_info=True)
        st.markdown("---")

    # --- 3) Analysis & Interaction Controls ---
    if isinstance(st.session_state.get("display_image"), Image.Image):
        st.subheader("AI Actions")
        # ... (Buttons and inputs as in the previous corrected version) ...
        if st.button("▶️ Run Initial Analysis", key="analyze_btn", help="Perform a general analysis of the entire image.", use_container_width=True):
            st.session_state.last_action = "analyze"; st.rerun()

        st.markdown("---")
        st.subheader("❓ Ask AI Question")

        if st.session_state.get("roi_coords"):
            rc = st.session_state.roi_coords
            roi_summary = f"L:{rc['left']}, T:{rc['top']}, W:{rc['width']}, H:{rc['height']}"
            st.info(f"✅ ROI Selected: [{roi_summary}]. Question will focus here.")
            if st.button("❌ Clear ROI", key="clear_roi_btn", help="Remove the highlighted region selection.", use_container_width=True):
                st.session_state.roi_coords = None
                st.session_state.canvas_drawing = None
                logger.info("ROI cleared by user.")
                st.rerun()
        else:
            st.caption("ℹ️ Optionally, draw a rectangle on the image (left panel) to focus your question on a specific region.")

        question_input = st.text_area(
            "Ask about the image or the highlighted region:",
            height=100, key="question_input_widget",
            placeholder="e.g., What type of scan is this? Are there any abnormalities in the selected region?",
            label_visibility="collapsed"
        )
        if st.button("💬 Ask AI", key="ask_btn", use_container_width=True):
            user_question = st.session_state.question_input_widget
            if user_question and user_question.strip():
                st.session_state.last_action = "ask"; st.rerun()
            else: st.warning("Please enter a question before asking the AI.")

        st.markdown("---")
        st.subheader("🎯 Focused Condition Analysis")

        DISEASE_OPTIONS = ["Pneumonia", "Lung Cancer", "Stroke (Ischemic/Hemorrhagic)", "Bone Fracture", "Appendicitis", "Tuberculosis", "COVID-19 Pneumonitis", "Pulmonary Embolism", "Brain Tumor (e.g., Glioblastoma, Meningioma)", "Arthritis Signs", "Osteoporosis Signs", "Cardiomegaly", "Aortic Aneurysm/Dissection Signs", "Bowel Obstruction Signs"]
        disease_options_sorted = [""] + sorted(DISEASE_OPTIONS)
        disease_select = st.selectbox("Select a specific condition to analyze for:", options=disease_options_sorted, key="disease_select_widget", help="The AI will look for findings relevant to the selected condition.")
        if st.button("🩺 Run Condition Analysis", key="disease_btn", use_container_width=True):
            selected_disease = st.session_state.disease_select_widget
            if selected_disease:
                st.session_state.last_action = "disease"; st.rerun()
            else: st.warning("Please select a condition from the dropdown first.")

        st.markdown("---")
        with st.expander("📊 Confidence & Report", expanded=True):
            can_estimate_confidence = bool(st.session_state.history or st.session_state.initial_analysis or st.session_state.disease_analysis)
            if st.button("📈 Estimate AI Confidence", key="confidence_btn", help="Estimate the AI's confidence based on the conversation history and image.", disabled=not can_estimate_confidence, use_container_width=True):
                st.session_state.last_action = "confidence"; st.rerun()
            if not can_estimate_confidence: st.caption("Perform an analysis or ask a question first to enable confidence estimation.")

            if st.button("📄 Generate PDF Report Data", key="generate_report_data_btn", help="Compile the session analysis into data ready for PDF download.", use_container_width=True):
                st.session_state.last_action = "generate_report_data"; st.rerun()

            if st.session_state.get("pdf_report_bytes"):
                report_filename = f"RadVisionAI_Report_{st.session_state.session_id}.pdf"
                st.download_button(label="⬇️ Download PDF Report", data=st.session_state.pdf_report_bytes, file_name=report_filename, mime="application/pdf", key="download_pdf_button", help=f"Download the generated report ({report_filename})", use_container_width=True)
                st.success("PDF report data ready for download.")

    else: st.info("👈 Please upload an image file (JPG, PNG, or DICOM) to begin analysis.")

# =============================================================================
# === MAIN PANEL DISPLAYS =====================================================
# =============================================================================
# ... (Main panel columns and displays - Canvas, Metadata, Tabs - remain the same as previous corrected version) ...
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("🖼️ Image Viewer")
    display_img_object = st.session_state.get("display_image")
    if isinstance(display_img_object, Image.Image):
        bg_image_pil = None
        try:
             if display_img_object.mode != 'RGB':
                 logger.debug(f"Canvas Prep: Converting display image from {display_img_object.mode} to RGB.")
                 bg_image_pil = display_img_object.convert('RGB')
             else: bg_image_pil = display_img_object
        except Exception as convert_err:
             st.error(f"Failed to prepare image for canvas: {convert_err}")
             logger.error(f"RGB conversion error for canvas background: {convert_err}", exc_info=True)

        if bg_image_pil:
            MAX_CANVAS_WIDTH = 600; MAX_CANVAS_HEIGHT = 550
            img_w, img_h = bg_image_pil.size
            aspect_ratio = img_w / img_h if img_h > 0 else 1
            canvas_width = min(img_w, MAX_CANVAS_WIDTH)
            canvas_height = int(canvas_width / aspect_ratio) if aspect_ratio > 0 else MAX_CANVAS_HEIGHT
            if canvas_height > MAX_CANVAS_HEIGHT: canvas_height = MAX_CANVAS_HEIGHT; canvas_width = int(canvas_height * aspect_ratio)
            canvas_width = max(canvas_width, 150); canvas_height = max(canvas_height, 150)
            logger.info(f"Canvas Prep: Image size={img_w}x{img_h}, Calculated canvas size W={canvas_width}, H={canvas_height}")

            if canvas_width > 0 and canvas_height > 0:
                st.caption("Click and drag on the image below to select a Region of Interest (ROI).")
                try:
                    initial_drawing = st.session_state.get("canvas_drawing")
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.2)", stroke_width=2, stroke_color="rgba(220, 50, 50, 0.9)",
                        background_image=bg_image_pil, update_streamlit=True,
                        height=int(canvas_height), width=int(canvas_width),
                        drawing_mode="rect", initial_drawing=initial_drawing,
                        key="drawable_canvas",
                    )

                    if canvas_result and canvas_result.json_data is not None and canvas_result.json_data.get("objects"):
                        logger.debug(f"Canvas updated. Raw JSON data: {canvas_result.json_data}")
                        last_object = canvas_result.json_data["objects"][-1]
                        if last_object["type"] == "rect":
                             scale_x = img_w / canvas_width; scale_y = img_h / canvas_height
                             raw_left = int(last_object["left"]); raw_top = int(last_object["top"])
                             raw_width = int(last_object["width"] * last_object.get("scaleX", 1))
                             raw_height = int(last_object["height"] * last_object.get("scaleY", 1))
                             orig_left = int(raw_left * scale_x); orig_top = int(raw_top * scale_y)
                             orig_width = int(raw_width * scale_x); orig_height = int(raw_height * scale_y)
                             logger.debug(f"Canvas Rect: L={raw_left}, T={raw_top}, W={raw_width}, H={raw_height}")
                             logger.debug(f"Canvas Scale: ScaleX={scale_x:.2f}, ScaleY={scale_y:.2f}")
                             logger.debug(f"Scaled ROI (Orig Img): L={orig_left}, T={orig_top}, W={orig_width}, H={orig_height}")

                             if orig_width > 10 and orig_height > 10:
                                 new_roi = {"left": orig_left, "top": orig_top, "width": orig_width, "height": orig_height}
                                 if st.session_state.roi_coords != new_roi:
                                     st.session_state.roi_coords = new_roi
                                     st.session_state.canvas_drawing = canvas_result.json_data
                                     logger.info(f"ROI coordinates updated based on canvas drawing: {new_roi}")
                                     st.rerun()
                             else: logger.debug("Ignoring rectangle: Dimensions too small after scaling.")
                except Exception as canvas_error:
                    st.error(f"Error initializing or processing the drawing canvas: {canvas_error}")
                    logger.error(f"st_canvas failed: {canvas_error}", exc_info=True)
                    st.warning("Drawing feature may be unavailable. Check Browser's Developer Console (F12) for details.")
            else:
                st.error("Calculated canvas dimensions are invalid (<= 0). Cannot display drawing canvas.")
                logger.error(f"Invalid canvas dimensions calculated: W={canvas_width}, H={canvas_height}")
        else:
             st.error("Image is invalid. Cannot display drawing canvas.")
             logger.error("Canvas Prep: bg_image_pil was invalid before attempting canvas setup.")

        if st.session_state.is_dicom and st.session_state.dicom_metadata:
             display_dicom_metadata(st.session_state.dicom_metadata)
    else:
        logger.debug(f"Image Viewer: No valid display_image found in session state (Type: {type(display_img_object)}).")
        st.markdown("---")
        if st.session_state.get("uploaded_file_info"): st.warning("Image processing may have failed. Please check application logs or try uploading again.")
        else: st.info("Image will appear here after uploading via the sidebar.")
        st.markdown("<div style='height: 400px; border: 2px dashed #ccc; display: flex; align-items: center; justify-content: center; color: #aaa;'>Image Display Area</div>", unsafe_allow_html=True)

with col2:
    st.subheader("📊 Analysis & Results")
    tabs = st.tabs(["🔬 Initial Analysis", "💬 Q&A History", "🩺 Disease Focus", "📈 Confidence"])
    with tabs[0]: st.text_area("Overall Findings & Impressions", value=st.session_state.get("initial_analysis", "No initial analysis performed yet."), height=450, key="output_initial_text", disabled=True, help="General analysis results based on the 'Run Initial Analysis' action.")
    with tabs[1]:
        st.text_area("AI Answer to Last Question", value=st.session_state.get("qa_answer", "Ask a question using the sidebar."), height=200, key="output_qa_text", disabled=True, help="AI's answer to your most recent question.")
        st.markdown("---")
        if st.session_state.history:
             with st.expander("View Full Conversation History", expanded=True):
                 for i, (q, a) in enumerate(reversed(st.session_state.history)):
                     st.markdown(f"**You ({len(st.session_state.history)-i}):**"); st.caption(q)
                     st.markdown(f"**AI ({len(st.session_state.history)-i}):**"); st.markdown(a, unsafe_allow_html=True)
                     if i < len(st.session_state.history) - 1: st.markdown("---")
        else: st.caption("No conversation history yet.")
    with tabs[2]: st.text_area("Disease-Specific Findings", value=st.session_state.get("disease_analysis", "No focused condition analysis performed yet."), height=450, key="output_disease_text", disabled=True, help="Analysis results related to the specific condition selected in the sidebar.")
    with tabs[3]: st.text_area("AI Confidence Estimation", value=st.session_state.get("confidence_score", "No confidence estimation performed yet."), height=450, key="output_confidence_text", disabled=True, help="AI's estimated confidence in its most recent relevant analysis or answer.")

# =============================================================================
# === ACTION HANDLING ===========================================================
# =============================================================================
# ... (The *full*, REWRITTEN Action Handling code block goes here) ...
# Retrieve the action requested by the user interface
current_action: Optional[str] = st.session_state.get("last_action")

if current_action:
    logger.info(f"Initiating action handling for: '{current_action}'")

    # --- Pre-Action Validation ---
    processed_image = st.session_state.get("processed_image")
    session_id = st.session_state.get("session_id")

    if not isinstance(processed_image, Image.Image) or not session_id:
        error_msg = f"Cannot perform action '{current_action}': A processed image is required, but not found or the session is invalid."
        st.error(error_msg)
        processed_image_type = type(processed_image).__name__ if processed_image else "None"
        logger.error(
            f"Action '{current_action}' aborted pre-check. "
            f"Processed image type: {processed_image_type}, "
            f"Session ID valid: {bool(session_id)}"
        )
        st.session_state.last_action = None
        st.stop()

    # Prepare common variables for actions
    img_for_llm: Image.Image = processed_image
    roi: Optional[Dict[str, int]] = st.session_state.get("roi_coords")
    roi_context_str = " (focusing on selected ROI)" if roi else ""

    # --- Action Execution Block ---
    try:
        # Reset relevant state variables *before* performing the new action
        if current_action != "ask": st.session_state.qa_answer = ""
        if current_action != "analyze": st.session_state.initial_analysis = ""
        if current_action != "disease": st.session_state.disease_analysis = ""
        if current_action != "confidence": st.session_state.confidence_score = ""
        if current_action != "generate_report_data": st.session_state.pdf_report_bytes = None

        # --- Action Logic ---
        if current_action == "analyze":
            st.info("🔬 Performing preliminary analysis on the image...")
            with st.spinner("AI is analyzing the full image..."):
                analysis_result = run_initial_analysis(img_for_llm)
            st.session_state.initial_analysis = analysis_result
            logger.info("Preliminary analysis action completed successfully.")
            st.success("Preliminary analysis finished.")

        elif current_action == "ask":
            question = st.session_state.get("question_input_widget", "").strip() # Read from widget key
            if not question:
                st.warning("Please enter a question before clicking 'Ask'.")
                logger.warning("'Ask' action triggered with empty question input.")
            else:
                st.info(f"❓ Asking the AI your question{roi_context_str}...")
                st.session_state.qa_answer = "" # Clear previous answer explicitly here
                primary_model_name = "Gemini" # Example
                with st.spinner(f"{primary_model_name} is processing your question{roi_context_str}..."):
                    gemini_answer, success = run_multimodal_qa(img_for_llm, question, st.session_state.history, roi)

                if success:
                    st.session_state.qa_answer = gemini_answer
                    if isinstance(st.session_state.history, list):
                         st.session_state.history.append((question, gemini_answer))
                    else: st.session_state.history = [(question, gemini_answer)]
                    logger.info(f"Multimodal QA successful (Primary: {primary_model_name}) for question: '{question}'{roi_context_str}")
                    st.success(f"{primary_model_name} answered your question.")
                else:
                    # --- Fallback Logic ---
                    error_message = f"Primary AI ({primary_model_name}) failed to answer. Reason: {gemini_answer}"
                    st.error(error_message)
                    logger.warning(f"Primary AI ({primary_model_name}) failed for question: '{question}'. Reason: {gemini_answer}")
                    st.session_state.qa_answer = f"**[Primary AI Error]** {gemini_answer}\n\n---\n"

                    hf_token_available = bool(os.environ.get("HF_API_TOKEN"))
                    # Check if HF model ID indicates it's actually available
                    hf_model_available = bool(HF_VQA_MODEL_ID and HF_VQA_MODEL_ID != "hf_model_not_found" and HF_VQA_MODEL_ID != "unavailable")

                    if hf_token_available and hf_model_available:
                        st.info(f"Attempting fallback using Hugging Face model ({HF_VQA_MODEL_ID})...")
                        with st.spinner(f"Trying fallback AI ({HF_VQA_MODEL_ID}){roi_context_str}..."):
                            # Pass ROI to the HF function as well
                            hf_answer, hf_success = query_hf_vqa_inference_api(img_for_llm, question, roi)

                        if hf_success:
                            fallback_display = f"**[Fallback Answer ({HF_VQA_MODEL_ID})]**\n\n{hf_answer}"
                            st.session_state.qa_answer += fallback_display
                            if isinstance(st.session_state.history, list):
                                st.session_state.history.append((question, fallback_display))
                            else: st.session_state.history = [(question, fallback_display)]
                            logger.info(f"HF VQA fallback successful for question: '{question}'{roi_context_str}")
                            st.success(f"Fallback AI ({HF_VQA_MODEL_ID}) provided an answer.")
                        else:
                            fallback_error_msg = f"Fallback AI ({HF_VQA_MODEL_ID}) also failed. Reason: {hf_answer}"
                            st.session_state.qa_answer += f"**[Fallback Failed]** {fallback_error_msg}"
                            st.error(fallback_error_msg)
                            logger.error(f"HF VQA fallback failed for question: '{question}'. Reason: {hf_answer}")
                    else:
                        missing_config = []
                        if not hf_token_available: missing_config.append("HF Token")
                        if not hf_model_available: missing_config.append("HF Model/Module")
                        fallback_unavailable_msg = f"Fallback AI unavailable (Configuration missing: {', '.join(missing_config)})."
                        st.session_state.qa_answer += f"**[Fallback Unavailable]** {fallback_unavailable_msg}"
                        st.warning(fallback_unavailable_msg)
                        logger.warning(f"HF VQA fallback skipped for question '{question}': Configuration missing.")

        elif current_action == "disease":
            selected_disease = st.session_state.get("disease_select_widget") # Read from widget key
            if not selected_disease:
                 st.warning("Please select a condition to analyze.")
                 logger.warning("'Disease Analysis' action triggered without a selected condition.")
            else:
                st.info(f"🩺 Analyzing the image for signs of '{selected_disease}'{roi_context_str}...")
                with st.spinner(f"AI is assessing for '{selected_disease}'{roi_context_str}..."):
                    disease_analysis_result = run_disease_analysis(img_for_llm, selected_disease, roi)
                st.session_state.disease_analysis = disease_analysis_result
                logger.info(f"Disease-specific analysis completed for '{selected_disease}'.")
                st.success(f"Analysis for '{selected_disease}' finished.")

        elif current_action == "confidence":
            st.info(f"📊 Estimating AI confidence based on recent interactions{roi_context_str}...")
            current_history = st.session_state.get("history", [])
            if not isinstance(current_history, list) or not current_history:
                 st.warning("Cannot estimate confidence without prior Q&A interaction.")
                 logger.warning("Confidence estimation skipped: No interaction history found.")
            else:
                with st.spinner(f"Calculating confidence score{roi_context_str}..."):
                     # Pass the image as context, history for interaction context
                    confidence_result = estimate_ai_confidence(img_for_llm, current_history)
                st.session_state.confidence_score = confidence_result
                logger.info("Confidence estimation action completed.")
                st.success("AI confidence estimation finished.")


        elif current_action == "generate_report_data":
            st.info("📄 Preparing data for PDF report generation...")
            st.session_state.pdf_report_bytes = None # Ensure clean state

            img_for_report: Optional[Image.Image] = st.session_state.get("display_image")
            img_with_roi_drawn = None

            if isinstance(img_for_report, Image.Image):
                current_roi = st.session_state.get("roi_coords")
                # Draw ROI on a *copy* if it exists
                if current_roi and isinstance(current_roi, dict) and all(k in current_roi for k in ['left', 'top', 'width', 'height']):
                    try:
                        img_copy = img_for_report.copy().convert("RGB")
                        draw = ImageDraw.Draw(img_copy)
                        # Use the scaled ROI coordinates stored in session state
                        x0, y0 = int(current_roi['left']), int(current_roi['top'])
                        x1, y1 = x0 + int(current_roi['width']), y0 + int(current_roi['height'])
                        # Ensure coordinates are within image bounds before drawing
                        img_w_rep, img_h_rep = img_copy.size
                        x0 = max(0, min(x0, img_w_rep -1))
                        y0 = max(0, min(y0, img_h_rep -1))
                        x1 = max(0, min(x1, img_w_rep))
                        y1 = max(0, min(y1, img_h_rep))
                        if x1 > x0 and y1 > y0: # Ensure valid rectangle
                             draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
                             img_with_roi_drawn = img_copy
                             logger.info("Successfully drew ROI onto image copy for PDF report.")
                        else:
                             logger.warning("ROI dimensions invalid after clamping to image bounds, not drawing.")
                             img_with_roi_drawn = img_for_report # Fallback
                    except Exception as e:
                        logger.error(f"Failed to draw ROI on report image: {e}", exc_info=True)
                        st.warning("Could not draw the ROI box onto the report image, using original.")
                        img_with_roi_drawn = img_for_report # Fallback
                else:
                    img_with_roi_drawn = img_for_report # Use original if no ROI

                # --- Prepare Report Content ---
                current_history = st.session_state.get("history", [])
                full_qa_history = "\n\n".join([f"User Q: {q}\n\nAI A: {a}" for q, a in current_history]) if current_history else "No Q&A interactions recorded."
                outputs_for_report = {
                    "Preliminary Analysis": st.session_state.get("initial_analysis", "Not performed."),
                    "Conversation History": full_qa_history,
                    "Condition-Specific Analysis": st.session_state.get("disease_analysis", "Not performed."),
                    "Last Confidence Estimate": st.session_state.get("confidence_score", "Not estimated.")
                }

                # --- DICOM Metadata Handling (with PHI Filtering - CRITICAL) ---
                if st.session_state.get("is_dicom") and st.session_state.get("dicom_metadata"):
                    logger.info("Processing DICOM metadata for report (with filtering).")
                    filtered_meta_str_list = []
                    # ** Define or import your comprehensive PHI exclusion list **
                    PHI_TAGS_TO_EXCLUDE = [
                        "PatientName", "PatientID", "PatientBirthDate", "PatientSex",
                        "OtherPatientIDs", "OtherPatientNames", "PatientAddress",
                        "PatientTelephoneNumbers", "ReferringPhysicianName", "InstitutionName",
                        "InstitutionAddress", "PhysicianOfRecord", "OperatorsName",
                        "IssuerOfPatientID", "PatientBirthTime", "PatientComments",
                        "PerformedProcedureStepStartDate", "PerformedProcedureStepStartTime",
                        "RequestingPhysician", "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate",
                        "StudyTime", "SeriesTime", "AcquisitionTime", "ContentTime",
                        "AccessionNumber", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
                        # Add any other potentially identifying tags
                    ]
                    dicom_meta = st.session_state.dicom_metadata # Use the tech metadata extracted earlier
                    for tag_name, tag_value in dicom_meta.items():
                         if tag_name in PHI_TAGS_TO_EXCLUDE:
                              logger.debug(f"Excluding potentially sensitive tag from report: {tag_name}")
                              continue

                         display_value = ""
                         try:
                             # Reuse formatting logic from ui_components or report_utils if abstracted
                             if isinstance(tag_value, list): display_value = ", ".join(map(str, tag_value))
                             elif isinstance(tag_value, pydicom.uid.UID): display_value = tag_value.name
                             elif isinstance(tag_value, bytes): display_value = f"[Binary Data ({len(tag_value)} bytes)]"
                             elif isinstance(tag_value, pydicom.valuerep.PersonName): display_value = "[Person Name]" # Explicitly filter PersonName objects
                             else: display_value = str(tag_value).strip()

                             if display_value and len(display_value) < 512: # Limit length
                                  filtered_meta_str_list.append(f"{tag_name}: {display_value}")
                         except Exception as tag_err:
                              logger.warning(f"Error processing report metadata tag '{tag_name}': {tag_err}")
                              filtered_meta_str_list.append(f"{tag_name}: [Error processing value]")


                    outputs_for_report["DICOM Metadata (Filtered)"] = "\n".join(filtered_meta_str_list) if filtered_meta_str_list else "No applicable metadata found or extracted."
                    logger.info(f"Included {len(filtered_meta_str_list)} filtered DICOM tags in report data.")
                # --- End DICOM Handling ---

                # --- Generate PDF Bytes ---
                with st.spinner("🎨 Generating PDF document..."):
                    pdf_bytes = generate_pdf_report_bytes(
                        session_id=session_id,
                        image=img_with_roi_drawn, # Use image with ROI drawn if available
                        analysis_outputs=outputs_for_report
                    )

                if pdf_bytes:
                    st.session_state.pdf_report_bytes = pdf_bytes
                    st.success("✅ PDF report data generated successfully!")
                    logger.info("PDF report data generation successful.")
                else:
                    st.error("❌ Failed to generate PDF report data. The generation function returned an error. Check logs.")
                    logger.error("PDF report generation failed (generate_pdf_report_bytes returned None or empty).")

            else:
                st.error("Cannot generate report: The image to include is not valid.")
                logger.error("PDF generation skipped: No valid PIL Image found in st.session_state.display_image.")

        # --- Unknown Action Handling ---
        else:
            st.warning(f"Action '{current_action}' is not recognized or implemented.")
            logger.warning(f"Attempted to handle unknown action: '{current_action}'")

    # --- General Exception Handling for Actions ---
    except Exception as e:
        st.error(f"An unexpected error occurred while trying to perform action '{current_action}'. Please check application logs.")
        logger.critical(f"Unhandled exception during action '{current_action}': {e}", exc_info=True)

    # --- Post-Action Cleanup & UI Update ---
    finally:
        st.session_state.last_action = None
        logger.debug(f"Action '{current_action}' handling finished. Resetting last_action state.")
        st.rerun() # Update the UI

# ------------------------------------------------------------------------------
# 8) Footer Disclaimer
# ------------------------------------------------------------------------------
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced | Session: {st.session_state.session_id or 'N/A'} | ⚠️ For Research, Informational & Educational Purposes Only. Not for clinical diagnosis.")