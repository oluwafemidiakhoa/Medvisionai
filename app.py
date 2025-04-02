# main_app.py (Example Name)

# --- Core Libraries ---
import io
import os
import uuid
import logging
import base64
from typing import Any, Dict, Optional, Tuple, List # Ensure all needed types are here

# --- Streamlit ---
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# --- Image & DICOM Processing ---
from PIL import Image, ImageDraw, UnidentifiedImageError
import pydicom # Keep import for type checking if needed, even if helpers use it

# ------------------------------------------------------------------------------
# 1) Monkey-Patch: Add image_to_url (Keep as provided, maybe add version check later)
#    Comment: This might not be required in recent Streamlit versions. Verify if needed.
# ------------------------------------------------------------------------------
import streamlit.elements.image as st_image
if not hasattr(st_image, "image_to_url"):
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
# 2) Import Custom Utilities & Fallbacks
# ------------------------------------------------------------------------------
# --- Attempt to import all custom utility modules ---
try:
    from dicom_utils import parse_dicom, extract_dicom_metadata, dicom_to_image, get_default_wl
    from llm_interactions import (
        run_initial_analysis, run_multimodal_qa, run_disease_analysis,
        estimate_ai_confidence
    )
    from report_utils import generate_pdf_report_bytes
    from ui_helpers import display_dicom_metadata, dicom_wl_sliders # Import UI helpers
    try:
        # Keep HF optional as it might rely on extra dependencies/secrets
        from hf_models import query_hf_vqa_inference_api, HF_VQA_MODEL_ID
    except ImportError:
        HF_VQA_MODEL_ID = "hf_model_not_found" # Indicate module missing
        def query_hf_vqa_inference_api(img: Image.Image, question: str, roi: Optional[Dict] = None) -> Tuple[str, bool]:
            logging.warning("Using MOCK Hugging Face VQA fallback (hf_models.py not found).")
            return f"[Fallback Unavailable] Hugging Face module not found.", False
        logging.warning("hf_models.py not found. Hugging Face VQA fallback is disabled.")

except ImportError as import_error:
    # Display a clear error in the UI and log it
    st.error(f"Critical Error: Failed to import utility modules ({import_error}). App functionality severely limited. Please ensure all required `.py` files are present.")
    logging.critical(f"Failed to import required utility modules: {import_error}", exc_info=True)
    # --- Provide Dummy Functions to Prevent Immediate Crash ---
    # (These should ideally never be called if the UI disables relevant buttons)
    def parse_dicom(b): logging.error("MOCK FUNCTION: parse_dicom"); return None
    def extract_dicom_metadata(d): logging.error("MOCK FUNCTION: extract_dicom_metadata"); return {"Error": "dicom_utils missing"}
    def dicom_to_image(d, wc=None, ww=None): logging.error("MOCK FUNCTION: dicom_to_image"); return Image.new("RGB", (100, 100), "grey")
    def get_default_wl(d): logging.error("MOCK FUNCTION: get_default_wl"); return None, None
    def run_initial_analysis(img): logging.error("MOCK FUNCTION: run_initial_analysis"); return "Error: llm_interactions missing."
    def run_multimodal_qa(img, q, h, roi): logging.error("MOCK FUNCTION: run_multimodal_qa"); return "Error: llm_interactions missing.", False
    def run_disease_analysis(img, d, roi): logging.error("MOCK FUNCTION: run_disease_analysis"); return "Error: llm_interactions missing."
    def estimate_ai_confidence(img, h): logging.error("MOCK FUNCTION: estimate_ai_confidence"); return "Error: llm_interactions missing."
    def generate_pdf_report_bytes(sid, img, data): logging.error("MOCK FUNCTION: generate_pdf_report_bytes"); return None
    def display_dicom_metadata(m): st.warning("Mock display_dicom_metadata (ui_helpers missing)")
    def dicom_wl_sliders(ds, m): st.warning("Mock dicom_wl_sliders (ui_helpers missing)"); return None, None
    HF_VQA_MODEL_ID = "unavailable"
    def query_hf_vqa_inference_api(img, q, roi): logging.error("MOCK FUNCTION: query_hf_vqa_inference_api"); return "HF Fallback Unavailable.", False
    # Stop execution if critical modules are missing? Or allow limited UI? For now, allow UI.
    # st.stop()

# ------------------------------------------------------------------------------
# 3) Setup Logging (Using Basic Config)
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
# 4) Configure Streamlit Page
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="RadVision AI Advanced",
    layout="wide", # Use wide layout for more space
    page_icon="⚕️", # Standard emoji icon
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# 5) Display Logo in Sidebar (Optional - Uncomment if logo exists)
# ------------------------------------------------------------------------------
# LOGO_PATH = "assets/radvisionai-logo.png" # Example path
# if os.path.exists(LOGO_PATH):
#     with st.sidebar:
#         try:
#             st.image(LOGO_PATH, width=200)
#         except Exception as e:
#             logger.warning(f"Logo image '{LOGO_PATH}' failed to load: {e}")
# else:
#     logger.debug(f"Logo image not found at '{LOGO_PATH}'")

# ------------------------------------------------------------------------------
# 6) Initialize Session State (Using Default Dictionary)
# ------------------------------------------------------------------------------
# Define all expected keys and their default values
DEFAULT_STATE = {
    "uploaded_file_info": None,     # Tracks the current file to detect changes
    "raw_image_bytes": None,        # Stores the raw bytes of the uploaded file
    "is_dicom": False,              # Flag indicating if the file is DICOM
    "dicom_dataset": None,          # Stores the parsed pydicom.Dataset
    "dicom_metadata": {},           # Extracted technical DICOM metadata
    # "dicom_wc": None,             # Default WC from DICOM (redundant if using slider state)
    # "dicom_ww": None,             # Default WW from DICOM (redundant if using slider state)
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
    # 'slider_wc': None,            # Current value from WC slider (set by dicom_wl_sliders)
    # 'slider_ww': None,            # Current value from WW slider (set by dicom_wl_sliders)
    # Note: Slider values are implicitly managed by Streamlit widgets via their keys
    # We only need to read them when applying W/L. No need to store explicitly here
    # unless needed for complex state interactions.
}
# Initialize session state keys if they don't exist
for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value
# Ensure history is always a list
if not isinstance(st.session_state.history, list):
    st.session_state.history = []

# ------------------------------------------------------------------------------
# 7) Page Title & Disclaimer
# ------------------------------------------------------------------------------
st.title("⚕️ RadVision QA Advanced: AI-Assisted Image Analysis")
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
                            logger.info(f"Default W/L from DICOM tags: WC={wc_default}, WW={ww_default}")

                            # Generate display image using default W/L
                            logger.debug("Generating display image from DICOM using default W/L...")
                            temp_display_image = dicom_to_image(ds, wc_default, ww_default)

                            # Generate processed image (e.g., for AI) - maybe using default W/L or normalized
                            # Option 1: Use default W/L for processed image too
                            # temp_processed_image = temp_display_image.copy() if temp_display_image else None
                            # Option 2: Use basic scaling/normalization (as in original code)
                            logger.debug("Generating processed image from DICOM using basic scaling...")
                            temp_processed_image = dicom_to_image(ds, window_center=None, window_width=None) # Let helper handle scaling

                            if temp_display_image and temp_processed_image:
                                logger.info("DICOM images (display & processed) generated successfully.")
                                processing_successful = True
                            else:
                                st.error("DICOM processing failed: Could not generate required image objects from pixel data.")
                                logger.error("dicom_to_image returned None for display and/or processed image.")
                        else:
                            # Error message handled by parse_dicom directly using st.error/warning
                            logger.error("parse_dicom returned None.")
                            # Keep processing_successful = False

                    except Exception as e:
                        st.error(f"An unexpected error occurred during DICOM processing: {e}")
                        logger.critical(f"DICOM processing pipeline error: {e}", exc_info=True)
                        processing_successful = False # Ensure failure state

                # --- Standard Image Processing Branch ---
                else:
                    try:
                        logger.debug("Attempting to open standard image file...")
                        img = Image.open(io.BytesIO(st.session_state.raw_image_bytes))
                        # Keep a copy for processing if needed (or just use the same)
                        temp_processed_image = img.copy()
                        # Ensure display image is RGB
                        if img.mode != 'RGB':
                            logger.info(f"Original image mode: {img.mode}. Converting to RGB for display.")
                            temp_display_image = img.convert("RGB")
                        else:
                            temp_display_image = img
                        st.session_state.dicom_dataset = None # Clear any stale DICOM data
                        st.session_state.dicom_metadata = {}
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
                    # Ensure final display image is RGB before storing
                    if temp_display_image.mode != 'RGB':
                         logger.warning(f"Final check: Converting display image from {temp_display_image.mode} to RGB.")
                         st.session_state.display_image = temp_display_image.convert('RGB')
                    else:
                         st.session_state.display_image = temp_display_image
                    # Store the processed image
                    st.session_state.processed_image = temp_processed_image
                    logger.info(f"Image processing complete. Display: {st.session_state.display_image.mode} {st.session_state.display_image.size}, Processed: {st.session_state.processed_image.mode} {st.session_state.processed_image.size}")
                    # Clear drawing state on new image
                    st.session_state.roi_coords = None
                    st.session_state.canvas_drawing = None
                    st.success(f"✅ Image '{uploaded_file.name}' loaded successfully!")
                    st.rerun() # Rerun immediately to update UI with the new image
                else:
                    logger.critical("Image loading failed after processing step.")
                    # Check if an error was already shown, if not, show a generic one
                    if processing_successful: # This case indicates images became invalid *after* processing step
                         st.error("❌ Image loading failed unexpectedly after processing. Please check logs.")
                    # Else: Specific error should have been shown already by the failed step.
                    # Clean up potentially partially set state on failure
                    st.session_state.uploaded_file_info = None
                    st.session_state.raw_image_bytes = None
                    st.session_state.display_image = None
                    st.session_state.processed_image = None
                    st.session_state.dicom_dataset = None
                    st.session_state.dicom_metadata = {}
            # End of `with st.spinner`
        # End of `if new_file_info != ...` (state update logic)
    # End of `if uploaded_file is not None` (widget check)

    # Separator in sidebar
    st.markdown("---")

    # --- 2) DICOM Window/Level Controls (Conditional using UI Helper) ---
    # Show controls only if a valid DICOM image is loaded
    if st.session_state.is_dicom and st.session_state.dicom_dataset and st.session_state.display_image:
        with st.expander("DICOM Window/Level", expanded=False):
            # Call the helper function which contains the sliders and reset button
            # It reads defaults from metadata and dataset internally
            try:
                # The helper returns the *current* values selected by the user on the sliders
                wc_current, ww_current = dicom_wl_sliders(st.session_state.dicom_dataset, st.session_state.dicom_metadata)

                # Check if the sliders returned valid values and if they changed significantly
                # Need to compare with the W/L that produced the *current* display_image
                # This requires storing the W/L used to generate the current display image,
                # or recalculating the image only if values change. Let's recalculate on change.

                # Retrieve the W/L values that *were used* to generate the current display image
                # If not stored, we can't reliably detect change without rerunning dicom_to_image every time.
                # Let's simplify: Assume the sliders manage their state. If the button wasn't pressed,
                # read the current slider values and regenerate the image if needed.

                # Read current slider values using their keys (if reset wasn't pressed)
                wc_slider_val = st.session_state.get("dicom_wc_slider")
                ww_slider_val = st.session_state.get("dicom_ww_slider")

                # Get the W/L used for the *currently displayed* image (Need to store this)
                # Let's add state variables for this:
                if 'current_display_wc' not in st.session_state: st.session_state.current_display_wc = None
                if 'current_display_ww' not in st.session_state: st.session_state.current_display_ww = None

                # If sliders exist and their values differ from what's displayed
                if wc_slider_val is not None and ww_slider_val is not None and \
                   (abs(wc_slider_val - st.session_state.current_display_wc) > 1e-3 or \
                    abs(ww_slider_val - st.session_state.current_display_ww) > 1e-3):

                    logger.info(f"Slider values changed. Applying new W/L: WC={wc_slider_val:.1f}, WW={ww_slider_val:.1f}")
                    with st.spinner("Applying Window/Level..."):
                         new_display_image = dicom_to_image(st.session_state.dicom_dataset, wc_slider_val, ww_slider_val)
                         if new_display_image:
                              # Ensure it's RGB
                              if new_display_image.mode != 'RGB':
                                   st.session_state.display_image = new_display_image.convert('RGB')
                                   logger.debug("Converted W/L adjusted image to RGB.")
                              else:
                                   st.session_state.display_image = new_display_image
                              # Store the W/L values that generated this image
                              st.session_state.current_display_wc = wc_slider_val
                              st.session_state.current_display_ww = ww_slider_val
                              st.rerun() # Update the main panel image display
                         else:
                              st.error("Failed to apply new Window/Level settings.")
                              logger.error("dicom_to_image returned None when applying new W/L.")
                # Initialize display W/L state if it's missing (first run after load)
                elif st.session_state.current_display_wc is None and st.session_state.current_display_ww is None and wc_slider_val is not None and ww_slider_val is not None:
                     st.session_state.current_display_wc = wc_slider_val
                     st.session_state.current_display_ww = ww_slider_val
                     logger.debug("Initialized current display W/L state from sliders.")


            except Exception as e:
                 st.error(f"Error in W/L control logic: {e}")
                 logger.error(f"W/L slider integration error: {e}", exc_info=True)
        st.markdown("---")


    # --- 3) Analysis & Interaction Controls (Conditional) ---
    # Show buttons only if a valid image is loaded and ready
    if isinstance(st.session_state.get("display_image"), Image.Image):
        st.subheader("AI Actions")

        if st.button("▶️ Run Initial Analysis", key="analyze_btn", help="Perform a general analysis of the entire image.", use_container_width=True):
            st.session_state.last_action = "analyze"; st.rerun()

        st.markdown("---")
        st.subheader("❓ Ask AI Question")

        # ROI Indicator and Clear Button
        if st.session_state.get("roi_coords"):
            # Display coordinates concisely
            rc = st.session_state.roi_coords
            roi_summary = f"L:{rc['left']}, T:{rc['top']}, W:{rc['width']}, H:{rc['height']}"
            st.info(f"✅ ROI Selected: [{roi_summary}]. Question will focus here.")
            if st.button("❌ Clear ROI", key="clear_roi_btn", help="Remove the highlighted region selection.", use_container_width=True):
                st.session_state.roi_coords = None
                st.session_state.canvas_drawing = None # Clear canvas state too
                logger.info("ROI cleared by user.")
                st.rerun()
        else:
            st.caption("ℹ️ Optionally, draw a rectangle on the image (left panel) to focus your question on a specific region.")

        # Question Input
        question_input = st.text_area( # Corrected variable name to avoid conflict
            "Ask about the image or the highlighted region:",
            height=100,
            key="question_input_widget", # Use distinct key
            placeholder="e.g., What type of scan is this? Are there any abnormalities in the selected region?",
            label_visibility="collapsed" # Hide redundant label
        )
        if st.button("💬 Ask AI", key="ask_btn", use_container_width=True):
            user_question = st.session_state.question_input_widget # Read from widget key
            if user_question and user_question.strip():
                st.session_state.last_action = "ask"
                st.rerun()
            else:
                st.warning("Please enter a question before asking the AI.")

        st.markdown("---")
        st.subheader("🎯 Focused Condition Analysis")

        # Define disease options (Consider loading from a config file)
        DISEASE_OPTIONS = [
            "Pneumonia", "Lung Cancer", "Stroke (Ischemic/Hemorrhagic)",
            "Bone Fracture", "Appendicitis", "Tuberculosis", "COVID-19 Pneumonitis",
            "Pulmonary Embolism", "Brain Tumor (e.g., Glioblastoma, Meningioma)",
            "Arthritis Signs", "Osteoporosis Signs", "Cardiomegaly",
            "Aortic Aneurysm/Dissection Signs", "Bowel Obstruction Signs"
        ]
        # Add a blank option first and sort the rest
        disease_options_sorted = [""] + sorted(DISEASE_OPTIONS)

        disease_select = st.selectbox(
            "Select a specific condition to analyze for:",
            options=disease_options_sorted,
            key="disease_select_widget", # Use distinct key
            help="The AI will look for findings relevant to the selected condition."
        )
        if st.button("🩺 Run Condition Analysis", key="disease_btn", use_container_width=True):
            selected_disease = st.session_state.disease_select_widget # Read from widget key
            if selected_disease:
                st.session_state.last_action = "disease"
                st.rerun()
            else:
                st.warning("Please select a condition from the dropdown first.")

        st.markdown("---")

        # --- Confidence & Report Section ---
        with st.expander("📊 Confidence & Report", expanded=True):
            # Confidence Button
            can_estimate_confidence = bool(st.session_state.history or st.session_state.initial_analysis or st.session_state.disease_analysis)
            if st.button("📈 Estimate AI Confidence", key="confidence_btn", help="Estimate the AI's confidence based on the conversation history and image.", disabled=not can_estimate_confidence, use_container_width=True):
                st.session_state.last_action = "confidence"
                st.rerun()
            if not can_estimate_confidence:
                st.caption("Perform an analysis or ask a question first to enable confidence estimation.")

            # Report Generation Button
            if st.button("📄 Generate PDF Report Data", key="generate_report_data_btn", help="Compile the session analysis into data ready for PDF download.", use_container_width=True):
                st.session_state.last_action = "generate_report_data"
                st.rerun()

            # Download Button (conditional)
            if st.session_state.get("pdf_report_bytes"):
                report_filename = f"RadVisionAI_Report_{st.session_state.session_id}.pdf"
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=st.session_state.pdf_report_bytes,
                    file_name=report_filename,
                    mime="application/pdf",
                    key="download_pdf_button",
                    help=f"Download the generated report ({report_filename})",
                    use_container_width=True
                )
                st.success("PDF report data ready for download.")
            # Indicate failure if report generation was the last action but bytes are missing
            elif st.session_state.get("last_action") == "generate_report_data" and not st.session_state.get("pdf_report_bytes"):
                # This state might be transient while the spinner is active, or permanent on failure.
                # The action handling section should display st.error on failure.
                # We can add a subtle indicator here if needed.
                # st.caption("PDF generation failed or in progress...")
                pass

    # Fallback message if no image is loaded
    else:
        st.info("👈 Please upload an image file (JPG, PNG, or DICOM) to begin analysis.")


# =============================================================================
# === MAIN PANEL DISPLAYS =====================================================
# =============================================================================
# Use columns for layout: Image/Canvas on left, Results on right
col1, col2 = st.columns([2, 3]) # Adjust ratio as needed (e.g., [1, 1] or [2, 3])

# --- Column 1: Image Viewer, Canvas, Metadata ---
with col1:
    st.subheader("🖼️ Image Viewer")

    display_img_object = st.session_state.get("display_image")

    if isinstance(display_img_object, Image.Image):
        # Prepare background image for canvas (ensure RGB)
        bg_image_pil = None
        try:
             if display_img_object.mode != 'RGB':
                 logger.debug(f"Canvas Prep: Converting display image from {display_img_object.mode} to RGB.")
                 bg_image_pil = display_img_object.convert('RGB')
             else:
                 bg_image_pil = display_img_object
        except Exception as convert_err:
             st.error(f"Failed to prepare image for canvas: {convert_err}")
             logger.error(f"RGB conversion error for canvas background: {convert_err}", exc_info=True)

        # Proceed only if background image preparation was successful
        if bg_image_pil:
            # --- Calculate Canvas Dimensions ---
            MAX_CANVAS_WIDTH = 600 # Max width for the canvas element
            MAX_CANVAS_HEIGHT = 550 # Max height for the canvas element
            img_w, img_h = bg_image_pil.size
            aspect_ratio = img_w / img_h if img_h > 0 else 1

            # Calculate initial width based on max width
            canvas_width = min(img_w, MAX_CANVAS_WIDTH)
            canvas_height = int(canvas_width / aspect_ratio) if aspect_ratio > 0 else MAX_CANVAS_HEIGHT

            # Adjust if height exceeds max height
            if canvas_height > MAX_CANVAS_HEIGHT:
                canvas_height = MAX_CANVAS_HEIGHT
                canvas_width = int(canvas_height * aspect_ratio)

            # Ensure minimum dimensions
            canvas_width = max(canvas_width, 150)
            canvas_height = max(canvas_height, 150)
            logger.info(f"Canvas Prep: Image size={img_w}x{img_h}, Calculated canvas size W={canvas_width}, H={canvas_height}")

            # --- Drawable Canvas ---
            if canvas_width > 0 and canvas_height > 0:
                st.caption("Click and drag on the image below to select a Region of Interest (ROI).")
                try:
                    # Retrieve previous drawing state if it exists
                    initial_drawing = st.session_state.get("canvas_drawing")

                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.2)",    # Orange fill for ROI
                        stroke_width=2,                         # Border width
                        stroke_color="rgba(220, 50, 50, 0.9)",  # Red border
                        background_image=bg_image_pil,          # The prepared PIL image
                        update_streamlit=True,                  # Send results back to Streamlit
                        height=int(canvas_height),
                        width=int(canvas_width),
                        drawing_mode="rect",                    # Allow drawing rectangles
                        initial_drawing=initial_drawing,        # Load previous rectangle if ROI exists
                        key="drawable_canvas",                  # Unique key for the widget
                    )

                    # --- ROI Processing Logic ---
                    if canvas_result and canvas_result.json_data is not None and canvas_result.json_data.get("objects"):
                        logger.debug(f"Canvas updated. Raw JSON data: {canvas_result.json_data}")
                        # Process the *last* drawn rectangle object
                        last_object = canvas_result.json_data["objects"][-1]
                        if last_object["type"] == "rect":
                             # Extract coordinates, applying canvas scaling factors
                             # IMPORTANT: These coordinates are relative to the *canvas* size,
                             # not the original image size. Need scaling if canvas size != image size.
                             scale_x = img_w / canvas_width
                             scale_y = img_h / canvas_height

                             # Raw canvas coordinates
                             raw_left = int(last_object["left"])
                             raw_top = int(last_object["top"])
                             # Apply object's own scale first (if user resized the rect)
                             raw_width = int(last_object["width"] * last_object.get("scaleX", 1))
                             raw_height = int(last_object["height"] * last_object.get("scaleY", 1))

                             # Scale to original image coordinates
                             orig_left = int(raw_left * scale_x)
                             orig_top = int(raw_top * scale_y)
                             orig_width = int(raw_width * scale_x)
                             orig_height = int(raw_height * scale_y)

                             logger.debug(f"Canvas Rect: L={raw_left}, T={raw_top}, W={raw_width}, H={raw_height}")
                             logger.debug(f"Canvas Scale: ScaleX={scale_x:.2f}, ScaleY={scale_y:.2f}")
                             logger.debug(f"Scaled ROI (Orig Img): L={orig_left}, T={orig_top}, W={orig_width}, H={orig_height}")

                             # Basic validation for minimum size on original image
                             if orig_width > 10 and orig_height > 10:
                                 new_roi = {"left": orig_left, "top": orig_top, "width": orig_width, "height": orig_height}
                                 # Update state only if ROI actually changed to prevent unnecessary reruns
                                 if st.session_state.roi_coords != new_roi:
                                     st.session_state.roi_coords = new_roi
                                     st.session_state.canvas_drawing = canvas_result.json_data # Store canvas state
                                     logger.info(f"ROI coordinates updated based on canvas drawing: {new_roi}")
                                     st.rerun() # Rerun to update the ROI indicator in sidebar
                             else:
                                 logger.debug("Ignoring rectangle: Dimensions too small after scaling to original image.")
                        #else: logger.debug(f"Ignoring last canvas object: Type is '{last_object['type']}', expected 'rect'.")
                    #else: logger.debug("Canvas result has no 'objects' or json_data is None.") # Frequent log, maybe remove

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

        # --- DICOM Metadata Display (using UI Helper) ---
        if st.session_state.is_dicom and st.session_state.dicom_metadata:
             # Call the helper function to display metadata in an expander
             display_dicom_metadata(st.session_state.dicom_metadata)

    # --- Fallback if display_image is NOT valid ---
    else:
        logger.debug(f"Image Viewer: No valid display_image found in session state (Type: {type(display_img_object)}).")
        st.markdown("---")
        # Show appropriate message based on whether a file was uploaded
        if st.session_state.get("uploaded_file_info"):
            st.warning("Image processing may have failed. Please check application logs or try uploading again.")
        else:
            st.info("Image will appear here after uploading via the sidebar.")
        # Simple placeholder box
        st.markdown("<div style='height: 400px; border: 2px dashed #ccc; display: flex; align-items: center; justify-content: center; color: #aaa;'>Image Display Area</div>", unsafe_allow_html=True)

# --- Column 2: Analysis Results Tabs ---
with col2:
    st.subheader("📊 Analysis & Results")
    tab_keys = ["Initial Analysis", "Q&A", "Disease Focus", "Confidence"]
    # Create tabs with icons
    tabs = st.tabs([
        "🔬 Initial Analysis",
        "💬 Q&A History",
        "🩺 Disease Focus",
        "📈 Confidence"
    ])

    # Initial Analysis Tab
    with tabs[0]:
        st.text_area(
            "Overall Findings & Impressions",
            value=st.session_state.get("initial_analysis", "No initial analysis performed yet."),
            height=450, # Adjust height as needed
            key="output_initial_text", # Unique key
            disabled=True,
            help="General analysis results based on the 'Run Initial Analysis' action."
        )

    # Q&A Tab
    with tabs[1]:
        st.text_area(
            "AI Answer to Last Question",
            value=st.session_state.get("qa_answer", "Ask a question using the sidebar."),
            height=200, # Shorter height for the last answer
            key="output_qa_text", # Unique key
            disabled=True,
            help="AI's answer to your most recent question."
        )
        st.markdown("---")
        # Display Conversation History
        if st.session_state.history:
             with st.expander("View Full Conversation History", expanded=True):
                 # Display history in reverse chronological order (most recent first)
                 for i, (q, a) in enumerate(reversed(st.session_state.history)):
                     st.markdown(f"**You ({len(st.session_state.history)-i}):**")
                     st.caption(q) # Display question concisely
                     st.markdown(f"**AI ({len(st.session_state.history)-i}):**")
                     # Use markdown with unsafe_allow_html=True if answers contain markdown formatting
                     st.markdown(a, unsafe_allow_html=True)
                     if i < len(st.session_state.history) - 1:
                         st.markdown("---") # Separator between Q&A pairs
        else:
            st.caption("No conversation history yet.")

    # Disease Focus Tab
    with tabs[2]:
        st.text_area(
            "Disease-Specific Findings",
            value=st.session_state.get("disease_analysis", "No focused condition analysis performed yet."),
            height=450,
            key="output_disease_text", # Unique key
            disabled=True,
            help="Analysis results related to the specific condition selected in the sidebar."
        )

    # Confidence Tab
    with tabs[3]:
        st.text_area(
            "AI Confidence Estimation",
            value=st.session_state.get("confidence_score", "No confidence estimation performed yet."),
            height=450,
            key="output_confidence_text", # Unique key
            disabled=True,
            help="AI's estimated confidence in its most recent relevant analysis or answer."
        )


# =============================================================================
# === ACTION HANDLING ===========================================================
# =============================================================================
# --- Include the REWRITTEN Action Handling Code Block Here ---
# --- (The one generated in the earlier step, starting with validation) ---

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
                    if hf_token_available and HF_VQA_MODEL_ID and HF_VQA_MODEL_ID != "hf_model_not_found" and HF_VQA_MODEL_ID != "unavailable":
                        st.info(f"Attempting fallback using Hugging Face model ({HF_VQA_MODEL_ID})...")
                        with st.spinner(f"Trying fallback AI ({HF_VQA_MODEL_ID}){roi_context_str}..."):
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
                        if not HF_VQA_MODEL_ID or HF_VQA_MODEL_ID == "hf_model_not_found" or HF_VQA_MODEL_ID == "unavailable": missing_config.append("Model ID/Module")
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
                    confidence_result = estimate_ai_confidence(img_for_llm, current_history) # Pass image and history
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
                        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
                        img_with_roi_drawn = img_copy
                        logger.info("Successfully drew ROI onto image copy for PDF report.")
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
                    # ** Reuse the PHI filtering logic from the report_utils refinement **
                    # ** Or, even better, call a dedicated filtering function **
                    # For brevity, assuming filtering logic similar to report_utils rewrite is applied here:
                    PHI_TAGS_TO_EXCLUDE = ["PatientName", "PatientID", "PatientBirthDate", "..."] # Example - Use comprehensive list
                    dicom_meta = st.session_state.dicom_metadata
                    for tag_name, tag_value in dicom_meta.items():
                         if tag_name in PHI_TAGS_TO_EXCLUDE: continue
                         # Simplified formatting for example:
                         try:
                              display_value = str(tag_value)
                              if isinstance(tag_value, list): display_value = ", ".join(map(str,tag_value))
                              elif isinstance(tag_value, pydicom.uid.UID): display_value = tag_value.name
                              elif isinstance(tag_value, bytes): display_value = f"[Binary Data ({len(tag_value)} bytes)]"
                              if display_value and len(display_value) < 512:
                                   filtered_meta_str_list.append(f"{tag_name}: {display_value.strip()}")
                         except: pass # Ignore errors in this simplified example

                    outputs_for_report["DICOM Metadata (Filtered)"] = "\n".join(filtered_meta_str_list) if filtered_meta_str_list else "No non-excluded metadata found."
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