# main_app.py (Revision - Force Data URL for Canvas Background)

# --- Core Libraries ---
import io
import os
import uuid
import logging
import base64
import copy
from typing import Any, Dict, Optional, Tuple, List

# --- Streamlit ---
import streamlit as st

# --- Drawable Canvas ---
try:
    from streamlit_drawable_canvas import st_canvas
    # Attempt to get version, handle if __version__ is not defined directly
    try:
        from streamlit_drawable_canvas import __version__ as CANVAS_VERSION
    except ImportError:
        import pkg_resources
        try:
            CANVAS_VERSION = pkg_resources.get_distribution("streamlit-drawable-canvas").version
        except pkg_resources.DistributionNotFound:
            CANVAS_VERSION = "Unknown" # Fallback if pkg_resources also fails
except ImportError:
    st.error("CRITICAL ERROR: streamlit-drawable-canvas is not installed. `pip install streamlit-drawable-canvas`")
    st.stop()

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
# <<< --- Configure Streamlit Page (MUST BE FIRST st COMMAND) --- >>>
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="RadVision AI Advanced",
    layout="wide",
    page_icon="⚕️",
    initial_sidebar_state="expanded"
)

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

if pydicom is None: logger.info("Pydicom module not found. DICOM functionality disabled.")
else:
    logger.info(f"Pydicom Version: {PYDICOM_VERSION}")
    try: import pylibjpeg; logger.info("pylibjpeg found.")
    except ImportError: logger.warning("pylibjpeg not found. `pip install pylibjpeg pylibjpeg-libjpeg` for wider DICOM compatibility.")
    try: import gdcm; logger.info("python-gdcm found.")
    except ImportError: logger.warning("python-gdcm not found. `pip install python-gdcm` for wider DICOM compatibility.")

logger.info(f"--- App Start ---")
logger.info(f"Logging level set to {LOG_LEVEL}")
logger.info(f"Streamlit Version: {st.__version__}")
logger.info(f"Pillow (PIL) Version: {PIL_VERSION}")
logger.info(f"Streamlit Drawable Canvas Version: {CANVAS_VERSION}")


# ------------------------------------------------------------------------------
# <<< --- Monkey-Patch to Add image_to_url if Missing (Keep Active) --- >>>
# ------------------------------------------------------------------------------
import streamlit.elements.image as st_image
logger.debug("Checking for 'image_to_url' attribute in streamlit.elements.image")
if not hasattr(st_image, "image_to_url"):
    logger.warning("Attribute 'image_to_url' NOT FOUND in streamlit.elements.image. Applying monkey-patch.")
    def image_to_url_monkey_patch( # Simplified version focusing on PIL
        image: Any, width: int, clamp: bool, channels: str,
        output_format: str, image_id: str, allow_emoji: bool = False,
    ) -> str:
        patch_func_name = "image_to_url_monkey_patch" # For logging clarity
        if isinstance(image, Image.Image):
            try:
                buffered = io.BytesIO()
                # Determine format, default to PNG
                fmt = output_format.upper() if output_format else "PNG"
                if fmt not in ["PNG", "JPEG"]: fmt = "PNG" # Ensure valid format

                img_to_save = image
                # Convert only if necessary for the target save format
                if image.mode not in ['RGB', 'RGBA', 'L'] and fmt in ['PNG', 'JPEG']:
                     logger.debug(f"[{patch_func_name}] Converting mode {image.mode} to RGB for saving as {fmt}")
                     img_to_save = image.convert("RGB")
                elif image.mode == 'P' and fmt == 'PNG':
                     # Palette to RGBA usually works well for PNG
                     logger.debug(f"[{patch_func_name}] Converting mode P to RGBA for PNG saving.")
                     img_to_save = image.convert("RGBA")
                elif image.mode != 'RGB' and fmt == 'JPEG':
                     # JPEG requires RGB
                     logger.debug(f"[{patch_func_name}] Converting mode {image.mode} to RGB for JPEG saving.")
                     img_to_save = image.convert("RGB")

                img_to_save.save(buffered, format=fmt)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                data_url = f"data:image/{fmt.lower()};base64,{img_str}"
                logger.debug(f"[{patch_func_name}] Successfully created data URL (format: {fmt}).")
                return data_url
            except Exception as e:
                logger.error(f"[{patch_func_name}] Failed during image conversion/saving: {e}", exc_info=True)
                return "" # Return empty on error
        else:
            # This patch primarily targets PIL Images for st_canvas compatibility.
            logger.warning(f"[{patch_func_name}] Received non-PIL image type: {type(image)}. Cannot create data URL.")
            return "" # Cannot handle other types with this patch
    try:
        # Apply the defined function as the attribute
        st_image.image_to_url = image_to_url_monkey_patch
        logger.info("Monkey-patch for 'image_to_url' applied successfully.")
    except Exception as patch_apply_err:
        # Catch errors during the patching process itself (less likely)
        logger.error(f"FATAL: Failed to apply monkey-patch to streamlit.elements.image: {patch_apply_err}", exc_info=True)
else:
    logger.info("Attribute 'image_to_url' FOUND in streamlit.elements.image. No patch needed.")


# ------------------------------------------------------------------------------
# <<< --- Import Custom Utilities & Fallbacks --- >>>
# ------------------------------------------------------------------------------
try:
    from dicom_utils import parse_dicom, extract_dicom_metadata, dicom_to_image, get_default_wl
    from llm_interactions import run_initial_analysis, run_multimodal_qa, run_disease_analysis, estimate_ai_confidence
    from report_utils import generate_pdf_report_bytes
    from ui_components import display_dicom_metadata, dicom_wl_sliders
    logger.info("Successfully imported custom utility modules.")
    try: from hf_models import query_hf_vqa_inference_api, HF_VQA_MODEL_ID
    except ImportError:
        HF_VQA_MODEL_ID = "hf_model_not_found"; query_hf_vqa_inference_api = None # Define fallback function/variable
        logger.warning("hf_models.py not found or failed to import. HF VQA fallback disabled.")
except ImportError as import_error:
    st.error(f"CRITICAL ERROR importing helper modules ({import_error}). Ensure all .py files are present and dependencies installed."); logger.critical(f"Failed import: {import_error}", exc_info=True); st.stop()


# ------------------------------------------------------------------------------
# <<< --- Helper Image Conversion (safe_image_to_data_url) --- >>>
# ------------------------------------------------------------------------------
def safe_image_to_data_url(img: Image.Image) -> str:
    """Converts a PIL Image to a base64 Data URL (PNG format)."""
    if not isinstance(img, Image.Image):
        logger.warning(f"safe_image_to_data_url: Received non-PIL Image (type: {type(img)}).")
        return ""
    buffered = io.BytesIO()
    format = "PNG" # Default to PNG for broad compatibility & transparency
    try:
        img_to_save = img
        # Ensure compatibility with PNG save format (RGB, RGBA, L, P modes usually ok)
        # If PIL struggles with a specific mode for PNG, convert to RGBA as a safe bet
        if img.mode not in ['RGB', 'L', 'RGBA', 'P']:
            logger.debug(f"safe_image_to_data_url: Converting mode {img.mode} to RGBA for PNG.")
            img_to_save = img.convert('RGBA')
        # P mode is okay for PNG, no conversion needed here normally.
        # L mode is okay for PNG.
        # RGB/RGBA are okay.

        img_to_save.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{format.lower()};base64,{img_str}"
    except Exception as e:
        logger.error(f"Failed converting image to data URL using helper: {e}", exc_info=True)
        # Attempt fallback conversion to RGB before giving up
        try:
            logger.warning("safe_image_to_data_url: Retrying conversion to RGB for data URL.")
            img_rgb = img.convert('RGB')
            buffered_rgb = io.BytesIO()
            img_rgb.save(buffered_rgb, format="PNG") # Still save as PNG
            img_str_rgb = base64.b64encode(buffered_rgb.getvalue()).decode()
            return f"data:image/png;base64,{img_str_rgb}"
        except Exception as e2:
             logger.error(f"Fallback conversion to RGB also failed: {e2}", exc_info=True)
             return ""


# ------------------------------------------------------------------------------
# <<< --- Initialize Session State --- >>>
# ------------------------------------------------------------------------------
DEFAULT_STATE = {
    "uploaded_file_info": None, "raw_image_bytes": None, "is_dicom": False,
    "dicom_dataset": None, "dicom_metadata": {}, "processed_image": None,
    "display_image": None, "session_id": None, "history": [],
    "initial_analysis": "", "qa_answer": "", "disease_analysis": "",
    "confidence_score": "", "last_action": None, "pdf_report_bytes": None,
    "canvas_drawing": None, "roi_coords": None, 'current_display_wc': None,
    'current_display_ww': None,
}
# Initialize state keys if they don't exist
for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(default_value) if isinstance(default_value, (list, dict)) else default_value
# Ensure history is always a list (important for robustness)
if not isinstance(st.session_state.history, list):
    logger.warning("Session state 'history' was not a list. Resetting to empty list.")
    st.session_state.history = []
logger.debug("Session state initialized/verified.")


# ------------------------------------------------------------------------------
# <<< --- Page Title & Disclaimer --- >>>
# ------------------------------------------------------------------------------
st.title("⚕️ RadVision QA Advanced: AI-Assisted Image Analysis")
with st.expander("⚠️ Important Disclaimer & Usage Guide", expanded=False):
    st.warning("""
        **Disclaimer:** This tool is intended for research and educational purposes ONLY.
        **It is NOT a medical device and MUST NOT be used for clinical diagnosis, patient management, or any medical decision-making.**
        AI outputs may be inaccurate or incomplete. Always rely on qualified medical professionals for health-related interpretations and decisions.
    """)
    st.info("""
        **Quick Guide:** 1. Upload... 2. DICOM W/L... 3. Analyze... 4. Review... 5. Report...
    """)
st.markdown("---")


# =============================================================================
# === SIDEBAR CONTROLS ========================================================
# =============================================================================
with st.sidebar:
    # --- Logo ---
    logo_path = "assets/radvisionai-hero.jpeg"
    if os.path.exists(logo_path):
        st.image(logo_path, width=200, caption="RadVision AI")
        st.markdown("---")
    else:
        logger.warning(f"Sidebar logo not found at: {logo_path}")
        st.markdown("### RadVision AI"); st.markdown("---")

    # --- File Uploader ---
    st.header("Image Upload & Controls")
    ALLOWED_TYPES = ["jpg", "jpeg", "png", "dcm", "dicom"]
    uploaded_file = st.file_uploader(
        f"Upload Image ({', '.join(type.upper() for type in ALLOWED_TYPES)})",
        type=ALLOWED_TYPES,
        key="file_uploader_widget", # Persistent key for the widget
        accept_multiple_files=False,
        help="Select a standard image (JPG, PNG) or DICOM file."
    )

    # --- File Processing Logic ---
    if uploaded_file is not None:
        # --- File Change Detection ---
        try:
             file_mtime = getattr(uploaded_file, 'last_modified', None)
             if file_mtime is None: import hashlib; hasher = hashlib.md5(); hasher.update(uploaded_file.getvalue()); file_unique_id = hasher.hexdigest(); uploaded_file.seek(0); logger.warning("Using MD5 for change detection.")
             else: file_unique_id = str(file_mtime)
             new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{file_unique_id}"
        except Exception as file_info_err: logger.error(f"Err getting file info: {file_info_err}", exc_info=True); new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{str(uuid.uuid4())[:8]}" # Fallback ID

        # --- Process If New File ---
        if new_file_info != st.session_state.get("uploaded_file_info"):
            logger.info(f"New file detected: {uploaded_file.name} (Size: {uploaded_file.size})")
            st.toast(f"Processing '{uploaded_file.name}'...", icon="⏳")

            # --- Reset State for New File ---
            logger.debug("Resetting relevant session state variables...")
            preserve_keys = {"file_uploader_widget"} # Don't reset the uploader itself
            for key, default_value in DEFAULT_STATE.items():
                if key not in preserve_keys:
                    st.session_state[key] = copy.deepcopy(default_value) if isinstance(default_value, (list, dict)) else default_value
            # Set info for the newly uploaded file
            st.session_state.uploaded_file_info = new_file_info
            st.session_state.session_id = str(uuid.uuid4())[:8] # Generate new session ID
            logger.info(f"New Session ID generated: {st.session_state.session_id}")

            # --- Image Reading and Processing ---
            with st.spinner("🔬 Reading and processing image data..."):
                st.session_state.raw_image_bytes = None # Clear previous bytes
                temp_display_image = None
                temp_processed_image = None
                processing_successful = False # Flag to track success

                try:
                    logger.debug("Reading file into memory...")
                    st.session_state.raw_image_bytes = uploaded_file.getvalue()
                    if not st.session_state.raw_image_bytes: raise ValueError("Uploaded file is empty.")
                    logger.info(f"Read {len(st.session_state.raw_image_bytes)} bytes successfully.")

                    # --- Determine File Type (DICOM vs Standard) ---
                    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                    is_dicom_magic = (len(st.session_state.raw_image_bytes) > 132 and st.session_state.raw_image_bytes[128:132] == b'DICM')
                    st.session_state.is_dicom = (pydicom is not None and (file_ext in (".dcm", ".dicom") or "dicom" in uploaded_file.type.lower() or is_dicom_magic))
                    logger.info(f"File identified as DICOM: {st.session_state.is_dicom}")

                    # --- DICOM Processing Branch ---
                    if st.session_state.is_dicom:
                        logger.debug("Attempting DICOM processing...")
                        ds = None
                        try:
                            ds = parse_dicom(st.session_state.raw_image_bytes, uploaded_file.name)
                            st.session_state.dicom_dataset = ds
                        except Exception as e:
                            st.error(f"Error parsing DICOM file: {e}")
                            logger.error(f"DICOM parse failed for {uploaded_file.name}: {e}", exc_info=True)
                            ds = None # Ensure ds is None on failure

                        if ds:
                            logger.info("DICOM parsed successfully.")
                            st.session_state.dicom_metadata = extract_dicom_metadata(ds)
                            wc, ww = get_default_wl(ds)
                            st.session_state.current_display_wc, st.session_state.current_display_ww = wc, ww
                            logger.info(f"Extracted metadata and default W/L (WC={wc}, WW={ww}).")

                            # Generate images (display with W/L, processed raw)
                            temp_display_image = dicom_to_image(ds, wc, ww)
                            temp_processed_image = dicom_to_image(ds, None, None) # Raw pixel data image

                            if isinstance(temp_display_image, Image.Image) and isinstance(temp_processed_image, Image.Image):
                                # Allow L/RGB/RGBA initially, conversion handled later if needed by specific components
                                logger.info(f"DICOM images generated. Display mode: {temp_display_image.mode}, Processed mode: {temp_processed_image.mode}")
                                processing_successful = True
                            else:
                                st.error("Failed to generate displayable images from DICOM data.")
                                logger.error("dicom_to_image did not return valid PIL Images.")
                        elif pydicom is None: # Should have been caught earlier, but double check
                            st.error("Cannot process DICOM: 'pydicom' library is not available.")

                    # --- Standard Image Processing Branch ---
                    else:
                        logger.debug("Attempting standard image processing...")
                        try:
                            img = Image.open(io.BytesIO(st.session_state.raw_image_bytes))
                            img.load() # Load image data to ensure validity

                            temp_display_image = img.copy()
                            temp_processed_image = img.copy()

                            # Prepare display image: Convert only if problematic mode (e.g., P, CMYK)
                            # Allow L, RGB, RGBA initially as they are generally displayable
                            if temp_display_image.mode not in ['RGB', 'RGBA', 'L']:
                                logger.info(f"Converting display image from problematic mode {temp_display_image.mode} to RGB.")
                                temp_display_image = temp_display_image.convert("RGB")
                            else:
                                logger.info(f"Display image kept as mode {temp_display_image.mode}.")

                            # Prepare processed image: Convert P/RGBA to RGB for LLM consistency, keep L as is
                            if temp_processed_image.mode in ['P', 'RGBA']:
                                logger.info(f"Converting processed image from {temp_processed_image.mode} to RGB.")
                                temp_processed_image = temp_processed_image.convert("RGB")
                            elif temp_processed_image.mode == 'L':
                                logger.info("Processed image kept as Grayscale (L).")
                            # Add conversion for other modes if needed, e.g., CMYK -> RGB
                            elif temp_processed_image.mode not in ['RGB', 'L']:
                                 logger.info(f"Converting processed image from {temp_processed_image.mode} to RGB.")
                                 temp_processed_image = temp_processed_image.convert("RGB")

                            # Clear any previous DICOM state
                            st.session_state.dicom_dataset = None; st.session_state.dicom_metadata = {}; st.session_state.current_display_wc = None; st.session_state.current_display_ww = None
                            processing_successful = True
                            logger.info(f"Standard image prepared. Display mode: {temp_display_image.mode}, Processed mode: {temp_processed_image.mode}")

                        except UnidentifiedImageError:
                            st.error(f"Cannot identify image file format for '{uploaded_file.name}'. File may be corrupted or unsupported.")
                            logger.error(f"UnidentifiedImageError processing {uploaded_file.name}", exc_info=True)
                        except Exception as e:
                            st.error(f"An error occurred processing standard image: {e}")
                            logger.error(f"Standard image processing error for {uploaded_file.name}: {e}", exc_info=True)

                except Exception as e:
                    # Catch-all for errors during byte reading or type determination phase
                    st.error(f"A critical error occurred during file loading: {e}")
                    logger.critical(f"Outer file processing error: {e}", exc_info=True)
                    processing_successful = False # Ensure flag is false

                # --- Final Check and State Update ---
                logger.debug(f"Post-processing check: Success Flag={processing_successful}, Display Type={type(temp_display_image)}, Processed Type={type(temp_processed_image)}")
                if processing_successful and isinstance(temp_display_image, Image.Image) and isinstance(temp_processed_image, Image.Image):
                    # Assign the prepared images to session state
                    st.session_state.display_image = temp_display_image
                    st.session_state.processed_image = temp_processed_image

                    # Log final modes being stored
                    logger.info(f"**SUCCESS**: Assigning images to session state. Display Mode: {getattr(st.session_state.display_image, 'mode', 'N/A')}, Processed Mode: {getattr(st.session_state.processed_image, 'mode', 'N/A')}")

                    # Reset analysis-specific parts of state
                    st.session_state.roi_coords = None; st.session_state.canvas_drawing = None; st.session_state.initial_analysis = ""; st.session_state.qa_answer = ""; st.session_state.disease_analysis = ""; st.session_state.confidence_score = ""; st.session_state.pdf_report_bytes = None; st.session_state.history = []

                    st.success(f"✅ Image '{uploaded_file.name}' processed successfully!")
                    st.rerun() # Trigger UI update with the new image state
                else:
                    # Handle failure: Log details, show error, reset state
                    logger.critical("Image loading/processing pipeline failed or produced invalid results.")
                    if processing_successful: # Means logic thought it worked, but PIL objects invalid
                        st.error("❌ Processing seemed successful, but final image data is invalid.")
                        logger.error(f"Final Check Failure Detail: Display Type={type(temp_display_image)}, Processed Type={type(temp_processed_image)}")
                    else: # Error occurred earlier
                         st.error("❌ Image processing failed. Please check file format and integrity.")
                    # Reset state fully on failure
                    st.session_state.uploaded_file_info = None; st.session_state.raw_image_bytes = None; st.session_state.display_image = None; st.session_state.processed_image = None; st.session_state.dicom_dataset = None; st.session_state.dicom_metadata = {}; st.session_state.is_dicom = False; st.session_state.current_display_wc = None; st.session_state.current_display_ww = None
                    # Do not rerun on failure, let user see error.

    # --- DICOM W/L Controls ---
    st.markdown("---")
    # Show only if DICOM, pydicom available, dataset loaded, and display image exists
    if st.session_state.is_dicom and pydicom is not None and st.session_state.dicom_dataset and isinstance(st.session_state.get("display_image"), Image.Image):
        with st.expander("DICOM Window/Level", expanded=False):
             try:
                # Get slider values using the UI component
                wc_slider, ww_slider = dicom_wl_sliders(
                    st.session_state.dicom_dataset,
                    st.session_state.dicom_metadata,
                    initial_wc=st.session_state.current_display_wc,
                    initial_ww=st.session_state.current_display_ww
                )
                current_wc, current_ww = st.session_state.current_display_wc, st.session_state.current_display_ww

                # Check if sliders are valid and if values have changed significantly
                sliders_valid = (wc_slider is not None and ww_slider is not None)
                current_valid = (current_wc is not None and current_ww is not None)
                needs_update = sliders_valid and (
                    not current_valid or
                    abs(wc_slider - current_wc) > 1e-3 or abs(ww_slider - current_ww) > 1e-3
                )

                if needs_update:
                    logger.info(f"DICOM W/L Change Detected: Applying WC={wc_slider:.1f}, WW={ww_slider:.1f}")
                    with st.spinner("Applying new Window/Level settings..."):
                         # Regenerate display image with new settings
                         new_display_img = dicom_to_image(st.session_state.dicom_dataset, wc_slider, ww_slider)
                         if isinstance(new_display_img, Image.Image):
                              # Assign directly. The canvas will use this updated display_image.
                              logger.info(f"W/L adjusted image generated. Mode: {new_display_img.mode}")
                              st.session_state.display_image = new_display_img
                              # Update state with new W/L values
                              st.session_state.current_display_wc = wc_slider
                              st.session_state.current_display_ww = ww_slider
                              # Clear canvas drawing/ROI as the background changed significantly
                              st.session_state.canvas_drawing = None
                              st.session_state.roi_coords = None
                              logger.debug("DICOM W/L applied successfully, canvas drawing cleared, triggering rerun.")
                              st.rerun()
                         else:
                              st.error("Failed to apply new Window/Level settings.")
                              logger.error("dicom_to_image failed during W/L update.")
             except Exception as e:
                 st.error(f"Error in DICOM Window/Level controls: {e}")
                 logger.error(f"DICOM W/L slider interaction error: {e}", exc_info=True)
        st.markdown("---") # Separator after W/L controls
    elif st.session_state.is_dicom and pydicom is None:
        st.warning("DICOM file detected, but 'pydicom' library missing. W/L controls unavailable.")

    # --- AI Actions ---
    # Enable actions only if a valid display image is available
    if isinstance(st.session_state.get("display_image"), Image.Image):
        st.subheader("AI Actions")
        # --- Initial Analysis Button ---
        if st.button("▶️ Run Initial Analysis", key="analyze_btn", use_container_width=True, help="Get a general AI analysis."):
            st.session_state.last_action = "analyze"; logger.info("Sidebar: Action set to 'analyze'"); st.rerun()

        st.markdown("---")
        st.subheader("❓ Ask AI Question")
        # --- ROI Display & Clear ---
        if st.session_state.roi_coords:
            rc = st.session_state.roi_coords; st.info(f"✅ ROI: [X:{rc['left']}, Y:{rc['top']}, W:{rc['width']}, H:{rc['height']}]")
            if st.button("❌ Clear ROI", key="clear_roi_btn", use_container_width=True, help="Remove selected ROI."):
                st.session_state.roi_coords = None; st.session_state.canvas_drawing = None; logger.info("Sidebar: ROI cleared."); st.rerun()
        else: st.caption("ℹ️ Optionally, draw ROI on image viewer.")
        # --- Question Input & Ask Button ---
        question_input = st.text_area("Ask about image/ROI:", height=100, key="question_input_widget", placeholder="e.g., Any abnormalities in the selected area?", label_visibility="collapsed")
        if st.button("💬 Ask AI", key="ask_btn", use_container_width=True, help="Ask AI about image or ROI."):
            q = st.session_state.question_input_widget.strip()
            if q: st.session_state.last_action = "ask"; logger.info(f"Sidebar: Action set to 'ask'. Q: '{q[:50]}...'"); st.rerun()
            else: st.warning("Please enter a question."); logger.warning("Sidebar 'Ask AI': Empty question.")

        st.markdown("---")
        st.subheader("🎯 Focused Condition Analysis")
        # --- Disease Selection & Button ---
        DISEASE_OPTIONS = ["Pneumonia", "Lung Cancer", "Stroke", "Fracture", "Appendicitis", "Tuberculosis", "COVID-19", "Pulmonary Embolism", "Brain Tumor", "Arthritis", "Osteoporosis", "Cardiomegaly", "Aortic Aneurysm", "Bowel Obstruction", "Mass/Nodule", "Effusion", "Normal Variation"]
        disease_options = [""] + sorted(DISEASE_OPTIONS)
        disease_select = st.selectbox("Select Condition:", options=disease_options, key="disease_select_widget", help="Focus AI analysis on a specific condition.")
        if st.button("🩺 Run Condition Analysis", key="disease_btn", use_container_width=True, help="Analyze for selected condition."):
            d = st.session_state.disease_select_widget
            if d: st.session_state.last_action = "disease"; logger.info(f"Sidebar: Action set to 'disease'. Condition: '{d}'"); st.rerun()
            else: st.warning("Please select a condition."); logger.warning("Sidebar 'Condition Analysis': No condition selected.")

        st.markdown("---")
        # --- Confidence & Report Expander ---
        with st.expander("📊 Confidence & Report", expanded=True):
            can_estimate = bool(st.session_state.history or st.session_state.initial_analysis or st.session_state.disease_analysis)
            if st.button("📈 Estimate Confidence", key="confidence_btn", disabled=not can_estimate, use_container_width=True, help="Estimate AI confidence based on context."):
                st.session_state.last_action = "confidence"; logger.info("Sidebar: Action set to 'confidence'"); st.rerun()
            if not can_estimate: st.caption("Run analysis/QA first.")
            if st.button("📄 Generate PDF Data", key="generate_report_data_btn", use_container_width=True, help="Prepare data for PDF report."):
                st.session_state.last_action = "generate_report_data"; logger.info("Sidebar: Action set to 'generate_report_data'"); st.rerun()
            # --- PDF Download Button ---
            if st.session_state.pdf_report_bytes:
                fname = f"RadVisionAI_Report_{st.session_state.session_id or 'session'}.pdf"
                st.download_button(label="⬇️ Download PDF Report", data=st.session_state.pdf_report_bytes, file_name=fname, mime="application/pdf", key="download_pdf_button", use_container_width=True)
    else:
        # Message if no image loaded
        st.info("👈 Upload an image using the controls above to enable AI actions.")


# =============================================================================
# === MAIN PANEL DISPLAYS =====================================================
# =============================================================================
col1, col2 = st.columns([2, 3]) # Adjust ratio if needed

with col1:
    st.subheader("🖼️ Image Viewer")
    # Get the image intended for display from session state
    display_img_object = st.session_state.get("display_image")

    # Log the state right before trying to display
    logger.info(f"Main Panel Render: Checking display_image state. Type: {type(display_img_object)}, Is PIL: {isinstance(display_img_object, Image.Image)}, Mode: {getattr(display_img_object, 'mode', 'N/A')}")

    # Proceed only if we have a valid PIL Image object
    if isinstance(display_img_object, Image.Image):
        # Add explicit warning about checking browser console - VERY IMPORTANT FOR CANVAS ISSUES
        st.warning("ℹ️ **If the image viewer below appears blank or drawing fails, please check the Browser Developer Console (press F12) for JavaScript errors.**", icon="⚠️")

        bg_image_data_url = None # Initialize
        try:
            # --- FORCE DATA URL GENERATION ---
            # Use the patched function or the helper
            logger.debug("Attempting to generate Data URL for canvas background...")
            # Option 1: Try using the potentially patched st_image.image_to_url first
            data_url_generated = False
            if hasattr(st_image, "image_to_url"):
                try:
                     bg_image_data_url = st_image.image_to_url(
                         display_img_object,
                         width=-1, # Let PIL handle size based on actual image for URL gen
                         clamp=False,
                         channels="RGBA", # Prefer RGBA for data URL to preserve transparency if any
                         output_format="PNG", # Use PNG for broad compatibility
                         image_id=f"canvas_bg_{st.session_state.session_id or uuid.uuid4()}" # Unique ID
                     )
                     if bg_image_data_url and bg_image_data_url.startswith("data:image"):
                        logger.info(f"Generated data URL via st_image.image_to_url (Length: {len(bg_image_data_url)} chars)")
                        data_url_generated = True
                     else:
                        logger.warning(f"st_image.image_to_url returned invalid data: {bg_image_data_url[:100]}...")
                        bg_image_data_url = None # Reset if invalid
                except Exception as url_err:
                     logger.error(f"Failed to generate Data URL via st_image.image_to_url: {url_err}", exc_info=False) # Log less verbosely maybe
                     bg_image_data_url = None # Ensure it's None on error

            # Option 2: Fallback to our helper function if Option 1 failed or wasn't available
            if not data_url_generated:
                 logger.warning("Falling back to safe_image_to_data_url helper for canvas background.")
                 bg_image_data_url = safe_image_to_data_url(display_img_object)
                 if bg_image_data_url and bg_image_data_url.startswith("data:image"):
                    logger.info(f"Generated data URL via safe_image_to_data_url (Length: {len(bg_image_data_url)} chars)")
                    data_url_generated = True
                 else:
                    logger.error("Helper safe_image_to_data_url also failed to generate a valid URL.")
                    bg_image_data_url = None

            if not data_url_generated:
                 raise ValueError("Failed to generate a valid Data URL for the canvas background after all attempts.")
            # ---------------------------------

            # --- Calculate optimal canvas dimensions based on image aspect ratio ---
            MAX_CANVAS_WIDTH, MAX_CANVAS_HEIGHT = 700, 600
            img_w, img_h = display_img_object.size
            if img_w <= 0 or img_h <= 0: raise ValueError(f"Invalid image dimensions: {img_w}x{img_h}")

            aspect_ratio = img_w / img_h
            canvas_w = min(img_w, MAX_CANVAS_WIDTH)
            canvas_h = int(canvas_w / aspect_ratio)

            # Adjust if height exceeds max
            if canvas_h > MAX_CANVAS_HEIGHT:
                canvas_h = MAX_CANVAS_HEIGHT
                canvas_w = int(canvas_h * aspect_ratio)
            # Ensure minimum reasonable size
            canvas_w, canvas_h = max(int(canvas_w), 150), max(int(canvas_h), 150)
            logger.info(f"Canvas Rendering: Calculated dims W={canvas_w}, H={canvas_h} for image size {img_w}x{img_h}")

            st.caption("Click and drag on the image below to select a Region of Interest (ROI).")

            # --- Get current drawing state for persistence ---
            initial_drawing = st.session_state.get("canvas_drawing")
            if initial_drawing and not isinstance(initial_drawing, dict):
                logger.warning(f"Invalid initial_drawing state found (type {type(initial_drawing)}), resetting.")
                initial_drawing = None # Reset if format is wrong

            # --- Render the Drawable Canvas ---
            # ***** CHANGE: Pass the generated Data URL string *****
            logger.info(f"Rendering st_canvas with background using generated Data URL. Initial drawing: {'Set' if initial_drawing else 'None'}")
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.2)",  # ROI fill
                stroke_width=2,                      # ROI border width
                stroke_color="rgba(220, 50, 50, 0.9)",# ROI border color
                background_image=bg_image_data_url,  # *** Pass the Data URL string ***
                update_streamlit=True,               # Send drawing updates back
                height=canvas_h,                     # Calculated height
                width=canvas_w,                      # Calculated width
                drawing_mode="rect",                 # Only allow rectangles
                initial_drawing=initial_drawing,     # Restore previous drawing
                key="drawable_canvas",               # Unique key
            )
            # Log success *after* the call, even if background rendering fails internally in JS
            logger.info("st_canvas Python call completed.")

            # --- Process Canvas Result (ROI Logic) ---
            if canvas_result and canvas_result.json_data is not None:
                # Store the latest drawing state back into session state for persistence
                st.session_state.canvas_drawing = canvas_result.json_data
                current_roi_state = st.session_state.get("roi_coords") # Get existing ROI

                # Check if user drew or modified a rectangle
                if canvas_result.json_data.get("objects"):
                    last_object = canvas_result.json_data["objects"][-1] # Get the latest object
                    if last_object["type"] == "rect":
                        # Calculate ROI coordinates relative to original image size
                        canvas_left, canvas_top = int(last_object["left"]), int(last_object["top"])
                        canvas_rect_width = int(last_object["width"] * last_object.get("scaleX", 1)) # Use rect width/height from object
                        canvas_rect_height = int(last_object["height"] * last_object.get("scaleY", 1))

                        # Scaling factors based on actual image size vs canvas render size
                        scale_x = img_w / canvas_w; scale_y = img_h / canvas_h
                        original_left = max(0, int(canvas_left * scale_x))
                        original_top = max(0, int(canvas_top * scale_y))
                        original_width = int(canvas_rect_width * scale_x)
                        original_height = int(canvas_rect_height * scale_y)

                        # Ensure coordinates are within image bounds
                        original_right = min(img_w, original_left + original_width)
                        original_bottom = min(img_h, original_top + original_height)
                        final_width = max(0, original_right - original_left)
                        final_height = max(0, original_bottom - original_top)

                        # Define minimum ROI size
                        MIN_ROI_DIM = 10
                        if final_width >= MIN_ROI_DIM and final_height >= MIN_ROI_DIM:
                            new_roi_dict = {"left": original_left, "top": original_top, "width": final_width, "height": final_height}
                            # Update state and rerun only if ROI actually changed
                            if current_roi_state != new_roi_dict:
                                st.session_state.roi_coords = new_roi_dict
                                logger.info(f"ROI Updated: {new_roi_dict}")
                                st.rerun() # Rerun to update sidebar info etc.
                        elif current_roi_state is not None:
                             # If new drawing is too small, clear existing ROI
                             logger.info("New drawing too small, clearing existing ROI.")
                             st.session_state.roi_coords = None; st.rerun()

                # If no objects exist anymore (e.g., user deleted rect), clear ROI state
                elif not canvas_result.json_data.get("objects") and current_roi_state is not None:
                    logger.info("Canvas drawing cleared, removing ROI state.")
                    st.session_state.roi_coords = None; st.rerun()

        except ValueError as ve:
             st.error(f"Image Viewer Error: {ve}")
             logger.error(f"Error preparing for canvas: {ve}", exc_info=True)
        except Exception as canvas_error:
            # Catch errors during st_canvas call or ROI processing
            st.error(f"An error occurred with the image viewer/canvas: {canvas_error}")
            logger.error(f"st_canvas rendering or processing failed: {canvas_error}", exc_info=True)
            # Remind user about console check again
            st.warning("Drawing might be unavailable. **Check Browser Console (F12) for details.**")

        # --- Display DICOM Metadata (if applicable) ---
        if st.session_state.is_dicom and st.session_state.dicom_metadata:
            st.markdown("---"); st.subheader("📄 DICOM Metadata")
            if pydicom is None: st.warning("'pydicom' library missing, cannot display metadata.")
            else: logger.debug("Displaying DICOM metadata."); display_dicom_metadata(st.session_state.dicom_metadata)

    # --- Fallback Placeholder ---
    # Show only if no image has been successfully processed and assigned yet
    elif not st.session_state.get("display_image"): # Check if display_image is None/missing
        logger.debug("Image Viewer: No valid display_image available.")
        st.markdown("---")
        if st.session_state.get("uploaded_file_info"): # Check if upload was attempted but failed
             st.warning("Image processing failed. Please check the file or try another.")
        else: # No upload attempted yet
             st.info("Upload an image file using the sidebar to begin.")
        # Display a simple placeholder box
        st.markdown(
            """<div style='height: 400px; border: 2px dashed #ccc; display: flex; align-items: center; justify-content: center; text-align: center; color: #aaa; font-style: italic; padding: 20px; border-radius: 8px;'>Image Display Area</div>""",
            unsafe_allow_html=True
        )

# --- Column 2: Analysis Results Tabs ---
with col2:
    st.subheader("📊 Analysis & Results")
    tab_titles = ["🔬 Initial Analysis", "💬 Q&A History", "🩺 Disease Focus", "📈 Confidence"]
    tabs = st.tabs(tab_titles)
    # --- Tab Contents ---
    with tabs[0]: st.text_area("Overall Findings", value=st.session_state.initial_analysis or "No initial analysis performed yet.", height=450, key="output_initial", disabled=True, help="General analysis results.")
    with tabs[1]:
        st.text_area("Latest AI Answer", value=st.session_state.qa_answer or "Ask a question using the sidebar.", height=200, key="output_qa", disabled=True, help="Latest answer from 'Ask AI'.")
        st.markdown("---")
        if st.session_state.history:
             with st.expander("Full Conversation History", expanded=True):
                 for i, (q, a) in enumerate(reversed(st.session_state.history)): # Newest first
                     st.markdown(f"**You ({len(st.session_state.history)-i}):**"); st.caption(f"{q}")
                     st.markdown(f"**AI ({len(st.session_state.history)-i}):**"); st.markdown(a, unsafe_allow_html=True)
                     if i < len(st.session_state.history) - 1: st.markdown("---")
        else: st.caption("No Q&A history for this session yet.")
    with tabs[2]: st.text_area("Focused Condition Findings", value=st.session_state.disease_analysis or "No focused condition analysis performed.", height=450, key="output_disease", disabled=True, help="Results from 'Run Condition Analysis'.")
    with tabs[3]: st.text_area("AI Confidence Estimation", value=st.session_state.confidence_score or "No confidence estimation performed.", height=450, key="output_confidence", disabled=True, help="Estimated AI confidence level.")


# =============================================================================
# === ACTION HANDLING LOGIC (No changes needed here, kept as is) =============
# =============================================================================
current_action: Optional[str] = st.session_state.get("last_action")
if current_action:
    logger.info(f"ACTION HANDLER: Triggered action '{current_action}'")
    # --- Prerequisites Check ---
    processed_image = st.session_state.get("processed_image")
    session_id = st.session_state.get("session_id")
    roi_coords = st.session_state.get("roi_coords")
    conversation_history = st.session_state.history # Assumed to be list by init

    # Validate image (except for report generation which uses display_image)
    if current_action != "generate_report_data":
        if not isinstance(processed_image, Image.Image):
            st.error(f"Action '{current_action}' requires a valid processed image. Please re-upload."); logger.error(f"Action '{current_action}' aborted: Invalid processed_image."); st.session_state.last_action = None; st.stop()
    # Validate session ID
    if not session_id: st.error(f"Action '{current_action}' requires a Session ID. Please re-upload."); logger.error(f"Action '{current_action}' aborted: Missing Session ID."); st.session_state.last_action = None; st.stop()
    # Ensure history is list (redundant check if init is correct)
    if not isinstance(conversation_history, list): conversation_history = []; st.session_state.history = conversation_history; logger.warning("History reset in action handler.")

    # Prepare common arguments
    image_for_llm = processed_image
    roi_info_str = " (with ROI)" if roi_coords else ""

    # --- Execute Actions ---
    try:
        if current_action == "analyze":
            st.info(f"🔬 Running initial analysis{roi_info_str}...")
            with st.spinner("AI analyzing..."): res = run_initial_analysis(image_for_llm) # Pass roi=roi_coords if supported
            st.session_state.initial_analysis = res; st.success("Analysis finished.")
        elif current_action == "ask":
            q = st.session_state.question_input_widget.strip();
            if not q: st.warning("Question cannot be empty."); logger.warning("Ask action: empty question.")
            else:
                st.info(f"❓ Asking AI{roi_info_str}: '{q[:70]}...'"); st.session_state.qa_answer = "" # Clear previous
                with st.spinner("AI processing question..."): ans, ok = run_multimodal_qa(image_for_llm, q, conversation_history, roi_coords)
                if ok: st.session_state.qa_answer = ans; st.session_state.history.append((q, ans)); st.success("AI answered.")
                else: # Fallback Logic
                    st.error(f"Primary AI failed: {ans}"); logger.warning(f"Primary AI failed: {ans}"); st.session_state.qa_answer = f"**[Primary AI Error]** {ans}\n\n---\n"
                    # Check Fallback availability
                    hf_ok = (HF_VQA_MODEL_ID != "hf_model_not_found" and query_hf_vqa_inference_api is not None and os.environ.get("HF_API_TOKEN"))
                    if hf_ok:
                         fb_model_name = HF_VQA_MODEL_ID.split('/')[-1]
                         st.info(f"Attempting fallback ({fb_model_name})..."); logger.info(f"Trying fallback: {HF_VQA_MODEL_ID}")
                         with st.spinner(f"Asking fallback ({fb_model_name})..."): fb_ans, fb_ok = query_hf_vqa_inference_api(image_for_llm, q, roi_coords)
                         if fb_ok: fb_disp = f"**[Fallback: {fb_model_name}]**\n\n{fb_ans}"; st.session_state.qa_answer += fb_disp; st.session_state.history.append((f"[Fallback] {q}", fb_disp)); st.success(f"Fallback ({fb_model_name}) answered.")
                         else: fb_err = f"Fallback ({fb_model_name}) failed: {fb_ans}"; st.session_state.qa_answer += f"**[Fallback Failed]** {fb_err}"; st.error(fb_err); logger.error(f"HF fallback failed: {fb_ans}")
                    else: fb_msg = "Fallback AI unavailable."; st.session_state.qa_answer += f"**[Fallback Unavailable]** {fb_msg}"; st.warning(fb_msg); logger.warning("HF fallback skipped.")
        elif current_action == "disease":
            d = st.session_state.disease_select_widget
            if not d: st.warning("Please select a condition first."); logger.warning("Disease action: no condition selected.")
            else:
                st.info(f"🩺 Analyzing for '{d}'{roi_info_str}...")
                with st.spinner(f"AI assessing for '{d}'..."): res = run_disease_analysis(image_for_llm, d, roi_coords)
                st.session_state.disease_analysis = res; st.success(f"Analysis for '{d}' complete.")
        elif current_action == "confidence":
            context_exists = bool(conversation_history or st.session_state.initial_analysis or st.session_state.disease_analysis)
            if not context_exists: st.warning("Cannot estimate confidence without prior analysis/Q&A."); logger.warning("Confidence action: no context.")
            else:
                st.info(f"📊 Estimating AI confidence{roi_info_str}...")
                with st.spinner("Calculating confidence score..."): res = estimate_ai_confidence(image_for_llm, conversation_history, st.session_state.initial_analysis, st.session_state.disease_analysis, roi_coords)
                st.session_state.confidence_score = res; st.success("Confidence estimation complete.")
        elif current_action == "generate_report_data":
            st.info("📄 Preparing PDF report data..."); st.session_state.pdf_report_bytes = None
            img_rep = st.session_state.get("display_image") # Use display image for visual consistency in report
            if not isinstance(img_rep, Image.Image): st.error("Cannot generate report: Display image is invalid."); logger.error("PDF generation failed: Invalid display_image.")
            else:
                 img_final = img_rep.copy() # Work on a copy
                 # Draw ROI on report image if it exists
                 if roi_coords:
                      try:
                          if img_final.mode not in ['RGB', 'RGBA']: img_final = img_final.convert("RGB") # Ensure drawable mode
                          draw = ImageDraw.Draw(img_final); x0,y0,w,h = roi_coords['left'],roi_coords['top'],roi_coords['width'],roi_coords['height']; draw.rectangle([x0,y0,x0+w,y0+h], outline="red", width=3); logger.info("Drew ROI on report image.")
                      except Exception as draw_e: logger.error(f"Failed drawing ROI on report image: {draw_e}", exc_info=True); st.warning("Could not draw ROI on report image.")
                 # Gather report text data
                 qa_hist_str = "\n\n".join([f"Q: {q}\nA: {a}" for q,a in conversation_history]) if conversation_history else "N/A"
                 outputs = { "Session ID": session_id or "N/A", "Preliminary Analysis": st.session_state.initial_analysis or "N/P", "Q&A History": qa_hist_str, "Condition Analysis": st.session_state.disease_analysis or "N/P", "Confidence": st.session_state.confidence_score or "N/E" }
                 # Add selected DICOM metadata if available
                 if st.session_state.is_dicom and st.session_state.dicom_metadata:
                      filtered_meta = { k: v for k, v in st.session_state.dicom_metadata.items() if k in ["Patient Name", "Patient ID", "Study Date", "Modality", "Study Description", "Window Center", "Window Width"]} # Example filter + WC/WW
                      # Format WC/WW if they exist from state
                      if st.session_state.current_display_wc is not None: filtered_meta["Current Window Center"] = f"{st.session_state.current_display_wc:.1f}"
                      if st.session_state.current_display_ww is not None: filtered_meta["Current Window Width"] = f"{st.session_state.current_display_ww:.1f}"
                      outputs["DICOM Metadata (Selected)"] = "\n".join([f"{k}: {v}" for k,v in filtered_meta.items()]) if filtered_meta else "No relevant metadata found."
                 # Generate PDF bytes
                 with st.spinner("🎨 Generating PDF document..."): pdf_bytes = generate_pdf_report_bytes(session_id, img_final, outputs)
                 if pdf_bytes: st.session_state.pdf_report_bytes = pdf_bytes; st.success("✅ PDF data ready for download!"); logger.info("PDF generation successful.")
                 else: st.error("❌ Failed to generate PDF report."); logger.error("PDF generation failed: generate_pdf_report_bytes returned None.")
        else:
            st.warning(f"Unknown action requested: '{current_action}'."); logger.warning(f"Handler encountered unknown action: '{current_action}'")

    except Exception as e:
        # Catch-all for errors during action execution
        st.error(f"An unexpected error occurred during action '{current_action}': {e}")
        logger.critical(f"Critical error during action '{current_action}': {e}", exc_info=True)
    finally:
        # Cleanup: Clear the action flag and rerun UI
        st.session_state.last_action = None
        logger.debug(f"Action '{current_action}' handler finished.")
        st.rerun()


# --- Footer ---
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced | Session: {st.session_state.get('session_id', 'N/A')} | v(dev)")
logger.info("--- App Render Cycle Complete ---")