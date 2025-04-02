# main_app.py (Revision - Removed Debug st.image Block)

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
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper() # Default back to INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

if pydicom is None: logger.error("Pydicom module not found. DICOM functionality disabled.")
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
# Monkey-Patch (Optional)
# ------------------------------------------------------------------------------
import streamlit.elements.image as st_image
if not hasattr(st_image, "image_to_url"):
    def image_to_url_monkey_patch(img_obj: Any, *args, **kwargs) -> str:
        if isinstance(img_obj, Image.Image):
            buffered = io.BytesIO(); format = "PNG"
            try:
                img_to_save = img_obj
                if img_obj.mode not in ['RGB', 'L', 'RGBA']: img_to_save = img_obj.convert('RGB')
                elif img_obj.mode == 'P': img_to_save = img_obj.convert('RGBA')
                img_to_save.save(buffered, format=format)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/{format.lower()};base64,{img_str}"
            except Exception as e: logger.error(f"Monkey-patch image_to_url failed: {e}", exc_info=True); return ""
        else: logger.warning(f"Monkey-patch image_to_url: Unsupported type {type(img_obj)}"); return ""
    st_image.image_to_url = image_to_url_monkey_patch
    logging.info("Applied monkey-patch for streamlit.elements.image.image_to_url (if missing)")


# ------------------------------------------------------------------------------
# <<< --- Import Custom Utilities & Fallbacks --- >>>
# ------------------------------------------------------------------------------
try:
    # Assume these helper modules exist in the same directory or are installable
    # Fallback functions are defined inline if imports fail
    try: from dicom_utils import parse_dicom, extract_dicom_metadata, dicom_to_image, get_default_wl
    except ImportError:
        logger.error("dicom_utils not found. Using basic fallbacks.")
        def parse_dicom(data, fname): raise NotImplementedError("DICOM parsing unavailable.")
        def extract_dicom_metadata(ds): return {"Error": "Metadata extraction unavailable."}
        def dicom_to_image(ds, wc, ww): return None # Indicate failure
        def get_default_wl(ds): return None, None
    try: from llm_interactions import run_initial_analysis, run_multimodal_qa, run_disease_analysis, estimate_ai_confidence
    except ImportError:
        logger.error("llm_interactions not found. Using basic fallbacks.")
        def run_initial_analysis(img, roi=None): return "LLM analysis unavailable."
        def run_multimodal_qa(img, q, hist, roi=None): return "LLM Q&A unavailable.", False
        def run_disease_analysis(img, disease, roi=None): return "LLM disease analysis unavailable."
        def estimate_ai_confidence(img, hist, init, disease, roi=None): return "Confidence estimation unavailable."
    try: from report_utils import generate_pdf_report_bytes
    except ImportError:
        logger.error("report_utils not found. Using basic fallbacks.")
        def generate_pdf_report_bytes(sid, img, outputs): return None # Indicate failure
    try: from ui_components import display_dicom_metadata, dicom_wl_sliders
    except ImportError:
        logger.error("ui_components not found. Using basic fallbacks.")
        def display_dicom_metadata(metadata): st.json({"Error": "Metadata display unavailable.", **metadata})
        def dicom_wl_sliders(ds, meta, initial_wc, initial_ww):
            st.warning("DICOM W/L controls unavailable.")
            return initial_wc, initial_ww
    logger.info("Attempted import of custom utility modules.")
    try: from hf_models import query_hf_vqa_inference_api, HF_VQA_MODEL_ID
    except ImportError:
        HF_VQA_MODEL_ID = "hf_model_not_found"
        def query_hf_vqa_inference_api(img: Image.Image, question: str, roi: Optional[Dict] = None) -> Tuple[str, bool]: return "[Fallback Unavailable] HF module not found.", False
        logger.warning("hf_models.py not found. HF VQA fallback disabled.")
except Exception as import_error: # Catch any other unexpected import error
    st.error(f"CRITICAL ERROR importing helpers ({import_error}). Ensure all .py files present or dependencies installed."); logger.critical(f"Failed import: {import_error}", exc_info=True); st.stop()

# --- Helper Image Conversion ---
def safe_image_to_data_url(img: Image.Image) -> str:
    if not isinstance(img, Image.Image): logger.warning(f"safe_image_to_data_url: Not PIL Image (type: {type(img)})."); return ""
    buffered = io.BytesIO(); format = "PNG"
    try:
        img_to_save = img
        # Ensure compatible mode for saving to PNG in base64
        if img.mode not in ['RGB', 'L', 'RGBA']: img_to_save = img.convert('RGB')
        elif img.mode == 'P': img_to_save = img.convert('RGBA') # Convert indexed palette to RGBA
        img_to_save.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{format.lower()};base64,{img_str}"
    except Exception as e: logger.error(f"Failed converting image to data URL: {e}", exc_info=True); return ""

# ------------------------------------------------------------------------------
# Initialize Session State
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
for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        # Use deepcopy for mutable defaults like lists/dicts to avoid shared references
        st.session_state[key] = copy.deepcopy(default_value) if isinstance(default_value, (list, dict)) else default_value
# Ensure history is always a list upon initialization or recovery
if not isinstance(st.session_state.history, list): st.session_state.history = []
logger.debug("Session state initialized.")

# ------------------------------------------------------------------------------
# Page Title & Disclaimer
# ------------------------------------------------------------------------------
st.title("⚕️ RadVision QA Advanced: AI-Assisted Image Analysis")
with st.expander("⚠️ Important Disclaimer & Usage Guide", expanded=False):
    st.warning("""
        **Disclaimer:** This tool is intended for research and educational purposes ONLY.
        **It is NOT a medical device and MUST NOT be used for clinical diagnosis, patient management, or any medical decision-making.**
        AI outputs may be inaccurate or incomplete. Always rely on qualified medical professionals for health-related interpretations and decisions.
    """)
    st.info("""
        **Quick Guide:**
        1.  **Upload:** Use the sidebar to upload a JPG, PNG, or DICOM file.
        2.  **DICOM W/L (Optional):** If a DICOM file is uploaded, adjust Window/Level settings in the sidebar for optimal viewing.
        3.  **Analyze:** Use the AI Actions in the sidebar:
            *   `Run Initial Analysis` for a general overview.
            *   Draw a Rectangle ROI (Region of Interest) on the image viewer.
            *   `Ask AI` a question about the image or the selected ROI.
            *   `Run Condition Analysis` to focus the AI on a specific potential condition.
        4.  **Review:** Check the results in the 'Analysis & Results' tabs.
        5.  **Report (Optional):** Estimate confidence and generate a non-clinical summary PDF.
    """)
st.markdown("---")

# =============================================================================
# === SIDEBAR CONTROLS ========================================================
# =============================================================================
with st.sidebar:
    # Logo
    logo_path = "assets/radvisionai-hero.jpeg" # Consider making this configurable
    if os.path.exists(logo_path):
        st.image(logo_path, width=200, caption="RadVision AI")
        st.markdown("---")
    else:
        logger.warning(f"Sidebar logo not found at: {logo_path}")
        st.markdown("### RadVision AI") # Fallback text if logo missing
        st.markdown("---")

    st.header("Image Upload & Controls")
    # Define allowed file types
    ALLOWED_TYPES = ["jpg", "jpeg", "png", "dcm", "dicom"]
    uploaded_file = st.file_uploader(
        f"Upload Image ({', '.join(type.upper() for type in ALLOWED_TYPES)})",
        type=ALLOWED_TYPES,
        key="file_uploader_widget",
        accept_multiple_files=False,
        help="Select a standard image file (JPG, PNG) or a medical DICOM file (.dcm, .dicom)."
    )

    # --- File Processing Logic ---
    if uploaded_file is not None:
        # Simple Change Detection: Use name and size. More robust might use hash.
        try:
             # Attempt to get modification time; fall back to hashing if unavailable (e.g., BytesIO in tests)
             file_mtime = getattr(uploaded_file, 'last_modified', None)
             if file_mtime is None: # Fallback for streams without mtime
                 import hashlib; hasher = hashlib.md5(); hasher.update(uploaded_file.getvalue()); file_unique_id = hasher.hexdigest(); uploaded_file.seek(0); # Read, hash, reset pointer
                 logger.warning("File modification time unavailable, using MD5 hash for change detection.")
             else:
                 file_unique_id = str(file_mtime) # Use modification time if available
             # Combine name, size, and unique ID for change detection key
             new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{file_unique_id}"
        except Exception as file_info_err:
             logger.error(f"Error getting file info/hash: {file_info_err}", exc_info=True)
             # Fallback unique identifier if all else fails
             new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{str(uuid.uuid4())[:8]}"

        # Check if the uploaded file is different from the one currently in state
        if new_file_info != st.session_state.get("uploaded_file_info"):
            logger.info(f"New file detected: {uploaded_file.name} (Size: {uploaded_file.size})")
            st.toast(f"Processing '{uploaded_file.name}'...", icon="⏳")

            # Reset relevant session state variables for the new file
            logger.debug("Resetting state for new file...")
            preserve_keys = {"file_uploader_widget"} # Keep the uploader state itself
            for key, default_value in DEFAULT_STATE.items():
                if key not in preserve_keys:
                    # Deepcopy for mutable types to prevent aliasing issues
                    st.session_state[key] = copy.deepcopy(default_value) if isinstance(default_value, (list, dict)) else default_value

            # Store new file info and generate a unique session ID for this file upload instance
            st.session_state.uploaded_file_info = new_file_info
            st.session_state.session_id = str(uuid.uuid4())[:8] # Short UUID for session tracking
            logger.info(f"New Session ID: {st.session_state.session_id}")

            # --- Image Processing ---
            with st.spinner("🔬 Processing image data..."):
                st.session_state.raw_image_bytes = None
                temp_display_image = None
                temp_processed_image = None
                processing_successful = False

                try:
                    logger.debug("Reading file bytes...")
                    st.session_state.raw_image_bytes = uploaded_file.getvalue()
                    if not st.session_state.raw_image_bytes:
                        raise ValueError("Uploaded file appears to be empty.")
                    logger.info(f"Successfully read {len(st.session_state.raw_image_bytes)} bytes from '{uploaded_file.name}'.")

                    # --- Determine File Type (DICOM vs. Standard Image) ---
                    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                    # Basic DICOM magic number check (DICM at byte offset 128)
                    is_dicom_magic = (len(st.session_state.raw_image_bytes) > 132 and
                                      st.session_state.raw_image_bytes[128:132] == b'DICM')
                    # Consider it DICOM if pydicom is available AND extension matches OR magic number present OR mime type hints at it
                    st.session_state.is_dicom = (
                        pydicom is not None and (
                            file_ext in (".dcm", ".dicom") or
                            "dicom" in uploaded_file.type.lower() or
                            is_dicom_magic
                        )
                    )
                    logger.info(f"File identified as DICOM: {st.session_state.is_dicom}")

                    # --- DICOM Processing Branch ---
                    if st.session_state.is_dicom:
                        logger.debug("Attempting DICOM processing...")
                        dicom_dataset = None
                        try:
                            # Use the utility function to parse the DICOM data
                            dicom_dataset = parse_dicom(st.session_state.raw_image_bytes, uploaded_file.name)
                            st.session_state.dicom_dataset = dicom_dataset # Store the parsed dataset
                        except pydicom.errors.InvalidDicomError as e:
                            st.error(f"Invalid DICOM file: {e}")
                            logger.error(f"InvalidDicomError during parsing: {e}", exc_info=True)
                            dicom_dataset = None # Ensure dataset is None on error
                        except NotImplementedError as e: # Catch fallback error
                            st.error(f"DICOM processing unavailable: {e}")
                            logger.error(f"DICOM processing function not available: {e}")
                            dicom_dataset = None
                        except Exception as e:
                            st.error(f"An unexpected error occurred while parsing DICOM: {e}")
                            logger.error(f"DICOM parse failed unexpectedly: {e}", exc_info=True)
                            dicom_dataset = None

                        if dicom_dataset:
                            logger.info("DICOM file parsed successfully.")
                            # Extract relevant metadata using the utility function
                            st.session_state.dicom_metadata = extract_dicom_metadata(dicom_dataset)
                            # Get default Window/Level values
                            default_wc, default_ww = get_default_wl(dicom_dataset)
                            st.session_state.current_display_wc = default_wc
                            st.session_state.current_display_ww = default_ww
                            logger.info(f"Default DICOM W/L values: WC={default_wc}, WW={default_ww}")

                            # Generate images: one for display (with initial W/L), one processed (raw pixel data for analysis)
                            temp_display_image = dicom_to_image(dicom_dataset, default_wc, default_ww)
                            temp_processed_image = dicom_to_image(dicom_dataset, None, None) # Get raw pixels image

                            # Validate that image generation was successful
                            if isinstance(temp_display_image, Image.Image) and isinstance(temp_processed_image, Image.Image):
                                processing_successful = True
                                logger.info("Successfully generated display and processed images from DICOM.")
                            else:
                                st.error("Failed to generate images from DICOM data. Pixel data might be missing or unsupported.")
                                logger.error("dicom_to_image did not return valid PIL Images.")
                        # Handle case where pydicom is missing entirely
                        elif pydicom is None:
                            st.error("Cannot process DICOM file: The 'pydicom' library is not installed.")
                            logger.error("DICOM processing skipped because pydicom is None.")

                    # --- Standard Image Processing Branch ---
                    else:
                        logger.debug("Attempting standard image processing...")
                        try:
                            # Open image using Pillow from bytes
                            img = Image.open(io.BytesIO(st.session_state.raw_image_bytes))
                            logger.info(f"Image opened successfully with Pillow. Format: {img.format}, Mode: {img.mode}, Size: {img.size}")

                            # Ensure images are in a usable format (e.g., RGB for display/analysis)
                            # Keep copies to avoid modifying the original Pillow object unintentionally
                            temp_display_image = img.copy()
                            temp_processed_image = img.copy()

                            # Convert display image to RGB if needed (common requirement for display)
                            if temp_display_image.mode != 'RGB':
                                logger.info(f"Converting display image from {temp_display_image.mode} to RGB.")
                                temp_display_image = temp_display_image.convert("RGB")

                            # Convert processed image to RGB if it's Paletted or has Alpha (simpler for some models)
                            if temp_processed_image.mode in ['P', 'RGBA']:
                                logger.info(f"Converting processed image from {temp_processed_image.mode} to RGB.")
                                temp_processed_image = temp_processed_image.convert("RGB")
                            elif temp_processed_image.mode == 'L': # Grayscale is often fine, log it.
                                logger.info("Processed image is Grayscale (L mode).")


                            # Reset DICOM-specific state variables
                            st.session_state.dicom_dataset = None
                            st.session_state.dicom_metadata = {}
                            st.session_state.current_display_wc = None
                            st.session_state.current_display_ww = None

                            processing_successful = True
                            logger.info("Successfully generated display and processed images from standard image file.")

                        except UnidentifiedImageError:
                            st.error(f"Cannot identify image format for '{uploaded_file.name}'. File may be corrupted or unsupported.")
                            logger.error(f"UnidentifiedImageError for file: {uploaded_file.name}", exc_info=True)
                        except Exception as e:
                            st.error(f"An error occurred while processing the image: {e}")
                            logger.error(f"Standard image processing failed: {e}", exc_info=True)

                except Exception as e:
                    # Catch-all for errors during byte reading or type determination
                    st.error(f"A critical error occurred during file processing: {e}")
                    logger.critical(f"Outer file processing error: {e}", exc_info=True)
                    processing_successful = False

                # --- Final Check and State Update ---
                logger.debug(f"Final Check: processing_successful={processing_successful}, display PIL={isinstance(temp_display_image, Image.Image)}, processed PIL={isinstance(temp_processed_image, Image.Image)}")
                if processing_successful and isinstance(temp_display_image, Image.Image) and isinstance(temp_processed_image, Image.Image):
                     # Final check to ensure display image is RGB (st_canvas prefers RGB)
                    if temp_display_image.mode != 'RGB':
                        try:
                            logger.warning(f"Final Check: Display image mode is {temp_display_image.mode}, converting to RGB.")
                            st.session_state.display_image = temp_display_image.convert('RGB')
                        except Exception as e:
                            logger.error(f"Final RGB conversion for display image failed: {e}", exc_info=True)
                            st.error("Failed final conversion of display image to RGB format.")
                            processing_successful = False # Mark as failed if conversion fails
                    else:
                        st.session_state.display_image = temp_display_image # Already RGB or successfully converted

                    # Only proceed if display image is valid RGB
                    if processing_successful:
                        logger.info(f"Assigning display_image to session state. Type: {type(st.session_state.display_image)}, Mode: {getattr(st.session_state.display_image, 'mode', 'N/A')}")
                        # Assign the processed image (might be L or RGB)
                        st.session_state.processed_image = temp_processed_image
                        logger.info(f"Assigning processed_image to session state. Type: {type(st.session_state.processed_image)}, Mode: {getattr(st.session_state.processed_image, 'mode', 'N/A')}")

                        logger.info(f"**SUCCESS**: Session state updated for file '{uploaded_file.name}'.")
                        # Reset analysis-specific state parts
                        st.session_state.roi_coords = None
                        st.session_state.canvas_drawing = None # Reset drawing state
                        st.session_state.initial_analysis = ""
                        st.session_state.qa_answer = ""
                        st.session_state.disease_analysis = ""
                        st.session_state.confidence_score = ""
                        st.session_state.pdf_report_bytes = None
                        st.session_state.history = [] # Clear history for new image

                        st.success(f"✅ Image '{uploaded_file.name}' processed successfully!")
                        st.rerun() # Rerun to update the UI reflecting the new state
                    else:
                         # Handle the case where final RGB conversion failed
                        st.error("Image processing failed during final preparation stage.")
                        logger.error("Processing failed (final conversion or assignment).")
                        st.session_state.display_image = None
                        st.session_state.processed_image = None

                else: # Processing failed earlier or resulted in invalid image objects
                    logger.critical("Image loading and processing pipeline failed.")
                    # Provide slightly more specific feedback if possible
                    if processing_successful: # Means processing logic thought it succeeded, but images are invalid
                        st.error("❌ Image was processed, but the resulting image data is invalid or corrupted.")
                        logger.error(f"Processing marked successful, but final check failed: display type {type(temp_display_image)}, processed type {type(temp_processed_image)}")
                    # Reset all relevant state if processing failed
                    st.session_state.uploaded_file_info = None
                    st.session_state.raw_image_bytes = None
                    st.session_state.display_image = None
                    st.session_state.processed_image = None
                    st.session_state.dicom_dataset = None
                    st.session_state.dicom_metadata = {}
                    st.session_state.current_display_wc = None
                    st.session_state.current_display_ww = None
                    st.session_state.is_dicom = False
                    # No rerun here, let the user see the error and potentially upload again.

    # --- DICOM Window/Level Controls ---
    st.markdown("---")
    # Show W/L controls only if it's a DICOM, pydicom is available, dataset was loaded, and we have a display image
    if st.session_state.is_dicom and pydicom is not None and st.session_state.dicom_dataset and isinstance(st.session_state.get("display_image"), Image.Image):
        with st.expander("DICOM Window/Level", expanded=False):
            try:
                # Use the custom UI component for sliders
                wc_slider, ww_slider = dicom_wl_sliders(
                    st.session_state.dicom_dataset,
                    st.session_state.dicom_metadata,
                    initial_wc=st.session_state.current_display_wc,
                    initial_ww=st.session_state.current_display_ww
                )

                # Get current display values from state
                current_wc_disp = st.session_state.current_display_wc
                current_ww_disp = st.session_state.current_display_ww

                # Check if sliders returned valid values and if they differ from current display values
                valid_sliders = (wc_slider is not None and ww_slider is not None)
                valid_current_disp = (current_wc_disp is not None and current_ww_disp is not None)

                # Update if sliders are valid AND either current display values are invalid OR slider values changed significantly
                needs_update = valid_sliders and (
                    not valid_current_disp or
                    (abs(wc_slider - current_wc_disp) > 1e-3 or abs(ww_slider - current_ww_disp) > 1e-3) # Use tolerance for float comparison
                )

                if needs_update:
                    logger.info(f"DICOM W/L Change Detected: Applying WC={wc_slider:.1f}, WW={ww_slider:.1f}")
                    with st.spinner("Applying new Window/Level settings..."):
                         # Regenerate the display image with new W/L settings
                         new_display_img = dicom_to_image(st.session_state.dicom_dataset, wc_slider, ww_slider)
                         if isinstance(new_display_img, Image.Image):
                              # Ensure the new image is RGB before updating state
                              st.session_state.display_image = new_display_img.convert('RGB') if new_display_img.mode != 'RGB' else new_display_img
                              # Update the current W/L values in state
                              st.session_state.current_display_wc = wc_slider
                              st.session_state.current_display_ww = ww_slider
                              logger.debug("DICOM W/L applied successfully, triggering rerun.")
                              st.rerun() # Rerun to display the updated image
                         else:
                              st.error("Failed to apply new Window/Level settings.")
                              logger.error("dicom_to_image failed during W/L update, did not return a valid PIL Image.")

            except Exception as e:
                st.error(f"Error in DICOM Window/Level controls: {e}")
                logger.error(f"DICOM W/L slider interaction error: {e}", exc_info=True)
        st.markdown("---") # Separator after W/L controls
    elif st.session_state.is_dicom and pydicom is None:
        # Show a warning if it's identified as DICOM but pydicom is missing
        st.warning("DICOM file detected, but the 'pydicom' library is missing. Window/Level controls are unavailable.")

    # --- AI Actions Section ---
    # Enable AI actions only if a valid display image exists in the session state
    if isinstance(st.session_state.get("display_image"), Image.Image):
        st.subheader("AI Actions")

        # Button to trigger initial analysis
        if st.button("▶️ Run Initial Analysis", key="analyze_btn", use_container_width=True, help="Get a general AI analysis of the entire image."):
            st.session_state.last_action = "analyze"
            logger.info("Sidebar Button: Setting action to 'analyze'")
            st.rerun() # Rerun to handle the action

        st.markdown("---") # Separator
        st.subheader("❓ Ask AI Question")

        # Display current ROI info if set
        current_roi = st.session_state.get("roi_coords")
        if current_roi:
            rc = current_roi
            st.info(f"✅ ROI Selected: [X:{rc['left']}, Y:{rc['top']}, W:{rc['width']}, H:{rc['height']}]")
            # Button to clear the ROI
            if st.button("❌ Clear ROI", key="clear_roi_btn", use_container_width=True, help="Remove the selected Region of Interest."):
                st.session_state.roi_coords = None
                st.session_state.canvas_drawing = None # Also clear the canvas drawing state
                logger.info("Sidebar Button: ROI cleared by user.")
                st.rerun()
        else:
            st.caption("ℹ️ Optionally, draw a rectangle on the image viewer to select a Region of Interest (ROI) before asking.")

        # Text area for user question input
        question_input = st.text_area(
            "Ask about the image or selected ROI:",
            height=100,
            key="question_input_widget",
            placeholder="e.g., Are there any anomalies in this area? What type of scan is this?",
            label_visibility="collapsed"
        )

        # Button to trigger Q&A
        if st.button("💬 Ask AI", key="ask_btn", use_container_width=True, help="Ask the AI a question about the image or the ROI."):
            user_question = st.session_state.question_input_widget
            if user_question and user_question.strip():
                st.session_state.last_action = "ask"
                logger.info(f"Sidebar Button: Setting action to 'ask' with question: '{user_question[:50]}...'")
                st.rerun()
            else:
                st.warning("Please enter a question before clicking 'Ask AI'.")
                logger.warning("Sidebar Button 'Ask AI': Clicked with empty question.")

        st.markdown("---") # Separator
        st.subheader("🎯 Focused Condition Analysis")

        # Predefined list of common conditions for the dropdown
        # Consider making this list configurable or dynamically generated
        DISEASE_OPTIONS = [
            "Pneumonia", "Lung Cancer", "Stroke", "Fracture", "Appendicitis",
            "Tuberculosis", "COVID-19", "Pulmonary Embolism", "Brain Tumor",
            "Arthritis", "Osteoporosis", "Cardiomegaly", "Aortic Aneurysm",
            "Bowel Obstruction", "Mass/Nodule", "Effusion", "Normal Variation" # Added Normal
        ]
        # Add a blank option at the start for default/unselected state
        disease_options = [""] + sorted(DISEASE_OPTIONS)

        # Dropdown for selecting a specific condition
        disease_select = st.selectbox(
            "Select Condition:",
            options=disease_options,
            key="disease_select_widget",
            help="Select a specific condition for the AI to focus on."
        )

        # Button to trigger disease-focused analysis
        if st.button("🩺 Run Condition Analysis", key="disease_btn", use_container_width=True, help="Ask the AI to analyze the image specifically for the selected condition."):
            selected_disease = st.session_state.disease_select_widget
            if selected_disease:
                st.session_state.last_action = "disease"
                logger.info(f"Sidebar Button: Setting action to 'disease' with condition: '{selected_disease}'")
                st.rerun()
            else:
                st.warning("Please select a condition from the dropdown list.")
                logger.warning("Sidebar Button 'Run Condition Analysis': Clicked with no condition selected.")

        st.markdown("---") # Separator
        # Expander for Confidence Estimation and Report Generation
        with st.expander("📊 Confidence & Report", expanded=True):
            # Determine if there's enough context for confidence estimation
            can_estimate_confidence = bool(
                st.session_state.history or
                st.session_state.initial_analysis or
                st.session_state.disease_analysis
            )
            # Button to estimate AI confidence
            if st.button("📈 Estimate Confidence", key="confidence_btn", disabled=not can_estimate_confidence, use_container_width=True, help="Estimate the AI's confidence based on the analysis performed so far."):
                st.session_state.last_action = "confidence"
                logger.info("Sidebar Button: Setting action to 'confidence'")
                st.rerun()
            if not can_estimate_confidence:
                st.caption("Run at least one analysis or ask a question first to enable confidence estimation.")

            # Button to generate data for the PDF report
            if st.button("📄 Generate PDF Data", key="generate_report_data_btn", use_container_width=True, help="Prepare the data needed to generate a summary PDF report (non-clinical)."):
                st.session_state.last_action = "generate_report_data"
                logger.info("Sidebar Button: Setting action to 'generate_report_data'")
                st.rerun()

            # If PDF report data (bytes) exists in state, show the download button
            if st.session_state.pdf_report_bytes:
                report_filename = f"RadVisionAI_Report_{st.session_state.session_id or 'session'}.pdf"
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=st.session_state.pdf_report_bytes,
                    file_name=report_filename,
                    mime="application/pdf",
                    key="download_pdf_button",
                    use_container_width=True,
                    help="Download the generated non-clinical summary report."
                )
                logger.debug(f"Offering PDF download: {report_filename}")

    else:
        # Message shown in the sidebar if no image is loaded yet
        st.info("👈 Upload an image using the controls above to begin analysis.")

# =============================================================================
# === MAIN PANEL DISPLAYS =====================================================
# =============================================================================
col1, col2 = st.columns([2, 3]) # Ratio for image viewer vs. results panel

# --- Column 1: Image Viewer, Canvas, Metadata ---
with col1:
    st.subheader("🖼️ Image Viewer")
    # Get the display image object from session state
    display_img_object = st.session_state.get("display_image")
    logger.debug(f"Main Panel Check: Retrieved display_image from state. Type: {type(display_img_object)}, Is PIL Image: {isinstance(display_img_object, Image.Image)}")

    # --- This is the main image display area using streamlit-drawable-canvas ---
    if isinstance(display_img_object, Image.Image):
        logger.debug(f"Image Viewer: Preparing to render canvas. Image Mode: {display_img_object.mode}, Size: {display_img_object.size}")

        # --- Prepare Background Image for Canvas ---
        # st_canvas requires an RGB(A) image. Convert if necessary.
        bg_image_pil = None
        try:
            if display_img_object.mode == 'RGB':
                bg_image_pil = display_img_object
                logger.debug("Canvas Prep: Image is already RGB.")
            elif display_img_object.mode == 'RGBA':
                 bg_image_pil = display_img_object
                 logger.debug("Canvas Prep: Image is RGBA, acceptable for canvas.")
            elif display_img_object.mode == 'L': # Grayscale
                 logger.info(f"Canvas Prep: Converting Grayscale (L) image to RGB for canvas.")
                 bg_image_pil = display_img_object.convert('RGB')
            else: # Other modes like P, CMYK etc.
                logger.info(f"Canvas Prep: Converting image mode {display_img_object.mode} to RGB for canvas.")
                bg_image_pil = display_img_object.convert('RGB')

            # Final check on the prepared image
            if not isinstance(bg_image_pil, Image.Image):
                raise TypeError(f"Image conversion for canvas resulted in invalid type: {type(bg_image_pil)}")
            logger.debug(f"Canvas Prep: Final background image type {type(bg_image_pil)}, mode {getattr(bg_image_pil, 'mode', 'N/A')}")

        except Exception as prep_err:
            st.error(f"Failed to prepare image for the viewer: {prep_err}")
            logger.error(f"Canvas background image preparation error: {prep_err}", exc_info=True)
            bg_image_pil = None # Ensure it's None if prep fails

        # --- Render Canvas if Background Image is Ready ---
        if isinstance(bg_image_pil, Image.Image):
            # --- Calculate Canvas Dimensions ---
            # Define max dimensions for the canvas container
            MAX_CANVAS_WIDTH, MAX_CANVAS_HEIGHT = 700, 600
            img_w, img_h = bg_image_pil.size
            aspect_ratio = img_w / img_h if img_h > 0 else 1 # Avoid division by zero

            # Calculate scaled dimensions preserving aspect ratio
            canvas_w = min(img_w, MAX_CANVAS_WIDTH)
            canvas_h = int(canvas_w / aspect_ratio) if aspect_ratio > 0 else MAX_CANVAS_HEIGHT

            # If calculated height exceeds max height, recalculate width based on max height
            if canvas_h > MAX_CANVAS_HEIGHT:
                canvas_h = MAX_CANVAS_HEIGHT
                canvas_w = int(canvas_h * aspect_ratio)

            # Ensure minimum dimensions for usability
            canvas_w = max(int(canvas_w), 150)
            canvas_h = max(int(canvas_h), 150)
            logger.info(f"Canvas Prep: Calculated canvas dimensions: Width={canvas_w}, Height={canvas_h} (Aspect Ratio: {aspect_ratio:.2f})")

            # Proceed only if dimensions are valid
            if canvas_w > 0 and canvas_h > 0:
                st.caption("Click and drag on the image below to select a Region of Interest (ROI).")
                try:
                    # Retrieve the current drawing state from session state (if any)
                    # Ensures the rectangle persists across reruns until cleared
                    initial_drawing_state = st.session_state.get("canvas_drawing")
                    # Basic validation of the drawing state format
                    if initial_drawing_state and not isinstance(initial_drawing_state, dict):
                        logger.warning(f"Invalid initial drawing state found (type: {type(initial_drawing_state)}), resetting.")
                        initial_drawing_state = None
                        st.session_state.canvas_drawing = None

                    logger.info(f"Rendering st_canvas. Background image mode: {bg_image_pil.mode}. Initial drawing state: {'Set' if initial_drawing_state else 'None'}")
                    if not isinstance(bg_image_pil, Image.Image): # Defensive check
                         raise ValueError("Background image became invalid just before st_canvas call.")

                    # --- Render the Drawable Canvas ---
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.2)",  # Semi-transparent orange fill for rectangle
                        stroke_width=2,                      # Border width of the rectangle
                        stroke_color="rgba(220, 50, 50, 0.9)",# Red border for rectangle
                        background_image=bg_image_pil,       # The prepared PIL image
                        update_streamlit=True,               # Send updates back to Streamlit dynamically
                        height=canvas_h,                     # Calculated canvas height
                        width=canvas_w,                      # Calculated canvas width
                        drawing_mode="rect",                 # Allow drawing rectangles only
                        initial_drawing=initial_drawing_state, # Load previous drawing state if exists
                        key="drawable_canvas",               # Unique key for the widget
                    )
                    logger.info("st_canvas rendered (or attempted). Waiting for user interaction or update.")

                    # --- Process Canvas Result ---
                    # This block executes when the user draws/modifies the rectangle
                    if canvas_result is not None and canvas_result.json_data is not None:
                        # Store the latest drawing state back into session state
                        st.session_state.canvas_drawing = canvas_result.json_data
                        current_roi_state = st.session_state.get("roi_coords")

                        # Check if there are any drawing objects (rectangles)
                        if canvas_result.json_data.get("objects"):
                            # Get the latest drawn object (assuming the last one is the current one)
                            last_object = canvas_result.json_data["objects"][-1]
                            if last_object["type"] == "rect":
                                # --- Calculate ROI Coordinates relative to the original image ---
                                # Get dimensions and position from the canvas object
                                canvas_left, canvas_top = int(last_object["left"]), int(last_object["top"])
                                # Account for potential scaling factors applied by the canvas
                                canvas_width = int(last_object["width"] * last_object.get("scaleX", 1))
                                canvas_height = int(last_object["height"] * last_object.get("scaleY", 1))

                                # Calculate scaling factors between original image and canvas dimensions
                                scale_x = img_w / canvas_w
                                scale_y = img_h / canvas_h

                                # Convert canvas coordinates to original image coordinates
                                original_left = max(0, int(canvas_left * scale_x))
                                original_top = max(0, int(canvas_top * scale_y))
                                original_width = int(canvas_width * scale_x)
                                original_height = int(canvas_height * scale_y)

                                # Ensure coordinates are within image bounds and width/height are non-negative
                                original_right = min(img_w, original_left + original_width)
                                original_bottom = min(img_h, original_top + original_height)
                                final_width = max(0, original_right - original_left)
                                final_height = max(0, original_bottom - original_top)

                                # Define a minimum ROI size to be considered valid (e.g., 10x10 pixels)
                                MIN_ROI_DIM = 10
                                if final_width >= MIN_ROI_DIM and final_height >= MIN_ROI_DIM:
                                    new_roi_dict = {
                                        "left": original_left, "top": original_top,
                                        "width": final_width, "height": final_height
                                    }
                                    # --- Update State and Rerun if ROI changed ---
                                    if current_roi_state != new_roi_dict:
                                        st.session_state.roi_coords = new_roi_dict
                                        logger.info(f"ROI Updated: {new_roi_dict}")
                                        st.rerun() # Rerun to update UI elements dependent on ROI (like sidebar info)
                                elif current_roi_state is not None:
                                     # If the new drawing is too small, but an ROI existed, clear the old ROI
                                     logger.info("New drawing is too small, clearing existing ROI.")
                                     st.session_state.roi_coords = None
                                     st.rerun()

                        # If no objects exist anymore (e.g., user deleted the rect), clear the ROI state
                        elif not canvas_result.json_data.get("objects") and current_roi_state is not None:
                            logger.info("Canvas drawing cleared by user, removing ROI state.")
                            st.session_state.roi_coords = None
                            st.rerun() # Rerun to update UI

                except Exception as canvas_error:
                    st.error(f"Error displaying the image canvas: {canvas_error}")
                    logger.error(f"st_canvas rendering or interaction failed: {canvas_error}", exc_info=True)
                    # Provide hint for potential browser issues
                    st.warning("Drawing functionality might be unavailable. Check browser console (F12) for JavaScript errors.")
            else:
                # This case should ideally not be reached if image prep is correct
                st.error("Cannot display image: Invalid canvas dimensions calculated.")
                logger.error(f"Invalid calculated canvas dimensions: W={canvas_w}, H={canvas_h}")
        else:
            # Error message if bg_image_pil preparation failed earlier
            st.info("Image could not be prepared for the interactive viewer.")
            logger.error("Cannot display canvas because bg_image_pil is not a valid PIL Image.")

        # --- Display DICOM Metadata (if applicable) ---
        # Show metadata below the canvas if it's a DICOM and metadata was extracted
        if st.session_state.is_dicom and st.session_state.dicom_metadata:
            st.markdown("---") # Separator
            st.subheader("📄 DICOM Metadata")
            if pydicom is None:
                st.warning("DICOM metadata might be available but cannot be displayed as 'pydicom' library is missing.")
            else:
                logger.debug("Displaying extracted DICOM metadata.")
                # Use the custom UI component to display metadata nicely
                display_dicom_metadata(st.session_state.dicom_metadata)

    # --- Fallback Placeholder if No Image ---
    else:
        logger.debug("Image Viewer: No valid display_image found in session state for rendering.")
        st.markdown("---") # Separator
        # Show different messages depending on whether a file was uploaded but failed, or no file yet
        if st.session_state.uploaded_file_info:
            st.warning("Image processing failed, or the uploaded file resulted in invalid image data. Please try a different file.")
        else:
            st.info("Upload an image file using the sidebar to get started.")
        # Simple HTML placeholder box
        st.markdown(
            """
            <div style='height: 400px; border: 2px dashed #ccc; display: flex;
                        align-items: center; justify-content: center; text-align: center;
                        color: #aaa; font-style: italic; padding: 20px; border-radius: 8px;'>
                Image Display Area<br/>(Waiting for image upload or encountered processing error)
            </div>
            """,
            unsafe_allow_html=True
        )

# --- Column 2: Analysis Results Tabs ---
with col2:
    st.subheader("📊 Analysis & Results")
    # Define tab titles
    tab_titles = ["🔬 Initial Analysis", "💬 Q&A History", "🩺 Disease Focus", "📈 Confidence"]
    # Create tabs
    tabs = st.tabs(tab_titles)

    # --- Initial Analysis Tab ---
    with tabs[0]:
        st.text_area(
            "Overall Findings",
            value=st.session_state.initial_analysis or "No initial analysis has been performed yet.",
            height=450,
            key="output_initial",
            disabled=True, # Read-only display
            help="Shows the general analysis results from 'Run Initial Analysis'."
        )

    # --- Q&A History Tab ---
    with tabs[1]:
        st.text_area(
            "Latest AI Answer",
            value=st.session_state.qa_answer or "Ask a question using the sidebar.",
            height=200,
            key="output_qa",
            disabled=True, # Read-only display
            help="Shows the answer to the most recent question asked."
        )
        st.markdown("---") # Separator
        # Display conversation history if it exists
        if st.session_state.history:
             with st.expander("Full Conversation History", expanded=True):
                 # Iterate through history (list of tuples: (question, answer))
                 for i, (question, answer) in enumerate(reversed(st.session_state.history)): # Show newest first
                     st.markdown(f"**You ({len(st.session_state.history)-i}):**")
                     st.caption(f"{question}") # Display question in caption style
                     st.markdown(f"**AI ({len(st.session_state.history)-i}):**")
                     st.markdown(answer, unsafe_allow_html=True) # Display AI answer, allowing basic markdown
                     # Add separator between Q&A pairs, but not after the last one
                     if i < len(st.session_state.history) - 1:
                         st.markdown("---")
        else:
            st.caption("No question history yet for this session.")

    # --- Disease Focus Tab ---
    with tabs[2]:
        st.text_area(
            "Focused Condition Findings",
            value=st.session_state.disease_analysis or "No focused condition analysis has been performed yet.",
            height=450,
            key="output_disease",
            disabled=True, # Read-only display
            help="Shows the results from 'Run Condition Analysis' for a specific condition."
        )

    # --- Confidence Tab ---
    with tabs[3]:
        st.text_area(
            "AI Confidence Estimation",
            value=st.session_state.confidence_score or "No confidence estimation has been performed yet.",
            height=450,
            key="output_confidence",
            disabled=True, # Read-only display
            help="Shows the AI's estimated confidence based on performed analyses and Q&A."
        )

# =============================================================================
# === ACTION HANDLING LOGIC (Triggered by st.rerun() after button clicks) ===
# =============================================================================
# Get the action that was set in the sidebar before the rerun
current_action: Optional[str] = st.session_state.get("last_action")

# Proceed only if an action is set
if current_action:
    logger.info(f"ACTION HANDLER: Detected action '{current_action}'")

    # --- Prerequisites Check ---
    # Most actions require a valid processed image and a session ID
    processed_image = st.session_state.get("processed_image")
    session_id = st.session_state.get("session_id")
    roi_coords = st.session_state.get("roi_coords") # Get current ROI, if any
    conversation_history = st.session_state.history # Get history (should be list)

    # Validate image for actions that need it
    if current_action != "generate_report_data": # Report generation uses display_image
        if not isinstance(processed_image, Image.Image):
            st.error(f"Cannot perform action '{current_action}': The processed image data is missing or invalid.")
            logger.error(f"Action '{current_action}' aborted: Required 'processed_image' is not a valid PIL Image (Type: {type(processed_image)}).")
            st.session_state.last_action = None # Clear the action
            st.stop() # Stop execution if image invalid

    # Validate session ID
    if not session_id:
        st.error(f"Cannot perform action '{current_action}': Session ID is missing. Please re-upload the image.")
        logger.error(f"Action '{current_action}' aborted: Required 'session_id' is missing.")
        st.session_state.last_action = None # Clear the action
        st.stop() # Stop execution if session ID invalid

    # Ensure history is a list (it should be, but defensive check)
    if not isinstance(conversation_history, list):
        logger.warning(f"Session state 'history' was not a list (Type: {type(conversation_history)}). Resetting to empty list.")
        conversation_history = []
        st.session_state.history = conversation_history

    # Prepare common arguments for LLM calls
    image_for_llm = processed_image # Use the potentially raw/processed image for LLMs
    roi_info_str = " (with ROI)" if roi_coords else "" # For logging/display purposes

    # --- Execute Action ---
    try:
        if current_action == "analyze":
            st.info(f"🔬 Performing initial analysis{roi_info_str}...")
            with st.spinner("AI is analyzing the image... Please wait."):
                # Call the appropriate LLM function (passing ROI if supported by the function)
                # Note: Adjust function signature if ROI needs to be passed explicitly
                analysis_result = run_initial_analysis(image_for_llm) # Pass roi=roi_coords if function supports it
            st.session_state.initial_analysis = analysis_result
            st.success("Initial analysis complete.")
            logger.info(f"Action 'analyze' completed.")

        elif current_action == "ask":
            user_question = st.session_state.question_input_widget.strip()
            if not user_question:
                st.warning("Question was empty. Please enter a question.")
                logger.warning("Action 'ask' skipped: User question was empty.")
            else:
                st.info(f"❓ Asking AI{roi_info_str}: '{user_question[:70]}...'")
                st.session_state.qa_answer = "" # Clear previous answer display
                with st.spinner("AI is processing your question..."):
                     # Call the multimodal Q&A function
                     ai_answer, success_flag = run_multimodal_qa(
                         image_for_llm,
                         user_question,
                         conversation_history, # Pass current history
                         roi_coords # Pass ROI data
                     )
                if success_flag:
                    st.session_state.qa_answer = ai_answer
                    # Append the new Q&A pair to the history
                    st.session_state.history.append((user_question, ai_answer))
                    st.success("AI answered your question.")
                    logger.info(f"Action 'ask' completed successfully.")
                else:
                    # Handle failure from the primary LLM/VQA model
                    error_message = f"Primary AI failed to answer: {ai_answer}"
                    st.error(error_message)
                    logger.warning(f"Action 'ask' failed with primary model: {ai_answer}")
                    st.session_state.qa_answer = f"**[Primary AI Error]** {ai_answer}\n\n---\n" # Display error

                    # --- Fallback Logic (Example using Hugging Face) ---
                    # Check if fallback is configured and available
                    hf_fallback_enabled = (
                        HF_VQA_MODEL_ID and HF_VQA_MODEL_ID != "hf_model_not_found" and
                        'query_hf_vqa_inference_api' in globals() and
                        os.environ.get("HF_API_TOKEN") # Check if API token is set (optional)
                    )
                    if hf_fallback_enabled:
                         fallback_model_name = HF_VQA_MODEL_ID.split('/')[-1] # Get short name for display
                         st.info(f"Attempting fallback using '{fallback_model_name}'...")
                         logger.info(f"Action 'ask': Trying fallback model {HF_VQA_MODEL_ID}")
                         with st.spinner(f"Asking fallback AI ({fallback_model_name})..."):
                              fallback_answer, fallback_success = query_hf_vqa_inference_api(
                                  image_for_llm, user_question, roi_coords
                              )
                         if fallback_success:
                              fallback_display = f"**[Fallback: {fallback_model_name}]**\n\n{fallback_answer}"
                              st.session_state.qa_answer += fallback_display # Append fallback answer
                              # Add fallback Q&A to history, marked as fallback
                              st.session_state.history.append((f"[Fallback Question] {user_question}", fallback_display))
                              st.success(f"Fallback AI ({fallback_model_name}) provided an answer.")
                              logger.info(f"Action 'ask': Fallback model successful.")
                         else:
                              fallback_error_msg = f"Fallback AI ({fallback_model_name}) also failed: {fallback_answer}"
                              st.session_state.qa_answer += f"**[Fallback Failed]** {fallback_error_msg}"
                              st.error(fallback_error_msg)
                              logger.error(f"Action 'ask': Fallback model failed: {fallback_answer}")
                    else:
                         # Inform user if fallback is not available/configured
                         fallback_unavailable_msg = "No fallback AI model is available or configured."
                         st.session_state.qa_answer += f"**[Fallback Unavailable]** {fallback_unavailable_msg}"
                         st.warning(fallback_unavailable_msg)
                         logger.warning("Action 'ask': Fallback model skipped (not configured or unavailable).")

        elif current_action == "disease":
            selected_disease = st.session_state.disease_select_widget
            if not selected_disease:
                st.warning("No condition selected. Please choose one from the dropdown.")
                logger.warning("Action 'disease' skipped: No condition selected.")
            else:
                st.info(f"🩺 Analyzing for '{selected_disease}'{roi_info_str}...")
                with st.spinner(f"AI is assessing the image for signs of '{selected_disease}'..."):
                    # Call the disease-specific analysis function
                    disease_result = run_disease_analysis(
                        image_for_llm,
                        selected_disease,
                        roi_coords # Pass ROI data
                    )
                st.session_state.disease_analysis = disease_result
                st.success(f"Analysis for '{selected_disease}' complete.")
                logger.info(f"Action 'disease' completed for condition '{selected_disease}'.")

        elif current_action == "confidence":
            # Check if there's any analysis context to base confidence on
            context_available = bool(
                conversation_history or
                st.session_state.initial_analysis or
                st.session_state.disease_analysis
            )
            if not context_available:
                st.warning("Cannot estimate confidence: No analysis or Q&A has been performed yet.")
                logger.warning("Action 'confidence' skipped: No context available.")
            else:
                st.info(f"📊 Estimating AI confidence based on current analysis{roi_info_str}...")
                with st.spinner("AI is calculating the confidence score..."):
                    # Call the confidence estimation function
                    confidence_result = estimate_ai_confidence(
                        image_for_llm,
                        conversation_history,
                        st.session_state.initial_analysis,
                        st.session_state.disease_analysis,
                        roi_coords # Pass ROI data
                    )
                st.session_state.confidence_score = confidence_result
                st.success("Confidence estimation complete.")
                logger.info(f"Action 'confidence' completed.")

        elif current_action == "generate_report_data":
            st.info("📄 Preparing data for PDF report generation...")
            st.session_state.pdf_report_bytes = None # Clear any previous report data

            # Use the display_image for the report, as it includes W/L adjustments
            report_image = st.session_state.get("display_image")

            if not isinstance(report_image, Image.Image):
                st.error("Cannot generate report: The display image is missing or invalid.")
                logger.error("Action 'generate_report_data' aborted: 'display_image' is not a valid PIL Image.")
            else:
                 # Create a copy to draw ROI on without modifying the main display image
                 image_for_report = report_image.copy()
                 # Draw ROI rectangle on the report image if ROI exists
                 if roi_coords:
                      try:
                          # Ensure image is drawable (needs RGB or RGBA usually)
                          if image_for_report.mode not in ['RGB', 'RGBA']:
                              logger.info(f"Converting report image from {image_for_report.mode} to RGB for drawing.")
                              image_for_report = image_for_report.convert("RGB")

                          draw = ImageDraw.Draw(image_for_report)
                          x0, y0 = roi_coords['left'], roi_coords['top']
                          x1 = x0 + roi_coords['width']
                          y1 = y0 + roi_coords['height']
                          # Draw a red rectangle with thickness 3
                          draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
                          logger.info("Successfully drew ROI rectangle on the image for the PDF report.")
                      except Exception as draw_error:
                          logger.error(f"Failed to draw ROI on report image: {draw_error}", exc_info=True)
                          st.warning("Could not draw the ROI on the report image, proceeding without it.")
                          # Keep using the original report_image copy without the drawing

                 # --- Gather Data for Report ---
                 # Format Q&A history nicely
                 qa_history_formatted = "\n\n".join([
                     f"Q: {q}\nA: {a}" for q, a in conversation_history
                 ]) if conversation_history else "No questions asked."

                 # Prepare dictionary with all output sections
                 report_outputs = {
                     "Session ID": session_id,
                     "Preliminary Analysis": st.session_state.initial_analysis or "Not Performed",
                     "Q&A History": qa_history_formatted,
                     "Focused Condition Analysis": st.session_state.disease_analysis or "Not Performed",
                     "AI Confidence Estimation": st.session_state.confidence_score or "Not Estimated"
                 }

                 # Add filtered DICOM metadata if applicable
                 if st.session_state.is_dicom and st.session_state.dicom_metadata:
                      # TODO: Implement actual filtering logic in report_utils or here
                      # Example: Select only specific important tags
                      filtered_meta = {
                          k: v for k, v in st.session_state.dicom_metadata.items()
                          if k in ["Patient Name", "Patient ID", "Study Date", "Modality", "Study Description"] # Example filter
                      }
                      report_outputs["DICOM Metadata (Selected)"] = "\n".join([f"{k}: {v}" for k,v in filtered_meta.items()]) if filtered_meta else "No relevant metadata found or extracted."
                      logger.debug("Added filtered DICOM metadata to report data.")


                 # --- Generate PDF Bytes ---
                 with st.spinner("🎨 Generating PDF document..."):
                     pdf_bytes = generate_pdf_report_bytes(
                         session_id,
                         image_for_report, # Pass the (potentially annotated) image
                         report_outputs     # Pass the dictionary of text outputs
                     )

                 if pdf_bytes:
                     st.session_state.pdf_report_bytes = pdf_bytes
                     st.success("✅ PDF report data generated successfully! Use the download button.")
                     logger.info("Action 'generate_report_data' completed successfully.")
                 else:
                     st.error("❌ Failed to generate the PDF report.")
                     logger.error("Action 'generate_report_data' failed: generate_pdf_report_bytes returned None.")
        else:
            # Handle unrecognized actions (should not happen with button logic)
            st.warning(f"Unknown action requested: '{current_action}'.")
            logger.warning(f"Handler encountered unknown action: '{current_action}'")

    except Exception as action_error:
        # Catch-all for unexpected errors during action execution
        st.error(f"An unexpected error occurred while performing action '{current_action}'.")
        logger.critical(f"Critical error during action '{current_action}': {action_error}", exc_info=True)

    finally:
        # --- Cleanup After Action ---
        # Clear the last action flag regardless of success or failure
        st.session_state.last_action = None
        logger.debug(f"Action '{current_action}' handler finished execution.")
        # Rerun one last time to update the UI (e.g., show results in tabs, update download button state)
        st.rerun()

# --- Footer ---
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced | Session ID: {st.session_state.get('session_id', 'N/A')} | Powered by Streamlit")
logger.info("--- App Render Cycle Complete ---")