import streamlit as st

# Ensure this is the very first command
st.set_page_config(
    page_title="RadVision AI Advanced",
    layout="wide",
    page_icon="⚕️",
    initial_sidebar_state="expanded"
)

# --- Core Libraries ---
import io
import os
import uuid
import logging
import base64
from typing import Any, Dict, Optional, Tuple, List
import copy
import random  # For Tip of the Day
import re      # For formatting the translation

# --- Drawable Canvas ---
try:
    from streamlit_drawable_canvas import st_canvas
    import streamlit_drawable_canvas as st_canvas_module
    CANVAS_VERSION = getattr(st_canvas_module, 'version', 'Unknown')
except ImportError:
    st.error("CRITICAL ERROR: streamlit-drawable-canvas is not installed. Run `pip install streamlit-drawable-canvas`.")
    st.stop()

# --- Custom CSS for a Polished Look & Tab Scrolling ---
st.markdown(
    """
    <style>
      body {
          font-family: 'Helvetica', sans-serif;
          background-color: #f9f9f9;
      }
      .css-1d391kg { background-color: #ffffff; }
      footer {
          text-align: center;
          font-size: 0.8em;
          color: #888888;
      }
      /* Ensure tab scrolling on overflow */
      div[role="tablist"] {
          overflow-x: auto;
          white-space: nowrap;
          /* Add some padding for better visual */
          padding-bottom: 10px;
      }
       /* Style for individual tabs to prevent wrapping */
      div[role="tablist"] button {
          white-space: nowrap;
      }
    </style>
    """, unsafe_allow_html=True
)

# --- Display Hero Logo ---
logo_path = os.path.join("assets", "radvisionai-hero.jpeg")
if os.path.exists(logo_path):
    st.image(logo_path, width=400)
else:
    st.warning("Hero logo not found in the assets folder.")

# --- Tip of the Day in Sidebar ---
TIPS = [
    "Tip: Use Demo Mode to quickly see how the analysis works.",
    "Tip: Draw an ROI to focus the AI on a specific region of the image.",
    "Tip: Adjust DICOM window/level sliders for optimal contrast.",
    "Tip: Review conversation history to refine your questions.",
    "Tip: Generate a PDF report to save your analysis.",
    "Tip: Use the Translation tab to view results in other languages." # Added tip
]
st.sidebar.info(random.choice(TIPS))

# --- Image & DICOM Processing ---
try:
    from PIL import Image, ImageDraw, UnidentifiedImageError
    import PIL
    PIL_VERSION = getattr(PIL, 'version', 'Unknown')
except ImportError:
    st.error("CRITICAL ERROR: Pillow (PIL) is not installed. Run `pip install Pillow`.")
    st.stop()

try:
    import pydicom
    import pydicom.errors
    PYDICOM_VERSION = getattr(pydicom, 'version', 'Unknown')
except ImportError:
    PYDICOM_VERSION = 'Not Installed'
    pydicom = None

# --- Set Up Logging ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
if pydicom is None:
    logger.error("pydicom not found. DICOM functionality disabled.")
else:
    logger.info(f"pydicom version: {PYDICOM_VERSION}")

try:
    import pylibjpeg
    logger.info("pylibjpeg found.")
except ImportError:
    logger.warning("pylibjpeg not found. Consider installing for extended DICOM support.")

try:
    import gdcm
    logger.info("python-gdcm found.")
except ImportError:
    logger.warning("python-gdcm not found. Consider installing for improved DICOM compatibility.")

logger.info("--- App Start ---")
logger.info(f"Logging level: {LOG_LEVEL}")
logger.info(f"Streamlit version: {st.version}")
logger.info(f"Pillow (PIL) version: {PIL_VERSION}")
logger.info(f"streamlit_drawable_canvas version: {CANVAS_VERSION}")

# --- Function to Post-Process Translated Text ---
def format_translation(translated_text: str) -> str:
    """Attempts to restore some list formatting that might be lost in translation."""
    # Simple replacements: Finds digits followed by a dot and ensures they start on a new line.
    formatted_text = re.sub(r'\s*(\d+\.)', r'\n\n\1', translated_text.strip())
    # Simple replacements: Finds hyphens/asterisks used as bullets
    formatted_text = re.sub(r'\s*([-*]\s)', r'\n\1', formatted_text)
    return formatted_text

# --- Monkey-Patch st.elements.image.image_to_url if missing ---
# (Your existing monkey-patch code remains unchanged here)
import streamlit.elements.image as st_image
if not hasattr(st_image, "image_to_url"):
    def image_to_url_monkey_patch(img_obj: Any, *args, **kwargs) -> str:
        if isinstance(img_obj, Image.Image):
            try:
                buffered = io.BytesIO()
                fmt = "PNG"
                temp_img = img_obj
                # Ensure compatibility with different modes before saving
                if img_obj.mode == 'P': # Palette mode often needs RGBA conversion for transparency
                    temp_img = img_obj.convert('RGBA')
                elif img_obj.mode not in ['RGB', 'L', 'RGBA']: # Convert other modes to RGB as a fallback
                    temp_img = img_obj.convert('RGB')

                temp_img.save(buffered, format=fmt)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/{fmt.lower()};base64,{img_str}"
            except Exception as e:
                logger.error(f"Monkey-patch failed during image conversion: {e}", exc_info=True)
                return ""
        else:
            logger.warning(f"Unsupported type for image_to_url monkey-patch: {type(img_obj)}")
            return ""
    st_image.image_to_url = image_to_url_monkey_patch
    logger.info("Applied monkey-patch for st.elements.image.image_to_url")


# --- Import Custom Utilities & Fallbacks ---
try:
    from dicom_utils import parse_dicom, extract_dicom_metadata, dicom_to_image, get_default_wl
    from llm_interactions import run_initial_analysis, run_multimodal_qa, run_disease_analysis, estimate_ai_confidence
    from report_utils import generate_pdf_report_bytes
    from ui_components import display_dicom_metadata, dicom_wl_sliders
    logger.info("Custom utility modules imported.")

    # HF fallback for Q&A
    try:
        from hf_models import query_hf_vqa_inference_api, HF_VQA_MODEL_ID
    except ImportError:
        HF_VQA_MODEL_ID = "hf_model_not_found"
        def query_hf_vqa_inference_api(img: Image.Image, question: str, roi: Optional[Dict] = None) -> Tuple[str, bool]:
            return "[Fallback Unavailable] HF module not found.", False
        logger.warning("HF models not found. Fallback disabled.")
except ImportError as e:
    st.error(f"CRITICAL ERROR importing helpers: {e}")
    logger.critical(f"Failed import: {e}", exc_info=True)
    st.stop()

# --- Import the Translation Module (with detection) ---
# Define default languages here - can be replaced by LANGUAGE_CODES from module
DEFAULT_LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Japanese": "ja",
    "Chinese (Simplified)": "zh-CN", "Russian": "ru", "Arabic": "ar", "Hindi": "hi"
}
LANGUAGE_CODES = None
translate = None
detect_language = None
try:
    from translation_models import translate, LANGUAGE_CODES, detect_language
    logger.info("Translation module imported successfully.")
    if not LANGUAGE_CODES: # Fallback if module exists but LANGUAGE_CODES is empty
        LANGUAGE_CODES = DEFAULT_LANGUAGES
        logger.warning("LANGUAGE_CODES not found in translation_models, using defaults.")
except ImportError as e:
    st.warning(f"Could not import translation_models: {e}. Translation features disabled.")
    logger.error("Translation module not found. Translation features disabled.", exc_info=True)
    LANGUAGE_CODES = DEFAULT_LANGUAGES # Use defaults even if module fails, UI can show them


# --- Additional UI: Clear ROI ---
# (Your existing Clear ROI button logic remains unchanged here)
if st.sidebar.button("🗑️ Clear ROI", help="Clear the current ROI selection"):
    st.session_state.roi_coords = None
    st.session_state.canvas_drawing = None # Reset canvas drawing state too
    st.session_state.clear_roi_triggered = True # Use a dedicated flag
    logger.info("Clear ROI button clicked.")
    st.rerun() # Rerun immediately to clear canvas visually

# Handle ROI clear confirmation after rerun
if st.session_state.get("clear_roi_triggered", False):
    st.success("ROI cleared!")
    st.balloons()
    st.session_state.clear_roi_triggered = False # Reset flag


# --- Demo Mode: Load a Demo Image ---
# (Your existing Demo Mode logic remains unchanged here)
demo_mode = st.sidebar.checkbox("Demo Mode", help="Load a demo image & sample analysis.")
if demo_mode and "demo_loaded" not in st.session_state:
    demo_path = os.path.join("assets", "demo.png")
    if os.path.exists(demo_path):
        try:
            demo_img = Image.open(demo_path).convert("RGB")
            # Reset relevant states when demo mode is activated
            for key, value in DEFAULT_STATE.items():
                 if key not in {"file_uploader_widget"}: # Don't reset the widget itself
                    st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value

            st.session_state.display_image = demo_img
            st.session_state.processed_image = demo_img
            st.session_state.session_id = "demo" + str(uuid.uuid4())[:4] # Unique demo session
            st.session_state.history = [("System", "Demo mode activated. Sample analysis loaded.")]
            st.session_state.initial_analysis = "This is a demonstration analysis.\n\nFindings:\n1. Potential area of interest noted in the upper lobe.\n2. No other significant abnormalities detected.\n\nImpression: Suspicious finding requires further investigation."
            st.session_state.demo_loaded = True
            st.success("Demo mode activated! Demo image loaded.")
            logger.info("Demo mode activated.")
            st.rerun() # Rerun to update UI with demo content
        except Exception as e:
            st.sidebar.error(f"Error loading demo image: {e}")
            logger.error(f"Failed to load demo image: {e}", exc_info=True)
            st.session_state.demo_loaded = False # Ensure flag is reset on error
    else:
        st.sidebar.warning("Demo image (demo.png) not found in assets folder.")
elif not demo_mode and st.session_state.get("demo_loaded"):
    # If demo mode is unchecked, clear demo state
    logger.info("Demo mode deactivated.")
    for key, value in DEFAULT_STATE.items():
        if key not in {"file_uploader_widget"}:
             st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value
    st.session_state.demo_loaded = False
    st.session_state.session_id = str(uuid.uuid4())[:8] # New real session ID
    st.info("Demo mode deactivated. Upload an image to begin.")
    st.rerun()


# --- Helper: Convert PIL Image to Data URL Safely ---
# (Your existing safe_image_to_data_url function remains unchanged here)
def safe_image_to_data_url(img: Image.Image) -> str:
    if not isinstance(img, Image.Image):
        logger.warning(f"Attempted to convert non-PIL Image to data URL: {type(img)}")
        return ""
    try:
        buffered = io.BytesIO()
        fmt = "PNG" # Always use PNG for data URLs for broad compatibility
        temp_img = img

        # Handle different image modes gracefully
        if img.mode == 'P': # Palette mode often needs RGBA for transparency
            temp_img = img.convert('RGBA')
        elif img.mode == 'CMYK' or img.mode == 'I': # Convert complex modes to RGB
             temp_img = img.convert('RGB')
        elif img.mode not in ['RGB', 'L', 'RGBA']: # Fallback for other modes
             temp_img = img.convert('RGB') # Convert to RGB if not already compatible

        temp_img.save(buffered, format=fmt)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{fmt.lower()};base64,{img_str}"
    except Exception as e:
        logger.error(f"Failed converting image to data URL: {e}", exc_info=True)
        return ""


# --- Initialize Session State Defaults ---
DEFAULT_STATE = {
    "uploaded_file_info": None,
    "raw_image_bytes": None,
    "is_dicom": False,
    "dicom_dataset": None,
    "dicom_metadata": {},
    "processed_image": None, # Image used for AI analysis (potentially normalized)
    "display_image": None,   # Image shown in viewer (can have W/L applied)
    "session_id": None,
    "history": [],           # List of tuples (question, answer)
    "initial_analysis": "",
    "qa_answer": "",         # Last Q&A answer
    "disease_analysis": "",
    "confidence_score": "",
    "last_action": None,     # Track button clicks
    "pdf_report_bytes": None,
    "canvas_drawing": None,  # Store canvas state for ROI
    "roi_coords": None,      # Processed ROI coordinates {left, top, width, height}
    "current_display_wc": None, # DICOM window center
    "current_display_ww": None, # DICOM window width
    "translation_output": "", # Store the last translation result
    "translation_src_lang": "Auto-Detect", # Default source language
    "translation_tgt_lang": "Spanish", # Default target language
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value

# Ensure history is always a list
if not isinstance(st.session_state.history, list):
    st.session_state.history = []

logger.debug("Session state initialized/verified.")

# --- Ensure a Session ID Exists ---
if not st.session_state.get("session_id"):
    st.session_state.session_id = str(uuid.uuid4())[:8]
    logger.info(f"New session started: {st.session_state.session_id}")


# --- Page Title & Usage Guide ---
st.title("⚕️ RadVision QA Advanced: AI-Assisted Image Analysis")
with st.expander("Usage Guide", expanded=False):
    st.info("⚠️ Disclaimer: This tool is for research/informational purposes only. AI-generated insights should always be verified by a qualified medical professional before making any clinical decisions.")
    st.markdown(
        "**Workflow:**\n"
        "1.  **Upload:** Use the sidebar to upload a JPG, PNG, or DICOM image.\n"
        "2.  **(Optional) DICOM Adjust:** If DICOM, use sliders (sidebar) to adjust Window Center (WC) / Window Width (WW) for optimal viewing.\n"
        "3.  **(Optional) ROI:** Draw a rectangle on the image (left panel) to define a Region of Interest for focused analysis.\n"
        "4.  **Analyze:** Use the 'AI Actions' in the sidebar:\n"
        "    *   `Run Initial Analysis`: Get overall findings.\n"
        "    *   `Ask AI a Question`: Query the AI about the image/ROI.\n"
        "    *   `Run Condition Analysis`: Check for specific conditions.\n"
        "5.  **Review:** Check results in the tabs (right panel): Initial Analysis, Q&A, Disease Focus.\n"
        "6.  **(Optional) Translate:** Go to the 'Translation' tab to translate results into other languages.\n"
        "7.  **Confidence & Report:** Estimate AI confidence and generate/download a PDF summary."
    )
st.markdown("---")

# --- Sidebar: Upload, DICOM W/L, and AI Actions ---
with st.sidebar:
    st.header("Image Upload & Controls")
    uploaded_file = st.file_uploader(
        "Upload Image (JPG, PNG, DCM)",
        type=["jpg", "jpeg", "png", "dcm", "dicom"],
        key="file_uploader_widget", # Consistent key
        help="Upload a JPG, PNG, or DICOM file to analyze."
    )

    # --- File Processing Logic ---
    if uploaded_file is not None:
        # Create a unique identifier for the uploaded file to detect changes
        try:
            # Use modification time if available, otherwise hash content
            file_mtime = getattr(uploaded_file, 'last_modified', None)
            if file_mtime is None:
                import hashlib
                hasher = hashlib.md5()
                file_content = uploaded_file.getvalue() # Read content once
                hasher.update(file_content)
                file_unique_id = hasher.hexdigest()
                uploaded_file.seek(0) # Reset cursor after reading
            else:
                file_unique_id = str(file_mtime)

            new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{file_unique_id}"
            raw_bytes = uploaded_file.getvalue() # Use pre-read content if hashed
            uploaded_file.seek(0) # Always reset cursor

        except Exception as err:
            logger.error(f"Error generating file info: {err}", exc_info=True)
            new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{str(uuid.uuid4())[:8]}" # Fallback ID
            raw_bytes = uploaded_file.getvalue()
            uploaded_file.seek(0)

        # Check if it's a new file compared to the last one processed
        if new_file_info != st.session_state.get("uploaded_file_info"):
            logger.info(f"Processing new file: {uploaded_file.name} (Size: {uploaded_file.size})")
            st.toast(f"Processing '{uploaded_file.name}'...", icon="⏳")

            # Reset state for the new file (keep file uploader widget state)
            current_widget_state = st.session_state.get("file_uploader_widget")
            current_demo_state = st.session_state.get("demo_loaded", False) # Preserve demo state if active

            for key, value in DEFAULT_STATE.items():
                 st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value

            st.session_state.file_uploader_widget = current_widget_state # Restore widget state
            st.session_state.demo_loaded = current_demo_state # Restore demo state

            st.session_state.uploaded_file_info = new_file_info
            st.session_state.session_id = str(uuid.uuid4())[:8] # New session for new file
            st.session_state.raw_image_bytes = raw_bytes
            st.session_state.demo_loaded = False # Turn off demo mode if a real file is uploaded

            # Determine if DICOM
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            is_dicom_likely = (pydicom is not None and ("dicom" in uploaded_file.type.lower() or file_ext in (".dcm", ".dicom")))

            with st.spinner("🔬 Processing image..."):
                temp_display = None
                temp_processed = None
                is_dicom_confirmed = False
                success = False

                if is_dicom_likely:
                    try:
                        ds = parse_dicom(st.session_state.raw_image_bytes, uploaded_file.name)
                        if ds:
                            st.session_state.dicom_dataset = ds
                            st.session_state.dicom_metadata = extract_dicom_metadata(ds)
                            wc, ww = get_default_wl(ds)
                            st.session_state.current_display_wc, st.session_state.current_display_ww = wc, ww
                            temp_display = dicom_to_image(ds, wc, ww) # Image for display with initial W/L
                            # Processed image: Use default W/L or normalized for AI consistency
                            temp_processed = dicom_to_image(ds, wc, ww, normalize=False) # Keep closer to original for AI
                            if isinstance(temp_display, Image.Image) and isinstance(temp_processed, Image.Image):
                                is_dicom_confirmed = True
                                success = True
                            else:
                                logger.warning("DICOM parsing succeeded but image conversion failed.")
                        else:
                             logger.warning("parse_dicom returned None, treating as non-DICOM.")
                    except pydicom.errors.InvalidDicomError:
                        logger.warning(f"File '{uploaded_file.name}' is not a valid DICOM. Attempting standard image processing.")
                        is_dicom_likely = False # Fallback to standard image processing
                    except Exception as e:
                        st.error(f"Error processing DICOM file: {e}")
                        logger.error(f"DICOM processing failed: {e}", exc_info=True)

                # If not DICOM or DICOM processing failed, try standard image processing
                if not is_dicom_confirmed:
                    st.session_state.is_dicom = False
                    try:
                        raw_img = Image.open(io.BytesIO(st.session_state.raw_image_bytes)).convert("RGB")
                        temp_display = raw_img.copy()
                        temp_processed = raw_img.copy()
                        success = True
                    except UnidentifiedImageError:
                        st.error("Unsupported image format. Please upload JPG, PNG, or a valid DICOM file.")
                        logger.error("UnidentifiedImageError during standard image processing.")
                    except Exception as e:
                        st.error(f"Error processing standard image file: {e}")
                        logger.error(f"Standard image processing failed: {e}", exc_info=True)

                # Finalize state update if processing was successful
                if success and isinstance(temp_display, Image.Image) and isinstance(temp_processed, Image.Image):
                    st.session_state.is_dicom = is_dicom_confirmed
                    # Ensure images are in RGB for display consistency if needed
                    if temp_display.mode != 'RGB': temp_display = temp_display.convert('RGB')
                    if temp_processed.mode != 'RGB': temp_processed = temp_processed.convert('RGB')

                    st.session_state.display_image = temp_display
                    st.session_state.processed_image = temp_processed
                    st.success(f"✅ Image '{uploaded_file.name}' processed successfully!")
                    logger.info(f"Image processed successfully. DICOM: {st.session_state.is_dicom}")
                    st.rerun() # Rerun to update the UI with the new image
                else:
                    st.error("Image loading failed. Please check the file format or integrity.")
                    st.session_state.uploaded_file_info = None # Reset file info on failure
                    logger.error("Image processing failed overall.")

    # --- DICOM Window/Level Sliders ---
    if st.session_state.is_dicom and st.session_state.dicom_dataset is not None:
        st.markdown("---")
        st.subheader("DICOM Windowing")
        # Use the ui_components function for sliders
        wc, ww = dicom_wl_sliders(
            st.session_state.dicom_dataset,
            st.session_state.current_display_wc,
            st.session_state.current_display_ww
        )
        # Check if values changed
        if wc != st.session_state.current_display_wc or ww != st.session_state.current_display_ww:
            logger.info(f"DICOM W/L changed: WC={wc}, WW={ww}")
            st.session_state.current_display_wc = wc
            st.session_state.current_display_ww = ww
            with st.spinner("Updating view..."):
                new_display_img = dicom_to_image(st.session_state.dicom_dataset, wc, ww)
                if isinstance(new_display_img, Image.Image):
                    if new_display_img.mode != 'RGB': new_display_img = new_display_img.convert('RGB')
                    st.session_state.display_image = new_display_img
                    # Note: We usually don't update st.session_state.processed_image here,
                    # as the AI should ideally work on a consistent representation.
                else:
                     st.warning("Failed to update DICOM view with new W/L settings.")
                     logger.warning("dicom_to_image failed during W/L update.")
            st.rerun() # Rerun to show the updated image

    st.markdown("---")
    st.header("AI Actions")

    # --- AI Action Buttons ---
    img_available = isinstance(st.session_state.get("processed_image"), Image.Image)

    if st.button("▶️ Run Initial Analysis", key="analyze_btn", help="Perform a general analysis of the image.", disabled=not img_available):
        st.session_state.last_action = "analyze"
        st.rerun()

    st.subheader("❓ Ask AI a Question")
    question_input = st.text_area(
        "Your Question:", height=80, key="question_input_widget",
        placeholder="e.g., 'Are there any signs of fracture?' or 'Describe the highlighted region.'",
        disabled=not img_available
    )
    if st.button("💬 Ask AI", key="ask_btn", help="Submit your question about the image/ROI to the AI.", disabled=not img_available):
        if question_input.strip():
            st.session_state.last_action = "ask"
            st.rerun()
        else:
            st.warning("Please enter a question before submitting.")

    st.subheader("🎯 Condition Analysis")
    # Common conditions - you might want to customize this list
    DISEASE_OPTIONS = [
        "Pneumonia", "Lung Cancer", "Nodule/Mass", "Effusion", "Pneumothorax",
        "Fracture", "Arthritis", "Osteoporosis",
        "Stroke", "Brain Tumor", "Hemorrhage",
        "Appendicitis", "Bowel Obstruction", "Cardiomegaly", "Aortic Aneurysm",
        "Tuberculosis", "COVID-19", "Pulmonary Embolism", "Other (Specify in Q&A)"
    ]
    disease_select = st.selectbox(
        "Select Condition:", [""] + sorted(DISEASE_OPTIONS), key="disease_select_widget",
        help="Select a potential condition for focused analysis.",
        disabled=not img_available
    )
    if st.button("🩺 Run Condition Analysis", key="disease_btn", help="Analyze the image specifically for the selected condition.", disabled=not img_available):
        if disease_select:
            st.session_state.last_action = "disease"
            st.rerun()
        else:
            st.warning("Please select a condition to analyze.")

    st.subheader("📊 Confidence & Report")
    # Enable confidence if there's any text output from the AI
    can_estimate = bool(st.session_state.history or st.session_state.initial_analysis or st.session_state.disease_analysis)
    if st.button("📈 Estimate Confidence", key="confidence_btn", disabled=not can_estimate, help="Estimate the AI's confidence based on the analysis performed so far."):
        if can_estimate:
            st.session_state.last_action = "confidence"
            st.rerun()
        # No warning needed here as button is disabled

    # Enable PDF generation if an image has been processed
    if st.button("📄 Generate PDF Data", key="generate_report_data_btn", help="Compile the analysis results into a PDF report.", disabled=not img_available):
        st.session_state.last_action = "generate_report_data"
        st.rerun()

    # Download button appears only after PDF bytes are generated
    if st.session_state.get("pdf_report_bytes"):
        fname = f"RadVisionAI_Report_{st.session_state.session_id or 'session'}.pdf"
        st.download_button(
            label="⬇️ Download PDF Report",
            data=st.session_state.pdf_report_bytes,
            file_name=fname,
            mime="application/pdf",
            key="download_pdf_button",
            help="Click to download the generated PDF report."
        )
        # Optionally clear the bytes after showing the button once to avoid clutter
        # Consider adding a button to explicitly clear the generated report state if needed
        # st.session_state.pdf_report_bytes = None

# --- Main Content: Two-Column Layout ---
col1, col2 = st.columns([2, 3]) # Adjust ratio if needed (e.g., [1, 1] or [3, 2])

with col1:
    st.subheader("🖼️ Image Viewer")
    display_img = st.session_state.get("display_image")

    if isinstance(display_img, Image.Image):
        # Display the image first
        st.image(display_img, caption="Current View" + (" (DICOM)" if st.session_state.is_dicom else ""), use_container_width=True)

        # ROI Drawing Canvas Section
        st.markdown("---")
        st.caption("Draw a rectangle below to select a Region of Interest (ROI). Use sidebar button to clear.")
        # Adjust canvas size dynamically but within limits
        MAX_CANVAS_WIDTH = 600 # Max width in pixels
        MAX_CANVAS_HEIGHT = 500 # Max height in pixels
        img_w, img_h = display_img.size

        if img_w > 0 and img_h > 0:
            aspect_ratio = img_w / img_h
            canvas_width = min(img_w, MAX_CANVAS_WIDTH)
            canvas_height = int(canvas_width / aspect_ratio)

            # Fit to max height if necessary
            if canvas_height > MAX_CANVAS_HEIGHT:
                canvas_height = MAX_CANVAS_HEIGHT
                canvas_width = int(canvas_height * aspect_ratio)

            # Ensure minimum dimensions
            canvas_width = max(int(canvas_width), 150)
            canvas_height = max(int(canvas_height), 150)

            try:
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.2)", # Semi-transparent orange fill
                    stroke_width=2,
                    stroke_color="rgba(230, 50, 50, 0.9)", # Red border
                    background_image=display_img, # Display the current view in canvas
                    update_streamlit=True, # Update Streamlit state on drawing events
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="rect", # Only allow rectangle drawing
                    initial_drawing=st.session_state.get("canvas_drawing", None), # Restore previous drawing if available
                    key="drawable_canvas" # Unique key for the canvas
                )

                # Process canvas results if drawing exists and has objects
                if canvas_result.json_data and canvas_result.json_data.get("objects"):
                    # Get the last drawn object (assuming only one ROI rectangle is intended)
                    last_obj = canvas_result.json_data["objects"][-1]
                    if last_obj["type"] == "rect":
                        # Extract rectangle properties, considering potential scaling factors
                        scaleX = last_obj.get("scaleX", 1.0)
                        scaleY = last_obj.get("scaleY", 1.0)
                        left_val = int(last_obj["left"])
                        top_val = int(last_obj["top"])
                        width_val = int(last_obj["width"] * scaleX)
                        height_val = int(last_obj["height"] * scaleY)

                        # Calculate scaling factor from displayed canvas size to original image size
                        scale_x_img = img_w / canvas_width
                        scale_y_img = img_h / canvas_height

                        # Calculate ROI coordinates in the original image dimensions
                        orig_left = max(0, int(left_val * scale_x_img))
                        orig_top = max(0, int(top_val * scale_y_img))
                        orig_width = min(img_w - orig_left, int(width_val * scale_x_img))
                        orig_height = min(img_h - orig_top, int(height_val * scale_y_img))

                        # Store the calculated ROI coordinates if they are valid
                        if orig_width > 0 and orig_height > 0:
                            new_roi = {"left": orig_left, "top": orig_top, "width": orig_width, "height": orig_height}
                            # Update session state only if ROI changed to avoid unnecessary reruns
                            if st.session_state.roi_coords != new_roi:
                                st.session_state.roi_coords = new_roi
                                st.session_state.canvas_drawing = canvas_result.json_data # Save canvas state
                                logger.info(f"ROI updated: {new_roi}")
                                st.rerun() # Rerun to reflect potential ROI-based changes elsewhere
                        else:
                             # If calculated ROI is invalid (e.g., zero width/height), clear existing ROI
                             if st.session_state.roi_coords is not None:
                                 st.session_state.roi_coords = None
                                 logger.info("ROI cleared due to invalid dimensions.")
                                 st.rerun()

                # Handle case where drawing exists but has no objects (e.g., cleared)
                elif canvas_result.json_data and not canvas_result.json_data.get("objects"):
                     if st.session_state.roi_coords is not None:
                         st.session_state.roi_coords = None
                         st.session_state.canvas_drawing = None # Clear canvas state as well
                         logger.info("ROI cleared via canvas interaction.")
                         st.rerun()

            except Exception as e:
                st.error(f"Error initializing drawing canvas: {e}")
                logger.error(f"Canvas error: {e}", exc_info=True)

        else:
            st.warning("Image has invalid dimensions, cannot enable ROI drawing.")

        # --- DICOM Metadata Display ---
        if st.session_state.is_dicom and st.session_state.dicom_metadata:
            st.markdown("---")
            with st.expander("DICOM Metadata", expanded=False):
                # Use the ui_components function to display metadata
                display_dicom_metadata(st.session_state.dicom_metadata)

    else:
        # Message when no image is loaded
        st.info("Upload an image using the sidebar to begin analysis.")
        if st.session_state.get("demo_loaded"):
            st.info("Demo mode is active. Image and sample analysis are loaded.")

with col2:
    st.subheader("📊 Analysis & Results")

    # Define tab titles including the new Translation tab
    tab_titles = ["🔬 Initial Analysis", "💬 Q&A History", "🩺 Disease Focus", "📈 Confidence", "🌐 Translation"]
    tabs = st.tabs(tab_titles)

    # Tab 0: Initial Analysis
    with tabs[0]:
        st.text_area(
            "Overall Findings & Impressions",
            value=st.session_state.initial_analysis or "Run 'Initial Analysis' from the sidebar.",
            height=450,
            key="output_initial",
            disabled=True, # Display only
            help="Results from the 'Run Initial Analysis' action."
        )

    # Tab 1: Q&A History
    with tabs[1]:
        st.text_area(
            "Latest AI Answer",
            value=st.session_state.qa_answer or "Ask a question using the sidebar.",
            height=200,
            key="output_qa",
            disabled=True, # Display only
            help="The most recent answer from the 'Ask AI' action."
        )
        st.markdown("---")
        if st.session_state.history:
            with st.expander("Full Conversation History", expanded=True):
                # Display history in reverse chronological order (newest first)
                for i, (q, a) in enumerate(reversed(st.session_state.history)):
                    st.markdown(f"**You ({len(st.session_state.history) - i}):** {q}")
                    # Use markdown with unsafe_allow_html=True if answers might contain HTML/Markdown formatting
                    st.markdown(f"**AI ({len(st.session_state.history) - i}):**\n{a}", unsafe_allow_html=True)
                    if i < len(st.session_state.history) - 1:
                        st.markdown("---") # Separator between entries
        else:
            st.caption("No questions asked yet in this session.")

    # Tab 2: Disease Focus
    with tabs[2]:
        st.text_area(
            "Disease-Specific Analysis",
            value=st.session_state.disease_analysis or "Run 'Condition Analysis' from the sidebar.",
            height=450,
            key="output_disease",
            disabled=True, # Display only
            help="Results from the 'Run Condition Analysis' action for a selected condition."
        )

    # Tab 3: Confidence
    with tabs[3]:
        st.text_area(
            "AI Confidence Estimation",
            value=st.session_state.confidence_score or "Run 'Estimate Confidence' from the sidebar.",
            height=450,
            key="output_confidence",
            disabled=True, # Display only
            help="An estimation of the AI's confidence based on the generated analyses."
        )

    # Tab 4: Translation - NEW TAB
    with tabs[4]:
        st.subheader("🌐 Translate Analysis Results")

        # Check if translation functions are available
        if not translate or not detect_language or not LANGUAGE_CODES:
            st.warning("Translation features are currently unavailable. Please check the application setup.")
            logger.warning("Translation tab accessed but translation functions/codes are missing.")
        else:
            st.caption("Translate analysis text or enter custom text.")

            # 1. Select text source
            text_options = {
                "Initial Analysis": st.session_state.initial_analysis,
                "Latest Q&A Answer": st.session_state.qa_answer,
                "Disease Analysis": st.session_state.disease_analysis,
                "Confidence Estimation": st.session_state.confidence_score,
                "(Enter Custom Text Below)": "" # Placeholder for custom input
            }
            selected_label = st.selectbox(
                "Select text to translate:",
                options=list(text_options.keys()),
                key="translate_source_select"
            )

            # 2. Text Area for custom input or display selected text
            if selected_label == "(Enter Custom Text Below)":
                text_to_translate_input = st.text_area(
                    "Enter text here:", "", height=150, key="translate_custom_input"
                )
            else:
                # Display the selected text in a disabled text area
                text_to_translate_input = st.text_area(
                    f"Selected text ({selected_label}):",
                    value=text_options[selected_label] or "Source text is empty.",
                    height=150,
                    key="translate_selected_display",
                    disabled=True
                )

            # Determine the actual text to use for translation
            text_to_translate = text_to_translate_input if selected_label == "(Enter Custom Text Below)" else text_options[selected_label]

            # 3. Language Selection
            lang_names = list(LANGUAGE_CODES.keys())
            try:
                 default_tgt_index = lang_names.index(st.session_state.translation_tgt_lang)
            except ValueError:
                 default_tgt_index = 0 # Default to first language if previous selection invalid

            col_lang1, col_lang2 = st.columns(2)
            with col_lang1:
                src_lang_options = ["Auto-Detect"] + lang_names
                try:
                    default_src_index = src_lang_options.index(st.session_state.translation_src_lang)
                except ValueError:
                    default_src_index = 0 # Default to Auto-Detect
                selected_src_lang_name = st.selectbox(
                    "Source Language:",
                    options=src_lang_options,
                    index=default_src_index,
                    key="translate_source_lang_select"
                )
            with col_lang2:
                selected_tgt_lang_name = st.selectbox(
                    "Target Language:",
                    options=lang_names,
                    index=default_tgt_index,
                    key="translate_target_lang_select"
                )

            # 4. Translate Button
            if st.button("Translate Now", key="translate_button_go"):
                st.session_state.translation_src_lang = selected_src_lang_name # Save selection
                st.session_state.translation_tgt_lang = selected_tgt_lang_name # Save selection

                if not text_to_translate or not text_to_translate.strip():
                    st.warning("Please select or enter text to translate.")
                elif selected_src_lang_name == selected_tgt_lang_name and selected_src_lang_name != "Auto-Detect":
                    st.info("Source and Target languages are the same. No translation needed.")
                    st.session_state.translation_output = text_to_translate
                else:
                    with st.spinner(f"Translating to {selected_tgt_lang_name}..."):
                        try:
                            actual_src_lang_name = selected_src_lang_name
                            detected_code = None

                            # Auto-Detection Logic
                            if selected_src_lang_name == "Auto-Detect":
                                detected_code = detect_language(text_to_translate[:500]) # Detect based on first 500 chars
                                logger.info(f"Attempting auto-detection, detected code: {detected_code}")
                                detected_lang_found = False
                                if detected_code:
                                    for name, code in LANGUAGE_CODES.items():
                                        # Handle variations like zh-CN vs zh
                                        if detected_code.lower() == code.lower() or detected_code.lower().split('-')[0] == code.lower().split('-')[0]:
                                            actual_src_lang_name = name
                                            st.info(f"Auto-detected source language: **{name}** ({detected_code})")
                                            detected_lang_found = True
                                            break
                                if not detected_lang_found:
                                    st.warning("Could not reliably auto-detect source language. Assuming English.")
                                    logger.warning(f"Auto-detection failed or code '{detected_code}' not in LANGUAGE_CODES. Defaulting to English.")
                                    actual_src_lang_name = "English" # Fallback if detection fails or not in our list

                            # Check again if source and target are now the same after detection
                            if actual_src_lang_name == selected_tgt_lang_name:
                                st.info(f"Detected source language ({actual_src_lang_name}) is the same as the target. No translation needed.")
                                st.session_state.translation_output = text_to_translate
                            else:
                                # Get language codes for the API call (if needed by your translate function)
                                # src_code = LANGUAGE_CODES.get(actual_src_lang_name)
                                # tgt_code = LANGUAGE_CODES.get(selected_tgt_lang_name)

                                # --- Call your translation function ---
                                # Adapt this call based on how your `translate` function works.
                                # This example assumes it takes text, target language name, and source language name.
                                raw_translation = translate(
                                    text_to_translate,
                                    target_language=selected_tgt_lang_name,
                                    source_language=actual_src_lang_name
                                )
                                # ---

                                if raw_translation:
                                    # Apply post-processing formatting
                                    final_trans = format_translation(raw_translation)
                                    st.session_state.translation_output = final_trans
                                    st.success("Translation complete!")
                                    logger.info(f"Translation successful: {actual_src_lang_name} -> {selected_tgt_lang_name}")
                                else:
                                    st.error("Translation failed. The translation service might be unavailable or returned an empty result.")
                                    logger.error(f"Translation failed for: {actual_src_lang_name} -> {selected_tgt_lang_name}")
                                    st.session_state.translation_output = "Error: Translation failed."

                        except Exception as e:
                            st.error(f"An error occurred during translation: {e}")
                            logger.error(f"Translation exception: {e}", exc_info=True)
                            st.session_state.translation_output = f"Error: {e}"

            # 5. Display Translation Output
            st.text_area(
                "Translated Text:",
                value=st.session_state.translation_output or "",
                height=250,
                key="translate_output_area",
                disabled=True, # Usually display only, user can copy
                help="The result of the translation will appear here."
            )


# --- ACTION HANDLING LOGIC ---
# (This section processes button clicks stored in st.session_state.last_action)
if 'last_action' in st.session_state and st.session_state.last_action:
    current_action = st.session_state.last_action
    logger.info(f"Handling action: {current_action}")

    # Reset action flag immediately to prevent re-triggering on subsequent reruns
    st.session_state.last_action = None

    # Ensure image exists for actions that require it
    if current_action not in ["generate_report_data"] and not isinstance(st.session_state.get("processed_image"), Image.Image):
        st.warning(f"Cannot perform '{current_action}': No valid image loaded.")
        logger.warning(f"Action '{current_action}' skipped: processed_image is invalid or None.")
        st.stop() # Stop processing this action

    # Ensure session ID exists
    if not st.session_state.get("session_id"):
        st.error(f"Cannot perform '{current_action}': Session ID is missing. Please reload.")
        logger.error(f"Action '{current_action}' aborted: missing session ID.")
        st.stop() # Stop processing this action

    # Prepare common variables
    img_llm = st.session_state.processed_image
    roi = st.session_state.get("roi_coords") # Use .get for safety
    roi_str = " (with ROI)" if roi else ""
    history = st.session_state.history if isinstance(st.session_state.history, list) else []
    st.session_state.history = history # Ensure it's assigned back if corrected

    try:
        if current_action == "analyze":
            st.info(f"🔬 Performing initial analysis{roi_str}...")
            with st.spinner("AI is analyzing the image..."):
                result = run_initial_analysis(img_llm, roi) # Pass ROI if available
            st.session_state.initial_analysis = result
            # Clear other analysis fields when starting a new initial analysis
            st.session_state.qa_answer = ""
            st.session_state.disease_analysis = ""
            st.session_state.confidence_score = ""
            logger.info(f"Initial analysis completed{roi_str}.")
            st.rerun() # Rerun to display results in the correct tab

        elif current_action == "ask":
            q = st.session_state.question_input_widget.strip() # Get question from widget state
            if not q:
                st.warning("Question field was empty.")
                logger.warning("Attempted 'ask' action with empty question.")
            else:
                st.info(f"❓ Asking AI: \"{q}\"{roi_str}...")
                st.session_state.qa_answer = "" # Clear previous answer display
                with st.spinner("AI is processing your question..."):
                    # Pass history for context
                    answer, ok = run_multimodal_qa(img_llm, q, history, roi)
                if ok:
                    st.session_state.qa_answer = answer
                    st.session_state.history.append((q, answer)) # Add to history
                    logger.info(f"Q&A successful for: '{q}'{roi_str}")
                else:
                    # Handle primary AI failure
                    fail_msg = f"Primary AI failed to answer: {answer}"
                    st.session_state.qa_answer = fail_msg
                    st.error("Primary AI query failed. Attempting fallback if configured...")
                    logger.warning(f"Primary AI Q&A failed for '{q}'. Reason: {answer}")

                    # Attempt Fallback (using HF example structure)
                    hf_token_exists = bool(os.environ.get("HF_API_TOKEN")) # Check if HF token is set
                    if hf_token_exists and HF_VQA_MODEL_ID != "hf_model_not_found":
                        logger.info(f"Attempting HF fallback using {HF_VQA_MODEL_ID}.")
                        with st.spinner(f"Trying fallback with {HF_VQA_MODEL_ID}..."):
                            fb_ans, fb_ok = query_hf_vqa_inference_api(img_llm, q, roi)
                        if fb_ok:
                            fb_disp = f"**[Fallback Result: {HF_VQA_MODEL_ID}]**\n\n{fb_ans}"
                            st.session_state.qa_answer += "\n\n" + fb_disp # Append fallback answer
                            st.session_state.history.append((f"[Fallback Query] {q}", fb_disp)) # Add fallback to history
                            st.info("Fallback AI provided an answer.")
                            logger.info(f"HF fallback successful for '{q}'.")
                        else:
                            fb_fail_msg = f"**[Fallback Error - {HF_VQA_MODEL_ID}]:** {fb_ans}"
                            st.session_state.qa_answer += "\n\n" + fb_fail_msg
                            st.error("Fallback AI also failed.")
                            logger.error(f"HF fallback failed for '{q}'. Reason: {fb_ans}")
                    else:
                         no_fb_msg = "\n\n**[Fallback Unavailable]** No fallback model configured or API token missing."
                         st.session_state.qa_answer += no_fb_msg
                         logger.warning("HF fallback skipped: Model/Token not configured.")
            st.rerun() # Rerun to display answer/errors

        elif current_action == "disease":
            d = st.session_state.disease_select_widget # Get selected disease from widget state
            if not d:
                st.warning("No condition was selected for analysis.")
                logger.warning("Attempted 'disease' action with no condition selected.")
            else:
                st.info(f"🩺 Running focused analysis for '{d}'{roi_str}...")
                with st.spinner(f"AI is analyzing for signs of {d}..."):
                    result = run_disease_analysis(img_llm, d, roi) # Pass ROI
                st.session_state.disease_analysis = result
                # Clear other analysis fields
                st.session_state.qa_answer = ""
                st.session_state.confidence_score = ""
                logger.info(f"Disease analysis completed for '{d}'{roi_str}.")
            st.rerun() # Rerun to display results

        elif current_action == "confidence":
            # Check again if there is context, though button should be disabled if not
            if not (history or st.session_state.initial_analysis or st.session_state.disease_analysis):
                st.warning("Cannot estimate confidence without prior analysis or Q&A.")
                logger.warning("Confidence estimation skipped: no context available.")
            else:
                st.info(f"📊 Estimating AI confidence{roi_str}...")
                context_summary = f"Initial: {bool(st.session_state.initial_analysis)}, Q&A: {len(history)} entries, Disease: {bool(st.session_state.disease_analysis)}"
                logger.info(f"Running confidence estimation with context: {context_summary}")
                with st.spinner("AI is assessing its confidence..."):
                    # Pass relevant context to the confidence function
                    res = estimate_ai_confidence(
                        img_llm,
                        history,
                        st.session_state.initial_analysis,
                        st.session_state.disease_analysis,
                        roi # Pass ROI if it influenced the analyses
                    )
                st.session_state.confidence_score = res
                logger.info("Confidence estimation completed.")
            st.rerun() # Rerun to display score

        elif current_action == "generate_report_data":
            st.info("📄 Generating PDF report data...")
            st.session_state.pdf_report_bytes = None # Clear previous report bytes
            img_for_report = st.session_state.get("display_image")

            if not isinstance(img_for_report, Image.Image):
                st.error("Cannot generate report: Invalid or missing display image.")
                logger.error("PDF generation aborted: invalid display_image.")
            else:
                img_final = img_for_report.copy() # Work on a copy
                # Draw ROI on the image copy if ROI exists
                if roi:
                    try:
                        draw = ImageDraw.Draw(img_final)
                        x0, y0 = roi['left'], roi['top']
                        x1 = x0 + roi['width']
                        y1 = y0 + roi['height']
                        # Draw a distinct rectangle (e.g., red, thickness 3)
                        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
                        logger.info("ROI drawn on image for PDF report.")
                    except Exception as e:
                        logger.error(f"Error drawing ROI on image for PDF: {e}", exc_info=True)
                        # Continue report generation without the drawn ROI

                # Compile all text outputs for the report
                full_history = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in history]) if history else "No conversation history recorded."
                outputs = {
                    "Session ID": st.session_state.session_id or "N/A",
                    "Initial Analysis": st.session_state.initial_analysis or "Not Performed",
                    "Conversation History": full_history,
                    "Condition Analysis": st.session_state.disease_analysis or "Not Performed",
                    "Confidence Estimation": st.session_state.confidence_score or "Not Estimated",
                }

                # Add DICOM metadata summary if available
                if st.session_state.is_dicom and st.session_state.dicom_metadata:
                    # Include a few key fields or just a note
                    meta_summary = {k: v for k, v in st.session_state.dicom_metadata.items() if k in ['PatientName', 'PatientID', 'StudyDescription', 'Modality']}
                    outputs["DICOM Summary"] = str(meta_summary) if meta_summary else "Basic DICOM metadata available."

                # Generate the PDF bytes
                with st.spinner("Generating PDF document..."):
                    pdf_bytes = generate_pdf_report_bytes(
                        st.session_state.session_id,
                        img_final, # Image with ROI drawn (if applicable)
                        outputs, # Dictionary of text results
                        st.session_state.dicom_metadata if st.session_state.is_dicom else None # Pass full metadata if needed
                    )

                if pdf_bytes:
                    st.session_state.pdf_report_bytes = pdf_bytes
                    st.success("PDF report generated successfully! Download button available.")
                    logger.info("PDF generation successful.")
                    st.balloons() # Fun success indicator!
                else:
                    st.error("Failed to generate PDF report. The report generation function might have encountered an issue.")
                    logger.error("PDF generation failed: generate_pdf_report_bytes returned None or empty.")
            st.rerun() # Rerun to show download button or error

        else:
            st.warning(f"Unknown action '{current_action}' encountered.")
            logger.warning(f"Attempted to handle unknown action: '{current_action}'")
            # No rerun needed if action is unknown

    except Exception as e:
        st.error(f"An error occurred while handling action '{current_action}': {e}")
        logger.critical(f"Unhandled exception during action '{current_action}': {e}", exc_info=True)
        # Rerun to clear the action state and potentially show error message correctly
        st.rerun()

# --- Footer ---
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced | Session ID: {st.session_state.get('session_id', 'N/A')}")

# Simple Footer Example (Update links as needed)
st.markdown(
    """
    <div style="text-align: center; font-size: 0.8em; color: #888; margin-top: 2em;">
      <a href="#" target="_blank">Privacy Policy</a> |
      <a href="#" target="_blank">Terms of Service</a> |
      <a href="#" target="_blank">Documentation</a>
      <br>
      RadVision AI is intended for informational purposes only. Consult a qualified healthcare professional for diagnosis.
    </div>
    """, unsafe_allow_html=True
)

logger.info("--- App Render Complete ---")