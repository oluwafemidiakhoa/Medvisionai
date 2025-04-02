import io
import os
import uuid
import logging
import base64
from typing import Any, Dict, Optional, Tuple

import streamlit as st
from PIL import Image, ImageDraw, UnidentifiedImageError
import pydicom
from streamlit_drawable_canvas import st_canvas

# ------------------------------------------------------------------------------
# 1) Monkey-Patch: Add image_to_url to streamlit.elements.image if missing
# ------------------------------------------------------------------------------
import streamlit.elements.image as st_image
if not hasattr(st_image, "image_to_url"):
    def image_to_url_monkey_patch(img_obj: Any, *args, **kwargs) -> str:
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
    logging.info("Monkey-patched streamlit.elements.image.image_to_url")

# ------------------------------------------------------------------------------
# 2) Import Custom Utilities (Placeholder - Assume these exist)
# ------------------------------------------------------------------------------
try:
    from dicom_utils import parse_dicom, extract_dicom_metadata, dicom_to_image, get_default_wl
    from llm_interactions import (
        run_initial_analysis, run_multimodal_qa, run_disease_analysis,
        estimate_ai_confidence
    )
    from report_utils import generate_pdf_report_bytes
    try:
        from hf_models import query_hf_vqa_inference_api, HF_VQA_MODEL_ID
    except ImportError:
        HF_VQA_MODEL_ID = "mock_hf_model (fallback)"
        def query_hf_vqa_inference_api(img: Image.Image, question: str, roi: Optional[Dict] = None) -> Tuple[str, bool]:
            logging.warning("Using MOCK Hugging Face VQA fallback.")
            return f"Mock HF Response to: '{question}' {'(ROI provided)' if roi else ''}", True
except ImportError as e:
    st.error(f"Critical Error: Failed to import utility modules: {e}. App functionality will be limited. Please ensure dicom_utils.py, llm_interactions.py, report_utils.py exist.")
    # Provide dummy functions so the app doesn't crash immediately during development
    def parse_dicom(b): logging.error("Using mock parse_dicom"); return None
    def extract_dicom_metadata(d): logging.error("Using mock extract_dicom_metadata"); return {"Error": "dicom_utils not found"}
    def dicom_to_image(d, wc, ww, normalize=False): logging.error("Using mock dicom_to_image"); return Image.new("RGB", (100, 100), "grey")
    def get_default_wl(d): logging.error("Using mock get_default_wl"); return 1024, 2048
    def run_initial_analysis(img): logging.error("Using mock run_initial_analysis"); return "Error: llm_interactions not found."
    def run_multimodal_qa(img, q, h, roi): logging.error("Using mock run_multimodal_qa"); return "Error: llm_interactions not found.", False
    def run_disease_analysis(img, d, roi): logging.error("Using mock run_disease_analysis"); return "Error: llm_interactions not found."
    def estimate_ai_confidence(h, img, roi): logging.error("Using mock estimate_ai_confidence"); return "Error: llm_interactions not found."
    def generate_pdf_report_bytes(sid, img, data): logging.error("Using mock generate_pdf_report_bytes"); return None
    HF_VQA_MODEL_ID = "unavailable (mock)"
    def query_hf_vqa_inference_api(img, q, roi): logging.error("Using mock query_hf_vqa_inference_api"); return "HF Fallback Unavailable.", False
    logging.error("Missing local utility modules. Using mock functions, functionality limited.")

# --- Helper Functions ---
def image_to_data_url(img: Image.Image) -> str:
    """Convert a PIL Image to a base64 encoded data URL (PNG format)."""
    buffered = io.BytesIO()
    try:
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Failed to convert image to data URL: {e}")
        return "" # Return empty string on failure

# ------------------------------------------------------------------------------
# 3) Setup Logging
# ------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, # Use INFO or DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ------------------------------------------------------------------------------
# 4) Configure Streamlit Page
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="RadVision AI Advanced",
    layout="centered",
    page_icon="⚕️",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# 5) Display Logo in Sidebar (Optional)
# ------------------------------------------------------------------------------
# with st.sidebar:
#     try:
#         st.image("assets/radvisionai-logo.png", width=200)
#     except Exception as e:
#         logger.warning(f"Logo image not found or failed to load: {e}")

# ------------------------------------------------------------------------------
# 6) Initialize Session State
# ------------------------------------------------------------------------------
DEFAULT_STATE = {
    "uploaded_file_info": None, "raw_image_bytes": None, "is_dicom": False,
    "dicom_dataset": None, "dicom_metadata": {}, "dicom_wc": None, "dicom_ww": None,
    "processed_image": None, "display_image": None, "session_id": None,
    "history": [], "initial_analysis": "", "qa_answer": "", "disease_analysis": "",
    "confidence_score": "", "last_action": None, "pdf_report_bytes": None,
    "canvas_drawing": None, "roi_coords": None, "slider_wc": None, "slider_ww": None,
}
for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# ------------------------------------------------------------------------------
# 7) Page Title & Disclaimer
# ------------------------------------------------------------------------------
st.title("⚕️ RadVision QA Advanced: AI")
with st.expander("⚠️ Important Disclaimer", expanded=False):
    st.warning(
        """
        **Disclaimer:** This tool utilizes Artificial Intelligence for medical image analysis
        and is intended for informational and research purposes only. [...]
        By using this tool, you acknowledge and agree to these terms.
        """ # Truncated for brevity
    )
st.markdown("---")

# =============================================================================
# === SIDEBAR CONTROLS ========================================================
# =============================================================================
with st.sidebar:
    st.header("Image & Controls")

    # --- 1) Upload Image ---
    uploaded_file = st.file_uploader(
        "Upload Image (JPG, PNG, DCM)",
        type=["jpg", "jpeg", "png", "dcm", "dicom"],
        key="file_uploader",
        help="Select a standard image format or a DICOM (.dcm) file. Max 500MB."
    )

    # --- Process Uploaded File ---
    if uploaded_file is not None:
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{uploaded_file.type}"
        if new_file_info != st.session_state.uploaded_file_info:
            logger.info(f"New file upload detected: {uploaded_file.name} ({uploaded_file.type}, {uploaded_file.size} bytes)")
            # Reset state
            for key, default_value in DEFAULT_STATE.items():
                if key not in ["file_uploader"]: st.session_state[key] = default_value
            logger.debug("Session state reset for new file.")
            st.session_state.uploaded_file_info = new_file_info
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.raw_image_bytes = uploaded_file.getvalue()
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            st.session_state.is_dicom = file_ext in (".dcm", ".dicom") or "dicom" in uploaded_file.type.lower()
            logger.info(f"File identified as DICOM: {st.session_state.is_dicom}")

            with st.spinner("🔬 Processing image... Please wait."):
                processing_successful = False
                temp_display_image = None
                temp_processed_image = None

                # --- DICOM Processing ---
                if st.session_state.is_dicom:
                    try:
                        logger.debug("Attempting to parse DICOM...")
                        st.session_state.dicom_dataset = parse_dicom(st.session_state.raw_image_bytes)
                        if st.session_state.dicom_dataset:
                            logger.info("DICOM parsed successfully.")
                            ds = st.session_state.dicom_dataset
                            st.session_state.dicom_metadata = extract_dicom_metadata(ds)
                            wc, ww = get_default_wl(ds)
                            st.session_state.dicom_wc, st.session_state.dicom_ww = wc, ww
                            logger.info(f"Default W/L from DICOM: WC={wc}, WW={ww}")

                            logger.debug("Attempting to generate display image from DICOM...")
                            temp_display_image = dicom_to_image(ds, wc, ww)
                            logger.debug("Attempting to generate processed image from DICOM...")
                            temp_processed_image = dicom_to_image(ds, window_center=None, window_width=None, normalize=True)

                            if temp_display_image and temp_processed_image:
                                logger.info("DICOM images (display & processed) generated.")
                                processing_successful = True
                                pixel_min, pixel_max = 0, 4095
                                try:
                                    arr = ds.pixel_array; pixel_min = float(arr.min()); pixel_max = float(arr.max())
                                    logger.info(f"DICOM pixel range: {pixel_min} to {pixel_max}")
                                except Exception as e: logger.warning(f"Could not get DICOM pixel range: {e}")
                                default_wc = (pixel_max + pixel_min) / 2
                                default_ww = (pixel_max - pixel_min) if pixel_max > pixel_min else 1024
                                st.session_state.slider_wc = wc if wc is not None else default_wc
                                st.session_state.slider_ww = ww if (ww is not None and ww > 0) else default_ww
                                logger.info(f"Slider initial values set: WC={st.session_state.slider_wc}, WW={st.session_state.slider_ww}")
                            else:
                                st.error("DICOM processing failed: Could not generate image objects.")
                                logger.error("dicom_to_image returned None for display or processed image.")
                        else:
                            st.error("Failed to parse the DICOM file. It might be corrupted or not a valid DICOM.")
                            logger.error("parse_dicom returned None.")
                    except Exception as e:
                        st.error(f"Error during DICOM processing: {e}")
                        logger.error(f"DICOM processing pipeline error: {e}", exc_info=True)
                # --- Non-DICOM Image Processing ---
                else:
                    try:
                        logger.debug("Attempting to open standard image...")
                        img = Image.open(io.BytesIO(st.session_state.raw_image_bytes))
                        temp_processed_image = img.copy()
                        logger.debug(f"Original image mode: {img.mode}. Converting to RGB for display.")
                        temp_display_image = img.convert("RGB")
                        st.session_state.dicom_dataset = None
                        st.session_state.dicom_metadata = {}
                        processing_successful = True
                        logger.info("Standard image opened and converted for display.")
                    except UnidentifiedImageError:
                        st.error("Cannot identify image file format. Please upload JPG, PNG, or DICOM.")
                        logger.error("UnidentifiedImageError during standard image processing.")
                    except Exception as e:
                        st.error(f"Error processing image file: {e}")
                        logger.error(f"Standard image processing error: {e}", exc_info=True)

                # --- Final Check and State Update ---
                if processing_successful and isinstance(temp_display_image, Image.Image) and isinstance(temp_processed_image, Image.Image):
                    if temp_display_image.mode != 'RGB':
                        logger.warning(f"Final check: Converting display image from {temp_display_image.mode} to RGB.")
                        st.session_state.display_image = temp_display_image.convert('RGB')
                    else:
                        st.session_state.display_image = temp_display_image
                    st.session_state.processed_image = temp_processed_image
                    logger.info(f"Processing complete. display_image: {st.session_state.display_image.mode} {st.session_state.display_image.size}, processed_image: {st.session_state.processed_image.mode} {st.session_state.processed_image.size}")
                    st.success("✅ Image loaded successfully!")
                    st.session_state.roi_coords = None
                    st.session_state.canvas_drawing = None
                    st.rerun()
                else:
                    st.error("❌ Image loading failed after processing. Please check logs or try a different file.")
                    logger.error(f"Processing marked successful={processing_successful} but image objects invalid: display={type(temp_display_image)}, processed={type(temp_processed_image)}")
                    st.session_state.uploaded_file_info = None
                    st.session_state.raw_image_bytes = None
                    st.session_state.display_image = None
                    st.session_state.processed_image = None
            # End of `with st.spinner`
        # End of `if new_file_info != ...`
    # End of `if uploaded_file is not None`

    st.markdown("---")

    # --- 2) DICOM Window/Level Controls (Conditional) ---
    if st.session_state.display_image and st.session_state.is_dicom and st.session_state.dicom_dataset:
        with st.expander("DICOM Window/Level", expanded=False):
            ds = st.session_state.dicom_dataset
            pixel_min, pixel_max = 0, 4095
            try: arr = ds.pixel_array; pixel_min = float(arr.min()); pixel_max = float(arr.max())
            except Exception: pass
            min_level=-1024; max_level=4095; max_width=8192; min_width=1.0 # Simplified ranges
            try: # More robust range calculation
                 if pixel_max > pixel_min:
                     px_range = pixel_max - pixel_min
                     min_level = pixel_min - px_range
                     max_level = pixel_max + px_range
                     max_width = px_range * 2
                 else: min_level = pixel_min - 512; max_level = pixel_max + 512; max_width = 1024
            except: pass
            max_width = max(max_width, 1)

            current_wc = st.session_state.get("slider_wc", (pixel_max + pixel_min) / 2)
            current_ww = st.session_state.get("slider_ww", (pixel_max - pixel_min) if pixel_max > pixel_min else 1024)
            current_ww = max(current_ww, min_width)

            new_wc = st.slider("Window Center (Level)", min_value=min_level, max_value=max_level, value=current_wc, step=1.0, key="wc_slider")
            new_ww = st.slider("Window Width", min_value=min_width, max_value=max_width, value=current_ww, step=1.0, key="ww_slider")

            if abs(new_wc - current_wc) > 1e-3 or abs(new_ww - current_ww) > 1e-3:
                st.session_state.slider_wc = new_wc; st.session_state.slider_ww = new_ww
                with st.spinner("Applying Window/Level..."):
                    try:
                        st.session_state.display_image = dicom_to_image(ds, new_wc, new_ww)
                        logger.info(f"Applied W/L: WC={new_wc}, WW={new_ww}")
                        if st.session_state.display_image and st.session_state.display_image.mode != 'RGB':
                             st.session_state.display_image = st.session_state.display_image.convert('RGB'); logger.info("Converted W/L image to RGB.")
                    except Exception as e: st.error(f"Failed to apply W/L: {e}"); logger.error(f"W/L application error: {e}", exc_info=True)
                st.rerun()
            if st.button("Reset W/L", key="reset_wl_btn"):
                with st.spinner("Resetting Window/Level..."):
                    try:
                        wc_reset, ww_reset = get_default_wl(ds)
                        px_min, px_max = 0, 4095
                        try: arr = ds.pixel_array; px_min = float(arr.min()); px_max = float(arr.max())
                        except Exception: pass
                        default_wc_reset = (px_max + px_min) / 2; default_ww_reset = (px_max - px_min) if px_max > px_min else 1024
                        final_wc = wc_reset if wc_reset is not None else default_wc_reset; final_ww = ww_reset if (ww_reset is not None and ww_reset > 0) else default_ww_reset
                        st.session_state.slider_wc = final_wc; st.session_state.slider_ww = final_ww
                        st.session_state.display_image = dicom_to_image(ds, final_wc, final_ww)
                        logger.info(f"Reset W/L to: WC={final_wc}, WW={final_ww}")
                        if st.session_state.display_image and st.session_state.display_image.mode != 'RGB':
                            st.session_state.display_image = st.session_state.display_image.convert('RGB'); logger.info("Converted reset W/L image to RGB.")
                    except Exception as e: st.error(f"Failed to reset W/L: {e}"); logger.error(f"W/L reset error: {e}", exc_info=True)
                st.rerun()
        st.markdown("---")

    # --- 3) Analysis & Interaction Controls (Conditional) ---
    if isinstance(st.session_state.display_image, Image.Image): # Check if image is valid
        if st.button("▶️ Run Initial Analysis", key="analyze_btn", help="Perform a general analysis of the entire image."):
            st.session_state.last_action = "analyze"; st.rerun()
        st.markdown("---")
        st.subheader("❓ Ask AI Question")
        if st.session_state.roi_coords:
            # Display the captured coordinates for debugging ROI issues
            st.info(f"✅ ROI Selected: {st.session_state.roi_coords}. Question will focus here.")
            if st.button("❌ Clear ROI", key="clear_roi_btn", help="Remove the highlighted region"):
                st.session_state.roi_coords = None; st.session_state.canvas_drawing = None
                logger.info("ROI cleared by user."); st.rerun()
        else: st.caption("Optionally, draw a rectangle on the image to focus your question.")
        question_input = st.text_area("Ask about the image or the highlighted region:", height=100, key="question_input", placeholder="e.g., What type of scan is this? Are there any abnormalities in the selected region?")
        if st.button("💬 Ask AI", key="ask_btn"):
            if st.session_state.question_input and st.session_state.question_input.strip(): st.session_state.last_action = "ask"; st.rerun()
            else: st.warning("Please enter a question before asking the AI.")
        st.markdown("---")
        st.subheader("🎯 Focused Condition Analysis")
        disease_options=["", "Pneumonia", "Lung Cancer", "Stroke (Ischemic/Hemorrhagic)", "Bone Fracture", "Appendicitis", "Tuberculosis", "COVID-19 Pneumonitis", "Pulmonary Embolism", "Brain Tumor (e.g., Glioblastoma, Meningioma)", "Arthritis Signs", "Osteoporosis Signs", "Cardiomegaly", "Aortic Aneurysm/Dissection Signs", "Bowel Obstruction Signs"]
        disease_options_sorted=[""] + sorted(list(set(opt for opt in disease_options if opt)))
        disease_select=st.selectbox("Select a specific condition to analyze for:", options=disease_options_sorted, key="disease_select", help="The AI will look for findings relevant to the selected condition.")
        if st.button("🩺 Run Condition Analysis", key="disease_btn"):
            if st.session_state.disease_select: st.session_state.last_action="disease"; st.rerun()
            else: st.warning("Please select a condition from the dropdown first.")
        st.markdown("---")
        with st.expander("📊 Confidence & Report", expanded=True):
            if st.button("📈 Estimate AI Confidence", key="confidence_btn", help="Estimate the AI's confidence based on the conversation history and image."):
                if st.session_state.history or st.session_state.initial_analysis or st.session_state.disease_analysis: st.session_state.last_action = "confidence"; st.rerun()
                else: st.warning("Please perform an analysis or ask a question first to estimate confidence.")
            if st.button("📄 Generate PDF Report Data", key="generate_report_data_btn", help="Compile the session analysis into data for a PDF report."):
                st.session_state.last_action = "generate_report_data"; st.rerun()
            if st.session_state.get("pdf_report_bytes"):
                report_filename = f"RadVisionAI_Report_{st.session_state.session_id}.pdf"
                st.download_button(label="⬇️ Download PDF Report", data=st.session_state.pdf_report_bytes, file_name=report_filename, mime="application/pdf", key="download_pdf_button", help=f"Download the generated report ({report_filename})")
                st.caption("PDF report data generated successfully.")
            elif st.session_state.last_action == "generate_report_data" and not st.session_state.pdf_report_bytes: st.caption("PDF generation failed or is in progress.")
    else: st.info("👈 Please upload an image file (JPG, PNG, or DICOM) to begin analysis.")

# =============================================================================
# === MAIN PANEL DISPLAYS =====================================================
# =============================================================================
col1, col2 = st.columns([2, 3])

# --- Column 1: Image Viewer and Canvas (Cleaned Debugging) ---
with col1:
    st.subheader("🖼️ Image Viewer")

    display_img_object = st.session_state.get("display_image")

    if isinstance(display_img_object, Image.Image):
        # Keep the direct st.image display as a visual confirmation the image object is valid
        try:
            st.image(display_img_object,
                     caption=f"Display Image (Mode: {display_img_object.mode}, Size: {display_img_object.size})",
                     use_column_width='always')
        except Exception as img_display_error:
            st.error(f"Error displaying image directly: {img_display_error}")
            logger.error(f"st.image failed: {img_display_error}", exc_info=True)

        bg_image_pil = display_img_object

        # Ensure RGB for canvas
        if bg_image_pil.mode != 'RGB':
            logger.warning(f"Canvas Prep: Converting image from {bg_image_pil.mode} to RGB.")
            try:
                bg_image_pil = bg_image_pil.convert('RGB')
            except Exception as convert_err:
                st.error(f"Failed to convert image to RGB for canvas: {convert_err}")
                logger.error(f"RGB conversion error: {convert_err}", exc_info=True)
                bg_image_pil = None # Invalidate

        if bg_image_pil:
            MAX_CANVAS_WIDTH = 600; MAX_CANVAS_HEIGHT = 550
            img_w, img_h = bg_image_pil.width, bg_image_pil.height
            aspect_ratio = img_w / img_h if img_h > 0 else 1
            canvas_width = min(img_w, MAX_CANVAS_WIDTH)
            canvas_height = int(canvas_width / aspect_ratio) if aspect_ratio > 0 else MAX_CANVAS_HEIGHT
            if canvas_height > MAX_CANVAS_HEIGHT: canvas_height = MAX_CANVAS_HEIGHT; canvas_width = int(canvas_height * aspect_ratio)
            canvas_width = max(canvas_width, 150); canvas_height = max(canvas_height, 150)
            logger.info(f"Canvas Prep: Calculated dimensions W={canvas_width}, H={canvas_height}")

            if canvas_width > 0 and canvas_height > 0:
                st.caption("Click and drag on the image below to select a Region of Interest (ROI).")
                try:
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.2)", stroke_width=2, stroke_color="rgba(220, 50, 50, 0.8)",
                        background_image=bg_image_pil, update_streamlit=True,
                        height=int(canvas_height), width=int(canvas_width),
                        drawing_mode="rect", key="canvas",
                    )

                    # --- ROI Processing Logic ---
                    # Add more logging here to debug ROI extraction
                    if canvas_result and canvas_result.json_data is not None:
                        logger.debug(f"Canvas Result JSON data: {canvas_result.json_data}") # Log the raw data
                        if canvas_result.json_data.get("objects"):
                            last_object = canvas_result.json_data["objects"][-1]
                            logger.debug(f"Last drawn object: {last_object}") # Log the object details
                            if last_object["type"] == "rect":
                                left = int(last_object["left"])
                                top = int(last_object["top"])
                                width = int(last_object["width"] * last_object.get("scaleX", 1))
                                height = int(last_object["height"] * last_object.get("scaleY", 1))
                                logger.debug(f"Extracted Rect Coords: L={left}, T={top}, W={width}, H={height}")

                                if width > 5 and height > 5: # Add a minimum size threshold
                                    new_roi = {"left": left, "top": top, "width": width, "height": height}
                                    if st.session_state.roi_coords != new_roi:
                                        logger.info(f"Setting new ROI coords: {new_roi}")
                                        st.session_state.roi_coords = new_roi
                                        st.session_state.canvas_drawing = canvas_result.json_data
                                        # Consider removing st.rerun() here if it causes state issues
                                        st.rerun()
                                else:
                                    logger.debug("Ignoring rectangle: Dimensions too small.")
                            else:
                                logger.debug(f"Ignoring last object: Type is '{last_object['type']}', expected 'rect'.")
                        else:
                            logger.debug("Canvas result has no 'objects'.")
                    else:
                         # This might occur on initial load or if drawing fails
                         logger.debug("Canvas result or its JSON data is None.")

                except Exception as canvas_error:
                    st.error(f"Error initializing the drawing canvas: {canvas_error}")
                    logger.error(f"st_canvas failed: {canvas_error}", exc_info=True)
                    st.warning("Please check the Browser's Developer Console (F12) for more details.")
            else:
                st.error("Calculated canvas dimensions are invalid (<= 0). Cannot draw canvas.")
                logger.error(f"Invalid canvas dimensions calculated: W={canvas_width}, H={canvas_height}")
        else:
             st.error("Background image became invalid after preparation for canvas.")
             logger.error("Canvas Prep: bg_image_pil became invalid before canvas call.")

    # --- Fallback if display_image is NOT valid ---
    else:
        logger.error(f"Image Viewer: st.session_state.display_image type is {type(display_img_object)}, expected PIL Image.")
        st.markdown("---")
        if st.session_state.uploaded_file_info: st.warning("Image processing likely failed. Check terminal logs.")
        else: st.info("Image will appear here after uploading.")
        st.markdown("<div style='height: 400px; border: 1px dashed #444; ...'>Image Display Area</div>", unsafe_allow_html=True) # Truncated style

    # --- DICOM Metadata Display ---
    if st.session_state.is_dicom and st.session_state.dicom_metadata:
        with st.expander("DICOM Metadata", expanded=False):
            meta_col1, meta_col2 = st.columns(2)
            meta_items = list(st.session_state.dicom_metadata.items())
            for i, (key, value) in enumerate(meta_items):
                col_target = meta_col1 if i % 2 == 0 else meta_col2
                if isinstance(value, list): display_val = ", ".join(map(str, value))
                elif isinstance(value, pydicom.uid.UID): display_val = f"{value.name} ({value})"
                elif isinstance(value, bytes):
                    try: display_val = value.decode('utf-8', errors='replace').strip()
                    except Exception: display_val = f"[Binary Data ({len(value)} bytes)]"
                else: display_val = str(value).strip()
                if len(display_val) > 100: display_val = display_val[:100] + "..."
                if display_val: col_target.markdown(f"**{key}:** `{display_val}`")

# --- Column 2: Analysis Results Tabs ---
with col2:
    st.subheader("📊 Analysis & Results")
    tab_keys = ["Initial Analysis", "Q&A", "Disease Focus", "Confidence"]
    tabs = st.tabs([f" {key} " for key in tab_keys])
    with tabs[0]: st.text_area("Overall Findings & Impressions", value=st.session_state.initial_analysis or "No initial analysis performed yet.", height=400, key="output_initial", disabled=True, help="General analysis results.")
    with tabs[1]:
        st.text_area("AI Answer", value=st.session_state.qa_answer or "No questions asked yet.", height=250, key="output_qa", disabled=True, help="AI's answer to your most recent question.")
        if st.session_state.history:
            with st.expander("View Full Conversation History", expanded=True):
                for i, (q, a) in enumerate(reversed(st.session_state.history)):
                    st.markdown(f"**You ({len(st.session_state.history)-i}):** {q}")
                    st.markdown(f"**AI ({len(st.session_state.history)-i}):**"); st.markdown(a, unsafe_allow_html=True)
                    if i < len(st.session_state.history) - 1: st.markdown("---")
    with tabs[2]: st.text_area("Disease-Specific Findings", value=st.session_state.disease_analysis or "No focused condition analysis performed yet.", height=400, key="output_disease", disabled=True, help="Analysis for selected condition.")
    with tabs[3]: st.text_area("AI Confidence Estimation", value=st.session_state.confidence_score or "No confidence estimation performed yet.", height=400, key="output_confidence", disabled=True, help="AI's estimated confidence.")

# << All code from the previous version UP TO the ACTION HANDLING section >>
# Imports, Setup, Sidebar, Main Panel col1 and col2 display logic remains the same...

# =============================================================================
# === ACTION HANDLING ===========================================================
# =============================================================================
current_action = st.session_state.get("last_action")
if current_action:
    logger.info(f"Handling action: '{current_action}'")
    if not isinstance(st.session_state.processed_image, Image.Image) or not st.session_state.session_id:
        st.error(f"Cannot perform action '{current_action}': Processed image is not available or session invalid.")
        logger.error(f"Action '{current_action}' aborted: Processed image type {type(st.session_state.processed_image)}, session_id={st.session_state.session_id}")
        st.session_state.last_action = None; st.stop()

    img_for_llm = st.session_state.processed_image
    roi: Optional[Dict[str, int]] = st.session_state.get("roi_coords")
    roi_info_str = " (focused on ROI)" if roi else ""

    try:
        # --- Action Logic ---
        if current_action == "analyze":
            with st.spinner("Performing initial analysis..."): result = run_initial_analysis(img_for_llm)
            st.session_state.initial_analysis=result; st.session_state.qa_answer=""; st.session_state.disease_analysis=""; st.session_state.confidence_score=""
            logger.info("Initial analysis completed.")
        elif current_action == "ask":
            question = st.session_state.question_input.strip(); st.session_state.qa_answer = ""
            with st.spinner(f"AI is thinking{roi_info_str}..."): gemini_answer, success = run_multimodal_qa(img_for_llm, question, st.session_state.history, roi)
            if success:
                st.session_state.qa_answer = gemini_answer; st.session_state.history.append((question, gemini_answer))
                logger.info(f"Multimodal QA successful for question: '{question}'{roi_info_str}")
            else: # Fallback Logic
                st.session_state.qa_answer = f"Primary AI failed: {gemini_answer}"; st.error("Primary AI query failed. Attempting fallback...")
                logger.warning(f"Primary AI failed for question: '{question}'. Reason: {gemini_answer}")
                hf_token_available = bool(os.environ.get("HF_API_TOKEN"))
                if hf_token_available:
                    with st.spinner(f"Attempting Fallback ({HF_VQA_MODEL_ID})..."): hf_answer, hf_success = query_hf_vqa_inference_api(img_for_llm, question, roi)
                    if hf_success: fallback_display = f"**[Fallback ({HF_VQA_MODEL_ID})]**\n\n{hf_answer}"; st.session_state.qa_answer = fallback_display; st.session_state.history.append((question, fallback_display)); st.info("Fallback successful."); logger.info(f"HF VQA fallback successful: '{question}'")
                    else: error_msg = f"\n\n---\n**Fallback Failed:** {hf_answer}"; st.session_state.qa_answer += error_msg; st.error(f"Fallback also failed: {hf_answer}"); logger.error(f"HF VQA fallback failed: '{question}'. Reason: {hf_answer}")
                else: st.session_state.qa_answer += "\n\n---\n**[Fallback Unavailable: HF Token Missing]**"; st.warning("HF token not configured."); logger.warning("HF VQA fallback skipped: Token missing.")
        elif current_action == "disease":
            disease = st.session_state.disease_select
            with st.spinner(f"Analyzing for '{disease}'{roi_info_str}..."): result = run_disease_analysis(img_for_llm, disease, roi)
            st.session_state.disease_analysis = result; st.session_state.qa_answer = ""; st.session_state.confidence_score = ""
            logger.info(f"Disease analysis completed for '{disease}'.")
        elif current_action == "confidence":
            with st.spinner("Estimating AI confidence..."): result = estimate_ai_confidence(st.session_state.history, st.session_state.processed_image, roi)
            st.session_state.confidence_score = result
            logger.info("Confidence estimation completed.")
        elif current_action == "generate_report_data":
            st.session_state.pdf_report_bytes = None
            with st.spinner("📄 Generating PDF report data..."):
                img_for_report = st.session_state.display_image; img_with_roi_for_report = img_for_report
                current_roi = st.session_state.get("roi_coords")
                if isinstance(img_for_report, Image.Image) and current_roi:
                    try:
                        img_copy = img_for_report.copy().convert("RGB"); draw = ImageDraw.Draw(img_copy)
                        x0, y0 = current_roi['left'], current_roi['top']; x1, y1 = x0 + current_roi['width'], y0 + current_roi['height']
                        draw.rectangle([x0, y0, x1, y1], outline="red", width=3); img_with_roi_for_report = img_copy
                        logger.info("Drew ROI onto image for PDF report.")
                    except Exception as e: logger.error(f"Failed to draw ROI on report image: {e}", exc_info=True)

                if isinstance(img_with_roi_for_report, Image.Image):
                    full_qa_history = "\n\n".join([f"User Q: {q}\n\nAI A: {a}" for q, a in st.session_state.history]) if st.session_state.history else "No Q&A history."
                    outputs_for_report = {"Initial Analysis": st.session_state.initial_analysis or "Not performed.", "Conversation History": full_qa_history, "Disease-Specific Analysis": st.session_state.disease_analysis or "Not performed.", "Last Confidence Estimate": st.session_state.confidence_score or "Not estimated."}

                    # --- CORRECTED DICOM Metadata Formatting for PDF ---
                    if st.session_state.is_dicom and st.session_state.dicom_metadata:
                         meta_str_list = []
                         for k, v in st.session_state.dicom_metadata.items():
                            display_v = "" # Initialize display_v
                            if isinstance(v, list):
                                display_v = ", ".join(map(str, v))
                            elif isinstance(v, pydicom.uid.UID):
                                display_v = f"{v.name} ({v})"
                            elif isinstance(v, bytes):
                                # CORRECTED: Put try/except on separate lines
                                try:
                                    display_v = v.decode("utf-8", errors="replace").strip()
                                except Exception:
                                    display_v = f"[Binary Data ({len(v)} bytes)]"
                            else:
                                display_v = str(v).strip()

                            # Check if display_v has content before adding
                            if display_v:
                                meta_str_list.append(f"{k}: {display_v}")

                         # Assign the formatted string or a default message
                         outputs_for_report["DICOM Metadata"] = "\n".join(meta_str_list) if meta_str_list else "No significant metadata found."
                    # --- End of Corrected DICOM Metadata Formatting ---

                    pdf_bytes = generate_pdf_report_bytes(st.session_state.session_id, img_with_roi_for_report, outputs_for_report)
                    if pdf_bytes: st.session_state.pdf_report_bytes = pdf_bytes; st.success("PDF report data generated."); logger.info("PDF report data generation successful.")
                    else: st.error("Failed to generate PDF data."); logger.error("PDF report generation failed (returned None).")
                else: st.error("Cannot generate report: No valid image."); logger.error("PDF generation skipped: No valid image.")
        # --- End of Action Logic ---
    except Exception as e:
        st.error(f"An unexpected error occurred during action '{current_action}': {e}")
        logger.error(f"Unhandled error during action '{current_action}': {e}", exc_info=True)
    finally:
        st.session_state.last_action = None
        logger.debug(f"Action '{current_action}' handling finished, resetting last_action.")
        st.rerun() # Update UI

# ------------------------------------------------------------------------------
# 8) Footer (Optional)
# ------------------------------------------------------------------------------
# st.markdown("---")
# st.caption("RadVision AI Advanced | For Research & Informational Purposes Only")