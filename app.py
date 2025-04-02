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
#    (Useful for some potential internal Streamlit uses or extensions)
# ------------------------------------------------------------------------------
import streamlit.elements.image as st_image
if not hasattr(st_image, "image_to_url"):
    def image_to_url(img_obj: Any, width: int = -1, clamp: bool = False,
                     channels: str = "RGB", output_format: str = "auto",
                     image_id: str = "") -> str:
        """Retrieve the URL for an image object."""
        # Simplified version for basic PIL Image support
        if isinstance(img_obj, Image.Image):
            buffered = io.BytesIO()
            format = "PNG" if output_format.lower() == "png" else "JPEG"
            img_obj.save(buffered, format=format)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/{format.lower()};base64,{img_str}"
        else:
            # Fallback or raise error for unsupported types
            logging.warning(f"image_to_url: Unsupported image type {type(img_obj)}")
            return "" # Or raise an appropriate error
    st_image.image_to_url = image_to_url
    logging.info("Monkey-patched streamlit.elements.image.image_to_url")

# ------------------------------------------------------------------------------
# 2) Import Custom Utilities (Placeholder - Assume these exist)
# ------------------------------------------------------------------------------
# Mock implementations for demonstration if real files aren't available
# Replace these with your actual imports
try:
    from dicom_utils import parse_dicom, extract_dicom_metadata, dicom_to_image, get_default_wl
    from llm_interactions import (
        run_initial_analysis, run_multimodal_qa, run_disease_analysis,
        estimate_ai_confidence # Assume this function exists now
        # get_gemini_api_url # Might not be needed directly in UI
    )
    # from hf_models import query_hf_vqa_inference_api, HF_VQA_MODEL_ID # If using HF fallback
    from report_utils import generate_pdf_report_bytes
    # --- Mock HF Fallback ---
    HF_VQA_MODEL_ID = "mock_hf_model"
    def query_hf_vqa_inference_api(img: Image.Image, question: str, roi: Optional[Dict] = None) -> Tuple[str, bool]:
        logging.warning("Using MOCK Hugging Face VQA fallback.")
        return f"Mock HF Response to: '{question}' {'(ROI provided)' if roi else ''}", True

except ImportError as e:
    st.error(f"Failed to import utility modules: {e}. Please ensure dicom_utils.py, llm_interactions.py, report_utils.py exist.")
    # Provide dummy functions so the app doesn't crash immediately
    def parse_dicom(b): return None
    def extract_dicom_metadata(d): return {"Error": "dicom_utils not found"}
    def dicom_to_image(d, wc, ww): return Image.new("RGB", (100, 100), "grey")
    def get_default_wl(d): return 1024, 2048
    def run_initial_analysis(img): return "Error: llm_interactions not found."
    def run_multimodal_qa(img, q, h, roi): return "Error: llm_interactions not found.", False
    def run_disease_analysis(img, d, roi): return "Error: llm_interactions not found."
    def estimate_ai_confidence(h, img, roi): return "Error: llm_interactions not found."
    def generate_pdf_report_bytes(sid, img, data): return None
    HF_VQA_MODEL_ID = "unavailable"
    def query_hf_vqa_inference_api(img, q, roi): return "HF Fallback Unavailable.", False
    logging.error("Missing local utility modules. Using mock functions.")


# --- Helper Functions ---
def image_to_data_url(img: Image.Image) -> str:
    """Convert a PIL Image to a base64 encoded data URL (PNG format)."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# ------------------------------------------------------------------------------
# 3) Setup Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# 4) Configure Streamlit Page
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="RadVision AI Advanced",
    layout="centered", # Use centered layout for a more contained feel
    page_icon="⚕️",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# 5) Display Logo in Sidebar (Optional)
# ------------------------------------------------------------------------------
# Ensure you have an 'assets' folder with the logo image
# Comment out if you don't have the logo file
# with st.sidebar:
#     try:
#         st.image("assets/radvisionai-logo.png", width=200) # Adjust path/name
#     except Exception as e:
#         logger.warning(f"Logo image not found or failed to load: {e}")
#         # st.sidebar.markdown("## RadVision AI") # Fallback text title

# ------------------------------------------------------------------------------
# 6) Initialize Session State
# ------------------------------------------------------------------------------
DEFAULT_STATE = {
    "uploaded_file_info": None,    # Stores name-size-type string to detect new uploads
    "raw_image_bytes": None,       # Raw bytes of the uploaded file
    "is_dicom": False,             # Flag if the uploaded file is DICOM
    "dicom_dataset": None,         # Parsed pydicom dataset object
    "dicom_metadata": {},          # Extracted key DICOM metadata
    "dicom_wc": None,              # Default or current window center
    "dicom_ww": None,              # Default or current window width
    "processed_image": None,       # PIL Image object (potentially full depth for DICOM) for backend processing
    "display_image": None,         # PIL Image object (potentially window/leveled) for display/canvas
    "session_id": None,            # Unique ID for the session/report
    "history": [],                 # List of (question, answer) tuples for Q&A
    "initial_analysis": "",        # Stores the result of the initial analysis
    "qa_answer": "",               # Stores the latest Q&A answer
    "disease_analysis": "",        # Stores the result of focused disease analysis
    "confidence_score": "",        # Stores the AI confidence estimation result
    "last_action": None,           # Tracks the last button pressed ('analyze', 'ask', etc.)
    "pdf_report_bytes": None,      # Stores the generated PDF report bytes
    "canvas_drawing": None,        # Stores the state of the drawable canvas (JSON)
    "roi_coords": None,            # Stores {'left': x, 'top': y, 'width': w, 'height': h} of the drawn ROI
    "slider_wc": None,             # Stores the current value of the WC slider
    "slider_ww": None,             # Stores the current value of the WW slider
}

# Initialize session state keys if they don't exist
for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# ------------------------------------------------------------------------------
# 7) Page Title & Disclaimer
# ------------------------------------------------------------------------------
st.title("⚕️ RadVision QA Advanced: AI")

# Use an expander for the disclaimer to keep the initial view cleaner
with st.expander("⚠️ Important Disclaimer", expanded=False):
    st.warning(
        """
        **Disclaimer:** This tool utilizes Artificial Intelligence for medical image analysis
        and is intended for informational and research purposes only.

        *   **Not a Medical Device:** This application has not been evaluated or approved by any regulatory authority (e.g., FDA, EMA) for clinical diagnosis or patient management.
        *   **Not a Substitute for Professional Judgment:** The AI-generated outputs must **never** replace the assessment and clinical judgment of a qualified healthcare professional (e.g., radiologist, physician).
        *   **Verification Required:** All findings, analyses, and suggestions generated by this AI **must** be carefully reviewed, verified, and interpreted by a licensed medical expert in the context of the patient's complete clinical information before making any decisions regarding diagnosis or treatment.
        *   **Limitations:** AI models can make mistakes, hallucinate information, or fail to identify subtle findings. The accuracy may vary depending on image quality, pathology, and other factors.
        *   **No Liability:** The creators and providers of this tool assume no liability for any decisions made or actions taken based on the information provided by this application. Use at your own risk and discretion.

        By using this tool, you acknowledge and agree to these terms.
        """
    )

st.markdown("---") # Visual separator

# =============================================================================
# === SIDEBAR CONTROLS ========================================================
# =============================================================================
with st.sidebar:
    st.header("Image & Controls")

    # --- 1) Upload Image ---
    uploaded_file = st.file_uploader(
        "Upload Image (JPG, PNG, DCM)",
        type=["jpg", "jpeg", "png", "dcm", "dicom"],
        key="file_uploader", # Assign a key to help manage state
        help="Select a standard image format or a DICOM (.dcm) file."
    )

    # --- Process Uploaded File ---
    if uploaded_file is not None:
        # Check if it's a NEW file upload, not just a rerun
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{uploaded_file.type}"
        if new_file_info != st.session_state.uploaded_file_info:
            logger.info(f"New file uploaded: {uploaded_file.name} ({uploaded_file.type}, {uploaded_file.size} bytes)")

            # Reset most of the session state for the new image
            for key, default_value in DEFAULT_STATE.items():
                # Keep the uploader state itself, reset others
                if key not in ["file_uploader"]:
                    st.session_state[key] = default_value

            # Store new file info and generate a session ID
            st.session_state.uploaded_file_info = new_file_info
            st.session_state.session_id = str(uuid.uuid4())[:8] # Shorter session ID
            st.session_state.raw_image_bytes = uploaded_file.getvalue()

            # Determine if DICOM based on extension or content type
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            st.session_state.is_dicom = file_ext in (".dcm", ".dicom") or "dicom" in uploaded_file.type.lower()

            # --- Image Processing ---
            with st.spinner("🔬 Processing image..."):
                processing_successful = False
                if st.session_state.is_dicom:
                    try:
                        st.session_state.dicom_dataset = parse_dicom(st.session_state.raw_image_bytes)
                        if st.session_state.dicom_dataset:
                            ds = st.session_state.dicom_dataset
                            st.session_state.dicom_metadata = extract_dicom_metadata(ds)
                            # Get default window/level
                            wc, ww = get_default_wl(ds)
                            st.session_state.dicom_wc, st.session_state.dicom_ww = wc, ww

                            # Generate initial display image using default W/L
                            st.session_state.display_image = dicom_to_image(ds, wc, ww)
                            # Generate 'processed' image (e.g., full range, maybe normalized) for AI
                            st.session_state.processed_image = dicom_to_image(ds, window_center=None, window_width=None, normalize=True) # Example processing

                            # Set initial slider values based on pixel range or defaults
                            pixel_min, pixel_max = 0, 4095 # Default reasonable range
                            try:
                                arr = ds.pixel_array
                                pixel_min = float(arr.min())
                                pixel_max = float(arr.max())
                                logger.info(f"DICOM pixel range: {pixel_min} to {pixel_max}")
                            except Exception as e:
                                logger.warning(f"Could not get DICOM pixel range: {e}")

                            # Sensible defaults if wc/ww are None or invalid
                            default_wc = (pixel_max + pixel_min) / 2
                            default_ww = (pixel_max - pixel_min) if pixel_max > pixel_min else 1024

                            st.session_state.slider_wc = wc if wc is not None else default_wc
                            st.session_state.slider_ww = ww if (ww is not None and ww > 0) else default_ww

                            processing_successful = True
                        else:
                            st.error("Failed to parse the DICOM file.")
                    except Exception as e:
                        st.error(f"Error processing DICOM file: {e}")
                        logger.error(f"DICOM processing error: {e}", exc_info=True)

                # --- Non-DICOM Image Processing ---
                else:
                    try:
                        img = Image.open(io.BytesIO(st.session_state.raw_image_bytes)).convert("RGB")
                        st.session_state.processed_image = img.copy() # Keep original for processing
                        st.session_state.display_image = img # Use same for display initially
                        # Reset DICOM specific state
                        st.session_state.dicom_dataset = None
                        st.session_state.dicom_metadata = {}
                        processing_successful = True
                    except UnidentifiedImageError:
                        st.error("Cannot identify image file format. Please upload JPG, PNG, or DICOM.")
                    except Exception as e:
                        st.error(f"Error processing image file: {e}")
                        logger.error(f"Image processing error: {e}", exc_info=True)

            if processing_successful:
                st.success("Image loaded and processed.")
                # Clear any previous ROI when a new image is loaded
                st.session_state.roi_coords = None
                st.session_state.canvas_drawing = None
                st.rerun() # Rerun to update the main panel with the new image
            else:
                st.error("Image loading failed. Please try a different file.")
                # Clear potentially problematic state
                st.session_state.uploaded_file_info = None
                st.session_state.raw_image_bytes = None

    st.markdown("---")

    # --- 2) DICOM Window/Level Controls (Conditional) ---
    if st.session_state.display_image and st.session_state.is_dicom and st.session_state.dicom_dataset:
        with st.expander("DICOM Window/Level", expanded=False):
            ds = st.session_state.dicom_dataset
            # Define slider ranges based on pixel data or sensible defaults
            pixel_min, pixel_max = 0, 4095
            try:
                arr = ds.pixel_array
                pixel_min = float(arr.min())
                pixel_max = float(arr.max())
            except Exception: pass # Keep defaults if pixel array fails

            # Define wider bounds for sliders to allow exploration
            min_level = pixel_min - (pixel_max - pixel_min) if pixel_max > pixel_min else -1024
            max_level = pixel_max + (pixel_max - pixel_min) if pixel_max > pixel_min else 4095
            max_width = (pixel_max - pixel_min) * 2 if pixel_max > pixel_min else 8192
            min_width = 1.0

            # Get current slider values from session state
            current_wc = st.session_state.get("slider_wc", (pixel_max + pixel_min) / 2)
            current_ww = st.session_state.get("slider_ww", (pixel_max - pixel_min) if pixel_max > pixel_min else 1024)

            # Sliders for WC and WW
            new_wc = st.slider("Window Center (Level)", min_value=min_level, max_value=max_level, value=current_wc, step=1.0, key="wc_slider")
            new_ww = st.slider("Window Width", min_value=min_width, max_value=max_width, value=current_ww, step=1.0, key="ww_slider")

            # Check if slider values changed significantly
            if abs(new_wc - current_wc) > 1e-3 or abs(new_ww - current_ww) > 1e-3:
                st.session_state.slider_wc = new_wc
                st.session_state.slider_ww = new_ww
                with st.spinner("Applying Window/Level..."):
                    try:
                        # Update only the display image with new W/L
                        st.session_state.display_image = dicom_to_image(ds, new_wc, new_ww)
                        logger.info(f"Applied W/L: WC={new_wc}, WW={new_ww}")
                    except Exception as e:
                        st.error(f"Failed to apply W/L: {e}")
                        logger.error(f"W/L application error: {e}", exc_info=True)
                st.rerun() # Rerun to update the canvas background

            # Button to reset W/L to default
            if st.button("Reset W/L", key="reset_wl_btn"):
                with st.spinner("Resetting Window/Level..."):
                    try:
                        wc_reset, ww_reset = get_default_wl(ds)
                        # Recalculate defaults if needed
                        px_min, px_max = 0, 4095
                        try:
                            arr = ds.pixel_array; px_min = float(arr.min()); px_max = float(arr.max())
                        except Exception: pass
                        default_wc_reset = (px_max + px_min) / 2
                        default_ww_reset = (px_max - px_min) if px_max > px_min else 1024

                        final_wc = wc_reset if wc_reset is not None else default_wc_reset
                        final_ww = ww_reset if (ww_reset is not None and ww_reset > 0) else default_ww_reset

                        st.session_state.slider_wc = final_wc
                        st.session_state.slider_ww = final_ww
                        st.session_state.display_image = dicom_to_image(ds, final_wc, final_ww)
                        logger.info(f"Reset W/L to: WC={final_wc}, WW={final_ww}")
                    except Exception as e:
                        st.error(f"Failed to reset W/L: {e}")
                        logger.error(f"W/L reset error: {e}", exc_info=True)
                st.rerun()

    st.markdown("---")

    # --- 3) Analysis & Interaction Controls (Conditional) ---
    if st.session_state.display_image: # Only show if an image is loaded
        # Initial Analysis Button
        if st.button("▶️ Run Initial Analysis", key="analyze_btn", help="Perform a general analysis of the entire image."):
            st.session_state.last_action = "analyze"
            st.rerun() # Trigger action handling

        st.markdown("---")

        # Ask Another Question Section
        st.subheader("❓ Ask AI Question")
        # Provide feedback about ROI selection status
        if st.session_state.roi_coords:
            st.info("✅ Region of Interest (ROI) is selected. Your question will focus on this area.")
            if st.button("❌ Clear ROI", key="clear_roi_btn", help="Remove the highlighted region"):
                st.session_state.roi_coords = None
                st.session_state.canvas_drawing = None # Clear canvas state too
                logger.info("ROI cleared by user.")
                st.rerun()
        else:
            st.caption("Optionally, draw a rectangle on the image to focus your question on a specific Region of Interest (ROI).")

        # Question input area
        question_input = st.text_area(
            "Ask about the image or the highlighted region:",
            height=100,
            key="question_input",
            placeholder="e.g., What type of scan is this? Are there any abnormalities in the selected region?"
        )

        # Ask AI button
        if st.button("💬 Ask AI", key="ask_btn"):
            if st.session_state.question_input and st.session_state.question_input.strip():
                st.session_state.last_action = "ask"
                st.rerun() # Trigger action handling
            else:
                st.warning("Please enter a question before asking the AI.")

        st.markdown("---")

        # Focused Condition Analysis Section
        st.subheader("🎯 Focused Condition Analysis")
        disease_options = [
            "", "Pneumonia", "Lung Cancer", "Stroke (Ischemic/Hemorrhagic)",
            "Bone Fracture", "Appendicitis", "Tuberculosis", "COVID-19 Pneumonitis",
            "Pulmonary Embolism", "Brain Tumor (e.g., Glioblastoma, Meningioma)",
            "Arthritis Signs", "Osteoporosis Signs", "Cardiomegaly",
            "Aortic Aneurysm/Dissection Signs", "Bowel Obstruction Signs"
        ]
        # Sort options, keeping "" at the start
        disease_options_sorted = [""] + sorted(list(set(opt for opt in disease_options if opt)))

        disease_select = st.selectbox(
            "Select a specific condition to analyze for:",
            options=disease_options_sorted,
            key="disease_select",
            help="The AI will look for findings relevant to the selected condition."
        )

        # Run Condition Analysis button
        if st.button("🩺 Run Condition Analysis", key="disease_btn"):
            if st.session_state.disease_select:
                st.session_state.last_action = "disease"
                st.rerun() # Trigger action handling
            else:
                st.warning("Please select a condition from the dropdown first.")

        st.markdown("---")

        # Confidence & PDF Section
        with st.expander("📊 Confidence & Report", expanded=True): # Keep expanded for visibility
            # Estimate Confidence button
            if st.button("📈 Estimate AI Confidence", key="confidence_btn", help="Estimate the AI's confidence based on the conversation history and image."):
                if st.session_state.history or st.session_state.initial_analysis or st.session_state.disease_analysis:
                    st.session_state.last_action = "confidence"
                    st.rerun() # Trigger action handling
                else:
                    st.warning("Please perform an analysis or ask a question first to estimate confidence.")

            # Generate PDF Report Data button
            if st.button("📄 Generate PDF Report Data", key="generate_report_data_btn", help="Compile the session analysis into data for a PDF report."):
                st.session_state.last_action = "generate_report_data"
                st.rerun() # Trigger action handling

            # Download PDF button (conditional)
            if st.session_state.get("pdf_report_bytes"):
                report_filename = f"RadVisionAI_Report_{st.session_state.session_id}.pdf"
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=st.session_state.pdf_report_bytes,
                    file_name=report_filename,
                    mime="application/pdf",
                    key="download_pdf_button",
                    help=f"Download the generated report ({report_filename})"
                )
                # Add a small note after generation
                st.caption("PDF report data generated successfully.")
            elif st.session_state.last_action == "generate_report_data": # If button was clicked but failed
                 if not st.session_state.pdf_report_bytes:
                    st.caption("PDF generation failed or is in progress.")


    else: # If no image is loaded yet
        st.info("👈 Please upload an image file (JPG, PNG, or DICOM) to begin.")

# =============================================================================
# === MAIN PANEL DISPLAYS =====================================================
# =============================================================================
# Use columns for layout: Image viewer on left, results on right
col1, col2 = st.columns([2, 3]) # Give slightly more space to results

# --- Column 1: Image Viewer and Canvas ---
with col1:
    st.subheader("🖼️ Image Viewer")

    if st.session_state.display_image:
        bg_image_pil = st.session_state.display_image

        # --- Dynamic Canvas Size Calculation ---
        MAX_CANVAS_WIDTH = 600  # Max width in pixels for the canvas
        MAX_CANVAS_HEIGHT = 550 # Max height in pixels
        img_w, img_h = bg_image_pil.width, bg_image_pil.height

        aspect_ratio = img_w / img_h if img_h > 0 else 1

        # Calculate width first, then height based on aspect ratio
        canvas_width = min(img_w, MAX_CANVAS_WIDTH)
        canvas_height = int(canvas_width / aspect_ratio) if aspect_ratio > 0 else MAX_CANVAS_HEIGHT

        # If calculated height exceeds max height, recalculate width based on max height
        if canvas_height > MAX_CANVAS_HEIGHT:
            canvas_height = MAX_CANVAS_HEIGHT
            canvas_width = int(canvas_height * aspect_ratio)

        # Ensure minimum dimensions
        canvas_width = max(canvas_width, 100)
        canvas_height = max(canvas_height, 100)

        # --- Display Drawable Canvas ---
        st.caption("Click and drag on the image below to select a Region of Interest (ROI).")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.2)",  # Semi-transparent orange fill
            stroke_width=2,                      # Border width of the drawing tool
            stroke_color="rgba(255, 99, 71, 0.8)", # Tomato color border
            background_image=bg_image_pil,       # The PIL image to draw on
            update_streamlit=True,               # Send results back to Streamlit when drawing changes
            height=int(canvas_height),
            width=int(canvas_width),
            drawing_mode="rect",                 # Allow drawing rectangles only
            key="canvas",                        # Unique key for the component
            # Try to restore previous drawing if needed (experimental)
            # initial_drawing=st.session_state.get("canvas_drawing", None),
        )

        # --- Process Drawn ROI ---
        if canvas_result.json_data is not None and canvas_result.json_data.get("objects"):
            # Check if the last object is a rectangle (robustness)
            last_object = canvas_result.json_data["objects"][-1]
            if last_object["type"] == "rect":
                # Extract coordinates, accounting for potential scaling applied by canvas
                left = int(last_object["left"])
                top = int(last_object["top"])
                # Width/Height might be scaled if canvas size != background image size internally
                # Multiply by scaleX/scaleY to get pixel dimensions relative to the background image
                width = int(last_object["width"] * last_object.get("scaleX", 1))
                height = int(last_object["height"] * last_object.get("scaleY", 1))

                # Basic validation: ensure width and height are positive
                if width > 0 and height > 0:
                    new_roi = {
                        "left": left, "top": top, "width": width, "height": height,
                    }

                    # Only update state and rerun if the ROI is actually different
                    # Comparing dictionaries directly works for simple structures
                    if st.session_state.roi_coords != new_roi:
                        logger.info(f"ROI selected/updated: {new_roi}")
                        st.session_state.roi_coords = new_roi
                        st.session_state.canvas_drawing = canvas_result.json_data # Store state
                        st.rerun() # Rerun to update sidebar info/button
                # else: ignore clicks without dragging (width/height 0)

        # --- Display DICOM Metadata (Conditional) ---
        if st.session_state.is_dicom and st.session_state.dicom_metadata:
            with st.expander("DICOM Metadata", expanded=False):
                # Use columns within expander for better layout
                meta_col1, meta_col2 = st.columns(2)
                meta_items = list(st.session_state.dicom_metadata.items())
                # Distribute items between columns
                for i, (key, value) in enumerate(meta_items):
                    col_target = meta_col1 if i % 2 == 0 else meta_col2
                    # Format value for display
                    if isinstance(value, list): display_val = ", ".join(map(str, value))
                    elif isinstance(value, pydicom.uid.UID): display_val = f"{value.name} ({value})"
                    elif isinstance(value, bytes):
                        try: display_val = value.decode('utf-8', errors='replace').strip()
                        except Exception: display_val = f"[Binary Data ({len(value)} bytes)]"
                    else: display_val = str(value).strip()

                    # Truncate long values for display
                    if len(display_val) > 100: display_val = display_val[:100] + "..."

                    if display_val: # Only show if value is not empty after stripping
                        col_target.markdown(f"**{key}:** `{display_val}`")

    else: # If no image is loaded
        st.markdown("---")
        st.info("Image will be displayed here after uploading.")
        # Placeholder visual
        st.markdown("<div style='height: 400px; border: 1px dashed grey; display: flex; align-items: center; justify-content: center; color: grey;'>Image Area</div>", unsafe_allow_html=True)


# --- Column 2: Analysis Results Tabs ---
with col2:
    st.subheader("📊 Analysis & Results")

    # Determine which tab should be active based on the last action or available results
    active_tab_key = "Initial Analysis" # Default
    if st.session_state.qa_answer: active_tab_key = "Q&A"
    elif st.session_state.disease_analysis: active_tab_key = "Disease Focus"
    elif st.session_state.confidence_score: active_tab_key = "Confidence"
    elif st.session_state.initial_analysis: active_tab_key = "Initial Analysis"

    # Define tabs
    tab_keys = ["Initial Analysis", "Q&A", "Disease Focus", "Confidence"]
    tabs = st.tabs([f" {key} " for key in tab_keys]) # Add spacing for visual separation

    # --- Tab 1: Initial Analysis ---
    with tabs[0]:
        st.text_area(
            "Overall Findings & Impressions",
            value=st.session_state.initial_analysis or "No initial analysis performed yet.",
            height=400, # Consistent height
            key="output_initial",
            disabled=True, # Read-only
            help="General analysis results for the entire image."
        )

    # --- Tab 2: Q&A ---
    with tabs[1]:
        st.text_area(
            "AI Answer",
            value=st.session_state.qa_answer or "No questions asked yet.",
            height=250, # Slightly smaller to make space for history
            key="output_qa",
            disabled=True,
            help="The AI's answer to your most recent question."
        )
        # Expander for Conversation History
        if st.session_state.history:
            with st.expander("View Full Conversation History", expanded=True):
                # Display history, latest first
                for i, (q, a) in enumerate(reversed(st.session_state.history)):
                    st.markdown(f"**You ({len(st.session_state.history)-i}):** {q}")
                    st.markdown(f"**AI ({len(st.session_state.history)-i}):**")
                    st.markdown(a, unsafe_allow_html=True) # Render potential markdown in answer
                    if i < len(st.session_state.history) - 1:
                        st.markdown("---") # Separator between Q&A pairs

    # --- Tab 3: Disease Focus ---
    with tabs[2]:
        st.text_area(
            "Disease-Specific Findings",
            value=st.session_state.disease_analysis or "No focused condition analysis performed yet.",
            height=400, # Consistent height
            key="output_disease",
            disabled=True,
            help="Analysis results specifically looking for the selected condition."
        )

    # --- Tab 4: Confidence ---
    with tabs[3]:
        st.text_area(
            "AI Confidence Estimation",
            value=st.session_state.confidence_score or "No confidence estimation performed yet.",
            height=400, # Consistent height
            key="output_confidence",
            disabled=True,
            help="The AI's estimated confidence in its analysis/answers, along with justification."
        )

# =============================================================================
# === ACTION HANDLING ===========================================================
# =============================================================================
# Check if an action was triggered in the last run
current_action = st.session_state.get("last_action")

if current_action:
    logger.info(f"Handling action: {current_action}")
    # Ensure an image is loaded before proceeding
    if not st.session_state.processed_image or not st.session_state.session_id:
        st.error("Cannot perform action: No image loaded or session not initialized.")
        st.session_state.last_action = None # Reset action
        st.stop() # Stop execution if no image

    # Get image and ROI for backend calls
    img_for_llm = st.session_state.processed_image
    roi: Optional[Dict[str, int]] = st.session_state.get("roi_coords")
    roi_info_str = " (focused on ROI)" if roi else "" # For user feedback in spinners

    try:
        # --- Handle 'analyze' action ---
        if current_action == "analyze":
            with st.spinner("Performing initial analysis..."):
                result = run_initial_analysis(img_for_llm)
                st.session_state.initial_analysis = result
                # Clear other potentially stale results
                st.session_state.qa_answer = ""
                st.session_state.disease_analysis = ""
                st.session_state.confidence_score = ""
            logger.info("Initial analysis completed.")

        # --- Handle 'ask' action ---
        elif current_action == "ask":
            question = st.session_state.question_input.strip()
            st.session_state.qa_answer = "" # Clear previous answer display
            with st.spinner(f"AI is thinking{roi_info_str}..."):
                # Call the primary multimodal QA function (e.g., Gemini)
                # Pass the current history and ROI
                gemini_answer, success = run_multimodal_qa(
                    img_for_llm, question, st.session_state.history, roi
                )

            if success:
                st.session_state.qa_answer = gemini_answer
                st.session_state.history.append((question, gemini_answer))
                logger.info(f"Multimodal QA successful for question: '{question}'{roi_info_str}")
                # Clear the input box after successful ask by resetting its key
                # st.session_state.question_input = "" # This might clear too early sometimes
                # A more robust way might involve forms or JS, but this often works:
                if "question_input" in st.session_state:
                     # Trigger a re-render which might pick up an empty default value
                     # Note: Direct reset might not always work as expected with st.rerun
                     pass # Often gets cleared implicitly or user just types over it
            else:
                # --- Handle Failure & Fallback (Optional: Hugging Face VQA) ---
                st.session_state.qa_answer = f"Primary AI failed: {gemini_answer}"
                st.error("Primary AI query failed. Attempting fallback...")
                logger.warning(f"Primary AI failed for question: '{question}'. Reason: {gemini_answer}")

                # Check if HF Token is available (replace with your actual check)
                hf_token_available = bool(os.environ.get("HF_API_TOKEN"))

                if hf_token_available:
                    with st.spinner(f"Attempting Hugging Face VQA Fallback ({HF_VQA_MODEL_ID})..."):
                        hf_answer, hf_success = query_hf_vqa_inference_api(img_for_llm, question, roi)
                        if hf_success:
                            fallback_display = (
                                f"**[Fallback Answer from Hugging Face VQA ({HF_VQA_MODEL_ID})]**\n\n{hf_answer}"
                            )
                            st.session_state.qa_answer = fallback_display # Overwrite previous failure message
                            st.session_state.history.append((question, fallback_display)) # Add fallback to history
                            st.info("Hugging Face VQA fallback successful.")
                            logger.info(f"HF VQA fallback successful for question: '{question}'")
                        else:
                            error_msg = f"\n\n---\n**Hugging Face Fallback Failed:** {hf_answer}"
                            st.session_state.qa_answer += error_msg
                            st.error(f"Hugging Face VQA fallback also failed: {hf_answer}")
                            logger.error(f"HF VQA fallback failed for question: '{question}'. Reason: {hf_answer}")
                else:
                    st.session_state.qa_answer += "\n\n---\n**[Fallback Unavailable: Hugging Face Token Missing]**"
                    st.warning("Hugging Face API token not configured. Cannot use VQA fallback.")
                    logger.warning("HF VQA fallback skipped: Token missing.")

        # --- Handle 'disease' action ---
        elif current_action == "disease":
            disease = st.session_state.disease_select
            with st.spinner(f"Analyzing for '{disease}'{roi_info_str}..."):
                result = run_disease_analysis(img_for_llm, disease, roi)
                st.session_state.disease_analysis = result
                # Clear other results
                st.session_state.qa_answer = ""
                st.session_state.confidence_score = ""
            logger.info(f"Disease analysis completed for '{disease}'.")

        # --- Handle 'confidence' action ---
        elif current_action == "confidence":
            with st.spinner("Estimating AI confidence..."):
                # Pass history, image, and ROI to the confidence estimation function
                result = estimate_ai_confidence(
                    st.session_state.history,
                    st.session_state.processed_image, # Or display_image? Decide based on function needs
                    roi
                )
                st.session_state.confidence_score = result
                # Clear other potentially stale results? Maybe not needed here.
            logger.info("Confidence estimation completed.")

        # --- Handle 'generate_report_data' action ---
        elif current_action == "generate_report_data":
            st.session_state.pdf_report_bytes = None # Clear previous report bytes
            with st.spinner("📄 Generating PDF report data..."):
                img_for_report = st.session_state.display_image
                img_with_roi_for_report = img_for_report # Default

                # --- Draw ROI on image for report ---
                current_roi = st.session_state.get("roi_coords")
                if img_for_report and current_roi:
                    try:
                        # Create a copy to draw on
                        img_copy = img_for_report.copy().convert("RGB") # Ensure RGB for drawing
                        draw = ImageDraw.Draw(img_copy)
                        x0, y0 = current_roi['left'], current_roi['top']
                        x1, y1 = x0 + current_roi['width'], y0 + current_roi['height']
                        # Draw rectangle (adjust color/width as needed)
                        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
                        img_with_roi_for_report = img_copy
                        logger.info("Drew ROI onto image for PDF report.")
                    except Exception as e:
                        logger.error(f"Failed to draw ROI on report image: {e}", exc_info=True)
                        # Fallback to using the original image without ROI drawn

                # --- Compile data for the report ---
                if img_with_roi_for_report:
                    full_qa_history = "\n\n".join(
                        [f"User Q: {q}\n\nAI A: {a}" for q, a in st.session_state.history]
                    ) if st.session_state.history else "No Q&A history for this session."

                    outputs_for_report = {
                        "Initial Analysis": st.session_state.initial_analysis or "Not performed.",
                        "Conversation History": full_qa_history,
                        "Disease-Specific Analysis": st.session_state.disease_analysis or "Not performed.",
                        "Last Confidence Estimate": st.session_state.confidence_score or "Not estimated.",
                    }
                    # Add DICOM metadata if available
                    if st.session_state.is_dicom and st.session_state.dicom_metadata:
                         meta_str_list = []
                         for k, v in st.session_state.dicom_metadata.items():
                            if isinstance(v, list): display_v = ", ".join(map(str, v))
                            elif isinstance(v, pydicom.uid.UID): display_v = f"{v.name} ({v})"
                            elif isinstance(v, bytes):
                                try: display_v = v.decode("utf-8", errors="replace").strip()
                                except Exception: display_v = f"[Binary Data ({len(v)} bytes)]"
                            else: display_v = str(v).strip()
                            if display_v: meta_str_list.append(f"{k}: {display_v}")
                         outputs_for_report["DICOM Metadata"] = "\n".join(meta_str_list) if meta_str_list else "No significant metadata found."

                    # --- Generate PDF Bytes ---
                    pdf_bytes = generate_pdf_report_bytes(
                        st.session_state.session_id,
                        img_with_roi_for_report, # Use image with potential ROI drawn
                        outputs_for_report
                    )

                    if pdf_bytes:
                        st.session_state.pdf_report_bytes = pdf_bytes
                        st.success("PDF report data generated successfully. Download button available.")
                        logger.info("PDF report data generation successful.")
                    else:
                        st.error("Failed to generate PDF report data.")
                        logger.error("PDF report data generation failed (returned None).")
                else:
                     st.error("Cannot generate report: No image available.")
                     logger.error("PDF report generation skipped: No image available.")

    except Exception as e:
        # General error catching for actions
        st.error(f"An error occurred during the '{current_action}' action: {e}")
        logger.error(f"Error during action '{current_action}': {e}", exc_info=True)

    finally:
        # --- Reset Action and Rerun ---
        # Crucial: Reset the last_action flag AFTER handling it
        st.session_state.last_action = None
        # Rerun to update the UI reflecting the results of the action
        st.rerun()

# ------------------------------------------------------------------------------
# 8) Footer (Optional)
# ------------------------------------------------------------------------------
# st.markdown("---")
# st.caption("RadVision AI Advanced | For Research & Informational Purposes Only")