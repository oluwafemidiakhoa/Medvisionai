import io
import os
import uuid
import logging
import base64
from typing import Any

import streamlit as st
from PIL import Image, UnidentifiedImageError
import pydicom
from streamlit_drawable_canvas import st_canvas

# ------------------------------------------------------------------------------
# Monkey-Patch: Add st.image.image_to_url if missing (needed by st_canvas)
# ------------------------------------------------------------------------------
if not hasattr(st.image, "image_to_url"):
    def image_to_url(img: Image.Image) -> str:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"
    st.image.image_to_url = image_to_url

# ------------------------------------------------------------------------------
# Import Custom Utilities
# ------------------------------------------------------------------------------
from dicom_utils import parse_dicom, extract_dicom_metadata, dicom_to_image, get_default_wl
from llm_interactions import (
    run_initial_analysis, run_multimodal_qa, run_disease_analysis,
    estimate_ai_confidence, get_gemini_api_url
)
from hf_models import query_hf_vqa_inference_api, HF_VQA_MODEL_ID
from report_utils import generate_pdf_report_bytes

# --- Helper Functions ---

def image_to_data_url(img: Image.Image) -> str:
    """
    Convert a PIL Image to a base64 encoded data URL (PNG format).
    """
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configure Streamlit Page ---
st.set_page_config(page_title="MediVision QA Advanced", layout="wide", page_icon="⚕️")

# --- Initialize Session State ---
DEFAULT_STATE = {
    "uploaded_file_info": None,
    "raw_image_bytes": None,
    "is_dicom": False,
    "dicom_dataset": None,
    "dicom_metadata": {},
    "dicom_wc": None,
    "dicom_ww": None,
    "processed_image": None,
    "display_image": None,
    "session_id": None,
    "history": [],
    "initial_analysis": "",
    "qa_answer": "",
    "disease_analysis": "",
    "confidence_score": "",
    "last_action": None,
    "pdf_report_bytes": None,
    "canvas_drawing": None,
    "roi_coords": None,
    "slider_wc": None,
    "slider_ww": None,
}

for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- App Header & Disclaimer ---
st.title("⚕️ MediVision QA Advanced: AI Image Analysis")
try:
    gemini_url = get_gemini_api_url()
    if gemini_url:
        model_name = gemini_url.split('/')[-1].split(':')[0]
        st.caption(f"Using Gemini Model: `{model_name}` | HF Fallback: `{HF_VQA_MODEL_ID}`")
    else:
        st.caption(f"Gemini Model: Not Configured | HF Fallback: `{HF_VQA_MODEL_ID}`")
except Exception:
    st.caption("Error reading model config.")

st.markdown("---")
st.warning(
    """
    **Disclaimer:** This tool provides AI-generated analysis for informational purposes only.
    It is **NOT** a substitute for professional medical advice, diagnosis, or treatment.
    All outputs **MUST** be reviewed and validated by qualified healthcare professionals.
    """
)

# =============================================================================
# === SIDEBAR CONTROLS ========================================================
# =============================================================================
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader(
        "1. Upload Image (JPG, PNG, DICOM)",
        type=["jpg", "jpeg", "png", "dcm", "dicom"],
        key="file_uploader"
    )

    if uploaded_file is not None:
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{uploaded_file.type}"
        if new_file_info != st.session_state.uploaded_file_info:
            logger.info(f"New file uploaded: {uploaded_file.name}")
            # Reset session state (except the uploader key)
            for key in list(st.session_state.keys()):
                if key not in ["file_uploader"]:
                    st.session_state[key] = DEFAULT_STATE.get(key)

            st.session_state.uploaded_file_info = new_file_info
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.raw_image_bytes = uploaded_file.getvalue()
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            st.session_state.is_dicom = file_ext in (".dcm", ".dicom")

            with st.spinner("Processing file..."):
                if st.session_state.is_dicom:
                    st.session_state.dicom_dataset = parse_dicom(st.session_state.raw_image_bytes)
                    if st.session_state.dicom_dataset:
                        ds = st.session_state.dicom_dataset
                        st.session_state.dicom_metadata = extract_dicom_metadata(ds)
                        wc, ww = get_default_wl(ds)
                        st.session_state.dicom_wc, st.session_state.dicom_ww = wc, ww
                        st.session_state.display_image = dicom_to_image(ds, wc, ww)
                        st.session_state.processed_image = dicom_to_image(ds, None, None)
                        # Get pixel range for slider defaults
                        pixel_min, pixel_max = 0, 4095
                        try:
                            arr = ds.pixel_array
                            pixel_min = float(arr.min())
                            pixel_max = float(arr.max())
                        except Exception:
                            pass
                        st.session_state.slider_wc = wc if wc is not None else (pixel_max + pixel_min) / 2
                        st.session_state.slider_ww = (
                            ww if (ww is not None and ww > 0)
                            else (pixel_max - pixel_min) * 0.8 if pixel_max > pixel_min else 1024
                        )
                    else:
                        st.error("Failed to parse DICOM file.")
                else:
                    try:
                        img = Image.open(io.BytesIO(st.session_state.raw_image_bytes)).convert("RGB")
                        st.session_state.processed_image = img
                        st.session_state.display_image = img
                        st.session_state.dicom_dataset = None
                        st.session_state.dicom_metadata = {}
                    except UnidentifiedImageError:
                        st.error("Cannot identify image file format.")
                    except Exception as e:
                        st.error(f"Error processing image: {e}")

            if st.session_state.processed_image:
                st.success("Image ready.")
                st.rerun()
            else:
                st.error("Image processing failed.")

    st.markdown("---")

    # --- DICOM Window/Level Controls ---
    if st.session_state.processed_image:
        if st.session_state.is_dicom and st.session_state.dicom_dataset:
            st.subheader("DICOM Window/Level")
            ds = st.session_state.dicom_dataset
            default_wc, default_ww = st.session_state.dicom_wc, st.session_state.dicom_ww

            pixel_min, pixel_max = 0, 4095
            try:
                arr = ds.pixel_array
                pixel_min = float(arr.min())
                pixel_max = float(arr.max())
            except Exception:
                pass

            min_level = pixel_min - (pixel_max - pixel_min)
            max_level = pixel_max + (pixel_max - pixel_min)
            max_width = (pixel_max - pixel_min) * 2 if pixel_max > pixel_min else 4096

            current_wc = st.session_state.get("slider_wc", (pixel_max + pixel_min) / 2)
            current_ww = st.session_state.get("slider_ww", (pixel_max - pixel_min) * 0.8 if pixel_max > pixel_min else 1024)

            new_wc = st.slider("Window Center (Level)", min_value=min_level, max_value=max_level, value=current_wc, step=1.0, key="wc_slider")
            new_ww = st.slider("Window Width", min_value=1.0, max_value=max_width, value=current_ww, step=1.0, key="ww_slider")

            if abs(new_wc - current_wc) > 1e-3 or abs(new_ww - current_ww) > 1e-3:
                st.session_state.slider_wc = new_wc
                st.session_state.slider_ww = new_ww
                with st.spinner("Applying Window/Level..."):
                    st.session_state.display_image = dicom_to_image(ds, new_wc, new_ww)
                st.rerun()

            if st.button("Reset W/L", key="reset_wl"):
                wc_reset, ww_reset = get_default_wl(ds)
                px_min, px_max = 0, 4095
                try:
                    arr = ds.pixel_array
                    px_min = float(arr.min())
                    px_max = float(arr.max())
                except Exception:
                    pass
                st.session_state.slider_wc = wc_reset if wc_reset is not None else (px_max + px_min) / 2
                st.session_state.slider_ww = (
                    ww_reset if (ww_reset is not None and ww_reset > 0)
                    else (px_max - px_min) * 0.8 if px_max > px_min else 1024
                )
                st.session_state.display_image = dicom_to_image(ds, wc_reset, ww_reset)
                st.rerun()

            st.markdown("---")

        # --- Analysis Actions ---
        st.subheader("Analysis Actions")
        if st.button("Analyze Image (Initial)", key="analyze_btn", type="primary"):
            st.session_state.last_action = "analyze"
            st.rerun()

        st.markdown("---")

        # --- Follow-up Q&A ---
        st.subheader("Follow-up Question")
        question_input = st.text_area("Ask about image / highlighted region:", height=100, key="question_input")
        if st.session_state.roi_coords:
            if st.button("Clear Highlighted Region (ROI)", key="clear_roi"):
                st.session_state.roi_coords = None
                st.session_state.canvas_drawing = None
                st.rerun()

        if st.button("Ask Gemini", key="ask_btn"):
            if st.session_state.question_input.strip():
                st.session_state.last_action = "ask"
                st.rerun()
            else:
                st.warning("Please enter a question.")

        st.markdown("---")

        # --- Disease-Specific Analysis ---
        st.subheader("Disease-Specific Check")
        disease_options = [
            "", "Pneumonia", "Lung cancer", "Stroke", "Fracture", "Appendicitis", "Tuberculosis",
            "COVID-19 Findings", "Pulmonary embolism", "Glioblastoma", "Meningioma",
            "Arthritis", "Osteoporosis signs", "Cardiomegaly", "Aortic aneurysm", "Bowel obstruction signs"
        ]
        disease_select = st.selectbox("Select Condition Focus:", options=sorted(set(disease_options)), key="disease_select")
        if st.button("Run Focused Analysis", key="disease_btn"):
            if st.session_state.disease_select:
                st.session_state.last_action = "disease"
                st.rerun()
            else:
                st.warning("Please select a condition.")

        st.markdown("---")

        # --- Confidence Estimation & PDF Report ---
        st.subheader("Confidence & Reporting")
        if st.button("Estimate AI Confidence", key="confidence_btn"):
            if st.session_state.history:
                st.session_state.last_action = "confidence"
                st.rerun()
            else:
                st.warning("No analysis/Q&A yet.")

        if st.button("Generate PDF Report Data", key="generate_report_data_btn"):
            st.session_state.last_action = "generate_report_data"
            st.rerun()

        if st.session_state.get("pdf_report_bytes"):
            report_filename = f"medivision_report_{st.session_state.session_id}.pdf"
            st.download_button(
                label="Download PDF Report",
                data=st.session_state.pdf_report_bytes,
                file_name=report_filename,
                mime="application/pdf",
                key="download_pdf_button"
            )
    else:
        st.info("Upload an image to enable controls.")

# =============================================================================
# === MAIN PANEL DISPLAYS =====================================================
# =============================================================================
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Image Viewer")
    if st.session_state.display_image:
        bg_image_pil = st.session_state.display_image
        # Calculate canvas dimensions based on image aspect ratio
        canvas_height = 450
        img_w, img_h = bg_image_pil.width, bg_image_pil.height
        aspect = img_w / img_h if img_h > 0 else 1
        canvas_width = min(int(canvas_height * aspect), 600)
        canvas_height = int(canvas_width / aspect) if aspect > 0 else 400

        st.caption("Click and drag to highlight a Region of Interest (ROI) for questions.")
        # Pass the PIL image directly as the canvas background
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="rgba(255, 165, 0, 0.8)",
            background_image=bg_image_pil,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="rect",
            key="canvas",
        )

        # Process drawn ROI (if any)
        if canvas_result.json_data is not None and canvas_result.json_data.get("objects"):
            last_rect = canvas_result.json_data["objects"][-1]
            rect_width = max(1, int(last_rect.get("width", 1) * last_rect.get("scaleX", 1)))
            rect_height = max(1, int(last_rect.get("height", 1) * last_rect.get("scaleY", 1)))
            st.session_state.roi_coords = {
                "left": int(last_rect.get("left", 0)),
                "top": int(last_rect.get("top", 0)),
                "width": rect_width,
                "height": rect_height,
            }

        # Display DICOM metadata if available
        if st.session_state.is_dicom and st.session_state.dicom_metadata:
            with st.expander("View DICOM Metadata"):
                meta_cols = st.columns(2)
                idx = 0
                for key, value in st.session_state.dicom_metadata.items():
                    if isinstance(value, list):
                        display_val = ", ".join(map(str, value))
                    elif isinstance(value, pydicom.uid.UID):
                        display_val = value.name
                    elif isinstance(value, bytes):
                        try:
                            display_val = value.decode("utf-8", errors="replace")
                        except Exception:
                            display_val = f"[Binary Data ({len(value)} bytes)]"
                    else:
                        display_val = str(value)
                    meta_cols[idx % 2].markdown(f"**{key}:** `{display_val}`")
                    idx += 1
    else:
        st.markdown("*Upload an image using the sidebar.*")

with col2:
    st.subheader("Analysis & Results")
    tabs = st.tabs(["Initial Analysis", "Q&A", "Disease Focus", "Confidence"])

    with tabs[0]:
        st.text_area("Caption, Findings, Differentials",
                     value=st.session_state.initial_analysis,
                     height=350,
                     key="output_initial",
                     disabled=True)
    with tabs[1]:
        st.text_area("Answer",
                     value=st.session_state.qa_answer,
                     height=350,
                     key="output_qa",
                     disabled=True)
        if st.session_state.history:
            with st.expander("View Conversation History"):
                for i, (q, a) in enumerate(reversed(st.session_state.history)):
                    st.markdown(f"**Q{len(st.session_state.history) - i}:** {q}")
                    st.markdown(f"**A{len(st.session_state.history) - i}:**")
                    st.markdown(a, unsafe_allow_html=True)
                    st.markdown("---")
    with tabs[2]:
        st.text_area("Disease-Specific Findings",
                     value=st.session_state.disease_analysis,
                     height=350,
                     key="output_disease",
                     disabled=True)
    with tabs[3]:
        st.text_area("AI Confidence Score & Justification",
                     value=st.session_state.confidence_score,
                     height=200,
                     key="output_confidence",
                     disabled=True)

# =============================================================================
# === ACTION HANDLING ===========================================================
# =============================================================================
current_action = st.session_state.get("last_action")

if current_action:
    logger.info(f"Handling action: {current_action}")
    if not st.session_state.processed_image or not st.session_state.session_id:
        st.error("Action cannot be performed: No processed image or session.")
        st.session_state.last_action = None
        st.stop()

    img_for_llm = st.session_state.processed_image
    roi = st.session_state.roi_coords

    if current_action == "analyze":
        with st.spinner("Performing initial analysis..."):
            result = run_initial_analysis(img_for_llm)
            st.session_state.initial_analysis = result
            st.session_state.qa_answer = ""
            st.session_state.disease_analysis = ""
            st.session_state.confidence_score = ""

    elif current_action == "ask":
        question = st.session_state.question_input.strip()
        st.session_state.qa_answer = ""
        with st.spinner("Thinking... (Querying Gemini)"):
            gemini_answer, success = run_multimodal_qa(img_for_llm, question, st.session_state.history, roi)

        if success:
            st.session_state.qa_answer = gemini_answer
            st.session_state.history.append((question, gemini_answer))
            st.session_state.question_input = ""
        else:
            st.session_state.qa_answer = f"Gemini Failed: {gemini_answer}"
            st.error("Gemini query failed. Attempting Hugging Face VQA fallback...")
            with st.spinner(f"Attempting HF Fallback ({HF_VQA_MODEL_ID})..."):
                if os.environ.get("HF_API_TOKEN"):
                    hf_answer, hf_success = query_hf_vqa_inference_api(img_for_llm, question)
                    if hf_success:
                        fallback_display = (
                            f"**[Fallback Answer from Hugging Face VQA ({HF_VQA_MODEL_ID})]**\n\n{hf_answer}"
                        )
                        st.session_state.qa_answer = fallback_display
                        st.session_state.history.append((question, fallback_display))
                        st.session_state.question_input = ""
                        st.info("Hugging Face VQA fallback successful.")
                    else:
                        st.session_state.qa_answer += f"\n\n---\n**Hugging Face Fallback Failed:** {hf_answer}"
                        st.error(f"Hugging Face VQA fallback also failed: {hf_answer}")
                else:
                    st.session_state.qa_answer += "\n\n---\n**[Fallback Unavailable: HF Token Missing]**"
                    st.warning("Hugging Face API token not configured. Cannot use VQA fallback.")

    elif current_action == "disease":
        disease = st.session_state.disease_select
        with st.spinner(f"Analyzing for {disease}..."):
            result = run_disease_analysis(img_for_llm, disease, roi)
            st.session_state.disease_analysis = result
            st.session_state.qa_answer = ""
            st.session_state.confidence_score = ""

    elif current_action == "confidence":
        with st.spinner("Estimating confidence..."):
            result = estimate_ai_confidence(img_for_llm, st.session_state.history)
            st.session_state.confidence_score = result

    elif current_action == "generate_report_data":
        st.session_state.pdf_report_bytes = None
        with st.spinner("Generating PDF report..."):
            img_for_report = st.session_state.display_image
            if img_for_report:
                full_qa_history = "\n\n".join(
                    [f"User Q: {q}\n\nAI A: {a}" for q, a in st.session_state.history]
                ) or "No Q&A history for this session."
                outputs_for_report = {
                    "Initial Analysis": st.session_state.initial_analysis or "Not performed.",
                    "Conversation History": full_qa_history,
                    "Disease-Specific Analysis": st.session_state.disease_analysis or "Not performed.",
                    "Last Confidence Estimate": st.session_state.confidence_score or "Not estimated.",
                }
                if st.session_state.is_dicom and st.session_state.dicom_metadata:
                    meta_str_list = []
                    for k, v in st.session_state.dicom_metadata.items():
                        if isinstance(v, list):
                            display_v = ", ".join(map(str, v))
                        elif isinstance(v, pydicom.uid.UID):
                            display_v = v.name
                        elif isinstance(v, bytes):
                            try:
                                display_v = v.decode("utf-8", errors="replace")
                            except Exception:
                                display_v = f"[Binary Data ({len(v)} bytes)]"
                        else:
                            display_v = str(v)
                        meta_str_list.append(f"{k}: {display_v}")
                    outputs_for_report["DICOM Metadata"] = "\n".join(meta_str_list)

                pdf_bytes = generate_pdf_report_bytes(
                    st.session_state.session_id, img_for_report, outputs_for_report
                )
                if pdf_bytes:
                    st.session_state.pdf_report_bytes = pdf_bytes
                    st.success("PDF data generated. Download button available.")
                else:
                    st.error("Failed to generate PDF report data.")

    st.session_state.last_action = None
    st.rerun()

# =============================================================================
# === FOOTER ==================================================================
# =============================================================================
st.markdown("---")
st.caption(f"Session ID: `{st.session_state.session_id if st.session_state.session_id else 'N/A'}`")
hf_token_status = "Configured" if os.environ.get("HF_API_TOKEN") else "Not Configured"
gemini_key_status = "Configured" if os.environ.get("GEMINI_API_KEY") else "Not Configured"
st.caption(f"Gemini API Key: {gemini_key_status} | Hugging Face Token: {hf_token_status}")
