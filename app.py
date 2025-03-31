import streamlit as st
from PIL import Image, UnidentifiedImageError
import uuid
import io
import os
import logging
from streamlit_drawable_canvas import st_canvas # Import canvas
from typing import Optional, Tuple, List, Dict, Any

# Import functions from our modules
from dicom_utils import parse_dicom, extract_dicom_metadata, dicom_to_image, get_default_wl
from llm_interactions import (
    run_initial_analysis, run_multimodal_qa, run_disease_analysis,
    estimate_ai_confidence
)
from hf_models import query_hf_vqa_inference_api # Import HF function
from report_utils import generate_pdf_report_bytes
# from ui_components import display_dicom_metadata, dicom_wl_sliders # Optional import

# --- Basic Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Page Config ---
st.set_page_config(page_title="MediVision QA Advanced", layout="wide", page_icon="⚕️")

# --- Initialize Session State ---
DEFAULT_STATE = {
    "uploaded_file_info": None, "raw_image_bytes": None, "is_dicom": False,
    "dicom_dataset": None, "dicom_metadata": {}, "dicom_wc": None, "dicom_ww": None,
    "processed_image": None, "display_image": None, # Separate processed (for LLM) and display (with W/L)
    "session_id": None, "history": [],
    "initial_analysis": "", "qa_answer": "", "disease_analysis": "", "confidence_score": "",
    "last_action": None, "pdf_report_bytes": None,
    "canvas_drawing": None, "roi_coords": None, # For drawable canvas
}
for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- UI: Title & Disclaimer ---
st.title("⚕️ MediVision QA Advanced: AI Image Analysis")
# st.caption(f"Using Gemini Model: `{llm_interactions.get_gemini_api_url().split('/')[-1].split(':')[0]}`") # Show model
st.markdown("---")
st.warning("Disclaimer: For informational purposes only. Not a substitute for professional medical advice. Validate all outputs.")

# ==============================================================================
# === SIDEBAR CONTROLS =========================================================
# ==============================================================================
with st.sidebar:
    st.header("Controls")

    # --- 1. File Upload ---
    uploaded_file = st.file_uploader(
        "1. Upload Image (JPG, PNG, DICOM)",
        type=["jpg", "jpeg", "png", "dcm", "dicom"],
        key="file_uploader"
    )

    # --- Process Upload ---
    if uploaded_file is not None:
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{uploaded_file.type}"
        # If it's a new file, reset state and process
        if new_file_info != st.session_state.uploaded_file_info:
            logger.info(f"New file uploaded: {uploaded_file.name}")
            # Reset relevant state variables
            for key in list(st.session_state.keys()): # Iterate over copy of keys
                 if key not in ["file_uploader"]: # Don't reset the uploader widget state itself
                      st.session_state[key] = DEFAULT_STATE.get(key) # Reset to default

            st.session_state.uploaded_file_info = new_file_info
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.raw_image_bytes = uploaded_file.getvalue()
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            st.session_state.is_dicom = file_ext in (".dcm", ".dicom")

            with st.spinner("Processing file..."):
                if st.session_state.is_dicom:
                    st.session_state.dicom_dataset = parse_dicom(st.session_state.raw_image_bytes)
                    if st.session_state.dicom_dataset:
                        st.session_state.dicom_metadata = extract_dicom_metadata(st.session_state.dicom_dataset)
                        # Get initial W/L for display
                        wc, ww = get_default_wl(st.session_state.dicom_dataset)
                        st.session_state.dicom_wc, st.session_state.dicom_ww = wc, ww
                        # Generate initial display image and base processed image
                        st.session_state.display_image = dicom_to_image(st.session_state.dicom_dataset, wc, ww)
                        st.session_state.processed_image = dicom_to_image(st.session_state.dicom_dataset) # Base image for LLM
                        if st.session_state.processed_image is None: st.error("Failed to create base image from DICOM.")
                    else:
                        st.error("Failed to parse DICOM file.")
                else: # Standard image
                    try:
                        img = Image.open(io.BytesIO(st.session_state.raw_image_bytes)).convert("RGB")
                        st.session_state.processed_image = img
                        st.session_state.display_image = img # Display and processed are the same
                    except UnidentifiedImageError: st.error("Cannot identify image file format.")
                    except Exception as e: st.error(f"Error processing image: {e}")

            if st.session_state.processed_image:
                 st.success("Image ready.")
                 # Force rerun to update UI immediately after processing
                 st.rerun()
            else:
                 st.error("Image processing failed.")


    st.markdown("---")

    # --- Controls requiring a processed image ---
    if st.session_state.processed_image:
        # --- 2. DICOM W/L Controls (Conditional) ---
        if st.session_state.is_dicom and st.session_state.dicom_dataset:
             # Using ui_components function (or place code here)
             # wc, ww = dicom_wl_sliders(st.session_state.dicom_dataset, st.session_state.dicom_metadata)

             # Or inline code:
             st.subheader("DICOM Window/Level")
             ds = st.session_state.dicom_dataset
             metadata = st.session_state.dicom_metadata
             default_wc, default_ww = st.session_state.dicom_wc, st.session_state.dicom_ww # Initial defaults

             pixel_min, pixel_max = 0, 4095
             try:
                 pixel_array = ds.pixel_array
                 pixel_min = float(pixel_array.min()); pixel_max = float(pixel_array.max())
             except Exception: pass # Ignore if fails

             min_level=pixel_min-(pixel_max-pixel_min); max_level=pixel_max+(pixel_max-pixel_min)
             max_width=(pixel_max-pixel_min)*2 if pixel_max>pixel_min else 4096

             # Use session state to store current slider values
             if 'slider_wc' not in st.session_state: st.session_state.slider_wc = default_wc if default_wc is not None else (pixel_max+pixel_min)/2
             if 'slider_ww' not in st.session_state: st.session_state.slider_ww = default_ww if default_ww is not None else (pixel_max-pixel_min)*0.8

             new_wc = st.slider("Window Center (Level)", min_value=min_level, max_value=max_level, value=st.session_state.slider_wc, step=1.0)
             new_ww = st.slider("Window Width", min_value=1.0, max_value=max_width, value=st.session_state.slider_ww, step=1.0)

             # Update display image only if W/L changed significantly
             threshold = 1e-3 # To avoid float comparison issues
             if abs(new_wc - st.session_state.slider_wc) > threshold or abs(new_ww - st.session_state.slider_ww) > threshold:
                  st.session_state.slider_wc = new_wc
                  st.session_state.slider_ww = new_ww
                  with st.spinner("Applying Window/Level..."):
                       st.session_state.display_image = dicom_to_image(ds, new_wc, new_ww)
                       st.rerun() # Update the main image display

             st.markdown("---")


        # --- 3. Analysis Buttons ---
        st.subheader("Analysis Actions")
        if st.button("Analyze Image (Initial)", key="analyze_btn", type="primary"):
            st.session_state.last_action = "analyze"
            st.rerun()

        st.markdown("---")

        # --- 4. Follow-up Q&A ---
        st.subheader("Follow-up Question")
        question_input = st.text_area(
            "Ask about the image (or highlighted region):",
            height=100,
            key="question_input"
        )
        # Clear ROI button
        if st.session_state.roi_coords:
             if st.button("Clear Highlighted Region (ROI)", key="clear_roi"):
                 st.session_state.roi_coords = None
                 st.session_state.canvas_drawing = None # Reset canvas state if needed
                 st.rerun()

        if st.button("Ask Gemini", key="ask_btn"):
             if st.session_state.question_input.strip():
                 st.session_state.last_action = "ask"
                 st.rerun()
             else: st.warning("Please enter a question.")

        st.markdown("---")

        # --- 5. Disease Specific Check ---
        st.subheader("Disease-Specific Check")
        # Simplified list for example
        disease_options = ["", "Pneumonia", "Lung cancer", "Stroke", "Fracture", "Appendicitis"]
        disease_select = st.selectbox("Select Condition Focus:", options=disease_options, key="disease_select")
        if st.button("Run Focused Analysis", key="disease_btn"):
            if st.session_state.disease_select:
                st.session_state.last_action = "disease"
                st.rerun()
            else: st.warning("Please select a condition.")

        st.markdown("---")

        # --- 6. Confidence & Reporting ---
        st.subheader("Confidence & Reporting")
        if st.button("Estimate AI Confidence", key="confidence_btn"):
             if st.session_state.history:
                 st.session_state.last_action = "confidence"
                 st.rerun()
             else: st.warning("No analysis/Q&A yet.")

        if st.button("Generate PDF Report Data", key="generate_report_data_btn"):
            st.session_state.last_action = "generate_report_data"
            st.rerun()

        if st.session_state.pdf_report_bytes:
            report_filename = f"medivision_report_{st.session_state.session_id}.pdf"
            st.download_button("Download PDF Report", data=st.session_state.pdf_report_bytes, file_name=report_filename, mime="application/pdf")

    else:
        st.info("Upload an image to enable controls.")


# ==============================================================================
# === MAIN PANEL DISPLAYS ======================================================
# ==============================================================================
col1, col2 = st.columns([2, 3]) # Give more space to results

with col1:
    st.subheader("Image Viewer")
    if st.session_state.display_image:
        # --- Drawable Canvas for ROI ---
        canvas_height = 450 # Adjust as needed
        canvas_width = int(canvas_height * (st.session_state.display_image.width / st.session_state.display_image.height))

        # Convert PIL image to format suitable for background
        # bg_image_bytes = io.BytesIO()
        # st.session_state.display_image.save(bg_image_bytes, format="PNG")
        # bg_image_b64 = base64.b64encode(bg_image_bytes.getvalue()).decode()
        # bg_image_uri = f"data:image/png;base64,{bg_image_b64}"

        st.caption("Click and drag to highlight a Region of Interest (ROI) for questions.")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Orange fill with transparency
            stroke_width=2,
            stroke_color="rgba(255, 165, 0, 0.8)", # Orange stroke
            background_image=st.session_state.display_image, # Use PIL image directly
            update_streamlit=True, # Trigger reruns on drawing change
            height=canvas_height,
            width=canvas_width,
            drawing_mode="rect", # Allow drawing rectangles
            key="canvas",
        )

        # Process canvas results to get ROI
        if canvas_result.json_data is not None:
             objects = canvas_result.json_data.get("objects", [])
             if objects: # If something was drawn
                  # Get the last drawn rectangle
                  last_rect = objects[-1]
                  # Store coordinates relative to the canvas size
                  st.session_state.roi_coords = {
                       "left": int(last_rect["left"]),
                       "top": int(last_rect["top"]),
                       "width": int(last_rect["width"] * last_rect.get("scaleX", 1)), # Account for potential scaling
                       "height": int(last_rect["height"] * last_rect.get("scaleY", 1))
                  }
                  # Optional: Display coords for debugging
                  # st.caption(f"ROI (approx): {st.session_state.roi_coords}")
             else: # If drawing was cleared
                  if st.session_state.roi_coords is not None: # Clear if it was previously set
                     st.session_state.roi_coords = None

        else: # Canvas is empty or reset
             if st.session_state.roi_coords is not None: # Clear if it was previously set
                 st.session_state.roi_coords = None


        # --- Display DICOM Metadata (Conditional) ---
        if st.session_state.is_dicom and st.session_state.dicom_metadata:
             # Using ui_components function (or place code here)
             # display_dicom_metadata(st.session_state.dicom_metadata)

             # Or inline code:
             with st.expander("View DICOM Metadata"):
                meta_cols = st.columns(2)
                idx = 0
                for key, value in st.session_state.dicom_metadata.items():
                    display_val = str(value)
                    if isinstance(value, list): display_val = ", ".join(map(str, value))
                    elif isinstance(value, pydicom.uid.UID): display_val = value.name
                    meta_cols[idx%2].markdown(f"**{key}:** `{display_val}`")
                    idx+=1

    else:
        st.markdown("*Upload an image using the sidebar.*")

with col2:
    st.subheader("Analysis & Results")
    tabs = st.tabs(["Initial Analysis", "Q&A", "Disease Focus", "Confidence"])

    with tabs[0]: st.text_area("Caption, Findings, Differentials", value=st.session_state.initial_analysis, height=300, key="output_initial", disabled=True)
    with tabs[1]:
        st.text_area("Answer", value=st.session_state.qa_answer, height=300, key="output_qa", disabled=True)
        if st.session_state.history:
            with st.expander("View Conversation History"):
                for i, (q, a) in enumerate(reversed(st.session_state.history)):
                    st.markdown(f"**Q{len(st.session_state.history)-i}:** {q}\n\n**A{len(st.session_state.history)-i}:** {a}\n\n---")
    with tabs[2]: st.text_area("Disease-Specific Findings", value=st.session_state.disease_analysis, height=300, key="output_disease", disabled=True)
    with tabs[3]: st.text_area("AI Confidence Score & Justification", value=st.session_state.confidence_score, height=150, key="output_confidence", disabled=True)


# ==============================================================================
# === ACTION HANDLING (EXECUTED ON RERUN AFTER BUTTON PRESS) ==================
# ==============================================================================
current_action = st.session_state.get("last_action")

if current_action:
    logger.info(f"Handling action: {current_action}")
    # Ensure image exists before proceeding
    if not st.session_state.processed_image or not st.session_state.session_id:
        st.error("Action cannot be performed: No processed image or session.")
        st.session_state.last_action = None # Reset action
        st.stop()

    img_for_llm = st.session_state.processed_image # Use the base processed image for LLM
    roi = st.session_state.roi_coords # Get current ROI

    # --- Initial Analysis ---
    if current_action == "analyze":
        with st.spinner("Performing initial analysis..."):
            result = run_initial_analysis(img_for_llm)
            st.session_state.initial_analysis = result
            # Optionally add to history:
            # st.session_state.history.append(("Initial Analysis Request", result))
    # --- Ask Question ---
    elif current_action == "ask":
        question = st.session_state.question_input.strip()
        with st.spinner("Thinking... (Querying Gemini)"):
            gemini_answer, success = run_multimodal_qa(img_for_llm, question, st.session_state.history, roi)

        if success:
            st.session_state.qa_answer = gemini_answer
            st.session_state.history.append((question, gemini_answer))
            st.session_state.question_input = "" # Clear box
        else:
            st.session_state.qa_answer = f"Gemini Failed: {gemini_answer}"
            st.error("Gemini query failed. Attempting Hugging Face VQA fallback...")
            # --- Hugging Face Fallback ---
            with st.spinner("Attempting Hugging Face VQA fallback..."):
                hf_api_token = st.secrets.get("HF_API_TOKEN") # Check token before calling
                if hf_api_token:
                     # Note: HF VQA doesn't inherently use history or complex ROI prompts like Gemini easily
                     # We pass the raw question and image.
                     hf_answer, hf_success = query_hf_vqa_inference_api(img_for_llm, question)
                     if hf_success:
                          fallback_display = f"**[Fallback Answer from Hugging Face VQA ({HF_VQA_MODEL_ID})]**\n{hf_answer}"
                          st.session_state.qa_answer = fallback_display # Display fallback answer
                          st.session_state.history.append((question, fallback_display)) # Store fallback in history
                          st.session_state.question_input = ""
                          st.info("Hugging Face VQA fallback successful.")
                     else:
                          st.session_state.qa_answer += f"\n\nHugging Face Fallback Failed: {hf_answer}"
                          st.error(f"Hugging Face VQA fallback also failed: {hf_answer}")
                else:
                     st.warning("Hugging Face API token not configured. Cannot use VQA fallback.")
                     st.session_state.qa_answer += "\n\n[Fallback Unavailable: HF Token Missing]"

    # --- Disease Analysis ---
    elif current_action == "disease":
        disease = st.session_state.disease_select
        with st.spinner(f"Analyzing for {disease}..."):
            result = run_disease_analysis(img_for_llm, disease, roi)
            st.session_state.disease_analysis = result
            # Optional history add:
            # st.session_state.history.append((f"Disease Focus: {disease}", result))

    # --- Confidence Estimation ---
    elif current_action == "confidence":
        with st.spinner("Estimating confidence..."):
            result = estimate_ai_confidence(img_for_llm, st.session_state.history)
            st.session_state.confidence_score = result

    # --- PDF Report Generation ---
    elif current_action == "generate_report_data":
        st.session_state.pdf_report_bytes = None # Clear old data
        with st.spinner("Generating PDF report..."):
            # Use the *currently displayed* image for the report visual
            img_for_report = st.session_state.display_image if st.session_state.display_image else st.session_state.processed_image

            full_qa_history = "\n\n".join([f"User Q: {q}\n\nAI A: {a}" for q, a in st.session_state.history]) or "No Q&A."
            outputs_for_report = {
                "Initial Analysis": st.session_state.initial_analysis or "N/A",
                "Conversation History": full_qa_history,
                "Disease-Specific Analysis": st.session_state.disease_analysis or "N/A",
                "Last Confidence Estimate": st.session_state.confidence_score or "N/A"
            }
            # Include DICOM metadata if available
            if st.session_state.is_dicom and st.session_state.dicom_metadata:
                 meta_str = "\n".join([f"{k}: {v}" for k,v in st.session_state.dicom_metadata.items()])
                 outputs_for_report["DICOM Metadata"] = meta_str


            pdf_bytes = generate_pdf_report_bytes(
                st.session_state.session_id,
                img_for_report,
                outputs_for_report
            )
            if pdf_bytes:
                st.session_state.pdf_report_bytes = pdf_bytes
                st.success("PDF data generated. Download button available.")
            else: st.error("Failed to generate PDF report.")

    # --- Reset Action and Rerun ---
    st.session_state.last_action = None
    st.rerun()


# ==============================================================================
# === FOOTER ===================================================================
# ==============================================================================
st.markdown("---")
st.caption(f"Session ID: `{st.session_state.session_id}`")
# Add other footer info if needed