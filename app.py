# Assume these imports are present at the top of your Streamlit script
import streamlit as st
from PIL import Image, ImageDraw
import pydicom
import os
import logging
from typing import Optional, Dict, Tuple, List, Any, Union # Added Union for clarity

# --- Assumptions ---
# Assume 'logger' is configured and available (e.g., using Python's logging module)
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO) # Example basic config if needed

# Assume HF_VQA_MODEL_ID is defined, e.g.:
HF_VQA_MODEL_ID: Optional[str] = "llava-hf/llava-1.5-7b-hf" # Example model

# Assume the following functions are defined elsewhere and handle the core logic:
# These functions should return results and potentially a boolean indicating success/failure
def run_initial_analysis(image: Image.Image) -> str:
    """Placeholder for initial image analysis AI call."""
    logger.debug("Placeholder: run_initial_analysis called")
    # Simulate AI processing
    import time; time.sleep(2)
    return "AI Preliminary Findings: Image appears unremarkable in this simulation."

def run_multimodal_qa(image: Image.Image, question: str, history: List[Tuple[str, str]], roi: Optional[Dict[str, int]]) -> Tuple[str, bool]:
    """Placeholder for multimodal QA call (e.g., Gemini)."""
    logger.debug(f"Placeholder: run_multimodal_qa called with question: '{question}', ROI: {bool(roi)}")
    # Simulate AI processing - potentially fail sometimes for fallback testing
    import random; import time; time.sleep(2)
    if "fail" in question.lower() or random.random() < 0.1: # Simulate failure condition
         return "Simulated primary AI processing error.", False
    answer = f"AI Simulation: Based on the image {'and ROI ' if roi else ''}regarding '{question}', the simulated finding is..."
    return answer, True

def query_hf_vqa_inference_api(image: Image.Image, question: str, roi: Optional[Dict[str, int]]) -> Tuple[str, bool]:
    """Placeholder for Hugging Face VQA fallback API call."""
    logger.debug(f"Placeholder: query_hf_vqa_inference_api called with question: '{question}', ROI: {bool(roi)}")
    # Simulate API call
    import time; time.sleep(3)
    # Simulate potential failure
    if "fail hard" in question.lower():
        return "Simulated fallback API connection error.", False
    # Crop image if ROI is provided (real implementation needed)
    processed_img = image
    if roi:
        try:
            box = (roi['left'], roi['top'], roi['left'] + roi['width'], roi['top'] + roi['height'])
            processed_img = image.crop(box)
            logger.debug("Cropped image to ROI for HF VQA.")
        except Exception as e:
            logger.error(f"Failed to crop image for HF VQA ROI: {e}")
            # Proceed with full image if crop fails

    answer = f"Simulated Fallback AI: Answering '{question}' based on the {'ROI' if roi else 'full image'}..."
    return answer, True

def run_disease_analysis(image: Image.Image, disease: str, roi: Optional[Dict[str, int]]) -> str:
    """Placeholder for disease-specific analysis AI call."""
    logger.debug(f"Placeholder: run_disease_analysis called for '{disease}', ROI: {bool(roi)}")
    # Simulate AI processing
    import time; time.sleep(2)
    return f"AI Simulation: Analysis for '{disease}' {'within the ROI ' if roi else ''}suggests [simulated results]."

def estimate_ai_confidence(history: List[Tuple[str, str]], image: Image.Image, roi: Optional[Dict[str, int]], **kwargs) -> str:
    """Placeholder for AI confidence estimation logic."""
    logger.debug(f"Placeholder: estimate_ai_confidence called. History length: {len(history)}, ROI: {bool(roi)}")
    # Simulate confidence calculation based on history length or other factors
    import time; time.sleep(1)
    confidence = min(95, 50 + len(history) * 10) # Example basic logic
    return f"Simulated Confidence Score: {confidence}%"

def generate_pdf_report_bytes(session_id: str, image: Image.Image, analysis_outputs: Dict[str, str]) -> Optional[bytes]:
    """Placeholder for PDF generation function using libraries like reportlab or fpdf."""
    logger.debug(f"Placeholder: generate_pdf_report_bytes called for session {session_id}")
    # Simulate PDF generation
    from io import BytesIO
    try:
        # In a real scenario, use reportlab, fpdf2, etc. to create a PDF
        buffer = BytesIO()
        # Simulate writing basic content
        buffer.write(f"--- Simulated PDF Report ---\n".encode('utf-8'))
        buffer.write(f"Session ID: {session_id}\n\n".encode('utf-8'))

        # Add text outputs
        for key, value in analysis_outputs.items():
            buffer.write(f"--- {key} ---\n".encode('utf-8'))
            # Basic sanitization/wrapping might be needed for real PDF libs
            buffer.write(f"{value}\n\n".encode('utf-8', errors='replace'))

        # Simulate adding the image (real libraries have methods for this)
        buffer.write(f"\n--- Image Included (Simulation) ---\n".encode('utf-8'))
        # img_buffer = BytesIO()
        # image.save(img_buffer, format="PNG")
        # buffer.write(img_buffer.getvalue()) # How you add images depends on the PDF lib

        buffer.write(f"\n--- Disclaimer ---\n".encode('utf-8'))
        buffer.write("For Research, Informational & Educational Purposes Only. Not for clinical diagnosis.\n".encode('utf-8'))

        buffer.seek(0)
        logger.info("Simulated PDF generation successful.")
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Simulated PDF generation failed: {e}", exc_info=True)
        return None

# --- Initialize Session State Keys (if not done elsewhere) ---
# It's good practice to initialize all expected keys at the start of the app
if "processed_image" not in st.session_state: st.session_state.processed_image = None
if "session_id" not in st.session_state: st.session_state.session_id = None # Should be set upon session start
if "roi_coords" not in st.session_state: st.session_state.roi_coords = None
if "last_action" not in st.session_state: st.session_state.last_action = None
if "question_input" not in st.session_state: st.session_state.question_input = ""
if "history" not in st.session_state: st.session_state.history = [] # List of (question, answer) tuples
if "disease_select" not in st.session_state: st.session_state.disease_select = None
if "initial_analysis" not in st.session_state: st.session_state.initial_analysis = ""
if "qa_answer" not in st.session_state: st.session_state.qa_answer = ""
if "disease_analysis" not in st.session_state: st.session_state.disease_analysis = ""
if "confidence_score" not in st.session_state: st.session_state.confidence_score = ""
if "pdf_report_bytes" not in st.session_state: st.session_state.pdf_report_bytes = None
if "display_image" not in st.session_state: st.session_state.display_image = None # Image shown in UI (may have ROI)
if "is_dicom" not in st.session_state: st.session_state.is_dicom = False
if "dicom_metadata" not in st.session_state: st.session_state.dicom_metadata = {}
# --- End Session State Initialization ---


# =============================================================================
# === ACTION HANDLING ===========================================================
# =============================================================================

# Retrieve the action requested by the user interface
current_action: Optional[str] = st.session_state.get("last_action")

if current_action:
    logger.info(f"Initiating action handling for: '{current_action}'")

    # --- Pre-Action Validation ---
    # Essential check: Ensure we have a valid image and session to work with.
    # Use .get() for safer access
    processed_image = st.session_state.get("processed_image")
    session_id = st.session_state.get("session_id")

    if not isinstance(processed_image, Image.Image) or not session_id:
        error_msg = f"Cannot perform action '{current_action}': A processed image is required, but not found or the session is invalid."
        st.error(error_msg)
        # Log detailed type information for debugging
        processed_image_type = type(processed_image).__name__ if processed_image else "None"
        logger.error(
            f"Action '{current_action}' aborted pre-check. "
            f"Processed image type: {processed_image_type}, "
            f"Session ID valid: {bool(session_id)}"
        )
        # Reset the action and stop execution for this run
        st.session_state.last_action = None
        st.stop() # Prevent further execution in this Streamlit run

    # Prepare common variables for actions
    img_for_llm: Image.Image = processed_image
    roi: Optional[Dict[str, int]] = st.session_state.get("roi_coords")
    # User-friendly string indicating if an ROI is active for spinner messages
    roi_context_str = " (focusing on selected ROI)" if roi else ""

    # --- Action Execution Block ---
    try:
        # Reset relevant state variables *before* performing the new action
        # This prevents stale data from previous actions being displayed.
        # We keep the result of the *current* action type, clearing others.
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
            question = st.session_state.get("question_input", "").strip()
            if not question:
                st.warning("Please enter a question before clicking 'Ask'.")
                logger.warning("'Ask' action triggered with empty question input.")
            else:
                st.info(f"❓ Asking the AI your question{roi_context_str}...")
                st.session_state.qa_answer = "" # Clear previous answer explicitly here
                primary_model_name = "Gemini" # Example: Make model name explicit
                with st.spinner(f"{primary_model_name} is processing your question{roi_context_str}..."):
                    gemini_answer, success = run_multimodal_qa(img_for_llm, question, st.session_state.history, roi)

                if success:
                    st.session_state.qa_answer = gemini_answer
                    # Ensure history stores tuples of (question, answer)
                    if isinstance(st.session_state.history, list):
                         st.session_state.history.append((question, gemini_answer))
                    else:
                         st.session_state.history = [(question, gemini_answer)] # Initialize if not list
                    logger.info(f"Multimodal QA successful (Primary: {primary_model_name}) for question: '{question}'{roi_context_str}")
                    st.success(f"{primary_model_name} answered your question.")
                else:
                    # --- Fallback Logic ---
                    error_message = f"Primary AI ({primary_model_name}) failed to answer. Reason: {gemini_answer}"
                    st.error(error_message)
                    logger.warning(f"Primary AI ({primary_model_name}) failed for question: '{question}'. Reason: {gemini_answer}")
                    # Display primary error before attempting fallback
                    st.session_state.qa_answer = f"**[Primary AI Error]** {gemini_answer}\n\n---\n"

                    # Check prerequisites for fallback (e.g., API token and Model ID)
                    hf_token_available = bool(os.environ.get("HF_API_TOKEN"))
                    if hf_token_available and HF_VQA_MODEL_ID:
                        st.info(f"Attempting fallback using Hugging Face model ({HF_VQA_MODEL_ID})...")
                        with st.spinner(f"Trying fallback AI ({HF_VQA_MODEL_ID}){roi_context_str}..."):
                            hf_answer, hf_success = query_hf_vqa_inference_api(img_for_llm, question, roi)

                        if hf_success:
                            fallback_display = f"**[Fallback Answer ({HF_VQA_MODEL_ID})]**\n\n{hf_answer}"
                            st.session_state.qa_answer += fallback_display # Append fallback answer
                            # Add fallback Q&A to history
                            if isinstance(st.session_state.history, list):
                                st.session_state.history.append((question, fallback_display))
                            else:
                                st.session_state.history = [(question, fallback_display)] # Initialize if needed
                            logger.info(f"HF VQA fallback successful for question: '{question}'{roi_context_str}")
                            st.success(f"Fallback AI ({HF_VQA_MODEL_ID}) provided an answer.")
                        else:
                            fallback_error_msg = f"Fallback AI ({HF_VQA_MODEL_ID}) also failed. Reason: {hf_answer}"
                            st.session_state.qa_answer += f"**[Fallback Failed]** {fallback_error_msg}" # Append fallback failure
                            st.error(fallback_error_msg)
                            logger.error(f"HF VQA fallback failed for question: '{question}'. Reason: {hf_answer}")
                    else:
                        missing_config = []
                        if not hf_token_available: missing_config.append("HF Token")
                        if not HF_VQA_MODEL_ID: missing_config.append("Model ID")
                        fallback_unavailable_msg = f"Fallback AI unavailable (Configuration missing: {', '.join(missing_config)})."
                        st.session_state.qa_answer += f"**[Fallback Unavailable]** {fallback_unavailable_msg}" # Append unavailability info
                        st.warning(fallback_unavailable_msg)
                        logger.warning(f"HF VQA fallback skipped for question '{question}': Configuration missing.")

        elif current_action == "disease":
            selected_disease = st.session_state.get("disease_select")
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
            # Check if there's history to base confidence on
            if not isinstance(current_history, list) or not current_history:
                 st.warning("Cannot estimate confidence without prior Q&A interaction.")
                 logger.warning("Confidence estimation skipped: No interaction history found.")
            else:
                with st.spinner(f"Calculating confidence score{roi_context_str}..."):
                    # Pass relevant context to the confidence estimation function
                    confidence_result = estimate_ai_confidence(
                        history=current_history,
                        image=img_for_llm, # Use the base processed image
                        roi=roi,
                        # Pass other state if the function needs it:
                        # initial_analysis=st.session_state.get("initial_analysis"),
                        # disease_analysis=st.session_state.get("disease_analysis")
                    )
                st.session_state.confidence_score = confidence_result
                logger.info("Confidence estimation action completed.")
                st.success("AI confidence estimation finished.")


        elif current_action == "generate_report_data":
            st.info("📄 Preparing data for PDF report generation...")
            st.session_state.pdf_report_bytes = None # Ensure clean state

            # Use the image currently displayed to the user (might have ROI drawn)
            img_for_report: Optional[Image.Image] = st.session_state.get("display_image")
            img_with_roi_drawn = None # Initialize

            if isinstance(img_for_report, Image.Image):
                # Attempt to draw ROI if it exists
                current_roi = st.session_state.get("roi_coords")
                if current_roi and isinstance(current_roi, dict) and all(k in current_roi for k in ['left', 'top', 'width', 'height']):
                    try:
                        # Work on a copy to avoid modifying the original display image state
                        img_copy = img_for_report.copy().convert("RGB") # Ensure RGB for color drawing
                        draw = ImageDraw.Draw(img_copy)
                        x0, y0 = int(current_roi['left']), int(current_roi['top'])
                        x1, y1 = x0 + int(current_roi['width']), y0 + int(current_roi['height'])
                        # Use a clearly visible color and thickness
                        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
                        img_with_roi_drawn = img_copy # Use this image for the report
                        logger.info("Successfully drew ROI onto image copy for PDF report.")
                    except Exception as e:
                        logger.error(f"Failed to draw ROI on report image: {e}", exc_info=True)
                        st.warning("Could not draw the ROI box onto the report image, using original.")
                        img_with_roi_drawn = img_for_report # Fallback to original if drawing fails
                else:
                    img_with_roi_drawn = img_for_report # Use original if no ROI or ROI invalid

                # --- Prepare Report Content ---
                # Format Q&A history nicely
                current_history = st.session_state.get("history", [])
                full_qa_history = "\n\n".join([f"User Q: {q}\n\nAI A: {a}" for q, a in current_history]) if current_history else "No Q&A interactions recorded."

                # Consolidate outputs using .get for safety
                outputs_for_report = {
                    "Preliminary Analysis": st.session_state.get("initial_analysis", "Not performed."),
                    "Conversation History": full_qa_history,
                    "Condition-Specific Analysis": st.session_state.get("disease_analysis", "Not performed."),
                    "Last Confidence Estimate": st.session_state.get("confidence_score", "Not estimated.")
                }

                # --- **CRITICAL: DICOM Metadata Handling for Privacy** ---
                if st.session_state.get("is_dicom") and st.session_state.get("dicom_metadata"):
                    logger.info("Processing DICOM metadata for report.")
                    filtered_meta_str_list = []
                    # Define tags to EXCLUDE (Example - **MUST BE ADAPTED** based on actual requirements and regulations like HIPAA/GDPR)
                    # This is NOT exhaustive and requires careful review for your specific context.
                    PHI_TAGS_TO_EXCLUDE = [
                        "PatientName", "PatientID", "PatientBirthDate", "PatientSex",
                        "OtherPatientIDs", "OtherPatientNames", "PatientAddress",
                        "PatientTelephoneNumbers", "ReferringPhysicianName", "InstitutionName",
                        "InstitutionAddress", "PhysicianOfRecord", "OperatorsName",
                        "IssuerOfPatientID", "PatientBirthTime", "PatientComments",
                        "PerformedProcedureStepStartDate", "PerformedProcedureStepStartTime",
                        "RequestingPhysician", "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate",
                        "StudyTime", "SeriesTime", "AcquisitionTime", "ContentTime",
                        "AccessionNumber", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", # UIDs might be considered quasi-identifiers
                        # Add any other potentially identifying tags specific to your data/region
                    ]
                    # Consider using an ALLOW list for specific technical parameters instead for maximum safety.

                    dicom_meta: Dict[str, Any] = st.session_state.dicom_metadata
                    for tag_name, tag_value in dicom_meta.items():
                        # **Apply Filtering**
                        if tag_name in PHI_TAGS_TO_EXCLUDE:
                            logger.debug(f"Excluding potentially sensitive tag from report: {tag_name}")
                            continue # Skip this tag

                        display_value = "" # Initialize display value for the tag
                        try:
                            # Handle pydicom specific types first
                            if isinstance(tag_value, pydicom.dataelem.DataElement):
                                value = tag_value.value
                            else:
                                value = tag_value # Assume it's already the value if not DataElement

                            if isinstance(value, list): # Handle multi-valued tags
                                display_value = ", ".join(map(str, value))
                            elif isinstance(value, pydicom.uid.UID): # Format UIDs nicely
                                # Decide if UIDs should be included - they can sometimes be linked back
                                # display_value = f"{value.name} ({value})" # Option 1: Include name and UID
                                display_value = f"{value.name}" # Option 2: Include only name if UID itself is sensitive
                                # display_value = "[UID Omitted]" # Option 3: Omit entirely
                            elif isinstance(value, bytes): # Handle raw byte data
                                # Try decoding, provide placeholder if fails or too long
                                if len(value) > 128: # Limit binary data in report
                                     display_value = f"[Binary Data ({len(value)} bytes)]"
                                else:
                                     try:
                                         # Attempt common encodings or use replace
                                         display_value = value.decode('utf-8', errors='replace').strip()
                                         if not display_value.isprintable(): # Check if result is printable
                                             display_value = f"[Binary Data ({len(value)} bytes, non-printable)]"
                                     except Exception:
                                         display_value = f"[Binary Data ({len(value)} bytes, undecodable)]"
                            elif isinstance(value, (int, float, str)): # Handle standard types
                                 display_value = str(value).strip()
                            elif isinstance(value, pydicom.valuerep.PersonName):
                                 # PersonName might have been missed by simple tag name filtering if under a different tag
                                 logger.debug(f"Excluding PersonName object found in tag: {tag_name}")
                                 continue # Skip PHI
                            else: # Fallback for other types
                                display_value = str(value).strip()

                            # Only add if the value is meaningful (not empty or just whitespace) and not excessively long
                            if display_value and len(display_value) < 512: # Limit length per tag
                                filtered_meta_str_list.append(f"{tag_name}: {display_value}")
                            elif display_value:
                                logger.debug(f"Truncating long value for tag {tag_name} in report.")
                                filtered_meta_str_list.append(f"{tag_name}: {display_value[:509]}...")


                        except Exception as e:
                             logger.warning(f"Error processing DICOM tag '{tag_name}' (Value type: {type(tag_value)}) for report: {e}", exc_info=False) # Avoid logging raw value in warning
                             # Optionally include a marker for failed tags:
                             # filtered_meta_str_list.append(f"{tag_name}: [Error processing value]")


                    # Add the filtered metadata section to the report content
                    if filtered_meta_str_list:
                         outputs_for_report["DICOM Metadata (Filtered)"] = "\n".join(filtered_meta_str_list)
                         logger.info(f"Included {len(filtered_meta_str_list)} filtered DICOM tags in report data.")
                    else:
                         outputs_for_report["DICOM Metadata (Filtered)"] = "No non-excluded metadata found or processed."
                         logger.info("No non-excluded DICOM metadata included in report.")
                    # --- End of DICOM Handling ---

                # --- Generate PDF Bytes ---
                with st.spinner("🎨 Generating PDF document..."):
                    pdf_bytes = generate_pdf_report_bytes(
                        session_id=session_id, # Use validated session_id
                        image=img_with_roi_drawn, # Use the image with ROI drawn (or original)
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
                st.error("Cannot generate report: The image to include is not valid or available in 'display_image' state.")
                logger.error("PDF generation skipped: No valid PIL Image found in st.session_state.display_image.")

        # --- Unknown Action Handling ---
        else:
            st.warning(f"Action '{current_action}' is not recognized or implemented.")
            logger.warning(f"Attempted to handle unknown action: '{current_action}'")

    # --- General Exception Handling for Actions ---
    except Exception as e:
        # Display a user-friendly error message
        st.error(f"An unexpected error occurred while trying to perform action '{current_action}'. Please check the application logs or contact support if the issue persists.")
        # Log the detailed exception including the traceback
        logger.error(f"Unhandled exception during action '{current_action}': {e}", exc_info=True)

    # --- Post-Action Cleanup & UI Update ---
    finally:
        # ALWAYS reset the last action to prevent re-execution on page interactions or reruns
        st.session_state.last_action = None
        logger.debug(f"Action '{current_action}' handling finished. Resetting last_action state.")
        # Rerun the script to update the UI immediately reflecting the changes from the action
        # This ensures results (or errors) are displayed promptly.
        st.rerun()

# ------------------------------------------------------------------------------
# 8) Footer (Displayed regardless of action state)
# ------------------------------------------------------------------------------
st.markdown("---")
# CRITICAL: Keep this disclaimer prominent
st.caption("🩺 **RadVision AI Advanced** | ⚠️ **Disclaimer:** For Research, Informational & Educational Purposes Only. This tool is NOT intended for clinical diagnosis, patient treatment decisions, or replacing professional medical advice.")