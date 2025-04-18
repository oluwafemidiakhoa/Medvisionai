# action_handlers.py

import streamlit as st
import logging
import os
import re
from PIL import Image, ImageDraw
from typing import Optional, Any, Tuple, List, Dict

# --- Import Backend Modules & Set Flags ---
try:
    from llm_interactions import (
        run_initial_analysis,
        run_multimodal_qa,
        run_disease_analysis,
        estimate_ai_confidence
    )
    LLM_INTERACTIONS_AVAILABLE = True
except ImportError:
    LLM_INTERACTIONS_AVAILABLE = False
    logging.error("llm_interactions not found.")

try:
    from umls_utils import (
        search_umls,
        UMLSAuthError,
        UMLSConcept
    )
    from config import DEFAULT_UMLS_HITS
    UMLS_APIKEY = os.getenv("UMLS_APIKEY", "")
    if not UMLS_APIKEY:
        raise ImportError("UMLS_APIKEY not set")
    UMLS_AVAILABLE = True
except ImportError as e:
    UMLS_AVAILABLE = False
    UMLSAuthError = RuntimeError
    search_umls = None
    DEFAULT_UMLS_HITS = 5
    UMLS_APIKEY = None
    UMLSConcept = None
    logging.warning(f"UMLS features disabled: {e}")

try:
    from report_utils import generate_pdf_report_bytes
    REPORT_UTILS_AVAILABLE = True
except ImportError:
    REPORT_UTILS_AVAILABLE = False
    generate_pdf_report_bytes = None
    logging.warning("report_utils not found.")

try:
    from hf_models import query_hf_vqa_inference_api, HF_VQA_MODEL_ID
    HF_MODELS_AVAILABLE = True
except ImportError:
    HF_MODELS_AVAILABLE = False
    HF_VQA_MODEL_ID = "N/A"
    query_hf_vqa_inference_api = None
    logging.warning("hf_models not found.")

logger = logging.getLogger(__name__)


# --- Helper for UMLS Auto-Enrichment ---
def _enrich_with_umls(text_to_search: str, state_key: str):
    """
    Run a UMLS search on text_to_search and store results in st.session_state[state_key].
    """
    if not text_to_search or not text_to_search.strip():
        logger.debug(f"Skip UMLS enrich '{state_key}': empty text.")
        st.session_state[state_key] = []
        return

    if UMLS_AVAILABLE and UMLS_APIKEY and search_umls:
        logger.info(f"Attempting UMLS enrichment for: {state_key}")
        try:
            concepts: List[UMLSConcept] = search_umls(
                text_to_search, UMLS_APIKEY, page_size=DEFAULT_UMLS_HITS
            )
            st.session_state[state_key] = concepts
            logger.info(f"UMLS enrich OK for '{state_key}' ({len(concepts)} found).")
        except UMLSAuthError as e:
            logger.error(f"UMLS Auth Err enrich '{state_key}': {e}")
            st.warning(f"UMLS Auth failed: {e}")
            st.session_state[state_key] = []
        except Exception as e:
            logger.error(f"UMLS Search Err enrich '{state_key}': {e}", exc_info=True)
            st.warning(f"UMLS Search failed: {e}")
            st.session_state[state_key] = []
    else:
        logger.debug(f"Skip UMLS enrich '{state_key}': unavailable.")
        st.session_state[state_key] = []


# --- Main Action Handler ---
def handle_action(action: str):
    logger.info(f"Handling action: '{action}'")

    # Determine prerequisites
    req_img = action not in ["generate_report_data", "umls_search"]
    req_llm = action in ["analyze", "ask", "disease", "confidence"]
    req_rpt = action == "generate_report_data"
    req_umls_manual = action == "umls_search"

    valid = True
    if req_img and not isinstance(st.session_state.get("processed_image"), Image.Image):
        st.error(f"No valid image for '{action}'.")
        valid = False
    if not st.session_state.get("session_id"):
        st.error("No session ID.")
        valid = False
    if req_llm and not LLM_INTERACTIONS_AVAILABLE:
        st.error(f"AI module unavailable for '{action}'.")
        valid = False
    if req_rpt and not REPORT_UTILS_AVAILABLE:
        st.error("Report module unavailable.")
        valid = False
    if req_umls_manual and not UMLS_AVAILABLE:
        st.error("UMLS module/key unavailable.")
        valid = False

    if not valid:
        st.session_state.last_action = None
        logger.warning(f"Action '{action}' cancelled (invalid).")
        return

    img = st.session_state.get("processed_image")
    roi = st.session_state.get("roi_coords")
    hist = st.session_state.get("history", [])

    try:
        if action == "analyze":
            st.info("🔬 Performing initial analysis…")
            with st.spinner("AI analyzing…"):
                result = run_initial_analysis(img, roi)
            st.session_state.initial_analysis = result
            st.session_state.qa_answer = ""
            st.session_state.disease_analysis = ""
            _enrich_with_umls(result, "initial_analysis_umls")
            st.success("Initial analysis complete!")

        elif action == "ask":
            question = st.session_state.question_input_widget.strip()
            if not question:
                st.warning("Please enter a question.")
            else:
                st.info(f"❓ Asking AI: '{question}'…")
                st.session_state.qa_answer = ""
                with st.spinner("AI thinking…"):
                    answer, ok = run_multimodal_qa(
                        img=img,
                        question=question,
                        history=hist,
                        roi=roi
                    )
                if ok:
                    st.session_state.qa_answer = answer
                    hist.extend([("User", question), ("AI", answer)])
                    _enrich_with_umls(answer, "qa_umls")
                    st.success("AI answered your question!")
                else:
                    # Primary AI failed
                    st.session_state.qa_answer = f"Primary AI error: {answer}"
                    st.error(st.session_state.qa_answer)
                    # Try fallback if available
                    token = os.getenv("HF_API_TOKEN", "")
                    if HF_MODELS_AVAILABLE and query_hf_vqa_inference_api and token:
                        st.info(f"Trying fallback model: {HF_VQA_MODEL_ID}")
                        with st.spinner("Fallback AI…"):
                            fb_answer, fb_ok = query_hf_vqa_inference_api(
                                img=img, question=question, roi=roi
                            )
                        if fb_ok:
                            fb_display = f"**[Fallback {HF_VQA_MODEL_ID}]**\n\n{fb_answer}"
                            st.session_state.qa_answer += f"\n\n{fb_display}"
                            hist.extend([("[FB] User", question), ("[FB] AI", fb_display)])
                            _enrich_with_umls(fb_answer, "qa_umls")
                            st.success("Fallback AI answered!")
                        else:
                            st.error(f"Fallback AI failed: {fb_answer}")
                    else:
                        st.warning("Fallback AI unavailable or missing token.")

        elif action == "disease":
            condition = st.session_state.disease_select_widget
            if not condition:
                st.warning("Please select a condition first.")
            else:
                st.info(f"🩺 Performing condition-specific analysis for '{condition}'…")
                with st.spinner("AI analyzing condition…"):
                    result = run_disease_analysis(img, condition, roi)
                st.session_state.disease_analysis = result
                st.session_state.qa_answer = ""
                _enrich_with_umls(result, "disease_umls")
                st.success(f"Analysis for '{condition}' complete!")

        elif action == "umls_search":
            # --- Manual UMLS Lookup ---
            term = st.session_state.get("lookup_term", "").strip()
            if not term:
                st.warning("🔍 Please enter a term to look up.")
            else:
                api_key = os.getenv("UMLS_APIKEY", "")
                if not api_key:
                    st.error("🔑 UMLS_APIKEY not set. Add it to your environment or Secrets.")
                else:
                    with st.spinner(f"🔍 Searching UMLS for '{term}'…"):
                        try:
                            concepts = search_umls(term, api_key, page_size=DEFAULT_UMLS_HITS)
                            st.session_state.lookup_results = concepts
                            if not concepts:
                                st.info("No UMLS concepts found.")
                        except Exception as e:
                            st.error(f"⚠️ UMLS lookup failed: {e}")
                            logger.error(f"UMLS search error: {e}", exc_info=True)

        elif action == "confidence":
            if not (hist or st.session_state.initial_analysis or st.session_state.disease_analysis):
                st.warning("Perform an analysis first before estimating confidence.")
            else:
                st.info("📈 Estimating AI confidence…")
                with st.spinner("Calculating confidence…"):
                    res = estimate_ai_confidence(
                        img=img,
                        history=hist,
                        initial_analysis=st.session_state.initial_analysis,
                        disease_analysis=st.session_state.disease_analysis,
                        roi=roi
                    )
                st.session_state.confidence_score = res
                st.success("Confidence estimation complete!")

        elif action == "generate_report_data":
            if not generate_pdf_report_bytes:
                st.error("Report generation function unavailable.")
                return
            st.info("📄 Generating PDF report…")
            st.session_state.pdf_report_bytes = None
            display_img = st.session_state.get("display_image")
            if not isinstance(display_img, Image.Image):
                st.error("No valid image for report.")
            else:
                # Draw ROI on PDF copy if present
                pdf_img = display_img.copy().convert("RGB")
                if st.session_state.get("roi_coords"):
                    try:
                        draw = ImageDraw.Draw(pdf_img)
                        x0, y0 = roi["left"], roi["top"]
                        x1, y1 = x0 + roi["width"], y0 + roi["height"]
                        draw.rectangle(
                            [x0, y0, x1, y1],
                            outline="red",
                            width=max(2, int(min(pdf_img.size) * 0.004))
                        )
                    except Exception as e:
                        logger.error(f"Error drawing ROI on PDF: {e}", exc_info=True)
                        st.warning("Could not draw ROI on report image.")

                # Prepare report data
                hist_fmt = "No Q&A history."
                if hist:
                    hist_fmt = "\n\n".join(
                        f"{role}: {re.sub('<[^<]+?>', '', msg)}"
                        for role, msg in hist
                    )

                report_data: Dict[str, str] = {
                    "Session ID": st.session_state.session_id,
                    "Image Filename": (st.session_state.uploaded_file_info or "N/A").split("-")[0],
                    "Initial Analysis": st.session_state.initial_analysis or "Not Performed",
                    "Conversation History": hist_fmt,
                    "Condition Analysis": st.session_state.disease_analysis or "Not Performed",
                    "AI Confidence": st.session_state.confidence_score or "Not Performed",
                }

                # Include DICOM summary if available
                if st.session_state.get("is_dicom") and st.session_state.get("dicom_metadata"):
                    meta = st.session_state.dicom_metadata
                    keys = ["PatientName", "PatientID", "StudyDate", "Modality", "StudyDescription"]
                    summary = {
                        k: meta.get(k, "N/A")
                        for k in keys
                        if meta.get(k)
                    }
                    if summary:
                        report_data["DICOM Summary"] = "\n".join(f"{k}: {v}" for k, v in summary.items())

                with st.spinner("Generating PDF bytes…"):
                    pdf_bytes = generate_pdf_report_bytes(
                        session_id=st.session_state.session_id,
                        image=pdf_img,
                        analysis_outputs=report_data
                    )
                if pdf_bytes:
                    st.session_state.pdf_report_bytes = pdf_bytes
                    st.success("PDF report ready for download!")
                    st.balloons()
                else:
                    st.error("PDF generation failed.")

        else:
            st.warning(f"Unknown action '{action}'.")

    except Exception as e:
        st.error(f"Error during '{action}': {e}")
        logger.critical(f"Action '{action}' error: {e}", exc_info=True)

    finally:
        # Clear the last_action flag if it matches
        if st.session_state.get("last_action") == action:
            st.session_state.last_action = None
        st.rerun()
