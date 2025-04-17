# action_handlers.py
import streamlit as st
import logging
import os
import re
from PIL import Image, ImageDraw # Needed for report generation

# --- Import Backend Modules ---
try: from llm_interactions import run_initial_analysis, run_multimodal_qa, run_disease_analysis, estimate_ai_confidence, LLM_INTERACTIONS_AVAILABLE
except ImportError: LLM_INTERACTIONS_AVAILABLE = False
# --- UMLS Imports for Action Handler ---
try:
    from umls_utils import search_umls, UMLSAuthError, UMLSConcept, UMLS_APIKEY, UMLS_AVAILABLE, umls_utils
    from config import DEFAULT_UMLS_HITS # Import hit count from config
    if not UMLS_APIKEY: UMLS_AVAILABLE = False # Ensure availability reflects key presence
except ImportError:
    UMLS_AVAILABLE = False; UMLSAuthError = RuntimeError; search_umls = None; DEFAULT_UMLS_HITS = 5; UMLS_APIKEY = None; UMLSConcept=None
# --- End UMLS Imports ---
try: from report_utils import generate_pdf_report_bytes, REPORT_UTILS_AVAILABLE
except ImportError: REPORT_UTILS_AVAILABLE = False
try: from hf_models import query_hf_vqa_inference_api, HF_VQA_MODEL_ID, HF_MODELS_AVAILABLE
except ImportError: HF_MODELS_AVAILABLE = False; HF_VQA_MODEL_ID = "N/A"


logger = logging.getLogger(__name__)

# --- Helper for UMLS Auto-Enrichment ---
def _enrich_with_umls(text_to_search: str, state_key: str):
    """Searches UMLS for terms in text and updates session state."""
    if not text_to_search or not text_to_search.strip():
        logger.debug(f"Skipping UMLS enrichment for '{state_key}': Input text is empty.")
        st.session_state[state_key] = [] # Ensure it's an empty list
        return

    if UMLS_AVAILABLE and UMLS_APIKEY and search_umls:
        logger.info(f"Attempting UMLS auto-enrichment for state key: {state_key}")
        try:
            # No spinner here to keep main action fast, potentially add later if slow
            concepts = search_umls(
                text_to_search,
                UMLS_APIKEY,
                page_size=DEFAULT_UMLS_HITS
            )
            st.session_state[state_key] = concepts # Store list of UMLSConcept objects
            logger.info(f"UMLS enrichment for '{state_key}' found {len(concepts)} concepts.")
        except UMLSAuthError as e:
            logger.error(f"UMLS Auth Error during auto-enrichment for '{state_key}': {e}")
            st.warning(f"UMLS Authentication failed during background enrichment: {e}")
            st.session_state[state_key] = [] # Set empty on error
        except Exception as e:
            logger.error(f"UMLS Search Error during auto-enrichment for '{state_key}': {e}", exc_info=True)
            st.warning(f"UMLS Search failed during background enrichment: {e}")
            st.session_state[state_key] = [] # Set empty on error
    else:
        logger.debug(f"Skipping UMLS enrichment for '{state_key}': UMLS unavailable or API key missing.")
        st.session_state[state_key] = [] # Ensure empty if unavailable

# --- Main Action Handler ---
def handle_action(action: str):
    """Handles the logic for actions triggered by sidebar buttons."""
    logger.info(f"Handling action: '{action}'")

    # Prerequisite Checks (keep existing logic)
    req_img = action not in ["generate_report_data", "umls_search"] # UMLS manual search doesn't req image
    req_llm = action in ["analyze", "ask", "disease", "confidence"]
    req_rpt = (action == "generate_report_data")
    req_umls_manual = (action == "umls_search") # Separate check for manual lookup
    valid = True
    if req_img and not isinstance(st.session_state.get("processed_image"), Image.Image): st.error(f"No valid image for {action}."); valid = False
    if not st.session_state.get("session_id"): st.error("No session ID."); valid = False
    if req_llm and not LLM_INTERACTIONS_AVAILABLE: st.error(f"AI module unavailable for {action}."); valid = False
    if req_rpt and not REPORT_UTILS_AVAILABLE: st.error(f"Report module unavailable."); valid = False
    if req_umls_manual and not UMLS_AVAILABLE: st.error(f"UMLS module/key unavailable for lookup."); valid = False # Check for manual lookup

    if valid:
        img = st.session_state.processed_image
        roi = st.session_state.roi_coords
        hist = st.session_state.history
        try:
            if action == "analyze":
                st.info("🔬 Analyzing...");
                with st.spinner("AI analyzing..."): res = run_initial_analysis(img, roi)
                st.session_state.initial_analysis = res
                st.session_state.qa_answer = ""; st.session_state.disease_analysis = ""
                # --- Auto-enrichment Call ---
                _enrich_with_umls(res, "initial_analysis_umls")
                # --- End Enrichment ---
                logger.info("Analyze OK."); st.success("Analysis complete!")

            elif action == "ask":
                q = st.session_state.question_input_widget.strip()
                if not q: st.warning("Question empty.")
                else:
                    st.info(f"❓ Asking: '{q}'..."); st.session_state.qa_answer = "";
                    with st.spinner("AI thinking..."): ans, ok = run_multimodal_qa(img=img, question=q, history=hist, roi=roi)
                    if ok:
                        st.session_state.qa_answer = ans
                        hist.append(("User", q)); hist.append(("AI", ans))
                        # --- Auto-enrichment Call ---
                        _enrich_with_umls(ans, "qa_umls")
                        # --- End Enrichment ---
                        logger.info("Ask OK."); st.success("AI answered!")
                    else: # Primary Fail + Fallback
                        err = f"Primary AI fail: {ans}"; st.session_state.qa_answer = err; st.error(err); logger.error(f"Ask fail: {ans}")
                        st.session_state.qa_umls = [] # Clear UMLS if answer failed
                        tok=os.getenv("HF_API_TOKEN")
                        if HF_MODELS_AVAILABLE and tok:
                            st.info(f"Trying fallback: {HF_VQA_MODEL_ID}");
                            with st.spinner("Fallback..."): fb_ans, fb_ok = query_hf_vqa_inference_api(img=img, question=q, roi=roi)
                            if fb_ok:
                                fb_disp=f"**[Fallback]**\n{fb_ans}"; st.session_state.qa_answer+=f"\n\n{fb_disp}"; hist.append(("[FB] User",q)); hist.append(("[FB] AI", fb_disp))
                                # --- Auto-enrichment for Fallback ---
                                _enrich_with_umls(fb_ans, "qa_umls") # Enrich based on fallback answer
                                # --- End Enrichment ---
                                logger.info("Fallback OK."); st.success("Fallback answered.")
                            else: fb_err=f"[FB Error]: {fb_ans}"; st.session_state.qa_answer+=f"\n\n{fb_err}"; logger.error(f"Fallback fail: {fb_ans}"); st.error(fb_err)
                        elif HF_MODELS_AVAILABLE: st.session_state.qa_answer+="\n\n[FB Skip: Token missing]"; st.warning("HF Token missing.")
                        else: st.session_state.qa_answer+="\n\n[FB Unavailable]"; st.warning("Fallback unavailable.")

            elif action == "disease":
                d = st.session_state.disease_select_widget
                if not d: st.warning("No condition selected.")
                else:
                    st.info(f"🩺 Analyzing for '{d}'...");
                    with st.spinner(f"Analyzing {d}..."): res = run_disease_analysis(img, d, roi)
                    st.session_state.disease_analysis = res
                    st.session_state.qa_answer = ""
                    # --- Auto-enrichment Call ---
                    _enrich_with_umls(res, "disease_umls")
                    # --- End Enrichment ---
                    logger.info(f"Disease '{d}' OK."); st.success(f"Analysis for '{d}' complete!")

            # --- UMLS Manual Search Action (No change needed here, handled in main_page_ui) ---
            elif action == "umls_search":
                # This action primarily triggers the UI in main_page_ui to perform the search
                # No backend logic needed here beyond triggering the rerun (which happens anyway)
                # We'll keep the state clear call here for clarity, although UI handles display
                st.session_state.umls_results = None
                st.session_state.umls_error = None
                logger.debug("UMLS manual search action triggered - UI will handle.")
                # NOTE: Rerun at the end of handle_action() will update the UI

            elif action == "confidence":
                # (Keep existing confidence logic)
                if not (hist or st.session_state.get("initial_analysis") or st.session_state.get("disease_analysis")): st.warning("No analysis for confidence.")
                else:
                    st.info("📊 Estimating confidence...");
                    with st.spinner("Calculating..."): res=estimate_ai_confidence(img=img,history=hist,initial_analysis=st.session_state.get("initial_analysis"),disease_analysis=st.session_state.get("disease_analysis"),roi=roi)
                    st.session_state.confidence_score=res; logger.info("Confidence OK."); st.success("Confidence estimated!")

            elif action == "generate_report_data":
                # (Keep existing report generation logic)
                st.info("📄 Generating report..."); st.session_state.pdf_report_bytes=None; img_rep=st.session_state.get("display_image")
                if not isinstance(img_rep, Image.Image): st.error("No valid image for report."); logger.error("Report fail: no image.")
                else:
                    pdf_img = img_rep.copy().convert("RGB")
                    if roi and ImageDraw:
                        try: draw=ImageDraw.Draw(pdf_img); x0,y0,x1,y1=roi['left'],roi['top'],roi['left']+roi['width'],roi['top']+roi['height']; draw.rectangle([x0,y0,x1,y1],outline="red",width=max(2,int(min(pdf_img.size)*0.004))); logger.info("Drew ROI for PDF.")
                        except Exception as e: logger.error(f"ROI draw err: {e}"); st.warning("Could not draw ROI.")
                    hist_fmt="No Q&A.";
                    if hist: hist_fmt="\n\n".join([f"{qt}: {re.sub('<[^<]+?>','',str(m))}" for qt, m in hist])
                    rep_data={"Session ID":st.session_state.get("session_id"), "Image Filename":(st.session_state.get("uploaded_file_info", "N/A")).split('-')[0], "Initial Analysis":st.session_state.get("initial_analysis", "N/P"), "Conversation History":hist_fmt, "Condition Analysis":st.session_state.get("disease_analysis", "N/P"), "AI Confidence":st.session_state.get("confidence_score", "N/P")}
                    if st.session_state.get("is_dicom") and st.session_state.get("dicom_metadata"): meta_keys=['PatientName','PatientID','StudyDate','Modality','StudyDescription']; meta_sum={k:st.session_state.dicom_metadata.get(k,'N/A') for k in meta_keys if st.session_state.dicom_metadata.get(k)}; if meta_sum: rep_data["DICOM Summary"]="\n".join([f"{k}: {v}" for k,v in meta_sum.items()])
                    with st.spinner("Generating PDF..."): pdf_bytes=generate_pdf_report_bytes(session_id=st.session_state.get("session_id"), image=pdf_img, analysis_outputs=rep_data, dicom_metadata=st.session_state.get("dicom_metadata") if st.session_state.get("is_dicom") else None)
                    if pdf_bytes: st.session_state.pdf_report_bytes=pdf_bytes; st.success("PDF ready!"); logger.info("PDF OK."); st.balloons()
                    else: st.error("PDF generation failed."); logger.error("PDF gen fail.")
            else:
                st.warning(f"Unknown action '{action}'.")

        except Exception as e:
            st.error(f"Error during '{action}': {e}")
            logger.critical(f"Action '{action}' error: {e}", exc_info=True)
        finally:
            # Ensure action trigger is always reset
            if st.session_state.get("last_action") == action:
                 st.session_state.last_action = None
            logger.debug(f"Action '{action}' handling potentially complete.")
            st.rerun() # Rerun AFTER action handling to show results/updates
    else:
        # Action prerequisites not met
        st.session_state.last_action = None # Reset trigger
        logger.warning(f"Action '{action}' cancelled (invalid prerequisites).")
        # No rerun needed, error already shown