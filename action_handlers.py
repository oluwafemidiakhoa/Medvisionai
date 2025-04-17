# action_handlers.py
import streamlit as st
import logging
import os
import re
from PIL import Image, ImageDraw # Needed for report generation

# --- Import Backend Modules ---
# Import functions and check availability flags
try: from llm_interactions import run_initial_analysis, run_multimodal_qa, run_disease_analysis, estimate_ai_confidence, LLM_INTERACTIONS_AVAILABLE
except ImportError: LLM_INTERACTIONS_AVAILABLE = False
try: from umls_utils import search_umls, UMLSAuthError, UMLSConcept, UMLS_APIKEY, UMLS_AVAILABLE, umls_utils # Need utils for search_umls call
except ImportError: UMLS_AVAILABLE = False; UMLSAuthError = RuntimeError
try: from report_utils import generate_pdf_report_bytes, REPORT_UTILS_AVAILABLE
except ImportError: REPORT_UTILS_AVAILABLE = False
try: from hf_models import query_hf_vqa_inference_api, HF_VQA_MODEL_ID, HF_MODELS_AVAILABLE
except ImportError: HF_MODELS_AVAILABLE = False; HF_VQA_MODEL_ID = "N/A"


logger = logging.getLogger(__name__)

def handle_action(action: str):
    """Handles the logic for actions triggered by sidebar buttons."""
    logger.info(f"Handling action: '{action}'")

    # --- Prerequisite Checks ---
    req_img = action not in ["generate_report_data", "umls_search"]
    req_llm = action in ["analyze", "ask", "disease", "confidence"]
    req_rpt = (action == "generate_report_data")
    req_umls = (action == "umls_search")
    valid = True

    if req_img and not isinstance(st.session_state.get("processed_image"), Image.Image):
        st.error(f"Cannot perform '{action}': No valid image loaded."); valid = False
    if not st.session_state.get("session_id"):
        st.error("Critical Error: No session ID."); valid = False
    if req_llm and not LLM_INTERACTIONS_AVAILABLE:
        st.error(f"Cannot perform '{action}': AI module unavailable."); valid = False
    if req_rpt and not REPORT_UTILS_AVAILABLE:
        st.error(f"Cannot perform '{action}': Report module unavailable."); valid = False
    if req_umls and not UMLS_AVAILABLE:
        st.error(f"Cannot perform '{action}': UMLS module/key unavailable."); valid = False

    # --- Execute Action ---
    if valid:
        img = st.session_state.processed_image
        roi = st.session_state.roi_coords
        hist = st.session_state.history
        try:
            if action == "analyze":
                st.info("🔬 Analyzing...");
                with st.spinner("AI analyzing..."): res = run_initial_analysis(img, roi)
                st.session_state.initial_analysis = res; st.session_state.qa_answer = ""; st.session_state.disease_analysis = ""; logger.info("Analyze OK."); st.success("Analysis complete!")

            elif action == "ask":
                q = st.session_state.question_input_widget.strip()
                if not q: st.warning("Question empty.")
                else:
                    st.info(f"❓ Asking: '{q}'..."); st.session_state.qa_answer = "";
                    with st.spinner("AI thinking..."): ans, ok = run_multimodal_qa(img=img, question=q, history=hist, roi=roi)
                    if ok: st.session_state.qa_answer = ans; hist.append(("User", q)); hist.append(("AI", ans)); logger.info("Ask OK."); st.success("AI answered!")
                    else: # Primary Fail + Fallback
                        err = f"Primary AI fail: {ans}"; st.session_state.qa_answer = err; st.error(err); logger.error(f"Ask fail: {ans}")
                        tok=os.getenv("HF_API_TOKEN")
                        if HF_MODELS_AVAILABLE and tok:
                            st.info(f"Trying fallback: {HF_VQA_MODEL_ID}");
                            with st.spinner("Fallback..."): fb_ans, fb_ok = query_hf_vqa_inference_api(img=img, question=q, roi=roi)
                            if fb_ok: fb_disp=f"**[Fallback]**\n{fb_ans}"; st.session_state.qa_answer+=f"\n\n{fb_disp}"; hist.append(("[FB] User",q)); hist.append(("[FB] AI", fb_disp)); logger.info("Fallback OK."); st.success("Fallback answered.")
                            else: fb_err=f"[FB Error]: {fb_ans}"; st.session_state.qa_answer+=f"\n\n{fb_err}"; logger.error(f"Fallback fail: {fb_ans}"); st.error(fb_err)
                        elif HF_MODELS_AVAILABLE: st.session_state.qa_answer+="\n\n[FB Skip: Token missing]"; st.warning("HF Token missing.")
                        else: st.session_state.qa_answer+="\n\n[FB Unavailable]"; st.warning("Fallback unavailable.")

            elif action == "disease":
                d = st.session_state.disease_select_widget
                if not d: st.warning("No condition selected.")
                else:
                    st.info(f"🩺 Analyzing for '{d}'...");
                    with st.spinner(f"Analyzing {d}..."): res = run_disease_analysis(img, d, roi)
                    st.session_state.disease_analysis = res; st.session_state.qa_answer = ""; logger.info(f"Disease '{d}' OK."); st.success(f"Analysis for '{d}' complete!")

            elif action == "umls_search":
                t = st.session_state.get("umls_search_term","").strip()
                st.session_state.umls_results=None; st.session_state.umls_error=None
                if not t: st.warning("UMLS term empty.")
                else:
                    st.info(f"🔎 UMLS: '{t}'...");
                    with st.spinner("Querying UMLS..."):
                        try:
                            # Ensure umls_utils is imported and usable
                            if umls_utils:
                                results = umls_utils.search_umls(t, UMLS_APIKEY) # Call search from utils
                                st.session_state.umls_results = results; logger.info(f"UMLS OK ({len(results)})."); st.success(f"Found {len(results)} concepts.")
                            else: raise RuntimeError("umls_utils module not loaded correctly")
                        except UMLSAuthError as e: err=f"UMLS Auth Fail: {e}"; st.error(err); logger.error(err); st.session_state.umls_error=f"Auth: {e}"
                        except RuntimeError as e: err=f"UMLS Search Fail: {e}"; st.error(err); logger.error(err,exc_info=True); st.session_state.umls_error=f"Search: {e}"
                        except Exception as e: err=f"UMLS Unexpected: {e}"; st.error(err); logger.critical(err,exc_info=True); st.session_state.umls_error=f"Unexpected: {e}"

            elif action == "confidence":
                if not (hist or st.session_state.get("initial_analysis") or st.session_state.get("disease_analysis")): st.warning("No analysis for confidence.")
                else:
                    st.info("📊 Estimating confidence...");
                    with st.spinner("Calculating..."): res=estimate_ai_confidence(img=img,history=hist,initial_analysis=st.session_state.get("initial_analysis"),disease_analysis=st.session_state.get("disease_analysis"),roi=roi)
                    st.session_state.confidence_score=res; logger.info("Confidence OK."); st.success("Confidence estimated!")

            elif action == "generate_report_data":
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
                    if st.session_state.get("is_dicom") and st.session_state.get("dicom_metadata"):
                        meta_keys=['PatientName','PatientID','StudyDate','Modality','StudyDescription']; meta_sum={k:st.session_state.dicom_metadata.get(k,'N/A') for k in meta_keys if st.session_state.dicom_metadata.get(k)}; if meta_sum: rep_data["DICOM Summary"]="\n".join([f"{k}: {v}" for k,v in meta_sum.items()])
                    with st.spinner("Generating PDF..."): pdf_bytes=generate_pdf_report_bytes(session_id=st.session_state.get("session_id"), image=pdf_img, analysis_outputs=rep_data, dicom_metadata=st.session_state.get("dicom_metadata") if st.session_state.get("is_dicom") else None)
                    if pdf_bytes: st.session_state.pdf_report_bytes=pdf_bytes; st.success("PDF ready!"); logger.info("PDF OK."); st.balloons()
                    else: st.error("PDF generation failed."); logger.error("PDF gen fail.")

            else: st.warning(f"Unknown action '{action}'.")

        except Exception as e:
            st.error(f"Error during '{action}': {e}")
            logger.critical(f"Action '{action}' error: {e}", exc_info=True)
        finally:
            st.session_state.last_action = None # Reset action trigger
            logger.debug(f"Action '{action}' handling complete.")
            st.rerun() # Rerun to show results/updates
    else:
        # Action prerequisites not met
        st.session_state.last_action = None # Reset trigger
        logger.warning(f"Action '{action}' cancelled (invalid prerequisites).")
        # No rerun needed, error already shown