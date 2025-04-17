# main_page_ui.py
import streamlit as st
import logging
import os
from PIL import Image, ImageDraw

# --- Import specific UI components needed ---
# Define fallbacks if imports fail
try: from ui_components import display_dicom_metadata
except ImportError: display_dicom_metadata = None; logging.warning("display_dicom_metadata not imported.") # Use logging directly
try: from ui_components import display_umls_concepts
except ImportError: display_umls_concepts = None; logging.warning("display_umls_concepts not imported.")
# --- End UI Component Imports ---

try: from streamlit_drawable_canvas import st_canvas; DRAWABLE_CANVAS_AVAILABLE = True
except ImportError: st_canvas = None; DRAWABLE_CANVAS_AVAILABLE = False

try: from translation_models import TRANSLATION_AVAILABLE, LANGUAGE_CODES, AUTO_DETECT_INDICATOR, translate
except ImportError: TRANSLATION_AVAILABLE = False; LANGUAGE_CODES={"En":"en"}; AUTO_DETECT_INDICATOR="Auto"; translate=None

try: from umls_utils import search_umls, UMLSAuthError, UMLSConcept, UMLS_APIKEY, UMLS_AVAILABLE
except ImportError: UMLS_AVAILABLE = False; search_umls=None; UMLSAuthError=RuntimeError; UMLSConcept=None; UMLS_APIKEY=None
try: from config import DEFAULT_UMLS_HITS
except ImportError: DEFAULT_UMLS_HITS = 5

# --- Define format_translation (either imported or fallback) ---
try:
    from session_state import format_translation
    logging.info("Imported format_translation from session_state.")
except ImportError:
    logging.warning("Could not import format_translation from session_state. Using basic fallback.")
    # Define fallback function HERE, outside the except block's direct execution
    def format_translation(t: Optional[str]) -> str:
        """Basic fallback for formatting translation."""
        return str(t) if t else ""
# --- End format_translation definition ---


logger = logging.getLogger(__name__) # Initialize logger after potential logging warnings

def _render_image_viewer(column):
    # (Keep existing _render_image_viewer logic - No changes needed here)
    with column:
        st.subheader("🖼️ Image Viewer")
        display_img = st.session_state.get("display_image")
        if isinstance(display_img, Image.Image):
            logger.debug("Rendering image viewer...")
            if DRAWABLE_CANVAS_AVAILABLE and st_canvas:
                st.caption("Draw rectangle for ROI.")
                MAX_W, MAX_H = 600, 500; img_w, img_h = display_img.size
                if img_w <= 0 or img_h <= 0: st.warning("Invalid image size.")
                else:
                    ar = img_w / img_h; can_w = min(img_w, MAX_W); can_h = int(can_w / ar)
                    if can_h > MAX_H: can_h = MAX_H; can_w = int(can_h * ar)
                    can_w, can_h = max(can_w, 150), max(can_h, 150); logger.debug(f"Canvas: {can_w}x{can_h}")
                    canvas_result = st_canvas(fill_color="rgba(255,165,0,0.2)", stroke_width=2, stroke_color="rgba(239,83,80,0.8)", background_image=display_img, update_streamlit=True, height=can_h, width=can_w, drawing_mode="rect", initial_drawing=st.session_state.get("canvas_drawing"), key="drawable_canvas")
                    if canvas_result.json_data and canvas_result.json_data.get("objects"):
                        if canvas_result.json_data["objects"]:
                            lobj = canvas_result.json_data["objects"][-1]
                            if lobj["type"] == "rect":
                                cl, ct = int(lobj["left"]), int(lobj["top"]); cws = int(lobj["width"]*lobj.get("scaleX",1)); chs = int(lobj["height"]*lobj.get("scaleY",1))
                                scx, scy = img_w/can_w, img_h/can_h; ol = max(0, int(cl*scx)); ot = max(0, int(ct*scy))
                                ow = max(1, min(img_w-ol, int(cws*scx))); oh = max(1, min(img_h-ot, int(chs*scy)))
                                new_roi = {"left":ol, "top":ot, "width":ow, "height":oh}
                                if st.session_state.get("roi_coords") != new_roi: st.session_state.roi_coords=new_roi; st.session_state.canvas_drawing=canvas_result.json_data; logger.info(f"ROI set: {new_roi}"); st.info(f"ROI Set: ({ol},{ot}), Size: {ow}x{oh}", icon="🎯")
            else: st.image(display_img, caption="Preview", use_container_width=True)
            if st.session_state.get("roi_coords"): r = st.session_state.roi_coords; st.caption(f"ROI: ({r['left']},{r['top']})-W:{r['width']},H:{r['height']}")
            st.markdown("---")
            if st.session_state.get("is_dicom") and st.session_state.get("dicom_metadata"):
                if display_dicom_metadata: display_dicom_metadata(st.session_state.dicom_metadata)
                else: st.json(st.session_state.dicom_metadata)
        elif st.session_state.get("uploaded_file_info") and not display_img: st.error("❌ Image preview unavailable (processing failed).")
        else: st.info("⬅️ Upload an image or use Demo Mode.")


def _render_results_tabs(column):
    """Renders the analysis results tabs, including UMLS sections."""
    with column:
        st.subheader("📊 Analysis & Results")
        tab_titles = ["🔬 Initial", "💬 Q&A", "🩺 Condition", "📈 Confidence", "🌐 Translate", "🧬 UMLS Lookup"]
        tabs = st.tabs(tab_titles)

        # --- Initial Analysis Tab (with UMLS) ---
        with tabs[0]:
            st.text_area("Findings", value=st.session_state.get("initial_analysis", "Run Initial Analysis."), height=400, disabled=True, key="init_disp")
            if display_umls_concepts and st.session_state.get("initial_analysis_umls"):
                st.markdown("---"); st.markdown("**Auto‑Enriched UMLS Concepts:**")
                display_umls_concepts(st.session_state.initial_analysis_umls)
            elif st.session_state.get("initial_analysis_umls"): # Fallback
                st.markdown("---"); st.markdown("**Auto‑Enriched UMLS Concepts (Raw):**")
                try: st.json([c.to_dict() for c in st.session_state.initial_analysis_umls])
                except AttributeError: st.json(st.session_state.initial_analysis_umls)

        # --- Q&A Tab (with UMLS) ---
        with tabs[1]:
            st.text_area("Latest Answer", value=st.session_state.get("qa_answer", "Ask a question."), height=150, disabled=True, key="qa_disp")
            if display_umls_concepts and st.session_state.get("qa_umls"):
                st.markdown("---"); st.markdown("**Auto‑Enriched UMLS Concepts (Latest Answer):**")
                display_umls_concepts(st.session_state.qa_umls)
            elif st.session_state.get("qa_umls"): # Fallback
                 st.markdown("---"); st.markdown("**Auto‑Enriched UMLS Concepts (Raw):**")
                 try: st.json([c.to_dict() for c in st.session_state.qa_umls])
                 except AttributeError: st.json(st.session_state.qa_umls)
            # Q&A History display
            st.markdown("---"); st.subheader("History")
            history = st.session_state.get("history", [])
            if history:
                 for i, (qt, m) in enumerate(reversed(history)): pfx = "👤:" if qt.lower().startswith("user") else "🤖:" if qt.lower().startswith("ai") else "ℹ️:" if qt.lower().startswith("sys") else f"**{qt}:**"; unsafe = "ai" in qt.lower(); st.markdown(f"{pfx} {m}", unsafe_allow_html=unsafe);
                 if i < len(history)-1: st.markdown("---")
            else: st.caption("No Q&A yet.")

        # --- Condition Tab (with UMLS) ---
        with tabs[2]:
            st.text_area("Condition Analysis", value=st.session_state.get("disease_analysis", "Run Condition Analysis."), height=400, disabled=True, key="dis_disp")
            if display_umls_concepts and st.session_state.get("disease_umls"):
                st.markdown("---"); st.markdown("**Auto‑Enriched UMLS Concepts:**")
                display_umls_concepts(st.session_state.disease_umls)
            elif st.session_state.get("disease_umls"): # Fallback
                 st.markdown("---"); st.markdown("**Auto‑Enriched UMLS Concepts (Raw):**")
                 try: st.json([c.to_dict() for c in st.session_state.disease_umls])
                 except AttributeError: st.json(st.session_state.disease_umls)

        # --- Confidence Tab ---
        with tabs[3]:
             st.text_area("Confidence", value=st.session_state.get("confidence_score", "Run Confidence Estimation."), height=400, disabled=True, key="conf_disp")

        # --- Translation Tab ---
        with tabs[4]:
            st.subheader("🌐 Translate")
            if not TRANSLATION_AVAILABLE: st.warning("Translation unavailable.")
            else:
                st.caption("Select text, languages, translate.")
                txt_opts = {"Initial": st.session_state.get("initial_analysis"), "Q&A": st.session_state.get("qa_answer"), "Condition": st.session_state.get("disease_analysis"), "Confidence": st.session_state.get("confidence_score"), "(Custom)": ""}
                avail_opts = {lbl: txt for lbl, txt in txt_opts.items() if (txt and txt.strip()) or lbl == "(Custom)"}
                if not avail_opts: st.info("No text to translate.")
                else:
                    sel_lbl = st.selectbox("Translate:", list(avail_opts.keys()), 0, key="trans_sel"); txt_to_trans = avail_opts.get(sel_lbl, "")
                    if sel_lbl == "(Custom)": txt_to_trans = st.text_area("Custom text:", "", 100, key="trans_cust_in")
                    else: st.text_area("Selected:", txt_to_trans, 100, disabled=True, key="trans_sel_disp")
                    cl1, cl2 = st.columns(2)
                    with cl1: src_opts = [AUTO_DETECT_INDICATOR] + sorted(LANGUAGE_CODES.keys()); src_lang = st.selectbox("From:", src_opts, 0, key="trans_src")
                    with cl2: tgt_opts = sorted(LANGUAGE_CODES.keys()); tgt_idx=0; common=["English","Spanish"];
                    for i, l in enumerate(tgt_opts):
                        if l in common: tgt_idx=i; break
                    tgt_lang = st.selectbox("To:", tgt_opts, tgt_idx, key="trans_tgt")
                    if st.button("🔄 Translate", key="trans_btn"):
                        st.session_state.translation_result=None; st.session_state.translation_error=None
                        if not txt_to_trans.strip(): st.warning("No text."); st.session_state.translation_error="Empty."
                        elif src_lang == tgt_lang and src_lang != AUTO_DETECT_INDICATOR: st.info("Same language."); st.session_state.translation_result=txt_to_trans
                        else:
                            with st.spinner("Translating..."):
                                try:
                                    if translate: out=translate(text=txt_to_trans, target_language=tgt_lang, source_language=src_lang) # Adjusted signature assumption
                                    else: raise RuntimeError("Translate fn missing")
                                    if out is not None: st.session_state.translation_result=out; st.success("OK!")
                                    else: st.error("No result."); st.session_state.translation_error="None."
                                except Exception as e: st.error(f"Fail:{e}"); st.session_state.translation_error=str(e)
                    if st.session_state.get("translation_result"): st.text_area("Result:", format_translation(st.session_state.translation_result), 200, key="trans_out") # Use format_translation here

        # --- UMLS Lookup Tab ---
        with tabs[5]:
            st.subheader("🧬 UMLS Manual Lookup")
            if not UMLS_AVAILABLE: st.warning("UMLS features unavailable.")
            elif not search_umls: st.error("UMLS search function unavailable.")
            else:
                term = st.text_input("Enter term to search UMLS:", value=st.session_state.get("umls_search_term", ""), key="umls_manual_term_input")
                st.session_state.umls_search_term = term
                if st.button("Search UMLS", key="umls_manual_search_btn"):
                    manual_search_term = term.strip(); st.session_state.umls_results = None; st.session_state.umls_error = None
                    if not manual_search_term: st.warning("Please enter lookup term.")
                    elif not UMLS_APIKEY: st.error("UMLS_APIKEY not set."); logger.error("Manual UMLS search: API key missing.")
                    else:
                        with st.spinner("Querying UMLS..."):
                            try: concepts = search_umls(manual_search_term, UMLS_APIKEY, page_size=DEFAULT_UMLS_HITS)
                            st.session_state.umls_results = concepts; logger.info(f"Manual UMLS found {len(concepts)} concepts.")
                            if not concepts: st.info("No concepts found.")
                            except UMLSAuthError as e: err=f"UMLS Auth Error: {e}"; st.error(err); st.session_state.umls_error = err
                            except Exception as e: err=f"UMLS Search Error: {e}"; st.error(err); logger.error(err, exc_info=True); st.session_state.umls_error = err
            # Display manual search results
            if st.session_state.get("umls_results") is not None:
                if display_umls_concepts: # Check if display function exists
                     st.markdown("---")
                     display_umls_concepts(st.session_state.umls_results)
                elif st.session_state.get("umls_results"): # Fallback
                     st.markdown("---")
                     try: st.json([c.to_dict() for c in st.session_state.umls_results])
                     except AttributeError: st.json(st.session_state.umls_results)


def render_main_content(col1, col2):
    """Renders the main page content within the provided columns."""
    _render_image_viewer(col1)
    _render_results_tabs(col2)