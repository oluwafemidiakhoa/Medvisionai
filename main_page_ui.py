# main_page_ui.py
import streamlit as st
import logging
from PIL import Image, ImageDraw # Need PIL for type checking and drawing

# Import UI components and check availability
try: from ui_components import display_dicom_metadata, display_umls_concepts, UI_COMPONENTS_AVAILABLE
except ImportError: UI_COMPONENTS_AVAILABLE = False; display_dicom_metadata=None; display_umls_concepts=None

# Import canvas if available
try: from streamlit_drawable_canvas import st_canvas; DRAWABLE_CANVAS_AVAILABLE = True
except ImportError: st_canvas = None; DRAWABLE_CANVAS_AVAILABLE = False

# Import translation helpers if needed within tabs
try: from translation_models import TRANSLATION_AVAILABLE, LANGUAGE_CODES, AUTO_DETECT_INDICATOR, translate
except ImportError: TRANSLATION_AVAILABLE = False; LANGUAGE_CODES={"En":"en"}; AUTO_DETECT_INDICATOR="Auto"; translate=None

from session_state import format_translation # Import helper

logger = logging.getLogger(__name__)

def _render_image_viewer(column):
    """Renders the image viewer (canvas or static image) and metadata."""
    with column:
        st.subheader("🖼️ Image Viewer")
        display_img = st.session_state.get("display_image")

        if isinstance(display_img, Image.Image):
            logger.debug("Rendering image viewer...")
            if DRAWABLE_CANVAS_AVAILABLE and st_canvas:
                st.caption("Draw rectangle for ROI.")
                MAX_W, MAX_H = 600, 500
                img_w, img_h = display_img.size
                if img_w <= 0 or img_h <= 0: st.warning("Invalid image size.")
                else:
                    ar = img_w / img_h; can_w = min(img_w, MAX_W); can_h = int(can_w / ar)
                    if can_h > MAX_H: can_h = MAX_H; can_w = int(can_h * ar)
                    can_w, can_h = max(can_w, 150), max(can_h, 150)
                    logger.debug(f"Canvas dimensions: {can_w}x{can_h}")

                    canvas_result = st_canvas(
                        fill_color="rgba(255,165,0,0.2)", stroke_width=2, stroke_color="rgba(239,83,80,0.8)",
                        background_image=display_img, update_streamlit=True, height=can_h, width=can_w,
                        drawing_mode="rect", initial_drawing=st.session_state.get("canvas_drawing"), key="drawable_canvas"
                    )
                    # ROI Processing Logic
                    if canvas_result.json_data and canvas_result.json_data.get("objects"):
                        if canvas_result.json_data["objects"]:
                            lobj = canvas_result.json_data["objects"][-1]
                            if lobj["type"] == "rect":
                                cl, ct = int(lobj["left"]), int(lobj["top"]); cws = int(lobj["width"]*lobj.get("scaleX",1)); chs = int(lobj["height"]*lobj.get("scaleY",1))
                                scx, scy = img_w/can_w, img_h/can_h; ol = max(0, int(cl*scx)); ot = max(0, int(ct*scy))
                                ow = max(1, min(img_w-ol, int(cws*scx))); oh = max(1, min(img_h-ot, int(chs*scy)))
                                new_roi = {"left":ol, "top":ot, "width":ow, "height":oh}
                                if st.session_state.get("roi_coords") != new_roi:
                                    st.session_state.roi_coords=new_roi; st.session_state.canvas_drawing=canvas_result.json_data
                                    logger.info(f"ROI set: {new_roi}"); st.info(f"ROI Set: ({ol},{ot}), Size: {ow}x{oh}", icon="🎯")
            else: # Fallback display
                st.image(display_img, caption="Preview", use_container_width=True)

            # Display ROI info and metadata
            if st.session_state.get("roi_coords"):
                r = st.session_state.roi_coords; st.caption(f"ROI: ({r['left']},{r['top']})-W:{r['width']},H:{r['height']}")
            st.markdown("---")
            if st.session_state.get("is_dicom") and st.session_state.get("dicom_metadata"):
                if UI_COMPONENTS_AVAILABLE and display_dicom_metadata: display_dicom_metadata(st.session_state.dicom_metadata)
                else: st.json(st.session_state.dicom_metadata) # Simple fallback

        elif st.session_state.get("uploaded_file_info") and not display_img:
            st.error("❌ Image preview unavailable (processing failed).")
        else:
            st.info("⬅️ Upload an image or use Demo Mode.")

def _render_results_tabs(column):
    """Renders the analysis results tabs."""
    with column:
        st.subheader("📊 Analysis & Results")
        tabs = st.tabs(["🔬 Initial", "💬 Q&A", "🩺 Condition", "📚 UMLS", "📈 Confidence", "🌐 Translate"])

        with tabs[0]: # Initial Analysis
            st.text_area("Findings", value=st.session_state.get("initial_analysis", "Run Initial Analysis."), height=400, disabled=True, key="init_disp")
        with tabs[1]: # Q&A
            st.text_area("Latest Answer", value=st.session_state.get("qa_answer", "Ask a question."), height=150, disabled=True, key="qa_disp")
            st.markdown("---"); st.subheader("History")
            history = st.session_state.get("history", [])
            if history:
                for i, (qt, m) in enumerate(reversed(history)):
                    pfx = "👤:" if qt.lower().startswith("user") else "🤖:" if qt.lower().startswith("ai") else "ℹ️:" if qt.lower().startswith("sys") else f"**{qt}:**"
                    unsafe = "ai" in qt.lower()
                    st.markdown(f"{pfx} {m}", unsafe_allow_html=unsafe)
                    if i < len(history)-1: st.markdown("---")
            else: st.caption("No Q&A yet.")
        with tabs[2]: # Condition
            st.text_area("Condition Analysis", value=st.session_state.get("disease_analysis", "Run Condition Analysis."), height=400, disabled=True, key="dis_disp")
        with tabs[3]: # UMLS
            st.subheader("📚 UMLS Search")
            from config import UMLS_AVAILABLE # Check availability here
            if not UMLS_AVAILABLE: st.warning("UMLS unavailable.")
            else:
                term = st.text_input("Search term:", value=st.session_state.get("umls_search_term", ""), key="umls_in", placeholder="e.g., lung nodule")
                if st.button("🔎 Search UMLS", key="umls_btn"):
                    if term.strip(): st.session_state.last_action="umls_search"; st.session_state.umls_search_term=term.strip(); st.rerun()
                    else: st.warning("Enter search term.")
                if st.session_state.get("umls_error"): st.error(f"UMLS Error: {st.session_state.umls_error}")
                # Use UI component if available
                if UI_COMPONENTS_AVAILABLE and display_umls_concepts: display_umls_concepts(st.session_state.get("umls_results"))
                elif st.session_state.get("umls_results"): st.json(st.session_state.umls_results) # Simple fallback
                else: st.caption("No results to display.")
        with tabs[4]: # Confidence
            st.text_area("Confidence", value=st.session_state.get("confidence_score", "Run Confidence Estimation."), height=400, disabled=True, key="conf_disp")
        with tabs[5]: # Translation
            st.subheader("🌐 Translate")
            if not TRANSLATION_AVAILABLE: st.warning("Translation unavailable.")
            else:
                st.caption("Select text, languages, translate.")
                txt_opts = {"Initial": st.session_state.get("initial_analysis"), "Q&A": st.session_state.get("qa_answer"), "Condition": st.session_state.get("disease_analysis"), "Confidence": st.session_state.get("confidence_score"), "(Custom)": ""}
                avail_opts = {lbl: txt for lbl, txt in txt_opts.items() if (txt and txt.strip()) or lbl == "(Custom)"}
                if not avail_opts: st.info("No text to translate.")
                else:
                    sel_lbl = st.selectbox("Translate:", list(avail_opts.keys()), 0, key="trans_sel")
                    txt_to_trans = avail_opts.get(sel_lbl, "")
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
                                try: out=translate(t=txt_to_trans, tl=tgt_lang, sl=src_lang); # Use shortened args if defined in translate
                                if out is not None: st.session_state.translation_result=out; st.success("OK!")
                                else: st.error("No result."); st.session_state.translation_error="None."
                                except Exception as e: st.error(f"Fail:{e}"); st.session_state.translation_error=str(e)
                    if st.session_state.get("translation_result"): st.text_area("Result:", format_translation(st.session_state.translation_result), 200, key="trans_out")


def render_main_content(col1, col2):
    """Renders the main page content within the provided columns."""
    _render_image_viewer(col1)
    _render_results_tabs(col2)