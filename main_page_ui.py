# main_page_ui.py
# -*- coding: utf‑8 -*-
"""
Streamlit UI layer for RadVision AI.

‑ Renders the Image‑Viewer (with drawable ROI)
‑ Shows analysis tabs (Initial / Q&A / Condition / Confidence / Translate / UMLS Lookup)
‑ Handles manual UMLS search and displays auto‑enriched concepts

This file assumes:
    • session_state.py has created all default keys
    • action_handlers.handle_action() drives back‑end work
    • ui_components.display_umls_concepts() pretty‑prints a list[UMLSConcept]
"""

from __future__ import annotations

import streamlit as st
import logging
import os
from typing import Optional

from PIL import Image, ImageDraw

# ──────────────────────────────
#  Optional UI helpers
# ──────────────────────────────
try:
    from ui_components import display_dicom_metadata
except ImportError:
    display_dicom_metadata = None
    logging.warning("display_dicom_metadata not imported.")

try:
    from ui_components import display_umls_concepts
except ImportError:
    display_umls_concepts = None
    logging.warning("display_umls_concepts not imported.")

# drawable‑canvas (optional)
try:
    from streamlit_drawable_canvas import st_canvas

    DRAWABLE_CANVAS_AVAILABLE = True
except ImportError:
    st_canvas = None
    DRAWABLE_CANVAS_AVAILABLE = False

# translation layer (optional)
try:
    from translation_models import (
        TRANSLATION_AVAILABLE,
        LANGUAGE_CODES,
        AUTO_DETECT_INDICATOR,
        translate,
    )
except ImportError:
    TRANSLATION_AVAILABLE = False
    LANGUAGE_CODES = {"English": "en"}
    AUTO_DETECT_INDICATOR = "Auto"
    translate = None

# UMLS helpers (optional)
try:
    from umls_utils import (
        search_umls,
        UMLSAuthError,
        UMLSConcept,
        UMLS_APIKEY,
        UMLS_AVAILABLE,
    )
except ImportError:  # pragma: no cover
    UMLS_AVAILABLE = False
    search_umls = None
    UMLSAuthError = RuntimeError
    UMLSConcept = None
    UMLS_APIKEY = None

# shared config
try:
    from config import DEFAULT_UMLS_HITS
except ImportError:
    DEFAULT_UMLS_HITS = 5

# fallback for format_translation
try:
    from session_state import format_translation
except ImportError:

    def format_translation(txt: Optional[str]) -> str:  # type: ignore
        return str(txt or "")


logger = logging.getLogger(__name__)


# ──────────────────────────────
#  IMAGE‑VIEWER (left column)
# ──────────────────────────────
def _render_image_viewer(col):
    with col:
        st.subheader("🖼️ Image Viewer")

        img: Image.Image | None = st.session_state.get("display_image")
        if not isinstance(img, Image.Image):
            if st.session_state.get("uploaded_file_info"):
                st.error("❌ Image preview unavailable.")
            else:
                st.info("⬅️ Upload an image or enable *Demo Mode* in the sidebar.")
            return

        # ------------ ROI canvas ------------
        if DRAWABLE_CANVAS_AVAILABLE and st_canvas:
            st.caption("Draw rectangle for ROI.")
            max_w, max_h = 600, 500
            w, h = img.size
            if w <= 0 or h <= 0:
                st.warning("Invalid image size.")
            else:
                aspect = w / h
                c_w = min(w, max_w)
                c_h = int(c_w / aspect)
                if c_h > max_h:
                    c_h = max_h
                    c_w = int(c_h * aspect)
                c_w = max(c_w, 150)
                c_h = max(c_h, 150)

                canvas = st_canvas(
                    fill_color="rgba(255,165,0,0.25)",
                    stroke_color="rgba(239,83,80,0.9)",
                    stroke_width=2,
                    background_image=img,
                    update_streamlit=True,
                    height=c_h,
                    width=c_w,
                    drawing_mode="rect",
                    initial_drawing=st.session_state.get("canvas_drawing"),
                    key="drawable_canvas",
                )

                if canvas.json_data and canvas.json_data.get("objects"):
                    obj = canvas.json_data["objects"][-1]
                    if obj["type"] == "rect":
                        # convert canvas coords → original pixels
                        left = int(obj["left"])
                        top = int(obj["top"])
                        ow = int(obj["width"] * obj.get("scaleX", 1))
                        oh = int(obj["height"] * obj.get("scaleY", 1))
                        scale_x = w / c_w
                        scale_y = h / c_h
                        roi = {
                            "left": max(0, int(left * scale_x)),
                            "top": max(0, int(top * scale_y)),
                            "width": max(1, int(ow * scale_x)),
                            "height": max(1, int(oh * scale_y)),
                        }
                        if roi != st.session_state.get("roi_coords"):
                            st.session_state.roi_coords = roi
                            st.session_state.canvas_drawing = canvas.json_data
                            logger.info(f"ROI set: {roi}")
                            st.info(
                                f"ROI: ({roi['left']}, {roi['top']}) "
                                f"{roi['width']}×{roi['height']}",
                                icon="🎯",
                            )
        else:
            st.image(img, caption="Preview", use_container_width=True)

        # show ROI coords
        if st.session_state.get("roi_coords"):
            r = st.session_state.roi_coords
            st.caption(
                f"Current ROI → ({r['left']}, {r['top']}) "
                f"{r['width']}×{r['height']} px"
            )

        st.markdown("---")

        # DICOM metadata (if any)
        if (
            st.session_state.get("is_dicom")
            and st.session_state.get("dicom_metadata")
            and display_dicom_metadata
        ):
            with st.expander("📄 DICOM Metadata", False):
                display_dicom_metadata(st.session_state.dicom_metadata)


# ──────────────────────────────
#  RESULTS‑TABS (right column)
# ──────────────────────────────
def _render_results_tabs(col):
    with col:
        st.subheader("📊 Analysis & Results")

        tabs = st.tabs(
            [
                "🔬 Initial",
                "💬 Q&A",
                "🩺 Condition",
                "📈 Confidence",
                "🌐 Translate",
                "🧬 UMLS Lookup",
            ]
        )

        # ---- Initial Analysis ----
        with tabs[0]:
            st.text_area(
                "Findings",
                value=st.session_state.get("initial_analysis", "Run *Initial Analysis*."),
                height=400,
                disabled=True,
            )
            umls = st.session_state.get("initial_analysis_umls", [])
            if umls:
                st.markdown("---")
                st.markdown("**Auto‑Enriched UMLS Concepts:**")
                if display_umls_concepts:
                    display_umls_concepts(umls)
                else:
                    st.json([c.to_dict() for c in umls])  # type: ignore[attr-defined]

        # ---- Q&A ----
        with tabs[1]:
            st.text_area(
                "Latest Answer",
                value=st.session_state.get("qa_answer", "Ask a question."),
                height=180,
                disabled=True,
            )

            umls = st.session_state.get("qa_umls", [])
            if umls:
                st.markdown("---")
                st.markdown("**Auto‑Enriched UMLS Concepts (Latest Answer):**")
                if display_umls_concepts:
                    display_umls_concepts(umls)
                else:
                    st.json([c.to_dict() for c in umls])  # type: ignore[attr-defined]

            # full history
            st.markdown("---")
            st.subheader("Conversation History")
            history = st.session_state.get("history", [])
            if not history:
                st.caption("No Q&A yet.")
            else:
                for role, msg in reversed(history):
                    pfx = "👤" if role.lower().startswith("user") else "🤖"
                    st.markdown(f"**{pfx}**  {msg}", unsafe_allow_html=True)
                    st.markdown("---")

        # ---- Condition analysis ----
        with tabs[2]:
            st.text_area(
                "Condition Analysis",
                value=st.session_state.get("disease_analysis", "Run *Condition Analysis*."),
                height=400,
                disabled=True,
            )
            umls = st.session_state.get("disease_umls", [])
            if umls:
                st.markdown("---")
                st.markdown("**Auto‑Enriched UMLS Concepts:**")
                if display_umls_concepts:
                    display_umls_concepts(umls)
                else:
                    st.json([c.to_dict() for c in umls])  # type: ignore[attr-defined]

        # ---- Confidence ----
        with tabs[3]:
            st.text_area(
                "Confidence",
                value=st.session_state.get("confidence_score", "Run *Estimate Confidence*."),
                height=400,
                disabled=True,
            )

        # ---- Translation ----
        with tabs[4]:
            st.subheader("🌐 Translate")
            if not TRANSLATION_AVAILABLE:
                st.warning("Translation features are unavailable.")
            else:
                _render_translation_panel()

        # ---- Manual UMLS Lookup ----
        with tabs[5]:
            _render_umls_lookup_panel()


# ──────────────────────────────
#  Translation panel
# ──────────────────────────────
def _render_translation_panel():
    st.caption("Select text, choose languages, click **Translate**.")

    blocks = {
        "Initial": st.session_state.get("initial_analysis", ""),
        "Q&A": st.session_state.get("qa_answer", ""),
        "Condition": st.session_state.get("disease_analysis", ""),
        "Confidence": st.session_state.get("confidence_score", ""),
        "(Custom)": "",
    }
    options = [k for k, v in blocks.items() if v.strip() or k == "(Custom)"]
    choice = st.selectbox("Text block:", options, index=0)
    text_src = blocks[choice]

    if choice == "(Custom)":
        text_src = st.text_area("Custom Text", value=text_src, height=120)
    else:
        st.text_area("Selected Text", value=text_src, height=120, disabled=True)

    col1, col2 = st.columns(2)
    with col1:
        src_lang = st.selectbox(
            "Source Language",
            [AUTO_DETECT_INDICATOR] + sorted(LANGUAGE_CODES.keys()),
            0,
        )
    with col2:
        tgt_lang = st.selectbox("Target Language", sorted(LANGUAGE_CODES.keys()), 0)

    if st.button("🔄 Translate"):
        if not text_src.strip():
            st.warning("No text selected.")
            return
        if src_lang == tgt_lang and src_lang != AUTO_DETECT_INDICATOR:
            st.info("Source and target languages are the same.")
            st.session_state.translation_result = text_src
            return

        with st.spinner("Translating…"):
            try:
                out = translate(
                    text=text_src,
                    target_language=tgt_lang,
                    source_language=src_lang,
                )
                st.session_state.translation_result = out
                st.success("Translation complete!")
            except Exception as e:
                st.error(f"Translation error: {e}")
                logger.error("Translation error", exc_info=True)
                st.session_state.translation_result = None

    if st.session_state.get("translation_result"):
        st.text_area(
            "Translated Text",
            value=format_translation(st.session_state.translation_result),
            height=200,
            disabled=True,
        )


# ──────────────────────────────
#  Manual UMLS lookup panel
# ──────────────────────────────
def _render_umls_lookup_panel():
    if not UMLS_AVAILABLE:
        st.warning("UMLS features are unavailable (no key or import error).")
        return
    if search_umls is None:
        st.error("UMLS search function missing.")
        return

    term = st.text_input(
        "Enter a term to look up in UMLS",
        value=st.session_state.get("lookup_term", ""),
    )
    st.session_state.lookup_term = term

    if st.button("Search UMLS"):
        if not term.strip():
            st.warning("Please enter a term.")
        elif not UMLS_APIKEY:
            st.error("Set **UMLS_APIKEY** in your environment / Space Secrets.")
        else:
            with st.spinner("Querying UMLS…"):
                try:
                    concepts = search_umls(term, UMLS_APIKEY, DEFAULT_UMLS_HITS)
                    st.session_state.lookup_results = concepts
                    if not concepts:
                        st.info("No concepts found.")
                except UMLSAuthError as e:
                    st.error(f"UMLS authentication failed: {e}")
                except Exception as e:
                    st.error(f"UMLS search error: {e}")
                    logger.error("UMLS search error", exc_info=True)

    # display results
    if st.session_state.get("lookup_results"):
        st.markdown("---")
        if display_umls_concepts:
            display_umls_concepts(st.session_state.lookup_results)
        else:
            st.json([c.to_dict() for c in st.session_state.lookup_results])  # type: ignore[attr-defined]


# ──────────────────────────────
#  Public entry point
# ──────────────────────────────
def render_main_content(col_left, col_right):
    """Render left (image viewer) and right (tabs) columns."""
    _render_image_viewer(col_left)
    _render_results_tabs(col_right)
