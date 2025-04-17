# sidebar_ui.py
# -*- coding: utf‑8 -*-
"""
RadVision AI – Sidebar UI

Handles:
  • file upload & demo‑mode toggle
  • ROI clearing
  • DICOM window/level sliders
  • action buttons (analysis, Q&A, disease, confidence, PDF)

Relies on:
  • config.py      → TIPS, DISEASE_OPTIONS
  • ui_components  → dicom_wl_sliders()
  • report_utils   → generate_pdf_report_bytes()  (availability flag)
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import streamlit as st
from PIL import Image

from config import TIPS, DISEASE_OPTIONS

logger = logging.getLogger(__name__)

# ──────────────────────────────
#  Optional imports
# ──────────────────────────────
try:
    from ui_components import dicom_wl_sliders  # ds + metadata → wc, ww
except ImportError:  # pragma: no cover
    dicom_wl_sliders = None
    logger.warning("dicom_wl_sliders not imported – DICOM W/L disabled.")

try:
    from report_utils import generate_pdf_report_bytes

    REPORT_UTILS_AVAILABLE = True
except ImportError:
    generate_pdf_report_bytes = None
    REPORT_UTILS_AVAILABLE = False
    logger.warning("report_utils not found – PDF reporting disabled.")

try:
    from dicom_utils import dicom_to_image, DICOM_UTILS_AVAILABLE  # bool flag
except ImportError:  # pragma: no cover
    dicom_to_image = None
    DICOM_UTILS_AVAILABLE = False


# ──────────────────────────────
#  Sidebar renderer
# ──────────────────────────────
def render_sidebar() -> Optional[st.runtime.uploaded_file_manager.UploadedFile]:
    """
    Draw the sidebar and return the uploaded file object (if any).
    """

    with st.sidebar:
        # ---------- header ----------
        st.header("⚕️ RadVision Controls")
        st.markdown("---")
        st.info(f"💡 {random.choice(TIPS)}")
        st.markdown("---")

        # ---------- upload ----------
        st.header("Image Upload & Settings")
        uploaded_file = st.file_uploader(
            "Upload Image (JPG, PNG, DCM)",
            type=["jpg", "jpeg", "png", "dcm", "dicom"],
            key="file_uploader_widget",
            help="Upload a medical image; DICOM (.dcm) preferred.",
        )

        st.checkbox(
            "🚀 Demo Mode",
            value=st.session_state.get("demo_loaded", False),
            key="demo_mode_checkbox",
            help="Load a sample chest‑X‑ray and pre‑run analysis.",
        )

        # ---------- clear ROI ----------
        if st.button("🗑️ Clear ROI", key="clear_roi_btn", help="Remove selected ROI"):
            st.session_state.roi_coords = None
            st.session_state.canvas_drawing = None
            st.session_state.clear_roi_feedback = True
            logger.info("ROI cleared.")
            st.rerun()

        if st.session_state.pop("clear_roi_feedback", False):
            st.success("✅ ROI cleared!")
            st.balloons()

        # ---------- DICOM window/level ----------
        if (
            st.session_state.get("is_dicom")
            and isinstance(st.session_state.get("display_image"), Image.Image)
            and dicom_wl_sliders
        ):
            st.markdown("---")
            st.subheader("DICOM Display")

            ds = st.session_state.get("dicom_dataset")
            meta = st.session_state.get("dicom_metadata", {})
            new_wc, new_ww = dicom_wl_sliders(ds, meta)

            # apply change?
            if (
                new_wc is not None
                and new_ww is not None
                and (
                    new_wc != st.session_state.get("current_display_wc")
                    or new_ww != st.session_state.get("current_display_ww")
                )
            ):
                logger.info(f"DICOM W/L changed → WC={new_wc}, WW={new_ww}")
                st.session_state.current_display_wc = new_wc
                st.session_state.current_display_ww = new_ww

                if (
                    DICOM_UTILS_AVAILABLE
                    and dicom_to_image
                    and ds is not None
                    and isinstance(ds, object)
                ):
                    with st.spinner("Applying window/level…"):
                        try:
                            disp = dicom_to_image(ds, wc=new_wc, ww=new_ww)
                            if isinstance(disp, Image.Image):
                                st.session_state.display_image = (
                                    disp.convert("RGB") if disp.mode != "RGB" else disp
                                )
                                st.rerun()
                            else:
                                st.error("Failed to update display image.")
                        except Exception as e:  # pragma: no cover
                            st.error(f"DICOM conversion error: {e}")
                            logger.error("dicom_to_image error", exc_info=True)
                else:
                    logger.warning("DICOM utils unavailable – cannot apply W/L.")

        # ======================================================================
        #  AI actions
        # ======================================================================
        st.markdown("---")
        st.header("🤖 AI Analysis Actions")

        img_ready = isinstance(st.session_state.get("processed_image"), Image.Image)

        # Initial analysis
        if st.button(
            "▶️ Run Initial Analysis",
            key="analyze_btn",
            disabled=not img_ready,
            help="Perform a general analysis of the image (or ROI).",
        ):
            st.session_state.last_action = "analyze"
            st.rerun()

        # Q&A
        st.subheader("❓ Ask AI a Question")
        q_txt = st.text_area(
            "Question:",
            key="question_input_widget",
            height=100,
            placeholder="E.g. “Are there any nodules?”",
            disabled=not img_ready,
        )
        if st.button("💬 Ask Question", key="ask_btn", disabled=not img_ready):
            if q_txt.strip():
                st.session_state.last_action = "ask"
                st.rerun()
            else:
                st.warning("Enter a question first.")

        # condition
        st.subheader("🎯 Condition‑Specific Analysis")
        cond = st.selectbox(
            "Condition:",
            [""] + sorted(DISEASE_OPTIONS),
            key="disease_select_widget",
            disabled=not img_ready,
        )
        if st.button("🩺 Analyze Condition", key="disease_btn", disabled=not img_ready):
            if cond:
                st.session_state.last_action = "disease"
                st.rerun()
            else:
                st.warning("Select a condition first.")

        # confidence + PDF
        st.markdown("---")
        st.header("📊 Confidence & Reporting")

        can_conf = bool(
            st.session_state.get("history")
            or st.session_state.get("initial_analysis")
            or st.session_state.get("disease_analysis")
        )

        if st.button(
            "📈 Estimate AI Confidence",
            key="confidence_btn",
            disabled=(not img_ready) or (not can_conf),
        ):
            st.session_state.last_action = "confidence"
            st.rerun()

        pdf_disabled = (not img_ready) or (not REPORT_UTILS_AVAILABLE)
        if st.button(
            "📄 Generate PDF Report Data",
            key="generate_report_data_btn",
            disabled=pdf_disabled,
        ):
            st.session_state.last_action = "generate_report_data"
            st.rerun()

        # download link
        if st.session_state.get("pdf_report_bytes"):
            fn = f"RadVisionAI_Report_{st.session_state.get('session_id','session')}.pdf"
            st.download_button(
                "⬇️ Download PDF Report",
                data=st.session_state.pdf_report_bytes,
                file_name=fn,
                mime="application/pdf",
                key="download_pdf_button",
            )

    # sidebar context finished
    return uploaded_file
