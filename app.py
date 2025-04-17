# -*- coding: utf-8 -*-
"""
app.py – RadVision AI Advanced (main entry‑point)
-------------------------------------------------
Split‑architecture version that wires together:
    • sidebar_ui.py          – upload & action buttons
    • main_page_ui.py        – viewer + tabbed results
    • file_processing.py     – image / DICOM ingestion
    • action_handlers.py     – runs AI, UMLS, report

This file only orchestrates the flow and shows high‑level
status banners (e.g. missing UMLS key). All heavy logic
is isolated in the helper modules so tweaking UI here is
safe.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard lib
# ---------------------------------------------------------------------------
import logging
import sys
import io
import base64
import os
from typing import Any

# ---------------------------------------------------------------------------
# Third‑party
# ---------------------------------------------------------------------------
import streamlit as st

# Pillow is required for the monkey‑patch below. Import *after* Streamlit so
# a missing Pillow surfaces in the browser rather than crashing the Space.
try:
    from PIL import Image  # noqa: WPS433 – external import
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None            # type: ignore

# ---------------------------------------------------------------------------
# Local sub‑modules (all tiny, focused)
# ---------------------------------------------------------------------------
from config import (
    LOG_LEVEL,
    LOG_FORMAT,
    DATE_FORMAT,
    APP_CSS,
    FOOTER_MARKDOWN,
)
from session_state import initialize_session_state
from sidebar_ui import render_sidebar
from main_page_ui import render_main_content
from file_processing import handle_file_upload
from action_handlers import handle_action

# Optional helpers (may be missing – we show banners instead of crashing)
try:
    from translation_models import TRANSLATION_AVAILABLE
except ImportError:
    TRANSLATION_AVAILABLE = False

try:
    from umls_utils import UMLS_AVAILABLE
except ImportError:
    UMLS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Streamlit page config (must be the *first* Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RadVision AI Advanced",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.info("--- RadVision AI bootstrapping (Streamlit v%s) ---", st.__version__)

# ---------------------------------------------------------------------------
# Initialise session‑state dict
# ---------------------------------------------------------------------------
initialize_session_state()

# ---------------------------------------------------------------------------
# Global CSS theme
# ---------------------------------------------------------------------------
st.markdown(APP_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Monkey‑patch 🖼 (st.elements.image.image_to_url) for st_canvas snapshots
# ---------------------------------------------------------------------------
import streamlit.elements.image as _st_image  # noqa: WPS433

if not hasattr(_st_image, "image_to_url"):

    def _image_to_url_monkey_patch(
        img_obj: Any,
        width: int = -1,
        clamp: bool = False,
        channels: str = "RGB",
        output_format: str = "auto",
        image_id: str = "",
    ) -> str:  # noqa: D401
        """Serialize PIL Image to data‑URL so st_canvas can re‑render."""
        if not (PIL_AVAILABLE and isinstance(img_obj, Image.Image)):
            logger.warning("[Patch] Unsupported object %s – returning empty URL", type(img_obj))
            return ""

        fmt = output_format.upper() if output_format != "auto" else (img_obj.format or "PNG")
        if fmt not in {"PNG", "JPEG", "WEBP", "GIF"}:
            fmt = "PNG"
        if img_obj.mode == "RGBA" and fmt == "JPEG":
            fmt = "PNG"  # JPEG has no alpha

        buf = io.BytesIO()
        if img_obj.mode == "P":  # palette → RGBA
            img_obj = img_obj.convert("RGBA")
        if channels == "RGB" and img_obj.mode not in {"RGB", "L"}:
            img_obj = img_obj.convert("RGB")
        img_obj.save(buf, format=fmt)
        data = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/{fmt.lower()};base64,{data}"

    _st_image.image_to_url = _image_to_url_monkey_patch  # type: ignore[attr-defined]
    logger.info("Applied monkey‑patch for st.image → data‑url")

# ---------------------------------------------------------------------------
# Sidebar (file upload & action buttons)
# ---------------------------------------------------------------------------
uploaded_file = render_sidebar()

# ---------------------------------------------------------------------------
# File processing (populate processed_image / display_image)
# ---------------------------------------------------------------------------
handle_file_upload(uploaded_file)

# ---------------------------------------------------------------------------
# Main content area (viewer & tabs)
# ---------------------------------------------------------------------------
st.markdown("---")
st.title("⚕️ RadVision AI Advanced · AI‑Assisted Image Analysis")
with st.expander("User Guide & Disclaimer", expanded=False):
    st.warning(
        "⚠️ **Disclaimer**: This tool is intended for research / educational use only."
        " It is **NOT** a substitute for professional medical evaluation.",
    )
    st.markdown(
        """
        **Typical workflow**
        1. **Upload** image (DICOM or PNG/JPG) – or enable *Demo Mode*.
        2. **(DICOM)** adjust *Window / Level* if required.
        3. *(optional)* draw an **ROI** rectangle.
        4. Trigger AI actions from the sidebar *(Initial, Q&A, Condition etc.)*.
        5. Explore results tabs (**UMLS**, **Translate**, **Confidence**) as needed.
        6. **Generate PDF** for a portable report.
        """,
    )

st.markdown("---")
col1, col2 = st.columns([2, 3], gap="large")
render_main_content(col1, col2)

# ---------------------------------------------------------------------------
# Run deferred action (set in sidebar buttons)
# ---------------------------------------------------------------------------
if (action := st.session_state.get("last_action")):
    handle_action(action)

# ---------------------------------------------------------------------------
# Status banners for missing optional features
# ---------------------------------------------------------------------------
if not TRANSLATION_AVAILABLE:
    st.warning("🌐 Translation backend not loaded – install `deep‑translator` & restart.")

if not UMLS_AVAILABLE:
    st.warning("🧬 UMLS features unavailable – add `UMLS_APIKEY` to HF Secrets & restart.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced | Session ID: {st.session_state.get('session_id', 'N/A')}")
st.markdown(FOOTER_MARKDOWN, unsafe_allow_html=True)
logger.info("--- Render complete – Session ID: %s ---", st.session_state.get("session_id"))
