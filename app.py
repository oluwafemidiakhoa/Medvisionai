# -*- coding: utf-8 -*-
"""
app.py – RadVision AI Advanced (streamlit main file)

Only orchestration lives here; heavy logic is in the helper modules.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Standard lib
# ─────────────────────────────────────────────────────────────────────────────
import base64
import io
import logging
import os
import sys
from typing import Any, TYPE_CHECKING

# ─────────────────────────────────────────────────────────────────────────────
# Third‑party
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st

try:
    from PIL import Image as PILImage               # real class for isinstance check
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    PILImage = None                                 # type: ignore[assignment]
    logging.getLogger(__name__).warning(
        "Pillow is missing – image display will fail; check requirements.txt")

if TYPE_CHECKING:                                  # nice type hints in editors
    from streamlit.runtime.uploaded_file_manager import UploadedFile

# ─────────────────────────────────────────────────────────────────────────────
# Local modules  (assumed present)
# ─────────────────────────────────────────────────────────────────────────────
from config import (
    LOG_LEVEL, LOG_FORMAT, DATE_FORMAT,
    APP_TITLE, APP_ICON, APP_CSS,
    USER_GUIDE_MARKDOWN, DISCLAIMER_WARNING,
    FOOTER_MARKDOWN, UMLS_CONFIG_MSG
)
from session_state import initialize_session_state
from sidebar_ui   import render_sidebar
from main_page_ui import render_main_content
from file_processing import handle_file_upload
from action_handlers import handle_action

# Optional modules – swallow import errors gracefully
try:
    from translation_models import TRANSLATION_AVAILABLE, TRANSLATION_CONFIG_MSG
except Exception:                                  # pragma: no cover
    TRANSLATION_AVAILABLE   = False
    TRANSLATION_CONFIG_MSG  = "deep‑translator not installed or failed to import."

try:
    from umls_utils import UMLS_UTILS_LOADED
except Exception:                                  # pragma: no cover
    UMLS_UTILS_LOADED = False

UMLS_API_KEY_PRESENT    = bool(os.getenv("UMLS_APIKEY"))
UMLS_FULLY_AVAILABLE    = UMLS_UTILS_LOADED and UMLS_API_KEY_PRESENT

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit page config ( MUST be the first st.* call )
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = APP_TITLE,
    page_icon  = APP_ICON,
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(
    level   = LOG_LEVEL,
    format  = LOG_FORMAT,
    datefmt = DATE_FORMAT,
    stream  = sys.stdout,
    force   = True,
)
logger = logging.getLogger(__name__)
logger.info("--- RadVision AI (Streamlit %s) ---", st.__version__)

# ─────────────────────────────────────────────────────────────────────────────
# Session‑state bootstrap
# ─────────────────────────────────────────────────────────────────────────────
initialize_session_state()

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(APP_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Monkey‑patch st.elements.image.image_to_url  (if Streamlit < 1.25 and/or HF Space)
# ─────────────────────────────────────────────────────────────────────────────
import streamlit.elements.image as _st_image       # noqa: WPS433

if not hasattr(_st_image, "image_to_url") and PIL_AVAILABLE:

    def _img_to_url(
        img:   Any,
        width: int  = -1,
        clamp: bool = False,
        channels: str = "RGB",
        output_format: str = "auto",
        image_id: str = "",
    ) -> str:
        # only accept PIL Image
        if not isinstance(img, PILImage.Image):     # ← FIXED isinstance check
            logger.debug("image_to_url: unsupported type %s", type(img))
            return ""

        fmt = output_format.upper() if output_format != "auto" else (img.format or "PNG")
        if fmt not in {"PNG", "JPEG", "WEBP", "GIF"}:
            fmt = "PNG"
        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode == "RGBA" and fmt == "JPEG":
            fmt = "PNG"
        if channels == "RGB" and img.mode not in {"RGB", "L"}:
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode()}"

    _st_image.image_to_url = _img_to_url            # type: ignore[attr-defined]
    logger.info("Applied image_to_url monkey‑patch")

# ─────────────────────────────────────────────────────────────────────────────
# 1 ● Sidebar – upload & actions
# ─────────────────────────────────────────────────────────────────────────────
uploaded_file: UploadedFile | None = render_sidebar()

# ─────────────────────────────────────────────────────────────────────────────
# 2 ● Handle uploaded file (populate session_state.display_image etc.)
# ─────────────────────────────────────────────────────────────────────────────
handle_file_upload(uploaded_file)

# ─────────────────────────────────────────────────────────────────────────────
# 3 ● Main content (viewer + tabbed results)
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.title(f"{APP_ICON} {APP_TITLE} · AI‑Assisted Image Analysis")

with st.expander("User Guide & Disclaimer", expanded=False):
    st.warning(f"⚠️ **Disclaimer**: {DISCLAIMER_WARNING}")
    st.markdown(USER_GUIDE_MARKDOWN, unsafe_allow_html=True)

st.divider()
col_left, col_right = st.columns([2, 3], gap="large")
render_main_content(col_left, col_right)

# ─────────────────────────────────────────────────────────────────────────────
# 4 ● Deferred action (set by buttons in sidebar)
# ─────────────────────────────────────────────────────────────────────────────
if (act := st.session_state.get("last_action")):
    logger.info("Executing deferred action: %s", act)
    handle_action(act)
    if st.session_state.get("last_action") == act:
        st.session_state.last_action = None

# ─────────────────────────────────────────────────────────────────────────────
# 5 ● Status banners (grouped neatly)
# ─────────────────────────────────────────────────────────────────────────────
if not TRANSLATION_AVAILABLE:
    st.warning(f"🌐 Translation unavailable – {TRANSLATION_CONFIG_MSG}")

if not UMLS_FULLY_AVAILABLE:
    msg = (
        "requests / helper module missing."          if not UMLS_UTILS_LOADED else
        UMLS_CONFIG_MSG                              if not UMLS_API_KEY_PRESENT else
        "unknown configuration error."
    )
    st.warning(f"🧬 UMLS unavailable – {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# 6 ● Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(f"{APP_ICON} {APP_TITLE} | Session ID: {st.session_state.get('session_id', 'N/A')}")
st.markdown(FOOTER_MARKDOWN, unsafe_allow_html=True)
logger.info("--- Render complete – session %s ---", st.session_state.get("session_id"))
