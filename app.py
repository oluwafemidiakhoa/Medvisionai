# ──────────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
app.py – RadVision AI Advanced
──────────────────────────────
Main entry‑point that wires together:

    • sidebar_ui.py          – upload & action buttons
    • main_page_ui.py        – viewer + tabbed results
    • file_processing.py     – image / DICOM ingestion
    • action_handlers.py     – AI, UMLS, reporting

It also:
    • sets Streamlit page config & global CSS
    • initialises / resets session‑state
    • applies a Pillow‑based monkey‑patch so
      `streamlit‑drawable‑canvas` can re‑render images
    • shows status banners for optional subsystems
"""
from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import logging
import sys
import io
import base64
from typing import Any, TYPE_CHECKING
import os

# ── third‑party ───────────────────────────────────────────────────────────────
import streamlit as st

# Pillow (image handling) – optional but strongly recommended
try:
    import PIL.Image as PIL_ImageModule
    from PIL.Image import Image as PILImageClass
    PIL_AVAILABLE = True
except ImportError:                       # graceful degradation
    PIL_ImageModule = None                # type: ignore
    PILImageClass = object                # dummy sentinel
    PIL_AVAILABLE = False
    logging.warning("Pillow not installed – image features limited.")

if TYPE_CHECKING:                         # mypy / IDE hints only
    from streamlit.runtime.uploaded_file_manager import UploadedFile

# ── local modules ─────────────────────────────────────────────────────────────
try:
    from config import (
        LOG_LEVEL, LOG_FORMAT, DATE_FORMAT,
        APP_TITLE, APP_ICON,
        APP_CSS, FOOTER_MARKDOWN,
        USER_GUIDE_MARKDOWN, DISCLAIMER_WARNING,
        UMLS_CONFIG_MSG,
    )
except ImportError as cfg_err:            # critical – exit early
    st.error(f"❌ Failed to import config.py – {cfg_err}")
    logging.critical("config import failed", exc_info=True)
    sys.exit(1)

from session_state   import initialize_session_state
from sidebar_ui      import render_sidebar
from main_page_ui    import render_main_content
from file_processing import handle_file_upload
from action_handlers import handle_action

# Optional subsystems ---------------------------------------------------------
try:
    from translation_models import TRANSLATION_AVAILABLE, TRANSLATION_CONFIG_MSG
except ImportError:
    TRANSLATION_AVAILABLE  = False
    TRANSLATION_CONFIG_MSG = "translation_models missing or `deep‑translator` not installed."
try:
    from umls_utils import UMLS_UTILS_LOADED     # indicates module + requests ok
except ImportError:
    UMLS_UTILS_LOADED = False

UMLS_API_KEY_PRESENT   = bool(os.getenv("UMLS_APIKEY"))
UMLS_FULLY_AVAILABLE   = UMLS_UTILS_LOADED and UMLS_API_KEY_PRESENT

# ──────────────────────────────────────────────────────────────────────────────
# 1  Streamlit page config – MUST be first Streamlit call
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# 2  Logging
# ──────────────────────────────────────────────────────────────────────────────
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)
logger.info("RadVision AI boot • Streamlit %s", st.__version__)

# ──────────────────────────────────────────────────────────────────────────────
# 3  Session‑state initialise
# ──────────────────────────────────────────────────────────────────────────────
initialize_session_state()
session_id = st.session_state.get("session_id", "N/A")
logger.debug("Session initialised – ID: %s", session_id)

# ──────────────────────────────────────────────────────────────────────────────
# 4  Global CSS theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(APP_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 5  Monkey‑patch  (image → base64 data‑URL for drawable‑canvas)
# ──────────────────────────────────────────────────────────────────────────────
import streamlit.elements.image as _st_image               # noqa: WPS433

if not hasattr(_st_image, "image_to_url"):

    def _image_to_url_monkey_patch(
        img_obj: Any,
        width: int = -1,
        clamp: bool = False,
        channels: str = "RGB",
        output_format: str = "auto",
        image_id: str = "",
    ) -> str:
        """Return a `data:image/...` URL for a PIL Image."""
        if not (PIL_AVAILABLE and isinstance(img_obj, PILImageClass)):
            logger.warning("image_to_url: unsupported type %s", type(img_obj))
            return ""

        # choose format
        fmt = (
            output_format.upper()
            if output_format != "auto"
            else (getattr(img_obj, "format", None) or "PNG")
        )
        if fmt not in {"PNG", "JPEG", "WEBP", "GIF"}:
            fmt = "PNG"
        if img_obj.mode == "RGBA" and fmt == "JPEG":
            fmt = "PNG"                            # keep alpha

        # convert modes as needed
        if img_obj.mode == "P":
            img_obj = img_obj.convert("RGBA")
        if channels == "RGB" and img_obj.mode not in {"RGB", "L"}:
            img_obj = img_obj.convert("RGB")

        buf = io.BytesIO()
        try:
            img_obj.save(buf, format=fmt)
        except Exception as exc:                   # pragma: no cover
            logger.error("image_to_url: %s", exc, exc_info=True)
            return ""
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/{fmt.lower()};base64,{b64}"

    _st_image.image_to_url = _image_to_url_monkey_patch  # type: ignore[attr-defined]
    logger.info("Patched st.image → data‑URL (drawable‑canvas support)")
else:
    logger.info("image_to_url already present – no patch applied.")

# ──────────────────────────────────────────────────────────────────────────────
# 6  Sidebar  (upload + action buttons)
# ──────────────────────────────────────────────────────────────────────────────
uploaded_file: UploadedFile | None = render_sidebar()

# ──────────────────────────────────────────────────────────────────────────────
# 7  File processing (populates display_image / processed_image)
# ──────────────────────────────────────────────────────────────────────────────
handle_file_upload(uploaded_file)

# ──────────────────────────────────────────────────────────────────────────────
# 8  Main layout
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.title(f"{APP_ICON} {APP_TITLE} · AI‑Assisted Image Analysis")
with st.expander("User Guide & Disclaimer", expanded=False):
    st.warning(f"⚠️ **Disclaimer**: {DISCLAIMER_WARNING}")
    st.markdown(USER_GUIDE_MARKDOWN, unsafe_allow_html=True)
st.divider()

col_view, col_results = st.columns([2, 3], gap="large")
render_main_content(col_view, col_results)

# ──────────────────────────────────────────────────────────────────────────────
# 9  Deferred actions
# ──────────────────────────────────────────────────────────────────────────────
if (action := st.session_state.get("last_action")):
    logger.info("Executing action: %s", action)
    handle_action(action)
    if st.session_state.get("last_action") == action:   # action handler may rerun
        st.session_state.last_action = None

# ──────────────────────────────────────────────────────────────────────────────
# 10  Status banners for optional systems
# ──────────────────────────────────────────────────────────────────────────────
if not TRANSLATION_AVAILABLE:
    st.warning(f"🌐 Translation unavailable – {TRANSLATION_CONFIG_MSG}")

if not UMLS_FULLY_AVAILABLE:
    reason = (
        "UMLS utilities failed to load."
        if not UMLS_UTILS_LOADED else
        UMLS_CONFIG_MSG
        if not UMLS_API_KEY_PRESENT else
        "Unknown configuration issue."
    )
    st.warning(f"🧬 UMLS unavailable – {reason}")

# ──────────────────────────────────────────────────────────────────────────────
# 11  Footer
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(f"{APP_ICON} {APP_TITLE} | Session ID: {session_id}")
st.markdown(FOOTER_MARKDOWN, unsafe_allow_html=True)
logger.info("Render complete • Session ID: %s", session_id)
