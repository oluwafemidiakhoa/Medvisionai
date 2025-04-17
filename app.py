# -*- coding: utf-8 -*-
"""
app.py – RadVision AI Advanced (main entry‑point)
-------------------------------------------------
Orchestrates the RadVision AI application modules and UI flow:
  • streamlit page config & logging
  • session‑state initialization
  • global CSS & monkey‑patch
  • sidebar (upload & controls)
  • file processing (DICOM/image ingestion)
  • main page UI (viewer + tabs)
  • deferred action handling (AI, UMLS, reports)
  • status banners for missing translation/UMLS
  • footer
"""
from __future__ import annotations

import logging
import sys
import io
import base64
import os
from typing import Any, TYPE_CHECKING

import streamlit as st

# Pillow is used in the monkey‑patch below
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None  # type: ignore
    logging.getLogger(__name__).warning(
        "Pillow not found; some image features will be disabled."
    )

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile

# ------------------------------------------------------------------------------
# Local configuration & messages
# ------------------------------------------------------------------------------
try:
    from config import (
        LOG_LEVEL,
        LOG_FORMAT,
        DATE_FORMAT,
        APP_CSS,
        FOOTER_MARKDOWN,
        APP_TITLE,
        APP_ICON,
        USER_GUIDE_MARKDOWN,
        DISCLAIMER_WARNING,
        UMLS_CONFIG_MSG,
    )
except ImportError as e:
    st.error(f"CRITICAL: Failed to load config.py: {e}")
    logging.critical(f"CRITICAL: Failed to load config.py: {e}", exc_info=True)
    sys.exit(1)

# ------------------------------------------------------------------------------
# Core modules
# ------------------------------------------------------------------------------
from session_state import initialize_session_state
from sidebar_ui    import render_sidebar
from file_processing import handle_file_upload
from main_page_ui import render_main_content
from action_handlers import handle_action

# ------------------------------------------------------------------------------
# Optional features
# ------------------------------------------------------------------------------
# Translation
try:
    from translation_models import TRANSLATION_AVAILABLE, TRANSLATION_CONFIG_MSG
    logging.getLogger(__name__).info(f"Translation Available: {TRANSLATION_AVAILABLE}")
except ImportError:
    TRANSLATION_AVAILABLE = False
    TRANSLATION_CONFIG_MSG = (
        "Translation module missing or dependencies (deep-translator) not installed."
    )
    logging.getLogger(__name__).warning(
        "translation_models.py import failed; translation disabled."
    )

# UMLS
try:
    from umls_utils import UMLS_UTILS_LOADED  # bool flag exported by umls_utils
    logging.getLogger(__name__).info(f"UMLS Utils Loaded: {UMLS_UTILS_LOADED}")
except ImportError:
    UMLS_UTILS_LOADED = False
    logging.getLogger(__name__).warning(
        "umls_utils.py import failed; UMLS features disabled."
    )

UMLS_API_KEY_PRESENT = bool(os.getenv("UMLS_APIKEY"))
IS_UMLS_FULLY_AVAILABLE = UMLS_UTILS_LOADED and UMLS_API_KEY_PRESENT
logging.getLogger(__name__).info(
    f"UMLS Key Present: {UMLS_API_KEY_PRESENT}, Fully Available: {IS_UMLS_FULLY_AVAILABLE}"
)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def get_session_id() -> str:
    """Return the current session ID from Streamlit state."""
    return st.session_state.get("session_id", "unknown")

# ------------------------------------------------------------------------------
# 1) Page config (must be first Streamlit call)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# 2) Logging
# ------------------------------------------------------------------------------
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
logger.info("=== RadVision AI start (Streamlit v%s) ===", st.__version__)

# ------------------------------------------------------------------------------
# 3) Session‑state init
# ------------------------------------------------------------------------------
initialize_session_state()
logger.debug("Session initialized. ID=%s", get_session_id())

# ------------------------------------------------------------------------------
# 4) Global CSS
# ------------------------------------------------------------------------------
st.markdown(APP_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 5) Monkey‑patch st.image for drawable canvas
# ------------------------------------------------------------------------------
import streamlit.elements.image as _st_image  # noqa: E402
if not hasattr(_st_image, "image_to_url"):
    if PIL_AVAILABLE:
        logger.info("Applying st.image_to_url monkey‑patch for drawable canvas.")
        def _image_to_url_monkey_patch(
            image: Any,
            width: int = -1,
            clamp: bool = False,
            channels: str = "RGB",
            output_format: str = "auto",
            image_id: str = "",
        ) -> str:
            if not (PIL_AVAILABLE and isinstance(image, Image)):
                logger.warning(
                    "image_to_url: unsupported type %s; returning empty URL",
                    type(image),
                )
                return ""
            fmt = (output_format.upper() if output_format != "auto"
                   else (image.format or "PNG"))
            if fmt not in {"PNG", "JPEG", "WEBP", "GIF"}:
                fmt = "PNG"
            if image.mode == "RGBA" and fmt == "JPEG":
                fmt = "PNG"
            img = image
            if img.mode == "P":
                img = img.convert("RGBA")
            if channels == "RGB" and img.mode not in {"RGB", "L"}:
                img = img.convert("RGB")
            buf = io.BytesIO()
            try:
                img.save(buf, format=fmt)
            except Exception as e:
                logger.error("Failed to buffer image (%s): %s", fmt, e)
                return ""
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/{fmt.lower()};base64,{b64}"
        _st_image.image_to_url = _image_to_url_monkey_patch  # type: ignore[attr-defined]
        logger.info("Monkey‑patch applied.")
    else:
        logger.warning(
            "Pillow missing; cannot apply image_to_url monkey‑patch."
        )
else:
    logger.debug("st.image_to_url already present; skipping patch.")

# ------------------------------------------------------------------------------
# 6) Sidebar & file upload
# ------------------------------------------------------------------------------
uploaded_file: UploadedFile | None = render_sidebar()
logger.debug("Sidebar rendered; uploaded_file=%s", type(uploaded_file).__name__)

# ------------------------------------------------------------------------------
# 7) File ingestion & state update
# ------------------------------------------------------------------------------
handle_file_upload(uploaded_file)

# ------------------------------------------------------------------------------
# 8) Main content area
# ------------------------------------------------------------------------------
st.divider()
st.title(f"{APP_ICON} {APP_TITLE} · AI‑Assisted Image Analysis")
with st.expander("User Guide & Disclaimer", expanded=False):
    st.warning(f"⚠️ {DISCLAIMER_WARNING}")
    st.markdown(USER_GUIDE_MARKDOWN, unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns([2, 3], gap="large")
render_main_content(col1, col2)

# ------------------------------------------------------------------------------
# 9) Deferred action handling
# ------------------------------------------------------------------------------
if (action := st.session_state.get("last_action")):
    logger.info("Handling action: %s", action)
    handle_action(action)
    # if still set, clear to avoid loops
    if st.session_state.get("last_action") == action:
        st.session_state.last_action = None
        logger.debug("Cleared last_action flag.")

# ------------------------------------------------------------------------------
# 10) Status banners for optional features
# ------------------------------------------------------------------------------
if not TRANSLATION_AVAILABLE:
    st.warning(f"🌐 Translation unavailable – {TRANSLATION_CONFIG_MSG}")
    logger.warning("Displayed translation banner.")

if not IS_UMLS_FULLY_AVAILABLE:
    reason = (UMLS_CONFIG_MSG
              if UMLS_API_KEY_PRESENT else
              "Missing UMLS dependencies or API key.")
    st.warning(f"🧬 UMLS unavailable – {reason}")
    logger.warning("Displayed UMLS banner: %s", reason)

# ------------------------------------------------------------------------------
# 11) Footer
# ------------------------------------------------------------------------------
st.divider()
session_id = get_session_id()
st.caption(f"{APP_ICON} {APP_TITLE} | Session ID: {session_id}")
st.markdown(FOOTER_MARKDOWN, unsafe_allow_html=True)
logger.info("=== Render complete; Session ID=%s ===", session_id)
