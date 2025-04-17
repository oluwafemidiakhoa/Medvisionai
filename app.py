# -*- coding: utf-8 -*-
"""
app.py – RadVision AI Advanced (main entry‑point)
-------------------------------------------------
Orchestrates UI flow, session state, and the drawable‐canvas monkey‑patch so
your uploaded PIL images actually show up in the ROI viewer.
"""
from __future__ import annotations

import logging
import sys
import io
import base64
import os
from typing import Any

import streamlit as st

# --- Pillow check ---
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None  # type: ignore

# --- Local imports ---
from config import (
    LOG_LEVEL, LOG_FORMAT, DATE_FORMAT,
    APP_CSS, FOOTER_MARKDOWN,
    APP_TITLE, APP_ICON,
    USER_GUIDE_MARKDOWN, DISCLAIMER_WARNING
)
from session_state import initialize_session_state
from sidebar_ui import render_sidebar
from file_processing import handle_file_upload
from main_page_ui import render_main_content
from action_handlers import handle_action

# --- Optional feature flags ---
try:
    from translation_models import TRANSLATION_AVAILABLE
except ImportError:
    TRANSLATION_AVAILABLE = False

try:
    from umls_utils import UMLS_UTILS_LOADED
except ImportError:
    UMLS_UTILS_LOADED = False

UMLS_API_KEY_PRESENT = bool(os.getenv("UMLS_APIKEY"))
IS_UMLS_FULLY_AVAILABLE = UMLS_UTILS_LOADED and UMLS_API_KEY_PRESENT

# --- Streamlit page config (first call) ---
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Logging ---
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)
logger.info(f"--- Starting RadVision AI (Streamlit v{st.__version__}) ---")

# --- Session state init ---
initialize_session_state()

# --- Global CSS ---
st.markdown(APP_CSS, unsafe_allow_html=True)

# --- Monkey‑patch for st_canvas background images ---
import streamlit.elements.image as _st_image  # noqa

if not hasattr(_st_image, "image_to_url"):
    if PIL_AVAILABLE:
        logger.info("Applying image_to_url monkey‑patch for drawable canvas.")

        def _image_to_url_monkey_patch(
            img_obj: Any,
            width: int = -1,
            clamp: bool = False,
            channels: str = "RGB",
            output_format: str = "auto",
            image_id: str = "",
        ) -> str:
            """
            Convert a PIL.Image.Image into a base64 data‑URL for st_canvas.
            """
            # **Correct** isinstance check:
            if not (PIL_AVAILABLE and isinstance(img_obj, Image.Image)):
                return ""

            fmt = (output_format.upper() if output_format != "auto"
                   else (img_obj.format or "PNG"))
            if fmt not in {"PNG", "JPEG", "WEBP", "GIF"}:
                fmt = "PNG"
            if img_obj.mode == "RGBA" and fmt == "JPEG":
                fmt = "PNG"

            # Palette → RGBA
            if img_obj.mode == "P":
                img_obj = img_obj.convert("RGBA")
            # Enforce RGB
            if channels == "RGB" and img_obj.mode not in {"RGB", "L"}:
                img_obj = img_obj.convert("RGB")

            buf = io.BytesIO()
            img_obj.save(buf, format=fmt)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/{fmt.lower()};base64,{b64}"

        _st_image.image_to_url = _image_to_url_monkey_patch  # type: ignore[attr-defined]
    else:
        logger.warning("Pillow not available; skipping image_to_url patch.")

# --- Sidebar & upload ---
uploaded_file = render_sidebar()
handle_file_upload(uploaded_file)

# --- Main UI ---
st.markdown("---")
st.title(f"{APP_ICON} {APP_TITLE} · AI‑Assisted Image Analysis")
with st.expander("User Guide & Disclaimer", expanded=False):
    st.warning(f"⚠️ **Disclaimer**: {DISCLAIMER_WARNING}")
    st.markdown(USER_GUIDE_MARKDOWN, unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([2, 3], gap="large")
render_main_content(col1, col2)

# --- Deferred action ---
if action := st.session_state.get("last_action"):
    handle_action(action)
    if st.session_state.get("last_action") == action:
        st.session_state.last_action = None

# --- Status banners ---
if not TRANSLATION_AVAILABLE:
    st.warning("🌐 Translation unavailable – install `deep‑translator` & restart.")
if not IS_UMLS_FULLY_AVAILABLE:
    reason = (
        "UMLS utils failed to load."
        if not UMLS_UTILS_LOADED
        else "UMLS_APIKEY missing; add to Secrets & restart."
    )
    st.warning(f"🧬 UMLS unavailable – {reason}")

# --- Footer ---
st.markdown("---")
sid = st.session_state.get("session_id", "N/A")
st.caption(f"{APP_ICON} {APP_TITLE} | Session ID: {sid}")
st.markdown(FOOTER_MARKDOWN, unsafe_allow_html=True)
