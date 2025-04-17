# -*- coding: utf-8 -*-
"""
app.py – RadVision AI Advanced (main entry‑point)
-------------------------------------------------
Split‑architecture version that wires together:
    • sidebar_ui.py          – upload & action buttons
    • main_page_ui.py        – viewer + tabbed results
    • file_processing.py     – image / DICOM ingestion
    • action_handlers.py     – runs AI, UMLS, report

This file primarily orchestrates the application flow and displays
high‑level status banners (e.g., missing optional dependencies).
Heavy logic resides within the imported helper modules.
"""
from __future__ import annotations

# Standard library imports should come first
import logging
import sys
import io
import base64
import os
from typing import Any, TYPE_CHECKING

# Third-party imports
import streamlit as st
# Removed get_script_run_ctx as session_id is now handled internally by session_state.py

# Pillow is checked for availability, crucial for the monkey-patch.
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None # type: ignore[assignment, misc]

# Conditional import for type checking improves static analysis
if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile

# Local application/library specific imports
# Configuration should be imported early
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
    UMLS_CONFIG_MSG # Now defined in config.py
)
# Core application modules
# Corrected session_state import strategy
from session_state import initialize_session_state # Import the function directly
from sidebar_ui import render_sidebar
from main_page_ui import render_main_content
from file_processing import handle_file_upload
from action_handlers import handle_action

# Optional helpers (check availability without crashing)
try:
    from translation_models import TRANSLATION_AVAILABLE, TRANSLATION_CONFIG_MSG
except ImportError:
    TRANSLATION_AVAILABLE = False
    TRANSLATION_CONFIG_MSG = "Install `deep-translator` & restart." # Provide default message

try:
    from umls_utils import UMLS_AVAILABLE # UMLS_CONFIG_MSG is now in config.py
except ImportError:
    UMLS_AVAILABLE = False
    # UMLS_CONFIG_MSG comes from config.py


# ---------------------------------------------------------------------------
# Helper Functions (Revised)
# ---------------------------------------------------------------------------
def get_session_id() -> str:
    """Retrieves the current session ID *from session state*."""
    # Now reads the ID managed by session_state.py
    return st.session_state.get("session_id", "N/A_STATE")

# ---------------------------------------------------------------------------
# Streamlit Page Configuration (Must be the *first* Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Logging Configuration (Clear existing handlers, set new format)
# ---------------------------------------------------------------------------
# Remove default Streamlit handlers to prevent duplicate logs
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure root logger
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)
logger.info("--- RadVision AI App Initializing (Streamlit v%s) ---", st.__version__)

# ---------------------------------------------------------------------------
# Session State Initialization (Corrected Call)
# ---------------------------------------------------------------------------
initialize_session_state() # <-- FIXED: Call without arguments
# get_session_id() will now read the ID set *inside* initialize_session_state
logger.debug(f"Session state initialized/verified. Session ID: {get_session_id()}")

# ---------------------------------------------------------------------------
# Apply Global CSS Styles
# ---------------------------------------------------------------------------
st.markdown(APP_CSS, unsafe_allow_html=True)
logger.debug("Global CSS applied.")

# ---------------------------------------------------------------------------
# Monkey-Patch for st.image (if needed and Pillow is available)
# Required for st_canvas snapshot rendering compatibility
# ---------------------------------------------------------------------------
import streamlit.elements.image as st_image

if not hasattr(st_image, "image_to_url"):
    if not PIL_AVAILABLE:
        logger.warning("Pillow library not found. Cannot apply image_to_url monkey-patch.")
    else:
        logger.info("Applying monkey-patch for st.image -> data-url generation.")
        # ... (Keep the _image_to_url_monkey_patch function definition as before) ...
        def _image_to_url_monkey_patch(
            image: Any, # Can be various types, PIL Image is key
            width: int = -1,
            clamp: bool = False,
            channels: str = "RGB",
            output_format: str = "auto",
            image_id: str = "", # May be used by Streamlit internally
        ) -> str:
            """
            Serializes PIL Image to a base64 data URL.
            Needed for components like streamlit-drawable-canvas to redisplay images.
            Handles basic format conversions (e.g., palette to RGBA, alpha channel).
            """
            if not isinstance(image, Image.Image):
                logger.warning("image_to_url: Input is not a PIL Image (%s). Returning empty URL.", type(image))
                return ""
            fmt = output_format.upper() if output_format != "auto" else (image.format or "PNG")
            if fmt not in {"PNG", "JPEG", "WEBP", "GIF"}:
                logger.debug("image_to_url: Unsupported format '%s', falling back to PNG.", fmt)
                fmt = "PNG"
            if image.mode == "RGBA" and fmt == "JPEG":
                logger.debug("image_to_url: RGBA image requested as JPEG, converting to PNG.")
                fmt = "PNG"
            if image.mode == "P":
                logger.debug("image_to_url: Converting Palette image (mode 'P') to RGBA.")
                image = image.convert("RGBA")
            if channels == "RGB" and image.mode not in {"RGB", "L"}:
                logger.debug("image_to_url: Converting image mode '%s' to RGB.", image.mode)
                image = image.convert("RGB")
            buffer = io.BytesIO()
            try:
                image.save(buffer, format=fmt)
                img_data = buffer.getvalue()
            except Exception as e:
                logger.error("image_to_url: Failed to save image to buffer (format: %s): %s", fmt, e)
                return ""
            b64_data = base64.b64encode(img_data).decode("utf-8")
            return f"data:image/{fmt.lower()};base64,{b64_data}"

        st_image.image_to_url = _image_to_url_monkey_patch # type: ignore[attr-defined]
        logger.info("Monkey-patch applied successfully.")
else:
    logger.info("Streamlit version detected that likely includes image_to_url. Skipping monkey-patch.")


# ---------------------------------------------------------------------------
# === Main Application Flow ===
# ---------------------------------------------------------------------------

# --- 1. Render Sidebar & Get Uploaded File ---
uploaded_file: UploadedFile | None = render_sidebar()
logger.debug("Sidebar rendered. Uploaded file: %s", uploaded_file.name if uploaded_file else "None")

# --- 2. Process Uploaded File ---
handle_file_upload(uploaded_file) # Assumes this calls reset_session_state_for_new_file if needed

# --- 3. Render Main Content Area ---
st.divider()
st.title(f"{APP_ICON} {APP_TITLE} · AI-Assisted Image Analysis")

with st.expander("User Guide & Disclaimer", expanded=False):
    st.warning(f"⚠️ **Disclaimer**: {DISCLAIMER_WARNING}")
    st.markdown(USER_GUIDE_MARKDOWN, unsafe_allow_html=True)

st.divider()

col_img_viewer, col_analysis_results = st.columns([2, 3], gap="large")
render_main_content(col_img_viewer, col_analysis_results)
logger.debug("Main content area rendered.")

# --- 4. Handle Deferred Actions ---
action = st.session_state.get("last_action")
if action:
    logger.info("Executing deferred action: '%s'", action)
    handle_action(action)
    if st.session_state.get("last_action") == action:
         st.session_state.last_action = None
         logger.debug("Action '%s' completed and reset.", action)
else:
    logger.debug("No deferred action pending.")


# --- 5. Display Status Banners for Optional Features ---
# Read availability flags and config messages
if not TRANSLATION_AVAILABLE:
    st.warning(f"🌐 Translation features unavailable – {TRANSLATION_CONFIG_MSG}")
    logger.warning("Optional feature unavailable: Translation (%s)", TRANSLATION_CONFIG_MSG)

if not UMLS_AVAILABLE:
    st.warning(f"🧬 UMLS features unavailable – {UMLS_CONFIG_MSG}") # Message comes from config.py
    logger.warning("Optional feature unavailable: UMLS (%s)", UMLS_CONFIG_MSG)

# --- 6. Render Footer ---
st.divider()
# Use the revised get_session_id() which reads from st.session_state
current_session_id = get_session_id()
st.caption(f"{APP_ICON} {APP_TITLE} | Session ID: {current_session_id}")
st.markdown(FOOTER_MARKDOWN, unsafe_allow_html=True)
logger.info("--- Render cycle complete – Session ID: %s ---", current_session_id)
# ---------------------------------------------------------------------------
# === End of Application Flow ===
# ---------------------------------------------------------------------------