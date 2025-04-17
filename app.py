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
from streamlit.runtime.scriptrunner import get_script_run_ctx # For reliable session ID

# Pillow is checked for availability, crucial for the monkey-patch.
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None # type: ignore[assignment, misc] # Define Image as None if Pillow isn't installed

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
    USER_GUIDE_MARKDOWN, # Assuming user guide text is moved to config
    DISCLAIMER_WARNING, # Assuming disclaimer is moved to config
)
# Core application modules
from session_state import initialize_session_state
from sidebar_ui import render_sidebar
from main_page_ui import render_main_content
from file_processing import handle_file_upload
from action_handlers import handle_action

# Optional helpers (check availability without crashing)
try:
    # Assuming translation_models exports this constant
    from translation_models import TRANSLATION_AVAILABLE
except ImportError:
    TRANSLATION_AVAILABLE = False

try:
    # Assuming umls_utils exports this constant
    from umls_utils import UMLS_AVAILABLE, UMLS_CONFIG_MSG # Add message for clarity
except ImportError:
    UMLS_AVAILABLE = False
    # Provide a default message if the module itself is missing
    UMLS_CONFIG_MSG = "Add `UMLS_APIKEY` to HF Secrets & restart."


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def get_session_id() -> str:
    """Retrieves the current Streamlit session ID reliably."""
    ctx = get_script_run_ctx()
    return ctx.session_id if ctx else "N/A"

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
    stream=sys.stdout, # Log to stdout for containerized environments
    force=True # Override any existing configuration
)
logger = logging.getLogger(__name__)
logger.info("--- RadVision AI App Initializing (Streamlit v%s) ---", st.__version__)

# ---------------------------------------------------------------------------
# Session State Initialization (Ensure it runs early)
# ---------------------------------------------------------------------------
initialize_session_state(get_session_id()) # Pass session ID during init if needed
logger.debug("Session state initialized.")

# ---------------------------------------------------------------------------
# Apply Global CSS Styles
# ---------------------------------------------------------------------------
st.markdown(APP_CSS, unsafe_allow_html=True)
logger.debug("Global CSS applied.")

# ---------------------------------------------------------------------------
# Monkey-Patch for st.image (if needed and Pillow is available)
# Required for st_canvas snapshot rendering compatibility
# ---------------------------------------------------------------------------
import streamlit.elements.image as st_image # Use alias for clarity

# Check if the function *already exists* (e.g., future Streamlit versions)
if not hasattr(st_image, "image_to_url"):
    if not PIL_AVAILABLE:
        logger.warning("Pillow library not found. Cannot apply image_to_url monkey-patch.")
    else:
        logger.info("Applying monkey-patch for st.image -> data-url generation.")
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
            # Ensure input is a PIL Image object
            if not isinstance(image, Image.Image):
                logger.warning("image_to_url: Input is not a PIL Image (%s). Returning empty URL.", type(image))
                return ""

            # Determine output format, defaulting to PNG
            fmt = output_format.upper() if output_format != "auto" else (image.format or "PNG")
            if fmt not in {"PNG", "JPEG", "WEBP", "GIF"}:
                logger.debug("image_to_url: Unsupported format '%s', falling back to PNG.", fmt)
                fmt = "PNG"

            # Handle alpha channel compatibility (JPEG doesn't support alpha)
            if image.mode == "RGBA" and fmt == "JPEG":
                logger.debug("image_to_url: RGBA image requested as JPEG, converting to PNG.")
                fmt = "PNG" # Switch to PNG to preserve alpha

            # Convert palette images (mode 'P') to RGBA for broader compatibility
            if image.mode == "P":
                logger.debug("image_to_url: Converting Palette image (mode 'P') to RGBA.")
                image = image.convert("RGBA")

            # Convert to RGB if requested and necessary
            if channels == "RGB" and image.mode not in {"RGB", "L"}: # 'L' (grayscale) is fine
                logger.debug("image_to_url: Converting image mode '%s' to RGB.", image.mode)
                image = image.convert("RGB")

            # Save image to an in-memory buffer
            buffer = io.BytesIO()
            try:
                image.save(buffer, format=fmt)
                img_data = buffer.getvalue()
            except Exception as e:
                logger.error("image_to_url: Failed to save image to buffer (format: %s): %s", fmt, e)
                return ""

            # Encode buffer to base64 and create data URL
            b64_data = base64.b64encode(img_data).decode("utf-8")
            return f"data:image/{fmt.lower()};base64,{b64_data}"

        # Apply the patch
        st_image.image_to_url = _image_to_url_monkey_patch # type: ignore[attr-defined]
        logger.info("Monkey-patch applied successfully.")
else:
    logger.info("Streamlit version >= X.Y.Z already has image_to_url. Skipping monkey-patch.")


# ---------------------------------------------------------------------------
# === Main Application Flow ===
# ---------------------------------------------------------------------------

# --- 1. Render Sidebar & Get Uploaded File ---
# The sidebar function handles its own elements and returns the file object
uploaded_file: UploadedFile | None = render_sidebar()
logger.debug("Sidebar rendered. Uploaded file: %s", uploaded_file.name if uploaded_file else "None")

# --- 2. Process Uploaded File ---
# This function updates session state with the processed image data
handle_file_upload(uploaded_file)
# Logging within handle_file_upload is assumed

# --- 3. Render Main Content Area ---
# Title, User Guide Expander, Columns, Viewer, Results Tabs
st.divider() # Use st.divider() for modern separator
st.title(f"{APP_ICON} {APP_TITLE} · AI-Assisted Image Analysis")

with st.expander("User Guide & Disclaimer", expanded=False):
    st.warning(f"⚠️ **Disclaimer**: {DISCLAIMER_WARNING}")
    st.markdown(USER_GUIDE_MARKDOWN, unsafe_allow_html=True) # Assuming markdown format

st.divider() # Use st.divider()

# Define layout columns
col_img_viewer, col_analysis_results = st.columns([2, 3], gap="large") # Use descriptive names

# Render the main page UI components into the columns
render_main_content(col_img_viewer, col_analysis_results)
logger.debug("Main content area rendered.")

# --- 4. Handle Deferred Actions ---
# Check if an action button was clicked (state set in sidebar_ui or action_handlers)
action = st.session_state.get("last_action")
if action:
    logger.info("Executing deferred action: '%s'", action)
    # Action handler manages its own logic, spinners, and state updates
    handle_action(action)
    # Reset the action trigger ONLY if the handler didn't redirect/rerun
    if st.session_state.get("last_action") == action: # Check if state changed during handler
         st.session_state.last_action = None # Prevent re-triggering on next rerun
         logger.debug("Action '%s' completed and reset.", action)
else:
    logger.debug("No deferred action pending.")


# --- 5. Display Status Banners for Optional Features ---
# These appear near the bottom, informed by the availability flags
if not TRANSLATION_AVAILABLE:
    st.warning("🌐 Translation backend not loaded – install `deep-translator` & restart.")
    logger.warning("Optional feature unavailable: Translation (deep-translator not found).")

if not UMLS_AVAILABLE:
    st.warning(f"🧬 UMLS features unavailable – {UMLS_CONFIG_MSG}")
    logger.warning("Optional feature unavailable: UMLS (%s)", UMLS_CONFIG_MSG)


# --- 6. Render Footer ---
st.divider() # Use st.divider()
# Use the reliable session ID getter
current_session_id = get_session_id()
st.caption(f"{APP_ICON} {APP_TITLE} | Session ID: {current_session_id}")
st.markdown(FOOTER_MARKDOWN, unsafe_allow_html=True)
logger.info("--- Render cycle complete – Session ID: %s ---", current_session_id)
# ---------------------------------------------------------------------------
# === End of Application Flow ===
# ---------------------------------------------------------------------------