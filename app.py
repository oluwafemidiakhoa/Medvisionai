# -*- coding: utf-8 -*-
"""
app.py – RadVision AI Advanced (main entry‑point)
-------------------------------------------------
Orchestrates the RadVision AI application modules and UI flow.
Handles page configuration, logging, session state, main layout,
and status reporting for optional features.
"""
from __future__ import annotations

# Standard library imports
import logging
import sys
import io
import base64
import os
from typing import Any, TYPE_CHECKING

# Third-party imports
import streamlit as st

# Pillow is checked for availability
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None # type: ignore[assignment, misc]
    logging.getLogger(__name__).warning("Pillow library not found. Image processing unavailable.")


# Conditional import for type checking
if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile

# --- Local Application Imports ---
# Configuration (ensure this file exists and has the needed constants)
try:
    from config import (
        LOG_LEVEL, LOG_FORMAT, DATE_FORMAT, APP_CSS, FOOTER_MARKDOWN,
        APP_TITLE, APP_ICON, USER_GUIDE_MARKDOWN, DISCLAIMER_WARNING,
        UMLS_CONFIG_MSG # Specific message for missing key/config
    )
except ImportError as cfg_err:
     # Fallback if config.py is missing - app will likely fail but gives a clue
     sys.exit(f"CRITICAL ERROR: Failed to import configuration from config.py: {cfg_err}")

# Core application modules
from session_state import initialize_session_state, reset_session_state_for_new_file # Import reset function too
from sidebar_ui import render_sidebar
from main_page_ui import render_main_content
from file_processing import handle_file_upload
from action_handlers import handle_action

# --- Check Optional Feature Availability ---
# Translation
try:
    # Assumes translation_models exports these
    from translation_models import TRANSLATION_AVAILABLE, TRANSLATION_CONFIG_MSG
    logging.getLogger(__name__).info(f"Translation Available: {TRANSLATION_AVAILABLE}")
except ImportError:
    TRANSLATION_AVAILABLE = False
    TRANSLATION_CONFIG_MSG = "Translation module not found." # Fallback message
    logging.getLogger(__name__).warning("translation_models.py not found or import failed.")

# UMLS
try:
    # Import only the flag indicating successful load of the module + its deps
    from umls_utils import UMLS_UTILS_LOADED
    logging.getLogger(__name__).info(f"UMLS Utils Loaded: {UMLS_UTILS_LOADED}")
except ImportError:
    UMLS_UTILS_LOADED = False
    logging.getLogger(__name__).warning("umls_utils.py not found or import failed.")

# UMLS Full Availability Check: Module MUST load AND API Key MUST be present
UMLS_API_KEY_PRESENT = bool(os.getenv("UMLS_APIKEY"))
IS_UMLS_FULLY_AVAILABLE = UMLS_UTILS_LOADED and UMLS_API_KEY_PRESENT
logging.getLogger(__name__).info(f"UMLS API Key Present: {UMLS_API_KEY_PRESENT}")
logging.getLogger(__name__).info(f"UMLS Fully Available (Utils Loaded + Key Present): {IS_UMLS_FULLY_AVAILABLE}")

# --- Helper Functions ---
def get_session_id() -> str:
    """Retrieves the current session ID from session state."""
    return st.session_state.get("session_id", "N/A_STATE") # Default if state not init yet

# ---------------------------------------------------------------------------
# Streamlit Page Configuration (MUST BE FIRST ST CALL)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
# Remove default Streamlit handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
# Configure root logger
logging.basicConfig(
    level=LOG_LEVEL, format=LOG_FORMAT, datefmt=DATE_FORMAT, stream=sys.stdout, force=True
)
logger = logging.getLogger(__name__)
logger.info("--- RadVision AI App Initializing (Streamlit v%s) ---", st.__version__)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
initialize_session_state() # Call without arguments
logger.debug(f"Session state initialization check complete. Session ID: {get_session_id()}")

# ---------------------------------------------------------------------------
# Apply Global CSS
# ---------------------------------------------------------------------------
st.markdown(APP_CSS, unsafe_allow_html=True)
logger.debug("Global CSS applied.")

# ---------------------------------------------------------------------------
# Monkey-Patch (Optional, if needed for st_canvas)
# ---------------------------------------------------------------------------
# ... (Keep the monkey-patch code exactly as before if you use streamlit-drawable-canvas) ...
# Example placeholder if not needed:
# logger.debug("Skipping image_to_url monkey-patch (not required or Pillow unavailable).")
# OR keep the full patch code from previous versions if you use it.


# ===========================================================================
# === Main Application Flow ===
# ===========================================================================

# --- 1. Render Sidebar & Get Uploaded File ---
# This function should handle its own UI elements and return the file object
# It might also trigger session state resets internally if needed.
uploaded_file: UploadedFile | None = render_sidebar()
logger.debug("Sidebar rendered. Uploaded file object: %s", type(uploaded_file).__name__)

# --- 2. Process Uploaded File ---
# This function takes the uploaded file, processes it (potentially resetting state),
# and updates session state with image data (e.g., st.session_state.display_image).
# Critical step for fixing the "image not showing" issue lies within this function's implementation.
# Ensure it calls `reset_session_state_for_new_file()` appropriately.
handle_file_upload(uploaded_file)

# --- 3. Render Main Content Area ---
st.divider()
st.title(f"{APP_ICON} {APP_TITLE} · AI-Assisted Image Analysis")

with st.expander("User Guide & Disclaimer", expanded=False):
    st.warning(f"⚠️ **Disclaimer**: {DISCLAIMER_WARNING}")
    st.markdown(USER_GUIDE_MARKDOWN, unsafe_allow_html=True)

st.divider()

# Define layout columns for viewer and results
col_img_viewer, col_analysis_results = st.columns([2, 3], gap="large")

# Render the main page UI components into the columns
# This function is responsible for calling st.image(st.session_state.get('display_image'), ...)
# Ensure it correctly reads the state set by handle_file_upload.
render_main_content(col_img_viewer, col_analysis_results)
logger.debug("Main content area rendered.")

# --- 4. Handle Deferred Actions ---
# Check if an action button (like 'Run Analysis') set a flag in session state
action = st.session_state.get("last_action")
if action:
    logger.info("Executing deferred action: '%s'", action)
    # action_handlers.py takes over, performs the action, updates state
    handle_action(action)
    # Reset the trigger flag *after* handling, if it wasn't changed by the handler
    if st.session_state.get("last_action") == action:
         st.session_state.last_action = None
         logger.debug("Action '%s' completed and trigger reset.", action)
else:
    logger.debug("No deferred action pending.")


# --- 5. Display Status Banners for Optional Features ---
# These appear near the bottom, providing feedback on configuration/dependencies.

# Translation Banner
if not TRANSLATION_AVAILABLE:
    st.warning(f"🌐 Translation features unavailable – {TRANSLATION_CONFIG_MSG}")
    logger.warning("Banner displayed: Translation unavailable (%s)", TRANSLATION_CONFIG_MSG)

# UMLS Banner (Uses the combined check)
if not IS_UMLS_FULLY_AVAILABLE:
    # Determine the specific reason for unavailability
    if not UMLS_UTILS_LOADED:
        # If the module/dependency itself failed
        reason = "Failed to load UMLS utilities (check imports/dependencies in umls_utils.py and requirements.txt)."
    elif not UMLS_API_KEY_PRESENT:
        # If module loaded but key is missing
        reason = UMLS_CONFIG_MSG # Use the specific message from config.py
    else:
        # Fallback, should ideally not be reached
         reason = "Unknown configuration issue."

    st.warning(f"🧬 UMLS features unavailable – {reason}")
    logger.warning("Banner displayed: UMLS unavailable (%s)", reason)


# --- 6. Render Footer ---
st.divider()
current_session_id = get_session_id() # Read ID from state
st.caption(f"{APP_ICON} {APP_TITLE} | Session ID: {current_session_id}")
st.markdown(FOOTER_MARKDOWN, unsafe_allow_html=True)
logger.info("--- Render cycle complete – Session ID: %s ---", current_session_id)
# ===========================================================================
# === End of Application Flow ===
# ===========================================================================