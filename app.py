# -*- coding: utf-8 -*-
"""
app.py – RadVision AI Advanced (main entry‑point)
-------------------------------------------------
Orchestrates the RadVision AI application modules and UI flow.
Handles page configuration, logging, session state, main layout,
status reporting for optional features, and necessary patches.
"""
from __future__ import annotations

# Standard library imports
import logging
import sys
import io          # Required for monkey-patch
import base64      # Required for monkey-patch
import os
from typing import Any, TYPE_CHECKING # Ensure Any is imported

# Third-party imports
import streamlit as st

# Pillow is checked for availability
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None # type: ignore[assignment, misc] # Define Image as None if Pillow isn't installed
    logging.getLogger(__name__).warning("Pillow library not found. Image processing/display might fail.")


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
     # Use st.error for visibility if possible early on, otherwise log and exit
     st.error(f"CRITICAL ERROR: Failed to import configuration from config.py: {cfg_err}")
     logging.critical(f"CRITICAL ERROR: Failed to import configuration from config.py: {cfg_err}", exc_info=True)
     sys.exit(f"CRITICAL ERROR: Failed to import configuration from config.py: {cfg_err}")


# Core application modules
from session_state import initialize_session_state, reset_session_state_for_new_file
from sidebar_ui import render_sidebar
from main_page_ui import render_main_content
from file_processing import handle_file_upload
from action_handlers import handle_action

# --- Check Optional Feature Availability ---
# Translation
try:
    from translation_models import TRANSLATION_AVAILABLE, TRANSLATION_CONFIG_MSG
    logging.getLogger(__name__).info(f"Translation Available: {TRANSLATION_AVAILABLE}")
except ImportError:
    TRANSLATION_AVAILABLE = False
    TRANSLATION_CONFIG_MSG = "Translation module not found or dependencies missing (e.g., `deep-translator`)."
    logging.getLogger(__name__).warning("translation_models.py not found or import failed.")

# UMLS
try:
    from umls_utils import UMLS_UTILS_LOADED # Flag indicating module+requests loaded
    logging.getLogger(__name__).info(f"UMLS Utils Loaded: {UMLS_UTILS_LOADED}")
except ImportError:
    UMLS_UTILS_LOADED = False
    logging.getLogger(__name__).warning("umls_utils.py not found or import failed.")

# Combined UMLS check
UMLS_API_KEY_PRESENT = bool(os.getenv("UMLS_APIKEY"))
IS_UMLS_FULLY_AVAILABLE = UMLS_UTILS_LOADED and UMLS_API_KEY_PRESENT
logging.getLogger(__name__).info(f"UMLS API Key Present: {UMLS_API_KEY_PRESENT}")
logging.getLogger(__name__).info(f"UMLS Fully Available (Utils Loaded + Key Present): {IS_UMLS_FULLY_AVAILABLE}")

# --- Helper Functions ---
def get_session_id() -> str:
    """Retrieves the current session ID from session state."""
    return st.session_state.get("session_id", "N/A_STATE")

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


# !!! --- MONKEY-PATCH SECTION --- !!!
# Ensure this section is present and correctly placed (after imports, before UI rendering)
# ---------------------------------------------------------------------------
# Monkey-Patch for st.image (if needed and Pillow is available)
# Required for streamlit-drawable-canvas background image rendering
# ---------------------------------------------------------------------------
import streamlit.elements.image as st_image # Use specific alias

# Check if the function *already exists* and if Pillow is available
if not hasattr(st_image, "image_to_url"):
    if not PIL_AVAILABLE:
        logger.warning("Pillow library not found. Cannot apply image_to_url monkey-patch for drawable canvas.")
    else:
        logger.info("Applying monkey-patch for st.image -> data-url generation (for drawable canvas).")
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
            # Need to handle case where PIL_AVAILABLE is False and Image is None
            if not PIL_AVAILABLE or not isinstance(image, Image.Image): # type: ignore # Ignore error if Image is None fallback
                logger.warning("image_to_url: Input is not a PIL Image or Pillow unavailable (%s). Returning empty URL.", type(image))
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
        logger.info("Monkey-patch 'image_to_url' applied successfully.")
else:
    logger.info("Streamlit version detected that likely includes image_to_url natively or patch already applied. Skipping monkey-patch.")
# !!! --- END OF MONKEY-PATCH SECTION --- !!!


# ===========================================================================
# === Main Application Flow ===
# ===========================================================================

# --- 1. Render Sidebar & Get Uploaded File ---
uploaded_file: UploadedFile | None = render_sidebar()
logger.debug("Sidebar rendered. Uploaded file object: %s", type(uploaded_file).__name__)

# --- 2. Process Uploaded File ---
# This function reads the file, processes it, updates session state (crucially 'display_image'),
# and potentially resets parts of the state.
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

# Render the UI components into the columns. This function calls st.image using
# st.session_state.display_image and potentially st_canvas which relies on the patch.
render_main_content(col_img_viewer, col_analysis_results)
logger.debug("Main content area rendered.")

# --- 4. Handle Deferred Actions ---
action = st.session_state.get("last_action")
if action:
    logger.info("Executing deferred action: '%s'", action)
    handle_action(action) # This module performs the action
    # Reset trigger if action handler didn't change it (e.g., via st.rerun)
    if st.session_state.get("last_action") == action:
         st.session_state.last_action = None
         logger.debug("Action '%s' completed and trigger reset.", action)
else:
    logger.debug("No deferred action pending.")


# --- 5. Display Status Banners for Optional Features ---
# Translation Banner
if not TRANSLATION_AVAILABLE:
    st.warning(f"🌐 Translation features unavailable – {TRANSLATION_CONFIG_MSG}")
    logger.warning("Banner displayed: Translation unavailable (%s)", TRANSLATION_CONFIG_MSG)

# UMLS Banner (Uses the combined check)
if not IS_UMLS_FULLY_AVAILABLE:
    # Determine the specific reason for unavailability
    if not UMLS_UTILS_LOADED:
        reason = "Failed to load UMLS utilities (check imports/dependencies like 'requests')."
    elif not UMLS_API_KEY_PRESENT:
        reason = UMLS_CONFIG_MSG # Use the specific message from config.py about the key
    else:
        reason = "Unknown configuration issue preventing UMLS use." # Fallback

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