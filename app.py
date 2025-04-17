# app.py (Main Application File)
import streamlit as st
import logging
import sys
import copy # For monkey patch if needed
import io   # For monkey patch if needed
import base64 # For monkey patch if needed
import os # Keep for getenv potentially used elsewhere
from typing import Any # Keep for monkey-patch type hint

# --- Configuration and Setup ---
from config import LOG_LEVEL, LOG_FORMAT, DATE_FORMAT, APP_CSS, FOOTER_MARKDOWN
from session_state import initialize_session_state
from sidebar_ui import render_sidebar
from main_page_ui import render_main_content
from file_processing import handle_file_upload
from action_handlers import handle_action

# --- Page Config (Must be first Streamlit command) ---
st.set_page_config(
    page_title="RadVision AI Advanced", layout="wide", page_icon="⚕️", initial_sidebar_state="expanded"
)

# --- Logging Setup ---
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=DATE_FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.info(f"--- RadVision AI Start (Streamlit v{st.__version__}) ---")

# --- Initialize Session State ---
initialize_session_state()

# --- Apply Custom CSS ---
st.markdown(APP_CSS, unsafe_allow_html=True)

# --- Check/Apply Monkey Patch (Important for st_canvas) ---
# (Keep the monkey-patch logic as is)
try: from PIL import Image; PIL_AVAILABLE = True
except ImportError: PIL_AVAILABLE = False; Image = None
import streamlit.elements.image as st_image
logger.debug(f"Checking for st_image.image_to_url. Available: {hasattr(st_image, 'image_to_url')}")
if not hasattr(st_image, "image_to_url"):
    logger.info("Applying monkey-patch for st.elements.image.image_to_url.")
    def image_to_url_monkey_patch(img_obj: Any,width: int = -1,clamp: bool = False,channels: str = "RGB",output_format: str = "auto",image_id: str = "") -> str:
        if PIL_AVAILABLE and isinstance(img_obj, Image.Image):
            try:
                buffered = io.BytesIO(); fmt = img_obj.format or "PNG"
                if output_format.lower() != "auto": fmt = output_format.upper()
                if fmt not in ["PNG", "JPEG", "GIF", "WEBP"]: fmt = "PNG"
                if img_obj.mode == 'RGBA' and fmt == 'JPEG': fmt = 'PNG'
                temp_img = img_obj
                if temp_img.mode == 'P': temp_img = temp_img.convert('RGBA'); fmt = "PNG"
                elif channels == "RGB" and temp_img.mode not in ['RGB', 'L']: temp_img = temp_img.convert('RGB')
                elif temp_img.mode == 'CMYK': temp_img = temp_img.convert('RGB')
                logger.debug(f"Saving image for data URL format: {fmt}")
                temp_img.save(buffered, format=fmt)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8"); return f"data:image/{fmt.lower()};base64,{img_str}"
            except Exception as e: logger.error(f"Monkey-patch fail: {e}", exc_info=True); return ""
        else: logger.warning(f"Monkey-patch skip: Type {type(img_obj)} / PIL {PIL_AVAILABLE}."); return ""
    st_image.image_to_url = image_to_url_monkey_patch; logger.info("Monkey-patch applied.")
else: logger.info("Monkey-patch not needed.")

# --- Render Sidebar and Get Uploaded File ---
uploaded_file = render_sidebar()

# --- Process Uploaded File ---
handle_file_upload(uploaded_file)

# --- Render Main Page Content ---
st.markdown("---"); st.title("⚕️ RadVision AI Advanced: AI-Assisted Image Analysis")
with st.expander("User Guide & Disclaimer", expanded=False):
    st.warning("⚠️ **Disclaimer**: For research/educational use ONLY. NOT medical advice.")
    st.markdown("""**Workflow:** 1.Upload 2.(DICOM) W/L 3.(Optional) ROI 4.AI Actions 5.Explore Tabs 6.UMLS Lookup 7.Translate 8.Confidence 9.Report""")
st.markdown("---")
col1, col2 = st.columns([2, 3])
render_main_content(col1, col2) # Renders viewer and results tabs

# --- Handle Actions ---
current_action = st.session_state.get("last_action")
if current_action:
    handle_action(current_action) # Delegate to action handler

# --- Render Footer ---
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced | Session ID: {st.session_state.get('session_id', 'N/A')}")
st.markdown(FOOTER_MARKDOWN, unsafe_allow_html=True)
logger.info(f"--- Render complete: {st.session_state.get('session_id')} ---")
