# file_processing.py
import streamlit as st
import hashlib
import uuid
import logging
import io
import os
from session_state import reset_session_state_for_new_file # Import state reset helper

# Import necessary libraries and utils (ensure they are installed/available)
try: from PIL import Image, UnidentifiedImageError; PIL_AVAILABLE = True
except ImportError: PIL_AVAILABLE = False; Image = None; UnidentifiedImageError = Exception

try: import pydicom; from pydicom.errors import InvalidDicomError; PYDICOM_AVAILABLE = True
except ImportError: PYDICOM_AVAILABLE = False; pydicom = None; InvalidDicomError = Exception

try: from dicom_utils import parse_dicom, extract_dicom_metadata, dicom_to_image, get_default_wl; DICOM_UTILS_AVAILABLE = True
except ImportError: DICOM_UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)

def handle_file_upload(uploaded_file):
    """Processes the uploaded file, updates session state, and triggers rerun."""
    if uploaded_file is None:
        # No file uploaded in this cycle
        return

    try:
        uploaded_file.seek(0)
        h = hashlib.sha256(uploaded_file.read()).hexdigest()[:16]
        uploaded_file.seek(0)
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{h}"
        logger.debug(f"Calculated file info: {new_file_info}")
    except Exception as e:
        logger.warning(f"Could not generate hash for {uploaded_file.name}: {e}")
        new_file_info = f"{uploaded_file.name}-{uploaded_file.size}-{uuid.uuid4().hex[:8]}" # Fallback
        logger.debug(f"Using fallback file info: {new_file_info}")

    if new_file_info != st.session_state.get("uploaded_file_info"):
        logger.info(f"New file detected: '{new_file_info}' (Prev: '{st.session_state.get('uploaded_file_info')}')")
        st.toast(f"Processing '{uploaded_file.name}'...", icon="⏳")

        # Reset state before processing
        reset_session_state_for_new_file()

        st.session_state.uploaded_file_info = new_file_info
        st.session_state.demo_loaded = False
        st.session_state.raw_image_bytes = uploaded_file.getvalue()
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()

        # Determine if DICOM (initial check)
        is_dicom_attempt = (
            PYDICOM_AVAILABLE and DICOM_UTILS_AVAILABLE and
            ("dicom" in uploaded_file.type.lower() or file_ext in (".dcm", ".dicom"))
        )
        st.session_state.is_dicom = is_dicom_attempt # Tentatively set
        logger.info(f"Attempting DICOM processing: {is_dicom_attempt}")

        with st.spinner("🔬 Analyzing file..."):
            img_disp, img_proc, success, err_msg = None, None, False, ""

            # --- DICOM Processing ---
            if is_dicom_attempt:
                try:
                    ds = parse_dicom(st.session_state.raw_image_bytes, uploaded_file.name)
                    st.session_state.dicom_dataset = ds
                    if ds:
                        st.session_state.dicom_metadata = extract_dicom_metadata(ds)
                        wc, ww = get_default_wl(ds)
                        st.session_state.current_display_wc, st.session_state.current_display_ww = wc, ww
                        img_disp = dicom_to_image(ds, wc=wc, ww=ww)
                        img_proc = dicom_to_image(ds, wc=None, ww=None, normalize=True)
                        if isinstance(img_disp, Image.Image) and isinstance(img_proc, Image.Image):
                            success = True; logger.info("DICOM processed successfully.")
                        else: err_msg = "DICOM conversion failed."
                    else: err_msg = "DICOM parse failed."; st.session_state.is_dicom = False
                except InvalidDicomError as e: err_msg = f"Invalid DICOM: {e}"; st.session_state.is_dicom = False
                except Exception as e: err_msg = f"DICOM error: {e}"; logger.error(err_msg, exc_info=True); st.session_state.is_dicom = False
                if not success: logger.error(f"DICOM processing failed: {err_msg}")

            # --- Standard Image Processing (if not DICOM or DICOM failed) ---
            if not success:
                st.session_state.is_dicom = False # Ensure flag is False if DICOM failed
                logger.info("Attempting standard image processing...")
                if not PIL_AVAILABLE: err_msg = "PIL unavailable."; logger.critical(err_msg)
                else:
                    try:
                        raw = Image.open(io.BytesIO(st.session_state.raw_image_bytes))
                        proc = raw.convert("RGB")
                        img_disp, img_proc = proc.copy(), proc.copy(); success = True
                        logger.info("Standard image processed successfully.")
                    except UnidentifiedImageError: err_msg = "Cannot ID format (use JPG/PNG/DCM)."
                    except Exception as e: err_msg = f"Standard image error: {e}"; logger.error(err_msg, exc_info=True)
                if not success: logger.error(f"Standard image processing failed: {err_msg}")

            # --- Update State Based on Success ---
            if success and isinstance(img_disp, Image.Image) and isinstance(img_proc, Image.Image):
                st.session_state.display_image = img_disp.convert('RGB') if img_disp.mode != 'RGB' else img_disp
                st.session_state.processed_image = img_proc
                logger.info("Session state updated with processed images.")
                st.success(f"✅ '{uploaded_file.name}' loaded!")
                st.rerun() # Trigger UI update with new image
            else:
                 logger.error(f"Failed to process file {uploaded_file.name}. Error: {err_msg}")
                 st.error(f"Failed to load '{uploaded_file.name}'. {err_msg or 'Processing error.'}")
                 # Clear potentially inconsistent state on failure
                 st.session_state.uploaded_file_info = None # Mark as failed
                 st.session_state.display_image=None; st.session_state.processed_image=None
                 st.session_state.is_dicom=False; st.session_state.dicom_dataset=None
                 st.session_state.dicom_metadata={}; st.session_state.raw_image_bytes=None
                 st.session_state.current_display_wc=None; st.session_state.current_display_ww=None
    # else: # File is the same as last run, do nothing here
    #     logger.debug("Uploaded file info matches session state. No reprocessing.")