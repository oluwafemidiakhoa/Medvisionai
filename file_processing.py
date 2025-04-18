# -*- coding: utf-8 -*-
"""
file_processing.py - Handles Uploaded File Processing for RadVision AI
=======================================================================

Reads uploaded files (standard images or DICOM), processes them into a
displayable format (PIL Image), extracts relevant metadata, and updates
the Streamlit session state accordingly. Includes basic DICOM windowing.
"""

import logging
import io
from typing import Optional, Dict, Any, Tuple

import streamlit as st
import numpy as np

# --- Dependencies ---
# Pillow for standard image handling
try:
    from PIL import Image, UnidentifiedImageError
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    # Define dummy class if PIL is missing
    class Image: Image = type('Image', (object,), {}) # type: ignore[misc, assignment]
    class UnidentifiedImageError(Exception): pass
    logging.getLogger(__name__).error("Pillow library not found. Install it (`pip install Pillow`)")

# Pydicom for DICOM handling
try:
    import pydicom
    from pydicom.errors import InvalidDicomError
    import pydicom.pixel_data_handlers.util as pydicom_utils # For windowing support
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    # Define dummy classes/exceptions if pydicom is missing
    class pydicom: Dataset = type('Dataset', (object,), {}) # type: ignore[misc, assignment]
    class InvalidDicomError(Exception): pass
    class pydicom_utils: pass # Dummy namespace
    logging.getLogger(__name__).warning("pydicom library not found. Install it (`pip install pydicom numpy`). DICOM processing disabled.")

# Session state reset function
try:
    from session_state import reset_session_state_for_new_file
except ImportError:
    def reset_session_state_for_new_file(): # type: ignore[misc]
        logging.error("reset_session_state_for_new_file function not found in session_state.py!")
        st.error("Internal Error: Cannot reset state for new file.")
    logging.warning("Could not import reset_session_state_for_new_file from session_state.py.")

# --- Logging ---
logger = logging.getLogger(__name__)

# =============================================================================
# Helper Functions
# =============================================================================

def _extract_dicom_metadata(ds: pydicom.Dataset) -> Dict[str, Any]:
    """Extracts a selection of common and useful DICOM tags into a dictionary."""
    metadata = {}
    tags_to_extract = [
        "PatientName", "PatientID", "PatientBirthDate", "PatientSex",
        "StudyInstanceUID", "StudyDate", "StudyTime", "StudyDescription",
        "SeriesInstanceUID", "SeriesNumber", "SeriesDescription", "Modality",
        "SOPInstanceUID", "InstanceNumber", "ImageType",
        "Manufacturer", "ManufacturerModelName", "SoftwareVersions",
        "PixelSpacing", "SliceThickness", "SpacingBetweenSlices",
        "Rows", "Columns", "BitsAllocated", "BitsStored", "HighBit",
        "PixelRepresentation", "PhotometricInterpretation",
        "WindowCenter", "WindowWidth",
        "RescaleIntercept", "RescaleSlope",
        "BodyPartExamined", "Laterality", "ViewPosition",
        # Add more tags as needed
    ]

    logger.debug(f"Extracting metadata for {len(tags_to_extract)} predefined tags...")
    for tag_name in tags_to_extract:
        if tag_name in ds:
            element = ds[tag_name]
            # Use element.repval which attempts to give a useful string representation
            # Handle multi-value safely using str(element.value) as fallback
            try:
                 value_repr = element.repval if hasattr(element, 'repval') else str(element.value)
                 # Limit length for display
                 MAX_LEN = 100
                 if len(value_repr) > MAX_LEN:
                      value_repr = value_repr[:MAX_LEN-3] + "..."
                 metadata[tag_name] = value_repr
            except Exception as e:
                 logger.warning(f"Could not get representation for tag '{tag_name}': {e}")
                 metadata[tag_name] = "[Error reading value]"
        # else: logger.debug(f"Tag '{tag_name}' not found in dataset.") # Optional: log missing tags

    logger.info(f"Extracted {len(metadata)} DICOM metadata tags.")
    return metadata

def _apply_dicom_windowing(
    pixel_array: np.ndarray,
    wc: Optional[float],
    ww: Optional[float],
    rescale_slope: Optional[float],
    rescale_intercept: Optional[float],
    photometric_interpretation: str
) -> np.ndarray:
    """Applies Rescale Slope/Intercept and Window Center/Width to a NumPy array."""
    logger.debug(f"Applying windowing: WC={wc}, WW={ww}, Slope={rescale_slope}, Intercept={rescale_intercept}")
    image_2d = pixel_array.astype(np.float64) # Use float64 for calculations

    # 1. Apply Rescale Slope/Intercept (Modality LUT)
    if rescale_slope is not None and rescale_intercept is not None:
        logger.debug("Applying Rescale Slope and Intercept.")
        # Ensure slope is not zero
        slope = float(rescale_slope) if rescale_slope != 0 else 1.0
        intercept = float(rescale_intercept)
        image_2d = image_2d * slope + intercept

    # 2. Apply Window Center/Width (VOI LUT)
    # Use provided WC/WW if valid, otherwise calculate default window covering the range
    min_pixel = np.min(image_2d)
    max_pixel = np.max(image_2d)
    pixel_range = max_pixel - min_pixel if max_pixel > min_pixel else 1.0

    effective_wc = float(wc) if wc is not None else min_pixel + pixel_range / 2.0
    effective_ww = float(ww) if ww is not None and ww >= 1 else pixel_range
    effective_ww = max(1.0, effective_ww) # Ensure width is at least 1

    logger.debug(f"Effective WC={effective_wc:.2f}, WW={effective_ww:.2f} (Range: [{min_pixel:.2f} - {max_pixel:.2f}])")

    # Calculate min/max bounds for windowing
    window_min = effective_wc - effective_ww / 2.0
    window_max = effective_wc + effective_ww / 2.0

    # Apply windowing transformation
    image_2d = np.clip(image_2d, window_min, window_max)

    # 3. Normalize to 0-255 for 8-bit display
    if window_max > window_min: # Avoid division by zero if WW was invalid/zero
        image_2d = ((image_2d - window_min) / (window_max - window_min)) * 255.0
    else:
        image_2d.fill(128) # If range is zero, display as mid-gray
        logger.warning("Window width resulted in zero range after clipping. Displaying gray image.")

    # Handle MONOCHROME1: Invert pixel values
    if photometric_interpretation == "MONOCHROME1":
        logger.debug("Applying MONOCHROME1 inversion.")
        image_2d = 255.0 - image_2d

    # Convert to 8-bit unsigned integer
    image_2d_scaled = image_2d.astype(np.uint8)
    logger.debug("Windowing and scaling to uint8 complete.")
    return image_2d_scaled

def _dicom_to_pil_image(
    ds: pydicom.Dataset,
    wc: Optional[float] = None,
    ww: Optional[float] = None
) -> Optional[Image.Image]:
    """Converts a pydicom Dataset to a displayable PIL Image using basic windowing."""
    if not PYDICOM_AVAILABLE or not PIL_AVAILABLE:
        logger.error("Cannot convert DICOM: pydicom or Pillow library not available.")
        return None
    if not hasattr(ds, 'PixelData'):
        logger.error("DICOM dataset missing 'PixelData' attribute.")
        return None

    logger.info("Converting DICOM pixel data to PIL Image...")
    try:
        # Use pydicom's utility function to apply LUTs if possible (more robust)
        # This handles Photometric Interpretation, rescale, basic VOI LUT
        # Note: This might not use the *exact* WC/WW passed if internal VOI LUT exists
        # arr = pydicom_utils.apply_voi_lut(ds.pixel_array, ds) # Preferred method

        # Manual method for more explicit control over WC/WW from state:
        pixel_array = ds.pixel_array
        rescale_slope = ds.get("RescaleSlope")
        rescale_intercept = ds.get("RescaleIntercept")
        photometric = ds.get("PhotometricInterpretation", "MONOCHROME2")

        # Use WC/WW passed in (from state), otherwise fall back to tags in dataset
        window_center = wc if wc is not None else ds.get("WindowCenter")
        window_width = ww if ww is not None else ds.get("WindowWidth")

        # Handle potential multi-value WC/WW (take the first value)
        if isinstance(window_center, pydicom.multival.MultiValue):
             window_center = window_center[0] if len(window_center) > 0 else None
        if isinstance(window_width, pydicom.multival.MultiValue):
             window_width = window_width[0] if len(window_width) > 0 else None

        # Convert potentially numeric string types
        wc_float = float(window_center) if window_center is not None else None
        ww_float = float(window_width) if window_width is not None else None

        image_scaled = _apply_dicom_windowing(
            pixel_array,
            wc=wc_float,
            ww=ww_float,
            rescale_slope=rescale_slope,
            rescale_intercept=rescale_intercept,
            photometric_interpretation=photometric
        )

        # Create PIL Image from the processed NumPy array
        if image_scaled.ndim == 2: # Grayscale
             pil_image = Image.fromarray(image_scaled, mode='L') # L mode for 8-bit grayscale
             logger.info("Successfully created grayscale PIL Image from DICOM.")
             return pil_image
        elif image_scaled.ndim == 3 and image_scaled.shape[2] == 3: # RGB
             pil_image = Image.fromarray(image_scaled, mode='RGB')
             logger.info("Successfully created RGB PIL Image from DICOM.")
             return pil_image
        else:
             logger.error(f"Processed DICOM array has unexpected dimensions: {image_scaled.shape}")
             return None

    except Exception as e:
        logger.exception("Failed to convert DICOM pixel data to PIL Image.") # Log full traceback
        st.warning(f"Could not process DICOM pixel data for display: {e}")
        return None


# =============================================================================
# Main File Handling Function
# =============================================================================

def handle_file_upload(uploaded_file: Optional[st.runtime.uploaded_file_manager.UploadedFile]) -> None:
    """
    Processes the uploaded file from Streamlit's file_uploader.

    Reads the file, determines type (DICOM or image), extracts data/metadata,
    updates session state, and resets state if it's a new file.

    Args:
        uploaded_file: The file object from st.file_uploader, or None.
    """
    if uploaded_file is None:
        # No file uploaded currently, do nothing here.
        # Clearing state should happen on explicit "Clear" action or demo mode toggle.
        return

    # --- Check if this is a NEW file compared to the last processed one ---
    new_file_info = {
        "name": uploaded_file.name,
        "size": uploaded_file.size,
        "type": uploaded_file.type, # MIME type reported by browser
    }
    last_file_info = st.session_state.get("uploaded_file_info")

    if new_file_info == last_file_info:
        logger.debug("Uploaded file is the same as already processed. Skipping.")
        return # Avoid reprocessing the same file

    # --- New File Detected: Reset State and Process ---
    logger.info(f"New file uploaded: '{new_file_info['name']}' ({new_file_info['size']} bytes)")
    reset_session_state_for_new_file() # Crucial step!

    # Store info about the *current* file being processed
    st.session_state.uploaded_file_info = new_file_info
    st.session_state.display_image = None # Ensure display image is cleared initially
    st.session_state.is_dicom = False
    st.session_state.dicom_dataset = None
    st.session_state.dicom_metadata = None
    st.session_state.raw_image_bytes = None # Clear previous raw bytes


    # --- Read File Content ---
    try:
        uploaded_file.seek(0) # Ensure reading from the start
        file_bytes = uploaded_file.getvalue()
        st.session_state.raw_image_bytes = file_bytes
        file_buffer = io.BytesIO(file_bytes) # Create buffer for pydicom/PIL
        logger.debug(f"Read {len(file_bytes)} bytes from uploaded file.")
    except Exception as e:
        logger.error(f"Failed to read bytes from uploaded file '{uploaded_file.name}': {e}", exc_info=True)
        st.error(f"Error reading file: {e}")
        st.session_state.uploaded_file_info = None # Clear info as read failed
        return

    # --- Attempt DICOM Processing ---
    dicom_success = False
    if PYDICOM_AVAILABLE:
        try:
            logger.debug("Attempting to read file as DICOM...")
            # Pass the buffer directly to dcmread
            ds = pydicom.dcmread(file_buffer, force=True) # force=True helps with some minor header issues

            # Basic validation: Check for PixelData
            if not hasattr(ds, 'PixelData'):
                 raise InvalidDicomError("File parsed as DICOM but missing PixelData.")

            logger.info("Successfully read file as DICOM.")
            st.session_state.is_dicom = True
            st.session_state.dicom_dataset = ds
            st.session_state.dicom_metadata = _extract_dicom_metadata(ds)

            # Convert to PIL Image for display (using initial WC/WW from DICOM or defaults)
            st.session_state.display_image = _dicom_to_pil_image(ds) # Pass dataset, let helper handle WC/WW initially
            if st.session_state.display_image is None:
                 logger.error("Failed to create display image from DICOM pixel data.")
                 st.error("Could not visualize DICOM pixel data.")
                 # Keep DICOM metadata even if image fails
            dicom_success = True

        except InvalidDicomError:
            logger.info("File is not a valid DICOM file or is missing PixelData.")
            # Reset buffer position in case it's a standard image
            file_buffer.seek(0)
        except Exception as e:
            logger.error(f"Unexpected error during DICOM processing: {e}", exc_info=True)
            st.error(f"Error processing DICOM: {e}")
            file_buffer.seek(0) # Reset buffer position

    # --- Attempt Standard Image Processing (if not DICOM or DICOM failed) ---
    if not dicom_success and PIL_AVAILABLE:
        try:
            logger.debug("Attempting to read file as standard image (PNG, JPG, etc.)...")
            img = Image.open(file_buffer)
            # img.load() # Force loading image data to catch potential errors early

            # Convert to a standard mode if needed (e.g., P, RGBA -> RGB for compatibility)
            if img.mode == 'P': # Palette mode
                img = img.convert('RGBA') # Convert via RGBA to preserve transparency info
                logger.debug("Converted image mode P -> RGBA")
            if img.mode == 'RGBA':
                # Create a white background and paste RGBA image onto it
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
                img = background
                logger.debug("Converted image mode RGBA -> RGB by pasting on white background")
            elif img.mode != 'RGB' and img.mode != 'L': # Grayscale 'L' is okay
                 img = img.convert('RGB')
                 logger.debug(f"Converted image mode {img.mode} -> RGB")


            st.session_state.is_dicom = False # Ensure this is False
            st.session_state.display_image = img # STORE THE IMAGE!
            logger.info(f"Successfully read file as standard image (mode: {img.mode}).")

        except UnidentifiedImageError:
            error_msg = f"Cannot identify image file: '{uploaded_file.name}'. It might be corrupted or an unsupported format."
            logger.error(error_msg)
            st.error(error_msg)
            st.session_state.uploaded_file_info = None # Clear info as it's unusable
            st.session_state.display_image = None
        except Exception as e:
            error_msg = f"Error processing image file '{uploaded_file.name}': {e}"
            logger.error(error_msg, exc_info=True)
            st.error(error_msg)
            st.session_state.uploaded_file_info = None
            st.session_state.display_image = None

    elif not dicom_success and not PIL_AVAILABLE:
         logger.error("Cannot process non-DICOM file because Pillow (PIL) is not installed.")
         st.error("Image processing library (Pillow) not available. Cannot display non-DICOM images.")
         st.session_state.uploaded_file_info = None # Clear info

    # Final check if display image was set
    if st.session_state.get("display_image") is None and st.session_state.get("uploaded_file_info") is not None:
         # Only show this if we think we should have an image but don't
         logger.error(f"Processing complete for '{uploaded_file.name}', but no display_image was generated.")
         # st.error("Failed to generate a displayable image from the uploaded file.") # Avoid double errors

    logger.debug(f"File processing complete for '{uploaded_file.name}'. Display image set: {st.session_state.get('display_image') is not None}")