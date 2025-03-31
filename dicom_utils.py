import pydicom
import pydicom.errors
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
from PIL import Image
import io
import logging
import streamlit as st
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

@st.cache_data(max_entries=10) # Cache DICOM parsing
def parse_dicom(dicom_bytes: bytes) -> Optional[pydicom.Dataset]:
    """Parses DICOM bytes into a pydicom Dataset object."""
    try:
        ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
        return ds
    except pydicom.errors.InvalidDicomError as e:
        logger.error(f"Invalid DICOM data: {e}")
        st.warning("The uploaded file is not a valid DICOM file or is corrupted.")
        return None
    except Exception as e:
        logger.error(f"Error reading DICOM data: {e}", exc_info=True)
        st.warning(f"Failed to read DICOM data: {e}")
        return None

@st.cache_data(max_entries=10) # Cache metadata extraction
def extract_dicom_metadata(ds: pydicom.Dataset) -> Dict[str, Any]:
    """Extracts selected, non-sensitive DICOM metadata."""
    metadata = {}
    tags_to_extract = {
        "Modality": (0x0008, 0x0060),
        "StudyDescription": (0x0008, 0x1030),
        "SeriesDescription": (0x0008, 0x103E),
        "PatientPosition": (0x0018, 0x5100),
        "Manufacturer": (0x0008, 0x0070),
        "ManufacturerModelName": (0x0008, 0x1090),
        "Rows": (0x0028, 0x0010),
        "Columns": (0x0028, 0x0011),
        "PixelSpacing": (0x0028, 0x0030),
        "WindowCenter": (0x0028, 0x1050),
        "WindowWidth": (0x0028, 0x1051),
        "PhotometricInterpretation": (0x0028, 0x0004),
    }

    for name, tag in tags_to_extract.items():
        try:
            element = ds[tag]
            # Handle multi-value tags appropriately
            if element.VM > 1:
                metadata[name] = list(element.value) if element.value else "N/A"
            else:
                metadata[name] = element.value if element.value else "N/A"
        except KeyError:
            metadata[name] = "Not Found"
        except Exception as e:
            logger.warning(f"Could not read tag {name} ({tag}): {e}")
            metadata[name] = "Error Reading"

    # Sanitize None values
    for key, value in metadata.items():
        if value is None:
            metadata[key] = "N/A"

    return metadata


@st.cache_data(max_entries=50) # Cache image generation based on bytes and W/L
def dicom_to_image(
    ds: pydicom.Dataset,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None
) -> Optional[Image.Image]:
    """
    Converts a pydicom Dataset to a PIL Image object (RGB), applying W/L.

    Args:
        ds: The pydicom Dataset.
        window_center: Window Center value for VOI LUT.
        window_width: Window Width value for VOI LUT.

    Returns:
        A PIL Image object (RGB) or None if processing fails.
    """
    if 'PixelData' not in ds:
        logger.error("DICOM dataset does not contain PixelData.")
        return None

    try:
        # Use pydicom's built-in VOI LUT application if W/L provided
        if window_center is not None and window_width is not None and window_width > 0:
            logger.info(f"Applying VOI LUT with WC: {window_center}, WW: {window_width}")
            # Ensure data type is suitable for apply_voi_lut if needed (often handles it)
            pixel_array = apply_voi_lut(ds.pixel_array, ds, window=window_width, level=window_center)
            # Scale to 0-255 after applying VOI LUT
            min_val, max_val = pixel_array.min(), pixel_array.max()
            if max_val > min_val:
                 pixel_array_scaled = ((pixel_array - min_val) / (max_val - min_val) * 255.0)
            else: # Avoid division by zero
                 pixel_array_scaled = np.zeros_like(pixel_array)

            pixel_array_uint8 = pixel_array_scaled.astype(np.uint8)

        else: # Fallback to basic scaling if no W/L or invalid W/L
            logger.info("No/Invalid W/L provided, using basic min/max scaling.")
            pixel_array = ds.pixel_array
            min_val, max_val = pixel_array.min(), pixel_array.max()
            if max_val == min_val:
                scaled_array = np.zeros_like(pixel_array, dtype=np.uint8)
            else:
                scaled_array = ((pixel_array - min_val) / (max_val - min_val) * 255.0)
            pixel_array_uint8 = scaled_array.astype(np.uint8)

        # --- Convert to RGB PIL Image ---
        if pixel_array_uint8.ndim == 2: # Grayscale
            image = Image.fromarray(pixel_array_uint8).convert("RGB")
        elif pixel_array_uint8.ndim == 3: # Multi-channel (assume first channel if RGB fails)
             logger.info(f"DICOM pixel array has 3 dimensions (shape: {pixel_array_uint8.shape}).")
             try:
                 image = Image.fromarray(pixel_array_uint8, 'RGB')
             except ValueError:
                 logger.warning("Direct RGB conversion failed. Using first channel as grayscale.")
                 if pixel_array_uint8.shape[2] >= 1:
                     image = Image.fromarray(pixel_array_uint8[:,:,0]).convert("RGB")
                 else: raise ValueError("Cannot process 3D array.")
        else:
            logger.error(f"Unsupported pixel array dimensions: {pixel_array_uint8.ndim}")
            return None

        return image

    except Exception as e:
        logger.error(f"Error converting DICOM pixel data to image: {e}", exc_info=True)
        st.warning(f"Failed to process DICOM pixel data: {e}")
        return None

def get_default_wl(ds: pydicom.Dataset) -> Tuple[Optional[float], Optional[float]]:
    """Gets default Window Center and Width from DICOM tags."""
    wc = ds.get("WindowCenter", None)
    ww = ds.get("WindowWidth", None)

    # Handle multi-value tags (use the first value)
    if isinstance(wc, pydicom.multival.MultiValue):
        wc = wc[0] if len(wc) > 0 else None
    if isinstance(ww, pydicom.multival.MultiValue):
        ww = ww[0] if len(ww) > 0 else None

    # Convert to float if possible
    try:
        wc = float(wc) if wc is not None else None
    except (ValueError, TypeError):
        wc = None
    try:
        ww = float(ww) if ww is not None else None
    except (ValueError, TypeError):
        ww = None

    # Basic sanity check
    if ww is not None and ww <= 0:
        logger.warning(f"Invalid default WindowWidth ({ww}), ignoring.")
        ww = None

    return wc, ww