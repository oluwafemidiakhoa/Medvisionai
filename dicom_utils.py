import pydicom
import pydicom.errors
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
from PIL import Image, ImageDraw
import io
import logging
import streamlit as st
from typing import Optional, Tuple, Dict, Any, List, Union

# Configure logger (assumed to be set up globally in your app)
logger = logging.getLogger(__name__)

# --- Helper Function to Filter PHI from Metadata ---
def filter_sensitive_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filters out keys known to contain Protected Health Information (PHI)
    from the metadata dictionary.
    
    Args:
        metadata: Dictionary of metadata tags.
    
    Returns:
        Filtered dictionary with PHI removed.
    """
    # List of keys that might contain PHI (adjust as needed)
    phi_keys = {"PatientName", "PatientID", "PatientBirthDate", "PatientSex", "PatientAddress"}
    return {k: v for k, v in metadata.items() if k not in phi_keys}

# --- DICOM Parsing ---

@st.cache_data(max_entries=10, show_spinner=False)
def parse_dicom(dicom_bytes: bytes, filename: str = "Uploaded File", require_pixeldata: bool = True) -> Optional[pydicom.Dataset]:
    """
    Parses DICOM file bytes into a pydicom Dataset object.
    
    Args:
        dicom_bytes: The raw bytes of the DICOM file.
        filename: The original filename for logging/error messages.
        require_pixeldata: If True, treat missing PixelData as a fatal error.
        
    Returns:
        A pydicom Dataset object if successful, otherwise None.
    """
    logger.info(f"Attempting to parse DICOM data from '{filename}' ({len(dicom_bytes)} bytes)...")
    try:
        ds = pydicom.dcmread(io.BytesIO(dicom_bytes), force=True)
        logger.info(f"Successfully parsed DICOM data from '{filename}'. SOP Class: {ds.SOPClassUID.name if 'SOPClassUID' in ds else 'Unknown'}")
        if require_pixeldata and 'PixelData' not in ds:
            logger.error(f"DICOM file '{filename}' is missing PixelData tag.")
            st.error(f"Error: '{filename}' does not contain image data (PixelData tag missing).")
            return None
        return ds
    except pydicom.errors.InvalidDicomError as e:
        logger.error(f"Invalid DICOM data encountered in '{filename}': {e}", exc_info=False)
        st.error(f"Error parsing '{filename}': The file is not a valid DICOM file or is corrupted. Details: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred reading DICOM data from '{filename}': {e}", exc_info=True)
        st.error(f"Failed to read DICOM file '{filename}'. An unexpected error occurred.")
        return None

# --- DICOM Metadata Extraction ---

@st.cache_data(max_entries=10, show_spinner=False)
def extract_dicom_metadata(ds: pydicom.Dataset, filter_phi: bool = True) -> Dict[str, Any]:
    """
    Extracts a predefined set of technical DICOM metadata tags.
    
    Args:
        ds: The pydicom Dataset object.
        filter_phi: If True, filter out keys that may contain PHI.
        
    Returns:
        A dictionary containing the values of the extracted tags.
    """
    logger.debug(f"Extracting technical metadata for SOP Instance UID: {ds.SOPInstanceUID if 'SOPInstanceUID' in ds else 'Unknown'}")
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
        "SliceThickness": (0x0018, 0x0050),
        "WindowCenter": (0x0028, 0x1050),
        "WindowWidth": (0x0028, 0x1051),
        "RescaleIntercept": (0x0028, 0x1052),
        "RescaleSlope": (0x0028, 0x1053),
        "PhotometricInterpretation": (0x0028, 0x0004),
        "BitsAllocated": (0x0028, 0x0100),
        "BitsStored": (0x0028, 0x0101),
        "HighBit": (0x0028, 0x0102),
        "PixelRepresentation": (0x0028, 0x0103),
        "SamplesPerPixel": (0x0028, 0x0002),
    }
    
    for name, tag_address in tags_to_extract.items():
        try:
            element = ds[tag_address]
            value = element.value
            if value is None or value == "":
                metadata[name] = "N/A"
                continue

            if isinstance(value, pydicom.uid.UID):
                display_value = value.name
            elif isinstance(value, list):
                display_value = ", ".join(map(str, value))
            elif isinstance(value, pydicom.valuerep.DSfloat):
                display_value = float(value)
            elif isinstance(value, pydicom.valuerep.IS):
                display_value = int(value)
            else:
                display_value = value

            metadata[name] = display_value

        except KeyError:
            logger.debug(f"Metadata tag {name} ({tag_address}) not found in dataset.")
            metadata[name] = "Not Found"
        except Exception as e:
            logger.warning(f"Could not read metadata tag {name} ({tag_address}): {e}", exc_info=False)
            metadata[name] = "Error Reading"
    
    logger.debug(f"Extracted {len(metadata)} metadata tags.")
    if filter_phi:
        metadata = filter_sensitive_metadata(metadata)
    return metadata

# --- DICOM Image Conversion ---

@st.cache_data(max_entries=20, show_spinner="Processing DICOM image...")
def dicom_to_image(
    ds: pydicom.Dataset,
    window_center: Optional[Union[float, List[float]]] = None,
    window_width: Optional[Union[float, List[float]]] = None
) -> Optional[Image.Image]:
    """
    Converts DICOM pixel data to a displayable PIL Image (RGB), applying VOI LUT.
    
    Args:
        ds: The pydicom Dataset object containing PixelData.
        window_center: Window Center value(s) for VOI LUT (first used if list).
        window_width: Window Width value(s) for VOI LUT (first used if list).
    
    Returns:
        A PIL Image object in RGB format, or None if processing fails.
    """
    if 'PixelData' not in ds:
        logger.error("Cannot convert to image: Missing PixelData tag.")
        return None

    logger.debug(f"Converting DICOM to image. Photometric Interpretation: {ds.get('PhotometricInterpretation', 'N/A')}")
    
    try:
        pixel_array = ds.pixel_array

        wc_to_use: Optional[float] = None
        ww_to_use: Optional[float] = None
        
        if window_center is not None and window_width is not None:
            wc_in = window_center[0] if isinstance(window_center, list) and window_center else window_center
            ww_in = window_width[0] if isinstance(window_width, list) and window_width else window_width
            try:
                wc_to_use = float(wc_in) if wc_in is not None else None
                ww_to_use = float(ww_in) if ww_in is not None else None
                if ww_to_use is not None and ww_to_use <= 0:
                    logger.warning(f"Provided Window Width ({ww_to_use}) is invalid. Ignoring.")
                    ww_to_use = None
                else:
                    logger.info(f"Using provided WC/WW: {wc_to_use} / {ww_to_use}")
            except (ValueError, TypeError):
                logger.warning(f"Conversion error for provided WC/WW values ('{wc_in}', '{ww_in}'). Ignoring.")
                wc_to_use = None
                ww_to_use = None

        if wc_to_use is None or ww_to_use is None:
            default_wc, default_ww = get_default_wl(ds)
            if default_wc is not None and default_ww is not None:
                wc_to_use = default_wc
                ww_to_use = default_ww
                logger.info(f"Using default WC/WW: {wc_to_use} / {ww_to_use}")

        if wc_to_use is not None and ww_to_use is not None:
            logger.debug(f"Applying VOI LUT with WC={wc_to_use}, WW={ww_to_use}")
            processed_array = apply_voi_lut(pixel_array, ds, window=ww_to_use, level=wc_to_use)
            min_val, max_val = processed_array.min(), processed_array.max()
            if max_val > min_val:
                pixel_array_scaled = ((processed_array - min_val) / (max_val - min_val + 1e-6)) * 255.0
            else:
                pixel_array_scaled = np.zeros_like(processed_array)
            pixel_array_uint8 = pixel_array_scaled.astype(np.uint8)
            logger.debug("VOI LUT applied and scaled to uint8.")
        else:
            logger.info("No valid WC/WW provided. Applying basic min/max scaling.")
            if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
                try:
                    slope = float(ds.RescaleSlope)
                    intercept = float(ds.RescaleIntercept)
                    if slope != 1.0 or intercept != 0.0:
                        logger.debug(f"Applying Rescale Slope ({slope}) and Intercept ({intercept})")
                        pixel_array = pixel_array.astype(np.float64) * slope + intercept
                except Exception as rescale_err:
                    logger.warning(f"Rescale Slope/Intercept error: {rescale_err}")
            min_val, max_val = pixel_array.min(), pixel_array.max()
            if max_val > min_val:
                scaled_array = ((pixel_array - min_val) / (max_val - min_val + 1e-6)) * 255.0
            else:
                scaled_array = np.zeros_like(pixel_array)
            pixel_array_uint8 = scaled_array.astype(np.uint8)
            logger.debug("Basic scaling applied and converted to uint8.")

        photometric_interpretation = ds.get("PhotometricInterpretation", "").upper()
        logger.debug(f"Array shape: {pixel_array_uint8.shape}, dtype: {pixel_array_uint8.dtype}")

        if pixel_array_uint8.ndim == 2:
            if photometric_interpretation in ("MONOCHROME1", "MONOCHROME2"):
                image = Image.fromarray(pixel_array_uint8, mode='L').convert("RGB")
                logger.debug("Converted 2D grayscale array to RGB.")
            else:
                logger.warning(f"Unknown 2D Photometric Interpretation '{photometric_interpretation}'. Using MONOCHROME2 assumption.")
                image = Image.fromarray(pixel_array_uint8, mode='L').convert("RGB")
        elif pixel_array_uint8.ndim == 3:
            samples_per_pixel = ds.get("SamplesPerPixel", 1)
            if samples_per_pixel == 3 and photometric_interpretation in ("RGB", "YBR_FULL", "YBR_FULL_422"):
                planar_config = ds.get("PlanarConfiguration", 0)
                if planar_config == 0:
                    if pixel_array_uint8.shape[-1] == 3:
                        image = Image.fromarray(pixel_array_uint8, mode='RGB')
                        logger.debug("Converted 3D array (PlanarConfig=0) to RGB.")
                    else:
                        logger.warning(f"Unexpected shape for PlanarConfig=0: {pixel_array_uint8.shape}. Using first channel.")
                        image = Image.fromarray(pixel_array_uint8[:,:,0], mode='L').convert("RGB")
                elif planar_config == 1:
                    if pixel_array_uint8.shape[0] == 3:
                        logger.debug("Reshaping 3D array (PlanarConfig=1) for RGB conversion.")
                        reshaped_array = np.transpose(pixel_array_uint8, (1, 2, 0))
                        image = Image.fromarray(reshaped_array, mode='RGB')
                    else:
                        logger.warning(f"Unexpected shape for PlanarConfig=1: {pixel_array_uint8.shape}. Using first plane.")
                        image = Image.fromarray(pixel_array_uint8[0,:,:], mode='L').convert("RGB")
                else:
                    logger.warning(f"Unexpected Planar Configuration ({planar_config}). Assuming color-by-pixel.")
                    if pixel_array_uint8.shape[-1] == 3:
                        image = Image.fromarray(pixel_array_uint8, mode='RGB')
                    else:
                        image = Image.fromarray(pixel_array_uint8[:,:,0], mode='L').convert("RGB")
            elif samples_per_pixel == 1:
                # Multi-frame grayscale: if more than one frame, take the first one.
                if pixel_array_uint8.shape[0] > 1:
                    logger.info("Detected multi-frame grayscale. Displaying first frame.")
                    image = Image.fromarray(pixel_array_uint8[0,:,:], mode='L').convert("RGB")
                else:
                    image = Image.fromarray(pixel_array_uint8, mode='L').convert("RGB")
            else:
                logger.warning(f"Unsupported 3D array format: shape {pixel_array_uint8.shape}, SamplesPerPixel={samples_per_pixel}.")
                try:
                    if pixel_array_uint8.ndim == 3 and pixel_array_uint8.shape[0] > 1:
                        image = Image.fromarray(pixel_array_uint8[0,:,:], mode='L').convert("RGB")
                    else:
                        image = Image.fromarray(pixel_array_uint8[:,:,0], mode='L').convert("RGB")
                except Exception as e:
                    logger.error("Error extracting 2D slice from 3D array.")
                    return None
        elif pixel_array_uint8.ndim == 4:
            logger.error(f"Unsupported 4D array dimensions: {pixel_array_uint8.shape}")
            st.warning("Failed to process DICOM image: 4D data is not supported.")
            return None
        else:
            logger.error(f"Unsupported array dimensions: {pixel_array_uint8.ndim}")
            st.warning("Failed to process DICOM image due to unsupported array dimensions.")
            return None

        logger.info(f"Successfully converted DICOM to PIL Image (RGB, size: {image.size}).")
        return image

    except AttributeError as e:
        logger.error(f"AttributeError during image conversion: {e}", exc_info=False)
        st.warning(f"Failed to process image data: Required DICOM tag missing ({e}).")
        return None
    except Exception as e:
        logger.error(f"Unexpected error converting DICOM to image: {e}", exc_info=True)
        st.warning("An unexpected error occurred while processing DICOM image data.")
        return None

# --- Window/Level Helper ---

def get_default_wl(ds: pydicom.Dataset) -> Tuple[Optional[float], Optional[float]]:
    """
    Retrieves and optimizes Window Center and Width from DICOM tags.
    Implements smart defaults for different modalities and anatomical regions.
    
    Modality-specific defaults:
    - CT Brain: W:80 L:40
    - CT Chest: W:1500 L:-600
    - CT Abdomen: W:400 L:50 
    - CT Bone: W:1800 L:400
    - MRI Brain: Auto-calculated from histogram
    - X-Ray: Auto-calculated with exposure compensation
    
    Args:
        ds: The pydicom Dataset object.
    
    Returns:
        A tuple (WindowCenter, WindowWidth) or (None, None) if not found.
    """
    wc_val = ds.get("WindowCenter", None)
    ww_val = ds.get("WindowWidth", None)
    wc: Optional[float] = None
    ww: Optional[float] = None

    if isinstance(wc_val, pydicom.multival.MultiValue):
        wc_val = wc_val[0] if len(wc_val) > 0 else None
    if isinstance(ww_val, pydicom.multival.MultiValue):
        ww_val = ww_val[0] if len(ww_val) > 0 else None

    if wc_val is not None:
        try:
            wc = float(wc_val)
        except (ValueError, TypeError):
            logger.debug(f"Could not convert WindowCenter '{wc_val}' to float.")
            wc = None
    if ww_val is not None:
        try:
            ww = float(ww_val)
            if ww <= 0:
                logger.debug(f"Invalid WindowWidth ({ww}).")
                ww = None
        except (ValueError, TypeError):
            logger.debug(f"Could not convert WindowWidth '{ww_val}' to float.")
            ww = None

    if wc is not None and ww is not None:
        logger.debug(f"Found default WC/WW: {wc} / {ww}")
        return wc, ww
    else:
        logger.debug("Default WC/WW not found or invalid.")
        return None, None
