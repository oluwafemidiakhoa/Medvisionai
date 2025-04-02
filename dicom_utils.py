import pydicom
import pydicom.errors
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
from PIL import Image
import io
import logging
import streamlit as st
from typing import Optional, Tuple, Dict, Any, List, Union

# Assume logger is configured elsewhere, consistent with the action handling section
logger = logging.getLogger(__name__)

# --- DICOM Parsing ---

@st.cache_data(max_entries=10, show_spinner=False) # Cache parsing, hide default spinner
def parse_dicom(dicom_bytes: bytes, filename: str = "Uploaded File") -> Optional[pydicom.Dataset]:
    """
    Parses DICOM file bytes into a pydicom Dataset object.

    Args:
        dicom_bytes: The raw bytes of the DICOM file.
        filename: The original filename for logging/error messages.

    Returns:
        A pydicom Dataset object if successful, otherwise None.
    """
    logger.info(f"Attempting to parse DICOM data from '{filename}' ({len(dicom_bytes)} bytes)...")
    try:
        # Use force=True to potentially read files with minor header issues,
        # but be aware this might allow slightly non-compliant files.
        # Consider if strict compliance is necessary for your use case.
        ds = pydicom.dcmread(io.BytesIO(dicom_bytes), force=True)
        logger.info(f"Successfully parsed DICOM data from '{filename}'. SOP Class: {ds.SOPClassUID.name if 'SOPClassUID' in ds else 'Unknown'}")
        # Basic validation: check for essential tags like PixelData if it's an image type
        if 'PixelData' not in ds:
             logger.warning(f"DICOM file '{filename}' parsed but lacks PixelData tag. May not be an image.")
             # Decide if this should be an error or just a warning depending on expected file types
             # st.warning(f"'{filename}' does not contain image data (PixelData tag missing).")
             # return None # Optionally return None if PixelData is mandatory

        return ds
    except pydicom.errors.InvalidDicomError as e:
        logger.error(f"Invalid DICOM data encountered in '{filename}': {e}", exc_info=False) # Keep log concise
        st.error(f"Error parsing '{filename}': The file is not a valid DICOM file or is corrupted. Details: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred reading DICOM data from '{filename}': {e}", exc_info=True)
        st.error(f"Failed to read DICOM file '{filename}'. An unexpected error occurred.")
        return None

# --- DICOM Metadata Extraction ---

@st.cache_data(max_entries=10, show_spinner=False) # Cache metadata extraction
def extract_dicom_metadata(ds: pydicom.Dataset) -> Dict[str, Any]:
    """
    Extracts a predefined set of technical DICOM metadata tags.

    Args:
        ds: The pydicom Dataset object.

    Returns:
        A dictionary containing the values of the extracted tags.
        Note: This function extracts technical parameters and does NOT perform
              comprehensive PHI filtering. Rely on specific filtering mechanisms
              (like in the report generation) before displaying sensitive data.
    """
    logger.debug(f"Extracting predefined technical metadata for SOP Instance UID: {ds.SOPInstanceUID if 'SOPInstanceUID' in ds else 'Unknown'}")
    metadata = {}
    # Define technical tags typically safe and useful for display/processing
    # **This is NOT a PHI filter list.**
    tags_to_extract = {
        # Tag Name (for dict key) : (Group, Element)
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
        "PixelRepresentation": (0x0028, 0x0103), # 0=unsigned, 1=signed
        "SamplesPerPixel": (0x0028, 0x0002),
    }

    for name, tag_address in tags_to_extract.items():
        try:
            element = ds[tag_address]
            value = element.value
            # Handle None, empty sequences, or empty strings explicitly
            if value is None or value == "":
                metadata[name] = "N/A"
                continue

            # Nicer representation for specific types
            if isinstance(value, pydicom.uid.UID):
                display_value = value.name # Use UID name representation
            elif isinstance(value, list): # Handle multi-valued tags
                 # Convert elements to string, join, handle potential nested lists/objects simply
                 display_value = ", ".join(map(str, value))
            elif isinstance(value, pydicom.valuerep.DSfloat): # Decimal String
                 display_value = float(value)
            elif isinstance(value, pydicom.valuerep.IS): # Integer String
                 display_value = int(value)
            else:
                 display_value = value # Use the value directly for others (int, float, str)

            metadata[name] = display_value

        except KeyError:
            logger.debug(f"Metadata tag {name} ({tag_address}) not found in dataset.")
            metadata[name] = "Not Found"
        except Exception as e:
            # Log error but don't crash metadata extraction for one bad tag
            logger.warning(f"Could not read or process metadata tag {name} ({tag_address}): {e}", exc_info=False)
            metadata[name] = "Error Reading"

    logger.debug(f"Finished extracting {len(metadata)} technical metadata tags.")
    return metadata

# --- DICOM Image Conversion ---

@st.cache_data(max_entries=20, show_spinner="Processing DICOM image...") # Cache image generation, show spinner
def dicom_to_image(
    ds: pydicom.Dataset,
    window_center: Optional[Union[float, List[float]]] = None,
    window_width: Optional[Union[float, List[float]]] = None
) -> Optional[Image.Image]:
    """
    Converts DICOM pixel data to a displayable PIL Image (RGB), applying VOI LUT.

    Handles grayscale and some basic color formats. Uses provided Window/Level
    values or falls back to dataset defaults or simple min/max scaling.

    Args:
        ds: The pydicom Dataset object containing PixelData.
        window_center: Window Center (Level) value(s) for VOI LUT. Uses first if list.
        window_width: Window Width value(s) for VOI LUT. Uses first if list.

    Returns:
        A PIL Image object in RGB format, or None if processing fails.
    """
    if 'PixelData' not in ds:
        logger.error("Cannot convert to image: DICOM dataset lacks PixelData tag.")
        # No st.error here as parse_dicom might have warned already, avoid duplication
        return None

    logger.debug(f"Starting DICOM to image conversion. Photometric Interpretation: {ds.get('PhotometricInterpretation', 'N/A')}")

    try:
        pixel_array = ds.pixel_array # Access the pixel data

        # --- Determine and Apply Window/Level ---
        wc_to_use: Optional[float] = None
        ww_to_use: Optional[float] = None

        # Prioritize user-provided W/L values
        if window_center is not None and window_width is not None:
             # Handle potential multi-value inputs, take the first valid one
             wc_in = window_center[0] if isinstance(window_center, list) and window_center else window_center
             ww_in = window_width[0] if isinstance(window_width, list) and window_width else window_width
             try:
                  wc_to_use = float(wc_in) if wc_in is not None else None
                  ww_to_use = float(ww_in) if ww_in is not None else None
                  if ww_to_use is not None and ww_to_use <= 0:
                      logger.warning(f"Provided Window Width ({ww_to_use}) is invalid, ignoring.")
                      ww_to_use = None # Invalidate if non-positive width
                  elif wc_to_use is not None and ww_to_use is not None:
                       logger.info(f"Using provided WC/WW: {wc_to_use} / {ww_to_use}")
             except (ValueError, TypeError):
                  logger.warning(f"Could not convert provided WC/WW ('{wc_in}', '{ww_in}') to float, ignoring.")
                  wc_to_use = None
                  ww_to_use = None

        # If user W/L not valid or not provided, try default W/L from DICOM tags
        if wc_to_use is None or ww_to_use is None:
            default_wc, default_ww = get_default_wl(ds) # Use helper function
            if default_wc is not None and default_ww is not None:
                 wc_to_use = default_wc
                 ww_to_use = default_ww
                 logger.info(f"Using default WC/WW from DICOM tags: {wc_to_use} / {ww_to_use}")

        # --- Apply Transformation ---
        if wc_to_use is not None and ww_to_use is not None:
            logger.debug(f"Applying VOI LUT with WC={wc_to_use}, WW={ww_to_use}")
            # Ensure data type is appropriate if necessary before apply_voi_lut
            # pixel_array = pixel_array.astype(np.float64) # Sometimes needed depending on pydicom/numpy versions
            processed_array = apply_voi_lut(pixel_array, ds, window=ww_to_use, level=wc_to_use)
            # Scale result of VOI LUT to 0-255
            min_val, max_val = processed_array.min(), processed_array.max()
            if max_val > min_val:
                # Add small epsilon to prevent division by zero if max_val == min_val after LUT
                pixel_array_scaled = ((processed_array - min_val) / (max_val - min_val + 1e-6)) * 255.0
            else:
                pixel_array_scaled = np.zeros_like(processed_array)
            pixel_array_uint8 = pixel_array_scaled.astype(np.uint8)
            logger.debug("VOI LUT applied and scaled to uint8.")

        else: # Fallback to basic min/max scaling if no valid W/L found
            logger.info("No valid Window/Level found. Applying basic min/max scaling.")
            # Apply Rescale Slope/Intercept if present, as VOI LUT wasn't used
            if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
                try:
                    slope = float(ds.RescaleSlope)
                    intercept = float(ds.RescaleIntercept)
                    if slope != 1.0 or intercept != 0.0:
                        logger.debug(f"Applying Rescale Slope ({slope}) and Intercept ({intercept})")
                        # Ensure float array for calculation
                        pixel_array = pixel_array.astype(np.float64) * slope + intercept
                    else:
                        logger.debug("Rescale Slope=1, Intercept=0, no rescale needed.")
                except Exception as rescale_err:
                    logger.warning(f"Could not apply Rescale Slope/Intercept: {rescale_err}")


            min_val, max_val = pixel_array.min(), pixel_array.max()
            if max_val > min_val:
                 # Add small epsilon to prevent division by zero
                 scaled_array = ((pixel_array - min_val) / (max_val - min_val + 1e-6) * 255.0)
            else: # Handle constant image
                 scaled_array = np.zeros_like(pixel_array)
            pixel_array_uint8 = scaled_array.astype(np.uint8)
            logger.debug("Basic min/max scaling applied and converted to uint8.")


        # --- Convert numpy array to PIL Image (RGB) ---
        photometric_interpretation = ds.get("PhotometricInterpretation", "").upper()
        logger.debug(f"Array shape for PIL conversion: {pixel_array_uint8.shape}, dtype: {pixel_array_uint8.dtype}")

        if pixel_array_uint8.ndim == 2: # Grayscale image
            # Common grayscale types
            if photometric_interpretation in ("MONOCHROME1", "MONOCHROME2"):
                image = Image.fromarray(pixel_array_uint8, mode='L').convert("RGB")
                logger.debug("Converted 2D Grayscale (MONOCHROME1/2) array to RGB PIL Image.")
            else: # Other 2D formats, treat as grayscale
                 logger.warning(f"Unknown 2D Photometric Interpretation '{photometric_interpretation}'. Treating as MONOCHROME2.")
                 image = Image.fromarray(pixel_array_uint8, mode='L').convert("RGB")

        elif pixel_array_uint8.ndim == 3: # Potentially color or multi-frame
            logger.debug(f"Input array has 3 dimensions. Photometric Interpretation: {photometric_interpretation}")
            # Check samples per pixel
            samples_per_pixel = ds.get("SamplesPerPixel", 1)

            if samples_per_pixel == 3 and photometric_interpretation in ("RGB", "YBR_FULL", "YBR_FULL_422"):
                 # Planar Configuration (0=Color-by-pixel, 1=Color-by-plane)
                 planar_config = ds.get("PlanarConfiguration", 0)
                 if planar_config == 0: # Color-by-pixel (RRR...GGG...BBB...) is unusual for numpy shape but check anyway
                      if pixel_array_uint8.shape[-1] == 3:
                           image = Image.fromarray(pixel_array_uint8, mode='RGB')
                           logger.debug("Converted 3D array (SamplesPerPixel=3, PlanarConfig=0) to RGB PIL Image.")
                      else:
                           logger.warning(f"Expected shape[2]=3 for Planar Config 0, got {pixel_array_uint8.shape}. Attempting first channel.")
                           image = Image.fromarray(pixel_array_uint8[:,:,0], mode='L').convert("RGB")

                 elif planar_config == 1: # Color-by-plane (RRR... GGG... BBB...) - needs reshaping
                     if pixel_array_uint8.shape[0] == 3: # Check if first dimension is 3 (planes)
                         logger.debug("Reshaping 3D array (PlanarConfig=1) for RGB conversion.")
                         # Reshape from (3, rows, cols) to (rows, cols, 3)
                         reshaped_array = np.transpose(pixel_array_uint8, (1, 2, 0))
                         image = Image.fromarray(reshaped_array, mode='RGB')
                         logger.debug("Converted 3D array (SamplesPerPixel=3, PlanarConfig=1) to RGB PIL Image.")
                     else:
                          logger.warning(f"Expected shape[0]=3 for Planar Config 1, got {pixel_array_uint8.shape}. Attempting first plane.")
                          image = Image.fromarray(pixel_array_uint8[0,:,:], mode='L').convert("RGB")
                 else: # Fallback if PlanarConfiguration is invalid or missing
                      logger.warning(f"Unexpected Planar Configuration ({planar_config}). Assuming color-by-pixel if last dim is 3.")
                      if pixel_array_uint8.shape[-1] == 3:
                           image = Image.fromarray(pixel_array_uint8, mode='RGB')
                      else: # Fallback to first channel/slice
                           logger.warning("Falling back to first channel/slice as grayscale.")
                           image = Image.fromarray(pixel_array_uint8[:,:,0] if pixel_array_uint8.shape[-1] != 3 else pixel_array_uint8[0,:,:], mode='L').convert("RGB")

            elif samples_per_pixel == 1 and pixel_array_uint8.ndim == 3: # Likely multi-frame grayscale
                logger.info("Detected 3D array with SamplesPerPixel=1. Displaying first frame.")
                image = Image.fromarray(pixel_array_uint8[0,:,:], mode='L').convert("RGB") # Display first frame
            else: # Unknown 3D format
                logger.warning(f"Unsupported 3D array format (Samples={samples_per_pixel}, PI='{photometric_interpretation}', ndim={pixel_array_uint8.ndim}). Attempting first slice/channel.")
                # Try slicing based on likely dimension order
                try:
                     if pixel_array_uint8.shape[0] > 1 and pixel_array_uint8.shape[0] < 5: # Likely planes first
                         image = Image.fromarray(pixel_array_uint8[0,:,:], mode='L').convert("RGB")
                     elif pixel_array_uint8.shape[-1] > 1 and pixel_array_uint8.shape[-1] < 5: # Likely channels last
                          image = Image.fromarray(pixel_array_uint8[:,:,0], mode='L').convert("RGB")
                     else: # Default guess: first slice/frame
                          image = Image.fromarray(pixel_array_uint8[0,:,:], mode='L').convert("RGB")
                except IndexError:
                     logger.error("Could not extract a 2D slice from the 3D array for display.")
                     return None


        else: # Unsupported dimensions
            logger.error(f"Cannot convert to image: Unsupported pixel array dimensions ({pixel_array_uint8.ndim})")
            st.warning("Failed to process DICOM image data due to unsupported array dimensions.")
            return None

        logger.info(f"Successfully converted DICOM to PIL Image (RGB format, size: {image.size}).")
        return image

    except AttributeError as e:
         # Often happens if ds is None or essential tags missing after force=True parsing
         logger.error(f"AttributeError during image conversion, likely missing DICOM tag: {e}", exc_info=False)
         st.warning(f"Failed to process image data: Required DICOM information missing ({e}).")
         return None
    except Exception as e:
        logger.error(f"Unexpected error converting DICOM pixel data to image: {e}", exc_info=True)
        st.warning(f"An unexpected error occurred while processing DICOM image data.")
        return None

# --- Window/Level Helper ---

def get_default_wl(ds: pydicom.Dataset) -> Tuple[Optional[float], Optional[float]]:
    """
    Safely retrieves default Window Center (Level) and Width from DICOM tags.

    Handles missing tags, multi-value entries (takes first), and non-numeric values.

    Args:
        ds: The pydicom Dataset object.

    Returns:
        A tuple containing (WindowCenter, WindowWidth), both Optional[float].
        Returns (None, None) if values are not found or invalid.
    """
    wc_val = ds.get("WindowCenter", None)
    ww_val = ds.get("WindowWidth", None)
    wc: Optional[float] = None
    ww: Optional[float] = None

    # Extract first value if multi-valued
    if isinstance(wc_val, pydicom.multival.MultiValue):
        wc_val = wc_val[0] if len(wc_val) > 0 else None
    if isinstance(ww_val, pydicom.multival.MultiValue):
        ww_val = ww_val[0] if len(ww_val) > 0 else None

    # Convert safely to float
    if wc_val is not None:
        try:
            wc = float(wc_val)
        except (ValueError, TypeError):
            logger.debug(f"Could not convert default WindowCenter ('{wc_val}') to float.")
            wc = None # Invalid format
    if ww_val is not None:
         try:
            ww = float(ww_val)
            # Basic sanity check for width
            if ww <= 0:
                 logger.debug(f"Invalid default WindowWidth found ({ww}), ignoring.")
                 ww = None
         except (ValueError, TypeError):
            logger.debug(f"Could not convert default WindowWidth ('{ww_val}') to float.")
            ww = None # Invalid format

    if wc is not None and ww is not None:
         logger.debug(f"Found default WC/WW in DICOM tags: {wc} / {ww}")
         return wc, ww
    else:
         logger.debug("Default WC/WW not found or invalid in DICOM tags.")
         return None, None