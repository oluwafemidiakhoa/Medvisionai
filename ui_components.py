# ui_helpers.py (Example filename)

import streamlit as st
from typing import Optional, Tuple, Dict, Any, List, Union
import pydicom
import pydicom.valuerep # Import for specific types like PersonName
import numpy as np
import logging

# Assume logger is configured elsewhere
logger = logging.getLogger(__name__)

# --- DICOM Metadata Display ---

def display_dicom_metadata(metadata: Optional[Dict[str, Any]]):
    """
    Displays formatted DICOM metadata in a Streamlit expander, arranged in columns.

    Args:
        metadata: A dictionary containing DICOM tags (keys) and their values.
                  Handles basic formatting for lists, UIDs, and bytes.
                  If None or empty, displays a placeholder message.
    """
    with st.expander("View DICOM Metadata", expanded=False):
        if not metadata:
            st.caption("No metadata extracted or available.")
            return

        cols = st.columns(2)
        col_idx = 0
        logger.debug(f"Displaying {len(metadata)} metadata items.")

        for key, value in metadata.items():
            display_value = "N/A" # Default display value
            try:
                # Format specific types for better readability
                if value is None:
                    display_value = "N/A"
                elif isinstance(value, list):
                    # Join list elements, ensuring individual elements are strings
                    display_value = ", ".join(map(str, value))
                elif isinstance(value, pydicom.uid.UID):
                    # Show the descriptive name for UIDs
                    display_value = value.name
                elif isinstance(value, bytes):
                    # Avoid displaying large byte strings directly
                    display_value = f"[Binary Data ({len(value)} bytes)]"
                elif isinstance(value, pydicom.valuerep.PersonName):
                    # Potentially sensitive, display generic placeholder or handle based on policy
                    display_value = "[Person Name]"
                else:
                    # Default to string representation, strip whitespace
                    display_value = str(value).strip()

                # Truncate very long strings for display
                if len(display_value) > 150:
                    display_value = display_value[:147] + "..."

            except Exception as e:
                logger.warning(f"Error formatting metadata key '{key}' for display: {e}", exc_info=False)
                display_value = "[Error formatting value]"

            # Use markdown for bold key and plain value
            cols[col_idx % 2].markdown(f"**{key}:** {display_value}")
            col_idx += 1

# --- DICOM Window/Level Sliders ---

def dicom_wl_sliders(
    ds: Optional[pydicom.Dataset],
    metadata: Dict[str, Any] # Assumes metadata is already extracted
    ) -> Tuple[Optional[float], Optional[float]]:
    """
    Creates Streamlit sliders for adjusting DICOM Window Center (Level) and Width.

    Derives slider ranges and defaults from the dataset's pixel data and metadata.
    Includes a button to reset sliders to their initial default values.

    Args:
        ds: The pydicom Dataset object (must contain PixelData).
        metadata: Dictionary containing extracted metadata, used for default W/L.

    Returns:
        A tuple (window_center, window_width) representing the current slider
        values as floats. Returns (None, None) if sliders cannot be created.
    """
    st.subheader("DICOM Window/Level Adjustment")

    if ds is None or 'PixelData' not in ds:
        st.caption("Cannot create W/L sliders: DICOM data or PixelData missing.")
        logger.warning("dicom_wl_sliders called with missing Dataset or PixelData.")
        return None, None

    # --- Determine Pixel Range ---
    pixel_min: float = 0.0
    pixel_max: float = 4095.0 # Fallback default range
    pixel_range_determined = False
    try:
        pixel_array = ds.pixel_array
        # Apply Rescale Slope/Intercept if present for accurate range
        if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
             slope = float(ds.RescaleSlope)
             intercept = float(ds.RescaleIntercept)
             logger.debug(f"Applying Rescale Slope ({slope}) / Intercept ({intercept}) for range calculation.")
             # Calculate on float copy to avoid modifying original array type if slope/intercept are floats
             rescaled_array = pixel_array.astype(np.float64) * slope + intercept
             pixel_min = float(rescaled_array.min())
             pixel_max = float(rescaled_array.max())
        else:
             pixel_min = float(pixel_array.min())
             pixel_max = float(pixel_array.max())

        pixel_range_determined = True
        logger.info(f"Determined pixel value range: Min={pixel_min}, Max={pixel_max}")
        if pixel_max == pixel_min:
             logger.warning("Pixel data range is zero (constant image). Sliders may not be meaningful.")
             # Adjust range slightly to avoid zero width/division issues
             pixel_max += 1.0

    except AttributeError:
         st.caption("Pixel data format not directly accessible (e.g., compressed). Using default range.")
         logger.warning("Could not directly access pixel_array (possibly compressed), using default range for sliders.")
    except Exception as e:
        st.caption(f"Could not determine pixel range (Error: {e}). Using default range.")
        logger.error(f"Error determining pixel range for sliders: {e}", exc_info=True)

    # --- Get and Validate Default W/L from Metadata ---
    default_wc_raw = metadata.get("WindowCenter", None)
    default_ww_raw = metadata.get("WindowWidth", None)
    default_wc: Optional[float] = None
    default_ww: Optional[float] = None

    # Helper to safely convert potential multi-value or string to float
    def safe_float_convert(value: Any) -> Optional[float]:
        if isinstance(value, (list, pydicom.multival.MultiValue)):
            val_to_convert = value[0] if len(value) > 0 else None
        else:
            val_to_convert = value
        if val_to_convert is None: return None
        try: return float(val_to_convert)
        except (ValueError, TypeError): return None

    default_wc = safe_float_convert(default_wc_raw)
    default_ww = safe_float_convert(default_ww_raw)

    # Use calculated center/width if defaults are missing or invalid
    calculated_center = (pixel_max + pixel_min) / 2.0
    # Use 80% of range or a fallback fixed width if range is very small/zero
    calculated_width = max(1.0, (pixel_max - pixel_min) * 0.8)

    if default_wc is None:
        default_wc = calculated_center
        logger.debug(f"Using calculated default Window Center: {default_wc:.2f}")
    if default_ww is None or default_ww <= 0:
        default_ww = calculated_width
        logger.debug(f"Using calculated default Window Width: {default_ww:.2f}")

    logger.info(f"Using defaults for sliders - WC: {default_wc:.2f}, WW: {default_ww:.2f}")

    # --- Calculate Sensible Slider Bounds ---
    data_range = pixel_max - pixel_min
    # Allow sliders to go slightly beyond the data range, but not excessively
    slider_min_level = pixel_min - data_range * 0.5 # Extend 50% below min
    slider_max_level = pixel_max + data_range * 0.5 # Extend 50% above max
    # Cap maximum width to avoid extreme values, e.g., 2x the data range or a fixed large value
    slider_max_width = max(1.0, data_range * 2.0)
    # Add absolute cap in case data_range is huge (e.g. for float data)
    slider_max_width = min(slider_max_width, 65536.0) # Example cap

    # Clamp default values to be within the calculated slider bounds
    clamped_default_wc = max(slider_min_level, min(slider_max_level, default_wc))
    clamped_default_ww = max(1.0, min(slider_max_width, default_ww))

    # --- Create Sliders ---
    # Use unique keys based on session state if these sliders persist across reruns
    # For simplicity here, using fixed keys assuming they are recreated each time needed
    wc = st.slider(
        "Window Center (Level)",
        min_value=slider_min_level,
        max_value=slider_max_level,
        value=clamped_default_wc,
        step=max(0.1, data_range / 1000.0), # Dynamic step based on range, minimum 0.1
        key="dicom_wc_slider",
        help=f"Adjusts the brightness center. Range based on pixel data [{pixel_min:.1f} - {pixel_max:.1f}]"
    )
    ww = st.slider(
        "Window Width",
        min_value=1.0, # Width must be positive
        max_value=slider_max_width,
        value=clamped_default_ww,
        step=max(0.1, data_range / 1000.0), # Dynamic step based on range
        key="dicom_ww_slider",
        help=f"Adjusts the contrast range. Based on pixel data range [{data_range:.1f}]"
    )

    # --- Reset Button ---
    if st.button("Reset W/L", key="reset_wl_button"):
         logger.info("Reset W/L button clicked. Triggering rerun to apply defaults.")
         # Clear potentially cached W/L values from session state if they are stored there
         # st.session_state.pop('manual_wc', None)
         # st.session_state.pop('manual_ww', None)
         # Rerun will cause sliders to re-render with their default values calculated above
         st.rerun()

    # Return the current values from the sliders
    return float(wc), float(ww)