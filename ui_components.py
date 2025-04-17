# -*- coding: utf-8 -*-
"""
ui_components.py - Streamlit UI helper components for RadVision AI.

Provides functions for displaying DICOM metadata, creating W/L sliders,
and showing UMLS concept search results.
"""

import streamlit as st
from typing import Optional, Tuple, Dict, Any, List, Union
import pydicom
import pydicom.valuerep
import numpy as np
import logging

# --- Import UMLSConcept ---
# Ensure umls_utils.py is accessible in the Python path
try:
    from umls_utils import UMLSConcept
    UMLS_CONCEPT_IMPORTED = True
except ImportError:
    # Define a dummy class if import fails, so the function signature still works
    # but the display function will show a warning.
    class UMLSConcept:
        def __init__(self, name="Import Error", ui="", uri="", rootSource=""):
            self.name = name
            self.ui = ui
            self.uri = uri
            self.rootSource = rootSource
    UMLS_CONCEPT_IMPORTED = False
    logging.warning("Could not import UMLSConcept from umls_utils. UMLS display might be limited.")


logger = logging.getLogger(__name__)

# --- DICOM Metadata Display ---

def display_dicom_metadata(metadata: Optional[Dict[str, Any]]) -> None:
    """
    Displays formatted DICOM metadata in a Streamlit expander, arranged in two columns.

    Args:
        metadata: A dictionary containing extracted DICOM tag names and their values.
    """
    with st.expander("View DICOM Metadata", expanded=False):
        if not metadata:
            st.caption("No metadata extracted or available.")
            return

        cols = st.columns(2)
        col_idx = 0
        logger.debug(f"Displaying {len(metadata)} metadata items.")

        # Sort metadata alphabetically by key for consistent display
        sorted_metadata = sorted(metadata.items())

        for key, value in sorted_metadata:
            display_value = "N/A"
            try:
                if value is None:
                    display_value = "N/A"
                elif isinstance(value, list):
                    # Handle lists, including potentially mixed types or PersonName objects
                    display_value = ", ".join(
                        str(v.original_string) if isinstance(v, pydicom.valuerep.PersonName) else str(v)
                        for v in value
                    )
                elif isinstance(value, pydicom.uid.UID):
                    display_value = f"{value.name} ({value})" # Show name and UID string
                elif isinstance(value, bytes):
                    display_value = f"[Binary Data ({len(value)} bytes)]"
                elif isinstance(value, pydicom.valuerep.PersonName):
                     # Use original_string for a simpler representation if available
                    display_value = value.original_string if hasattr(value, 'original_string') else str(value)
                elif isinstance(value, (pydicom.valuerep.DSfloat, pydicom.valuerep.IS)):
                     display_value = str(value) # Ensure numeric types are strings
                else:
                    display_value = str(value).strip()

                # Truncate very long values
                if len(display_value) > 150:
                    display_value = display_value[:147] + "..."

            except Exception as e:
                logger.warning(f"Error formatting metadata key '{key}' (Value: {value}): {e}", exc_info=False) # Less verbose logging
                display_value = "[Error formatting value]"

            # Use markdown for better formatting control
            cols[col_idx % 2].markdown(f"**{key}:** `{display_value}`")
            col_idx += 1

# --- DICOM Window/Level Sliders ---

def dicom_wl_sliders(
    ds: Optional[pydicom.Dataset],
    current_wc: Optional[float] = None,
    current_ww: Optional[float] = None
) -> Tuple[Optional[float], Optional[float]]:
    """
    Creates Streamlit sliders for adjusting DICOM Window Center (Level) and Width.
    Uses provided current values if available, otherwise calculates defaults from the dataset.

    Args:
        ds: The pydicom Dataset object.
        current_wc: The current window center value from session state (optional).
        current_ww: The current window width value from session state (optional).

    Returns:
        A tuple containing the selected (or current) window center and window width.
    """
    # st.subheader("DICOM Window/Level Adjustment") # Removed subheader, place in calling code if needed

    if ds is None or not hasattr(ds, 'PixelData'):
        st.caption("W/L sliders unavailable: DICOM data missing.")
        logger.warning("dicom_wl_sliders called with missing Dataset or PixelData.")
        return current_wc, current_ww # Return current values if no dataset

    pixel_min: float = 0.0
    pixel_max: float = 4095.0 # Default fallback range
    data_range_calculated = False

    try:
        pixel_array = ds.pixel_array
        # Prefer calculating range *after* applying rescale slope/intercept if present
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            slope = float(ds.RescaleSlope)
            intercept = float(ds.RescaleIntercept)
            # Ensure calculations are done in float to avoid overflow/precision issues
            rescaled_array = pixel_array.astype(np.float64) * slope + intercept
            pixel_min = float(np.min(rescaled_array))
            pixel_max = float(np.max(rescaled_array))
            logger.debug(f"Applied Rescale Slope ({slope}) / Intercept ({intercept}). Calculated range: {pixel_min:.2f} - {pixel_max:.2f}")
        else:
            # If no rescale, use raw pixel values
            pixel_min = float(np.min(pixel_array))
            pixel_max = float(np.max(pixel_array))
            logger.debug(f"No Rescale Slope/Intercept found. Using raw pixel range: {pixel_min:.2f} - {pixel_max:.2f}")

        # Handle edge case of a constant image
        if pixel_max == pixel_min:
            logger.warning("Pixel data range is zero (constant image). Adjusting max range slightly.")
            pixel_max += 1.0
        data_range_calculated = True

    except AttributeError:
        st.caption("Warning: Could not access RescaleSlope/Intercept. Using default range.")
        logger.warning("AttributeError accessing RescaleSlope/Intercept. Falling back.")
        # Attempt range from raw pixels as fallback
        try:
            pixel_array = ds.pixel_array
            pixel_min = float(np.min(pixel_array))
            pixel_max = float(np.max(pixel_array))
            if pixel_max == pixel_min: pixel_max += 1.0
            data_range_calculated = True
        except Exception as fallback_e:
             st.caption(f"Error calculating pixel range: {fallback_e}. Using default range [0-4095].")
             logger.error(f"Error determining pixel range for sliders: {fallback_e}", exc_info=True)
    except Exception as e:
        st.caption(f"Error calculating pixel range: {e}. Using default range [0-4095].")
        logger.error(f"Error determining pixel range for sliders: {e}", exc_info=True)


    # --- Determine Default and Current W/L ---
    def safe_float_convert(value: Any, index: int = 0) -> Optional[float]:
        """Safely convert DICOM numeric values (potentially multi-value) to float."""
        val_to_convert = value
        if isinstance(value, (list, pydicom.multival.MultiValue)):
            if len(value) > index:
                val_to_convert = value[index]
            else:
                return None # Index out of bounds
        try:
            # Handle specific DICOM number string types if necessary
            if isinstance(val_to_convert, (pydicom.valuerep.DSfloat, pydicom.valuerep.DSdecimal)):
                 return float(val_to_convert)
            elif isinstance(val_to_convert, pydicom.valuerep.IS):
                 return float(int(val_to_convert)) # Convert IS (Integer String) to int first
            return float(val_to_convert) if val_to_convert is not None else None
        except (ValueError, TypeError, AttributeError):
            logger.warning(f"Could not convert DICOM value '{value}' to float.", exc_info=False)
            return None

    # Use current values from state if provided and valid, otherwise calculate defaults
    wc_to_use: Optional[float] = current_wc
    ww_to_use: Optional[float] = current_ww

    if wc_to_use is None:
        wc_to_use = safe_float_convert(ds.get("WindowCenter"))
        if wc_to_use is None and data_range_calculated:
             wc_to_use = (pixel_max + pixel_min) / 2.0
        elif wc_to_use is None: # Absolute fallback if range calc failed
             wc_to_use = 2048.0
        logger.debug(f"Calculated default WC: {wc_to_use}")


    if ww_to_use is None:
        ww_to_use = safe_float_convert(ds.get("WindowWidth"))
        if ww_to_use is None and data_range_calculated:
             ww_to_use = max(1.0, (pixel_max - pixel_min)) # Default width covers the whole range
        elif ww_to_use is None: # Absolute fallback
             ww_to_use = 4096.0
        logger.debug(f"Calculated default WW: {ww_to_use}")

    # Ensure defaults are at least 1.0 for width
    ww_to_use = max(1.0, ww_to_use) if ww_to_use is not None else 1.0

    # --- Define Slider Ranges ---
    data_dynamic_range = pixel_max - pixel_min if data_range_calculated else 4095.0
    # Allow sliders to go a bit beyond the calculated min/max
    slider_min_level = pixel_min - data_dynamic_range * 0.25 if data_range_calculated else -1024.0
    slider_max_level = pixel_max + data_dynamic_range * 0.25 if data_range_calculated else 5120.0
    # Set a reasonable max width, avoiding excessively large values
    slider_max_width = max(1.0, data_dynamic_range * 2.0) if data_range_calculated else 8192.0
    slider_max_width = min(slider_max_width, 65536.0) # Cap maximum width


    # Use a smaller step for finer control, related to the data range
    step_size = max(0.1, data_dynamic_range / 2000.0) if data_range_calculated and data_dynamic_range > 0 else 1.0

    # --- Create Sliders ---
    selected_wc = st.slider(
        "Window Center (Level)",
        min_value=slider_min_level,
        max_value=slider_max_level,
        value=float(wc_to_use), # Ensure value is float for slider
        step=step_size,
        key="dicom_wc_slider",
        help=f"Adjust brightness center. Data range approx: [{pixel_min:.1f} - {pixel_max:.1f}]"
    )
    selected_ww = st.slider(
        "Window Width",
        min_value=1.0, # Width must be at least 1
        max_value=slider_max_width,
        value=float(ww_to_use), # Ensure value is float
        step=step_size,
        key="dicom_ww_slider",
        help=f"Adjust contrast range. Data range approx: {data_dynamic_range:.1f}"
    )

    # Reset button logic remains outside this function, handled by session state/rerun in app.py

    # Return the values selected by the user on the sliders
    return float(selected_wc), float(selected_ww)


# --- UMLS Concepts Display ---

def display_umls_concepts(concepts: Optional[List[UMLSConcept]]) -> None:
    """
    Displays a list of standardized UMLS concepts in an expandable panel.
    Uses the UMLSConcept dataclass imported from umls_utils.

    Args:
        concepts: A list of UMLSConcept dataclass instances, or None.
    """
    # Title moved outside expander for visibility
    st.subheader("📚 UMLS Concept Results")
    with st.expander("View Details", expanded=True): # Expand by default if results exist
        if not UMLS_CONCEPT_IMPORTED:
             st.warning("UMLS display unavailable: Could not import `UMLSConcept` from `umls_utils.py`.")
             return

        if concepts is None:
            st.caption("Perform a UMLS search to see results here.")
            return

        if not concepts:
            st.info("No UMLS concepts found for the search term.")
            return

        logger.debug(f"Displaying {len(concepts)} UMLS concepts.")
        st.success(f"Found {len(concepts)} UMLS concept(s):")

        for i, concept in enumerate(concepts):
            # Ensure we have a valid UMLSConcept object
            if not isinstance(concept, UMLSConcept):
                st.warning(f"Item {i+1} is not a valid UMLSConcept object. Skipping.")
                logger.warning(f"Invalid object type found in concepts list: {type(concept)}")
                continue

            # Format the output using markdown for links and code formatting
            st.markdown(f"**{concept.name}**")
            st.markdown(f"    CUI: `{concept.ui}`") # Indent details
            st.markdown(f"    Source: `{concept.rootSource}`")
            if concept.uri:
                st.markdown(f"    URI: [{concept.uri}]({concept.uri})")
            if concept.uriLabel: # Display label if available
                 st.markdown(f"    URI Label: `{concept.uriLabel}`")

            if i < len(concepts) - 1:
                st.markdown("---") # Add separator between concepts