import streamlit as st
from typing import Optional, Tuple, Dict, Any, List, Union
import pydicom
import pydicom.valuerep
import numpy as np
import logging

from umls_utils import UMLSConcept  # For UMLS concept display

logger = logging.getLogger(__name__)

# --- DICOM Metadata Display ---

def display_dicom_metadata(metadata: Optional[Dict[str, Any]]) -> None:
    """
    Displays formatted DICOM metadata in a Streamlit expander, arranged in two columns.
    """
    with st.expander("View DICOM Metadata", expanded=False):
        if not metadata:
            st.caption("No metadata extracted or available.")
            return

        cols = st.columns(2)
        col_idx = 0
        logger.debug(f"Displaying {len(metadata)} metadata items.")

        for key, value in metadata.items():
            display_value = "N/A"
            try:
                if value is None:
                    display_value = "N/A"
                elif isinstance(value, list):
                    display_value = ", ".join(map(str, value))
                elif isinstance(value, pydicom.uid.UID):
                    display_value = value.name
                elif isinstance(value, bytes):
                    display_value = f"[Binary Data ({len(value)} bytes)]"
                elif isinstance(value, pydicom.valuerep.PersonName):
                    display_value = "[Person Name]"
                else:
                    display_value = str(value).strip()

                if len(display_value) > 150:
                    display_value = display_value[:147] + "..."
            except Exception as e:
                logger.warning(f"Error formatting metadata key '{key}': {e}", exc_info=True)
                display_value = "[Error formatting value]"

            cols[col_idx % 2].markdown(f"**{key}:** {display_value}")
            col_idx += 1

# --- DICOM Window/Level Sliders ---

def dicom_wl_sliders(
    ds: Optional[pydicom.Dataset],
    metadata: Dict[str, Any]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Creates Streamlit sliders for adjusting DICOM Window Center (Level) and Width.
    """
    st.subheader("DICOM Window/Level Adjustment")

    if ds is None or 'PixelData' not in ds:
        st.caption("Cannot create W/L sliders: DICOM data or PixelData missing.")
        logger.warning("dicom_wl_sliders called with missing Dataset or PixelData.")
        return None, None

    pixel_min: float = 0.0
    pixel_max: float = 4095.0
    try:
        pixel_array = ds.pixel_array
        if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
            slope = float(ds.RescaleSlope)
            intercept = float(ds.RescaleIntercept)
            logger.debug(f"Applying Rescale Slope ({slope}) / Intercept ({intercept}) for range calculation.")
            rescaled_array = pixel_array.astype(np.float64) * slope + intercept
            pixel_min = float(rescaled_array.min())
            pixel_max = float(rescaled_array.max())
        else:
            pixel_min = float(pixel_array.min())
            pixel_max = float(pixel_array.max())
        logger.info(f"Determined pixel value range: Min={pixel_min}, Max={pixel_max}")

        if pixel_max == pixel_min:
            logger.warning("Pixel data range is zero (constant image). Adjusting range.")
            pixel_max += 1.0

    except Exception as e:
        st.caption(f"Could not determine pixel range (Error: {e}). Using default range.")
        logger.error(f"Error determining pixel range for sliders: {e}", exc_info=True)

    def safe_float_convert(value: Any) -> Optional[float]:
        if isinstance(value, (list, pydicom.multival.MultiValue)):
            val_to_convert = value[0] if len(value) > 0 else None
        else:
            val_to_convert = value
        try:
            return float(val_to_convert) if val_to_convert is not None else None
        except (ValueError, TypeError):
            return None

    default_wc = safe_float_convert(metadata.get("WindowCenter")) or ((pixel_max + pixel_min) / 2.0)
    default_ww = safe_float_convert(metadata.get("WindowWidth")) or max(1.0, (pixel_max - pixel_min) * 0.8)

    data_range = pixel_max - pixel_min
    slider_min_level = pixel_min - data_range * 0.5
    slider_max_level = pixel_max + data_range * 0.5
    slider_max_width = min(max(1.0, data_range * 2.0), 65536.0)

    wc = st.slider(
        "Window Center (Level)",
        min_value=slider_min_level,
        max_value=slider_max_level,
        value=max(slider_min_level, min(slider_max_level, default_wc)),
        step=max(0.1, data_range / 1000.0),
        key="dicom_wc_slider",
        help=f"Adjust brightness center. Range: [{pixel_min:.1f} - {pixel_max:.1f}]"
    )
    ww = st.slider(
        "Window Width",
        min_value=1.0,
        max_value=slider_max_width,
        value=max(1.0, min(slider_max_width, default_ww)),
        step=max(0.1, data_range / 1000.0),
        key="dicom_ww_slider",
        help=f"Adjust contrast range. Data range: {data_range:.1f}"
    )

    if st.button("Reset W/L", key="reset_wl_button"):
        logger.info("Reset W/L button clicked. Rerunning to apply default values.")
        st.rerun()

    return float(wc), float(ww)

# --- UMLS Concepts Display ---

def display_umls_concepts(concepts: List[UMLSConcept]) -> None:
    """
    Displays a list of standardized UMLS concepts in an expandable panel.

    Args:
        concepts: A list of UMLSConcept dataclass instances.
    """
    with st.expander("Standardized UMLS Concepts", expanded=False):
        if not concepts:
            st.write("No UMLS concepts available.")
            return
        for concept in concepts:
            st.markdown(f"- [{concept.name}]({concept.uri})  
                CUI: `{concept.ui}`, Source: {concept.rootSource}")
