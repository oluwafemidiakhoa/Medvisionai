import streamlit as st
from typing import Optional, Tuple, Dict, Any
import pydicom

# This file is optional. You can place these helper functions directly in main_app.py
# if the UI logic isn't too complex yet.

def display_dicom_metadata(metadata: Dict[str, Any]):
    """Displays formatted DICOM metadata in an expander."""
    with st.expander("View DICOM Metadata", expanded=False):
        # Format metadata for display (e.g., in columns or as a list)
        cols = st.columns(2)
        col_idx = 0
        for key, value in metadata.items():
            # Simple conversion for display
            display_value = str(value)
            if isinstance(value, list):
                 display_value = ", ".join(map(str, value))
            elif isinstance(value, pydicom.uid.UID):
                 display_value = value.name # Show name for UIDs if available

            cols[col_idx % 2].markdown(f"**{key}:** `{display_value}`")
            col_idx += 1
        if not metadata:
            st.caption("No metadata extracted or available.")


def dicom_wl_sliders(
    ds: pydicom.Dataset,
    metadata: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[float]]:
    """Creates sliders for DICOM Window/Level adjustment."""
    st.subheader("DICOM Window/Level")

    # Get defaults from metadata or calculate reasonable range
    default_wc, default_ww = metadata.get("WindowCenter", None), metadata.get("WindowWidth", None)

    # Try to get pixel range for better slider limits
    pixel_min, pixel_max = 0, 4095 # Default reasonable range
    try:
        pixel_array = ds.pixel_array
        pixel_min = float(pixel_array.min())
        pixel_max = float(pixel_array.max())
        # Adjust defaults if they seem unreasonable for the pixel range
        if default_wc is not None and not (pixel_min <= default_wc <= pixel_max):
             default_wc = (pixel_max + pixel_min) / 2
        if default_ww is not None and (default_ww <= 0 or default_ww > (pixel_max - pixel_min)):
             default_ww = (pixel_max - pixel_min) * 0.8 # Default to 80% of range if default is bad
    except Exception:
        st.caption("Could not determine pixel range for optimal slider limits.")


    # Set reasonable slider ranges based on pixel data or defaults
    min_level = pixel_min - (pixel_max-pixel_min) # Allow going below min
    max_level = pixel_max + (pixel_max-pixel_min) # Allow going above max
    max_width = (pixel_max - pixel_min) * 2 if pixel_max > pixel_min else 4096

    wc = st.slider(
        "Window Center (Level)",
        min_value=min_level,
        max_value=max_level,
        value=default_wc if default_wc is not None else (pixel_max + pixel_min) / 2,
        step=1.0,
        key="dicom_wc"
    )
    ww = st.slider(
        "Window Width",
        min_value=1.0, # Width must be positive
        max_value=max_width,
        value=default_ww if default_ww is not None else (pixel_max - pixel_min) * 0.8,
        step=1.0,
        key="dicom_ww"
    )

    if st.button("Reset W/L", key="reset_wl"):
         # Rerun will reset sliders to default values based on metadata next time
         st.rerun()

    return float(wc), float(ww)