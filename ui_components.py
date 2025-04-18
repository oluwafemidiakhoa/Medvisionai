# -*- coding: utf-8 -*-
"""
ui_components.py - Streamlit UI Helper Components for RadVision AI
==================================================================

Provides reusable functions for creating specific UI elements within the
Streamlit application, such as displaying DICOM metadata in a structured
format, generating interactive Window/Level sliders for DICOM images,
and presenting UMLS concept search results clearly.
"""

import logging
from typing import Optional, Tuple, Dict, Any, List, Union

import numpy as np
import pydicom
import pydicom.valuerep
import pydicom.multival
import pydicom.uid
import streamlit as st

# --- Dependency: UMLSConcept ---
# Attempt to import from umls_utils. Handle gracefully if unavailable.
try:
    # Assuming umls_utils.py is structured to export UMLSConcept
    from umls_utils import UMLSConcept
    _UMLS_CONCEPT_IMPORTED = True
except ImportError:
    # Define a dummy class if import fails. This allows function signatures
    # using UMLSConcept to remain valid, preventing NameErrors elsewhere.
    # The display function itself will check _UMLS_CONCEPT_IMPORTED.
    class UMLSConcept: # type: ignore[no-redef]
        """Dummy class if umls_utils.UMLSConcept cannot be imported."""
        def __init__(self, name: str = "Import Error", ui: str = "", uri: str = "", rootSource: str = ""):
            self.name = name
            self.ui = ui
            self.uri = uri
            self.rootSource = rootSource

    _UMLS_CONCEPT_IMPORTED = False
    logging.warning(
        "Could not import UMLSConcept from umls_utils.py. "
        "UMLS display functionality will be disabled or limited."
    )

# --- Logging Configuration ---
logger = logging.getLogger(__name__)

# =============================================================================
# DICOM Metadata Display Component
# =============================================================================

def display_dicom_metadata(metadata: Optional[Dict[str, Any]]) -> None:
    """
    Displays formatted DICOM metadata within a Streamlit expander,
    arranging the key-value pairs neatly into two columns.

    Args:
        metadata: A dictionary where keys are DICOM tag names (str)
                  and values are the extracted tag values (Any type).
                  If None or empty, displays a placeholder message.
    """
    # Expander provides a clean way to optionally view potentially long metadata
    with st.expander("View DICOM Metadata", expanded=False):
        if not metadata:
            st.caption("No DICOM metadata extracted or provided.")
            return

        logger.debug(f"Attempting to display {len(metadata)} DICOM metadata items.")
        cols = st.columns(2)  # Use two columns for better layout
        col_idx = 0

        # Sort metadata alphabetically by tag name for consistent order
        try:
            # Sorting might fail if keys are not consistently strings (unlikely)
            sorted_metadata_items = sorted(metadata.items())
        except Exception as sort_err:
            logger.error(f"Failed to sort metadata keys: {sort_err}. Displaying unsorted.")
            st.warning("Could not sort metadata items.")
            sorted_metadata_items = metadata.items()


        for key, value in sorted_metadata_items:
            display_value = "N/A"  # Default display value
            try:
                # Handle different potential value types from pydicom
                if value is None:
                    display_value = "_(empty)_"
                elif isinstance(value, (list, pydicom.multival.MultiValue)):
                    # Join list items, handling PersonName objects specifically
                    str_values = []
                    for v in value:
                        if isinstance(v, pydicom.valuerep.PersonName):
                            # Use formatted() for a standard representation, fallback to str()
                            str_values.append(v.formatted("%L, %F %M") if hasattr(v, 'formatted') else str(v))
                        else:
                            str_values.append(str(v))
                    display_value = "; ".join(str_values) # Use semicolon for multi-value clarity
                elif isinstance(value, pydicom.uid.UID):
                    # Show UID name (if known) and the UID value itself
                    display_value = f"{value.name} ({value})"
                elif isinstance(value, bytes):
                    # Indicate binary data without displaying it directly
                    display_value = f"[Binary Data: {len(value)} bytes]"
                elif isinstance(value, pydicom.valuerep.PersonName):
                    display_value = value.formatted("%L, %F %M") if hasattr(v, 'formatted') else str(value)
                elif isinstance(value, (pydicom.valuerep.DSfloat, pydicom.valuerep.DSdecimal, pydicom.valuerep.IS)):
                    # Ensure numeric types represented as strings are correctly displayed
                    display_value = str(value)
                else:
                    # General case: convert to string and remove leading/trailing whitespace
                    display_value = str(value).strip()

                # Truncate excessively long values to keep UI clean
                MAX_LEN = 150
                if len(display_value) > MAX_LEN:
                    display_value = display_value[:MAX_LEN - 3] + "..."

            except Exception as fmt_err:
                # Log formatting errors but don't crash the UI
                logger.warning(
                    f"Error formatting DICOM metadata key '{key}' (Type: {type(value)}): {fmt_err}",
                    exc_info=False # Avoid full traceback unless debugging
                )
                display_value = "[Error formatting value]"

            # Use markdown for consistent styling (bold key, code-formatted value)
            try:
                 # Ensure key is string for markdown
                safe_key = str(key)
                cols[col_idx % 2].markdown(f"**{safe_key}:** `{display_value}`")
                col_idx += 1
            except Exception as md_err:
                 logger.error(f"Failed to render markdown for key '{key}': {md_err}")
                 cols[col_idx % 2].text(f"{key}: {display_value} (Render Error)")
                 col_idx += 1

# =============================================================================
# DICOM Window/Level (W/L) Slider Component
# =============================================================================

def _safe_float_convert(value: Any, index: int = 0) -> Optional[float]:
    """Safely converts DICOM numeric string types or lists thereof to float."""
    target_value = value
    if isinstance(value, (list, pydicom.multival.MultiValue)):
        if len(value) > index:
            target_value = value[index]
        else:
            logger.warning(f"Attempted to access index {index} in multi-value '{value}', but length is {len(value)}.")
            return None # Index out of bounds for multi-value list

    if target_value is None:
        return None

    try:
        # Handle specific DICOM numeric representations
        if isinstance(target_value, (pydicom.valuerep.DSfloat, pydicom.valuerep.DSdecimal)):
            return float(target_value)
        elif isinstance(target_value, pydicom.valuerep.IS): # Integer String
            return float(int(target_value))
        # General case: attempt direct float conversion
        return float(target_value)
    except (ValueError, TypeError) as e:
        logger.debug(f"Could not convert DICOM value '{target_value}' (type: {type(target_value)}) to float: {e}", exc_info=False)
        return None

def dicom_wl_sliders(
    ds: Optional[pydicom.Dataset],
    current_wc: Optional[float] = None,
    current_ww: Optional[float] = None
) -> Tuple[Optional[float], Optional[float]]:
    """
    Creates Streamlit sliders for adjusting DICOM Window Center (Level) and Width.

    Calculates appropriate ranges and defaults based on the DICOM dataset,
    considering pixel data range and Rescale Slope/Intercept if available.
    Uses provided `current_wc` and `current_ww` from session state as initial
    values if provided.

    Args:
        ds: The pydicom Dataset object. Must contain 'PixelData'.
        current_wc: The current window center value (optional).
        current_ww: The current window width value (optional).

    Returns:
        A tuple containing the selected (float) window center and window width.
        Returns (current_wc, current_ww) if sliders cannot be created (e.g., no dataset).
    """
    # --- Input Validation ---
    if ds is None or not hasattr(ds, 'PixelData'):
        # Display placeholder instead of sliders if data is missing
        st.caption("ℹ️ W/L adjustment sliders require loaded DICOM pixel data.")
        if ds is None:
             logger.warning("dicom_wl_sliders called with ds=None.")
        else:
             logger.warning("dicom_wl_sliders called, but Dataset missing 'PixelData'.")
        # Return the passed-in values as no change is possible
        return current_wc, current_ww

    # --- Calculate Pixel Data Range (considering Rescale Slope/Intercept) ---
    pixel_min: float = 0.0
    pixel_max: float = 4095.0 # Default fallback range (common for 12-bit CT/MRI)
    data_range_calculated = False

    try:
        pixel_array = ds.pixel_array
        # Use float64 for calculations to prevent overflow/precision issues
        pixel_array_float = pixel_array.astype(np.float64)

        # Apply rescale slope/intercept if they exist and are valid numbers
        slope = _safe_float_convert(ds.get("RescaleSlope"))
        intercept = _safe_float_convert(ds.get("RescaleIntercept"))

        if slope is not None and intercept is not None:
            # Ensure slope is not zero to avoid division issues later if needed
            slope = float(slope) if slope != 0 else 1.0
            intercept = float(intercept)
            rescaled_array = pixel_array_float * slope + intercept
            pixel_min = float(np.min(rescaled_array))
            pixel_max = float(np.max(rescaled_array))
            logger.debug(f"Applied Rescale Slope ({slope}) / Intercept ({intercept}). Calculated range: {pixel_min:.2f} - {pixel_max:.2f}")
            data_range_calculated = True
        else:
            # If no rescale tags or they are invalid, use raw pixel values
            pixel_min = float(np.min(pixel_array_float))
            pixel_max = float(np.max(pixel_array_float))
            logger.debug(f"Using raw pixel range (no valid Rescale Slope/Intercept): {pixel_min:.2f} - {pixel_max:.2f}")
            data_range_calculated = True

        # Handle edge case: image where all pixels have the same value
        if pixel_max == pixel_min:
            logger.warning("Pixel data range is zero (constant image detected). Adjusting max range slightly for slider.")
            pixel_max += 1.0 # Add 1 to allow some width range

    except AttributeError as ae:
         # This might happen if tags like RescaleSlope exist but are invalid format
         st.caption("⚠️ Warning: Could not access DICOM rescale tags correctly. Using raw pixel range.")
         logger.warning(f"AttributeError accessing pixel data or rescale tags: {ae}. Falling back to raw range.")
         # Attempt raw range as fallback
         try:
             pixel_min = float(np.min(ds.pixel_array.astype(np.float64)))
             pixel_max = float(np.max(ds.pixel_array.astype(np.float64)))
             if pixel_max == pixel_min: pixel_max += 1.0
             data_range_calculated = True
         except Exception as fallback_e:
              st.caption(f"⚠️ Error calculating pixel range: {fallback_e}. Using default range [0-4095].")
              logger.error(f"Error determining pixel range for sliders (fallback failed): {fallback_e}", exc_info=True)
              # Reset to hardcoded defaults if all else fails
              pixel_min, pixel_max = 0.0, 4095.0
              data_range_calculated = False # Indicate failure
    except Exception as e:
        st.caption(f"⚠️ Error calculating pixel range: {e}. Using default range [0-4095].")
        logger.error(f"Unexpected error determining pixel range for sliders: {e}", exc_info=True)
        pixel_min, pixel_max = 0.0, 4095.0
        data_range_calculated = False

    data_dynamic_range = max(1.0, pixel_max - pixel_min) # Ensure range is at least 1.0

    # --- Determine Initial W/L Values ---
    # Use values from session state if valid, otherwise DICOM tags, otherwise calculate from range
    wc_initial: float
    ww_initial: float

    if current_wc is not None:
        wc_initial = float(current_wc)
        logger.debug(f"Using provided current WC: {wc_initial:.2f}")
    else:
        wc_from_dicom = _safe_float_convert(ds.get("WindowCenter"))
        if wc_from_dicom is not None:
            wc_initial = wc_from_dicom
            logger.debug(f"Using WC from DICOM tag: {wc_initial:.2f}")
        else:
            wc_initial = pixel_min + data_dynamic_range / 2.0 # Center of the calculated range
            logger.debug(f"Calculated default WC from range: {wc_initial:.2f}")

    if current_ww is not None:
        ww_initial = max(1.0, float(current_ww)) # Ensure width >= 1
        logger.debug(f"Using provided current WW: {ww_initial:.2f}")
    else:
        ww_from_dicom = _safe_float_convert(ds.get("WindowWidth"))
        if ww_from_dicom is not None:
            ww_initial = max(1.0, ww_from_dicom) # Ensure width >= 1
            logger.debug(f"Using WW from DICOM tag: {ww_initial:.2f}")
        else:
            ww_initial = data_dynamic_range # Default width covers the full calculated range
            logger.debug(f"Calculated default WW from range: {ww_initial:.2f}")

    # --- Define Slider Properties (Min, Max, Step) ---
    # Allow sliders to go slightly beyond the calculated data range for flexibility
    margin = data_dynamic_range * 0.5 # Allow going 50% beyond calculated range
    slider_min_level = pixel_min - margin
    slider_max_level = pixel_max + margin

    # Set a sensible max width (e.g., 2x data range), capped to avoid extreme values
    slider_max_width = max(1.0, data_dynamic_range * 2.0)
    slider_max_width = min(slider_max_width, 65535.0) # Absolute cap

    # Determine step size based on range for finer control, ensure it's reasonable
    step_size = max(0.01, round(data_dynamic_range / 2000.0, 2)) if data_dynamic_range > 0 else 1.0
    # Ensure initial values are within the calculated slider bounds, clamp if necessary
    wc_initial = max(slider_min_level, min(slider_max_level, wc_initial))
    ww_initial = max(1.0, min(slider_max_width, ww_initial))


    # --- Create Sliders using st.slider ---
    # Tooltips provide context about the data range
    help_text = f"Data range ≈ [{pixel_min:.1f} – {pixel_max:.1f}] | Dynamic range ≈ {data_dynamic_range:.1f}"

    selected_wc = st.slider(
        label="Window Center (Level)",
        min_value=slider_min_level,
        max_value=slider_max_level,
        value=wc_initial,
        step=step_size,
        key="dicom_wc_slider", # Unique key for session state
        help=f"Adjusts image brightness. {help_text}"
    )
    selected_ww = st.slider(
        label="Window Width",
        min_value=1.0, # Width must be positive
        max_value=slider_max_width,
        value=ww_initial,
        step=step_size,
        key="dicom_ww_slider", # Unique key for session state
        help=f"Adjusts image contrast. {help_text}"
    )

    # Return the current values from the sliders
    return float(selected_wc), float(selected_ww)

# =============================================================================
# UMLS Concept Display Component
# =============================================================================

def display_umls_concepts(concepts: Optional[List[UMLSConcept]], search_term: Optional[str] = None) -> None:
    """
    Displays a list of UMLS concepts in a structured format within an expander.

    Args:
        concepts: A list of UMLSConcept objects (or None/empty).
        search_term: The term used for the search (optional, for context).
    """
    # Add a subheader for context, visible even when collapsed
    st.subheader("📚 UMLS Concept Lookup Results")

    # Default to expanded if there are concepts, otherwise collapsed
    expanded_default = bool(concepts)

    with st.expander("View Concept Details", expanded=expanded_default):
        if not _UMLS_CONCEPT_IMPORTED:
            # Show warning if the necessary class couldn't be imported
            st.warning(
                "UMLS display unavailable: Could not import `UMLSConcept` from `umls_utils.py`. "
                "Ensure the file exists and dependencies are installed."
            )
            return

        if concepts is None:
            # Placeholder if no search has been performed yet
            st.caption("Perform a UMLS search or analysis to see mapped concepts here.")
            return

        if not concepts:
            # Message if search was done but yielded no results
            search_context = f" for '{search_term}'" if search_term else ""
            st.info(f"No UMLS concepts found{search_context}.")
            return

        # Confirmation message with count
        search_context = f" for '{search_term}'" if search_term else ""
        st.success(f"Found {len(concepts)} UMLS concept(s){search_context}:")
        logger.debug(f"Displaying {len(concepts)} UMLS concepts.")

        # Iterate and display each concept
        for i, concept in enumerate(concepts):
            # Type check for safety, especially relevant if import failed/dummy used
            if not isinstance(concept, UMLSConcept):
                st.warning(f"Skipping invalid item #{i+1} in concepts list (expected UMLSConcept, got {type(concept).__name__}).")
                logger.warning(f"Invalid object type found in concepts list: {type(concept)}")
                continue

            # Use markdown for rich formatting (bolding, links, code style)
            col1, col2 = st.columns([3, 1]) # Layout for name and source/CUI
            with col1:
                 # Display name as bold, make it a link if URI exists
                 name_display = f"**[{concept.name}]({concept.uri})**" if concept.uri else f"**{concept.name}**"
                 st.markdown(name_display, unsafe_allow_html=True) # Allow HTML for potential link target
            with col2:
                 # Display Source and CUI more compactly
                 st.markdown(f"`{concept.rootSource}` | `{concept.ui}`", unsafe_allow_html=True)

            # Display URI label if available
            # if concept.uriLabel:
            #      st.markdown(f"    *Label:* `{concept.uriLabel}`", unsafe_allow_html=True)

            # Add a divider between concepts for readability, except after the last one
            if i < len(concepts) - 1:
                st.divider() # Use Streamlit's built-in divider

def _display_concept_table(concepts: List[Any]) -> None:
    """
    Helper function to display a table of UMLS concepts with interactive features.

    Args:
        concepts: List of UMLS concept objects
    """
    # Create a dataframe for better display
    import pandas as pd

    rows = []
    for concept in concepts:
        try:
            # Extract concept data
            concept_name = getattr(concept, "name", str(concept))
            concept_ui = getattr(concept, "ui", "")
            concept_source = getattr(concept, "rootSource", "")
            concept_uri = getattr(concept, "uri", "")

            # Extract semantic types (if available)
            semantic_types = getattr(concept, "semTypes", [""])
            if isinstance(semantic_types, list):
                semantic_types = ", ".join(semantic_types)

            # Add to rows
            rows.append({
                "Concept Name": concept_name,
                "CUI": concept_ui,
                "Source": concept_source,
                "Semantic Type": semantic_types,
                "URI": concept_uri
            })
        except Exception as e:
            logger.error(f"Error processing UMLS concept for table: {e}")

    if not rows:
        st.info("No concepts available in this category.")
        return

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Define column configuration for improved display
    column_config = {
        "Concept Name": st.column_config.TextColumn(
            "Concept Name",
            width="large",
            help="The standardized name of the medical concept"
        ),
        "CUI": st.column_config.TextColumn(
            "CUI",
            width="small",
            help="Concept Unique Identifier in UMLS"
        ),
        "Source": st.column_config.TextColumn(
            "Source",
            width="small",
            help="Source vocabulary (e.g., SNOMEDCT_US, ICD10)"
        ),
        "Semantic Type": st.column_config.TextColumn(
            "Semantic Type",
            width="medium",
            help="Category of the concept in the UMLS semantic network"
        )
    }

    # Display as interactive table
    st.dataframe(
        df,
        column_config=column_config,
        hide_index=True,
        use_container_width=True
    )

    # Add links to UMLS Browser for each concept
    for concept in concepts:
        try:
            concept_name = getattr(concept, "name", str(concept))
            concept_ui = getattr(concept, "ui", "")
            concept_uri = getattr(concept, "uri", "")

            if concept_uri:
                st.markdown(f"[{concept_name} ({concept_ui}) - View in UMLS Browser]({concept_uri})")
            else:
                # Create a fallback link to UMLS Browser
                umls_browser_url = f"https://uts.nlm.nih.gov/uts/umls/concept/{concept_ui}"
                st.markdown(f"[{concept_name} ({concept_ui}) - View in UMLS Browser]({umls_browser_url})")
        except Exception as e:
            logger.error(f"Error creating UMLS concept link: {e}")


def display_umls_concepts(concepts: List[Any], search_term: str = None) -> None:
    """
    Display UMLS concepts with expandable details and interactive features.

    Args:
        concepts: List of UMLS concept objects
        search_term: Optional search term used to find these concepts
    """
    if not concepts:
        st.info("No UMLS concepts available.")
        return

    # Display header with search term if provided
    if search_term:
        st.markdown(f"##### UMLS Concepts for '{search_term}'")
    else:
        st.markdown("##### UMLS Concepts")

    # Group concepts by semantic type if available
    concept_groups = {}

    for concept in concepts:
        try:
            # Extract semantic types (if available)
            semantic_types = getattr(concept, "semTypes", ["Uncategorized"])
            if not semantic_types:
                semantic_types = ["Uncategorized"]

            # Use first semantic type for grouping
            semantic_type = semantic_types[0] if isinstance(semantic_types, list) else semantic_types

            # Add to appropriate group
            if semantic_type not in concept_groups:
                concept_groups[semantic_type] = []
            concept_groups[semantic_type].append(concept)
        except Exception as e:
            logger.error(f"Error processing UMLS concept for display: {e}")
            # If grouping fails, ensure concept is still displayed
            if "Uncategorized" not in concept_groups:
                concept_groups["Uncategorized"] = []
            concept_groups["Uncategorized"].append(concept)

    # Display concepts by semantic type in tabs if multiple types exist
    if len(concept_groups) > 1:
        tabs = st.tabs(list(concept_groups.keys()))

        for i, (semantic_type, concepts_in_group) in enumerate(concept_groups.items()):
            with tabs[i]:
                _display_concept_table(concepts_in_group)
    else:
        # Just display the table directly if only one semantic type
        _display_concept_table(concepts)

    # Add export options
    if len(concepts) > 0:
        # Create a CSV in memory
        import io
        import csv
        import base64

        csv_buffer = io.StringIO()
        csv_writer = csv.writer(csv_buffer)

        # Write header
        csv_writer.writerow(["Concept Name", "CUI", "Source", "Semantic Types"])

        # Write concepts
        for concept in concepts:
            try:
                concept_name = getattr(concept, "name", str(concept))
                concept_ui = getattr(concept, "ui", "")
                concept_source = getattr(concept, "rootSource", "")
                semantic_types = getattr(concept, "semTypes", [""])
                if isinstance(semantic_types, list):
                    semantic_types = ", ".join(semantic_types)

                csv_writer.writerow([concept_name, concept_ui, concept_source, semantic_types])
            except Exception as e:
                logger.error(f"Error exporting UMLS concept: {e}")

        # Create download link
        csv_string = csv_buffer.getvalue()
        b64 = base64.b64encode(csv_string.encode()).decode()

        col1, col2 = st.columns([3, 1])
        with col2:
            href = f'<a href="data:file/csv;base64,{b64}" download="umls_concepts.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)