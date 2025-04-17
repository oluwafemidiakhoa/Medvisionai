# -*- coding: utf-8 -*-
"""
main_page_ui.py - Renders the Main Content Area for RadVision AI
=================================================================

Responsible for displaying the primary user interface elements within the
main page area, typically organized into two columns:
1.  Image Viewer: Displays the uploaded/demo image, handles optional ROI
    drawing via streamlit-drawable-canvas, and shows DICOM metadata.
2.  Analysis & Results: Presents analysis outputs (Initial, Q&A, Condition,
    Confidence) in tabs, provides translation functionality, and includes
    a manual UMLS concept lookup tool.

Dependencies:
- Streamlit for UI elements.
- PIL (Pillow) for image handling.
- Optional: streamlit-drawable-canvas for ROI.
- Optional: ui_components.py for displaying DICOM metadata and UMLS concepts.
- Optional: translation_models.py for translation features.
- Optional: umls_utils.py for UMLS lookup features.
- config.py for constants (like DEFAULT_UMLS_HITS).
- session_state.py (implicitly, as it reads/writes `st.session_state`).
"""

from __future__ import annotations

import logging
from typing import Optional, List, Tuple, Any, Dict

import streamlit as st
from PIL import Image  # Assumes Pillow is installed

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Import Optional UI & Logic Components ---

# ui_components (DICOM Metadata display, UMLS Concept display)
_UI_COMPONENTS_LOADED = False
try:
    # Import specific functions if they exist
    from ui_components import display_dicom_metadata, display_umls_concepts
    _UI_COMPONENTS_LOADED = True
    logger.debug("Successfully imported ui_components.")
except ImportError:
    # Define dummy functions if the module or functions are missing
    def display_dicom_metadata(*args, **kwargs): # type: ignore[misc]
        st.warning("DICOM metadata display component not available.")
    def display_umls_concepts(*args, **kwargs): # type: ignore[misc]
        st.warning("UMLS concepts display component not available.")
    logger.warning("ui_components.py not found or missing expected functions.")

# streamlit-drawable-canvas (for ROI drawing)
_DRAWABLE_CANVAS_AVAILABLE = False
try:
    from streamlit_drawable_canvas import st_canvas
    _DRAWABLE_CANVAS_AVAILABLE = True
    logger.debug("Successfully imported streamlit-drawable-canvas.")
except ImportError:
    st_canvas = None # Keep variable defined as None if not available
    logger.warning("streamlit-drawable-canvas not found. ROI drawing disabled.")

# Translation features
_TRANSLATION_AVAILABLE = False
_LANGUAGE_CODES = {"English": "en"} # Default fallback
_AUTO_DETECT_INDICATOR = "Auto"
try:
    from translation_models import (
        TRANSLATION_AVAILABLE as _T_AVAILABLE,
        LANGUAGE_CODES as _T_LANG_CODES,
        AUTO_DETECT_INDICATOR as _T_AUTO_DETECT,
        translate as _translate_func,
    )
    # Only override defaults if import was successful
    _TRANSLATION_AVAILABLE = _T_AVAILABLE
    _LANGUAGE_CODES = _T_LANG_CODES
    _AUTO_DETECT_INDICATOR = _T_AUTO_DETECT
    translate = _translate_func
    logger.debug(f"Translation features available: {_TRANSLATION_AVAILABLE}")
except ImportError:
    translate = None # Keep variable defined as None if not available
    logger.warning("translation_models.py not found. Translation disabled.")

# UMLS features (check both module load and API key presence)
_UMLS_UTILS_LOADED = False
_UMLS_API_KEY_PRESENT = False
search_umls = None # Define as None initially
UMLSAuthError = RuntimeError # Define fallback exception
UMLSConcept = Any # Define fallback type
try:
    # Check if the utils module itself loaded successfully
    from umls_utils import UMLS_UTILS_LOADED as _U_LOADED, search_umls as _search_umls_func, UMLSAuthError as _UAuthError, UMLSConcept as _UConcept
    _UMLS_UTILS_LOADED = _U_LOADED
    if _UMLS_UTILS_LOADED:
         # Only assign functions/types if the module really loaded
         search_umls = _search_umls_func
         UMLSAuthError = _UAuthError
         UMLSConcept = _UConcept
         # Check for API key presence (can be done here or in app.py)
         import os
         _UMLS_API_KEY_PRESENT = bool(os.getenv("UMLS_APIKEY"))
         logger.debug(f"UMLS Utils loaded: {_UMLS_UTILS_LOADED}, API Key Present: {_UMLS_API_KEY_PRESENT}")
    else:
         logger.warning("umls_utils reported it did not load correctly (e.g., missing 'requests').")
except ImportError:
    logger.warning("umls_utils.py not found. UMLS lookup disabled.")

# Determine overall UMLS availability for the UI
IS_UMLS_LOOKUP_AVAILABLE = _UMLS_UTILS_LOADED and _UMLS_API_KEY_PRESENT

# Configuration (e.g., default number of UMLS hits)
try:
    from config import DEFAULT_UMLS_HITS
except ImportError:
    DEFAULT_UMLS_HITS = 3 # Fallback value
    logger.warning("Could not import DEFAULT_UMLS_HITS from config.py, using default.")


# =============================================================================
#  1. IMAGE VIEWER (Left Column)
# =============================================================================

def _render_image_viewer(col) -> None:
    """Renders the image display area, including ROI canvas if available."""
    with col:
        st.subheader("🖼️ Image Viewer")

        # --- Check for Image in Session State ---
        # Use a specific key expected to be set by file_processing.py
        img_to_display: Optional[Image.Image] = st.session_state.get("display_image")

        if not isinstance(img_to_display, Image.Image):
            # No valid image object found in state
            if st.session_state.get("uploaded_file_info"):
                # File info exists, but image processing likely failed
                st.error(
                    "❌ Image preview failed to load. Check file format or logs.",
                    icon="🖼️"
                 )
                logger.warning("Image viewer: display_image key exists but is not a PIL Image.")
            else:
                # No file uploaded or demo mode active yet
                st.info(
                    "⬅️ Upload an image or enable *Demo Mode* in the sidebar to begin.",
                    icon="☝️"
                )
            # Stop rendering the viewer if no image
            return

        # --- Image Available: Render Canvas or Static Image ---
        logger.debug(f"Rendering image viewer with image size: {img_to_display.size}")

        if _DRAWABLE_CANVAS_AVAILABLE and st_canvas:
            st.caption("Draw a rectangle on the image to define a Region of Interest (ROI).")
            # --- Calculate Canvas Dimensions (Fit within constraints) ---
            max_w, max_h = 600, 500 # Max dimensions for the canvas UI element
            img_w, img_h = img_to_display.size

            if img_w <= 0 or img_h <= 0:
                st.warning("Image has invalid dimensions (<= 0). Cannot display.")
                logger.error(f"Invalid image dimensions for display: {img_to_display.size}")
                return

            aspect_ratio = img_w / img_h
            # Calculate width first, respecting max_w
            canvas_w = min(img_w, max_w)
            # Calculate corresponding height based on aspect ratio
            canvas_h = int(canvas_w / aspect_ratio)

            # If calculated height exceeds max_h, recalculate width based on max_h
            if canvas_h > max_h:
                canvas_h = max_h
                canvas_w = int(canvas_h * aspect_ratio)

            # Ensure minimum dimensions
            canvas_w = max(canvas_w, 150)
            canvas_h = max(canvas_h, 150)
            logger.debug(f"Canvas dimensions set to: {canvas_w}x{canvas_h}")

            # --- Render the Canvas ---
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.2)",  # Semi-transparent orange fill
                stroke_width=2,
                stroke_color="rgba(255, 87, 34, 0.9)", # Darker orange stroke
                background_image=img_to_display,    # The crucial part!
                update_streamlit=True,        # Send updates back to Streamlit
                height=canvas_h,
                width=canvas_w,
                drawing_mode="rect",          # Allow drawing rectangles
                # Load initial drawing from state if it exists (e.g., after rerun)
                initial_drawing=st.session_state.get("canvas_drawing"),
                key="drawable_canvas",        # Unique key for this widget
            )

            # --- Process ROI from Canvas Result ---
            if canvas_result.json_data and isinstance(canvas_result.json_data, dict) and canvas_result.json_data.get("objects"):
                try:
                    # Get the last drawn object (usually the current rectangle)
                    last_object = canvas_result.json_data["objects"][-1]
                    if last_object.get("type") == "rect":
                        # Extract coordinates from the canvas object
                        canvas_left = int(last_object.get("left", 0))
                        canvas_top = int(last_object.get("top", 0))
                        # Account for potential scaling within the object itself
                        rect_w = int(last_object.get("width", 0) * last_object.get("scaleX", 1))
                        rect_h = int(last_object.get("height", 0) * last_object.get("scaleY", 1))

                        # Scale canvas coordinates back to original image dimensions
                        scale_x = img_w / canvas_w
                        scale_y = img_h / canvas_h

                        roi_coords: Dict[str, int] = {
                            "left": max(0, int(canvas_left * scale_x)),
                            "top": max(0, int(canvas_top * scale_y)),
                            "width": max(1, int(rect_w * scale_x)), # Ensure width/height >= 1
                            "height": max(1, int(rect_h * scale_y)),
                        }

                        # Update session state only if ROI has changed
                        if roi_coords != st.session_state.get("roi_coords"):
                            st.session_state.roi_coords = roi_coords
                            # Store the drawing state to redraw the rectangle on rerun
                            st.session_state.canvas_drawing = canvas_result.json_data
                            logger.info(f"ROI updated via canvas: {roi_coords}")
                            # Provide immediate feedback (optional)
                            # st.info(f"ROI set: ({roi_coords['left']},{roi_coords['top']}) {roi_coords['width']}x{roi_coords['height']}px", icon="🎯")
                except (IndexError, KeyError, ValueError, TypeError) as e:
                     logger.error(f"Error processing canvas JSON data for ROI: {e}", exc_info=True)
                     st.warning("Could not process the drawn ROI coordinates.")

        else:
            # Fallback: Display static image if canvas is unavailable
            st.image(img_to_display, caption="Image Preview", use_column_width='auto')
            if not _DRAWABLE_CANVAS_AVAILABLE:
                st.caption("Install `streamlit-drawable-canvas` for ROI drawing.")

        # --- Display Current ROI Coordinates ---
        current_roi = st.session_state.get("roi_coords")
        if isinstance(current_roi, dict):
            try:
                roi_text = (f"Current ROI: ({current_roi['left']}, {current_roi['top']}), "
                            f"Size: {current_roi['width']} x {current_roi['height']} px")
                st.caption(roi_text, help="Region of Interest coordinates on the original image.")
            except KeyError:
                 logger.warning("roi_coords in session state missing expected keys.")
                 st.caption("Current ROI: [Invalid Format]")


        # --- Display DICOM Metadata ---
        st.divider()
        if st.session_state.get("is_dicom"):
             dicom_meta = st.session_state.get("dicom_metadata")
             if dicom_meta and _UI_COMPONENTS_LOADED:
                 # Call the display function imported from ui_components
                 display_dicom_metadata(dicom_meta)
             elif dicom_meta:
                 # Fallback if ui_components didn't load
                 with st.expander("📄 DICOM Metadata (Raw)", expanded=False):
                      st.json(dicom_meta, expanded=False)
             else:
                  st.caption("No DICOM metadata extracted.")


# =============================================================================
#  2. ANALYSIS & RESULTS TABS (Right Column)
# =============================================================================

def _render_results_tabs(col) -> None:
    """Renders the tabbed interface for displaying analysis results."""
    with col:
        st.subheader("📊 Analysis & Results")

        # Define tab names and corresponding icons
        tab_titles = [
            "🔬 Initial",
            "💬 Q&A",
            "🩺 Condition",
            "📈 Confidence",
            "🌐 Translate",
            "🧬 UMLS Lookup",
        ]
        tabs = st.tabs(tab_titles)

        # --- Tab 1: Initial Analysis ---
        with tabs[0]:
            st.markdown("**Initial Analysis Findings**")
            analysis_text = st.session_state.get("initial_analysis", "")
            st.text_area(
                "Initial Analysis Output", # Use label for accessibility
                value=analysis_text if analysis_text else "Run *Initial Analysis* from the sidebar.",
                height=350,
                disabled=True,
                key="initial_analysis_textarea",
                label_visibility="collapsed" # Hide label visually, but keep for screen readers
            )
            # Display automatically mapped UMLS concepts for this analysis
            initial_umls_concepts = st.session_state.get("initial_analysis_umls", [])
            if initial_umls_concepts and _UI_COMPONENTS_LOADED:
                st.divider()
                st.markdown("**Mapped UMLS Concepts:**")
                display_umls_concepts(initial_umls_concepts) # Use component
            elif initial_umls_concepts: # Fallback display
                 st.divider()
                 st.markdown("**Mapped UMLS Concepts (Raw):**")
                 st.json([c.to_dict() for c in initial_umls_concepts], expanded=False)


        # --- Tab 2: Q&A ---
        with tabs[1]:
            st.markdown("**Latest Q&A Response**")
            qa_text = st.session_state.get("qa_answer", "")
            st.text_area(
                "Latest Answer",
                value=qa_text if qa_text else "Ask a question using the sidebar.",
                height=150,
                disabled=True,
                key="qa_answer_textarea",
                label_visibility="collapsed"
            )
            # Display automatically mapped UMLS concepts for the latest answer
            qa_umls_concepts = st.session_state.get("qa_umls", [])
            if qa_umls_concepts and _UI_COMPONENTS_LOADED:
                st.divider()
                st.markdown("**Mapped UMLS Concepts (Latest Answer):**")
                display_umls_concepts(qa_umls_concepts)
            elif qa_umls_concepts: # Fallback
                 st.divider()
                 st.markdown("**Mapped UMLS Concepts (Latest Answer - Raw):**")
                 st.json([c.to_dict() for c in qa_umls_concepts], expanded=False)

            # Display full conversation history
            st.divider()
            st.markdown("##### Conversation History")
            history: List[Tuple[str, str, Any]] = st.session_state.get("history", [])
            if not history:
                st.caption("No Q&A history yet for this session.")
            else:
                # Display in reverse chronological order (most recent first)
                # Limit number shown directly? Or use expander? For now, show all.
                with st.container(height=300): # Make history scrollable if long
                    for i, (question, answer, _) in enumerate(reversed(history)):
                        st.markdown(f"**You:** {question}")
                        # Using expander for potentially long answers
                        with st.expander(f"**AI Response #{len(history) - i}**", expanded=(i==0)): # Expand latest
                            st.markdown(answer) # Let markdown handle formatting
                        st.divider()


        # --- Tab 3: Condition-Specific Analysis ---
        with tabs[2]:
            st.markdown("**Condition-Specific Analysis**")
            condition_text = st.session_state.get("disease_analysis", "")
            st.text_area(
                "Condition Analysis Output",
                value=condition_text if condition_text else "Select a condition and run *Analyze Condition* from the sidebar.",
                height=350,
                disabled=True,
                key="condition_analysis_textarea",
                label_visibility="collapsed"
            )
            # Display automatically mapped UMLS concepts for this analysis
            disease_umls_concepts = st.session_state.get("disease_umls", [])
            if disease_umls_concepts and _UI_COMPONENTS_LOADED:
                st.divider()
                st.markdown("**Mapped UMLS Concepts:**")
                display_umls_concepts(disease_umls_concepts)
            elif disease_umls_concepts: # Fallback
                 st.divider()
                 st.markdown("**Mapped UMLS Concepts (Raw):**")
                 st.json([c.to_dict() for c in disease_umls_concepts], expanded=False)


        # --- Tab 4: Confidence Score ---
        with tabs[3]:
            st.markdown("**AI Confidence Assessment**")
            confidence_text = st.session_state.get("confidence_score", "")
            st.text_area(
                "Confidence Score Output",
                value=confidence_text if confidence_text else "Run *Estimate AI Confidence* from the sidebar after asking a question.",
                height=300,
                disabled=True,
                key="confidence_score_textarea",
                label_visibility="collapsed"
            )


        # --- Tab 5: Translate ---
        with tabs[4]:
            st.markdown("**Translate Analysis Text**")
            if not _TRANSLATION_AVAILABLE:
                # Use the config message from translation_models if available
                try:
                    from translation_models import TRANSLATION_CONFIG_MSG
                except ImportError:
                    TRANSLATION_CONFIG_MSG = "Install `deep-translator` & restart."
                st.warning(f"Translation features unavailable – {TRANSLATION_CONFIG_MSG}")
                st.caption("Ensure `deep-translator` is in `requirements.txt` and the Space is restarted.")
            else:
                # Render the translation UI panel (defined below)
                _render_translation_panel()


        # --- Tab 6: UMLS Lookup ---
        with tabs[5]:
            st.markdown("**Manual UMLS Concept Lookup**")
            if not IS_UMLS_LOOKUP_AVAILABLE:
                 # Use the specific message from config.py
                 try:
                      from config import UMLS_CONFIG_MSG
                 except ImportError:
                      UMLS_CONFIG_MSG = "Check configuration (API Key/Secrets)."
                 # Determine specific reason
                 reason = UMLS_CONFIG_MSG
                 if not _UMLS_UTILS_LOADED:
                      reason = "UMLS utilities failed to load (check dependencies)."
                 elif not _UMLS_API_KEY_PRESENT:
                      reason = "UMLS_APIKEY not found in environment/secrets."

                 st.warning(f"UMLS lookup features unavailable – {reason}")
                 st.caption("Ensure `requests` is installed, `umls_utils.py` is present, and `UMLS_APIKEY` is set in Space Secrets.")
            else:
                 # Render the UMLS lookup UI panel (defined below)
                 _render_umls_lookup_panel()


# =============================================================================
#  Helper: Translation Panel UI (_render_results_tabs -> Tab 5)
# =============================================================================

def _render_translation_panel() -> None:
    """Renders the UI elements for the translation feature."""
    if not translate: # Guard against calling if import failed
        logger.error("Attempted to render translation panel, but translate function is unavailable.")
        st.error("Internal Error: Translation function not loaded.")
        return

    st.caption("Select an analysis block or enter custom text, choose languages, and click Translate.")

    # --- Source Text Selection ---
    # Provide choices based on available analysis results in session state
    analysis_blocks = {
        "Initial Analysis": st.session_state.get("initial_analysis", ""),
        "Latest Q&A Answer": st.session_state.get("qa_answer", ""),
        "Condition Analysis": st.session_state.get("disease_analysis", ""),
        "Confidence Score": st.session_state.get("confidence_score", ""),
        "(Enter Custom Text)": "", # Option for manual input
    }
    # Only offer options that actually have text, plus the custom option
    available_options = [k for k, v in analysis_blocks.items() if v or k == "(Enter Custom Text)"]
    if not available_options: available_options = ["(Enter Custom Text)"] # Ensure custom is always there

    selected_block_key = st.selectbox(
        "Select text to translate:",
        options=available_options,
        index=0, # Default to the first available option
        key="translate_block_choice"
    )
    source_text = analysis_blocks[selected_block_key]

    # --- Text Areas ---
    # Show selected text (disabled) or allow custom input
    if selected_block_key == "(Enter Custom Text)":
        source_text_input = st.text_area(
            "Enter or paste text to translate:",
            value=st.session_state.get("custom_translate_text", ""), # Persist custom text
            height=150,
            key="custom_translate_text_area",
            placeholder="Paste text here..."
        )
        # Update session state if custom text changes
        if source_text_input != st.session_state.get("custom_translate_text", ""):
            st.session_state.custom_translate_text = source_text_input
            source_text = source_text_input # Use the latest input
    else:
        st.text_area(
            "Selected text:",
            value=source_text,
            height=150,
            disabled=True,
            key="selected_translate_text_area"
        )

    # --- Language Selection ---
    lang_col1, lang_col2 = st.columns(2)
    with lang_col1:
        # Get language names sorted alphabetically, add Auto-Detect first
        lang_options = sorted(_LANGUAGE_CODES.keys())
        source_lang_name = st.selectbox(
            "Source Language:",
            options=[_AUTO_DETECT_INDICATOR] + lang_options,
            index=0, # Default to Auto-Detect
            key="translate_source_lang",
        )
    with lang_col2:
        # Target language cannot be Auto-Detect
        target_lang_name = st.selectbox(
            "Target Language:",
            options=lang_options, # Exclude Auto-Detect
            index=lang_options.index("English") if "English" in lang_options else 0, # Default to English if available
            key="translate_target_lang",
        )

    # --- Translate Button & Action ---
    if st.button("🔄 Translate Text", key="translate_button", use_container_width=True):
        # Input validation
        text_to_translate = source_text.strip()
        if not text_to_translate:
            st.warning("Please select or enter text to translate.")
            st.stop() # Stop execution for this button press

        source_code = _AUTO_DETECT_INDICATOR if source_lang_name == _AUTO_DETECT_INDICATOR else _LANGUAGE_CODES.get(source_lang_name, "auto")
        target_code = _LANGUAGE_CODES.get(target_lang_name)

        if not target_code:
             st.error(f"Invalid target language selected: {target_lang_name}")
             st.stop()
        if source_code != _AUTO_DETECT_INDICATOR and source_code == target_code:
            st.info("Source and target languages are the same. No translation needed.")
            st.session_state.translation_result = text_to_translate # Display original text
            st.stop()

        # --- Perform Translation ---
        with st.spinner(f"Translating from {source_lang_name} to {target_lang_name}..."):
            try:
                logger.info(f"Translating text (src: {source_code}, tgt: {target_code}). Length: {len(text_to_translate)}")
                translated_text = translate(
                    text=text_to_translate,
                    target_language=target_code,
                    source_language=source_code, # Pass code or 'Auto' string
                )
                st.session_state.translation_result = translated_text
                st.session_state.translation_error = None # Clear previous errors
                st.success("Translation complete!")
                logger.info("Translation successful.")
            except Exception as e:
                error_message = f"Translation failed: {e}"
                st.error(error_message)
                logger.error(f"Translation failed: {e}", exc_info=True)
                st.session_state.translation_result = None # Clear result on error
                st.session_state.translation_error = error_message # Store error message

    # --- Display Translation Result or Error ---
    if "translation_result" in st.session_state and st.session_state.translation_result is not None:
        st.text_area(
            "Translated Text:",
            value=st.session_state.translation_result, # Directly display result
            height=200,
            disabled=True,
            key="translation_result_textarea",
        )
    elif "translation_error" in st.session_state and st.session_state.translation_error:
         # Optionally redisplay the error if needed, though the st.error above might suffice
         # st.info(f"Previous translation error: {st.session_state.translation_error}")
         pass


# =============================================================================
#  Helper: UMLS Lookup Panel UI (_render_results_tabs -> Tab 6)
# =============================================================================

def _render_umls_lookup_panel() -> None:
    """Renders the UI elements for the manual UMLS lookup feature."""
    if not search_umls: # Guard against calling if import failed
        logger.error("Attempted to render UMLS panel, but search_umls function is unavailable.")
        st.error("Internal Error: UMLS search function not loaded.")
        return

    # Input field for search term
    search_term_input = st.text_input(
        label="Enter a medical term to look up in UMLS:",
        # Use session state to remember the last search term
        value=st.session_state.get("umls_lookup_term", ""),
        key="umls_lookup_input",
        placeholder="e.g., pneumonia, myocardial infarction",
    )
    # Update session state immediately if input changes
    if search_term_input != st.session_state.get("umls_lookup_term", ""):
         st.session_state.umls_lookup_term = search_term_input

    # --- Search Button & Action ---
    if st.button("Search UMLS", key="umls_lookup_button", use_container_width=True):
        term_to_search = st.session_state.get("umls_lookup_term", "").strip()
        if not term_to_search:
            st.warning("Please enter a term to search.")
            st.stop()

        # Retrieve API key (should be present if this panel is rendered)
        umls_api_key = os.getenv("UMLS_APIKEY")
        if not umls_api_key:
             # This check is slightly redundant if IS_UMLS_LOOKUP_AVAILABLE is correct, but safe
             st.error("UMLS API Key configuration error.")
             logger.error("UMLS API Key not found despite UI being rendered.")
             st.stop()

        # --- Perform UMLS Search ---
        with st.spinner(f"Searching UMLS for '{term_to_search}'..."):
            try:
                logger.info(f"Performing manual UMLS lookup for: '{term_to_search}'")
                # Use configured default hits, allow override if needed later
                concepts_found: List[UMLSConcept] = search_umls(
                    term=term_to_search,
                    apikey=umls_api_key,
                    page_size=DEFAULT_UMLS_HITS
                )
                # Store results (even if empty) in session state
                st.session_state.umls_lookup_results = concepts_found
                st.session_state.umls_lookup_error = None # Clear previous errors
                logger.info(f"Manual UMLS lookup found {len(concepts_found)} concepts.")
                if not concepts_found:
                     # Provide feedback directly if no results
                     st.info(f"No UMLS concepts found matching '{term_to_search}'.")

            except UMLSAuthError as auth_err:
                error_msg = f"UMLS Authentication Failed: {auth_err}. Please check your UMLS_APIKEY."
                st.error(error_msg)
                logger.error(f"Manual UMLS lookup failed: {error_msg}")
                st.session_state.umls_lookup_results = None
                st.session_state.umls_lookup_error = error_msg
            except Exception as e:
                error_msg = f"UMLS Search Error: {e}"
                st.error(error_msg)
                logger.error(f"Manual UMLS lookup failed: {e}", exc_info=True)
                st.session_state.umls_lookup_results = None
                st.session_state.umls_lookup_error = error_msg

    # --- Display UMLS Results ---
    # Check for results stored in session state from the last successful search
    if "umls_lookup_results" in st.session_state and st.session_state.umls_lookup_results is not None:
        results_to_display = st.session_state.umls_lookup_results
        if results_to_display: # Only display if list is not empty
             st.divider()
             if _UI_COMPONENTS_LOADED:
                  # Use the dedicated display component
                  display_umls_concepts(results_to_display, search_term=st.session_state.get("umls_lookup_term"))
             else:
                  # Fallback raw display if component missing
                  st.markdown("**UMLS Concepts Found (Raw):**")
                  st.json([c.to_dict() for c in results_to_display], expanded=False) # Use concept's own dict method
    elif "umls_lookup_error" in st.session_state and st.session_state.umls_lookup_error:
         # Optionally redisplay the error if needed
         # st.info(f"Previous lookup error: {st.session_state.umls_lookup_error}")
         pass


# =============================================================================
#  Public Entry Point Function
# =============================================================================

def render_main_content(col_left, col_right) -> None:
    """
    Renders the main content area of the application, populating the
    provided left and right columns with the image viewer and results tabs,
    respectively.

    Args:
        col_left: The Streamlit column object for the left side (image viewer).
        col_right: The Streamlit column object for the right side (results tabs).
    """
    logger.debug("Rendering main content area...")
    _render_image_viewer(col_left)
    _render_results_tabs(col_right)
    logger.debug("Main content area rendering complete.")