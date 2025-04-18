# -*- coding: utf-8 -*-
"""
app.py - Main Streamlit application for RadVision AI Advanced.

Handles image uploading (DICOM, JPG, PNG), display, ROI selection,
interaction with AI models for analysis and Q&A, translation,
and report generation.  Now also includes UMLS lookup.

IMPORTANT CHANGES:
- Integrated UMLS lookup functionality.
- Restructured UI to include UMLS tab and sidebar elements.
- Enhanced error handling for UMLS interactions.
"""

import os
import logging
import streamlit as st
import io
import uuid
import hashlib
import copy
import re
from PIL import Image
from typing import Optional, List, Dict, Any

# Import necessary variables from file_processing
try:
    from file_processing import PYDICOM_AVAILABLE, PIL_AVAILABLE
    DICOM_UTILS_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    PIL_AVAILABLE = False
    DICOM_UTILS_AVAILABLE = False
    logging.warning("Could not import variables from file_processing.py")

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="RadVision AI Advanced",
    layout="wide",
    page_icon="⚕️",
    initial_sidebar_state="expanded"
)

# Import authentication module
try:
    from auth import is_authenticated, render_login_page
    AUTH_ENABLED = True
except ImportError:
    logger.warning("Authentication module not found. Running without auth.")
    AUTH_ENABLED = False

# Check authentication if enabled
if AUTH_ENABLED:
    is_auth, username = is_authenticated()
    if not is_auth:
        render_login_page()
        st.stop()  # Stop execution if not authenticated
    else:
        logger.info(f"User authenticated: {username}")

# Import UMLS utilities
try:
    from umls_utils import search_umls, UMLSAuthError, UMLSConcept
    UMLS_UTILS_AVAILABLE = True
    # Check if API key is available
    api_key = os.environ.get("UMLS_APIKEY")
    if api_key:
        logger.info("UMLS utilities imported successfully with API key.")
    else:
        logger.warning("UMLS utilities imported but UMLS_APIKEY is not set.")
except ImportError as e:
    UMLS_UTILS_AVAILABLE = False
    logger.error(f"Failed to import UMLS utilities: {e}")

# Import other modules as needed
try:
    from config import DEFAULT_STATE, TIPS, DISEASE_OPTIONS, APP_CSS
    from dicom_utils import parse_dicom, extract_dicom_metadata, dicom_to_image, get_default_wl
    from translation_models import translate, detect_language, LANGUAGE_CODES, AUTO_DETECT_INDICATOR
    from ui_components import display_dicom_metadata, display_umls_concepts
    from report_utils import generate_pdf_report_bytes, REPORT_UTILS_AVAILABLE
    from llm_interactions import run_initial_analysis, run_multimodal_qa, run_disease_analysis, estimate_ai_confidence
    from action_handlers import LLM_INTERACTIONS_AVAILABLE
    from image_processor import process_uploaded_file
    from error_recovery import recover_on_error, safe_execution
    from caching import clear_expired_cache_files
    from performance_profiler import profile_time

    # Add import status logs
    logger.info("Core modules imported successfully.")
except ImportError as e:
    logger.critical(f"Failed to import essential modules: {e}")
    st.error(f"Critical error: Failed to load essential modules. {e}")
    st.stop()

# Apply custom CSS
st.markdown(APP_CSS, unsafe_allow_html=True)

# Initialize session state
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Create a unique session ID if not exists
if not st.session_state.get("session_id"):
    import uuid
    st.session_state.session_id = str(uuid.uuid4())[:8]
    logger.info(f"New session initialized: {st.session_state.session_id}")

# Render sidebar using the dedicated module
try:
    from sidebar_ui import render_sidebar
    render_sidebar()  # This function handles all sidebar UI elements
except ImportError:
    # Fallback if sidebar_ui.py is missing or has errors
    logger.error("Could not import render_sidebar from sidebar_ui.py. Using fallback sidebar.")

    # Fallback sidebar implementation
    with st.sidebar:
        st.header("⚕️ RadVision Controls")
        st.markdown("---")

        # Display random tip
        import random
        st.info(f"💡 {random.choice(TIPS)}")
        st.markdown("---")

        # Image upload section
        st.header("Image Upload & Settings")
        uploaded_file = st.file_uploader(
            "Upload Image (JPG, PNG, DCM)",
            type=["jpg", "jpeg", "png", "dcm", "dicom"],
            key="file_uploader_widget",
            help="Upload a medical image file for analysis. DICOM (.dcm) is preferred."
        )

        # Demo mode and ROI controls
        demo_mode = st.checkbox("🚀 Demo Mode", value=st.session_state.get("demo_loaded", False), 
                              help="Load demo image from assets/demo.png")
        if demo_mode and not st.session_state.get("demo_loaded"):
            # Load demo image when demo mode is enabled
            try:
                with open("assets/demo.png", "rb") as f:
                    st.session_state.demo_loaded = True
                    st.toast("Demo image loaded", icon="✅")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to load demo image: {e}")

        if st.button("🗑️ Clear ROI", help="Remove the selected ROI"):
            st.session_state.roi_coords = None
            st.session_state.canvas_drawing = None
            st.session_state.clear_roi_feedback = True
            st.rerun()

        # AI analysis actions
        st.markdown("---")
        st.header("🤖 AI Analysis Actions")

        action_disabled = not isinstance(st.session_state.get("display_image"), Image.Image)

        if st.button("▶️ Run Initial Analysis", disabled=action_disabled):
            st.session_state.last_action = "analyze"
            st.rerun()

        st.subheader("❓ Ask AI a Question")
        question_input = st.text_area(
            "Enter your question:",
            height=100,
            key="question_input_widget",
            placeholder="E.g., 'Are there any nodules in the upper right lobe?'",
            disabled=action_disabled
        )
        if st.button("💬 Ask Question", key="ask_btn", disabled=action_disabled):
            if question_input.strip():
                st.session_state.last_action = "ask"
                st.rerun()
            else:
                st.warning("Please enter a question before submitting.")

        st.subheader("🎯 Condition-Specific Analysis")
        disease_select = st.selectbox(
            "Select condition to focus on:",
            options=[""] + sorted(DISEASE_OPTIONS),
            key="disease_select_widget",
            disabled=action_disabled
        )
        if st.button("🩺 Analyze Condition", key="disease_btn", disabled=action_disabled):
            if disease_select:
                st.session_state.last_action = "disease"
                st.rerun()
            else:
                st.warning("Please select a condition first.")

        # Add confidence button
        st.markdown("---")
        st.header("📊 Confidence & Reporting")

        can_conf = bool(
            st.session_state.get("history") or
            st.session_state.get("initial_analysis") or
            st.session_state.get("disease_analysis")
        )

        if st.button(
            "📈 Estimate AI Confidence",
            key="confidence_btn",
            disabled=(not action_disabled) or (not can_conf),
            help="Assess how confident the AI is in its analysis."
        ):
            st.session_state.last_action = "confidence"
            st.rerun()

        # UMLS Lookup
        if UMLS_UTILS_AVAILABLE:
            st.markdown("---")
            st.header("🔍 UMLS Lookup")
            umls_term = st.text_input(
                "Enter medical term to lookup:",
                key="umls_lookup_term",
                disabled=not UMLS_UTILS_AVAILABLE
            )
            if st.button("🔍 Search UMLS", disabled=not UMLS_UTILS_AVAILABLE):
                if umls_term.strip():
                    st.session_state.last_action = "umls_lookup"
                    st.rerun()
                else:
                    st.warning("Please enter a term to search.")


# --- Handle Different View States ---
active_view = st.session_state.get("active_view", "main")

# Handle different dashboard views
if active_view == "umls_dashboard" and UMLS_UTILS_AVAILABLE:
    try:
        from umls_dashboard import render_umls_dashboard
        render_umls_dashboard()
        st.stop()  # Stop further rendering
    except ImportError as e:
        st.error(f"Could not load UMLS dashboard: {e}")
        # Reset to main view if dashboard fails
        st.session_state.active_view = "main"
elif active_view == "fda_dashboard":
    try:
        from fda_dashboard import render_fda_dashboard
        render_fda_dashboard()
        st.stop()  # Stop further rendering
    except ImportError as e:
        st.error(f"Could not load FDA dashboard: {e}")
        # Reset to main view if dashboard fails
        st.session_state.active_view = "main"

# --- Standard Main UI: ROI Selection Logic ---
col1, col2 = st.columns([2, 3])

with col1:
    if st.session_state.get("display_image") is not None:
        st.subheader("🖼️ Image Viewer")

        # Create a canvas for ROI selection
        try:
            from streamlit_drawable_canvas import st_canvas
            canvas_available = True
        except ImportError:
            canvas_available = False

        # Import fallback canvas
        try:
            from custom_canvas import st_roi_selector
            fallback_canvas_available = True
        except ImportError:
            fallback_canvas_available = False

        if canvas_available:
            # Resize image for canvas if too large
            img = st.session_state.display_image
            max_dim = 800
            if img.width > max_dim or img.height > max_dim:
                aspect = img.width / img.height
                if img.width > img.height:
                    new_w, new_h = max_dim, int(max_dim / aspect)
                else:
                    new_w, new_h = int(max_dim * aspect), max_dim
                img = img.resize((new_w, new_h))

            # Draw canvas for ROI selection
            try:
                # Create a temporary buffer to save the image
                import io
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
                img_bytes = img_buffer.getvalue()

                # Show static image first as fallback
                st.image(img, caption="Image Preview", use_container_width=True)

                # Then try canvas with direct bytes
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",
                    stroke_width=2,
                    stroke_color="#FF0000",
                    background_image=img,
                    update_streamlit=True,
                    height=img.height,
                    width=img.width,
                    drawing_mode="rect",
                    key="canvas",
                    display_toolbar=True,
                )

                # Process ROI if drawn
                if canvas_result and canvas_result.json_data and "objects" in canvas_result.json_data:
                    objects = canvas_result.json_data["objects"]
                    if objects:
                        # Get the last drawn rectangle
                        last_rect = objects[-1]
                        # Scale ROI back to original image if resized
                        scale_x = st.session_state.display_image.width / img.width
                        scale_y = st.session_state.display_image.height / img.height

                        st.session_state.roi_coords = {
                            "left": int(last_rect.get("left", 0) * scale_x),
                            "top": int(last_rect.get("top", 0) * scale_y),
                            "width": int(last_rect.get("width", 0) * scale_x),
                            "height": int(last_rect.get("height", 0) * scale_y)
                        }
                        st.success("ROI selected! Use the AI analysis buttons to analyze this region.")
            except Exception as e:
                st.error(f"Canvas drawing feature unavailable: {str(e)}")
                st.image(img, caption="Uploaded Image (ROI selection unavailable)", use_container_width=True)
                canvas_result = None
                if fallback_canvas_available:
                    st.info("Falling back to alternative ROI selection method.")
                    try:
                        roi_coords = st_roi_selector(img)
                        if roi_coords:
                            st.session_state.roi_coords = roi_coords
                            st.success("ROI selected using fallback method!")
                    except Exception as fe:
                        st.error(f"Fallback ROI selection failed: {fe}")

        elif fallback_canvas_available:
            st.info("Using fallback ROI selection method (streamlit_drawable_canvas not found).")
            try:
                roi_coords = st_roi_selector(img)
                if roi_coords:
                    st.session_state.roi_coords = roi_coords
                    st.success("ROI selected!")
            except Exception as fe:
                st.error(f"Fallback ROI selection failed: {fe}")
        else:
            st.image(st.session_state.display_image, caption="Uploaded Image", use_container_width=True)
            st.warning("ROI selection requires streamlit_drawable_canvas or a custom fallback. Install it for ROI features.")

        # Display DICOM metadata if available
        if st.session_state.is_dicom and st.session_state.dicom_metadata:
            with st.expander("📄 DICOM Metadata", expanded=False):
                display_dicom_metadata(st.session_state.dicom_metadata)

# --- File Upload Logic ---
# Get the uploaded_file from session state (set by sidebar)
uploaded_file = st.session_state.get("file_uploader_widget")

# Import the image processor module
from image_processor import process_uploaded_file
from error_recovery import recover_on_error, safe_execution
from caching import clear_expired_cache_files
from performance_profiler import profile_time

# Clean expired cache files
clear_expired_cache_files()

# Process uploaded file
@profile_time
@recover_on_error('file_upload')
def handle_uploaded_file(uploaded_file):
    """Process the uploaded file and update session state."""
    try:
        # Process the file
        result = process_uploaded_file(uploaded_file)

        if not result['success']:
            st.error(f"Error processing file: {result['error']}")
            return False

        # Update session state with processing results
        keys_to_preserve = {"file_uploader_widget", "session_id", "uploaded_file_info", "demo_loaded", "umls_lookup_term"}
        st.session_state.session_id = st.session_state.get("session_id", str(uuid.uuid4())[:8])

        for key, value in DEFAULT_STATE.items():
            if key not in keys_to_preserve:
                st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value

        # Store processing results in session state
        st.session_state.uploaded_file_info = result['file_info']
        st.session_state.demo_loaded = False
        st.session_state.is_dicom = result['is_dicom']
        st.session_state.display_image = result['display_image']
        st.session_state.processed_image = result['processed_image']

        if result['is_dicom']:
            st.session_state.dicom_dataset = result['dicom_dataset']
            st.session_state.dicom_metadata = result['dicom_metadata']

            # Get default window/level if available
            if result['dicom_dataset'] and 'WindowCenter' in dir(result['dicom_dataset']):
                try:
                    from dicom_utils import get_default_wl
                    default_wc, default_ww = get_default_wl(result['dicom_dataset'])
                    st.session_state.current_display_wc = default_wc
                    st.session_state.current_display_ww = default_ww
                except Exception as e:
                    logger.warning(f"Could not get default window/level: {e}")

        # Store raw bytes for possible reprocessing
        st.session_state.raw_image_bytes = uploaded_file.getvalue()

        logger.info(f"File processed successfully: {uploaded_file.name}")
        return True

    except Exception as e:
        logger.error(f"Error in handle_uploaded_file: {e}", exc_info=True)
        st.error(f"Error processing file: {e}")
        return False

# Handle the file upload
if uploaded_file is not None:
    file_info = None

    try:
        # Get a unique identifier for the file
        uploaded_file.seek(0)
        file_content_hash = hashlib.sha256(uploaded_file.read()).hexdigest()[:16]
        uploaded_file.seek(0)
        file_info = f"{uploaded_file.name}-{uploaded_file.size}-{file_content_hash}"
    except Exception as e:
        logger.warning(f"Could not generate hash for file: {e}")
        file_info = f"{uploaded_file.name}-{uploaded_file.size}-{uuid.uuid4().hex[:8]}"

    # Check if this is a new file
    if file_info != st.session_state.get("uploaded_file_info"):
        logger.info(f"New file uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
        st.toast(f"Processing '{uploaded_file.name}'...", icon="⏳")

        with st.spinner("🔬 Analyzing file..."):
            processing_success = handle_uploaded_file(uploaded_file)

            if processing_success:
                st.success(f"✅ '{uploaded_file.name}' loaded successfully!")
                logger.info(f"Image processed: {uploaded_file.name}")
                st.rerun()
            else:
                st.error("Image loading failed. Check format or try again.")
                logger.error(f"Image processing failed for file: {uploaded_file.name}")
                st.session_state.uploaded_file_info = None
                st.session_state.display_image = None
                st.session_state.processed_image = None
                st.session_state.is_dicom = False

# --- Display Uploaded Image (Minimal Addition) ---
# If an image has been processed, immediately display it.
if st.session_state.get("display_image") is not None:
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("🖼️ Image Viewer")
        st.image(st.session_state.display_image, caption="Uploaded Image",use_container_width=True)

        # Display DICOM metadata if available
        if st.session_state.is_dicom and st.session_state.dicom_metadata:
            with st.expander("📄 DICOM Metadata", expanded=False):
                display_dicom_metadata(st.session_state.dicom_metadata)

    with col2:
        st.subheader("📊 Analysis & Results")
        tab_titles = [
            "🔬 Initial Analysis",
            "💬 Q&A History",
            "🩺 Condition Focus",
            "📈 Confidence",
            "🌐 Translation",
            "🧬 UMLS Lookup"
        ]
        tabs = st.tabs(tab_titles)

        # Initial Analysis tab
        with tabs[0]:
            st.text_area(
                "Overall Findings & Impressions",
                value=st.session_state.initial_analysis or "Run 'Initial Analysis' to see results here.",
                height=400,
                disabled=True
            )

            # Display UMLS concepts if available
            if st.session_state.get("initial_analysis_umls") and UMLS_UTILS_AVAILABLE:
                with st.expander("🧬 Linked UMLS Concepts", expanded=False):
                    display_umls_concepts(st.session_state.initial_analysis_umls)

        # Q&A History tab
        with tabs[1]:
            st.text_area(
                "Latest AI Answer",
                value=st.session_state.qa_answer or "Ask a question to see AI's response here.",
                height=200,
                disabled=True
            )

            # Display conversation history
            if st.session_state.history:
                with st.expander("Full Conversation History", expanded=True):
                    for i, (q_type, message) in enumerate(reversed(st.session_state.history)):
                        if q_type.lower() == "user question":
                            st.markdown(f"**You:** {message}")
                        elif q_type.lower() == "ai answer":
                            st.markdown(f"**AI:** {message}")
                        elif q_type.lower() == "system":
                            st.info(f"*{message}*", icon="ℹ️")
                        else:
                            st.markdown(f"**{q_type}:** {message}")
                        if i < len(st.session_state.history) - 1:
                            st.markdown("---")
            else:
                st.caption("No questions asked yet.")

            # Display UMLS concepts if available
            if st.session_state.get("qa_umls") and UMLS_UTILS_AVAILABLE:
                with st.expander("🧬 Linked UMLS Concepts", expanded=False):
                    display_umls_concepts(st.session_state.qa_umls)

        # Condition Focus tab
        with tabs[2]:
            st.text_area(
                "Condition-Specific Analysis",
                value=st.session_state.disease_analysis or "Select a condition and click 'Analyze Condition'.",
                height=400,
                disabled=True
            )

            # Display UMLS concepts if available
            if st.session_state.get("disease_umls") and UMLS_UTILS_AVAILABLE:
                with st.expander("🧬 Linked UMLS Concepts", expanded=False):
                    display_umls_concepts(st.session_state.disease_umls)

        # Confidence tab
        with tabs[3]:
            st.text_area(
                "Estimated AI Confidence",
                value=st.session_state.confidence_score or "Run 'Estimate AI Confidence' after analysis.",
                height=400,
                disabled=True
            )

        # Translation tab
        with tabs[4]:
            st.subheader("🌐 Translate Analysis Text")

            if not "translate" in globals():
                st.warning("Translation features are unavailable.")
            else:
                # Translation source selection
                text_options = {
                    "Initial Analysis": st.session_state.initial_analysis,
                    "Latest Q&A Answer": st.session_state.qa_answer,
                    "Condition Analysis": st.session_state.disease_analysis,
                    "Confidence Estimation": st.session_state.confidence_score,
                    "(Enter Custom Text Below)": ""
                }

                available_options = {
                    label: txt for label, txt in text_options.items() if txt or label == "(Enter Custom Text Below)"
                }

                selected_label = st.selectbox(
                    "Select text to translate:",
                    list(available_options.keys()),
                    index=0
                )

                text_to_translate = available_options.get(selected_label, "")
                if selected_label == "(Enter Custom Text Below)":
                    text_to_translate = st.text_area(
                        "Enter text to translate:",
                        value="",
                        height=150
                    )

                # Display selected text
                st.text_area(
                    "Text selected for translation:",
                    value=text_to_translate,
                    height=100,
                    disabled=True
                )

                # Language selection
                col_lang1, col_lang2 = st.columns(2)
                with col_lang1:
                    source_language_options = [AUTO_DETECT_INDICATOR] + sorted(list(LANGUAGE_CODES.keys()))
                    source_language_name = st.selectbox(
                        "Source Language:",
                        source_language_options,
                        index=0
                    )
                with col_lang2:
                    target_language_options = sorted(list(LANGUAGE_CODES.keys()))
                    default_target_index = 0
                    if "English" in target_language_options:
                        default_target_index = target_language_options.index("English")

                    target_language_name = st.selectbox(
                        "Translate To:",
                        target_language_options,
                        index=default_target_index
                    )

                # Translation button
                if st.button("🔄 Translate Now"):
                    if not text_to_translate.strip():
                        st.warning("Please select or enter some text first.")
                    elif source_language_name == target_language_name and source_language_name != AUTO_DETECT_INDICATOR:
                        st.info("Source and target are the same; no translation needed.")
                        st.session_state.translation_result = text_to_translate
                    else:
                        with st.spinner(f"Translating from '{source_language_name}' to '{target_language_name}'..."):
                            try:
                                translation_output = translate(
                                    text=text_to_translate,
                                    target_language=target_language_name,
                                    source_language=source_language_name
                                )
                                if translation_output is not None:
                                    st.session_state.translation_result = translation_output
                                    st.success("Translation complete!")
                                else:
                                    st.error("Translation returned no result.")
                                    st.session_state.translation_error = "Service returned None."
                            except Exception as e:
                                st.error(f"Translation error: {e}")
                                st.session_state.translation_error = str(e)

                # Display translation result
                if st.session_state.get("translation_result"):
                    st.text_area("Translated Text:", value=st.session_state.translation_result, height=200)


        # UMLS Lookup tab
        with tabs[5]:
            st.subheader("🧬 UMLS Concept Lookup")

            if not UMLS_UTILS_AVAILABLE:
                st.warning("UMLS utilities are not available. Check your configuration.")
            else:
                st.info("Search for medical terms to find standardized UMLS concepts.")

                # Display UMLS lookup results if available
                if st.session_state.get("umls_lookup_results"):
                    st.success(f"Found {len(st.session_state.umls_lookup_results)} concepts")
                    display_umls_concepts(st.session_state.umls_lookup_results)
                elif st.session_state.get("umls_lookup_error"):
                    st.error(f"Lookup error: {st.session_state.umls_lookup_error}")

# --- Button Action Handlers ---
current_action = st.session_state.get("last_action")
if current_action:
    logger.info(f"Handling action: '{current_action}' for session: {st.session_state.session_id}")
    action_requires_image = current_action not in ["generate_report_data", "umls_lookup"]
    action_requires_llm = current_action in ["analyze", "ask", "disease", "confidence"]
    action_requires_report_util = (current_action == "generate_report_data")

    # Import report_utils module first to ensure it's defined
    try:
        from report_utils import REPORT_UTILS_AVAILABLE
    except ImportError:
        REPORT_UTILS_AVAILABLE = False
        logger.error("Failed to import report_utils module.")

    if action_requires_report_util and not REPORT_UTILS_AVAILABLE:
        st.error("Report generation module unavailable.")
        st.session_state.last_action = None 
        st.stop()

    if action_requires_image and not isinstance(st.session_state.get("processed_image"), Image.Image):
        st.error(f"No valid image for '{current_action}'. Please upload an image.")
        st.session_state.last_action = None
        st.stop()
    if not st.session_state.session_id:
        st.error("No session ID available—cannot continue.")
        st.session_state.last_action = None
        st.stop()
    if action_requires_llm and not LLM_INTERACTIONS_AVAILABLE:
        st.error("Core AI module unavailable.")
        st.session_state.last_action = None
        st.stop()

    img_for_llm = st.session_state.processed_image
    roi_coords = st.session_state.roi_coords
    current_history = st.session_state.history
    if not isinstance(current_history, list):
        current_history = []
        st.session_state.history = current_history

    try:
        if current_action == "analyze":
            st.info("🔬 Performing initial analysis...")
            with st.spinner("AI analyzing..."):
                analysis_result = run_initial_analysis(img_for_llm, roi=roi_coords)
                #Added UMLS lookup for initial analysis
                if UMLS_UTILS_AVAILABLE:
                    api_key = os.environ.get("UMLS_APIKEY")
                    if api_key:
                        try:
                            # Extract key terms from the analysis for better UMLS lookup
                            import re
                            # Look for diagnostic phrases often found in medical reports
                            diagnostic_findings = re.findall(r'(?:diagnosis|impression|finding)s?[:\s]+([^\.;]+)[\.;]', 
                                                        analysis_result, re.IGNORECASE)
                            # Look for anatomical structures with abnormalities
                            anatomical_terms = re.findall(r'(?:opacity|mass|lesion|effusion|consolidation|nodule)s?\s+(?:in|of|at)?\s+(?:the)?\s+([^\.;,]+)', 
                                                        analysis_result, re.IGNORECASE)
                            # Extract common disease names that might appear
                            disease_terms = re.findall(r'(?:pneumonia|edema|fracture|cancer|tumor|malignancy|inflammation)(?:[^\.\n;]*)', 
                                                    analysis_result, re.IGNORECASE)

                            # Build search text from key terms
                            search_terms = []
                            if diagnostic_findings: search_terms.extend(diagnostic_findings[:2])
                            if anatomical_terms: search_terms.extend(anatomical_terms[:2])
                            if disease_terms: search_terms.extend(disease_terms[:2])

                            search_text = " ".join(search_terms[:5]) if search_terms else analysis_result[:300]
                            logger.info(f"UMLS search text: {search_text}")

                            # Get UMLS concepts with max results and vocabulary filter
                            max_results = st.session_state.get("umls_max_results", 5)
                            vocab_filters = st.session_state.get("umls_vocab_filter")

                            umls_concepts = search_umls(
                                term=search_text, 
                                apikey=api_key, 
                                page_size=max_results,
                                source_vocabs=vocab_filters
                            )
                            st.session_state.initial_analysis_umls =umls_concepts
                            if umls_concepts:
                                logger.info(f"Found {len(umlsconcepts)} UMLS concepts for initial analysis")
                            else:
                                logger.info("No UMLS concepts found for initial analysis")
                        except Exception as umls_e:
                            logger.error(f"UMLS lookup failed for initial analysis: {umls_e}", exc_info=True)
                            st.warning(f"UMLS lookup failed for initial analysis: {str(umls_e)[:100]}")
                    else:
                        logger.warning("UMLS_APIKEY not set, skipping UMLS lookup.")
            st.session_state.initial_analysis = analysis_result
            st.session_state.qa_answer = ""
            st.session_state.disease_analysis = ""
            logger.info("Initial analysis complete.")
            st.success("Initial analysis complete!")

        elif current_action == "ask":
            question_text = st.session_state.question_input_widget.strip()
            if not question_text:
                st.warning("Question is empty.")
            else:
                st.info(f"Asking AI: '{question_text}'...")
                st.session_state.qa_answer = ""
                with st.spinner("Thinking..."):
                    answer, success_flag = run_multimodal_qa(
                        img_for_llm,
                        question_text,
                        current_history,
                        roi=roi_coords
                    )
                    #Added UMLS lookup for QA
                    if UMLS_UTILS_AVAILABLE:
                        api_key = os.environ.get("UMLS_APIKEY")
                        if api_key:
                            try:
                                umls_concepts = search_umls(text=answer, apikey=api_key)
                                st.session_state.qa_umls = umls_concepts
                            except Exception as umls_e:
                                st.warning(f"UMLS lookup failed for QA: {umls_e}")
                        else:
                            st.warning("UMLS_APIKEY not set, skipping UMLS lookup.")

                if success_flag:
                    st.session_state.qa_answer = answer
                    st.session_state.history.append(("User Question", question_text))
                    st.session_state.history.append(("AI Answer", answer))
                    st.success("AI answered your question!")
                else:
                    primary_error_msg = f"Primary AI failed: {answer}"
                    st.session_state.qa_answer = primary_error_msg
                    st.error(primary_error_msg)
                    hf_token = os.environ.get("HF_API_TOKEN")
                    if HF_MODELS_AVAILABLE and hf_token:
                        st.info(f"Attempting fallback HF model: {HF_VQA_MODEL_ID}")
                        with st.spinner("Trying fallback..."):
                            fallback_answer, fallback_success = query_hf_vqa_inference_api(
                                img_for_llm, question_text, roi=roi_coords
                            )
                        if fallback_success:
                            fallback_display = f"**[Fallback: {HF_VQA_MODEL_ID}]**\n\n{fallback_answer}"
                            st.session_state.qa_answer += "\n\n" + fallback_display
                            st.session_state.history.append(("[Fallback] User Question", question_text))
                            st.session_state.history.append(("[Fallback] AI Answer", fallback_display))
                            st.success("Fallback AI answered.")
                        else:
                            fallback_error_msg = f"[Fallback Error - {HF_VQA_MODEL_ID}]: {fallback_answer}"
                            st.session_state.qa_answer += f"\n\n{fallback_error_msg}"
                            st.error("Fallback AI also failed.")
                    elif HF_MODELS_AVAILABLE and not hf_token:
                        st.session_state.qa_answer += "\n\n[Fallback Skipped: HF_API_TOKEN missing]"
                        st.warning("Hugging Face API token needed for fallback.")
                    else:
                        st.session_state.qa_answer += "\n\n[Fallback Unavailable]"
                        st.warning("No fallback AI is configured.")

        elif current_action == "disease":
            selected_disease = st.session_state.disease_select_widget
            if not selected_disease:
                st.warning("No condition selected.")
            else:
                st.info(f"Analyzing for '{selected_disease}'...")
                with st.spinner("AI analyzing condition..."):
                    disease_result = run_disease_analysis(img_for_llm, selected_disease, roi=roi_coords)
                                        #Added UMLS lookup for disease analysis
                    if UMLS_UTILS_AVAILABLE:
                        api_key = os.environ.get("UMLS_APIKEY")
                        if api_key:
                            try:
                                umls_concepts = search_umls(text=disease_result, apikey=api_key)
                                st.session_state.disease_umls = umls_concepts
                            except Exception as umls_e:
                                st.warning(f"UMLS lookup failed for disease analysis: {umls_e}")
                        else:
                            st.warning("UMLS_APIKEY not set, skipping UMLS lookup.")
                st.session_state.disease_analysis = disease_result
                st.session_state.qa_answer = ""
                logger.info(f"Disease analysis for {selected_disease} complete.")
                st.success(f"Analysis for '{selected_disease}' complete!")

        elif current_action == "confidence":
            if not (current_history or st.session_state.initial_analysis or st.session_state.disease_analysis):
                st.warning("No prior analysis to estimate confidence.")
            else:
                st.info("📊 Estimating confidence...")
                with st.spinner("Calculating confidence..."):
                    confidence_result = estimate_ai_confidence(
                        img_for_llm,
                        current_history,
                        roi=roi_coords
                    )
                st.session_state.confidence_score = confidence_result
                st.success("Confidence estimation complete!")
                logger.info("Confidence estimation complete.")

        elif current_action == "generate_report_data":
            st.info("📄 Generating PDF report data...")
            st.session_state.pdf_report_bytes = None
            image_for_report = st.session_state.get("display_image")
            if not isinstance(image_for_report, Image.Image):
                st.error("Cannot generate report: No valid image in memory.")
            else:
                final_image_for_pdf = image_for_report.copy().convert("RGB")
                if roi_coords:
                    try:
                        from PIL import ImageDraw #Corrected import location
                        draw = ImageDraw.Draw(final_image_for_pdf)
                        x0, y0 = roi_coords['left'], roi_coords['top']
                        x1, y1 = x0 + roi_coords['width'], y0 + roi_coords['height']
                        draw.rectangle(
                            [x0, y0, x1, y1],
                            outline="red",
                            width=max(3, int(min(final_image_for_pdf.size) * 0.005))
                        )
                        logger.info("ROI box drawn on PDF image.")
                    except Exception as e:
                        logger.error(f"Error drawing ROI on PDF image: {e}", exc_info=True)
                        st.warning("Could not draw ROI on the PDF image.")

                formatted_history = "No Q&A history available."
                if current_history:
                    lines = []
                    for q_type, msg in current_history:
                        cleaned_msg = re.sub('<[^<]+?>', '', str(msg))
                        lines.append(f"{q_type}: {cleaned_msg}")
                    formatted_history = "\n\n".join(lines)

                report_data = {
                    "Session ID": st.session_state.session_id,
                    "Image Filename": (st.session_state.uploaded_file_info or "N/A").split('-')[0],
                    "Initial Analysis": st.session_state.initial_analysis or "Not Performed",
                    "Conversation History": formatted_history,
                    "Condition Analysis": st.session_state.disease_analysis or "Not Performed",
                    "AI Confidence Estimation": st.session_state.confidence_score or "Not Performed",
                }
                if st.session_state.is_dicom and st.session_state.dicom_metadata:
                    meta_summary = {k: v for k, v in st.session_state.dicom_metadata.items() if k in [
                        'Patient Name', 'Patient ID', 'Study Date', 'Modality', 'Study Description'
                    ]}
                    if meta_summary:
                        lines = [f"{k}: {v}" for k, v in meta_summary.items()]
                        report_data["DICOM Summary"] = "\n".join(lines)

                with st.spinner("Generating PDF..."):
                    pdf_bytes = generate_pdf_report_bytes(
                        session_id=st.session_state.session_id,
                        image=final_image_for_pdf,
                        analysis_outputs=report_data,
                        dicom_metadata=st.session_state.dicom_metadata if st.session_state.is_dicom else None,
                        include_anatomical_diagram=True  # Enable the new smart features
                    )
                if pdf_bytes:
                    st.session_state.pdf_report_bytes = pdf_bytes
                    st.success("PDF data ready! Download in sidebar.")
                    logger.info("PDF report generated.")
                    st.balloons()
                else:
                    st.error("Failed to generate PDF.")
                    logger.error("PDF generator returned no data.")

        elif current_action == "umls_lookup" and UMLS_UTILS_AVAILABLE:
            term = st.session_state.umls_lookup_term.strip()
            if term:
                with st.spinner(f"Looking up '{term}' in UMLS..."):
                    try:
                        api_key = os.environ.get("UMLS_APIKEY")
                        if not api_key:
                            raise ValueError("UMLS_APIKEY environment variable is not set")

                        # Get vocabulary filters if set
                        vocab_filters = st.session_state.get("umls_vocab_filter")
                        max_results = st.session_state.get("umls_max_results", 5)

                        # Search UMLS with the term and filters
                        concepts = search_umls(
                            term=term, 
                            apikey=api_key,
                            page_size=max_results,
                            source_vocabs=vocab_filters
                        )
                        st.session_state.umls_lookup_results = concepts
                        st.session_state.umls_lookup_error = None

                        if not concepts:
                            st.info(f"No UMLS concepts found for '{term}'")
                    except UMLSAuthError as auth_err:
                        error_msg = f"UMLS Authentication Failed: {auth_err}"
                        st.error(error_msg)
                        st.session_state.umls_lookup_results = None
                        st.session_state.umls_lookup_error = error_msg
                    except Exception as e:
                        error_msg = f"UMLS Search Error: {e}"
                        st.error(error_msg)
                        st.session_state.umls_lookup_results = None
                        st.session_state.umls_lookup_error = error_msg

            # Clear the action
            st.session_state.last_action = None
            st.rerun()

        elif current_action == "umls_dashboard" and UMLS_UTILS_AVAILABLE:
            # Launch the advanced UMLS dashboard
            try:
                from umls_dashboard import render_umls_dashboard
                render_umls_dashboard()

                # Keep the dashboard mode active until user navigates away
                st.session_state.active_view = "umls_dashboard"
                logger.info("UMLS dashboard rendered successfully")
            except ImportError as e:
                st.error(f"Could not load UMLS dashboard: {e}")
                logger.error(f"Failed to import umls_dashboard: {e}")

            # Clear the action but don't rerun to avoid refreshing the dashboard
            st.session_state.last_action = None
            # Note: Not calling st.rerun() here to let dashboard render

        else:
            st.warning(f"Unknown action '{current_action}' triggered.")
    except Exception as e:
        st.error(f"Error during '{current_action}': {e}")
        logger.critical(f"Action '{current_action}' error: {e}", exc_info=True)
    finally:
        st.session_state.last_action = None
        logger.debug(f"Action '{current_action}' complete.")
        st.rerun()

# --- Footer ---
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced | Session ID: {st.session_state.get('session_id', 'N/A')}")
st.markdown(
    """
    <footer>
      <p>RadVision AI is for informational purposes only. Not a substitute for professional evaluation.</p>
      <p><a href="#" target="_blank">Privacy Policy</a> | <a href="#" target="_blank">Terms of Service</a></p>
    </footer>
    """,
    unsafe_allow_html=True
)
logger.info(f"--- Application render complete for session: {st.session_state.session_id} ---")