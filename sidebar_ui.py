# sidebar_ui.py
# -*- coding: utf-8 -*-
"""
RadVision AI – Sidebar UI

Handles:
  • file upload & demo-mode toggle
  • ROI clearing
  • DICOM window/level sliders
  • action buttons (analysis, Q&A, disease, confidence, PDF)
"""

import os
import logging
import random
from typing import Optional, Dict, Any

import streamlit as st
from PIL import Image  # Main import for Image class to be accessible globally

# Import configuration
try:
    from config import TIPS, DISEASE_OPTIONS
except ImportError:
    TIPS = ["Use ROI selection to focus analysis on a specific area."]
    DISEASE_OPTIONS = ["Pneumonia", "Pulmonary Nodule", "Pleural Effusion"]
    logging.warning("Could not import from config.py, using defaults.")

# Setup logging
logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from ui_components import dicom_wl_sliders
    DICOM_WL_AVAILABLE = True
except ImportError:
    dicom_wl_sliders = None
    DICOM_WL_AVAILABLE = False
    logger.warning("dicom_wl_sliders not imported – DICOM W/L disabled.")

try:
    from report_utils import generate_pdf_report_bytes
    REPORT_UTILS_AVAILABLE = True
except ImportError:
    generate_pdf_report_bytes = None
    REPORT_UTILS_AVAILABLE = False
    logger.warning("report_utils not found – PDF reporting disabled.")

try:
    from dicom_utils import dicom_to_image
    DICOM_UTILS_AVAILABLE = True
except ImportError:
    dicom_to_image = None
    DICOM_UTILS_AVAILABLE = False
    logger.warning("dicom_utils not imported – DICOM conversion disabled.")

def render_sidebar() -> None:
    """Render the complete sidebar UI for RadVision AI."""

    with st.sidebar:
        # Auth controls
        try:
            from auth import is_authenticated, logout, render_user_management
            is_auth, username = is_authenticated()
            if is_auth:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"Logged in as: {username}")
                with col2:
                    if st.button("Logout", key="logout_btn"):
                        logout()
                        st.rerun()
        except ImportError:
            pass  # Skip auth if not available

        st.header("⚕️ RadVision Controls")
        st.markdown("---")

        # Display logo with robust error handling
        try:
            if os.path.exists("assets/radvisionai-hero.jpeg"):
                st.image("assets/radvisionai-hero.jpeg", use_container_width=True)
            else:
                st.info("⚕️ RadVision AI")
        except Exception as e:
            logger.warning(f"Failed to load logo image: {e}")
            st.info("⚕️ RadVision AI")

        # Only display demo image when demo mode is active
        if st.session_state.get("demo_loaded", False):
            try:
                if os.path.exists("assets/demo.png"):
                    st.sidebar.image("assets/demo.png", caption="Demo Image", use_container_width=True)
            except Exception as e:
                logger.warning(f"Failed to load demo image: {e}")

        # Display random tip
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

        # Demo mode toggle
        demo_mode = st.checkbox(
            "🚀 Demo Mode", 
            value=st.session_state.get("demo_loaded", False),
            help="Load a demo image for testing the application.",
            on_change=lambda: st.rerun()  # Force rerun when checkbox changes
        )

        # Update demo state when checkbox changes
        if demo_mode != st.session_state.get("demo_loaded", False):
            st.session_state.demo_loaded = demo_mode

        # Load demo image when demo mode is enabled
        if demo_mode and not st.session_state.get("display_image"):
            try:
                if os.path.exists("assets/demo.png"):
                    # Image is already imported at the top of the file
                    demo_img = Image.open("assets/demo.png")
                    st.session_state.demo_loaded = True
                    st.session_state.display_image = demo_img
                    st.session_state.processed_image = demo_img.copy()
                    st.toast("Demo image loaded", icon="✅")
                    st.rerun()
                else:
                    st.error("Demo image not found in assets/demo.png")
            except Exception as e:
                st.error(f"Failed to load demo image: {e}")
                logger.error(f"Demo image loading error: {e}", exc_info=True)

        # ROI controls
        if st.button(
            "🗑️ Clear ROI", 
            help="Remove the selected Region of Interest"
        ):
            st.session_state.roi_coords = None
            st.session_state.canvas_drawing = None
            st.session_state.clear_roi_feedback = True
            st.rerun()

        # DICOM window/level controls if available
        if (DICOM_WL_AVAILABLE and 
            st.session_state.get("is_dicom") and 
            st.session_state.get("dicom_dataset")):

            st.markdown("---")
            st.header("🎛️ DICOM Window/Level")

            # Create sliders to adjust Window/Level values
            wc, ww = dicom_wl_sliders(
                ds=st.session_state.get("dicom_dataset"),
                current_wc=st.session_state.get("current_display_wc"),
                current_ww=st.session_state.get("current_display_ww")
            )

            # Update image if W/L changed
            if (wc != st.session_state.get("current_display_wc") or 
                ww != st.session_state.get("current_display_ww")):

                if DICOM_UTILS_AVAILABLE and dicom_to_image:
                    try:
                        ds = st.session_state.get("dicom_dataset")
                        if ds:
                            with st.spinner("Updating display..."):
                                display_img = dicom_to_image(ds, wc=wc, ww=ww)
                                if isinstance(display_img, Image.Image):
                                    st.session_state.display_image = display_img
                                    st.session_state.current_display_wc = wc
                                    st.session_state.current_display_ww = ww
                                    st.rerun()
                    except Exception as e:
                        logger.error(f"Error updating image with new W/L: {e}", exc_info=True)
                        st.error(f"Could not apply new window settings: {e}")

        # ---------------------------------------------------------------
        # AI Analysis Actions
        # ---------------------------------------------------------------
        st.markdown("---")
        st.header("🤖 AI Analysis Actions")

        # Check if image is loaded
        img_ready = isinstance(st.session_state.get("processed_image"), Image.Image)

        # Initial analysis button
        if st.button(
            "▶️ Run Initial Analysis", 
            key="analyze_btn",
            disabled=not img_ready,
            help="Perform a general analysis of the image (or ROI)."
        ):
            st.session_state.last_action = "analyze"
            st.rerun()

        # Q&A section
        st.subheader("❓ Ask AI a Question")
        q_txt = st.text_area(
            "Enter your question:",
            key="question_input_widget",
            height=100,
            placeholder="E.g., 'Are there any nodules in the upper right lobe?'",
            disabled=not img_ready
        )

        if st.button("💬 Ask Question", key="ask_btn", disabled=not img_ready):
            if q_txt.strip():
                st.session_state.last_action = "ask"
                st.rerun()
            else:
                st.warning("Please enter a question before submitting.")

        # Condition analysis
        st.subheader("🎯 Condition-Specific Analysis")
        cond = st.selectbox(
            "Select condition to focus on:",
            options=[""] + sorted(DISEASE_OPTIONS),
            key="disease_select_widget",
            disabled=not img_ready
        )

        if st.button("🩺 Analyze Condition", key="disease_btn", disabled=not img_ready):
            if cond:
                st.session_state.last_action = "disease"
                st.rerun()
            else:
                st.warning("Please select a condition first.")

        # ---------------------------------------------------------------
        # Confidence & Reporting
        # ---------------------------------------------------------------
        st.markdown("---")
        st.header("📊 Confidence & Reporting")

        # Check if there's analysis data to estimate confidence on
        can_conf = bool(
            st.session_state.get("history") or
            st.session_state.get("initial_analysis") or
            st.session_state.get("disease_analysis")
        )

        # Confidence estimation button
        if st.button(
            "📈 Estimate AI Confidence",
            key="confidence_btn",
            disabled=(not img_ready) or (not can_conf),
            help="Assess how confident the AI is in its analysis."
        ):
            st.session_state.last_action = "confidence"
            st.rerun()

        # FDA Compliance dashboard button
        try:
            # Only render if the user has admin privileges
            from auth import is_authenticated
            is_auth, username = is_authenticated()

            if is_auth:
                # Check if user is admin in auth_config.json
                try:
                    import json
                    import os

                    if os.path.exists("auth_config.json"):
                        with open("auth_config.json", "r") as f:
                            auth_config = json.load(f)

                        is_admin = auth_config.get("users", {}).get(username, {}).get("is_admin", False)

                        if is_admin:
                            st.markdown("---")
                            st.header("🏥 FDA Compliance")

                            if st.button("View FDA Compliance Dashboard", use_container_width=True):
                                st.session_state.active_view = "fda_dashboard"
                                st.rerun()

                except Exception as e:
                    logger.error(f"Error checking admin status: {e}")
        except ImportError:
            pass  # Skip if auth module not available

        # PDF report generation
        pdf_disabled = (not img_ready) or (not REPORT_UTILS_AVAILABLE)
        if st.button(
            "📄 Generate PDF Report",
            key="generate_report_data_btn",
            disabled=pdf_disabled,
            help="Create a downloadable PDF report with the analysis results."
        ):
            st.session_state.last_action = "generate_report_data"
            st.rerun()

        # Download PDF if available
        if st.session_state.get("pdf_report_bytes"):
            try:
                session_id = st.session_state.get("session_id", "report")
                file_name = f"RadVision_Report_{session_id}.pdf"

                # Ensure we have valid bytes for Streamlit download button
                pdf_bytes = st.session_state.pdf_report_bytes
                if isinstance(pdf_bytes, bytearray):
                    pdf_bytes = bytes(pdf_bytes)

                # Convert to bytes if it's not already
                if isinstance(pdf_bytes, bytearray):
                    pdf_bytes = bytes(pdf_bytes)
                
                # Handle any other type conversion if needed
                if not isinstance(pdf_bytes, bytes):
                    try:
                        pdf_bytes = bytes(pdf_bytes)
                    except Exception as e:
                        logger.error(f"Failed to convert PDF bytes: {e}")
                        st.error("Invalid PDF data format. Please regenerate the report.")
                        st.session_state.pdf_report_bytes = None
                        return
                
                # Use download button with bytes data
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=file_name,
                    mime="application/pdf",
                    help="Download the generated PDF report."
                )
            except Exception as e:
                st.error(f"Error displaying PDF download button: {e}")
                logger.error(f"PDF download button error: {e}", exc_info=True)

            # Save to Google Sheets option
            try:
                # Import required packages
                import os
                # Try to import from sheets_integration
                try:
                    from sheets_integration import log_analysis_to_sheet, SHEETS_AVAILABLE, get_sheet_client
                    # Check if sheets is properly configured with credentials
                    client = get_sheet_client()
                    sheets_disabled = not bool(client)
                    
                    # Set appropriate message based on configuration status
                    if not client:
                        sheets_status_message = "Google Sheets integration not fully configured"
                    else:
                        sheets_status_message = ""
                except ImportError:
                    SHEETS_AVAILABLE = False
                    sheets_disabled = True
                    sheets_status_message = "Google Sheets integration module not available"
                    logger.warning("Google Sheets integration module not available")
                # Show button even if disabled so user can click and see explanation
                if st.button(
                    "📊 Save Analysis to Sheets",
                    key="save_to_sheets_btn",
                    disabled=sheets_disabled,
                    help="Save this analysis to Google Sheets for tracking"
                ):
                    if sheets_disabled:
                        st.info("Google Sheets integration not fully configured. Please check configuration.")
                    else:
                        with st.spinner("Saving to Google Sheets..."):
                            # Create a summary of the analysis
                            initial_analysis = st.session_state.get("initial_analysis", "")
                            summary = initial_analysis[:500] + "..." if len(initial_analysis) > 500 else initial_analysis

                            # Get relevant data for logging
                            analysis_data = {
                                "session_id": st.session_state.get("session_id", "unknown"),
                                "image_name": (st.session_state.get("uploaded_file_info") or "demo_image").split('-')[0],
                                "analysis_type": "Initial Analysis",
                                "key_findings": summary,
                                "confidence": st.session_state.get("confidence_score", "Not estimated"),
                                "umls_concepts": st.session_state.get("initial_analysis_umls", [])
                            }

                            # Log to Google Sheets
                            try:
                                success = log_analysis_to_sheet(analysis_data)
                                
                                if success:
                                    st.success("Analysis saved to Google Sheets!")
                                else:
                                    st.error("Failed to save analysis. Check Google Sheets configuration.")
                            except Exception as e:
                                st.error(f"Error saving to sheets: {str(e)}")
                                logger.error(f"Sheets integration error: {e}")
                
                # Show explanatory message about sheets status
                if sheets_disabled:
                    st.info("Google Sheets integration not configured. See Case History tab.")
            except ImportError:
                pass  # Silently ignore if sheets_integration is not available

        # UMLS Advanced Integration
        try:
            from umls_utils import UMLS_UTILS_LOADED
            if UMLS_UTILS_LOADED and os.environ.get("UMLS_APIKEY"):
                st.markdown("---")
                st.header("🧬 UMLS Integration")

                # Quick term lookup
                umls_term = st.text_input(
                    "Quick medical term lookup:",
                    key="umls_lookup_term",
                    placeholder="e.g., pneumonia"
                )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔍 Search", use_container_width=True):
                        if umls_term.strip():
                            st.session_state.last_action = "umls_lookup"
                            st.rerun()
                        else:
                            st.warning("Please enter a term")

                with col2:
                    if st.button("🔬 Advanced", use_container_width=True, key="umls_advanced_btn"):
                        st.session_state.last_action = "umls_dashboard"
                        st.session_state.active_view = "umls_dashboard"
                        st.rerun()

                # UMLS vocabulary filters
                with st.expander("UMLS Options", expanded=False):
                    st.multiselect(
                        "Preferred Vocabularies:", 
                        ["SNOMEDCT_US", "ICD10CM", "RXNORM", "LOINC"],
                        default=["SNOMEDCT_US", "ICD10CM"],
                        key="umls_vocab_filter"
                    )

                    st.number_input(
                        "Max Concepts:", 
                        min_value=1, 
                        max_value=50, 
                        value=5,
                        key="umls_max_results"
                    )
        except ImportError:
            pass  # Skip UMLS section if not available