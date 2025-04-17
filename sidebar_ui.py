# sidebar_ui.py
import streamlit as st
import random
import logging
from config import TIPS, DISEASE_OPTIONS
from PIL import Image # Check image type
# Import UI components directly needed
from ui_components import dicom_wl_sliders, UI_COMPONENTS_AVAILABLE

logger = logging.getLogger(__name__)

# --- Check for Report Utils Availability HERE ---
try:
    from report_utils import generate_pdf_report_bytes # Import the function needed
    REPORT_UTILS_AVAILABLE = True
    logger.info("Report utilities (generate_pdf_report_bytes) found.")
except ImportError:
    REPORT_UTILS_AVAILABLE = False
    logger.warning("report_utils not found or generate_pdf_report_bytes missing. PDF reporting disabled in sidebar.")
# --- End Check ---


def render_sidebar():
    """Renders the sidebar UI elements and returns the uploaded file object."""
    uploaded_file = None

    with st.sidebar:
        st.header("⚕️ RadVision Controls")
        st.markdown("---")
        st.info(f"💡 {random.choice(TIPS)}")
        st.markdown("---")

        st.header("Image Upload & Settings")
        uploaded_file = st.file_uploader(
            "Upload Image (JPG, PNG, DCM)",
            type=["jpg","jpeg","png","dcm","dicom"],
            key="file_uploader_widget",
            help="Upload medical image. DICOM preferred."
        )

        demo_mode = st.checkbox(
            "🚀 Demo Mode",
            value=st.session_state.get("demo_loaded", False),
            help="Load sample X-ray."
        )

        if st.button("🗑️ Clear ROI", help="Remove selected ROI", key="clear_roi_btn"):
            st.session_state.roi_coords = None
            st.session_state.canvas_drawing = None
            st.session_state.clear_roi_feedback = True
            logger.info("ROI cleared.")
            st.rerun()

        if st.session_state.get("clear_roi_feedback"):
            st.success("✅ ROI cleared!")
            st.balloons()
            st.session_state.clear_roi_feedback = False

        # --- DICOM W/L ---
        if st.session_state.get("is_dicom") and isinstance(st.session_state.get("display_image"), Image.Image) and UI_COMPONENTS_AVAILABLE:
            st.markdown("---")
            st.subheader("DICOM Display")
            try:
                from dicom_utils import DICOM_UTILS_AVAILABLE, dicom_to_image # Import here too
            except ImportError:
                DICOM_UTILS_AVAILABLE = False
                dicom_to_image = None
                logger.warning("dicom_utils unavailable within sidebar W/L logic.")

            new_wc, new_ww = dicom_wl_sliders(
                st.session_state.get("dicom_dataset"),
                st.session_state.get("current_display_wc"),
                st.session_state.get("current_display_ww")
            )

            if new_wc != st.session_state.get("current_display_wc") or new_ww != st.session_state.get("current_display_ww"):
                logger.info(f"W/L changed: WC={new_wc}, WW={new_ww}")
                st.session_state.current_display_wc = new_wc
                st.session_state.current_display_ww = new_ww

                if DICOM_UTILS_AVAILABLE and dicom_to_image and st.session_state.get("dicom_dataset") and PIL_AVAILABLE:
                    with st.spinner("Applying W/L..."):
                        logger.debug("Regenerating display image...")
                        new_display_img = dicom_to_image(st.session_state.dicom_dataset, wc=new_wc, ww=new_ww)
                        if isinstance(new_display_img, Image.Image):
                            st.session_state.display_image = new_display_img.convert('RGB') if new_display_img.mode != 'RGB' else new_display_img
                            logger.info("Display image updated."); st.rerun()
                        else: st.error("Failed to update DICOM image."); logger.error(f"dicom_to_image invalid type ({type(new_display_img)}).")
                elif not DICOM_UTILS_AVAILABLE: st.warning("DICOM utils unavailable."); logger.warning("W/L changed; DICOM utils missing.")
                elif not st.session_state.dicom_dataset: st.warning("DICOM dataset missing."); logger.warning("W/L changed; DICOM dataset missing.")

        # --- AI Actions ---
        st.markdown("---"); st.header("🤖 AI Analysis Actions")
        action_disabled = not isinstance(st.session_state.get("processed_image"), Image.Image)

        if st.button("▶️ Run Initial Analysis", key="analyze_btn", disabled=action_disabled, help="General analysis."):
            st.session_state.last_action = "analyze"; st.rerun()

        st.subheader("❓ Ask AI a Question")
        question_input = st.text_area("Enter question:", height=100, key="question_input_widget", placeholder="E.g., 'Any nodules?'", disabled=action_disabled)
        if st.button("💬 Ask Question", key="ask_btn", disabled=action_disabled):
            if question_input.strip(): st.session_state.last_action = "ask"; st.rerun()
            else: st.warning("Please enter a question.")

        st.subheader("🎯 Condition-Specific Analysis")
        disease_select = st.selectbox("Select condition:", options=[""] + sorted(DISEASE_OPTIONS), key="disease_select_widget", disabled=action_disabled)
        if st.button("🩺 Analyze Condition", key="disease_btn", disabled=action_disabled):
            if disease_select: st.session_state.last_action = "disease"; st.rerun()
            else: st.warning("Please select a condition.")

        st.markdown("---"); st.header("📊 Confidence & Reporting")
        can_estimate = bool(st.session_state.get("history") or st.session_state.get("initial_analysis") or st.session_state.get("disease_analysis")) and not action_disabled
        if st.button("📈 Estimate AI Confidence", key="confidence_btn", disabled=not can_estimate):
            st.session_state.last_action = "confidence"; st.rerun()

        # Use the locally defined REPORT_UTILS_AVAILABLE flag
        report_disabled = action_disabled or not REPORT_UTILS_AVAILABLE
        if st.button("📄 Generate PDF Report Data", key="generate_report_data_btn", disabled=report_disabled):
            st.session_state.last_action = "generate_report_data"; st.rerun()

        if st.session_state.get("pdf_report_bytes"):
            report_filename = f"RadVisionAI_Report_{st.session_state.get('session_id', 'session')}.pdf"
            st.download_button("⬇️ Download PDF Report", st.session_state.pdf_report_bytes, report_filename, "application/pdf", key="download_pdf_button", help="Download generated PDF.")

    return uploaded_file