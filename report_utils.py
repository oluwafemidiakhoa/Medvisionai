# -*- coding: utf-8 -*-
"""
report_utils.py - Generates PDF reports for RadVision AI Advanced.

Focuses on clear presentation, including prominent disclaimers and limitations
of the AI analysis, aligning with responsible AI demonstration principles.
"""

import os
import logging
import tempfile
from fpdf import FPDF # RECOMMENDATION: Consider using fpdf2 for better UTF-8 support (pip install fpdf2)
from PIL import Image
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# --- Constants for Styling and Content ---
REPORT_TITLE = "RadVision AI - Analysis Report (Demonstration)"
PROMINENT_DISCLAIMER_TITLE = "IMPORTANT DISCLAIMER & USAGE NOTICE"
PROMINENT_DISCLAIMER_TEXT = (
    "This report contains AI-generated analysis intended SOLELY for research, educational, and demonstration purposes.\n"
    "It is NOT a medical diagnosis, radiological interpretation, or substitute for professional medical advice.\n"
    "AI analysis may be inaccurate, incomplete, and lacks full clinical context.\n"
    "ALWAYS consult qualified healthcare professionals for any medical decisions.\n"
    "Do NOT use this report for patient care without independent expert validation.\n"
    "Ensure compliance with privacy regulations (e.g., HIPAA) regarding any data shown."
)
LIMITATIONS_TITLE = "Limitations of This AI Analysis"
LIMITATIONS_TEXT = (
    "- Analysis is based SOLELY on the provided image(s) and text inputs.\n"
    "- Performance depends heavily on image quality, view, and potential artifacts.\n"
    "- The AI lacks critical clinical context (patient history, symptoms, labs, etc.).\n"
    "- Absence of prior imaging studies prevents comparison over time.\n"
    "- AI functions primarily on visual pattern recognition, not clinical reasoning.\n"
    "- Outputs may contain errors or omissions."
)
CONFIDENCE_WARNING_TEXT = (
    "Note: This reflects the AI's internal self-assessment (EXPERIMENTAL) and DOES NOT represent "
    "clinical certainty or diagnostic accuracy. Treat with extreme caution."
)
FOOTER_TEXT = (
    "RadVision AI Demo Report | AI analysis requires expert clinical correlation and validation."
)

# Colors (RGB tuples)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)       # For critical errors/warnings
COLOR_GREY = (128, 128, 128)  # For footer/minor text
COLOR_DARK_BLUE = (0, 0, 128) # For disclaimer title

# Temporary file prefix
TEMP_IMAGE_PREFIX = "radvision_report_img_"

# --- PDF Generation Function ---

def generate_pdf_report_bytes(
    session_id: str,
    image: Optional[Image.Image],
    analysis_outputs: Dict[str, str],
    # Add dicom_metadata if needed for the report, similar to app.py
    dicom_metadata: Optional[Dict[str, Any]] = None
) -> Optional[bytes]:
    """
    Generates a PDF report summarizing the analysis session responsibly.

    Includes session ID, prominent disclaimers, limitations, an embedded
    image (if provided), and formatted AI analysis outputs with appropriate caveats.

    Args:
        session_id: The unique identifier for the analysis session.
        image: A PIL Image object to embed. Placeholder if None.
        analysis_outputs: Dictionary with section titles (keys) and AI-generated
                          content (values). Special handling for confidence scores.
        dicom_metadata: Optional dictionary of key DICOM tags to include.

    Returns:
        The generated PDF as bytes if successful, otherwise None.
    """
    logger.info(f"Starting PDF report generation for session ID: {session_id}")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(15, 15, 15) # Left, Top, Right margins in mm
    pdf.set_auto_page_break(auto=True, margin=15) # Bottom margin for auto page break

    # --- Report Header ---
    pdf.set_font("Arial", 'B', 16) # Use standard fonts available in FPDF
    pdf.set_text_color(*COLOR_BLACK)
    pdf.cell(0, 10, txt=REPORT_TITLE, ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, txt=f"Session ID: {session_id}", ln=True, align='C')
    pdf.ln(5)

    # --- *** Prominent Disclaimer Section *** ---
    pdf.set_font("Arial", 'B', 11)
    pdf.set_text_color(*COLOR_DARK_BLUE)
    pdf.cell(0, 6, txt=PROMINENT_DISCLAIMER_TITLE, ln=True, align='L')
    pdf.set_font("Arial", size=9)
    pdf.set_text_color(*COLOR_BLACK)
    pdf.multi_cell(0, 4.5, txt=PROMINENT_DISCLAIMER_TEXT)
    pdf.ln(6)

    # --- Embed Image ---
    temp_image_path = None
    try:
        if image is None:
            logger.warning("No image provided for PDF report.")
            pdf.set_font("Arial", 'I', 10)
            pdf.set_text_color(*COLOR_GREY)
            pdf.cell(0, 10, "[No image associated with this session]", ln=True, align='C')
            pdf.set_text_color(*COLOR_BLACK) # Reset color
            pdf.ln(5)
        else:
            # Save image temporarily to embed
            # RECOMMENDATION: Use PNG for lossless quality if possible
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix=TEMP_IMAGE_PREFIX) as tmpfile:
                temp_image_path = tmpfile.name
                logger.debug(f"Saving temporary image for PDF to: {temp_image_path}")
                # Ensure image is RGB for broader compatibility, though PNG supports others
                img_to_save = image.convert("RGB") if image.mode != "RGB" else image
                img_to_save.save(temp_image_path, format="PNG")

            # Calculate display size to fit page width (e.g., 80% of usable width)
            page_width_mm = pdf.w - pdf.l_margin - pdf.r_margin
            max_img_width_mm = page_width_mm * 0.8

            img_width_px, img_height_px = image.size
            if img_width_px <= 0 or img_height_px <= 0:
                raise ValueError(f"Invalid source image dimensions: {image.size}")

            aspect_ratio = img_height_px / img_width_px
            # Convert pixels to mm heuristic (adjust 3.78 based on DPI if known, roughly 96 DPI)
            # Or just use the width constraint
            display_width_mm = max_img_width_mm
            display_height_mm = display_width_mm * aspect_ratio

            # Check if calculated height exceeds available page space
            available_height_mm = pdf.h - pdf.get_y() - pdf.b_margin - 10 # Leave some buffer
            if display_height_mm > available_height_mm and available_height_mm > 20: # Avoid tiny images
                logger.debug(f"Image height ({display_height_mm:.1f}mm) exceeds available space ({available_height_mm:.1f}mm). Resizing.")
                display_height_mm = available_height_mm
                display_width_mm = display_height_mm / aspect_ratio

            # Center image horizontally
            x_pos = (pdf.w - display_width_mm) / 2
            logger.debug(f"Embedding image: {os.path.basename(temp_image_path)} at ({x_pos:.1f}, {pdf.get_y():.1f}) size ({display_width_mm:.1f}x{display_height_mm:.1f} mm)")
            pdf.image(temp_image_path, x=x_pos, y=pdf.get_y(), w=display_width_mm, h=display_height_mm)
            pdf.ln(display_height_mm + 5) # Move below image + spacing

    except Exception as e:
        err_msg = f"Error processing or embedding image in PDF: {e}"
        logger.error(err_msg, exc_info=True)
        pdf.set_text_color(*COLOR_RED)
        pdf.set_font("Arial", 'B', 10)
        pdf.multi_cell(0, 6, "[Error displaying image in report - See application logs]", align='C')
        pdf.set_text_color(*COLOR_BLACK) # Reset color
        pdf.set_font("Arial", size=10)
        pdf.ln(5)
    finally:
        # Clean up temporary image file
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                logger.debug(f"Removed temporary image file: {temp_image_path}")
            except OSError as e_os:
                logger.warning(f"Could not remove temp image file '{temp_image_path}': {e_os}")

    # --- *** Limitations Section *** ---
    pdf.set_font("Arial", 'B', 11)
    pdf.set_text_color(*COLOR_DARK_BLUE)
    pdf.cell(0, 6, txt=LIMITATIONS_TITLE, ln=True, align='L')
    pdf.set_font("Arial", size=9)
    pdf.set_text_color(*COLOR_BLACK)
    pdf.multi_cell(0, 4.5, txt=LIMITATIONS_TEXT)
    pdf.ln(6)

    # --- Optional: DICOM Metadata Summary ---
    if dicom_metadata:
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 6, txt="DICOM Metadata Summary", ln=True, align='L')
        pdf.set_font("Arial", size=9)
        meta_lines = []
        # Select and format key tags - use .get() for safety
        key_tags = ['PatientName', 'PatientID', 'StudyDate', 'Modality', 'StudyDescription', 'InstitutionName']
        for tag in key_tags:
            value = dicom_metadata.get(tag, 'N/A')
            # Improve tag display name slightly
            display_tag = tag.replace('PatientName', 'Patient Name').replace('PatientID', 'Patient ID').replace('StudyDate', 'Study Date').replace('StudyDescription', 'Study Desc.')
            meta_lines.append(f"- {display_tag}: {value}")

        if meta_lines:
             pdf.multi_cell(0, 4.5, txt="\n".join(meta_lines))
        else:
             pdf.multi_cell(0, 4.5, txt="No relevant DICOM summary tags found.")
        pdf.ln(6)


    # --- AI Analysis Sections ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="AI-Generated Analysis Sections", ln=True) # Clear attribution

    if not analysis_outputs:
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 6, "No AI analysis results were recorded for this session.", ln=True)
        pdf.ln(4)
    else:
        for section_title, content in analysis_outputs.items():
            pdf.set_font("Arial", 'B', 11)
            pdf.set_text_color(*COLOR_BLACK) # Ensure default color
            pdf.multi_cell(0, 6, f"{section_title}:") # Section title

            # --- Special Handling for Experimental Confidence Score ---
            if "confidence" in section_title.lower() or "self-assessment" in section_title.lower():
                pdf.set_font("Arial", 'I', 9) # Italic, smaller font for warning
                pdf.set_text_color(*COLOR_RED) # Red color for warning
                pdf.multi_cell(0, 4.5, txt=CONFIDENCE_WARNING_TEXT)
                pdf.set_font("Arial", size=10) # Reset font for content
                pdf.set_text_color(*COLOR_BLACK) # Reset color

            # --- Write Section Content ---
            pdf.set_font("Arial", size=10)
            content_to_write = str(content).strip() if content else "[Not Available]"

            # RECOMMENDATION: Use fpdf2 and UTF-8 fonts for better character support.
            # The following uses latin-1 for compatibility with the original code,
            # but will replace unsupported characters.
            try:
                # Encode/decode using latin-1 with replacement
                encoded_content = content_to_write.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 5, txt=encoded_content) # Write content
            except Exception as write_e:
                # Fallback if writing fails
                logger.error(f"Error writing PDF content for section '{section_title}': {write_e}", exc_info=True)
                pdf.set_text_color(*COLOR_RED)
                pdf.set_font("Arial", 'I', 9)
                pdf.multi_cell(0, 5, "[Error rendering content for this section - Please check application logs]")
                pdf.set_text_color(*COLOR_BLACK) # Reset color

            pdf.ln(5)  # Add vertical space between sections

    # --- Footer ---
    pdf.set_y(-20)  # Position 20 mm from the bottom
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(*COLOR_GREY)
    pdf.multi_cell(0, 4, txt=FOOTER_TEXT, align='C')
    pdf.set_text_color(*COLOR_BLACK) # Reset to default

    # --- Generate PDF Bytes ---
    try:
        # Output PDF as bytes ( 'S' destination )
        pdf_bytes = pdf.output(dest='S')

        # FPDF might return a string in some environments, ensure bytes
        if isinstance(pdf_bytes, str):
            pdf_bytes = pdf_bytes.encode('latin-1') # Use latin-1 consistent with content encoding

        logger.info(f"PDF report ({len(pdf_bytes)} bytes) generated successfully for session {session_id}.")
        return pdf_bytes
    except Exception as e_pdf:
        logger.error(f"Critical error during final PDF byte generation: {e_pdf}", exc_info=True)
        return None # Return None on failure