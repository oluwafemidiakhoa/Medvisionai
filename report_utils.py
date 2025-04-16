import os
import logging
import tempfile
from fpdf import FPDF  # Recommend using fpdf2: pip install fpdf2
from PIL import Image
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Define constants for styling or fixed text
REPORT_TITLE = "MediVision QA - AI Analysis Report"
FOOTER_TEXT = (
    "Disclaimer: AI-generated analysis. For informational purposes only. "
    "Requires expert clinical validation."
)
ERROR_COLOR = (255, 0, 0)  # Red for errors in PDF
DEFAULT_COLOR = (0, 0, 0)  # Black
TEMP_IMAGE_PREFIX = "medivision_report_img_"

def generate_pdf_report_bytes(
    session_id: str,
    image: Optional[Image.Image],
    analysis_outputs: Dict[str, str]
) -> Optional[bytes]:
    """
    Generates a PDF report summarizing the analysis session.

    Includes session ID, an embedded image (if provided), and formatted
    analysis outputs. Handles errors during generation gracefully.

    Args:
        session_id: The unique identifier for the analysis session.
        image: A PIL Image object to embed in the report. If None, a
               placeholder message is added.
        analysis_outputs: A dictionary with section titles as keys and
                          content strings as values.

    Returns:
        The generated PDF as bytes if successful, otherwise None.
    """
    logger.info(f"Starting PDF report generation for session ID: {session_id}")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Header ---
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=REPORT_TITLE, ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, txt=f"Session ID: {session_id}", ln=True, align='C')
    pdf.ln(10)  # Space after header

    # --- Embed Image ---
    temp_image_path = None
    try:
        if image is None:
            logger.warning("No image provided for PDF report.")
            pdf.set_font("Arial", 'I', 10)
            pdf.set_text_color(*ERROR_COLOR)
            pdf.cell(0, 10, "[No image available for this report]", ln=True, align='C')
            pdf.set_text_color(*DEFAULT_COLOR)
            pdf.ln(5)
        else:
            # Save the image temporarily to a file for embedding
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix=TEMP_IMAGE_PREFIX) as tmpfile:
                temp_image_path = tmpfile.name
                logger.debug(f"Saving temporary image for PDF report to: {temp_image_path}")
                image.save(temp_image_path, format="PNG")
            
            # Calculate display size: fit within 90% of page width
            page_width_mm = pdf.w - pdf.l_margin - pdf.r_margin
            max_img_width_mm = page_width_mm * 0.9

            img_width_px, img_height_px = image.size
            if img_width_px <= 0 or img_height_px <= 0:
                raise ValueError(f"Invalid image dimensions: {image.size}")

            aspect_ratio = img_height_px / img_width_px
            # Use a basic heuristic: convert pixel dimensions to mm (adjustable)
            display_width_mm = min(max_img_width_mm, img_width_px / 3)
            display_height_mm = display_width_mm * aspect_ratio

            # Ensure the image fits vertically on the page
            available_height_mm = pdf.h - pdf.get_y() - pdf.b_margin - 10
            if display_height_mm > available_height_mm:
                logger.debug(f"Image height ({display_height_mm:.1f}mm) exceeds available space ({available_height_mm:.1f}mm). Resizing.")
                display_height_mm = available_height_mm
                display_width_mm = display_height_mm / aspect_ratio

            # Center image horizontally
            x_pos = (pdf.w - display_width_mm) / 2
            logger.debug(f"Embedding image: {os.path.basename(temp_image_path)} at position ({x_pos:.1f}, {pdf.get_y():.1f}) with size ({display_width_mm:.1f}x{display_height_mm:.1f} mm)")
            pdf.image(temp_image_path, x=x_pos, y=pdf.get_y(), w=display_width_mm, h=display_height_mm)
            pdf.ln(display_height_mm + 5)

    except Exception as e:
        err_msg = f"Error processing or embedding image in PDF: {e}"
        logger.error(err_msg, exc_info=True)
        pdf.set_text_color(*ERROR_COLOR)
        pdf.set_font("Arial", 'B', 10)
        pdf.multi_cell(0, 6, "[Error displaying image in report - See application logs for details]", align='C')
        pdf.set_text_color(*DEFAULT_COLOR)
        pdf.set_font("Arial", size=10)
        pdf.ln(5)
    finally:
        # Ensure temporary image file is removed
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                logger.debug(f"Removed temporary image file: {temp_image_path}")
            except OSError as e:
                logger.warning(f"Could not remove temporary image file '{temp_image_path}': {e}")

    # --- Analysis Sections ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Analysis Results", ln=True)

    if not analysis_outputs:
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 6, "No analysis results were generated for this session.", ln=True)
        pdf.ln(4)
    else:
        for section_title, content in analysis_outputs.items():
            # Section Title
            pdf.set_font("Arial", 'B', 11)
            pdf.multi_cell(0, 6, f"{section_title}:")
            pdf.set_font("Arial", size=10)

            content_to_write = content.strip() if content else "N/A"
            try:
                # Encode content using Latin-1 with replacement for unsupported characters
                encoded_content = content_to_write.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 5, txt=encoded_content)
            except Exception as e:
                logger.error(f"Error writing section '{section_title}' to PDF: {e}", exc_info=True)
                pdf.set_text_color(*ERROR_COLOR)
                pdf.multi_cell(0, 5, "[Error rendering content for this section - See application logs]")
                pdf.set_text_color(*DEFAULT_COLOR)

            pdf.ln(4)  # Space between sections

    # --- Footer Disclaimer ---
    pdf.set_y(-20)  # Position footer near the bottom
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(128, 128, 128)  # Gray footer text
    pdf.multi_cell(0, 4, txt=FOOTER_TEXT, align='C')
    pdf.set_text_color(*DEFAULT_COLOR)  # Reset text color

    # --- Output PDF as Bytes ---
    try:
        pdf_bytes = pdf.output(dest='S')
        if isinstance(pdf_bytes, str):
            pdf_bytes = pdf_bytes.encode('latin-1')
        logger.info(f"PDF report ({len(pdf_bytes)} bytes) generated successfully for session {session_id}.")
        return pdf_bytes
    except Exception as e:
        logger.error(f"Critical error during final PDF byte generation: {e}", exc_info=True)
        return None
