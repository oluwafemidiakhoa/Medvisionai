import os
import logging
from fpdf import FPDF
from PIL import Image
from typing import Optional, Dict, Any # Added Any for robustness if needed elsewhere
import tempfile
import pydicom # Import pydicom here if used for metadata formatting

logger = logging.getLogger(__name__)

IMAGE_FILENAME_PREFIX = "medivision_image_"

def generate_pdf_report_bytes(
    session_id: str,
    image: Image.Image,
    analysis_outputs: Dict[str, str]
    ) -> Optional[bytes]:
    """Generates a PDF report summarizing the analysis session and returns its bytes."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt="MediVision QA - AI Analysis Report", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, txt=f"Session ID: {session_id}", ln=True, align='C')
    pdf.ln(10)

    # --- Embed Image using a temporary file ---
    temp_image_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix=IMAGE_FILENAME_PREFIX) as tmpfile:
            temp_image_path = tmpfile.name
            logger.info(f"Saving temporary image for PDF report to: {temp_image_path}")
            if image is None:
                raise ValueError("Cannot save None image to PDF.")
            image.save(temp_image_path, format="PNG")

        page_width_mm = pdf.w - 2 * pdf.l_margin
        max_img_width_mm = page_width_mm * 0.8
        img_width_px, img_height_px = image.size
        if img_width_px <= 0 or img_height_px <= 0: raise ValueError("Invalid image dimensions")
        aspect_ratio = img_height_px / img_width_px
        display_width_mm = max_img_width_mm
        display_height_mm = display_width_mm * aspect_ratio
        max_img_height_mm = (pdf.h - pdf.get_y() - pdf.b_margin - 20)
        if display_height_mm > max_img_height_mm:
            display_height_mm = max_img_height_mm
            display_width_mm = display_height_mm / aspect_ratio
        x_pos = (pdf.w - display_width_mm) / 2
        pdf.image(temp_image_path, x=x_pos, y=pdf.get_y(), w=display_width_mm)
        pdf.ln(display_height_mm + 5)

    except Exception as e:
        err_msg = f"Error embedding image in PDF: {e}"
        logger.error(err_msg, exc_info=True)
        pdf.set_text_color(255, 0, 0)
        pdf.multi_cell(0, 5, f"[Image Embedding Error: Check Logs]", align='C')
        pdf.set_text_color(0, 0, 0)
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                logger.info(f"Removed temporary image file: {temp_image_path}")
            except OSError as e:
                logger.warning(f"Could not remove temporary image file {temp_image_path}: {e}")

    # --- Add Analysis Sections ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Analysis Results", ln=True)
    pdf.set_font("Arial", size=10)

    for section, content in analysis_outputs.items():
        pdf.set_font("Arial", 'B', 11)
        pdf.multi_cell(0, 6, f"{section}:") # Section Title
        pdf.set_font("Arial", size=10)
        content_to_write = content if content else "N/A"
        try:
            encoded_content = content_to_write.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 5, encoded_content)
        except Exception as e: # <--- THIS IS LINE ~40
            # VV V V V V V V V V V V V V V V V V V V V V V V V V V V V
            # MAKE SURE THESE LINES ARE INDENTED LIKE THIS (4 spaces)
            logger.error(f"Error writing section '{section}' to PDF: {e}", exc_info=True)
            pdf.set_text_color(255,0,0)
            pdf.multi_cell(0, 5, f"[Error rendering content for '{section}']") # More specific msg
            pdf.set_text_color(0,0,0)
            # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
        # This line's indentation is relative to the 'for' loop
        pdf.ln(4) # <--- THIS IS LINE ~42

    # --- Footer Disclaimer ---
    pdf.set_y(-15)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Disclaimer: AI-generated analysis. For informational purposes only. Requires expert clinical validation.", 0, 0, 'C')

    # --- Output PDF as Bytes ---
    try:
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        logger.info("PDF report generated successfully as bytes.")
        return pdf_bytes
    except Exception as e:
        logger.error(f"Error generating PDF bytes: {e}", exc_info=True)
        return None