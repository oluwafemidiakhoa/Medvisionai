# report_utils.py (Example filename)

import os
import logging
import tempfile
from fpdf import FPDF # Recommend using fpdf2: pip install fpdf2
from PIL import Image
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Define constants for styling or fixed text
REPORT_TITLE = "MediVision QA - AI Analysis Report"
FOOTER_TEXT = "Disclaimer: AI-generated analysis. For informational purposes only. Requires expert clinical validation."
ERROR_COLOR = (255, 0, 0) # Red for errors in PDF
DEFAULT_COLOR = (0, 0, 0) # Black
TEMP_IMAGE_PREFIX = "medivision_report_img_"

def generate_pdf_report_bytes(
    session_id: str,
    image: Optional[Image.Image], # Allow None explicitly
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
        analysis_outputs: A dictionary where keys are section titles (str)
                          and values are the content (str) for the report.

    Returns:
        The generated PDF as bytes if successful, otherwise None.
    """
    logger.info(f"Starting PDF report generation for session ID: {session_id}")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(15, 15, 15) # Set consistent margins
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Header ---
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=REPORT_TITLE, ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, txt=f"Session ID: {session_id}", ln=True, align='C')
    pdf.ln(10) # Space after header

    # --- Embed Image ---
    temp_image_path = None # Define path variable outside try
    try:
        if image is None:
             logger.warning("No image provided for PDF report.")
             pdf.set_font("Arial", 'I', 10)
             pdf.set_text_color(*ERROR_COLOR)
             pdf.cell(0, 10, "[No image available for this report]", ln=True, align='C')
             pdf.set_text_color(*DEFAULT_COLOR)
             pdf.ln(5)
        else:
            # Save image temporarily to embed it by path (most compatible fpdf method)
            # Using delete=False requires manual cleanup in 'finally'
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix=TEMP_IMAGE_PREFIX) as tmpfile:
                temp_image_path = tmpfile.name
                logger.debug(f"Saving temporary image for PDF report to: {temp_image_path}")
                image.save(temp_image_path, format="PNG")
                # tmpfile is closed here, but file still exists due to delete=False

            # Calculate image display size to fit page width while maintaining aspect ratio
            page_width_mm = pdf.w - pdf.l_margin - pdf.r_margin # Usable page width
            max_img_width_mm = page_width_mm * 0.9 # Allow some padding

            img_width_px, img_height_px = image.size
            if img_width_px <= 0 or img_height_px <= 0:
                 raise ValueError(f"Invalid image dimensions: {image.size}")

            aspect_ratio = img_height_px / img_width_px
            display_width_mm = min(max_img_width_mm, img_width_px / 3) # Basic heuristic for pixel to mm, adjust if needed
            display_height_mm = display_width_mm * aspect_ratio

            # Ensure image doesn't overflow page height
            available_height_mm = pdf.h - pdf.get_y() - pdf.b_margin - 10 # Available vertical space minus padding
            if display_height_mm > available_height_mm:
                logger.debug(f"Image height ({display_height_mm:.1f}mm) exceeds available space ({available_height_mm:.1f}mm), resizing.")
                display_height_mm = available_height_mm
                display_width_mm = display_height_mm / aspect_ratio # Recalculate width based on new height

            # Center the image horizontally
            x_pos = (pdf.w - display_width_mm) / 2
            logger.debug(f"Embedding image: Path='{os.path.basename(temp_image_path)}', Pos=({x_pos:.1f}, {pdf.get_y():.1f}), Size=({display_width_mm:.1f}x{display_height_mm:.1f} mm)")
            pdf.image(temp_image_path, x=x_pos, y=pdf.get_y(), w=display_width_mm, h=display_height_mm)
            pdf.ln(display_height_mm + 5) # Move cursor below image plus padding

    except Exception as e:
        err_msg = f"Error processing or embedding image in PDF: {e}"
        logger.error(err_msg, exc_info=True)
        # Add an error message within the PDF itself
        pdf.set_text_color(*ERROR_COLOR)
        pdf.set_font("Arial", 'B', 10)
        pdf.multi_cell(0, 6, "[Error displaying image in report - See application logs for details]", align='C')
        pdf.set_text_color(*DEFAULT_COLOR)
        pdf.set_font("Arial", size=10) # Reset font
        pdf.ln(5)
    finally:
        # --- Crucial Cleanup for delete=False ---
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                logger.debug(f"Successfully removed temporary image file: {temp_image_path}")
            except OSError as e:
                # Log warning but don't crash PDF generation if removal fails
                logger.warning(f"Could not remove temporary image file '{temp_image_path}': {e}")

    # --- Add Analysis Sections ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Analysis Results", ln=True)

    if not analysis_outputs:
         pdf.set_font("Arial", 'I', 10)
         pdf.cell(0, 6, txt="No analysis results were generated for this session.", ln=True)
         pdf.ln(4)
    else:
        for section_title, content in analysis_outputs.items():
            # Section Title
            pdf.set_font("Arial", 'B', 11)
            # Use multi_cell for title in case it's long
            pdf.multi_cell(0, 6, f"{section_title}:")
            pdf.set_font("Arial", size=10)

            # Section Content
            content_to_write = content.strip() if content else "N/A"
            try:
                # FPDF(2) primarily supports Latin-1. Encode/decode with 'replace'
                # handles unsupported characters gracefully for PDF generation.
                encoded_content = content_to_write.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 5, txt=encoded_content)
            except Exception as e:
                # Handle potential errors during string processing or writing
                logger.error(f"Error writing section '{section_title}' to PDF: {e}", exc_info=True)
                pdf.set_text_color(*ERROR_COLOR)
                pdf.multi_cell(0, 5, f"[Error rendering content for this section - See application logs]")
                pdf.set_text_color(*DEFAULT_COLOR)

            pdf.ln(4) # Space between sections

    # --- Footer Disclaimer ---
    pdf.set_y(-20) # Position footer slightly higher
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(128, 128, 128) # Gray color for footer
    pdf.multi_cell(0, 4, txt=FOOTER_TEXT, align='C') # Use multi_cell for potential wrapping
    pdf.set_text_color(*DEFAULT_COLOR) # Reset color

    # --- Output PDF as Bytes ---
    try:
        # Output 'S' returns bytes (implicitly latin-1 encoded string first)
        pdf_bytes = pdf.output(dest='S')
        if isinstance(pdf_bytes, str):
             # If fpdf returned str (older versions?), encode explicitly
             pdf_bytes = pdf_bytes.encode('latin-1')
        logger.info(f"PDF report ({len(pdf_bytes)} bytes) generated successfully for session {session_id}.")
        return pdf_bytes
    except Exception as e:
        logger.error(f"Critical error during final PDF byte generation: {e}", exc_info=True)
        return None # Indicate failure