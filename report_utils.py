import os
import logging
import tempfile
from fpdf import FPDF  # Recommend using fpdf2: pip install fpdf2
from PIL import Image
from typing import Optional, Dict, Any

from umls_utils import search_umls, UMLSConcept, DEFAULT_UMLS_HITS

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

    Includes session ID, an embedded image (if provided), formatted
    analysis outputs, and standardized UMLS concept section.
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
    pdf.ln(10)

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
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix=TEMP_IMAGE_PREFIX) as tmpfile:
                temp_image_path = tmpfile.name
                logger.debug(f"Saving temporary image for PDF report to: {temp_image_path}")
                image.save(temp_image_path, format="PNG")

            page_width_mm = pdf.w - pdf.l_margin - pdf.r_margin
            max_img_width_mm = page_width_mm * 0.9
            img_w_px, img_h_px = image.size
            aspect = img_h_px / img_w_px if img_w_px else 1
            display_w_mm = min(max_img_width_mm, img_w_px / 3)
            display_h_mm = display_w_mm * aspect
            avail_h_mm = pdf.h - pdf.get_y() - pdf.b_margin - 10
            if display_h_mm > avail_h_mm:
                display_h_mm = avail_h_mm
                display_w_mm = display_h_mm / aspect

            x_pos = (pdf.w - display_w_mm) / 2
            pdf.image(temp_image_path, x=x_pos, y=pdf.get_y(), w=display_w_mm, h=display_h_mm)
            pdf.ln(display_h_mm + 5)
    except Exception as e:
        logger.error(f"Error processing or embedding image in PDF: {e}", exc_info=True)
        pdf.set_text_color(*ERROR_COLOR)
        pdf.set_font("Arial", 'B', 10)
        pdf.multi_cell(0, 6, "[Error displaying image in report - See application logs for details]", align='C')
        pdf.set_text_color(*DEFAULT_COLOR)
        pdf.set_font("Arial", size=10)
        pdf.ln(5)
    finally:
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
        for section, content in analysis_outputs.items():
            pdf.set_font("Arial", 'B', 11)
            pdf.multi_cell(0, 6, f"{section}:")
            pdf.set_font("Arial", size=10)
            text = content.strip() if content else "N/A"
            try:
                encoded = text.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 5, txt=encoded)
            except Exception as e:
                logger.error(f"Error writing section '{section}' to PDF: {e}", exc_info=True)
                pdf.set_text_color(*ERROR_COLOR)
                pdf.multi_cell(0, 5, "[Error rendering content for this section - See application logs]")
                pdf.set_text_color(*DEFAULT_COLOR)
            pdf.ln(4)

    # --- UMLS Concepts Section ---
    api_key = os.getenv("UMLS_APIKEY")
    if api_key:
        try:
            combined = "\n".join(analysis_outputs.values())
            concepts: list[UMLSConcept] = search_umls(combined, api_key, page_size=DEFAULT_UMLS_HITS)
            if concepts:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, txt="Standardized UMLS Concepts", ln=True)
                pdf.set_font("Arial", size=10)
                for c in concepts:
                    line = f"{c.name} (CUI: {c.ui}, Source: {c.rootSource})"
                    pdf.multi_cell(0, 5, txt=line)
                pdf.ln(4)
        except Exception as e:
            logger.error(f"UMLS PDF mapping failed: {e}", exc_info=True)

    # --- Footer ---
    pdf.set_y(-20)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 4, txt=FOOTER_TEXT, align='C')
    pdf.set_text_color(*DEFAULT_COLOR)

    # --- Output PDF as Bytes ---
    try:
        out = pdf.output(dest='S')
        pdf_bytes = out.encode('latin-1') if isinstance(out, str) else out
        logger.info(f"PDF report ({len(pdf_bytes)} bytes) generated successfully for session {session_id}.")
        return pdf_bytes
    except Exception as e:
        logger.error(f"Critical error during final PDF byte generation: {e}", exc_info=True)
        return None
