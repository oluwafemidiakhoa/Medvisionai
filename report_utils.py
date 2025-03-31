import os
import logging
from fpdf import FPDF
from PIL import Image
from typing import Optional, Dict
import tempfile

logger = logging.getLogger(__name__)

IMAGE_FILENAME_PREFIX = "medivision_image_"

# Keep the generate_pdf_report_bytes function exactly as in the previous refactored version.
# It takes session_id, image (PIL), and analysis_outputs dict, returns bytes or None.
# (Function omitted here for brevity, copy it from the previous response)

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
    # ... (rest of the PDF generation logic, including image embedding with tempfile) ...

    # --- Add Analysis Sections ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Analysis Results", ln=True)
    pdf.set_font("Arial", size=10)
    for section, content in analysis_outputs.items():
        pdf.set_font("Arial", 'B', 11)
        pdf.multi_cell(0, 6, f"{section}:")
        pdf.set_font("Arial", size=10)
        content_to_write = content if content else "N/A"
        try:
            encoded_content = content_to_write.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 5, encoded_content)
        except Exception as e:
            # ... error handling ...
        pdf.ln(4)

    # --- Footer Disclaimer ---
    # ...

    # --- Output PDF as Bytes ---
    try:
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        logger.info("PDF report generated successfully as bytes.")
        return pdf_bytes
    except Exception as e:
        logger.error(f"Error generating PDF bytes: {e}", exc_info=True)
        # st.error(f"Failed to generate PDF report: {e}") # Error displayed in main app
        return None