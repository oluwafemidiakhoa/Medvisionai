import os
import logging
import tempfile
from fpdf import FPDF  # Ensure fpdf2 (pip install fpdf2)
from PIL import Image, ImageDraw
from typing import Optional, Dict, Any, List

from umls_utils import search_umls, UMLSConcept, DEFAULT_UMLS_HITS

logger = logging.getLogger(__name__)

# --- Constants ---
REPORT_TITLE = "MediVision QA - AI Analysis Report"
FOOTER_TEXT = (
    "Disclaimer: AI-generated analysis. For informational purposes only. "
    "Requires expert clinical validation."
)
ERROR_COLOR = (255, 0, 0)
DEFAULT_COLOR = (0, 0, 0)
TEMP_IMAGE_PREFIX = "medivision_report_img_"

# --- PDF Generation ---
def generate_pdf_report_bytes(
    session_id: str,
    image: Optional[Image.Image],
    analysis_outputs: Dict[str, str]
) -> Optional[bytes]:
    """
    Generates a PDF report summarizing the analysis session with:
      - Header and session metadata
      - Embedded image (with ROI box if drawn)
      - Analysis sections
      - Standardized UMLS Concepts + SNOMED/ICD sections
      - Footer disclaimer
    """
    logger.info(f"Starting PDF report generation for session ID: {session_id}")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)

    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, REPORT_TITLE, ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, f"Session ID: {session_id}", ln=True, align='C')
    pdf.ln(10)

    # Embed Image
    temp_image_path = None
    try:
        if image is None:
            pdf.set_font("Arial", 'I', 10)
            pdf.set_text_color(*ERROR_COLOR)
            pdf.cell(0, 10, "[No image available for this report]", ln=True, align='C')
            pdf.set_text_color(*DEFAULT_COLOR)
            pdf.ln(5)
        else:
            # Save temp image
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix=TEMP_IMAGE_PREFIX) as tmp:
                temp_image_path = tmp.name
                image.save(temp_image_path, format="PNG")

            page_w = pdf.w - pdf.l_margin - pdf.r_margin
            max_w = page_w * 0.9
            iw, ih = image.size
            ar = ih / iw if iw else 1
            display_w = min(max_w, iw / 3)
            display_h = display_w * ar
            avail_h = pdf.h - pdf.get_y() - pdf.b_margin - 10
            if display_h > avail_h:
                display_h = avail_h
                display_w = display_h / ar

            x_pos = (pdf.w - display_w) / 2
            pdf.image(temp_image_path, x=x_pos, y=pdf.get_y(), w=display_w, h=display_h)
            pdf.ln(display_h + 5)
    except Exception as e:
        logger.error(f"Error embedding image: {e}", exc_info=True)
        pdf.set_text_color(*ERROR_COLOR)
        pdf.set_font("Arial", 'B', 10)
        pdf.multi_cell(0, 6, "[Error displaying image in report]")
        pdf.set_text_color(*DEFAULT_COLOR)
        pdf.set_font("Arial", size=10)
        pdf.ln(5)
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except Exception:
                pass

    # Analysis sections
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Analysis Results", ln=True)
    pdf.ln(2)
    if not analysis_outputs:
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 6, "No analysis results for this session.", ln=True)
        pdf.ln(4)
    else:
        for title, text in analysis_outputs.items():
            pdf.set_font("Arial", 'B', 11)
            pdf.multi_cell(0, 6, f"{title}:")
            pdf.set_font("Arial", size=10)
            content = text.strip() if text else "N/A"
            try:
                pdf.multi_cell(0, 5, txt=content)
            except Exception:
                pdf.multi_cell(0, 5, "[Error rendering section]")
            pdf.ln(4)

    # UMLS Concepts
    api_key = os.getenv("UMLS_APIKEY")
    if api_key:
        try:
            combined = "\n".join(analysis_outputs.values())
            concepts: List[UMLSConcept] = search_umls(combined, api_key, page_size=DEFAULT_UMLS_HITS)
            if concepts:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Standardized UMLS Concepts", ln=True)
                pdf.set_font("Arial", size=10)
                for c in concepts:
                    pdf.multi_cell(0, 5, f"{c.name} (CUI: {c.ui}, Source: {c.rootSource})")
                pdf.ln(4)
                # SNOMED / ICD-10 sections
                snomed = [c for c in concepts if c.rootSource.upper().startswith("SNOMEDCT")]
                icd10  = [c for c in concepts if c.rootSource.upper().startswith("ICD10")]
                if snomed:
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 8, "SNOMED CT Codes", ln=True)
                    pdf.set_font("Arial", size=10)
                    for c in snomed:
                        pdf.multi_cell(0, 5, f"{c.name}: {c.ui}")
                    pdf.ln(4)
                if icd10:
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 8, "ICD-10 Codes", ln=True)
                    pdf.set_font("Arial", size=10)
                    for c in icd10:
                        pdf.multi_cell(0, 5, f"{c.name}: {c.ui}")
                    pdf.ln(4)
        except Exception as e:
            logger.error(f"UMLS mapping section failed: {e}", exc_info=True)

    # Footer
    pdf.set_y(-20)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 4, FOOTER_TEXT, align='C')
    pdf.set_text_color(*DEFAULT_COLOR)

    # Output
    try:
        raw = pdf.output(dest='S')
        result = raw.encode('latin-1') if isinstance(raw, str) else raw
        return result
    except Exception as e:
        logger.error(f"Error generating PDF bytes: {e}")
        return None
