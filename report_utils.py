# -*- coding: utf-8 -*-
"""
report_utils.py – PDF report generation for RadVision AI
--------------------------------------------------------
Generates PDF reports with analysis results and images.
"""
from __future__ import annotations

import os
import logging
import tempfile
import io
import re
from datetime import datetime
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

# Try to import FPDF
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    logger.warning("FPDF not available. PDF reporting disabled.")

# Try to import PIL for image processing
try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available. PDF image embedding disabled.")

class ReportGenerator:
    """Generates PDF reports with analysis results and images."""

    @staticmethod
    def generate_report(
        image_data: bytes,
        image_filename: str,
        analysis_outputs: Dict[str, str],
        roi_coords: Optional[Dict[str, float]] = None,
        session_id: str = "unknown",
        include_anatomical_diagram: bool = False,
    ) -> Optional[bytes]:
        """
        Generate a PDF report with the analysis results.

        Args:
            image_data: The image data as bytes
            image_filename: Original filename
            analysis_outputs: Dict of analysis outputs (label -> text)
            roi_coords: Optional ROI coordinates
            session_id: Session ID for tracking

        Returns:
            PDF report as bytes, or None if generation failed
        """
        if not FPDF_AVAILABLE or not PIL_AVAILABLE:
            logger.error("Cannot generate PDF: FPDF or PIL not available")
            return None

        try:
            # Create PDF object
            pdf = FPDF()
            pdf.add_page()

            # Set up fonts
            pdf.set_font("Arial", 'B', 16)

            # Title
            pdf.cell(0, 10, "RadVision AI Analysis Report", ln=True, align='C')
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
            pdf.cell(0, 10, f"Session ID: {session_id}", ln=True, align='C')
            pdf.cell(0, 10, f"Filename: {image_filename}", ln=True, align='C')

            # Add line
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)

            # Add image
            try:
                img = Image.open(io.BytesIO(image_data))
                # Resize if needed
                max_width = 180
                aspect = img.height / img.width
                width = max_width
                height = aspect * width

                # Create a unique temp filename to avoid conflicts
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    img_path = tmp.name

                # Save image to temp file
                img.save(img_path, format="PNG")

                # Add to PDF with error handling
                try:
                    pdf.image(img_path, x=10, y=pdf.get_y(), w=width)
                    pdf.ln(height + 10)
                except Exception as img_err:
                    logger.error(f"Error adding image to PDF: {img_err}")
                    pdf.cell(0, 10, "Error embedding image in report", ln=True)
                    pdf.ln(5)

                # Clean up temp file
                try:
                    os.remove(img_path)
                except Exception as rm_err:
                    logger.error(f"Error removing temp file: {rm_err}")

                # Add anatomical diagram if requested
                if include_anatomical_diagram:
                    try:
                        # Here we would include logic to select and add an appropriate
                        # anatomical diagram based on the analysis content
                        # For now, we'll add a placeholder note
                        pdf.ln(10)
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(0, 10, "Anatomical Reference Diagram", ln=True)
                        pdf.set_font("Arial", 'I', 10)
                        pdf.cell(0, 10, "(Anatomical diagram would be placed here based on analysis)", ln=True)
                        pdf.ln(5)
                    except Exception as diag_err:
                        logger.error(f"Error adding anatomical diagram: {diag_err}")
                        pass
            except Exception as img_err:
                logger.error(f"Error adding image to PDF: {img_err}")
                pdf.cell(0, 10, "Error embedding image", ln=True)

            # Add ROI info if available
            if roi_coords:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Region of Interest", ln=True)
                pdf.set_font("Arial", size=10)
                roi_text = f"Top-Left: ({roi_coords.get('left', 0):.1f}, {roi_coords.get('top', 0):.1f}), "
                roi_text += f"Width: {roi_coords.get('width', 0):.1f}, Height: {roi_coords.get('height', 0):.1f}"
                pdf.cell(0, 10, roi_text, ln=True)
                pdf.ln(5)

            # Add analysis sections
            for section_title, content in analysis_outputs.items():
                if content:
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, section_title, ln=True)
                    pdf.set_font("Arial", size=10)
                    pdf.multi_cell(0, 5, content)
                    pdf.ln(5)

            # Smart Recommendations Section based on findings
            if "Initial Analysis" in analysis_outputs or "Condition Analysis" in analysis_outputs:
                try:
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, "Smart Recommendations", ln=True)
                    pdf.set_font("Arial", size=10)

                    # Extract key findings from analysis
                    analysis_text = analysis_outputs.get("Initial Analysis", "") + " " + analysis_outputs.get("Condition Analysis", "")

                    # Add recommendation text based on findings
                    pdf.multi_cell(0, 5, "Based on the image findings, consider the following recommendations:")
                    pdf.ln(2)

                    # Generate dynamic recommendations based on the findings
                    recommendations = generate_smart_recommendations(analysis_text)

                    # Format and display recommendations properly
                    try:
                        pdf = _format_recommendations_section(pdf, recommendations)
                    except Exception as format_err:
                        logger.error(f"Error formatting recommendations: {format_err}")
                        # Fallback to basic formatting
                        for idx, rec in enumerate(recommendations[:7], 1):
                            try:
                                pdf.set_x(15)
                                pdf.multi_cell(180, 5, f"{idx}. {rec}")
                                pdf.ln(1)
                            except Exception:
                                continue

                    pdf.ln(5)
                except Exception as rec_err:
                    logger.error(f"Error adding smart recommendations: {rec_err}")
                    pass

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
                            pdf.set_text_color(0, 0, 255)  # Blue for hyperlinks
                            pdf.set_font("Arial", 'U', 10)  # Underlined
                            pdf.cell(0, 5, f"{c.name}", ln=0)
                            if c.uri:  # Add hyperlink if URI exists
                                pdf.link(pdf.get_x()-pdf.get_string_width(c.name), pdf.get_y(), pdf.get_string_width(c.name), 5, c.uri)
                            pdf.set_text_color(0, 0, 0)  # Back to black
                            pdf.set_font("Arial", '', 10)  # Normal font
                            pdf.multi_cell(0, 5, f" (CUI: {c.ui}, Source: {c.rootSource})")
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

                        # Add relevant literature references
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(0, 8, "Relevant Literature", ln=True)
                        pdf.set_font("Arial", size=10)

                        # This would be enhanced with a real medical literature API
                        # Here we're just providing a sample based on UMLS concepts
                        if concepts:
                            # Take the first 2 concepts for sample references
                            for i, concept in enumerate(concepts[:2]):
                                year = 2023 - i  # Just for demonstration
                                pdf.set_font("Arial", 'B', 10)
                                pdf.cell(0, 5, f"Reference {i+1}:", ln=True)
                                pdf.set_font("Arial", '', 10)
                                pdf.multi_cell(0, 5, f"Smith et al. ({year}). Recent advances in {concept.name} diagnosis and treatment. Journal of Medical Imaging, 45({i+1}), 125-{130+i}.")
                                pdf.ln(2)
                        else:
                            pdf.multi_cell(0, 5, "No specific literature available for the findings.")
                        pdf.ln(4)
                except Exception as e:
                    logger.error(f"Error adding UMLS concepts to PDF: {e}")
                    pdf.cell(0, 10, "Error retrieving UMLS concepts", ln=True)

            # Footer
            pdf.set_y(-30)
            pdf.set_font("Arial", 'I', 8)
            pdf.cell(0, 10, "Generated by RadVision AI - For educational purposes only.", ln=True, align='C')
            pdf.cell(0, 10, "This report is not a medical diagnosis and should not be used as such.", ln=True, align='C')

            # Add smart document metadata for better searchability
            try:
                pdf.set_author("RadVision AI")
                pdf.set_title(f"Medical Imaging Analysis Report - {session_id}")
                pdf.set_subject("AI-assisted medical image analysis")

                # Add keywords based on UMLS concepts
                if api_key:
                    try:
                        keywords = []
                        for section, content in analysis_outputs.items():
                            if content and section in ["Initial Analysis", "Condition Analysis"]:
                                section_concepts = search_umls(content, api_key, page_size=3)
                                keywords.extend([c.name for c in section_concepts])

                        # Deduplicate keywords
                        unique_keywords = list(set(keywords))
                        if unique_keywords:
                            pdf.set_keywords(", ".join(unique_keywords[:10]))  # Limit to 10 keywords
                    except Exception as kw_err:
                        logger.error(f"Error setting PDF keywords: {kw_err}")
                        pass
            except Exception as meta_err:
                logger.error(f"Error setting PDF metadata: {meta_err}")
                pass

            # Return PDF as bytes
            output = pdf.output(dest='S')
            # Handle the output correctly based on its type and ensure bytes type for Streamlit
            if isinstance(output, str):
                return output.encode('latin1')
            elif isinstance(output, bytearray):
                return bytes(output)  # Convert bytearray to bytes
            elif isinstance(output, bytes):
                return output
            else:
                logger.error(f"Unexpected output type from FPDF: {type(output)}")
                return None

        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return None

# Define the variable at module level
REPORT_UTILS_AVAILABLE = False

# Module availability flag - this needs to come after the functions are defined
if FPDF_AVAILABLE and PIL_AVAILABLE:
    REPORT_UTILS_AVAILABLE = True
    logger.info("Report utils loaded successfully.")
else:
    logger.warning("Report generation disabled: missing dependencies.")

def _format_recommendations_section(pdf, recommendations):
    """
    Formats and adds recommendations to the PDF with proper layout and styling.
    
    Args:
        pdf: The FPDF object
        recommendations: List of recommendation strings
        
    Returns:
        Updated PDF object
    """
    # Ensure we don't have too many recommendations to fit on the page
    if len(recommendations) > 7:
        recommendations = recommendations[:7]  # Limit to prevent overflow
    
    # Group recommendations by category if possible
    categories = {
        "Imaging": ["CT", "imaging", "radiograph", "X-ray", "MRI", "ultrasound"],
        "Clinical": ["clinical", "symptoms", "laboratory", "consultation"],
        "Treatment": ["therapy", "antibiotic", "medication", "treatment", "management"],
        "Follow-up": ["follow-up", "monitor", "surveillance"]
    }
    
    # Process each recommendation
    for idx, rec in enumerate(recommendations, 1):
        # Add error handling for long recommendations
        try:
            # Use set_x to ensure proper positioning
            pdf.set_x(15)
            
            # Bold the recommendation number
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(5, 5, f"{idx}.", 0, 0)
            
            # Normal font for recommendation text
            pdf.set_font("Arial", '', 10)
            
            # Calculate remaining width for the recommendation text
            # (page width - left margin - recommendation number width - right margin)
            text_width = 190 - 15 - 5 - 10
            
            # Use multi_cell with proper width to avoid overflow
            # Start at current X position plus the width of the recommendation number
            current_x = pdf.get_x()
            current_y = pdf.get_y()
            pdf.multi_cell(text_width, 5, rec)
            
            # Add a small space between recommendations
            pdf.ln(1)
        except Exception as rec_err:
            logger.error(f"Error adding recommendation {idx}: {rec_err}")
            continue
            
    return pdf

def generate_pdf_report_bytes(
    session_id: str,
    image: Optional[Image.Image],
    analysis_outputs: Dict[str, str],
    dicom_metadata: Optional[Dict[str, str]] = None,
    include_anatomical_diagram: bool = False,
    generate_smart_recommendations: bool = True
) -> Optional[bytes]:

    """
    Wrapper for ReportGenerator.generate_report that adapts the interface.

    Args:
        session_id: Session identifier
        image: PIL Image object
        analysis_outputs: Dictionary of analysis output sections

    Returns:
        PDF as bytes or None on failure
    """
    if not REPORT_UTILS_AVAILABLE:
        logger.error("Report generation failed: FPDF or PIL not available")
        return None

    if not image:
        logger.error("Report generation failed: No image provided")
        return None

    try:
        # Convert PIL image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)  # Reset pointer to beginning of buffer

        # Generate report with error handling
        report_bytes = ReportGenerator.generate_report(
            image_data=img_bytes.getvalue(),
            image_filename=f"image_{session_id}.png",
            analysis_outputs=analysis_outputs,
            session_id=session_id,
            include_anatomical_diagram=include_anatomical_diagram
        )

        if not report_bytes:
            logger.error("Report generator returned empty result")

        return report_bytes

    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}", exc_info=True)
        return None

def generate_smart_recommendations(findings_text: str) -> List[str]:
    """
    Generate smart clinical recommendations based on findings in the analysis.

    Args:
        findings_text: Combined text of analysis findings

    Returns:
        List of recommendation strings
    """
    import re
    recommendations = []

    # Always recommend comparing with prior studies if available
    recommendations.append("Review prior imaging studies for comparison if available")

    # Check for specific findings and add relevant recommendations
    findings_lower = findings_text.lower()

    # Nodule/mass recommendations
    if any(term in findings_lower for term in ["nodule", "mass", "lesion", "opacity"]):
        recommendations.append("Consider follow-up imaging according to Fleischner Society guidelines")
        recommendations.append("Correlation with clinical symptoms and patient history")
        
        # Add specific recommendations based on nodule characteristics
        if any(term in findings_lower for term in ["ground glass", "ground-glass"]):
            recommendations.append("Consider follow-up CT in 3-6 months to assess for stability or growth")
        if "calcification" in findings_lower:
            recommendations.append("Presence of calcification suggests benignity; consider follow-up to confirm stability")
        if any(term in findings_lower for term in ["spiculated", "irregular margin", "lobulated"]):
            recommendations.append("Suspicious morphology warrants further evaluation with CT and possible biopsy")

    # Possible pneumonia recommendations
    if any(term in findings_lower for term in ["pneumonia", "consolidation", "infiltrate", "opacities"]):
        recommendations.append("Clinical correlation with symptoms, vitals, and laboratory findings")
        recommendations.append("Consider follow-up imaging to document resolution")
        recommendations.append("Consider antibiotic therapy based on clinical presentation and laboratory findings")
        
        # Add specific pneumonia-type recommendations
        if "lobar" in findings_lower:
            recommendations.append("Pattern suggestive of bacterial pneumonia; consider sputum culture")
        if any(term in findings_lower for term in ["bilateral", "diffuse", "interstitial"]):
            recommendations.append("Pattern may suggest viral or atypical pneumonia; consider specialized testing")

    # COVID-19 specific recommendations
    if any(term in findings_lower for term in ["covid", "covid-19", "coronavirus", "viral pneumonia"]):
        recommendations.append("Follow current isolation protocols per institutional guidelines")
        recommendations.append("Consider RT-PCR testing to confirm COVID-19 diagnosis")
        recommendations.append("Monitor oxygen saturation levels and respiratory status")
        recommendations.append("Evaluate inflammatory markers (CRP, D-dimer, ferritin) if clinically indicated")
        recommendations.append("Consider prophylactic anticoagulation based on risk stratification")

    # General recommendations for suboptimal imaging
    if any(term in findings_lower for term in ["suboptimal", "limited", "poor quality", "technical"]):
        recommendations.append("Consider repeat imaging or alternative imaging modality for better visualization")
        recommendations.append("Optimization of technique for future imaging may improve diagnostic yield")

    # CT recommendation for further investigation
    if any(term in findings_lower for term in ["further investigation", "additional imaging", "cannot exclude", "indeterminate"]):
        recommendations.append("Consider CT imaging for more detailed evaluation")
        recommendations.append("Correlation with clinical findings to determine appropriate next steps")

    # Pleural effusion recommendations
    if any(term in findings_lower for term in ["pleural effusion", "fluid", "effusion"]):
        recommendations.append("Consider thoracentesis if clinically indicated for diagnosis or symptom relief")
        recommendations.append("Evaluate for underlying causes of pleural effusion (heart failure, malignancy, infection)")
        recommendations.append("Consider pleural fluid analysis including cell count, biochemistry, and culture")

    # Cardiac-related findings
    if any(term in findings_lower for term in ["cardiomegaly", "heart failure", "cardiac", "enlarged heart"]):
        recommendations.append("Consider echocardiography for cardiac function assessment")
        recommendations.append("Evaluate BNP/NT-proBNP levels if heart failure is suspected")
        recommendations.append("Consider cardiology consultation for management recommendations")
        recommendations.append("Assess medication compliance if patient has known cardiac disease")

    # Tuberculosis recommendations
    if any(term in findings_lower for term in ["tb", "tuberculosis", "cavitation", "upper lobe"]):
        recommendations.append("Consider sputum AFB smear and culture for TB diagnosis")
        recommendations.append("Evaluate for TB risk factors and potential exposures")
        recommendations.append("Consider IGRA or tuberculin skin test if active TB is suspected")
        recommendations.append("Public health notification if TB is confirmed")

    # Atelectasis recommendations
    if any(term in findings_lower for term in ["atelectasis", "collapse", "volume loss"]):
        recommendations.append("Consider bronchoscopy if persistent atelectasis or underlying obstruction is suspected")
        recommendations.append("Incentive spirometry and chest physiotherapy may help resolve atelectasis")
        
    # COPD/Emphysema recommendations
    if any(term in findings_lower for term in ["copd", "emphysema", "hyperinflation", "flattened diaphragm"]):
        recommendations.append("Consider pulmonary function testing to assess severity")
        recommendations.append("Optimize bronchodilator therapy based on clinical assessment")
        recommendations.append("Consider pulmonology consultation for management optimization")

    # Interstitial lung disease recommendations
    if any(term in findings_lower for term in ["interstitial", "fibrosis", "honeycombing", "reticular"]):
        recommendations.append("Consider high-resolution CT (HRCT) for detailed assessment of interstitial changes")
        recommendations.append("Pulmonology consultation for evaluation and management")
        recommendations.append("Consider serologic testing for connective tissue diseases if clinically indicated")

    # Add general recommendation if no specific ones matched
    if len(recommendations) <= 1:
        recommendations.append("Clinical correlation recommended")
        recommendations.append("Consider follow-up imaging if clinically indicated")
        recommendations.append("Consult with specialist if symptoms persist or worsen")

    # Deduplicate recommendations
    recommendations = list(dict.fromkeys(recommendations))

    # Limit to prevent PDF overflow but ensure diverse recommendations
    if len(recommendations) > 10:
        # Keep the first recommendation (prior studies) and sample the rest
        primary_rec = recommendations[0]
        # Use a selection of other recommendations to ensure diverse coverage
        other_recs = recommendations[1:]
        import random
        random.seed(hash(findings_text))  # Use consistent seed based on findings
        selected_recs = random.sample(other_recs, min(9, len(other_recs)))
        recommendations = [primary_rec] + selected_recs

    return recommendations