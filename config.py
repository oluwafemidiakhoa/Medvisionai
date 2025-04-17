# config.py
import os

# --- Session State Defaults ---
DEFAULT_STATE = {
    "uploaded_file_info": None, "raw_image_bytes": None, "is_dicom": False,
    "dicom_dataset": None, "dicom_metadata": {}, "processed_image": None,
    "display_image": None, "session_id": None, "history": [],
    "initial_analysis": "", "qa_answer": "", "disease_analysis": "",
    "confidence_score": "", "last_action": None, "pdf_report_bytes": None,
    "canvas_drawing": None, "roi_coords": None, "current_display_wc": None,
    "current_display_ww": None, "clear_roi_feedback": False, "demo_loaded": False,
    "translation_result": None, "translation_error": None, "umls_search_term": "",
    "umls_results": None, "umls_error": None,
}

# --- Sidebar Options ---
TIPS = [
    "Tip: Use 'Demo Mode' for a quick walkthrough.", "Tip: Draw an ROI rectangle.",
    "Tip: Adjust DICOM W/L.", "Tip: Ask follow-up questions.", "Tip: Generate a PDF report.",
    "Tip: Use 'Translation' tab.", "Tip: Clear the ROI.", "Tip: Use 'UMLS Lookup'."
]

DISEASE_OPTIONS = [
    "Pneumonia", "Lung Cancer", "Nodule/Mass", "Effusion", "Fracture", "Stroke",
    "Appendicitis", "Bowel Obstruction", "Cardiomegaly", "Aortic Aneurysm",
    "Pulmonary Embolism", "Tuberculosis", "COVID-19", "Brain Tumor",
    "Arthritis", "Osteoporosis", "Other..."
]

# --- Logging ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# --- CSS Styling ---
# (Keep the CSS string here)
APP_CSS = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f0f2f6; }
      .main .block-container { padding: 2rem 1.5rem; }
      .css-1d391kg { background-color: #ffffff; border-right: 1px solid #e0e0e0; } /* Sidebar */
      .stButton>button { border-radius: 8px; padding: 0.5rem 1rem; font-weight: 500; width: 100%; margin-bottom: 0.5rem; }
      .stButton>button:hover { filter: brightness(95%); }
      /* --- FIX for faint text in sidebar inputs --- */
      .css-1d391kg .stTextArea textarea::placeholder { color: #6c757d !important; opacity: 1; }
      .css-1d391kg div[data-baseweb="select"] > div:first-child > div:first-child { color: #31333F !important; }
      .css-1d391kg div[data-baseweb="select"] svg { fill: #31333F !important; }
      /* --- End FIX --- */
      div[role="tablist"] { overflow-x: auto; white-space: nowrap; border-bottom: 1px solid #e0e0e0; scrollbar-width: thin; scrollbar-color: #ccc #f0f2f6; }
      div[role="tablist"]::-webkit-scrollbar { height: 6px; }
      div[role="tablist"]::-webkit-scrollbar-track { background: #f0f2f6; }
      div[role="tablist"]::-webkit-scrollbar-thumb { background-color: #ccc; border-radius: 10px; border: 2px solid #f0f2f6; }
      footer { text-align: center; font-size: 0.8em; color: #6c757d; margin-top: 2rem; padding: 1rem 0; border-top: 1px solid #e0e0e0; }
      footer a { color: #007bff; text-decoration: none; }
      footer a:hover { text-decoration: underline; }
    </style>
    """

# --- Footer ---
FOOTER_MARKDOWN = """
    <footer>
      <p>RadVision AI is for informational/educational use ONLY. Not medical advice.</p>
      <p> <a href="#" target="_blank">Privacy</a> | <a href="#" target="_blank">Terms</a> | <a href="https://github.com/mgbam/radvisionai" target="_blank">GitHub</a> </p>
    </footer>
    """