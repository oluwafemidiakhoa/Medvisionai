# config.py
import os
import logging # Import logging to use its constants like logging.INFO

# --- General App Settings ---
APP_TITLE = "RadVision AI Advanced"
APP_ICON = "⚕️" # Or a URL to an image file, e.g., "/static/icon.png"

# --- Session State Defaults ---
# Defines the initial structure and default values for st.session_state
# Keys here should match the keys used throughout the application modules.
DEFAULT_STATE = {
    # File/Upload Info
    "uploaded_file_info": None,     # Dict: {name, size, type} of the last processed file
    "raw_image_bytes": None,        # Bytes of the uploaded file (useful for reprocessing)
    "demo_loaded": False,           # Flag if demo mode is active

    # Image Processing & Display
    "is_dicom": False,              # Boolean indicating if the loaded file is DICOM
    "dicom_dataset": None,          # pydicom.Dataset object if DICOM
    "dicom_metadata": None,         # Dict extracted from DICOM headers
    "display_image": None,          # PIL.Image object ready for display (viewer uses this!)
    "current_display_wc": None,     # Currently applied DICOM Window Center (Level)
    "current_display_ww": None,     # Currently applied DICOM Window Width

    # ROI & Canvas
    "canvas_drawing": None,         # State for streamlit-drawable-canvas (to redraw ROI)
    "roi_coords": None,             # Dict: {left, top, width, height} of the defined ROI

    # AI Analysis Results
    "initial_analysis": "",         # Text result from initial analysis LLM call
    "initial_analysis_umls": [],    # List[UMLSConcept] mapped from initial_analysis
    "qa_answer": "",                # Text result from the latest Q&A LLM call
    "qa_umls": [],                  # List[UMLSConcept] mapped from qa_answer
    "disease_analysis": "",         # Text result from condition-specific analysis
    "disease_umls": [],             # List[UMLSConcept] mapped from disease_analysis
    "confidence_score": "",         # Text result from confidence estimation call

    # Interaction State
    "session_id": None,             # Unique identifier for the session
    "history": [],                  # List of tuples: (question, answer, umls_concepts) for Q&A
    "last_action": None,            # String indicating the last sidebar action clicked (e.g., "run_analysis")

    # Optional Features State
    "pdf_report_bytes": None,       # Bytes of the generated PDF report
    "translation_result": None,     # Text result from the translation tab
    "translation_error": None,      # Error message from translation attempt
    "umls_lookup_term": "",         # Term entered in the manual UMLS lookup tab
    "umls_lookup_results": None,    # List[UMLSConcept] from manual lookup search
    "umls_lookup_error": None,      # Error message from manual lookup attempt

    # UI Feedback Flags (Optional)
    "clear_roi_feedback": False,    # Flag to potentially show a "ROI cleared" message

    # REMOVED: "processed_image" - This seems redundant if "display_image" is the primary
    # key used for showing the image after all processing (like windowing).
    # If you have specific intermediate steps, you might keep it, but ensure clarity.
}

# --- Sidebar UI Elements ---
# Tips shown randomly or cyclically in the sidebar
TIPS = [
    "Tip: Use 'Demo Mode' for a quick walkthrough with a sample image.",
    "Tip: Draw an ROI rectangle directly on the image viewer to focus analysis.",
    "Tip: Adjust DICOM Window/Level (W/L) sliders for optimal image contrast.",
    "Tip: Ask specific follow-up questions using the 'Ask AI a Question' section.",
    "Tip: Generate a PDF report to save or share the current analysis findings.",
    "Tip: Use the 'Translate' tab to view findings in different languages.",
    "Tip: Click 'Clear Image / ROI' in the sidebar to start fresh.",
    "Tip: Use the 'UMLS Lookup' tab to find definitions for medical terms.",
]

# Options for the condition-specific analysis dropdown
DISEASE_OPTIONS = [
    "Pneumonia", "Lung Cancer", "Nodule/Mass", "Effusion", "Fracture", "Stroke",
    "Appendicitis", "Bowel Obstruction", "Cardiomegaly", "Aortic Aneurysm",
    "Pulmonary Embolism", "Tuberculosis", "COVID-19", "Brain Tumor",
    "Arthritis", "Osteoporosis", "Other..."
]

# --- UI Content Strings ---
# Markdown content for the user guide expander
USER_GUIDE_MARKDOWN = """
**Typical workflow:**
1.  **Upload** a medical image (DICOM, PNG, JPG) using the sidebar, or enable **Demo Mode**.
2.  **(If DICOM):** Adjust **Window/Level (W/L)** sliders in the sidebar for best contrast *after* the image loads.
3.  **(Optional):** Draw a rectangular **Region of Interest (ROI)** directly on the image viewer below.
4.  Trigger AI actions using the buttons in the sidebar (**Run Initial Analysis**, **Ask AI a Question**, **Analyze Condition**).
5.  Explore the results, translations, and mapped medical terms (**UMLS**) in the tabs on the right.
6.  **(Optional):** Generate a **PDF Report** to save the findings.
"""

# Warning message displayed prominently
DISCLAIMER_WARNING = "This tool is intended for research/educational use ONLY. It is **NOT** a substitute for professional medical evaluation or diagnosis by qualified healthcare providers."

# --- UMLS Configuration ---
DEFAULT_UMLS_HITS = 5 # Default number of concepts for manual lookup & auto-mapping
# User-facing message shown if UMLS API key is missing/invalid
UMLS_CONFIG_MSG = "Ensure `UMLS_APIKEY` is correctly set in Hugging Face Secrets and restart the Space."

# --- Logging Configuration ---
# Determine log level from environment variable, defaulting to INFO
LOG_LEVEL_STR = os.environ.get("LOG_LEVEL", "INFO").upper()
# Use getattr for safe conversion from string to logging level constant
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)

# Define log format and date format
LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# --- CSS Styling ---
# Keep your existing CSS block here. Ensure it defines necessary styles.
APP_CSS = """
    <style>
      /* Base styling */
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
      .main .block-container { padding: 2rem 1.5rem; max-width: 95%; } /* Adjust padding/width */

      /* Sidebar styling */
      /* Use Streamlit theme variables where possible, but override if needed */
      /* .css-1d391kg { background-color: var(--streamlit-secondary-background-color); } */

      /* Button styling */
      .stButton>button { border-radius: 8px; font-weight: 500; width: 100%; margin-bottom: 0.5rem; }
      .stButton>button:hover { filter: brightness(95%); }

      /* Text Area Placeholder Fix (especially for dark mode) */
      .stTextArea textarea::placeholder { color: #888 !important; opacity: 1; }

      /* Tab styling */
      div[role="tablist"] {
          overflow-x: auto; white-space: nowrap; border-bottom: 1px solid var(--streamlit-gray-30);
          scrollbar-width: thin; scrollbar-color: var(--streamlit-gray-40) var(--streamlit-secondary-background-color);
      }
      div[role="tablist"]::-webkit-scrollbar { height: 6px; }
      div[role="tablist"]::-webkit-scrollbar-track { background: var(--streamlit-secondary-background-color); }
      div[role="tablist"]::-webkit-scrollbar-thumb { background-color: var(--streamlit-gray-40); border-radius: 10px; border: 2px solid var(--streamlit-secondary-background-color); }

      /* Footer styling */
      footer { text-align: center; font-size: 0.8em; color: var(--streamlit-gray-60); margin-top: 3rem; padding: 1rem 0; border-top: 1px solid var(--streamlit-gray-30); }
      footer p { margin-bottom: 0.25rem; }
      footer a { color: var(--streamlit-link-color); text-decoration: none; }
      footer a:hover { text-decoration: underline; }
    </style>
    """

# --- Footer Content ---
# Remember to replace placeholder URLs
FOOTER_MARKDOWN = """
    <footer>
      <p>RadVision AI is for informational/educational use ONLY. Not medical advice.</p>
      <p>
          <a href="YOUR_PRIVACY_POLICY_URL_HERE" target="_blank">Privacy</a> |
          <a href="YOUR_TERMS_OF_SERVICE_URL_HERE" target="_blank">Terms</a> |
          <a href="https://github.com/mgbam/radvisionai" target="_blank">GitHub</a>
     </p>
    </footer>
    """