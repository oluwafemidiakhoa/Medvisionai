# config.py
import os
import logging # Import logging to use its constants like logging.INFO

# --- General App Settings ---
APP_TITLE = "RadVision AI Advanced"
APP_ICON = "⚕️" # Or a URL to an image file, e.g., "/static/icon.png"

# --- Session State Defaults ---
DEFAULT_STATE = {
    "uploaded_file_info": None,
    "raw_image_bytes": None,
    "is_dicom": False,
    "dicom_dataset": None,
    "dicom_metadata": {},
    "processed_image": None,    # Image after processing (e.g., windowing)
    "display_image": None,      # Image currently shown in viewer
    "session_id": None,
    "history": [],              # Could store tuples of (action, result)
    "initial_analysis": "",
    "qa_answer": "",
    "disease_analysis": "",
    "confidence_score": "",     # Could be numeric or text
    "last_action": None,        # Tracks the last button pressed
    "pdf_report_bytes": None,   # Stores generated PDF data
    "canvas_drawing": None,     # State from streamlit-drawable-canvas
    "roi_coords": None,         # Extracted ROI (x, y, w, h)
    "current_display_wc": None, # DICOM Window Center
    "current_display_ww": None, # DICOM Window Width
    "clear_roi_feedback": False,# Flag to show ROI cleared message
    "demo_loaded": False,       # Flag if demo mode is active
    "translation_result": None,
    "translation_error": None,
    # --- UMLS Specific State ---
    "umls_search_term": "",     # For manual lookup input
    "umls_results": None,       # For manual lookup results (list or dict)
    "umls_error": None,         # Error message from manual lookup
    "initial_analysis_umls": [],# List of UMLS concepts from initial analysis text
    "qa_umls": [],              # List of UMLS concepts from latest Q&A answer
    "disease_umls": [],         # List of UMLS concepts from disease analysis text
    # --- End UMLS ---
}

# --- Sidebar UI Elements ---
TIPS = [ # Keep existing TIPS
    "Tip: Use 'Demo Mode' for a quick walkthrough.",
    "Tip: Draw an ROI rectangle on the image viewer.",
    "Tip: Adjust DICOM Window/Level (W/L) for optimal contrast.",
    "Tip: Ask specific follow-up questions about the findings.",
    "Tip: Generate a PDF report to save or share the analysis.",
    "Tip: Use the 'Translate' tab for findings in other languages.",
    "Tip: Click 'Clear Image / ROI' to reset the viewer and analysis.",
    "Tip: Use the 'UMLS Lookup' tab to find medical term definitions.",
]
DISEASE_OPTIONS = [ # Keep existing options
    "Pneumonia", "Lung Cancer", "Nodule/Mass", "Effusion", "Fracture", "Stroke",
    "Appendicitis", "Bowel Obstruction", "Cardiomegaly", "Aortic Aneurysm",
    "Pulmonary Embolism", "Tuberculosis", "COVID-19", "Brain Tumor",
    "Arthritis", "Osteoporosis", "Other..."
]

# --- UI Content Strings ---
USER_GUIDE_MARKDOWN = """
**Typical workflow**
1.  **Upload** image (DICOM or PNG/JPG) – or enable *Demo Mode*.
2.  **(DICOM)** adjust *Window / Level* if required using the sidebar controls after upload.
3.  *(optional)* draw a rectangular **Region of Interest (ROI)** directly on the image.
4.  Trigger AI actions from the sidebar *(Initial Analysis, Ask a Question, Condition-Specific Analysis)*.
5.  Explore results in the different tabs (**UMLS**, **Translate**, **Confidence**) as needed.
6.  Generate a **PDF Report** summarizing the findings.
"""

DISCLAIMER_WARNING = "This tool is intended for research / educational use only. It is **NOT** a substitute for professional medical evaluation or diagnosis."

# --- UMLS Configuration ---
DEFAULT_UMLS_HITS = 5 # Number of concepts to fetch/display by default in lookups
# Message displayed if UMLS integration fails or isn't configured
UMLS_CONFIG_MSG = "Ensure `UMLS_APIKEY` is set in Hugging Face Secrets & restart."

# --- Logging Configuration ---
# Set level from environment variable or default to INFO
# Use logging constants for clarity (logging.INFO, logging.DEBUG, etc.)
LOG_LEVEL_STR = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)

LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# --- CSS Styling ---
# Keep existing CSS block
APP_CSS = """
    <style>
      /* Base styling */
      body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        /* background-color: #f0f2f6; */ /* Using Streamlit's theme is often better */
      }
      .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
      }

      /* Sidebar styling */
      .css-1d391kg { /* Specific selector for Streamlit sidebar, might change */
        /* background-color: #ffffff; */ /* Let theme handle */
        /* border-right: 1px solid #e0e0e0; */ /* Let theme handle */
      }

      /* Button styling */
      .stButton>button {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        width: 100%;
        margin-bottom: 0.5rem; /* Spacing between buttons */
      }
      .stButton>button:hover {
        filter: brightness(95%); /* Subtle hover effect */
      }

      /* --- FIX for faint placeholder text in dark mode sidebar inputs --- */
      /* Adjust selectors based on browser inspection if needed */
      .stTextArea textarea::placeholder {
         color: #6c757d !important; /* A slightly darker grey */
         opacity: 1;
      }
      div[data-baseweb="select"] > div:first-child > div:first-child {
         /* Target selectbox text */
         /* color: #31333F !important; */ /* Let theme handle text color */
      }
      div[data-baseweb="select"] svg {
         /* Target selectbox dropdown arrow */
         /* fill: #31333F !important; */ /* Let theme handle icon color */
      }
      /* --- End FIX --- */

      /* Tab styling */
      div[role="tablist"] {
        overflow-x: auto; /* Allow horizontal scrolling for many tabs */
        white-space: nowrap; /* Prevent tabs from wrapping */
        border-bottom: 1px solid #e0e0e0; /* Separator line */
        scrollbar-width: thin; /* Firefox scrollbar */
        scrollbar-color: #ccc #f0f2f6; /* Firefox scrollbar color */
      }
      /* Webkit (Chrome, Safari) scrollbar styling */
      div[role="tablist"]::-webkit-scrollbar {
        height: 6px;
      }
      div[role="tablist"]::-webkit-scrollbar-track {
        background: #f0f2f6; /* Scrollbar track color */
      }
      div[role="tablist"]::-webkit-scrollbar-thumb {
        background-color: #ccc; /* Scrollbar handle color */
        border-radius: 10px;
        border: 2px solid #f0f2f6; /* Padding around thumb */
      }

      /* Footer styling */
      footer {
        text-align: center;
        font-size: 0.8em;
        color: #6c757d; /* Grey text */
        margin-top: 2rem;
        padding: 1rem 0;
        border-top: 1px solid #e0e0e0; /* Separator line */
      }
      footer p {
          margin-bottom: 0.25rem; /* Reduce spacing between footer lines */
      }
      footer a {
        color: #007bff; /* Standard link blue */
        text-decoration: none;
      }
      footer a:hover {
        text-decoration: underline;
      }
    </style>
    """

# --- Footer Content ---
# Keep existing Footer Markdown/HTML
FOOTER_MARKDOWN = """
    <footer>
      <p>RadVision AI is for informational/educational use ONLY. Not medical advice.</p>
      <p>
          <a href="YOUR_PRIVACY_POLICY_URL" target="_blank">Privacy</a> |
          <a href="YOUR_TERMS_OF_SERVICE_URL" target="_blank">Terms</a> |
          <a href="https://github.com/mgbam/radvisionai" target="_blank">GitHub</a>
     </p>
    </footer>
    """