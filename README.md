---
title: MedVisionAI
emoji: 👀
colorFrom: blue
colorTo: pink
sdk: streamlit
sdk_version: 1.44.1
app_file: app.py
pinned: false
short_description: Medical Image diagnostic
license: mit
---
RadVision AI Advanced

RadVision AI Advanced is a Streamlit-based medical imaging analysis application designed to assist clinicians and researchers with rapid, AI-powered interpretation of DICOM and standard image formats. With integrated multi-modal AI analysis, translation, and standardized terminology mapping via UMLS, this tool streamlines diagnostics and reporting workflows.

Table of Contents

Overview

Features

File Structure

Installation

Configuration

Running the Application

Usage Guide

UMLS Feature Enhancements

Contributing

License

Overview

RadVision AI Advanced leverages state-of-the-art AI models and clinical knowledge sources to:

Process medical images (DICOM, PNG, JPG)

Perform initial and condition-specific analyses

Enable interactive Q&A with context

Estimate model confidence

Translate findings

Map key terms to standardized medical concepts (UMLS, SNOMED CT, ICD-10)

Generate comprehensive PDF reports

Disclaimer: For research and educational use only. Always verify AI outputs with qualified clinical experts.

Features

Multi-Format Image Processing: Automatic support for DICOM metadata, window/level adjustment, and common image formats.

Drawable ROI Canvas: Focus AI analysis on user-selected regions of interest.

Initial Analysis: Automated, structured interpretation of image findings.

Contextual Q&A: Ask follow-up questions with full conversation history.

Condition-Focused Evaluation: Disease-specific analyses (e.g., pneumonia, tuberculosis).

AI Confidence Estimation: Numerical and qualitative confidence scoring.

Translation Module: Detect and translate AI text outputs while preserving formatting.

PDF Report Generation: Downloadable reports with embedded images, analyses, UMLS mappings, and medical codes.

UMLS Lookup & Auto-Enrichment: Standardized terminology support and interactive concept exploration.

File Structure

radvisionai/
├── app.py                  # Main Streamlit application
├── action_handlers.py      # Handles mapping of UI actions to backend functions
├── config.py               # Central config (API keys, model names, UMLS settings)
├── dicom_utils.py          # DICOM parsing & image conversion
├── file_processing.py      # File hashing & caching utilities
├── hf_models.py            # Hugging Face VQA fallback integration
├── llm_interactions.py     # LLM prompts, UMLS mapping, analysis functions
├── main_page_ui.py         # Refactored main UI components for Streamlit
├── report_utils.py         # PDF report generation with UMLS/SNOMED/ICD sections
├── requirements.txt        # Python dependencies
├── session_state.py        # Streamlit session-state initialization
├── sidebar_ui.py           # Refactored sidebar controls (upload, ROI, actions)
├── test_translation.py     # Unit tests for translation features
├── translation_models.py   # deep-translator integration
├── ui_components.py        # Streamlit UI helpers & UMLS concept display
└── umls_utils.py           # UMLS REST API client & concept dataclass

Installation

Clone the repository



git clone https://github.com/yourusername/radvision-ai-advanced.git
cd radvision-ai-advanced

2. **(Optional) Create a virtual environment**
   ```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

Install dependencies



pip install -r requirements.txt


Ensure these core packages are installed:
- streamlit >= 1.44.1
- Pillow
- pydicom
- deep-translator
- fpdf2
- requests
- spacy, scispacy (for advanced UMLS filtering)

## Configuration

Before running the app, set environment variables (or add them to `.env` / `secrets.toml`):

```dotenv
HF_API_TOKEN=<your Hugging Face token>
GEMINI_API_KEY=<your Google Gemini API key>
GEMINI_MODEL_OVERRIDE=<optional model override>

# UMLS Integration
UMLS_APIKEY=<your UMLS REST API key>
UMLS_HITS=5                # Number of concepts to fetch
UMLS_SOURCE_FILTER=SNOMEDCT_US,ICD10CM  # Comma-separated filter (optional)

HF_API_TOKEN: For Hugging Face VQA fallback.

GEMINI_API_KEY: For primary LLM vision calls.

UMLS_APIKEY: To enable standardized terminology mapping.

UMLS_HITS: Controls number of UMLS concepts per result.

UMLS_SOURCE_FILTER: Restrict to specific vocabularies.

Running the Application

Launch the Streamlit app:

streamlit run app.py

The application will open in your default browser.

Usage Guide

Upload an Image: DICOM (.dcm) or standard (JPG/PNG).

Adjust DICOM Window/Level: If applicable.

Draw ROI: Use the canvas to focus AI on a region.

AI Analysis: Use sidebar actions:

▶️ Run Initial Analysis

💬 Ask Question

🩺 Analyze Condition

📈 Estimate AI Confidence

📄 Generate PDF Report

Translation: Switch to the Translation tab, select text, choose languages, and translate.

Review UMLS Concepts:

Auto-Enrichment: Under each AI results tab, view standardized concepts.

Lookup Tab: Ad-hoc UMLS searches in the 🧬 UMLS Lookup tab.

Download Report: PDF with embedded images, analyses, UMLS concepts, SNOMED CT, and ICD-10 codes.

UMLS Feature Enhancements

Feature

Description

🔍 UMLS Concept Search Bar

Interactive tab: search terms (e.g., “COPD”) → view definitions, synonyms, SNOMED CT/ICD-10 mappings, RxNorm.

💡 Auto-Enrich AI Outputs

AI responses auto-annotated with clickable UMLS concept links.

🧠 Synonym Expansion for Q&A

Synonym detection maps user queries to same UMLS concept for robust Q&A.

📚 Clinical Guidelines Integration

From UMLS concept, link out to PubMed/guidelines pages for treatment recommendations.

🏷️ Smart PDF Report

Annotated PDF: lists SNOMED CT and ICD-10 codes for each detected concept.

Tools & APIs

UMLS REST API (NLM): Metathesaurus searches & concept metadata.

QuickUMLS (optional): Local fuzzy matching for speed.

Metathesaurus: Vocabularies integration.

RxNorm API: Map drug-related concepts to RxNorm codes.

Contributing

Contributions welcome! Please submit issues and pull requests. Follow project style, include tests, and update documentation.

License

MIT License. See LICENSE for details.

"# Medvisionai" 
