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
# RadVision AI Advanced

RadVision AI Advanced is a cutting‑edge, Streamlit‑based medical imaging analysis application designed to assist clinicians and researchers with rapid, AI‑powered interpretation of both DICOM and standard image formats. The tool integrates advanced image processing, region‑of‑interest (ROI) selection, and multiple AI services—including language models for analysis, Q&A, and disease‑specific evaluations—to generate detailed reports and insights on medical images.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Usage Guide](#usage-guide)
- [Contributing](#contributing)
- [License](#license)
- [Configuration Reference](#configuration-reference)

## Overview

RadVision AI Advanced leverages state‑of‑the‑art AI models to process and analyze medical images. The application supports both DICOM files and common image formats (JPG, PNG) and provides a user‑friendly, interactive interface. Key capabilities include:

- **Multi‑Format Image Processing:** Automatic detection and handling of DICOM images as well as standard image formats.
- **ROI Selection:** Users can draw regions of interest on images using an integrated drawable canvas.
- **Multi‑Modal AI Analysis:** Provides initial analyses, interactive Q&A sessions, disease‑specific evaluations, and confidence estimations.
- **PDF Report Generation:** Summarizes analysis outputs in a downloadable PDF report.
- **Advanced Translation Functionality:** Uses the deep‑translator library with a Google Translate backend to detect and translate analysis text into multiple languages, preserving the original formatting (bullet points, numbering, spacing).

> **Note:** This application is intended for research and educational use only. Always verify results with clinical experts.

## Features

### Image Processing

- **DICOM Support:** Parse DICOM files and extract metadata.
- **Window/Level Adjustment:** Interactive sliders to optimize image visualization.
- **Standard Image Processing:** Utilizes the Python Imaging Library (PIL) for common image formats.

### AI Integration

- **Initial Analysis:** Automated interpretation of the uploaded image.
- **Q&A Interface:** Enables users to ask questions about the image with region‑of‑interest support.
- **Disease‑Specific Evaluation:** Focused analysis for conditions such as pneumonia, tuberculosis, etc.
- **Confidence Estimation:** Provides an AI‑generated confidence score for the analysis.
- **Fallback Mechanisms:** Uses external models (e.g., Hugging Face VQA APIs) when primary methods fail.

### Translation & Language Detection

- **Translation Module:** Implements translation using the deep‑translator library (Google Translate backend) with robust dependency checks and workarounds for known issues.
- **Language Detection:** Detects the language of provided text snippets before translation.
- **Formatting Preservation:** Uses a few‑shot prompt with examples to ensure bullet points, numbering, and spacing are preserved in the translation.

### Reporting

- **PDF Report Generation:** Generates downloadable PDF reports that include embedded images, session IDs, and formatted text summaries.

### User Interface

- **Streamlit‑Based Layout:** Clean two‑column design.
  - **Left Panel:** Image viewer with ROI selection and DICOM metadata.
  - **Right Panel:** Analysis results, Q&A history, condition evaluation, confidence estimation, translation features, and UMLS concept display.

## File Structure

```plaintext
radvisionai/
├── app.py                  # Main Streamlit application entry point.
├── dicom_utils.py          # DICOM parsing, metadata extraction, and image conversion functions.
├── hf_models.py            # Integration with external VQA models (e.g., Hugging Face) as a fallback.
├── llm_interactions.py     # Functions for interfacing with language models for analysis and Q&A, with UMLS mapping.
├── report_utils.py         # Functions to generate PDF reports for analysis sessions, including UMLS concepts.
├── ui_components.py        # Helper functions for UI elements (e.g., metadata display, window/level sliders, UMLS concept panel).
├── translation_models.py   # Translation and language detection using deep‑translator (Google Translate backend).
├── umls_utils.py           # Utilities for UMLS API integration and concept retrieval.
├── requirements.txt        # List of Python dependencies.
└── README.md               # Project documentation.
```

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/radvision-ai-advanced.git
   cd radvision-ai-advanced
   ```
2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

Ensure you have the required libraries such as Streamlit, Pillow, pydicom, deep-translator, fpdf2, transformers, scispacy, and requests installed.

## Configuration

Before running the application, configure the following environment variables or add them to a `secrets.toml` / `.env` file:

- `HF_API_TOKEN`  
  Your Hugging Face API token for VQA fallback.

- `GEMINI_API_KEY`  
  API key for the Gemini language‑model service.

- `GEMINI_MODEL_OVERRIDE` *(Optional)*  
  Override for the default Gemini model name (e.g., `"gemini-2.5-pro-exp-03-25"`).

### UMLS Integration *(New)*

To enable standardized terminology mapping in both the Streamlit UI and generated PDF reports:

- `UMLS_APIKEY`  
  Your UMLS REST API key (from https://uts.nlm.nih.gov/).

- `UMLS_HITS` *(Optional)*  
  Number of top UMLS concepts to retrieve per response. Defaults to `3`.

- `UMLS_SOURCE_FILTER` *(Optional)*  
  Comma‑separated list of preferred source vocabularies (e.g., `SNOMEDCT_US,ICD10CM`). If empty, all sources are allowed.

**Example `.env`:**
```dotenv
HF_API_TOKEN=hf_xxx
GEMINI_API_KEY=ya29.xxx
UMLS_APIKEY=YOUR_UMLS_KEY
UMLS_HITS=5
UMLS_SOURCE_FILTER=SNOMEDCT_US,ICD10CM
```

Once set, every AI output will include a “Standardized UMLS Concepts” panel in the app **and** a corresponding section in the PDF report.

## Running the Application

To start the application locally:

```bash
streamlit run app.py
```

The app will open in your default browser. From there, you can upload images, adjust DICOM settings, perform AI analysis, access translation features, view UMLS concepts, and generate PDF reports.

## Usage Guide

### Upload an Image

Use the sidebar to upload a JPG, PNG, or DICOM file.

### Adjust DICOM Settings

For DICOM images, use interactive window/level sliders to optimize visualization.

### Run AI Analysis

Click the action buttons (e.g., “Run Initial Analysis”, “Ask AI”, “Run Condition Analysis”) in the sidebar. Optionally, draw an ROI on the image.

### View UMLS Concepts

In the analysis output, expand the “Standardized UMLS Concepts” panel to see mapped concepts with clickable links.

### Translation Functionality

In the Translation tab, select the text to translate (e.g., “Initial Analysis”).

Choose “Auto‑Detect” for the source language (or select a language manually) and choose a target language.

### Generate a Report

Use the “Generate PDF Report” button to create a downloadable PDF summarizing your session, including UMLS mappings.

## Contributing

Contributions are welcome! Please submit pull requests or open issues for bug fixes, improvements, or new features. Follow standard coding practices and document your changes.

## License

This project is open source and available under the MIT License.

## Configuration Reference

For advanced configuration options for Hugging Face Spaces and similar deployment scenarios, please refer to the Hugging Face Spaces configuration reference.

