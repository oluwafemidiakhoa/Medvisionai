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
RadVision AI Advanced is a cutting‑edge, Streamlit‑based medical imaging analysis application designed to assist clinicians and researchers with rapid, AI‑powered interpretation of both DICOM and standard image formats. The tool integrates advanced image processing, region-of‑interest (ROI) selection, and multiple AI services—including language models for analysis, Q&A, and disease‑specific evaluations—to generate detailed reports and insights on medical images.

Table of Contents
Overview

Features

File Structure

Installation

Configuration

Running the Application

Usage Guide

Contributing

License

Configuration Reference

Overview
RadVision AI Advanced leverages state‑of‑the‑art AI models to process and analyze medical images. The application supports both DICOM files and common image formats (JPG, PNG) and provides a user‑friendly, interactive interface. Key capabilities include:

Multi‑Format Image Processing: Automatic detection and handling of DICOM images as well as standard image formats.

ROI Selection: Users can draw regions of interest on images using an integrated drawable canvas.

Multi‑Modal AI Analysis: Provides initial analyses, interactive Q&A sessions, disease‑specific evaluations, and confidence estimations.

PDF Report Generation: Summarizes analysis outputs in a downloadable PDF report.

Advanced Translation Functionality: Uses the deep‑translator library with a Google Translate backend to detect and translate analysis text into multiple languages, preserving the original formatting (bullet points, numbering, spacing).

Note: This application is intended for research and educational use only. Always verify results with clinical experts.

Features
Image Processing
DICOM Support: Parse DICOM files and extract metadata.

Window/Level Adjustment: Interactive sliders to optimize image visualization.

Standard Image Processing: Utilizes the Python Imaging Library (PIL) for common image formats.

AI Integration
Initial Analysis: Automated interpretation of the uploaded image.

Q&A Interface: Enables users to ask questions about the image with region-of‑interest support.

Disease‑Specific Evaluation: Focused analysis for conditions such as pneumonia, tuberculosis, etc.

Confidence Estimation: Provides an AI‑generated confidence score for the analysis.

Fallback Mechanisms: Uses external models (e.g., Hugging Face VQA APIs) when primary methods fail.

Translation & Language Detection
Translation Module: Implements translation using the deep‑translator library (Google Translate backend) with robust dependency checks and workarounds for known issues.

Language Detection: Detects the language of provided text snippets before translation.

Formatting Preservation: Uses a few‑shot prompt with examples to ensure bullet points, numbering, and spacing are preserved in the translation.

Reporting
PDF Report Generation: Generates downloadable PDF reports that include embedded images, session IDs, and formatted text summaries.

User Interface
Streamlit‑Based Layout: Clean two‑column design.

Left Panel: Image viewer with ROI selection and DICOM metadata.

Right Panel: Analysis results, Q&A history, disease evaluation, confidence estimation, and translation features.

File Structure
pgsql
Copy code
├── app.py                  # Main Streamlit application entry point.
├── dicom_utils.py          # DICOM parsing, metadata extraction, and image conversion functions.
├── hf_models.py            # Integration with external VQA models (e.g., Hugging Face) as a fallback.
├── llm_interactions.py     # Functions for interfacing with language models for analysis and Q&A.
├── report_utils.py         # Functions to generate PDF reports for analysis sessions.
├── ui_helpers.py           # Helper functions for UI elements (e.g., metadata display, window/level sliders).
├── translation_models.py   # Translation and language detection using deep‑translator (Google Translate backend).
├── requirements.txt        # List of Python dependencies.
└── README.md               # Project documentation.
app.py: Initializes the Streamlit interface, processes image uploads, integrates all modules, and controls the overall workflow.

dicom_utils.py: Handles DICOM file parsing, metadata extraction, image conversion, and window/level adjustments.

hf_models.py: Provides integration with external VQA models for fallback in multimodal analysis.

llm_interactions.py: Contains functions for communicating with large language models for initial analysis, Q&A, and confidence scoring.

report_utils.py: Creates PDF reports summarizing the analysis session.

ui_helpers.py: Contains functions for UI enhancements like metadata display and interactive sliders.

translation_models.py: Implements translation and language detection using the deep‑translator library.

Dependency Handling: Attempts to import deep‑translator and gracefully degrades translation features if unavailable.

Workarounds: Applies a workaround for known issues with certain exceptions.

Installation
1. Clone the Repository
bash
Copy code
git clone https://github.com/yourusername/radvision-ai-advanced.git
cd radvision-ai-advanced
2. Create a Virtual Environment (Optional but Recommended)
bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
bash
Copy code
pip install -r requirements.txt
Ensure you have the required libraries such as Streamlit, Pillow, pydicom, deep-translator, fpdf2, and transformers installed.

Configuration
Before running the application, configure the following environment variables or add them to a secrets.toml file:

HF_API_TOKEN: Your Hugging Face API token for VQA fallback.

GEMINI_API_KEY: API key for the Gemini language model service.

GEMINI_MODEL_OVERRIDE (Optional): Override for the default Gemini model name (e.g., "gemini-2.5-pro-exp-03-25").

For local testing, these variables can be added to a .env file or set in your terminal session.

Running the Application
To start the application locally, run:

bash
Copy code
streamlit run app.py
The app will open in your default browser. From there, you can upload images, adjust DICOM settings, perform AI analysis, access translation features, and generate PDF reports.

Usage Guide
Upload an Image
Use the sidebar to upload a JPG, PNG, or DICOM file.

Adjust DICOM Settings
For DICOM images, use interactive window/level sliders to optimize visualization.

Run AI Analysis
Click the action buttons (e.g., "Run Initial Analysis", "Ask AI", "Run Condition Analysis") in the sidebar. Optionally, draw an ROI on the image.

Translation Functionality
In the Translation tab, select the text to translate (e.g., “Initial Analysis”).

Choose "Auto‑Detect" for the source language (or select a language manually) and choose a target language.

The system uses deep‑translator to detect the source language and then translates the text. The few‑shot prompt provided in the app helps preserve formatting such as bullet points and numbering.

View Analysis Results
The right panel displays analysis results—including initial analysis, Q&A history, condition evaluation, confidence scores, and translations—in a clean, tabbed layout.

Generate a Report
Use the "Generate PDF Data" button to create a downloadable PDF report summarizing your session.

Contributing
Contributions are welcome! Please submit pull requests or open issues for bug fixes, improvements, or new features. Follow standard coding practices and document your changes.

License
This project is open source and available under the MIT License.

Configuration Reference
For advanced configuration options for Hugging Face Spaces and similar deployment scenarios, please refer to the Hugging Face Spaces configuration reference.

Disclaimer
This tool is intended for research and informational purposes only. The AI outputs should be verified by clinical experts, and it is not intended for clinical decision-making without professional validation.