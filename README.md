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
RadVision AI Advanced is a cutting-edge, Streamlit-based medical imaging analysis application designed to assist clinicians and researchers with rapid, AI-powered interpretation of both DICOM and standard image formats. The tool integrates advanced image processing, region-of-interest (ROI) selection, and multiple AI services—including language models for analysis, Q&A, and disease-specific evaluations—to generate detailed reports and insights on medical images.

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

Overview
RadVision AI Advanced leverages state-of-the-art AI models to process and analyze medical images. The application supports:

DICOM and Standard Images: Automatically detects and processes both DICOM and common image file formats (JPG, PNG).

ROI Selection: Users can draw on images to define regions of interest using an integrated drawable canvas.

Multi-Modal AI Analysis: Provides initial analyses, Q&A interactions, disease-specific evaluations, and confidence estimations.

PDF Report Generation: Summarizes analysis results in a downloadable PDF report.

The app is designed for research and educational purposes, and its outputs should be verified by clinical experts.

Features
Image Processing:

DICOM parsing and metadata extraction.

Window/Level adjustment with interactive sliders.

Standard image processing using the Python Imaging Library (PIL).

AI Integration:

Initial analysis to describe and interpret the image.

Q&A interface for detailed inquiries.

Condition/disease-specific analysis.

Confidence estimation on AI outputs.

Fallback mechanisms using Hugging Face’s VQA models when primary methods fail.

Reporting:

Generation of PDF reports that include embedded images, session IDs, and formatted analysis results.

User Interface:

Streamlit-based UI with a clean two-column layout:

Left: Image viewer and ROI selection.

Right: Analysis results and interactive controls.

File Structure
graphql
Copy
├── app.py                  # Main Streamlit application entry point.
├── dicom_utils.py          # Functions to parse DICOM files, extract metadata, and convert images.
├── hf_models.py            # Integration with Hugging Face Inference API for VQA fallback.
├── llm_interactions.py     # Functions to interact with Gemini (and other LLMs) for analysis, Q&A, and more.
├── report_utils.py         # Functions to generate PDF reports summarizing the session's analysis.
├── ui_helpers.py           # Helper functions for the UI, including metadata display and window/level sliders.
├── requirements.txt        # List of Python dependencies.
└── README.md               # Project documentation.
app.py:
Initializes the Streamlit interface, processes uploads, integrates all helper modules, and controls the overall workflow.

dicom_utils.py:
Contains functions for DICOM file parsing, metadata extraction, image conversion, and window/level handling.

hf_models.py:
Handles querying external VQA models (e.g., from Hugging Face) as a fallback for multimodal analysis.

llm_interactions.py:
Provides functions that interact with language models (like Gemini) to generate initial analyses, answer questions, run disease-specific evaluations, and estimate AI confidence.

report_utils.py:
Generates PDF reports summarizing the analysis session, including embedded images and formatted text.

ui_helpers.py:
Contains UI-related helper functions such as displaying DICOM metadata and creating interactive window/level sliders.

Installation
Clone the Repository:

bash
Copy
git clone https://github.com/yourusername/radvision-ai-advanced.git
cd radvision-ai-advanced
Create a Virtual Environment (Optional but Recommended):

bash
Copy
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Install Dependencies:

bash
Copy
pip install -r requirements.txt
Note: Ensure you have the required libraries such as Streamlit, Pillow, pydicom, fpdf2, and requests installed.

Configuration
Before running the application, configure the following environment variables or add them to a secrets.toml file for deployment:

HF_API_TOKEN:
Your Hugging Face API token for accessing VQA models.

GEMINI_API_KEY:
API key for the Gemini language model service.

GEMINI_MODEL_OVERRIDE (Optional):
To override the default Gemini model name (e.g., "gemini-2.5-pro-exp-03-25").

For local testing with Streamlit, you can add these variables to a .env file or configure them in your terminal session.

Running the Application
To start the application locally, run:

bash
Copy
streamlit run app.py
The app will open in your default browser. You can then upload images, adjust DICOM window/level settings, run various AI analyses, and generate PDF reports.

Usage Guide
Upload an Image:
Use the sidebar to upload a JPG, PNG, or DICOM file.

Adjust DICOM Settings:
If a DICOM image is detected, adjust the window center and width using the sliders.

Run AI Analysis:
Click the appropriate action buttons (e.g., "Run Initial Analysis", "Ask AI", "Run Condition Analysis") in the sidebar. You can also draw on the image to define a region of interest (ROI).

View Results:
Analysis results, Q&A responses, disease-specific insights, and confidence estimations will appear in the two-column layout.

Generate a Report:
Use the "Generate PDF Data" button to create a downloadable report summarizing your session.

Contributing
Contributions are welcome! Feel free to submit pull requests or open issues with suggestions, bug reports, or feature requests. Please adhere to standard coding practices and document your changes accordingly.

License
This project is open source and available under the MIT License.

Disclaimer: This tool is intended for research and informational purposes only. Always consult a qualified healthcare professional for clinical interpretations and decisions.


Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference