
# RadVision AI: Advanced Medical Imaging Analysis Platform

![RadVision AI](assets/radvisionai-hero.jpeg)

## Overview

RadVision AI is an advanced Streamlit-based medical imaging analysis platform designed to assist healthcare professionals with AI-powered interpretation of medical images. The system leverages multi-modal AI analysis, translation capabilities, and standardized medical terminology mapping via UMLS to streamline diagnostic workflows and improve clinical decision-making.

## Key Features

### 🖼️ Multi-Format Image Processing
- Support for DICOM medical imaging format with metadata extraction
- Automatic window/level adjustment for optimal viewing
- Compatible with common image formats (PNG, JPG, JPEG)

### 🎯 Region of Interest (ROI) Analysis
- Interactive ROI selection via drawable canvas
- Focus AI analysis on specific regions of concern
- Enhances precision and relevance of analysis results

### 🔬 Comprehensive AI Analysis
- **Initial Analysis**: Automated, structured interpretation of image findings
- **Contextual Q&A**: Ask follow-up questions with full conversation history
- **Condition-Focused Evaluation**: Disease-specific analyses (e.g., pneumonia, tuberculosis)
- **AI Confidence Estimation**: Numerical and qualitative confidence scoring

### 🧠 Smart Recommendations
- Context-aware clinical recommendations based on findings
- Categorized suggestions for imaging, clinical follow-up, and treatment
- Evidence-based recommendations aligned with clinical guidelines

### 🌐 Language Translation
- Detect and translate AI text outputs while preserving formatting
- Support for multiple languages to improve global accessibility
- Seamless integration within the analysis workflow

### 📄 PDF Report Generation
- Comprehensive, professional clinical reports
- Embedded images with annotations and findings
- UMLS concept mapping and medical codes (SNOMED CT, ICD-10)
- Smart recommendations section for clinical decision support

### 🧬 UMLS Integration & Medical Coding
- Automatic mapping to Unified Medical Language System (UMLS) concepts
- Interactive exploration of medical terminology relationships
- Standardized medical codes for improved interoperability

### 📊 Evaluation & Quality Assurance
- Comprehensive tools for measuring AI performance
- Accuracy testing against gold standard datasets
- Compliance checking for healthcare standards

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RadVision AI Platform                    │
├─────────────┬─────────────┬────────────────┬────────────────────┤
│             │             │                │                    │
│  Image      │  Analysis   │  Medical       │  Report            │
│  Processing │  Pipeline   │  Knowledge     │  Generation        │
│  Module     │             │  Integration   │                    │
│             │             │                │                    │
├─────────────┼─────────────┼────────────────┼────────────────────┤
│ - DICOM     │ - Initial   │ - UMLS         │ - PDF              │
│   Parser    │   Analysis  │   Concept      │   Generation       │
│ - Image     │ - Q&A       │   Lookup       │ - Smart            │
│   Processing│   Engine    │ - Medical      │   Recommendations  │
│ - ROI       │ - Disease   │   Coding       │ - Standardized     │
│   Selection │   Analysis  │   (SNOMED,     │   Formatting       │
│             │ - Confidence│    ICD-10)     │                    │
│             │   Estimation│                │                    │
└─────────────┴─────────────┴────────────────┴────────────────────┘
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Required Python packages (installed automatically via requirements.txt)

### Installation

1. Clone the repository or fork the Repl
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   - Set up Google Cloud credentials for Gemini AI integration
   - Configure UMLS API access for terminology mapping

4. Launch the application:
   ```
   streamlit run app.py
   ```

5. Default authentication credentials:
   - Username: `admin`
   - Password: `radvision`

## Usage Guide

### Uploading Images
- Use the file uploader to select DICOM (.dcm) or standard image formats (JPG/PNG)
- Enable demo mode to use sample images for testing

### Analysis Workflow
1. **View uploaded image** in the image viewer panel
2. Optionally **draw a Region of Interest (ROI)** to focus analysis
3. **Run Initial Analysis** to get AI-generated findings
4. **Ask follow-up questions** to explore specific aspects
5. **Analyze for specific conditions** using the condition selector
6. **Estimate AI confidence** to understand reliability

### Generated Reports
- Create comprehensive PDF reports with the "Generate Report" button
- View standardized medical codes (UMLS, SNOMED CT, ICD-10)
- Access Smart Recommendations based on image findings

## Evaluation Tools

RadVision AI includes a suite of evaluation tools accessible via:
```
streamlit run evaluation_suite.py
```

The evaluation suite provides:
1. **AI Accuracy Testing**: Measurement against gold standard datasets
2. **Performance Benchmarking**: Analysis of load times and processing speed
3. **Compliance Checking**: Verification of healthcare standards adherence
4. **Usability Testing**: Collection and analysis of user feedback

## Smart Recommendations Testing

Test the Smart Recommendations feature independently:
```
streamlit run smart_recommendations_ui.py
```

## Google Sheets Integration

RadVision AI supports saving analysis results to Google Sheets:
1. Configure your Google Cloud service account
2. Save analysis data for long-term tracking
3. Enable collaborative workflows

## Case Studies

| Case | Image Type | Key Findings | AI Confidence | Clinical Impact |
|------|------------|--------------|---------------|-----------------|
| Pneumonia | Chest X-ray | Bilateral infiltrates, Consolidation | 92% | Early detection and treatment |
| Tuberculosis | Chest CT | Cavitary lesions, Upper lobe involvement | 89% | Infection control measures |
| Bone Fracture | X-ray | Distal radius fracture, No displacement | 95% | Appropriate treatment planning |
| Brain Tumor | MRI | 2.3 cm mass, Perilesional edema | 87% | Surgical planning assistance |
| COVID-19 | Chest X-ray | Ground-glass opacities, Peripheral distribution | 91% | Rapid triage and isolation |

## Compliance Information

- For research and educational use only
- Not FDA approved for clinical diagnostics
- See `fda_requirements.md` for compliance information

## Disclaimer

RadVision AI is for informational purposes only and is not a substitute for professional medical evaluation. All AI analyses should be verified by qualified healthcare providers.

## License

MIT License. See LICENSE for details.
