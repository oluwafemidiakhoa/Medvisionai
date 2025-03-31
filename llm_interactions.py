# llm_interactions.py

import requests
import base64
import io
import logging
import os # Use os module for environment variables
import streamlit as st
from PIL import Image
# Corrected import: Added Dict, List, Optional, Tuple were already needed
from typing import Optional, Tuple, List, Dict

# Circular import was removed previously.

logger = logging.getLogger(__name__)

# --- Constants ---
IMAGE_MIME_TYPE = "image/png"
MAX_HISTORY_LEN = 5
API_TIMEOUT = 180

# --- Helper ---
@st.cache_data(max_entries=100)
def image_to_base64_str(image_bytes: bytes, format: str = "PNG") -> str:
    """Converts image bytes to a base64 encoded string."""
    img_byte = image_bytes
    return base64.b64encode(img_byte).decode("utf-8")

def get_gemini_api_url() -> Optional[str]:
    """Retrieves Gemini API URL from environment variables."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set.")
        st.error("Configuration Error: GEMINI_API_KEY is not set in the environment variables (or Space Secrets).")
        return None

    # --- Specify the desired Gemini model here ---  <<<<< CORRECTED MODEL NAME
    # model_name = "gemini-1.5-pro-latest" # Generally available Pro model
    model_name = "gemini-2.5-pro-exp-03-25" # Use the requested experimental model
    # -------------------------------------------

    logger.info(f"Targeting Gemini Model: {model_name}") # Log the model being used
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"


# --- Prompt Templates ---
INITIAL_ANALYSIS_PROMPT = """
You are an expert AI assistant simulating a radiologist. Your task is to analyze the provided medical image.
Based *only* on the visual information present in the image, provide the following:
1.  **Detailed Description:** Describe the image content, view (if discernible), and any visible anatomical structures.
2.  **Key Findings:** List any notable abnormalities or significant features observed. Be specific about location, size, and characteristics if possible. If no abnormalities are seen, state "No significant abnormalities detected."
3.  **Potential Differential Diagnoses:** Based *strictly* on the findings, list potential conditions or diagnoses, ordered from most likely to least likely. If findings are normal, indicate "Consistent with normal findings."
4.  **Reasoning for Top Diagnosis:** Briefly explain the visual evidence supporting the most likely diagnosis (if abnormalities were found).
Structure your response clearly using these headings. Do not provide medical advice or treatment recommendations.
"""
QA_CONTEXT_PROMPT_TEMPLATE = """
You are a medical expert AI assisting with the interpretation of a medical image.

**Region of Interest:** {roi_info}
**Conversation History (Most Recent First):**
{history_text}

**Current Question:** "{question}"

Analyze the provided medical image again, considering the conversation history, the highlighted region (if any), and the specific question asked.
Provide a concise, clinically relevant answer based *primarily* on the visual information in the image and the question's context.
If the image does not contain sufficient information to answer the question, clearly state that. Avoid speculation beyond the visual evidence. Do not give medical advice.
"""
CONFIDENCE_PROMPT_TEMPLATE = """
Based on your most recent analysis or answer provided below:
---
Last Question/Task: {last_q}
Your Last Response:
{last_a}
---

Critically evaluate your confidence in the accuracy and completeness of your last response regarding the analysis of the provided medical image.
Respond *only* in the following format:
**Confidence:** [Score]/10
**Justification:** [Provide a brief explanation for your score. Mention factors like image quality (clarity, artifacts, view), conspicuity of findings, ambiguity, or reliance on specific visual features.]

(Score interpretation: 1=Very Low, 5=Moderate, 10=Very High)
"""
DISEASE_SPECIFIC_PROMPT_TEMPLATE = """
You are an expert radiologist AI. Focus *specifically* on evaluating the provided medical image for signs of **{disease}**.

**Region of Interest:** {roi_info}

Address the following points based *only* on the visual evidence:
1.  **Presence/Absence of Findings:** Are there typical visual indicators associated with {disease}? (State clearly: Present, Absent, or Indeterminate).
2.  **Description of Findings:** If present, describe the relevant findings in detail (e.g., location, size, morphology, pattern, distribution).
3.  **Severity Assessment (if applicable):** Based purely on the image, provide a qualitative assessment of severity (e.g., mild, moderate, severe), if possible. If not possible, state so.
4.  **Image-Based Next Steps:** Suggest potential next steps in the diagnostic workup that are directly indicated by *these specific image findings* (e.g., "consider contrast-enhanced scan for better characterization", "comparison with prior images recommended", "further views may be helpful"). Do **not** suggest treatments or give definitive medical advice.

If no signs related to {disease} are visible, state that clearly. Focus on the highlighted region if specified by the user.
"""

# --- Core Gemini Interaction ---
def query_gemini_vision(
    image: Image.Image,
    text_prompt: str,
    ) -> Tuple[Optional[str], bool]:
    """Sends the image and text prompt to the configured Gemini API."""
    gemini_api_url = get_gemini_api_url()
    if not gemini_api_url:
        return "Error: Gemini API URL not configured.", False

    # Log the specific model endpoint being hit (without key)
    logger.info(f"Querying Gemini API endpoint: {gemini_api_url.split('?')[0]}")
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = image_to_base64_str(img_bytes)
    except Exception as e:
        logger.error(f"Failed to encode image to base64: {e}", exc_info=True)
        return f"Error: Failed to process image for API request ({e}).", False

    payload = {
        "contents": [{"parts": [{"text": text_prompt}, {"inline_data": {"mime_type": IMAGE_MIME_TYPE, "data": img_base64}}]}],
        # Adjust generation config if needed for the experimental model
        "generation_config": {"temperature": 0.3, "top_k": 32, "top_p": 0.9, "max_output_tokens": 8192, "stop_sequences": []},
        "safety_settings": [
            {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        ]
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(gemini_api_url, headers=headers, json=payload, timeout=API_TIMEOUT)
        response.raise_for_status()
        response_data = response.json()
        logger.debug(f"Gemini Raw Response: {response_data}")

        # Robust Response Parsing (remains the same logic)
        if 'candidates' in response_data and len(response_data['candidates']) > 0:
            candidate = response_data['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content'] and len(candidate['content']['parts']) > 0:
                if 'text' in candidate['content']['parts'][0]:
                    logger.info("Successfully received text response from Gemini.")
                    return candidate['content']['parts'][0]['text'], True
                else:
                    finish_reason = candidate.get('finishReason', 'UNKNOWN')
                    safety_ratings = candidate.get('safetyRatings', [])
                    if finish_reason == 'SAFETY':
                        blocked_categories = [r['category'] for r in safety_ratings if r.get('probability') not in ['NEGLIGIBLE', 'LOW']]
                        error_msg = f"Error: Gemini response blocked due to safety reasons (Categories: {', '.join(blocked_categories)})."
                        logger.warning(error_msg)
                        return error_msg, False
                    else:
                        error_msg = f"Error: Gemini response structure issue - no 'text' found. Finish Reason: {finish_reason}."
                        logger.error(error_msg)
                        return error_msg, False
            else:
                 finish_reason = candidate.get('finishReason')
                 if finish_reason == 'SAFETY':
                     safety_ratings = candidate.get('safetyRatings', [])
                     blocked_categories = [r['category'] for r in safety_ratings if r.get('probability') not in ['NEGLIGIBLE', 'LOW']]
                     error_msg = f"Error: Content generation stopped (SAFETY). Categories: {', '.join(blocked_categories)}."
                     logger.warning(error_msg)
                     return error_msg, False
                 elif finish_reason:
                     error_msg = f"Error: Gemini finished unexpectedly (Reason: {finish_reason}). Candidate: {candidate}"
                     logger.error(error_msg)
                     return error_msg, False
                 else:
                     error_msg = f"Error: Unexpected Gemini response structure - missing 'content' or 'parts'. Candidate: {candidate}"
                     logger.error(error_msg)
                     return error_msg, False
        elif 'promptFeedback' in response_data:
             block_reason = response_data['promptFeedback'].get('blockReason', 'UNKNOWN')
             safety_ratings = response_data['promptFeedback'].get('safetyRatings', [])
             blocked_categories = [r['category'] for r in safety_ratings if r.get('probability') not in ['NEGLIGIBLE', 'LOW']]
             error_msg = f"Error: Prompt blocked by Gemini API (Reason: {block_reason}, Categories: {', '.join(blocked_categories)})."
             logger.warning(error_msg)
             return error_msg, False
        else:
            error_msg = f"Error: No candidates or feedback found in Gemini response. Response: {response_data}"
            logger.error(error_msg)
            return error_msg, False

    except requests.exceptions.Timeout:
         error_msg = f"Error: Gemini API request timed out after {API_TIMEOUT} seconds."
         logger.error(error_msg)
         return error_msg, False
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else "N/A"
        error_detail = ""
        if e.response is not None:
             try: error_detail = e.response.json().get('error', {}).get('message', e.response.text)
             except: error_detail = e.response.text
        logger.error(f"Gemini API request failed: {e} (Status Code: {status_code})", exc_info=True)
        # Add more specific status code checks if needed
        return f"Error connecting to Gemini API ({status_code}): {error_detail or e}", False
    except Exception as e:
        logger.error(f"Unexpected error processing Gemini response: {e}", exc_info=True)
        return f"Error: Failed to process the response from Gemini ({e}).", False


# --- Functions for Specific Tasks ---
# These functions use the templates defined above and call query_gemini_vision

def run_initial_analysis(image: Image.Image) -> str:
    """Generates the initial analysis using Gemini."""
    logger.info("Running initial analysis...")
    prompt = INITIAL_ANALYSIS_PROMPT
    result_text, success = query_gemini_vision(image, prompt)
    if success:
        return result_text or "[Analysis OK, No Text Returned]"
    else:
        return f"Initial Analysis Failed:\n{result_text}"

def run_multimodal_qa(
    image: Image.Image,
    question: str,
    history: List[Tuple[str, str]],
    roi_coords: Optional[Dict] = None
    ) -> Tuple[str, bool]:
    """Handles QA, potentially using ROI and history."""
    logger.info(f"Received question: {question}")
    roi_info = "No specific region highlighted."
    if roi_coords:
        roi_info = f"User highlighted region: Top-Left({roi_coords['left']},{roi_coords['top']}), Bottom-Right({roi_coords['left']+roi_coords['width']},{roi_coords['top']+roi_coords['height']}). Focus on this region if relevant."
    history_str = "\n---\n".join([f"User: {q}\nAI: {a}" for q, a in history[-MAX_HISTORY_LEN:]]) or "No previous questions."
    prompt = QA_CONTEXT_PROMPT_TEMPLATE.format(history_text=history_str, question=question, roi_info=roi_info)
    gemini_result, gemini_success = query_gemini_vision(image, prompt)
    if gemini_success:
        return gemini_result or "[QA OK, No Text Returned]", True
    else:
        return f"Gemini QA Failed:\n{gemini_result}", False

def run_disease_analysis(
    image: Image.Image,
    disease: str,
    roi_coords: Optional[Dict] = None
    ) -> str:
    """Runs disease-specific analysis."""
    logger.info(f"Running disease analysis for: {disease}")
    roi_info = "No specific region highlighted."
    if roi_coords:
        roi_info = f"User highlighted region: Top-Left({roi_coords['left']},{roi_coords['top']}), Bottom-Right({roi_coords['left']+roi_coords['width']},{roi_coords['top']+roi_coords['height']}). Focus on this region if relevant."
    prompt = DISEASE_SPECIFIC_PROMPT_TEMPLATE.format(disease=disease, roi_info=roi_info)
    result_text, success = query_gemini_vision(image, prompt)
    if success:
        return result_text or "[Analysis OK, No Text Returned]"
    else:
        return f"Disease Analysis Failed ({disease}):\n{result_text}"

def estimate_ai_confidence(
    image: Image.Image,
    history: List[Tuple[str, str]]
    ) -> str:
    """Estimates confidence based on the last interaction."""
    if not history: return "No history available."
    logger.info("Requesting confidence estimation...")
    last_q, last_a = history[-1]
    prompt = CONFIDENCE_PROMPT_TEMPLATE.format(last_q=last_q, last_a=last_a)
    result_text, success = query_gemini_vision(image, prompt)
    if success:
        return result_text or "[Estimation OK, No Text Returned]"
    else:
        return f"Confidence Estimation Failed:\n{result_text}"