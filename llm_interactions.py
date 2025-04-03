import requests
import base64
import io
import logging
import os
import streamlit as st
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any

logger = logging.getLogger(__name__)

# --- Constants ---
IMAGE_MIME_TYPE: str = "image/png"  # Use PNG for high-quality medical images
MAX_HISTORY_LEN: int = 5            # Limit conversation history sent to LLM
API_TIMEOUT: int = 180              # Seconds to wait for API response
DEFAULT_MODEL_NAME: str = "gemini-1.5-pro-latest"  # Default stable model name
GEMINI_MODEL_NAME: str = os.environ.get("GEMINI_MODEL_OVERRIDE", "gemini-2.5-pro-exp-03-25")  # Experimental model override

# --- Helper Functions ---

@st.cache_data(max_entries=50, show_spinner=False)
def _image_to_base64_str(image_bytes: bytes, image_format: str = "PNG") -> str:
    """
    Converts image bytes to a base64 encoded string.

    Args:
        image_bytes: Raw bytes of the image.
        image_format: The format of the image (e.g., 'PNG', 'JPEG').

    Returns:
        The base64 encoded string.
    """
    if not image_bytes:
        logger.error("No image bytes provided for encoding.")
        return ""
    base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
    logger.debug(f"Encoded image ({len(image_bytes)} bytes) to base64 string of length {len(base64_encoded)}.")
    return base64_encoded

def get_gemini_api_url_and_key() -> Tuple[Optional[str], Optional[str]]:
    """
    Retrieves the Gemini API Key and constructs the API URL.

    Returns:
        A tuple containing (api_url, api_key). Both can be None if the key is missing.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set.")
        return None, None
    logger.debug(f"Using Gemini model: {GEMINI_MODEL_NAME}")
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"
    return api_url, api_key

# --- Prompt Templates ---
INITIAL_ANALYSIS_PROMPT = """
You are an expert AI assistant simulating a radiologist. Your task is to analyze the provided medical image.
Based *only* on the visual information present in the image, provide the following:
1. **Detailed Description:** Describe the image content, view (if discernible), and any visible anatomical structures.
2. **Key Findings:** List any notable abnormalities or significant features observed. Be specific about location, size, and characteristics if possible. If no abnormalities are seen, state "No significant abnormalities detected."
3. **Potential Differential Diagnoses:** Based *strictly* on the findings, list potential conditions or diagnoses, ordered from most likely to least likely. If findings are normal, indicate "Consistent with normal findings."
4. **Reasoning for Top Diagnosis:** Briefly explain the visual evidence supporting the most likely diagnosis (if abnormalities were found).
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
1. **Presence/Absence of Findings:** Are there typical visual indicators associated with {disease}? (State clearly: Present, Absent, or Indeterminate).
2. **Description of Findings:** If present, describe the relevant findings in detail (e.g., location, size, morphology, pattern, distribution).
3. **Severity Assessment (if applicable):** Based purely on the image, provide a qualitative assessment of severity (e.g., mild, moderate, severe), if possible. If not possible, state so.
4. **Image-Based Next Steps:** Suggest potential next steps in the diagnostic workup that are directly indicated by *these specific image findings* (e.g., "consider contrast-enhanced scan for better characterization", "comparison with prior images recommended", "further views may be helpful"). Do **not** suggest treatments or give definitive medical advice.

If no signs related to {disease} are visible, state that clearly. Focus on the highlighted region if specified by the user.
"""

# --- Core Gemini Interaction ---
def query_gemini_vision(
    image: Image.Image,
    text_prompt: str,
) -> Tuple[Optional[str], bool]:
    """
    Sends the image and text prompt to the Gemini API endpoint.

    Handles API key retrieval, image encoding, payload construction,
    API call, and response parsing with robust error checks.

    Args:
        image: The PIL Image object to send.
        text_prompt: The text prompt to accompany the image.

    Returns:
        A tuple containing:
            - The generated text response or an error message.
            - A boolean indicating success (True) or failure (False).
    """
    gemini_api_url, api_key = get_gemini_api_url_and_key()
    if not gemini_api_url or not api_key:
        st.error("Configuration Error: GEMINI_API_KEY is not set.")
        return "Error: Gemini API key not configured.", False

    logger.info(f"Querying Gemini endpoint: {gemini_api_url} (Model: {GEMINI_MODEL_NAME})")

    # --- Prepare Image ---
    try:
        buffered = io.BytesIO()
        image_format = "PNG"
        image.save(buffered, format=image_format)
        img_bytes = buffered.getvalue()
        img_base64 = _image_to_base64_str(img_bytes, image_format)
        logger.debug(f"Prepared image for Gemini API (Base64 length: {len(img_base64)}).")
    except Exception as e:
        logger.error(f"Failed to process image for Gemini API: {e}", exc_info=True)
        return f"Error: Failed to process image for API request ({e}).", False

    # --- Construct Payload ---
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": text_prompt},
                    {"inline_data": {"mime_type": IMAGE_MIME_TYPE, "data": img_base64}}
                ]
            }
        ],
        "generation_config": {
            "temperature": 0.3,
            "top_k": 32,
            "top_p": 0.9,
            "max_output_tokens": 8192,
            "stop_sequences": []
        },
        "safety_settings": [
            {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            for c in [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT"
            ]
        ]
    }
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }
    logger.debug(f"Sending Gemini API request. Payload keys: {list(payload.keys())}.")

    # --- Make API Call ---
    try:
        response = requests.post(gemini_api_url, headers=headers, json=payload, timeout=API_TIMEOUT)
        response.raise_for_status()
        response_data: Dict[str, Any] = response.json()
        logger.debug(f"Gemini Raw Response JSON: {response_data}")

        # --- Parse Response ---
        if 'candidates' in response_data and isinstance(response_data['candidates'], list) and response_data['candidates']:
            candidate = response_data['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content'] and candidate['content']['parts']:
                if 'text' in candidate['content']['parts'][0]:
                    parsed_text = candidate['content']['parts'][0]['text']
                    logger.info("Successfully parsed text response from Gemini.")
                    return parsed_text, True
                else:
                    finish_reason = candidate.get('finishReason', 'UNKNOWN')
                    safety_ratings = candidate.get('safetyRatings', [])
                    logger.warning(f"Response missing 'text'. Finish Reason: {finish_reason}. Safety Ratings: {safety_ratings}")
                    if finish_reason == 'SAFETY':
                        blocked_categories = [r.get('category', 'UNKNOWN') for r in safety_ratings if r.get('probability') not in ['NEGLIGIBLE', 'LOW']]
                        error_msg = f"Error: Response blocked due to safety filters (Categories: {', '.join(blocked_categories) or 'N/A'})."
                    else:
                        error_msg = f"Error: Expected 'text' not found. Finish Reason: {finish_reason}."
                    return error_msg, False
            else:
                finish_reason = candidate.get('finishReason', 'UNKNOWN')
                safety_ratings = candidate.get('safetyRatings', [])
                logger.warning(f"Candidate missing valid 'content' or 'parts'. Finish Reason: {finish_reason}. Safety Ratings: {safety_ratings}")
                if finish_reason == 'SAFETY':
                    blocked_categories = [r.get('category', 'UNKNOWN') for r in safety_ratings if r.get('probability') not in ['NEGLIGIBLE', 'LOW']]
                    error_msg = f"Error: Generation halted due to safety filters (Categories: {', '.join(blocked_categories) or 'N/A'})."
                elif finish_reason:
                    error_msg = f"Error: Gemini finished unexpectedly (Reason: {finish_reason})."
                else:
                    error_msg = "Error: Invalid response structure from Gemini (missing content/parts)."
                return error_msg, False
        elif 'promptFeedback' in response_data:
            feedback = response_data['promptFeedback']
            block_reason = feedback.get('blockReason', 'UNKNOWN')
            safety_ratings = feedback.get('safetyRatings', [])
            blocked_categories = [r.get('category', 'UNKNOWN') for r in safety_ratings if r.get('probability') not in ['NEGLIGIBLE', 'LOW']]
            logger.warning(f"Prompt blocked. Reason: {block_reason}. Safety Ratings: {safety_ratings}")
            error_msg = f"Error: Request prompt blocked (Reason: {block_reason}, Categories: {', '.join(blocked_categories) or 'N/A'})."
            return error_msg, False
        else:
            logger.error(f"Unexpected response format: {response_data}")
            return "Error: Received unexpected response format from Gemini API.", False

    except requests.exceptions.Timeout:
        error_msg = f"Error: Gemini API request timed out after {API_TIMEOUT} seconds."
        logger.error(error_msg)
        return error_msg, False
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        try:
            error_detail = e.response.json().get('error', e.response.text)
        except Exception:
            error_detail = e.response.text
        log_message = f"Gemini API HTTP Error ({status_code}). Details: {error_detail}"
        user_message = f"Error: API request failed (Status: {status_code})."
        if status_code == 400:
            user_message += " Check request format or parameters."
            logger.error(log_message, exc_info=False)
        elif status_code == 401:
            user_message += " Check API Key."
            logger.error(log_message, exc_info=False)
        elif status_code == 403:
            user_message += " Check API Key permissions or quota."
            logger.error(log_message, exc_info=False)
        elif status_code == 429:
            user_message += " Rate limit exceeded. Please try again later."
            logger.warning(log_message, exc_info=False)
        elif status_code >= 500:
            user_message += " Server error occurred. Please try again later."
            logger.error(log_message, exc_info=True)
        else:
            user_message += " Please check logs for details."
            logger.error(log_message, exc_info=True)
        return user_message, False
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during Gemini API request: {e}", exc_info=True)
        return "Error: Network error occurred while contacting Gemini API.", False
    except Exception as e:
        logger.error(f"Unexpected error during Gemini interaction: {e}", exc_info=True)
        return "Error: An unexpected error occurred during the Gemini interaction.", False

# --- Functions for Specific Tasks ---

def run_initial_analysis(image: Image.Image) -> str:
    """
    Generates the initial analysis for the given image using Gemini.

    Args:
        image: The PIL Image object for analysis.

    Returns:
        A string containing the analysis results or an error message.
    """
    logger.info("Running initial analysis via Gemini...")
    prompt = INITIAL_ANALYSIS_PROMPT
    result_text, success = query_gemini_vision(image, prompt)
    if success:
        return result_text if result_text else "[Analysis completed but no text returned by API]"
    else:
        return f"Initial Analysis Failed:\n{result_text}"

def run_multimodal_qa(
    image: Image.Image,
    question: str,
    history: List[Tuple[str, str]],
    roi_coords: Optional[Dict[str, int]] = None
) -> Tuple[str, bool]:
    """
    Handles a multimodal Q&A interaction using Gemini, taking conversation history and ROI into account.

    Args:
        image: The PIL Image object.
        question: The question to ask.
        history: List of previous (question, answer) tuples.
        roi_coords: Optional dictionary with ROI coordinates ('left', 'top', 'width', 'height').

    Returns:
        A tuple containing:
            - The generated answer or error message.
            - A boolean indicating success (True) or failure (False).
    """
    logger.info(f"Running multimodal QA. Question: '{question}', History length: {len(history)}, ROI provided: {bool(roi_coords)}")
    
    # Format ROI information robustly
    roi_info = "No specific region highlighted."
    if roi_coords and all(k in roi_coords for k in ['left', 'top', 'width', 'height']):
        try:
            roi_info = (
                f"User highlighted region: Top-Left=(x:{int(roi_coords['left'])}, y:{int(roi_coords['top'])}), "
                f"Width={int(roi_coords['width'])}, Height={int(roi_coords['height'])}."
            )
        except Exception as e:
            logger.warning(f"Error formatting ROI coordinates {roi_coords}: {e}. Using default message.")
            roi_info = "Issue processing highlighted region information."
    
    # Format conversation history (most recent first, limited to MAX_HISTORY_LEN)
    history_str = "\n---\n".join([f"User: {q}\nAI: {a}" for q, a in history[-MAX_HISTORY_LEN:][::-1]])
    if not history_str:
        history_str = "No previous conversation history."
    
    prompt = QA_CONTEXT_PROMPT_TEMPLATE.format(history_text=history_str, question=question, roi_info=roi_info)
    logger.debug(f"Generated QA prompt (excluding image):\n{prompt}")
    
    gemini_result, gemini_success = query_gemini_vision(image, prompt)
    if gemini_success:
        return gemini_result if gemini_result else "[QA processed but no text returned by API]", True
    else:
        return f"Gemini QA Failed:\n{gemini_result}", False

def run_disease_analysis(
    image: Image.Image,
    disease: str,
    roi_coords: Optional[Dict[str, int]] = None
) -> str:
    """
    Runs disease-specific analysis using Gemini, considering ROI if provided.

    Args:
        image: The PIL Image object.
        disease: The disease/condition to focus on.
        roi_coords: Optional ROI coordinates.

    Returns:
        A string containing the analysis results or an error message.
    """
    logger.info(f"Running disease analysis for '{disease}'. ROI provided: {bool(roi_coords)}")
    
    roi_info = "Analysis applies to the entire image."
    if roi_coords and all(k in roi_coords for k in ['left', 'top', 'width', 'height']):
        try:
            roi_info = (
                f"User highlighted region: Top-Left=(x:{int(roi_coords['left'])}, y:{int(roi_coords['top'])}), "
                f"Width={int(roi_coords['width'])}, Height={int(roi_coords['height'])}."
            )
        except Exception as e:
            logger.warning(f"Error formatting ROI coordinates {roi_coords}: {e}. Using default message.")
            roi_info = "Issue processing highlighted region information; analysis applies to entire image."
    
    prompt = DISEASE_SPECIFIC_PROMPT_TEMPLATE.format(disease=disease, roi_info=roi_info)
    logger.debug(f"Generated Disease Analysis prompt (excluding image):\n{prompt}")
    
    result_text, success = query_gemini_vision(image, prompt)
    if success:
        return result_text if result_text else "[Analysis completed but no text returned by API]"
    else:
        return f"Disease Analysis Failed ({disease}):\n{result_text}"

def estimate_ai_confidence(
    image: Image.Image,
    history: List[Tuple[str, str]]
) -> str:
    """
    Asks Gemini to estimate its confidence in its last response based on the conversation history.

    Args:
        image: The PIL Image associated with the last interaction.
        history: List of previous (question, answer) tuples.

    Returns:
        A string containing the confidence estimation or an error message.
    """
    if not history:
        logger.info("No history available for confidence estimation.")
        return "[Confidence estimation not performed: No previous interaction found]"
    
    logger.info("Estimating AI confidence via Gemini...")
    last_q, last_a = history[-1]  # Most recent interaction
    prompt = CONFIDENCE_PROMPT_TEMPLATE.format(last_q=last_q, last_a=last_a)
    logger.debug(f"Generated Confidence prompt (excluding image):\n{prompt}")
    
    result_text, success = query_gemini_vision(image, prompt)
    if success:
        return result_text if result_text else "[Confidence estimation processed but no text returned by API]"
    else:
        return f"Confidence Estimation Failed:\n{result_text}"
