# llm_interactions.py

import requests
import base64
import io
import logging
import os # Using os module for environment variables (as per original code)
import streamlit as st
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any # Added Any for response data typing

logger = logging.getLogger(__name__)

# --- Constants ---
IMAGE_MIME_TYPE: str = "image/png" # Use PNG for potentially better quality with medical images
MAX_HISTORY_LEN: int = 5 # Limit conversation history sent to LLM
API_TIMEOUT: int = 180 # Seconds to wait for API response
DEFAULT_MODEL_NAME: str = "gemini-1.5-pro-latest" # Default stable model
# Allow overriding via environment variable if needed
# Using the specified experimental model from original code:
GEMINI_MODEL_NAME: str = os.environ.get("GEMINI_MODEL_OVERRIDE", "gemini-2.5-pro-exp-03-25")

# --- Helper Functions ---

# Caching image conversion is generally safe and can save processing
@st.cache_data(max_entries=50, show_spinner=False)
def _image_to_base64_str(image_bytes: bytes, image_format: str = "PNG") -> str:
    """
    Converts image bytes to a base64 encoded string.

    Args:
        image_bytes: Raw bytes of the image.
        image_format: The format of the image ('PNG', 'JPEG', etc.).

    Returns:
        The base64 encoded string.
    """
    # No need to read bytes again, they are passed directly
    base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
    logger.debug(f"Encoded {len(image_bytes)} bytes image ({image_format}) to base64 string ({len(base64_encoded)} chars).")
    return base64_encoded

def get_gemini_api_url_and_key() -> Tuple[Optional[str], Optional[str]]:
    """
    Retrieves the Gemini API Key and constructs the API URL.

    Returns:
        A tuple containing (api_url, api_key). Both can be None if the key is missing.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # Log the error here, but let the caller handle st.error display
        logger.error("GEMINI_API_KEY environment variable not set.")
        return None, None

    logger.debug(f"Retrieved GEMINI_API_KEY. Targeting Model: {GEMINI_MODEL_NAME}")
    # Construct the API URL using the globally defined model name
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"
    return api_url, api_key


# --- Prompt Templates (Content remains the same as provided, assumed correct) ---
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
    """
    Sends the image and text prompt to the configured Gemini API endpoint.

    Handles API key retrieval, image encoding, payload construction, API call,
    and robust response parsing including safety checks.

    Args:
        image: The PIL Image object to send.
        text_prompt: The text prompt to accompany the image.

    Returns:
        A tuple containing:
            - str: The generated text content if successful, otherwise an error
                   message prefixed with "Error:". Can be None on success if
                   the API returns no text unexpectedly.
            - bool: True if the API call was successful and text was parsed,
                    False otherwise.
    """
    gemini_api_url, api_key = get_gemini_api_url_and_key()
    if not gemini_api_url or not api_key:
        # Error logged in get_gemini_api_url_and_key
        # Display error in UI from the calling context (e.g., Action Handling) if needed
        st.error("Configuration Error: GEMINI_API_KEY is not set. Cannot query Gemini API.")
        return "Error: Gemini API key not configured.", False

    logger.info(f"Querying Gemini endpoint: {gemini_api_url} (Model: {GEMINI_MODEL_NAME})")

    # --- Prepare Image ---
    try:
        # Save image to bytes in memory
        buffered = io.BytesIO()
        image_format = "PNG" # Stick to PNG
        image.save(buffered, format=image_format)
        img_bytes = buffered.getvalue()
        # Encode bytes to base64
        img_base64 = _image_to_base64_str(img_bytes, image_format)
        logger.debug(f"Image prepared for Gemini API (Base64 Length: {len(img_base64)}).")
    except Exception as e:
        logger.error(f"Failed to process image for Gemini API: {e}", exc_info=True)
        return f"Error: Failed to process image for API request ({e}).", False

    # --- Construct Payload ---
    # Standard Gemini Vision payload structure
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": text_prompt},
                    {"inline_data": {"mime_type": IMAGE_MIME_TYPE, "data": img_base64}}
                ]
            }
        ],
        # Generation configuration - adjust as needed for the specific model
        "generation_config": {
            "temperature": 0.3,         # Lower temperature for more deterministic medical answers
            "top_k": 32,                # Consider adjusting based on model docs
            "top_p": 0.9,               # Consider adjusting based on model docs
            "max_output_tokens": 8192,  # Max allowed by many models, adjust if needed
            "stop_sequences": []        # No specific stop sequences defined here
        },
        # Safety settings - BLOCK_MEDIUM_AND_ABOVE is a reasonable default
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
        "x-goog-api-key": api_key # Send API key via header (alternative to query param)
                                  # Note: Query param method used in URL construction still works too. Header is often preferred.
    }

    # Use the URL without the key in query params if using header auth
    api_url_for_request = gemini_api_url # Keep URL built previously (which might have key) or use base URL if key is only in header

    logger.debug(f"Sending request to Gemini. Payload keys: {list(payload.keys())}. Content length (approx): {len(text_prompt) + len(img_base64)}")

    # --- Make API Call ---
    try:
        response = requests.post(api_url_for_request, headers=headers, json=payload, timeout=API_TIMEOUT)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        response_data: Dict[str, Any] = response.json()
        logger.debug(f"Gemini Raw Response JSON: {response_data}")

        # --- Response Parsing ---
        # Based on standard Gemini API structure v1beta
        if 'candidates' in response_data and isinstance(response_data['candidates'], list) and len(response_data['candidates']) > 0:
            candidate = response_data['candidates'][0]
            # Check for valid content structure
            if 'content' in candidate and 'parts' in candidate['content'] and isinstance(candidate['content']['parts'], list) and len(candidate['content']['parts']) > 0:
                # Check if the first part contains text
                if 'text' in candidate['content']['parts'][0]:
                    parsed_text = candidate['content']['parts'][0]['text']
                    logger.info("Successfully received and parsed text response from Gemini.")
                    return parsed_text, True
                else:
                    # No text found, check finish reason (e.g., safety)
                    finish_reason = candidate.get('finishReason', 'UNKNOWN')
                    safety_ratings = candidate.get('safetyRatings', [])
                    logger.warning(f"Gemini response part exists but lacks 'text'. Finish Reason: {finish_reason}. Safety Ratings: {safety_ratings}")
                    if finish_reason == 'SAFETY':
                         blocked_categories = [r.get('category', 'UNKNOWN') for r in safety_ratings if r.get('probability') not in ['NEGLIGIBLE', 'LOW']]
                         error_msg = f"Error: Gemini response blocked due to safety reasons (Categories: {', '.join(blocked_categories) or 'N/A'})."
                    else:
                         error_msg = f"Error: Gemini response structure issue - expected 'text' not found in parts. Finish Reason: {finish_reason}."
                    return error_msg, False
            else:
                # Content/parts structure is missing or invalid, check finish reason
                finish_reason = candidate.get('finishReason')
                safety_ratings = candidate.get('safetyRatings', [])
                logger.warning(f"Gemini response candidate lacks valid 'content' or 'parts'. Finish Reason: {finish_reason}. Safety Ratings: {safety_ratings}")
                if finish_reason == 'SAFETY':
                     blocked_categories = [r.get('category', 'UNKNOWN') for r in safety_ratings if r.get('probability') not in ['NEGLIGIBLE', 'LOW']]
                     error_msg = f"Error: Content generation stopped due to safety filters (Categories: {', '.join(blocked_categories) or 'N/A'})."
                elif finish_reason:
                     error_msg = f"Error: Gemini finished unexpectedly (Reason: {finish_reason})."
                else:
                     error_msg = f"Error: Invalid Gemini response structure (missing content/parts)."
                return error_msg, False

        # Handle cases where the prompt itself was blocked
        elif 'promptFeedback' in response_data:
             feedback = response_data['promptFeedback']
             block_reason = feedback.get('blockReason', 'UNKNOWN')
             safety_ratings = feedback.get('safetyRatings', [])
             blocked_categories = [r.get('category', 'UNKNOWN') for r in safety_ratings if r.get('probability') not in ['NEGLIGIBLE', 'LOW']]
             logger.warning(f"Gemini prompt blocked. Reason: {block_reason}. Safety Ratings: {safety_ratings}")
             error_msg = f"Error: The request prompt was blocked by Gemini (Reason: {block_reason}, Categories: {', '.join(blocked_categories) or 'N/A'})."
             return error_msg, False

        # Fallback if response structure is completely unexpected
        else:
            logger.error(f"Unexpected Gemini response format: No 'candidates' or 'promptFeedback'. Response: {response_data}")
            return "Error: Received an unexpected response format from Gemini API.", False

    except requests.exceptions.Timeout:
         error_msg = f"Error: Gemini API request timed out after {API_TIMEOUT} seconds."
         logger.error(error_msg)
         return error_msg, False # Keep user message concise
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        error_detail = "No specific error message provided."
        try:
            # Try to extract Google's structured error message
            error_data = e.response.json().get('error', {})
            error_detail = error_data.get('message', e.response.text)
            # Log specific details if available
            logger.debug(f"Gemini API HTTP Error Details: Status={error_data.get('status')}, Code={error_data.get('code')}")
        except: # Fallback if response is not JSON or structure differs
            error_detail = e.response.text

        log_message = f"Gemini API HTTP Error ({status_code}). Details: {error_detail}"
        user_message = f"Error: API request failed (Status: {status_code})."

        if status_code == 400: # Bad Request (often malformed payload or invalid args)
             user_message += " Check request format or parameters."
             logger.error(log_message, exc_info=False)
        elif status_code == 401: # Unauthorized
             user_message += " Check API Key."
             logger.error(log_message, exc_info=False)
        elif status_code == 403: # Forbidden (often API key valid but lacks permission/quota)
             user_message += " Check API Key permissions or quota."
             logger.error(log_message, exc_info=False)
        elif status_code == 429: # Rate limit exceeded
            user_message += " Rate limit exceeded. Please try again later."
            logger.warning(log_message, exc_info=False) # Warning as it's temporary
        elif status_code >= 500: # Server-side error
             user_message += " Server error occurred. Please try again later."
             logger.error(log_message, exc_info=True) # Include traceback for server errors
        else: # Other client-side errors
             user_message += " Please check logs for details."
             logger.error(log_message, exc_info=True)

        return user_message, False
    except requests.exceptions.RequestException as e:
        # Catch other network-related errors
        logger.error(f"Network error during Gemini API request: {e}", exc_info=True)
        return "Error: Network error occurred while contacting the Gemini API.", False
    except Exception as e:
        # Catch-all for any other unexpected errors during processing
        logger.error(f"Unexpected error processing Gemini request or response: {e}", exc_info=True)
        return "Error: An unexpected error occurred during the Gemini interaction.", False


# --- Functions for Specific Tasks ---
# These wrappers call query_gemini_vision with specific prompts.

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
        # Return the text, or a placeholder if the API unexpectedly returns empty text
        return result_text if result_text else "[Analysis completed but no text returned by API]"
    else:
        # result_text already contains the error message from query_gemini_vision
        return f"Initial Analysis Failed:\n{result_text}"

def run_multimodal_qa(
    image: Image.Image,
    question: str,
    history: List[Tuple[str, str]],
    roi_coords: Optional[Dict[str, int]] = None # Changed type hint for keys
    ) -> Tuple[str, bool]:
    """
    Handles a Question/Answer interaction using Gemini, considering history and ROI.

    Args:
        image: The PIL Image object.
        question: The user's question.
        history: List of previous (question, answer) tuples.
        roi_coords: Optional dictionary with ROI coordinates ('left', 'top', 'width', 'height').

    Returns:
        A tuple containing:
            - str: The generated answer or an error message.
            - bool: True if successful, False otherwise.
    """
    logger.info(f"Running multimodal QA via Gemini. Question: '{question}', History length: {len(history)}, ROI: {bool(roi_coords)}")

    # Format ROI information for the prompt
    roi_info = "No specific region highlighted."
    if roi_coords and all(k in roi_coords for k in ['left', 'top', 'width', 'height']):
        try:
             roi_info = (f"User highlighted a region of interest: "
                         f"Approx. Bounding Box Top-Left=(x:{int(roi_coords['left'])}, y:{int(roi_coords['top'])}), "
                         f"Width={int(roi_coords['width'])}, Height={int(roi_coords['height'])}. "
                         f"Focus analysis on this region if relevant to the question.")
        except (ValueError, TypeError, KeyError) as e:
             logger.warning(f"Could not format ROI coordinates ({roi_coords}): {e}. Using default message.")
             roi_info = "Issue processing highlighted region information."

    # Format conversation history (most recent first)
    history_str = "\n---\n".join([f"User: {q}\nAI: {a}" for q, a in history[-MAX_HISTORY_LEN:][::-1]]) # Reverse for most recent first
    if not history_str:
        history_str = "No previous conversation history."

    prompt = QA_CONTEXT_PROMPT_TEMPLATE.format(history_text=history_str, question=question, roi_info=roi_info)
    logger.debug(f"Generated QA Prompt (excluding image):\n{prompt}")

    gemini_result, gemini_success = query_gemini_vision(image, prompt)

    if gemini_success:
        # Return the text, or a placeholder if the API unexpectedly returns empty text
        return gemini_result if gemini_result else "[QA processed but no text returned by API]", True
    else:
        # gemini_result already contains the error message
        return f"Gemini QA Failed:\n{gemini_result}", False

def run_disease_analysis(
    image: Image.Image,
    disease: str,
    roi_coords: Optional[Dict[str, int]] = None # Changed type hint for keys
    ) -> str:
    """
    Runs disease-specific analysis using Gemini, considering ROI.

    Args:
        image: The PIL Image object.
        disease: The name of the disease/condition to focus on.
        roi_coords: Optional dictionary with ROI coordinates.

    Returns:
        A string containing the analysis results or an error message.
    """
    logger.info(f"Running disease-specific analysis for '{disease}' via Gemini. ROI: {bool(roi_coords)}")

    # Format ROI information
    roi_info = "Analysis applies to the entire image."
    if roi_coords and all(k in roi_coords for k in ['left', 'top', 'width', 'height']):
        try:
            roi_info = (f"User highlighted a region of interest: "
                        f"Approx. Bounding Box Top-Left=(x:{int(roi_coords['left'])}, y:{int(roi_coords['top'])}), "
                        f"Width={int(roi_coords['width'])}, Height={int(roi_coords['height'])}. "
                        f"Focus the assessment for '{disease}' primarily within this region.")
        except (ValueError, TypeError, KeyError) as e:
             logger.warning(f"Could not format ROI coordinates ({roi_coords}): {e}. Using default message.")
             roi_info = "Issue processing highlighted region information; analysis applies to the entire image."

    prompt = DISEASE_SPECIFIC_PROMPT_TEMPLATE.format(disease=disease, roi_info=roi_info)
    logger.debug(f"Generated Disease Analysis Prompt (excluding image):\n{prompt}")

    result_text, success = query_gemini_vision(image, prompt)

    if success:
        return result_text if result_text else "[Analysis completed but no text returned by API]"
    else:
        return f"Disease Analysis Failed ({disease}):\n{result_text}"

def estimate_ai_confidence(
    image: Image.Image, # Image might provide context for confidence (e.g., quality)
    history: List[Tuple[str, str]]
    ) -> str:
    """
    Asks Gemini to estimate its confidence based on the last interaction.

    Args:
        image: The PIL Image associated with the last interaction.
        history: List of previous (question, answer) tuples.

    Returns:
        A string containing the confidence estimation or an error message.
    """
    if not history:
        logger.info("Confidence estimation skipped: No history available.")
        # Return a clearer message indicating why it wasn't performed
        return "[Confidence estimation not performed: No previous interaction found]"

    logger.info("Requesting confidence estimation via Gemini...")
    last_q, last_a = history[-1] # Get the most recent interaction

    prompt = CONFIDENCE_PROMPT_TEMPLATE.format(last_q=last_q, last_a=last_a)
    logger.debug(f"Generated Confidence Prompt (excluding image):\n{prompt}")

    result_text, success = query_gemini_vision(image, prompt)

    if success:
        return result_text if result_text else "[Confidence estimation processed but no text returned by API]"
    else:
        return f"Confidence Estimation Failed:\n{result_text}"