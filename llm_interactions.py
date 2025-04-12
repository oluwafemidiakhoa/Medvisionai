# llm_interactions.py

"""
llm_interactions.py

This module handles interactions with the Gemini API (or other LLM backends)
for medical image analysis. It provides functions for initial analysis, Q&A,
condition-specific analysis, and confidence estimation.
"""

import requests
import base64
import io
import logging
import os
from typing import Optional, Tuple, List, Dict, Any
from PIL import Image

logger = logging.getLogger(__name__)

# --- Constants ---
IMAGE_MIME_TYPE: str = "image/png"  # Use PNG for image transmission
API_TIMEOUT: int = 180              # Timeout for API calls in seconds

# Allow overriding the model via environment variable, otherwise use a default
# Check Google AI Studio or documentation for the latest recommended models.
# Using "gemini-1.5-pro-latest" or "gemini-1.5-flash-latest" are often good choices.
# The specific experimental one below might change or be deprecated.
DEFAULT_MODEL_NAME: str = "gemini-1.5-pro-latest"
GEMINI_MODEL_NAME: str = os.environ.get("GEMINI_MODEL_OVERRIDE", DEFAULT_MODEL_NAME) # Use default if not overridden
# GEMINI_MODEL_NAME: str = os.environ.get("GEMINI_MODEL_OVERRIDE", "gemini-1.5-pro-exp-03-25") # Previous experimental

# --- Prompt Templates ---
INITIAL_ANALYSIS_PROMPT_TEMPLATE = """
You are an expert AI assistant simulating a radiologist. Your task is to analyze the provided medical image.

**Region of Interest:** {roi_info}

Based *only* on the visual information in the image (paying attention to the ROI if specified), provide:
1. **Detailed Description:** Describe the image content, anatomical structures, and view.
2. **Key Findings:** List any notable abnormalities or significant observations.
3. **Potential Differential Diagnoses:** List potential diagnoses based on the findings, ordered from most likely to least likely.
4. **Reasoning for Top Diagnosis:** Explain the evidence supporting the top differential diagnosis.

Structure your response clearly using these headings. Do not provide definitive medical advice. Focus solely on image interpretation.
"""

QA_CONTEXT_PROMPT_TEMPLATE = """
You are a medical expert AI assisting with the interpretation of a medical image.

**Region of Interest:** {roi_info}
**Conversation History (Most Recent First):**
{history_text}

**Current Question:** "{question}"

Analyze the provided image in light of the above context and provide a concise, clinically relevant answer to the Current Question.
If the image lacks sufficient information to answer the question, state that clearly.
"""

CONFIDENCE_PROMPT_TEMPLATE = """
You are an AI evaluating the confidence of a previous analysis of a medical image.
Based on the last interaction provided below:
---
Last Question/Task: {last_q}
Your Last Response:
{last_a}
---
Critically evaluate your confidence in the **last response's** analysis or answer. Consider factors like image quality (if discernible from the response), clarity of findings, ambiguity, and the nature of the question.
Respond strictly in this format, providing a numerical score and a brief justification:
**Confidence:** [Score]/10
**Justification:** [Brief explanation of factors affecting confidence in the *last response*].
"""

DISEASE_SPECIFIC_PROMPT_TEMPLATE = """
You are an expert radiologist AI. Focus exclusively on analyzing the provided image for signs of **{disease}**.

**Region of Interest:** {roi_info}

Provide:
1. **Findings:** Indicate whether signs of {disease} are present, absent, or indeterminate based on the image.
2. **Description:** Describe any visual findings relevant to {disease}.
3. **Severity Assessment (if applicable):** Qualitatively assess the severity of findings related to {disease}, if present.
4. **Imaging Recommendations (Optional):** Suggest potential next imaging steps based only on the findings for {disease} (without giving definitive medical advice).

If no signs related to {disease} are present, state so clearly.
"""

# --- Core Gemini Interaction Function ---
def query_gemini_vision(image: Image.Image, text_prompt: str) -> Tuple[Optional[str], bool]:
    """
    Sends an image and a text prompt to the Gemini API for analysis.

    Args:
        image: A PIL Image object to analyze.
        text_prompt: The prompt text accompanying the image.

    Returns:
        A tuple (response_text, success_flag).
    """
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.error("GEMINI_API_KEY environment variable is not set.")
        return "Error: Gemini API key not configured.", False

    # Use f-string formatting correctly for the API URL
    gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"
    headers = {
        "Content-Type": "application/json",
        # The API key is usually passed as a query parameter, not a header for this specific Google API
        # "x-goog-api-key": gemini_api_key # Remove this line
    }
    params = {"key": gemini_api_key} # Pass key as query parameter

    # Convert the image to PNG and then base64-encode it.
    try:
        buffered = io.BytesIO()
        # Ensure image is RGB before saving as PNG if it has an alpha channel or other modes
        if image.mode == 'RGBA' or image.mode == 'P':
            image_to_save = image.convert('RGB')
        else:
            image_to_save = image
        image_to_save.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        logger.error(f"Error encoding image to PNG/base64: {e}", exc_info=True)
        return f"Error encoding image: {e}", False

    # Construct payload matching Gemini API requirements
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
            "temperature": 0.3,         # Lower temperature for more deterministic medical analysis
            "top_k": 32,
            "top_p": 0.9,               # Using top_p as well
            "max_output_tokens": 8192, # Gemini 1.5 has large context window
            "stop_sequences": []
        },
        "safety_settings": [
            # Adjust safety settings as needed for medical context, be cautious
            {"category": cat, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} # Or BLOCK_LOW_AND_ABOVE / BLOCK_ONLY_HIGH
            for cat in [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT"
            ]
        ]
    }

    logger.info(f"Sending request to Gemini API: {gemini_api_url}")
    logger.debug(f"Prompt length: {len(text_prompt)}")
    logger.debug(f"Payload keys: {list(payload.keys())}") # Log top-level keys

    try:
        response = requests.post(
            gemini_api_url,
            headers=headers,
            params=params, # Pass key via params
            json=payload,
            timeout=API_TIMEOUT
        )
        response.raise_for_status() # Check for HTTP errors (4xx, 5xx)
        response_data: Dict[str, Any] = response.json()
        logger.debug(f"Raw Gemini API response received.") # Avoid logging potentially large response data by default

        # --- Parse Response ---
        # Carefully navigate the response structure based on Gemini documentation
        if 'candidates' in response_data and isinstance(response_data['candidates'], list) and response_data['candidates']:
            candidate = response_data['candidates'][0]
            # Check for finish reason (e.g., "STOP", "MAX_TOKENS", "SAFETY")
            finish_reason = candidate.get('finishReason', 'UNKNOWN')
            if finish_reason not in ["STOP", "MAX_TOKENS"]: # Treat other reasons (like SAFETY) as potential issues
                logger.warning(f"Gemini response finished with reason: {finish_reason}")
                # Check for safety ratings specifically if finish reason indicates it
                safety_ratings = candidate.get('safetyRatings', [])
                blocked_categories = [sr['category'] for sr in safety_ratings if sr.get('probability') in ['MEDIUM', 'HIGH']] # Adjust levels as needed
                if blocked_categories:
                    return f"Error: Response potentially blocked due to safety settings ({finish_reason}). Categories: {blocked_categories}", False

            # Extract text content
            if 'content' in candidate and 'parts' in candidate['content'] and isinstance(candidate['content']['parts'], list) and candidate['content']['parts']:
                if 'text' in candidate['content']['parts'][0]:
                    parsed_text = candidate['content']['parts'][0]['text'].strip()
                    if parsed_text:
                        logger.info("Successfully received and parsed text from Gemini API.")
                        return parsed_text, True
                    else:
                        logger.warning("Gemini response contained an empty text part.")
                        return "Error: Response received but contained no text.", False
                else:
                    logger.warning("First part of candidate content did not contain 'text' key.")
                    return "Error: Response structure invalid (missing text in part).", False
            else:
                logger.warning("Candidate content structure is invalid (missing 'content' or 'parts').")
                return "Error: Invalid response structure (candidate content).", False
        # Check for prompt feedback / blocking *before* candidates if applicable
        elif 'promptFeedback' in response_data:
            feedback = response_data['promptFeedback']
            block_reason = feedback.get('blockReason', 'UNKNOWN')
            logger.warning(f"Prompt blocked by API. Reason: {block_reason}")
            # Provide more details if available
            safety_feedback = feedback.get('safetyRatings', [])
            blocked_categories = [sr['category'] for sr in safety_feedback if sr.get('probability') in ['MEDIUM', 'HIGH']]
            return f"Error: Prompt blocked by API (Reason: {block_reason}). Categories: {blocked_categories}", False
        else:
            logger.error(f"Unexpected response format from Gemini API. Keys: {response_data.keys()}")
            return "Error: Unexpected response format from API.", False
    except requests.exceptions.Timeout:
        logger.error(f"Gemini API request timed out after {API_TIMEOUT} seconds.")
        return "Error: Request timed out.", False
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        error_text = e.response.text # Get error details from response body
        logger.error(f"HTTP error {status_code} querying Gemini: {e}. Response: {error_text}", exc_info=True)
        if status_code == 429:
             return "Error: API Rate Limit Exceeded (429). Please try again later.", False
        elif status_code in [401, 403]:
             return f"Error: Authentication/Permission Failed ({status_code}). Check API Key and permissions.", False
        elif status_code == 400:
             return f"Error: Bad Request (400). Check API endpoint, model name, and request format. Details: {error_text[:200]}", False
        else:
            return f"Error: HTTP error {status_code}. Details: {error_text[:200]}", False
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during Gemini API request: {e}", exc_info=True)
        return "Error: Network error connecting to API.", False
    except Exception as e:
        logger.critical(f"Unexpected error during Gemini API interaction: {e}", exc_info=True)
        return f"Error: An unexpected error occurred ({type(e).__name__}).", False

# --- LLM Interaction Functions ---
# UPDATED function signature
def run_initial_analysis(image: Image.Image, roi: Optional[Dict] = None) -> str:
    """
    Performs an initial analysis of the image using the Gemini API.

    Args:
        image: The PIL Image object.
        roi: Optional dictionary with ROI coordinates.

    Returns:
        The analysis response text or an error message.
    """
    logger.info(f"Requesting initial analysis. ROI provided: {bool(roi)}")
    roi_info = "No specific region highlighted."
    if roi and all(key in roi for key in ["left", "top", "width", "height"]):
        roi_info = f"ROI coordinates: Top-Left=({roi['left']}, {roi['top']}), Width={roi['width']}, Height={roi['height']}"

    prompt = INITIAL_ANALYSIS_PROMPT_TEMPLATE.format(roi_info=roi_info)
    response_text, success = query_gemini_vision(image, prompt)

    if success and response_text:
        return response_text
    else:
        # Prefixing with error type for clarity in UI
        return f"Initial Analysis Failed: {response_text}"

# UPDATED function signature (already correct in previous user version)
def run_multimodal_qa(image: Image.Image, question: str, history: List[Tuple[str, str]], roi: Optional[Dict] = None) -> Tuple[str, bool]:
    """
    Performs a Q&A interaction using the Gemini API, incorporating conversation history and ROI.

    Args:
        image: The PIL Image to analyze.
        question: The user question.
        history: List of previous (question, answer) pairs.
        roi: Optional dictionary for region of interest.

    Returns:
        A tuple (response_text, success_flag).
    """
    logger.info(f"Requesting multimodal Q&A. ROI provided: {bool(roi)}. History length: {len(history)}")
    roi_info = "No specific region highlighted."
    if roi and all(key in roi for key in ["left", "top", "width", "height"]):
        roi_info = f"ROI coordinates: Top-Left=({roi['left']}, {roi['top']}), Width={roi['width']}, Height={roi['height']}"

    # Format history nicely
    history_text = "\n---\n".join([f"User: {q}\nAI: {a}" for q, a in history[-3:][::-1]]) # Last 3 turns, newest first
    if not history_text:
        history_text = "No previous conversation history available."

    prompt = QA_CONTEXT_PROMPT_TEMPLATE.format(
        roi_info=roi_info,
        history_text=history_text,
        question=question
    )

    response_text, success = query_gemini_vision(image, prompt)
    # Return the actual response or error message, and the success flag
    return response_text if response_text else "Error: No response text received.", success

# UPDATED function signature
def run_disease_analysis(image: Image.Image, disease: str, roi: Optional[Dict] = None) -> str:
    """
    Runs a focused analysis for a specified condition using the Gemini API.

    Args:
        image: The PIL Image to analyze.
        disease: The condition to focus on.
        roi: Optional ROI dictionary.

    Returns:
        The analysis response text or an error message.
    """
    logger.info(f"Requesting disease analysis for '{disease}'. ROI provided: {bool(roi)}")
    roi_info = "No specific region highlighted."
    if roi and all(key in roi for key in ["left", "top", "width", "height"]):
        roi_info = f"ROI coordinates: Top-Left=({roi['left']}, {roi['top']}), Width={roi['width']}, Height={roi['height']}"

    prompt = DISEASE_SPECIFIC_PROMPT_TEMPLATE.format(
        disease=disease,
        roi_info=roi_info
    )

    response_text, success = query_gemini_vision(image, prompt)

    if success and response_text:
        return response_text
    else:
        return f"Disease Analysis Failed ({disease}): {response_text}"

# UPDATED function signature (added roi, though might not be used directly in prompt)
def estimate_ai_confidence(
    image: Image.Image,
    history: List[Tuple[str, str]],
    initial_analysis: Optional[str] = None, # Added context params from app.py
    disease_analysis: Optional[str] = None, # Added context params from app.py
    roi: Optional[Dict] = None
    ) -> str:
    """
    Requests an estimation of AI confidence based on the most recent analysis.
    Note: This calls the LLM again to evaluate its *previous* response.

    Args:
        image: The PIL Image that was analyzed (needed for the confidence query).
        history: List of previous (question, answer) pairs. Must not be empty.
        initial_analysis: String of the initial analysis (optional context).
        disease_analysis: String of the disease-specific analysis (optional context).
        roi: Optional ROI dict if used for the last analysis.

    Returns:
        A string containing the confidence score and justification, or an error message.
    """
    logger.info(f"Requesting AI confidence estimation. History length: {len(history)}. ROI used previously: {bool(roi)}")
    if not history:
        logger.warning("Confidence estimation requested without history.")
        return "Confidence Estimation Failed: No conversation history available to evaluate."

    # Evaluate confidence based on the LAST interaction in the history
    last_q, last_a = history[-1]

    # Simple check if the last answer itself was an error message
    if last_a is None or last_a.startswith("Error:") or "Failed:" in last_a:
        logger.warning(f"Last answer was an error/failure, reporting low confidence. Last Answer: '{last_a}'")
        return "**Confidence:** 1/10\n**Justification:** The previous step failed or returned an error."

    prompt = CONFIDENCE_PROMPT_TEMPLATE.format(
        last_q=last_q,
        last_a=last_a
    )

    # We use the same image that the previous response was based on
    response_text, success = query_gemini_vision(image, prompt)

    if success and response_text:
        # Basic validation of expected format
        if "**Confidence:**" in response_text and "/10" in response_text and "**Justification:**" in response_text:
            return response_text
        else:
            logger.warning(f"Confidence response did not match expected format: '{response_text}'")
            # Return the raw response anyway, but log a warning
            return f"Confidence Response (Format Warning):\n{response_text}"
    else:
        return f"Confidence Estimation Failed: {response_text}"