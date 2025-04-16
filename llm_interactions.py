# -*- coding: utf-8 -*-
"""
llm_interactions.py

This module handles interactions with the Gemini API (or potentially other
LLM backends configured similarly) for medical image analysis within the
RadVision AI application. It provides functions for generating initial analysis,
answering user questions in context, performing condition-specific analysis,
and estimating the AI's confidence in its previous responses.
"""

import requests
import base64
import io
import logging
import os
from typing import Optional, Tuple, List, Dict, Any
from PIL import Image
import PIL # Ensure PIL namespace is available if needed elsewhere implicitly

# Configure logging for this module
logger = logging.getLogger(__name__)

# --- Constants ---
IMAGE_MIME_TYPE: str = "image/png"  # Standardized format for transmission
API_TIMEOUT: int = 180              # Generous timeout for potentially complex API calls (seconds)

# --- Model Configuration ---
# Allow overriding the model via environment variable.
# Check Google AI Studio or official documentation for the latest recommended models.
# Models like "gemini-1.5-pro-latest" or "gemini-1.5-flash-latest" are generally good choices.
DEFAULT_MODEL_NAME: str = "gemini-1.5-pro-latest" # A stable, powerful default
GEMINI_MODEL_NAME: str = os.environ.get("GEMINI_MODEL_OVERRIDE", DEFAULT_MODEL_NAME)
logger.info(f"Using Gemini model: {GEMINI_MODEL_NAME}")

# --- Prompt Templates (Enhanced for Clarity and Role Definition) ---

# Common instructions for all prompts emphasizing the assistant's role and limitations
_BASE_ROLE_PROMPT = """
You are a highly specialized AI assistant simulating an expert radiologist. Your primary function is to analyze medical images and provide interpretations based *solely* on the visual information presented.

**IMPORTANT:**
- You **must not** provide definitive medical diagnoses or treatment advice.
- Your analysis is for informational and educational purposes only and requires verification by a qualified human expert.
- Focus exclusively on interpreting the visual data within the image.
- If the image quality is insufficient or information is lacking to answer confidently, state this clearly.
"""

INITIAL_ANALYSIS_PROMPT_TEMPLATE = f"""{_BASE_ROLE_PROMPT}

**Task:** Perform a comprehensive initial analysis of the provided medical image.

**Region of Interest (ROI):** {{roi_info}}

**Analysis Structure:**
Please provide the following sections in your response:
1.  **Image Description:** Detail the modality, view, and anatomical structures visible. Mention image quality if noteworthy (e.g., suboptimal, motion artifact).
2.  **Key Findings:** Enumerate significant observations or abnormalities detected in the image. If an ROI is specified, pay particular attention to that area. If no significant findings are present, state "No significant abnormalities detected."
3.  **Potential Differential Diagnoses:** Based *only* on the Key Findings, list potential conditions or interpretations, ordered from most likely to least likely.
4.  **Reasoning for Top Differential:** Briefly explain the visual evidence supporting the most likely differential diagnosis listed above.

Structure your response clearly using markdown formatting with these numbered headings.
"""

QA_CONTEXT_PROMPT_TEMPLATE = f"""{_BASE_ROLE_PROMPT}

**Task:** Answer the user's question regarding the provided medical image, considering the context.

**Region of Interest (ROI):** {{roi_info}}
**Conversation History (Recent turns, most recent first):**
{{history_text}}
---
**Current Question:** "{{question}}"
---

Analyze the provided image, taking into account the ROI (if specified) and the preceding conversation history. Provide a concise, clinically relevant answer directly addressing the "Current Question".

If the question cannot be answered based on the visual information in the image or the provided context, clearly state that and explain why (e.g., "The image does not show the requested anatomical region," or "The findings are indeterminate for answering this question").
"""

# Note: This prompt asks the AI to evaluate *itself* based on its prior output.
CONFIDENCE_PROMPT_TEMPLATE = f"""{_BASE_ROLE_PROMPT}

**Task:** Evaluate the confidence level of the *previous* AI response provided below, considering the context of the question/task it was addressing.

**Context of the Last Interaction:**
*   **Last Question/Task:** {{last_q}}
*   **Region of Interest (ROI) at the time:** {{roi_info}}
*   **Previous AI Response to Evaluate:**
    ```
    {{last_a}}
    ```
---
**Evaluation Request:**
Critically evaluate your confidence in the **"Previous AI Response to Evaluate"** ONLY. Consider factors such as:
*   **Clarity of Findings:** Were the described findings distinct and unambiguous in the context of the image?
*   **Sufficiency of Information:** Was the visual information likely sufficient to support the statements made in the response?
*   **Ambiguity:** Was there inherent ambiguity in the image or findings that affects certainty?
*   **Scope of Question:** Did the response fully address the scope of the "Last Question/Task"?

**Output Format:**
Respond *strictly* in the following format:
**Confidence:** [Numerical score from 1 (Very Low) to 10 (Very High)]/10
**Justification:** [A brief, concise explanation addressing the factors above to justify the score. Focus on the limitations or strengths related to the *previous response's* certainty.]
"""

DISEASE_SPECIFIC_PROMPT_TEMPLATE = f"""{_BASE_ROLE_PROMPT}

**Task:** Analyze the provided image *exclusively* for findings potentially related to **{{disease}}**.

**Region of Interest (ROI):** {{roi_info}}

**Analysis Structure:**
Provide the following information specific to **{{disease}}**:
1.  **Presence of Findings:** State whether definitive signs suggestive of {{disease}} are Present, Absent, or Indeterminate based *only* on this image.
2.  **Description of Relevant Findings:** If signs are present or indeterminate, describe the specific visual findings observed that are relevant to {{disease}}. If absent, state "No specific findings suggestive of {{disease}} were identified." Pay attention to the ROI if specified.
3.  **Severity Assessment (if applicable and findings present):** If findings suggestive of {{disease}} are present, provide a qualitative assessment of their severity (e.g., Mild, Moderate, Severe, Extensive), if possible from the image.
4.  **Limitations/Recommendations (Optional):** Briefly mention any limitations of this image for assessing {{disease}} or suggest potential *imaging* follow-up relevant to clarifying findings for {{disease}} (e.g., "Recommend comparison with prior studies," "CT correlation may be helpful"). Do not suggest clinical management.

Focus solely on the specified condition: **{{disease}}**.
"""

# --- Helper Function for Image Encoding ---
def _encode_image(image: Image.Image) -> Tuple[Optional[str], str]:
    """Encodes a PIL Image to base64 PNG string."""
    try:
        buffered = io.BytesIO()
        # Create a copy to avoid modifying the original object during conversion
        image_copy = image.copy()
        # Ensure image is in RGB format for broader compatibility before saving as PNG
        # This handles modes like RGBA, P (palette), LA (Luminance+Alpha), etc.
        if image_copy.mode not in ('RGB', 'L'): # Allow Luminance (grayscale) as well
            logger.debug(f"Converting image from mode {image_copy.mode} to RGB for encoding.")
            image_copy = image_copy.convert('RGB')

        image_copy.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        logger.debug(f"Image successfully encoded to base64 PNG (Size: {len(img_bytes)} bytes)")
        return img_base64, IMAGE_MIME_TYPE
    except Exception as e:
        logger.error(f"Error encoding image to PNG/base64: {e}", exc_info=True)
        return None, f"Error encoding image: {e}"

# --- Core Gemini Interaction Function ---
def query_gemini_vision(image: Image.Image, text_prompt: str) -> Tuple[Optional[str], bool]:
    """
    Sends an image and a text prompt to the configured Gemini Vision API.

    Args:
        image: A PIL Image object to analyze.
        text_prompt: The prompt text accompanying the image.

    Returns:
        A tuple containing:
            - The response text (str) if successful, or an error message (str) if failed.
            - A boolean success flag (True if successful, False otherwise).
    """
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.error("CRITICAL: GEMINI_API_KEY environment variable is not set.")
        return "Configuration Error: Gemini API key not found.", False

    # Construct the correct API endpoint URL
    # Using v1beta as it often has the latest vision models
    gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"

    # Prepare headers and parameters (API key goes in query params for this Google API)
    headers = {"Content-Type": "application/json"}
    params = {"key": gemini_api_key}

    # Encode the image
    img_base64, mime_type = _encode_image(image)
    if img_base64 is None:
        # Encoding failed, error message is in the 'mime_type' variable here
        return mime_type, False

    # Construct the payload according to Gemini API specifications
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": text_prompt},
                    {"inline_data": {"mime_type": mime_type, "data": img_base64}}
                ]
            }
        ],
        "generation_config": {
            "temperature": 0.2,         # Lower temperature for more factual/consistent medical interpretation
            "top_k": 32,                # Reasonable value for Top-K sampling
            "top_p": 0.95,              # Use Top-P sampling
            "max_output_tokens": 8192,  # Utilize large context window of Gemini 1.5 models
            "stop_sequences": []        # No specific stop sequences needed typically
        },
        "safety_settings": [
            # Configure safety settings - BLOCK_MEDIUM_AND_ABOVE is a reasonable default.
            # Consider BLOCK_LOW_AND_ABOVE for stricter filtering if needed,
            # but be aware it might block borderline medical content. Test thoroughly.
            {"category": cat, "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            for cat in [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT"
            ]
        ]
    }

    logger.info(f"Sending request to Gemini API model: {GEMINI_MODEL_NAME}")
    # Avoid logging full prompt/payload unless DEBUG level is enabled due to size/potential PII
    logger.debug(f"Prompt length: {len(text_prompt)} chars")
    logger.debug(f"Payload keys: {list(payload.keys())}")

    try:
        response = requests.post(
            gemini_api_url,
            headers=headers,
            params=params, # Key is passed as a query parameter
            json=payload,
            timeout=API_TIMEOUT
        )

        # Check for HTTP errors first
        response.raise_for_status()

        response_data: Dict[str, Any] = response.json()
        logger.debug("Raw Gemini API response received successfully.")

        # --- Parse the Gemini Response ---
        # Check for blocked prompt first (before candidates)
        if 'promptFeedback' in response_data and 'blockReason' in response_data['promptFeedback']:
            reason = response_data['promptFeedback']['blockReason']
            safety_ratings = response_data['promptFeedback'].get('safetyRatings', [])
            blocked_categories = [sr.get('category', 'UNKNOWN') for sr in safety_ratings if sr.get('blocked')]
            error_msg = f"API Error: Prompt blocked due to safety settings (Reason: {reason}). Categories: {blocked_categories}"
            logger.warning(error_msg)
            return error_msg, False

        # Check for candidates and content
        if 'candidates' in response_data and isinstance(response_data['candidates'], list) and response_data['candidates']:
            candidate = response_data['candidates'][0]

            # Check finish reason
            finish_reason = candidate.get('finishReason', 'UNKNOWN')
            if finish_reason not in ["STOP", "MAX_TOKENS"]:
                # Log unusual finish reasons like SAFETY, RECITATION, OTHER
                safety_ratings = candidate.get('safetyRatings', [])
                blocked_categories = [sr.get('category', 'UNKNOWN') for sr in safety_ratings if sr.get('blocked')]
                warn_msg = f"Gemini response finished with reason: {finish_reason}."
                if blocked_categories:
                     warn_msg += f" Associated safety categories: {blocked_categories}"
                logger.warning(warn_msg)
                # Decide if this should be an error or just a warning. Let's treat SAFETY as error.
                if finish_reason == "SAFETY":
                     return f"API Error: Response generation stopped due to safety settings. Categories: {blocked_categories}", False
                # For other reasons like RECITATION, maybe proceed but log warning.

            # Extract text content safely
            content = candidate.get('content', {})
            parts = content.get('parts', [])
            if parts and isinstance(parts, list) and 'text' in parts[0]:
                parsed_text = parts[0]['text'].strip()
                if parsed_text:
                    logger.info("Successfully received and parsed text response from Gemini API.")
                    return parsed_text, True
                else:
                    logger.warning("Gemini response contained an empty text part.")
                    return "API Error: Response received but contained no text content.", False
            else:
                logger.warning(f"Could not extract text from candidate part structure: {parts}")
                return "API Error: Response structure invalid (missing text in part).", False
        else:
            # No candidates found, indicates an issue
            logger.error(f"Unexpected response format: No candidates found. Keys: {response_data.keys()}")
            return "API Error: No valid candidates found in the API response.", False

    except requests.exceptions.Timeout:
        logger.error(f"Gemini API request timed out after {API_TIMEOUT} seconds.")
        return "API Error: Request timed out.", False
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        try:
            error_details = e.response.json() # Try to get JSON error details
            error_message = error_details.get("error", {}).get("message", e.response.text)
        except ValueError: # If response is not JSON
            error_message = e.response.text[:500] # Limit length

        logger.error(f"HTTP error {status_code} querying Gemini: {error_message}", exc_info=(status_code >= 500)) # Show trace for server errors

        if status_code == 400:
            return f"API Error: Bad Request (400). Check model name and request format. Details: {error_message}", False
        elif status_code == 401 or status_code == 403:
            return f"API Error: Authentication/Permission Failed ({status_code}). Check API Key and permissions.", False
        elif status_code == 429:
            return "API Error: Rate Limit Exceeded (429). Please try again later.", False
        elif status_code >= 500:
            return f"API Error: Server Error ({status_code}). Please try again later. Details: {error_message}", False
        else:
            return f"API Error: HTTP error {status_code}. Details: {error_message}", False
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during Gemini API request: {e}", exc_info=True)
        return "Network Error: Could not connect to the API service.", False
    except Exception as e:
        logger.critical(f"Unexpected critical error during Gemini API interaction: {e}", exc_info=True)
        return f"Internal Error: An unexpected error occurred ({type(e).__name__}).", False


# --- Specific LLM Interaction Functions ---

def run_initial_analysis(image: Image.Image, roi: Optional[Dict] = None) -> str:
    """
    Performs an initial analysis of the image using the Gemini API.

    Args:
        image: The PIL Image object.
        roi: Optional dictionary with ROI coordinates {left, top, width, height}.

    Returns:
        The analysis response text if successful, otherwise a string prefixed with "Initial Analysis Failed: ".
    """
    action_name = "Initial Analysis"
    logger.info(f"Requesting {action_name}. ROI provided: {bool(roi)}")
    roi_info = "No specific region highlighted by user."
    if roi and isinstance(roi, dict) and all(key in roi for key in ["left", "top", "width", "height"]):
        try:
            # Format ROI info clearly for the prompt
            roi_info = (f"User has highlighted a Region of Interest (ROI) at "
                        f"Top-Left=({int(roi['left'])}, {int(roi['top'])}) with "
                        f"Width={int(roi['width'])}, Height={int(roi['height'])} pixels.")
        except (TypeError, ValueError):
            logger.warning("ROI dictionary contained non-integer values.", exc_info=True)
            roi_info = "ROI provided but coordinates are invalid."

    prompt = INITIAL_ANALYSIS_PROMPT_TEMPLATE.format(roi_info=roi_info)
    response_text, success = query_gemini_vision(image, prompt)

    if success and response_text:
        return response_text
    else:
        # Prefix error message for clarity in the calling application (app.py)
        error_prefix = f"{action_name} Failed: "
        # Avoid duplicating the prefix if query_gemini_vision already added one
        if response_text and any(err in response_text for err in ["Error:", "Failed:"]):
             return f"{action_name} Failed - {response_text}" # Append if specific error type already exists
        else:
             return error_prefix + (response_text or "Unknown error from API.")

def run_multimodal_qa(
    image: Image.Image,
    question: str,
    history: List[Tuple[str, str, Any]], # Expecting (question, answer, timestamp) potentially
    roi: Optional[Dict] = None
    ) -> Tuple[str, bool]:
    """
    Performs a Q&A interaction using the Gemini API, considering history and ROI.

    Args:
        image: The PIL Image to analyze.
        question: The user's current question.
        history: List of previous interaction tuples (e.g., (question, answer, timestamp)).
        roi: Optional dictionary for region of interest.

    Returns:
        A tuple containing:
            - The response text (str) or an error message (str).
            - A boolean success flag (True if successful, False otherwise).
    """
    action_name = "Multimodal Q&A"
    logger.info(f"Requesting {action_name}. ROI provided: {bool(roi)}. History length: {len(history)}")
    roi_info = "No specific region highlighted by user."
    if roi and isinstance(roi, dict) and all(key in roi for key in ["left", "top", "width", "height"]):
        try:
             roi_info = (f"User has highlighted ROI at Top-Left=({int(roi['left'])}, {int(roi['top'])}) "
                         f"with Width={int(roi['width'])}, Height={int(roi['height'])}.")
        except (TypeError, ValueError):
             roi_info = "ROI provided but coordinates are invalid."


    # Format history: Use last N turns, ensure correct extraction from tuple
    HISTORY_TURNS_FOR_CONTEXT = 3
    history_text = ""
    if history:
        formatted_history = []
        # Iterate safely, handling potential variations in tuple length
        for entry in history[-HISTORY_TURNS_FOR_CONTEXT:][::-1]: # Last N turns, newest first
            q = entry[0] if len(entry) > 0 else "[Missing Question]"
            a = entry[1] if len(entry) > 1 else "[Missing Answer]"
            # Don't include timestamp in the prompt context unless needed
            formatted_history.append(f"User: {q}\nAI: {a}")
        history_text = "\n---\n".join(formatted_history)
    if not history_text:
        history_text = "No previous conversation history available for context."

    prompt = QA_CONTEXT_PROMPT_TEMPLATE.format(
        roi_info=roi_info,
        history_text=history_text,
        question=question
    )

    # Call the core API function
    response_text, success = query_gemini_vision(image, prompt)

    # Return the result tuple directly; query_gemini_vision handles error formatting
    return response_text if response_text else "Error: No response text received from API.", success


def run_disease_analysis(image: Image.Image, disease: str, roi: Optional[Dict] = None) -> str:
    """
    Runs a focused analysis for a specified condition using the Gemini API.

    Args:
        image: The PIL Image to analyze.
        disease: The condition/disease name to focus on.
        roi: Optional ROI dictionary.

    Returns:
        The analysis response text if successful, otherwise a string prefixed with "Disease Analysis Failed ({disease}): ".
    """
    action_name = "Disease Analysis"
    logger.info(f"Requesting {action_name} for '{disease}'. ROI provided: {bool(roi)}")
    roi_info = "No specific region highlighted by user."
    if roi and isinstance(roi, dict) and all(key in roi for key in ["left", "top", "width", "height"]):
        try:
            roi_info = (f"User has highlighted ROI at Top-Left=({int(roi['left'])}, {int(roi['top'])}) "
                        f"with Width={int(roi['width'])}, Height={int(roi['height'])}.")
        except (TypeError, ValueError):
            roi_info = "ROI provided but coordinates are invalid."

    prompt = DISEASE_SPECIFIC_PROMPT_TEMPLATE.format(
        disease=disease,
        roi_info=roi_info
    )

    response_text, success = query_gemini_vision(image, prompt)

    if success and response_text:
        return response_text
    else:
        error_prefix = f"{action_name} Failed ({disease}): "
        if response_text and any(err in response_text for err in ["Error:", "Failed:"]):
            return f"{action_name} Failed ({disease}) - {response_text}"
        else:
            return error_prefix + (response_text or "Unknown error from API.")


def estimate_ai_confidence(
    image: Image.Image, # Image is needed for the API call even if prompt doesn't reference it directly
    history: List[Tuple[str, str, Any]], # Expecting (question, answer, timestamp)
    initial_analysis: Optional[str] = None, # Added context param (though not used in current prompt)
    disease_analysis: Optional[str] = None, # Added context param (though not used in current prompt)
    roi: Optional[Dict] = None # ROI active during the last interaction being evaluated
    ) -> str:
    """
    Requests the Gemini API to estimate confidence in its *most recent* response.

    Args:
        image: The PIL Image corresponding to the last interaction.
        history: List of previous interaction tuples. Must not be empty.
        initial_analysis: The initial analysis text (currently unused in prompt, for potential future enhancement).
        disease_analysis: The disease analysis text (currently unused in prompt, for potential future enhancement).
        roi: The ROI dictionary active during the last interaction being evaluated.

    Returns:
        A string containing the confidence score and justification,
        or a string prefixed with "Confidence Estimation Failed: ".
    """
    action_name = "Confidence Estimation"
    logger.info(f"Requesting {action_name}. History length: {len(history)}. ROI used previously: {bool(roi)}")

    if not history:
        logger.warning(f"{action_name} requested without history.")
        return f"{action_name} Failed: No conversation history available to evaluate."

    # Extract the last interaction safely
    try:
        last_entry = history[-1]
        last_q = last_entry[0] if len(last_entry) > 0 else "[Missing Last Question]"
        last_a = last_entry[1] if len(last_entry) > 1 else "[Missing Last Answer]"
    except IndexError:
         logger.error("History list seems malformed or unexpectedly empty during confidence estimation.")
         return f"{action_name} Failed: Could not retrieve last interaction from history."

    # Pre-check: If the last answer was clearly an error, return low confidence immediately
    if last_a is None or isinstance(last_a, str) and any(err in last_a for err in ["Error:", "Failed:", "Blocked", "Unavailable"]):
        logger.warning(f"Last interaction appears to be an error/failure ('{last_a[:100]}...'), reporting low confidence directly.")
        return "**Confidence:** 1/10\n**Justification:** The previous step resulted in an error or failed response."

    # Format ROI info relevant to the *last interaction*
    roi_info = "No specific region highlighted by user during last interaction."
    if roi and isinstance(roi, dict) and all(key in roi for key in ["left", "top", "width", "height"]):
         try:
            roi_info = (f"User highlighted ROI at Top-Left=({int(roi['left'])}, {int(roi['top'])}) "
                         f"with Width={int(roi['width'])}, Height={int(roi['height'])} during last interaction.")
         except (TypeError, ValueError):
             roi_info = "ROI was provided but coordinates invalid during last interaction."


    prompt = CONFIDENCE_PROMPT_TEMPLATE.format(
        last_q=last_q,
        last_a=last_a,
        roi_info=roi_info
    )

    # Call the API using the *same image* as the last interaction
    response_text, success = query_gemini_vision(image, prompt)

    if success and response_text:
        # Basic validation of the expected format
        resp_lower = response_text.lower()
        if "**confidence:**" in resp_lower and "/10" in resp_lower and "**justification:**" in resp_lower:
            logger.info("Confidence estimation received in expected format.")
            return response_text
        else:
            logger.warning(f"Confidence response did not strictly match expected format:\n'''{response_text}'''")
            # Return the raw response anyway, prefixing it to indicate potential format issue
            return f"Confidence Response (Format Warning):\n{response_text}"
    else:
        error_prefix = f"{action_name} Failed: "
        if response_text and any(err in response_text for err in ["Error:", "Failed:"]):
             return f"{action_name} Failed - {response_text}"
        else:
             return error_prefix + (response_text or "Unknown error from API.")