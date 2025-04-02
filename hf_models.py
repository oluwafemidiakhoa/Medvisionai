import requests
import streamlit as st
import logging
from PIL import Image
import base64
import io
from typing import Optional, Tuple, Dict, Any # Added Dict, Any

# Assume logger is configured elsewhere, consistent with previous sections
logger = logging.getLogger(__name__)

# --- Constants ---
# Specify the VQA model identifier from Hugging Face Hub (ensure it supports the Inference API)
# LLaVA is a common choice for general VQA. Verify compatibility and expected payload/response.
HF_VQA_MODEL_ID: str = "llava-hf/llava-1.5-7b-hf"
# HF_VQA_MODEL_ID: str = "Salesforce/blip-vqa-base" # Another example, payload/response differs!

# Timeout for the API request in seconds. Adjust based on expected model inference time.
HF_API_TIMEOUT: int = 60 # Slightly shorter timeout, adjust if needed

# --- Helper Functions ---

def get_hf_api_token() -> Optional[str]:
    """
    Retrieves the Hugging Face API Token securely from Streamlit secrets.

    Returns:
        The API token string if found, otherwise None.
    """
    try:
        # Access the token defined in Streamlit's secrets management
        # (e.g., in secrets.toml or environment variables for deployed apps)
        token = st.secrets.get("HF_API_TOKEN")
        if token:
            logger.debug("Hugging Face API Token retrieved successfully from secrets.")
            return token
        else:
            # Log the absence, but the user-facing warning happens in the main query function
            logger.warning("HF_API_TOKEN not found in Streamlit secrets.")
            return None
    except Exception as e:
        # Avoid exposing detailed error related to secrets management to the user here
        logger.error(f"Error accessing Streamlit secrets for HF API Token: {e}", exc_info=True)
        # The calling function should inform the user about the configuration issue
        return None

def _crop_image_to_roi(image: Image.Image, roi: Dict[str, int]) -> Optional[Image.Image]:
    """Crops a PIL Image to the specified ROI dictionary."""
    try:
        x0, y0 = int(roi['left']), int(roi['top'])
        x1, y1 = x0 + int(roi['width']), y0 + int(roi['height'])
        box = (x0, y0, x1, y1)
        cropped_img = image.crop(box)
        logger.debug(f"Cropped image to ROI box: {box}")
        return cropped_img
    except KeyError as e:
        logger.error(f"ROI dictionary is missing required key: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to crop image to ROI ({roi}): {e}", exc_info=True)
        return None


def _image_to_base64(image: Image.Image) -> str:
    """
    Converts a PIL Image object to a base64 encoded string (PNG format).

    Args:
        image: The PIL Image object.

    Returns:
        The base64 encoded string representation of the image.

    Raises:
        Exception: If image saving or encoding fails.
    """
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG") # PNG is generally preferred for lossless quality
        img_byte = buffered.getvalue()
        base64_str = base64.b64encode(img_byte).decode("utf-8")
        logger.debug(f"Image successfully encoded to base64 string ({len(base64_str)} chars).")
        return base64_str
    except Exception as e:
        logger.error(f"Error during image to base64 conversion: {e}", exc_info=True)
        # Re-raise to be caught by the calling function for user feedback
        raise Exception(f"Failed to process image for API request: {e}")


# Note: Caching API calls is generally complex due to external factors (API status, model updates)
# and potentially dynamic inputs (image content, question). Avoid simple Streamlit caching here.
def query_hf_vqa_inference_api(
    image: Image.Image,
    question: str,
    roi: Optional[Dict[str, int]] = None # Added roi parameter
) -> Tuple[str, bool]:
    """
    Queries a specified Hugging Face VQA model via the serverless Inference API.

    Handles API token retrieval, optional ROI cropping, image encoding, request
    construction (model-specific payload), API call, and response parsing.

    Args:
        image: The PIL Image object for analysis.
        question: The question to ask about the image.
        roi: An optional dictionary defining the region of interest to focus on.
             Expected keys: {'left', 'top', 'width', 'height'}.

    Returns:
        A tuple containing:
            - str: The generated answer string, or an error message prefixed
                   with "[Fallback Error]" or "[Fallback Unavailable]".
            - bool: True if the query was successful and an answer was parsed,
                    False otherwise.
    """
    hf_api_token = get_hf_api_token()
    if not hf_api_token:
        # Return a user-friendly message indicating configuration issue
        return "[Fallback Unavailable] Hugging Face API Token not configured.", False

    # Construct the API endpoint URL
    api_url = f"https://api-inference.huggingface.co/models/{HF_VQA_MODEL_ID}"
    headers = {"Authorization": f"Bearer {hf_api_token}"}

    logger.info(f"Preparing Hugging Face VQA query. Model: {HF_VQA_MODEL_ID}, ROI: {bool(roi)}")

    # --- Prepare Image ---
    image_to_send = image
    if roi:
        cropped_image = _crop_image_to_roi(image, roi)
        if cropped_image:
            image_to_send = cropped_image
            logger.info("Using ROI-cropped image for HF VQA query.")
        else:
            # Inform user/log that cropping failed, but proceed with full image
            logger.warning("Failed to crop image to ROI, proceeding with full image for HF VQA.")
            # Optionally, return an error if ROI processing is critical:
            # return "[Fallback Error] Failed processing ROI for image.", False

    try:
        img_base64 = _image_to_base64(image_to_send)
    except Exception as e:
        # Error already logged in _image_to_base64
        return f"[Fallback Error] {e}", False # Return the error message raised by the helper

    # --- Construct Payload (CRITICAL: Model-Dependent) ---
    # The structure of the 'payload' MUST match the specific model's requirements
    # as documented on its Hugging Face model card. Examples below.

    # Example Payload for LLaVA models (e.g., llava-hf/llava-1.5-7b-hf):
    payload = {
        "inputs": f"USER: <image>\n{question}\nASSISTANT:", # Prompt includes placeholder and question
         "parameters": {"max_new_tokens": 250} # Optional: control output length
    }

    # Example Payload for BLIP models (e.g., Salesforce/blip-vqa-base):
    # payload = {
    #     "inputs": {
    #         "image": img_base64,
    #         "question": question
    #     }
    # }

    # Example Payload for some other models might just need image bytes directly:
    # headers = {"Authorization": f"Bearer {hf_api_token}", "Content-Type": "image/png"} # Different headers!
    # payload = image_to_send.tobytes() # Send raw bytes

    logger.debug(f"Sending request to HF VQA API: {api_url}. Payload keys: {list(payload.keys()) if isinstance(payload, dict) else 'Raw Bytes'}")

    # --- Make API Call ---
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=HF_API_TIMEOUT) # Use json=payload for dicts
        # For raw bytes payload: requests.post(api_url, headers=headers, data=payload, timeout=HF_API_TIMEOUT)

        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        response_data = response.json()
        logger.debug(f"HF VQA Raw Response JSON: {response_data}")

        # --- Response Parsing (CRITICAL: Model-Dependent) ---
        # Adapt this section based on the JSON structure returned by HF_VQA_MODEL_ID
        parsed_answer: Optional[str] = None

        # Example Parsing for LLaVA style response (often a list with generated_text)
        if isinstance(response_data, list) and len(response_data) > 0 and "generated_text" in response_data[0]:
            full_text = response_data[0]["generated_text"]
            # Extract only the generated part after the "ASSISTANT:" marker
            assistant_marker = "ASSISTANT:"
            if assistant_marker in full_text:
                 parsed_answer = full_text.split(assistant_marker, 1)[-1].strip()
            else:
                 parsed_answer = full_text.strip() # Fallback if marker isn't found

        # Example Parsing for BLIP style response (dict with "answer")
        elif isinstance(response_data, dict) and "answer" in response_data:
             parsed_answer = response_data["answer"]

        # Add more 'elif' blocks here for other expected response structures

        # --- Validate and Return Parsed Answer ---
        if parsed_answer is not None and parsed_answer.strip():
            logger.info(f"Successfully parsed answer from HF VQA ({HF_VQA_MODEL_ID}).")
            return parsed_answer.strip(), True
        else:
            logger.warning(f"HF VQA response received, but failed to parse a valid answer. Response: {response_data}")
            return "[Fallback Error] Could not parse a valid answer from the model's response.", False

    except requests.exceptions.Timeout:
        error_msg = f"Request to Hugging Face VQA API timed out after {HF_API_TIMEOUT} seconds ({api_url}). The model might be taking too long."
        logger.error(error_msg)
        return f"[Fallback Error] Request timed out.", False # Keep user message concise
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        error_detail = ""
        try:
            # Try to get specific error message from JSON response
            error_detail = e.response.json().get('error', e.response.text)
        except: # Fallback if response is not JSON or parsing fails
            error_detail = e.response.text

        log_message = f"HF API HTTP Error ({status_code}) for {api_url}. Details: {error_detail}"
        user_message = f"[Fallback Error] API request failed (Status: {status_code})."

        if status_code == 401:
            user_message += " Check Hugging Face API Token configuration."
            logger.error(log_message, exc_info=False) # Don't need traceback for auth error
        elif status_code == 404:
            user_message += f" Check if Model ID '{HF_VQA_MODEL_ID}' is correct and supports Inference API."
            logger.error(log_message, exc_info=False)
        elif status_code == 503: # Model loading or unavailable
            user_message += " The model may be loading, please wait and try again."
            logger.warning(log_message, exc_info=False) # Warning, as it might be temporary
        else: # Other HTTP errors
            user_message += " Please check logs for details."
            logger.error(log_message, exc_info=True) # Include traceback for unexpected HTTP errors

        return user_message, False
    except requests.exceptions.RequestException as e:
        # Catch other network-related errors (DNS, connection refused, etc.)
        logger.error(f"Network error during HF API request to {api_url}: {e}", exc_info=True)
        return f"[Fallback Error] Network error occurred while contacting the API.", False
    except Exception as e:
        # Catch-all for any other unexpected errors (e.g., JSON decoding, parsing logic)
        logger.error(f"Unexpected error during HF VQA query or response processing: {e}", exc_info=True)
        return f"[Fallback Error] An unexpected error occurred during processing.", False