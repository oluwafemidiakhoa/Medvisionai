import requests
import streamlit as st
import logging
from PIL import Image
import base64
import io
from typing import Optional, Tuple, Dict, Any

# Configure logger (assumed to be set up globally in your app)
logger = logging.getLogger(__name__)

# --- Constants ---
HF_VQA_MODEL_ID: str = "llava-hf/llava-1.5-7b-hf"  # Example model supporting VQA via the Hugging Face Inference API
HF_API_TIMEOUT: int = 60  # API request timeout in seconds

# --- Helper Functions ---

def get_hf_api_token() -> Optional[str]:
    """
    Retrieves the Hugging Face API Token securely from Streamlit secrets.

    Returns:
        The API token string if found, otherwise None.
    """
    try:
        token = st.secrets.get("HF_API_TOKEN")
        if token:
            logger.debug("Hugging Face API Token retrieved successfully from secrets.")
            return token
        else:
            logger.warning("HF_API_TOKEN not found in Streamlit secrets.")
            return None
    except Exception as e:
        logger.error(f"Error accessing Streamlit secrets for HF API Token: {e}", exc_info=True)
        return None

def _crop_image_to_roi(image: Image.Image, roi: Dict[str, int]) -> Optional[Image.Image]:
    """
    Crops a PIL Image to the specified ROI.

    Args:
        image: The PIL Image object.
        roi: A dictionary with keys 'left', 'top', 'width', and 'height'.

    Returns:
        A cropped Image if successful, or None if cropping fails.
    """
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
    Converts a PIL Image object to a base64 encoded PNG string.

    Args:
        image: The PIL Image object.

    Returns:
        The base64 encoded string representation of the image.

    Raises:
        Exception: If the image encoding fails.
    """
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_byte = buffered.getvalue()
        base64_str = base64.b64encode(img_byte).decode("utf-8")
        logger.debug(f"Image successfully encoded to base64 string ({len(base64_str)} chars).")
        return base64_str
    except Exception as e:
        logger.error(f"Error during image to base64 conversion: {e}", exc_info=True)
        raise Exception(f"Failed to process image for API request: {e}")

def query_hf_vqa_inference_api(
    image: Image.Image,
    question: str,
    roi: Optional[Dict[str, int]] = None
) -> Tuple[str, bool]:
    """
    Queries the Hugging Face VQA model via the Inference API.

    This function handles API token retrieval, optional ROI cropping,
    image encoding, payload construction (model-specific), API call,
    and response parsing.

    Args:
        image: The PIL Image object to analyze.
        question: The question to ask about the image.
        roi: An optional dictionary specifying the region of interest.
             Expected keys: 'left', 'top', 'width', 'height'.

    Returns:
        A tuple containing:
            - A string with the generated answer or an error message.
            - A boolean indicating success (True) or failure (False).
    """
    hf_api_token = get_hf_api_token()
    if not hf_api_token:
        return "[Fallback Unavailable] Hugging Face API Token not configured.", False

    api_url = f"https://api-inference.huggingface.co/models/{HF_VQA_MODEL_ID}"
    headers = {"Authorization": f"Bearer {hf_api_token}"}

    logger.info(f"Preparing HF VQA query. Model: {HF_VQA_MODEL_ID}, Using ROI: {bool(roi)}")

    # --- Prepare Image: Apply ROI if provided ---
    image_to_send = image
    if roi:
        cropped_image = _crop_image_to_roi(image, roi)
        if cropped_image:
            image_to_send = cropped_image
            logger.info("Using ROI-cropped image for HF VQA query.")
        else:
            logger.warning("ROI cropping failed; proceeding with full image.")

    try:
        img_base64 = _image_to_base64(image_to_send)
    except Exception as e:
        return f"[Fallback Error] {e}", False

    # --- Construct Payload ---
    # Adjust the payload structure as required by the specific model.
    payload = {
        "inputs": f"USER: <image>\n{question}\nASSISTANT:",
        "parameters": {"max_new_tokens": 250}
    }
    logger.debug(f"Payload prepared with keys: {list(payload.keys())}")

    # --- Make API Call ---
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=HF_API_TIMEOUT)
        response.raise_for_status()
        response_data = response.json()
        logger.debug(f"HF VQA API response: {response_data}")

        # --- Parse Response ---
        parsed_answer: Optional[str] = None

        # Example parsing for LLaVA-style responses:
        if isinstance(response_data, list) and response_data and "generated_text" in response_data[0]:
            full_text = response_data[0]["generated_text"]
            assistant_marker = "ASSISTANT:"
            if assistant_marker in full_text:
                parsed_answer = full_text.split(assistant_marker, 1)[-1].strip()
            else:
                parsed_answer = full_text.strip()
        # Example parsing for BLIP-style responses:
        elif isinstance(response_data, dict) and "answer" in response_data:
            parsed_answer = response_data["answer"]

        if parsed_answer and parsed_answer.strip():
            logger.info(f"Successfully parsed answer from HF VQA ({HF_VQA_MODEL_ID}).")
            return parsed_answer.strip(), True
        else:
            logger.warning(f"Response received but no valid answer parsed. Response: {response_data}")
            return "[Fallback Error] Could not parse a valid answer from the model's response.", False

    except requests.exceptions.Timeout:
        error_msg = f"Request to HF VQA API timed out after {HF_API_TIMEOUT} seconds."
        logger.error(error_msg)
        return "[Fallback Error] Request timed out.", False
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        error_detail = ""
        try:
            error_detail = e.response.json().get('error', e.response.text)
        except Exception:
            error_detail = e.response.text

        log_message = f"HTTP Error ({status_code}) for {api_url}. Details: {error_detail}"
        user_message = f"[Fallback Error] API request failed (Status: {status_code})."

        if status_code == 401:
            user_message += " Check HF API Token configuration."
            logger.error(log_message, exc_info=False)
        elif status_code == 404:
            user_message += f" Verify that Model ID '{HF_VQA_MODEL_ID}' is correct."
            logger.error(log_message, exc_info=False)
        elif status_code == 503:
            user_message += " The model may be loading; please try again later."
            logger.warning(log_message, exc_info=False)
        else:
            user_message += " Please check logs for details."
            logger.error(log_message, exc_info=True)
        return user_message, False
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during HF API request: {e}", exc_info=True)
        return "[Fallback Error] Network error occurred while contacting the API.", False
    except Exception as e:
        logger.error(f"Unexpected error during HF VQA query: {e}", exc_info=True)
        return "[Fallback Error] An unexpected error occurred during processing.", False
