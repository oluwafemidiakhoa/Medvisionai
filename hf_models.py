import requests
import streamlit as st
import logging
from PIL import Image
import base64
import io
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# --- Constants ---
# Choose a VQA model available on Hugging Face Inference API
# Option 1: General LLaVA model (good starting point)
HF_VQA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
# Option 2: Medical specific (if available/suitable) - Check Inference API support
# HF_VQA_MODEL_ID = "microsoft/BiomedVQA" # Example, verify API compatibility
HF_API_TIMEOUT = 90 # Shorter timeout for potentially faster models

def get_hf_api_token() -> Optional[str]:
    """Retrieves Hugging Face API Token from secrets."""
    try:
        token = st.secrets.get("HF_API_TOKEN")
        if not token:
            # Only warn if the feature is attempted to be used
            # st.warning("HF_API_TOKEN not found in Streamlit secrets. Hugging Face fallback disabled.")
            pass
        return token
    except Exception as e:
        st.error(f"Error accessing Streamlit secrets for HF API Token: {e}")
        return None

def image_to_base64_hf(image: Image.Image) -> str:
    """Converts PIL Image to base64 string for HF API."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG") # Or JPEG depending on model preference
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode("utf-8")

# @st.cache_data # Caching API calls is complex
def query_hf_vqa_inference_api(
    image: Image.Image,
    question: str
) -> Tuple[Optional[str], bool]:
    """
    Queries a Hugging Face VQA model via the Inference API.

    Args:
        image: PIL Image object.
        question: The question to ask about the image.

    Returns:
        Tuple (answer_text, success_boolean)
    """
    hf_api_token = get_hf_api_token()
    if not hf_api_token:
        return "[Fallback Unavailable] Hugging Face API Token not configured.", False

    api_url = f"https://api-inference.huggingface.co/models/{HF_VQA_MODEL_ID}"
    headers = {"Authorization": f"Bearer {hf_api_token}"}

    logger.info(f"Querying Hugging Face VQA API: {HF_VQA_MODEL_ID}")

    try:
        img_base64 = image_to_base64_hf(image)
    except Exception as e:
        logger.error(f"Failed to encode image for HF API: {e}", exc_info=True)
        return f"[Fallback Error] Failed to process image for HF API: {e}", False

    # Payload structure depends on the specific model API. LLaVA often uses text prompts combining image placeholders.
    # Check the model card on Hugging Face for the correct API usage.
    # Example for a model expecting separate image and text:
    # This payload structure might need adjustment based on the chosen HF_VQA_MODEL_ID
    payload = {
        "inputs": {
            "image": img_base64,
            "prompt": f"USER: <image>\n{question}\nASSISTANT:" # LLaVA style prompt
            # For other models it might be:
            # "question": question,
        },
        # Add parameters if needed by the model API
        # "parameters": {"max_new_tokens": 200}
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=HF_API_TIMEOUT)
        response.raise_for_status()
        response_data = response.json()
        logger.debug(f"HF VQA Raw Response: {response_data}")

        # --- Response Parsing (Highly Model Dependent) ---
        # Adapt this based on the actual JSON structure returned by the chosen model's API
        # Example for LLaVA style response (often a list with generated_text)
        if isinstance(response_data, list) and len(response_data) > 0 and "generated_text" in response_data[0]:
            answer = response_data[0]["generated_text"]
            # Clean up the answer if needed (e.g., remove the input prompt part)
            answer = answer.split("ASSISTANT:")[-1].strip()
            logger.info("Successfully received response from HF VQA.")
            return answer, True
        # Example for other potential structures
        elif isinstance(response_data, dict) and "answer" in response_data:
             answer = response_data["answer"]
             logger.info("Successfully received response from HF VQA.")
             return answer, True
        else:
            logger.warning(f"HF VQA response received, but couldn't parse answer. Response: {response_data}")
            return "[Fallback Error] Could not parse answer from HF VQA response.", False

    except requests.exceptions.Timeout:
        error_msg = f"[Fallback Error] Request to HF VQA API timed out ({api_url})."
        logger.error(error_msg)
        return error_msg, False
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else "N/A"
        error_detail = ""
        if e.response is not None:
             try: error_detail = e.response.json().get('error', e.response.text)
             except: error_detail = e.response.text # Fallback to raw text

        if status_code == 401: error_msg = "[Fallback Error] Unauthorized (401). Check HF API Token."
        elif status_code == 404: error_msg = f"[Fallback Error] Model Not Found (404). Check Model ID: {HF_VQA_MODEL_ID}."
        elif status_code == 503: error_msg = f"[Fallback Error] Model Loading (503). Wait and retry. Details: {error_detail}"
        else: error_msg = f"[Fallback Error] HF API Error ({e}, Status: {status_code}). Details: {error_detail}"

        logger.error(error_msg, exc_info=True)
        return error_msg, False
    except Exception as e:
        error_msg = f"[Fallback Error] Failed processing HF VQA response: {e}"
        logger.error(error_msg, exc_info=True)
        return error_msg, False