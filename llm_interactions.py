import requests
import base64
import io
import logging
import streamlit as st
from PIL import Image
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

# --- Constants ---
IMAGE_MIME_TYPE = "image/png"
MAX_HISTORY_LEN = 5
API_TIMEOUT = 180

# --- Helper ---
@st.cache_data(max_entries=100) # Cache image encoding
def image_to_base64_str(image_bytes: bytes, format: str = "PNG") -> str:
    """Converts image bytes to a base64 encoded string."""
    img_byte = image_bytes # Assume already in correct format bytes
    return base64.b64encode(img_byte).decode("utf-8")

def get_gemini_api_url() -> Optional[str]:
    """Retrieves Gemini API URL from secrets."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            st.error("GEMINI_API_KEY not found in Streamlit secrets.")
            return None
        # --- Specify the desired Gemini model here ---
        # model_name = "gemini-1.5-flash-latest" # Example: Use Flash
        model_name = "gemini-1.5-pro-latest"  # Example: Use latest Pro (recommended)
        # model_name = "gemini-2.5-pro-exp-03-25" # Example: Use experimental if needed
        # -------------------------------------------
        return f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    except Exception as e:
        st.error(f"Error accessing Streamlit secrets for Gemini API Key: {e}")
        return None


# --- Prompt Templates ---
# (Keep the same templates as before: INITIAL_ANALYSIS_PROMPT, QA_CONTEXT_PROMPT_TEMPLATE, etc.)
# For brevity, they are omitted here, but should be included in this file.

INITIAL_ANALYSIS_PROMPT = """
You are an expert AI assistant simulating a radiologist... (full prompt)
"""

QA_CONTEXT_PROMPT_TEMPLATE = """
You are a medical expert AI assisting with the interpretation of a medical image...

**Region of Interest:** {roi_info}
**Conversation History (Most Recent First):**
{history_text}

**Current Question:** "{question}"

Analyze the provided medical image again... (rest of prompt)
"""

CONFIDENCE_PROMPT_TEMPLATE = """
Based on your most recent analysis or answer provided below... (full prompt)
"""

DISEASE_SPECIFIC_PROMPT_TEMPLATE = """
You are an expert radiologist AI. Focus *specifically* on evaluating the provided medical image for signs of **{disease}**...

**Region of Interest:** {roi_info}

Address the following points... (rest of prompt)
"""


# --- Core Gemini Interaction ---
# @st.cache_data # Caching API calls can be complex due to changing history/prompts
def query_gemini_vision(
    image: Image.Image,
    text_prompt: str,
    ) -> Tuple[Optional[str], bool]:
    """Sends the image and text prompt to the configured Gemini API."""
    gemini_api_url = get_gemini_api_url()
    if not gemini_api_url:
        return "Error: Gemini API URL not configured.", False

    logger.info(f"Querying Gemini API: {gemini_api_url.split('?')[0]}") # Log endpoint without key
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
        "generation_config": {"temperature": 0.3, "top_k": 32, "top_p": 0.9, "max_output_tokens": 8192, "stop_sequences": []}, # Increased tokens for Pro
        "safety_settings": [
            {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        ]
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(gemini_api_url, headers=headers, json=payload, timeout=API_TIMEOUT)
        # Robust Error Handling & Response Parsing (keep the detailed logic from the previous version)
        # ... (include the full try/except block with response parsing and error handling here) ...
        # Example snippet:
        response.raise_for_status()
        response_data = response.json()
        # ... (rest of parsing logic)
        if 'candidates' in response_data and response_data['candidates']:
             # ... check parts, text, finishReason, etc.
             part = response_data['candidates'][0].get('content', {}).get('parts', [{}])[0]
             if 'text' in part:
                 logger.info("Successfully received text response from Gemini.")
                 return part['text'], True
             else: # Handle blocked response, errors etc.
                 finish_reason = response_data['candidates'][0].get('finishReason', 'UNKNOWN')
                 # ... (detailed handling)
                 error_msg = f"Gemini Error: No text in response. Finish Reason: {finish_reason}"
                 logger.error(error_msg)
                 return error_msg, False
        # ... (handle promptFeedback, other errors)

        error_msg = f"Error: Unexpected Gemini response format. Response: {response_data}"
        logger.error(error_msg)
        return error_msg, False


    except requests.exceptions.Timeout:
         error_msg = f"Error: Gemini API request timed out after {API_TIMEOUT} seconds."
         logger.error(error_msg)
         return error_msg, False
    except requests.exceptions.RequestException as e:
        # ... (Keep the detailed RequestException handling from previous version)
        status_code = e.response.status_code if e.response is not None else "N/A"
        logger.error(f"Gemini API request failed: {e} (Status Code: {status_code})", exc_info=True)
        # ... detailed status code messages ...
        return f"Error connecting to Gemini API: {e}", False
    except Exception as e:
        logger.error(f"Unexpected error processing Gemini response: {e}", exc_info=True)
        return f"Error: Failed to process the response from Gemini ({e}).", False


# --- Functions for Specific Tasks ---

def run_initial_analysis(image: Image.Image) -> str:
    """Generates the initial analysis using Gemini."""
    logger.info("Running initial analysis...")
    # Add ROI info if available? Maybe not for initial general analysis.
    roi_info = "Not specified." # Placeholder for initial analysis
    prompt = INITIAL_ANALYSIS_PROMPT # Use the basic prompt
    result_text, success = query_gemini_vision(image, prompt)
    if success:
        logger.info("Initial analysis successful.")
        return result_text or "[Analysis OK, No Text Returned]"
    else:
        logger.error(f"Initial analysis failed: {result_text}")
        return f"Initial Analysis Failed:\n{result_text}"

def run_multimodal_qa(
    image: Image.Image,
    question: str,
    history: List[Tuple[str, str]],
    roi_coords: Optional[Dict] = None # Add ROI coordinates
    ) -> Tuple[str, bool]:
    """Handles QA, potentially using ROI and history."""
    logger.info(f"Received question: {question}")

    # Format ROI info for the prompt
    if roi_coords:
        roi_info = f"User has highlighted a region of interest with bounding box: Top-Left ({roi_coords['left']},{roi_coords['top']}), Bottom-Right ({roi_coords['left'] + roi_coords['width']},{roi_coords['top'] + roi_coords['height']}). Focus your answer on this region if relevant."
    else:
        roi_info = "No specific region highlighted by the user."

    history_str = "\n---\n".join([f"User: {q}\nAI: {a}" for q, a in history[-MAX_HISTORY_LEN:]]) or "No previous questions."
    prompt = QA_CONTEXT_PROMPT_TEMPLATE.format(history_text=history_str, question=question, roi_info=roi_info)

    # Try Gemini first
    gemini_result, gemini_success = query_gemini_vision(image, prompt)

    if gemini_success:
        logger.info("Gemini answered QA successfully.")
        return gemini_result or "[QA OK, No Text Returned]", True
    else:
        # Return the error from Gemini
        logger.warning(f"Gemini QA failed: {gemini_result}. No fallback configured in this module.")
        return f"Gemini QA Failed:\n{gemini_result}", False
        # Fallback logic would now be handled in the main app using hf_models.py

def run_disease_analysis(
    image: Image.Image,
    disease: str,
    roi_coords: Optional[Dict] = None
    ) -> str:
    """Runs disease-specific analysis."""
    logger.info(f"Running disease analysis for: {disease}")
    if roi_coords:
        roi_info = f"User has highlighted a region of interest with bounding box: Top-Left ({roi_coords['left']},{roi_coords['top']}), Bottom-Right ({roi_coords['left'] + roi_coords['width']},{roi_coords['top'] + roi_coords['height']}). Focus analysis on this region if relevant."
    else:
        roi_info = "No specific region highlighted."

    prompt = DISEASE_SPECIFIC_PROMPT_TEMPLATE.format(disease=disease, roi_info=roi_info)
    result_text, success = query_gemini_vision(image, prompt)
    if success:
        logger.info(f"Disease analysis for '{disease}' successful.")
        return result_text or "[Analysis OK, No Text Returned]"
    else:
        logger.error(f"Disease analysis for '{disease}' failed: {result_text}")
        return f"Disease Analysis Failed ({disease}):\n{result_text}"

def estimate_ai_confidence(
    image: Image.Image,
    history: List[Tuple[str, str]]
    ) -> str:
    """Estimates confidence based on the last interaction."""
    if not history:
        return "No history available to estimate confidence on."
    logger.info("Requesting confidence estimation...")
    last_q, last_a = history[-1]
    prompt = CONFIDENCE_PROMPT_TEMPLATE.format(last_q=last_q, last_a=last_a)
    result_text, success = query_gemini_vision(image, prompt) # Image context helps
    if success:
        logger.info("Confidence estimation successful.")
        return result_text or "[Estimation OK, No Text Returned]"
    else:
        logger.error(f"Confidence estimation failed: {result_text}")
        return f"Confidence Estimation Failed:\n{result_text}"