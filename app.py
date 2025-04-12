# llm_interactions.py

import os
import io
import base64
import logging
import requests # For making API calls (if using REST)
# import google.generativeai as genai # Uncomment if using the official SDK
from PIL import Image
from typing import Optional, Dict, List, Tuple, Any

# --- Configuration ---
# Try to get the API key from Hugging Face secrets
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# Replace with the specific Gemini model endpoint you are using
# Check Google AI Studio or documentation for the correct model name/endpoint
GEMINI_VISION_API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent" # Example endpoint

# --- Logging Setup ---
logger = logging.getLogger(__name__)
# Basic logging configuration (adjust level and format as needed)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

# --- Helper: Convert PIL Image to Base64 ---
def image_to_base64(img: Image.Image, format="JPEG") -> Optional[str]:
    """Converts a PIL Image to a base64 encoded string."""
    if not isinstance(img, Image.Image):
        logger.error("Invalid input: not a PIL Image.")
        return None
    try:
        buffered = io.BytesIO()
        # Ensure image is in RGB format for JPEG compatibility
        rgb_img = img.convert("RGB")
        rgb_img.save(buffered, format=format)
        img_byte = buffered.getvalue()
        img_base64 = base64.b64encode(img_byte).decode("utf-8")
        return img_base64
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}", exc_info=True)
        return None

# --- Core Function to Query Gemini Vision API (using requests) ---
def query_gemini_vision(
    image: Image.Image,
    prompt: str,
    api_key: Optional[str] = GEMINI_API_KEY,
    api_endpoint: str = GEMINI_VISION_API_ENDPOINT
) -> Tuple[Optional[str], bool]:
    """
    Queries the Gemini Vision API with an image and a prompt using requests.

    Args:
        image: The PIL Image object.
        prompt: The text prompt for the model.
        api_key: The API key for authentication.
        api_endpoint: The specific Gemini model API endpoint URL.

    Returns:
        A tuple containing:
        - The extracted text response from the model (str) or an error message (str).
        - A boolean indicating success (True) or failure (False).
    """
    if not api_key:
        logger.error("Gemini API Key not found. Please set it as a Secret.")
        return "Error: API Key not configured.", False
    if not image:
        logger.error("No image provided for Gemini query.")
        return "Error: No image provided.", False

    logger.info(f"Querying Gemini Vision API: {api_endpoint}")
    logger.debug(f"Prompt: {prompt[:100]}...") # Log start of prompt

    img_base64 = image_to_base64(image)
    if not img_base64:
        return "Error: Failed to encode image.", False

    headers = {
        "Content-Type": "application/json",
    }
    # Construct the request data payload according to the specific Gemini API documentation
    # This is a common structure for gemini-pro-vision, adjust if needed
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_base64
                        }
                    }
                ]
            }
        ],
        # Add generationConfig if needed (e.g., temperature, max_output_tokens)
        # "generationConfig": {
        #     "temperature": 0.4,
        #     "topK": 32,
        #     "topP": 1,
        #     "maxOutputTokens": 4096,
        #     "stopSequences": []
        # }
    }
    params = {"key": api_key}

    try:
        response = requests.post(api_endpoint, headers=headers, json=data, params=params, timeout=90) # Increased timeout
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        response_data = response.json()
        # --- Parse the response ---
        # This parsing logic depends heavily on the Gemini API response structure.
        # Inspect the actual response or consult documentation.
        # Example for gemini-pro-vision:
        if "candidates" in response_data and response_data["candidates"]:
            first_candidate = response_data["candidates"][0]
            if "content" in first_candidate and "parts" in first_candidate["content"]:
                 text_parts = [part["text"] for part in first_candidate["content"]["parts"] if "text" in part]
                 full_text = "".join(text_parts).strip()
                 if full_text:
                     logger.info("Gemini query successful.")
                     logger.debug(f"Gemini response: {full_text[:100]}...")
                     return full_text, True
                 else:
                     logger.warning("Gemini response parsed but contained no text.")
                     return "Error: Received empty text response from API.", False
            else:
                logger.warning("Unexpected response structure: Missing 'content' or 'parts'.")
                return f"Error: Unexpected API response structure (missing content/parts). Response: {response_data}", False
        elif "promptFeedback" in response_data and "blockReason" in response_data["promptFeedback"]:
            reason = response_data["promptFeedback"]["blockReason"]
            logger.warning(f"Gemini request blocked. Reason: {reason}")
            return f"Error: Request blocked by API. Reason: {reason}", False
        else:
            logger.warning(f"Unexpected response structure: {response_data}")
            return f"Error: Unexpected API response structure. Response: {response_data}", False
        # --- End Parsing ---

    except requests.exceptions.RequestException as e:
        # Handle specific HTTP errors if possible
        http_status = e.response.status_code if e.response is not None else "N/A"
        logger.error(f"HTTP error occurred querying Gemini: {e} (Status: {http_status})", exc_info=True)
        # Provide more specific user feedback for common errors
        if http_status == 429:
             return "Error: API Rate Limit Exceeded. Please try again later.", False
        elif http_status == 400:
             return f"Error: Bad Request (400). Check API endpoint and request format. Details: {e.response.text if e.response is not None else 'N/A'}", False
        elif http_status in [401, 403]:
             return "Error: Authentication Failed (401/403). Check your API Key.", False
        else:
             return f"Error: Network or API error occurred (Status: {http_status}).", False
    except Exception as e:
        logger.critical(f"An unexpected error occurred during Gemini query: {e}", exc_info=True)
        return f"Error: An unexpected error occurred: {e}", False

# --- Main Analysis Functions ---

def run_initial_analysis(image: Image.Image, roi: Optional[Dict] = None) -> str:
    """
    Runs the initial overview analysis on the image, potentially focusing on an ROI.

    Args:
        image: The PIL Image object.
        roi: Optional dictionary with ROI coordinates {'left', 'top', 'width', 'height'}.

    Returns:
        A string containing the analysis findings or an error message.
    """
    logger.info(f"Running initial analysis. ROI provided: {bool(roi)}")

    # --- Define the prompt, potentially incorporating ROI info ---
    base_prompt = (
        "You are a helpful AI assistant analyzing medical images for informational purposes only. "
        "Provide a structured analysis of the key findings in this image. "
        "Describe any potential abnormalities or notable features. "
        "Structure your response with 'Findings:' and 'Impression:' sections."
    )

    if roi:
        # Example: Modify prompt to mention ROI
        # You could make this more detailed based on the roi dict keys
        roi_coords = f"left={roi.get('left')}, top={roi.get('top')}, width={roi.get('width')}, height={roi.get('height')}"
        prompt = f"{base_prompt} Pay special attention to the region of interest defined by the approximate coordinates: {roi_coords}."
        logger.info("ROI provided, modifying prompt for focused analysis.")
    else:
        prompt = base_prompt

    # --- Call the Gemini API ---
    result_text, success = query_gemini_vision(image, prompt)

    if success:
        return result_text
    else:
        # query_gemini_vision already includes "Error:" prefix in its error messages
        return result_text # Return the error message from the query function

def run_multimodal_qa(
    image: Image.Image,
    question: str,
    history: List[Tuple[str, str]],
    roi: Optional[Dict] = None
) -> Tuple[str, bool]:
    """
    Answers a specific question about the image, considering history and potential ROI.

    Args:
        image: The PIL Image object.
        question: The user's question string.
        history: List of previous (question, answer) tuples for context.
        roi: Optional dictionary with ROI coordinates.

    Returns:
        A tuple containing:
        - The answer string or an error message.
        - Boolean indicating success.
    """
    logger.info(f"Running multimodal Q&A. ROI provided: {bool(roi)}. History length: {len(history)}")

    # --- Construct the prompt, including history and question ---
    prompt_parts = [
        "You are a helpful AI assistant answering questions about a medical image. Answer the following question based on the image provided.",
        "Conversation History:"
    ]
    if history:
        for i, (q, a) in enumerate(history[-3:]): # Include last 3 turns for context
            prompt_parts.append(f"Q{i+1}: {q}")
            prompt_parts.append(f"A{i+1}: {a}")
    else:
        prompt_parts.append("(No history)")

    prompt_parts.append("\nCurrent Question:")
    prompt_parts.append(question)

    if roi:
        roi_coords = f"left={roi.get('left')}, top={roi.get('top')}, width={roi.get('width')}, height={roi.get('height')}"
        prompt_parts.append(f"\nPlease consider the region of interest defined by coordinates: {roi_coords} when answering.")
        logger.info("ROI provided, adding ROI context to Q&A prompt.")

    full_prompt = "\n".join(prompt_parts)

    # --- Call the Gemini API ---
    answer_text, success = query_gemini_vision(image, full_prompt)

    # Return the result directly (includes error messages on failure)
    return answer_text, success

def run_disease_analysis(
    image: Image.Image,
    disease: str,
    roi: Optional[Dict] = None
) -> str:
    """
    Analyzes the image specifically for signs of a given disease/condition.

    Args:
        image: The PIL Image object.
        disease: The name of the disease/condition to check for.
        roi: Optional dictionary with ROI coordinates.

    Returns:
        A string with the focused analysis or an error message.
    """
    logger.info(f"Running disease-specific analysis for '{disease}'. ROI provided: {bool(roi)}")

    # --- Construct the prompt ---
    base_prompt = (
        f"Analyze the provided medical image specifically for signs of **{disease}**. "
        f"Describe any findings consistent or inconsistent with {disease}. "
        "Provide a clear assessment regarding this specific condition."
    )
    if roi:
        roi_coords = f"left={roi.get('left')}, top={roi.get('top')}, width={roi.get('width')}, height={roi.get('height')}"
        prompt = f"{base_prompt} Focus your analysis within the region of interest defined by coordinates: {roi_coords} if relevant, but also consider the overall image context."
        logger.info("ROI provided, adding ROI context to disease analysis prompt.")
    else:
        prompt = base_prompt

    # --- Call the Gemini API ---
    result_text, success = query_gemini_vision(image, prompt)

    if success:
        return result_text
    else:
        return result_text # Return the error message

def estimate_ai_confidence(
    image: Image.Image, # Image might be needed if confidence depends on image quality/features
    history: List[Tuple[str, str]],
    initial_analysis: Optional[str] = None,
    disease_analysis: Optional[str] = None,
    roi: Optional[Dict] = None # Note if ROI was used in generating the analyses
    ) -> str:
    """
    Estimates the AI's confidence based on the analyses performed.
    (This is a simplified example - real confidence estimation is complex).

    Args:
        image: The PIL image object (may be unused in simple implementations).
        history: List of Q&A tuples.
        initial_analysis: String of the initial analysis.
        disease_analysis: String of the disease-specific analysis.
        roi: Optional ROI dict used for the analyses.

    Returns:
        A string describing the estimated confidence level and rationale.
    """
    logger.info(f"Estimating AI confidence. ROI used in preceding analyses: {bool(roi)}")

    # --- Simple Example Confidence Logic (Placeholder) ---
    # This needs to be replaced with a more sophisticated method, potentially
    # involving analyzing the text for uncertainty markers, checking for API errors, etc.
    # Or calling another LLM prompt specifically designed for confidence assessment.

    confidence_level = "Moderate" # Default guess
    rationale = []

    analysis_items = []
    if initial_analysis and not initial_analysis.startswith("Error:"):
        analysis_items.append(initial_analysis)
        rationale.append("Initial analysis provided some findings.")
    if disease_analysis and not disease_analysis.startswith("Error:"):
        analysis_items.append(disease_analysis)
        rationale.append("Disease-specific analysis was performed.")
    if history:
        # Check if last answer was successful
        if history[-1][1] and not history[-1][1].startswith("Error:"):
             rationale.append(f"{len(history)} question(s) were answered.")
        else:
             rationale.append(f"{len(history)} question(s) were asked, but the last answer may have failed.")
             confidence_level = "Low-Moderate" # Downgrade if last Q&A failed

    if not analysis_items and not history:
         return "Confidence cannot be estimated: No analysis or Q&A performed yet."

    # Example: Look for uncertainty words in the analyses (very basic)
    uncertainty_markers = ["possible", "suggests", "could be", "unclear", "difficult to ascertain", "consider"]
    found_uncertainty = False
    combined_text = " ".join(analysis_items)
    for marker in uncertainty_markers:
        if marker in combined_text.lower():
            found_uncertainty = True
            rationale.append(f"Analysis includes terms indicating uncertainty (e.g., '{marker}').")
            break

    if not found_uncertainty and analysis_items:
        confidence_level = "Moderate-High"
        rationale.append("Analyses did not contain strong uncertainty markers.")

    if bool(roi):
        rationale.append("Analysis may have been focused on a specific Region of Interest.")


    # --- Construct the confidence response ---
    response = f"**Estimated Confidence:** {confidence_level}\n\n**Rationale:**\n"
    for i, r in enumerate(rationale):
        response += f"- {r}\n"
    response += "\n**Disclaimer:** This is a heuristic estimation. Always verify findings with a qualified professional."

    return response