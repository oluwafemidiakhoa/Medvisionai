# -*- coding: utf-8 -*-
"""
llm_interactions.py - Gemini API Interaction Module
===================================================

Handles interactions with the Google Gemini API for multimodal analysis
within the RadVision AI application.

Core functionalities:
- Generating initial radiological analysis from images.
- Answering user questions in the context of an image and conversation history.
- Performing analysis focused on specific suspected conditions.
- Estimating the AI's confidence in its previous statements.
- Optionally enriching responses with mapped UMLS concepts using umls_utils.
"""

import os
import io
import base64
import logging
import requests # Dependency: Make sure 'requests' is installed
from typing import Optional, Tuple, List, Dict, Any, TypeAlias
from PIL import Image # Dependency: Make sure 'Pillow' is installed

# Local application imports (assuming they are in the python path)
try:
    # Import specific exceptions and the search function
    from umls_utils import search_umls, UMLSConcept, UMLSAuthError, UMLSSearchError
    _UMLS_UTILS_AVAILABLE = True
except ImportError:
    _UMLS_UTILS_AVAILABLE = False
    # Define dummy types/exceptions if umls_utils is missing, so the rest loads
    UMLSConcept = Any
    UMLSAuthError = RuntimeError
    UMLSSearchError = RuntimeError
    def search_umls(*args, **kwargs):
        raise NotImplementedError("umls_utils module not found.")

# --- Module Configuration & Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
IMAGE_MIME_TYPE: str = "image/png"  # Gemini prefers PNG, JPEG, WEBP
API_TIMEOUT: int = 180              # Max time (seconds) for API request
DEFAULT_MODEL_NAME: str = "gemini-1.5-flash-latest" # Default model (Flash is often faster/cheaper)

# --- Environment Variable Configuration ---
def_env = os.getenv # Shortcut

# Gemini Configuration
GEMINI_API_KEY: Optional[str] = def_env("GEMINI_API_KEY")
GEMINI_MODEL_NAME: str = def_env("GEMINI_MODEL_OVERRIDE", DEFAULT_MODEL_NAME)

# UMLS Configuration (defaults sourced from environment or hardcoded)
UMLS_API_KEY: Optional[str] = def_env("UMLS_APIKEY")
DEFAULT_UMLS_HITS: int = int(def_env("UMLS_HITS", "3")) # Number of UMLS concepts to map
# Optional filter for specific UMLS sources (comma-separated string -> list)
UMLS_SOURCE_FILTER_STR: Optional[str] = def_env("UMLS_SOURCE_FILTER")
UMLS_SOURCE_FILTER: List[str] = UMLS_SOURCE_FILTER_STR.split(',') if UMLS_SOURCE_FILTER_STR else []

# --- Availability Checks & Messages (for app integration) ---
GEMINI_AVAILABLE: bool = bool(GEMINI_API_KEY)
GEMINI_CONFIG_MSG: str = "Set `GEMINI_API_KEY` in Hugging Face Secrets & restart." if not GEMINI_AVAILABLE else "Gemini API key configured."

UMLS_AVAILABLE_FOR_LLM: bool = _UMLS_UTILS_AVAILABLE and bool(UMLS_API_KEY)
UMLS_LLM_CONFIG_MSG: str = ("`umls_utils` not found or `requests` not installed." if not _UMLS_UTILS_AVAILABLE
                           else "Set `UMLS_APIKEY` in Hugging Face Secrets & restart." if not UMLS_API_KEY
                           else "UMLS mapping configured.")

logger.info(f"Using Gemini model: {GEMINI_MODEL_NAME}")
logger.info(f"Gemini Available: {GEMINI_AVAILABLE} ({GEMINI_CONFIG_MSG})")
logger.info(f"UMLS Available for LLM: {UMLS_AVAILABLE_FOR_LLM} ({UMLS_LLM_CONFIG_MSG})")
if UMLS_SOURCE_FILTER:
    logger.info(f"UMLS Source Filter Applied: {UMLS_SOURCE_FILTER}")

# --- Type Aliases for Clarity ---
# Assuming history stores (Question: str, Answer: str, Optional[UMLSConcepts]: Any)
HistoryEntry: TypeAlias = Tuple[str, str, Any]
RoiDict: TypeAlias = Dict[str, Any] # Expected keys: left, top, width, height

# --- Prompt Templates (using f-string interpolation) ---
# Base prompt defining the AI's persona
_BASE_ROLE_PROMPT: str = """
You are an advanced AI assistant with expertise in medical imaging analysis, incorporating knowledge from radiology, pathology, and clinical diagnostics.
Your capabilities include:
1. Multi-modality image analysis (X-ray, CT, MRI, Ultrasound, Pathology)
2. Standardized reporting using DICOM standards
3. Integration with medical terminology (UMLS, SNOMED-CT, ICD-10)
4. Region-specific analysis with anatomical precision

IMPORTANT MEDICAL DISCLAIMER: 
- This analysis is for research and educational purposes ONLY
- NOT for clinical diagnosis or medical decision-making
- Always consult qualified healthcare professionals
- Analysis is based on single images without full clinical context

Guidelines:
1. Use precise medical terminology with anatomical references
2. Provide structured, hierarchical findings
3. Include measurement estimates when relevant
4. Note image quality and technical limitations
5. Suggest differential diagnoses with confidence levels
6. Reference standardized reporting systems when applicable
"""

# Template for initial, comprehensive analysis
INITIAL_ANALYSIS_PROMPT_TEMPLATE: str = f"""{_BASE_ROLE_PROMPT}
**Task:** Provide a comprehensive initial radiological analysis of the provided medical image.
**Region of Interest (ROI):** {{roi_info}}

**Output Structure:**
1.  **Image Description:** Briefly describe the type of image, view, and overall quality.
2.  **Key Findings:** List the most significant observations, both normal and abnormal. Be specific about location, size, and characteristics. Use bullet points.
3.  **Potential Differential Diagnoses:** Based on the findings, suggest a ranked list of possible conditions or diagnoses. Briefly explain the reasoning for the top 1-2 differentials.
4.  **Recommendations/Limitations:** Suggest potential next steps (e.g., comparison with priors, other imaging modalities) and mention any limitations of the analysis based on the single image provided.
"""

# Template for answering questions based on image and history
QA_CONTEXT_PROMPT_TEMPLATE: str = f"""{_BASE_ROLE_PROMPT}
**Task:** Answer the user's question accurately based on the provided medical image and the preceding conversation context.
**Region of Interest (ROI):** {{roi_info}}

**Relevant Conversation History (Last few turns):**
{{history_text}}
---
**User's Current Question:** "{{question}}"

**Instructions:** Address the user's question directly. Refer back to the image findings and previous statements if relevant. If the question cannot be answered from the image, state that clearly. Maintain a helpful and informative tone.
"""

# Template for estimating confidence in the *previous* response
CONFIDENCE_PROMPT_TEMPLATE: str = f"""{_BASE_ROLE_PROMPT}
**Task:** Assess the confidence level of your *immediately preceding* response, considering the image information and the nature of the query.
**Region of Interest (ROI):** {{roi_info}}

**Context:**
*   **User's Last Question:** "{{last_q}}"
*   **Your Last Answer:** "{{last_a}}"

**Output:** Provide a confidence score (e.g., High, Medium, Low) and a brief justification (1-2 sentences). Consider factors like ambiguity in the image, the specificity of the question, and the reliance on interpretation versus direct observation.
"""

# Template for analyzing the image specifically for a given disease/condition
DISEASE_SPECIFIC_PROMPT_TEMPLATE: str = f"""{_BASE_ROLE_PROMPT}
**Task:** Focus the analysis of the provided medical image *specifically* for signs of **{{disease}}**. Ignore findings unrelated to this condition unless critically relevant.
**Region of Interest (ROI):** {{roi_info}}

**Output Structure:**
1.  **Presence of Findings Consistent with {{disease}}:** State clearly whether signs consistent with the specified disease are present, absent, or indeterminate.
2.  **Description of Relevant Findings:** If findings are present, describe them in detail (location, size, characteristics relevant to {{disease}}). Use bullet points.
3.  **Assessment:** Based ONLY on findings relevant to {{disease}}, provide a brief assessment (e.g., 'Findings highly suggestive of...', 'Possible subtle signs...', 'No clear evidence of...').
4.  **Limitations:** Note any limitations in assessing for {{disease}} based on this specific image (e.g., view, quality, need for other tests).
"""

# --- Helper Functions ---

def _encode_image(image: Image.Image) -> Tuple[Optional[str], str]:
    """Encodes a PIL Image to base64 PNG string for Gemini API.

    Args:
        image: The PIL Image object.

    Returns:
        A tuple containing the base64 encoded string (or None on error)
        and the MIME type string (or an error message).
    """
    if not isinstance(image, Image.Image):
        logger.error("Invalid input: _encode_image expects a PIL Image.")
        return None, "Error: Input must be a PIL Image object."
    try:
        # Convert to RGB if necessary (e.g., RGBA, P mode) - Gemini supports various modes but PNG often works best
        if image.mode not in ('RGB', 'L'): # L is grayscale
             img_to_save = image.convert('RGB')
             logger.debug(f"Converted image mode from {image.mode} to RGB for encoding.")
        else:
             img_to_save = image

        buffered = io.BytesIO()
        img_to_save.save(buffered, format="PNG") # Save as PNG
        encoded_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
        logger.info(f"Image successfully encoded to base64 PNG ({len(encoded_data)} chars).")
        return encoded_data, IMAGE_MIME_TYPE
    except Exception as e:
        logger.exception("Image encoding to base64 PNG failed.") # Log full traceback
        return None, f"Error encoding image: {e}"

def _format_roi(roi: Optional[RoiDict]) -> str:
    """Formats ROI information for inclusion in prompts."""
    if not roi or not isinstance(roi, dict):
        return "Whole image analysis (no specific region highlighted by user)."
    try:
        # Validate keys and attempt conversion to int, handle potential errors
        left = int(roi.get("left", 0))
        top = int(roi.get("top", 0))
        width = int(roi.get("width", 0))
        height = int(roi.get("height", 0))
        if width <= 0 or height <= 0:
            return "ROI provided but has invalid dimensions (width/height <= 0)."
        return (f"Analysis focused on Region of Interest (ROI) at coordinates "
                f"(left={left}, top={top}) with size (width={width}, height={height}) pixels.")
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not format ROI dictionary due to invalid value type: {roi}. Error: {e}")
        return "ROI coordinates provided but could not be interpreted (invalid format)."
    except Exception as e: # Catch unexpected errors
        logger.error(f"Unexpected error formatting ROI: {roi}. Error: {e}")
        return "Error processing provided ROI information."

def _format_history(history: List[HistoryEntry]) -> str:
    """Formats the last few turns of conversation history for prompts."""
    if not history:
        return "No previous conversation history."
    # Take last N turns (e.g., 3) to keep prompt concise
    num_turns_to_include = 3
    recent_history = history[-num_turns_to_include:]
    formatted_turns = []
    for i, (question, answer, _) in enumerate(recent_history):
        # Simple Q/A format
        turn = f"Turn {len(history) - len(recent_history) + i + 1}:\nUser: {question}\nAI: {answer}"
        formatted_turns.append(turn)
    return "\n---\n".join(formatted_turns) if formatted_turns else "No recent history to display."

# --- Core Gemini API Interaction ---

def query_gemini_vision(image: Image.Image, text_prompt: str) -> Tuple[str, bool]:
    """Sends image and text prompt to Gemini Vision API and returns the response.

    Args:
        image: The PIL Image object.
        text_prompt: The text prompt to accompany the image.

    Returns:
        A tuple containing:
        - The response text from the API (str). On error, this contains an error message.
        - A boolean indicating success (True) or failure (False).
    """
    if not GEMINI_AVAILABLE:
        logger.error("Gemini API key not configured. Cannot query API.")
        return "Configuration Error: Gemini API key not found.", False
    if not isinstance(image, Image.Image):
         logger.error("Invalid image object provided to query_gemini_vision.")
         return "Internal Error: Invalid image object.", False

    # Encode the image
    img_base64_data, mime_type_or_error = _encode_image(image)
    if img_base64_data is None:
        logger.error(f"Image encoding failed: {mime_type_or_error}")
        return f"Internal Error: {mime_type_or_error}", False # Return the encoding error

    # --- Prepare API Request ---
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}

    # Construct the payload according to Gemini API spec
    payload = {
        "contents": [{
            "parts": [
                {"text": text_prompt},
                {"inline_data": {"mime_type": mime_type_or_error, "data": img_base64_data}}
            ]
        }],
        # Configuration for generation - adjust as needed
        "generation_config": {
            "temperature": 0.3,      # Lower temperature for more factual/less creative responses
            "top_k": 32,
            "top_p": 0.95,
            "max_output_tokens": 8192 # Max possible tokens
        },
        # Safety settings to block harmful content
        "safety_settings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]
    }

    # --- Make API Call ---
    try:
        logger.info(f"Sending request to Gemini model {GEMINI_MODEL_NAME}...")
        response = requests.post(
            api_url,
            headers=headers,
            params=params,
            json=payload,
            timeout=API_TIMEOUT
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # --- Process Successful Response ---
        data = response.json()
        candidates = data.get('candidates', [])
        if not candidates:
             prompt_feedback = data.get("promptFeedback")
             if prompt_feedback and prompt_feedback.get("blockReason"):
                 block_reason = prompt_feedback.get("blockReason")
                 block_details = prompt_feedback.get("safetyRatings")
                 logger.warning(f"Gemini request blocked. Reason: {block_reason}. Details: {block_details}")
                 return f"API Error: Request blocked due to safety settings (Reason: {block_reason}).", False
             else:
                 logger.warning("Gemini API returned no candidates and no block reason.")
                 return "API Error: No response candidates received from Gemini.", False

        # Access the first candidate's content
        content = candidates[0].get('content', {})
        parts = content.get('parts', [{}])
        text_response = parts[0].get('text', '').strip()

        # Check for finish reason (e.g., safety, recitation, length)
        finish_reason = candidates[0].get("finishReason")
        if finish_reason and finish_reason != "STOP":
             logger.warning(f"Gemini generation finished unexpectedly. Reason: {finish_reason}")
             if text_response:
                 return f"{text_response}\n\n(Warning: Output may be incomplete due to finish reason: {finish_reason})", True
             else:
                 return f"API Error: Generation failed or was stopped. Reason: {finish_reason}.", False

        if not text_response:
            logger.warning("Gemini API returned empty text response in candidates.")
            return "API Error: Received an empty response from Gemini.", False

        logger.info("Successfully received response from Gemini.")
        return text_response, True

    # --- Handle Errors ---
    except requests.exceptions.Timeout:
        logger.error(f"Gemini API request timed out after {API_TIMEOUT} seconds.")
        return f"API Error: Request timed out after {API_TIMEOUT} seconds.", False
    except requests.exceptions.HTTPError as http_err:
        error_msg = f"HTTP Error {http_err.response.status_code} calling Gemini API."
        try:
            error_details = http_err.response.json()
            api_error = error_details.get("error", {})
            message = api_error.get("message", http_err.response.text)
            error_msg += f" Details: {message}"
            logger.error(f"{error_msg} | Response: {error_details}")
        except Exception:
             logger.error(f"{error_msg} | Raw Response: {http_err.response.text}", exc_info=True)
        return error_msg, False
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Network error connecting to Gemini API: {req_err}", exc_info=True)
        return f"API Network Error: {req_err}", False
    except Exception as e:
        logger.exception("An unexpected error occurred during Gemini API interaction.")
        return f"Internal Error: An unexpected error occurred ({type(e).__name__}).", False

# --- UMLS Integration Helper ---

def _append_umls_info(
    response_text: str,
    num_hits: Optional[int] = None
) -> str:
    """Appends formatted UMLS concepts found in the response text.

    Args:
        response_text: The text generated by the LLM.
        num_hits: Max number of UMLS concepts per search (overrides default).

    Returns:
        The original text potentially appended with a UMLS concepts section.
        Returns original text if UMLS is unavailable, fails, or finds no concepts.
    """
    if not UMLS_AVAILABLE_FOR_LLM:
        logger.debug("UMLS mapping skipped: UMLS not available or API key missing.")
        return response_text
    if not _UMLS_UTILS_AVAILABLE:
        logger.error("Attempted UMLS mapping but umls_utils is not available.")
        return response_text

    hits_to_get = num_hits if isinstance(num_hits, int) and num_hits > 0 else DEFAULT_UMLS_HITS

    logger.info(f"Attempting to map UMLS concepts in response (max hits: {hits_to_get})...")
    try:
        # Extract key medical terms instead of using the entire response text
        # This is more reliable and less likely to cause API errors
        import re
        
        # Find key medical terms (diagnostic terms, anatomical structures, etc.)
        # Extract terms that might be diseases, findings, or important medical concepts
        potential_terms = []
        
        # Look for diagnostic phrases often found in medical reports
        diagnostic_findings = re.findall(r'(?:diagnosis|impression|finding)s?[:\s]+([^\.;]+)[\.;]', 
                                        response_text, re.IGNORECASE)
        if diagnostic_findings:
            potential_terms.extend(diagnostic_findings)
            
        # Look for anatomical structures with abnormalities
        anatomical_terms = re.findall(r'(?:opacity|mass|lesion|effusion|consolidation|nodule)s?\s+(?:in|of|at)?\s+(?:the)?\s+([^\.;,]+)', 
                                     response_text, re.IGNORECASE)
        if anatomical_terms:
            potential_terms.extend(anatomical_terms)
            
        # Extract common disease names that might appear
        disease_terms = re.findall(r'(?:pneumonia|edema|fracture|cancer|tumor|malignancy|inflammation)(?:[^\.\n;]*)', 
                                  response_text, re.IGNORECASE)
        if disease_terms:
            potential_terms.extend(disease_terms)
        
        # If we found specific terms, use them
        if potential_terms:
            search_text = " ".join(potential_terms[:3])  # Limit to top 3 terms
            logger.info(f"Searching UMLS with extracted key terms: {search_text}")
        else:
            # Use a shorter version of the response as fallback
            search_text = response_text[:300]
            logger.info("No specific medical terms extracted, using truncated response")
        
        concepts: List[UMLSConcept] = search_umls(search_text, UMLS_API_KEY, page_size=hits_to_get)

        if UMLS_SOURCE_FILTER:
            original_count = len(concepts)
            concepts = [c for c in concepts if c.rootSource in UMLS_SOURCE_FILTER]
            logger.debug(f"Filtered UMLS concepts by source ({UMLS_SOURCE_FILTER}): {original_count} -> {len(concepts)}")

        if not concepts:
            logger.info("No relevant UMLS concepts found or mapped for the response text.")
            return response_text

        entries = []
        seen_cuis = set()
        for concept in concepts:
            if concept.ui not in seen_cuis:
                name_display = f"[{concept.name}]({concept.uri})" if concept.uri else concept.name
                entries.append(
                    f"- {name_display} | CUI: `{concept.ui}` | Source: {concept.rootSource}"
                )
                seen_cuis.add(concept.ui)

        if not entries:
            return response_text

        umls_block = "\n\n---\n**Mapped UMLS Concepts (Top Matches):**\n" + "\n".join(entries)
        logger.info(f"Successfully appended {len(entries)} unique UMLS concepts to response.")
        return response_text + umls_block

    except (UMLSAuthError, UMLSSearchError) as umls_err:
        logger.error(f"UMLS mapping failed due to UMLS error: {umls_err}")
        return response_text
    except Exception as e:
        logger.exception("An unexpected error occurred during UMLS mapping.")
        return response_text

# --- Public Interaction Functions ---

def run_initial_analysis(
    image: Image.Image,
    roi: Optional[RoiDict] = None,
    umls_hits: Optional[int] = None
) -> str:
    """Generates the initial comprehensive analysis for an image.

    Args:
        image: The PIL Image to analyze.
        roi: Optional dictionary describing the region of interest.
        umls_hits: Optional override for number of UMLS concepts to map.

    Returns:
        The formatted analysis string, potentially including UMLS terms.
        Returns an error message string on failure.
    """
    logger.info("Running initial analysis...")
    roi_info = _format_roi(roi)
    prompt = INITIAL_ANALYSIS_PROMPT_TEMPLATE.format(roi_info=roi_info)

    response_text, success = query_gemini_vision(image, prompt)

    if not success:
        logger.error(f"Initial analysis failed: {response_text}")
        return f"**Initial Analysis Failed:** {response_text}"

    return _append_umls_info(response_text, num_hits=umls_hits)


def run_multimodal_qa(
    image: Image.Image,
    question: str,
    history: List[HistoryEntry],
    roi: Optional[RoiDict] = None,
    umls_hits: Optional[int] = None
) -> Tuple[str, bool]:
    """Answers a question based on the image, history, and optional ROI.

    Args:
        image: The PIL Image context.
        question: The user's question string.
        history: List of previous (question, answer, umls) tuples.
        roi: Optional dictionary describing the region of interest.
        umls_hits: Optional override for number of UMLS concepts to map.

    Returns:
        A tuple containing:
            - The answer string (potentially with UMLS terms) or an error message.
            - Boolean indicating success.
    """
    logger.info(f"Running multimodal Q&A for question: '{question}'")
    if not question:
        logger.warning("Multimodal QA called with empty question.")
        return "Error: Please provide a question.", False

    roi_info = _format_roi(roi)
    history_text = _format_history(history)
    prompt = QA_CONTEXT_PROMPT_TEMPLATE.format(
        roi_info=roi_info,
        history_text=history_text,
        question=question
    )

    response_text, success = query_gemini_vision(image, prompt)

    if not success:
        logger.error(f"Multimodal QA failed: {response_text}")
        return f"**Q&A Failed:** {response_text}", False

    final_response = _append_umls_info(response_text, num_hits=umls_hits)
    return final_response, True


def run_disease_analysis(
    image: Image.Image,
    disease: str,
    roi: Optional[RoiDict] = None,
    umls_hits: Optional[int] = None
) -> str:
    """Performs analysis focused on a specific disease.

    Args:
        image: The PIL Image to analyze.
        disease: The name of the disease/condition to focus on.
        roi: Optional dictionary describing the region of interest.
        umls_hits: Optional override for number of UMLS concepts to map.

    Returns:
        The formatted analysis string, potentially including UMLS terms.
        Returns an error message string on failure.
    """
    logger.info(f"Running disease-specific analysis for: '{disease}'")
    if not disease:
        logger.warning("Disease-specific analysis called with empty disease name.")
        return "Error: Please specify a disease or condition for analysis."

    roi_info = _format_roi(roi)
    disease_clean = disease.strip()
    prompt = DISEASE_SPECIFIC_PROMPT_TEMPLATE.format(disease=disease_clean, roi_info=roi_info)

    response_text, success = query_gemini_vision(image, prompt)

    if not success:
        logger.error(f"Disease analysis failed ({disease_clean}): {response_text}")
        return f"**Disease Analysis Failed ({disease_clean}):** {response_text}"

    return _append_umls_info(response_text, num_hits=umls_hits)


def estimate_ai_confidence(
    image: Image.Image,
    history: List[Any],
    roi: Optional[RoiDict] = None
) -> str:
    """Estimates the AI's confidence in its last response based on history.

    Args:
        image: The PIL Image context (may be used by LLM).
        history: List of previous interactions. 
                Can be (question, answer, umls) or (role, message) tuples.
        roi: Optional dictionary describing the region of interest.

    Returns:
        A string describing the confidence level and justification.
        Returns an error message string on failure or if history is empty.
    """
    logger.info("Estimating AI confidence for the last response...")
    if not history:
        logger.warning("Confidence estimation called with empty history.")
        return "Confidence Estimation Failed: No conversation history available to evaluate."

    # First determine the format of the history entries - could be (q, a, umls) or (role, msg)
    last_q = "No question available"
    last_a = "No previous analysis available"
    
    try:
        # Try to get the last question/answer pair
        if history and isinstance(history[-1], tuple):
            if len(history[-1]) == 3:  # (question, answer, umls) format
                last_q, last_a, _ = history[-1]
            elif len(history[-1]) == 2:  # (role, message) format
                # Find last user question and AI answer
                for i in range(len(history)-1, -1, -1):
                    role, message = history[i]
                    if isinstance(role, str) and "user" in role.lower():
                        last_q = message
                        break
                        
                for i in range(len(history)-1, -1, -1):
                    role, message = history[i]
                    if isinstance(role, str) and "ai" in role.lower():
                        last_a = message
                        break
    except Exception as e:
        logger.warning(f"Error parsing history for confidence estimation: {e}")
        # Continue with defaults instead of failing completely
    
    # If no history is usable, use other available analysis
    if last_q == "No question available" or last_a == "No previous analysis available":
        logger.info("Using available analysis for confidence estimation since history format is unclear")
        
    roi_info = _format_roi(roi)
    prompt = CONFIDENCE_PROMPT_TEMPLATE.format(last_q=last_q, last_a=last_a, roi_info=roi_info)

    response_text, success = query_gemini_vision(image, prompt)

    if not success:
        logger.error(f"Confidence estimation failed: {response_text}")
        return f"**Confidence Estimation Failed:** {response_text}"
    else:
        return f"**AI Confidence Assessment:**\n{response_text}"
