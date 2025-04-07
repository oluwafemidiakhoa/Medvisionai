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
DEFAULT_MODEL_NAME: str = "gemini-1.5-pro-latest"  # Default stable model name
GEMINI_MODEL_NAME: str = os.environ.get("GEMINI_MODEL_OVERRIDE", "gemini-2.5-pro-exp-03-25")

# --- Prompt Templates ---
INITIAL_ANALYSIS_PROMPT = """
You are an expert AI assistant simulating a radiologist. Your task is to analyze the provided medical image.
Based *only* on the visual information in the image, provide:
1. **Detailed Description:** Describe the image content, anatomical structures, and view.
2. **Key Findings:** List any notable abnormalities.
3. **Potential Differential Diagnoses:** List potential diagnoses, ordered from most likely to least likely.
4. **Reasoning for Top Diagnosis:** Explain the evidence supporting the top diagnosis.
Structure your response clearly using these headings. Do not provide medical advice.
"""

QA_CONTEXT_PROMPT_TEMPLATE = """
You are a medical expert AI assisting with the interpretation of a medical image.

**Region of Interest:** {roi_info}
**Conversation History (Most Recent First):**
{history_text}

**Current Question:** "{question}"

Analyze the provided image in light of the above context and provide a concise, clinically relevant answer.
If the image lacks sufficient information, state that clearly.
"""

CONFIDENCE_PROMPT_TEMPLATE = """
Based on your most recent response:
---
Last Question/Task: {last_q}
Your Last Response:
{last_a}
---
Critically evaluate your confidence in your analysis on a scale of 1 to 10.
Respond in this format:
**Confidence:** [Score]/10
**Justification:** [Brief explanation of factors affecting confidence].
"""

DISEASE_SPECIFIC_PROMPT_TEMPLATE = """
You are an expert radiologist AI. Focus exclusively on analyzing the provided image for signs of **{disease}**.

**Region of Interest:** {roi_info}

Provide:
1. **Findings:** Indicate whether signs of {disease} are present, absent, or indeterminate.
2. **Description:** Describe any relevant visual findings.
3. **Severity Assessment:** Qualitatively assess severity if applicable.
4. **Next Steps:** Suggest image-based recommendations (without giving definitive medical advice).
If no signs are present, state so clearly.
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
        logger.error("GEMINI_API_KEY is not set.")
        return "Error: Gemini API key not configured.", False

    gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": gemini_api_key
    }
    
    # Convert the image to PNG and then base64-encode it.
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        logger.error(f"Error encoding image: {e}", exc_info=True)
        return f"Error encoding image: {e}", False
    
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
            {"category": cat, "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            for cat in [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT"
            ]
        ]
    }
    
    logger.debug(f"Payload constructed with keys: {list(payload.keys())}")
    
    try:
        response = requests.post(gemini_api_url, headers=headers, json=payload, timeout=API_TIMEOUT)
        response.raise_for_status()
        response_data: Dict[str, Any] = response.json()
        logger.debug(f"Gemini API response: {response_data}")
        
        if 'candidates' in response_data and isinstance(response_data['candidates'], list) and response_data['candidates']:
            candidate = response_data['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content'] and candidate['content']['parts']:
                if 'text' in candidate['content']['parts'][0]:
                    parsed_text = candidate['content']['parts'][0]['text']
                    return parsed_text, True
                else:
                    logger.warning("Text not found in candidate parts.")
                    return "Error: Response did not contain text.", False
            else:
                logger.warning("Candidate content structure is invalid.")
                return "Error: Invalid response structure.", False
        elif 'promptFeedback' in response_data:
            feedback = response_data['promptFeedback']
            block_reason = feedback.get('blockReason', 'UNKNOWN')
            logger.warning(f"Prompt blocked: {block_reason}")
            return f"Error: Prompt blocked (Reason: {block_reason}).", False
        else:
            logger.error("Unexpected response format from Gemini API.")
            return "Error: Unexpected response format.", False
    except requests.exceptions.Timeout:
        logger.error(f"Gemini API request timed out after {API_TIMEOUT} seconds.")
        return "Error: Gemini API request timed out.", False
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error: {e}", exc_info=True)
        return f"Error: HTTP error {e.response.status_code}.", False
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during Gemini API request: {e}", exc_info=True)
        return "Error: Network error occurred during API request.", False
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        return "Error: An unexpected error occurred.", False

# --- LLM Interaction Functions ---
def run_initial_analysis(image: Image.Image) -> str:
    """
    Performs an initial analysis of the image using the Gemini API.
    
    Returns:
        The analysis response text.
    """
    prompt = INITIAL_ANALYSIS_PROMPT
    response_text, success = query_gemini_vision(image, prompt)
    if success and response_text:
        return response_text
    else:
        return f"Initial analysis failed: {response_text}"

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
    roi_info = "No specific region highlighted."
    if roi and all(key in roi for key in ["left", "top", "width", "height"]):
        roi_info = f"ROI: Top-Left=({roi['left']}, {roi['top']}), Width={roi['width']}, Height={roi['height']}"
    
    history_text = "\n---\n".join([f"User: {q}\nAI: {a}" for q, a in history[::-1]])
    if not history_text:
        history_text = "No previous conversation history."
    
    prompt = QA_CONTEXT_PROMPT_TEMPLATE.format(
        roi_info=roi_info,
        history_text=history_text,
        question=question
    )
    
    response_text, success = query_gemini_vision(image, prompt)
    return response_text if response_text else "No response received.", success

def run_disease_analysis(image: Image.Image, disease: str, roi: Optional[Dict] = None) -> str:
    """
    Runs a focused analysis for a specified condition using the Gemini API.
    
    Args:
        image: The PIL Image to analyze.
        disease: The condition to focus on.
        roi: Optional ROI dictionary.
    
    Returns:
        The analysis response text.
    """
    roi_info = "No specific region highlighted."
    if roi and all(key in roi for key in ["left", "top", "width", "height"]):
        roi_info = f"ROI: Top-Left=({roi['left']}, {roi['top']}), Width={roi['width']}, Height={roi['height']}"
    
    prompt = DISEASE_SPECIFIC_PROMPT_TEMPLATE.format(
        disease=disease,
        roi_info=roi_info
    )
    
    response_text, success = query_gemini_vision(image, prompt)
    if success and response_text:
        return response_text
    else:
        return f"Disease analysis failed: {response_text}"

def estimate_ai_confidence(image: Image.Image, history: List[Tuple[str, str]]) -> str:
    """
    Requests an estimation of AI confidence based on the most recent analysis.
    
    Args:
        image: The PIL Image that was analyzed.
        history: List of previous (question, answer) pairs.
    
    Returns:
        A string containing the confidence score and justification.
    """
    if not history:
        return "Confidence estimation not performed: no conversation history available."
    
    last_q, last_a = history[-1]
    prompt = CONFIDENCE_PROMPT_TEMPLATE.format(
        last_q=last_q,
        last_a=last_a
    )
    response_text, success = query_gemini_vision(image, prompt)
    if success and response_text:
        return response_text
    else:
        return f"Confidence estimation failed: {response_text}"
