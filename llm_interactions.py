# -*- coding: utf-8 -*-
"""
llm_interactions.py

This module handles interactions with the Gemini API (or other LLM backends)
for medical image analysis within the RadVision AI application. It provides
functions for generating initial analysis, answering questions in context,
performing condition-specific analysis, estimating AI confidence, and
mapping key terms to standardized UMLS concepts with advanced customization.
"""
import os
import io
import base64
import logging
import requests
from typing import Optional, Tuple, List, Dict, Any
from PIL import Image

from umls_utils import search_umls, UMLSConcept

# Configure logging
logger = logging.getLogger(__name__)

# --- Constants ---
IMAGE_MIME_TYPE: str = "image/png"
API_TIMEOUT: int = 180  # seconds
def_env = os.getenv
# Gemini model config
DEFAULT_MODEL_NAME: str = "gemini-1.5-pro-latest"
GEMINI_MODEL_NAME: str = def_env("GEMINI_MODEL_OVERRIDE", DEFAULT_MODEL_NAME)
logger.info(f"Using Gemini model: {GEMINI_MODEL_NAME}")

# UMLS mapping defaults
DEFAULT_UMLS_HITS: int = int(def_env("UMLS_HITS", "3"))
# Optional filter for specific UMLS sources (comma-separated)
SOURCE_FILTER = def_env("UMLS_SOURCE_FILTER", "").split(",") if def_env("UMLS_SOURCE_FILTER") else []

# --- Prompt Templates (as defined previously) ---
_BASE_ROLE_PROMPT = """
You are a highly specialized AI assistant simulating an expert radiologist...
"""
INITIAL_ANALYSIS_PROMPT_TEMPLATE = f"""{_BASE_ROLE_PROMPT}
**Task:** Perform a comprehensive initial analysis...
**Region of Interest (ROI):** {{roi_info}}
..."""
QA_CONTEXT_PROMPT_TEMPLATE = f"""{_BASE_ROLE_PROMPT}
**Task:** Answer the user's question...
..."""
CONFIDENCE_PROMPT_TEMPLATE = f"""{_BASE_ROLE_PROMPT}
**Task:** Evaluate the confidence level...
..."""
DISEASE_SPECIFIC_PROMPT_TEMPLATE = f"""{_BASE_ROLE_PROMPT}
**Task:** Analyze exclusively for **{{disease}}**...
..."""

# --- Helper: Encode image to base64 ---
def _encode_image(image: Image.Image) -> Tuple[Optional[str], str]:
    try:
        buffered = io.BytesIO()
        img = image.convert('RGB') if image.mode not in ('RGB', 'L') else image
        img.save(buffered, format="PNG")
        data = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return data, IMAGE_MIME_TYPE
    except Exception as e:
        logger.error(f"Image encoding failed: {e}")
        return None, f"Error encoding image: {e}"

# --- Core API interaction ---
def query_gemini_vision(image: Image.Image, text_prompt: str) -> Tuple[str, bool]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Configuration Error: Gemini API key not found.", False
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    img_data, mime = _encode_image(image)
    if img_data is None:
        return mime, False
    payload = {
        "contents": [{"parts": [{"text": text_prompt}, {"inline_data": {"mime_type": mime, "data": img_data}}]}],
        "generation_config": {"temperature": 0.2, "top_k": 32, "top_p": 0.95, "max_output_tokens": 8192},
        "safety_settings": [{"category": cat, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for cat in [
            "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
    }
    try:
        resp = requests.post(url, headers=headers, params=params, json=payload, timeout=API_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get('candidates', [])
        if not candidates:
            return "API Error: No candidates returned.", False
        text = candidates[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
        return (text, True) if text else ("API Error: Empty response.", False)
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return f"API Error: {e}", False

# --- Integration of UMLS concepts (advanced) ---
def append_umls_info(
    response: str,
    num_hits: Optional[int] = None,
    include_types: bool = False
) -> str:
    """Query UMLS for top concepts based on response text and append them.

    Args:
        response: The AI-generated text to map.
        num_hits: Number of top UMLS concepts to retrieve (overrides DEFAULT_UMLS_HITS).
        include_types: If True, include semantic type info if available.
    """
    api_key = os.getenv("UMLS_APIKEY")
    if not api_key:
        logger.warning("UMLS_APIKEY not set; skipping UMLS mapping.")
        return response

    hits = num_hits if isinstance(num_hits, int) and num_hits > 0 else DEFAULT_UMLS_HITS
    try:
        concepts: List[UMLSConcept] = search_umls(response, api_key, page_size=hits)
        # Optional filtering by source
        if SOURCE_FILTER:
            concepts = [c for c in concepts if c.rootSource in SOURCE_FILTER]
        if not concepts:
            return response
        entries = []
        for c in concepts:
            label = f"{c.name} (CUI: {c.ui}, Source: {c.rootSource})"
            if include_types and hasattr(c, 'uriLabel') and c.uriLabel:
                label += f", Type: {c.uriLabel}"
            # Hyperlink name to UMLS browser
            entries.append(f"- [{c.name}]({c.uri}) | CUI: {c.ui} | Source: {c.rootSource}")
        block = "\n\n**Standardized UMLS Concepts:**\n" + "\n".join(entries)
        return response + block
    except Exception as e:
        logger.error(f"UMLS mapping failed: {e}")
        return response

# --- LLM Interaction Functions ---
def run_initial_analysis(
    image: Image.Image,
    roi: Optional[Dict] = None,
    umls_hits: Optional[int] = None
) -> str:
    """Performs initial analysis and appends UMLS mapping."""
    action = "Initial Analysis"
    roi_info = _format_roi(roi)
    prompt = INITIAL_ANALYSIS_PROMPT_TEMPLATE.format(roi_info=roi_info)
    text, ok = query_gemini_vision(image, prompt)
    if not ok:
        return f"{action} Failed: {text}"
    return append_umls_info(text, num_hits=umls_hits)


def run_multimodal_qa(
    image: Image.Image,
    question: str,
    history: List[Tuple[str, str, Any]],
    roi: Optional[Dict] = None,
    umls_hits: Optional[int] = None
) -> Tuple[str, bool]:
    """Performs Q&A in context and appends UMLS mapping."""
    action = "Multimodal Q&A"
    roi_info = _format_roi(roi)
    history_text = _format_history(history)
    prompt = QA_CONTEXT_PROMPT_TEMPLATE.format(
        roi_info=roi_info,
        history_text=history_text,
        question=question
    )
    text, ok = query_gemini_vision(image, prompt)
    return (append_umls_info(text, num_hits=umls_hits) if ok else text, ok)


def run_disease_analysis(
    image: Image.Image,
    disease: str,
    roi: Optional[Dict] = None,
    umls_hits: Optional[int] = None
) -> str:
    """Performs disease-focused analysis and appends UMLS mapping."""
    action = f"Disease Analysis ({disease})"
    roi_info = _format_roi(roi)
    prompt = DISEASE_SPECIFIC_PROMPT_TEMPLATE.format(disease=disease, roi_info=roi_info)
    text, ok = query_gemini_vision(image, prompt)
    if not ok:
        return f"{action} Failed: {text}"
    return append_umls_info(text, num_hits=umls_hits)


def estimate_ai_confidence(
    image: Image.Image,
    history: List[Tuple[str, str, Any]],
    roi: Optional[Dict] = None
) -> str:
    """Estimates AI confidence without UMLS mapping."""
    action = "Confidence Estimation"
    if not history:
        return f"{action} Failed: No history available."
    last_q, last_a = history[-1][0], history[-1][1]
    roi_info = _format_roi(roi)
    prompt = CONFIDENCE_PROMPT_TEMPLATE.format(last_q=last_q, last_a=last_a, roi_info=roi_info)
    text, ok = query_gemini_vision(image, prompt)
    return text if ok else f"{action} Failed: {text}"

# --- Utility formatters ---
def _format_roi(roi: Optional[Dict]) -> str:
    if not roi or not all(k in roi for k in ("left", "top", "width", "height")):
        return "No specific region highlighted by user."
    try:
        return (f"ROI at ({int(roi['left'])},{int(roi['top'])}) size "
                f"{int(roi['width'])}x{int(roi['height'])} pixels.")
    except Exception:
        return "ROI provided but invalid."


def _format_history(history: List[Tuple[str, str, Any]]) -> str:
    if not history:
        return "No previous history."
    turns = []
    for q, a, _ in history[-3:]:
        turns.append(f"User: {q}\nAI: {a}")
    return "\n---\n".join(turns)
