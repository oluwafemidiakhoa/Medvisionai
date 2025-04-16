---
**Current User Question:** "{{question}}"
---

**Instructions:**
- Analyze the image visually, paying attention to the ROI if specified.
- Use the conversation history for context if relevant to the question.
- Provide a concise, factual, and direct answer to the "Current User Question" based *only* on visual evidence.
- If the question cannot be answered visually from the image (e.g., requires clinical data, different view, finding is ambiguous), clearly state this and explain why.
- Adhere strictly to all constraints defined in your base role (no diagnosis, caution, etc.).
"""

# Prompt for focused analysis on a specific condition
DISEASE_SPECIFIC_PROMPT_TEMPLATE = f"""{_BASE_ROLE_PROMPT}

**Task:** Analyze the provided image *specifically and exclusively* for visual findings potentially related to **{{disease}}**.

**Region of Interest (ROI):** {{roi_info}}

**Output Structure (Use these Markdown headings):**

1.  **## 1. Findings Suggestive of {{disease}}:**
    *   State whether visual findings potentially suggestive of {{disease}} are **Identified**, **Not Identified**, or **Indeterminate** based *only* on this image. Use cautious language.

2.  **## 2. Description of Relevant Visual Findings:**
    *   If suggestive findings are identified or indeterminate, describe the specific visual observations relevant to {{disease}} (e.g., "Patchy airspace opacity in the right lower lobe," "Linear lucency crossing the distal radius"). Use anatomical locators. Focus on the ROI if specified.
    *   If no relevant findings are identified, state: "No specific visual findings clearly suggestive of {{disease}} were identified in the analyzed image/ROI."

3.  **## 3. Qualitative Severity Impression (Optional & Cautious):**
    *   If relevant findings are *clearly* identified and severity can be reasonably judged *visually*, provide a *qualitative impression* (e.g., Mild, Moderate, Extensive).
    *   **State explicitly if severity cannot be determined** from the image alone. **Avoid definitive grading.**

4.  **## 4. Imaging Limitations & Disclaimer:**
    *   Briefly mention any limitations specific to assessing **{{disease}}** with this particular image (e.g., "2D view limits assessment of X," "Further views or modalities like CT might be needed for confirmation").
    *   Include the mandatory disclaimer: This AI analysis requires expert clinical correlation and validation.

**Focus solely on visual findings related to {{disease}}.**
"""

# Prompt for AI self-assessment (QUALITATIVE ONLY - Renamed from Confidence)
SELF_ASSESSMENT_PROMPT_TEMPLATE = f"""{_BASE_ROLE_PROMPT}

**Task:** Perform a qualitative self-assessment of the *previous AI response* provided below, considering the context it was generated in. **This is an internal check, not a clinical confidence score.**

**Context of the Interaction Being Assessed:**
*   **Question/Task Addressed:** {{last_q}}
*   **Region of Interest (ROI) Used:** {{roi_info}}
*   **Previous AI Response (The one to assess):**
    ```
    {{last_a}}
    ```
---
**Self-Assessment Request:**
Critically evaluate the **"Previous AI Response"** based on the following factors related to the visual information in the image and the task context. Provide a brief justification for each factor.

**Output Format (Use these Markdown headings):**

1.  **## 1. Clarity of Findings:**
    *   Justification: [Were the visual findings described in the previous response distinct and unambiguous in the image? Or were they subtle/equivocal?]

2.  **## 2. Sufficiency of Visual Information:**
    *   Justification: [Was the visual information in the image likely sufficient to support the statements made? Were there significant limitations ignored?]

3.  **## 3. Potential Ambiguity:**
    *   Justification: [Was there inherent ambiguity in the image or findings that limits certainty (e.g., overlapping structures, poor quality, non-specific appearance)?]

4.  **## 4. Scope Alignment:**
    *   Justification: [Did the previous response directly and fully address the user's question or the requested task's scope? Or did it deviate or miss aspects?]

5.  **## 5. Overall Assessment Impression:**
    *   Impression: [Provide a brief qualitative summary impression - e.g., "Assessment suggests response was well-supported by clear visual evidence," "Assessment indicates moderate confidence due to some ambiguity," "Assessment suggests low confidence due to poor image quality/non-specific findings."]

**Reminder:** This assessment reflects the AI's perspective on its previous output's limitations and alignment, **not clinical certainty.**
"""

# --- Helper Function for Image Encoding ---
def _encode_image(image: Image.Image) -> Tuple[Optional[str], str]:
    """Encodes a PIL Image to base64 PNG string, ensuring RGB or L compatibility."""
    try:
        buffered = io.BytesIO()
        # Work on a copy to avoid modifying the original
        image_copy = image.copy()
        # Ensure image is in a compatible mode (RGB or Grayscale 'L')
        if image_copy.mode not in ('RGB', 'L'):
            logger.debug(f"Converting image from mode '{image_copy.mode}' to 'RGB' for encoding.")
            image_copy = image_copy.convert('RGB')

        # Save as PNG (lossless, widely supported)
        image_copy.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        logger.debug(f"Image successfully encoded to base64 PNG (Original mode: {image.mode}, Size: {len(img_bytes)} bytes)")
        return img_base64, IMAGE_MIME_TYPE
    except Exception as e:
        logger.error(f"Error encoding image to PNG/base64: {e}", exc_info=True)
        return None, f"Error encoding image: {e}"


# --- Core Gemini Interaction Function ---
def query_gemini_vision(image: Image.Image, text_prompt: str) -> Tuple[Optional[str], bool]:
    """
    Sends an image and text prompt to the configured Gemini API, handling errors.

    Args:
        image: PIL Image object.
        text_prompt: The prompt for the AI.

    Returns:
        Tuple: (response_text or error_message, success_flag)
    """
    # Retrieve API Key securely (ensure it's set in environment or secrets management)
    gemini_api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.critical("GEMINI_API_KEY not found in environment or Streamlit secrets.")
        return "Configuration Error: Gemini API key not found.", False

    # Use v1beta for potentially newer features/models like Gemini 1.5
    gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": gemini_api_key}

    # Encode image
    img_base64, mime_type_or_error = _encode_image(image)
    if img_base64 is None:
        return f"Image Encoding Failed: {mime_type_or_error}", False

    # Construct API payload
    payload = {
        "contents": [{"parts": [{"text": text_prompt}, {"inline_data": {"mime_type": IMAGE_MIME_TYPE, "data": img_base64}}]}],
        "generation_config": {
            "temperature": 0.25, # Low temp for more deterministic medical description
            "top_k": 32,
            "top_p": 0.95,
            "max_output_tokens": 8192, # Leverage large context
        },
        "safety_settings": [
            # BLOCK_MEDIUM_AND_ABOVE is a reasonable default. Test carefully for medical content.
            {"category": cat, "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            for cat in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        ]
    }

    logger.info(f"Sending request to Gemini API: {GEMINI_MODEL_NAME}")
    logger.debug(f"Prompt length: {len(text_prompt)} chars. Payload keys: {list(payload.keys())}")

    try:
        response = requests.post(gemini_api_url, headers=headers, params=params, json=payload, timeout=API_TIMEOUT)
        response.raise_for_status() # Check for HTTP errors (4xx, 5xx)

        response_data: Dict[str, Any] = response.json()
        logger.debug("Raw Gemini API response received.")

        # --- Parse Gemini Response ---
        # 1. Check for Prompt Blocking
        prompt_feedback = response_data.get('promptFeedback')
        if prompt_feedback and prompt_feedback.get('blockReason'):
            reason = prompt_feedback['blockReason']
            blocked_cats = [sr.get('category', 'UNKNOWN') for sr in prompt_feedback.get('safetyRatings', []) if sr.get('blocked')]
            error_msg = f"API Error: Input prompt blocked by safety filters (Reason: {reason}). Categories: {blocked_cats}"
            logger.warning(error_msg)
            return error_msg, False

        # 2. Check for Candidates
        candidates = response_data.get('candidates')
        if not candidates or not isinstance(candidates, list):
            logger.error(f"API Error: No 'candidates' field found in response. Keys: {response_data.keys()}")
            return "API Error: Invalid response structure (no candidates).", False

        candidate = candidates[0] # Assume first candidate is the primary one

        # 3. Check Finish Reason (Includes safety blocking of response)
        finish_reason = candidate.get('finishReason', 'UNKNOWN')
        if finish_reason != "STOP":
             safety_ratings = candidate.get('safetyRatings', [])
             blocked_cats = [sr.get('category', 'UNKNOWN') for sr in safety_ratings if sr.get('blocked')]
             warn_msg = f"Gemini response finished unexpectedly (Reason: {finish_reason})."
             if blocked_cats: warn_msg += f" Blocked safety categories: {blocked_cats}"
             logger.warning(warn_msg)
             if finish_reason == "SAFETY":
                 return f"API Error: Response generation blocked by safety filters. Categories: {blocked_cats}", False
             elif finish_reason != "MAX_TOKENS": # Treat MAX_TOKENS as potentially usable partial response
                  return f"API Error: Response generation stopped unexpectedly (Reason: {finish_reason}).", False

        # 4. Extract Text Content
        content = candidate.get('content', {})
        parts = content.get('parts', [])
        if parts and isinstance(parts, list) and 'text' in parts[0]:
            parsed_text = parts[0]['text'].strip()
            if parsed_text:
                logger.info("Successfully received and parsed text response from Gemini API.")
                return parsed_text, True
            else:
                logger.warning("Gemini response contained an empty text part, though finish reason was okay.")
                return "API Error: Response received but contained no text content.", False
        else:
            logger.warning(f"Could not extract text from candidate part structure: {parts}")
            return "API Error: Response structure invalid (missing text part).", False

    # --- Exception Handling ---
    except requests.exceptions.Timeout:
        logger.error(f"Gemini API request timed out after {API_TIMEOUT} seconds.")
        return "API Error: Request timed out. The AI took too long to respond.", False
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        try: error_details = e.response.json().get("error", {})
        except: error_details = {}
        error_message = error_details.get("message", e.response.text[:500])
        logger.error(f"HTTP error {status_code} querying Gemini: {error_message}", exc_info=(status_code >= 500))
        if status_code in [401, 403]: return f"API Error: Authentication/Permission Failed ({status_code}). Check API Key.", False
        if status_code == 429: return "API Error: Rate limit exceeded (429). Please try again later.", False
        return f"API Error: HTTP {status_code}. Details: {error_message}", False
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during Gemini API request: {e}", exc_info=True)
        return "Network Error: Could not connect to the AI service.", False
    except Exception as e:
        logger.critical(f"Unexpected critical error during Gemini API interaction: {e}", exc_info=True)
        return f"Internal Error: An unexpected error occurred ({type(e).__name__}).", False


# --- Specific Interaction Function Wrappers ---

def _format_roi_info(roi: Optional[Dict]) -> str:
    """Formats ROI dictionary into a string for prompts, handling invalid input."""
    if roi and isinstance(roi, dict) and all(key in roi for key in ["left", "top", "width", "height"]):
        try:
            # Ensure values are integers for clean display
            left, top, width, height = map(int, [roi['left'], roi['top'], roi['width'], roi['height']])
            return (f"User has highlighted a Region of Interest (ROI) at "
                    f"Top-Left=({left}, {top}) with "
                    f"Width={width}, Height={height} pixels.")
        except (TypeError, ValueError, KeyError):
            logger.warning("ROI dictionary contained invalid/missing keys or non-numeric values.", exc_info=True)
            return "ROI provided but coordinates appear invalid."
    return "Analysis applies to the entire image (no specific ROI highlighted)." # More explicit default


def _format_history_text(history: List[Tuple[str, str, Any]]) -> str:
    """Formats recent conversation history for the prompt context."""
    if not history:
        return "No previous conversation history available."
    formatted_entries = []
    # Iterate safely over last N turns, newest first in prompt
    for entry in history[-HISTORY_TURNS_FOR_CONTEXT:][::-1]:
        try:
            q_type = entry[0] if len(entry) > 0 else "[Type Missing]"
            msg = entry[1] if len(entry) > 1 else "[Message Missing]"
            # Simple formatting, indicate source clearly
            if "[fallback]" in q_type.lower():
                # Extract user question part if available
                user_q_part = q_type.split(']')[1].strip() if ']' in q_type else "[User Question]"
                formatted_entries.append(f"User: {user_q_part}\nAI (Fallback): {msg}")
            elif "user" in q_type.lower():
                 formatted_entries.append(f"User: {msg}")
            elif "ai" in q_type.lower():
                 formatted_entries.append(f"AI: {msg}")
            else: # Handle system messages or other types if necessary
                 formatted_entries.append(f"{q_type}: {msg}")

        except Exception as e:
            logger.warning(f"Skipping malformed history entry {entry}: {e}")
            continue
    return "\n---\n".join(formatted_entries) if formatted_entries else "No processable conversation history available."


def run_initial_analysis(image: Image.Image, roi: Optional[Dict] = None) -> str:
    """Performs initial structured analysis, returning result or formatted error message."""
    action_name = "Initial Analysis"
    logger.info(f"Requesting {action_name}. ROI: {bool(roi)}")
    roi_info = _format_roi_info(roi)
    prompt = INITIAL_ANALYSIS_PROMPT_TEMPLATE.format(roi_info=roi_info)
    response_text, success = query_gemini_vision(image, prompt)
    # Prefix error clearly for the UI
    return response_text if success else f"**{action_name} Failed:** {response_text or 'Unknown API error.'}"


def run_multimodal_qa(
    image: Image.Image, question: str, history: List[Tuple[str, str, Any]], roi: Optional[Dict] = None
) -> Tuple[str, bool]:
    """Handles Q&A, returning result/error message and success flag."""
    action_name = "Multimodal Q&A"
    logger.info(f"Requesting {action_name}. ROI: {bool(roi)}. History: {len(history)} turns.")
    roi_info = _format_roi_info(roi)
    history_text = _format_history_text(history)
    prompt = QA_CONTEXT_PROMPT_TEMPLATE.format(roi_info=roi_info, history_text=history_text, question=question)
    response_text, success = query_gemini_vision(image, prompt)
    # Return tuple directly; error message already formatted by query_gemini_vision if needed
    return response_text if response_text else "Error: No response text received from API.", success


def run_disease_analysis(image: Image.Image, disease: str, roi: Optional[Dict] = None) -> str:
    """Performs disease-focused analysis, returning result or formatted error."""
    action_name = "Disease Analysis"
    logger.info(f"Requesting {action_name} for '{disease}'. ROI: {bool(roi)}")
    roi_info = _format_roi_info(roi)
    prompt = DISEASE_SPECIFIC_PROMPT_TEMPLATE.format(disease=disease, roi_info=roi_info)
    response_text, success = query_gemini_vision(image, prompt)
    # Prefix error clearly for the UI
    return response_text if success else f"**{action_name} Failed ({disease}):** {response_text or 'Unknown API error.'}"


def run_llm_self_assessment(
    image: Image.Image, # Image associated with the interaction being assessed
    history: List[Tuple[str, str, Any]],
    roi: Optional[Dict] = None # ROI state during the interaction being assessed
    ) -> str:
    """
    Requests the AI to perform a qualitative self-assessment of its last response.
    This is experimental and NOT a clinical confidence score.

    Args:
        image: The PIL Image corresponding to the last interaction.
        history: List of previous interaction tuples. Must not be empty.
        roi: The ROI dictionary active during the last interaction being evaluated.

    Returns:
        A string containing the AI's self-assessment based on the defined factors,
        or a string prefixed with "LLM Self-Assessment Failed: ".
    """ # <<< ENSURED THIS CLOSING QUOTE IS PRESENT
    action_name = "LLM Self-Assessment (Experimental)"
    logger.info(f"Requesting {action_name}. History length: {len(history)}. ROI used previously: {bool(roi)}")

    if not history:
        logger.warning(f"{action_name} requested without history.")
        return f"**{action_name} Failed:** No conversation history available to assess."

    # --- Safely extract the last Q/A pair to assess ---
    last_q, last_a = "[Question Missing]", "[Answer Missing]"
    try:
        last_ai_answer_entry = None
        last_user_question_entry = None
        for entry in reversed(history): # Search backwards
             entry_type = entry[0].lower() if len(entry) > 0 else ""
             entry_msg = entry[1] if len(entry) > 1 else ""
             # Find the most recent AI answer first
             if "ai answer" in entry_type and not last_ai_answer_entry:
                 last_ai_answer_entry = (entry_type, entry_msg)
             # Then find the user question that likely preceded it
             elif "user question" in entry_type and not last_user_question_entry:
                 last_user_question_entry = (entry_type, entry_msg)
             # Stop once we have the pair related to the last AI answer
             if last_ai_answer_entry and last_user_question_entry:
                 break

        if not last_ai_answer_entry or not last_user_question_entry:
             raise ValueError("Could not find a preceding User/AI pair in history.")

        last_q = last_user_question_entry[1]
        last_a = last_ai_answer_entry[1]

        # --- Pre-check: If last answer was an error, provide direct feedback ---
        if isinstance(last_a, str) and any(err in last_a.lower() for err in ["error:", "failed:", "blocked", "unavailable", "could not"]):
            logger.warning(f"Last interaction was an error ('{last_a[:100]}...'). Reporting low assessment directly.")
            # Return formatted assessment indicating failure
            return (f"**{action_name} Result:**\n\n"
                    f"## 1. Clarity of Findings:\n   Justification: N/A - Previous step resulted in an error.\n\n"
                    f"## 2. Sufficiency of Visual Information:\n   Justification: N/A - Error state.\n\n"
                    f"## 3. Potential Ambiguity:\n   Justification: N/A - Error state.\n\n"
                    f"## 4. Scope Alignment:\n   Justification: N/A - The previous request failed.\n\n"
                    f"## 5. Overall Assessment Impression:\n   Impression: Assessment not possible due to prior error.")

    except Exception as e: # Catch broader exceptions during history processing
         logger.error(f"Error processing history for self-assessment: {e}", exc_info=True)
         return f"**{action_name} Failed:** Error processing interaction history."

    # --- Prepare and Run Assessment Prompt ---
    roi_info = _format_roi_info(roi) # Format ROI state from the time of the last answer
    prompt = SELF_ASSESSMENT_PROMPT_TEMPLATE.format(last_q=last_q, last_a=last_a, roi_info=roi_info)

    # Call the API using the *same image* associated with the last interaction
    response_text, success = query_gemini_vision(image, prompt)

    if success and response_text:
        # Basic check for expected structure
        if "## 1. Clarity" in response_text and "## 5. Overall Assessment" in response_text:
            logger.info("LLM Self-Assessment received in expected format.")
            # Prepend title for clarity in UI
            return f"**{action_name} Result:**\n\n{response_text}"
        else:
            logger.warning(f"Self-assessment response did not match expected Markdown format:\n'''{response_text}'''")
            # Return raw but flag it and add title
            return f"**{action_name} Result (Format Warning):**\n\n{response_text}"
    else:
        # Prefix error clearly for the UI
        return f"**{action_name} Failed:** {response_text or 'Unknown API error.'}"