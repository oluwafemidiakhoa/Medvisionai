import streamlit as st
import google.generativeai as genai
import zipfile
import io
import json
import os  # Still needed for API key potentially, but not model names
from pathlib import Path
import time

# --- Configuration ---
# Model names are now discovered dynamically. Remove hardcoded names.
MAX_PROMPT_TOKENS_ESTIMATE = 800000  # Keep this estimate
RESULTS_PAGE_SIZE = 25

AVAILABLE_ANALYSES = {  # Keep analyses config
    "generate_docs": "Generate Missing Docstrings/Comments",
    "find_bugs": "Identify Potential Bugs & Anti-patterns",
    "check_style": "Check Style Guide Compliance (General)",
    "summarize_modules": "Summarize Complex Modules/Files",
    "suggest_refactoring": "Suggest Refactoring Opportunities",
}
CODE_EXTENSIONS = {
    '.py', '.js', '.java', '.c', '.cpp', '.h', '.cs', '.go', '.rb', 
    '.php', '.swift', '.kt', '.ts', '.html', '.css', '.scss', '.sql'
}  # Keep extensions

# --- Session State Initialization ---
# (Keep most session state, add one for the selected model)
if 'mock_api_call' not in st.session_state:
    st.session_state.mock_api_call = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'analysis_requested' not in st.session_state:
    st.session_state.analysis_requested = False
if 'selected_model_name' not in st.session_state:
    st.session_state.selected_model_name = None  # Will hold the "models/..." name
if 'available_models_dict' not in st.session_state:
    st.session_state.available_models_dict = {}  # Store display_name -> name mapping

# --- Gemini API Setup & Model Discovery ---
model = None  # Global variable for the initialized model instance

# --- NEW: Function to list available models ---
@st.cache_data(ttl=3600)  # Cache model list for an hour
def get_available_models():
    """Lists models supporting 'generateContent' using the API key."""
    model_dict = {}
    try:
        if 'GEMINI_API_KEY' not in st.secrets:
            # Don't stop here, let the main part handle it, but return empty
            print("API key not found in secrets during model listing attempt.")
            return {}
        # Configure API key temporarily just for listing
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        print("Listing available models via API...")
        for m in genai.list_models():
            # Check if the model supports the 'generateContent' method
            if 'generateContent' in m.supported_generation_methods:
                # Store mapping: user-friendly name -> internal name
                model_dict[m.display_name] = m.name
        print(f"Found {len(model_dict)} compatible models.")
        return model_dict
    except Exception as e:
        st.error(f"🚨 Error listing available models: {e}")
        return {}  # Return empty on error

def initialize_gemini_model():
    """Initializes the Gemini model based on the selected name."""
    global model
    selected_name = st.session_state.get('selected_model_name')

    if selected_name and model is None and not st.session_state.mock_api_call:
        try:
            if 'GEMINI_API_KEY' not in st.secrets:
                st.error("🚨 Gemini API Key not found. Add it to `.streamlit/secrets.toml`.")
                st.stop()  # Stop if key missing for initialization
            # Configure API key (might be redundant if list_models worked, but safe)
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            print(f"Initializing Gemini Model: {selected_name}")
            # Use the selected model name from session state
            model = genai.GenerativeModel(model_name=selected_name)
            print(f"Gemini Model Initialized ({selected_name}).")
            return True
        except Exception as e:
            st.error(f"🚨 Error initializing selected Gemini model '{selected_name}': {e}")
            st.session_state.selected_model_name = None  # Reset selection on error
            st.stop()
            return False
    elif st.session_state.mock_api_call:
        return True  # No init needed for mock mode
    elif model is not None and model.model_name == selected_name:
        return True  # Already initialized with the correct model
    elif model is not None and model.model_name != selected_name:
        print("Model changed. Re-initializing...")
        model = None  # Reset model instance
        return initialize_gemini_model()  # Recurse to re-initialize with new name
    elif not selected_name and not st.session_state.mock_api_call:
        # This case happens if no model is selected yet
        return False  # Cannot initialize without a selection
    return False  # Default case

# --- Helper Functions ---
# (estimate_token_count, process_zip_file_cached, construct_analysis_prompt,
#  call_gemini_api, display_results - remain the same as the optimized version)

def estimate_token_count(text):
    """Estimates the number of tokens based on text length."""
    return len(text) // 3

@st.cache_data(max_entries=5)
def process_zip_file_cached(file_id, file_size, file_content_bytes):
    """
    Processes a ZIP file and extracts code files.
    Returns a tuple of (code_files dict, total_chars, file_count, ignored_files list).
    """
    code_files = {}
    total_chars = 0
    file_count = 0
    ignored_files = []
    status_placeholder = st.empty()
    progress_bar = status_placeholder.progress(0)
    try:
        with zipfile.ZipFile(io.BytesIO(file_content_bytes), 'r') as zip_ref:
            members = zip_ref.infolist()
            total_members = len(members)
            for i, member in enumerate(members):
                if i % 10 == 0:
                    progress_bar.progress(int((i / total_members) * 100))
                if member.is_dir() or any(p.startswith('.') for p in Path(member.filename).parts) or '__' in member.filename:
                    continue
                file_path = Path(member.filename)
                if file_path.suffix.lower() in CODE_EXTENSIONS:
                    try:
                        with zip_ref.open(member) as file:
                            file_bytes = file.read()
                            try:
                                content = file_bytes.decode('utf-8')
                            except UnicodeDecodeError:
                                try:
                                    content = file_bytes.decode('latin-1')
                                except Exception as decode_err:
                                    ignored_files.append(f"{member.filename} (Decode Error: {decode_err})")
                                    continue
                            code_files[member.filename] = content
                            total_chars += len(content)
                            file_count += 1
                    except Exception as read_err:
                        ignored_files.append(f"{member.filename} (Read Error: {read_err})")
                else:
                    if not (any(p.startswith('.') for p in Path(member.filename).parts) or '__' in member.filename):
                        ignored_files.append(f"{member.filename} (Skipped Extension: {file_path.suffix})")
            progress_bar.progress(100)
            status_placeholder.empty()
    except zipfile.BadZipFile:
        status_placeholder.empty()
        st.error("🚨 Invalid ZIP.")
        return None, 0, 0, []
    except Exception as e:
        status_placeholder.empty()
        st.error(f"🚨 ZIP Error: {e}")
        return None, 0, 0, []
    if file_count == 0:
        if not ignored_files:
            st.warning("No code files found.")
        else:
            st.warning("No code files found; some skipped.")
    return code_files, total_chars, file_count, ignored_files

def construct_analysis_prompt(code_files_dict, requested_analyses):
    """
    Constructs the prompt for analysis by including code files and JSON structure for expected output.
    Returns the full prompt and a list of included files.
    """
    prompt_parts = ["Analyze the following codebase...\n\n"]
    current_token_estimate = estimate_token_count(prompt_parts[0])
    included_files = []
    code_segments = []
    prompt_status = st.empty()
    
    if len(code_files_dict) > 50:
        prompt_status.info("Constructing prompt...")
        
    for filename, content in code_files_dict.items():
        segment = f"--- START FILE: {filename} ---\n{content}\n--- END FILE: {filename} ---\n\n"
        segment_token_estimate = estimate_token_count(segment)
        if current_token_estimate + segment_token_estimate <= MAX_PROMPT_TOKENS_ESTIMATE:
            code_segments.append(segment)
            current_token_estimate += segment_token_estimate
            included_files.append(filename)
        else:
            st.warning(f"⚠️ Codebase may exceed context limit. Analyzed first {len(included_files)} files (~{current_token_estimate:,} tokens).")
            break
    prompt_status.empty()
    
    if not included_files:
        st.error("🚨 No code files included in prompt.")
        return None, []
    
    prompt_parts.append("".join(code_segments))
    json_structure_description = "{\n"
    structure_parts = []
    
    if "generate_docs" in requested_analyses:
        structure_parts.append('    "documentation_suggestions": [...]')
    if "find_bugs" in requested_analyses:
        structure_parts.append('    "potential_bugs": [...]')
    if "check_style" in requested_analyses:
        structure_parts.append('    "style_issues": [...]')
    if "summarize_modules" in requested_analyses:
        structure_parts.append('    "module_summaries": [...]')
    if "suggest_refactoring" in requested_analyses:
        structure_parts.append('    "refactoring_suggestions": [...]')
        
    json_structure_description += ",\n".join(structure_parts) + "\n}"
    prompt_footer = f"\n**Analysis Task:**...\n**Output Format:**...\n{json_structure_description}\n**JSON Output Only:**\n"
    prompt_parts.append(prompt_footer)
    
    full_prompt = "".join(prompt_parts)
    return full_prompt, included_files

def call_gemini_api(prompt):
    """
    Calls the Gemini API using the provided prompt.
    Returns the parsed JSON insights or an error message.
    """
    if not prompt:
        return None, "Prompt generation failed."
    
    # MOCK MODE
    if st.session_state.mock_api_call:
        st.info(" MOCK MODE: Simulating API call...")
        time.sleep(1)
        mock_json_response = json.dumps({
            "documentation_suggestions": [],
            "potential_bugs": [],
            "style_issues": [],
            "module_summaries": [],
            "refactoring_suggestions": []
        })
        st.success("Mock response generated.")
        return json.loads(mock_json_response), None
    # REAL API CALL
    else:
        if not initialize_gemini_model():
            return None, "Gemini Model Initialization Failed."
        if model is None:
            return None, "Gemini model not selected or available."  # Added check
        try:
            api_status = st.empty()
            api_status.info(f"📡 Sending request to {model.model_name} (Est. prompt tokens: {estimate_token_count(prompt):,})... Please wait.")
            start_time = time.time()
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.2),
                safety_settings=[
                    {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
                    for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", 
                              "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
                ]
            )
            end_time = time.time()
            api_status.success(f"✅ Response received from AI ({model.model_name}) in {end_time - start_time:.2f}s.")
            time.sleep(1)
            api_status.empty()
            try:
                json_response_text = response.text.strip()
                # Remove markdown code fences if present
                if json_response_text.startswith("```json"):
                    json_response_text = json_response_text[7:]
                if json_response_text.startswith("```"):
                    json_response_text = json_response_text[3:]
                if json_response_text.endswith("```"):
                    json_response_text = json_response_text[:-3]
                json_start = json_response_text.find('{')
                json_end = json_response_text.rfind('}') + 1
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    final_json_text = json_response_text[json_start:json_end]
                    insights = json.loads(final_json_text)
                    return insights, None
                else:
                    st.warning("⚠️ Could not find valid JSON object.")
                    return {"raw_response": response.text}, "AI response did not contain clear JSON object."
            except json.JSONDecodeError as json_err:
                st.error(f"🚨 Error parsing JSON: {json_err}")
                st.code(response.text, language='text')
                return None, f"AI response not valid JSON: {json_err}"
            except AttributeError:
                st.error("🚨 Unexpected API response structure (AttributeError).")
                st.code(f"Response object: {response}", language='text')
                return None, "Unexpected response structure (AttributeError)."
            except Exception as e:
                st.error(f"🚨 Unexpected issue processing response: {e}")
                try:
                    st.code(f"Response object: {response}", language='text')
                except Exception:
                    pass
                return None, f"Unexpected response structure: {e}"
        except Exception as e:
            api_status.empty()
            st.error(f"🚨 API call error: {e}")
            error_msg = f"API call failed: {e}"
            if hasattr(e, 'message'):
                if "429" in e.message:
                    error_msg = "API Quota Exceeded or Rate Limit hit."
                elif "API key not valid" in e.message:
                    error_msg = "Invalid Gemini API Key."
                elif "permission denied" in e.message.lower():
                    error_msg = f"Permission Denied for model '{st.session_state.selected_model_name}'. Check API key access."
                elif "blocked" in e.message.lower():
                    error_msg = "Content blocked due to safety settings."
            elif "block_reason: SAFETY" in str(e):
                error_msg = "Content blocked due to safety settings."
            return None, error_msg

def display_results(results_json, requested_analyses):
    """
    Displays the analysis results with pagination and allows JSON download.
    """
    st.header("📊 Analysis Report")
    if not isinstance(results_json, dict):
        st.error("Invalid results format.")
        st.json(results_json)
        return
    if "raw_response" in results_json:
        st.subheader("Raw AI Response (JSON Parsing Failed)")
        st.code(results_json["raw_response"], language='text')
        return

    display_config = {
        "generate_docs": {
            "key": "documentation_suggestions",
            "title": AVAILABLE_ANALYSES["generate_docs"],
            "fields": {"file": "File", "line": "Line"}
        },
        "find_bugs": {
            "key": "potential_bugs",
            "title": AVAILABLE_ANALYSES["find_bugs"],
            "fields": {"file": "File", "line": "Line", "severity": "Severity"}
        },
        "check_style": {
            "key": "style_issues",
            "title": AVAILABLE_ANALYSES["check_style"],
            "fields": {"file": "File", "line": "Line"}
        },
        "summarize_modules": {
            "key": "module_summaries",
            "title": AVAILABLE_ANALYSES["summarize_modules"],
            "fields": {"file": "File"}
        },
        "suggest_refactoring": {
            "key": "refactoring_suggestions",
            "title": AVAILABLE_ANALYSES["suggest_refactoring"],
            "fields": {"file": "File", "line": "Line", "area": "Area"}
        },
    }
    any_results_found = False
    for analysis_key in requested_analyses:
        if analysis_key in display_config:
            config = display_config[analysis_key]
            items = results_json.get(config["key"], [])
            total_items = len(items)
            st.subheader(f"{config['title']} ({total_items} found)")
            if items:
                any_results_found = True
                state_key = f"visible_{analysis_key}"
                if state_key not in st.session_state:
                    st.session_state[state_key] = RESULTS_PAGE_SIZE
                visible_count = st.session_state[state_key]
                items_to_display = items[:visible_count]
                for item in items_to_display:
                    details = [
                        f"**{field_label}:** `{item.get(field_key, 'N/A')}`" if field_key == 'file'
                        else f"**{field_label}:** {item.get(field_key, 'N/A')}"
                        for field_key, field_label in config["fields"].items()
                        if item.get(field_key, 'N/A') != 'N/A'
                    ]
                    st.markdown("- " + " - ".join(details))
                    if 'suggestion' in item:
                        st.code(item['suggestion'], language='text')
                    elif 'description' in item:
                        st.markdown(f"  > {item['description']}")
                    elif 'summary' in item:
                        st.markdown(f"  > {item['summary']}")
                if total_items > visible_count:
                    if st.button(f"Show more ({total_items - visible_count} remaining)", key=f"more_{analysis_key}"):
                        st.session_state[state_key] += RESULTS_PAGE_SIZE
                        st.rerun()
            else:
                st.markdown("_No items found for this category._")
            st.divider()
    if not any_results_found:
        st.info("No specific findings were identified.")
    st.download_button(
        label="Download Full Report (JSON)",
        data=json.dumps(results_json, indent=4),
        file_name="code_audit_report.json",
        mime="application/json"
    )

# --- Streamlit App Main Interface ---
st.set_page_config(page_title="Codebase Audit Assistant", layout="wide")
st.title("🤖 Codebase Audit & Documentation Assistant")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Analysis Controls")
    st.session_state.mock_api_call = st.toggle(
        "🧪 Enable Mock API Mode",
        value=st.session_state.mock_api_call,
        help="Use fake data instead of calling Gemini API."
    )

    st.divider()
    st.header("♊ Select Model")
    # --- NEW: Dynamic Model Selection ---
    if not st.session_state.mock_api_call:
        # Get available models (uses cache)
        st.session_state.available_models_dict = get_available_models()
        model_display_names = list(st.session_state.available_models_dict.keys())

        if model_display_names:
            # Try to find the index of the previously selected model
            current_model_display_name = None
            if st.session_state.selected_model_name:
                # Find display name matching the stored internal name
                for disp_name, internal_name in st.session_state.available_models_dict.items():
                    if internal_name == st.session_state.selected_model_name:
                        current_model_display_name = disp_name
                        break

            try:
                selected_index = model_display_names.index(current_model_display_name) if current_model_display_name in model_display_names else 0
            except ValueError:
                selected_index = 0  # Default to first if previous selection not found

            selected_display_name = st.selectbox(
                "Choose Gemini model:",
                options=model_display_names,
                index=selected_index,
                key="model_selector",
                help="Select the Gemini model to use for analysis."
            )
            # Update session state with the internal name based on selection
            st.session_state.selected_model_name = st.session_state.available_models_dict.get(selected_display_name)
            st.info(f"Using REAL Gemini API ({st.session_state.selected_model_name})")
        elif 'GEMINI_API_KEY' in st.secrets:
            st.warning("No compatible models found or error listing models. Check API Key permissions.")
            st.session_state.selected_model_name = None  # Ensure no model selected
        else:
            st.warning("Add GEMINI_API_KEY to secrets to list models.")
            st.session_state.selected_model_name = None
    else:  # Mock mode is active
        st.info("Mock API Mode ACTIVE")
        st.session_state.selected_model_name = "mock_model"  # Use a placeholder name for mock mode
    # --- End Dynamic Model Selection ---

    st.divider()
    st.header("🔎 Select Analyses")
    selected_analyses = [
        key for key, name in AVAILABLE_ANALYSES.items() 
        if st.checkbox(name, value=True, key=f"cb_{key}")
    ]
    st.divider()
    st.header("📄 How To Use")
    st.info(
        "1. Set API Key.\n"
        "2. Toggle Mock Mode if needed.\n"
        "3. Select Model (if not Mock).\n"
        "4. Select analyses.\n"
        "5. Upload ZIP.\n"
        "6. Click 'Analyze'.\n"
        "7. Review report."
    )
    st.info(f"Note: Limited by token estimates (~{MAX_PROMPT_TOKENS_ESTIMATE:,} est. tokens).")
    st.divider()
    st.warning("⚠️ **Privacy:** Code sent to Google API if Mock Mode is OFF.")

# Update title dynamically based on selected model
if st.session_state.selected_model_name and not st.session_state.mock_api_call:
    st.markdown(f"Upload codebase (`.zip`) for analysis via **{st.session_state.selected_model_name}**.")
elif st.session_state.mock_api_call:
    st.markdown("Upload codebase (`.zip`) for analysis (Using **Mock Data**).")
else:
    st.markdown("Upload codebase (`.zip`) for analysis.")

# --- Main Content Area ---
uploaded_file = st.file_uploader(
    "📁 Upload Codebase ZIP File",
    type=['zip'],
    key="file_uploader",
    on_change=lambda: st.session_state.update(
        analysis_results=None,
        error_message=None,
        analysis_requested=False
    )
)
analysis_button_placeholder = st.empty()
results_placeholder = st.container()

if uploaded_file:
    st.success(f"✅ File '{uploaded_file.name}' uploaded.")
    uploaded_file_bytes = uploaded_file.getvalue()
    file_id = f"{uploaded_file.name}-{uploaded_file.size}"
    code_files, total_chars, file_count, ignored_files = process_zip_file_cached(
        file_id, uploaded_file.size, uploaded_file_bytes
    )
    if code_files is not None:
        st.info(f"Found **{file_count}** code files ({total_chars:,} chars). Est. tokens: ~{estimate_token_count(total_chars):,}")
        if ignored_files:
            with st.expander(f"View {len(ignored_files)} Skipped/Ignored Files"):
                st.code("\n".join(ignored_files), language='text')

        # Disable button if no model selected (and not in mock mode)
        model_ready = bool(st.session_state.selected_model_name) or st.session_state.mock_api_call
        analyze_button_disabled = (not selected_analyses or file_count == 0 or not model_ready)
        analyze_button_label = "Analyze Codebase"
        if not model_ready:
            analyze_button_label = "Select Model First"
        elif analyze_button_disabled:
            analyze_button_label = "Select Analyses or Upload Valid Code"

        if analysis_button_placeholder.button(
            analyze_button_label,
            type="primary",
            disabled=analyze_button_disabled
        ):
            st.session_state.analysis_requested = True
            st.session_state.analysis_results = None
            st.session_state.error_message = None
            if not selected_analyses:
                st.warning("Please select analysis types.")
            elif file_count == 0:
                st.warning("No relevant code files found.")
            elif not model_ready:
                st.warning("Please select a Gemini model from the sidebar.")
            else:
                with results_placeholder:
                    spinner_model_name = (
                        st.session_state.selected_model_name 
                        if not st.session_state.mock_api_call 
                        else "Mock Mode"
                    )
                    spinner_msg = f"🚀 Preparing prompt & contacting AI ({spinner_model_name})... Please wait."
                    with st.spinner(spinner_msg):
                        analysis_prompt, included_files_in_prompt = construct_analysis_prompt(code_files, selected_analyses)
                        if analysis_prompt and included_files_in_prompt:
                            results_json, error_msg = call_gemini_api(analysis_prompt)
                            st.session_state.analysis_results = results_json
                            st.session_state.error_message = error_msg
                        elif not included_files_in_prompt:
                            st.session_state.error_message = "Could not proceed: No files included."
                        else:
                            st.session_state.error_message = "Failed to generate analysis prompt."
                st.rerun()

# Display results (Keep the same logic)
if st.session_state.analysis_requested:
    with results_placeholder:
        st.divider()
        if st.session_state.error_message:
            st.error(f"Analysis Failed: {st.session_state.error_message}")
            if isinstance(st.session_state.analysis_results, dict) and "raw_response" in st.session_state.analysis_results:
                st.subheader("Raw AI Response")
                st.code(st.session_state.analysis_results["raw_response"], language='text')
        elif st.session_state.analysis_results:
            display_results(st.session_state.analysis_results, selected_analyses)
        else:
            st.info("Analysis initiated, but no results/errors stored.")
elif not uploaded_file:
    results_placeholder.info("Upload a ZIP file to begin.")

results_placeholder.divider()
results_placeholder.markdown("_Assistant powered by Google Gemini._")
