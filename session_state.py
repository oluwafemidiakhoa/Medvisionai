# session_state.py
import streamlit as st
import copy
import uuid
import logging
from config import DEFAULT_STATE # Import defaults from config

logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
        logger.info(f"New session started: {st.session_state.session_id}")

    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value

    # Ensure history is always a list (safety check)
    if not isinstance(st.session_state.get("history"), list):
        st.session_state.history = []

    logger.debug(f"Session state initialized/verified for session ID: {st.session_state.session_id}")

def reset_session_state_for_new_file():
    """Resets relevant state when a new file is uploaded, preserving session ID."""
    logger.debug("Resetting session state for new file...")
    keys_to_preserve = {"session_id"} # Only preserve session ID
    preserved_values = {k: st.session_state.get(k) for k in keys_to_preserve if k in st.session_state}

    for key, default_value in DEFAULT_STATE.items():
        if key not in keys_to_preserve:
            st.session_state[key] = copy.deepcopy(default_value) if isinstance(default_value, (dict, list)) else default_value

    for k, v in preserved_values.items():
        st.session_state[k] = v
    logger.debug("Session state reset complete.")