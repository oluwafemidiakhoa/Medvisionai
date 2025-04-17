# -*- coding: utf-8 -*-
"""
session_state.py - Manages Streamlit Session State Initialization and Reset
===========================================================================

Provides functions to initialize the application's session state with
default values and reset specific parts of the state when needed (e.g.,
upon new file upload), while preserving essential information like the session ID.
"""

import streamlit as st
import copy
import uuid
import logging
from typing import Dict, Any, Set

# Import default state structure from config
try:
    from config import DEFAULT_STATE
except ImportError:
    # Fallback if config is missing or DEFAULT_STATE isn't defined
    DEFAULT_STATE: Dict[str, Any] = {"session_id": None, "history": []}
    logging.error("Could not import DEFAULT_STATE from config.py. Using minimal fallback.")

logger = logging.getLogger(__name__)

def initialize_session_state() -> None:
    """
    Initializes Streamlit's session state dictionary.

    If 'session_id' doesn't exist, it generates a new one.
    It then iterates through DEFAULT_STATE, adding any missing keys
    to st.session_state with their default values. Uses deepcopy for
    mutable types (dicts, lists) to prevent unintended sharing.
    """
    # --- Ensure Session ID Exists ---
    if "session_id" not in st.session_state or not st.session_state.session_id:
        # Generate a new short UUID if no session ID exists
        new_id = str(uuid.uuid4())[:8]
        st.session_state.session_id = new_id
        logger.info(f"New session started or ID missing. Assigned Session ID: {new_id}")
    else:
        # Log existing ID if already present
        logger.debug(f"Existing Session ID found: {st.session_state.session_id}")


    # --- Initialize other state variables from defaults ---
    initialized_keys = []
    for key, default_value in DEFAULT_STATE.items():
        if key not in st.session_state:
            # Use deepcopy for mutable defaults to avoid shared references
            if isinstance(default_value, (dict, list)):
                st.session_state[key] = copy.deepcopy(default_value)
            else:
                st.session_state[key] = default_value
            initialized_keys.append(key)

    if initialized_keys:
         logger.debug(f"Initialized missing session state keys: {initialized_keys}")


    # --- Ensure specific keys have correct types (Safety Check) ---
    # Example: Ensure 'history' is always a list, even if DEFAULT_STATE was wrong
    if "history" not in st.session_state or not isinstance(st.session_state.history, list):
        if "history" in st.session_state:
             logger.warning("Session state 'history' was not a list. Resetting to empty list.")
        st.session_state.history = []


    logger.debug(f"Session state initialization check complete for Session ID: {st.session_state.session_id}")


def reset_session_state_for_new_file() -> None:
    """
    Resets specific session state keys to their defaults.

    Typically called when a new file is uploaded or demo mode is toggled.
    It preserves essential keys like 'session_id'.
    """
    logger.info(f"Resetting session state for new file (Session ID: {st.session_state.session_id})...")

    # Define keys that should NOT be reset
    keys_to_preserve: Set[str] = {"session_id"}

    # Store values of keys to preserve
    preserved_values: Dict[str, Any] = {}
    for key in keys_to_preserve:
        if key in st.session_state:
            preserved_values[key] = st.session_state[key]
            logger.debug(f"Preserving key: '{key}'")

    # Reset all other keys defined in DEFAULT_STATE
    reset_keys = []
    for key, default_value in DEFAULT_STATE.items():
        if key not in keys_to_preserve:
            if isinstance(default_value, (dict, list)):
                st.session_state[key] = copy.deepcopy(default_value)
            else:
                st.session_state[key] = default_value
            reset_keys.append(key)

    if reset_keys:
        logger.debug(f"Reset keys to defaults: {reset_keys}")

    # Restore preserved values (should be redundant if logic above is correct, but safe)
    for key, value in preserved_values.items():
        st.session_state[key] = value

    logger.info("Session state reset for new file complete.")