
# -*- coding: utf-8 -*-
"""
error_handler.py - Error handling utilities for RadVision AI
===========================================================

Provides centralized error handling, logging, and reporting
for improved reliability and troubleshooting.
"""

import sys
import traceback
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Type

import streamlit as st

# Configure logger
logger = logging.getLogger(__name__)

# Constants
ERROR_LOG_DIR = "error_logs"
MAX_STORED_ERRORS = 50

class ErrorStore:
    """Stores recent errors for easy debugging and display in UI."""
    
    _errors: List[Dict[str, Any]] = []
    
    @classmethod
    def add_error(cls, error_type: str, error_msg: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an error to the store.
        
        Args:
            error_type: Category/type of error
            error_msg: Main error message
            details: Additional structured details
        """
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": error_msg,
            "details": details or {}
        }
        
        cls._errors.append(error_entry)
        
        # Trim list if it gets too long
        if len(cls._errors) > MAX_STORED_ERRORS:
            cls._errors = cls._errors[-MAX_STORED_ERRORS:]
            
        # Log to file if directory exists
        cls._log_to_file(error_entry)
    
    @classmethod
    def get_recent_errors(cls, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent errors.
        
        Args:
            limit: Maximum number of errors to return
            
        Returns:
            List of error dictionaries
        """
        return cls._errors[-limit:]
    
    @classmethod
    def clear_errors(cls) -> None:
        """Clear all stored errors."""
        cls._errors = []
    
    @classmethod
    def _log_to_file(cls, error_entry: Dict[str, Any]) -> None:
        """Save error to JSON log file if logging directory exists."""
        try:
            if not os.path.exists(ERROR_LOG_DIR):
                return
                
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = os.path.join(ERROR_LOG_DIR, f"error_log_{timestamp}.json")
            
            # Read existing logs
            existing_logs = []
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    try:
                        existing_logs = json.load(f)
                    except json.JSONDecodeError:
                        existing_logs = []
            
            # Append new error
            existing_logs.append(error_entry)
            
            # Write back to file
            with open(log_file, 'w') as f:
                json.dump(existing_logs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to log error to file: {e}")

def capture_exception(error_type: str) -> Callable:
    """
    Decorator to capture and log exceptions.
    
    Args:
        error_type: Category/type of error
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Format traceback
                tb = traceback.format_exc()
                
                # Log error
                logger.error(f"{error_type} Error in {func.__name__}: {e}\n{tb}")
                
                # Store error
                ErrorStore.add_error(
                    error_type=error_type,
                    error_msg=str(e),
                    details={
                        "function": func.__name__,
                        "traceback": tb,
                        "args": str(args),
                        "kwargs": str(kwargs)
                    }
                )
                
                # Re-raise for proper handling
                raise
        return wrapper
    return decorator

def render_error_dashboard() -> None:
    """Render a UI component showing recent errors."""
    st.subheader("🐞 Error Dashboard")
    
    if st.button("Clear Error History"):
        ErrorStore.clear_errors()
        st.success("Error history cleared")
        st.rerun()
        
    recent_errors = ErrorStore.get_recent_errors()
    
    if not recent_errors:
        st.info("No errors recorded. That's great! 🎉")
        return
        
    st.warning(f"Found {len(recent_errors)} recent errors")
    
    for i, error in enumerate(reversed(recent_errors)):
        with st.expander(f"{error['type']}: {error['message'][:50]}...", expanded=(i==0)):
            st.text(f"Time: {error['timestamp']}")
            st.text(f"Type: {error['type']}")
            st.text(f"Message: {error['message']}")
            
            if "traceback" in error["details"]:
                with st.expander("View Traceback"):
                    st.code(error["details"]["traceback"])
            
            st.json(error["details"])
