
# -*- coding: utf-8 -*-
"""
error_recovery.py - Error Recovery Module for RadVision AI
=========================================================

Provides mechanisms for robust error recovery, state preservation,
and automatic retries for failed operations.
"""

import time
import logging
import traceback
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from functools import wraps

import streamlit as st

# Import existing error handler if available
try:
    from error_handler import ErrorStore
    ERROR_STORE_AVAILABLE = True
except ImportError:
    ERROR_STORE_AVAILABLE = False
    
# Configure logger
logger = logging.getLogger(__name__)

# Type variable for generic function return
T = TypeVar('T')

# Constants
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.0
RECOVERABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    IOError,
)

class RecoveryState:
    """Manages application state for recovery purposes."""
    
    _preserved_states: Dict[str, Dict[str, Any]] = {}
    _recovery_lock = threading.RLock()
    
    @classmethod
    def preserve_state(cls, state_key: str) -> None:
        """
        Save current session state for potential recovery.
        
        Args:
            state_key: Unique identifier for this state snapshot
        """
        with cls._recovery_lock:
            # Create a shallow copy of session state (deep copy may fail for some objects)
            state_copy = {}
            for key, value in st.session_state.items():
                # Skip large objects that don't need recovery (display_image, etc.)
                if key in ['display_image', 'processed_image', 'raw_image_bytes']:
                    continue
                state_copy[key] = value
                
            cls._preserved_states[state_key] = state_copy
            logger.debug(f"Preserved state '{state_key}' with {len(state_copy)} items")
    
    @classmethod
    def restore_state(cls, state_key: str) -> bool:
        """
        Restore a previously preserved state.
        
        Args:
            state_key: Identifier of state to restore
            
        Returns:
            Boolean indicating if restoration was successful
        """
        with cls._recovery_lock:
            if state_key not in cls._preserved_states:
                logger.warning(f"No preserved state found for key '{state_key}'")
                return False
                
            preserved = cls._preserved_states[state_key]
            
            # Don't overwrite newer state if present
            for key, value in preserved.items():
                if key not in st.session_state:
                    st.session_state[key] = value
            
            logger.info(f"Restored state '{state_key}' with {len(preserved)} items")
            return True
    
    @classmethod
    def clear_preserved_state(cls, state_key: str) -> None:
        """
        Remove a preserved state when no longer needed.
        
        Args:
            state_key: Identifier of state to clear
        """
        with cls._recovery_lock:
            if state_key in cls._preserved_states:
                del cls._preserved_states[state_key]
                logger.debug(f"Cleared preserved state '{state_key}'")

def with_retries(max_retries: int = MAX_RETRIES, 
                delay_seconds: float = RETRY_DELAY_SECONDS,
                recoverable_exceptions: Tuple = RECOVERABLE_EXCEPTIONS) -> Callable:
    """
    Decorator to automatically retry functions on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay_seconds: Delay between retries
        recoverable_exceptions: Exception types that trigger retries
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            retries = 0
            last_exception = None
            
            while retries <= max_retries:
                try:
                    if retries > 0:
                        logger.info(f"Retry {retries}/{max_retries} for {func.__name__}")
                    return func(*args, **kwargs)
                except recoverable_exceptions as e:
                    last_exception = e
                    retries += 1
                    if retries <= max_retries:
                        logger.warning(f"Recoverable error in {func.__name__}, retrying: {e}")
                        time.sleep(delay_seconds * retries)  # Exponential backoff
                    else:
                        logger.error(f"Maximum retries reached for {func.__name__}: {e}")
                except Exception as e:
                    # Non-recoverable exception, don't retry
                    logger.error(f"Non-recoverable error in {func.__name__}: {e}")
                    raise
            
            # If we get here, all retries failed
            if last_exception:
                raise last_exception
            raise RuntimeError(f"All retries failed for {func.__name__}")
        
        return wrapper
    return decorator

def recover_on_error(recovery_state_key: Optional[str] = None) -> Callable:
    """
    Decorator to handle errors and preserve/restore state.
    
    Args:
        recovery_state_key: Key for state preservation, defaults to function name
        
    Returns:
        Decorated function with error recovery
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Use function name as default state key if not provided
            state_key = recovery_state_key or func.__name__
            
            try:
                # Preserve current state before execution
                RecoveryState.preserve_state(state_key)
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Success, clear preserved state
                RecoveryState.clear_preserved_state(state_key)
                return result
                
            except Exception as e:
                # Log the error
                error_details = {
                    "function": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs),
                    "traceback": traceback.format_exc()
                }
                
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                
                # Store error if ErrorStore is available
                if ERROR_STORE_AVAILABLE:
                    ErrorStore.add_error(
                        error_type=f"Function Error: {func.__name__}",
                        error_msg=str(e),
                        details=error_details
                    )
                
                # Attempt to restore state
                RecoveryState.restore_state(state_key)
                
                # Re-raise the exception
                raise
                
        return wrapper
    return decorator

def safe_execution(fallback_value: Any = None) -> Callable:
    """
    Decorator to continue execution despite errors, returning a fallback value.
    
    Args:
        fallback_value: Value to return if function raises an exception
        
    Returns:
        Decorated function that never raises exceptions
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Union[T, Any]]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Union[T, Any]:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Error in {func.__name__}, using fallback: {e}")
                return fallback_value
        return wrapper
    return decorator
