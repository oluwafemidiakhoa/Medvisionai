
# -*- coding: utf-8 -*-
"""
caching.py - Caching utilities for RadVision AI
===============================================

Provides memory and disk-based caching mechanisms for expensive operations
like image processing, API calls, and LLM interactions.
"""

import os
import json
import hashlib
import logging
import pickle
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from functools import wraps
from datetime import datetime, timedelta

import streamlit as st

# Configure logger
logger = logging.getLogger(__name__)

# Type variable for generic function return
T = TypeVar('T')

# Constants
CACHE_DIR = "cache"
MEMORY_CACHE_MAX_ITEMS = 100  # Max items to keep in memory cache
DISK_CACHE_MAX_AGE_HOURS = 24  # Max age for disk cached items

# Ensure cache directory exists
if not os.path.exists(CACHE_DIR):
    try:
        os.makedirs(CACHE_DIR)
        logger.info(f"Created cache directory: {CACHE_DIR}")
    except Exception as e:
        logger.warning(f"Failed to create cache directory: {e}")

# Memory cache dictionary
_memory_cache: Dict[str, Tuple[Any, float]] = {}

def generate_cache_key(func_name: str, args: Tuple, kwargs: Dict) -> str:
    """
    Generate a unique cache key based on function name and arguments.
    
    Args:
        func_name: Name of the function being cached
        args: Positional arguments
        kwargs: Keyword arguments
        
    Returns:
        A unique hash string to use as cache key
    """
    # Convert args and kwargs to string representation
    args_str = str(args)
    kwargs_str = str(sorted(kwargs.items()))
    
    # Generate hash
    key_str = f"{func_name}:{args_str}:{kwargs_str}"
    return hashlib.md5(key_str.encode()).hexdigest()

def memory_cache(max_age_seconds: int = 300) -> Callable:
    """
    Decorator for in-memory caching with time expiration.
    
    Args:
        max_age_seconds: Maximum age of cached items in seconds
        
    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Skip cache if disabled in session state
            if st.session_state.get("disable_cache", False):
                return func(*args, **kwargs)
            
            # Generate cache key
            cache_key = generate_cache_key(func.__name__, args, kwargs)
            
            # Check if in cache and not expired
            current_time = time.time()
            if cache_key in _memory_cache:
                result, timestamp = _memory_cache[cache_key]
                if current_time - timestamp <= max_age_seconds:
                    logger.debug(f"Memory cache hit for {func.__name__}")
                    return result
            
            # Execute function if not cached or expired
            result = func(*args, **kwargs)
            
            # Store in cache with timestamp
            _memory_cache[cache_key] = (result, current_time)
            
            # Clean up old items if cache is too large
            if len(_memory_cache) > MEMORY_CACHE_MAX_ITEMS:
                # Sort by timestamp (oldest first) and remove oldest items
                sorted_items = sorted(_memory_cache.items(), key=lambda x: x[1][1])
                items_to_remove = len(_memory_cache) - MEMORY_CACHE_MAX_ITEMS
                for k, _ in sorted_items[:items_to_remove]:
                    del _memory_cache[k]
            
            return result
        return wrapper
    return decorator

def disk_cache(max_age_hours: int = DISK_CACHE_MAX_AGE_HOURS) -> Callable:
    """
    Decorator for disk-based caching with time expiration.
    
    Args:
        max_age_hours: Maximum age of cached items in hours
        
    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Skip cache if disabled in session state
            if st.session_state.get("disable_cache", False):
                return func(*args, **kwargs)
            
            # Skip cache if cache directory doesn't exist
            if not os.path.exists(CACHE_DIR):
                return func(*args, **kwargs)
            
            # Generate cache key
            cache_key = generate_cache_key(func.__name__, args, kwargs)
            cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
            
            # Check if cache file exists and not expired
            if os.path.exists(cache_file):
                file_time = os.path.getmtime(cache_file)
                file_age = datetime.now() - datetime.fromtimestamp(file_time)
                
                if file_age < timedelta(hours=max_age_hours):
                    try:
                        with open(cache_file, 'rb') as f:
                            logger.debug(f"Disk cache hit for {func.__name__}")
                            return pickle.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to load cache file {cache_file}: {e}")
            
            # Execute function if not cached or expired
            result = func(*args, **kwargs)
            
            # Save to cache file
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except Exception as e:
                logger.warning(f"Failed to write cache file {cache_file}: {e}")
            
            return result
        return wrapper
    return decorator

def clear_all_caches() -> None:
    """Clear both memory and disk caches."""
    # Clear memory cache
    global _memory_cache
    _memory_cache = {}
    
    # Clear disk cache
    if os.path.exists(CACHE_DIR):
        try:
            for filename in os.listdir(CACHE_DIR):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(CACHE_DIR, filename))
            logger.info("All caches cleared")
        except Exception as e:
            logger.error(f"Error clearing disk cache: {e}")

def clear_expired_cache_files() -> None:
    """Remove expired cache files from disk."""
    if not os.path.exists(CACHE_DIR):
        return
        
    try:
        current_time = time.time()
        max_age_seconds = DISK_CACHE_MAX_AGE_HOURS * 3600
        
        for filename in os.listdir(CACHE_DIR):
            if not filename.endswith('.pkl'):
                continue
                
            file_path = os.path.join(CACHE_DIR, filename)
            file_time = os.path.getmtime(file_path)
            
            if current_time - file_time > max_age_seconds:
                os.remove(file_path)
                logger.debug(f"Removed expired cache file: {filename}")
    except Exception as e:
        logger.error(f"Error cleaning expired cache files: {e}")

# Clean expired cache files on module import
clear_expired_cache_files()
