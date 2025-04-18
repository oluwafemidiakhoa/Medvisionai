
# -*- coding: utf-8 -*-
"""
profiling_ui.py - Performance Profiling UI for RadVision AI
==========================================================

Provides a Streamlit interface for monitoring application performance,
identifying bottlenecks, and optimizing resource usage.
"""

import streamlit as st
import logging
import time
from datetime import datetime
import os
import sys
import gc
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional

# Import performance profiler
from performance_profiler import (
    get_performance_summary,
    render_performance_dashboard,
    MEMORY_TRACKING_ENABLED
)

# Import caching module
from caching import (
    clear_all_caches,
    clear_expired_cache_files
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def get_system_stats() -> Dict[str, Any]:
    """
    Get current system resource usage statistics.
    
    Returns:
        Dictionary with resource statistics
    """
    process = psutil.Process(os.getpid())
    
    # Collect memory stats
    memory_info = process.memory_info()
    system_memory = psutil.virtual_memory()
    
    # CPU usage (last 1 second)
    process_cpu = process.cpu_percent(interval=0.5)
    system_cpu = psutil.cpu_percent(interval=None)
    
    # Thread count
    thread_count = process.num_threads()
    
    # Open files count
    try:
        open_files = len(process.open_files())
    except (psutil.AccessDenied, Exception) as e:
        open_files = -1
        
    # File descriptors (Linux/Unix only)
    try:
        if hasattr(process, 'num_fds'):
            file_descriptors = process.num_fds()
        else:
            file_descriptors = None
    except (psutil.AccessDenied, Exception):
        file_descriptors = None
    
    return {
        "timestamp": datetime.now().isoformat(),
        "memory": {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent": process.memory_percent(),
            "system_percent": system_memory.percent,
            "system_available_gb": system_memory.available / (1024 * 1024 * 1024)
        },
        "cpu": {
            "process_percent": process_cpu,
            "system_percent": system_cpu,
            "num_threads": thread_count
        },
        "io": {
            "open_files": open_files,
            "file_descriptors": file_descriptors
        },
        "runtime": {
            "python_version": sys.version,
            "gc_enabled": gc.isenabled(),
            "gc_threshold": gc.get_threshold()
        }
    }

def main():
    """Main function for the profiling UI."""
    st.set_page_config(
        page_title="RadVision AI - Performance Profiler",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 RadVision AI Performance Profiler")
    st.markdown("---")
    
    # System stats
    st.header("System Resources")
    
    # Get system stats
    stats = get_system_stats()
    
    # Display memory metrics
    memory_col1, memory_col2, memory_col3 = st.columns(3)
    
    with memory_col1:
        st.metric(
            "Process Memory (RSS)",
            f"{stats['memory']['rss_mb']:.1f} MB",
            delta=None
        )
    
    with memory_col2:
        st.metric(
            "Process Memory Usage",
            f"{stats['memory']['percent']:.1f}%",
            delta=None
        )
    
    with memory_col3:
        st.metric(
            "System Memory Usage",
            f"{stats['memory']['system_percent']:.1f}%",
            delta=None
        )
    
    # Display CPU metrics
    cpu_col1, cpu_col2, cpu_col3 = st.columns(3)
    
    with cpu_col1:
        st.metric(
            "Process CPU Usage",
            f"{stats['cpu']['process_percent']:.1f}%",
            delta=None
        )
    
    with cpu_col2:
        st.metric(
            "System CPU Usage",
            f"{stats['cpu']['system_percent']:.1f}%",
            delta=None
        )
    
    with cpu_col3:
        st.metric(
            "Process Threads",
            f"{stats['cpu']['num_threads']}",
            delta=None
        )
    
    # Cache management
    st.header("Cache Management")
    
    cache_col1, cache_col2 = st.columns(2)
    
    with cache_col1:
        if st.button("Clear All Caches"):
            clear_all_caches()
            st.success("All caches cleared successfully")
    
    with cache_col2:
        if st.button("Clear Expired Cache Files"):
            clear_expired_cache_files()
            st.success("Expired cache files cleared")
    
    # Function profiling section
    st.markdown("---")
    render_performance_dashboard()
    
    # Memory tracking note
    if not MEMORY_TRACKING_ENABLED:
        st.warning(
            "Memory tracking is disabled. To enable detailed memory profiling, "
            "set MEMORY_TRACKING_ENABLED = True in performance_profiler.py"
        )
    
    # Runtime info
    with st.expander("Runtime Information"):
        st.json(stats["runtime"])
    
    # Auto-refresh option
    auto_refresh = st.checkbox("Auto-refresh (every 10s)")
    
    if auto_refresh:
        time.sleep(10)
        st.rerun()

if __name__ == "__main__":
    main()
