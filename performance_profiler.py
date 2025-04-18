
# -*- coding: utf-8 -*-
"""
performance_profiler.py - Performance Monitoring for RadVision AI
================================================================

Provides tools for monitoring and optimizing application performance,
including function timing, memory usage tracking, and bottleneck
identification.
"""

import time
import logging
import functools
import threading
import tracemalloc
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from functools import wraps

import streamlit as st

# Configure logger
logger = logging.getLogger(__name__)

# Type variable for generic function return
T = TypeVar('T')

# Constants
PROFILE_LOG_FILE = "performance_logs.csv"
MEMORY_TRACKING_ENABLED = False  # Set to True to enable memory tracking (has performance overhead)

# Global profiling data
_performance_data: Dict[str, List[Dict[str, Any]]] = {}
_profiling_lock = threading.RLock()

# Start memory tracking if enabled
if MEMORY_TRACKING_ENABLED:
    tracemalloc.start()

def profile_time(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to profile function execution time.
    
    Args:
        func: Function to be profiled
        
    Returns:
        Decorated function with timing
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            success = False
            raise
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Record performance data
            with _profiling_lock:
                if func.__name__ not in _performance_data:
                    _performance_data[func.__name__] = []
                
                _performance_data[func.__name__].append({
                    'timestamp': datetime.now().isoformat(),
                    'execution_time': execution_time,
                    'success': success
                })
            
            # Log performance info
            log_level = logging.DEBUG if execution_time < 1.0 else logging.INFO
            logger.log(log_level, f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        
        return result
    
    return wrapper

def profile_memory(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to profile function memory usage.
    
    Args:
        func: Function to be profiled
        
    Returns:
        Decorated function with memory profiling
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        if not MEMORY_TRACKING_ENABLED:
            return func(*args, **kwargs)
        
        # Take memory snapshot before
        tracemalloc.stop()
        tracemalloc.start()
        start_snapshot = tracemalloc.take_snapshot()
        
        result = func(*args, **kwargs)
        
        # Take memory snapshot after
        end_snapshot = tracemalloc.take_snapshot()
        
        # Compare snapshots
        top_stats = end_snapshot.compare_to(start_snapshot, 'lineno')
        
        # Log memory usage
        total_diff = sum(stat.size_diff for stat in top_stats)
        logger.info(f"Memory usage for '{func.__name__}': {total_diff / 1024:.2f} KB")
        
        # Log top 3 memory consumers if significant
        if total_diff > 1024 * 1024:  # More than 1MB
            for i, stat in enumerate(top_stats[:3]):
                logger.info(f"  {i+1}. {stat}")
        
        return result
    
    return wrapper

def get_performance_summary() -> Dict[str, Dict[str, float]]:
    """
    Get performance summary statistics.
    
    Returns:
        Dictionary with performance statistics by function
    """
    with _profiling_lock:
        summary = {}
        
        for func_name, calls in _performance_data.items():
            if not calls:
                continue
                
            # Calculate statistics
            total_calls = len(calls)
            successful_calls = sum(1 for call in calls if call.get('success', False))
            execution_times = [call['execution_time'] for call in calls]
            
            avg_time = sum(execution_times) / len(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            
            # Sort execution times and get percentiles
            sorted_times = sorted(execution_times)
            p50 = sorted_times[int(len(sorted_times) * 0.5)]
            p90 = sorted_times[int(len(sorted_times) * 0.9)]
            p99 = sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) >= 100 else max_time
            
            summary[func_name] = {
                'total_calls': total_calls,
                'success_rate': successful_calls / total_calls if total_calls > 0 else 0,
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'p50': p50,
                'p90': p90,
                'p99': p99
            }
            
        return summary

def render_performance_dashboard() -> None:
    """Render a Streamlit UI showing performance metrics."""
    st.subheader("📊 Performance Dashboard")
    
    if st.button("Reset Performance Data"):
        with _profiling_lock:
            _performance_data.clear()
        st.success("Performance data cleared")
        st.rerun()
    
    summary = get_performance_summary()
    
    if not summary:
        st.info("No performance data collected yet.")
        return
        
    # Display summary table
    st.write("Function Performance Summary")
    
    # Prepare data for table
    table_data = []
    for func_name, stats in summary.items():
        table_data.append({
            "Function": func_name,
            "Calls": stats['total_calls'],
            "Success Rate": f"{stats['success_rate']:.1%}",
            "Avg Time (s)": f"{stats['avg_time']:.4f}",
            "Min (s)": f"{stats['min_time']:.4f}",
            "Max (s)": f"{stats['max_time']:.4f}",
            "P90 (s)": f"{stats['p90']:.4f}",
        })
    
    # Sort by average time (slowest first)
    table_data.sort(key=lambda x: float(x["Avg Time (s)"].replace(',', '')), reverse=True)
    
    # Display table
    st.table(table_data)
    
    # Identify potential bottlenecks
    bottlenecks = []
    for func_name, stats in summary.items():
        if stats['avg_time'] > 1.0:  # Functions taking > 1 second on average
            bottlenecks.append((func_name, stats['avg_time']))
    
    if bottlenecks:
        st.subheader("🚨 Potential Bottlenecks")
        for func_name, avg_time in sorted(bottlenecks, key=lambda x: x[1], reverse=True):
            st.warning(f"{func_name}: {avg_time:.2f}s average execution time")
    
    # Memory tracking info
    if MEMORY_TRACKING_ENABLED:
        st.subheader("Memory Usage")
        current, peak = tracemalloc.get_traced_memory()
        st.info(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
        st.info(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
    else:
        st.info("Memory tracking is disabled. Enable it in performance_profiler.py")
