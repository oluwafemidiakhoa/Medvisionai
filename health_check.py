
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
health_check.py - System Health Check for RadVision AI
======================================================

Provides a comprehensive health check of the RadVision AI system,
rating various aspects of performance and resource usage.
"""

import os
import sys
import time
import json
import logging
import psutil
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import performance profiler
from performance_profiler import get_performance_summary
from error_handler import ErrorStore

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("health_check")

def get_system_health():
    """Get comprehensive system health metrics"""
    process = psutil.Process(os.getpid())
    
    # Collect memory stats
    memory_info = process.memory_info()
    system_memory = psutil.virtual_memory()
    
    # CPU usage 
    process_cpu = process.cpu_percent(interval=0.5)
    system_cpu = psutil.cpu_percent(interval=None)
    
    # Disk usage
    disk = psutil.disk_usage('/')
    
    # Performance summary from profiler
    perf_summary = get_performance_summary()
    
    # Error history
    recent_errors = []
    try:
        recent_errors = ErrorStore.get_recent_errors(limit=5)
    except:
        pass
    
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
        },
        "disk": {
            "total_gb": disk.total / (1024 * 1024 * 1024),
            "used_gb": disk.used / (1024 * 1024 * 1024),
            "free_gb": disk.free / (1024 * 1024 * 1024),
            "percent": disk.percent
        },
        "performance": perf_summary,
        "errors": {
            "count": len(recent_errors),
            "recent": recent_errors[:3]  # First 3 errors
        }
    }

def rate_system_health(health_data):
    """
    Rate system health on a scale of 1-10
    Returns dict with ratings and overall score
    """
    ratings = {}
    
    # Rate memory usage (lower is better)
    memory_percent = health_data["memory"]["system_percent"]
    if memory_percent < 50:
        ratings["memory"] = 10
    elif memory_percent < 70:
        ratings["memory"] = 8
    elif memory_percent < 85:
        ratings["memory"] = 5
    else:
        ratings["memory"] = 2
    
    # Rate CPU usage (lower is better)
    cpu_percent = health_data["cpu"]["system_percent"]
    if cpu_percent < 30:
        ratings["cpu"] = 10
    elif cpu_percent < 60:
        ratings["cpu"] = 7
    elif cpu_percent < 85:
        ratings["cpu"] = 4
    else:
        ratings["cpu"] = 2
    
    # Rate disk space (higher free % is better)
    disk_percent = health_data["disk"]["percent"]
    if disk_percent < 50:
        ratings["disk"] = 10
    elif disk_percent < 75:
        ratings["disk"] = 7
    elif disk_percent < 90:
        ratings["disk"] = 4
    else:
        ratings["disk"] = 1
    
    # Rate error count (lower is better)
    error_count = health_data["errors"]["count"]
    if error_count == 0:
        ratings["errors"] = 10
    elif error_count < 3:
        ratings["errors"] = 7
    elif error_count < 10:
        ratings["errors"] = 4
    else:
        ratings["errors"] = 1
    
    # Calculate overall score (weighted average)
    weights = {
        "memory": 0.3,
        "cpu": 0.3,
        "disk": 0.2,
        "errors": 0.2
    }
    
    overall_score = sum(ratings[key] * weights[key] for key in weights)
    ratings["overall"] = round(overall_score, 1)
    
    return ratings

def display_health_check():
    """Display health check in Streamlit UI"""
    st.set_page_config(
        page_title="RadVision AI - Health Check",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 RadVision AI System Health Check")
    st.markdown("---")
    
    # Run health check
    with st.spinner("Performing health check..."):
        health_data = get_system_health()
        ratings = rate_system_health(health_data)
    
    # Display overall health score
    st.header("System Health Score")
    
    overall_score = ratings["overall"]
    
    # Determine status color based on score
    if overall_score >= 8:
        score_color = "green"
        status = "Excellent"
    elif overall_score >= 6:
        score_color = "blue"
        status = "Good"
    elif overall_score >= 4:
        score_color = "orange"
        status = "Fair"
    else:
        score_color = "red"
        status = "Poor"
    
    # Display score with color
    st.markdown(f"<h1 style='color: {score_color}; text-align: center;'>{overall_score}/10 - {status}</h1>", unsafe_allow_html=True)
    
    # Detailed metrics
    st.header("Detailed Health Metrics")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Memory Usage",
            f"{health_data['memory']['system_percent']:.1f}%",
            f"{ratings['memory']}/10",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "CPU Usage",
            f"{health_data['cpu']['system_percent']:.1f}%",
            f"{ratings['cpu']}/10",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Disk Usage",
            f"{health_data['disk']['percent']:.1f}%",
            f"{ratings['disk']}/10",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            "Error Count",
            f"{health_data['errors']['count']}",
            f"{ratings['errors']}/10",
            delta_color="inverse"
        )
    
    # System resource details
    with st.expander("System Resource Details"):
        st.json(health_data)
    
    # Performance recommendations
    st.header("Recommendations")
    
    recommendations = []
    
    if ratings["memory"] < 5:
        recommendations.append("High memory usage detected. Consider optimizing memory-intensive operations or increasing available memory.")
    
    if ratings["cpu"] < 5:
        recommendations.append("High CPU usage detected. Look for computationally expensive operations and optimize if possible.")
    
    if ratings["disk"] < 5:
        recommendations.append("Low disk space. Clean up unnecessary files or increase available storage.")
    
    if ratings["errors"] < 7:
        recommendations.append("Multiple errors detected. Review the error logs to address recurring issues.")
    
    if not recommendations:
        st.success("No critical issues detected. System is performing well!")
    else:
        for i, rec in enumerate(recommendations):
            st.warning(f"{i+1}. {rec}")
    
    # Auto-refresh option
    auto_refresh = st.checkbox("Auto-refresh (every 60s)")
    
    if auto_refresh:
        time.sleep(60)
        st.rerun()

if __name__ == "__main__":
    if "streamlit" in sys.modules:
        display_health_check()
    else:
        # Command line mode
        health_data = get_system_health()
        ratings = rate_system_health(health_data)
        
        print("\n=== RADVISION AI HEALTH CHECK ===")
        print(f"Overall Health Score: {ratings['overall']}/10")
        print("\nDetailed Ratings:")
        print(f"- Memory: {ratings['memory']}/10 ({health_data['memory']['system_percent']:.1f}% used)")
        print(f"- CPU: {ratings['cpu']}/10 ({health_data['cpu']['system_percent']:.1f}% used)")
        print(f"- Disk: {ratings['disk']}/10 ({health_data['disk']['percent']:.1f}% used)")
        print(f"- Errors: {ratings['errors']}/10 ({health_data['errors']['count']} recent errors)")
        
        print("\nFor detailed metrics, run with Streamlit: streamlit run health_check.py")
import os
import json
import streamlit as st

def health_check():
    """Check if all required environment variables are set properly"""
    st.title("🏥 RadVision AI Health Check")
    
    # Check service account
    service_account_json = os.environ.get("SERVICE_ACCOUNT_JSON")
    if service_account_json:
        try:
            # Try to parse JSON
            service_account_data = json.loads(service_account_json)
            required_fields = ["type", "project_id", "private_key", "client_email"]
            missing_fields = [field for field in required_fields if field not in service_account_data]
            
            if not missing_fields:
                st.success("✅ Google Service Account is properly configured")
                st.info(f"Project ID: {service_account_data.get('project_id')}")
                st.info(f"Client Email: {service_account_data.get('client_email')}")
            else:
                st.error(f"❌ Service Account JSON is missing fields: {', '.join(missing_fields)}")
        except json.JSONDecodeError:
            st.error("❌ Service Account JSON is not valid JSON")
    else:
        st.error("❌ SERVICE_ACCOUNT_JSON environment variable is not set")
    
    # Check other secrets
    api_keys = {
        "GEMINI_API_KEY": "Gemini API",
        "UMLS_APIKEY": "UMLS API",
        "HF_API_TOKEN": "Hugging Face API"
    }
    
    for env_var, name in api_keys.items():
        value = os.environ.get(env_var)
        if value:
            st.success(f"✅ {name} key is configured")
        else:
            st.warning(f"⚠️ {name} key is not configured")
    
    # Check workflow configuration
    if os.path.exists(".replit"):
        st.success("✅ .replit configuration file exists")
    else:
        st.warning("⚠️ .replit configuration file does not exist")
    
    # Git status check
    st.subheader("Git Status")
    if os.path.exists(".git"):
        st.info("Git repository is initialized")
        st.info("To verify Git pull works correctly, run the following commands in the shell:")
        st.code("git status\ngit pull")
    else:
        st.info("No Git repository found in this workspace")
    
    # Health advice for .env file
    st.subheader("Environment File Status")
    if os.path.exists(".env"):
        st.warning("⚠️ .env file exists and may interfere with Git operations")
        st.info("Consider removing the .env file after confirming all secrets are in Replit Secrets:")
        st.code("rm .env")
    else:
        st.success("✅ No .env file found - good for Git operations")

if __name__ == "__main__":
    health_check()
