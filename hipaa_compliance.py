
# -*- coding: utf-8 -*-
"""
hipaa_compliance.py - HIPAA Compliance Tools for RadVision AI
=============================================================

This module implements tools and features to help ensure HIPAA compliance
for protected health information (PHI) when used in a healthcare setting.

Key features:
1. PHI detection and anonymization
2. Audit logging for HIPAA compliance
3. Access controls documentation
4. Data storage and transmission security
"""

import os
import re
import json
import logging
import datetime
import hashlib
import uuid
from typing import Dict, List, Any, Optional, Tuple

import streamlit as st

# Configure logger
logger = logging.getLogger(__name__)

# Constants
HIPAA_CONFIG_FILE = "hipaa_config.json"
HIPAA_AUDIT_LOG_FILE = "hipaa_audit_log.json"

# HIPAA audit event types
class AuditEventType:
    LOGIN = "LOGIN"
    LOGOUT = "LOGOUT"
    IMAGE_UPLOAD = "IMAGE_UPLOAD"
    IMAGE_VIEW = "IMAGE_VIEW"
    REPORT_GENERATION = "REPORT_GENERATION"
    PHI_ACCESS = "PHI_ACCESS"
    SYSTEM_CONFIG = "SYSTEM_CONFIG"
    SECURITY_EVENT = "SECURITY_EVENT"

# Default HIPAA configuration
DEFAULT_HIPAA_CONFIG = {
    "phi_handling": {
        "detect_phi_in_images": True,
        "detect_phi_in_text": True,
        "anonymize_dicom": True,
        "anonymize_reports": True,
        "phi_retention_days": 30,  # 0 = don't retain
    },
    "audit_logging": {
        "enabled": True,
        "log_login_events": True,
        "log_phi_access": True,
        "log_report_generation": True,
        "retention_days": 90,
    },
    "security": {
        "require_authentication": True,
        "password_expiration_days": 90,
        "session_timeout_minutes": 30,
        "failed_login_lockout": True,
        "max_failed_attempts": 5,
    },
    "last_updated": datetime.datetime.now().isoformat(),
}

def init_hipaa_config() -> None:
    """Initialize the HIPAA configuration if it doesn't exist."""
    if not os.path.exists(HIPAA_CONFIG_FILE):
        with open(HIPAA_CONFIG_FILE, 'w') as f:
            json.dump(DEFAULT_HIPAA_CONFIG, f, indent=2)
        logger.info("Created default HIPAA configuration")

def get_hipaa_config() -> Dict[str, Any]:
    """
    Get current HIPAA configuration.
    
    Returns:
        Dictionary with HIPAA configuration
    """
    if not os.path.exists(HIPAA_CONFIG_FILE):
        init_hipaa_config()
        
    try:
        with open(HIPAA_CONFIG_FILE, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading HIPAA config: {e}")
        return DEFAULT_HIPAA_CONFIG

def update_hipaa_config(section: str, key: str, value: Any) -> bool:
    """
    Update a specific section of the HIPAA configuration.
    
    Args:
        section: Main section of the config ('phi_handling', 'audit_logging', etc.)
        key: Key within the section to update
        value: New value to set
        
    Returns:
        True if update successful, False otherwise
    """
    if not os.path.exists(HIPAA_CONFIG_FILE):
        init_hipaa_config()
        
    try:
        with open(HIPAA_CONFIG_FILE, 'r') as f:
            config = json.load(f)
            
        if section in config and key in config[section]:
            config[section][key] = value
            config["last_updated"] = datetime.datetime.now().isoformat()
            
            with open(HIPAA_CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Updated HIPAA config: {section}.{key}")
            return True
        else:
            logger.error(f"Invalid section or key: {section}.{key}")
            return False
    except Exception as e:
        logger.error(f"Error updating HIPAA config: {e}")
        return False

def log_hipaa_audit_event(
    event_type: str,
    user_id: Optional[str] = None,
    description: str = "",
    resource_id: Optional[str] = None,
    success: bool = True,
    additional_data: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Log a HIPAA audit event.
    
    Args:
        event_type: Type of event (see AuditEventType class)
        user_id: User identifier
        description: Description of the event
        resource_id: Identifier of the resource being accessed
        success: Whether the action was successful
        additional_data: Any additional data to log
        
    Returns:
        True if logging successful, False otherwise
    """
    config = get_hipaa_config()
    if not config["audit_logging"]["enabled"]:
        return True  # Silently succeed if audit logging is disabled
    
    # Get username from session state if not provided
    if user_id is None and "auth_data" in st.session_state:
        user_id = st.session_state.auth_data.get("username", "unknown")
    
    # Create audit event
    audit_event = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event_type": event_type,
        "user_id": user_id or "unknown",
        "description": description,
        "resource_id": resource_id,
        "success": success,
        "session_id": st.session_state.get("session_id", "unknown"),
        "client_ip": st.session_state.get("client_ip", "unknown"),
        "additional_data": additional_data or {}
    }
    
    # Load existing audit log
    audit_log = []
    if os.path.exists(HIPAA_AUDIT_LOG_FILE):
        try:
            with open(HIPAA_AUDIT_LOG_FILE, 'r') as f:
                audit_log = json.load(f)
                if not isinstance(audit_log, list):
                    audit_log = []
        except Exception as e:
            logger.error(f"Error loading audit log: {e}")
    
    # Add new event and save
    audit_log.append(audit_event)
    try:
        with open(HIPAA_AUDIT_LOG_FILE, 'w') as f:
            json.dump(audit_log, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving audit log: {e}")
        return False

def detect_phi_in_text(text: str) -> List[Dict[str, Any]]:
    """
    Detect potential PHI in text.
    
    Args:
        text: Text to analyze
        
    Returns:
        List of detected PHI instances with type and position
    """
    if not text:
        return []
    
    detected_phi = []
    
    # Define PHI patterns
    phi_patterns = [
        # MRN (Medical Record Number)
        (r'MRN:?\s*(\d{5,10})', 'MRN'),
        # SSN
        (r'(?:\d{3}-\d{2}-\d{4})', 'SSN'),
        # Names (simplified)
        (r'(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})', 'NAME'),
        # Dates of birth
        (r'(?:DOB|Date of Birth)[\s:]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', 'DOB'),
        (r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', 'DATE'),
        # Phone numbers
        (r'(?:\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}', 'PHONE'),
        # Addresses (simplified)
        (r'\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)', 'ADDRESS'),
        # Email addresses
        (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', 'EMAIL'),
    ]
    
    # Detect PHI using patterns
    for pattern, phi_type in phi_patterns:
        for match in re.finditer(pattern, text):
            detected_phi.append({
                'type': phi_type,
                'value': match.group(0),
                'start': match.start(),
                'end': match.end()
            })
    
    return detected_phi

def anonymize_text(text: str, replace_with: str = '[REDACTED]') -> Tuple[str, List[Dict[str, Any]]]:
    """
    Anonymize PHI in text.
    
    Args:
        text: Text to anonymize
        replace_with: Replacement string for PHI
        
    Returns:
        Tuple of (anonymized_text, list_of_redactions)
    """
    if not text:
        return text, []
    
    # Detect PHI
    phi_instances = detect_phi_in_text(text)
    
    # Sort PHI instances by start position in reverse order
    # (to avoid changing positions when replacing)
    phi_instances.sort(key=lambda x: x['start'], reverse=True)
    
    # Create anonymized text by replacing PHI
    anonymized_text = text
    for phi in phi_instances:
        start, end = phi['start'], phi['end']
        anonymized_text = anonymized_text[:start] + replace_with + anonymized_text[end:]
    
    return anonymized_text, phi_instances

def render_hipaa_compliance_ui() -> None:
    """Render the HIPAA compliance UI for administrators."""
    st.title("🔒 HIPAA Compliance Dashboard")
    
    # Check if user is admin
    is_admin = False
    if "auth_data" in st.session_state:
        username = st.session_state.auth_data.get("username")
        if username:
            try:
                with open("auth_config.json", 'r') as f:
                    auth_config = json.load(f)
                is_admin = auth_config.get("users", {}).get(username, {}).get("is_admin", False)
            except Exception as e:
                st.error(f"Error checking admin status: {e}")
    
    if not is_admin:
        st.warning("You must be an administrator to view the HIPAA compliance dashboard.")
        return
    
    # Get current HIPAA configuration
    hipaa_config = get_hipaa_config()
    
    # Display configuration
    st.header("⚙️ HIPAA Configuration")
    
    # PHI Handling tab
    tabs = st.tabs(["PHI Handling", "Audit Logging", "Security", "Access Control"])
    
    # PHI Handling tab
    with tabs[0]:
        st.subheader("Protected Health Information (PHI) Handling")
        
        phi_detect_image = st.checkbox(
            "Detect PHI in uploaded images",
            value=hipaa_config["phi_handling"]["detect_phi_in_images"]
        )
        
        phi_detect_text = st.checkbox(
            "Detect PHI in text inputs and outputs",
            value=hipaa_config["phi_handling"]["detect_phi_in_text"]
        )
        
        anonymize_dicom = st.checkbox(
            "Automatically anonymize DICOM metadata",
            value=hipaa_config["phi_handling"]["anonymize_dicom"]
        )
        
        anonymize_reports = st.checkbox(
            "Anonymize generated reports",
            value=hipaa_config["phi_handling"]["anonymize_reports"]
        )
        
        phi_retention = st.number_input(
            "PHI retention period (days, 0 = don't retain)",
            value=hipaa_config["phi_handling"]["phi_retention_days"],
            min_value=0,
            max_value=365
        )
        
        if st.button("Save PHI Settings"):
            update_hipaa_config("phi_handling", "detect_phi_in_images", phi_detect_image)
            update_hipaa_config("phi_handling", "detect_phi_in_text", phi_detect_text)
            update_hipaa_config("phi_handling", "anonymize_dicom", anonymize_dicom)
            update_hipaa_config("phi_handling", "anonymize_reports", anonymize_reports)
            update_hipaa_config("phi_handling", "phi_retention_days", phi_retention)
            st.success("PHI handling settings updated!")
            
            # Log the configuration change
            log_hipaa_audit_event(
                AuditEventType.SYSTEM_CONFIG,
                description="Updated PHI handling settings",
                additional_data={"section": "phi_handling"}
            )
    
    # Audit Logging tab
    with tabs[1]:
        st.subheader("Audit Logging")
        
        audit_enabled = st.checkbox(
            "Enable audit logging",
            value=hipaa_config["audit_logging"]["enabled"]
        )
        
        log_login = st.checkbox(
            "Log login/logout events",
            value=hipaa_config["audit_logging"]["log_login_events"]
        )
        
        log_phi_access = st.checkbox(
            "Log PHI access events",
            value=hipaa_config["audit_logging"]["log_phi_access"]
        )
        
        log_reports = st.checkbox(
            "Log report generation events",
            value=hipaa_config["audit_logging"]["log_report_generation"]
        )
        
        log_retention = st.number_input(
            "Audit log retention period (days)",
            value=hipaa_config["audit_logging"]["retention_days"],
            min_value=30,
            max_value=3650
        )
        
        if st.button("Save Audit Settings"):
            update_hipaa_config("audit_logging", "enabled", audit_enabled)
            update_hipaa_config("audit_logging", "log_login_events", log_login)
            update_hipaa_config("audit_logging", "log_phi_access", log_phi_access)
            update_hipaa_config("audit_logging", "log_report_generation", log_reports)
            update_hipaa_config("audit_logging", "retention_days", log_retention)
            st.success("Audit logging settings updated!")
            
            # Log the configuration change
            log_hipaa_audit_event(
                AuditEventType.SYSTEM_CONFIG,
                description="Updated audit logging settings",
                additional_data={"section": "audit_logging"}
            )
        
        # Display recent audit logs
        st.subheader("Recent Audit Logs")
        if os.path.exists(HIPAA_AUDIT_LOG_FILE):
            try:
                with open(HIPAA_AUDIT_LOG_FILE, 'r') as f:
                    audit_log = json.load(f)
                
                # Display the most recent 10 events
                if audit_log:
                    recent_events = sorted(
                        audit_log, 
                        key=lambda x: x.get("timestamp", ""), 
                        reverse=True
                    )[:10]
                    
                    for event in recent_events:
                        with st.expander(
                            f"{event.get('timestamp', 'Unknown')} - {event.get('event_type', 'Unknown')} - {event.get('user_id', 'Unknown')}"
                        ):
                            st.json(event)
                else:
                    st.info("No audit events recorded yet.")
            except Exception as e:
                st.error(f"Error loading audit log: {e}")
        else:
            st.info("No audit log file exists yet.")
    
    # Security tab
    with tabs[2]:
        st.subheader("Security Settings")
        
        require_auth = st.checkbox(
            "Require authentication",
            value=hipaa_config["security"]["require_authentication"]
        )
        
        password_expiry = st.number_input(
            "Password expiration (days)",
            value=hipaa_config["security"]["password_expiration_days"],
            min_value=0,
            max_value=365
        )
        
        session_timeout = st.number_input(
            "Session timeout (minutes)",
            value=hipaa_config["security"]["session_timeout_minutes"],
            min_value=5,
            max_value=480
        )
        
        lockout_enabled = st.checkbox(
            "Enable failed login lockout",
            value=hipaa_config["security"]["failed_login_lockout"]
        )
        
        max_attempts = st.number_input(
            "Max failed login attempts before lockout",
            value=hipaa_config["security"]["max_failed_attempts"],
            min_value=3,
            max_value=10
        )
        
        if st.button("Save Security Settings"):
            update_hipaa_config("security", "require_authentication", require_auth)
            update_hipaa_config("security", "password_expiration_days", password_expiry)
            update_hipaa_config("security", "session_timeout_minutes", session_timeout)
            update_hipaa_config("security", "failed_login_lockout", lockout_enabled)
            update_hipaa_config("security", "max_failed_attempts", max_attempts)
            st.success("Security settings updated!")
            
            # Log the configuration change
            log_hipaa_audit_event(
                AuditEventType.SYSTEM_CONFIG,
                description="Updated security settings",
                additional_data={"section": "security"}
            )
    
    # Access Control tab
    with tabs[3]:
        st.subheader("Access Control")
        
        # Display user access
        st.markdown("#### User Access Control")
        
        # Load users from auth config
        try:
            with open("auth_config.json", 'r') as f:
                auth_config = json.load(f)
                users = auth_config.get("users", {})
                
                for username, user_data in users.items():
                    cols = st.columns([3, 1, 1])
                    with cols[0]:
                        st.text(f"Username: {username}")
                    with cols[1]:
                        st.text(f"Admin: {'Yes' if user_data.get('is_admin') else 'No'}")
                    with cols[2]:
                        if st.button("Audit Log", key=f"audit_{username}"):
                            # Show user-specific audit log
                            user_events = []
                            if os.path.exists(HIPAA_AUDIT_LOG_FILE):
                                try:
                                    with open(HIPAA_AUDIT_LOG_FILE, 'r') as f:
                                        audit_log = json.load(f)
                                        user_events = [
                                            event for event in audit_log 
                                            if event.get("user_id") == username
                                        ]
                                except Exception as e:
                                    st.error(f"Error loading audit log: {e}")
                            
                            if user_events:
                                st.json(user_events[:5])  # Show the 5 most recent events
                            else:
                                st.info(f"No audit events found for user {username}")
        except Exception as e:
            st.error(f"Error loading user data: {e}")
    
    # PHI Detection Test
    st.header("🧪 PHI Detection Test")
    st.info("Enter text below to test the PHI detection algorithm")
    
    test_text = st.text_area(
        "Enter text with potential PHI:",
        value="Patient John Smith (DOB: 01/15/1965) was seen on 04/22/2025. Contact: (555) 123-4567. MRN: 12345678"
    )
    
    if st.button("Detect PHI"):
        detected_phi = detect_phi_in_text(test_text)
        
        if detected_phi:
            st.warning(f"Detected {len(detected_phi)} instances of potential PHI:")
            for phi in detected_phi:
                st.markdown(f"- **{phi['type']}**: {phi['value']}")
            
            anonymized_text, _ = anonymize_text(test_text)
            st.subheader("Anonymized Version:")
            st.text(anonymized_text)
        else:
            st.success("No PHI detected in the text")
    
    # Disclaimer
    st.markdown("---")
    st.warning(
        "**Disclaimer:** This HIPAA compliance module is for educational purposes only. "
        "A complete HIPAA compliance program requires comprehensive administrative, technical, "
        "and physical safeguards beyond what this module provides. Consult with HIPAA compliance "
        "experts before implementing any healthcare application with PHI."
    )
