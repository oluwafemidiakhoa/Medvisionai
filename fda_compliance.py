
# -*- coding: utf-8 -*-
"""
fda_compliance.py - FDA Compliance Documentation and Validation
==============================================================

This module implements FDA compliance features for RadVision AI, including:
1. 21 CFR Part 820 Quality System Regulation documentation
2. IEC 62304 Software Development Life Cycle tracking
3. Risk management according to ISO 14971
4. Validation & verification protocols for software as a medical device (SaMD)

FDA Classification: This software would likely be considered a Class II medical device
under 21 CFR 892.2050 (Picture archiving and communications system).
"""

import os
import logging
import datetime
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple

import streamlit as st

# Configure logger
logger = logging.getLogger(__name__)

# Constants
FDA_CONFIG_FILE = "fda_compliance_config.json"
FDA_VALIDATION_RESULTS_FILE = "validation_results.json"

# Default FDA compliance configuration
DEFAULT_FDA_CONFIG = {
    "device_info": {
        "name": "RadVision AI",
        "version": "1.0.0",
        "intended_use": "Research and educational use only. Not for diagnostic purposes.",
        "classification": "Unclassified - Not FDA Cleared/Approved",
        "regulation_number": "21 CFR 892.2050 (If submitted for FDA clearance)",
    },
    "quality_system": {
        "design_controls": {
            "design_inputs": [],
            "design_outputs": [],
            "design_verification": [],
            "design_validation": [],
            "design_transfer": [],
            "design_changes": [],
        },
        "risk_management": {
            "hazards": [],
            "mitigations": [],
            "residual_risks": [],
        },
        "verification_validation": {
            "test_protocols": [],
            "test_results": [],
            "validation_studies": [],
        },
    },
    "regulatory_status": {
        "fda_submission_type": None,  # 510(k), De Novo, PMA, etc.
        "fda_submission_date": None,
        "fda_clearance_date": None,
        "fda_clearance_number": None,
    },
    "labeling": {
        "warning_statement": "CAUTION: Research Device. For Research and Educational Use Only. Not for use in diagnostic procedures.",
        "last_updated": datetime.datetime.now().isoformat(),
    },
    "version_history": []
}

def init_fda_compliance_config() -> None:
    """Initialize the FDA compliance configuration if it doesn't exist."""
    if not os.path.exists(FDA_CONFIG_FILE):
        with open(FDA_CONFIG_FILE, 'w') as f:
            json.dump(DEFAULT_FDA_CONFIG, f, indent=2)
        logger.info("Created default FDA compliance configuration")

def get_fda_compliance_status() -> Dict[str, Any]:
    """
    Get current FDA compliance status.
    
    Returns:
        Dictionary with compliance status information
    """
    if not os.path.exists(FDA_CONFIG_FILE):
        init_fda_compliance_config()
        
    try:
        with open(FDA_CONFIG_FILE, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading FDA compliance config: {e}")
        return DEFAULT_FDA_CONFIG

def update_fda_compliance(section: str, key: str, value: Any) -> bool:
    """
    Update a specific section of the FDA compliance configuration.
    
    Args:
        section: Main section of the config ('device_info', 'quality_system', etc.)
        key: Key within the section to update
        value: New value to set
        
    Returns:
        True if update successful, False otherwise
    """
    if not os.path.exists(FDA_CONFIG_FILE):
        init_fda_compliance_config()
        
    try:
        with open(FDA_CONFIG_FILE, 'r') as f:
            config = json.load(f)
            
        if section in config and key in config[section]:
            config[section][key] = value
            config["version_history"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "change": f"Updated {section}.{key}",
                "user": st.session_state.get("auth_data", {}).get("username", "unknown")
            })
            
            with open(FDA_CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Updated FDA compliance config: {section}.{key}")
            return True
        else:
            logger.error(f"Invalid section or key: {section}.{key}")
            return False
    except Exception as e:
        logger.error(f"Error updating FDA compliance config: {e}")
        return False

def validate_software_integrity() -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Validate software integrity by checking file hashes against expected values.
    This is part of 21 CFR Part 11 compliance for software validation.
    
    Returns:
        Tuple of (passed, validation_results)
    """
    critical_files = [
        "app.py", 
        "llm_interactions.py", 
        "umls_utils.py",
        "dicom_utils.py",
        "report_utils.py"
    ]
    
    results = []
    all_passed = True
    
    # Get baseline hashes if available
    baseline_hashes = {}
    if os.path.exists(FDA_VALIDATION_RESULTS_FILE):
        try:
            with open(FDA_VALIDATION_RESULTS_FILE, 'r') as f:
                validation_data = json.load(f)
                baseline_hashes = validation_data.get("file_hashes", {})
        except Exception as e:
            logger.error(f"Error loading validation results: {e}")
    
    # Compute current hashes
    current_hashes = {}
    for file_path in critical_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                    file_hash = hashlib.sha256(file_content).hexdigest()
                    current_hashes[file_path] = file_hash
                    
                    # Compare with baseline if available
                    if file_path in baseline_hashes:
                        match = file_hash == baseline_hashes[file_path]
                        if not match:
                            all_passed = False
                    else:
                        match = None  # No baseline to compare
                        
                    results.append({
                        "file_path": file_path,
                        "hash": file_hash,
                        "expected_hash": baseline_hashes.get(file_path),
                        "hash_match": match,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
            except Exception as e:
                logger.error(f"Error computing hash for {file_path}: {e}")
                results.append({
                    "file_path": file_path,
                    "error": str(e),
                    "timestamp": datetime.datetime.now().isoformat()
                })
                all_passed = False
        else:
            logger.warning(f"File not found: {file_path}")
            results.append({
                "file_path": file_path,
                "error": "File not found",
                "timestamp": datetime.datetime.now().isoformat()
            })
            all_passed = False
    
    # Save validation results
    validation_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "passed": all_passed,
        "results": results,
        "file_hashes": current_hashes
    }
    
    try:
        with open(FDA_VALIDATION_RESULTS_FILE, 'w') as f:
            json.dump(validation_data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving validation results: {e}")
    
    return all_passed, results

def create_validation_baseline() -> bool:
    """
    Create a baseline for software validation by storing current file hashes.
    
    Returns:
        True if baseline created successfully, False otherwise
    """
    _, results = validate_software_integrity()
    
    # Extract hashes from results
    file_hashes = {}
    for result in results:
        if "hash" in result:
            file_hashes[result["file_path"]] = result["hash"]
    
    # Save baseline
    validation_data = {
        "baseline_timestamp": datetime.datetime.now().isoformat(),
        "created_by": st.session_state.get("auth_data", {}).get("username", "unknown"),
        "file_hashes": file_hashes
    }
    
    try:
        with open(FDA_VALIDATION_RESULTS_FILE, 'w') as f:
            json.dump(validation_data, f, indent=2)
        logger.info("Created software validation baseline")
        return True
    except Exception as e:
        logger.error(f"Error creating validation baseline: {e}")
        return False

def render_fda_compliance_ui() -> None:
    """Render the FDA compliance UI for administrators."""
    st.title("🏥 FDA Compliance Dashboard")
    
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
        st.warning("You must be an administrator to view the FDA compliance dashboard.")
        return
    
    # Get current compliance status
    compliance_status = get_fda_compliance_status()
    
    # Display compliance status
    st.header("📋 Device Information")
    st.json(compliance_status["device_info"])
    
    st.header("⚠️ Regulatory Status")
    st.json(compliance_status["regulatory_status"])
    
    st.header("🔬 Validation Status")
    
    # Add validation button
    if st.button("Run Software Validation"):
        passed, results = validate_software_integrity()
        if passed:
            st.success("Software validation passed!")
        else:
            st.error("Software validation failed. See results below.")
        st.json(results)
    
    # Add baseline creation button
    if st.button("Create Validation Baseline"):
        success = create_validation_baseline()
        if success:
            st.success("Created new validation baseline!")
        else:
            st.error("Failed to create validation baseline.")
    
    # Add compliance report generation
    st.header("📑 Compliance Documentation")
    if st.button("Generate Compliance Report"):
        st.info("Generating compliance report...")
        # In a real implementation, this would generate a PDF report
        st.download_button(
            label="Download Compliance Report",
            data=json.dumps(compliance_status, indent=2),
            file_name=f"fda_compliance_report_{datetime.datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
    
    # FDA compliance checklist
    st.header("✅ Compliance Checklist")
    
    checklist_items = [
        {"name": "Design Controls Documentation", "key": "design_controls", "required": True},
        {"name": "Risk Analysis", "key": "risk_analysis", "required": True},
        {"name": "Software Validation", "key": "software_validation", "required": True},
        {"name": "Labeling Review", "key": "labeling", "required": True},
        {"name": "Clinical Validation", "key": "clinical_validation", "required": True},
        {"name": "Cybersecurity Assessment", "key": "cybersecurity", "required": True},
        {"name": "Regulatory Strategy", "key": "regulatory_strategy", "required": False},
        {"name": "Post-Market Surveillance Plan", "key": "postmarket", "required": False}
    ]
    
    for item in checklist_items:
        status = st.checkbox(
            f"{item['name']} {'(Required)' if item['required'] else '(Recommended)'}", 
            value=False,
            key=f"checklist_{item['key']}"
        )
        if status:
            st.success(f"{item['name']} completed")
        elif item["required"]:
            st.warning(f"{item['name']} required for FDA submission")
        else:
            st.info(f"{item['name']} recommended for FDA submission")
    
    # Disclaimer
    st.markdown("---")
    st.warning(
        "**Disclaimer:** This FDA compliance dashboard is for educational purposes only. "
        "A real FDA submission requires comprehensive documentation and expert regulatory guidance. "
        "Software claiming to be FDA compliant must meet all applicable requirements of 21 CFR Part 820, "
        "and if utilizing AI/ML, should follow FDA guidance on Artificial Intelligence and Machine Learning in Software."
    )
