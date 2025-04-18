
# -*- coding: utf-8 -*-
"""
image_processor.py - Image Processing Module for RadVision AI
=============================================================

Handles the processing of uploaded images, including DICOM parsing,
image conversion, and preparation for AI analysis.
"""

import os
import io
import uuid
import hashlib
import logging
from typing import Optional, Dict, Any, Tuple, Union, BinaryIO

import streamlit as st
from PIL import Image, UnidentifiedImageError
import numpy as np

# Import caching utilities
from caching import memory_cache, disk_cache

# Configure logger
logger = logging.getLogger(__name__)

# Try to import DICOM utilities
try:
    import pydicom
    from dicom_utils import parse_dicom, extract_dicom_metadata, dicom_to_image, get_default_wl
    DICOM_UTILS_AVAILABLE = True
    PYDICOM_AVAILABLE = True
except ImportError:
    DICOM_UTILS_AVAILABLE = False
    PYDICOM_AVAILABLE = False
    logger.warning("DICOM utilities not available")

# Check for PIL availability
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL (Pillow) not available")

@memory_cache(max_age_seconds=600)
def hash_file_content(file_content: bytes) -> str:
    """
    Generate a hash from file content for caching and identification.
    
    Args:
        file_content: Binary content of the file
        
    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(file_content).hexdigest()[:16]

def is_dicom_file(file_name: str, file_type: str) -> bool:
    """
    Check if a file is likely a DICOM file based on name and type.
    
    Args:
        file_name: Name of the file
        file_type: MIME type of the file
        
    Returns:
        Boolean indicating if file is DICOM
    """
    if not PYDICOM_AVAILABLE or not DICOM_UTILS_AVAILABLE:
        return False
        
    file_ext = os.path.splitext(file_name)[1].lower()
    return ("dicom" in file_type.lower() or file_ext in (".dcm", ".dicom"))

@disk_cache(max_age_hours=24)
def process_standard_image(image_bytes: bytes) -> Optional[Image.Image]:
    """
    Process standard image formats (JPG, PNG, etc.).
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        PIL Image object or None if processing fails
    """
    if not PIL_AVAILABLE:
        logger.error("Cannot process standard images: Pillow missing")
        return None
    
    try:
        raw_img = Image.open(io.BytesIO(image_bytes))
        processed_img = raw_img.convert("RGB")
        logger.info("Standard image processed successfully")
        return processed_img
    except UnidentifiedImageError:
        logger.error("Could not identify image format")
        return None
    except Exception as e:
        logger.error(f"Error processing standard image: {e}", exc_info=True)
        return None

@disk_cache(max_age_hours=24)
def process_dicom_image(dicom_bytes: bytes, filename: str = "unknown.dcm") -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[Dict], Optional[Any]]:
    """
    Process DICOM image including metadata extraction.
    
    Args:
        dicom_bytes: Raw DICOM file bytes
        filename: Original filename for logging
        
    Returns:
        Tuple of (display_image, processed_image, metadata_dict, dicom_dataset)
    """
    if not DICOM_UTILS_AVAILABLE:
        logger.error("DICOM utilities not available")
        return None, None, None, None
    
    try:
        dicom_dataset = parse_dicom(dicom_bytes, filename=filename)
        if not dicom_dataset:
            logger.error("Could not parse DICOM dataset")
            return None, None, None, None
            
        metadata = extract_dicom_metadata(dicom_dataset)
        default_wc, default_ww = get_default_wl(dicom_dataset)
        
        display_img = dicom_to_image(dicom_dataset, wc=default_wc, ww=default_ww)
        processed_img = dicom_to_image(dicom_dataset, wc=None, ww=None, normalize=True)
        
        if not isinstance(display_img, Image.Image) or not isinstance(processed_img, Image.Image):
            logger.error("Failed to convert DICOM to images")
            return None, None, metadata, dicom_dataset
            
        logger.info(f"DICOM image processed successfully: {filename}")
        return display_img, processed_img, metadata, dicom_dataset
        
    except Exception as e:
        logger.error(f"Error processing DICOM image: {e}", exc_info=True)
        return None, None, None, None

def process_uploaded_file(uploaded_file) -> Dict[str, Any]:
    """
    Process an uploaded file (standard image or DICOM).
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Dictionary with processing results and metadata
    """
    result = {
        "success": False,
        "is_dicom": False,
        "display_image": None,
        "processed_image": None,
        "dicom_metadata": None,
        "dicom_dataset": None,
        "file_info": None,
        "error": None
    }
    
    if uploaded_file is None:
        result["error"] = "No file uploaded"
        return result
    
    try:
        # Read file content and generate unique identifier
        uploaded_file.seek(0)
        file_content = uploaded_file.read()
        uploaded_file.seek(0)
        
        # Generate file info for caching/tracking
        file_hash = hash_file_content(file_content)
        result["file_info"] = f"{uploaded_file.name}-{uploaded_file.size}-{file_hash}"
        
        # Check if DICOM
        is_dicom = is_dicom_file(uploaded_file.name, uploaded_file.type)
        result["is_dicom"] = is_dicom
        
        # Process based on file type
        if is_dicom:
            display_img, processed_img, metadata, dicom_dataset = process_dicom_image(
                file_content, filename=uploaded_file.name
            )
            
            if display_img and processed_img:
                result["display_image"] = display_img
                result["processed_image"] = processed_img
                result["dicom_metadata"] = metadata
                result["dicom_dataset"] = dicom_dataset
                result["success"] = True
            else:
                result["error"] = "Failed to process DICOM image"
                
        else:
            # Process as standard image
            processed_img = process_standard_image(file_content)
            
            if processed_img:
                result["display_image"] = processed_img.copy()
                result["processed_image"] = processed_img
                result["success"] = True
            else:
                result["error"] = "Failed to process standard image"
    
    except Exception as e:
        logger.error(f"Error in process_uploaded_file: {e}", exc_info=True)
        result["error"] = str(e)
    
    return result
