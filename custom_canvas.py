
# -*- coding: utf-8 -*-
"""
custom_canvas.py - Alternative ROI selection for RadVision AI
=============================================================
This module provides a fallback ROI selection method using regular Streamlit
components when streamlit-drawable-canvas is unavailable or incompatible.
"""

import streamlit as st
from PIL import Image
import json
from typing import Dict, Any, Optional, Tuple

def st_roi_selector(
    background_image: Image.Image, 
    key: str = "roi_selector",
    initial_coords: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """
    A fallback ROI selector using standard Streamlit components.
    
    Args:
        background_image: PIL Image to display and select ROI from
        key: Unique key for the component
        initial_coords: Optional initial coordinates (left, top, width, height)
        
    Returns:
        Dict with json_data containing the ROI selection (compatible with st_canvas format)
    """
    img_width, img_height = background_image.size
    
    st.image(background_image, caption="Image for ROI selection", use_column_width=True)
    
    # Get default values from initial_coords if provided
    default_left = initial_coords.get("left", 0) if initial_coords else 0
    default_top = initial_coords.get("top", 0) if initial_coords else 0
    default_width = initial_coords.get("width", img_width // 4) if initial_coords else img_width // 4
    default_height = initial_coords.get("height", img_height // 4) if initial_coords else img_height // 4
    
    # Show sliders for ROI selection
    st.write("Adjust ROI coordinates:")
    col1, col2 = st.columns(2)
    
    with col1:
        left = st.slider("Left position (x)", 0, img_width-10, default_left, key=f"{key}_left")
        width = st.slider("Width", 10, img_width-left, default_width, key=f"{key}_width")
    
    with col2:
        top = st.slider("Top position (y)", 0, img_height-10, default_top, key=f"{key}_top")
        height = st.slider("Height", 10, img_height-top, default_height, key=f"{key}_height")
    
    # Create a dict that mimics the format of st_canvas result
    objects = [{
        "type": "rect",
        "left": left,
        "top": top,
        "width": width,
        "height": height,
    }]
    
    result = {
        "json_data": {
            "objects": objects
        }
    }
    
    # Return in a format compatible with st_canvas
    return result
