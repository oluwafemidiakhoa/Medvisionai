# -*- coding: utf-8 -*-
"""
app.py – RadVision AI Advanced (main entry‑point)
================================================
Lightweight orchestrator that wires together:

• sidebar_ui.py    – upload panel & action buttons  
• main_page_ui.py  – viewer + tabbed results (AI / UMLS / Translate …)  
• file_processing.py – DICOM / image ingestion + ROI plumbing  
• action_handlers.py – runs Gemini, UMLS enrichment, PDF reporting  

Heavy logic lives in the helper modules; this file focuses on flow control.
"""
from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Standard library
# ──────────────────────────────────────────────────────────────────────────────
import logging
import sys
import io
import base64
from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
# 3rd‑party
# ──────────────────────────────────────────────────────────────────────────────
import streamlit as st

# Pillow is required for the image‑to‑data‑URL monkey‑patch (for st_canvas).
try:
    from PIL import Image  # noqa: WPS433 – external import
    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover – Space will show UI error banner
    PIL_AVAILABLE = False
    Image = None  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# Internal modules (each kept small & focused)
# ──────────────────────────────────────────────────────────────────────────────
from config import LOG_LEVEL, LOG_FORMAT, DATE_FORMAT, APP_CSS, FOOTER_MARKDOWN
from session_state import initialize_session_state
from sidebar_ui import render_sidebar
from main_page_ui import render_main_content
from file_processing import handle_file_upload
from action_handlers import handle_action

# -----------------------------------------------------------------------------
# Streamlit page config – **must** be first Streamlit call
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RadVision AI Advanced",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.info("↪︎  RadVision AI bootstrap (Streamlit v%s)", st.__version__)

# -----------------------------------------------------------------------------
# Session‑state defaults
# -----------------------------------------------------------------------------
initialize_session_state()

# -----------------------------------------------------------------------------
# Inject custom global CSS
# -----------------------------------------------------------------------------
st.markdown(APP_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Monkey‑patch: st.elements.image.image_to_url  ⇢  data‑URL encoder
# Needed for streamlit‑drawable‑canvas screenshots
# -----------------------------------------------------------------------------
import streamlit.elements.image as _st_image  # noqa: WPS433

if not hasattr(_st_image, "image_to_url"):

    def _image_to_url_monkey_patch(       # noqa: C901 – kept simple & explicit
        img_obj: Any,
        width: int = -1,
        clamp: bool = False,
        channels: str = "RGB",
        output_format: str = "auto",
        image_id: str = "",
    ) -> str:
        """Return a **data:URL** for a PIL Image so st_canvas can re‑render."""
        if not (PIL_AVAILABLE and isinstance(img_obj, Image.Image)):
            logger.warning("[patch] image_to_url: unsupported type %s", type(img_obj))
            return ""

        # ---------- normalise format ----------
        fmt = (output_format.upper() if output_format != "auto" else (img_obj.format or "PNG"))
        if fmt not in {"PNG", "JPEG", "WEBP", "GIF"}:
            fmt = "PNG"
        if img_obj.mode == "RGBA" and fmt == "JPEG":  # JPEG has no alpha
            fmt = "PNG"

        # ---------- convert if needed ----------
        if img_obj.mode == "P":             # palette → RGBA
            img_obj = img_obj.convert("RGBA")
        if channels == "RGB" and img_obj.mode not in {"RGB", "L"}:
            img_obj = img_obj.convert("RGB")

        # ---------- encode ----------
        buf = io.BytesIO()
        img_obj.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/{fmt.lower()};base64,{b64}"

    _st_image.image_to_url = _image_to_url_monkey_patch  # type: ignore[attr-defined]
    logger.info("✔︎  Patched st.image → data‑URL encoder")

# ──────────────────────────────────────────────────────────────────────────────
# UI flow
# ──────────────────────────────────────────────────────────────────────────────
# 1️⃣  Sidebar (upload, actions)  → returns file‑uploader object
uploaded_file = render_sidebar()

# 2️⃣  Process uploaded file (populate session‑state images / DICOM data)
handle_file_upload(uploaded_file)

# 3️⃣  Main page layout (left: viewer · right: tabbed results)
st.markdown("---")
st.title("⚕️ RadVision AI Advanced · AI‑Assisted Image Analysis")
with st.expander("User Guide & Disclaimer", expanded=False):
    st.warning(
        "⚠️ **Disclaimer**: For research / educational use only – "
        "not intended for primary diagnostic decisions.",
    )
    st.markdown(
        """
        **Workflow**
        1. **Upload** image (or enable **Demo Mode**).  
        2. *(DICOM)* adjust **Window / Level** if needed.  
        3. *(optional)* draw an **ROI** rectangle on the viewer.  
        4. Trigger **AI actions** from the sidebar.  
        5. Explore tabs – **Translate**, **UMLS**, **Confidence** …  
        6. **Generate PDF** for a portable report.
        """,
    )
st.markdown("---")

col1, col2 = st.columns([2, 3], gap="large")
render_main_content(col1, col2)

# 4️⃣  Perform deferred action (set via sidebar buttons)
if (action := st.session_state.get("last_action")):
    handle_action(action)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(f"⚕️ RadVision AI Advanced | Session ID: {st.session_state.get('session_id', 'N/A')}")
st.markdown(FOOTER_MARKDOWN, unsafe_allow_html=True)
logger.info("✓  Render cycle complete – Session ID %s", st.session_state.get("session_id"))
