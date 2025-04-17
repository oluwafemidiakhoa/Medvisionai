# -*- coding: utf-8 -*-
"""
app.py · RadVision AI Advanced
──────────────────────────────
Main orchestrator: pulls together the sidebar, viewer/tabs, file‑loader and
action handler modules.  No heavy logic lives here; that keeps the UI snappy
and individual parts testable.
"""
from __future__ import annotations

# ─────────────────────────────  std‑lib  ──────────────────────────────
import io, sys, os, base64, logging
from typing import Any, TYPE_CHECKING

# ──────────────────────────  third‑party  ─────────────────────────────
import streamlit as st

# Pillow is optional but strongly recommended – required by the monkey‑patch
try:
    from PIL import Image  # noqa: WPS433
    PIL_AVAILABLE = True
except ImportError:  # fallback dummy so isinstance() never crashes
    PIL_AVAILABLE = False
    class _ImgProxy: ...     # noqa: WPS604
    Image = _ImgProxy        # type: ignore

# ───────────────────────────  local code  ─────────────────────────────
from config          import (
    LOG_LEVEL, LOG_FORMAT, DATE_FORMAT,
    APP_TITLE, APP_ICON, APP_CSS, FOOTER_MARKDOWN,
    USER_GUIDE_MARKDOWN, DISCLAIMER_WARNING, UMLS_CONFIG_MSG,
)
from session_state   import (
    initialize_session_state,
)
from sidebar_ui      import render_sidebar
from file_processing import handle_file_upload
from main_page_ui    import render_main_content
from action_handlers import handle_action

# Optional (show status banners if missing)
try:
    from translation_models import TRANSLATION_AVAILABLE, TRANSLATION_CONFIG_MSG
except Exception:  # pragma: no cover
    TRANSLATION_AVAILABLE, TRANSLATION_CONFIG_MSG = False, (
        "Translation module not found or `deep‑translator` missing."
    )
try:
    from umls_utils import UMLS_UTILS_LOADED
except Exception:  # pragma: no cover
    UMLS_UTILS_LOADED = False

# ─────────────────────────  Streamlit page  ───────────────────────────
st.set_page_config(
    page_title=APP_TITLE, page_icon=APP_ICON,
    layout="wide", initial_sidebar_state="expanded",
)

# ─────────────────────────────  logging  ──────────────────────────────
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(
    level=LOG_LEVEL, format=LOG_FORMAT, datefmt=DATE_FORMAT,
    stream=sys.stdout, force=True,
)
log = logging.getLogger(__name__)
log.info("⇢ Booting RadVision AI (Streamlit %s)", st.__version__)

# ────────────────────────  session initialisation  ────────────────────
initialize_session_state()
get_sid = lambda: st.session_state.get("session_id", "N/A")               # noqa: E731
log.debug("Session initialised → ID %s", get_sid())

# ────────────────────────────  global CSS  ────────────────────────────
st.markdown(APP_CSS, unsafe_allow_html=True)

# ─────────────────────  monkey‑patch for streamlit‑canvas  ────────────
import streamlit.elements.image as _st_img  # noqa: WPS433

if not hasattr(_st_img, "image_to_url") and PIL_AVAILABLE:
    def _img_to_url(img: Any, *_, **__) -> str:                           # noqa: D401
        if not isinstance(img, Image):
            log.warning("Patch: object %s not a PIL Image", type(img))
            return ""
        buf = io.BytesIO()
        fmt = (img.format or "PNG").upper()
        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode == "RGBA" and fmt == "JPEG":
            fmt = "PNG"
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/{fmt.lower()};base64,{b64}"
    _st_img.image_to_url = _img_to_url        # type: ignore[attr-defined]
    log.info("Applied image→data‑URL monkey‑patch")
elif not PIL_AVAILABLE:
    log.warning("Pillow missing – drawable‑canvas will not work")

# ─────────────────────────────  SIDEBAR  ──────────────────────────────
uploaded = render_sidebar()            # <class 'UploadedFile'> | None
handle_file_upload(uploaded)           # sets .display_image & friends

# ──────────────────────────────  MAIN UI  ─────────────────────────────
st.divider()
st.title(f"{APP_ICON} {APP_TITLE} · AI‑Assisted Image Analysis")

with st.expander("User Guide & Disclaimer"):
    st.warning(f"⚠️ **Disclaimer:** {DISCLAIMER_WARNING}")
    st.markdown(USER_GUIDE_MARKDOWN, unsafe_allow_html=True)

st.divider()
col_left, col_right = st.columns([2, 3], gap="large")
render_main_content(col_left, col_right)

# ─────────────────────────────  ACTIONS  ──────────────────────────────
if (act := st.session_state.get("last_action")):
    log.info("Running deferred action «%s»", act)
    handle_action(act)
    st.session_state.last_action = None       # reset trigger

# ───────────────────────────  status banners  ─────────────────────────
if not TRANSLATION_AVAILABLE:
    st.warning(f"🌐 Translation unavailable – {TRANSLATION_CONFIG_MSG}")

if not (UMLS_UTILS_LOADED and os.getenv("UMLS_APIKEY")):
    reason = (
        "UMLS utilities failed to import."
        if not UMLS_UTILS_LOADED else UMLS_CONFIG_MSG
    )
    st.warning(f"🧬 UMLS features unavailable – {reason}")

# ──────────────────────────────  FOOTER  ──────────────────────────────
st.divider()
st.caption(f"{APP_ICON} {APP_TITLE} | Session ID {get_sid()}")
st.markdown(FOOTER_MARKDOWN, unsafe_allow_html=True)
log.info("⇠ Render finished (session %s)", get_sid())
