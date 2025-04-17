# -*- coding: utf-8 -*-
"""
umls_utils.py – Tiny helper around the UMLS REST API
===================================================

Public surface
--------------
search_umls(term: str, apikey: str, *, page_size: int = 5) -> list[UMLSConcept]
get_cached_tgt(apikey: str) -> str
flush_tgt_cache() -> None

A minimal, dependency‑free cache keeps the TGT for 8 hours inside the module
process; Hugging Face Spaces restart often enough that this is sufficient.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Std‑lib
# ─────────────────────────────────────────────────────────────────────────────
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Third‑party
# ─────────────────────────────────────────────────────────────────────────────
try:
    import requests
    from requests.exceptions import RequestException
except ImportError as exc:                                        # pragma: no cover
    raise ImportError(
        "'requests' library is required for umls_utils.py – add it to requirements.txt"
    ) from exc

# ─────────────────────────────────────────────────────────────────────────────
# Constants – override in tests if you like
# ─────────────────────────────────────────────────────────────────────────────
TGT_URL              = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
STS_URL              = "https://uts-ws.nlm.nih.gov/rest/search/current"
DEFAULT_SERVICE      = "http://umlsks.nlm.nih.gov"
USER_AGENT           = "RadVisionAI/1.0 (+https://hf.co/spaces/mgbam/radvisionai)"
TIMEOUT              = 15          # s
SLEEP_BETWEEN_CALLS  = 0.06        # ~16 req/s overall
TGT_CACHE_SECONDS    = 8 * 60 * 60 # 8 h

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────
class UMLSError(RuntimeError):          """Base error for this module."""
class UMLSAuthError(UMLSError):          """Authentication (TGT / ST) failed."""
class UMLSSearchError(UMLSError):        """Search endpoint returned an error."""
class UMLSConnectionError(UMLSError):    """Network‑level error."""

# ─────────────────────────────────────────────────────────────────────────────
# Dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class UMLSConcept:
    ui:          str
    name:        str
    rootSource:  str
    uri:         str
    uriLabel:    Optional[str] = None

    @classmethod
    def from_json(cls, item: Dict[str, Any]) -> "UMLSConcept":
        return cls(
            ui         = item.get("ui", ""),
            name       = item.get("name", ""),
            rootSource = item.get("rootSource", ""),
            uri        = item.get("uri", ""),
            uriLabel   = item.get("uriLabel"),
        )

    def to_dict(self) -> Dict[str, str]:
        return {
            "ui":         self.ui,
            "name":       self.name,
            "rootSource": self.rootSource,
            "uri":        self.uri,
            "uriLabel":   self.uriLabel or "",
        }

# ─────────────────────────────────────────────────────────────────────────────
# Simple in‑memory TGT cache
# ─────────────────────────────────────────────────────────────────────────────
_tgt_cache: Dict[str, tuple[str, float]] = {}      # {apikey: (tgt_url, expiry_ts)}

def flush_tgt_cache() -> None:
    """Erase the in‑memory TGT cache (mainly for unit tests)."""
    _tgt_cache.clear()
    logger.debug("TGT cache flushed.")

def get_cached_tgt(apikey: str) -> str:
    """Return a cached TGT or fetch a new one if expired/absent."""
    tgt, exp = _tgt_cache.get(apikey, ("", 0.0))
    if time.time() < exp:
        logger.debug("Re‑using cached TGT (expires in %.0f s).", exp - time.time())
        return tgt

    logger.debug("Requesting new TGT…")
    try:
        resp = _SESSION.post(
            TGT_URL,
            data={"apikey": apikey},
            timeout=TIMEOUT,
        )
        time.sleep(SLEEP_BETWEEN_CALLS)
        resp.raise_for_status()
        if resp.status_code != 201 or "location" not in resp.headers:
            raise UMLSAuthError(f"TGT request failed: HTTP {resp.status_code}")
        tgt_url = resp.headers["location"]
    except RequestException as exc:
        raise UMLSConnectionError(f"Network error obtaining TGT – {exc}") from exc

    _tgt_cache[apikey] = (tgt_url, time.time() + TGT_CACHE_SECONDS)
    logger.info("Obtained new TGT; cached for %dh.", TGT_CACHE_SECONDS // 3600)
    return tgt_url

# ─────────────────────────────────────────────────────────────────────────────
# Shared requests.Session
# ─────────────────────────────────────────────────────────────────────────────
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": USER_AGENT})

# ─────────────────────────────────────────────────────────────────────────────
# Public function
# ─────────────────────────────────────────────────────────────────────────────
def search_umls(term: str, apikey: str, *, page_size: int = 5) -> List[UMLSConcept]:
    """
    Query the UMLS Metathesaurus search endpoint.

    Parameters
    ----------
    term : str
        Search phrase, e.g. “pulmonary embolism”.
    apikey : str
        Your personal UMLS API key (will be exchanged for TGT → ST).
    page_size : int, default 5
        Maximum number of results.

    Returns
    -------
    list[UMLSConcept]
    """
    if not term.strip():
        raise ValueError("search_umls: term may not be empty.")
    if page_size <= 0:
        raise ValueError("search_umls: page_size must be positive.")

    # 1 • get TGT (cached) → ST
    tgt_url = get_cached_tgt(apikey)                    # may raise
    try:
        resp_st = _SESSION.post(
            tgt_url, data={"service": DEFAULT_SERVICE}, timeout=TIMEOUT
        )
        time.sleep(SLEEP_BETWEEN_CALLS)
        resp_st.raise_for_status()
        st_ticket = resp_st.text
    except RequestException as exc:
        raise UMLSConnectionError(f"Network error obtaining ST – {exc}") from exc
    if not st_ticket:
        raise UMLSAuthError("Received empty ST ticket from UMLS.")

    # 2 • search
    params = {"string": term, "ticket": st_ticket, "pageSize": str(page_size)}
    try:
        resp = _SESSION.get(STS_URL, params=params, timeout=TIMEOUT)
        time.sleep(SLEEP_BETWEEN_CALLS)
        resp.raise_for_status()
        data = resp.json()
    except RequestException as exc:
        raise UMLSConnectionError(f"Network error during search – {exc}") from exc
    except ValueError as exc:
        raise UMLSSearchError("Invalid JSON from UMLS search.") from exc

    items = data.get("result", {}).get("results", [])
    if not isinstance(items, list):
        logger.warning("UMLS search: unexpected JSON structure.")
        return []

    return [UMLSConcept.from_json(it) for it in items]

# ─────────────────────────────────────────────────────────────────────────────
# Quick CLI test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":                              # pragma: no cover
    import os, pprint, sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    key  = os.getenv("UMLS_APIKEY") or sys.exit("Set UMLS_APIKEY first.")
    term = " ".join(sys.argv[1:]) or "pulmonary embolism"
    print(f"Searching UMLS for: {term!r} …")
    try:
        concepts = search_umls(term, key, page_size=3)
    except UMLSError as exc:
        sys.exit(f"UMLS error: {exc}")

    if concepts:
        pprint.pp([c.to_dict() for c in concepts])
    else:
        print("No concepts found.")
