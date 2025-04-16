"""
umls_utils.py
================
Utility helpers for interacting with the UMLS REST API (https://documentation.uts.nlm.nih.gov/) from within
RadVision AI.

Main public functions
---------------------
get_tgt(apikey: str) -> str
    Acquire a Ticket‑Granting Ticket (TGT) required for subsequent requests.
get_service_ticket(tgt: str, service: str = "http://umlsks.nlm.nih.gov") -> str
    Exchange a TGT for a short‑lived Service Ticket (ST).
search_umls(term: str, apikey: str, page_size: int = 5) -> list[dict]
    Search the UMLS Metathesaurus for *term* and return the top *page_size* hits.

Typical Usage
-------------
>>> from umls_utils import search_umls
>>> concepts = search_umls("lung nodule", umls_api_key)
>>> for c in concepts:
...     print(c["name"], c["ui"], c["rootSource"])

The functions perform minimal error‑handling and rate‑limiting; callers should
further wrap them in application‑specific retry / caching logic if needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import time
import requests

# =====================
# Constants & Settings
# =====================
TGT_URL = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
SEARCH_URL = "https://uts-ws.nlm.nih.gov/rest/search/current"
DEFAULT_SERVICE = "http://umlsks.nlm.nih.gov"
USER_AGENT = "RadVisionAI/1.0 (+https://huggingface.co/spaces/mgbam/radvisionai)"
# UMLS recommends not to exceed 20 req/s. We use a conservative sleep to be safe.
RATE_LIMIT_SLEEP = 0.05  # 20 requests per second
TIMEOUT = 15  # seconds


class UMLSAuthError(RuntimeError):
    """Raised when there is an authentication problem with UMLS."""


@dataclass
class UMLSConcept:
    """Dataclass representing a single UMLS search hit."""

    ui: str
    name: str
    rootSource: str
    uri: str
    uriLabel: str | None = None

    @classmethod
    def from_json(cls, item: Dict[str, Any]) -> "UMLSConcept":
        return cls(
            ui=item.get("ui", ""),
            name=item.get("name", ""),
            rootSource=item.get("rootSource", ""),
            uri=item.get("uri", ""),
            uriLabel=item.get("uriLabel"),
        )

    def to_dict(self) -> Dict[str, str]:
        return {
            "ui": self.ui,
            "name": self.name,
            "rootSource": self.rootSource,
            "uri": self.uri,
            "uriLabel": self.uriLabel or "",
        }


# ================
# Auth Helpers
# ================

def get_tgt(apikey: str) -> str:
    """Request a Ticket‑Granting Ticket (TGT).

    Parameters
    ----------
    apikey : str
        Your UMLS API key.

    Returns
    -------
    str
        The URL of the TGT resource.
    """
    headers = {"User-Agent": USER_AGENT}
    resp = requests.post(TGT_URL, data={"apikey": apikey}, headers=headers, timeout=TIMEOUT)
    if resp.status_code != 201:
        raise UMLSAuthError(f"Failed to obtain TGT: {resp.status_code} {resp.text}")
    return resp.headers["location"]


def get_service_ticket(tgt: str, service: str = DEFAULT_SERVICE) -> str:
    """Exchange a TGT for a single‑use Service Ticket (ST)."""
    headers = {"User-Agent": USER_AGENT}
    resp = requests.post(tgt, data={"service": service}, headers=headers, timeout=TIMEOUT)
    if resp.status_code != 200:
        raise UMLSAuthError(f"Failed to obtain ST: {resp.status_code} {resp.text}")
    return resp.text


# ====================
# Core Search Helper
# ====================

def search_umls(term: str, apikey: str, page_size: int = 5) -> List[UMLSConcept]:
    """Search UMLS for *term* and return a list of top concepts.

    This function is purposely lightweight. In a production setting you may want
    to add caching (e.g., with functools.lru_cache or an external cache) to
    reduce latency and quota consumption.
    """
    tgt = get_tgt(apikey)
    st = get_service_ticket(tgt)

    headers = {
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    params = {
        "string": term,
        "ticket": st,
        "pageSize": str(page_size),
    }
    resp = requests.get(SEARCH_URL, params=params, headers=headers, timeout=TIMEOUT)
    time.sleep(RATE_LIMIT_SLEEP)  # simple rate‑limit guard

    if resp.status_code != 200:
        raise RuntimeError(f"UMLS search failed: {resp.status_code} {resp.text}")

    data = resp.json()
    results = data.get("result", {}).get("results", [])
    return [UMLSConcept.from_json(item) for item in results]


# =============
# CLI Testing
# =============
if __name__ == "__main__":
    import os, pprint, sys

    key = os.getenv("UMLS_APIKEY")
    if not key:
        sys.exit("Set UMLS_APIKEY environment variable to test.")

    term = "lung nodule" if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    concepts = search_umls(term, key)
    pprint.pp([c.to_dict() for c in concepts])
