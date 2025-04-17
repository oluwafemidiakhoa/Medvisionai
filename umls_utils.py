from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import time
import os
import requests

# =====================
# Constants & Settings
# =====================
TGT_URL = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
SEARCH_URL = "https://uts-ws.nlm.nih.gov/rest/search/current"
DEFAULT_SERVICE = "http://umlsks.nlm.nih.gov"
USER_AGENT = "RadVisionAI/1.0 (+https://huggingface.co/spaces/mgbam/radvisionai)"
# UMLS recommends not to exceed 20 requests/sec
RATE_LIMIT_SLEEP = 0.05  # seconds between calls
TIMEOUT = 15  # seconds for HTTP requests

# Default number of UMLS hits
DEFAULT_UMLS_HITS: int = int(os.getenv("UMLS_HITS", "5"))
# Optional source filtering (comma-separated string in env)
SOURCE_FILTER: List[str] = (
    os.getenv("UMLS_SOURCE_FILTER", "").split(',')
    if os.getenv("UMLS_SOURCE_FILTER") else []
)


class UMLSAuthError(RuntimeError):
    """Raised when there is an authentication problem with UMLS."""


@dataclass
class UMLSConcept:
    """Dataclass representing a single UMLS search hit."""
    ui: str
    name: str
    rootSource: str
    uri: str
    uriLabel: Optional[str] = None

    @classmethod
    def from_json(cls, item: Dict[str, Any]) -> UMLSConcept:
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
# Authentication
# ================

def get_tgt(apikey: str) -> str:
    """Acquire a Ticket-Granting Ticket (TGT) for UMLS API calls."""
    headers = {"User-Agent": USER_AGENT}
    resp = requests.post(TGT_URL, data={"apikey": apikey}, headers=headers, timeout=TIMEOUT)
    if resp.status_code != 201:
        raise UMLSAuthError(f"Failed to obtain TGT: {resp.status_code} {resp.text}")
    return resp.headers.get("location", "")


def get_service_ticket(tgt: str, service: str = DEFAULT_SERVICE) -> str:
    """Exchange a TGT for a single-use Service Ticket (ST)."""
    headers = {"User-Agent": USER_AGENT}
    resp = requests.post(tgt, data={"service": service}, headers=headers, timeout=TIMEOUT)
    if resp.status_code != 200:
        raise UMLSAuthError(f"Failed to obtain ST: {resp.status_code} {resp.text}")
    return resp.text


# ====================
# Core Search Helper
# ====================

def search_umls(
    term: str,
    apikey: str,
    page_size: int = DEFAULT_UMLS_HITS,
    source_filter: Optional[List[str]] = None
) -> List[UMLSConcept]:
    """
    Search the UMLS Metathesaurus for *term* and return top concepts.

    Args:
        term: The search string.
        apikey: UMLS API key.
        page_size: Number of results to fetch.
        source_filter: Optional list of source vocabularies to filter by.
    Returns:
        A list of UMLSConcept objects.
    """
    # Authenticate
    tgt = get_tgt(apikey)
    st = get_service_ticket(tgt)

    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    params = {"string": term, "ticket": st, "pageSize": str(page_size)}

    resp = requests.get(SEARCH_URL, params=params, headers=headers, timeout=TIMEOUT)
    time.sleep(RATE_LIMIT_SLEEP)

    if resp.status_code != 200:
        raise RuntimeError(f"UMLS search failed: {resp.status_code} {resp.text}")

    data = resp.json()
    results = data.get("result", {}).get("results", [])

    concepts = [UMLSConcept.from_json(item) for item in results]
    # Apply default and override filters
    filter_list = source_filter if source_filter is not None else SOURCE_FILTER
    if filter_list:
        concepts = [c for c in concepts if c.rootSource in filter_list]
    return concepts


# =============
# CLI Testing
# =============
if __name__ == "__main__":
    import sys, pprint

    key = os.getenv("UMLS_APIKEY")
    if not key:
        sys.exit("Set UMLS_APIKEY environment variable to test.")

    term = "lung nodule" if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    concepts = search_umls(term, key)
    pprint.pp([c.to_dict() for c in concepts])
