# -*- coding: utf-8 -*-
"""
umls_utils.py - UMLS REST API Interaction Utilities
===================================================

Handles authentication and search requests to the UMLS Terminology Services API.
Exports the UMLSConcept dataclass and checks/reports its own load status.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# --- Dependency Check & Availability ---
_MODULE_LOAD_SUCCESS = False
_REQUESTS_AVAILABLE = False
try:
    import requests
    from requests.exceptions import RequestException
    _REQUESTS_AVAILABLE = True
    _MODULE_LOAD_SUCCESS = True  # Module structure loaded, requests is available
    logging.getLogger(__name__).debug("Successfully imported 'requests' library for UMLS.")
except ImportError:
    logging.getLogger(__name__).warning(
        "Failed to import 'requests' library. UMLS functionality requires 'requests'. "
        "Ensure it is listed in requirements.txt."
    )
    # Define dummy classes/exceptions if requests is not installed,
    # so the rest of the file can be parsed without errors.
    class RequestException(Exception): pass
    class requests: Session = type('Session', (object,), {}) # type: ignore[misc]

# --- Constants and Configuration ---
logger = logging.getLogger(__name__)

TGT_URL: str = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
SEARCH_URL: str = "https://uts-ws.nlm.nih.gov/rest/search/current"
DEFAULT_SERVICE: str = "http://umlsks.nlm.nih.gov"
USER_AGENT: str = "RadVisionAI/1.0 (+https://huggingface.co/spaces/mgbam/radvisionai)" # Customize if needed
TIMEOUT: int = 15  # seconds
RATE_LIMIT_SLEEP: float = 0.06  # ~16 requests/sec max overall

# --- Module Load Status ---
# This flag indicates if the module AND its essential 'requests' dependency loaded.
UMLS_UTILS_LOADED: bool = _MODULE_LOAD_SUCCESS

# --- Custom Exceptions ---
class UMLSError(RuntimeError):
    """Base exception for UMLS utility errors."""
    pass
class UMLSAuthError(UMLSError):
    """Raised for authentication problems (TGT or ST acquisition)."""
    pass
class UMLSConnectionError(UMLSError):
    """Raised for network connectivity issues during API calls."""
    pass
class UMLSSearchError(UMLSError):
    """Raised specifically for errors during the search phase."""
    pass

# --- Data Structures ---
@dataclass
class UMLSConcept:
    """Represents a single concept returned by the UMLS search."""
    ui: str = field(default="")
    name: str = field(default="")
    rootSource: str = field(default="")
    uri: str = field(default="")
    uriLabel: Optional[str] = field(default=None)

    @classmethod
    def from_json(cls, item: Dict[str, Any]) -> UMLSConcept:
        if not isinstance(item, dict):
            logger.warning("Received non-dict item for UMLSConcept creation: %s", type(item))
            return cls()
        return cls(
            ui=item.get("ui", ""),
            name=item.get("name", ""),
            rootSource=item.get("rootSource", ""),
            uri=item.get("uri", ""),
            uriLabel=item.get("uriLabel"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"ui": self.ui, "name": self.name, "rootSource": self.rootSource, "uri": self.uri, "uriLabel": self.uriLabel}

# --- Internal Helper Functions ---

def _create_session() -> requests.Session:
    """Creates a requests Session with default headers."""
    if not _REQUESTS_AVAILABLE:
        raise UMLSError("Cannot create session: 'requests' library is not available.")
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session

def _get_tgt(apikey: str, session: requests.Session) -> str:
    """Requests a Ticket-Granting Ticket (TGT)."""
    if not _REQUESTS_AVAILABLE: raise UMLSError("Cannot get TGT: 'requests' not available.")
    if not apikey: raise UMLSAuthError("UMLS API key is missing or empty.")
    logger.debug("Requesting UMLS TGT...")
    try:
        resp = session.post(TGT_URL, data={"apikey": apikey}, timeout=TIMEOUT)
        time.sleep(RATE_LIMIT_SLEEP)
        resp.raise_for_status()
        if resp.status_code == 201 and "location" in resp.headers:
            logger.debug("Successfully obtained TGT.")
            return resp.headers["location"]
        else:
            raise UMLSAuthError(f"Failed to obtain TGT: Status {resp.status_code}, Headers: {resp.headers}")
    except RequestException as e:
        logger.error("Network error obtaining TGT: %s", e)
        raise UMLSConnectionError(f"Network error obtaining TGT: {e}") from e
    except Exception as e:
        logger.error("Error obtaining TGT: %s", e, exc_info=True)
        if isinstance(e, UMLSAuthError): raise
        raise UMLSError(f"Failed to obtain TGT: {e}") from e

def _get_service_ticket(tgt_location: str, session: requests.Session, service: str = DEFAULT_SERVICE) -> str:
    """Exchanges a TGT for a single-use Service Ticket (ST)."""
    if not _REQUESTS_AVAILABLE: raise UMLSError("Cannot get ST: 'requests' not available.")
    logger.debug("Requesting UMLS Service Ticket (ST)...")
    try:
        resp = session.post(tgt_location, data={"service": service}, timeout=TIMEOUT)
        time.sleep(RATE_LIMIT_SLEEP)
        resp.raise_for_status()
        if resp.status_code == 200 and resp.text:
            logger.debug("Successfully obtained Service Ticket.")
            return resp.text
        else:
             raise UMLSAuthError(f"Failed to obtain ST: Status {resp.status_code}, Response empty: {not resp.text}")
    except RequestException as e:
        logger.error("Network error obtaining ST: %s", e)
        raise UMLSConnectionError(f"Network error obtaining ST: {e}") from e
    except Exception as e:
        logger.error("Error obtaining ST: %s", e, exc_info=True)
        if isinstance(e, UMLSAuthError): raise
        raise UMLSError(f"Failed to obtain ST: {e}") from e

# --- Public Search Function ---

def search_umls(term: str, apikey: str, page_size: int = 5) -> List[UMLSConcept]:
    """Searches the UMLS Metathesaurus for a given term."""
    if not UMLS_UTILS_LOADED:
        raise UMLSError("UMLS search cannot proceed: UMLS Utils or 'requests' library not loaded.")
    if not term: raise ValueError("Search term cannot be empty.")
    if page_size <= 0: raise ValueError("Page size must be positive.")
    if not apikey: raise UMLSAuthError("UMLS API key must be provided for search.")

    session = _create_session()
    try:
        tgt_location = _get_tgt(apikey, session)
        service_ticket = _get_service_ticket(tgt_location, session)

        logger.info("Searching UMLS for '%s' (page size: %d)...", term, page_size)
        headers = {"Accept": "application/json"}
        params = {"string": term, "ticket": service_ticket, "pageSize": str(page_size)}

        resp = session.get(SEARCH_URL, params=params, headers=headers, timeout=TIMEOUT)
        time.sleep(RATE_LIMIT_SLEEP)
        resp.raise_for_status()

        if resp.status_code == 200:
            try:
                data = resp.json()
                results_data = data.get("result", {}).get("results", [])
                if not isinstance(results_data, list):
                     logger.warning("UMLS search 'results' field is not a list: %s", type(results_data))
                     return []
                concepts = [UMLSConcept.from_json(item) for item in results_data]
                logger.info("Found %d UMLS concepts for '%s'.", len(concepts), term)
                return concepts
            except ValueError as json_err:
                logger.error("Failed to decode JSON response from UMLS search: %s", resp.text[:500])
                raise UMLSSearchError("Invalid JSON response received from UMLS search.") from json_err
        else:
            raise UMLSSearchError(f"UMLS search failed with status code {resp.status_code}.")

    except RequestException as e:
        logger.error("Network error during UMLS search for '%s': %s", term, e)
        raise UMLSConnectionError(f"Network error during UMLS search: {e}") from e
    except (UMLSAuthError, UMLSConnectionError, UMLSSearchError, ValueError) as e:
        logger.error("UMLS search failed for '%s': %s", term, e)
        raise e # Re-raise specific known errors
    except Exception as e:
        logger.exception("An unexpected error occurred during UMLS search for '%s'", term)
        raise UMLSError(f"An unexpected error occurred during UMLS search: {e}") from e
    finally:
        session.close()

# --- CLI Testing ---
if __name__ == "__main__":
    # Keep the __main__ block as previously refined for testing
    import os, sys, pprint
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    if not UMLS_UTILS_LOADED: sys.exit("Error: UMLS Utils require 'requests'. Install it.")
    key = os.getenv("UMLS_APIKEY")
    if not key: sys.exit("Error: Set UMLS_APIKEY environment variable.")
    term_to_search = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "myocardial infarction"
    print(f"--- Testing UMLS Search for: '{term_to_search}' ---")
    try:
        results = search_umls(term_to_search, key, page_size=3)
        if results:
             print("--- Results ---")
             pprint.pp([c.to_dict() for c in results])
        else:
             print("--- No results found ---")
    except Exception as e:
        print(f"--- Error during test: {e} ---")