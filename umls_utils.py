# -*- coding: utf-8 -*-
"""
umls_utils.py
================
Utility helpers for interacting with the UMLS REST API (https://documentation.uts.nlm.nih.gov/)
from within RadVision AI.

Provides functions for authentication (TGT/ST acquisition) and Metathesaurus search.

Main public functions
---------------------
get_tgt(apikey: str) -> str
    Acquire a Ticket-Granting Ticket (TGT).
get_service_ticket(tgt_location: str) -> str
    Exchange a TGT for a short-lived Service Ticket (ST).
search_umls(term: str, apikey: str, page_size: int = 5) -> list[UMLSConcept]
    Search the UMLS Metathesaurus for *term*.

Usage Notes
-----------
- Requires a UMLS API key (obtainable from the UTS website).
- API key should be securely managed (e.g., environment variables, secrets management).
- Basic rate limiting is included, but robust applications might need more sophisticated
  strategies (e.g., exponential backoff, token buckets).
- Consider adding caching (e.g., `functools.lru_cache` or Redis/Memcached) around
  `search_umls` in high-volume scenarios to reduce API calls and latency.
"""
from __future__ import annotations

import logging
import time
import os
from typing import List, Dict, Any
from dataclasses import dataclass

import requests # Dependency: Make sure 'requests' is installed

# =====================
# Module Configuration
# =====================

# --- API Endpoints ---
TGT_URL = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
SEARCH_URL = "https://uts-ws.nlm.nih.gov/rest/search/current"
DEFAULT_SERVICE = "http://umlsks.nlm.nih.gov" # Service URL for which ST is requested

# --- Request Settings ---
USER_AGENT = "RadVisionAI/1.0 (+https://huggingface.co/spaces/mgbam/radvisionai)" # Identify client
# UMLS recommends not exceeding 20 req/s. Sleep ensures ~compliance.
RATE_LIMIT_SLEEP_SECONDS = 0.06 # Slightly > 1/20s
REQUEST_TIMEOUT_SECONDS = 15

# --- Logging ---
logger = logging.getLogger(__name__) # Get logger instance for this module

# --- Availability Check & Configuration Message (for app.py) ---
# Check if the required dependency (requests) is installed.
# The presence of the API key is checked at runtime when functions are called.
try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

# Check for API Key in environment - determines practical availability
_API_KEY_PRESENT = bool(os.getenv("UMLS_APIKEY"))

UMLS_AVAILABLE = _REQUESTS_AVAILABLE and _API_KEY_PRESENT
UMLS_CONFIG_MSG = (
    "Install `requests` library and set `UMLS_APIKEY` in HF Secrets."
    if not _REQUESTS_AVAILABLE
    else "Set `UMLS_APIKEY` in Hugging Face Secrets & restart."
    if not _API_KEY_PRESENT
    else "UMLS configured." # Should not be seen if UMLS_AVAILABLE is False
)

# =====================
# Custom Exceptions
# =====================

class UMLSError(RuntimeError):
    """Base class for UMLS utility errors."""
    pass

class UMLSAuthError(UMLSError):
    """Raised for authentication problems (TGT or ST acquisition)."""
    pass

class UMLSSearchError(UMLSError):
    """Raised for errors during the search operation."""
    pass

# =====================
# Data Structure
# =====================

@dataclass(frozen=True) # Immutable dataclass
class UMLSConcept:
    """Represents a single concept returned by the UMLS search."""
    ui: str         # Unique Identifier
    name: str       # Preferred Name
    rootSource: str # Source Vocabulary (e.g., SNOMEDCT_US, MSH)
    uri: str        # REST API URI for this concept
    uriLabel: str | None = None # Optional human-readable label for the URI

    @classmethod
    def from_json(cls, item: Dict[str, Any]) -> UMLSConcept:
        """Factory method to create a UMLSConcept from a JSON dictionary item."""
        if not isinstance(item, dict):
            logger.warning("Received non-dict item for UMLSConcept creation: %s", type(item))
            return cls(ui="", name="Invalid Data", rootSource="", uri="") # Return dummy object

        return cls(
            ui=item.get("ui", "N/A"), # Provide defaults if keys missing
            name=item.get("name", "Unknown Name"),
            rootSource=item.get("rootSource", "Unknown Source"),
            uri=item.get("uri", ""),
            uriLabel=item.get("uriLabel"), # Can be None
        )

    def to_dict(self) -> Dict[str, str]:
        """Serializes the concept to a dictionary (useful for display/storage)."""
        return {
            "ui": self.ui,
            "name": self.name,
            "rootSource": self.rootSource,
            "uri": self.uri,
            "uriLabel": self.uriLabel or "", # Ensure uriLabel is string or empty string
        }

# =====================
# Authentication Helpers
# =====================

def get_tgt(apikey: str) -> str:
    """Requests a Ticket-Granting Ticket (TGT) using the API key.

    Args:
        apikey: Your UMLS API key.

    Returns:
        The location URL of the TGT resource.

    Raises:
        UMLSAuthError: If TGT acquisition fails due to API errors, network issues,
                       or non-201 status code.
    """
    headers = {"User-Agent": USER_AGENT}
    payload = {"apikey": apikey}
    tgt_location = ""

    try:
        logger.debug("Requesting UMLS TGT...")
        resp = requests.post(
            TGT_URL,
            data=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS
        )
        resp.raise_for_status() # Raises HTTPError for 4xx/5xx responses

        if resp.status_code == 201:
            tgt_location = resp.headers.get("location", "")
            if not tgt_location:
                 raise UMLSAuthError("TGT request successful (201) but 'location' header missing.")
            logger.info("Successfully obtained UMLS TGT.")
            return tgt_location
        else:
            # Should be caught by raise_for_status, but as a safeguard:
            raise UMLSAuthError(f"Unexpected status code for TGT request: {resp.status_code}")

    except requests.exceptions.Timeout:
        logger.error("Timeout occurred while requesting UMLS TGT.")
        raise UMLSAuthError("Timeout during TGT request.") from None
    except requests.exceptions.RequestException as e:
        logger.error("Network error during UMLS TGT request: %s", e, exc_info=True)
        raise UMLSAuthError(f"Network error during TGT request: {e}") from e
    except Exception as e: # Catch unexpected errors
        logger.exception("An unexpected error occurred during TGT request.")
        raise UMLSAuthError(f"An unexpected error occurred during TGT request: {e}") from e


def get_service_ticket(tgt_location: str, service: str = DEFAULT_SERVICE) -> str:
    """Exchanges a TGT for a single-use Service Ticket (ST).

    Args:
        tgt_location: The URL of the TGT obtained from get_tgt().
        service: The service URL for which the ST is requested.

    Returns:
        The Service Ticket string.

    Raises:
        UMLSAuthError: If ST acquisition fails due to API errors, network issues,
                       or non-200 status code.
    """
    headers = {"User-Agent": USER_AGENT}
    payload = {"service": service}
    service_ticket = ""

    try:
        logger.debug("Requesting UMLS Service Ticket (ST)...")
        resp = requests.post(
            tgt_location,
            data=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS
        )
        resp.raise_for_status() # Check for 4xx/5xx errors

        if resp.status_code == 200:
            service_ticket = resp.text
            if not service_ticket:
                 raise UMLSAuthError("ST request successful (200) but response body is empty.")
            logger.info("Successfully obtained UMLS Service Ticket.")
            return service_ticket
        else:
            # Should be caught by raise_for_status:
            raise UMLSAuthError(f"Unexpected status code for ST request: {resp.status_code}")

    except requests.exceptions.Timeout:
        logger.error("Timeout occurred while requesting UMLS ST.")
        raise UMLSAuthError("Timeout during ST request.") from None
    except requests.exceptions.RequestException as e:
        logger.error("Network error during UMLS ST request: %s", e, exc_info=True)
        raise UMLSAuthError(f"Network error during ST request: {e}") from e
    except Exception as e:
        logger.exception("An unexpected error occurred during ST request.")
        raise UMLSAuthError(f"An unexpected error occurred during ST request: {e}") from e


# ====================
# Core Search Function
# ====================

def search_umls(
    term: str,
    apikey: str,
    page_size: int = 5,
    search_type: str = "exact" # Changed default to 'exact' for potentially better precision
) -> List[UMLSConcept]:
    """Searches the UMLS Metathesaurus for a given term.

    Handles TGT and ST acquisition internally.

    Args:
        term: The search string.
        apikey: Your UMLS API key.
        page_size: Maximum number of results to return.
        search_type: Type of search (e.g., 'exact', 'words', 'approximate').
                     See UMLS docs for options. Default: 'exact'.

    Returns:
        A list of UMLSConcept objects representing the search results.

    Raises:
        UMLSAuthError: If authentication fails.
        UMLSSearchError: If the search request fails, returns unexpected data,
                         or encounters network issues.
        ValueError: If page_size is non-positive.
    """
    if not _REQUESTS_AVAILABLE:
        raise UMLSError("The 'requests' library is not installed. Cannot perform UMLS search.")
    if not apikey:
        raise UMLSAuthError("UMLS API key is missing or empty.")
    if page_size <= 0:
        raise ValueError("page_size must be a positive integer.")

    logger.info("Starting UMLS search for term: '%s' (page_size=%d, type=%s)", term, page_size, search_type)

    try:
        # --- Authentication Step ---
        # Note: TGT/ST are fetched for *every* search call in this simple design.
        # Consider caching TGTs for efficiency in high-frequency scenarios.
        tgt_location = get_tgt(apikey)
        service_ticket = get_service_ticket(tgt_location)

        # --- Search Step ---
        headers = {
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }
        params = {
            "string": term,
            "ticket": service_ticket,
            "pageSize": str(page_size),
            "searchType": search_type,
        }

        logger.debug("Executing UMLS search request...")
        time.sleep(RATE_LIMIT_SLEEP_SECONDS) # Basic rate limiting (before request)

        resp = requests.get(
            SEARCH_URL,
            params=params,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS
        )
        resp.raise_for_status() # Check for 4xx/5xx first

        if resp.status_code != 200:
            # Should be caught by raise_for_status
             raise UMLSSearchError(f"UMLS search returned unexpected status: {resp.status_code}")

        # --- Process Results ---
        try:
            data = resp.json()
        except requests.exceptions.JSONDecodeError:
            logger.error("Failed to decode JSON response from UMLS search.")
            raise UMLSSearchError("Invalid JSON response received from UMLS search.")

        if not isinstance(data, dict):
             raise UMLSSearchError(f"Expected JSON dictionary, received {type(data)}")

        result_data = data.get("result")
        if not isinstance(result_data, dict):
             logger.warning("UMLS response missing 'result' dictionary or it's not a dict.")
             return [] # Return empty list if 'result' is missing/wrong type

        results_list = result_data.get("results")
        if not isinstance(results_list, list):
             logger.warning("UMLS response 'result' missing 'results' list or it's not a list.")
             return [] # Return empty list if 'results' is missing/wrong type

        # Convert JSON items to UMLSConcept objects
        concepts = [UMLSConcept.from_json(item) for item in results_list]
        logger.info("UMLS search completed successfully, found %d concepts.", len(concepts))
        return concepts

    except requests.exceptions.Timeout:
        logger.error("Timeout occurred during UMLS search request for term: %s", term)
        raise UMLSSearchError(f"Timeout during UMLS search for '{term}'.") from None
    except requests.exceptions.RequestException as e:
        logger.error("Network error during UMLS search for term '%s': %s", term, e, exc_info=True)
        raise UMLSSearchError(f"Network error during UMLS search for '{term}': {e}") from e
    except UMLSAuthError: # Re-raise auth errors directly
        raise
    except Exception as e:
        logger.exception("An unexpected error occurred during UMLS search for term '%s'.", term)
        raise UMLSSearchError(f"An unexpected error occurred during UMLS search for '{term}': {e}") from e


# =============
# CLI Test Stub
# =============
if __name__ == "__main__":
    import sys
    import pprint

    # Basic logging setup for CLI testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # Check for API key from environment variable
    api_key = os.getenv("UMLS_APIKEY")
    if not api_key:
        print("ERROR: Set the UMLS_APIKEY environment variable to run this test.", file=sys.stderr)
        sys.exit(1)

    # Determine search term from command line arguments or use default
    search_term = "lung nodule" if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    print(f"--- Testing UMLS Search ---")
    print(f"Search Term: '{search_term}'")
    print(f"API Key Present: {'Yes' if api_key else 'No'}")
    print(f"UMLS Available Flag: {UMLS_AVAILABLE}")
    print(f"Config Message: {UMLS_CONFIG_MSG}")
    print("-" * 25)

    if not UMLS_AVAILABLE:
         print("UMLS utilities are not available based on checks. Exiting test.")
         sys.exit(1)

    try:
        # Call the search function
        concepts_found = search_umls(search_term, api_key, page_size=5)

        # Pretty print the results
        if concepts_found:
            print(f"Found {len(concepts_found)} concepts:")
            pprint.pprint([c.to_dict() for c in concepts_found], indent=2)
        else:
            print("No concepts found for the given term.")

    except (UMLSAuthError, UMLSSearchError, ValueError) as e:
        print(f"\nERROR during UMLS search: {e}", file=sys.stderr)
        # For debugging, you might want the full traceback:
        # import traceback
        # traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\nUNEXPECTED ERROR during UMLS search: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)