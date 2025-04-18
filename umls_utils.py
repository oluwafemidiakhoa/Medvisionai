
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
umls_utils.py – Tiny helper around the UMLS REST API
===================================================

Public surface
--------------
search_umls(term: str, apikey: str, *, page_size: int = 5) -> list[UMLSConcept]
get_cached_tgt(apikey: str) -> str
flush_tgt_cache() -> None

A minimal, dependency‑free cache keeps the TGT for 8 hours inside the module
process; Hugging Face Spaces restart often enough that this is sufficient.
"""

# Default number of UMLS hits to return
DEFAULT_UMLS_HITS = 5

# ─────────────────────────────────────────────────────────────────────────────
# Std‑lib
# ─────────────────────────────────────────────────────────────────────────────
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Third‑party
# ─────────────────────────────────────────────────────────────────────────────
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TGT_URL              = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
STS_URL              = "https://uts-ws.nlm.nih.gov/rest/search/current"
DEFAULT_SERVICE      = "http://umlsks.nlm.nih.gov"
USER_AGENT           = "RadVisionAI/1.0 (+https://hf.co/spaces/mgbam/radvisionai)"
TIMEOUT              = 15          # s
SLEEP_BETWEEN_CALLS  = 0.06        # ~16 req/s overall
TGT_CACHE_SECONDS    = 8 * 60 * 60 # 8 h

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
_SESSION = requests.Session() if REQUESTS_AVAILABLE else None
if _SESSION:
    _SESSION.headers.update({"User-Agent": USER_AGENT})

def flush_tgt_cache() -> None:
    """Erase the in‑memory TGT cache (mainly for unit tests)."""
    _tgt_cache.clear()
    logger.debug("TGT cache flushed.")

def get_cached_tgt(apikey: str) -> str:
    """Return a cached TGT or fetch a new one if expired/absent."""
    if not REQUESTS_AVAILABLE:
        raise UMLSConnectionError("requests library not available")
    
    tgt, exp = _tgt_cache.get(apikey, ("", 0.0))
    if time.time() < exp:
        logger.debug("Re‑using cached TGT (expires in %.0f s).", exp - time.time())
        return tgt

    logger.debug("Requesting new TGT…")
    try:
        resp = _SESSION.post(
            TGT_URL,
            data={"apikey": apikey},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        tgt = resp.headers.get("location") or resp.text
        if not tgt:
            raise UMLSAuthError("Missing TGT in response!")
        
        _tgt_cache[apikey] = (tgt, time.time() + TGT_CACHE_SECONDS)
        logger.debug("TGT cached.")
        return tgt
    except requests.RequestException as e:
        raise UMLSConnectionError(f"TGT request failed: {e}") from e

def _get_service_ticket(tgt_url: str, service: str = DEFAULT_SERVICE) -> str:
    """Get a service ticket from a TGT."""
    if not REQUESTS_AVAILABLE:
        raise UMLSConnectionError("requests library not available")
    
    try:
        resp = _SESSION.post(
            tgt_url,
            data={"service": service},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        raise UMLSConnectionError(f"Service ticket request failed: {e}") from e

def search_umls(term: str, apikey: str, *, page_size: int = 5, 
                source_vocabs: Optional[List[str]] = None,
                semantic_types: Optional[List[str]] = None) -> List[UMLSConcept]:
    """
    Search UMLS for concepts matching term with enhanced filtering options.
    
    Args:
        term: The search term
        apikey: UMLS API key
        page_size: Max number of results to return
        source_vocabs: Optional list of source vocabularies to filter results (e.g., ["SNOMEDCT_US", "ICD10CM"])
        semantic_types: Optional list of semantic types to filter results
        
    Returns:
        List of UMLSConcept objects from the search result
    """
    if not REQUESTS_AVAILABLE:
        raise UMLSConnectionError("requests library not available")
    
    if not term:
        return []
    
    # Limit search term length to avoid long URLs that may cause 403 errors
    term_truncated = term[:500] if len(term) > 500 else term
    
    try:
        tgt_url = get_cached_tgt(apikey)
        service_ticket = _get_service_ticket(tgt_url)
        
        params = {
            "string": term_truncated,
            "ticket": service_ticket,
            "pageSize": page_size,
        }
        
        # Add source vocabulary filter if specified
        if source_vocabs:
            params["sabs"] = ",".join(source_vocabs)
        
        # Add semantic type filter if specified
        if semantic_types:
            params["stypes"] = ",".join(semantic_types)
        
        # Handle potential API errors gracefully
        try:
            resp = _SESSION.get(STS_URL, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            
            results = data.get("result", {}).get("results", [])
            concepts = [UMLSConcept.from_json(item) for item in results]
            
            # Add small delay to respect rate limits
            time.sleep(SLEEP_BETWEEN_CALLS)
            
            return concepts
        except requests.exceptions.HTTPError as http_err:
            # If we get a 403 Forbidden, the string might be too long or there's an auth issue
            if http_err.response.status_code == 403:
                logger.warning(f"UMLS API returned 403 Forbidden. String may be too long or auth issue.")
                # Try with keywords instead of full text (fallback behavior)
                if len(term_truncated) > 50:
                    # Extract keywords instead of using full text
                    import re
                    keywords = re.findall(r'\b[A-Za-z]{4,}\b', term_truncated)
                    if keywords:
                        logger.info(f"Retrying UMLS search with keywords: {' '.join(keywords[:5])}")
                        return search_umls(" ".join(keywords[:5]), apikey, page_size=page_size,
                                          source_vocabs=source_vocabs, semantic_types=semantic_types)
            raise UMLSConnectionError(f"UMLS search failed: {http_err}") from http_err
        except requests.RequestException as e:
            raise UMLSConnectionError(f"UMLS search failed: {e}") from e
        except (ValueError, KeyError) as e:
            raise UMLSSearchError(f"Invalid response: {e}") from e
    except UMLSAuthError as auth_err:
        logger.error(f"UMLS authentication failed: {auth_err}")
        # Return empty results instead of crashing
        return []
    except Exception as e:
        logger.error(f"Unexpected error in UMLS search: {e}")
        # Return empty results for non-critical UMLS operations
        return []

# New function to get concept definitions
def get_concept_definitions(cui: str, apikey: str) -> List[Dict[str, str]]:
    """
    Get definitions for a specific concept by CUI.
    
    Args:
        cui: Concept Unique Identifier
        apikey: UMLS API key
        
    Returns:
        List of dictionaries containing definitions from various sources
    """
    if not REQUESTS_AVAILABLE:
        raise UMLSConnectionError("requests library not available")
    
    tgt_url = get_cached_tgt(apikey)
    service_ticket = _get_service_ticket(tgt_url)
    
    CONCEPT_URL = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}/definitions"
    
    params = {
        "ticket": service_ticket,
    }
    
    try:
        resp = _SESSION.get(CONCEPT_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        
        definitions = []
        for result in data.get("result", []):
            definition = {
                "source": result.get("rootSource", ""),
                "value": result.get("value", ""),
                "sourceOriginated": result.get("sourceOriginated", False)
            }
            definitions.append(definition)
        
        # Add small delay to respect rate limits
        time.sleep(SLEEP_BETWEEN_CALLS)
        
        return definitions
    except requests.RequestException as e:
        raise UMLSConnectionError(f"UMLS definition lookup failed: {e}") from e
    except (ValueError, KeyError) as e:
        raise UMLSSearchError(f"Invalid response: {e}") from e

# New function to get concept relationships
def get_concept_relationships(cui: str, apikey: str, relationship_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Get relationships for a specific concept by CUI.
    
    Args:
        cui: Concept Unique Identifier
        apikey: UMLS API key
        relationship_types: Optional list of relationship types to filter by
        
    Returns:
        List of dictionaries containing relationship data
    """
    if not REQUESTS_AVAILABLE:
        raise UMLSConnectionError("requests library not available")
    
    tgt_url = get_cached_tgt(apikey)
    service_ticket = _get_service_ticket(tgt_url)
    
    RELATIONS_URL = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}/relations"
    
    params = {
        "ticket": service_ticket,
    }
    
    if relationship_types:
        params["relationTypes"] = ",".join(relationship_types)
    
    try:
        resp = _SESSION.get(RELATIONS_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        
        relationships = []
        for result in data.get("result", []):
            relationship = {
                "relation_type": result.get("relationLabel", ""),
                "related_id": result.get("relatedId", ""),
                "ui": result.get("ui", ""),
                "source": result.get("rootSource", "")
            }
            relationships.append(relationship)
        
        # Add small delay to respect rate limits
        time.sleep(SLEEP_BETWEEN_CALLS)
        
        return relationships
    except requests.RequestException as e:
        raise UMLSConnectionError(f"UMLS relationship lookup failed: {e}") from e
    except (ValueError, KeyError) as e:
        raise UMLSSearchError(f"Invalid response: {e}") from e

# Function to get semantic types for a concept
def get_semantic_types(cui: str, apikey: str) -> List[Dict[str, str]]:
    """
    Get semantic types for a specific concept by CUI.
    
    Args:
        cui: Concept Unique Identifier
        apikey: UMLS API key
        
    Returns:
        List of dictionaries containing semantic type data
    """
    if not REQUESTS_AVAILABLE:
        raise UMLSConnectionError("requests library not available")
    
    tgt_url = get_cached_tgt(apikey)
    service_ticket = _get_service_ticket(tgt_url)
    
    SEMTYPES_URL = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}/semantictypes"
    
    params = {
        "ticket": service_ticket,
    }
    
    try:
        resp = _SESSION.get(SEMTYPES_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        
        semantic_types = []
        for result in data.get("result", []):
            semantic_type = {
                "name": result.get("name", ""),
                "ui": result.get("ui", ""),
                "uri": result.get("uri", "")
            }
            semantic_types.append(semantic_type)
        
        # Add small delay to respect rate limits
        time.sleep(SLEEP_BETWEEN_CALLS)
        
        return semantic_types
    except requests.RequestException as e:
        raise UMLSConnectionError(f"UMLS semantic type lookup failed: {e}") from e
    except (ValueError, KeyError) as e:
        raise UMLSSearchError(f"Invalid response: {e}") from e

# ─────────────────────────────────────────────────────────────────────────────
# Module load check
# ─────────────────────────────────────────────────────────────────────────────
UMLS_UTILS_LOADED = REQUESTS_AVAILABLE

if not UMLS_UTILS_LOADED:
    logger.warning("UMLS utils failed to load: 'requests' package is missing.")
else:
    logger.info("UMLS utils loaded successfully.")
