"""Semantic Scholar enrichment — citation counts, references, TLDR.

Requires S2_API_KEY env var for reasonable rate limits (100 req/sec vs 100 req/5min).
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

import httpx

from arxiv_search_kit.exceptions import EnrichmentError
from arxiv_search_kit.models import Paper

logger = logging.getLogger(__name__)

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_BATCH_LIMIT = 500  # Max IDs per batch request

DEFAULT_FIELDS = [
    "citationCount",
    "influentialCitationCount",
    "references.externalIds",
    "tldr",
]


def _get_headers() -> dict[str, str]:
    """Get S2 API headers with optional API key."""
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("S2_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _arxiv_to_s2_id(arxiv_id: str) -> str:
    """Convert ArXiv ID to S2 external ID format."""
    return f"ArXiv:{arxiv_id}"


def enrich_papers(
    papers: list[Paper],
    fields: list[str] | None = None,
    timeout: float = 30.0,
) -> list[Paper]:
    """Enrich papers with Semantic Scholar metadata (sync).

    Args:
        papers: Papers to enrich.
        fields: S2 fields to request. Defaults to DEFAULT_FIELDS.
        timeout: HTTP request timeout.

    Returns:
        Same papers with enrichment fields populated.
    """
    if not papers:
        return papers

    fields = fields or DEFAULT_FIELDS
    fields_str = ",".join(fields)
    headers = _get_headers()

    with httpx.Client(timeout=timeout) as http:
        for batch_start in range(0, len(papers), S2_BATCH_LIMIT):
            batch = papers[batch_start : batch_start + S2_BATCH_LIMIT]
            s2_ids = [_arxiv_to_s2_id(p.arxiv_id) for p in batch]

            try:
                response = http.post(
                    f"{S2_API_BASE}/paper/batch",
                    headers=headers,
                    params={"fields": fields_str},
                    json={"ids": s2_ids},
                )
                response.raise_for_status()
                results = response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning("S2 rate limited — set S2_API_KEY for higher limits")
                else:
                    logger.warning("S2 batch failed: %s", e)
                continue
            except Exception as e:
                logger.warning("S2 enrichment failed: %s", e)
                continue

            for paper, s2_data in zip(batch, results):
                if s2_data is not None:
                    _apply_enrichment(paper, s2_data)

    return papers


async def async_enrich_papers(
    papers: list[Paper],
    fields: list[str] | None = None,
    timeout: float = 30.0,
) -> list[Paper]:
    """Enrich papers with Semantic Scholar metadata (async).

    Args:
        papers: Papers to enrich.
        fields: S2 fields to request.
        timeout: HTTP request timeout.

    Returns:
        Same papers with enrichment fields populated.
    """
    if not papers:
        return papers

    fields = fields or DEFAULT_FIELDS
    fields_str = ",".join(fields)
    headers = _get_headers()

    async with httpx.AsyncClient(timeout=timeout) as client:
        for batch_start in range(0, len(papers), S2_BATCH_LIMIT):
            batch = papers[batch_start : batch_start + S2_BATCH_LIMIT]
            s2_ids = [_arxiv_to_s2_id(p.arxiv_id) for p in batch]

            try:
                response = await client.post(
                    f"{S2_API_BASE}/paper/batch",
                    headers=headers,
                    params={"fields": fields_str},
                    json={"ids": s2_ids},
                )
                response.raise_for_status()
                results = response.json()

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning("S2 rate limit hit. Set S2_API_KEY for higher limits.")
                else:
                    logger.warning(f"S2 batch request failed: {e}")
                continue
            except Exception as e:
                logger.warning(f"S2 enrichment failed: {e}")
                continue

            for paper, s2_data in zip(batch, results):
                if s2_data is None:
                    continue
                _apply_enrichment(paper, s2_data)

    return papers


def get_citations(
    arxiv_id: str,
    limit: int = 100,
    timeout: float = 30.0,
) -> list[dict]:
    """Get papers that cite the given paper (via S2 API).

    Args:
        arxiv_id: ArXiv ID of the paper.
        limit: Max citations to return (S2 supports up to 1000).
        timeout: HTTP timeout.

    Returns:
        List of dicts with keys: arxiv_id, title, year, citation_count.
        Papers without ArXiv IDs are included with arxiv_id=None.
    """
    return _get_citation_graph(arxiv_id, direction="citations", limit=limit, timeout=timeout)


def get_references(
    arxiv_id: str,
    limit: int = 100,
    timeout: float = 30.0,
) -> list[dict]:
    """Get papers referenced by the given paper (via S2 API).

    Args:
        arxiv_id: ArXiv ID of the paper.
        limit: Max references to return.
        timeout: HTTP timeout.

    Returns:
        List of dicts with keys: arxiv_id, title, year, citation_count.
    """
    return _get_citation_graph(arxiv_id, direction="references", limit=limit, timeout=timeout)


def _get_citation_graph(
    arxiv_id: str,
    direction: str,
    limit: int = 100,
    timeout: float = 30.0,
) -> list[dict]:
    """Fetch citations or references from S2 API."""
    s2_id = _arxiv_to_s2_id(arxiv_id)
    fields = "title,year,citationCount,externalIds"
    url = f"{S2_API_BASE}/paper/{s2_id}/{direction}?fields={fields}&limit={limit}"
    headers = _get_headers()

    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logger.warning("S2 rate limited — set S2_API_KEY for higher limits")
        else:
            logger.warning("S2 %s failed: %s", direction, e)
        return []
    except Exception as e:
        logger.warning("S2 %s failed: %s", direction, e)
        return []

    results = []
    key = "citingPaper" if direction == "citations" else "citedPaper"
    for entry in data.get("data", []):
        paper_data = entry.get(key, {})
        if not paper_data:
            continue
        ext_ids = paper_data.get("externalIds", {}) or {}
        results.append({
            "arxiv_id": ext_ids.get("ArXiv"),
            "title": paper_data.get("title", ""),
            "year": paper_data.get("year"),
            "citation_count": paper_data.get("citationCount", 0),
        })

    return results


async def async_get_citations(
    arxiv_id: str,
    limit: int = 100,
    timeout: float = 30.0,
) -> list[dict]:
    """Async version of get_citations."""
    return await _async_get_citation_graph(arxiv_id, "citations", limit, timeout)


async def async_get_references(
    arxiv_id: str,
    limit: int = 100,
    timeout: float = 30.0,
) -> list[dict]:
    """Async version of get_references."""
    return await _async_get_citation_graph(arxiv_id, "references", limit, timeout)


async def _async_get_citation_graph(
    arxiv_id: str,
    direction: str,
    limit: int = 100,
    timeout: float = 30.0,
) -> list[dict]:
    """Async fetch citations or references from S2 API."""
    s2_id = _arxiv_to_s2_id(arxiv_id)
    fields = "title,year,citationCount,externalIds"
    url = f"{S2_API_BASE}/paper/{s2_id}/{direction}?fields={fields}&limit={limit}"
    headers = _get_headers()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        logger.warning(f"S2 async {direction} failed: {e}")
        return []

    results = []
    key = "citingPaper" if direction == "citations" else "citedPaper"
    for entry in data.get("data", []):
        paper_data = entry.get(key, {})
        if not paper_data:
            continue
        ext_ids = paper_data.get("externalIds", {}) or {}
        results.append({
            "arxiv_id": ext_ids.get("ArXiv"),
            "title": paper_data.get("title", ""),
            "year": paper_data.get("year"),
            "citation_count": paper_data.get("citationCount", 0),
        })

    return results


def _apply_enrichment(paper: Paper, s2_data: dict) -> None:
    """Apply S2 enrichment data to a Paper object."""
    if "citationCount" in s2_data:
        paper.citation_count = s2_data["citationCount"]

    if "influentialCitationCount" in s2_data:
        paper.influential_citation_count = s2_data["influentialCitationCount"]

    if "tldr" in s2_data and s2_data["tldr"]:
        paper.tldr = s2_data["tldr"].get("text")

    if "references" in s2_data and s2_data["references"]:
        ref_ids = []
        for ref in s2_data["references"]:
            ext_ids = ref.get("externalIds", {})
            if ext_ids and "ArXiv" in ext_ids:
                ref_ids.append(ext_ids["ArXiv"])
        paper.references = ref_ids if ref_ids else None