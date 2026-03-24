"""ArxivClient — main entry point for the arxiv_search_kit SDK."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Sequence

from arxiv_search_kit.index.embedder import Specter2Embedder
from arxiv_search_kit.index.store import IndexStore
from arxiv_search_kit.models import Paper, SearchResult
from arxiv_search_kit.search.engine import SearchEngine

logger = logging.getLogger(__name__)


class ArxivClient:
    """ArXiv paper search over a local SPECTER2 + LanceDB index.

    Usage::

        client = ArxivClient()  # auto-downloads index from HuggingFace
        papers = client.search("attention mechanism transformers")
    """

    def __init__(
        self,
        index_dir: str | Path | None = None,
        device: str = "cpu",
        rerank: bool = True,
        embedding_batch_size: int = 32,
        cache_dir: str | Path | None = None,
        eager_load: bool = True,
    ):
        """
        Args:
            index_dir: Path to LanceDB index. If None, downloads from HuggingFace.
            device: Torch device for query embedding ("cpu" or "cuda").
            rerank: Apply graph-based re-ranking by default.
            embedding_batch_size: Batch size for SPECTER2.
            cache_dir: Where to cache the HF-downloaded index.
            eager_load: Load SPECTER2 immediately. Set False if only using get_paper().
        """
        if index_dir is None:
            from arxiv_search_kit.hub import download_index
            index_dir = download_index(cache_dir=cache_dir)

        self._store = IndexStore(index_dir)
        self._embedder = Specter2Embedder(device=device, batch_size=embedding_batch_size)
        self._engine = SearchEngine(store=self._store, embedder=self._embedder, rerank=rerank)

        if eager_load:
            self._embedder.warmup()

        logger.info("ArxivClient ready: %s papers, device=%s", f"{self._store.num_papers:,}", device)

    @property
    def num_papers(self) -> int:
        return self._store.num_papers

    def search(
        self,
        query: str,
        max_results: int = 20,
        categories: list[str] | None = None,
        conference: str | None = None,
        year: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        context_paper_id: str | None = None,
        context_title: str | None = None,
        context_abstract: str | None = None,
        rerank: bool | None = None,
        sort_by: str | None = None,
        min_citations: int | None = None,
    ) -> SearchResult:
        """Search for papers by keywords.

        Args:
            query: Search keywords (e.g. "vision transformers object detection").
            max_results: Max papers to return.
            categories: ArXiv categories to filter (e.g. ["cs.CV", "cs.LG"]).
            conference: Conference name to filter (e.g. "CVPR"), maps to categories.
            year: Filter to this publication year.
            date_from: Published after this date (YYYY-MM-DD).
            date_to: Published before this date (YYYY-MM-DD).
            context_paper_id: ArXiv ID of paper being reviewed — biases results
                toward this paper's neighborhood.
            context_title: Title of context paper (alternative to ID).
            context_abstract: Abstract of context paper.
            rerank: Override default graph re-ranking.
            sort_by: "relevance" (default), "citations" (S2 API), or "date".
            min_citations: Minimum citation count filter (uses S2 API).
        """
        return self._engine.search(
            query=query, max_results=max_results, categories=categories,
            conference=conference, year=year, date_from=date_from, date_to=date_to,
            context_paper_id=context_paper_id, context_title=context_title,
            context_abstract=context_abstract, rerank=rerank,
            sort_by=sort_by, min_citations=min_citations,
        )

    def batch_search(
        self,
        queries: list[str],
        max_results: int = 20,
        categories: list[str] | None = None,
        conference: str | None = None,
        year: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        context_paper_id: str | None = None,
        context_title: str | None = None,
        context_abstract: str | None = None,
    ) -> SearchResult:
        """Run multiple queries, merge and deduplicate results.

        Args:
            queries: List of keyword queries.
            max_results: Results per query (all unique papers returned after merge).
        """
        return self._engine.batch_search(
            queries=queries, max_results=max_results, categories=categories,
            conference=conference, year=year, date_from=date_from, date_to=date_to,
            context_paper_id=context_paper_id, context_title=context_title,
            context_abstract=context_abstract,
        )

    def get_paper(self, arxiv_id: str) -> Paper:
        """Look up a paper by ArXiv ID. Raises PaperNotFoundError if missing."""
        return self._store.get_paper(arxiv_id)

    def get_papers(self, arxiv_ids: list[str]) -> list[Paper]:
        """Get multiple papers by ID. Missing papers are silently skipped."""
        return self._store.get_papers(arxiv_ids)

    def find_related(
        self,
        arxiv_id: str,
        max_results: int = 20,
        categories: list[str] | None = None,
        rerank: bool | None = None,
    ) -> SearchResult:
        """Find papers related to a given paper via SPECTER2 similarity."""
        return self._engine.find_related(
            arxiv_id=arxiv_id, max_results=max_results,
            categories=categories, rerank=rerank,
        )

    def get_citations(self, arxiv_id: str, limit: int = 100) -> list[dict]:
        """Papers that cite this paper (via Semantic Scholar API)."""
        from arxiv_search_kit.enrichment import get_citations
        return get_citations(arxiv_id, limit=limit)

    def get_references(self, arxiv_id: str, limit: int = 100) -> list[dict]:
        """Papers referenced by this paper (via Semantic Scholar API)."""
        from arxiv_search_kit.enrichment import get_references
        return get_references(arxiv_id, limit=limit)

    def enrich(
        self,
        papers: Sequence[Paper] | SearchResult,
        fields: list[str] | None = None,
    ) -> list[Paper]:
        """Add S2 metadata (citation_count, references, tldr) to papers.

        Set S2_API_KEY env var for higher rate limits.
        """
        from arxiv_search_kit.enrichment import enrich_papers
        paper_list = list(papers) if not isinstance(papers, list) else papers
        return enrich_papers(paper_list, fields=fields)

    # Unused — kept for potential future use. Prefer S2 /paper/search/match.
    def search_title(self, title: str, threshold: float = 0.90) -> Paper | None:
        return self._engine.search_title(title, threshold=threshold)

    async def async_search(self, **kwargs) -> SearchResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.search(**kwargs))

    async def async_batch_search(self, **kwargs) -> SearchResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.batch_search(**kwargs))

    async def async_find_related(self, **kwargs) -> SearchResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.find_related(**kwargs))

    async def async_enrich(self, papers: Sequence[Paper] | SearchResult, fields: list[str] | None = None) -> list[Paper]:
        from arxiv_search_kit.enrichment import async_enrich_papers
        paper_list = list(papers) if not isinstance(papers, list) else papers
        return await async_enrich_papers(paper_list, fields=fields)