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

        client = ArxivClient()
        results = client.search("attention mechanism transformers")
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
            index_dir: Path to LanceDB index. Auto-downloads from HF if None.
            device: Torch device for SPECTER2 (``"cpu"`` or ``"cuda"``).
            rerank: Enable graph-based PageRank re-ranking.
            embedding_batch_size: Batch size for SPECTER2 inference.
            cache_dir: Where to cache the HF-downloaded index.
            eager_load: Load SPECTER2 immediately (False if only using get_paper).
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
        """Total number of papers in the index."""
        return self._store.num_papers

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

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
        details: str = "default",
    ) -> SearchResult:
        """Search papers by keywords.

        Args:
            query: Search keywords (e.g. ``"vision transformers"``).
            max_results: Max papers to return.
            categories: ArXiv categories to filter (e.g. ``["cs.CV"]``).
            conference: Conference name (e.g. ``"CVPR"``), maps to categories.
            year: Filter to this publication year.
            date_from: Published after this date (YYYY-MM-DD).
            date_to: Published before this date (YYYY-MM-DD).
            context_paper_id: ArXiv ID to bias results toward.
            context_title: Title of context paper (alternative to ID).
            context_abstract: Abstract of context paper.
            rerank: Override default PageRank re-ranking.
            sort_by: ``"relevance"`` | ``"citations"`` | ``"date"`` | ``"importance"``.
            min_citations: Minimum citation count filter (uses S2 API).
            details: ``"default"`` returns arxiv_id, title, abstract, year, citation_count.
                ``"extra"`` returns all fields.
        """
        return self._engine.search(
            query=query, max_results=max_results, categories=categories,
            conference=conference, year=year, date_from=date_from, date_to=date_to,
            context_paper_id=context_paper_id, context_title=context_title,
            context_abstract=context_abstract, rerank=rerank,
            sort_by=sort_by, min_citations=min_citations, details=details,
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
        sort_by: str | None = None,
        min_citations: int | None = None,
        details: str = "default",
    ) -> SearchResult:
        """Run multiple queries, merge and deduplicate results.

        Args:
            queries: List of keyword queries.
            max_results: Results per query (all unique papers returned after merge).
            sort_by: ``"relevance"`` | ``"citations"`` | ``"date"`` | ``"importance"``.
            min_citations: Minimum citation count filter (uses S2 API).
            details: ``"default"`` returns arxiv_id, title, abstract, year, citation_count.
                ``"extra"`` returns all fields.

        Other arguments are the same as :meth:`search`.
        """
        return self._engine.batch_search(
            queries=queries, max_results=max_results, categories=categories,
            conference=conference, year=year, date_from=date_from, date_to=date_to,
            context_paper_id=context_paper_id, context_title=context_title,
            context_abstract=context_abstract, sort_by=sort_by,
            min_citations=min_citations, details=details,
        )

    def find_related(
        self,
        arxiv_id: str,
        max_results: int = 20,
        categories: list[str] | None = None,
        rerank: bool | None = None,
        details: str = "default",
    ) -> SearchResult:
        """Find papers related to a given paper via SPECTER2 embedding similarity.

        Args:
            arxiv_id: ArXiv ID of the source paper.
            max_results: Number of results.
            categories: Filter to these categories.
            rerank: Override default re-ranking.
            details: ``"default"`` returns arxiv_id, title, abstract, year, citation_count.
                ``"extra"`` returns all fields.
        """
        return self._engine.find_related(
            arxiv_id=arxiv_id, max_results=max_results,
            categories=categories, rerank=rerank, details=details,
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_paper(self, arxiv_id: str) -> Paper:
        """Look up a single paper. Raises PaperNotFoundError if missing."""
        return self._store.get_paper(arxiv_id)

    def get_papers(self, arxiv_ids: list[str]) -> list[Paper]:
        """Bulk lookup by ArXiv IDs. Missing papers are silently skipped."""
        return self._store.get_papers(arxiv_ids)

    # ------------------------------------------------------------------
    # Enrichment (Semantic Scholar API)
    # ------------------------------------------------------------------

    def enrich(
        self,
        papers: Sequence[Paper] | SearchResult,
        fields: list[str] | None = None,
    ) -> list[Paper]:
        """Add S2 metadata (citations, venue, tldr, references) to papers.

        Args:
            papers: Papers or SearchResult to enrich.
            fields: S2 fields to request. Defaults to all available.
        """
        from arxiv_search_kit.enrichment import enrich_papers
        return enrich_papers(list(papers) if not isinstance(papers, list) else papers, fields=fields)

    def get_citations(self, arxiv_id: str, limit: int = 100) -> list[dict]:
        """Get papers that cite this paper (via S2 API)."""
        from arxiv_search_kit.enrichment import get_citations
        return get_citations(arxiv_id, limit=limit)

    def get_references(self, arxiv_id: str, limit: int = 100) -> list[dict]:
        """Get papers referenced by this paper (via S2 API)."""
        from arxiv_search_kit.enrichment import get_references
        return get_references(arxiv_id, limit=limit)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_pdf(
        self,
        paper: Paper | str,
        output_dir: str | Path = ".",
        filename: str | None = None,
    ) -> Path:
        """Download a paper's PDF from ArXiv.

        Args:
            paper: Paper object or ArXiv ID string.
            output_dir: Directory to save into.
            filename: Custom filename. Defaults to ``{arxiv_id}.pdf``.
        """
        from arxiv_search_kit.papers import download_pdf
        return download_pdf(paper, output_dir=output_dir, filename=filename)

    def download_source(
        self,
        paper: Paper | str,
        output_dir: str | Path = ".",
        filename: str | None = None,
    ) -> Path:
        """Download a paper's LaTeX source archive from ArXiv.

        Args:
            paper: Paper object or ArXiv ID string.
            output_dir: Directory to save into.
            filename: Custom filename. Defaults to ``{arxiv_id}.tar.gz``.
        """
        from arxiv_search_kit.papers import download_source
        return download_source(paper, output_dir=output_dir, filename=filename)

    def download_papers(
        self,
        papers: list[Paper] | list[str],
        output_dir: str | Path = ".",
        format: str = "pdf",
    ) -> list[Path]:
        """Batch download papers. Skips failures with a warning.

        Args:
            papers: List of Paper objects or ArXiv ID strings.
            output_dir: Directory to save into.
            format: ``"pdf"`` or ``"source"``.
        """
        from arxiv_search_kit.papers import download_papers
        return download_papers(papers, output_dir=output_dir, format=format)

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def summarize_paper(
        self,
        paper: Paper | str | list[Paper] | list[str],
        api_key: str | None = None,
        model: str = "gemini-3-flash-preview",
        max_concurrent: int = 5,
    ) -> str | dict[str, str]:
        """Download paper(s) LaTeX source and summarize using Google Gemini.

        Downloads the source from ArXiv, extracts the primary .tex file,
        trims content after the conclusion, and sends it to Gemini for
        a comprehensive summary covering contributions, methodology,
        results, and more.

        Args:
            paper: Paper object, ArXiv ID string, or a list of either.
            api_key: Google AI API key. Falls back to ``GEMINI_API_KEY`` env var.
            model: Gemini model to use.
            max_concurrent: Max parallel requests (only for multiple papers).

        Returns:
            A summary string for a single paper, or a dict mapping
            ArXiv ID to summary string for multiple papers.
        """
        from arxiv_search_kit.summarizer import summarize_paper
        return summarize_paper(paper, api_key=api_key, model=model, max_concurrent=max_concurrent)

    # ------------------------------------------------------------------
    # Async variants
    # ------------------------------------------------------------------

    async def async_search(self, **kwargs) -> SearchResult:
        """Async variant of :meth:`search`."""
        return await asyncio.get_running_loop().run_in_executor(None, lambda: self.search(**kwargs))

    async def async_batch_search(self, **kwargs) -> SearchResult:
        """Async variant of :meth:`batch_search`."""
        return await asyncio.get_running_loop().run_in_executor(None, lambda: self.batch_search(**kwargs))

    async def async_find_related(self, **kwargs) -> SearchResult:
        """Async variant of :meth:`find_related`."""
        return await asyncio.get_running_loop().run_in_executor(None, lambda: self.find_related(**kwargs))

    async def async_enrich(
        self,
        papers: Sequence[Paper] | SearchResult,
        fields: list[str] | None = None,
    ) -> list[Paper]:
        """Async variant of :meth:`enrich`."""
        from arxiv_search_kit.enrichment import async_enrich_papers
        return await async_enrich_papers(
            list(papers) if not isinstance(papers, list) else papers, fields=fields,
        )

    async def async_summarize_paper(
        self,
        paper: Paper | str | list[Paper] | list[str],
        api_key: str | None = None,
        model: str = "gemini-3-flash-preview",
        max_concurrent: int = 5,
    ) -> str | dict[str, str]:
        """Async variant of :meth:`summarize_paper`."""
        from arxiv_search_kit.summarizer import async_summarize_paper
        return await async_summarize_paper(
            paper, api_key=api_key, model=model, max_concurrent=max_concurrent,
        )
