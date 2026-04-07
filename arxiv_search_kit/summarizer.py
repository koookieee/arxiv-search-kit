"""Summarize ArXiv papers using Google Gemini from their LaTeX source."""

from __future__ import annotations

import gzip
import io
import logging
import re
import tarfile
from pathlib import Path

import httpx

from arxiv_search_kit.exceptions import DownloadError, SummarizationError
from arxiv_search_kit.models import Paper

logger = logging.getLogger(__name__)

ARXIV_SOURCE_URL = "https://arxiv.org/e-print/{arxiv_id}"

SYSTEM_PROMPT = """\
You are an expert academic researcher who writes concise, dense paper summaries. \
You will be given the LaTeX source of a research paper. Your job is to distill it into \
a tight summary of at most 2000 tokens — every sentence must earn its place. \
Be specific and precise, not verbose."""

USER_PROMPT = """\
Write a concise summary of the following paper in at most 2000 tokens. \
Cover each section in 1–3 sentences max. Be dense and specific — include key numbers, \
method names, and results but cut all padding and repetition.

1. **Title & Authors** — title, authors, affiliations (one line).
2. **Problem** — what problem is solved and why it matters (2–3 sentences).
3. **Contributions** — bullet list of novel claims, each in one precise sentence.
4. **Method** — core approach, key equations or architecture components (3–5 sentences).
5. **Experiments** — datasets, baselines, and the most important quantitative results with numbers.
6. **Ablations** — key findings from ablation studies in 1–2 sentences.
7. **Limitations & Conclusion** — main limitations and takeaways (2–3 sentences).

Stop when you have covered all sections. Do not pad or repeat.

Here is the LaTeX source:

{tex_content}
"""


def _fetch_source(arxiv_id: str, timeout: float = 60.0) -> bytes:
    """Download the e-print source archive as raw bytes."""
    url = ARXIV_SOURCE_URL.format(arxiv_id=arxiv_id)
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        return resp.content
    except httpx.HTTPStatusError as e:
        raise DownloadError(f"ArXiv returned {e.response.status_code} for {url}") from e
    except Exception as e:
        raise DownloadError(f"Source download failed for {url}: {e}") from e


def _find_primary_tex(archive_bytes: bytes) -> str:
    """Extract the primary .tex file content from an e-print archive.

    Tries tar.gz first, then plain gzip (single-file submissions).
    The primary file is identified by looking for ``\\documentclass``.
    """
    # Try as tar.gz
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
            tex_files: list[tuple[str, str]] = []
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".tex"):
                    f = tar.extractfile(member)
                    if f is not None:
                        content = f.read().decode("utf-8", errors="replace")
                        tex_files.append((member.name, content))

            if not tex_files:
                raise SummarizationError("No .tex files found in source archive")

            # Find the primary file (the one with \documentclass)
            for name, content in tex_files:
                if r"\documentclass" in content:
                    logger.info("Primary tex file: %s", name)
                    return content

            # Fallback: largest .tex file
            tex_files.sort(key=lambda x: len(x[1]), reverse=True)
            logger.info("No \\documentclass found, using largest: %s", tex_files[0][0])
            return tex_files[0][1]
    except tarfile.TarError:
        pass

    # Try as plain gzip (single-file submissions)
    try:
        content = gzip.decompress(archive_bytes).decode("utf-8", errors="replace")
        if r"\documentclass" in content or r"\begin{document}" in content:
            logger.info("Single-file gzip submission")
            return content
    except Exception:
        pass

    raise SummarizationError("Could not extract .tex content from source archive")


def _trim_after_conclusion(tex: str) -> str:
    """Trim the tex content to keep everything up to and including the conclusion section.

    Removes appendices, acknowledgements after conclusion, bibliography, etc.
    """
    # Common patterns for conclusion sections (case insensitive)
    conclusion_patterns = [
        r"\\section\*?\{[^}]*[Cc]onclusion[^}]*\}",
        r"\\section\*?\{[^}]*[Ss]ummary[^}]*\}",
        r"\\section\*?\{[^}]*[Cc]oncluding [Rr]emarks[^}]*\}",
    ]

    conclusion_start = -1
    for pattern in conclusion_patterns:
        match = re.search(pattern, tex)
        if match:
            conclusion_start = match.start()
            break

    if conclusion_start == -1:
        # No conclusion found — return full content (minus bibliography)
        bib_match = re.search(r"\\bibliography\{|\\begin\{thebibliography\}", tex)
        if bib_match:
            return tex[:bib_match.start()].rstrip()
        return tex

    # Find the next \section or \appendix or \bibliography after conclusion
    after_conclusion = tex[conclusion_start:]
    # Match the next section-level command after the conclusion section header
    next_section = re.search(
        r"\n\\(?:section|appendix|bibliography|begin\{thebibliography\}|end\{document\})",
        after_conclusion[1:],  # skip the conclusion \section itself
    )

    if next_section:
        # Keep up to the end of the conclusion section
        end_pos = conclusion_start + 1 + next_section.start()
        return tex[:end_pos].rstrip()

    return tex


def _clean_tex(tex: str) -> str:
    """Light cleaning of LaTeX source to reduce token usage while preserving content."""
    # Remove comments (lines starting with %)
    tex = re.sub(r"(?m)^%.*$", "", tex)
    # Remove inline comments (but not escaped \%)
    tex = re.sub(r"(?<!\\)%.*$", "", tex, flags=re.MULTILINE)
    # Collapse multiple blank lines
    tex = re.sub(r"\n{3,}", "\n\n", tex)
    return tex.strip()


QA_SYSTEM_PROMPT = """\
You are an expert academic researcher. You will be given the LaTeX source of a research paper \
and a question about it. Answer the question accurately and in detail, grounding every claim \
in the paper's content. If the paper does not contain enough information to answer, say so clearly."""

QA_USER_PROMPT = """\
Answer the following question based solely on the content of the paper below. \
Be specific and thorough — cite sections, equations, tables, or numbers from the paper where relevant. \
Do not speculate beyond what the paper says.

Question: {question}

Paper LaTeX source:

{tex_content}
"""


def ask_paper(
    paper: Paper | str,
    question: str,
    api_key: str | None = None,
    model: str = "gemini-3-flash-preview",
    timeout: float = 120.0,
) -> str:
    """Answer a question about a paper using its LaTeX source and Google Gemini.

    Downloads the paper's LaTeX source from ArXiv, extracts the primary .tex
    file, trims after the conclusion, and asks Gemini to answer the question
    grounded in the paper's content.

    Args:
        paper: Paper object or ArXiv ID string.
        question: The question to answer about the paper.
        api_key: Google AI API key. Falls back to ``GEMINI_API_KEY`` env var.
        model: Gemini model to use.
        timeout: HTTP timeout for source download.

    Returns:
        Answer string grounded in the paper's content.

    Raises:
        SummarizationError: If the API call fails.
        DownloadError: If source download fails.
    """
    import os

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise SummarizationError(
            "google-genai is required for ask_paper. "
            "Install it with: pip install arxiv-search-kit[summarize]"
        )

    arxiv_id = paper.arxiv_id if isinstance(paper, Paper) else paper
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SummarizationError(
            "Gemini API key required. Pass api_key= or set GEMINI_API_KEY env var."
        )

    logger.info("Downloading source for %s ...", arxiv_id)
    archive_bytes = _fetch_source(arxiv_id, timeout=timeout)
    tex_content = _find_primary_tex(archive_bytes)
    tex_content = _trim_after_conclusion(tex_content)
    tex_content = _clean_tex(tex_content)

    logger.info("Asking question about %s (%d chars tex)", arxiv_id, len(tex_content))

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=QA_SYSTEM_PROMPT),
                        types.Part(text=QA_USER_PROMPT.format(
                            question=question,
                            tex_content=tex_content,
                        )),
                    ],
                ),
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="medium"),
            ),
        )
        answer = response.text
    except Exception as e:
        raise SummarizationError(f"Gemini API call failed: {e}") from e

    if not answer:
        raise SummarizationError("Gemini returned an empty response")

    logger.info("Answer generated (%d chars)", len(answer))
    return answer


async def async_ask_paper(
    paper: Paper | str,
    question: str,
    api_key: str | None = None,
    model: str = "gemini-3-flash-preview",
    timeout: float = 120.0,
) -> str:
    """Async variant of :func:`ask_paper`."""
    import asyncio
    return await asyncio.get_running_loop().run_in_executor(
        None, lambda: ask_paper(paper, question, api_key=api_key, model=model, timeout=timeout)
    )


def _summarize_one(
    arxiv_id: str,
    api_key: str,
    model: str,
    timeout: float,
) -> str:
    """Summarize a single paper (internal). Assumes imports and api_key are validated."""
    from google import genai
    from google.genai import types

    # 1. Download source
    logger.info("Downloading source for %s ...", arxiv_id)
    archive_bytes = _fetch_source(arxiv_id, timeout=timeout)

    # 2. Extract primary .tex
    tex_content = _find_primary_tex(archive_bytes)

    # 3. Trim after conclusion and clean
    tex_content = _trim_after_conclusion(tex_content)
    tex_content = _clean_tex(tex_content)

    logger.info("Tex content length: %d chars", len(tex_content))

    # 4. Call Gemini
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=SYSTEM_PROMPT),
                        types.Part(text=USER_PROMPT.format(tex_content=tex_content)),
                    ],
                ),
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="medium"),
            ),
        )
        summary = response.text
    except Exception as e:
        raise SummarizationError(f"Gemini API call failed: {e}") from e

    if not summary:
        raise SummarizationError("Gemini returned an empty response")

    logger.info("Summary generated for %s (%d chars)", arxiv_id, len(summary))
    return summary


def summarize_paper(
    paper: Paper | str | list[Paper] | list[str],
    api_key: str | None = None,
    model: str = "gemini-3-flash-preview",
    max_concurrent: int = 5,
    timeout: float = 120.0,
) -> str | dict[str, str]:
    """Download paper(s) LaTeX source and summarize using Google Gemini.

    Accepts a single paper or a list of papers. When given a list,
    papers are summarized in parallel using a thread pool.

    Args:
        paper: Paper object, ArXiv ID string, or a list of either.
        api_key: Google AI API key. Falls back to ``GEMINI_API_KEY`` env var.
        model: Gemini model to use.
        max_concurrent: Max parallel requests (only used for multiple papers).
        timeout: HTTP timeout for source download per paper.

    Returns:
        A summary string for a single paper, or a dict mapping
        ArXiv ID to summary string for multiple papers.
        Failed papers in a batch are logged and omitted from the dict.

    Raises:
        SummarizationError: If summarization fails (single paper mode).
        DownloadError: If source download fails (single paper mode).
    """
    import os

    try:
        from google import genai  # noqa: F401
    except ImportError:
        raise SummarizationError(
            "google-genai is required for summarization. "
            "Install it with: pip install arxiv-search-kit[summarize]"
        )

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SummarizationError(
            "Gemini API key required. Pass api_key= or set GEMINI_API_KEY env var."
        )

    # Single paper
    if not isinstance(paper, list):
        arxiv_id = paper.arxiv_id if isinstance(paper, Paper) else paper
        return _summarize_one(arxiv_id, api_key, model, timeout)

    # Multiple papers — parallel via thread pool
    from concurrent.futures import ThreadPoolExecutor, as_completed

    papers = paper
    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = {}
        for p in papers:
            aid = p.arxiv_id if isinstance(p, Paper) else p
            futures[pool.submit(_summarize_one, aid, api_key, model, timeout)] = aid

        for future in as_completed(futures):
            aid = futures[future]
            try:
                results[aid] = future.result()
            except Exception as e:
                logger.warning("Failed to summarize %s: %s", aid, e)

    return results


async def async_summarize_paper(
    paper: Paper | str | list[Paper] | list[str],
    api_key: str | None = None,
    model: str = "gemini-3-flash-preview",
    max_concurrent: int = 5,
    timeout: float = 120.0,
) -> str | dict[str, str]:
    """Async variant of :func:`summarize_paper`."""
    import asyncio
    return await asyncio.get_running_loop().run_in_executor(
        None, lambda: summarize_paper(
            paper, api_key=api_key, model=model,
            max_concurrent=max_concurrent, timeout=timeout,
        )
    )
