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
You are an expert academic researcher who produces exhaustive, publication-quality paper summaries. \
You will be given the LaTeX source of a research paper. Your job is to produce a summary so \
detailed and thorough that a reader would not need to read the original paper to fully understand \
every important aspect of the work. Do not be brief — be comprehensive. Extract every significant \
detail from the source."""

USER_PROMPT = """\
Produce an exhaustive, highly detailed summary of the following LaTeX paper source. \
A reader of your summary should walk away understanding this paper as well as if they read it. \
Your summary MUST cover ALL of the following sections in depth:

1. **Paper Title & Authors** — Full title, all authors, and their affiliations if available.

2. **Problem Statement & Motivation** — What problem does this paper address? Why is it important? \
What are the specific shortcomings of prior work that motivated this research? Include any concrete \
examples or statistics the authors use to motivate their work.

3. **Key Contributions** — List every novel contribution the paper claims, exactly as framed by \
the authors. Be specific — not "improved performance" but exactly what was improved, by how much, \
and on what.

4. **Related Work** — Summarize how the paper positions itself relative to prior work. What are \
the key prior approaches discussed, and how does this work differ from or build upon each?

5. **Methodology / Proposed Approach** — Describe the full technical approach in detail. Include:
   - Architecture/system design with all components
   - Key equations, formulations, and mathematical definitions
   - Training procedure, loss functions, optimization details
   - Any novel techniques, tricks, or design choices and their justifications
   - Hyperparameters and configuration details mentioned

6. **Experimental Setup** — Describe all datasets, evaluation metrics, baselines/comparisons, \
implementation details (hardware, training time, batch size, learning rate schedules, etc.).

7. **Results & Benchmark Performance** — Report ALL quantitative results from every table and \
figure discussed. Include specific numbers (accuracy, BLEU, F1, latency, FLOPs, etc.), which \
model variants were tested, and how they compare to each baseline. Do not omit any result table.

8. **Ablation Studies & Analysis** — Summarize every ablation experiment: what was varied, what \
was the effect, and what conclusions were drawn. Include any qualitative analysis, visualizations, \
or case studies the authors discuss.

9. **Limitations & Failure Cases** — Note any limitations, failure modes, or weaknesses the authors \
acknowledge. If they don't explicitly state limitations, note any that are apparent from the results.

10. **Conclusion & Future Work** — The main takeaways, broader impact if discussed, and any \
future research directions the authors propose.

Be extremely thorough. Every specific number, every comparison, every design decision matters. \
Use bullet points and sub-sections for clarity. Keep the language precise and technical.

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
