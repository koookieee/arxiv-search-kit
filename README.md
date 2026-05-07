# arxiv-search-kit

Offline ArXiv paper search over 928K CS papers. LanceDB vector index + BM25 hybrid retrieval.

## Contents

- [Which backend should I use?](#which-backend-should-i-use)
- [Install](#install)
- [Gemini Embeddings](#gemini-embeddings) ← recommended
- [SPECTER2 Embeddings](#specter2-embeddings)
- [Search Options](#search-options)
- [HTTP API Server](#http-api-server)
- [Paper Q&A](#paper-qa)
- [Enrichment & Citations](#enrichment--citations)
- [Download Papers](#download-papers)
- [Paper Object](#paper-object)

---

## Which backend should I use?

| | Gemini-2 | SPECTER2 |
|---|---|---|
| **Input** | query string only | query string (+ optional context) |
| **Quality** | higher — handles free-text natively | good, improves with context |
| **Speed** | ~200ms (API call) | ~40ms on GPU, ~500ms on CPU |
| **Cost** | Gemini API key required | free, local |
| **Index size** | ~10GB | ~4GB |

**Use Gemini** if you have an API key and want the best out-of-the-box quality — just pass queries, no abstract needed.

**Use SPECTER2** if you want fully local search with no API dependencies. Add `context_title` + `context_abstract` for better results when you have them.

---

## Install

```bash
pip install arxiv-search-kit[cpu]
# or
uv add "arxiv-search-kit[cpu]"
```

GPU (CUDA):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install arxiv-search-kit[gpu]
```

The pre-built index auto-downloads from HuggingFace on first use (~4GB for SPECTER2, ~10GB for Gemini).

---

## Gemini Embeddings

**Just pass query strings — no title or abstract needed.** The Gemini embedding model handles asymmetric retrieval internally.

```bash
export GEMINI_API_KEY=AIza...
```

```python
from arxiv_search_kit import ArxivClient

client = ArxivClient(embedding="gemini")
# or: ArxivClient(embedding="gemini", gemini_api_key="AIza...")
```

### `batch_search` — recommended

Pass multiple queries covering different angles of a topic. Results are merged and deduplicated. All queries are embedded in a **single API call**.


With filters and importance ranking:

```python
results = client.batch_search(
    queries=[
        "efficient LLM fine-tuning",
        "parameter-efficient transfer learning",
        "LoRA low-rank adaptation",
    ],
    max_results=6,
    categories=["cs.LG", "cs.CL"],
    sort_by="importance",   # blends relevance with citations + venue prestige
)
```

### `search` — single query

```python
results = client.search("efficient fine-tuning of large language models", max_results=20)

# with filters
results = client.search(
    "vision transformers object detection",
    categories=["cs.CV"],
)
```

### `find_related` — by paper ID

```python
related = client.find_related("1706.03762", max_results=10)  # Attention Is All You Need
```

---

## SPECTER2 Embeddings

Local model, no API key required. Works well for standalone queries; add `context_title` + `context_abstract` when you have them for better precision.

```python
from arxiv_search_kit import ArxivClient

client = ArxivClient()  # SPECTER2 is the default

# basic query
results = client.search("vision transformers object detection")

# with context — biases results toward your paper's neighborhood
results = client.search(
    "self-supervised learning",
    context_title="My Paper Title",
    context_abstract="We propose a method for...",
)

# by ArXiv ID (uses stored embedding from the index)
results = client.search("self-supervised learning", context_paper_id="2010.11929")
```

---

## Search Options

These options work with both backends.

### Filters

```python
results = client.search("graph neural networks", categories=["cs.LG", "cs.AI"], year=2024)
results = client.search("LLM safety", date_from="2024-01-01", date_to="2024-06-30")
results = client.search("object detection", conference="CVPR", year=2024)
results = client.search("transformers", min_citations=50)
```

### Sorting

```python
results = client.search("diffusion models", sort_by="relevance")   # default
results = client.search("diffusion models", sort_by="citations")   # citation count via S2
results = client.search("diffusion models", sort_by="date")
results = client.search("diffusion models", sort_by="importance")  # relevance × citations × venue
```

`sort_by="importance"` blends relevance with citation count, venue prestige, and influential citation ratio:

```
importance = 0.55 * log_citation_score + 0.30 * venue_score + 0.15 * influential_ratio
final_score = 0.6 * relevance + 0.4 * importance
```

### Fields

By default results include: `arxiv_id`, `title`, `abstract`, `citation_count` (only when using `sort_by="importance"` or `"citations"`).

```python
results = client.search("transformers", details="extra")
# extra adds: authors, categories, doi, venue, tldr, publication_types, ...
```

### Async

```python
results = await client.async_search("transformers")
results = await client.async_batch_search(queries=[...], sort_by="importance")
related  = await client.async_find_related("1706.03762")
```

---

## HTTP API Server

`search_api.py` exposes the full kit as an HTTP API — useful for running on a GPU machine and querying from agents or remote clients.

### Start

```bash
python search_api.py --port 8081 --gemini-api-key AIza... --gemini-index-dir /path/to/gemini_index
```

Default port is `8081`. Pass `GEMINI_API_KEY` as an env var instead of `--gemini-api-key` if preferred.

### Endpoints

All endpoints accept JSON POST bodies. The `"embedding"` field selects the backend (`"gemini"` or `"specter2"`); **default is `"gemini"`**.

#### `POST /batch_search` — recommended

```python
import requests

resp = requests.post("http://localhost:8081/batch_search", json={
    "queries": [
        "reinforcement learning from human feedback",
        "process reward models",
        "on-policy distillation",
    ],
    "max_results": 20,
    # "embedding": "gemini",    # default
    # "categories": ["cs.LG"],
    # "year": 2024,
    # "sort_by": "importance",
})
papers = resp.json()["papers"]
```

#### `POST /search` — single query

```python
resp = requests.post("http://localhost:8081/search", json={
    "query": "efficient fine-tuning of LLMs",
    "max_results": 10,
})
papers = resp.json()["papers"]
```

#### `POST /find_related`

```python
resp = requests.post("http://localhost:8081/find_related", json={
    "arxiv_id": "1706.03762",
    "max_results": 10,
})
```

#### `POST /query_paper`

```python
resp = requests.post("http://localhost:8081/query_paper", json={
    "arxiv_id": "1706.03762",
    "query": "summarize this paper",
    "api_key": "AIza...",  # or set GEMINI_API_KEY on the server
})
print(resp.json()["results"]["1706.03762"])
```

Batch — pass `arxiv_ids` list:

```python
resp = requests.post("http://localhost:8081/query_paper", json={
    "arxiv_ids": ["1706.03762", "2005.14165"],
    "query": "what datasets were used?",
})
print(resp.json()["results"])  # {"1706.03762": "...", "2005.14165": "..."}
```

#### Other endpoints

| Endpoint | Body | Description |
|---|---|---|
| `GET /health` | — | returns `{"status": "ok", "clients_loaded": [...]}` |
| `POST /enrich` | `{"arxiv_ids": [...]}` | fetch citation/venue data from Semantic Scholar |
| `POST /citations` | `{"arxiv_id": "...", "limit": 50}` | papers that cite this paper |
| `POST /references` | `{"arxiv_id": "...", "limit": 50}` | papers referenced by this paper |
| `POST /get_paper` | `{"arxiv_id": "..."}` | fetch a single paper by ID |
| `POST /download_source` | `{"arxiv_id": "..."}` | list files in the LaTeX source archive |
| `POST /read_file` | `{"arxiv_id": "...", "file_path": "main.tex"}` | read a file from the source archive |

---

## Paper Q&A

Query any paper with natural language. Downloads the LaTeX source, converts to Markdown via pandoc, then queries Gemini.

```bash
pip install arxiv-search-kit[summarize]
apt install pandoc
```

```python
client = ArxivClient()  # embedding backend doesn't matter here

# summarize
response = client.query_paper("1706.03762", "summarize this paper")

# specific questions
response = client.query_paper("1706.03762", "what datasets were used?")
response = client.query_paper("1706.03762", "explain the loss function")

# batch — parallel across multiple papers
results = client.search("vision transformers", max_results=5)
responses = client.query_paper(results.papers, "summarize this paper")
# {"2401.12345": "...", "2312.67890": "...", ...}

# control parallelism (default: 5)
responses = client.query_paper(results.papers, "what optimizer was used?", max_concurrent=3)
```

Set `GEMINI_API_KEY` to avoid passing `api_key=` each time.

---

## Enrichment & Citations

```python
# enrich search results with citation data, venue, TL;DR
results = client.search("attention mechanism")
client.enrich(results)

results[0].citation_count   # 95421
results[0].venue            # "Neural Information Processing Systems"
results[0].tldr             # "A new architecture based solely on attention..."

# enrich specific fields only
client.enrich(results, fields=["citationCount", "venue"])

# citation graph
citations  = client.get_citations("1706.03762", limit=100)
references = client.get_references("1706.03762", limit=100)
```

The Semantic Scholar API works without a key (5,000 req / 5 min shared). For heavier use:

```bash
export S2_API_KEY=your_key_here
```

---

## Download Papers

```python
# single paper
path = client.download_pdf("1706.03762", output_dir="./papers")
path = client.download_source("1706.03762", output_dir="./sources")

# from search results
results = client.search("vision transformers", max_results=5)
paths = client.download_papers(results.papers, output_dir="./papers", format="pdf")
```

---

## Paper Object

```python
paper = results[0]

paper.arxiv_id          # "2401.12345"
paper.title             # "Paper Title"
paper.abstract          # "We propose..."
paper.authors           # [Author(name="Alice", affiliation="MIT"), ...]
paper.author_names      # ["Alice", "Bob"]
paper.categories        # ["cs.CV", "cs.LG"]
paper.primary_category  # "cs.CV"
paper.published         # datetime(2024, 1, 15)
paper.year              # 2024
paper.pdf_url           # "https://arxiv.org/pdf/2401.12345"
paper.abs_url           # "https://arxiv.org/abs/2401.12345"
paper.doi               # "10.1234/..." or None
paper.journal_ref       # "NeurIPS 2024" or None
paper.similarity_score  # 0.87

# enrichment fields (after client.enrich() or sort_by="importance"/"citations")
paper.citation_count              # 142
paper.influential_citation_count  # 23
paper.venue                       # "Neural Information Processing Systems"
paper.tldr                        # "This paper proposes..."

paper.to_dict()         # dict with all fields
paper.to_bibtex()       # BibTeX string
paper.to_bibtex("acl")  # ACL-style BibTeX
```

---

## Coverage

928K papers: **cs.CV** (144K), **cs.LG** (129K), **cs.CL** (78K), **cs.AI** (36K), **cs.RO** (38K), **cs.CR** (32K), **stat.ML** (20K), and 40+ more subcategories.

Conference mappings: CVPR, NeurIPS, ICML, ICLR, ACL, EMNLP, NAACL, AAAI, IJCAI, CHI, KDD, SIGIR, RSS, ICRA, and [many more](arxiv_search_kit/categories.py).

## How It Works

1. **Index**: 928K papers embedded with SPECTER2 or Gemini-2, stored in [LanceDB](https://lancedb.github.io/lancedb/)
2. **Retrieval**: Hybrid — dense (cosine) + sparse (BM25) fused via Reciprocal Rank Fusion
3. **Re-ranking**: Personalized PageRank on a k-NN similarity graph built from candidate embeddings
4. **Enrichment**: Optional citation/venue data from [Semantic Scholar API](https://api.semanticscholar.org/api-docs/graph)

## License

MIT
