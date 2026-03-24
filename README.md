# arxiv-search-kit

Offline ArXiv paper search over 928K CS papers. SPECTER2 embeddings + LanceDB vector index + BM25 hybrid retrieval.

**40ms per search on GPU. 99% precision@10. No API keys. No rate limits.**

## Install

```bash
pip install arxiv-search-kit[gpu]   # with CUDA
pip install arxiv-search-kit[cpu]   # CPU only
```

## Quick start

```python
from arxiv_search_kit import ArxivClient

client = ArxivClient()  # auto-downloads 4GB index on first run

# keyword search
papers = client.search("attention mechanism transformers", categories=["cs.CL", "cs.LG"])

# find related papers
related = client.find_related("1706.03762")  # Attention Is All You Need

# search with context paper (biases results toward your paper's neighborhood)
papers = client.search(
    "self-supervised learning",
    context_paper_id="2010.11929",  # ViT
)

# batch search (returns all unique papers across queries)
papers = client.batch_search([
    "vision transformers",
    "neural radiance fields",
    "RLHF alignment",
], max_results=10)

# conference-aware search
papers = client.search("object detection", conference="CVPR", year=2024)

# sort by citations (calls Semantic Scholar API)
papers = client.search("diffusion models", sort_by="citations", min_citations=50)
```

## What you get back

```python
paper = papers[0]
paper.arxiv_id      # "2401.12345"
paper.title          # "..."
paper.abstract       # "..."
paper.authors        # [Author(name="...", affiliation="..."), ...]
paper.categories     # ["cs.CV", "cs.LG"]
paper.published      # datetime
paper.pdf_url        # "https://arxiv.org/pdf/2401.12345"
paper.to_bibtex()    # BibTeX string
```

## Citation graph (via Semantic Scholar)

```python
citations = client.get_citations("1706.03762")
references = client.get_references("1706.03762")

# enrich search results with citation counts
client.enrich(papers)
papers[0].citation_count  # 95421
```

## Coverage

928K papers across all major CS + stat.ML categories:

cs.CV (144K), cs.LG (129K), cs.CL (78K), cs.AI (36K), cs.RO (38K), cs.CR (32K), stat.ML (20K), and 40+ more subcategories.

Maps conferences to categories: CVPR, NeurIPS, ICML, ICLR, ACL, EMNLP, AAAI, CHI, KDD, SIGIR, and more.

## How it works

1. Pre-built index: 928K papers embedded with SPECTER2 (same model as Semantic Scholar), stored in LanceDB
2. At query time: embed query with SPECTER2, hybrid retrieval (vector + BM25), graph-based re-ranking via Personalized PageRank
3. Index auto-downloads from [HuggingFace](https://huggingface.co/datasets/anonymousatom/arxiv-search-index) on first use (~4GB)

## Building your own index

Only needed if you want to customize the paper set or update to latest papers.

```bash
pip install arxiv-search-kit[index]

# download metadata from ArXiv OAI-PMH (takes ~2 hours)
python -m arxiv_search_kit.scripts.build_index download --output metadata.jsonl

# build index (needs GPU, takes ~45 min)
python -m arxiv_search_kit.scripts.build_index all --output-dir ./my_index --device cuda
```

## License

MIT
