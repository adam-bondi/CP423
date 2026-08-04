# CP-423-B - Text Retrieval & Search Engine

Course Project: Build and Evaluate a Retrieval-Augmented Generation System

Team Members: Adam Bondi & Carson Yee

## 1. Background

A chatbot on its own can confidently generate incorrect information (often referred to as hallucinations) and cannot answer questions about documents it has never seen. Retrieval-Augmented Generation (RAG) addresses this limitation by first retrieving the most relevant passages from a document collection, then providing those passages to a Large Language Model (LLM) as context for answer generation.

The central question our project answers, supported by experimental evidence, is: *Does your system retrieve the appropriate context, and does the LLM effectively use that context to generate accurate answers?*

## 2. Corpus Selection

Among the suitable examples provided, we chose Wilfrid Laurier University's official **Student Homepage**, which contains university policies, academic calendars, regulations, co-op, residence, tuition, and related topics, as the corpus for our RAG system. As Data Science students, we are familiar with the layout of the academic calendars and wanted to choose something that demonstrates retrieval rather than just generation. This corpus is primarily text-based, written in English, publicly available, and a narrow specialized subset of information. This corpus is...

- Large enough that retrieval is necessary
- Includes hundreds of rich, highly factual, detailed pages/documents
- Not a standard IR benchmark
- Not something a modern LLM likely memorized well (unlike Wikipedia pages, arXiv abstracts, or books in the public domain)
- Easy to create evaluation questions and confirm evaluation set results

Overall, we think this corpus can successfully demonstrate the contribution of retrieval. To verify that our corpus is suitable, we performed a diagnostic experiment that demonstrates how the performance of our RAG system comes from effective retrieval rather than the LLM's memorized knowledge (see Section 3d).

## 3. Tasks

### a. Prepare the retrieval corpus

The retrieval corpus consists of official pages from the Wilfrid Laurier University Students website. URLs were obtained from the site's XML sitemap and filtered to include student information related to academics, career and experiential learning, campus services, programs, finances, and support and wellness. The `https://students.wlu.ca/sitemap.xml` contained 2,841 HTML documents. To filter our document collection, we filtered by page type. Rather than excluding entire sections, we filtered out page types whose URLs contain things like `/news/` and other non-reference pages that primarily contain time-sensitive announcements rather than stable factual information. After filtering, 1,959 candidate URLs remained.

Each HTML page was treated as a source document, with metadata including the page URL, title, section, last-modified date, and a unique document identifier preserved throughout preprocessing.

**Fetching and cleaning:** Pages were downloaded with a polite crawl delay (0.5s between requests). Raw HTML was parsed with BeautifulSoup; navigation, footer, header, and other non-content elements were stripped, along with WLU-specific boilerplate (breadcrumbs, banners, site header/footer). Common HTML entity and encoding artifacts introduced during scraping were normalized. Pages under 50 words after cleaning were dropped as low-content. This produced **1,872 final documents** from 1,957 successfully fetched pages (2 pages failed with 401 Unauthorized errors, restricted board/senate meeting pages).

**Chunking:** Each document's cleaned text was split into overlapping chunks of 400 words with a 50-word overlap between consecutive chunks, preserving the parent document's ID and metadata on every chunk. This produced **5,456 chunks** across the 1,872 documents (avg. 2.91 chunks/document).

Pipeline: `01_parse_sitemap.py` → `02_fetch_pages.py` → `03_extract_text.py` → `04_chunk_documents.py`

### b. Retrieval

Two retrieval methods were implemented over the 5,456-chunk index:

**Classical retrieval (BM25):** Implemented with `rank_bm25`, using NLTK word tokenization on lowercased chunk text. Built with `05_build_bm25.py`, queryable interactively with `06_test_bm25.py`, and wrapped in `retrieval/bm25_retriever.py` for use in the full pipeline.

**Dense retrieval:** Implemented using the pretrained `sentence-transformers/all-MiniLM-L6-v2` embedding model (384-dimensional embeddings), indexed with FAISS (`IndexFlatIP` over normalized embeddings, equivalent to cosine similarity). Built with `07_build_dense_index.py`, queryable interactively with `08_test_dense.py`, and wrapped in `retrieval/dense_retriever.py`.

Both retrievers expose a common `.retrieve(query, top_k)` interface returning ranked chunks with score, chunk ID, document ID, title, URL, and chunk text, so either can be swapped into the generation pipeline interchangeably.

**Observed retrieval behavior:** During testing, BM25 and dense retrieval showed different strengths depending on query phrasing. For short, keyword-style queries (e.g. `"USRA GPA requirement"`), BM25 retrieved highly precise, on-topic results. For the same information need phrased as a natural-language question (e.g. `"What GPA do I need to apply for a USRA?"`), both retrievers initially failed to surface the correct chunk at `top_k=5`; increasing to `top_k=10` allowed the correct chunk to be retrieved. This is discussed further in Section 3d and in our error analysis.

### c. Answer Generation

Answer generation uses a locally deployed, free LLM via [Ollama](https://ollama.com): **Meta Llama 3.2 (3B)**, accessed through Ollama's local REST API (`http://localhost:11434/api/generate`), with `temperature=0` for deterministic outputs.

The model is prompted with the retrieved chunks (labeled `[Chunk ID]`) and the user's question, and instructed to:
- Answer **only** using the provided context, never outside/parametric knowledge
- Reply with exactly `"I don't know."` when the context does not contain the answer
- Cite every fact used with an inline `[Chunk ID]` tag

Implementation: `generation/ollama_generator.py` (`OllamaGenerator` class). The full retrieval-to-generation pipeline is wired together in `rag_pipeline.py`, which can be run interactively and configured to use either `BM25Retriever` or `DenseRetriever`.

**Example (working correctly, `top_k=10`, dense retriever):**

> **Q:** What GPA do I need to apply for a USRA?
> **A:** A minimum cumulative B- average/GPA of 7.0.

> **Q:** What is the capital of Mongolia? (not in corpus)
> **A:** I don't know.

### d. Evaluation

**Diagnostic experiment (corpus validation):** Per the assignment requirements, we tested whether Llama 3.2 already "knows" our corpus content from pretraining. Ten factual questions were written from actual corpus pages (informed consent guidelines, USRA award details, Romeo research portal) and asked to the model with **no retrieved context** (`data/evaluation/run_diagnostic.py`, questions and reference answers in `data/evaluation/diagnostic_questions.csv`).

**Result: 0/10 correct.** The model consistently reported it had no specific knowledge of Laurier's guidelines, USRA figures, or internal systems like Romeo, rather than guessing (see `data/evaluation/diagnostic_results.csv` for full model responses). This confirms the corpus is sufficiently narrow and specialized that any correct answers from our full RAG pipeline are attributable to retrieval, not the model's pretraining — directly supporting our corpus selection rationale in Section 2.

**Gold evaluation set and full system evaluation:** *(in progress)* — a hand-written gold-standard question set (≥10 questions, including ≥2 multi-hop and ≥2 unanswerable questions, each with a reference answer and ground-truth source chunk ID) will be used to score both retrieval and generation quality across BM25 and dense retrieval. Results and error analysis will be reported here once complete.

## 4. Repository Structure

```
CP423/
├── src/
│   ├── 01_parse_sitemap.py      # Parse sitemap.xml, filter, assign doc_id
│   ├── 02_fetch_pages.py        # Download raw HTML per document
│   ├── 03_extract_text.py       # Clean HTML, extract text + metadata
│   ├── 04_chunk_documents.py    # Split documents into overlapping chunks
│   ├── 05_build_bm25.py         # Build BM25 index
│   ├── 06_test_bm25.py          # Interactive BM25 query tool
│   ├── 07_build_dense_index.py  # Build dense (FAISS) index
│   ├── 08_test_dense.py         # Interactive dense query tool
│   └── rag_pipeline.py          # Full retrieval + generation pipeline
├── retrieval/
│   ├── __init__.py
│   ├── bm25_retriever.py
│   └── dense_retriever.py
├── generation/
│   └── ollama_generator.py
├── data/
│   ├── raw/
│   │   ├── sitemap.xml
│   │   └── pages/                # Raw fetched HTML - not committed (see .gitignore) (added in this case for testing purposes)
│   ├── metadata/
│   │   ├── documents.csv
│   │   └── fetch_failures.csv
│   ├── processed/
│   │   ├── documents_text.csv
│   │   └── chunks.csv
│   ├── indexes/                  # BM25 + FAISS index files - not committed (see .gitignore)
│   └── evaluation/
│       ├── diagnostic_questions.csv
│       ├── diagnostic_results.csv
│       └── run_diagnostic.py
├── .gitignore
└── README.md
```

`data/raw/pages/` (raw fetched HTML) and `data/indexes/` (BM25 pickle, FAISS index, and embeddings pickle) are excluded from version control via `.gitignore`, since both are fully reconstructible by rerunning the pipeline scripts below. `__pycache__/` directories are also excluded.

## 5. Setup & Reproduction

**Dependencies:**
```
pip install pandas requests beautifulsoup4 tqdm nltk rank_bm25 faiss-cpu sentence-transformers
python -c "import nltk; nltk.download('punkt')"
```

**Local LLM:** Install [Ollama](https://ollama.com), then pull the model used in this project:
```
ollama pull llama3.2 or ollama list 
```
Ensure the Ollama app/service is running before running generation or the diagnostic script.

**Reproduce the full pipeline from scratch:**
```
python src/01_parse_sitemap.py
python src/02_fetch_pages.py
python src/03_extract_text.py
python src/04_chunk_documents.py
python src/05_build_bm25.py
python src/07_build_dense_index.py
```

**Run the diagnostic experiment:**
```
python data/evaluation/run_diagnostic.py
```

**Run the interactive RAG pipeline:**
```
python -m src.rag_pipeline
```
(Edit the retriever selection at the top of `rag_pipeline.py` to switch between `BM25Retriever` and `DenseRetriever`.)