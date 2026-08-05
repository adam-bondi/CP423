# CP-423-B - Text Retrieval & Search Engine

Course Project: Build and Evaluate a Retrieval-Augmented Generation System

Team Members: Adam Bondi & Carson Yee

## Overview
This project builds and evaluates a Retrieval-Augmented Generation (RAG) system over Wilfrid Laurier University's public Students website. Given a question, the system retrieves relevant text chunks using either BM25 or dense semantic retrieval and supplies those chunks to a locally deployed Llama 3.2 model. The model is instructed to answer only from the retrieved evidence, cite chunk IDs inline, and return "I don't know." when the context is insufficient.

The project investigates two questions:
- Does the retriever return the source evidence needed to answer the question?
- Does the language model use that evidence accurately and cite it correctly?

## 1. Background

A chatbot on its own can confidently generate incorrect information (often referred to as hallucinations) and cannot answer questions about documents it has never seen. Retrieval-Augmented Generation (RAG) addresses this limitation by first retrieving the most relevant passages from a document collection, then providing those passages to a Large Language Model (LLM) as context for answer generation.

The central question our project answers, supported by experimental evidence, is: *Does your system retrieve the appropriate context, and does the LLM effectively use that context to generate accurate answers?*

## 2. Corpus Selection

Among the suitable examples provided, we chose Wilfrid Laurier University's official **Student Homepage**, which contains university policies, academic calendars, regulations, co-op, residence, tuition, and related topics, as the source corpus for our RAG system. As Data Science students, we are familiar with the layout of the academic calendars and wanted to choose something that demonstrates retrieval rather than just generation. This corpus is primarily text-based, written in English, publicly available, and a narrow specialized subset of information. This corpus is...

- Large enough that retrieval is necessary
- Includes hundreds of rich, highly factual, detailed pages/documents
- Not a standard IR benchmark
- Not something a modern LLM likely memorized well (unlike Wikipedia pages, arXiv abstracts, or books in the public domain)
- Easy to create evaluation questions and confirm evaluation set results

Overall, we think this corpus can successfully demonstrate the contribution of retrieval. To verify that our corpus is suitable, we performed a diagnostic experiment that demonstrates how the performance of our RAG system comes from effective retrieval rather than the LLM's memorized knowledge.

### System Design
```
sitemap.xml
    -> URL filtering and document IDs
    -> HTML crawling
    -> main-text extraction and cleaning
    -> 400-word chunks with 50-word overlap
    -> BM25 index / Sentence-Transformer + FAISS index
    -> top-5 retrieved chunks
    -> Llama 3.2 through Ollama
    -> grounded answer with inline chunk citations
```

## 3. Tasks

### a. Prepare the Retrieval Corpus

The retrieval corpus consists of official pages from the Wilfrid Laurier University Students website. URLs were obtained from the site's XML sitemap and filtered to include student information related to academics, career and experiential learning, campus services, programs, finances, and support and wellness. The `https://students.wlu.ca/sitemap.xml` contained 2,841 HTML documents. To filter our document collection, we filtered by page type. Rather than excluding entire sections, we filtered out page types whose URLs contain things like /news/ and other non-reference pages that primarily contain time-sensitive announcements rather than stable factual information. After filtering, 1,957 candidate URLs remained.

Each HTML page was treated as a source document, with metadata including the page URL, title, section, last-modified date, and a unique document identifier preserved throughout preprocessing.

**Fetching and Preprocessing:** Pages were downloaded with a polite crawl delay (0.5s between requests). Raw HTML was parsed with BeautifulSoup and cleaned; navigation, footer, header, and other non-content elements were stripped, along with WLU-specific boilerplate (breadcrumbs, banners, site header/footer). Common HTML entity and encoding artifacts introduced during scraping were normalized. Pages under 50 words after cleaning were dropped as low-content. This produced **1,872 clean source documents** from 1,957 successfully fetched pages. 
- Note: 2 pages (restricted board/senate meeting pages) failed with HTTP 401 Unauthorized errors.

**Chunking:** Each document's cleaned text was split into overlapping chunks of 400 words with a 50-word overlap between consecutive chunks, preserving the parent document's ID and metadata on every chunk. This produced **5,456 chunks** across the 1,872 documents (avg. 2.91 chunks/document). Each chunk preserves its unique chunk ID, parent document ID, title, URL, section, last-modified date, and chunk index.

Pipeline: `01_parse_sitemap.py` → `02_fetch_pages.py` → `03_extract_text.py` → `04_chunk_documents.py`

### b. Retrieval

Two retrieval methods were implemented over the 5,456-chunk index:

**Classical Retrieval (BM25):** Implemented with rank_bm25, using NLTK word tokenization on lowercased chunk text. Built with `05_build_bm25.py`, queryable interactively with `06_test_bm25.py`, and wrapped in `retrieval/bm25_retriever.py` for use in the full pipeline.

**Dense Retrieval:** Implemented using the pretrained sentence-transformers/all-MiniLM-L6-v2 embedding model (384-dimensional embeddings), indexed with FAISS (IndexFlatIP over normalized embeddings, equivalent to cosine similarity). Built with `07_build_dense_index.py`, queryable interactively with `08_test_dense.py`, and wrapped in `retrieval/dense_retriever.py`.

Both retrievers expose a common `.retrieve(query, top_k)` interface returning ranked chunks with score, chunk ID, document ID, title, URL, and chunk text, so either can be swapped into the generation pipeline interchangeably. The command for such would be: results = retriever.retrieve(query, top_k=5)

### c. Answer Generation

Answer generation uses a locally deployed, free LLM via [Ollama](https://ollama.com): **Meta Llama 3.2 (3B)**, accessed through Ollama's local REST API (`http://localhost:11434/api/generate`), with `temperature=0` for deterministic outputs.

The model is prompted with the retrieved chunks (labelled `[Chunk ID]`) and the user's question, and instructed to:
- Answer **only** using the provided context, never outside/parametric knowledge
- Reply with exactly `"I don't know."` when the context does not contain the answer
- Cite every fact used with an inline `[Chunk ID]` tag

Implementation: `generation/ollama_generator.py` (`OllamaGenerator` class). The full retrieval-to-generation pipeline is wired together in `rag_pipeline.py`, which can be run interactively and configured to use either `BM25Retriever` or `DenseRetriever`.

**Example (ran during trial, working correctly, `top_k=5`, dense retriever):**

**Q:** What GPA do I need to apply for a USRA?
**A:** A minimum cumulative B- average/GPA of 7.0.

**Q:** What is the capital of Canada? (not in corpus)
**A:** I don't know.

### d. Evaluation

**Diagnostic Experiment (Corpus Validation):** Per the assignment requirements, we tested whether Llama 3.2 already "knows" our corpus content from pretraining. Ten corpus-specific factual questions were written from actual corpus pages and asked to the model with **no retrieved context**. Our chosen LLM answered these questions without any retrieved context. The final repository includes the diagnostic questions and responses under `./data/evaluation/diagnostic_questions.csv` and `./data/evaluation/diagnostic_results.csv`. This experiment verified that our corpus was suitable and that the performance of our RAG system comes from effective retrieval rather than the LLM's memorized knowledge.

**Gold Evaluation Set and Full System Evaluation:**
The gold set contains ten manually written and verified questions:
- 6 factoid questions;
- 2 multi-hop questions requiring evidence from multiple chunks or documents; and
- 2 unanswerable questions

Each answerable question includes a reference answer and one or more ground-truth chunk IDs. Both retrieval systems were evaluated with top_k=5 and the same Llama model, prompt, and generation settings. Retrieval metrics were computed automatically. Generated answers were graded manually for correctness, support, citation accuracy, and correct refusal on unanswerable questions.

**Results:** Results are shown in `data/evaluation/evaluation_results.csv`

BM25 performed best on this evaluation set, particularly for questions using terminology closely matching the source pages. Dense retrieval missed the correct evidence for two date-related questions. Both systems frequently produced correct answers while citing a nearby or related chunk instead of the exact supporting chunk. Citation reliability was therefore substantially lower than answer correctness.

## 4. Repository Structure

```
CP423/
├── generation/
│   ├── ollama_generator.py
├── retrieval/
│   ├── __init__.py
│   ├── bm25_retriever.py
│   └── dense_retriever.py
├── src/
│   ├── 01_parse_sitemap.py
│   ├── 02_fetch_pages.py
│   ├── 03_extract_text.py
│   ├── 04_chunk_documents.py
│   ├── 05_build_bm25.py
│   ├── 06_test_bm25.py
│   ├── 07_build_dense_index.py
│   ├── 08_test_dense.py
│   ├── evaluation.py
│   ├── rag_pipeline.py
│   └── summarize_results.py
├── data/
│   ├── raw/
│   │   ├── sitemap.xml
│   │   └── pages/                 # Raw fetched HTML - reconstructed by the crawler; not committed (see .gitignore)
│   ├── metadata/
│   │   ├── documents.csv
│   │   └── fetch_failures.csv
│   ├── processed/
│   │   ├── documents_text.csv
│   │   └── chunks.csv
│   ├── indexes/                   # BM25 + dense index files - reconstructed during 05_build_bm25.py and 07_build_dense_index.py; not committed (see .gitignore)
│   └── evaluation/
│       ├── diagnostic_questions.csv
│       ├── diagnostic_results.csv
│       ├── run_diagnostic.py
│       ├── gold_questions.csv # Hand-written gold-standard evaluation set
│       ├── evaluation_results.csv # Full graded results generated by evaluation.py
│       └── metrics.txt    # Generated by summarize_results.py
├── .gitignore
├── README.md
└── requirements.txt
```

`./data/raw/pages/` (raw fetched HTML) and `./data/indexes/` — BM25 pickle (bm25_index.pkl), FAISS index (dense.index), and embeddings pickle (dense_embeddings.pkl) — are excluded from version control via `.gitignore`, since both are fully reconstructible by rerunning the pipeline scripts below. `__pycache__/` directories are also excluded.

## 5. Setup & Reproduction

**Dependencies:**
```
pip install pandas requests beautifulsoup4 tqdm nltk rank_bm25 faiss-cpu sentence-transformers
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

**Local LLM:** Install [Ollama](https://ollama.com), start its service, and pull the exact model used in the experiments:
```
ollama pull llama3.2
```
Ensure the Ollama app/service is running before running generation or the diagnostic script.

**Reproduce the Full Pipeline (& Dataset) From Scratch:** 
The sitemap is included in data/raw/sitemap.xml. Raw HTML and index files are excluded from the GitHub repository because they are reconstructible. There is also no need to rerun the diagnostic experiment, as the corpus has already been verified.

1. Run the preprocessing and chunking:
```
python src/01_parse_sitemap.py
python src/02_fetch_pages.py
python src/03_extract_text.py
python src/04_chunk_documents.py
```
Note that fetching the pages and extracting the text takes a considerable amount of time.

2. Build the two retrieval indexes:
```
python src/05_build_bm25.py
python src/07_build_dense_index.py
```

3. Run the system
- Interactive BM25 testing: ```python src/06_test_bm25.py```
- Interactive dense testing: ```python src/08_test_dense.py```

4. Run the interactive RAG:
```python -m src.rag_pipeline```

Edit the retriever selection (BM25Retriever or DenseRetriever) at the top of `src/rag_pipeline.py` file to quickly switch between models. This is where trial testing was performed. Currently set to `top-k=5`, our chosen standard.

5. Gold-standard set evaluation questions:

```python -m src.evaluation``` 

Run this single command to reproduce the Experimental Results. This runs all 10 gold-standard questions through both BM25 and dense retrieval, generates responses, and saves raw results to `./data/evalaution/evaluation_results.csv`. 

This command:
- imports the BM25 and dense retrievers
- imports the Ollama generator
- loads and runs the gold evaluation set through BM25 and dense RAG systems; and
- generates `.data/evaluation/summary_results.csv`

_Note_: The evaluation script does not preserve existing human-grading columns (correct, supported, citation_correct, idk_correct) when it is rerun. Because generation correctness and citation support require human judgment, those labels should be reviewed whenever generated answers change. Running this command will clear those column values, and they should be re-graded by the team whenever generated answers change. Thus, the graded results currently in the repository reflect our team's manual evaluation.

**Reproducibility**
- Llama generation uses temperature=0.
- BM25 is deterministic.
- Dense embeddings are produced in inference mode without model training.
- FAISS IndexFlatIP performs deterministic exact search.
- The same model, prompt, top_k=5, and generation settings are used for both retrieval systems.

The process to reproduce all experimental results includes a required human-grading step, so it cannot be fully reproduced automatically from scratch with one command. Instead, the raw retrieval and generation outputs can be reproduced with `python -m src.evaluation`. Because generation correctness, evidential support, and citation accuracy were evaluated manually, the graded `evaluation_results.csv` is included in the repository. Running `python src/summarize_results.py` reproduces all reported tables and metrics from the recorded human judgments.

Follow the structure below:

- 1. ```python -m src.evaluation``` reruns both retrievers and the Ollama generation pipeline.
- 2. ```python src/summarize_results.py``` reproduces the reported summary metrics

**Known Limitations**
- The gold set contains only ten questions and focuses on one university website.
- Questions were written after reading the corpus, so wording may favour lexical retrieval.
- Multi-hop retrieval was only partially successful: retrieving at least one relevant chunk did not always retrieve every required chunk.
- Correct answers did not guarantee correct citations. The LLM often cited a nearby or topically related chunk rather than the exact evidence source.
- The system has no reranking, query expansion, or post-generation citation validator.
- Website content can change after the crawl date, so future runs may produce different documents and answers.
