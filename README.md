# CP-423-B - Text Retrieval & Search Engine
Course Project: Build and Evaluate a Retrieval-Augmented Generation System

Team Members: Adam Bondi & Carson Yee

## 1. Background
A chatbot on its own can confidently generate incorrect information (often referred to as hallucinations) and cannot answer questions about documents it has never seen. Retrieval-Augmented Generation (RAG) addresses this limitation by first retrieving the most relevant passages from a document collection, then providing those passages to a Large Language Model (LLM) as context for answer generation.

The central question our project answers, supported by experimental evidence, is: _Does your system retrieve the appropriate context, and does the LLM effectively use that context to generate accurate answers?_

## 2. Corpus Selection
Among the suitable examples provided, we chose Wilfrid Laurier University's official **Student Homepage**, which contains university policies, academic calendars, regulations, co-op, residence, tuition, and related topics, as the corpus for our RAG system. As Data Science students, we are familiar with the layout of the academic calendars and wanted to choose something that demonstrates retrieval rather than just generation. This corpus is primarily text-based, written in English, publicly available, and a narrow specialized subset of information. This corpus is...

- Large enough that retrieval is necessary
- Includes hundreds of rich, highly factual, detailed pages/documents)
- Not a standard IR benchmark
- Not something a modern LLM likely memorized well (unlike Wikipedia pages, arXiv abstracts, or books in the public domain)
- Easy to create evaluation questions and confirm evaluation set results

Overall, we think this corpus can successfully demonstrate the contribution of retrieval. To verify that our corpus is suitable, we  perform a diagnostic experiment that demonstrates how the performance of your RAG system comes from effective retrieval rather than the LLM's memorized knowledge.

## 3. Tasks

### a. Prepare the retrieval corpus:
The retrieval corpus consists of official pages from the Wilfrid Laurier University Students website. URLs were obtained from the site's XML sitemap and filtered to include student information related to academics, career and experiential learning, campus services, programs, finances, and support and wellness. The _https://students.wlu.ca/sitemap.xml_ contained 2,841 HTML documents. To filter our document collection, we filtered by page type. Rather than excluding entire sections, we filtered out page types whose URLs contain things like /news/ and other non-reference pages that primarily contain time-sensitive announcements rather than stable factual information. Each HTML page was treated as a source document, with metadata including the page URL, title, section, last-modified date, and a unique document identifier preserved throughout preprocessing.

### b. Retrieval:

### c. Answer Generation: Prompt the LLM using the retrieved document chunks

### d. Evaluation: 


