# Technical Interview Guide & Deep-Dive Architecture

This document provides an exhaustive, line-by-line explanation of the **Documentation Assistant RAG Project** to help you ace your technical interview.

---

## Table of Contents

1. [Entire Workflow of the Project](#1-entire-workflow-of-the-project)
2. [Modules & Inter-Module Interactions](#2-modules--inter-module-interactions)
3. [Hyperparameters & Critical Variables](#3-hyperparameters--critical-variables)
4. [LLM, Embedding Models & Infrastructure](#4-llm-embedding-models--infrastructure)
5. [Exact Chunking Strategy](#5-exact-chunking-strategy)
6. [Web Crawler to Chunk Conversion Pipeline](#6-web-crawler-to-chunk-conversion-pipeline)
7. [Interview Speaking Points & Edge-Case Q&A](#7-interview-speaking-points--edge-case-qa)

---

## 1. Entire Workflow of the Project

The system is an **Agentic Retrieval-Augmented Generation (RAG)** documentation assistant. It operates via two completely decoupled pipelines: an **Offline Ingestion Pipeline** and an **Online Retrieval & Response Pipeline**.

```
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
                                       [ PHASE 1: OFFLINE INGESTION PIPELINE ]
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Seed URL             TavilyMap               TavilyExtract            Recursive Splitter        Gemini Embeddings       Pinecone Index
 (config.py) ───> (Discover URLs) ───> (Async Content Fetch) ───> (4000 char / 200 ov) ───> (gemini-embedding-001) ───> (Vector Store)

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
                                      [ PHASE 2: ONLINE AGENTIC RAG PIPELINE ]
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 User Query           Streamlit UI            LangChain Agent           retrieval_context Tool                     Vector Store
  (main.py)  ───> (Chat Interface) ───> (gemini-pro-latest) ─────────> (Top k=5 chunks) ──────────────────────────────┘
                                                    │
                                                    ▼
                                        Response + Source Attribution
```

### High-Level Summary of Execution:
1. **Offline Ingestion** ([`ingestion.py`](file:///d:/CODE/LLM/LangChain/documentation-assistant/ingestion.py)):
   - Discovers domain URLs starting from `KNOWLEDGE_BASE_URL` using `TavilyMap`.
   - Filters out unwanted administrative/forum URLs via regex patterns (`BAD_URLS`).
   - Asynchronously fetches raw text using `TavilyExtract` in batches of 20.
   - Splits extracted text into 4000-character semantic chunks with 200-character overlap using `RecursiveCharacterTextSplitter`.
   - Converts chunks into 1536-dimensional embeddings using `gemini-embedding-001`.
   - Indexes vectors into Pinecone DB in rate-limit-protected batches of 25 chunks.

2. **Online Agentic RAG** ([`backend/core.py`](file:///d:/CODE/LLM/LangChain/documentation-assistant/backend/core.py) & [`main.py`](file:///d:/CODE/LLM/LangChain/documentation-assistant/main.py)):
   - User inputs a prompt into the Streamlit chat UI.
   - Prompt is passed to a LangChain agent using `ChatGoogleGenerativeAI(model="gemini-pro-latest", temperature=0)`.
   - Agent dynamically invokes the `@tool` `retrieval_context(query)`.
   - The tool performs a similarity search on Pinecone for the top $k=5$ vector matches.
   - The tool returns serialized text to the LLM and raw document objects as artifacts to the application.
   - The LLM synthesizes a grounded response, and Streamlit displays the response along with expandable source links.

---

## 2. Modules & Inter-Module Interactions

```
                            ┌──────────────┐
                            │  config.py   │
                            └──────┬───────┘
                                   │ (Imports Constants)
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  ingestion.py   │       │ backend/core.py │       │     main.py     │
└────────┬────────┘       └────────┬────────┘       └────────┬────────┘
         │                         │                         │
         │ (Populates Vectors)     │ (Queries Vectors)       │ (Calls run_llm)
         ▼                         ▼                         │
┌─────────────────┐       ┌─────────────────┐                │
│   Pinecone DB   │ ◄──── │ LangChain Agent │ ◄──────────────┘
└─────────────────┘       └─────────────────┘
```

### Module Breakdown:

1. **[`config.py`](file:///d:/CODE/LLM/LangChain/documentation-assistant/config.py)**:
   - **Role**: Configuration source of truth.
   - **Key Variables**: `KNOWLEDGE_BASE_SUBJECT`, `KNOWLEDGE_BASE_URL`, `SCRAPING_INSTRUCTIONS`.
   - **Interactions**: Imported by `ingestion.py` (target URL & scraping instructions), `backend/core.py` (system prompt customization), and `main.py` (UI titles).

2. **[`logger.py`](file:///d:/CODE/LLM/LangChain/documentation-assistant/logger.py)**:
   - **Role**: Terminal logger utility.
   - **Functions**: `log_info`, `log_success`, `log_error`, `log_warning`, `log_header` using ANSI color formatting.
   - **Interactions**: Imported by `ingestion.py` to give real-time terminal feedback during extraction and indexing.

3. **[`ingestion.py`](file:///d:/CODE/LLM/LangChain/documentation-assistant/ingestion.py)**:
   - **Role**: Async ETL (Extract, Transform, Load) pipeline script.
   - **Interactions**:
     - Calls Tavily API (`TavilyMap`, `TavilyExtract`).
     - Uses `langchain_classic.text_splitter.RecursiveCharacterTextSplitter`.
     - Calls `GoogleGenerativeAIEmbeddings` to vectorize chunks.
     - Writes embeddings directly to `PineconeVectorStore`.

4. **[`backend/core.py`](file:///d:/CODE/LLM/LangChain/documentation-assistant/backend/core.py)**:
   - **Role**: RAG reasoning & execution engine.
   - **Interactions**:
     - Defines `retrieval_context(query)` decorated with `@tool(response_format="content_and_artifact")`.
     - Connects to Pinecone DB (`PineconeVectorStore`).
     - Initializes `ChatGoogleGenerativeAI`.
     - Builds the agent via `create_agent()` and exports `run_llm(query: str)`.

5. **[`main.py`](file:///d:/CODE/LLM/LangChain/documentation-assistant/main.py)**:
   - **Role**: Frontend presentation layer.
   - **Interactions**:
     - Calls `run_llm(prompt)` from `backend/core.py`.
     - Extracts source metadata via `_format_sources()`.
     - Renders chat messages and expandable source links in Streamlit.

---

## 3. Hyperparameters & Critical Variables

| Variable | Value | Location in Code | Deep Technical Explanation & Interview Rationale |
| :--- | :--- | :--- | :--- |
| **`chunk_size`** | `4000` | [`ingestion.py:308`](file:///d:/CODE/LLM/LangChain/documentation-assistant/ingestion.py#L308) | Specifies max characters per chunk. 4000 chars (~1000 tokens) captures complete wiki sections without fragmenting context. |
| **`chunk_overlap`** | `200` | [`ingestion.py:308`](file:///d:/CODE/LLM/LangChain/documentation-assistant/ingestion.py#L308) | 200 character overlap (5% ratio) between adjacent chunks ensures sentences spanning split boundaries maintain semantic continuity. |
| **`k` (Retrieval Count)** | `5` | [`backend/core.py:26`](file:///d:/CODE/LLM/LangChain/documentation-assistant/backend/core.py#L26) | Number of top vector matches returned during similarity search. 5 chunks yield comprehensive context while preventing LLM context bloat. |
| **`batch_size` (Extract)** | `20` | [`ingestion.py:297`](file:///d:/CODE/LLM/LangChain/documentation-assistant/ingestion.py#L297) | Batches URLs into groups of 20 for `TavilyExtract` to prevent HTTP request timeouts. |
| **`batch_size` (Indexing)** | `25` | [`ingestion.py:353`](file:///d:/CODE/LLM/LangChain/documentation-assistant/ingestion.py#L353) | Batches document chunks into groups of 25 for vector indexing into Pinecone. |
| **`semaphore`** | `2` | [`ingestion.py:24`](file:///d:/CODE/LLM/LangChain/documentation-assistant/ingestion.py#L24) | `asyncio.Semaphore(2)` restricts active concurrent extraction tasks, preventing HTTP 429 rate limit exceptions. |
| **`embedding_dimensionality`** | `1536` | [`ingestion.py:60`](file:///d:/CODE/LLM/LangChain/documentation-assistant/ingestion.py#L60) | Output dimension for `gemini-embedding-001`, matching standard high-density vector space metrics in Pinecone. |
| **`temperature`** | `0` | [`backend/core.py:20`](file:///d:/CODE/LLM/LangChain/documentation-assistant/backend/core.py#L20) | Set to 0 to eliminate LLM randomness and ensure strictly factual, deterministic answers. |
| **`max_depth` & `limit`** | `2` / `200` | [`ingestion.py:67`](file:///d:/CODE/LLM/LangChain/documentation-assistant/ingestion.py#L67), [`:283`](file:///d:/CODE/LLM/LangChain/documentation-assistant/ingestion.py#L283) | `TavilyMap` parameters controlling crawler link traversal depth (2 clicks) and max link count (200 URLs). |

---

## 4. LLM, Embedding Models & Infrastructure

### 1. Large Language Model (LLM)
- **Model**: `ChatGoogleGenerativeAI(model="gemini-pro-latest", temperature=0)` ([`backend/core.py:20`](file:///d:/CODE/LLM/LangChain/documentation-assistant/backend/core.py#L20)).
- **Role**: Generates grounded answers based on context supplied by the tool call.
- **Guardrails**: System prompt defense rules enforce grounding:
  > *"Use the tool to find relevant information... Always mention sources... If you cannot find the answer in retrieved documents, say so."*

### 2. Embeddings Model
- **Model**: `GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", task_type="RETRIEVAL_DOCUMENT", output_dimensionality=1536)` ([`ingestion.py:57-61`](file:///d:/CODE/LLM/LangChain/documentation-assistant/ingestion.py#L57-L61)).
- **Optimization**: Setting `task_type="RETRIEVAL_DOCUMENT"` optimizes vector generation specifically for asymmetric text retrieval (matching short user queries against longer text documents).

### 3. Vector Database
- **Database**: **Pinecone** (`PineconeVectorStore`).
- **Index**: `document-assistant` (or set via `INDEX_NAME` env variable).
- **Function**: Stores 1536-dimensional vector representations of chunks and computes cosine similarity for top $k=5$ retrieval.

### 4. Web Crawler Engine
- **Tools**: `TavilyMap`, `TavilyExtract`, `TavilyCrawl` ([`ingestion.py:66-68`](file:///d:/CODE/LLM/LangChain/documentation-assistant/ingestion.py#L66-L68)).
- **Role**: Automatically discovers site structures and extracts clean text without raw HTML/DOM noise.

---

## 5. Exact Chunking Strategy

### Algorithm: Recursive Character Text Splitting
- **Class**: `RecursiveCharacterTextSplitter` from `langchain_classic.text_splitter`.
- **Config**: `chunk_size=4000`, `chunk_overlap=200`.

### Splitting Hierarchy:
The algorithm uses a prioritized separator list: `["\n\n", "\n", " ", ""]`.
1. **Paragraph Level (`\n\n`)**: Tries to split text on double line breaks.
2. **Line Level (`\n`)**: If a paragraph exceeds 4000 chars, it splits on single line breaks.
3. **Word Level (`" "`)**: If a line exceeds 4000 chars, it splits on spaces.
4. **Character Level (`""`)**: Fallback character cut if word splitting is impossible.

### Why This Strategy?
Unlike naive character splitters that cut text blindly every $N$ characters, recursive character splitting keeps logical paragraphs, code snippets, and section headers intact. The 200-character overlap guarantees that context at chunk boundaries is preserved.

### Observability: Chunk Debug Report
In [`ingestion.py:313-350`](file:///d:/CODE/LLM/LangChain/documentation-assistant/ingestion.py#L313-L350), custom code profiles chunks before indexing:
- Calculates per-document character count.
- Estimates tokens (`char_length / 4`).
- Computes average characters per chunk.
- Displays the top 10 documents generating the highest chunk counts to detect abnormal pages.

---

## 6. Web Crawler to Chunk Conversion Pipeline

Here is the exact step-by-step breakdown of how raw web pages become vector chunks in Pinecone:

```
 Step 1: TavilyMap ──> Step 2: URL Filtering ──> Step 3: URL Batching (20)
                                                                │
 Step 6: Pinecone Index <── Step 5: Chunking <── Step 4: Async TavilyExtract
```

### Detailed Execution Steps:

1. **Step 1 — URL Discovery (`TavilyMap`)**:
   - `tavily_map.invoke()` crawls starting at `MAIN_URL` (`https://darksouls.fandom.com/wiki/Lore`).
   - Discovers candidate page URLs up to limit 200.

2. **Step 2 — URL Filtering**:
   - Compares URLs against `BAD_URLS` list:
     ```python
     BAD_URLS = ["veaction=edit", "action=edit", "/wiki/Special:", "/wiki/File:", "/wiki/Category:", ...]
     ```
   - Filters out non-content URLs (edit pages, categories, talk pages, administrative pages).

3. **Step 3 — URL Batching**:
   - `chunk_batches(urls, batch_size=20)` slices discovered URLs into sub-lists of 20 URLs each.

4. **Step 4 — Asynchronous Extraction (`TavilyExtract` + `asyncio`)**:
   - `async_extract()` spawns coroutines for each batch:
     ```python
     async def extract_batch(urls, batch_num):
         return await tavily_extract.ainvoke(input={"urls": urls, "extract_depth": "basic"})
     ```
   - Standardizes response payloads into LangChain `Document` objects:
     ```python
     Document(page_content=raw_content, metadata={"url": url})
     ```

5. **Step 5 — Text Chunking**:
   - `text_splitter.split_documents(all_docs)` partitions raw `Document` objects into 4000-character chunk `Document` objects while inheriting source `url` metadata.

6. **Step 6 — Vector Embedding & Batch Indexing**:
   - `batch_indexing(docs_split, batch_size=25)` splits chunks into batches of 25.
   - Converts text to 1536-dim embeddings via `GoogleGenerativeAIEmbeddings` and uploads vectors to Pinecone:
     ```python
     vector_store.add_documents(batch)
     await asyncio.sleep(1) # Rate limit safety buffer
     ```

---

## 7. Interview Speaking Points & Edge-Case Q&A

### Key Architecture Pitch (1-Minute Elevator Pitch)
> *"I designed an Agentic RAG Documentation Assistant using LangChain, Google Gemini, Pinecone, and Tavily. The project features a decoupled ingestion pipeline that maps, filters, and extracts web pages asynchronously, chunks them using recursive character splitting, and indexes 1536-dimensional embeddings in Pinecone. For retrieval, I built an Agentic RAG pipeline using LangChain's tool calling system. The LLM dynamically retrieves context via a custom `@tool` decorator that separates content for the model from document artifacts for UI source attribution."*

### Key Technical Interview Q&A

#### Q: How does the tool format differ from standard RAG chains?
**Answer**: Standard RAG chains manually inject retrieved documents into a rigid prompt template. My architecture uses `@tool(response_format="content_and_artifact")`. The tool returns a 2-tuple:
1. `serialized`: Text context formatted as `Source: {url} \nContext: {text}` passed to the LLM.
2. `retrieved_docs`: Raw `Document` objects returned as artifacts. The application inspects `ToolMessage.artifact` to dynamically display clickable sources in Streamlit.

#### Q: How do you prevent rate limits during bulk ingestion?
**Answer**: Rate limiting is handled at three layers:
1. `asyncio.Semaphore(2)` limits active concurrent network calls.
2. URL batching (20 per batch for extraction).
3. Embedding batching (25 chunks per batch) with `await asyncio.sleep(1)` delay between Pinecone uploads.

#### Q: How do you prevent hallucinations in answers?
**Answer**: Hallucination defense is enforced via three mechanisms:
1. `temperature=0` on `ChatGoogleGenerativeAI` to eliminate stochastic sampling.
2. System prompt instructions explicitly forbidding speculation: *"If you cannot find the answer in the retrieved documents, say so."*
3. Grounding via top $k=5$ similarity retrieval from Pinecone.
