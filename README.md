# Documentation Assistant

A reusable **RAG-based documentation assistant** that can turn a website or wiki into a searchable chatbot.

It works by:

1. **Scraping and mapping** pages from a base URL
2. **Extracting raw content** from those pages
3. **Chunking** the content into smaller sections
4. **Embedding** those chunks into vectors
5. **Storing** them in a **Pinecone** vector database
6. **Retrieving relevant chunks** for a user query
7. Sending the retrieved context to an **LLM**
8. Returning a grounded answer in a **Streamlit chat UI**

This project is designed as a **template**.  
By changing a few values in `config.py`, you can create a new assistant for a different documentation site, wiki, or knowledge source.

---

## Features

- End-to-end **RAG pipeline**
- Async ingestion pipeline with batching
- Vector search with **Pinecone**
- Embeddings via **Google Gemini**
- Website mapping and extraction via **Tavily**
- Chat interface built with **Streamlit**
- Source display for retrieved documents
- Easy to reuse for other sites by editing `config.py`

---

# Documentation Assistant (RAG)

A reusable **RAG-based documentation assistant** that turns any documentation site or wiki into a searchable chatbot.

The system scrapes a site, embeds the content into a vector database, and retrieves relevant chunks to answer user queries with an LLM.

Example use cases:
- Documentation assistants
- Wiki/lore assistants
- API doc chatbots
- Internal knowledge bases

---

# Architecture

### Ingestion pipeline

Base URL → Crawl → Extract → Chunk → Embed → Pinecone

### Retrieval pipeline

User Query → Embed → Vector Search → Top-K Chunks → LLM → Response

---

# Tech Stack

- Python
- LangChain
- Google Gemini
- Pinecone
- Tavily
- Streamlit
- uv (package manager)

---

# Setup

## Install dependencies

```
bash
uv sync
```

## Add API keys to .env
```
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
INDEX_NAME=your_pinecone_index_name
TAVILY_API_KEY=your_tavily_api_key
```

## Edit configure.py
```
SUBJECT = "assistant_subject"
BASE_URL = "URL_from_where_to_scrape_documentation"
```

## Run ingestion.py and streamlit
`uv run python ingestion.py`
`uv run python ingestion.py`
open `http://localhost:8501`

