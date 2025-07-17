# RagWizard

> Master the Magic of Retrieval-Augmented Generation (RAG)  
> A production-ready, modular pipeline for document ingestion, embedding, retrieval, and generation.

---

## Overview

RagWizard is a comprehensive RAG framework designed to include all essential features needed to master RAG workflows—from multi-format document loading and chunking to vector search, prompt engineering, and LLM integration. Built for scalability and production use.

---

## Features

- Multi-format document loaders (PDF, DOCX, HTML, CSV, JSON)  
- Advanced chunking (recursive, token-aware, semantic, hybrid)  
- Support for multiple embedding providers (OpenAI, HuggingFace, custom)  
- Vector stores integration (Qdrant, FAISS, Pinecone)  
- Flexible retrieval with similarity search, MMR, and metadata filtering  
- LLM integrations with function calling and streaming support  
- Chat history management and follow-up question handling  
- Automated ingestion pipelines and monitoring  
- Docker-ready with FastAPI backend and optional UI demos

---

## Quickstart

### Requirements

- Python 3.9+  
- Access to OpenAI or other embedding/LLM APIs (optional)  
- Docker (optional)

### Install

```bash
git clone https://github.com/yourusername/ragwizard.git
cd ragwizard
pip install -r requirements.txt
