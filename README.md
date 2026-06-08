# 🚀 RAG Document Retrieval System

AI-powered document retrieval system using **LangChain, FAISS, BM25, and advanced retrieval techniques** to build a scalable, production-ready RAG pipeline with quantitative evaluation.

---

## 🧠 Overview

This project implements a **full RAG pipeline** from scratch, covering:

- Document ingestion & preprocessing
- Three chunking strategies (recursive, semantic, agentic)
- Hybrid retrieval (dense vector + sparse BM25)
- Multi-query expansion + Reciprocal Rank Fusion
- Cross-encoder reranking + context compression
- LLM answer generation (Ollama, local)
- Streamlit UI with source attribution
- **Quantitative eval** comparing vector vs hybrid vs hybrid+rerank

---

## 📐 Architecture