# Repository Guidelines

## Project Structure & Module Organization
This repository implements a modular RAG (Retrieval-Augmented Generation) system for document processing and querying.

- **`docs/`**: Contains raw source text files used for ingestion.
- **`db/`**: Stores the persistent vector database (ChromaDB).
- **`ingestion_pipeline.py`**: Handles loading, chunking (using `RecursiveCharacterTextSplitter`), and embedding documents into the vector store.
- **`retrieval_pipeline.py`**: Implements the retrieval logic using vector similarity search.
- **`answer_generation.py`** & **`history_aware_generation.py`**: Manage the LLM interaction and context-aware response generation.
- **`multi_query_retrieval.py`** & **`reciprocal_rank_fusion.py`**: Provide advanced retrieval strategies for improved accuracy.

## Build, Test, and Development Commands
The project is built with Python and utilizes LangChain, Chroma, and Ollama. There is no formal build system; scripts are executed directly.

- **Ingest documents**: `python ingestion_pipeline.py`
- **Execute retrieval test**: `python retrieval_pipeline.py`
- **Run answer generation**: `python answer_generation.py`
- **Advanced retrieval**: `python multi_query_retrieval.py` or `python reciprocal_rank_fusion.py`

*Note: Ensure Ollama is running locally with the `nomic-embed-text` model pulled.*

## Coding Style & Naming Conventions
- **Naming**: Use snake_case for Python scripts and functions. Script names reflect their specific role in the RAG pipeline.
- **Dependencies**: LangChain community and core packages are used extensively.
- **Models**: Defaults to `nomic-embed-text` for embeddings and `Ollama` for chat models.

## Testing Guidelines
There is currently no automated test suite. Validation is performed by running the pipeline scripts and verifying the console output for retrieval accuracy and generation quality.

## Commit & Pull Request Guidelines
Commit messages should be concise and focus on the specific component updated.
- Use descriptive summaries (e.g., "Updated readme file", "Initial commit with proper gitignore").
- Follow standard git practices for merging and synchronization.
