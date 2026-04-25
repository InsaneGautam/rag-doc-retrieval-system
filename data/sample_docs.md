# RAG System Overview
The Retrieval-Augmented Generation (RAG) system is designed to provide accurate answers by retrieving relevant documents before generating a response.

## Hybrid Retrieval
Hybrid retrieval combines keyword-based search (BM25) with semantic vector search. This ensures that both exact term matches and conceptually related content are found.

## Re-ranking
After initial retrieval, a Cross-Encoder model re-ranks the documents. Cross-encoders are more computationally expensive but significantly more accurate than bi-encoders for determining document relevance.

## Context Compression
Context compression reduces the token count by extracting only the most relevant sentences from the retrieved chunks. This helps stay within the LLM's context window and reduces noise.
