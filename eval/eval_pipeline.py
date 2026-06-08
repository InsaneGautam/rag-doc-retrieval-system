"""
RAG Pipeline Evaluation
Compares three retrieval strategies on a fixed test set:
  - Vector-only
  - Hybrid (BM25 + Vector)
  - Hybrid + Reranker

Metrics: Precision@K, MRR, Avg Rerank Score
"""

import sys
import os
import time
import json
from typing import List, Dict

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion import DataIngestor
from src.retriever import HybridRetriever
from src.reranker import Reranker

# ---------------------------------------------------------------------------
# Test dataset — ground truth (query → expected keywords in relevant chunks)
# Edit these to match your actual documents in data/
# ---------------------------------------------------------------------------
TEST_CASES: List[Dict] = [
    {
        "query": "How does reciprocal rank fusion work?",
        "relevant_keywords": ["reciprocal", "rank", "fusion", "rrf", "ranking"],
    },
    {
        "query": "What is hybrid search?",
        "relevant_keywords": ["hybrid", "bm25", "keyword", "semantic", "sparse", "dense"],
    },
    {
        "query": "Explain semantic chunking",
        "relevant_keywords": ["semantic", "chunk", "embedding", "split", "boundary"],
    },
    {
        "query": "How does agentic chunking work?",
        "relevant_keywords": ["agentic", "llm", "proposition", "chunk"],
    },
    {
        "query": "What is context compression in RAG?",
        "relevant_keywords": ["compress", "context", "relevant", "filter", "extraction"],
    },
    {
        "query": "How does the reranker improve retrieval?",
        "relevant_keywords": ["rerank", "cross-encoder", "score", "relevance", "top"],
    },
    {
        "query": "What embedding model is used?",
        "relevant_keywords": ["embedding", "sentence", "transformer", "encode", "model"],
    },
    {
        "query": "How is FAISS used for vector search?",
        "relevant_keywords": ["faiss", "vector", "index", "similarity", "search"],
    },
    {
        "query": "What is multi-query retrieval?",
        "relevant_keywords": ["multi", "query", "perspective", "recall", "expand"],
    },
    {
        "query": "How does the ingestion pipeline work?",
        "relevant_keywords": ["ingest", "load", "document", "chunk", "embed", "store"],
    },
]

K = 5  # Precision@K and MRR@K


def chunk_is_relevant(chunk_text: str, keywords: List[str]) -> bool:
    text_lower = chunk_text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def precision_at_k(docs, keywords: List[str], k: int) -> float:
    hits = sum(1 for doc in docs[:k] if chunk_is_relevant(doc.page_content, keywords))
    return hits / k


def mrr_at_k(docs, keywords: List[str], k: int) -> float:
    for rank, doc in enumerate(docs[:k], start=1):
        if chunk_is_relevant(doc.page_content, keywords):
            return 1.0 / rank
    return 0.0


def run_evaluation():
    print("Initializing pipeline components...")
    ingestor = DataIngestor()
    if not os.path.exists("faiss_index"):
        ingestor.ingest()

    hybrid_retriever = HybridRetriever()
    reranker = Reranker()

    results = []

    for tc in TEST_CASES:
        query = tc["query"]
        keywords = tc["relevant_keywords"]
        print(f"\nQuery: {query}")

        # --- Strategy 1: Vector-only (top-k from retriever without BM25) ---
        t0 = time.time()
        vector_retriever = hybrid_retriever.get_vector_retriever(k=K)
        vector_docs = vector_retriever.invoke(query)
        vector_latency = time.time() - t0

        # --- Strategy 2: Hybrid (BM25 + Vector via EnsembleRetriever) ---
        t0 = time.time()
        hybrid_ret = hybrid_retriever.get_retriever(k=10)
        hybrid_docs = hybrid_ret.invoke(query)
        hybrid_latency = time.time() - t0
        hybrid_docs_k = hybrid_docs[:K]

        # --- Strategy 3: Hybrid + Reranker ---
        t0 = time.time()
        reranked_docs = reranker.rerank(query, hybrid_docs, top_k=K)
        rerank_latency = time.time() - t0
        total_hybrid_rerank_latency = hybrid_latency + rerank_latency

        avg_rerank_score = 0.0
        if reranked_docs:
            scores = [
                doc.metadata.get("rerank_score", 0.0)
                for doc in reranked_docs
                if isinstance(doc.metadata.get("rerank_score"), float)
            ]
            avg_rerank_score = sum(scores) / len(scores) if scores else 0.0

        row = {
            "query": query,
            # Vector-only
            "vector_p@k": precision_at_k(vector_docs, keywords, K),
            "vector_mrr": mrr_at_k(vector_docs, keywords, K),
            "vector_latency_s": round(vector_latency, 3),
            # Hybrid
            "hybrid_p@k": precision_at_k(hybrid_docs_k, keywords, K),
            "hybrid_mrr": mrr_at_k(hybrid_docs_k, keywords, K),
            "hybrid_latency_s": round(hybrid_latency, 3),
            # Hybrid + Reranker
            "hybrid_rerank_p@k": precision_at_k(reranked_docs, keywords, K),
            "hybrid_rerank_mrr": mrr_at_k(reranked_docs, keywords, K),
            "hybrid_rerank_latency_s": round(total_hybrid_rerank_latency, 3),
            "avg_rerank_score": round(avg_rerank_score, 4),
        }

        results.append(row)
        print(
            f"  Vector   P@{K}={row['vector_p@k']:.2f}  MRR={row['vector_mrr']:.2f}  "
            f"lat={row['vector_latency_s']}s"
        )
        print(
            f"  Hybrid   P@{K}={row['hybrid_p@k']:.2f}  MRR={row['hybrid_mrr']:.2f}  "
            f"lat={row['hybrid_latency_s']}s"
        )
        print(
            f"  H+Rerank P@{K}={row['hybrid_rerank_p@k']:.2f}  MRR={row['hybrid_rerank_mrr']:.2f}  "
            f"lat={row['hybrid_rerank_latency_s']}s  avg_score={row['avg_rerank_score']}"
        )

    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    summary = {
        "Strategy": ["Vector-only", "Hybrid", "Hybrid + Reranker"],
        f"Avg P@{K}": [
            df["vector_p@k"].mean(),
            df["hybrid_p@k"].mean(),
            df["hybrid_rerank_p@k"].mean(),
        ],
        f"Avg MRR@{K}": [
            df["vector_mrr"].mean(),
            df["hybrid_mrr"].mean(),
            df["hybrid_rerank_mrr"].mean(),
        ],
        "Avg Latency (s)": [
            df["vector_latency_s"].mean(),
            df["hybrid_latency_s"].mean(),
            df["hybrid_rerank_latency_s"].mean(),
        ],
    }

    summary_df = pd.DataFrame(summary).set_index("Strategy")
    print(summary_df.to_string(float_format="{:.3f}".format))

    # Save results
    os.makedirs("eval", exist_ok=True)
    df.to_csv("eval/eval_results.csv", index=False)
    summary_df.to_csv("eval/eval_summary.csv")
    print("\nResults saved to eval/eval_results.csv and eval/eval_summary.csv")

    return df, summary_df


if __name__ == "__main__":
    run_evaluation()