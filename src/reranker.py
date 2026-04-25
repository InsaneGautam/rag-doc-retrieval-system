from sentence_transformers import CrossEncoder
import numpy as np

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"🔄 Loading Re-ranker model: {model_name}...")
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_k=3):
        if not documents:
            return []
            
        # Prepare pairs for the cross-encoder
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        # Sort documents by scores in descending order
        ranked_indices = np.argsort(scores)[::-1]
        
        ranked_docs = []
        for idx in ranked_indices[:top_k]:
            doc = documents[idx]
            doc.metadata["rerank_score"] = float(scores[idx])
            ranked_docs.append(doc)
            
        print(f"✅ Re-ranked {len(documents)} docs, returning top {top_k}.")
        return ranked_docs

if __name__ == "__main__":
    # Test reranker
    from langchain.docstore.document import Document
    reranker = Reranker()
    query = "How to split documents?"
    docs = [
        Document(page_content="You can use RecursiveCharacterTextSplitter for splitting."),
        Document(page_content="Cooking pasta requires boiling water."),
        Document(page_content="LangChain provides tools for ingestion.")
    ]
    results = reranker.rerank(query, docs)
    for i, res in enumerate(results):
        print(f"Rank {i+1} (Score: {res.metadata['rerank_score']:.4f}): {res.page_content}")
