from src.ingestion import DataIngestor
from src.retriever import HybridRetriever
from src.reranker import Reranker
from src.compressor import ContextCompressor
from src.generator import AnswerGenerator
import os

class RAGPipeline:
    def __init__(self):
        self.ingestor = DataIngestor()
        self.retriever_factory = None
        self.reranker = Reranker()
        self.compressor = ContextCompressor()
        self.generator = AnswerGenerator()

    def initialize(self, force_ingest=False):
        if force_ingest or not os.path.exists("faiss_index"):
            self.ingestor.ingest()
        self.retriever_factory = HybridRetriever()

    def run(self, query):
        print(f"\n--- Processing Query: {query} ---")
        
        # 1. Retrieval (Hybrid)
        retriever = self.retriever_factory.get_retriever(k=10)
        retrieved_docs = retriever.invoke(query)
        
        # 2. Re-ranking
        ranked_docs = self.reranker.rerank(query, retrieved_docs, top_k=5)
        
        # 3. Context Compression
        compressed_docs = self.compressor.compress(query, ranked_docs)
        
        # 4. Answer Generation
        answer = self.generator.generate(query, compressed_docs)
        
        return {
            "answer": answer,
            "source_documents": compressed_docs
        }

if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.initialize()
    
    query = "How do I implement a hybrid retrieval system?"
    result = pipeline.run(query)
    
    print("\nFINAL ANSWER:")
    print(result["answer"])
    
    print("\nSOURCES USED:")
    for i, doc in enumerate(result["source_documents"]):
        print(f"[{i+1}] {doc.metadata.get('source', 'Unknown')}")
