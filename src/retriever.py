from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import os

class HybridRetriever:
    def __init__(self, index_path="faiss_index"):
        self.index_path = index_path
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index path {index_path} not found. Please run ingestion first.")
            
        self.vector_store = FAISS.load_local(
            self.index_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )

    def get_retriever(self, k=5):
        # 1. Vector Retriever
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        
        # 2. BM25 Retriever
        # We need all documents from the vector store to initialize BM25
        # This is a bit hacky but works for local FAISS
        all_docs = []
        for i in range(self.vector_store.index.ntotal):
            doc = self.vector_store.docstore.search(self.vector_store.index_to_docstore_id[i])
            all_docs.append(doc)
            
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = k
        
        # 3. Ensemble (Hybrid) Retriever
        # Weighting: 0.5 Vector, 0.5 BM25
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )
        
        return ensemble_retriever

if __name__ == "__main__":
    # Test retrieval
    try:
        retriever_obj = HybridRetriever()
        retriever = retriever_obj.get_retriever()
        query = "What is the ingestion pipeline?"
        results = retriever.invoke(query)
        for i, res in enumerate(results):
            print(f"Result {i+1}: {res.page_content[:100]}...")
    except Exception as e:
        print(f"Error: {e}")
