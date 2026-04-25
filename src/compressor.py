import re
from sentence_transformers import SentenceTransformer, util
import torch

class ContextCompressor:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", threshold=0.4):
        print(f"🗜️ Loading Compressor model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def compress(self, query, documents):
        if not documents:
            return []
            
        compressed_docs = []
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        for doc in documents:
            # Split document into sentences
            sentences = re.split(r'(?<=[.!?])\s+', doc.page_content)
            if not sentences:
                continue
                
            sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)
            
            # Calculate cosine similarity
            cos_scores = util.cos_sim(query_embedding, sentence_embeddings)[0]
            
            # Select sentences above threshold
            relevant_sentences = [
                sentences[i] for i in range(len(sentences)) 
                if cos_scores[i] >= self.threshold
            ]
            
            if relevant_sentences:
                new_content = " ".join(relevant_sentences)
                doc.page_content = new_content
                compressed_docs.append(doc)
                
        print(f"✅ Compressed {len(documents)} documents.")
        return compressed_docs

if __name__ == "__main__":
    # Test compressor
    from langchain.docstore.document import Document
    compressor = ContextCompressor(threshold=0.3)
    query = "FAISS storage"
    docs = [
        Document(page_content="FAISS is a library for efficient similarity search. It can store vector embeddings on disk. I like eating apples in the morning. Vector databases are cool.")
    ]
    results = compressor.compress(query, docs)
    for res in results:
        print(f"Compressed Content: {res.page_content}")
        # Expected to see only FAISS/Vector related sentences, not the apple sentence.
