import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class DataIngestor:
    def __init__(self, docs_dir="data", index_path="faiss_index"):
        self.docs_dir = docs_dir
        self.index_path = index_path
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def load_documents(self):
        print(f"📂 Loading documents from {self.docs_dir}...")
        loaders = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
            ".md": TextLoader,
            ".py": TextLoader
        }
        
        all_docs = []
        for ext, loader_cls in loaders.items():
            loader = DirectoryLoader(self.docs_dir, glob=f"**/*{ext}", loader_cls=loader_cls)
            all_docs.extend(loader.load())
        
        print(f"✅ Loaded {len(all_docs)} documents.")
        return all_docs

    def ingest(self):
        docs = self.load_documents()
        if not docs:
            print("⚠️ No documents found to ingest.")
            return None

        print("✂️ Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(docs)
        print(f"✅ Created {len(chunks)} chunks.")

        print("🧠 Creating FAISS vector store...")
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        vector_store.save_local(self.index_path)
        print(f"✅ Vector store saved to {self.index_path}")
        
        return vector_store

if __name__ == "__main__":
    ingestor = DataIngestor()
    ingestor.ingest()
