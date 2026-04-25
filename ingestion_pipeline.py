import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama 
DOCS_PATH = "docs"
PERSIST_DIR = "db/chroma_db"

def load_documents():
    print("Loading documents...")

    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError("Docs folder not found")

    loader = DirectoryLoader(DOCS_PATH, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    print(f"Loaded {len(documents)} documents")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDoc {i+1} preview:")
        print(doc.page_content[:150])

    return documents


def split_documents(documents):
    print("\nSplitting documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    return chunks


def create_vector_store(chunks):
    print("\nCreating embeddings (Ollama)...")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    print("Vector DB created")
    return db


def main():
    print("INGESTION START")

    if os.path.exists(PERSIST_DIR):
        print("DB already exists, loading...")

        db = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=OllamaEmbeddings(model="nomic-embed-text")
        )

        print(f"Loaded {db._collection.count()} docs")
        return db

    docs = load_documents()
    chunks = split_documents(docs)
    create_vector_store(chunks)

    print("DONE")


if __name__ == "__main__":
    main()