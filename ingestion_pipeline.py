import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()


def load_documents(docs_path="docs"):
    """Load all text files safely (no encoding errors)"""
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"{docs_path} does not exist")

    documents = []

    for filename in os.listdir(docs_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(docs_path, filename)

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": file_path}
                    )
                )

                print(f"✅ Loaded: {filename}")

            except Exception as e:
                print(f"❌ Skipped {filename}: {e}")

    if not documents:
        raise ValueError("No documents loaded")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into chunks"""
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    print(f"✅ Created {len(chunks)} chunks")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and store embeddings"""
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

    print(f"✅ Vector store saved at {persist_directory}")
    return vectorstore

def main():
    print("=== RAG Document Ingestion Pipeline ===\n")

    docs_path = "docs"
    persistent_directory = "db/chroma_db"

    # If already exists → load
    if os.path.exists(persistent_directory):
        print("✅ Vector store already exists")

        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model
        )

        print(f"Loaded existing DB with {vectorstore._collection.count()} documents")
        return vectorstore

    print("Initializing new vector store...\n")

    # Step 1
    documents = load_documents(docs_path)

    # Step 2
    chunks = split_documents(documents)

    # Step 3
    vectorstore = create_vector_store(chunks, persistent_directory)

    print("\n✅ Ingestion complete! Ready for querying.")
    return vectorstore

if __name__ == "__main__":
    main()