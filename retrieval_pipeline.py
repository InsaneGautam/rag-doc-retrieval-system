from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
PERSIST_DIR = "db/chroma_db"

print("🔍 Loading DB...")

db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=OllamaEmbeddings(model="nomic-embed-text")
)

query = "How much did Microsoft pay to acquire GitHub?"
print(f"\nQuery: {query}")

retriever = db.as_retriever(search_kwargs={"k": 3})

docs = retriever.invoke(query)

print(f"\nRetrieved {len(docs)} docs:\n")

for i, doc in enumerate(docs, 1):
    print(f"Doc {i}:")
    print(doc.page_content)
    print("-" * 50)