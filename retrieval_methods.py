from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
db = Chroma(
    persist_directory="db/chroma_db",
    embedding_function=OllamaEmbeddings(model="nomic-embed-text")
)

query = "How much did Microsoft pay to acquire GitHub?"

print("\n=== BASIC RETRIEVAL ===")

retriever = db.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke(query)

for d in docs:
    print(d.page_content)