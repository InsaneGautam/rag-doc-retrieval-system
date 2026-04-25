from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
db = Chroma(
    persist_directory="db/chroma_db",
    embedding_function=OllamaEmbeddings(model="nomic-embed-text")
)

# Keep your RRF logic SAME