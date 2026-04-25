from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

db = Chroma(
    persist_directory="db/chroma_db",
    embedding_function=OllamaEmbeddings(model="nomic-embed-text")
)

llm = ChatOllama(model="llama3")

query = "How does Tesla make money?"

print("Generating variations...")

prompt = f"Generate 3 variations of: {query}"
variations = llm.invoke(prompt).content.split("\n")

retriever = db.as_retriever(search_kwargs={"k": 5})

for v in variations:
    if v.strip():
        print(f"\nQuery: {v}")
        docs = retriever.invoke(v)
        for d in docs:
            print("-", d.page_content[:100])