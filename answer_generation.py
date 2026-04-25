from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
PERSIST_DIR = "db/chroma_db"

print("🚀 Starting Answer Generator")

db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=OllamaEmbeddings(model="nomic-embed-text")
)

llm = ChatOllama(model="llama3", temperature=0)

def generate_answer(query):
    print(f"\n🔎 Query: {query}")

    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)

    print(f"📄 Retrieved {len(docs)} docs")

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a RAG assistant.

Use ONLY the context below.

Context:
{context}

Question: {query}

If answer not found, say: Not found in documents.
"""

    response = llm.invoke(prompt)

    print("\n💡 Answer:\n")
    print(response.content)


if __name__ == "__main__":
    while True:
        q = input("\nAsk: ")
        if q.lower() == "quit":
            break
        generate_answer(q)