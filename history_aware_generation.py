from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

PERSIST_DIR = "db/chroma_db"

db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=OllamaEmbeddings(model="nomic-embed-text")
)

llm = ChatOllama(model="llama3")

chat_history = []

def ask(user_query):
    print(f"\n🧑 User: {user_query}")

    if chat_history:
        rewrite_prompt = [
            SystemMessage(content="Rewrite into standalone query"),
            *chat_history,
            HumanMessage(content=user_query)
        ]
        standalone_query = llm.invoke(rewrite_prompt).content
    else:
        standalone_query = user_query

    print(f"🔍 Searching: {standalone_query}")

    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(standalone_query)

    context = "\n".join([d.page_content for d in docs])

    final_prompt = [
        SystemMessage(content="Answer only from context"),
        *chat_history,
        HumanMessage(content=f"Context:\n{context}\n\nQ:{user_query}")
    ]

    response = llm.invoke(final_prompt)

    print("\n🤖 Answer:\n", response.content)

    chat_history.append(HumanMessage(content=user_query))
    chat_history.append(AIMessage(content=response.content))


while True:
    q = input("\nAsk: ")
    if q == "quit":
        break
    ask(q)