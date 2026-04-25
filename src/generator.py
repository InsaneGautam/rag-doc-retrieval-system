from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class AnswerGenerator:
    def __init__(self, model_name="llama3"):
        print(f"🤖 Initializing Generator with model: {model_name}...")
        self.llm = ChatOllama(model=model_name, temperature=0)
        self.prompt = ChatPromptTemplate.from_template("""
        You are a helpful and professional AI Assistant. 
        Answer the following question using ONLY the provided context.
        If the answer is not contained within the context, strictly say: "I don't know based on the provided documentation."
        Do not use external knowledge. Keep the answer concise and grounded.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate(self, query, documents):
        if not documents:
            return "I don't know based on the provided documentation (no relevant documents found)."
            
        context_text = "\n\n".join([doc.page_content for doc in documents])
        
        response = self.chain.invoke({
            "context": context_text,
            "question": query
        })
        
        return response

if __name__ == "__main__":
    # Test generator
    from langchain.docstore.document import Document
    generator = AnswerGenerator()
    query = "How to ingest data?"
    docs = [Document(page_content="To ingest data, use the DataIngestor class which loads and chunks documents.")]
    print(generator.generate(query, docs))
