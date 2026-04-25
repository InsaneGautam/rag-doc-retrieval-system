from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
import pandas as pd

class RagasEvaluator:
    def __init__(self, model_name="llama3"):
        print(f"📊 Initializing Evaluator with model: {model_name}...")
        self.llm = ChatOllama(model=model_name)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
    def evaluate_rag(self, query, response, retrieved_docs, ground_truth=None):
        # Prepare data for Ragas
        data = {
            "question": [query],
            "answer": [response],
            "contexts": [[doc.page_content for doc in retrieved_docs]],
        }
        
        if ground_truth:
            data["ground_truth"] = [ground_truth]
            
        dataset = Dataset.from_dict(data)
        
        # In newer Ragas, we pass llm and embeddings to evaluate
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=self.llm,
            embeddings=self.embeddings
        )
        
        return result.to_pandas()

if __name__ == "__main__":
    # Mock data for testing
    evaluator = RagasEvaluator()
    query = "How does RAG work?"
    response = "RAG works by retrieving documents and then generating an answer."
    from langchain.docstore.document import Document
    docs = [Document(page_content="RAG combines retrieval and generation.")]
    
    # This might take time if Ollama is slow
    try:
        results = evaluator.evaluate_rag(query, response, docs)
        print(results)
    except Exception as e:
        print(f"Skipping evaluation test due to: {e}")
