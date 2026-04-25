import streamlit as st
from main import RAGPipeline
import os

st.set_page_config(page_title="Advanced RAG Assistant", layout="wide")

st.title("🚀 Advanced RAG Documentation Assistant")
st.markdown("""
This system uses **Hybrid Retrieval**, **Re-ranking**, and **Context Compression** to provide grounded answers.
""")

@st.cache_resource
def get_pipeline():
    pipeline = RAGPipeline()
    pipeline.initialize()
    return pipeline

try:
    pipeline = get_pipeline()
    
    with st.sidebar:
        st.header("Settings")
        if st.button("Re-ingest Documents"):
            with st.spinner("Ingesting..."):
                pipeline.initialize(force_ingest=True)
                st.success("Ingestion complete!")

    query = st.text_input("Ask a question about the documentation:", placeholder="e.g., How does the reranker work?")

    if query:
        with st.spinner("Thinking..."):
            result = pipeline.run(query)
            
            st.subheader("Answer")
            st.write(result["answer"])
            
            with st.expander("View Source Contexts"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Source {i+1}:** `{doc.metadata.get('source', 'N/A')}` (Score: {doc.metadata.get('rerank_score', 'N/A'):.4f})")
                    st.text_area(f"Content {i+1}", doc.page_content, height=150)
                    
except Exception as e:
    st.error(f"Error initializing pipeline: {e}")
    st.info("Make sure you have documents in the 'data' folder and Ollama is running.")
