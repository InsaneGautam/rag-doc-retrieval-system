import os
import streamlit as st
from dotenv import load_dotenv
from main import RAGPipeline

load_dotenv()

st.set_page_config(page_title="Advanced RAG Assistant", layout="wide")

st.title("🚀 Advanced RAG Documentation Assistant")
st.markdown(
    "Uses **Hybrid Retrieval** (BM25 + Vector), **Cross-encoder Reranking**, "
    "and **Context Compression** to deliver grounded, source-attributed answers."
)

# Show tracing status in sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    tracing_on = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    st.info(f"LangSmith tracing: {'✅ ON' if tracing_on else '⬜ OFF'}")

    if st.button("🔄 Re-ingest Documents"):
        with st.spinner("Ingesting documents..."):
            st.session_state.pipeline.initialize(force_ingest=True)
        st.success("Ingestion complete!")

    st.markdown("---")
    st.markdown("**Pipeline steps:**")
    st.markdown("1. Hybrid Retrieval (k=10)")
    st.markdown("2. Cross-encoder Reranking (top 5)")
    st.markdown("3. Context Compression")
    st.markdown("4. LLM Generation (Ollama)")


@st.cache_resource
def load_pipeline():
    pipeline = RAGPipeline()
    pipeline.initialize()
    return pipeline


try:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = load_pipeline()

    query = st.text_input(
        "Ask a question about the documentation:",
        placeholder="e.g., How does reciprocal rank fusion work?",
    )

    if query:
        with st.spinner("Retrieving, reranking, and generating..."):
            result = st.session_state.pipeline.run(query)

        st.subheader("📝 Answer")
        st.write(result["answer"])

        with st.expander(f"📚 View {len(result['source_documents'])} Source Chunks"):
            for i, doc in enumerate(result["source_documents"]):
                score = doc.metadata.get("rerank_score", None)
                source = doc.metadata.get("source", "N/A")
                score_str = f"{score:.4f}" if isinstance(score, float) else "N/A"
                st.markdown(f"**Chunk {i+1}** | Source: `{source}` | Rerank score: `{score_str}`")
                st.text_area(f"content_{i}", doc.page_content, height=120, label_visibility="collapsed")
                st.divider()

except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.info("Make sure: (1) documents exist in `data/`, (2) Ollama is running locally.")