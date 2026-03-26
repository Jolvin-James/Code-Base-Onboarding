import streamlit as st

from ingestion.loader import load_documents
from processing.chunker import MarkdownChunker
from processing.embedder import EmbeddingGenerator
from retrieval.faiss_index import FAISSIndex
from retrieval.query_processor import QueryProcessor
from retrieval.context_builder import ContextBuilder
from main import build_prompt, call_llm, check_stale_documents


st.set_page_config(page_title="Codebase Assistant", layout="wide")
st.title("Codebase Onboarding Assistant")


@st.cache_resource
def initialize():
    documents = load_documents("docs")

    chunker = MarkdownChunker()
    chunks = chunker.chunk_documents(documents)

    embedder = EmbeddingGenerator()
    embeddings, _ = embedder.embed_chunks(chunks)

    index = FAISSIndex(embeddings.shape[1])
    index.add_embeddings(embeddings, chunks)

    qp = QueryProcessor(embedder, index)
    cb = ContextBuilder()

    return qp, cb


query_processor, context_builder = initialize()

query = st.text_input("Ask a question about your codebase:")

if query:
    results = query_processor.process_query(query)

    if not results:
        st.error("No relevant results found.")
    else:
        context = context_builder.build_context(results)

        system_prompt, user_prompt = build_prompt(context, query)
        response = call_llm(system_prompt, user_prompt)

        st.subheader("Answer")
        st.write(response)

        # Confidence
        if results[0]["similarity"] < 0.2:
            st.warning("Low confidence result")

        # Stale warning
        warnings = check_stale_documents(results)
        for w in warnings:
            st.warning(
                f"{w['source']} outdated ({w['age_days']} days old)"
            )

        # Sources
        st.subheader("Sources")
        for r in results:
            st.caption(f"{r['source']} — {r['header']}")

        # Debug
        with st.expander("Context"):
            st.text(context)