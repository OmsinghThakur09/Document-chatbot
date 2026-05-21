# src/app.py
"""
Streamlit UI for Document Q&A chatbot.
Run with: streamlit run src/app.py
"""

import streamlit as st
import sys
import os

# including path explicitly so that our app won't crash due to 'file not found' error.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))  # python will look here first before anywhere else.

from ingest import ingest
from retriever import retrieve
from chatbot import ask_with_context

# --page config-- (must be first streamlit call).
st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="wide",
)

# session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []  # full chat history. dict({role, content})

if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False  # Ture if a PDF has been successfully loaded False otherwise.

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""  # name of loaded PDF.

if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []  # chunks used by LLM to answer

if "is_thinking" not in st.session_state:
    st.session_state.is_thinking = False      # True while LLM is generating

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None  # holds the question being processed

# sidebar
with st.sidebar:
    st.title("📄 Document Q&A")
    st.markdown("---")

    # pdf upload widget
    uploaded_file = st.file_uploader(
        label="Upload a pdf",
        type=["pdf"],
        help="Upload any pdf document to chat with it"
    )

    if uploaded_file is not None:
        # Only re-ingest if it's a different file from what's already loaded
        if uploaded_file.name != st.session_state.pdf_name:
            # Save uploaded file to disk
            save_path = os.path.join(ROOT_DIR, "data", uploaded_file.name)
            os.makedirs(os.path.join(ROOT_DIR, "data"), exist_ok=True)

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Ingesting PDF... this may take a moment"):
                ingest(save_path, clear_first=True)

            st.session_state.pdf_loaded = True
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.messages = []
            st.session_state.last_chunks = []
            st.success(f"✅ Loaded: {uploaded_file.name}")

    st.markdown("---")
    st.subheader("⚙️ Settings")

    temperature = st.slider(
        label="Temperature",
        min_value=0.0,
        max_value=1.5,
        value=0.7,
        step=0.1,
        help="Higher = more creative. Lower = more factual."
    )

    n_results = st.slider(
        label="Chunks to retrieve",
        min_value=1,
        max_value=6,
        value=3,
        step=1,
        help="How many document sections to use as context"
    )

    # clear chat button
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.last_chunks = []
        st.rerun()

# main area
st.title("💬 Chat with your Document")

if st.session_state.pdf_loaded:
    st.caption(f"Currently loaded: **{st.session_state.pdf_name}**")
else:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()


def render_sources(chunks):
    """
    Render a collapsible sources panel below an assistant answer.
    :param chunks: list of dicts returned by retriever.retrieve()
    :return: None
    """
    if not chunks:
        return

    with st.expander("📚 Sources used for last answer", expanded=False):
        for i, chunk in enumerate(chunks, 1):
            chunk_type = chunk.get("type", "text")
            label = "Table" if chunk_type == "table" else "Text"
            with st.expander(f"Chunk {i} — {label}  — Page {chunk['page']} "
                             f"from {chunk['source']}"
                             f"*(similarity distance: {chunk['distance']:.3f})*"):
                st.caption(chunk['text'][:300] + "...")
                st.markdown("---")


# Render full chat history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    if message["role"] == "assistant":
        render_sources(message.get("chunks", []))

# Chat input - renders at bottom of page
question = st.chat_input("Ask anything about your document...",
                         disabled=st.session_state.is_thinking
                         )

# Block A — capture new question (disabled while thinking)
if question and not st.session_state.is_thinking:
    # store question, flip flag, rerun so input renders as disabled
    st.session_state.pending_question = question
    st.session_state.is_thinking = True
    st.rerun()

# Block B — process pending question (runs on the rerun after Block A)
if st.session_state.is_thinking and st.session_state.pending_question:
    q = st.session_state.pending_question

    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chunks = retrieve(q, n_results=n_results)
            answer = ask_with_context(q, chunks, temperature=temperature)
        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chunks": chunks,
    })

    render_sources(chunks)

    # Clear pending state and re-enable input
    st.session_state.pending_question = None
    st.session_state.is_thinking = False
    st.rerun()