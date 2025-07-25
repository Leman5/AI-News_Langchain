"""Streamlit application for daily Australian news highlights.

This app ties together the news fetching, processing and retrieval
components into a simple web interface.  When the app loads it
automatically fetches recent news articles from Australian sources via
Event Registry, classifies and summarises them using HuggingFace
models, clusters duplicates and extracts the most important stories
per category.  The highlights are displayed in expandable panels and
a chat interface allows users to ask questions about the daily news.

The chatbot uses Retrieval Augmented Generation (RAG):  we embed
each article summary into a vector space using a sentence transformer
and build a FAISS index over these embeddings.  When the user asks
a question the top‑k most relevant summaries are retrieved and fed
into an LLM hosted on HuggingFace's inference API.  The answer
generated by the LLM is then displayed back to the user.

Because API calls to external services can be slow, we cache the
expensive operations using Streamlit's caching facilities.  The
``@st.cache_data`` decorator ensures that articles are only fetched
once per session and the heavy processing steps are not repeated
unnecessarily.

Before running this app you must set two environment variables or
define them inline below:

``NEWSAPI_KEY`` – your NewsAPI.ai/Event Registry API key.
``HF_API_TOKEN`` – your HuggingFace Inference API token.

Alternatively you can edit the variables near the top of this file.
"""

from __future__ import annotations

import datetime
import os
from typing import List

import streamlit as st
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain import HuggingFaceHub

# Import helper classes from within the package.  When running via
# ``streamlit run news_highlights_app/app.py`` the package root is on
# Python's module search path, so these imports resolve correctly.
from news_fetcher import NewsFetcher
from article_processor import ArticleProcessor, ProcessedArticle, Highlight


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Retrieve API keys from environment variables or set them here directly.
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "7d16fb57-b8ef-4371-b6eb-522a4540048b")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "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")

# Date window for fetching articles (past day)
TODAY = datetime.date.today()
START_DATE = TODAY - datetime.timedelta(days=1)


# ---------------------------------------------------------------------------
# Streamlit helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=True)
def load_and_process() -> (List[Highlight], List[str], FAISS):
    """Fetch, process and index news articles.

    This function is cached so that the expensive work (network calls,
    classification, summarisation and embedding) only happens once per
    session.  It returns the highlights, a list of all processed
    summaries and the FAISS vector store used for retrieval.
    """
    fetcher = NewsFetcher(api_key=NEWSAPI_KEY)
    # fetch articles from Australian sources within the past day
    # restrict to English and skip duplicates
    try:
        articles = fetcher.fetch_articles(
            start_date=START_DATE,
            end_date=TODAY,
            page=1,
            count=100,
            source_location_uri="http://en.wikipedia.org/wiki/Australia",
            language="eng",
            duplicate_filter="skipDuplicates",
            include_pr_and_blogs=False,
        )
    except Exception as e:
        st.error(f"Error fetching articles: {e}")
        return [], [], None
    processor = ArticleProcessor(hf_token=HF_API_TOKEN)
    # process raw articles
    processed = processor.process_articles(articles)
    # cluster to group duplicates
    clusters = processor.cluster_articles(processed)
    highlights = processor.build_highlights(clusters, top_k=5)
    # prepare retrieval index – use article summaries as the content
    docs = []
    metadata = []
    for art in processed:
        docs.append(art.summary)
        metadata.append({"title": art.title, "source": art.source, "category": art.category})
    # embed and index
    # Note: we instantiate a new embedding model here; FAISS
    # automatically persists the embeddings in memory
    embedding_model = processor.embedding_model
    vector_store = FAISS.from_texts(docs, embedding_model, metadatas=metadata)
    return highlights, docs, vector_store


def build_qa_chain(vector_store: FAISS) -> RetrievalQA:
    """Create a Retrieval‑augmented QA chain using HuggingFaceHub LLM.

    We use the ``google/flan-t5-large`` model hosted on HuggingFace Hub
    because it provides a good trade‑off between quality and speed.
    The retriever returns the top 5 most relevant summaries which are
    concatenated and passed to the LLM alongside the user question.
    """
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.2, "max_length": 256},
        huggingfacehub_api_token=HF_API_TOKEN,
    )
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
    )
    return qa_chain


def main() -> None:
    st.set_page_config(page_title="Australian News Highlights", page_icon="📰", layout="wide")
    st.title("🇦🇺 Daily Australian News Highlights")
    st.markdown(
        """
        This dashboard automatically gathers recent articles from major
        Australian news outlets, groups duplicate stories, summarises
        them and surfaces the top highlights across sports, lifestyle,
        music and finance categories.  You can expand any section
        below to read a short summary and see which outlets covered
        the story.  At the bottom there is a chat box – ask any
        question about today's news and the assistant will answer
        using the retrieved summaries.
        """
    )
    with st.spinner("Fetching and processing news articles…"):
        highlights, docs, vector_store = load_and_process()
    if not highlights:
        st.warning("No highlights available. Please check your API keys and try again.")
        return
    # Display highlights grouped by category
    categories = sorted({hl.category for hl in highlights})
    for cat in categories:
        st.subheader(cat.capitalize())
        cat_highs = [hl for hl in highlights if hl.category == cat]
        for idx, hl in enumerate(cat_highs, start=1):
            with st.expander(f"{idx}. {hl.title} ({hl.frequency} sources)"):
                st.write(hl.summary)
                st.write(f"**Sources:** {', '.join(hl.sources)}")
    st.divider()
    # Chatbot section
    st.header("Ask the assistant about today's news")
    if vector_store is None:
        st.warning("The retrieval index is unavailable.")
        return
    qa_chain = build_qa_chain(vector_store)
    # Initialise session state to store chat history across reruns
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # Accept user input
    query = st.chat_input("Ask a question about the news…")
    if query:
        # Append user query to history
        st.session_state.messages.append({"role": "user", "content": query})
        # Generate answer via QA chain
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    response = qa_chain.run(query)
                except Exception as e:
                    response = f"Sorry, an error occurred: {e}"
                st.markdown(response)
                # Append assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()