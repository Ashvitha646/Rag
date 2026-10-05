import sys
import os
from pathlib import Path

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "src"
    )
)

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

from chunk_papers import load_all_papers, split_documents
from hybrid_retrieval import create_bm25_index, hybrid_retrieve



load_dotenv()

BGE_CHROMA_FOLDER = "results/chroma_db_bge"



st.set_page_config(
    page_title="Research Paper Answer Bot",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown(
    """
    <style>

    .stApp {
        background-color: #f5f7fb;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    .dashboard-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 35px;
        border-radius: 20px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0px 8px 25px rgba(102, 126, 234, 0.25);
    }

    .dashboard-header h1 {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .dashboard-header p {
        font-size: 16px;
        margin: 0;
        opacity: 0.92;
    }

    .stat-card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        text-align: center;
        border: 1px solid #e5e7eb;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.05);
        min-height: 110px;
    }

    .stat-number {
        font-size: 28px;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 6px;
    }

    .stat-label {
        font-size: 14px;
        color: #6b7280;
    }

    .section-title {
        font-size: 22px;
        font-weight: 600;
        color: #1f2937;
        margin-top: 28px;
        margin-bottom: 12px;
    }

    .answer-box {
        background: white;
        padding: 25px;
        border-radius: 16px;
        border-left: 5px solid #667eea;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.05);
        line-height: 1.8;
        font-size: 16px;
        color: #374151;
    }

    .stTextInput input {
        border-radius: 12px;
        border: 1px solid #d1d5db;
        padding: 14px;
        font-size: 16px;
    }

    .stButton > button {
        border-radius: 12px;
        min-height: 45px;
        font-weight: 600;
        border: 1px solid #d1d5db;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
    }

    .dashboard-footer {
        text-align: center;
        color: #9ca3af;
        font-size: 13px;
        margin-top: 40px;
        padding: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)



@st.cache_resource
def load_vector_database():

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma(
        persist_directory=BGE_CHROMA_FOLDER,
        collection_name="research_papers_bge",
        embedding_function=embeddings
    )

    return vectorstore



@st.cache_resource
def load_research_data():

    documents = load_all_papers()

    chunks = split_documents(documents)

    bm25 = create_bm25_index(chunks)

    return chunks, bm25



@st.cache_resource
def load_reranker():

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )



def rerank_documents(question, documents, top_k=3):

    reranker = load_reranker()

    pairs = [
        [question, document.page_content]
        for document in documents
    ]

    scores = reranker.predict(pairs)

    ranked_results = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_results[:top_k]



def format_context(reranked_results):

    context_parts = []

    for i, (document, score) in enumerate(
        reranked_results,
        start=1
    ):

        source = Path(
            document.metadata.get("source", "Unknown")
        ).stem

        page = document.metadata.get("page", 0) + 1

        context_parts.append(
            f"[Source {i}: {source}, Page {page}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(context_parts)


def answer_question(question):

    chunks, bm25 = load_research_data()

    vectorstore = load_vector_database()

    candidate_documents = hybrid_retrieve(
        question=question,
        vectorstore=vectorstore,
        bm25=bm25,
        documents=chunks,
        k=10
    )

    reranked_results = rerank_documents(
        question=question,
        documents=candidate_documents,
        top_k=3
    )

    context = format_context(reranked_results)

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    prompt = f"""
You are a research paper question-answering assistant.

Answer the user's question using only the provided research paper context.

Rules:
1. Give a clear and simple explanation.
2. Do not invent information.
3. If the answer is not available in the context, say:
   "I could not find enough information in the indexed research papers."
4. Refer to sources using [Source 1], [Source 2], or [Source 3] only when supported by the context.
5. Do not create fake citations or references.
6. Keep the answer concise but informative.

User Question:
{question}

Research Paper Context:
{context}
"""

    response = llm.invoke(prompt)

    return response.content, reranked_results


with st.sidebar:

    st.markdown("## Research Bot")

    st.markdown("---")

    st.markdown("### About")

    st.write(
        "An AI-powered research assistant that answers questions "
        "from indexed research papers."
    )

    st.markdown("---")

    st.markdown("### System Pipeline")

    st.markdown("""
    PDF Processing

    Text Chunking

    BGE Embeddings

    ChromaDB

    BM25 Hybrid Search

    CrossEncoder Reranking

    Groq LLM
    """)

    st.markdown("---")

    st.markdown("### Indexed Papers")

    st.markdown("""
    - Attention Is All You Need
    - BERT
    - GPT-3
    - LoRA
    - RAG
    """)

    st.markdown("---")

    st.caption("Research Paper Answer Bot")
    st.caption("Powered by RAG + AI")


st.markdown(
    """
    <div class="dashboard-header">
        <h1>Research Paper Answer Bot</h1>
        <p>
            Intelligent question answering powered by Retrieval-Augmented Generation.
            Explore research papers and get AI-powered answers with source references.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-number">5</div>
            <div class="stat-label">Research Papers</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-number">654</div>
            <div class="stat-label">Text Chunks</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-number">2</div>
            <div class="stat-label">Embedding Models</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:

    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-number">AI</div>
            <div class="stat-label">Groq Powered</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    '<div class="section-title">Ask Your Research Question</div>',
    unsafe_allow_html=True
)

question = st.text_input(
    "Research Question",
    placeholder="Example: What is the purpose of LoRA?",
    label_visibility="collapsed"
)


col1, col2, col3 = st.columns([1, 1, 3])

with col1:

    ask_button = st.button(
        "Ask Question",
        type="primary",
        use_container_width=True
    )

with col2:

    clear_button = st.button(
        "Clear",
        use_container_width=True
    )


if clear_button:

    st.rerun()




st.markdown(
    '<div class="section-title">Sample Questions</div>',
    unsafe_allow_html=True
)

sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)


with sample_col1:

    if st.button(
        "What is LoRA?",
        use_container_width=True
    ):

        question = "What is LoRA?"
        ask_button = True


with sample_col2:

    if st.button(
        "Transformer Architecture",
        use_container_width=True
    ):

        question = "What is the main idea of the Transformer architecture?"
        ask_button = True


with sample_col3:

    if st.button(
        "How does BERT work?",
        use_container_width=True
    ):

        question = "How does BERT work?"
        ask_button = True


with sample_col4:

    if st.button(
        "What is RAG?",
        use_container_width=True
    ):

        question = "What is Retrieval-Augmented Generation?"
        ask_button = True


if ask_button:

    if question.strip():

        with st.spinner(
            "Searching papers, reranking results, and generating answer..."
        ):

            try:

                answer, sources = answer_question(
                    question.strip()
                )

                st.markdown(
                    '<div class="section-title">AI Generated Answer</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="answer-box">
                        {answer}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="section-title">Top 3 Relevant Sources</div>',
                    unsafe_allow_html=True
                )

                for i, (document, score) in enumerate(
                    sources,
                    start=1
                ):

                    source = Path(
                        document.metadata.get(
                            "source",
                            "Unknown"
                        )
                    ).stem

                    page = document.metadata.get(
                        "page",
                        0
                    ) + 1

                    with st.expander(
                        f"Source {i}: {source} — Page {page}"
                    ):

                        st.write(
                            document.page_content
                        )

                        st.caption(
                            f"CrossEncoder Reranker Score: {score:.4f}"
                        )

            except Exception as e:

                st.error(
                    f"An error occurred: {str(e)}"
                )

    else:

        st.warning("Please enter a research question.")



st.markdown(
    """
    <div class="dashboard-footer">
        Research Paper Answer Bot • Retrieval-Augmented Generation • AI-Powered Research Assistant
    </div>
    """,
    unsafe_allow_html=True
)