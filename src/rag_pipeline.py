import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables from .env
load_dotenv()


CHROMA_FOLDER = "results/chroma_db"
COLLECTION_NAME = "research_papers"


def load_vector_database():
    """
    Load the existing Chroma vector database.
    """

    print("Loading Hugging Face embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Connecting to ChromaDB...")

    vectorstore = Chroma(
        persist_directory=CHROMA_FOLDER,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    print("Vector database loaded successfully.")

    return vectorstore


def format_source(document):
    """
    Convert document metadata into a readable source citation.
    """

    source_path = document.metadata.get("source", "Unknown source")
    page_number = document.metadata.get("page", 0) + 1

    paper_title = Path(source_path).stem.replace("_", " ").title()

    return f"{paper_title} — Page {page_number}"


def answer_question(question):
    """
    Retrieve relevant research paper chunks and generate an answer using Groq.
    """

    vectorstore = load_vector_database()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    print("\nSearching research papers...")

    documents = retriever.invoke(question)

    context_parts = []

    for document in documents:
        source = format_source(document)

        context_parts.append(
            f"Source: {source}\n"
            f"Content:\n{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_template(
        """
You are a research paper assistant.

Answer the user's question using only the provided research paper context.

Instructions:
- Give a clear and simple explanation.
- Do not invent information.
- If the context does not contain enough information, say so.
- Mention the relevant paper titles and page numbers.
- Keep the answer concise but informative.

Research Paper Context:
{context}

User Question:
{question}
"""
    )

    print("Generating answer using Groq...")

    llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return response.content, documents


if __name__ == "__main__":

    print("=" * 70)
    print("RESEARCH PAPER ANSWER BOT")
    print("=" * 70)

    question = input("\nAsk a question about the research papers: ")

    try:
        answer, documents = answer_question(question)

        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)
        print(answer)

        print("\n" + "=" * 70)
        print("TOP 3 SOURCES")
        print("=" * 70)

        for i, document in enumerate(documents, start=1):
            print(f"{i}. {format_source(document)}")

    except Exception as e:
        print("\nAn error occurred:")
        print(e)