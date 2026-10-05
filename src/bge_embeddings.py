from langchain_huggingface import HuggingFaceEmbeddings


def create_bge_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return embeddings


if __name__ == "__main__":
    embeddings = create_bge_embeddings()

    test_embedding = embeddings.embed_query(
        "What is the main idea of the Transformer architecture?"
    )

    print("BGE embedding model loaded successfully!")
    print("Embedding dimensions:", len(test_embedding))