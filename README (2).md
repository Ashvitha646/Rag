# Smart Research Chatbot Using Hybrid RAG

## 1. Project Overview

Smart Research Chatbot is a research paper question-answering system based on Retrieval-Augmented Generation (RAG).

The system allows users to ask natural-language questions about a collection of research papers. Instead of relying only on the language model's pretrained knowledge, the system retrieves relevant passages from the indexed research papers and provides them as context to the language model.

The system combines semantic retrieval, keyword-based retrieval, hybrid retrieval, Cross-Encoder reranking, and large language model generation to produce context-grounded answers.

## 2. Objectives

- Build a question-answering system for research papers.
- Process and index research papers automatically.
- Store document embeddings in a vector database.
- Compare different embedding models.
- Implement semantic and keyword-based retrieval.
- Combine semantic retrieval with BM25 using hybrid retrieval.
- Improve retrieval using Cross-Encoder reranking.
- Generate answers using a large language model.
- Display relevant research paper and page information.
- Evaluate the performance of different retrieval approaches.

## 3. System Architecture

```text
                         User Question
                              |
                              v
                     Streamlit Interface
                              |
                              v
                       Query Processing
                              |
                 +------------+------------+
                 |                         |
                 v                         v
          BGE Semantic Search         BM25 Search
                 |                         |
                 +------------+------------+
                              |
                              v
                     Hybrid Retrieval
                              |
                              v
                     Candidate Chunks
                              |
                              v
                  Cross-Encoder Reranking
                              |
                              v
                       Top 3 Chunks
                              |
                              v
                    Context Construction
                              |
                              v
                         Groq LLM
                              |
                              v
                     Generated Answer
                              |
                              v
                 Paper and Page Sources
```

## 4. Research Papers

| Research Paper | File |
|---|---|
| Attention Is All You Need | `attention_is_all_you_need.pdf` |
| BERT | `bert.pdf` |
| GPT-3 | `gpt3.pdf` |
| LoRA | `lora.pdf` |
| Retrieval-Augmented Generation | `rag.pdf` |

The current dataset contains:

- 5 research papers
- 151 total pages
- 654 document chunks

## 5. Project Structure

```text
RAG/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
│
├── data/
│   └── papers/
│       ├── attention_is_all_you_need.pdf
│       ├── bert.pdf
│       ├── gpt3.pdf
│       ├── lora.pdf
│       └── rag.pdf
│
├── results/
│   ├── chroma_db/
│   └── chroma_db_bge/
│
├── src/
│   ├── load_papers.py
│   ├── chunk_papers.py
│   ├── create_embeddings.py
│   ├── bge_embeddings.py
│   ├── vector_database.py
│   ├── bge_vector_database.py
│   ├── semantic_search.py
│   ├── hybrid_retrieval.py
│   ├── reranking.py
│   ├── rag_pipeline.py
│   ├── rag_pipeline_reranked.py
│   └── compare_embeddings.py
│
└── evaluation/
    ├── test_questions.csv
    ├── evaluate_embeddings.py
    ├── embedding_results.csv
    ├── evaluate_retrieval.py
    ├── evaluate_rag.py
    └── rag_evaluation_results.csv
```

## 6. Document Processing

### 6.1 PDF Loading

Research papers are loaded from the `data/papers/` directory.

The extracted content retains metadata such as:

- Source PDF
- Page number

This metadata is used later to identify the source of retrieved information.

### 6.2 Document Chunking

The extracted documents are divided into smaller chunks using recursive text splitting.

Current configuration:

```text
Chunk Size: 1000
Chunk Overlap: 200
```

Page-level metadata is retained during the chunking process.

## 7. Embedding Models

### 7.1 MiniLM

```text
sentence-transformers/all-MiniLM-L6-v2
```

MiniLM was used as the initial semantic embedding model.

### 7.2 BGE

```text
BAAI/bge-small-en-v1.5
```

BGE was evaluated as an alternative embedding model and is used in the final Hybrid RAG pipeline.

## 8. Vector Database

ChromaDB is used as the vector database for storing document embeddings.

The project contains two collections:

```text
research_papers
research_papers_bge
```

The BGE collection is used by the Hybrid RAG pipeline.

## 9. Retrieval Methods

### 9.1 Dense Semantic Retrieval

Dense retrieval converts the user question into an embedding and searches for semantically similar document chunks in ChromaDB.

### 9.2 BM25 Retrieval

BM25 is used for keyword-based retrieval. It is useful when specific technical terms or phrases in the query need to be matched directly against the research papers.

### 9.3 Hybrid Retrieval

Hybrid retrieval combines dense semantic retrieval and BM25 retrieval.

```text
Dense Semantic Retrieval
          +
      BM25 Retrieval
          |
          v
    Combined Results
```

Duplicate chunks are removed from the combined results.

### 9.4 Cross-Encoder Reranking

The retrieved candidate chunks are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The Cross-Encoder evaluates the relevance between the question and each retrieved document chunk. The highest-ranked three chunks are selected as the final context for the language model.

## 10. Large Language Model

The final answer is generated using Groq through LangChain.

Current model:

```text
openai/gpt-oss-20b
```

Temperature:

```text
0
```

The prompt instructs the model to:

- Use only the provided research paper context.
- Give a clear explanation.
- Avoid inventing information.
- State when sufficient information is unavailable.
- Refer to retrieved sources.
- Keep the answer concise and informative.

## 11. RAG Pipeline

```text
Research Papers
       |
       v
PDF Loading
       |
       v
Document Chunking
       |
       v
BGE Embeddings
       |
       v
ChromaDB
       |
       +-------------------+
       |                   |
       v                   v
Dense Retrieval       BM25 Retrieval
       |                   |
       +---------+---------+
                 |
                 v
        Hybrid Retrieval
                 |
                 v
        Candidate Chunks
                 |
                 v
      Cross-Encoder Reranking
                 |
                 v
          Top 3 Chunks
                 |
                 v
       Context Construction
                 |
                 v
             Groq LLM
                 |
                 v
          Final Answer
                 |
                 v
        Source Information
```

## 12. User Interface

The project includes a Streamlit interface through which users can:

1. Enter a research question.
2. Retrieve relevant research paper content.
3. Generate an answer using the Groq LLM.
4. View the generated answer.
5. View the top relevant sources.
6. View paper and page information.

## 13. Evaluation Methodology

The project uses a test set containing 10 questions, with two questions for each of the five research papers.

The evaluation focuses on:

- Embedding performance
- Retrieval performance
- Final RAG paper retrieval

## 14. Embedding Evaluation Results

| Embedding Model | Top-1 Accuracy | Top-3 Accuracy |
|---|---:|---:|
| MiniLM | 90% | 100% |
| BGE | 100% | 100% |

BGE achieved 100% Top-1 and Top-3 accuracy on the current 10-question evaluation set.

## 15. Retrieval Evaluation Results

| Retrieval Method | Top-1 Accuracy | Top-3 Accuracy |
|---|---:|---:|
| BGE Dense Retrieval | 100% | 100% |
| BM25 | 70% | 80% |
| Hybrid Retrieval | 100% | 100% |
| Hybrid + Cross-Encoder Reranking | 90% | 100% |

### Observations

BGE performed strongly on the current evaluation dataset.

BM25 alone produced lower retrieval accuracy compared with dense semantic retrieval.

Hybrid retrieval achieved 100% Top-1 and Top-3 accuracy on the current test set.

Cross-Encoder reranking achieved 90% Top-1 and 100% Top-3 accuracy. Therefore, based on this evaluation, reranking did not improve Top-1 accuracy, although it maintained perfect Top-3 retrieval.

## 16. RAG Evaluation Results

The final RAG pipeline was evaluated using the same 10 test questions.

The evaluation checks whether the expected research paper appears among the retrieved sources.

### Result

```text
Correct Expected Paper Retrieved: 100%
```

The expected research paper was retrieved correctly for all 10 evaluation questions.

The results are stored in:

```text
evaluation/rag_evaluation_results.csv
```

## 17. Example

### Question

```text
What problem does LoRA address?
```

### Expected Paper

```text
lora
```

### Retrieved Sources

```text
1. lora — Page 12
2. lora — Page 1
3. lora — Page 4
```

### Another Example

Question:

```text
What is Retrieval-Augmented Generation?
```

Retrieved sources:

```text
1. rag — Page 1
2. rag — Page 1
3. rag — Page 2
```

The expected RAG paper was correctly retrieved and used to generate the answer.

## 18. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming |
| LangChain | RAG pipeline orchestration |
| ChromaDB | Vector database |
| Hugging Face | Embedding models |
| BGE | Semantic embeddings |
| MiniLM | Embedding comparison |
| BM25 | Keyword retrieval |
| Cross-Encoder | Reranking |
| Groq | Language model generation |
| Streamlit | User interface |
| PyPDFLoader | PDF processing |
| Pandas | Evaluation and data processing |

## 19. Installation

### Clone the Repository

```bash
git clone <repository-url>
cd RAG
```

### Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 20. Environment Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

The API key should not be committed to the repository.

The `.gitignore` file should contain:

```text
.env
venv/
__pycache__/
*.pyc
```

## 21. Running the Application

Run the Streamlit application using:

```bash
streamlit run app.py
```

## 22. Running the Evaluations

### Embedding Evaluation

```bash
python evaluation/evaluate_embeddings.py
```

### Retrieval Evaluation

```bash
python evaluation/evaluate_retrieval.py
```

This evaluates:

- Dense BGE retrieval
- BM25 retrieval
- Hybrid retrieval
- Hybrid retrieval with Cross-Encoder reranking

### RAG Evaluation

```bash
python evaluation/evaluate_rag.py
```

This evaluates whether the expected research paper is retrieved for each test question.

## 23. Evaluation Files

```text
evaluation/
│
├── test_questions.csv
├── evaluate_embeddings.py
├── embedding_results.csv
├── evaluate_retrieval.py
├── evaluate_rag.py
└── rag_evaluation_results.csv
```

## 24. Limitations

The current evaluation uses a relatively small test set of 10 questions.

Therefore, the reported results represent performance on the current evaluation dataset and should not be interpreted as a general benchmark.

The current RAG evaluation measures whether the expected research paper is retrieved. It does not independently measure the complete factual correctness or groundedness of every generated answer.

The Cross-Encoder reranking experiment did not improve Top-1 retrieval accuracy on the current test set.

## 25. Future Enhancements

Possible future improvements include:

- Conversational memory
- Multi-turn research conversations
- Multi-query retrieval
- Semantic chunking
- CRAG with web search
- Additional research papers
- Larger embedding models
- Automated answer-quality evaluation
- LLM-based evaluation
- Improved citation formatting
- Research paper comparison
- Automatic paper summarization
- Question generation from research papers

## 26. Key Results

| Evaluation | Result |
|---|---:|
| MiniLM Top-1 | 90% |
| MiniLM Top-3 | 100% |
| BGE Top-1 | 100% |
| BGE Top-3 | 100% |
| BM25 Top-1 | 70% |
| BM25 Top-3 | 80% |
| Hybrid Top-1 | 100% |
| Hybrid Top-3 | 100% |
| Hybrid + Reranker Top-1 | 90% |
| Hybrid + Reranker Top-3 | 100% |
| RAG Expected-Paper Retrieval | 100% |

## 27. Conclusion

Smart Research Chatbot using Hybrid RAG provides a research paper question-answering system that combines semantic retrieval, keyword retrieval, reranking, and large language model generation.

The system uses BGE embeddings with ChromaDB for semantic retrieval, BM25 for keyword retrieval, hybrid retrieval for combining both approaches, and a Cross-Encoder for reranking candidate passages.

The final retrieved context is provided to a Groq language model to generate an answer based on the indexed research papers.

On the current 10-question evaluation set, BGE and Hybrid Retrieval achieved 100% Top-1 and Top-3 retrieval accuracy. The final RAG evaluation also achieved 100% expected-paper retrieval.

The project demonstrates the application of Retrieval-Augmented Generation for building a research-oriented question-answering system over a curated collection of research papers.
