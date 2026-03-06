# Swiggy Annual Report RAG AI Assistant

## Overview
A production-ready Retrieval Augmented Generation (RAG) system that answers questions based strictly on the Swiggy Annual Report PDF. It processes the document offline to build a locally stored FAISS vector database and serves an interactive Streamlit application for end-user querying. It completely prevents hallucinations by using a strict LLM prompting strategy.

## Architecture
1. **Document Loading**: Instructs `PyPDFLoader` to parse the Swiggy Annual Report.
2. **Chunking**: Uses `RecursiveCharacterTextSplitter` splitting the text into blocks (800 chars, 150 overlap).
3. **Embeddings**: Employs `sentence-transformers/all-MiniLM-L6-v2` locally via HuggingFace for encoding semantic representations.
4. **Vector Database**: Persists embeddings via `FAISS` to the local storage.
5. **Retrieval**: Uses FAISS retriever (k=3) during inference to fetch the top 3 relevant text chunks.
6. **Generation**: Interacts with Groq LLMs via LangChain passing a strict system set of instructions strictly utilizing the context chunks.

## Tech Stack
* **Python**: Core programming language
* **LangChain**: Orchestration framework for LLMs and Document pipelines
* **FAISS**: Local Vector Database
* **HuggingFace Sentence Transformers**: Local embedding model
* **Groq API**: Large Language Model API provider for fast generation
* **PyPDF**: PDF document parsing
* **Streamlit**: Web Frontend Interface

## Project Structure
```text
rag-swiggy-ai/
├── data/
│   └── swiggy_annual_report.pdf
├── scripts/
│   └── build_vector_db.py
├── src/
│   ├── loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── rag_pipeline.py
│   └── prompts.py
├── vector_db/
│   ├── index.faiss
│   └── index.pkl
├── app/
│   └── streamlit_app.py
├── requirements.txt
├── .env
└── README.md
```

## Setup Instructions

1. **Clone or Download the Repository**
2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure Environment Variables**
   Create a `.env` file in the root directory and add your Groq API Key:
   ```env
   GROQ_API_KEY="your-groq-api-key"
   ```
5. **Provide the PDF file**
   Place your document at `data/swiggy_annual_report.pdf`.
6. **Build Vector Database (Run Offline First)**
   ```bash
   python scripts/build_vector_db.py
   ```
7. **Run the Streamlit App**
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Example Questions
* Who is the CEO of Swiggy?
* How many cities does Swiggy operate in?
* What was Swiggy's net loss in FY24?
* What is Instamart?

## Future Improvements
* Improve retrieval quality by experimenting with larger sentence transformer models.
* Implement a hybrid search (Keywords + Standard Semantic).
* Try different metadata filtering strategies (e.g. by section, or page type).
