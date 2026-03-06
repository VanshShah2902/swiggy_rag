import os
import sys

# Add the root project directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.vectorstores import FAISS
from src.loader import load_pdf
from src.chunker import split_documents
from src.embeddings import get_huggingface_embeddings

def main():
    pdf_path = os.path.join("data", "swiggy_annual_report.pdf")
    vector_db_path = "vector_db"
    
    print(f"Loading PDF from {pdf_path}...")
    try:
        documents = load_pdf(pdf_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please place 'swiggy_annual_report.pdf' inside the 'data' directory.")
        return

    print("Splitting document into chunks...")
    chunks = split_documents(documents, chunk_size=800, chunk_overlap=150)
    print(f"Created {len(chunks)} chunks.")

    print("Initializing embeddings model...")
    embeddings = get_huggingface_embeddings()

    print("Creating FAISS vector store and generating embeddings...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    print(f"Saving FAISS index locally to '{vector_db_path}/'...")
    os.makedirs(vector_db_path, exist_ok=True)
    vector_store.save_local(vector_db_path)
    
    print("Vector database built successfully!")

if __name__ == "__main__":
    main()
