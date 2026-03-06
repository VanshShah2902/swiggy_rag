from langchain_community.vectorstores import FAISS
from src.embeddings import get_huggingface_embeddings

def get_retriever(vector_db_path: str, k: int = 3):
    """
    Loads the FAISS vector database from disk and returns a retriever.
    """
    embeddings = get_huggingface_embeddings()
    
    # Load the FAISS index (allow_dangerous_deserialization is required for local pkl files)
    vector_store = FAISS.load_local(
        folder_path=vector_db_path, 
        embeddings=embeddings, 
        allow_dangerous_deserialization=True
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever
