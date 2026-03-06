import os
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path: str):
    """
    Loads a PDF document using PyPDFLoader.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents
