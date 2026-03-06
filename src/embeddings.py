from langchain_community.embeddings import HuggingFaceEmbeddings

def get_huggingface_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Initializes and returns the HuggingFace sentence transformer embeddings.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
