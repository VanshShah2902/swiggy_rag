from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=800, chunk_overlap=150):
    """
    Splits the loaded documents into smaller semantic chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
