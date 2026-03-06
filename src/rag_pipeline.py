import os
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.retriever import get_retriever
from src.prompts import rag_prompt

def format_docs(docs):
    """
    Formats the retrieved documents into a single string combining text and page numbers.
    """
    formatted = []
    for doc in docs:
        page_num = doc.metadata.get('page', 'Unknown')
        content = doc.page_content.strip()
        formatted.append(f"[Page {page_num}]: {content}")
    return "\n\n".join(formatted)

def build_rag_pipeline(vector_db_path: str):
    """
    Builds the full RAG pipeline using LangChain expression language (LCEL).
    """
    # 1. Initialize the LLM (requires GROQ_API_KEY environment variable)
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    
    # 2. Get the retriever
    retriever = get_retriever(vector_db_path=vector_db_path, k=3)
    
    # 3. Create the pipeline
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever
