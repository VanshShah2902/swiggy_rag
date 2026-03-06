from langchain_core.prompts import PromptTemplate

RAG_PROMPT_TEMPLATE = """You are an AI assistant that answers questions using the Swiggy Annual Report.

You must ONLY answer using the provided context.

If the answer cannot be found in the context, respond exactly with:
'I could not find the answer in the Swiggy Annual Report.'

Context:
{context}

Question:
{question}

Answer clearly and mention the page number if available."""

rag_prompt = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)
