import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Add the root project directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag_pipeline import build_rag_pipeline

# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()

st.set_page_config(page_title="Swiggy Annual Report AI Assistant", page_icon="🍔", layout="centered")

# Inject Custom CSS for modern styling
st.markdown("""
<style>
    /* Styling the main container for a card-like effect */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 800px;
    }
    
    /* Title styling */
    h1 {
        color: #FC8019 !important;
        text-align: center;
        font-weight: 800 !important;
        font-size: 2.5rem !important;
        margin-bottom: 0rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* User Message Bubble */
    [data-testid="chatAvatarIcon-user"] {
        background-color: #3b82f6 !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        border-bottom-right-radius: 5px;
        padding: 5px 15px;
        margin-left: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Assistant Message Bubble */
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: #FC8019 !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #fffaf5;
        border: 1px solid #fed7aa;
        border-radius: 15px;
        border-bottom-left-radius: 5px;
        padding: 5px 15px;
        margin-right: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Overriding Streamlit's default chat input padding */
    .stChatInputContainer {
        border-radius: 20px !important;
        border: 2px solid #fdba74 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }
    .stChatInputContainer:focus-within {
        border-color: #FC8019 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading application resources...")
def init_rag_system():
    """
    Initializes the RAG pipeline once and caches it for the Streamlit session.
    It expects the FAISS index to be already generated.
    """
    # Resolve the absolute path to the vector dictionary relative to this script
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    vector_db_dir = os.path.join(root_dir, "vector_db")
    if not os.path.exists(vector_db_dir) or not os.listdir(vector_db_dir):
        raise FileNotFoundError("Vector DB not found. Please run 'python scripts/build_vector_db.py' first.")
    
    rag_chain, retriever = build_rag_pipeline(vector_db_path=vector_db_dir)
    return rag_chain, retriever

st.title("🍔 Swiggy Annual Report AI Assistant")
st.markdown('<p class="subtitle">Ask any questions related to the Swiggy Annual Report. 🚀</p>', unsafe_allow_html=True)

try:
    rag_chain, retriever = init_rag_system()
except Exception as e:
    st.error(f"Failed to initialize the RAG system: {e}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am the Swiggy Annual Report AI Assistant. Ask me anything about the report!"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources and context if available in the message history
        if "sources" in message:
            with st.expander("Sources & Context"):
                st.markdown(f"**Pages:** {', '.join(map(str, message['source_pages']))}")
                for i, chunk in enumerate(message["sources"], start=1):
                    st.markdown(f"**Chunk {i} (Page {chunk['page']}):**")
                    st.write(chunk["content"])
                    st.markdown("---")

# Accept user input
if prompt := st.chat_input("Ask a question about the Swiggy Annual Report..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Analyzing the report and generating answer..."):
            # Retrieve context for displaying
            retrieved_docs = retriever.invoke(prompt)
            
            # Run pipeline to get the answer
            try:
                answer = rag_chain.invoke(prompt)
                
                # Render the answer
                st.markdown(answer)
                
                # Format sources for state and display
                source_pages = list(set([doc.metadata.get('page', 'Unknown') for doc in retrieved_docs]))
                sources_data = [{"page": doc.metadata.get('page', 'Unknown'), "content": doc.page_content} for doc in retrieved_docs]
                
                with st.expander("Sources & Context"):
                    st.markdown(f"**Pages:** {', '.join(map(str, source_pages))}")
                    for i, chunks in enumerate(sources_data, start=1):
                        st.markdown(f"**Chunk {i} (Page {chunks['page']}):**")
                        st.write(chunks["content"])
                        st.markdown("---")
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "source_pages": source_pages,
                    "sources": sources_data
                })
                
            except Exception as e:
                st.error(f"Error calling LLM. Ensure GROQ_API_KEY is set correctly. Error details: {e}")
