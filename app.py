import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Chat with Website", page_icon="🌐")

st.title("🌐 Chat with any Website")
st.caption("Powered by Groq & LangChain")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("Settings")
    # Paste your actual key inside the quotes below for the 'value'
    groq_api_key = st.text_input("Groq API Key", type="password", value="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")
    
    # Paste your website link inside the quotes below
    website_url = st.text_input("Website URL", value="http://ctrlplustech.com/")
    
    if st.button("Load Website Data"):
        with st.spinner("Scraping and indexing..."):
            # This triggers the caching logic below
            st.session_state.vector_store = None 
            st.success("Website loaded! You can now chat.")

# --- HELPER FUNCTIONS ---

@st.cache_resource
def get_vectorstore_from_url(url):
    """
    Scrapes the URL, splits text, and creates a vector store.
    Cached so we don't re-scrape on every message.
    """
    try:
        # 1. Load Data
        loader = WebBaseLoader(url)
        docs = loader.load()
        
        # 2. Split Data
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # 3. Create Embeddings & Vector Store
        # Using HuggingFace (runs locally/free)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error loading website: {e}")
        return None

def get_rag_chain(vectorstore, api_key):
    """
    Creates the RAG chain using Groq
    """
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile"
    )

    # --- ENTER YOUR DETAILS HERE ---
    my_contact_info = """
    Email: support@ctrlplustech.com
    Phone: +91 8220123488
    Website: http://ctrlplustech.com/
    """
    # -------------------------------

    # We use an f-string (f"...") to inject your contact info into the instructions
    prompt = ChatPromptTemplate.from_template(f"""
    You are a helpful assistant.
    
    IMPORTANT: If the user asks for contact details, owner information, email, phone, 
    or how to reach us, ALWAYS use the following information:
    {my_contact_info}

    For all other questions, answer based only on the provided context below.
    Think step by step before providing a detailed answer.
    If you cannot answer based on the context, say so.

    <context>
    {{context}}
    </context>

    Question: {{input}}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# --- MAIN APP LOGIC ---

# 1. Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Handle User Input
if user_input := st.chat_input("Ask a question about the website..."):
    
    # Check if API Key and URL are provided
    if not groq_api_key:
        st.info("Please add your Groq API Key in the sidebar to continue.")
        st.stop()
        
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process the response
    with st.chat_message("assistant"):
        # Load vector store (cached)
        if "vector_store_obj" not in st.session_state or st.session_state.vector_store_obj is None:
             st.session_state.vector_store_obj = get_vectorstore_from_url(website_url)
        
        if st.session_state.vector_store_obj:
            rag_chain = get_rag_chain(st.session_state.vector_store_obj, groq_api_key)
            
            response = rag_chain.invoke({"input": user_input})
            answer = response['answer']
            
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})