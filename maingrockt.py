import streamlit as st
import requests
import os
import tempfile
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import time

# Try to import markdown, but don't fail if it's not installed
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

# Set page config as the first Streamlit command
st.set_page_config(page_title="Groq LLM RAG Chatbot", page_icon="🤖")

# Groq API configuration
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Initialize SentenceTransformer model for embeddings
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# Initialize FAISS index
vector_dimension = 384  # Dimension of the embedding model output
index = faiss.IndexFlatL2(vector_dimension)

# Function to process text content
def process_text(text: str) -> Tuple[List[str], int]:
    lines = text.split('\n')
    return [text[i:i+512] for i in range(0, len(text), 512)], len(lines)

# Function to process PDF content
def process_pdf(file) -> Tuple[List[str], int]:
    pdf_reader = PdfReader(file)
    text = ""
    total_lines = 0
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        text += page_text
        total_lines += len(page_text.split('\n'))
    chunks, _ = process_text(text)
    return chunks, total_lines

# Function to process Markdown content
def process_markdown(text: str) -> Tuple[List[str], int]:
    if MARKDOWN_AVAILABLE:
        html = markdown.markdown(text)
        plain_text = ''.join(html.split('>')[1].split('<')[0] for _ in html.split('<')[1:])
        return process_text(plain_text)
    else:
        st.warning("Markdown processing is not available. Processing as plain text.")
        return process_text(text)

# Function to add documents to FAISS index
def add_to_index(chunks: List[str]):
    start_time = time.time()
    embeddings = embedding_model.encode(chunks)
    embedding_time = time.time() - start_time
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return chunks, embedding_time

# Function to search FAISS index
def search_index(query: str, k: int = 5) -> List[Tuple[int, float]]:
    query_vector = embedding_model.encode([query])
    faiss.normalize_L2(query_vector)
    distances, indices = index.search(query_vector, k)
    return list(zip(indices[0], distances[0]))

# Function to clear the vector database
def clear_vector_db():
    global index
    index = faiss.IndexFlatL2(vector_dimension)
    st.session_state.total_chunks = 0
    st.session_state.total_lines = 0
    st.session_state.total_embedding_time = 0
    st.session_state.all_chunks = []

# Streamlit app setup
st.title("Groq LLM RAG Chatbot")

# Initialize session state for metrics
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "total_lines" not in st.session_state:
    st.session_state.total_lines = 0
if "total_embedding_time" not in st.session_state:
    st.session_state.total_embedding_time = 0
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []

# Sidebar for API key input and file upload
with st.sidebar:
    st.header("Configuration")
    user_api_key = st.text_input("Enter your Groq API Key:", type="password")
    if user_api_key:
        GROQ_API_KEY = user_api_key

    st.header("Upload Files")
    uploaded_files = st.file_uploader("Choose text, PDF, or Markdown files", accept_multiple_files=True, type=['txt', 'pdf', 'md'])

    if uploaded_files:
        for file in uploaded_files:
            if file.type == "text/plain":
                text_content = file.read().decode()
                chunks, lines = process_text(text_content)
            elif file.type == "application/pdf":
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file.read())
                    temp_file_path = temp_file.name
                chunks, lines = process_pdf(temp_file_path)
                os.unlink(temp_file_path)
            elif file.type == "text/markdown":
                text_content = file.read().decode()
                chunks, lines = process_markdown(text_content)
            st.session_state.all_chunks.extend(chunks)
            st.session_state.total_lines += lines
        
        if st.session_state.all_chunks:
            with st.spinner("Processing and indexing documents..."):
                indexed_chunks, embedding_time = add_to_index(st.session_state.all_chunks)
                st.session_state.total_chunks = len(indexed_chunks)
                st.session_state.total_embedding_time += embedding_time
            st.success(f"Indexed {len(indexed_chunks)} chunks from {len(uploaded_files)} file(s).")

    # Clear Vector DB button
    if st.button("Clear Vector Database"):
        clear_vector_db()
        st.success("Vector database cleared successfully!")

    # Display metrics
    st.header("Vector DB Metrics")
    st.write(f"Total chunks indexed: {st.session_state.total_chunks}")
    st.write(f"Total lines processed: {st.session_state.total_lines}")
    st.write(f"Vector DB size: {index.ntotal} vectors")
    st.write(f"Total embedding time: {st.session_state.total_embedding_time:.2f} seconds")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to interact with Groq API
def get_groq_response(prompt, context=""):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Use the provided context to answer questions if available, but don't mention the context explicitly in your response unless asked. If no context is provided, answer to the best of your ability."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}" if context else prompt}
    ]
    
    data = {
        "model": "mixtral-8x7b-32768",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(GROQ_API_URL, json=data, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        return f"Error: Unable to get response from Groq API. {str(e)}"

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to ask?"):
    if not GROQ_API_KEY:
        st.error("Please enter your Groq API Key in the sidebar.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Perform RAG if vector database is not empty
        context = ""
        if index.ntotal > 0:
            with st.spinner("Searching knowledge base..."):
                search_results = search_index(prompt)
                context = "\n".join([st.session_state.all_chunks[i] for i, _ in search_results if i < len(st.session_state.all_chunks)])

        # Get and display Groq LLM response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = get_groq_response(prompt, context)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Instructions
st.sidebar.markdown("---")
st.sidebar.subheader("How to use:")
st.sidebar.markdown("""
1. Enter your Groq API Key in the sidebar.
2. (Optional) Upload text, PDF, or Markdown files to build the knowledge base.
3. Type your question in the chat input at the bottom.
4. Press Enter to send your message.
5. The chatbot will search the knowledge base (if available) and use the Groq LLM to generate a response.
6. Use the 'Clear Vector Database' button to remove all indexed documents.
""")

# Markdown module installation instructions
if not MARKDOWN_AVAILABLE:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Markdown Support")
    st.sidebar.warning("""
    The 'markdown' module is not installed. To enable Markdown processing:
    1. Open a terminal or command prompt.
    2. Run the following command:
       ```
       pip install markdown
       ```
    3. Restart this Streamlit application.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created with Streamlit, Groq LLM, and FAISS")
