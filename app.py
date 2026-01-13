"""
Streamlit web interface for the Resume Chatbot.
Run with: streamlit run app.py
"""

import streamlit as st
from chatbot import create_chatbot, chat

st.set_page_config(
    page_title="Resume Chatbot",
    page_icon="💼",
    layout="centered",
)

st.title("💼 Resume Chatbot")
st.caption("Ask me anything about my professional experience!")


@st.cache_resource
def get_chatbot():
    """Load chatbot once and cache it."""
    return create_chatbot()


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load chatbot
try:
    chain = get_chatbot()
except ValueError as e:
    st.error(str(e))
    st.info("Add your resume/experience files to the `data/` folder, then refresh this page.")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat(chain, prompt, st.session_state.chat_history)
            st.markdown(response)
    
    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.chat_history.append((prompt, response))

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This chatbot answers questions about professional experience 
        using RAG (Retrieval-Augmented Generation).
        
        **How it works:**
        1. Your resume files are loaded from `data/`
        2. Text is converted to vector embeddings
        3. Relevant context is retrieved for each question
        4. GPT generates a natural response
        """
    )
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
