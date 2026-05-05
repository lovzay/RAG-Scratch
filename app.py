import streamlit as st
import os
from dotenv import load_dotenv

# 1. Load environment variables (API Keys)
load_dotenv()

# 2. Page Configuration
st.set_page_config(page_title="Thesis RAG Assistant", page_icon="📚")

st.title("📚 Thesis RAG Assistant")
st.markdown("Query your 100-page thesis paper using OpenAI and ChromaDB.")

# 3. Sidebar for status and file management
with st.sidebar:
    st.header("System Status")
    # We will use this later to show if the database is populated
    st.info("Vector Database: Ready (Placeholder)")
    
    if st.button("Refresh Knowledge Base"):
        st.warning("This will trigger populate_database.py logic soon!")

# 4. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Chat Input logic
if prompt := st.chat_input("Ask something about the thesis..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Placeholder for RAG Logic
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = "I see your question! Once we finish the 'populate_database.py' script, I'll be able to retrieve the exact answer from your PDF here."
        response_placeholder.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})