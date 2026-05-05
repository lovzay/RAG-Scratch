import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()

CHROMA_PATH = "chromadb"

# Define the Prompt Template (The "Augmentation" part of RAG)
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# 2. Page Configuration
st.set_page_config(page_title="Thesis RAG Assistant", page_icon="📚")
st.title("📚 Thesis RAG Assistant")

# 3. Initialize the Vector DB connection
# We do this once to keep the app fast
embedding_function = OpenAIEmbeddings()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# 4. Chat Interface Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something about the thesis..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- THE RAG LOGIC ---
    with st.chat_message("assistant"):
        with st.spinner("Searching thesis..."):
            # A. Search for relevant chunks (Retrieval)
            results = db.similarity_search_with_relevance_scores(prompt, k=5)
            
            # B. Prepare context text
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            
            # C. Fill the prompt template
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            final_prompt = prompt_template.format(context=context_text, question=prompt)
            
            # D. Generate answer with OpenAI
            model = ChatOpenAI(model="gpt-4o") # or gpt-3.5-turbo
            response_text = model.invoke(final_prompt).content
            
            # E. Get sources for transparency
            sources = [doc.metadata.get("id", None) for doc, _score in results]
            formatted_response = f"{response_text}\n\n**Sources:** {', '.join(sources)}"
            
            st.markdown(formatted_response)
        
    st.session_state.messages.append({"role": "assistant", "content": formatted_response})