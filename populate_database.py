import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Load API Key
load_dotenv()

CHROMA_PATH = "chromadb"
DATA_PATH = "data"

def main():
    # 1. Load Documents
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {DATA_PATH}.")

    # 2. Split Text into Chunks
    # We use 1000 characters with a 10% overlap to keep context between chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} pages into {len(chunks)} chunks.")

    # 3. Create/Update Vector Database
    save_to_chroma(chunks)

def save_to_chroma(chunks):
    # Initialize OpenAI Embeddings
    embeddings = OpenAIEmbeddings()

    # Create the vector store and persist it to the chromadb folder
    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_PATH
    )
    
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()