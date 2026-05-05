import os
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
    # 1. Load and Split Documents
    documents = load_documents()
    chunks = split_documents(documents)
    
    # 2. Add Unique IDs to the chunks
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    # 3. Intelligently Add to Chroma
    add_to_chroma(chunks_with_ids)

def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks):
    # This will create IDs like "data/thesis.pdf:5:2"
    # Page Source : Page Number : Chunk Index
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If it's the same page, increment the index
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the chunk metadata
        chunk.metadata["id"] = chunk_id

    return chunks

def add_to_chroma(chunks):
    # Load the existing database
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=OpenAIEmbeddings()
    )

    # Calculate Page IDs
    new_chunk_ids = [chunk.metadata["id"] for chunk in chunks]

    # Get existing documents from the DB
    existing_items = db.get(include=[])  # include=[] makes it faster
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add chunks that don't exist in the DB
    new_chunks = []
    for chunk in chunks:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        print("✅ Database updated successfully.")
    else:
        print("✅ No new documents to add.")

if __name__ == "__main__":
    main()