import argparse
import os
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings



def update_database():
    print()
    # Have user input whether the database should be reset before updating
    while True:
        reset_input = input("Please state whether the database should be reset before updating it. Please state either 'Yes' or 'No'.\n")
        # If user inputs "Yes" reset the database. If they input "No", continue on.
        if reset_input == 'Yes':
            reset_database()
            break
        elif reset_input == 'No':
            break
        # If they don't input "Yes" or "No", reprompt them.
        else:
            print("\nPlease state either 'Yes' or 'No'.\n")

    print()

    # Load the documents in the "data" folder
    doc_loader = PyPDFDirectoryLoader("data")
    documents = doc_loader.load()

    # Split the loaded documents into chunks for adding
    # to the database
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 900,
        chunk_overlap = 80,
        length_function=len,
        is_separator_regex=False
    )
    chunks = text_splitter.split_documents(documents)

    # Define the embedding function
    def get_embedding_function():
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        return embeddings

    # Load the database
    database = Chroma(
        persist_directory="chroma", embedding_function=get_embedding_function()
    )

    # Get page ids
    chunks_with_ids = get_chunk_ids(chunks)

    # Get existing ids
    existing_items = database.get(include=[])
    existing_ids = set(existing_items['ids'])

    # Add new chunks to database
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata['id'] not in existing_ids:
            new_chunks.append(chunk)
    if len(new_chunks):
        print(f"Adding new documents to database: {len(new_chunks)}")
        # Make sure number of new chunks does not exceed max batch size of 5461, else
        # split the new chunks list into subsets, each max size 5461
        if len(new_chunks) >= 5462:
            # Form a batch of the first 5461 chunks
            new_chunks_batch = new_chunks[:5461]
            # Set next batch 
            new_chunks = new_chunks[5461:]
            at_end = False
            while True:
                # Add current batch to database
                new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks_batch]
                database.add_documents(new_chunks_batch, ids=new_chunk_ids)
                if at_end: # Stop adding new documents if we have added all the chunks
                    break
                if len(new_chunks) >= 5462: # Check if we have enough for another full batch, if so
                    # create another full batch and prepare the one after that, else let the next batch be the last one
                    new_chunks_batch = new_chunks[:5461]
                    new_chunks = new_chunks[5461:]
                else:
                    new_chunks_batch = new_chunks
                    at_end = True
        else:
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            database.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("There are no new documents to be added.\n")

    return

# Function for resetting the database
def reset_database():
    # Clear the database if it exists
    if os.path.exists("chroma"):
        shutil.rmtree("chroma")

# Function for getting the ids for chunks
def get_chunk_ids(chunks):
    # ID's are in the structure of
    # Page Source: Page Number: Index of Chunk
    # Example: "Army_FM_Intelligence":3:0
    
    previous_page_id = None
    current_chunk_index = 0

    # Go through each chunk
    for chunk in chunks:
        # Get metadata for chunk
        source = chunk.metadata.get("source").replace("data\\", "")
        page = chunk.metadata.get("page")
        current_page_id = f"{source} : Page {page}"

        # Increment chunk index if page ID is same as previous one
        if current_page_id == previous_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        # Get the chunk id by appending Page ID and chunk index
        chunk_id = f"{current_page_id} : Chunk Index {current_chunk_index}"
        previous_page_id = current_page_id

        # Add id to chunk meta-data
        chunk.metadata["id"] = chunk_id
    
    return chunks