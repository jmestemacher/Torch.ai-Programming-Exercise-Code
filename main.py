import argparse
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings

from get_embedding_function import get_embedding_function


def main():

    # Have user input whether the database should be updated
    while True:
        


    # Have user give prompt for AI agent
    while True:
        user_prompt = input('Please give a prompt to the LLM: \n')
        # Make sure inputted prompt isn't empty, otherwise reprompt user
        if user_prompt:
            break
        else:
            print("Please give a valid prompt.")

    system_prompt = """
    Answer the given question based just on this:

    {text}

    ---

    Answer this question based on the above text: {user_prompt}. If you cannot
    answer the question, please state as such and explain why. Please answer
    in a structured format.
    """


    # Define the embedding function
    def get_embedding_function():
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        return embeddings

    # Set up the database
    database = Chroma(persist_directory="chroma", embedding_function=get_embedding_function)

    # Search through the database
    results = database.similarity_search_with_score(user_prompt, k=6)

    # Format the results
    context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(system_prompt)
    prompt = prompt_template.format(context=context, question="")

    # Send our prompt to our local LLM
    model = Ollama(model="llama3.1")
    response = model.invoke(prompt)

    # Format and show the response to the user
    citations = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Here is the response: {response}\nHere are the Sources: {citations}"
    print(formatted_response)



if __name__ == "__main__":
    main()

