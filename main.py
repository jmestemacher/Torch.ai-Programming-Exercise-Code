import argparse
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from update_database import update_database
from langchain_ollama import OllamaLLM

def main():
    # Have user input whether the database should be updated
    while True:
        question = "Please state whether the database should be updated. Please state either 'Yes' or 'No'.\nYou will get an error if the database is empty and you choose to not populate it and continue on.\n"

        update_input = input(question)
        # If user inputs "Yes" update the database. If they input "No", continue on and don't update it.
        if update_input == 'Yes':
            update_database()
            break
        elif update_input == 'No':
            break
        # If they don't input "Yes" or "No", reprompt them.
        else:
            print("\nPlease state either 'Yes' or 'No'.\n")

    print()

    # Have user give prompt for AI agent
    while True:
        user_prompt = input('Please give a prompt to the LLM: \n')
        # Make sure inputted prompt isn't empty, otherwise reprompt user
        if user_prompt:
            break
        else:
            print("Please give a valid prompt.\n")

    print()

    system_prompt = """
    Answer the given question based on either your general knowledge or this:

    {text}

    ---

    Answer this question based on the above text: {prompt}. If you cannot
    answer the question, please state as such and explain why. Please answer
    in a structured format.
    """


    # Define the embedding function
    def get_embedding_function():
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        return embeddings

    # Set up the database
    database = Chroma(persist_directory="chroma", embedding_function=get_embedding_function())

    # Search through the database
    results = database.similarity_search_with_score(user_prompt, k=6)

    # Format the results
    context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(system_prompt)
    prompt = prompt_template.format(text=context, prompt=user_prompt)

    # Send our prompt to our local LLM
    model = OllamaLLM(model="llama3.1")
    response = model.invoke(prompt)

    # Format and show the response to the user
    citations = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"\nHere is the response:\n{response}\n\nHere are the Sources: {citations}"
    print(formatted_response)



if __name__ == "__main__":
    main()

