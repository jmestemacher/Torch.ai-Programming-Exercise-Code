import argparse
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from update_database import update_database
from langchain_ollama import OllamaLLM
import textwrap

def main():
    print()

    # Have user input whether the database should be updated
    while True:
        question = "Please state whether the database should be updated. Please state either 'Yes' or 'No'.\n"

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

    # Define the embedding function
    def get_embedding_function():
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        return embeddings

    # Set up the database
    database = Chroma(persist_directory="chroma", embedding_function=get_embedding_function())

    # Set up our local LLM model
    model = OllamaLLM(model="llama3.1")

    print()

    # Set up chat history for this session
    chat_history = []

    # Set up to allow multiple prompts (as many as the user wants until they close the application)
    while True:
        # Have user give prompt for AI agent
        while True:
            user_prompt = input('Please give a prompt to the LLM: \n')
            # Make sure inputted prompt isn't empty, otherwise reprompt user
            if user_prompt:
                break
            else:
                print("Please give a valid prompt.\n")

        print()

        # Format chat history for system prompt below
        chat_history_formatted = "\n\n---\n\n".join(chat_history)

        system_prompt = textwrap.dedent("""
        Answer the given question based on either your general knowledge or this (or both, if applicable):

        {text}

        ---

        Answer this question based on the above text: {prompt}. If you cannot
        answer the question, please state as such and explain why. Please answer
        in a structured format. If the above text had no text provided or it does not have the answer to the question state that
        and just answer based on your general knowledge. Do try to distinguish between what is your general knowledge and what is provided by the text.
        Also, the current chat history for this session is given below:

        {history}
        """)

        # Search through the database and get the top 6 results
        results = database.similarity_search_with_score(user_prompt, k=6)

        # Format the results
        context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(system_prompt)
        prompt = prompt_template.format(text=context, prompt=user_prompt, history=chat_history_formatted)

        # Send our prompt to our local LLM
        model = OllamaLLM(model="llama3.1")
        response = model.invoke(prompt)

        # Format and show the response to the user
        citations = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = f"\nHere is the response:\n{response}\n\nHere are the top {len(citations)} sources possibly related to your prompt: {citations}\n"
        print(formatted_response)

        # Append prompt and response to chat history
        chat_history.append(f"Prompt: {user_prompt}\n\nResponse: {response}") # I don't want the sources in the chat history.

        # Check if the user wants to see the actual source text of any citations
        while True:
            # Make sure database isn't empty
            if len(citations) == 0:
                break
            question = textwrap.dedent("""
            Please state which of the sources you want the full text of.
            Give the index starting from 0 and going from left to right for the list shown (0 for the leftmost entry, etc).
            If you want to get another prompt, input 'Next'.\n""")
            input_index = input(question)
            # If user inputs a valid index, give the text of the selected source.
            # Otherwise, reask them.
            try:
                if input_index == "Next":
                    print()
                    break
                input_index = int(input_index)
                print(f"\nHere is the source text:\n\n{results[input_index][0].page_content}\n\nThe id for this chunk is:\n {citations[input_index]}\n")
                print(f"Here is the citation list again for ease of reading: {citations}")
            except:
                # Reask user if they put in an invalid input
                print("\nPlease input either a valid index or 'Next'.\n")

if __name__ == "__main__":
    main()

