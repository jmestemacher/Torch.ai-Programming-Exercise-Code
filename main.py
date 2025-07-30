


# Have user give prompt to AI agent
while True:
    user_prompt = input('Please give a prompt to the LLM: \n')
    # Make sure inputted prompt isn't empty, otherwise reprompt user
    if user_prompt:
        break
    else:
        print("Please give a valid prompt.")


 
