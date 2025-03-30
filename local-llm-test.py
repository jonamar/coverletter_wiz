import ollama

# This script queries a local LLM server using the Ollama Python library
def query_local_llm(prompt, model="llava-llama3"):
    try:
        # Use the Ollama library to query the model
        response = ollama.generate(model=model, prompt=prompt)
        # Extract the response text from the tuples
        completion = ""
        for chunk in response:
            if isinstance(chunk, tuple) and chunk[0] == "response":
                completion += chunk[1]  # Append the response text
        return completion
    except Exception as e:
        print(f"Error connecting to the local LLM: {e}")
        return None

# This is a simple command-line interface to interact with the local LLM
if __name__ == "__main__":
    user_prompt = input("Enter your prompt: ")
    response = query_local_llm(user_prompt)
    if response:
        print("LLM Response:")
        print(response)