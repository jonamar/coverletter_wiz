import ollama
import yaml

# Load categories from a YAML file
def load_categories(yaml_file):
    try:
        with open(yaml_file, "r") as file:
            categories = yaml.safe_load(file)
        return categories
    except Exception as e:
        print(f"Error loading categories from YAML file: {e}")
        return None

# Query the local LLM to classify the text
def query_local_llm(prompt, model="llava-llama3", max_tags=5):
    try:
        response = ollama.generate(model=model, prompt=prompt)
        completion = ""
        for chunk in response:
            if isinstance(chunk, tuple) and chunk[0] == "response":
                completion += chunk[1]
        return completion
    except Exception as e:
        print(f"Error connecting to the local LLM: {e}")
        return None

# Main function to classify text based on categories
if __name__ == "__main__":
    # Load categories from the YAML file
    yaml_file = "categories.yaml"  # Ensure this file is in the same directory
    categories = load_categories(yaml_file)
    
    if not categories:
        print("Failed to load categories. Exiting.")
        exit(1)

    # Prepare the list of tags as a string
    tags = []
    for category, items in categories.items():
        tags.extend(items)
    tags_string = ", ".join(tags)

    # Get user input
    user_text = input("Enter the text to classify: ")

    # Prepare the prompt for the LLM
    max_tags = 3  # Limit the number of tags to return
    prompt = f"""
You are a text classifier. Based on the following list of tags, pick {max_tags} tags that are the most appropriate to describe the given text.

Tags: {tags_string}

Text: "{user_text}"

Respond with a comma-separated list of {max_tags} of the most relevant tags.
"""

    # Query the LLM
    response = query_local_llm(prompt)
    if response:
        print("Classified Tags:")
        print(response)