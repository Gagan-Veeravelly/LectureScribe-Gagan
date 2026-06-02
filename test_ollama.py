from langchain_ollama import ChatOllama

# Initialize the model
# We use "llama3.2" because that matches the model you just pulled
llm = ChatOllama(
    model="llama3.2",
    temperature=0  # Set to 0 for deterministic outputs
)

print("Sending request to Llama 3.2...")

# Run a simple query
try:
    response = llm.invoke("Summarize the concept of a 'Rate Limiter' in one sentence.")
    print("\nResponse from Llama 3.2:")
    print("-" * 30)
    print(response.content)
    print("-" * 30)
except Exception as e:
    print(f"Error: {e}")
