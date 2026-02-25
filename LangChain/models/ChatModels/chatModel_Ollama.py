from langchain_ollama import ChatOllama

llm = ChatOllama(
  model="llama3",
  temperature=0.5
)

response = llm.invoke("Explain the concept of quantum entanglement")

print(response.content)