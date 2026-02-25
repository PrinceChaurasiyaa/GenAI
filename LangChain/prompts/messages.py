from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

role = "You are a funny physics assistant"
user = "Tell me about Gravity."

messages = [
  SystemMessage(content=role),
  HumanMessage(content=user)
]

model = ChatOllama(model="llama3")

response = model.invoke(messages)

messages.append(AIMessage(content=response.content))

print(messages)