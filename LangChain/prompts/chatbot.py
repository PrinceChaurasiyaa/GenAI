from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage



model = ChatOllama(model="llama3")

chat_history = [
    SystemMessage(content="You are a funny story narrator.")
]

while True:
    user_input = input('\nYou: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("\nAI: ", result.content)
print(chat_history)