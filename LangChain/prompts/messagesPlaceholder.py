from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3")

chat_template = ChatPromptTemplate([
  ('system', "you are a customer support agent (who advice in short sentences)."),
  MessagesPlaceholder(variable_name="chat_history"),
  ('human', '{query}')
])

chat_history = []
with open('history.txt') as f:
  chat_history.extend(f.readlines())

user_query = "Where is my refund?"

prompts = chat_template.invoke({
  'chat_history':chat_history,
  'query':user_query
})

print(prompts)

response = model.invoke(prompts)

print(response.content)