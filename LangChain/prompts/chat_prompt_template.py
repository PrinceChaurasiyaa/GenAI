from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

model = ChatOllama(model="llama3")

chat_template = ChatPromptTemplate([
  ('system', 'You are a helpful {domain} expert.'),
  ('user', 'Explain in simple terms, what is {topic}')
  # SystemMessage(content="You are a helpful {domain} expert."),
  # HumanMessage(content="Explain in simple terms, what is {topic}")
])

prompt =  chat_template.invoke({'domain': 'football', 'topic': 'offside rule'})

print(prompt)

response = model.invoke(prompt)
print(response.content)
