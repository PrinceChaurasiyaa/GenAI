from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="llama3")

template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="Write a 3 line summary on the following text. /n {text}",
    input_variables=['text']
)

# prompt1 = template1.invoke({'topic': 'Quantum Mechanics'})

# response = model.invoke(prompt1)

# prompt2 = template2.invoke({'text': response.content})

# response1 = model.invoke(prompt2)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

response = chain.invoke({'topic': 'black hole'})