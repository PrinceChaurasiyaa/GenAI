from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="llama3")

prompt1 = PromptTemplate(
    template="Generate a summary including dialogues of a Classic book name {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Give me list of Characters name and dialogue from the story {title}",
    input_variables=["title"]
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'White Night'})

print(result)
chain.get_graph().print_ascii()
