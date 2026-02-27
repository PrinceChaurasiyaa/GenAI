from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="llama3")

prompt = PromptTemplate(
    template="Tell me a Joke on {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic': 'AI'})



print(result)
chain.get_graph().print_ascii()
