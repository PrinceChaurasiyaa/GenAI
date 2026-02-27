from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableSequence

model = ChatOllama(model="llama3")

prompt1 = PromptTemplate(
    template="Describe the equation of {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Author of that equation of {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser,  prompt2, model, parser)

result = chain.invoke({'topic': 'Schrödinger Equation'})

print(result)