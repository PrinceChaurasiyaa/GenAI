from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch

model = ChatOllama(model="llama3")

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Summarize the following text \n {text}",
    input_variables=['text'])

parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 200, prompt2 | model | parser),
    RunnablePassthrough()
)

final_chain = RunnableSequence(chain, branch_chain)

result = final_chain.invoke({'topic': 'Schrödinger Cat'})

print(result)