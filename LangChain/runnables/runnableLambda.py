from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

model = ChatOllama(model="llama3")

prompt1 = PromptTemplate(
    template="Tell me a joke on {topic}",
    input_variables=['topic']
)

def word_count(text):
    return len(text.split())

parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser)
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

final_chain = RunnableSequence(chain, parallel_chain)

result = final_chain.invoke({'topic': 'Schrödinger Cat'})

print(result)