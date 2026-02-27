from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="llama3")

parser = StrOutputParser()

loader = TextLoader("data.txt", encoding="utf-8")
documents = loader.load()

prompt = PromptTemplate(
    template="Write a summary of the following Poem - \n {poem}",
    input_variables=['poem']
)

chain = prompt | model | parser

result = chain.invoke({'poem': documents[0].page_content})

print(result)