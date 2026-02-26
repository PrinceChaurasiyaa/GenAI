from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

model = ChatOllama(model="llama3")

class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke")

parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Tell me a joke.\n{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

result = chain.invoke({})

print(result)