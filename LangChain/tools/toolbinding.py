from langchain_core.tools import tool
from langchain_ollama import ChatOllama

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two number"""
    return a * b

print(multiply.invoke({"a": 3, "b": 5}))
print(multiply.description)
print(multiply.name)
print(multiply.args)
print(multiply.args_schema)

# tool binding

llm = ChatOllama(model="llama3")

model = llm.bind_tools([multiply])

print(model)
