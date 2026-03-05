# Using Decorator

from langchain.tools import tool
@tool
def add(a: int, b: int):
    """Add two numbers"""
    return a+b

result = add.invoke({"a":3, "b": 5})
print(result)
print(add.name)
print(add.description)
print(add.args)
print(add.args_schema.model_json_schema())
# Using StructuredTool
from langchain_core.tools import Tool, StructuredTool

def multiply(a: int, b: int):
    """Multiply two numbers"""
    return a * b

multiply_tool = Tool(
    name="multiply",
    func=multiply,
    description="Useful for multiplying two numbers"
)

multiply_tools = StructuredTool.from_function(multiply)

response = multiply_tools.invoke({"a": 3, "b": 5})
print(response)
print(multiply_tools.name)
print(multiply_tools.description)
print(multiply_tools.args)
print(multiply_tools.args_schema.model_json_schema())