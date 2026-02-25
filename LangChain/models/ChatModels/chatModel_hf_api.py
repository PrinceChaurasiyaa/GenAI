from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=100
)

response = llm.invoke("What is the full form of CME from the Sun?")
print(response)