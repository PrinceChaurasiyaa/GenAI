from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
  model="llama3"
)

text = "LangChain is the framework"
text2 = "LangGraph is a library for building stateful, multi-actor applications with LLMs"

single_vector = embeddings.embed_query(text)
print(str(single_vector)[:100])

multiple_vector = embeddings.embed_documents([text, text2])
for vector in multiple_vector:
  print(str(vector)[:100])
